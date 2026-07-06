"""Strict Parquet-backed market data reader for runtime strategy paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_investor.agent_protocol import DataQualityIssue
from quant_investor.market.read_result import MarketDataReadResult


class MarketDataUnavailableError(RuntimeError):
    """Raised when strict Parquet market data is not healthy enough to read."""


def _normalize_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_symbol_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol and symbol not in result:
            result.append(symbol)
    return result


@dataclass(frozen=True)
class ParquetSnapshot:
    snapshot_id: str
    latest_complete_trade_date: str
    latest_trade_date: str
    table_root: Path
    serving_root: Path
    manifest_path: Path
    latest_pointer_path: Path


class MarketDataReader:
    """Unified local market data reader.

    CN runtime paths use the strict Parquet canonical pointer plus symbol-serving
    files.  CSV remains available elsewhere for exports and legacy maintenance,
    but this reader never falls back to CSV when the Parquet snapshot is missing
    or unhealthy.
    """

    def __init__(
        self,
        *,
        market: str = "CN",
        data_root: str | Path | None = None,
        mode_policy: str = "strict",
    ) -> None:
        self.market = str(market or "").strip().upper()
        self.data_root = Path(data_root or "data")
        self.mode_policy = str(mode_policy or "strict").strip().lower() or "strict"
        self.parquet_market_root = self.data_root / "parquet" / self.market.lower()
        self.latest_pointer_path = self.parquet_market_root / "_latest.json"
        self.catalog_path = self.parquet_market_root / "_catalog.json"
        self.issues: list[DataQualityIssue] = []
        self._latest_payload: dict[str, Any] | None = None
        self._snapshot_gate_cache: dict[str, Any] | None = None
        self._serving_symbols_cache: tuple[tuple[str, str], list[str]] | None = None
        self._components_payload: dict[str, Any] | None = None
        self._catalog_payload: dict[str, Any] | None = None

    def _resolve_data_path(self, raw_path: Any, fallback: Path) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            return fallback
        path = Path(text)
        if path.is_absolute():
            return path
        candidates = [
            Path.cwd() / path,
            self.data_root / path,
        ]
        if self.data_root.name == "data":
            candidates.append(self.data_root.parent / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_latest_payload(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._latest_payload is not None and not refresh:
            return dict(self._latest_payload)
        if not self.latest_pointer_path.exists():
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer missing: {self.latest_pointer_path}"
            )
        try:
            payload = json.loads(self.latest_pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer unreadable: {self.latest_pointer_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer invalid: {self.latest_pointer_path}"
            )
        self._latest_payload = dict(payload)
        return dict(payload)

    def _snapshot_from_payload(self, payload: Mapping[str, Any]) -> ParquetSnapshot:
        table_root = self._resolve_data_path(
            payload.get("table_root"),
            self.parquet_market_root / "bars",
        )
        serving_root = self._resolve_data_path(
            payload.get("derived_serving_root"),
            self.data_root / "parquet_serving" / self.market.lower() / "bars",
        )
        manifest_path = self._resolve_data_path(
            payload.get("manifest_path") or payload.get("clean_manifest_path"),
            self.parquet_market_root / "_snapshots" / f"{payload.get('snapshot_id', '')}.json",
        )
        return ParquetSnapshot(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            latest_complete_trade_date=_normalize_trade_date(
                payload.get("latest_complete_trade_date") or payload.get("latest_trade_date")
            ),
            latest_trade_date=_normalize_trade_date(
                payload.get("latest_trade_date") or payload.get("latest_complete_trade_date")
            ),
            table_root=table_root,
            serving_root=serving_root,
            manifest_path=manifest_path,
            latest_pointer_path=self.latest_pointer_path,
        )

    def clean_snapshot_gate(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._snapshot_gate_cache is not None and not refresh:
            return dict(self._snapshot_gate_cache)
        try:
            payload = self._load_latest_payload(refresh=refresh)
        except MarketDataUnavailableError as exc:
            return {
                "status": "blocked",
                "healthy": False,
                "blockers": [str(exc)],
                "latest_pointer_path": str(self.latest_pointer_path),
                "mode_policy": self.mode_policy,
            }

        snapshot = self._snapshot_from_payload(payload)
        blockers: list[str] = []
        if str(payload.get("status") or "").upper() != "OK":
            blockers.append(f"latest pointer status is {payload.get('status')!r}")
        blockers.extend(str(item) for item in list(payload.get("blockers", []) or []) if str(item).strip())
        if not snapshot.snapshot_id:
            blockers.append("snapshot_id missing")
        if not snapshot.latest_complete_trade_date:
            blockers.append("latest_complete_trade_date missing")
        if not snapshot.table_root.exists():
            blockers.append(f"canonical bars table_root missing: {snapshot.table_root}")
        elif not any(snapshot.table_root.rglob("*.parquet")):
            blockers.append(f"canonical bars table_root has no parquet files: {snapshot.table_root}")
        if not snapshot.serving_root.exists():
            blockers.append(f"serving bars root missing: {snapshot.serving_root}")
        elif not any(snapshot.serving_root.glob("symbol=*/bars.parquet")):
            blockers.append(f"serving bars root has no symbol parquet files: {snapshot.serving_root}")
        if not snapshot.manifest_path.exists():
            blockers.append(f"manifest missing: {snapshot.manifest_path}")

        gate_payload = {
            "status": "ok" if not blockers else "blocked",
            "healthy": not blockers,
            "blockers": blockers,
            "snapshot_id": snapshot.snapshot_id,
            "latest_complete_trade_date": snapshot.latest_complete_trade_date,
            "latest_trade_date": snapshot.latest_trade_date,
            "table_root": str(snapshot.table_root),
            "serving_root": str(snapshot.serving_root),
            "manifest_path": str(snapshot.manifest_path),
            "latest_pointer_path": str(snapshot.latest_pointer_path),
            "mode_policy": self.mode_policy,
        }
        self._snapshot_gate_cache = dict(gate_payload)
        return dict(gate_payload)

    def _require_snapshot(self) -> ParquetSnapshot:
        gate = self.clean_snapshot_gate()
        if not gate.get("healthy"):
            blockers = "; ".join(str(item) for item in gate.get("blockers", []) if str(item).strip())
            raise MarketDataUnavailableError(blockers or "strict Parquet snapshot is not healthy")
        return self._snapshot_from_payload(self._load_latest_payload())

    def snapshot(self) -> dict[str, Any]:
        gate = self.clean_snapshot_gate()
        payload = {
            "backend": "parquet",
            "storage_layer": "canonical+serving",
            "mode_policy": self.mode_policy,
            "resolution_strategy": "strict_parquet_serving",
            "fallback_used": False,
        }
        payload.update(gate)
        return payload

    def physical_directories_for_full_a(self) -> list[Path]:
        snapshot = self._require_snapshot()
        return [snapshot.serving_root]

    def _serving_symbols(self, snapshot: ParquetSnapshot) -> list[str]:
        cache_key = (snapshot.snapshot_id, str(snapshot.serving_root))
        if self._serving_symbols_cache is not None:
            cached_key, cached_symbols = self._serving_symbols_cache
            if cached_key == cache_key:
                return list(cached_symbols)
        symbols = []
        for path in sorted(snapshot.serving_root.glob("symbol=*/bars.parquet")):
            symbol = path.parent.name.split("symbol=", 1)[-1].strip().upper()
            if symbol:
                symbols.append(symbol)
        self._serving_symbols_cache = (cache_key, list(symbols))
        return list(symbols)

    def _load_components(self) -> dict[str, Any]:
        if self._components_payload is not None:
            return dict(self._components_payload)
        candidates = [
            self.data_root / "cn_universe" / "cn_index_components.json",
            Path.cwd() / "data" / "cn_universe" / "cn_index_components.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                self._components_payload = dict(payload)
                return dict(self._components_payload)
        self._components_payload = {}
        return {}

    def list_symbols(
        self,
        universe_key: str = "full_a",
        category: str | None = None,
        as_of: str | None = None,
    ) -> list[str]:
        snapshot = self._require_snapshot()
        serving_symbols = self._serving_symbols(snapshot)
        key = str(category or universe_key or "full_a").strip().lower()
        if key in {"", "all", "full", "full_a", "all_a", "full_market"}:
            if self.market == "CN" and as_of:
                from quant_investor.config import config
                from quant_investor.market.pit_universe import PITUniverseStore

                if bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
                    pit_symbols = PITUniverseStore.from_config().listed_symbols(as_of)
                    if pit_symbols:
                        serving_set = set(serving_symbols)
                        return [symbol for symbol in pit_symbols if symbol in serving_set]
                    if bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)):
                        return []
            return serving_symbols

        components = self._load_components()
        component_symbols = [
            _normalize_symbol(symbol)
            for symbol in list(components.get(key, []) or [])
            if _normalize_symbol(symbol)
        ]
        if component_symbols:
            serving_set = set(serving_symbols)
            symbols = [symbol for symbol in _dedupe(component_symbols) if symbol in serving_set]
            if self.market == "CN" and as_of:
                from quant_investor.config import config
                from quant_investor.market.pit_universe import filter_symbols_by_pit_status, PITUniverseStore

                if bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
                    records = PITUniverseStore.from_config().records_by_symbol()
                    filtered = filter_symbols_by_pit_status(
                        symbols,
                        as_of=as_of,
                        records=records,
                        required=bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)),
                    )
                    return filtered.symbols
            return symbols
        if self.market != "CN":
            return serving_symbols
        return []

    def resolve_symbol_path(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        for_write: bool = False,
    ) -> Path | None:
        snapshot = self._require_snapshot()
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return None
        path = snapshot.serving_root / f"symbol={normalized}" / "bars.parquet"
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return path if path.exists() else None

    def _metadata(self, snapshot: ParquetSnapshot, **overrides: Any) -> dict[str, Any]:
        metadata = {
            "backend": "parquet",
            "storage_layer": "serving",
            "mode_policy": self.mode_policy,
            "snapshot_id": snapshot.snapshot_id,
            "latest_complete_trade_date": snapshot.latest_complete_trade_date,
            "latest_trade_date": snapshot.latest_trade_date,
            "fallback_used": False,
        }
        metadata.update(overrides)
        return metadata

    def _read_symbol_parquet(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return frame
        if "trade_date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        elif "date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        elif "Date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["Date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame = frame.copy()
            frame["symbol"] = frame["ts_code"].map(_normalize_symbol)
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.copy()
            frame["ts_code"] = frame["symbol"].map(_normalize_symbol)
        return frame

    def _filter_frame(
        self,
        frame: pd.DataFrame,
        *,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        result = frame.copy()
        if not result.empty and "trade_date" in result.columns:
            normalized = result["trade_date"].map(_normalize_trade_date)
            mask = normalized.str.len().eq(8)
            start = _normalize_trade_date(start_date)
            end = _normalize_trade_date(end_date)
            if start:
                mask &= normalized >= start
            if end:
                mask &= normalized <= end
            result = result.loc[mask].copy()
            result["trade_date"] = result["trade_date"].map(_normalize_trade_date)
        if columns is not None:
            wanted = [str(column) for column in columns if str(column)]
            available = [column for column in wanted if column in result.columns]
            result = result.loc[:, available].copy()
        return result.reset_index(drop=True)

    def _missing_symbol_result(
        self,
        *,
        snapshot: ParquetSnapshot,
        symbol: str,
        universe_key: str,
        category: str | None = None,
        resolver_trace: Mapping[str, Any] | None = None,
        path: str = "",
        message: str = "symbol not resolved to an existing Parquet serving file",
        issue_type: str = "missing_file",
    ) -> MarketDataReadResult:
        normalized = _normalize_symbol(symbol)
        trace = dict(resolver_trace or self.snapshot())
        issue = DataQualityIssue(
            path=path,
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            issue_type=issue_type,
            severity="error",
            message=message,
            resolver_strategy=str(trace.get("resolution_strategy") or "strict_parquet_serving"),
            metadata={"snapshot": trace},
        )
        self.issues.append(issue)
        return MarketDataReadResult(
            path=path,
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=trace,
            issues=[issue],
            metadata=self._metadata(snapshot, resolved=False),
        )

    def peek_symbol_latest_date(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
    ) -> str:
        path = self.resolve_symbol_path(symbol, universe_key=universe_key, category=category)
        if path is None:
            return ""
        try:
            frame = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=["date"])
            except Exception:
                try:
                    frame = pd.read_parquet(path, columns=["Date"])
                except Exception:
                    return ""
        if frame.empty:
            return ""
        date_column = (
            "trade_date"
            if "trade_date" in frame.columns
            else "date"
            if "date" in frame.columns
            else "Date"
            if "Date" in frame.columns
            else ""
        )
        if not date_column:
            return ""
        return max((_normalize_trade_date(value) for value in frame[date_column]), default="")

    def read_symbol_frame(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> MarketDataReadResult:
        snapshot = self._require_snapshot()
        normalized = _normalize_symbol(symbol)
        path = self.resolve_symbol_path(
            normalized,
            universe_key=universe_key,
            category=category,
        )
        resolver_trace = self.snapshot()
        if path is None:
            return self._missing_symbol_result(
                snapshot=snapshot,
                symbol=normalized,
                universe_key=universe_key,
                category=category,
                resolver_trace=resolver_trace,
            )
        try:
            frame = self._filter_frame(
                self._read_symbol_parquet(path),
                start_date=start_date,
                end_date=end_date,
                columns=columns,
            )
        except Exception as exc:
            issue = DataQualityIssue(
                path=str(path),
                symbol=normalized,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                issue_type="read_error",
                severity="error",
                message=str(exc),
                resolver_strategy="strict_parquet_serving",
                metadata={"snapshot": resolver_trace},
            )
            self.issues.append(issue)
            return MarketDataReadResult(
                path=str(path),
                symbol=normalized,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=resolver_trace,
                issues=[issue],
                metadata=self._metadata(snapshot, resolved=True, row_count=0),
            )
        return MarketDataReadResult(
            frame=frame,
            path=str(path),
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=resolver_trace,
            issues=[],
            metadata=self._metadata(snapshot, resolved=True, row_count=int(len(frame))),
        )

    def read_symbol_frames(
        self,
        symbols: Iterable[str],
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> dict[str, MarketDataReadResult]:
        snapshot = self._require_snapshot()
        normalized_symbols = _normalize_symbol_list(symbols)
        if not normalized_symbols:
            return {}

        resolver_trace = self.snapshot()
        resolver_trace["resolution_strategy"] = "strict_parquet_canonical_batch"
        try:
            batch_frame = self._read_dataset(
                snapshot.table_root,
                date_column="trade_date",
                date_range=(
                    start_date or "",
                    end_date or "",
                ) if start_date or end_date else None,
                columns=columns,
                symbol_filter=normalized_symbols,
                derive_symbol_column=False,
            )
        except Exception as exc:
            return {
                symbol: self._missing_symbol_result(
                    snapshot=snapshot,
                    symbol=symbol,
                    universe_key=universe_key,
                    category=category,
                    resolver_trace=resolver_trace,
                    path=str(snapshot.table_root),
                    message=f"canonical Parquet batch read failed: {exc}",
                    issue_type="read_error",
                )
                for symbol in normalized_symbols
            }

        results: dict[str, MarketDataReadResult] = {}
        symbol_column = (
            "symbol"
            if "symbol" in batch_frame.columns
            else "ts_code"
            if "ts_code" in batch_frame.columns
            else ""
        )
        frames_by_symbol: dict[str, pd.DataFrame] = {}
        if symbol_column and not batch_frame.empty:
            for raw_symbol, group in batch_frame.groupby(symbol_column, sort=False):
                normalized_group_symbol = _normalize_symbol(raw_symbol)
                if normalized_group_symbol:
                    frames_by_symbol[normalized_group_symbol] = group
        serving_symbols: set[str] | None = None
        for symbol in normalized_symbols:
            symbol_frame = frames_by_symbol.get(symbol)
            if symbol_frame is None:
                symbol_frame = pd.DataFrame()
            if symbol_frame.empty:
                if serving_symbols is None:
                    serving_symbols = set(self._serving_symbols(snapshot))
                if symbol not in serving_symbols:
                    results[symbol] = self._missing_symbol_result(
                        snapshot=snapshot,
                        symbol=symbol,
                        universe_key=universe_key,
                        category=category,
                        resolver_trace=resolver_trace,
                        path=str(snapshot.table_root),
                        message="symbol not found in canonical Parquet batch dataset",
                    )
                    continue
            if not symbol_frame.empty and "trade_date" in symbol_frame.columns:
                trade_dates = symbol_frame["trade_date"]
                if not trade_dates.is_monotonic_increasing:
                    symbol_frame = symbol_frame.sort_values("trade_date", kind="stable")
            if columns is not None:
                wanted = [str(column) for column in columns if str(column)]
                available = [column for column in wanted if column in symbol_frame.columns]
                if list(symbol_frame.columns) != available:
                    symbol_frame = symbol_frame.loc[:, available]
            symbol_frame = symbol_frame.reset_index(drop=True)
            results[symbol] = MarketDataReadResult(
                frame=symbol_frame,
                path=str(snapshot.table_root),
                symbol=symbol,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=resolver_trace,
                issues=[],
                metadata=self._metadata(
                    snapshot,
                    storage_layer="canonical_batch",
                    resolution_strategy="strict_parquet_canonical_batch",
                    resolved=True,
                    batch_read=True,
                    row_count=int(len(symbol_frame)),
                ),
            )
        return results

    def read_path(
        self,
        path: str | Path,
        *,
        symbol: str = "",
        category: str = "",
        universe_key: str = "",
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> MarketDataReadResult:
        snapshot = self._require_snapshot()
        parquet_path = Path(path)
        if parquet_path.suffix.lower() != ".parquet":
            issue = DataQualityIssue(
                path=str(parquet_path),
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                issue_type="unsupported_runtime_format",
                severity="error",
                message="strict market data reader does not read CSV runtime paths",
                resolver_strategy="strict_parquet_serving",
                metadata={"fallback_used": False},
            )
            self.issues.append(issue)
            return MarketDataReadResult(
                path=str(parquet_path),
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=self.snapshot(),
                issues=[issue],
                metadata=self._metadata(snapshot, resolved=False),
            )
        frame = self._filter_frame(
            self._read_symbol_parquet(parquet_path),
            start_date=start_date,
            end_date=end_date,
            columns=columns,
        )
        return MarketDataReadResult(
            frame=frame,
            path=str(parquet_path),
            symbol=str(symbol or ""),
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=self.snapshot(),
            issues=[],
            metadata=self._metadata(snapshot, resolved=True, row_count=int(len(frame))),
        )

    def latest_trade_date(self, universe_key: str = "full_a", category: str | None = None) -> str:
        snapshot = self._require_snapshot()
        return snapshot.latest_complete_trade_date or snapshot.latest_trade_date

    def _read_dataset(
        self,
        path: Path,
        *,
        date_column: str = "trade_date",
        as_of: str = "",
        date_range: tuple[str, str] | None = None,
        columns: Sequence[str] | None = None,
        symbol_filter: Sequence[str] | None = None,
        derive_symbol_column: bool = True,
    ) -> pd.DataFrame:
        read_columns = list(columns or [])
        if read_columns and date_column not in read_columns:
            read_columns.append(date_column)
        if read_columns and "ts_code" not in read_columns and "symbol" not in read_columns:
            read_columns.append("ts_code")
        normalized_symbols = _normalize_symbol_list(symbol_filter or [])
        if normalized_symbols and read_columns and "ts_code" not in read_columns and "symbol" not in read_columns:
            read_columns.append("ts_code")

        dataset_symbol_filter_applied = False
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            filter_expr = None
            normalized_as_of = _normalize_trade_date(as_of)
            if normalized_as_of:
                filter_expr = ds.field(date_column) == normalized_as_of
            elif date_range is not None:
                start = _normalize_trade_date(date_range[0])
                end = _normalize_trade_date(date_range[1])
                if start and end:
                    filter_expr = (ds.field(date_column) >= start) & (ds.field(date_column) <= end)
                elif start:
                    filter_expr = ds.field(date_column) >= start
                elif end:
                    filter_expr = ds.field(date_column) <= end
            if normalized_symbols:
                schema_names = set(getattr(dataset.schema, "names", []) or [])
                symbol_field = "ts_code" if "ts_code" in schema_names else "symbol" if "symbol" in schema_names else ""
                if symbol_field:
                    symbol_expr = ds.field(symbol_field).isin(normalized_symbols)
                    filter_expr = symbol_expr if filter_expr is None else filter_expr & symbol_expr
                    dataset_symbol_filter_applied = True
            table = dataset.to_table(
                columns=_dedupe(read_columns) if read_columns else None,
                filter=filter_expr,
            )
            frame = table.to_pandas()
        except Exception:
            files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
            frames = [pd.read_parquet(file) for file in files if file.exists()]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            start_date = date_range[0] if date_range else as_of
            end_date = date_range[1] if date_range else as_of
            frame = self._filter_frame(
                frame,
                start_date=start_date,
                end_date=end_date,
                columns=None,
            )

        if not frame.empty and date_column in frame.columns:
            frame = frame.copy()
            frame[date_column] = frame[date_column].map(_normalize_trade_date)
        if (
            derive_symbol_column
            and not frame.empty
            and "symbol" not in frame.columns
            and "ts_code" in frame.columns
        ):
            frame = frame.copy()
            frame["symbol"] = frame["ts_code"].map(_normalize_symbol)
        if normalized_symbols and "symbol" in frame.columns and not dataset_symbol_filter_applied:
            frame = frame[frame["symbol"].isin(set(normalized_symbols))].copy()
        if columns is not None:
            wanted = [str(column) for column in columns if str(column)]
            for derived in ("symbol",):
                if derived in frame.columns and derived not in wanted:
                    wanted.append(derived)
            available = [column for column in wanted if column in frame.columns]
            frame = frame.loc[:, available].copy()
        return frame.reset_index(drop=True)

    def read_cross_section(
        self,
        trade_date: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        snapshot = self._require_snapshot()
        frame = self._read_dataset(
            snapshot.table_root,
            date_column="trade_date",
            as_of=trade_date,
            columns=columns,
        )
        key = str(category or universe_key or "full_a").strip().lower()
        if key not in {"", "all", "full", "full_a", "all_a", "full_market"} and "symbol" in frame.columns:
            symbols = set(self.list_symbols(key))
            frame = frame[frame["symbol"].isin(symbols)].copy()
        return frame.reset_index(drop=True)

    def _load_catalog(self) -> dict[str, Any]:
        if self._catalog_payload is not None:
            return dict(self._catalog_payload)
        if not self.catalog_path.exists():
            self._catalog_payload = {}
            return {}
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except Exception:
            self._catalog_payload = {}
            return {}
        self._catalog_payload = dict(payload) if isinstance(payload, dict) else {}
        return dict(self._catalog_payload)

    def read_table(
        self,
        logical_table: str,
        *,
        as_of: str = "",
        date_range: tuple[str, str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._require_snapshot()
        key = str(logical_table or "").strip()
        catalog = self._load_catalog()
        tables = dict(catalog.get("tables", {}) or {})
        table_meta = tables.get(key)
        if not isinstance(table_meta, dict):
            raise MarketDataUnavailableError(f"Parquet logical table not found in catalog: {key}")
        path = self._resolve_data_path(
            table_meta.get("path") or table_meta.get("table_root"),
            self.parquet_market_root / key,
        )
        date_column = str(table_meta.get("date_column") or "trade_date")
        return self._read_dataset(
            path,
            date_column=date_column,
            as_of=as_of,
            date_range=date_range,
            columns=columns,
        )


__all__ = [
    "MarketDataReadResult",
    "MarketDataReader",
    "MarketDataUnavailableError",
]
