"""Parquet market data storage validation and materialization helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from quant_investor.market.market_data_reader import (
    MarketDataReader,
    MarketDataUnavailableError,
    _complete_coverage_blockers,
)


class MarketDataStore:
    """Validate and materialize local Parquet market-data layers."""

    def __init__(
        self,
        *,
        market: str = "CN",
        data_root: str | Path | None = None,
    ) -> None:
        self.market = str(market or "").strip().upper()
        self.data_root = Path(data_root or "data")
        self.reader = MarketDataReader(market=self.market, data_root=self.data_root)

    def validate_latest(self) -> dict[str, Any]:
        gate = self.reader.clean_snapshot_gate(refresh=True)
        blockers = list(gate.get("blockers", []) or [])
        macro_generation: dict[str, Any] = {}
        if self.market == "CN" and self.reader.catalog_path.exists():
            try:
                catalog_payload = json.loads(
                    self.reader.catalog_path.read_text(encoding="utf-8")
                )
            except Exception:
                catalog_payload = {}
            tables = (
                catalog_payload.get("tables", {})
                if isinstance(catalog_payload, Mapping)
                else {}
            )
            required_tables = (
                catalog_payload.get("required_tables", [])
                if isinstance(catalog_payload, Mapping)
                else []
            )
            macro_declared = (
                isinstance(tables, Mapping)
                and "macro_daily" in tables
            ) or (
                isinstance(required_tables, list)
                and "macro_daily" in required_tables
            )
            catalog_schema_version = str(
                catalog_payload.get("schema_version") or ""
            )
            if macro_declared and catalog_schema_version == "strict-parquet-catalog.v1":
                from quant_investor.market.macro_mart import (
                    MacroMartPromotionError,
                    read_macro_mart,
                )

                try:
                    _frame, macro_generation = read_macro_mart(
                        data_root=self.reader.parquet_market_root / "macro_daily"
                    )
                except (MacroMartPromotionError, OSError, ValueError) as exc:
                    blockers.append(str(exc) or "macro_catalog_generation_invalid")
            elif (
                macro_declared
                and catalog_schema_version == "myquant-cn-clean-catalog.v1"
            ):
                macro_generation = {
                    "status": "legacy_catalog_entry_not_v14_generation",
                    "catalog_schema_version": catalog_schema_version,
                    "production_eligible": False,
                    "branch_readiness": "blocked",
                    "blockers": ["macro_v14_generation_unavailable"],
                }
            elif macro_declared:
                blockers.append("macro_catalog_schema_invalid")
        blockers = list(dict.fromkeys(blockers))
        status = "passed" if gate.get("healthy") and not blockers else "failed"
        coverage: dict[str, Any] = {}
        try:
            latest_payload = self.reader._load_latest_payload(refresh=True)
            raw_coverage = latest_payload.get("coverage", {}) if isinstance(latest_payload, dict) else {}
            coverage = dict(raw_coverage or {}) if isinstance(raw_coverage, dict) else {}
        except Exception:
            coverage = {}
        latest_complete_trade_date = self._normalize_trade_date(
            gate.get("latest_complete_trade_date")
        )
        blockers.extend(
            str(item)
            for item in list(gate.get("coverage_provenance_blockers", []) or [])
            if str(item).strip()
        )
        blockers.extend(
            _complete_coverage_blockers(
                coverage,
                latest_complete_trade_date=latest_complete_trade_date,
            )
        )
        blockers = list(dict.fromkeys(str(item) for item in blockers if str(item).strip()))
        coverage_provenance_blockers = list(
            dict.fromkeys(
                str(item)
                for item in list(
                    gate.get("coverage_provenance_blockers", []) or []
                )
                if str(item).strip()
            )
        )
        status = "passed" if not blockers else "failed"
        return {
            "market": self.market,
            "status": status,
            "blockers": blockers,
            "snapshot_id": gate.get("snapshot_id", ""),
            "latest_complete_trade_date": gate.get("latest_complete_trade_date", ""),
            "latest_trade_date": gate.get("latest_trade_date", ""),
            "latest_pointer_path": gate.get("latest_pointer_path", ""),
            "table_root": gate.get("table_root", ""),
            "serving_root": gate.get("serving_root", ""),
            "manifest_path": gate.get("manifest_path", ""),
            "mode_policy": gate.get("mode_policy", "strict"),
            "coverage": coverage,
            "coverage_ratio": coverage.get("coverage_ratio"),
            "macro_generation": macro_generation,
            "coverage_provenance_blockers": coverage_provenance_blockers,
        }

    def _atomic_write_parquet(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_parquet(tmp_path, index=False)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        self._fsync_directory(path.parent)

    def _atomic_write_json(self, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode("utf-8")
        with tmp_path.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @contextmanager
    def _market_writer_lock(self) -> Iterator[None]:
        lock_path = (
            self.data_root
            / "parquet"
            / self.market.lower()
            / ".market_writer.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _normalize_trade_date(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "nat", "none"}:
            return ""
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:8] if len(digits) >= 8 else ""

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _normalize_bars_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "vol",
            "Amount": "amount",
            "Adj Close": "adj_close",
            "AdjClose": "adj_close",
        }
        if "trade_date" not in work.columns:
            if "Date" in work.columns:
                rename_map["Date"] = "trade_date"
            elif "date" in work.columns:
                rename_map["date"] = "trade_date"
        if "ts_code" not in work.columns:
            if "Symbol" in work.columns:
                rename_map["Symbol"] = "ts_code"
            elif "symbol" in work.columns:
                rename_map["symbol"] = "ts_code"
        work = work.rename(columns=rename_map)
        missing = [column for column in ["ts_code", "trade_date"] if column not in work.columns]
        if missing:
            raise ValueError(f"missing required bars columns: {missing}")
        work["ts_code"] = work["ts_code"].map(self._normalize_symbol)
        work["trade_date"] = work["trade_date"].map(self._normalize_trade_date)
        work = work.loc[work["ts_code"].ne("") & work["trade_date"].ne("")].copy()
        if work.empty:
            raise ValueError("no valid bars rows after symbol/date normalization")
        for column in [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "adj_factor",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "turnover_rate",
            "volume_ratio",
            "pe",
            "pb",
            "total_mv",
            "circ_mv",
        ]:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce")
        work = (
            work.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )
        work["_year"] = work["trade_date"].str.slice(0, 4).astype(int)
        work["_month"] = work["trade_date"].str.slice(4, 6).astype(int)
        return work

    def _latest_dates_from_bars(self, frame: pd.DataFrame) -> tuple[str, str]:
        dates = sorted(
            {
                self._normalize_trade_date(value)
                for value in frame.get("trade_date", pd.Series(dtype=str)).tolist()
                if self._normalize_trade_date(value)
            }
        )
        latest = dates[-1] if dates else ""
        return latest, latest

    def _validate_adj_factor(self, frame: pd.DataFrame) -> None:
        if "adj_factor" not in frame.columns:
            raise ValueError("adj_factor is required for parquet-direct bars upsert")
        values = pd.to_numeric(frame["adj_factor"], errors="coerce")
        if values.isna().any() or not (values > 0).all():
            raise ValueError("adj_factor must be present and positive for parquet-direct bars upsert")

    def _merge_bars(self, existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if existing is not None and not existing.empty:
            frames.append(self._normalize_bars_frame(existing))
        if incoming is not None and not incoming.empty:
            frames.append(incoming.copy())
        if not frames:
            raise ValueError("no bars rows to merge")
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged = (
            merged.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )
        merged["_year"] = merged["trade_date"].str.slice(0, 4).astype(int)
        merged["_month"] = merged["trade_date"].str.slice(4, 6).astype(int)
        return merged

    def _copytree_hardlink_or_copy(self, source: Path, target: Path) -> None:
        if not source.exists():
            raise ValueError(f"missing source directory: {source}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _copy_file(src: str, dst: str) -> str:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            return dst

        shutil.copytree(source, target, copy_function=_copy_file)

    def _replace_directories(
        self,
        replacements: list[tuple[Path, Path]],
        *,
        after_replace: Callable[[], None] | None = None,
    ) -> None:
        backups: list[tuple[Path, Path]] = []
        moved: list[Path] = []
        try:
            for source, target in replacements:
                if not source.exists():
                    raise ValueError(f"missing staged directory: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = target.with_name(f".{target.name}.previous")
                if backup.exists():
                    shutil.rmtree(backup)
                if target.exists():
                    shutil.move(str(target), str(backup))
                    backups.append((target, backup))
                shutil.move(str(source), str(target))
                moved.append(target)
            if after_replace is not None:
                after_replace()
        except Exception:
            for target in reversed(moved):
                if target.exists():
                    shutil.rmtree(target)
            for target, backup in reversed(backups):
                if backup.exists():
                    shutil.move(str(backup), str(target))
            raise
        else:
            for _target, backup in backups:
                if backup.exists():
                    shutil.rmtree(backup)

    def append_health_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        health_path = self.data_root / "parquet" / self.market.lower() / "_health_ledger.jsonl"
        health_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_type": str(event_type or ""),
            "generated_at": self._utc_now(),
            "market": self.market,
            "snapshot_id": str(payload.get("snapshot_id") or ""),
            "status": str(payload.get("status") or ""),
            "payload": dict(payload),
        }
        with health_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def upsert_bars(
        self,
        frame: pd.DataFrame,
        *,
        target_trade_date: str,
        target_trade_dates: Iterable[str] | None = None,
        source: str,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_latest_pointer_sha256: str = "",
    ) -> dict[str, Any]:
        """Publish one immutable bars snapshot behind a locked pointer CAS."""

        with self._market_writer_lock():
            return self._upsert_bars_locked(
                frame,
                target_trade_date=target_trade_date,
                target_trade_dates=target_trade_dates,
                source=source,
                snapshot_id=snapshot_id,
                metadata=metadata,
                expected_latest_pointer_sha256=expected_latest_pointer_sha256,
            )

    def _upsert_bars_locked(
        self,
        frame: pd.DataFrame,
        *,
        target_trade_date: str,
        target_trade_dates: Iterable[str] | None = None,
        source: str,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_latest_pointer_sha256: str = "",
    ) -> dict[str, Any]:
        target_date = self._normalize_trade_date(target_trade_date)
        if not target_date:
            raise ValueError("target_trade_date is required for bars upsert")
        incoming = self._normalize_bars_frame(frame)
        incoming_dates = set(incoming["trade_date"].astype(str))
        declared_target_dates = {
            self._normalize_trade_date(value)
            for value in (target_trade_dates or [target_date])
            if self._normalize_trade_date(value)
        }
        if not declared_target_dates or target_date != max(declared_target_dates):
            raise ValueError(
                "target_trade_date must equal the latest declared target date"
            )
        if incoming_dates != declared_target_dates:
            raise ValueError(
                "upsert_bars incoming dates must equal declared target dates: "
                f"{sorted(incoming_dates)} != {sorted(declared_target_dates)}"
            )
        self._validate_adj_factor(incoming)

        expected_pointer_sha256 = str(
            expected_latest_pointer_sha256 or ""
        ).strip().lower()
        if expected_pointer_sha256 and (
            len(expected_pointer_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_pointer_sha256
            )
        ):
            raise ValueError("expected_latest_pointer_sha256_invalid")
        if expected_pointer_sha256:
            actual_pointer_sha256 = self._file_sha256(
                self.reader.latest_pointer_path
            )
            if actual_pointer_sha256 != expected_pointer_sha256:
                raise ValueError(
                    "market_pointer_cas_mismatch:"
                    f"{actual_pointer_sha256}!={expected_pointer_sha256}"
                )

        gate = self.reader.clean_snapshot_gate(refresh=True)
        if not gate.get("healthy"):
            blockers = "; ".join(str(item) for item in gate.get("blockers", []) if str(item).strip())
            raise ValueError(blockers or "cannot upsert without a healthy latest Parquet snapshot")
        previous_latest_payload = self.reader._load_latest_payload(refresh=True)
        snapshot = self.reader._snapshot_from_payload(previous_latest_payload)
        previous_coverage = previous_latest_payload.get("coverage", {})
        if not isinstance(previous_coverage, dict):
            previous_coverage = {}
        previous_coverage = dict(previous_coverage)
        previous_coverage_provenance_blockers = [
            str(item)
            for item in list(gate.get("coverage_provenance_blockers", []) or [])
            if str(item).strip()
        ]
        try:
            previous_snapshot_manifest = json.loads(
                snapshot.manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            previous_snapshot_manifest = {}
        if (
            isinstance(previous_snapshot_manifest, dict)
            and previous_snapshot_manifest.get("historical_scope_hash_backfilled")
            is True
        ):
            previous_coverage_provenance_blockers.append(
                "coverage_scope_hash_backfilled_from_historical_target"
            )
        previous_coverage_provenance_blockers = list(
            dict.fromkeys(previous_coverage_provenance_blockers)
        )

        historical_target = bool(
            snapshot.latest_complete_trade_date
            and target_date < snapshot.latest_complete_trade_date
        )
        if historical_target:
            latest_coverage_blockers = list(previous_coverage_provenance_blockers)
            if previous_coverage.get("complete") is not True:
                latest_coverage_blockers.append("coverage_complete_claim_missing")
            latest_coverage_blockers.extend(
                _complete_coverage_blockers(
                    previous_coverage,
                    latest_complete_trade_date=snapshot.latest_complete_trade_date,
                )
            )
            latest_coverage_blockers = list(
                dict.fromkeys(
                    str(item)
                    for item in latest_coverage_blockers
                    if str(item).strip()
                )
            )
            if latest_coverage_blockers:
                raise ValueError(
                    "historical_upsert_requires_verified_latest_coverage:"
                    + ",".join(latest_coverage_blockers)
                )

        resolved_snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging_base = self.data_root / "parquet_staging" / self.market.lower() / resolved_snapshot_id
        staged_table = staging_base / "table" / "bars"
        staged_serving = staging_base / "serving" / "bars"
        if staging_base.exists():
            shutil.rmtree(staging_base)
        try:
            self._copytree_hardlink_or_copy(snapshot.table_root, staged_table)
            self._copytree_hardlink_or_copy(snapshot.serving_root, staged_serving)

            for (year, month), month_incoming in incoming.groupby(["_year", "_month"], sort=True):
                month_dir = staged_table / f"year={int(year)}" / f"month={int(month):02d}"
                month_path = month_dir / "part.parquet"
                existing = pd.read_parquet(month_path) if month_path.exists() else pd.DataFrame()
                merged = self._merge_bars(existing, month_incoming)
                self._atomic_write_parquet(merged.drop(columns=["_year", "_month"]), month_path)

            for symbol, symbol_incoming in incoming.groupby("ts_code", sort=True):
                symbol_dir = staged_serving / f"symbol={symbol}"
                symbol_path = symbol_dir / "bars.parquet"
                existing = pd.read_parquet(symbol_path) if symbol_path.exists() else pd.DataFrame()
                merged = self._merge_bars(existing, symbol_incoming)
                self._atomic_write_parquet(merged.drop(columns=["_year", "_month"]), symbol_path)

            table_files = sorted(staged_table.rglob("*.parquet"))
            serving_files = sorted(staged_serving.rglob("*.parquet"))
            if not table_files or not serving_files:
                raise ValueError("staged upsert produced no parquet files")
            table_rows = sum(len(pd.read_parquet(path)) for path in table_files)
            serving_rows = sum(len(pd.read_parquet(path)) for path in serving_files)
            if table_rows != serving_rows:
                raise ValueError(f"table/serving row mismatch after upsert: {table_rows} != {serving_rows}")

            snapshot_manifest_dir = self.data_root / "parquet" / self.market.lower() / "_snapshots"
            snapshot_manifest_path = snapshot_manifest_dir / f"{resolved_snapshot_id}.json"
            snapshot_payload_dir = snapshot_manifest_dir / resolved_snapshot_id
            published_table_root = snapshot_payload_dir / "table" / "bars"
            published_serving_root = snapshot_payload_dir / "serving" / "bars"
            if snapshot_manifest_path.exists() or snapshot_payload_dir.exists():
                raise ValueError(
                    "immutable_snapshot_generation_already_exists:"
                    f"{resolved_snapshot_id}"
                )
            parquet_size = sum(path.stat().st_size for path in table_files)
            latest_available = self._normalize_trade_date(
                (metadata or {}).get("latest_available_trade_date") or target_date
            )
            latest_complete = self._normalize_trade_date(
                (metadata or {}).get("latest_complete_trade_date") or latest_available
            )
            full_frame = pd.concat((pd.read_parquet(path) for path in table_files), ignore_index=True)
            coverage = (metadata or {}).get("coverage")
            if not isinstance(coverage, dict):
                coverage = {
                    "latest_available_trade_date": latest_available,
                    "latest_complete_trade_date": latest_complete,
                    "row_count": int(len(full_frame)),
                    "symbol_count": int(full_frame["ts_code"].nunique()) if "ts_code" in full_frame.columns else 0,
                }
            else:
                coverage = dict(coverage)
            coverage.setdefault("coverage_trade_date", target_date)
            incoming_target_coverage = dict(coverage)
            historical_coverage_preserved = False
            previous_coverage_date = self._normalize_trade_date(
                previous_coverage.get("coverage_trade_date")
                or previous_coverage.get("latest_complete_trade_date")
                or snapshot.latest_complete_trade_date
            )
            incoming_coverage_date = self._normalize_trade_date(
                coverage.get("coverage_trade_date") or target_date
            )
            if (
                previous_coverage
                and not previous_coverage_provenance_blockers
                and previous_coverage_date
                and incoming_coverage_date
                and incoming_coverage_date < previous_coverage_date
            ):
                coverage = previous_coverage
                historical_coverage_preserved = True
            manifest = {
                "snapshot_id": resolved_snapshot_id,
                "market": self.market,
                "status": "OK",
                "source": str(source or ""),
                "row_count": int(len(full_frame)),
                "symbol_count": int(full_frame["ts_code"].nunique()) if "ts_code" in full_frame.columns else 0,
                "latest_trade_date": latest_complete,
                "latest_available_trade_date": latest_available,
                "latest_complete_trade_date": latest_complete,
                "table_root": str(published_table_root),
                "derived_serving_root": str(published_serving_root),
                "manifest_path": str(snapshot_manifest_path),
                "readback_validated": True,
                "parquet_size_bytes": int(parquet_size),
                "quarantined_tail_dates": list((metadata or {}).get("quarantined_tail_dates") or []),
                "coverage": coverage,
                "historical_upsert_coverage_preserved": historical_coverage_preserved,
                "historical_upsert_target_coverage": (
                    incoming_target_coverage if historical_coverage_preserved else {}
                ),
                "previous_coverage_provenance_blockers": (
                    previous_coverage_provenance_blockers
                ),
                "blockers": list((metadata or {}).get("blockers") or []),
                "metadata": {
                    **dict(metadata or {}),
                    "previous_snapshot_id": snapshot.snapshot_id,
                    "expected_previous_latest_pointer_sha256": (
                        expected_pointer_sha256
                    ),
                    "upsert_target_trade_date": target_date,
                    "upsert_target_trade_dates": sorted(declared_target_dates),
                    "upsert_affected_symbols": sorted(set(incoming["ts_code"].astype(str))),
                },
            }
            latest_payload = {
                "snapshot_id": resolved_snapshot_id,
                "status": "OK",
                "manifest_path": str(snapshot_manifest_path),
                "table_root": str(published_table_root),
                "derived_serving_root": str(published_serving_root),
                "latest_available_trade_date": latest_available,
                "latest_complete_trade_date": latest_complete,
                "latest_trade_date": latest_complete,
                "quarantined_tail_dates": manifest["quarantined_tail_dates"],
                "coverage": coverage,
                "blockers": [],
                "updated_at": self._utc_now(),
            }
            self._copytree_hardlink_or_copy(staged_table, published_table_root)
            self._copytree_hardlink_or_copy(
                staged_serving,
                published_serving_root,
            )
            self._atomic_write_json(manifest, snapshot_manifest_path)

            def _write_pointer_and_validate() -> None:
                pointer_written = False
                try:
                    if expected_pointer_sha256:
                        actual_pointer_sha256 = self._file_sha256(
                            self.reader.latest_pointer_path
                        )
                        if actual_pointer_sha256 != expected_pointer_sha256:
                            raise ValueError(
                                "market_pointer_cas_mismatch:"
                                f"{actual_pointer_sha256}!="
                                f"{expected_pointer_sha256}"
                            )
                    self._atomic_write_json(
                        latest_payload,
                        self.reader.latest_pointer_path,
                    )
                    pointer_written = True
                    self.reader._latest_payload = None
                    self.reader._snapshot_gate_cache = None
                    self.reader._serving_symbols_cache = None
                    validation = self.validate_latest()
                    if validation.get("status") != "passed":
                        raise ValueError(
                            "post_commit_storage_validation_failed:"
                            + ",".join(
                                str(item)
                                for item in list(validation.get("blockers", []) or [])
                                if str(item).strip()
                            )
                        )
                except Exception:
                    if pointer_written:
                        self._atomic_write_json(
                            dict(previous_latest_payload),
                            self.reader.latest_pointer_path,
                        )
                    self.reader._latest_payload = None
                    self.reader._snapshot_gate_cache = None
                    self.reader._serving_symbols_cache = None
                    raise

            _write_pointer_and_validate()
            self.reader._latest_payload = None
            self.reader._snapshot_gate_cache = None
            self.reader._serving_symbols_cache = None
            return manifest
        finally:
            if staging_base.exists():
                shutil.rmtree(staging_base)

    def write_full_history_bars(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_bars_frame(frame)
        if self.market == "CN":
            self._validate_adj_factor(normalized)
        resolved_snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        table_root = self.data_root / "parquet" / self.market.lower() / "bars"
        serving_root = self.data_root / "parquet_serving" / self.market.lower() / "bars"
        manifest_path = self.data_root / "parquet" / self.market.lower() / "_snapshots" / f"{resolved_snapshot_id}.json"

        for (year, month), month_frame in normalized.groupby(["_year", "_month"], sort=True):
            month_path = table_root / f"year={int(year)}" / f"month={int(month):02d}" / "part.parquet"
            existing = pd.read_parquet(month_path) if month_path.exists() else pd.DataFrame()
            merged = self._merge_bars(existing, month_frame)
            self._atomic_write_parquet(merged.drop(columns=["_year", "_month"]), month_path)

        for symbol, symbol_frame in normalized.groupby("ts_code", sort=True):
            normalized_symbol = self._normalize_symbol(symbol)
            if not normalized_symbol:
                continue
            symbol_path = serving_root / f"symbol={normalized_symbol}" / "bars.parquet"
            existing = pd.read_parquet(symbol_path) if symbol_path.exists() else pd.DataFrame()
            merged = self._merge_bars(existing, symbol_frame)
            self._atomic_write_parquet(merged.drop(columns=["_year", "_month"]), symbol_path)

        table_files = sorted(table_root.rglob("*.parquet"))
        serving_files = sorted(serving_root.rglob("*.parquet"))
        if not table_files or not serving_files:
            raise ValueError("full-history write produced no parquet files")
        full_frame = pd.concat((pd.read_parquet(path) for path in table_files), ignore_index=True)
        latest_available, latest_complete = self._latest_dates_from_bars(full_frame)
        parquet_size = sum(path.stat().st_size for path in table_files)
        coverage = {
            "latest_available_trade_date": latest_available,
            "latest_complete_trade_date": latest_complete,
            "row_count": int(len(full_frame)),
            "symbol_count": int(full_frame["ts_code"].nunique()) if "ts_code" in full_frame.columns else 0,
        }
        manifest = {
            "snapshot_id": resolved_snapshot_id,
            "market": self.market,
            "status": "OK",
            "source": str(source or ""),
            "row_count": int(len(full_frame)),
            "symbol_count": int(full_frame["ts_code"].nunique()) if "ts_code" in full_frame.columns else 0,
            "latest_trade_date": latest_complete,
            "latest_available_trade_date": latest_available,
            "latest_complete_trade_date": latest_complete,
            "table_root": str(table_root),
            "derived_serving_root": str(serving_root),
            "manifest_path": str(manifest_path),
            "readback_validated": True,
            "parquet_size_bytes": int(parquet_size),
            "coverage": coverage,
            "blockers": [],
            "metadata": dict(metadata or {}),
        }
        latest_payload = {
            "snapshot_id": resolved_snapshot_id,
            "status": "OK",
            "manifest_path": str(manifest_path),
            "table_root": str(table_root),
            "derived_serving_root": str(serving_root),
            "latest_available_trade_date": latest_available,
            "latest_complete_trade_date": latest_complete,
            "latest_trade_date": latest_complete,
            "coverage": coverage,
            "blockers": [],
            "updated_at": self._utc_now(),
        }
        self._atomic_write_json(manifest, manifest_path)
        self._atomic_write_json(latest_payload, self.reader.latest_pointer_path)
        self.reader._latest_payload = None
        self.reader._snapshot_gate_cache = None
        self.reader._serving_symbols_cache = None
        return manifest

    def materialize_cross_section(
        self,
        *,
        trade_date: str,
        universe_key: str = "full_a",
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        frame = self.reader.read_cross_section(
            trade_date,
            universe_key=universe_key,
            columns=columns,
        )
        target_dir = (
            self.data_root
            / "parquet_cache"
            / self.market.lower()
            / "daily_cross_section"
            / f"trade_date={trade_date}"
        )
        target_path = target_dir / "part.parquet"
        self._atomic_write_parquet(frame, target_path)
        meta_path = target_dir / "manifest.json"
        snapshot = self.reader.snapshot()
        self._atomic_write_json(
            {
                "schema_version": "myquant-daily-cross-section-cache.v1",
                "market": self.market,
                "trade_date": str(trade_date),
                "universe_key": str(universe_key),
                "row_count": int(len(frame)),
                "snapshot_id": snapshot.get("snapshot_id", ""),
                "source_latest_pointer": snapshot.get("latest_pointer_path", ""),
                "path": str(target_path),
            },
            meta_path,
        )
        return {
            "status": "materialized",
            "market": self.market,
            "trade_date": str(trade_date),
            "universe_key": str(universe_key),
            "row_count": int(len(frame)),
            "path": str(target_path),
            "manifest_path": str(meta_path),
        }

    def materialize_serving(self) -> dict[str, Any]:
        snapshot = self.reader._require_snapshot()
        frame = self.reader._read_dataset(snapshot.table_root, date_column="trade_date")
        if frame.empty or "ts_code" not in frame.columns:
            raise MarketDataUnavailableError("canonical bars table cannot materialize serving without ts_code rows")
        row_count = 0
        symbol_count = 0
        for symbol, group in frame.groupby(frame["ts_code"].astype(str).str.upper(), sort=True):
            normalized = str(symbol or "").strip().upper()
            if not normalized:
                continue
            target = snapshot.serving_root / f"symbol={normalized}" / "bars.parquet"
            self._atomic_write_parquet(group.sort_values("trade_date").reset_index(drop=True), target)
            row_count += int(len(group))
            symbol_count += 1
        return {
            "status": "materialized",
            "market": self.market,
            "snapshot_id": snapshot.snapshot_id,
            "symbol_count": symbol_count,
            "row_count": row_count,
            "serving_root": str(snapshot.serving_root),
        }

    def materialize_features(
        self,
        *,
        trade_date: str,
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        cross_section = self.materialize_cross_section(
            trade_date=trade_date,
            universe_key="full_a",
            columns=columns,
        )
        return {
            "status": "materialized",
            "market": self.market,
            "trade_date": str(trade_date),
            "daily_cross_section": cross_section,
        }

    def storage_diff(self) -> dict[str, Any]:
        validation = self.validate_latest()
        if validation["status"] != "passed":
            return {
                "market": self.market,
                "status": "failed",
                "validation": validation,
                "diff": {},
            }
        snapshot = self.reader.snapshot()
        serving_symbols = self.reader.list_symbols("full_a")
        coverage = {}
        try:
            latest_payload = self.reader._load_latest_payload()
            coverage = dict(latest_payload.get("coverage", {}) or {})
        except Exception:
            coverage = {}
        expected_symbol_count = int(coverage.get("symbol_count", 0) or 0)
        actual_symbol_count = len(serving_symbols)
        return {
            "market": self.market,
            "status": "passed" if not expected_symbol_count or expected_symbol_count == actual_symbol_count else "diff",
            "snapshot_id": snapshot.get("snapshot_id", ""),
            "diff": {
                "coverage_symbol_count": expected_symbol_count,
                "serving_symbol_count": actual_symbol_count,
                "symbol_count_delta": actual_symbol_count - expected_symbol_count,
                "latest_complete_trade_date": snapshot.get("latest_complete_trade_date", ""),
            },
        }


def run_storage_validate(*, market: str = "CN", data_root: str | Path | None = None) -> dict[str, Any]:
    return MarketDataStore(market=market, data_root=data_root).validate_latest()


def _bounded_files(root: Path, pattern: str, *, limit: int) -> tuple[list[Path], bool]:
    if not root.exists():
        return [], False
    files: list[Path] = []
    truncated = False
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        if len(files) >= limit:
            truncated = True
            break
        files.append(path)
    return files, truncated


def _validate_clean_root(
    *,
    name: str,
    root: Path,
    data_root: Path,
    json_required: bool,
    sample_limit: int,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    payload: dict[str, Any] = {
        "name": name,
        "path": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "sample_limit": int(sample_limit),
        "sample_files": [],
        "sample_file_count": 0,
        "truncated": False,
        "json_validated_count": 0,
        "invalid_json_count": 0,
    }
    if not root.exists() or not root.is_dir():
        blockers.append(f"{name} root missing: {root}")
        return payload, blockers

    all_samples: list[Path] = []
    truncated_any = False
    for pattern in ("*.json", "*.csv", "*.parquet", "*.md"):
        remaining = max(0, sample_limit - len(all_samples))
        if remaining <= 0:
            truncated_any = True
            break
        found, truncated = _bounded_files(root, pattern, limit=remaining)
        all_samples.extend(found)
        truncated_any = bool(truncated_any or truncated)

    payload["sample_files"] = [
        path.relative_to(data_root).as_posix()
        if path.is_relative_to(data_root)
        else str(path)
        for path in all_samples[:sample_limit]
    ]
    payload["sample_file_count"] = len(all_samples[:sample_limit])
    payload["truncated"] = truncated_any
    if not all_samples:
        blockers.append(f"{name} root has no bounded sample files: {root}")

    json_files, json_truncated = _bounded_files(root, "*.json", limit=sample_limit)
    payload["json_truncated"] = json_truncated
    for json_path in json_files:
        try:
            json.loads(json_path.read_text(encoding="utf-8"))
            payload["json_validated_count"] += 1
        except Exception:
            payload["invalid_json_count"] += 1
    if payload["invalid_json_count"]:
        blockers.append(f"{name} root has invalid JSON lineage files")
    if json_required and not payload["json_validated_count"]:
        blockers.append(f"{name} root has no valid JSON lineage sample")
    return payload, blockers


def run_storage_validate_clean(
    *,
    market: str = "CN",
    data_root: str | Path | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    scoped_market = str(market or "").strip().upper()
    if scoped_market != "CN":
        return {
            "market": scoped_market,
            "status": "failed",
            "blockers": ["storage-validate-clean currently supports CN only"],
            "local_read_only": True,
            "roots": {},
        }
    scoped_root = Path(data_root or "data")
    roots_to_check = {
        "clean": {
            "path": scoped_root / "clean",
            "json_required": False,
        },
        "factor_readiness": {
            "path": scoped_root / "factor_readiness" / "tushare",
            "json_required": True,
        },
        "cleaning_reports": {
            "path": scoped_root / "cleaning_reports" / "tushare",
            "json_required": True,
        },
    }
    roots: dict[str, Any] = {}
    blockers: list[str] = []
    bounded_limit = max(1, int(sample_limit or 20))
    for name, config in roots_to_check.items():
        payload, root_blockers = _validate_clean_root(
            name=name,
            root=Path(config["path"]),
            data_root=scoped_root,
            json_required=bool(config["json_required"]),
            sample_limit=bounded_limit,
        )
        roots[name] = payload
        blockers.extend(root_blockers)
    return {
        "market": scoped_market,
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "local_read_only": True,
        "schema_version": "myquant-clean-storage-validate.v1",
        "data_root": str(scoped_root),
        "sample_limit": bounded_limit,
        "roots": roots,
    }


def run_materialize_serving(*, market: str = "CN", data_root: str | Path | None = None) -> dict[str, Any]:
    return MarketDataStore(market=market, data_root=data_root).materialize_serving()


def run_materialize_features(
    *,
    market: str = "CN",
    trade_date: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    return MarketDataStore(market=market, data_root=data_root).materialize_features(trade_date=trade_date)


def run_storage_diff(*, market: str = "CN", data_root: str | Path | None = None) -> dict[str, Any]:
    return MarketDataStore(market=market, data_root=data_root).storage_diff()


__all__ = [
    "MarketDataStore",
    "run_materialize_features",
    "run_materialize_serving",
    "run_storage_diff",
    "run_storage_validate",
    "run_storage_validate_clean",
]
