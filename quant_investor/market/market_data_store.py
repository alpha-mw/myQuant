"""Parquet market data storage validation and materialization helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
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


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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

    @staticmethod
    def _lexical_absolute(path: str | Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def _uses_repository_data_root(self) -> bool:
        return self._lexical_absolute(self.data_root) == self._lexical_absolute(
            _REPOSITORY_ROOT / "data"
        )

    def _sealed_recovery_path(self, path: str | Path, *, label: str) -> str:
        """Return repository-relative data paths for durable recovery evidence."""

        if not self._uses_repository_data_root():
            return str(path)
        repository_root = self._lexical_absolute(_REPOSITORY_ROOT)
        target = self._lexical_absolute(path)
        try:
            relative = target.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"{label}_outside_repository_data_root") from exc
        if not relative.parts or relative.parts[0] != "data":
            raise ValueError(f"{label}_outside_repository_data_root")
        return relative.as_posix()

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
                    detail = str(exc) or "macro_catalog_generation_invalid"
                    if detail == "macro_generation_manifest_schema_invalid":
                        blockers.append("macro_v15_generation_unavailable")
                    blockers.append(detail)
            elif (
                macro_declared
                and catalog_schema_version == "myquant-cn-clean-catalog.v1"
            ):
                macro_generation = {
                    "status": "legacy_catalog_entry_not_v15_generation",
                    "catalog_schema_version": catalog_schema_version,
                    "production_eligible": False,
                    "branch_readiness": "blocked",
                    "blockers": ["macro_v15_generation_unavailable"],
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
        self._atomic_write_bytes(self._json_bytes(payload), path)

    @staticmethod
    def _json_bytes(payload: Mapping[str, Any]) -> bytes:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode("utf-8")

    def _atomic_write_bytes(self, encoded: bytes, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
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

    @staticmethod
    def _canonical_json_bytes(payload: Any) -> bytes:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _valid_sha256(value: Any) -> str:
        digest = str(value or "").strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("sha256_invalid")
        return digest

    @staticmethod
    def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    def _read_fd_stable_bytes(self, path: Path, *, label: str) -> bytes:
        try:
            before = os.lstat(path)
        except OSError as exc:
            raise ValueError(f"{label}_missing") from exc
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"{label}_symlink_rejected")
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}_regular_file_required")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            opened_signature = self._stat_signature(opened)
            if self._stat_signature(before) != opened_signature:
                raise ValueError(f"{label}_identity_changed_before_read")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_opened = os.fstat(descriptor)
            try:
                after_path = os.lstat(path)
            except OSError as exc:
                raise ValueError(f"{label}_path_replaced_during_read") from exc
            if (
                self._stat_signature(after_opened) != opened_signature
                or self._stat_signature(after_path) != opened_signature
            ):
                raise ValueError(f"{label}_changed_during_read")
            return b"".join(chunks)
        except OSError as exc:
            raise ValueError(f"{label}_open_failed") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _read_fd_stable_json(self, path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
        encoded = self._read_fd_stable_bytes(path, label=label)
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}_json_invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label}_json_object_required")
        return dict(payload), encoded

    def _write_new_bytes(self, encoded: bytes, path: Path, *, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"{label}_already_exists") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        self._fsync_directory(path.parent)
        if self._read_fd_stable_bytes(path, label=label) != encoded:
            raise ValueError(f"{label}_readback_mismatch")

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
        previous_pointer_bytes = self.reader.latest_pointer_path.read_bytes()
        previous_pointer_sha256 = hashlib.sha256(
            previous_pointer_bytes
        ).hexdigest()
        if (
            expected_pointer_sha256
            and previous_pointer_sha256 != expected_pointer_sha256
        ):
            raise ValueError(
                "market_pointer_cas_mismatch:"
                f"{previous_pointer_sha256}!={expected_pointer_sha256}"
            )

        gate = self.reader.clean_snapshot_gate(refresh=True)
        if not gate.get("healthy"):
            blockers = "; ".join(str(item) for item in gate.get("blockers", []) if str(item).strip())
            raise ValueError(blockers or "cannot upsert without a healthy latest Parquet snapshot")
        preflight_pointer_sha256 = self._file_sha256(
            self.reader.latest_pointer_path
        )
        if preflight_pointer_sha256 != previous_pointer_sha256:
            raise ValueError(
                "market_pointer_cas_mismatch:"
                f"{preflight_pointer_sha256}!={previous_pointer_sha256}"
            )
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
        if (
            not resolved_snapshot_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in resolved_snapshot_id
            )
        ):
            raise ValueError("snapshot_id_invalid")
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
                latest_pointer_bytes = self._json_bytes(latest_payload)
                latest_pointer_sha256 = hashlib.sha256(
                    latest_pointer_bytes
                ).hexdigest()
                try:
                    actual_pointer_sha256 = self._file_sha256(
                        self.reader.latest_pointer_path
                    )
                    if actual_pointer_sha256 != previous_pointer_sha256:
                        raise ValueError(
                            "market_pointer_cas_mismatch:"
                            f"{actual_pointer_sha256}!="
                            f"{previous_pointer_sha256}"
                        )
                    self._atomic_write_json(
                        latest_payload,
                        self.reader.latest_pointer_path,
                    )
                    self.reader._latest_payload = None
                    self.reader._snapshot_gate_cache = None
                    self.reader._serving_symbols_cache = None
                    validation = self.reader.clean_snapshot_gate(refresh=True)
                    bars_blockers = [
                        str(item)
                        for item in list(validation.get("blockers", []) or [])
                        if str(item).strip()
                    ]
                    if str(validation.get("snapshot_id") or "") != (
                        resolved_snapshot_id
                    ):
                        bars_blockers.append(
                            "post_commit_snapshot_id_mismatch"
                        )
                    if self._normalize_trade_date(
                        validation.get("latest_complete_trade_date")
                    ) != latest_complete:
                        bars_blockers.append(
                            "post_commit_latest_complete_trade_date_mismatch"
                        )
                    for field_name, actual, expected in (
                        (
                            "table_root",
                            validation.get("table_root"),
                            published_table_root,
                        ),
                        (
                            "serving_root",
                            validation.get("serving_root"),
                            published_serving_root,
                        ),
                        (
                            "manifest_path",
                            validation.get("manifest_path"),
                            snapshot_manifest_path,
                        ),
                    ):
                        try:
                            same_path = Path(str(actual or "")).resolve(
                                strict=True
                            ) == expected.resolve(strict=True)
                        except (OSError, RuntimeError):
                            same_path = False
                        if not same_path:
                            bars_blockers.append(
                                f"post_commit_{field_name}_mismatch"
                            )
                    if not validation.get("healthy") or bars_blockers:
                        raise ValueError(
                            "post_commit_bars_validation_failed:"
                            + ",".join(dict.fromkeys(bars_blockers))
                        )
                except Exception:
                    try:
                        current_pointer_bytes = (
                            self.reader.latest_pointer_path.read_bytes()
                        )
                    except OSError:
                        current_pointer_bytes = b""
                    if (
                        current_pointer_bytes == latest_pointer_bytes
                        and hashlib.sha256(current_pointer_bytes).hexdigest()
                        == latest_pointer_sha256
                    ):
                        self._atomic_write_bytes(
                            previous_pointer_bytes,
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

    def _snapshot_file_inventory(
        self,
        root: Path,
        *,
        label: str,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        def _is_orphaned_atomic_parquet_temp(path: Path) -> bool:
            prefix, separator, process_id = path.name.rpartition(".tmp-")
            return bool(
                separator
                and process_id.isdigit()
                and prefix.startswith(".")
                and prefix[1:].endswith(".parquet")
            )

        try:
            self.reader._assert_path_has_no_symlink(
                root,
                boundary=self.data_root,
                label=label,
            )
        except MarketDataUnavailableError as exc:
            raise ValueError(f"{label}_path_invalid:{exc}") from exc
        if not root.exists() or not root.is_dir():
            raise ValueError(f"{label}_root_missing")

        paths: list[Path] = []
        residual_paths: list[Path] = []
        for directory, directory_names, file_names in os.walk(
            root,
            followlinks=False,
        ):
            current = Path(directory)
            for name in directory_names:
                child = current / name
                if child.is_symlink():
                    raise ValueError(f"{label}_nested_symlink_rejected")
            for name in file_names:
                child = current / name
                if child.is_symlink():
                    raise ValueError(f"{label}_nested_symlink_rejected")
                if child.suffix.lower() != ".parquet":
                    if _is_orphaned_atomic_parquet_temp(child):
                        residual_paths.append(child)
                        continue
                    raise ValueError(
                        f"{label}_non_parquet_file_rejected:"
                        f"{child.relative_to(root).as_posix()}"
                    )
                paths.append(child)
        paths = sorted(paths, key=lambda item: item.relative_to(root).as_posix())
        residual_paths = sorted(
            residual_paths,
            key=lambda item: item.relative_to(root).as_posix(),
        )
        if not paths:
            raise ValueError(f"{label}_parquet_inventory_empty")

        records: list[dict[str, Any]] = []
        for path in paths:
            encoded = self._read_fd_stable_bytes(path, label=f"{label}_file")
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )

        residual_records = []
        for path in residual_paths:
            encoded = self._read_fd_stable_bytes(
                path,
                label=f"{label}_orphaned_atomic_temp",
            )
            residual_records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )

        after_files = sorted(path for path in root.rglob("*") if path.is_file())
        after_paths = sorted(
            path.relative_to(root).as_posix()
            for path in after_files
            if path.suffix.lower() == ".parquet"
        )
        after_residual_paths = sorted(
            path.relative_to(root).as_posix()
            for path in after_files
            if _is_orphaned_atomic_parquet_temp(path)
        )
        after_unexpected_paths = sorted(
            path.relative_to(root).as_posix()
            for path in after_files
            if path.suffix.lower() != ".parquet"
            and not _is_orphaned_atomic_parquet_temp(path)
        )
        before_paths = [str(record["relative_path"]) for record in records]
        before_residual_paths = [
            str(record["relative_path"]) for record in residual_records
        ]
        if (
            after_paths != before_paths
            or after_residual_paths != before_residual_paths
            or after_unexpected_paths
        ):
            raise ValueError(f"{label}_file_set_changed_during_inventory")
        residual_records_after = []
        for path in residual_paths:
            encoded = self._read_fd_stable_bytes(
                path,
                label=f"{label}_orphaned_atomic_temp_postscan",
            )
            residual_records_after.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        if residual_records_after != residual_records:
            raise ValueError(f"{label}_orphaned_atomic_temp_changed")
        return records, paths

    @staticmethod
    def _arrow_type_family(data_type: Any) -> str:
        import pyarrow as pa

        if pa.types.is_dictionary(data_type):
            return MarketDataStore._arrow_type_family(data_type.value_type)
        if pa.types.is_null(data_type):
            return "null"
        if (
            pa.types.is_integer(data_type)
            or pa.types.is_floating(data_type)
            or pa.types.is_decimal(data_type)
        ):
            return "numeric"
        if pa.types.is_boolean(data_type):
            return "bool"
        if (
            pa.types.is_date(data_type)
            or pa.types.is_time(data_type)
            or pa.types.is_timestamp(data_type)
            or pa.types.is_duration(data_type)
        ):
            return "datetime"
        if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            return "string"
        if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
            return "binary"
        raise ValueError(f"snapshot_logical_type_unsupported:{data_type}")

    @staticmethod
    def _logical_column_name(name: Any) -> str:
        column = str(name or "").strip()
        if column in {"Date", "date"}:
            return "trade_date"
        if column in {"Symbol", "symbol"}:
            return "ts_code"
        return column

    def _snapshot_logical_schema(
        self,
        paths: Sequence[Path],
    ) -> tuple[list[str], dict[str, str]]:
        import pyarrow.parquet as pq

        families: dict[str, set[str]] = {}
        for path in paths:
            try:
                schema = pq.read_schema(path)
            except Exception as exc:
                raise ValueError(f"snapshot_parquet_schema_unreadable:{path}") from exc
            physical_names = set(schema.names)
            for field in schema:
                physical_name = str(field.name)
                if physical_name in {"_year", "_month", "year", "month"}:
                    continue
                if physical_name in {"Symbol", "symbol"} and (
                    "ts_code" in physical_names or "Symbol" in physical_names
                    and physical_name == "symbol"
                ):
                    continue
                logical_name = self._logical_column_name(physical_name)
                if not logical_name:
                    raise ValueError("snapshot_logical_column_name_empty")
                families.setdefault(logical_name, set()).add(
                    self._arrow_type_family(field.type)
                )

        for required in ("ts_code", "trade_date"):
            if required not in families:
                raise ValueError(f"snapshot_logical_column_missing:{required}")
        resolved: dict[str, str] = {}
        for name, observed in families.items():
            non_null = observed - {"null"}
            if len(non_null) > 1:
                raise ValueError(
                    "snapshot_logical_type_conflict:"
                    f"{name}:{','.join(sorted(non_null))}"
                )
            resolved[name] = next(iter(non_null), "string")
        resolved["ts_code"] = "string"
        resolved["trade_date"] = "string"
        ordered = ["ts_code", "trade_date"] + sorted(
            name for name in resolved if name not in {"ts_code", "trade_date"}
        )
        return ordered, {name: resolved[name] for name in ordered}

    def _read_parquet_frame_fd_stable(self, path: Path, *, label: str) -> pd.DataFrame:
        import pyarrow.parquet as pq

        try:
            before = os.lstat(path)
        except OSError as exc:
            raise ValueError(f"{label}_missing") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}_regular_file_required")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            opened_signature = self._stat_signature(opened)
            if self._stat_signature(before) != opened_signature:
                raise ValueError(f"{label}_identity_changed_before_read")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                frame = pq.ParquetFile(handle).read().to_pandas()
            after_opened = os.fstat(descriptor)
            after_path = os.lstat(path)
            if (
                self._stat_signature(after_opened) != opened_signature
                or self._stat_signature(after_path) != opened_signature
            ):
                raise ValueError(f"{label}_changed_during_read")
            return frame
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{label}_unreadable") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _normalize_logical_frame(
        self,
        frame: pd.DataFrame,
        *,
        logical_columns: Sequence[str],
    ) -> pd.DataFrame:
        work = frame.copy()
        if "ts_code" in work.columns and "symbol" in work.columns:
            primary = work["ts_code"].map(self._normalize_symbol)
            redundant = work["symbol"].where(work["symbol"].notna(), "").map(
                self._normalize_symbol
            )
            declared = redundant.ne("")
            if not primary.loc[declared].equals(redundant.loc[declared]):
                raise ValueError("snapshot_redundant_symbol_mismatch")
            work = work.drop(columns=["symbol"])
        elif "ts_code" not in work.columns:
            for alias in ("Symbol", "symbol"):
                if alias in work.columns:
                    work = work.rename(columns={alias: "ts_code"})
                    break
        if "trade_date" not in work.columns:
            for alias in ("Date", "date"):
                if alias in work.columns:
                    work = work.rename(columns={alias: "trade_date"})
                    break
        for ignored in ("Symbol", "_year", "_month", "year", "month"):
            if ignored in work.columns:
                work = work.drop(columns=[ignored])
        if "ts_code" not in work.columns or "trade_date" not in work.columns:
            raise ValueError("snapshot_logical_key_columns_missing")
        work["ts_code"] = work["ts_code"].map(self._normalize_symbol)
        work["trade_date"] = work["trade_date"].map(self._normalize_trade_date)
        if work["ts_code"].eq("").any() or work["trade_date"].str.len().ne(8).any():
            raise ValueError("snapshot_logical_key_invalid")
        for column in logical_columns:
            if column not in work.columns:
                work[column] = None
        unexpected = set(work.columns) - set(logical_columns)
        if unexpected:
            raise ValueError(
                "snapshot_logical_columns_unexpected:"
                + ",".join(sorted(str(item) for item in unexpected))
            )
        return work.loc[:, list(logical_columns)].copy()

    @staticmethod
    def _canonical_logical_value(value: Any, *, family: str) -> str:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            return "N"
        if family == "numeric":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("snapshot_numeric_value_nonfinite")
            if number == 0.0:
                number = 0.0
            return "F" + number.hex()
        if family == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered not in {"true", "false", "1", "0"}:
                    raise ValueError("snapshot_boolean_value_invalid")
                return "B1" if lowered in {"true", "1"} else "B0"
            return "B1" if bool(value) else "B0"
        if family == "datetime":
            return "T" + pd.Timestamp(value).isoformat()
        if family == "binary":
            return "X" + bytes(value).hex()
        return "S" + str(value)

    def _snapshot_logical_summary(
        self,
        *,
        root: Path,
        paths: Sequence[Path],
        layout: str,
        logical_columns: Sequence[str],
        logical_types: Mapping[str, str],
        acknowledged_trade_date: str,
    ) -> dict[str, Any]:
        symbol_hashers: dict[str, Any] = {}
        symbol_counts: dict[str, int] = {}
        symbol_first_dates: dict[str, str] = {}
        symbol_last_dates: dict[str, str] = {}
        row_count = 0

        if layout == "table":
            ordered_paths: list[tuple[str, str, Path]] = []
            for path in paths:
                parts = path.relative_to(root).parts
                if (
                    len(parts) != 3
                    or not parts[0].startswith("year=")
                    or not parts[1].startswith("month=")
                    or parts[2] != "part.parquet"
                ):
                    raise ValueError(
                        "snapshot_table_layout_invalid:"
                        f"{path.relative_to(root).as_posix()}"
                    )
                year = parts[0].split("=", 1)[1]
                month = parts[1].split("=", 1)[1]
                if len(year) != 4 or len(month) != 2 or not (year + month).isdigit():
                    raise ValueError("snapshot_table_partition_invalid")
                ordered_paths.append((year, month, path))
            work_paths = [item[2] for item in sorted(ordered_paths)]
        elif layout == "serving":
            work_paths = []
            for path in paths:
                parts = path.relative_to(root).parts
                if (
                    len(parts) != 2
                    or not parts[0].startswith("symbol=")
                    or parts[1] != "bars.parquet"
                ):
                    raise ValueError(
                        "snapshot_serving_layout_invalid:"
                        f"{path.relative_to(root).as_posix()}"
                    )
                work_paths.append(path)
            work_paths.sort(
                key=lambda item: item.parent.name.split("=", 1)[1].upper()
            )
        else:
            raise ValueError("snapshot_layout_invalid")

        for path in work_paths:
            raw = self._read_parquet_frame_fd_stable(
                path,
                label=f"snapshot_{layout}_parquet",
            )
            frame = self._normalize_logical_frame(
                raw,
                logical_columns=logical_columns,
            )
            if layout == "table":
                parts = path.relative_to(root).parts
                expected_partition = (
                    parts[0].split("=", 1)[1] + parts[1].split("=", 1)[1]
                )
                if not frame["trade_date"].str.startswith(expected_partition).all():
                    raise ValueError("snapshot_table_partition_row_mismatch")
            else:
                expected_symbol = self._normalize_symbol(
                    path.parent.name.split("=", 1)[1]
                )
                if not frame["ts_code"].eq(expected_symbol).all():
                    raise ValueError("snapshot_serving_symbol_row_mismatch")
            frame = frame.sort_values(
                ["ts_code", "trade_date"],
                kind="mergesort",
            ).reset_index(drop=True)
            if frame.duplicated(subset=["ts_code", "trade_date"]).any():
                raise ValueError(f"snapshot_{layout}_duplicate_key")

            for symbol, group in frame.groupby("ts_code", sort=True):
                normalized_symbol = self._normalize_symbol(symbol)
                dates = group["trade_date"].astype(str).tolist()
                previous_last = symbol_last_dates.get(normalized_symbol, "")
                if previous_last and dates and dates[0] <= previous_last:
                    raise ValueError(f"snapshot_{layout}_duplicate_or_unordered_key")
                if dates:
                    symbol_first_dates.setdefault(normalized_symbol, dates[0])
                    symbol_last_dates[normalized_symbol] = dates[-1]
                hasher = symbol_hashers.setdefault(normalized_symbol, hashlib.sha256())
                for values in group.itertuples(index=False, name=None):
                    tokens = [
                        self._canonical_logical_value(
                            value,
                            family=str(logical_types[column]),
                        )
                        for column, value in zip(logical_columns, values)
                    ]
                    hasher.update(
                        self._canonical_json_bytes(tokens)
                    )
                count = int(len(group))
                symbol_counts[normalized_symbol] = (
                    symbol_counts.get(normalized_symbol, 0) + count
                )
                row_count += count

        symbol_digests = {
            symbol: symbol_hashers[symbol].hexdigest()
            for symbol in sorted(symbol_hashers)
        }
        root_hasher = hashlib.sha256()
        for symbol in sorted(symbol_digests):
            root_hasher.update(
                (
                    f"{symbol}\t{symbol_counts[symbol]}\t"
                    f"{symbol_first_dates[symbol]}\t{symbol_last_dates[symbol]}\t"
                    f"{symbol_digests[symbol]}\n"
                ).encode("utf-8")
            )
        exact_date_symbols = sorted(
            symbol
            for symbol, last_date in symbol_last_dates.items()
            if last_date == acknowledged_trade_date
        )
        return {
            "logical_rowset_sha256": root_hasher.hexdigest(),
            "row_count": row_count,
            "key_count": row_count,
            "symbol_count": len(symbol_digests),
            "latest_trade_date": max(symbol_last_dates.values(), default=""),
            "exact_date_symbol_count": len(exact_date_symbols),
            "exact_date_symbols_sha256": hashlib.sha256(
                "".join(f"{symbol}\n" for symbol in exact_date_symbols).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "symbol_counts": symbol_counts,
            "symbol_first_dates": symbol_first_dates,
            "symbol_last_dates": symbol_last_dates,
            "symbol_digests": symbol_digests,
        }

    def _candidate_pointer_from_snapshot_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        recovery: Mapping[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        coverage = manifest.get("coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError("snapshot_manifest_coverage_missing")
        latest_complete = self._normalize_trade_date(
            manifest.get("latest_complete_trade_date")
            or manifest.get("latest_trade_date")
        )
        latest_available = self._normalize_trade_date(
            manifest.get("latest_available_trade_date") or latest_complete
        )
        payload: dict[str, Any] = {
            "snapshot_id": str(manifest.get("snapshot_id") or ""),
            "status": "OK",
            "manifest_path": str(manifest.get("manifest_path") or ""),
            "table_root": str(manifest.get("table_root") or ""),
            "derived_serving_root": str(
                manifest.get("derived_serving_root") or ""
            ),
            "latest_available_trade_date": latest_available,
            "latest_complete_trade_date": latest_complete,
            "latest_trade_date": latest_complete,
            "quarantined_tail_dates": list(
                manifest.get("quarantined_tail_dates") or []
            ),
            "coverage": dict(coverage),
            "blockers": [],
            "updated_at": updated_at or self._utc_now(),
        }
        if recovery is not None:
            payload["recovery"] = dict(recovery)
        return payload

    def _validate_snapshot_reactivation_source(
        self,
        *,
        snapshot_id: str,
        expected_snapshot_manifest_sha256: str,
        acknowledged_trade_date: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshot_root = self.data_root / "parquet" / "cn" / "_snapshots"
        manifest_path = snapshot_root / f"{snapshot_id}.json"
        manifest, manifest_bytes = self._read_fd_stable_json(
            manifest_path,
            label="source_snapshot_manifest",
        )
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_sha256 != expected_snapshot_manifest_sha256:
            raise ValueError(
                "source_snapshot_manifest_sha256_mismatch:"
                f"{actual_manifest_sha256}!={expected_snapshot_manifest_sha256}"
            )
        if str(manifest.get("snapshot_id") or "") != snapshot_id:
            raise ValueError("source_snapshot_manifest_snapshot_id_mismatch")
        if str(manifest.get("market") or "").strip().upper() != "CN":
            raise ValueError("source_snapshot_manifest_market_mismatch")
        if str(manifest.get("status") or "").strip().upper() != "OK":
            raise ValueError("source_snapshot_manifest_status_invalid")
        coverage = manifest.get("coverage")
        if not isinstance(coverage, dict) or str(
            coverage.get("coverage_schema_version") or ""
        ).strip() != "cn-full-a-coverage.v4":
            raise ValueError("source_snapshot_exact_v4_required")
        manifest_trade_date = self._normalize_trade_date(
            manifest.get("latest_complete_trade_date")
            or manifest.get("latest_trade_date")
        )
        if manifest_trade_date != acknowledged_trade_date:
            raise ValueError(
                "acknowledged_trade_date_mismatch:"
                f"{acknowledged_trade_date}!={manifest_trade_date}"
            )

        candidate_pointer = self._candidate_pointer_from_snapshot_manifest(
            manifest
        )
        candidate_reader = MarketDataReader(market="CN", data_root=self.data_root)
        candidate_reader._latest_payload = dict(candidate_pointer)
        candidate_gate = candidate_reader.clean_snapshot_gate(refresh=False)
        candidate_blockers = [
            str(item)
            for item in list(candidate_gate.get("blockers", []) or [])
            if str(item).strip()
        ]
        if not candidate_gate.get("healthy") or candidate_blockers:
            raise ValueError(
                "source_snapshot_candidate_validation_failed:"
                + ",".join(dict.fromkeys(candidate_blockers))
            )
        snapshot = candidate_reader._snapshot_from_payload(candidate_pointer)
        expected_manifest_path = snapshot_root / f"{snapshot_id}.json"
        expected_table_root = snapshot_root / snapshot_id / "table" / "bars"
        expected_serving_root = snapshot_root / snapshot_id / "serving" / "bars"
        for label, actual, expected in (
            ("manifest", snapshot.manifest_path, expected_manifest_path),
            ("table", snapshot.table_root, expected_table_root),
            ("serving", snapshot.serving_root, expected_serving_root),
        ):
            try:
                same_path = actual.resolve(strict=True) == expected.resolve(
                    strict=True
                )
            except (OSError, RuntimeError):
                same_path = False
            if not same_path:
                raise ValueError(f"source_snapshot_{label}_path_mismatch")

        table_inventory, table_paths = self._snapshot_file_inventory(
            snapshot.table_root,
            label="source_snapshot_table",
        )
        serving_inventory, serving_paths = self._snapshot_file_inventory(
            snapshot.serving_root,
            label="source_snapshot_serving",
        )
        logical_columns, logical_types = self._snapshot_logical_schema(
            [*table_paths, *serving_paths]
        )
        table_summary = self._snapshot_logical_summary(
            root=snapshot.table_root,
            paths=table_paths,
            layout="table",
            logical_columns=logical_columns,
            logical_types=logical_types,
            acknowledged_trade_date=acknowledged_trade_date,
        )
        serving_summary = self._snapshot_logical_summary(
            root=snapshot.serving_root,
            paths=serving_paths,
            layout="serving",
            logical_columns=logical_columns,
            logical_types=logical_types,
            acknowledged_trade_date=acknowledged_trade_date,
        )
        for field in (
            "logical_rowset_sha256",
            "row_count",
            "key_count",
            "symbol_count",
            "latest_trade_date",
            "exact_date_symbol_count",
            "exact_date_symbols_sha256",
            "symbol_counts",
            "symbol_first_dates",
            "symbol_last_dates",
            "symbol_digests",
        ):
            if table_summary[field] != serving_summary[field]:
                raise ValueError(f"snapshot_table_serving_{field}_mismatch")
        if int(manifest.get("row_count") or -1) != int(
            table_summary["row_count"]
        ):
            raise ValueError("source_snapshot_manifest_row_count_mismatch")
        if int(manifest.get("symbol_count") or -1) != int(
            table_summary["symbol_count"]
        ):
            raise ValueError("source_snapshot_manifest_symbol_count_mismatch")
        table_total_size_bytes = sum(
            int(item["size_bytes"]) for item in table_inventory
        )
        if int(manifest.get("parquet_size_bytes") or -1) != table_total_size_bytes:
            raise ValueError("source_snapshot_manifest_parquet_size_mismatch")
        if table_summary["latest_trade_date"] != acknowledged_trade_date:
            raise ValueError("source_snapshot_latest_trade_date_mismatch")
        observed_bar_count = int(coverage.get("observed_bar_count") or -1)
        exact_date_symbol_count = int(table_summary["exact_date_symbol_count"])
        if observed_bar_count < 0 or observed_bar_count > exact_date_symbol_count:
            raise ValueError("source_snapshot_observed_bar_count_invalid")
        for coverage_key in ("daily_basic_coverage", "adj_factor_coverage"):
            coverage_summary = coverage.get(coverage_key)
            if not isinstance(coverage_summary, Mapping):
                raise ValueError(
                    f"source_snapshot_{coverage_key}_summary_missing"
                )
            if int(coverage_summary.get("covered_count") or -1) != (
                exact_date_symbol_count
            ):
                raise ValueError(
                    f"source_snapshot_{coverage_key}_count_mismatch"
                )

        pit_binding = candidate_reader.coverage_bound_pit(refresh=False)
        pit_blockers = [
            str(item)
            for item in list(pit_binding.get("blockers", []) or [])
            if str(item).strip()
        ]
        if str(pit_binding.get("status") or "") != "passed" or pit_blockers:
            raise ValueError(
                "source_snapshot_pit_validation_failed:"
                + ",".join(dict.fromkeys(pit_blockers))
            )
        pit_membership_path = Path(str(pit_binding.get("canonical_path") or ""))
        pit_manifest_path = Path(
            str(pit_binding.get("generation_manifest_path") or "")
        )
        pit_membership_bytes = self._read_fd_stable_bytes(
            pit_membership_path,
            label="source_snapshot_pit_membership",
        )
        pit_manifest_bytes = self._read_fd_stable_bytes(
            pit_manifest_path,
            label="source_snapshot_pit_generation_manifest",
        )
        pit_membership_sha256 = hashlib.sha256(pit_membership_bytes).hexdigest()
        pit_manifest_sha256 = hashlib.sha256(pit_manifest_bytes).hexdigest()
        if pit_membership_sha256 != str(
            pit_binding.get("canonical_sha256") or ""
        ):
            raise ValueError("source_snapshot_pit_membership_sha256_mismatch")
        if pit_manifest_sha256 != str(
            pit_binding.get("generation_manifest_sha256") or ""
        ):
            raise ValueError(
                "source_snapshot_pit_generation_manifest_sha256_mismatch"
            )

        table_inventory_after, _ = self._snapshot_file_inventory(
            snapshot.table_root,
            label="source_snapshot_table_postscan",
        )
        serving_inventory_after, _ = self._snapshot_file_inventory(
            snapshot.serving_root,
            label="source_snapshot_serving_postscan",
        )
        if table_inventory_after != table_inventory:
            raise ValueError("source_snapshot_table_inventory_changed")
        if serving_inventory_after != serving_inventory:
            raise ValueError("source_snapshot_serving_inventory_changed")
        manifest_after = self._read_fd_stable_bytes(
            manifest_path,
            label="source_snapshot_manifest_postscan",
        )
        if manifest_after != manifest_bytes:
            raise ValueError("source_snapshot_manifest_changed")
        if self._read_fd_stable_bytes(
            pit_membership_path,
            label="source_snapshot_pit_membership_postscan",
        ) != pit_membership_bytes:
            raise ValueError("source_snapshot_pit_membership_changed")
        if self._read_fd_stable_bytes(
            pit_manifest_path,
            label="source_snapshot_pit_generation_manifest_postscan",
        ) != pit_manifest_bytes:
            raise ValueError("source_snapshot_pit_generation_manifest_changed")

        table_inventory_sha256 = hashlib.sha256(
            self._canonical_json_bytes(table_inventory)
        ).hexdigest()
        serving_inventory_sha256 = hashlib.sha256(
            self._canonical_json_bytes(serving_inventory)
        ).hexdigest()
        source_validation = {
            "table_inventory_sha256": table_inventory_sha256,
            "serving_inventory_sha256": serving_inventory_sha256,
            "logical_column_names": list(logical_columns),
            "table_logical_rowset_sha256": table_summary[
                "logical_rowset_sha256"
            ],
            "serving_logical_rowset_sha256": serving_summary[
                "logical_rowset_sha256"
            ],
            "row_count": int(table_summary["row_count"]),
            "key_count": int(table_summary["key_count"]),
            "symbol_count": int(table_summary["symbol_count"]),
            "latest_trade_date": str(table_summary["latest_trade_date"]),
            "exact_date_symbol_count": int(
                table_summary["exact_date_symbol_count"]
            ),
            "pit_membership_path": str(pit_membership_path),
            "pit_membership_sha256": pit_membership_sha256,
            "pit_generation_manifest_path": str(pit_manifest_path),
            "pit_generation_manifest_sha256": pit_manifest_sha256,
        }
        return manifest, source_validation

    def _validate_reactivated_pointer(
        self,
        *,
        attempted_pointer_bytes: bytes,
        snapshot_id: str,
        expected_snapshot_manifest_sha256: str,
        acknowledged_trade_date: str,
        expected_source_validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        actual_pointer_bytes = self._read_fd_stable_bytes(
            self.reader.latest_pointer_path,
            label="reactivated_market_pointer",
        )
        if actual_pointer_bytes != attempted_pointer_bytes:
            raise ValueError("reactivated_market_pointer_readback_mismatch")
        validation_reader = MarketDataReader(
            market="CN",
            data_root=self.data_root,
        )
        gate = validation_reader.clean_snapshot_gate(refresh=True)
        blockers = [
            str(item)
            for item in list(gate.get("blockers", []) or [])
            if str(item).strip()
        ]
        if not gate.get("healthy") or blockers:
            raise ValueError(
                "reactivated_market_pointer_validation_failed:"
                + ",".join(dict.fromkeys(blockers))
            )
        if str(gate.get("snapshot_id") or "") != snapshot_id:
            raise ValueError("reactivated_market_pointer_snapshot_id_mismatch")
        _manifest, observed_source_validation = (
            self._validate_snapshot_reactivation_source(
                snapshot_id=snapshot_id,
                expected_snapshot_manifest_sha256=(
                    expected_snapshot_manifest_sha256
                ),
                acknowledged_trade_date=acknowledged_trade_date,
            )
        )
        if self._canonical_json_bytes(observed_source_validation) != (
            self._canonical_json_bytes(dict(expected_source_validation))
        ):
            raise ValueError("reactivated_snapshot_source_validation_changed")
        return observed_source_validation

    def reactivate_snapshot(
        self,
        *,
        snapshot_id: str,
        expected_snapshot_manifest_sha256: str,
        expected_market_pointer_sha256: str,
        acknowledge_trade_date: str,
        reason: str,
        commit: bool = False,
    ) -> dict[str, Any]:
        if self.market != "CN":
            raise ValueError("storage_reactivate_snapshot_cn_only")
        resolved_snapshot_id = str(snapshot_id or "").strip()
        if (
            not resolved_snapshot_id
            or Path(resolved_snapshot_id).name != resolved_snapshot_id
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in resolved_snapshot_id
            )
        ):
            raise ValueError("snapshot_id_invalid")
        try:
            manifest_sha256 = self._valid_sha256(
                expected_snapshot_manifest_sha256
            )
        except ValueError as exc:
            raise ValueError("expected_snapshot_manifest_sha256_invalid") from exc
        try:
            expected_pointer_sha256 = self._valid_sha256(
                expected_market_pointer_sha256
            )
        except ValueError as exc:
            raise ValueError("expected_market_pointer_sha256_invalid") from exc
        acknowledged_trade_date = self._normalize_trade_date(
            acknowledge_trade_date
        )
        if not acknowledged_trade_date:
            raise ValueError("acknowledge_trade_date_invalid")
        resolved_reason = str(reason or "").strip()
        if not resolved_reason:
            raise ValueError("snapshot_reactivation_reason_required")

        if commit:
            if not self._uses_repository_data_root():
                raise ValueError(
                    "snapshot_reactivation_commit_requires_repository_data_root"
                )
            with self._market_writer_lock():
                return self._reactivate_snapshot_locked(
                    snapshot_id=resolved_snapshot_id,
                    expected_snapshot_manifest_sha256=manifest_sha256,
                    expected_market_pointer_sha256=expected_pointer_sha256,
                    acknowledged_trade_date=acknowledged_trade_date,
                    reason=resolved_reason,
                    commit=True,
                )
        return self._reactivate_snapshot_locked(
            snapshot_id=resolved_snapshot_id,
            expected_snapshot_manifest_sha256=manifest_sha256,
            expected_market_pointer_sha256=expected_pointer_sha256,
            acknowledged_trade_date=acknowledged_trade_date,
            reason=resolved_reason,
            commit=False,
        )

    def _reactivate_snapshot_locked(
        self,
        *,
        snapshot_id: str,
        expected_snapshot_manifest_sha256: str,
        expected_market_pointer_sha256: str,
        acknowledged_trade_date: str,
        reason: str,
        commit: bool,
    ) -> dict[str, Any]:
        previous_pointer_bytes = self._read_fd_stable_bytes(
            self.reader.latest_pointer_path,
            label="current_market_pointer",
        )
        previous_pointer_sha256 = hashlib.sha256(previous_pointer_bytes).hexdigest()
        if previous_pointer_sha256 != expected_market_pointer_sha256:
            raise ValueError(
                "market_pointer_cas_mismatch:"
                f"{previous_pointer_sha256}!={expected_market_pointer_sha256}"
            )
        manifest, source_validation = self._validate_snapshot_reactivation_source(
            snapshot_id=snapshot_id,
            expected_snapshot_manifest_sha256=(
                expected_snapshot_manifest_sha256
            ),
            acknowledged_trade_date=acknowledged_trade_date,
        )
        snapshot_root = self.data_root / "parquet" / "cn" / "_snapshots"
        source_snapshot_manifest_path = self._sealed_recovery_path(
            snapshot_root / f"{snapshot_id}.json",
            label="source_snapshot_manifest",
        )
        source_snapshot_table_path = self._sealed_recovery_path(
            snapshot_root / snapshot_id / "table" / "bars",
            label="source_snapshot_table",
        )
        source_snapshot_serving_path = self._sealed_recovery_path(
            snapshot_root / snapshot_id / "serving" / "bars",
            label="source_snapshot_serving",
        )
        if commit:
            canonical_manifest_paths = {
                "manifest_path": source_snapshot_manifest_path,
                "table_root": source_snapshot_table_path,
                "derived_serving_root": source_snapshot_serving_path,
            }
            for field, expected_path in canonical_manifest_paths.items():
                if str(manifest.get(field) or "") != expected_path:
                    raise ValueError(
                        f"source_snapshot_{field}_not_repository_relative"
                    )
        pointer_manifest = dict(manifest)
        pointer_manifest["manifest_path"] = source_snapshot_manifest_path
        pointer_manifest["table_root"] = source_snapshot_table_path
        pointer_manifest["derived_serving_root"] = source_snapshot_serving_path
        pointer_coverage = dict(manifest.get("coverage") or {})
        pointer_coverage["pit_membership_path"] = source_validation[
            "pit_membership_path"
        ]
        pointer_coverage["pit_generation_manifest_path"] = source_validation[
            "pit_generation_manifest_path"
        ]
        pointer_manifest["coverage"] = pointer_coverage

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        recovery_id = (
            "recovery-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:12]
        )
        recovery_root = (
            self.data_root / "parquet" / "cn" / "_recoveries" / recovery_id
        )
        intent_path = recovery_root / "intent.json"
        receipt_path = recovery_root / "receipt.json"
        sealed_intent_path = self._sealed_recovery_path(
            intent_path,
            label="recovery_intent",
        )
        sealed_receipt_path = self._sealed_recovery_path(
            receipt_path,
            label="recovery_receipt",
        )
        intent = {
            "schema_version": "cn-market-snapshot-recovery-intent.v1",
            "recovery_id": recovery_id,
            "market": "CN",
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "previous_market_pointer_sha256": previous_pointer_sha256,
            "source_snapshot_manifest_path": source_snapshot_manifest_path,
            "source_snapshot_manifest_sha256": (
                expected_snapshot_manifest_sha256
            ),
            "acknowledged_trade_date": acknowledged_trade_date,
            "reason": reason,
            "intent_path": sealed_intent_path,
            "receipt_path": sealed_receipt_path,
            "source_validation": source_validation,
        }
        intent_bytes = self._canonical_json_bytes(intent)
        intent_sha256 = hashlib.sha256(intent_bytes).hexdigest()
        recovery_metadata = {
            "schema_version": "cn-market-snapshot-recovery-pointer.v1",
            "recovery_id": recovery_id,
            "previous_market_pointer_sha256": previous_pointer_sha256,
            "source_snapshot_manifest_sha256": (
                expected_snapshot_manifest_sha256
            ),
            "acknowledged_trade_date": acknowledged_trade_date,
            "reason": reason,
            "intent_path": sealed_intent_path,
            "intent_sha256": intent_sha256,
            "receipt_path": sealed_receipt_path,
        }
        candidate_pointer = self._candidate_pointer_from_snapshot_manifest(
            pointer_manifest,
            recovery=recovery_metadata,
            updated_at=created_at,
        )
        candidate_pointer_bytes = self._json_bytes(candidate_pointer)
        candidate_pointer_sha256 = hashlib.sha256(
            candidate_pointer_bytes
        ).hexdigest()
        result = {
            "schema_version": "cn-market-snapshot-recovery-result.v1",
            "status": "validated_dry_run" if not commit else "activated",
            "commit": bool(commit),
            "market": "CN",
            "snapshot_id": snapshot_id,
            "acknowledged_trade_date": acknowledged_trade_date,
            "previous_market_pointer_sha256": previous_pointer_sha256,
            "new_market_pointer_sha256": candidate_pointer_sha256,
            "source_snapshot_manifest_path": source_snapshot_manifest_path,
            "source_snapshot_manifest_sha256": (
                expected_snapshot_manifest_sha256
            ),
            "recovery_id": recovery_id,
            "intent_path": sealed_intent_path,
            "intent_sha256": intent_sha256,
            "receipt_path": sealed_receipt_path,
            "source_validation": source_validation,
        }
        if not commit:
            return result

        if recovery_root.exists():
            raise ValueError("snapshot_recovery_generation_already_exists")
        recovery_root.mkdir(parents=True, exist_ok=False)
        self._fsync_directory(recovery_root.parent)
        self._write_new_bytes(intent_bytes, intent_path, label="recovery_intent")

        attempted_pointer_written = False
        failure: Exception | None = None
        failure_status = "activation_failed"
        try:
            current_pointer_bytes = self._read_fd_stable_bytes(
                self.reader.latest_pointer_path,
                label="current_market_pointer_precommit",
            )
            if current_pointer_bytes != previous_pointer_bytes:
                failure_status = "cas_failed"
                raise ValueError("market_pointer_cas_mismatch_precommit")
            self._atomic_write_bytes(
                candidate_pointer_bytes,
                self.reader.latest_pointer_path,
            )
            attempted_pointer_written = True
            self.reader._latest_payload = None
            self.reader._snapshot_gate_cache = None
            self.reader._serving_symbols_cache = None
            self._validate_reactivated_pointer(
                attempted_pointer_bytes=candidate_pointer_bytes,
                snapshot_id=snapshot_id,
                expected_snapshot_manifest_sha256=(
                    expected_snapshot_manifest_sha256
                ),
                acknowledged_trade_date=acknowledged_trade_date,
                expected_source_validation=source_validation,
            )
            receipt = {
                "schema_version": "cn-market-snapshot-recovery-receipt.v1",
                "status": "activated",
                "recovery_id": recovery_id,
                "market": "CN",
                "snapshot_id": snapshot_id,
                "activated_at": self._utc_now(),
                "previous_market_pointer_sha256": previous_pointer_sha256,
                "new_market_pointer_sha256": candidate_pointer_sha256,
                "source_snapshot_manifest_path": source_snapshot_manifest_path,
                "source_snapshot_manifest_sha256": (
                    expected_snapshot_manifest_sha256
                ),
                "acknowledged_trade_date": acknowledged_trade_date,
                "reason": reason,
                "intent_path": sealed_intent_path,
                "intent_sha256": intent_sha256,
                "receipt_path": sealed_receipt_path,
                "source_validation": source_validation,
            }
            receipt_bytes = self._canonical_json_bytes(receipt)
            self._write_new_bytes(
                receipt_bytes,
                receipt_path,
                label="recovery_receipt",
            )
            if self._read_fd_stable_bytes(
                self.reader.latest_pointer_path,
                label="reactivated_market_pointer_final",
            ) != candidate_pointer_bytes:
                raise ValueError("reactivated_market_pointer_final_mismatch")
            result["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
            return result
        except Exception as exc:
            failure = exc
            try:
                current_pointer_bytes = self.reader.latest_pointer_path.read_bytes()
            except OSError:
                current_pointer_bytes = b""
            rolled_back = False
            if attempted_pointer_written and current_pointer_bytes == candidate_pointer_bytes:
                self._atomic_write_bytes(
                    previous_pointer_bytes,
                    self.reader.latest_pointer_path,
                )
                rolled_back = True
                failure_status = "rolled_back"
            elif attempted_pointer_written:
                failure_status = "activation_failed_pointer_changed"
            self.reader._latest_payload = None
            self.reader._snapshot_gate_cache = None
            self.reader._serving_symbols_cache = None
            if not receipt_path.exists():
                failure_receipt = {
                    "schema_version": (
                        "cn-market-snapshot-recovery-receipt.v1"
                    ),
                    "status": failure_status,
                    "recovery_id": recovery_id,
                    "market": "CN",
                    "snapshot_id": snapshot_id,
                    "finished_at": self._utc_now(),
                    "previous_market_pointer_sha256": previous_pointer_sha256,
                    "attempted_market_pointer_sha256": (
                        candidate_pointer_sha256
                    ),
                    "source_snapshot_manifest_sha256": (
                        expected_snapshot_manifest_sha256
                    ),
                    "intent_path": sealed_intent_path,
                    "intent_sha256": intent_sha256,
                    "receipt_path": sealed_receipt_path,
                    "rolled_back": rolled_back,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                self._write_new_bytes(
                    self._canonical_json_bytes(failure_receipt),
                    receipt_path,
                    label="recovery_failure_receipt",
                )
            raise failure

    def write_full_history_bars(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        snapshot_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.market == "CN":
            raise ValueError("cn_full_history_writer_retired_use_parquet_direct")
        normalized = self._normalize_bars_frame(frame)
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


def run_storage_reactivate_snapshot(
    *,
    market: str = "CN",
    snapshot_id: str,
    expected_snapshot_manifest_sha256: str,
    expected_market_pointer_sha256: str,
    acknowledge_trade_date: str,
    reason: str,
    commit: bool = False,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    return MarketDataStore(market=market, data_root=data_root).reactivate_snapshot(
        snapshot_id=snapshot_id,
        expected_snapshot_manifest_sha256=(
            expected_snapshot_manifest_sha256
        ),
        expected_market_pointer_sha256=expected_market_pointer_sha256,
        acknowledge_trade_date=acknowledge_trade_date,
        reason=reason,
        commit=commit,
    )


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
    roots_to_check: dict[str, dict[str, Any]] = {
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
    "run_storage_reactivate_snapshot",
    "run_storage_validate",
    "run_storage_validate_clean",
]
