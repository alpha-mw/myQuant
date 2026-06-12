"""Parquet market data storage validation and materialization helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_investor.market.market_data_reader import (
    MarketDataReader,
    MarketDataUnavailableError,
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
        status = "passed" if gate.get("healthy") else "failed"
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
        }

    def _atomic_write_parquet(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _atomic_write_json(self, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)

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
