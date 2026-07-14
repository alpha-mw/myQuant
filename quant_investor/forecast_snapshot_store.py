#!/usr/bin/env python3
"""
离线盈利预测快照存储。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.branch_contracts import ForecastSnapshot


def _normalize_as_of(as_of: Any) -> str:
    ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _build_missing_snapshot(
    *,
    symbol: str,
    as_of: str,
    reason: str,
    note: str,
    cached_as_of: str = "",
) -> ForecastSnapshot:
    data_quality = {
        "status": "neutral_snapshot",
        "reason": reason,
        "provider_missing": False,
        "provider_name": "offline_forecast_cache",
        "missing_scope": "symbol",
    }
    provenance = {
        "snapshot_type": "forecast",
        "provider_name": "offline_forecast_cache",
        "reason": reason,
        "provider_missing": False,
        "missing_scope": "symbol",
    }
    metadata: dict[str, Any] = {}
    if cached_as_of:
        metadata["cached_as_of"] = cached_as_of

    return ForecastSnapshot(
        symbol=symbol,
        as_of=as_of,
        available=False,
        source="offline_forecast_cache",
        provider="offline_forecast_cache",
        publish_time=f"{as_of}T00:00:00" if as_of else "",
        effective_time=f"{as_of}T00:00:00" if as_of else "",
        ingest_time="",
        revision_id=f"forecast:offline_forecast_cache:{as_of}",
        is_estimated=True,
        notes=[note],
        data_quality=data_quality,
        provenance=provenance,
        metadata=metadata,
    )


class ForecastSnapshotStore:
    """读取/写入离线 forecast snapshots。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _symbol_path(self, symbol: str) -> Path:
        normalized = str(symbol).replace("/", "_").replace(":", "_")
        return self.base_dir / f"{normalized}.json"

    def _version_dir(self, symbol: str) -> Path:
        normalized = str(symbol).replace("/", "_").replace(":", "_")
        return self.base_dir / "versions" / normalized

    def _version_path(self, symbol: str, as_of: str) -> Path:
        return self._version_dir(symbol) / f"{_normalize_as_of(as_of)}.json"

    @staticmethod
    def _read_snapshot(path: Path, symbol: str) -> ForecastSnapshot:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ForecastSnapshot(
            symbol=str(payload.get("symbol", symbol)),
            horizon_days=int(payload.get("horizon_days", 5)),
            expected_return=float(payload.get("expected_return", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            available=bool(payload.get("available", False)),
            as_of=str(payload.get("as_of", "")),
            source=str(payload.get("source", "offline_forecast_cache")),
            provider=str(payload.get("provider", "offline_forecast_cache")),
            publish_time=str(payload.get("publish_time", "")),
            effective_time=str(payload.get("effective_time", "")),
            ingest_time=str(payload.get("ingest_time", "")),
            revision_id=str(payload.get("revision_id", "")),
            is_estimated=bool(payload.get("is_estimated", False)),
            eps_growth=float(payload.get("eps_growth", 0.0)),
            revenue_growth_forecast=float(payload.get("revenue_growth_forecast", 0.0)),
            forecast_revision=float(payload.get("forecast_revision", 0.0)),
            coverage_count=int(payload.get("coverage_count", 0)),
            notes=[str(item) for item in payload.get("notes", [])],
            data_quality=dict(payload.get("data_quality", {})),
            provenance=dict(payload.get("provenance", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def load_snapshot(self, symbol: str, as_of: str | None = None) -> ForecastSnapshot | None:
        requested_as_of = _normalize_as_of(as_of) if as_of is not None else ""
        if requested_as_of:
            version_dir = self._version_dir(symbol)
            if version_dir.exists():
                eligible = sorted(
                    path
                    for path in version_dir.glob("*.json")
                    if _normalize_as_of(path.stem) and _normalize_as_of(path.stem) <= requested_as_of
                )
                if eligible:
                    return self._read_snapshot(eligible[-1], symbol)

        path = self._symbol_path(symbol)
        if not path.exists():
            return None
        snapshot = self._read_snapshot(path, symbol)
        if requested_as_of and _normalize_as_of(snapshot.as_of) > requested_as_of:
            return None
        return snapshot

    def inspect_snapshot(self, symbol: str, as_of: str) -> dict[str, Any]:
        requested_as_of = _normalize_as_of(as_of)
        latest = self.load_snapshot(symbol)
        snapshot = self.load_snapshot(symbol, requested_as_of)
        if snapshot is None:
            latest_as_of = _normalize_as_of(latest.as_of) if latest is not None else ""
            return {
                "status": "future" if latest_as_of and latest_as_of > requested_as_of else "missing",
                "requested_as_of": requested_as_of,
                "cached_as_of": latest_as_of,
                "available": False,
                "path": str(self._symbol_path(symbol)),
            }

        cached_as_of = _normalize_as_of(snapshot.as_of)
        is_fresh = not requested_as_of or cached_as_of == requested_as_of
        return {
            "status": "fresh" if is_fresh else "stale",
            "requested_as_of": requested_as_of,
            "cached_as_of": cached_as_of,
            "available": bool(snapshot.available),
            "path": str(self._symbol_path(symbol)),
        }

    def get_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
        requested_as_of = _normalize_as_of(as_of)
        latest = self.load_snapshot(symbol)
        snapshot = self.load_snapshot(symbol, requested_as_of)
        if snapshot is None:
            latest_as_of = _normalize_as_of(latest.as_of) if latest is not None else ""
            future_only = bool(requested_as_of and latest_as_of and latest_as_of > requested_as_of)
            return _build_missing_snapshot(
                symbol=symbol,
                as_of=requested_as_of,
                reason="forecast_cache_future_snapshot" if future_only else "forecast_cache_missing_or_stale",
                note="forecast_cache_future_snapshot" if future_only else "forecast_cache_missing_or_stale",
                cached_as_of=latest_as_of,
            )

        cached_as_of = _normalize_as_of(snapshot.as_of)
        if requested_as_of and (not cached_as_of or cached_as_of < requested_as_of):
            return _build_missing_snapshot(
                symbol=symbol,
                as_of=requested_as_of,
                reason="forecast_cache_missing_or_stale",
                note="forecast_cache_missing_or_stale",
                cached_as_of=cached_as_of,
            )
        return snapshot

    def save_snapshot(self, snapshot: ForecastSnapshot | dict[str, Any]) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(snapshot, ForecastSnapshot):
            payload = asdict(snapshot)
            symbol = snapshot.symbol
        else:
            payload = dict(snapshot)
            symbol = str(payload.get("symbol", "unknown"))

        normalized_as_of = _normalize_as_of(payload.get("as_of"))
        if not normalized_as_of:
            raise ValueError("forecast snapshot requires a valid as_of date")
        version_path = self._version_path(symbol, normalized_as_of)
        path = self._symbol_path(symbol)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._atomic_write(version_path, serialized)
        latest = self.load_snapshot(symbol)
        if latest is None or _normalize_as_of(latest.as_of) <= normalized_as_of:
            self._atomic_write(path, serialized)
        return path

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
