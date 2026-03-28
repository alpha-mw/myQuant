#!/usr/bin/env python3
"""
离线盈利预测快照存储。
"""

from __future__ import annotations

import json
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

    def load_snapshot(self, symbol: str) -> ForecastSnapshot | None:
        path = self._symbol_path(symbol)
        if not path.exists():
            return None

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

    def inspect_snapshot(self, symbol: str, as_of: str) -> dict[str, Any]:
        requested_as_of = _normalize_as_of(as_of)
        snapshot = self.load_snapshot(symbol)
        if snapshot is None:
            return {
                "status": "missing",
                "requested_as_of": requested_as_of,
                "cached_as_of": "",
                "available": False,
                "path": str(self._symbol_path(symbol)),
            }

        cached_as_of = _normalize_as_of(snapshot.as_of)
        is_fresh = not requested_as_of or (cached_as_of and cached_as_of >= requested_as_of)
        return {
            "status": "fresh" if is_fresh else "stale",
            "requested_as_of": requested_as_of,
            "cached_as_of": cached_as_of,
            "available": bool(snapshot.available),
            "path": str(self._symbol_path(symbol)),
        }

    def get_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
        requested_as_of = _normalize_as_of(as_of)
        snapshot = self.load_snapshot(symbol)
        if snapshot is None:
            return _build_missing_snapshot(
                symbol=symbol,
                as_of=requested_as_of,
                reason="forecast_cache_missing_or_stale",
                note="forecast_cache_missing_or_stale",
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

        path = self._symbol_path(symbol)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
