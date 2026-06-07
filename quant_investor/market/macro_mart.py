"""Offline CN macro mart builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.branch_readiness import SOURCE_TUSHARE

DEFAULT_MACRO_ROOT = Path("data/clean/cn_macro")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/macro")
MACRO_FIELDS = ("macro_score", "liquidity_score", "volatility_percentile", "policy_signal")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or "").strip() or datetime.now().strftime("%Y%m%d"), errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(datetime.now())
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def write_macro_mart(
    indicators: Mapping[str, Any] | None = None,
    *,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    run_id: str = "",
    provider_status: str = "offline_input",
    source_priority: str = SOURCE_TUSHARE,
    provider_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"cn_macro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = Path(data_root)
    snapshot_dir = Path(raw_snapshot_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    row = dict(indicators or {})
    if row:
        row.setdefault("trade_date", _date_text(as_of))
        row.setdefault("source", provider_status)
        row.setdefault("source_priority", source_priority)
        row.setdefault("pit_status", "market_point_in_time")
        row.setdefault("fetched_at", _now_utc())
        frame = pd.DataFrame([row])
    else:
        frame = pd.DataFrame(columns=["trade_date", *MACRO_FIELDS, "source", "source_priority", "pit_status", "fetched_at"])
    daily_path = data_dir / "macro_daily.csv"
    raw_path = snapshot_dir / f"{run_id}.csv"
    frame.to_csv(daily_path, index=False)
    frame.to_csv(raw_path, index=False)
    coverage = 0.0 if frame.empty else float(frame[list(MACRO_FIELDS)].notna().sum().sum() / max(len(frame) * len(MACRO_FIELDS), 1))
    manifest = {
        "run_id": run_id,
        "schema_version": "cn-macro-mart.v1",
        "provider_status": provider_status,
        "source_priority": source_priority,
        "daily_rows": int(len(frame)),
        "field_set": list(MACRO_FIELDS),
        "coverage_rate": coverage,
        "macro_daily": str(daily_path),
        "raw_snapshot": str(raw_path),
        "provider_manifest": dict(provider_manifest or {}),
    }
    (data_dir / "latest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_cn_macro_maintenance(
    *,
    indicators: Mapping[str, Any] | None = None,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    provider_status = "offline_input" if indicators else "not_requested"
    if allow_live and not indicators:
        provider_status = "live_tushare_not_implemented"
    elif allow_public_fallback and not indicators:
        provider_status = "public_fallback_not_implemented"
    manifest = write_macro_mart(
        indicators or {},
        as_of=as_of,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        run_id=run_id,
        provider_status=provider_status,
        source_priority=SOURCE_TUSHARE if indicators else "manual_offline_snapshot",
        provider_manifest={"allow_live": bool(allow_live), "allow_public_fallback": bool(allow_public_fallback)},
    )
    return {"run_id": manifest["run_id"], "provider_status": provider_status, "manifest": manifest}


__all__ = [
    "DEFAULT_MACRO_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "MACRO_FIELDS",
    "run_cn_macro_maintenance",
    "write_macro_mart",
]
