"""Offline CN intelligence mart builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.branch_readiness import SOURCE_TUSHARE

DEFAULT_INTELLIGENCE_ROOT = Path("data/parquet/cn/intelligence_daily")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/intelligence")
INTELLIGENCE_FIELDS = (
    "intelligence_score",
    "event_risk_score",
    "sentiment_score",
    "money_flow_score",
    "breadth_score",
    "rotation_score",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if "." in text:
        return text
    if text.startswith(("6", "9")):
        return f"{text}.SH"
    return f"{text}.SZ" if text else ""


def _latest_date(frame: pd.DataFrame) -> str:
    for column in ("trade_date", "date"):
        if column in frame.columns and not frame.empty:
            values = pd.to_datetime(frame[column], errors="coerce")
            if values.notna().any():
                return values.max().strftime("%Y-%m-%d")
    return ""


def build_intelligence_daily(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    source: str = "tushare_moneyflow_margin_breadth",
    source_priority: str = SOURCE_TUSHARE,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in symbol_frames.items():
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
        volume_col = "volume" if "volume" in working.columns else "vol" if "vol" in working.columns else ""
        amount_col = "amount" if "amount" in working.columns else "amt" if "amt" in working.columns else ""
        close = pd.to_numeric(working[close_col], errors="coerce").dropna() if close_col else pd.Series(dtype=float)
        volume = pd.to_numeric(working[volume_col], errors="coerce").dropna() if volume_col else pd.Series(dtype=float)
        amount = pd.to_numeric(working[amount_col], errors="coerce").dropna() if amount_col else pd.Series(dtype=float)
        momentum = float(close.pct_change().tail(5).mean()) if len(close) >= 6 else 0.0
        if len(volume) >= 5 and float(volume.tail(20).mean() or 0.0) > 0:
            money_flow = float(volume.iloc[-1] / volume.tail(20).mean() - 1.0)
        elif len(amount) >= 5 and float(amount.tail(20).mean() or 0.0) > 0:
            money_flow = float(amount.iloc[-1] / amount.tail(20).mean() - 1.0)
        else:
            money_flow = 0.0
        score = max(-1.0, min(1.0, momentum * 8.0 + money_flow * 0.15))
        rows.append(
            {
                "ts_code": _normalize_symbol(symbol),
                "trade_date": _latest_date(working),
                "intelligence_score": score,
                "event_risk_score": 0.0,
                "sentiment_score": 0.0,
                "money_flow_score": max(-1.0, min(1.0, money_flow)),
                "breadth_score": 1.0 if momentum > 0 else 0.0,
                "rotation_score": 0.0,
                "source": source,
                "source_priority": source_priority,
                "pit_status": "point_in_time",
                "fetched_at": _now_utc(),
            }
        )
    columns = ["ts_code", "trade_date", *INTELLIGENCE_FIELDS, "source", "source_priority", "pit_status", "fetched_at"]
    return pd.DataFrame(rows, columns=columns)


def write_intelligence_mart(
    daily: pd.DataFrame,
    *,
    data_root: str | Path = DEFAULT_INTELLIGENCE_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    run_id: str = "",
    provider_status: str = "offline_input",
    provider_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"cn_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = Path(data_root)
    snapshot_dir = Path(raw_snapshot_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    frame = daily.copy() if daily is not None else pd.DataFrame()
    daily_path = data_dir / "part.parquet"
    raw_path = snapshot_dir / f"{run_id}.csv"
    frame.to_parquet(daily_path, index=False)
    frame.to_csv(raw_path, index=False)
    coverage = 0.0 if frame.empty else float(frame[list(INTELLIGENCE_FIELDS)].notna().sum().sum() / max(len(frame) * len(INTELLIGENCE_FIELDS), 1))
    frame_priority = ""
    if "source_priority" in frame.columns and not frame.empty:
        priorities = frame["source_priority"].dropna().astype(str).str.strip()
        frame_priority = str(priorities.iloc[0]) if not priorities.empty else ""
    manifest_priority = (
        frame_priority
        or (SOURCE_TUSHARE if provider_status.startswith("live") or provider_status == "offline_input" else "manual_offline_snapshot")
    )
    manifest = {
        "run_id": run_id,
        "schema_version": "cn-intelligence-mart.v1",
        "provider_status": provider_status,
        "source_priority": manifest_priority,
        "daily_rows": int(len(frame)),
        "field_set": list(INTELLIGENCE_FIELDS),
        "coverage_rate": coverage,
        "storage_backend": "parquet_canonical",
        "table": "intelligence_daily",
        "intelligence_daily": str(daily_path),
        "raw_snapshot": str(raw_path),
        "provider_manifest": dict(provider_manifest or {}),
    }
    (data_dir / "latest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_cn_intelligence_maintenance(
    *,
    symbol_frames: Mapping[str, pd.DataFrame] | None = None,
    data_root: str | Path = DEFAULT_INTELLIGENCE_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    frames = dict(symbol_frames or {})
    provider_status = "offline_input" if frames else "not_requested"
    if allow_live and not frames:
        provider_status = "live_tushare_not_implemented"
    elif allow_public_fallback and not frames:
        provider_status = "public_fallback_not_implemented"
    daily = build_intelligence_daily(frames) if frames else pd.DataFrame(columns=["ts_code", "trade_date", *INTELLIGENCE_FIELDS])
    manifest = write_intelligence_mart(
        daily,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        run_id=run_id,
        provider_status=provider_status,
        provider_manifest={"allow_live": bool(allow_live), "allow_public_fallback": bool(allow_public_fallback)},
    )
    return {"run_id": manifest["run_id"], "provider_status": provider_status, "manifest": manifest}


__all__ = [
    "DEFAULT_INTELLIGENCE_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "INTELLIGENCE_FIELDS",
    "build_intelligence_daily",
    "run_cn_intelligence_maintenance",
    "write_intelligence_mart",
]
