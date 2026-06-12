"""Full-market metrics cache for the CN aggressive tracker."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.market.name_map import load_cn_stock_names


MARKET_METRICS_CACHE_SCHEMA_VERSION = "cn_aggressive_market_metrics_cache.v1"
MARKET_METRICS_COMPONENT_KEYS = ("full_a", "hs300", "zz500", "zz1000")
MARKET_METRICS_CATEGORIES = ("hs300", "zz500", "zz1000")
MARKET_METRICS_OUTPUT_COLUMNS = (
    "symbol",
    "name",
    "category",
    "ret1",
    "ret5",
    "ret20",
    "ret60",
    "close_vs_ma20",
    "ma20_vs_ma60",
    "ma60_vs_ma120",
    "dd20",
    "latest_close",
    "stage_target_price",
    "stage_stop_price",
    "score_full_market",
    "rank_full_market",
)
MARKET_METRICS_REQUIRED_COLUMNS = {
    "symbol",
    "name",
    "category",
    "ret5",
    "ret20",
    "ret60",
    "close_vs_ma20",
    "ma20_vs_ma60",
    "ma60_vs_ma120",
    "dd20",
    "latest_close",
    "stage_target_price",
    "stage_stop_price",
    "score_full_market",
    "rank_full_market",
}


@dataclass
class MarketMetricsBundle:
    full_metrics: pd.DataFrame
    breadth: dict[str, Any]
    cache_meta: dict[str, Any]


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return value / base


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _price_series(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].astype(float)


def _load_history_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def _metric_return(close: pd.Series, periods: int) -> float:
    if close.empty:
        return 0.0
    if len(close) <= periods:
        base = float(close.iloc[0])
    else:
        base = float(close.iloc[-(periods + 1)])
    current = float(close.iloc[-1])
    return _safe_pct(current - base, base)


def _derive_stage_levels(
    frame: pd.DataFrame,
    current_price: float,
) -> tuple[float, float]:
    if frame.empty or current_price <= 0:
        return round(current_price * 1.08, 2), round(current_price * 0.94, 2)

    recent = frame.tail(60).copy()
    high = (
        recent["high"].astype(float)
        if "high" in recent.columns
        else recent["close"].astype(float)
    )
    low = (
        recent["low"].astype(float)
        if "low" in recent.columns
        else recent["close"].astype(float)
    )
    close = recent["close"].astype(float)
    prev_close = close.shift(1).fillna(close)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = (
        float(true_range.tail(14).mean())
        if len(true_range) >= 2
        else current_price * 0.02
    )
    atr = max(atr, current_price * 0.005)

    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    low20 = float(low.tail(20).min()) if len(low) >= 5 else current_price * 0.96
    high20 = float(high.tail(20).max()) if len(high) >= 5 else current_price * 1.06

    support = min(current_price, max(low20, ma20 - 0.75 * atr))
    resistance = max(high20, current_price + 1.5 * atr)

    stop_price = max(
        current_price * 0.75,
        min(support * 0.99, current_price * 0.985),
    )
    target_price = max(current_price * 1.05, resistance)
    return round(target_price, 2), round(stop_price, 2)


def _score_full_market_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=MARKET_METRICS_OUTPUT_COLUMNS)

    scored = metrics.copy()
    rank_weights = {
        "ret1": 0.08,
        "ret5": 0.14,
        "ret20": 0.24,
        "ret60": 0.22,
        "close_vs_ma20": 0.12,
        "ma20_vs_ma60": 0.10,
        "ma60_vs_ma120": 0.06,
        "dd20": 0.04,
    }
    for column in rank_weights:
        scored[f"{column}_pct"] = scored[column].rank(
            method="average",
            pct=True,
        )

    scored["score_full_market"] = 0.0
    for column, weight in rank_weights.items():
        scored["score_full_market"] += scored[f"{column}_pct"] * weight

    scored["score_full_market"] = scored["score_full_market"].round(6)
    scored = scored.sort_values(
        by=["score_full_market", "ret20", "ret60", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["rank_full_market"] = range(1, len(scored) + 1)
    return scored


def _components_fingerprint(components: dict[str, Any]) -> str:
    payload = {
        category: sorted(
            str(symbol).strip().upper()
            for symbol in list(components.get(category, []) or [])
        )
        for category in MARKET_METRICS_COMPONENT_KEYS
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _reader_snapshot_payload(reader: Any) -> dict[str, Any]:
    snapshot_fn = getattr(reader, "snapshot", None)
    if callable(snapshot_fn):
        payload = snapshot_fn()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _market_metrics_cache_dir(
    base_dir: Path,
    *,
    snapshot_id: str,
    latest_trade_date: str,
    components_fingerprint: str,
) -> Path:
    del components_fingerprint
    safe_snapshot = str(snapshot_id or "unknown_snapshot").replace("/", "_")
    safe_trade_date = str(latest_trade_date or "unknown_date").replace("/", "_")
    return (
        base_dir
        / "_cache"
        / "market_metrics"
        / f"{safe_snapshot}_{safe_trade_date}"
    )


def _validate_market_metrics_frame(metrics: pd.DataFrame) -> list[str]:
    if not isinstance(metrics, pd.DataFrame):
        return sorted(MARKET_METRICS_REQUIRED_COLUMNS)
    return sorted(
        column
        for column in MARKET_METRICS_REQUIRED_COLUMNS
        if column not in metrics.columns
    )


def _normalize_market_metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame) and metrics.empty:
        return metrics.reindex(columns=MARKET_METRICS_OUTPUT_COLUMNS)
    return metrics


def _load_cached_market_metrics_bundle(
    *,
    cache_dir: Path,
    snapshot_id: str,
    latest_trade_date: str,
    components_fingerprint: str,
) -> MarketMetricsBundle | None:
    metrics_path = cache_dir / "full_metrics.parquet"
    breadth_path = cache_dir / "breadth.json"
    if not (metrics_path.exists() and breadth_path.exists()):
        return None
    try:
        meta = json.loads(breadth_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    expected = {
        "schema_version": MARKET_METRICS_CACHE_SCHEMA_VERSION,
        "snapshot_id": str(snapshot_id),
        "analysis_trade_date": str(latest_trade_date),
        "components_fingerprint": str(components_fingerprint),
    }
    for key, value in expected.items():
        if str(meta.get(key, "")) != value:
            return None
    try:
        metrics = pd.read_parquet(metrics_path)
    except Exception:
        return None
    if _validate_market_metrics_frame(metrics):
        return None
    breadth_payload = dict(meta.get("breadth", {}) or {})
    if int(meta.get("row_count", len(metrics)) or 0) != len(metrics):
        return None
    cache_meta = {
        **{key: value for key, value in meta.items() if key != "breadth"},
        "status": "cache_hit",
        "cache_hit": True,
        "cache_dir": str(cache_dir),
        "full_metrics_path": str(metrics_path),
        "breadth_path": str(breadth_path),
        "row_count": int(len(metrics)),
        "compute_elapsed_sec": 0.0,
    }
    return MarketMetricsBundle(
        full_metrics=metrics,
        breadth=breadth_payload,
        cache_meta=cache_meta,
    )


def _write_market_metrics_cache(
    *,
    cache_dir: Path,
    full_metrics: pd.DataFrame,
    breadth: dict[str, Any],
    snapshot_id: str,
    latest_trade_date: str,
    components_fingerprint: str,
    compute_elapsed_sec: float,
) -> dict[str, Any]:
    full_metrics = _normalize_market_metrics_frame(full_metrics)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cache_dir / "full_metrics.parquet"
    breadth_path = cache_dir / "breadth.json"
    tmp_metrics = cache_dir / f".full_metrics.parquet.tmp-{os.getpid()}"
    tmp_breadth = cache_dir / f".breadth.json.tmp-{os.getpid()}"
    disk_meta = {
        "schema_version": MARKET_METRICS_CACHE_SCHEMA_VERSION,
        "snapshot_id": str(snapshot_id),
        "analysis_trade_date": str(latest_trade_date),
        "components_fingerprint": str(components_fingerprint),
        "cache_dir": str(cache_dir),
        "full_metrics_path": str(metrics_path),
        "breadth_path": str(breadth_path),
        "row_count": int(len(full_metrics)),
        "compute_elapsed_sec": round(float(compute_elapsed_sec), 3),
        "breadth": _jsonable(breadth),
    }
    try:
        full_metrics.to_parquet(tmp_metrics, index=False)
        readback = pd.read_parquet(tmp_metrics)
        missing_columns = _validate_market_metrics_frame(readback)
        if missing_columns:
            raise RuntimeError(
                "full-market metrics cache readback missing required columns: "
                + ", ".join(missing_columns)
            )
        disk_meta["generated_at"] = _now_local().isoformat()
        tmp_breadth.write_text(
            json.dumps(_jsonable(disk_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_metrics, metrics_path)
        os.replace(tmp_breadth, breadth_path)
    finally:
        tmp_metrics.unlink(missing_ok=True)
        tmp_breadth.unlink(missing_ok=True)
    return {
        **{key: value for key, value in disk_meta.items() if key != "breadth"},
        "status": "blocking_generated",
        "cache_hit": False,
    }


def _read_frame_from_result(read_result: Any) -> pd.DataFrame:
    frame = getattr(read_result, "frame", pd.DataFrame())
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if "trade_date" in frame.columns:
        frame = frame.copy()
        frame["trade_date"] = frame["trade_date"].map(
            lambda value: str(value).replace("-", "")[:8]
        )
        frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def _compute_category_breadth(
    category: str,
    symbols: list[str],
    reader: Any,
    latest_trade_date: str,
    completeness_report: dict[str, Any],
    read_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    covered = 0
    adv_1d = 0
    adv_20d = 0
    ma20_gt_ma60 = 0
    ret_1d_values: list[float] = []
    ret_20d_values: list[float] = []
    ret_60d_values: list[float] = []
    batch_results = read_results or {}

    for symbol in symbols:
        read_result = batch_results.get(symbol)
        if read_result is None:
            read_result = reader.read_symbol_frame(
                symbol,
                universe_key="full_a",
                category=category,
            )
        frame = _read_frame_from_result(read_result)
        if frame.empty:
            continue
        latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
        if latest_local_date != latest_trade_date:
            continue

        close = _price_series(frame).dropna().astype(float)
        if len(close) < 2:
            continue

        ret1 = _metric_return(close, 1)
        ret20 = _metric_return(close, 20)
        ret60 = _metric_return(close, 60)
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
        ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())

        covered += 1
        adv_1d += int(ret1 > 0)
        adv_20d += int(ret20 > 0)
        ma20_gt_ma60 += int(ma20 > ma60)
        ret_1d_values.append(ret1)
        ret_20d_values.append(ret20)
        ret_60d_values.append(ret60)

    payload = dict(
        (completeness_report.get("categories", {}) or {}).get(category, {}) or {}
    )
    return {
        "ret1_positive_ratio": adv_1d / covered if covered else 0.0,
        "ret20_positive_ratio": adv_20d / covered if covered else 0.0,
        "ma20_gt_ma60_ratio": ma20_gt_ma60 / covered if covered else 0.0,
        "avg_ret1": sum(ret_1d_values) / len(ret_1d_values)
        if ret_1d_values
        else 0.0,
        "avg_ret20": sum(ret_20d_values) / len(ret_20d_values)
        if ret_20d_values
        else 0.0,
        "avg_ret60": sum(ret_60d_values) / len(ret_60d_values)
        if ret_60d_values
        else 0.0,
        "latest_count": covered,
        "expected": int(payload.get("expected", len(symbols))),
        "suspended_stale_count": len(payload.get("suspended_stale_symbols", [])),
    }


def _compute_full_market_metrics(
    components: dict[str, Any],
    reader: Any,
    latest_trade_date: str,
    read_results: dict[str, Any] | None = None,
) -> pd.DataFrame:
    stock_names = load_cn_stock_names()
    rows: list[dict[str, Any]] = []
    batch_results = read_results or {}
    for category in MARKET_METRICS_CATEGORIES:
        for symbol in components.get(category, []):
            read_result = batch_results.get(symbol)
            if read_result is None:
                read_result = reader.read_symbol_frame(
                    symbol,
                    universe_key="full_a",
                    category=category,
                )
            frame = _read_frame_from_result(read_result)
            if (
                frame.empty
                or "trade_date" not in frame.columns
                or "close" not in frame.columns
            ):
                continue

            latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
            if latest_local_date != latest_trade_date:
                continue

            close = _price_series(frame).dropna().astype(float)
            if len(close) < 20:
                continue

            ma20 = float(close.tail(20).mean())
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
            ma120 = float(close.tail(120).mean()) if len(close) >= 120 else ma60
            latest_close = float(close.iloc[-1])
            target_price, stop_price = _derive_stage_levels(frame, latest_close)
            high20 = float(close.tail(20).max())

            rows.append(
                {
                    "symbol": symbol,
                    "name": stock_names.get(symbol, symbol),
                    "category": category,
                    "ret1": _metric_return(close, 1),
                    "ret5": _metric_return(close, 5),
                    "ret20": _metric_return(close, 20),
                    "ret60": _metric_return(close, 60),
                    "close_vs_ma20": _safe_pct(latest_close - ma20, ma20),
                    "ma20_vs_ma60": _safe_pct(ma20 - ma60, ma60),
                    "ma60_vs_ma120": _safe_pct(ma60 - ma120, ma120),
                    "dd20": _safe_pct(latest_close - high20, high20),
                    "latest_close": latest_close,
                    "stage_target_price": target_price,
                    "stage_stop_price": stop_price,
                }
            )

    metrics = pd.DataFrame(rows)
    return _score_full_market_metrics(metrics)


def _compute_market_metrics_and_breadth(
    *,
    components: dict[str, Any],
    reader: Any,
    latest_trade_date: str,
    completeness_report: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbols = list(
        dict.fromkeys(
            str(symbol).strip().upper()
            for category in MARKET_METRICS_CATEGORIES
            for symbol in list(components.get(category, []) or [])
            if str(symbol).strip()
        )
    )
    batch_reader = getattr(reader, "read_symbol_frames", None)
    if callable(batch_reader):
        read_results = dict(
            batch_reader(
                symbols,
                universe_key="full_a",
                end_date=latest_trade_date,
                columns=[
                    "ts_code",
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                ],
            )
            or {}
        )
    else:
        read_results = {}

    full_metrics = _normalize_market_metrics_frame(
        _compute_full_market_metrics(
            components=components,
            reader=reader,
            latest_trade_date=latest_trade_date,
            read_results=read_results,
        )
    )
    breadth = {
        category: _compute_category_breadth(
            category=category,
            symbols=components.get(category, []),
            reader=reader,
            latest_trade_date=latest_trade_date,
            completeness_report=completeness_report,
            read_results=read_results,
        )
        for category in MARKET_METRICS_CATEGORIES
    }
    return full_metrics, breadth


def load_or_compute_market_metrics_bundle(
    *,
    base_dir: Path,
    components: dict[str, Any],
    reader: Any,
    latest_trade_date: str,
    completeness_report: dict[str, Any],
    skip_prewarm: bool = False,
    compute_fn: Any | None = None,
) -> MarketMetricsBundle:
    snapshot_payload = _reader_snapshot_payload(reader)
    if snapshot_payload.get("healthy") is False:
        blockers = "; ".join(
            str(item)
            for item in snapshot_payload.get("blockers", [])
            if str(item).strip()
        )
        raise RuntimeError(blockers or "strict Parquet snapshot is not healthy")
    snapshot_id = str(snapshot_payload.get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise RuntimeError("strict Parquet snapshot_id is missing")
    fingerprint = _components_fingerprint(components)
    cache_dir = _market_metrics_cache_dir(
        base_dir,
        snapshot_id=snapshot_id,
        latest_trade_date=latest_trade_date,
        components_fingerprint=fingerprint,
    )
    compute = compute_fn or _compute_market_metrics_and_breadth
    if skip_prewarm:
        started = time.time()
        computed = compute(
            components=components,
            reader=reader,
            latest_trade_date=latest_trade_date,
            completeness_report=completeness_report,
        )
        if isinstance(computed, MarketMetricsBundle):
            full_metrics = computed.full_metrics
            breadth = computed.breadth
        else:
            full_metrics, breadth = computed
        full_metrics = _normalize_market_metrics_frame(full_metrics)
        return MarketMetricsBundle(
            full_metrics=full_metrics,
            breadth=breadth,
            cache_meta={
                "schema_version": MARKET_METRICS_CACHE_SCHEMA_VERSION,
                "status": "skipped",
                "cache_hit": False,
                "cache_dir": str(cache_dir),
                "row_count": int(len(full_metrics)),
                "compute_elapsed_sec": round(time.time() - started, 3),
                "snapshot_id": snapshot_id,
                "analysis_trade_date": str(latest_trade_date),
                "components_fingerprint": fingerprint,
                "reason": "skip_market_metrics_prewarm",
            },
        )
    cached = _load_cached_market_metrics_bundle(
        cache_dir=cache_dir,
        snapshot_id=snapshot_id,
        latest_trade_date=latest_trade_date,
        components_fingerprint=fingerprint,
    )
    if cached is not None:
        return cached

    started = time.time()
    computed = compute(
        components=components,
        reader=reader,
        latest_trade_date=latest_trade_date,
        completeness_report=completeness_report,
    )
    if isinstance(computed, MarketMetricsBundle):
        full_metrics = computed.full_metrics
        breadth = computed.breadth
    else:
        full_metrics, breadth = computed
    full_metrics = _normalize_market_metrics_frame(full_metrics)
    cache_meta = _write_market_metrics_cache(
        cache_dir=cache_dir,
        full_metrics=full_metrics,
        breadth=breadth,
        snapshot_id=snapshot_id,
        latest_trade_date=latest_trade_date,
        components_fingerprint=fingerprint,
        compute_elapsed_sec=time.time() - started,
    )
    return MarketMetricsBundle(
        full_metrics=full_metrics,
        breadth=breadth,
        cache_meta=cache_meta,
    )


_load_or_compute_market_metrics_bundle = load_or_compute_market_metrics_bundle


__all__ = [
    "MARKET_METRICS_CACHE_SCHEMA_VERSION",
    "MARKET_METRICS_CATEGORIES",
    "MARKET_METRICS_COMPONENT_KEYS",
    "MARKET_METRICS_OUTPUT_COLUMNS",
    "MARKET_METRICS_REQUIRED_COLUMNS",
    "MarketMetricsBundle",
    "load_or_compute_market_metrics_bundle",
    "_load_or_compute_market_metrics_bundle",
    "_compute_category_breadth",
    "_compute_full_market_metrics",
    "_compute_market_metrics_and_breadth",
    "_components_fingerprint",
    "_derive_stage_levels",
    "_load_cached_market_metrics_bundle",
    "_load_history_frame",
    "_market_metrics_cache_dir",
    "_metric_return",
    "_normalize_market_metrics_frame",
    "_price_series",
    "_read_frame_from_result",
    "_reader_snapshot_payload",
    "_score_full_market_metrics",
    "_validate_market_metrics_frame",
    "_write_market_metrics_cache",
]
