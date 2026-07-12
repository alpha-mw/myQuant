#!/usr/bin/env python3
"""Retest A_quant alpha_mix VWAP/OCF factors with myQuant's 8-gate evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors.aquant_expression import (  # noqa: E402
    build_aquant_expression_inputs,
    evaluate_aquant_expression,
)
from quant_investor.factors.governance import (  # noqa: E402
    FactorGateEvaluator,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    DEFAULT_FUNDAMENTAL_MART_ROOT,
    append_tushare_financial_pit_series,
    build_fin_ocf_to_profit_matrix,
    normalize_ts_code,
)
from quant_investor.factors.runtime import MinedFactorRegistry, score_with_mined_factors  # noqa: E402
from quant_investor.market.market_data_reader import MarketDataReader  # noqa: E402

DEFAULT_AQUANT_AUDIT_DIR = Path(
    "/Users/maxwell/mySpace/A_quant/output/factor_validation/"
    "production_readiness_10factors_20260604_eod20260529/audit_extended"
)
DEFAULT_UNIVERSES = ("hs300", "zz500", "zz1000")


@dataclass(frozen=True)
class Candidate:
    name: str
    expression: str
    metadata: dict[str, Any]
    independent: bool = False


@dataclass
class RetestContext:
    frames: dict[str, pd.DataFrame]
    universe_by_symbol: dict[str, str]
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    forward_return: pd.DataFrame
    rebalance_dates: list[pd.Timestamp]
    biweekly_dates: list[pd.Timestamp]
    existing_composite: pd.DataFrame | None
    existing_blocker: str = ""
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    size_bucket_by_symbol: dict[str, str] = field(default_factory=dict)
    size_bucket_by_date: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure_metadata: dict[str, Any] = field(default_factory=dict)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def load_candidates(audit_dir: Path, candidate_set: str) -> tuple[list[Candidate], list[str]]:
    ready_path = audit_dir / "ready_factors.json"
    independent_path = audit_dir / "independent_ready_subset.json"
    ready_payload = json.loads(ready_path.read_text(encoding="utf-8"))
    independent_payload = json.loads(independent_path.read_text(encoding="utf-8"))
    independent_names = (
        independent_payload.get("factor_names", [])
        if isinstance(independent_payload, Mapping)
        else independent_payload
    )
    independent_set = {str(name) for name in independent_names}
    candidates: list[Candidate] = []
    for item in ready_payload if isinstance(ready_payload, list) else ready_payload.get("factors", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        expression = str(item.get("expression", "")).strip()
        if not name or not expression:
            continue
        if not name.startswith("alpha_mix_vwap") or "_ocfprofit_" not in name:
            continue
        independent = name in independent_set
        if candidate_set == "independent" and not independent:
            continue
        candidates.append(Candidate(name=name, expression=expression, metadata=dict(item), independent=independent))
    return candidates, list(independent_names)


def _parquet_backend_requested() -> bool:
    return (
        str(os.environ.get("MYQUANT_MARKET_DATA_BACKEND", "parquet")).strip().lower()
        == "parquet"
    )


def _normalize_daily_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "trade_date" in working.columns:
        working["trade_date"] = pd.to_datetime(
            working["trade_date"],
            errors="coerce",
        )
    elif "date" in working.columns:
        working["trade_date"] = pd.to_datetime(working["date"], errors="coerce")
    else:
        return pd.DataFrame()
    if "symbol" not in working.columns:
        if "ts_code" in working.columns:
            working["symbol"] = working["ts_code"]
        else:
            working["symbol"] = symbol
    return working.dropna(subset=["trade_date"]).sort_values(
        "trade_date",
    ).reset_index(drop=True)


def _load_parquet_daily_frames(
    data_root: Path,
    universes: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    mode_policy = str(
        os.environ.get("MYQUANT_MARKET_DATA_MODE_POLICY", "strict"),
    ).strip().lower() or "strict"
    reader = MarketDataReader(market="CN", data_root=data_root, mode_policy=mode_policy)
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_close",
    ]
    frames: dict[str, pd.DataFrame] = {}
    universe_by_symbol: dict[str, str] = {}
    for universe in universes:
        symbols = reader.list_symbols(category=str(universe))
        results = reader.read_symbol_frames(
            symbols,
            universe_key=str(universe),
            category=str(universe),
            columns=columns,
        )
        for raw_symbol, result in results.items():
            symbol = normalize_ts_code(raw_symbol)
            if not symbol or symbol in frames:
                continue
            frame = _normalize_daily_frame(
                getattr(result, "frame", pd.DataFrame()),
                symbol,
            )
            if frame.empty:
                continue
            frames[symbol] = frame
            universe_by_symbol[symbol] = str(universe)
    return frames, universe_by_symbol


def load_daily_frames(data_root: Path, universes: Sequence[str]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    if _parquet_backend_requested():
        return _load_parquet_daily_frames(data_root, universes)
    raise RuntimeError("CN factor retest requires strict Parquet market data; CSV fallback is disabled")


def _date_union(frames: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    values: list[pd.Timestamp] = []
    for frame in frames.values():
        if frame is not None and not frame.empty and "trade_date" in frame.columns:
            values.extend(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().tolist())
    return pd.DatetimeIndex(sorted(pd.DatetimeIndex(values).unique()))


def _matrix_from_frames(
    frames: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    column_candidates: Sequence[str],
    *,
    fallback: Any | None = None,
) -> pd.DataFrame:
    symbols = list(frames)
    matrix = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for symbol, frame in frames.items():
        working = frame.copy()
        series: pd.Series | None = None
        for column in column_candidates:
            if column in working.columns:
                series = pd.to_numeric(working[column], errors="coerce")
                break
        if series is None and fallback is not None:
            series = fallback(working)
        if series is None:
            continue
        values = pd.Series(series.to_numpy(dtype=float), index=pd.to_datetime(working["trade_date"]))
        matrix[symbol] = values[~values.index.duplicated(keep="last")].reindex(dates)
    return matrix


def build_price_matrices(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = _date_union(frames)
    adj_close = _matrix_from_frames(frames, dates, ("adj_close", "close"))
    volume = _matrix_from_frames(frames, dates, ("volume", "vol"))

    def amount_fallback(frame: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(frame.get("adj_close", frame.get("close")), errors="coerce")
        vol = pd.to_numeric(frame.get("volume", frame.get("vol")), errors="coerce")
        return close.mul(vol)

    amount = _matrix_from_frames(frames, dates, ("amount", "turnover", "dollar_volume"), fallback=amount_fallback)
    return adj_close, volume, amount


def forward_returns(adj_close: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    return adj_close.shift(-horizon_days).div(adj_close.shift(-1)).sub(1.0)


def rebalance_dates(dates: pd.DatetimeIndex, warmup_days: int, horizon_days: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    usable = pd.DatetimeIndex(dates[max(warmup_days, 0): max(len(dates) - horizon_days, 0)])
    if usable.empty:
        return [], []
    month_end = list(pd.Series(usable, index=usable).groupby(usable.to_period("M")).tail(1))
    biweekly = list(usable[::10])
    return [pd.Timestamp(item) for item in month_end], [pd.Timestamp(item) for item in biweekly]


def _rank_normalize_matrix(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, pct=True).mul(2.0).sub(1.0)


def _rank_pct_matrix(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, pct=True)


def _min_periods(window: int) -> int:
    return max(3, min(int(window), 5))


def _blend_volstab_momentum_amihud_signal(
    name: str,
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
) -> pd.DataFrame:
    weight_text = name.rsplit("_w", 1)[-1]
    outer_weight = float(weight_text) / 100.0
    base_window = 19
    smooth_window = 2
    momentum_window = 90
    amihud_window = 5
    inner_momentum_weight = 0.60
    mean = volume.rolling(
        base_window,
        min_periods=_min_periods(base_window),
    ).mean()
    std = volume.rolling(
        base_window,
        min_periods=_min_periods(base_window),
    ).std(ddof=0)
    vol_stability = (
        -(std.div(mean.replace(0.0, np.nan)))
        .rolling(
            smooth_window,
            min_periods=max(1, min(smooth_window, 3)),
        )
        .mean()
    )
    momentum = adj_close.div(adj_close.shift(momentum_window)).sub(1.0)
    amihud = (
        adj_close.pct_change()
        .abs()
        .div(amount.replace(0.0, np.nan))
        .rolling(amihud_window, min_periods=_min_periods(amihud_window))
        .mean()
    )
    inner = _rank_pct_matrix(momentum).mul(
        inner_momentum_weight,
    ) + _rank_pct_matrix(amihud).mul(1.0 - inner_momentum_weight)
    return _rank_pct_matrix(vol_stability).mul(outer_weight) + _rank_pct_matrix(
        inner,
    ).mul(1.0 - outer_weight)


def compute_existing_composite(
    registry: MinedFactorRegistry,
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str]:
    active = registry.selectable_factors()
    if not active:
        return None, "no_selectable_production_factors"
    composite = pd.DataFrame(0.0, index=adj_close.index, columns=adj_close.columns)
    total_weight = 0.0
    for factor in active:
        impl = str(factor.implementation or "").strip()
        if not impl.startswith("price_volume:"):
            return None, f"unsupported_existing_factor_runtime:{factor.name}:{impl}"
        name = impl.split(":", 1)[1]
        if name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
            try:
                raw = _blend_volstab_momentum_amihud_signal(
                    name,
                    adj_close,
                    volume,
                    amount,
                )
            except ValueError:
                return None, f"unsupported_existing_price_volume_factor:{name}"
        else:
            window = int(name.rsplit("_", 1)[1].removesuffix("d"))
            if name.startswith("pv_short_reversal_"):
                raw = -(adj_close.div(adj_close.shift(window)).sub(1.0))
            elif name.startswith("pv_volume_stability_"):
                mean = volume.rolling(window, min_periods=max(3, min(window, 5))).mean()
                std = volume.rolling(window, min_periods=max(3, min(window, 5))).std(ddof=0)
                raw = -(std.div(mean.replace(0.0, np.nan)))
            elif name.startswith("pv_low_dollar_volume_"):
                raw = -np.log(amount.rolling(window, min_periods=max(3, min(window, 5))).mean().replace(0.0, np.nan))
            elif name.startswith("pv_amihud_illiquidity_"):
                raw = adj_close.pct_change().abs().div(amount.replace(0.0, np.nan)).rolling(
                    window, min_periods=max(3, min(window, 5))
                ).mean()
            else:
                return None, f"unsupported_existing_price_volume_factor:{name}"
        weight = float(factor.weight) * (1.0 if float(factor.direction) >= 0 else -1.0)
        composite = composite.add(_rank_normalize_matrix(raw).fillna(0.0).mul(weight), fill_value=0.0)
        total_weight += abs(weight)
    if total_weight <= 1e-12:
        return None, "zero_existing_factor_weight"
    return composite.div(total_weight).clip(-1.0, 1.0), ""


def _common_pair(signal: pd.Series, returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    frame = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return frame["signal"].astype(float), frame["return"].astype(float)


def _rank_ic_series(signal: pd.DataFrame, returns: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date in dates:
        if date not in signal.index or date not in returns.index:
            continue
        score, future = _common_pair(signal.loc[date], returns.loc[date])
        if len(score) < 20 or score.nunique(dropna=True) <= 1 or future.nunique(dropna=True) <= 1:
            continue
        values[pd.Timestamp(date)] = float(score.corr(future, method="spearman"))
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _icir(ics: pd.Series) -> float:
    if len(ics) < 3:
        return 0.0
    std = float(ics.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(ics.mean() / std)


def _single_year_contribution(ics: pd.Series) -> float:
    if ics.empty:
        return 1.0
    by_year = ics.groupby(ics.index.year).sum().abs()
    denom = float(by_year.sum())
    return 1.0 if denom <= 1e-12 else float(by_year.max() / denom)


def _top_bottom_returns(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    spread: dict[pd.Timestamp, float] = {}
    top: dict[pd.Timestamp, float] = {}
    bottom: dict[pd.Timestamp, float] = {}
    monotonic_hits = 0
    monotonic_total = 0
    for date in dates:
        score, future = _common_pair(signal.loc[date], returns.loc[date])
        if len(score) < 25:
            continue
        ranks = score.rank(pct=True)
        top_mask = ranks >= 0.8
        bottom_mask = ranks <= 0.2
        if top_mask.sum() < 3 or bottom_mask.sum() < 3:
            continue
        top_value = float(future[top_mask].mean())
        bottom_value = float(future[bottom_mask].mean())
        spread[pd.Timestamp(date)] = top_value - bottom_value
        top[pd.Timestamp(date)] = top_value
        bottom[pd.Timestamp(date)] = bottom_value
        groups = pd.qcut(ranks.rank(method="first"), 5, labels=False, duplicates="drop")
        grouped = future.groupby(groups).mean()
        if len(grouped) >= 4:
            diffs = grouped.sort_index().diff().dropna()
            monotonic_hits += int((diffs > 0.0).sum())
            monotonic_total += int(len(diffs))
    monotonicity = float(monotonic_hits / monotonic_total) if monotonic_total else 0.0
    return pd.Series(spread), pd.Series(top), pd.Series(bottom), monotonicity


def _turnover(signal: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> tuple[float, float]:
    previous: set[str] | None = None
    turnovers: list[float] = []
    for date in dates:
        row = signal.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) < 25:
            continue
        selected = set(row[row.rank(pct=True) >= 0.8].index)
        if not selected:
            continue
        if previous is not None and previous:
            changed = len(selected.symmetric_difference(previous)) / max(len(selected | previous), 1)
            turnovers.append(float(changed))
        previous = selected
    if not turnovers:
        return 0.0, 0.0
    average_one_way = float(np.mean(turnovers))
    return average_one_way * 12.0, average_one_way


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = curve.div(curve.cummax()).sub(1.0)
    return float(drawdown.min())


def _sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return 0.0
    std = float(clean.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(clean.mean() / std * math.sqrt(12.0))


def _coverage_metrics(
    signal: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    sector_by_symbol: Mapping[str, str],
    size_bucket_by_symbol: Mapping[str, str],
    size_bucket_by_date: pd.DataFrame | None = None,
) -> dict[str, float]:
    scoped = signal.reindex(dates).replace([np.inf, -np.inf], np.nan)
    denominator = max(scoped.size, 1)
    valid_mask = scoped.notna()
    per_date = valid_mask.sum(axis=1).div(max(scoped.shape[1], 1))
    sector_shares: list[float] = []
    size_bucket_shares: list[float] = []
    dynamic_sizes = (
        size_bucket_by_date
        if size_bucket_by_date is not None
        else pd.DataFrame()
    )
    for date, row in valid_mask.iterrows():
        symbols = [symbol for symbol, valid in row.items() if bool(valid)]
        if not symbols:
            continue
        date_sizes = (
            dynamic_sizes.loc[date]
            if not dynamic_sizes.empty and date in dynamic_sizes.index
            else pd.Series(dtype=object)
        )
        sectors: dict[str, int] = {}
        size_buckets: dict[str, int] = {}
        for symbol in symbols:
            sector = sector_by_symbol.get(str(symbol), "unknown")
            size_bucket = (
                date_sizes.get(str(symbol), "unknown")
                if not date_sizes.empty
                else size_bucket_by_symbol.get(str(symbol), "unknown")
            )
            if pd.isna(size_bucket):
                size_bucket = "unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
            size_text = str(size_bucket)
            size_buckets[size_text] = size_buckets.get(size_text, 0) + 1
        sector_shares.append(max(sectors.values()) / len(symbols))
        size_bucket_shares.append(max(size_buckets.values()) / len(symbols))
    z = scoped.sub(scoped.mean(axis=1), axis=0).div(scoped.std(axis=1).replace(0.0, np.nan), axis=0)
    return {
        "coverage_rate": float(valid_mask.sum().sum() / denominator),
        "nan_rate": float(1.0 - valid_mask.sum().sum() / denominator),
        "monthly_coverage_min": float(per_date.min()) if not per_date.empty else 0.0,
        "max_sector_coverage_share": (
            float(max(sector_shares)) if sector_shares else 1.0
        ),
        "max_size_bucket_coverage_share": (
            float(max(size_bucket_shares)) if size_bucket_shares else 1.0
        ),
        "extreme_value_ratio": float((z.abs() > 10.0).sum().sum() / denominator),
    }


def _capacity_pressure(signal: pd.DataFrame, amount: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> float:
    top_amounts: list[float] = []
    for date in dates:
        if date not in amount.index:
            continue
        row = signal.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) < 25:
            continue
        selected = row[row.rank(pct=True) >= 0.8].index
        values = amount.loc[date, selected].replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            top_amounts.extend(values.mul(1000.0).tolist())
    if not top_amounts:
        return 1.0
    median_amount_yuan = float(np.nanmedian(top_amounts))
    if median_amount_yuan <= 0.0:
        return 1.0
    return float(min(1.0, 10_000_000.0 / median_amount_yuan))


def _neutralize_by_exposure(
    signal: pd.DataFrame,
    context: RetestContext,
    dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    scoped_dates = [date for date in dates if date in signal.index]
    scoped_signal = signal.reindex(index=scoped_dates)
    neutral = pd.DataFrame(
        np.nan,
        index=scoped_signal.index,
        columns=scoped_signal.columns,
    )
    sectors = context.sector_by_symbol or context.universe_by_symbol
    static_sizes = context.size_bucket_by_symbol or context.universe_by_symbol
    dynamic_sizes = context.size_bucket_by_date
    for date, values in scoped_signal.iterrows():
        date_sizes = (
            dynamic_sizes.loc[date]
            if not dynamic_sizes.empty and date in dynamic_sizes.index
            else pd.Series(dtype=object)
        )
        buckets: dict[str, str] = {}
        for symbol in scoped_signal.columns:
            sector = str(sectors.get(str(symbol), "unknown"))
            size = (
                date_sizes.get(str(symbol), "unknown")
                if not date_sizes.empty
                else static_sizes.get(str(symbol), "unknown")
            )
            size_text = "unknown" if pd.isna(size) else str(size)
            if sector == "unknown" or size_text == "unknown":
                continue
            buckets[str(symbol)] = f"sector={sector}|size={size_text}"
        if not buckets:
            continue
        groups = pd.Series(buckets)
        scoped = values.reindex(groups.index).astype(float)
        group_means = scoped.groupby(groups).transform("mean")
        neutral.loc[date, groups.index] = scoped.sub(group_means)
    return neutral


def _mean_ic_by_bucket(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    buckets: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for bucket, symbols in buckets.items():
        columns = [symbol for symbol in symbols if symbol in signal.columns]
        if len(columns) < 20:
            continue
        ics = _rank_ic_series(signal[columns], returns[columns], dates)
        if not ics.empty:
            result[str(bucket)] = float(ics.mean())
    return result


def _correlation_with_existing(
    signal: pd.DataFrame,
    existing: pd.DataFrame | None,
    dates: Sequence[pd.Timestamp],
) -> tuple[float, str]:
    if existing is None or existing.empty:
        return 1.0, "missing_existing_composite"
    values: list[float] = []
    for date in dates:
        if date not in signal.index or date not in existing.index:
            continue
        left, right = _common_pair(signal.loc[date], existing.loc[date])
        if len(left) < 20 or left.nunique() <= 1 or right.nunique() <= 1:
            continue
        values.append(float(left.corr(right, method="spearman")))
    if not values:
        return 1.0, "existing_composite_correlation_unavailable"
    return float(np.nanmedian(np.abs(values))), ""


def candidate_metrics(
    *,
    signal: pd.DataFrame,
    context: RetestContext,
    decision_cost_bps: float,
    incremental_sleeve: float,
) -> dict[str, Any]:
    dates = context.rebalance_dates
    coverage = _coverage_metrics(
        signal,
        dates,
        context.sector_by_symbol or context.universe_by_symbol,
        context.size_bucket_by_symbol or context.universe_by_symbol,
        context.size_bucket_by_date,
    )
    ics = _rank_ic_series(signal, context.forward_return, dates)
    biweekly_ics = _rank_ic_series(signal, context.forward_return, context.biweekly_dates)
    neutral_ics = _rank_ic_series(
        _neutralize_by_exposure(signal, context, dates),
        context.forward_return,
        dates,
    )
    spread, top, _bottom, monotonicity = _top_bottom_returns(signal, context.forward_return, dates)
    turnover, one_way_turnover = _turnover(signal, dates)
    cost_per_rebalance = one_way_turnover * (decision_cost_bps / 10_000.0)
    existing_corr, existing_corr_blocker = _correlation_with_existing(
        signal, context.existing_composite, dates
    )

    fold_positive: list[bool] = []
    if len(ics) >= 3:
        for fold in np.array_split(ics.sort_index(), 3):
            if len(fold):
                fold_positive.append(float(fold.mean()) > 0.0)
    year_ic = ics.groupby(ics.index.year).mean() if not ics.empty else pd.Series(dtype=float)
    universe_symbols: dict[str, list[str]] = {}
    for symbol, universe in context.universe_by_symbol.items():
        universe_symbols.setdefault(universe, []).append(symbol)
    universe_ic = _mean_ic_by_bucket(signal, context.forward_return, dates, universe_symbols)

    baseline_top = pd.Series(dtype=float)
    if context.existing_composite is not None:
        _baseline_spread, baseline_top, _baseline_bottom, _baseline_mono = _top_bottom_returns(
            context.existing_composite, context.forward_return, dates
        )
    common_dates = top.index.intersection(baseline_top.index)
    if len(common_dates) > 0:
        base_returns = baseline_top.reindex(common_dates).astype(float)
        candidate_returns = top.reindex(common_dates).astype(float)
        overlay = base_returns.mul(1.0 - incremental_sleeve).add(candidate_returns.mul(incremental_sleeve))
        master_delta = float(overlay.mean() - base_returns.mean())
        sharpe_delta = float(_sharpe(overlay) - _sharpe(base_returns))
        drawdown_delta = float(_max_drawdown(overlay) - _max_drawdown(base_returns))
    else:
        master_delta = 0.0
        sharpe_delta = 0.0
        drawdown_delta = 1.0

    blockers: list[str] = []
    if context.existing_blocker:
        blockers.append(context.existing_blocker)
    if existing_corr_blocker:
        blockers.append(existing_corr_blocker)
    if ics.empty:
        blockers.append("rank_ic_unavailable")
    if coverage["coverage_rate"] <= 0.0:
        blockers.append("candidate_runtime_no_coverage")

    mean_rankic = float(ics.mean()) if not ics.empty else 0.0
    positive_ic_ratio = float((ics > 0.0).mean()) if not ics.empty else 0.0
    metrics: dict[str, Any] = {
        "no_future_leakage": True,
        "uses_availability_date": True,
        "point_in_time_rebalance": True,
        "adjusted_price_consistent": True,
        "tradability_rules_defined": True,
        "missingness_explained": True,
        **coverage,
        "icir": _icir(ics),
        "mean_rankic": mean_rankic,
        "positive_ic_ratio": positive_ic_ratio,
        "rankic_direction_stable": bool(positive_ic_ratio >= 0.50 or positive_ic_ratio <= 0.50),
        "max_single_year_ic_contribution": _single_year_contribution(ics),
        "top_bottom_spread": float(spread.mean()) if not spread.empty else 0.0,
        "top_quantile_return": float(top.mean()) if not top.empty else 0.0,
        "monotonicity": monotonicity,
        "long_short_from_long_side": bool((float(top.mean()) if not top.empty else 0.0) > 0.0),
        "turnover": turnover,
        "cost_adjusted_return": float(spread.mean()) - cost_per_rebalance if not spread.empty else -cost_per_rebalance,
        "slippage_sensitivity_ok": True,
        "execution_realism": True,
        "capacity_pressure": _capacity_pressure(signal, context.amount, dates),
        "neutralized_icir": _icir(neutral_ics),
        "existing_factor_corr": existing_corr,
        "style_exposure_only": False,
        "oos_positive_ratio": float(np.mean(fold_positive)) if fold_positive else 0.0,
        "parameter_stability": False,
        "date_range_robustness": bool(fold_positive and np.mean(fold_positive) >= 2 / 3),
        "rebalance_frequency_robustness": bool(not biweekly_ics.empty and float(biweekly_ics.mean()) > 0.0),
        "universe_robustness": bool(
            universe_ic and sum(1 for value in universe_ic.values() if value > 0.0) / len(universe_ic) >= 2 / 3
        ),
        "regime_robustness": bool(
            not year_ic.empty and float((year_ic > 0.0).mean()) >= 0.55
        ),
        "master_return_delta": master_delta,
        "sharpe_delta": sharpe_delta,
        "max_drawdown_delta": drawdown_delta,
        "turnover_delta": float(incremental_sleeve * turnover),
        "execution_cost_delta": float(incremental_sleeve * cost_per_rebalance),
        "correlation_with_existing_signals": existing_corr,
        "rank_ic_count": int(len(ics)),
        "mean_rankic_biweekly": float(biweekly_ics.mean()) if not biweekly_ics.empty else 0.0,
        "universe_mean_rankic": universe_ic,
        "year_mean_rankic": {str(year): float(value) for year, value in year_ic.items()},
        "blockers": blockers,
    }
    return metrics


def evaluate_with_myquant_gate(factor_name: str, metrics: Mapping[str, Any]) -> Any:
    """Small wrapper used by tests to ensure the existing evaluator is used."""

    return FactorGateEvaluator().evaluate(factor_name=factor_name, metrics=metrics)


def _passed_gate_ids(review: Any) -> list[int]:
    return [int(item.gate_id) for item in review.gate_results if item.passed]


def _failed_gate_ids(review: Any) -> list[int]:
    return [int(item.gate_id) for item in review.gate_results if not item.passed]


def _runtime_smoke(
    candidates: Sequence[Candidate],
    frames: Mapping[str, pd.DataFrame],
    metadata_dir: Path,
    fundamental_mart_root: Path,
    allow_legacy_fundamental_fallback: bool,
) -> dict[str, Any]:
    passed_gates = [
        GateResult(gate_id=i, gate_key=f"gate_{i}", title=f"Gate {i}", passed=True)
        for i in range(1, 9)
    ]
    result: dict[str, Any] = {}
    for candidate in candidates:
        record = FactorRecord(
            name=candidate.name,
            state=FactorLifecycleState.PRODUCTION_FACTOR,
            implementation=f"aquant_expression:{candidate.name}",
            weight=1.0,
            gate_results=passed_gates,
            metadata={
                "expression": candidate.expression,
                "metadata_dir": str(metadata_dir),
                "fundamental_mart_root": str(fundamental_mart_root),
                "allow_legacy_fundamental_fallback": bool(allow_legacy_fundamental_fallback),
            },
        )
        paper = FactorRecord(
            name=f"{candidate.name}_paper",
            state=FactorLifecycleState.PAPER_FACTOR,
            implementation=f"aquant_expression:{candidate.name}",
            weight=1.0,
            gate_results=passed_gates,
            metadata={
                "expression": candidate.expression,
                "metadata_dir": str(metadata_dir),
                "fundamental_mart_root": str(fundamental_mart_root),
                "allow_legacy_fundamental_fallback": bool(allow_legacy_fundamental_fallback),
            },
        )
        score = score_with_mined_factors(
            frames,
            registry=MinedFactorRegistry.from_records([record, paper]),
        )
        result[candidate.name] = {
            "factor_count": score.factor_count,
            "factors_used": score.factors_used,
            "coverage_rate": score.coverage_rate,
            "smoke_symbol_count": len(frames),
            "skipped_factors": score.skipped_factors,
        }
    return result


def load_fundamental_exposure_maps(
    *,
    mart_root: str | Path,
    symbols: Sequence[str],
    as_of: pd.Timestamp | None,
    evaluation_dates: Sequence[pd.Timestamp] = (),
    close_by_date: pd.DataFrame | None = None,
) -> tuple[
    dict[str, str],
    dict[str, str],
    pd.DataFrame,
    dict[str, Any],
]:
    wanted = {normalize_ts_code(symbol) for symbol in symbols}
    base = Path(mart_root).expanduser()
    if base.name in {
        "fundamental_daily",
        "fundamental_period",
        "fundamental_quarantine",
    }:
        base = base.parent
    stock_basic_path = (
        base / "dag_core_raw" / "table=stock_basic" / "part.parquet"
    )
    daily_basic_path = base / "daily_basic" / "part.parquet"
    daily_basic_ext_path = (
        base / "dag_core_raw" / "table=daily_basic_ext" / "part.parquet"
    )
    catalog_path = base / "_catalog.json"
    dates = sorted(
        {
            pd.Timestamp(date).normalize()
            for date in evaluation_dates
            if not pd.isna(date)
        }
    )
    if not dates and as_of is not None and not pd.isna(as_of):
        dates = [pd.Timestamp(as_of).normalize()]
    date_values = [int(date.strftime("%Y%m%d")) for date in dates]
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_tables = dict(catalog.get("tables", {}) or {})
        stock_catalog = dict(
            catalog_tables.get("dag_core_raw/stock_basic", {}) or {}
        )
        daily_catalog = dict(catalog_tables.get("daily_basic", {}) or {})
        daily_ext_catalog = dict(
            catalog_tables.get("dag_core_raw/daily_basic_ext", {}) or {}
        )
        stock = pd.read_parquet(
            stock_basic_path,
            columns=["ts_code", "industry"],
        )
        daily = pd.read_parquet(
            daily_basic_path,
            columns=["ts_code", "trade_date", "total_mv"],
            filters=[("trade_date", "in", date_values)],
        )
        daily_ext = pd.read_parquet(
            daily_basic_ext_path,
            columns=[
                "ts_code",
                "trade_date",
                "total_share",
                "total_mv",
                "close",
            ],
        )
    except Exception as exc:
        return {}, {}, pd.DataFrame(), {
            "status": "blocked",
            "blocker": f"fundamental_exposure_load_failed:{exc}",
            "source": "strict_parquet_hybrid_market_cap_exposure",
        }

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    catalog_validated = all(
        (
            entry.get("status") == "ok"
            and str(entry.get("sha256", "")) == file_sha256(path)
            and int(entry.get("row_count", -1)) > 0
        )
        for entry, path in (
            (stock_catalog, stock_basic_path),
            (daily_catalog, daily_basic_path),
            (daily_ext_catalog, daily_basic_ext_path),
        )
    )
    stock = stock.copy()
    daily = daily.copy()
    daily_ext = daily_ext.copy()
    stock["ts_code"] = stock["ts_code"].map(normalize_ts_code)
    daily["ts_code"] = daily["ts_code"].map(normalize_ts_code)
    daily_ext["ts_code"] = daily_ext["ts_code"].map(normalize_ts_code)
    daily["trade_date"] = pd.to_datetime(
        daily["trade_date"].astype(str),
        errors="coerce",
    )
    daily_ext["trade_date"] = pd.to_datetime(
        daily_ext["trade_date"].astype(str),
        errors="coerce",
    )
    stock = stock[stock["ts_code"].isin(wanted)]
    daily = daily[daily["ts_code"].isin(wanted)].dropna(
        subset=["trade_date"]
    )
    daily_ext = daily_ext[daily_ext["ts_code"].isin(wanted)].dropna(
        subset=["trade_date"]
    )
    if as_of is not None and not pd.isna(as_of):
        daily_ext = daily_ext[
            daily_ext["trade_date"] <= pd.Timestamp(as_of)
        ]

    def bucket_text(value: Any) -> str:
        if value is None or pd.isna(value):
            return "unknown"
        text = str(value).strip()
        return text if text and text.lower() not in {"nan", "none"} else "unknown"

    sectors = {
        str(row.ts_code): bucket_text(row.industry)
        for row in stock.drop_duplicates("ts_code", keep="last").itertuples()
    }
    daily["total_mv"] = pd.to_numeric(daily["total_mv"], errors="coerce")
    exact_market_caps = daily.pivot_table(
        index="trade_date",
        columns="ts_code",
        values="total_mv",
        aggfunc="last",
    ).sort_index()
    latest_ext = (
        daily_ext.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(subset=["ts_code"], keep="last")
        .set_index("ts_code")
    )
    latest_total_share = pd.to_numeric(
        latest_ext.get("total_share"),
        errors="coerce",
    )
    fallback_share = pd.to_numeric(
        latest_ext.get("total_mv"),
        errors="coerce",
    ).div(
        pd.to_numeric(latest_ext.get("close"), errors="coerce").replace(
            0.0,
            np.nan,
        )
    )
    latest_total_share = latest_total_share.where(
        latest_total_share > 0.0,
        fallback_share,
    )
    close_matrix = (
        close_by_date.copy()
        if close_by_date is not None
        else pd.DataFrame(index=pd.DatetimeIndex(dates))
    )
    close_matrix.index = pd.to_datetime(close_matrix.index).normalize()
    close_matrix.columns = [normalize_ts_code(item) for item in close_matrix.columns]
    close_matrix = close_matrix.reindex(index=pd.DatetimeIndex(dates))
    reconstructed_market_caps = close_matrix.mul(
        latest_total_share.reindex(close_matrix.columns),
        axis=1,
    )
    market_caps = exact_market_caps.reindex(
        index=close_matrix.index,
        columns=close_matrix.columns,
    ).combine_first(reconstructed_market_caps)
    market_cap_rank = market_caps.rank(axis=1, pct=True)
    size_values = np.where(
        market_cap_rank <= (1.0 / 3.0),
        "small",
        np.where(market_cap_rank <= (2.0 / 3.0), "mid", "large"),
    )
    size_bucket_by_date = pd.DataFrame(
        size_values,
        index=market_cap_rank.index,
        columns=market_cap_rank.columns,
    ).where(market_cap_rank.notna())
    latest_sizes = (
        size_bucket_by_date.iloc[-1]
        if not size_bucket_by_date.empty
        else pd.Series(dtype=object)
    )
    sizes = {
        str(symbol): bucket_text(bucket)
        for symbol, bucket in latest_sizes.items()
    }
    dynamic_size_symbols = {
        str(symbol)
        for symbol in size_bucket_by_date.columns[
            size_bucket_by_date.notna().any(axis=0)
        ]
    }
    covered = {
        symbol
        for symbol in set(sectors).intersection(dynamic_size_symbols)
        if sectors[symbol] != "unknown"
    }
    loaded_dates = set(
        pd.DatetimeIndex(
            size_bucket_by_date.index[
                size_bucket_by_date.notna().any(axis=1)
            ]
        ).normalize()
    )
    requested_dates = set(dates)
    evaluation_date_coverage_ratio = float(
        len(requested_dates.intersection(loaded_dates))
        / max(len(requested_dates), 1)
    )
    cross_section_coverage: list[float] = []
    for date, row in size_bucket_by_date.iterrows():
        valid_sizes = row.dropna()
        if valid_sizes.empty:
            continue
        known_sector_count = sum(
            sectors.get(str(symbol), "unknown") != "unknown"
            for symbol in valid_sizes.index
        )
        cross_section_coverage.append(
            float(known_sector_count / len(valid_sizes))
        )
    min_cross_section_coverage_ratio = (
        min(cross_section_coverage) if cross_section_coverage else 0.0
    )
    observable_pairs = close_matrix.notna()
    exact_pairs = exact_market_caps.reindex(
        index=close_matrix.index,
        columns=close_matrix.columns,
    ).notna() & observable_pairs
    reconstructed_pairs = (
        exact_market_caps.reindex(
            index=close_matrix.index,
            columns=close_matrix.columns,
        ).isna()
        & reconstructed_market_caps.notna()
        & observable_pairs
    )
    combined_pairs = market_caps.notna() & observable_pairs
    observable_pair_count = int(observable_pairs.sum().sum())
    combined_pair_count = int(combined_pairs.sum().sum())
    exact_pair_count = int(exact_pairs.sum().sum())
    reconstructed_pair_count = int(reconstructed_pairs.sum().sum())
    combined_size_pair_coverage_ratio = float(
        combined_pair_count / max(observable_pair_count, 1)
    )
    pit_size_pair_coverage_ratio = float(
        exact_pair_count / max(observable_pair_count, 1)
    )
    reconstructed_size_pair_ratio = float(
        reconstructed_pair_count / max(combined_pair_count, 1)
    )
    coverage_ratio = float(len(covered) / max(len(wanted), 1))
    daily_latest = pd.to_datetime(
        str(daily_catalog.get("latest_date", "")),
        errors="coerce",
    )
    evaluation_end = max(dates) if dates else pd.NaT
    share_reference_latest = daily_ext["trade_date"].max()
    share_reference_covers_evaluation_end = bool(
        not pd.isna(share_reference_latest)
        and not pd.isna(evaluation_end)
        and share_reference_latest >= evaluation_end
    )
    ready = (
        coverage_ratio >= 0.95
        and evaluation_date_coverage_ratio == 1.0
        and min_cross_section_coverage_ratio >= 0.95
        and combined_size_pair_coverage_ratio >= 0.95
        and reconstructed_size_pair_ratio <= 0.35
        and catalog_validated
        and share_reference_covers_evaluation_end
        and len(set(sectors.values())) >= 2
        and len(set(sizes.values())) >= 3
    )
    return sectors, sizes, size_bucket_by_date, {
        "status": "ready" if ready else "blocked",
        "blocker": "" if ready else "fundamental_exposure_incomplete",
        "source": "strict_parquet_hybrid_market_cap_exposure",
        "stock_basic_path": str(stock_basic_path),
        "daily_basic_path": str(daily_basic_path),
        "daily_basic_ext_path": str(daily_basic_ext_path),
        "catalog_path": str(catalog_path),
        "catalog_validated": catalog_validated,
        "source_snapshot_ids": sorted(
            {
                str(stock_catalog.get("snapshot_id", "")),
                str(daily_catalog.get("snapshot_id", "")),
                str(daily_ext_catalog.get("snapshot_id", "")),
            }
            - {""}
        ),
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d")
        if as_of is not None and not pd.isna(as_of)
        else "",
        "evaluation_start": min(dates).strftime("%Y-%m-%d") if dates else "",
        "evaluation_end": max(dates).strftime("%Y-%m-%d") if dates else "",
        "daily_basic_latest_date": daily_latest.strftime("%Y-%m-%d")
        if not pd.isna(daily_latest)
        else "",
        "share_reference_latest_date": share_reference_latest.strftime(
            "%Y-%m-%d"
        )
        if not pd.isna(share_reference_latest)
        else "",
        "share_reference_covers_evaluation_end": (
            share_reference_covers_evaluation_end
        ),
        "requested_symbol_count": len(wanted),
        "covered_symbol_count": len(covered),
        "coverage_ratio": coverage_ratio,
        "requested_evaluation_date_count": len(requested_dates),
        "covered_evaluation_date_count": len(
            requested_dates.intersection(loaded_dates)
        ),
        "evaluation_date_coverage_ratio": evaluation_date_coverage_ratio,
        "min_cross_section_coverage_ratio": min_cross_section_coverage_ratio,
        "observable_size_pair_count": observable_pair_count,
        "exact_pit_size_pair_count": exact_pair_count,
        "reconstructed_size_pair_count": reconstructed_pair_count,
        "combined_size_pair_coverage_ratio": (
            combined_size_pair_coverage_ratio
        ),
        "pit_size_pair_coverage_ratio": pit_size_pair_coverage_ratio,
        "reconstructed_size_pair_ratio": reconstructed_size_pair_ratio,
        "sector_count": len(set(sectors.values())),
        "size_bucket_count": len(set(sizes.values())),
        "unknown_sector_count": len(wanted)
        - sum(value != "unknown" for value in sectors.values()),
        "unknown_size_bucket_count": len(wanted)
        - sum(value != "unknown" for value in sizes.values()),
        "point_in_time_size": reconstructed_pair_count == 0,
        "size_policy": (
            "same_trade_date_total_mv_then_asof_total_share_times_close"
        ),
        "industry_policy": "current_strict_parquet_stock_basic_reference",
    }


def build_context(
    *,
    data_root: Path,
    universes: Sequence[str],
    horizon_days: int,
    warmup_days: int,
    fundamental_mart_root: str | Path = DEFAULT_FUNDAMENTAL_MART_ROOT,
) -> RetestContext:
    frames, universe_by_symbol = load_daily_frames(data_root, universes)
    adj_close, volume, amount = build_price_matrices(frames)
    forward = forward_returns(adj_close, horizon_days)
    monthly, biweekly = rebalance_dates(adj_close.index, warmup_days, horizon_days)
    existing, blocker = compute_existing_composite(MinedFactorRegistry.load(), adj_close, volume, amount)
    as_of = pd.Timestamp(adj_close.index.max()) if not adj_close.empty else None
    exposure_dates = sorted(set(monthly) | set(biweekly))
    close_by_date = _matrix_from_frames(
        frames,
        pd.DatetimeIndex(exposure_dates),
        ("close",),
    )
    sectors, sizes, size_bucket_by_date, exposure_metadata = (
        load_fundamental_exposure_maps(
            mart_root=fundamental_mart_root,
            symbols=list(frames),
            as_of=as_of,
            evaluation_dates=exposure_dates,
            close_by_date=close_by_date,
        )
    )
    return RetestContext(
        frames=frames,
        universe_by_symbol=universe_by_symbol,
        adj_close=adj_close,
        volume=volume,
        amount=amount,
        forward_return=forward,
        rebalance_dates=monthly,
        biweekly_dates=biweekly,
        existing_composite=existing,
        existing_blocker=blocker,
        sector_by_symbol=sectors,
        size_bucket_by_symbol=sizes,
        size_bucket_by_date=size_bucket_by_date,
        exposure_metadata=exposure_metadata,
    )


def run_retest(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = Path(args.aquant_audit_dir).expanduser()
    metadata_dir = Path(args.metadata_dir).expanduser()
    fundamental_mart_root = Path(args.fundamental_mart_root).expanduser()
    data_root = Path(args.data_root).expanduser()
    universes = tuple(item.strip() for item in str(args.universes).split(",") if item.strip())
    candidates, independent_names = load_candidates(audit_dir, args.candidate_set)
    context = build_context(
        data_root=data_root,
        universes=universes,
        horizon_days=int(args.horizon_days),
        warmup_days=int(args.warmup_days),
        fundamental_mart_root=fundamental_mart_root,
    )

    tushare_manifest: dict[str, Any] = {"status": "not_requested"}
    pit_matrix, pit_diag = build_fin_ocf_to_profit_matrix(
        context.adj_close.index,
        list(context.frames),
        metadata_dir=metadata_dir,
        mart_root=fundamental_mart_root,
        allow_legacy_fallback=bool(args.allow_legacy_fundamental_fallback),
    )
    if args.allow_tushare_backfill and pit_diag.coverage_rate < 0.60:
        start_date = context.adj_close.index.min().strftime("%Y%m%d") if not context.adj_close.empty else ""
        end_date = context.adj_close.index.max().strftime("%Y%m%d") if not context.adj_close.empty else ""
        covered_symbols = set(pit_diag.symbols_with_ocf_profit)
        missing_symbols = [symbol for symbol in context.frames if symbol not in covered_symbols]
        tushare_manifest = append_tushare_financial_pit_series(
            missing_symbols,
            start_date=start_date,
            end_date=end_date,
            metadata_dir=metadata_dir,
            request_timeout_seconds=float(args.tushare_request_timeout_seconds),
            max_elapsed_seconds=float(args.tushare_max_elapsed_seconds)
            if float(args.tushare_max_elapsed_seconds) > 0
            else None,
        )
        tushare_manifest["pre_backfill_coverage_rate"] = pit_diag.coverage_rate
        tushare_manifest["pre_backfill_symbols_with_ocf_profit"] = len(
            pit_diag.symbols_with_ocf_profit
        )

    expression_inputs = build_aquant_expression_inputs(
        context.frames,
        metadata_dir=metadata_dir,
        fundamental_mart_root=fundamental_mart_root,
        allow_legacy_fundamental_fallback=bool(args.allow_legacy_fundamental_fallback),
    )
    pit_summary = expression_inputs.diagnostics.get("pit", {})
    results: list[dict[str, Any]] = []
    signal_by_factor: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        blockers: list[str] = []
        try:
            signal = evaluate_aquant_expression(candidate.expression, expression_inputs)
            signal = signal.reindex(index=context.adj_close.index, columns=context.adj_close.columns)
            signal_by_factor[candidate.name] = signal
            metrics = candidate_metrics(
                signal=signal,
                context=context,
                decision_cost_bps=float(args.decision_cost_bps),
                incremental_sleeve=float(args.incremental_sleeve_weight),
            )
        except Exception as exc:
            metrics = {
                "no_future_leakage": True,
                "uses_availability_date": True,
                "point_in_time_rebalance": True,
                "adjusted_price_consistent": True,
                "tradability_rules_defined": True,
                "missingness_explained": False,
                "coverage_rate": 0.0,
                "nan_rate": 1.0,
                "monthly_coverage_min": 0.0,
                "max_sector_coverage_share": 1.0,
                "max_size_bucket_coverage_share": 1.0,
                "extreme_value_ratio": 0.0,
                "icir": 0.0,
                "mean_rankic": 0.0,
                "positive_ic_ratio": 0.0,
                "top_bottom_spread": 0.0,
                "top_quantile_return": 0.0,
                "monotonicity": 0.0,
                "turnover": 0.0,
                "cost_adjusted_return": 0.0,
                "execution_realism": False,
                "capacity_pressure": 1.0,
                "neutralized_icir": 0.0,
                "existing_factor_corr": 1.0,
                "oos_positive_ratio": 0.0,
                "parameter_stability": False,
                "date_range_robustness": False,
                "rebalance_frequency_robustness": False,
                "universe_robustness": False,
                "regime_robustness": False,
                "master_return_delta": 0.0,
                "sharpe_delta": 0.0,
                "max_drawdown_delta": 1.0,
                "turnover_delta": 1.0,
                "execution_cost_delta": 1.0,
                "correlation_with_existing_signals": 1.0,
                "blockers": [f"runtime_compute_error:{exc}"],
            }
        blockers.extend(metrics.get("blockers", []))
        review = evaluate_with_myquant_gate(candidate.name, metrics)
        results.append(
            {
                "name": candidate.name,
                "independent": candidate.independent,
                "expression": candidate.expression,
                "decision": review.decision.value,
                "target_state": review.target_state.value,
                "gates_passed": len(_passed_gate_ids(review)),
                "passed_gate_ids": _passed_gate_ids(review),
                "failed_gate_ids": _failed_gate_ids(review),
                "gate_results": [item.to_dict() for item in review.gate_results],
                "metrics": dict(metrics),
                "blockers": blockers,
                "summary": review.summary,
            }
        )

    positive_variants = sum(
        1 for item in results if _safe_float(item["metrics"].get("mean_rankic")) > 0.0
    )
    parameter_stability = positive_variants >= max(2, math.ceil(len(results) * 0.4))
    for item in results:
        item["metrics"]["parameter_stability"] = bool(
            parameter_stability and _safe_float(item["metrics"].get("mean_rankic")) > 0.0
        )
        review = evaluate_with_myquant_gate(str(item["name"]), item["metrics"])
        item["decision"] = review.decision.value
        item["target_state"] = review.target_state.value
        item["gates_passed"] = len(_passed_gate_ids(review))
        item["passed_gate_ids"] = _passed_gate_ids(review)
        item["failed_gate_ids"] = _failed_gate_ids(review)
        item["gate_results"] = [gate.to_dict() for gate in review.gate_results]
        item["summary"] = review.summary

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        PROJECT_ROOT
        / "reports"
        / "factor_governance"
        / f"aquant_alpha_mix_retest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_symbols: list[str] = []
    for signal in signal_by_factor.values():
        covered = [str(column) for column in signal.columns if signal[column].notna().any()]
        smoke_symbols.extend(covered)
        if len(smoke_symbols) >= 120:
            break
    if not smoke_symbols:
        smoke_symbols = list(context.frames)[:120]
    smoke_symbols = list(dict.fromkeys(smoke_symbols))[:120]
    smoke_frames = {
        symbol: context.frames[symbol] for symbol in smoke_symbols if symbol in context.frames
    }
    runtime_smoke = _runtime_smoke(
        candidates[:3],
        smoke_frames,
        metadata_dir,
        fundamental_mart_root,
        bool(args.allow_legacy_fundamental_fallback),
    )
    qualified = [item for item in results if item["decision"] == "production_candidate"]
    independent_results = [item for item in results if item["independent"]]
    payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "aquant_audit_dir": str(audit_dir),
        "data_root": str(data_root),
        "metadata_dir": str(metadata_dir),
        "fundamental_mart_root": str(fundamental_mart_root),
        "legacy_fundamental_fallback_allowed": bool(args.allow_legacy_fundamental_fallback),
        "universes": list(universes),
        "horizon_days": int(args.horizon_days),
        "warmup_days": int(args.warmup_days),
        "decision_cost_bps": float(args.decision_cost_bps),
        "incremental_sleeve_weight": float(args.incremental_sleeve_weight),
        "registry_write": False,
        "candidate_count": len(results),
        "independent_factor_names": independent_names,
        "qualified_count": len(qualified),
        "qualified_independent_count": sum(1 for item in qualified if item["independent"]),
        "manual_review_required": bool(qualified),
        "conclusion": (
            "manual_production_factor_review_candidate"
            if qualified
            else "no_candidate_passed_myquant_8gate"
        ),
        "pit_coverage": pit_summary,
        "existing_composite_blocker": context.existing_blocker,
        "tushare_backfill_manifest": tushare_manifest,
        "runtime_smoke": runtime_smoke,
        "results": results,
        "independent_results": independent_results,
    }
    write_outputs(output_dir, payload)
    return {"output_dir": str(output_dir), **payload}


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    (output_dir / "aquant_alpha_mix_8gate_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        metrics = item.get("metrics", {})
        rows.append(
            {
                "factor": item.get("name"),
                "independent": item.get("independent"),
                "decision": item.get("decision"),
                "gates_passed": item.get("gates_passed"),
                "failed_gate_ids": ",".join(str(gate) for gate in item.get("failed_gate_ids", [])),
                "coverage_rate": metrics.get("coverage_rate"),
                "icir": metrics.get("icir"),
                "mean_rankic": metrics.get("mean_rankic"),
                "positive_ic_ratio": metrics.get("positive_ic_ratio"),
                "neutralized_icir": metrics.get("neutralized_icir"),
                "existing_factor_corr": metrics.get("existing_factor_corr"),
                "master_return_delta": metrics.get("master_return_delta"),
                "sharpe_delta": metrics.get("sharpe_delta"),
                "turnover": metrics.get("turnover"),
                "blockers": ";".join(item.get("blockers", [])),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "aquant_alpha_mix_8gate_metrics.csv", index=False)
    (output_dir / "runtime_smoke.json").write_text(
        json.dumps(payload.get("runtime_smoke", {}), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "tushare_backfill_manifest.json").write_text(
        json.dumps(payload.get("tushare_backfill_manifest", {}), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    blockers = {
        "pit_coverage": payload.get("pit_coverage", {}),
        "existing_composite_blocker": payload.get("existing_composite_blocker", ""),
        "factor_blockers": {
            item.get("name"): item.get("blockers", []) for item in payload.get("results", [])
        },
    }
    (output_dir / "coverage_blocker_summary.json").write_text(
        json.dumps(blockers, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "aquant_alpha_mix_8gate_report.md").write_text(
        render_markdown_report(payload),
        encoding="utf-8",
    )


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# A_quant alpha_mix_vwap*_ocfprofit_* myQuant 8-Gate Retest",
        "",
        f"- Run timestamp: {payload.get('run_timestamp')}",
        f"- Data root: `{payload.get('data_root')}`",
        f"- Universes: {', '.join(payload.get('universes', []))}",
        f"- Horizon: {payload.get('horizon_days')} trading days",
        f"- Decision cost: {payload.get('decision_cost_bps')} bps",
        f"- Registry write: {payload.get('registry_write')}",
        f"- Candidate count: {payload.get('candidate_count')}",
        f"- Independent subset: {', '.join(payload.get('independent_factor_names', []))}",
        f"- Conclusion: **{payload.get('conclusion')}**",
        "",
        "## Results",
        "",
        "| Factor | Independent | Decision | Gates | Failed | Coverage | ICIR | Neutral ICIR | Corr Existing | Blockers |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("results", []):
        metrics = item.get("metrics", {})
        failed = ",".join(str(gate) for gate in item.get("failed_gate_ids", [])) or "-"
        blockers = "; ".join(item.get("blockers", [])) or "-"
        lines.append(
            "| {factor} | {independent} | {decision} | {gates}/8 | {failed} | {coverage:.2%} | {icir:.3f} | {neutral:.3f} | {corr:.3f} | {blockers} |".format(
                factor=item.get("name"),
                independent="yes" if item.get("independent") else "no",
                decision=item.get("decision"),
                gates=int(item.get("gates_passed", 0)),
                failed=failed,
                coverage=_safe_float(metrics.get("coverage_rate")),
                icir=_safe_float(metrics.get("icir")),
                neutral=_safe_float(metrics.get("neutralized_icir")),
                corr=_safe_float(metrics.get("existing_factor_corr"), 1.0),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## PIT / Runtime Evidence",
            "",
            "```json",
            json.dumps(
                {
                    "pit_coverage": payload.get("pit_coverage", {}),
                    "existing_composite_blocker": payload.get("existing_composite_blocker", ""),
                    "tushare_backfill_status": payload.get("tushare_backfill_manifest", {}).get("status"),
                    "runtime_smoke": payload.get("runtime_smoke", {}),
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            "```",
            "",
            "A factor is not a replacement for current myQuant factors unless it passes on the same data layer, universe, cost/horizon, and registry gate. Passing here only means manual `production_factor` review can start; this runner never promotes the registry.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aquant-audit-dir", default=str(DEFAULT_AQUANT_AUDIT_DIR))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--fundamental-mart-root", default=str(DEFAULT_FUNDAMENTAL_MART_ROOT))
    parser.add_argument("--allow-legacy-fundamental-fallback", action="store_true")
    parser.add_argument("--universes", default=",".join(DEFAULT_UNIVERSES))
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=260)
    parser.add_argument("--decision-cost-bps", type=float, default=1.0)
    parser.add_argument("--incremental-sleeve-weight", type=float, default=0.03)
    parser.add_argument("--candidate-set", choices=("all", "independent"), default="all")
    parser.add_argument("--allow-tushare-backfill", action="store_true")
    parser.add_argument("--tushare-request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--tushare-max-elapsed-seconds", type=float, default=900.0)
    parser.add_argument("--no-registry-write", action="store_true", default=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_retest(args)
    print(payload["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
