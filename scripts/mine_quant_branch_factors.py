#!/usr/bin/env python3
"""Mine myQuant quant-branch candidates with the governed 8-gate policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    DEFAULT_FUNDAMENTAL_MART_ROOT,
)
from quant_investor.factors.governance import (  # noqa: E402
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import MinedFactorRegistry  # noqa: E402
from scripts.retest_aquant_alpha_mix_8gate import (  # noqa: E402
    _failed_gate_ids,
    _json_default,
    _passed_gate_ids,
    _safe_float,
    RetestContext,
    build_context,
    candidate_metrics,
    evaluate_with_myquant_gate,
)

DEFAULT_UNIVERSES = ("hs300", "zz500", "zz1000")
DEFAULT_REGISTRY_PATH = "quant_investor/factor_registry/mined_factors.json"


@dataclass(frozen=True)
class MiningCandidate:
    name: str
    family: str
    category: str
    implementation: str
    description: str
    expression: str = ""
    window: int | None = None
    params: Mapping[str, Any] | None = None


def _min_periods(window: int) -> int:
    return max(3, min(int(window), 5))


def _compact_pit_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    daily = dict(diagnostics.get("daily", {}) or {})
    ratio = dict(diagnostics.get("fin_ocf_to_profit_ratio_fallback", {}) or {})
    coverage = diagnostics.get("coverage_by_metric", {}) or {}
    return {
        "mart_root": daily.get("mart_root"),
        "daily_rows": daily.get("daily_rows"),
        "daily_blocker": daily.get("blocker", ""),
        "metrics_requested": diagnostics.get("metrics_requested", []),
        "coverage_by_metric": coverage,
        "fin_ocf_to_profit_symbols": len(
            ratio.get("symbols_with_ocf_profit", []) or []
        ),
        "fin_ocf_to_profit_ratio_rows": ratio.get("ratio_rows"),
        "legacy_fallback_allowed": diagnostics.get("legacy_fallback_allowed"),
        "pit_rows": diagnostics.get("pit_rows"),
    }


def _auto_analysis_start_date(
    context: RetestContext,
    min_price_coverage: float,
) -> pd.Timestamp | None:
    if context.adj_close.empty:
        return None
    valid = (
        context.adj_close.notna()
        .sum(axis=1)
        .div(max(context.adj_close.shape[1], 1))
    )
    ready = valid[valid >= float(min_price_coverage)]
    if ready.empty:
        return None
    return pd.Timestamp(ready.index[0])


def restrict_context_to_analysis_window(
    context: RetestContext,
    *,
    analysis_start_date: str,
    min_price_coverage: float,
) -> tuple[RetestContext, str]:
    start_text = str(analysis_start_date or "").strip()
    if start_text.lower() in {"", "none", "full"}:
        return context, ""
    if start_text.lower() == "auto":
        start = _auto_analysis_start_date(context, min_price_coverage)
        if start is None:
            return context, ""
    else:
        start = pd.Timestamp(pd.to_datetime(start_text, errors="raise"))
    dates = context.adj_close.index[context.adj_close.index >= start]
    existing = (
        context.existing_composite.reindex(dates)
        if context.existing_composite is not None
        else None
    )
    return (
        RetestContext(
            frames=context.frames,
            universe_by_symbol=context.universe_by_symbol,
            adj_close=context.adj_close.reindex(dates),
            volume=context.volume.reindex(dates),
            amount=context.amount.reindex(dates),
            forward_return=context.forward_return.reindex(dates),
            rebalance_dates=[
                date for date in context.rebalance_dates if date in dates
            ],
            biweekly_dates=[
                date for date in context.biweekly_dates if date in dates
            ],
            existing_composite=existing,
            existing_blocker=context.existing_blocker,
        ),
        start.strftime("%Y-%m-%d"),
    )


def _restrict_context_from_start(
    context: RetestContext,
    *,
    start: pd.Timestamp,
) -> RetestContext:
    dates = context.adj_close.index[context.adj_close.index >= start]
    existing = (
        context.existing_composite.reindex(dates)
        if context.existing_composite is not None
        else None
    )
    return RetestContext(
        frames=context.frames,
        universe_by_symbol=context.universe_by_symbol,
        adj_close=context.adj_close.reindex(dates),
        volume=context.volume.reindex(dates),
        amount=context.amount.reindex(dates),
        forward_return=context.forward_return.reindex(dates),
        rebalance_dates=[
            date for date in context.rebalance_dates if date in dates
        ],
        biweekly_dates=[
            date for date in context.biweekly_dates if date in dates
        ],
        existing_composite=existing,
        existing_blocker=context.existing_blocker,
    )


def candidate_maturity_context(
    context: RetestContext,
    signal: pd.DataFrame,
    *,
    base_start: str,
    min_signal_coverage: float,
) -> tuple[RetestContext, str]:
    """Start only once the candidate has enough non-null signal coverage.

    The start date is coverage-driven only. It does not inspect returns, IC, or
    any gate outcome, so it prevents long lookbacks from failing Gate 2 due to
    warmup without cherry-picking a profitable window.
    """

    start_text = str(base_start or "").strip()
    if start_text:
        floor = pd.Timestamp(pd.to_datetime(start_text, errors="raise"))
    elif not context.adj_close.empty:
        floor = pd.Timestamp(context.adj_close.index.min())
    else:
        return context, ""

    clean_signal = signal.replace([np.inf, -np.inf], np.nan).reindex(
        index=context.adj_close.index,
        columns=context.adj_close.columns,
    )
    coverage = clean_signal.notna().sum(axis=1).div(
        max(clean_signal.shape[1], 1)
    )
    for date in context.rebalance_dates:
        stamp = pd.Timestamp(date)
        if stamp < floor:
            continue
        if _safe_float(coverage.get(stamp, 0.0)) >= float(
            min_signal_coverage
        ):
            return (
                _restrict_context_from_start(context, start=stamp),
                stamp.strftime("%Y-%m-%d"),
            )
    return context, start_text


def price_volume_candidates(windows: Sequence[int]) -> list[MiningCandidate]:
    candidates: list[MiningCandidate] = []
    for window in windows:
        candidates.extend(
            [
                MiningCandidate(
                    name=f"pv_momentum_{window}d",
                    family="momentum",
                    category="momentum",
                    implementation=f"price_volume:pv_momentum_{window}d",
                    window=int(window),
                    description=f"{window}-day adjusted-price momentum.",
                ),
                MiningCandidate(
                    name=f"pv_short_reversal_{window}d",
                    family="short_reversal",
                    category="reversal",
                    implementation=(
                        f"price_volume:pv_short_reversal_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"Negative {window}-day adjusted return; recent "
                        "losers score higher."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_volume_stability_{window}d",
                    family="volume_stability",
                    category="trading_activity",
                    implementation=(
                        f"price_volume:pv_volume_stability_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"Negative {window}-day volume variation; "
                        "stable participation scores higher."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_low_dollar_volume_{window}d",
                    family="low_dollar_volume",
                    category="liquidity",
                    implementation=(
                        f"price_volume:pv_low_dollar_volume_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"Negative log average {window}-day amount; lower "
                        "dollar volume scores higher after capacity gate."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_high_dollar_volume_{window}d",
                    family="high_dollar_volume",
                    category="capacity",
                    implementation=(
                        f"price_volume:pv_high_dollar_volume_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"Log average {window}-day amount; higher capacity "
                        "scores higher after IC and return gates."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_amihud_illiquidity_{window}d",
                    family="amihud_illiquidity",
                    category="liquidity",
                    implementation=(
                        f"price_volume:pv_amihud_illiquidity_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"{window}-day abs return over traded amount; "
                        "higher illiquidity scores higher after capacity gate."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_volatility_penalty_{window}d",
                    family="volatility_penalty",
                    category="risk",
                    implementation=(
                        f"price_volume:pv_volatility_penalty_{window}d"
                    ),
                    window=int(window),
                    description=f"Negative trailing {window}-day volatility.",
                ),
                MiningCandidate(
                    name=f"pv_downside_volatility_{window}d",
                    family="downside_volatility",
                    category="risk",
                    implementation=(
                        f"price_volume:pv_downside_volatility_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"Negative trailing {window}-day downside volatility."
                    ),
                ),
                MiningCandidate(
                    name=f"pv_price_efficiency_{window}d",
                    family="price_efficiency",
                    category="trend_quality",
                    implementation=(
                        f"price_volume:pv_price_efficiency_{window}d"
                    ),
                    window=int(window),
                    description=(
                        f"{window}-day directional efficiency: net move "
                        "divided by absolute path length."
                    ),
                ),
            ]
        )
    for short_window, long_window in ((5, 20), (10, 40), (20, 60), (20, 120)):
        candidates.append(
            MiningCandidate(
                name=(
                    "pv_dollar_volume_growth_"
                    f"{short_window}d_{long_window}d"
                ),
                family="dollar_volume_growth",
                category="trading_activity",
                implementation=(
                    "price_volume:pv_dollar_volume_growth_"
                    f"{short_window}d_{long_window}d"
                ),
                window=long_window,
                params={
                    "short_window": short_window,
                    "long_window": long_window,
                },
                description=(
                    f"{short_window}-day traded amount relative to "
                    f"{long_window}-day traded amount."
                ),
            )
        )
    candidates.extend(
        [
            MiningCandidate(
                name="builtin_short_term_return_20d",
                family="short_term_return",
                category="momentum",
                implementation="builtin:short_term_return",
                window=20,
                description=(
                    "20-day adjusted-price momentum using the builtin runtime."
                ),
            ),
            MiningCandidate(
                name="builtin_volatility_penalty_60d",
                family="volatility_penalty",
                category="risk",
                implementation="builtin:volatility_penalty",
                window=60,
                description=("Negative trailing 60-day volatility."),
            ),
        ]
    )
    for base_window in (16, 18, 19, 20, 21, 22, 24, 25, 27, 30):
        for smooth_window in (2, 3, 5, 7, 10, 15, 20):
            candidates.append(
                MiningCandidate(
                    name=(
                        "pv_volume_stability_smooth_"
                        f"{base_window}d_{smooth_window}d"
                    ),
                    family="volume_stability_smooth",
                    category="trading_activity",
                    implementation=(
                        "price_volume:pv_volume_stability_smooth_"
                        f"{base_window}d_{smooth_window}d"
                    ),
                    window=base_window,
                    description=(
                        f"{base_window}-day volume stability smoothed over "
                        f"{smooth_window} trading days."
                    ),
                )
            )
    for weight in (0.65, 0.70, 0.75, 0.80):
        weight_pct = int(round(weight * 100))
        candidates.append(
            MiningCandidate(
                name=f"pv_blend_volstab19x2_mom90_amihud5_w{weight_pct}",
                family="volstab_momentum_illiquidity_blend",
                category="trading_activity_momentum_liquidity",
                implementation=(
                    "price_volume:"
                    f"pv_blend_volstab19x2_mom90_amihud5_w{weight_pct}"
                ),
                window=90,
                params={
                    "volume_stability_base_window": 19,
                    "volume_stability_smooth_window": 2,
                    "momentum_window": 90,
                    "amihud_window": 5,
                    "outer_volume_stability_weight": weight,
                    "inner_momentum_weight": 0.60,
                },
                description=(
                    "Rank blend of smoothed volume stability with a "
                    "momentum/Amihud composite."
                ),
            )
        )
    return candidates


def formulaic_candidates() -> list[MiningCandidate]:
    """Focused research candidates from formulaic alpha mining.

    These are not registry/runtime production implementations. They are a
    reproducible mining surface for ideas that survived the first quick screen:
    momentum x liquidity and momentum x PIT fundamental residual blends.
    """

    specs: list[tuple[str, str, str, str, str, Sequence[float]]] = [
        (
            "mom60_vol60",
            "momentum_risk",
            "momentum_60",
            "volatility_60",
            "60-day momentum blended with negative 60-day volatility.",
            (0.40, 0.50, 0.60),
        ),
        (
            "mom120_np_yoy_resid",
            "momentum_fundamental_residual",
            "momentum_120",
            "fin_net_profit_yoy_resid_existing",
            (
                "120-day momentum blended with PIT net-profit YoY "
                "residualized against the existing composite."
            ),
            (0.20, 0.25, 0.30, 0.40, 0.50),
        ),
        (
            "mom90_amihud5",
            "momentum_liquidity",
            "momentum_90",
            "amihud_5",
            "90-day momentum blended with 5-day Amihud illiquidity.",
            (0.40, 0.50, 0.60),
        ),
        (
            "mom90_resid_amihud5",
            "momentum_liquidity_residual",
            "momentum_90_resid_existing",
            "amihud_5",
            (
                "Existing-composite residual 90-day momentum blended with "
                "5-day Amihud illiquidity."
            ),
            (0.40, 0.50, 0.60),
        ),
        (
            "low_amount20_mom90",
            "liquidity_momentum",
            "low_amount_20",
            "momentum_90",
            "Low traded amount blended with 90-day momentum.",
            (0.35, 0.50, 0.65),
        ),
        (
            "high_amount20_mom60",
            "capacity_momentum",
            "high_amount_20",
            "momentum_60",
            "High traded amount blended with 60-day momentum.",
            (0.35, 0.50, 0.65),
        ),
        (
            "efficiency60_mom120",
            "trend_quality_momentum",
            "price_efficiency_60",
            "momentum_120",
            "60-day path efficiency blended with 120-day momentum.",
            (0.35, 0.50, 0.65),
        ),
        (
            "volgrowth20x60_mom90",
            "participation_momentum",
            "volume_growth_20_60",
            "momentum_90",
            "20/60-day traded amount growth blended with 90-day momentum.",
            (0.35, 0.50, 0.65),
        ),
        (
            "volstab20_ocf_resid",
            "volume_cash_quality_residual",
            "volume_stability_20",
            "fin_ocf_to_profit_resid_existing",
            (
                "20-day volume stability blended with PIT OCF-to-profit "
                "residualized against the existing composite."
            ),
            (0.35, 0.50, 0.65),
        ),
        (
            "quality_value",
            "fundamental_quality_value",
            "fin_roe",
            "fcf_to_price",
            "PIT ROE blended with free-cashflow yield.",
            (0.35, 0.50, 0.65),
        ),
        (
            "cash_growth_lowlev",
            "fundamental_cash_growth_balance",
            "fin_ocf_to_profit",
            "low_debt_to_assets",
            "PIT cash conversion blended with lower leverage.",
            (0.35, 0.50, 0.65),
        ),
    ]
    candidates: list[MiningCandidate] = []
    for slug, family, left, right, description, weights in specs:
        for weight in weights:
            weight_pct = int(round(float(weight) * 100))
            candidates.append(
                MiningCandidate(
                    name=f"formula_{slug}_w{weight_pct}",
                    family=family,
                    category="formulaic_research",
                    implementation="research_formula:rank_blend",
                    description=description,
                    params={
                        "left": left,
                        "right": right,
                        "left_weight": float(weight),
                    },
                )
            )
    return candidates


def fundamental_candidates() -> list[MiningCandidate]:
    specs = [
        (
            "fin_roe",
            "quality",
            "cs_rank(fin_roe)",
            "Higher PIT return on equity.",
        ),
        (
            "fin_roa",
            "quality",
            "cs_rank(fin_roa)",
            "Higher PIT return on assets.",
        ),
        (
            "fin_debt_to_assets",
            "balance_sheet",
            "0 - cs_rank(fin_debt_to_assets)",
            "Lower PIT debt-to-assets ratio.",
        ),
        (
            "fin_net_profit_yoy",
            "growth",
            "cs_rank(fin_net_profit_yoy)",
            "Higher PIT net-profit year-over-year growth.",
        ),
        (
            "fin_ocf_to_profit",
            "cash_quality",
            "cs_rank(fin_ocf_to_profit)",
            "Higher PIT operating-cashflow-to-profit ratio.",
        ),
        (
            "fin_fcf_to_profit",
            "cash_quality",
            "cs_rank(fin_fcf_to_profit)",
            "Higher PIT free-cashflow-to-profit ratio.",
        ),
        (
            "fcf_to_price",
            "valuation",
            "cs_rank(fcf_to_price)",
            "Higher PIT free-cashflow yield.",
        ),
    ]
    candidates: list[MiningCandidate] = []
    for field, family, expression, description in specs:
        candidates.append(
            MiningCandidate(
                name=f"fund_{field}",
                family=field,
                category=family,
                implementation=f"aquant_expression:fund_{field}",
                expression=expression,
                description=description,
            )
        )
        for window in (20, 60):
            smooth_expr = expression.replace(
                field, f"ts_mean({field}, {window})"
            )
            candidates.append(
                MiningCandidate(
                    name=f"fund_{field}_{window}d",
                    family=field,
                    category=family,
                    implementation=f"aquant_expression:fund_{field}_{window}d",
                    expression=smooth_expr,
                    window=window,
                    description=(
                        f"{description} Smoothed over {window} trading days."
                    ),
                )
            )
    candidates.extend(
        [
            MiningCandidate(
                name="fund_quality_cash_combo",
                family="fundamental_combo",
                category="quality_cash",
                implementation="aquant_expression:fund_quality_cash_combo",
                expression="cs_rank(fin_roe) + cs_rank(fin_ocf_to_profit)",
                description="PIT ROE plus cash conversion composite.",
            ),
            MiningCandidate(
                name="fund_fcf_value_growth_combo",
                family="fundamental_combo",
                category="value_growth",
                implementation=(
                    "aquant_expression:fund_fcf_value_growth_combo"
                ),
                expression=(
                    "cs_rank(fcf_to_price) + " "cs_rank(fin_net_profit_yoy)"
                ),
                description="PIT FCF yield plus net-profit growth composite.",
            ),
            MiningCandidate(
                name="fund_quality_low_leverage_combo",
                family="fundamental_combo",
                category="quality_balance",
                implementation=(
                    "aquant_expression:fund_quality_low_leverage_combo"
                ),
                expression="cs_rank(fin_roe) - cs_rank(fin_debt_to_assets)",
                description="PIT quality adjusted by balance-sheet leverage.",
            ),
            MiningCandidate(
                name="fund_cash_value_combo",
                family="fundamental_combo",
                category="cash_value",
                implementation="aquant_expression:fund_cash_value_combo",
                expression=(
                    "cs_rank(fin_fcf_to_profit) + cs_rank(fcf_to_price)"
                ),
                description="PIT FCF conversion plus FCF yield composite.",
            ),
        ]
    )
    return candidates


def compute_price_volume_signal(
    candidate: MiningCandidate, context: Any
) -> pd.DataFrame:
    window = int(candidate.window or 20)
    close = context.adj_close
    volume = context.volume
    amount = context.amount
    if candidate.family == "short_reversal":
        return -(close.div(close.shift(window)).sub(1.0))
    if candidate.family == "momentum":
        return close.div(close.shift(window)).sub(1.0)
    if candidate.family == "volume_stability":
        mean = volume.rolling(window, min_periods=_min_periods(window)).mean()
        std = volume.rolling(window, min_periods=_min_periods(window)).std(
            ddof=0
        )
        return -(std.div(mean.replace(0.0, np.nan)))
    if candidate.family == "volume_stability_smooth":
        parts = candidate.name.rsplit("_", 2)
        base_window = int(parts[-2].removesuffix("d"))
        smooth_window = int(parts[-1].removesuffix("d"))
        mean = volume.rolling(
            base_window,
            min_periods=_min_periods(base_window),
        ).mean()
        std = volume.rolling(
            base_window,
            min_periods=_min_periods(base_window),
        ).std(ddof=0)
        raw = -(std.div(mean.replace(0.0, np.nan)))
        return raw.rolling(
            smooth_window,
            min_periods=max(1, min(smooth_window, 3)),
        ).mean()
    if candidate.family == "low_dollar_volume":
        avg_amount = amount.rolling(
            window, min_periods=_min_periods(window)
        ).mean()
        return -np.log(avg_amount.replace(0.0, np.nan))
    if candidate.family == "high_dollar_volume":
        avg_amount = amount.rolling(
            window, min_periods=_min_periods(window)
        ).mean()
        return np.log(avg_amount.replace(0.0, np.nan))
    if candidate.family == "amihud_illiquidity":
        returns = close.pct_change().abs()
        return (
            returns.div(amount.replace(0.0, np.nan))
            .rolling(window, min_periods=_min_periods(window))
            .mean()
        )
    if candidate.family == "volatility_penalty":
        return (
            -close.pct_change()
            .rolling(window, min_periods=_min_periods(window))
            .std()
        )
    if candidate.family == "downside_volatility":
        returns = close.pct_change()
        downside = returns.where(returns < 0.0, 0.0)
        return -downside.rolling(
            window, min_periods=_min_periods(window)
        ).std()
    if candidate.family == "price_efficiency":
        net_move = close.div(close.shift(window)).sub(1.0).abs()
        path = close.pct_change().abs().rolling(
            window, min_periods=_min_periods(window)
        ).sum()
        return net_move.div(path.replace(0.0, np.nan))
    if candidate.family == "dollar_volume_growth":
        params = dict(candidate.params or {})
        short_window = int(params.get("short_window", 20))
        long_window = int(params.get("long_window", window))
        short_avg = amount.rolling(
            short_window, min_periods=_min_periods(short_window)
        ).mean()
        long_avg = amount.rolling(
            long_window, min_periods=_min_periods(long_window)
        ).mean()
        return short_avg.div(long_avg.replace(0.0, np.nan)).sub(1.0)
    if candidate.family == "short_term_return":
        return close.div(close.shift(window)).sub(1.0)
    if candidate.family == "volatility_penalty":
        return (
            -close.pct_change()
            .rolling(window, min_periods=_min_periods(window))
            .std()
        )
    if candidate.family == "volstab_momentum_illiquidity_blend":
        params = dict(candidate.params or {})
        base_window = int(params.get("volume_stability_base_window", 19))
        smooth_window = int(params.get("volume_stability_smooth_window", 2))
        momentum_window = int(params.get("momentum_window", 90))
        amihud_window = int(params.get("amihud_window", 5))
        outer_weight = _safe_float(
            params.get("outer_volume_stability_weight"),
            0.75,
        )
        inner_momentum_weight = _safe_float(
            params.get("inner_momentum_weight"),
            0.60,
        )
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
        momentum = close.div(close.shift(momentum_window)).sub(1.0)
        amihud = (
            close.pct_change()
            .abs()
            .div(amount.replace(0.0, np.nan))
            .rolling(amihud_window, min_periods=_min_periods(amihud_window))
            .mean()
        )
        inner = _cs_rank(momentum).mul(inner_momentum_weight) + _cs_rank(
            amihud
        ).mul(1.0 - inner_momentum_weight)
        return _cs_rank(vol_stability).mul(outer_weight) + _cs_rank(inner).mul(
            1.0 - outer_weight
        )
    raise ValueError(f"unsupported price/volume candidate: {candidate.name}")


def _cs_rank(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, pct=True)


def _residualize_against_existing(
    signal: pd.DataFrame,
    existing_composite: pd.DataFrame | None,
) -> pd.DataFrame:
    if existing_composite is None or existing_composite.empty:
        return signal * np.nan
    clean_signal = signal.replace([np.inf, -np.inf], np.nan)
    clean_existing = existing_composite.replace([np.inf, -np.inf], np.nan)
    common_dates = clean_signal.index.intersection(clean_existing.index)
    common_columns = clean_signal.columns.intersection(clean_existing.columns)
    residual = pd.DataFrame(
        index=clean_signal.index,
        columns=clean_signal.columns,
        dtype=float,
    )
    for date in common_dates:
        y = clean_signal.loc[date, common_columns].astype(float)
        x = clean_existing.loc[date, common_columns].astype(float)
        valid = y.notna() & x.notna()
        if int(valid.sum()) < 20:
            continue
        x_values = x[valid].to_numpy(dtype=float)
        y_values = y[valid].to_numpy(dtype=float)
        variance = float(np.var(x_values))
        if variance <= 1e-18:
            fitted = float(np.nanmean(y_values))
        else:
            beta = float(np.cov(x_values, y_values, ddof=0)[0, 1] / variance)
            alpha = float(np.nanmean(y_values) - beta * np.nanmean(x_values))
            fitted = alpha + beta * x_values
        residual.loc[date, valid.index[valid]] = y_values - fitted
    return residual


def _formulaic_primitives(
    context: RetestContext,
    expression_inputs: Any | None,
) -> dict[str, pd.DataFrame]:
    close = context.adj_close
    volume = context.volume
    amount = context.amount.replace(0.0, np.nan)
    returns = close.pct_change()

    def min_periods(window: int) -> int:
        return _min_periods(window)

    def momentum(window: int) -> pd.DataFrame:
        return close.div(close.shift(window)).sub(1.0)

    def amihud(window: int) -> pd.DataFrame:
        return (
            returns.abs()
            .div(amount)
            .rolling(window, min_periods=min_periods(window))
            .mean()
        )

    def low_amount(window: int) -> pd.DataFrame:
        average = amount.rolling(
            window,
            min_periods=min_periods(window),
        ).mean()
        return -np.log(average.replace(0.0, np.nan))

    def high_amount(window: int) -> pd.DataFrame:
        average = amount.rolling(
            window,
            min_periods=min_periods(window),
        ).mean()
        return np.log(average.replace(0.0, np.nan))

    def volume_stability(window: int) -> pd.DataFrame:
        mean = volume.rolling(
            window,
            min_periods=min_periods(window),
        ).mean()
        std = volume.rolling(
            window,
            min_periods=min_periods(window),
        ).std(ddof=0)
        return -(std.div(mean.replace(0.0, np.nan)))

    def volatility(window: int) -> pd.DataFrame:
        return -returns.rolling(window, min_periods=min_periods(window)).std()

    def volume_growth(short_window: int, long_window: int) -> pd.DataFrame:
        short_avg = amount.rolling(
            short_window,
            min_periods=min_periods(short_window),
        ).mean()
        long_avg = amount.rolling(
            long_window,
            min_periods=min_periods(long_window),
        ).mean()
        return short_avg.div(long_avg.replace(0.0, np.nan)).sub(1.0)

    def price_efficiency(window: int) -> pd.DataFrame:
        net_move = close.div(close.shift(window)).sub(1.0).abs()
        path = returns.abs().rolling(
            window,
            min_periods=min_periods(window),
        ).sum()
        return net_move.div(path.replace(0.0, np.nan))

    primitives: dict[str, pd.DataFrame] = {
        "momentum_60": momentum(60),
        "momentum_90": momentum(90),
        "momentum_120": momentum(120),
        "amihud_5": amihud(5),
        "low_amount_20": low_amount(20),
        "high_amount_20": high_amount(20),
        "volume_stability_20": volume_stability(20),
        "volatility_60": volatility(60),
        "volume_growth_20_60": volume_growth(20, 60),
        "price_efficiency_60": price_efficiency(60),
    }
    if expression_inputs is not None:
        for field in (
            "fin_net_profit_yoy",
            "fin_ocf_to_profit",
            "fin_roe",
            "fcf_to_price",
            "fin_debt_to_assets",
        ):
            primitives[field] = getattr(expression_inputs, field).reindex(
                index=close.index,
                columns=close.columns,
            )
        primitives["low_debt_to_assets"] = -primitives[
            "fin_debt_to_assets"
        ]

    for name, matrix in list(primitives.items()):
        if name.endswith("_resid_existing"):
            continue
        primitives[f"{name}_resid_existing"] = _residualize_against_existing(
            matrix,
            context.existing_composite,
        )
    return primitives


def compute_formulaic_signal(
    candidate: MiningCandidate,
    primitives: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    params = dict(candidate.params or {})
    if candidate.implementation != "research_formula:rank_blend":
        raise ValueError(f"unsupported formulaic candidate: {candidate.name}")
    left_name = str(params.get("left") or "")
    right_name = str(params.get("right") or "")
    if left_name not in primitives or right_name not in primitives:
        raise ValueError(
            f"missing formulaic primitive: {left_name} or {right_name}"
        )
    weight = _safe_float(params.get("left_weight"), 0.5)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"invalid rank blend weight: {weight}")
    return (
        _cs_rank(primitives[left_name]).mul(weight)
        + _cs_rank(primitives[right_name]).mul(1.0 - weight)
    )


def build_candidate_catalog(
    windows: Sequence[int],
    *,
    include_price_volume: bool = True,
    include_fundamental: bool = True,
    include_formulaic: bool = True,
) -> list[MiningCandidate]:
    candidates: list[MiningCandidate] = []
    if include_price_volume:
        candidates.extend(price_volume_candidates(windows))
    if include_fundamental:
        candidates.extend(fundamental_candidates())
    if include_formulaic:
        candidates.extend(formulaic_candidates())
    return candidates


def _coerce_signal(signal: pd.DataFrame, context: Any) -> pd.DataFrame:
    return signal.replace([np.inf, -np.inf], np.nan).reindex(
        index=context.adj_close.index,
        columns=context.adj_close.columns,
    )


def _gate_rows(review: Any) -> list[dict[str, Any]]:
    return [item.to_dict() for item in review.gate_results]


def _candidate_result(
    candidate: MiningCandidate,
    metrics: Mapping[str, Any],
    review: Any,
    blockers: Sequence[str],
    *,
    effective_analysis_start_date: str = "",
) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "family": candidate.family,
        "category": candidate.category,
        "implementation": candidate.implementation,
        "expression": candidate.expression,
        "window": candidate.window,
        "params": dict(candidate.params or {}),
        "description": candidate.description,
        "effective_analysis_start_date": effective_analysis_start_date,
        "decision": review.decision.value,
        "target_state": review.target_state.value,
        "gates_passed": len(_passed_gate_ids(review)),
        "passed_gate_ids": _passed_gate_ids(review),
        "failed_gate_ids": _failed_gate_ids(review),
        "gate_results": _gate_rows(review),
        "metrics": dict(metrics),
        "blockers": list(blockers),
        "summary": review.summary,
    }


def _set_parameter_stability(results: list[dict[str, Any]]) -> None:
    family_positive: dict[str, int] = {}
    family_total: dict[str, int] = {}
    for item in results:
        family = str(item.get("family", ""))
        metrics = item.get("metrics", {})
        family_total[family] = family_total.get(family, 0) + 1
        if (
            _safe_float(metrics.get("mean_rankic")) > 0.0
            and _safe_float(metrics.get("icir")) > 0.0
        ):
            family_positive[family] = family_positive.get(family, 0) + 1
    for item in results:
        family = str(item.get("family", ""))
        metrics = dict(item.get("metrics", {}))
        enough_siblings = family_total.get(family, 0) >= 2
        metrics["parameter_stability"] = bool(
            enough_siblings
            and family_positive.get(family, 0) >= 2
            and _safe_float(metrics.get("mean_rankic")) > 0.0
        )
        review = evaluate_with_myquant_gate(str(item["name"]), metrics)
        item["metrics"] = metrics
        item["decision"] = review.decision.value
        item["target_state"] = review.target_state.value
        item["gates_passed"] = len(_passed_gate_ids(review))
        item["passed_gate_ids"] = _passed_gate_ids(review)
        item["failed_gate_ids"] = _failed_gate_ids(review)
        item["gate_results"] = _gate_rows(review)
        item["summary"] = review.summary


def _load_source_notes(path_text: str) -> list[dict[str, Any]]:
    path = Path(str(path_text or "")).expanduser()
    if not str(path_text or "").strip() or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        notes = payload.get("source_notes", payload.get("ideas", []))
        if isinstance(notes, list):
            return [dict(item) for item in notes if isinstance(item, Mapping)]
        return [dict(payload)]
    return []


def _write_registry(path: Path, registry: MinedFactorRegistry) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": registry.schema_version,
                "metadata": registry.metadata,
                "factors": [record.to_dict() for record in registry.factors],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _record_from_result(
    result: Mapping[str, Any],
    *,
    run_timestamp: str,
    run_id: str,
    report_path: str,
    owner: str,
    source_notes: Sequence[Mapping[str, Any]],
    horizon_days: int,
) -> FactorRecord:
    metrics = dict(result.get("metrics", {}) or {})
    metadata = {
        "source_report": report_path,
        "registry_write_source": "daily_factor_mining_automation",
        "registry_write_run_id": run_id,
        "registry_write_at": run_timestamp,
        "effective_analysis_start_date": result.get(
            "effective_analysis_start_date", ""
        ),
        "expression": result.get("expression", ""),
        "params": dict(result.get("params", {}) or {}),
        "source_notes": [dict(item) for item in source_notes],
        "manual_promotion_required": True,
        "runtime_effect": "none_until_manual_production_factor_promotion",
    }
    return FactorRecord(
        name=str(result.get("name", "")).strip(),
        version="v" + run_timestamp[:10].replace("-", ""),
        state=FactorLifecycleState.PRODUCTION_CANDIDATE,
        category=str(result.get("category", "custom") or "custom"),
        implementation=str(result.get("implementation", "") or ""),
        weight=0.0,
        direction=1.0,
        horizon_days=int(metrics.get("horizon_days") or horizon_days),
        owner=owner,
        description=str(result.get("description", "") or ""),
        tags=[
            "daily_factor_mining",
            str(result.get("family", "") or "unknown_family"),
            str(result.get("category", "") or "custom"),
        ],
        gate_results=[
            GateResult.from_dict(item)
            for item in result.get("gate_results", [])
            if isinstance(item, Mapping)
        ],
        metrics=metrics,
        admission_decision=FactorAdmissionDecision.PRODUCTION_CANDIDATE,
        metadata=metadata,
    )


def apply_production_candidate_registry_updates(
    *,
    registry_path: str | Path,
    qualified_results: Sequence[Mapping[str, Any]],
    run_timestamp: str,
    run_id: str,
    report_path: str,
    owner: str,
    source_notes: Sequence[Mapping[str, Any]] = (),
    horizon_days: int = 30,
    max_candidates: int = 5,
    write: bool = True,
) -> dict[str, Any]:
    """Write qualified factors as zero-weight production candidates only."""

    path = Path(registry_path).expanduser()
    manifest: dict[str, Any] = {
        "requested": bool(write),
        "registry_path": str(path),
        "run_id": run_id,
        "source_report": report_path,
        "max_candidates": int(max_candidates),
        "qualified_count": len(qualified_results),
        "written_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
        "written_factors": [],
        "updated_factors": [],
        "skipped_factors": [],
        "status": "not_requested" if not write else "pending",
    }
    if not write:
        return manifest
    if not qualified_results:
        manifest["status"] = "no_qualified_candidates"
        return manifest

    registry = MinedFactorRegistry.load(path)
    by_name = {factor.name: factor for factor in registry.factors}
    changed = False
    selected = list(qualified_results)[: max(int(max_candidates), 0)]
    for item in selected:
        name = str(item.get("name", "") or "").strip()
        if not name:
            manifest["skipped_factors"].append(
                {"name": "", "reason": "missing_name"}
            )
            continue
        if item.get("decision") != FactorAdmissionDecision.PRODUCTION_CANDIDATE.value:
            manifest["skipped_factors"].append(
                {"name": name, "reason": f"decision={item.get('decision')}"}
            )
            continue
        existing = by_name.get(name)
        if existing and existing.state == FactorLifecycleState.PRODUCTION_FACTOR:
            manifest["skipped_factors"].append(
                {"name": name, "reason": "existing_production_factor"}
            )
            continue
        record = _record_from_result(
            item,
            run_timestamp=run_timestamp,
            run_id=run_id,
            report_path=report_path,
            owner=owner,
            source_notes=source_notes,
            horizon_days=horizon_days,
        )
        if existing:
            registry.factors = [
                record if factor.name == name else factor
                for factor in registry.factors
            ]
            manifest["updated_factors"].append(name)
            manifest["updated_count"] += 1
        else:
            registry.factors.append(record)
            manifest["written_factors"].append(name)
            manifest["written_count"] += 1
        changed = True

    manifest["skipped_count"] = len(manifest["skipped_factors"])
    if changed:
        registry.metadata.update(
            {
                "last_daily_factor_mining_at": run_timestamp,
                "last_daily_factor_mining_run_id": run_id,
                "last_daily_factor_mining_report": report_path,
                "last_daily_factor_mining_written": list(
                    manifest["written_factors"]
                ),
                "last_daily_factor_mining_updated": list(
                    manifest["updated_factors"]
                ),
                "daily_factor_mining_policy": (
                    "writes zero-weight production_candidate records only; "
                    "manual production_factor promotion required"
                ),
            }
        )
        _write_registry(path, registry)
        manifest["status"] = "updated"
    else:
        manifest["status"] = "no_registry_changes"
    return manifest


def run_mining(args: argparse.Namespace) -> dict[str, Any]:
    universes = tuple(
        item.strip() for item in str(args.universes).split(",") if item.strip()
    )
    windows = tuple(
        int(item.strip())
        for item in str(args.windows).split(",")
        if item.strip()
    )
    full_context = build_context(
        data_root=Path(args.data_root).expanduser(),
        universes=universes,
        horizon_days=int(args.horizon_days),
        warmup_days=int(args.warmup_days),
    )
    context, resolved_analysis_start = restrict_context_to_analysis_window(
        full_context,
        analysis_start_date=str(args.analysis_start_date),
        min_price_coverage=float(args.min_analysis_price_coverage),
    )
    expression_inputs = None
    candidates: list[MiningCandidate] = []
    if args.include_price_volume:
        candidates.extend(price_volume_candidates(windows))
    if args.include_fundamental or args.include_formulaic:
        expression_inputs = build_aquant_expression_inputs(
            full_context.frames,
            fundamental_mart_root=Path(
                args.fundamental_mart_root
            ).expanduser(),
            allow_legacy_fundamental_fallback=False,
        )
    if args.include_fundamental:
        candidates.extend(fundamental_candidates())
    if args.include_formulaic:
        candidates.extend(formulaic_candidates())

    formulaic_primitives = (
        _formulaic_primitives(full_context, expression_inputs)
        if args.include_formulaic
        else {}
    )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        blockers: list[str] = []
        effective_start = resolved_analysis_start
        try:
            if candidate.implementation.startswith("aquant_expression:"):
                if expression_inputs is None:
                    raise ValueError(
                        "fundamental expression inputs were not built"
                    )
                signal = evaluate_aquant_expression(
                    candidate.expression, expression_inputs
                )
            elif candidate.implementation.startswith("research_formula:"):
                signal = compute_formulaic_signal(
                    candidate,
                    formulaic_primitives,
                )
            else:
                signal = compute_price_volume_signal(candidate, full_context)
            signal = _coerce_signal(signal, full_context)
            metrics_context = context
            if args.candidate_maturity_start:
                metrics_context, effective_start = candidate_maturity_context(
                    full_context,
                    signal,
                    base_start=resolved_analysis_start,
                    min_signal_coverage=float(
                        args.min_candidate_signal_coverage
                    ),
                )
            metrics = candidate_metrics(
                signal=signal,
                context=metrics_context,
                decision_cost_bps=float(args.decision_cost_bps),
                incremental_sleeve=float(args.incremental_sleeve_weight),
            )
        except Exception as exc:
            blockers.append(f"candidate_compute_error:{exc}")
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
                "blockers": list(blockers),
            }
        blockers.extend(str(item) for item in metrics.get("blockers", []))
        review = evaluate_with_myquant_gate(candidate.name, metrics)
        results.append(
            _candidate_result(
                candidate,
                metrics,
                review,
                blockers,
                effective_analysis_start_date=effective_start,
            )
        )

    _set_parameter_stability(results)
    results.sort(
        key=lambda item: (
            int(item.get("gates_passed", 0)),
            _safe_float(item.get("metrics", {}).get("icir")),
            _safe_float(item.get("metrics", {}).get("master_return_delta")),
        ),
        reverse=True,
    )
    qualified = [
        item for item in results if item["decision"] == "production_candidate"
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROJECT_ROOT
        / "reports"
        / "factor_governance"
        / f"quant_branch_factor_mining_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    run_id = str(args.run_id or "").strip() or f"quant_branch_factor_mining_{timestamp}"
    source_notes = _load_source_notes(str(args.source_notes_json))
    results_json_path = output_dir / "quant_branch_factor_mining_results.json"
    registry_write_requested = bool(args.write_production_candidates)
    registry_manifest = apply_production_candidate_registry_updates(
        registry_path=str(args.registry_path),
        qualified_results=qualified,
        run_timestamp=run_timestamp,
        run_id=run_id,
        report_path=str(results_json_path),
        owner=str(args.registry_owner),
        source_notes=source_notes,
        horizon_days=int(args.horizon_days),
        max_candidates=int(args.max_registry_candidates),
        write=registry_write_requested,
    )
    payload = {
        "run_timestamp": run_timestamp,
        "run_id": run_id,
        "data_root": str(args.data_root),
        "fundamental_mart_root": str(args.fundamental_mart_root),
        "legacy_fundamental_fallback_allowed": False,
        "universes": list(universes),
        "horizon_days": int(args.horizon_days),
        "warmup_days": int(args.warmup_days),
        "analysis_start_date": str(args.analysis_start_date),
        "resolved_analysis_start_date": resolved_analysis_start,
        "candidate_maturity_start": bool(args.candidate_maturity_start),
        "min_candidate_signal_coverage": float(
            args.min_candidate_signal_coverage
        ),
        "min_analysis_price_coverage": float(args.min_analysis_price_coverage),
        "windows": list(windows),
        "decision_cost_bps": float(args.decision_cost_bps),
        "incremental_sleeve_weight": float(args.incremental_sleeve_weight),
        "registry_write": registry_manifest["status"] == "updated",
        "registry_write_requested": registry_write_requested,
        "registry_update_manifest": registry_manifest,
        "source_notes": source_notes,
        "existing_composite_blocker": context.existing_blocker,
        "candidate_count": len(results),
        "qualified_count": len(qualified),
        "qualified_factors": [item["name"] for item in qualified],
        "manual_review_required": bool(qualified),
        "conclusion": (
            "manual_production_factor_review_candidate"
            if qualified
            else "no_candidate_passed_myquant_8gate"
        ),
        "pit_coverage": _compact_pit_diagnostics(
            expression_inputs.diagnostics.get("pit", {})
            if expression_inputs is not None
            else {"status": "not_requested"}
        ),
        "results": results,
    }
    write_outputs(output_dir, payload)
    return {"output_dir": str(output_dir), **payload}


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    (output_dir / "quant_branch_factor_mining_results.json").write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, default=_json_default
        ),
        encoding="utf-8",
    )
    (output_dir / "registry_update_manifest.json").write_text(
        json.dumps(
            payload.get("registry_update_manifest", {}),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        metrics = item.get("metrics", {})
        rows.append(
            {
                "factor": item.get("name"),
                "family": item.get("family"),
                "category": item.get("category"),
                "implementation": item.get("implementation"),
                "effective_analysis_start_date": item.get(
                    "effective_analysis_start_date"
                ),
                "decision": item.get("decision"),
                "gates_passed": item.get("gates_passed"),
                "failed_gate_ids": ",".join(
                    str(gate) for gate in item.get("failed_gate_ids", [])
                ),
                "coverage_rate": metrics.get("coverage_rate"),
                "nan_rate": metrics.get("nan_rate"),
                "monthly_coverage_min": metrics.get("monthly_coverage_min"),
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
    pd.DataFrame(rows).to_csv(
        output_dir / "quant_branch_factor_mining_metrics.csv", index=False
    )
    (output_dir / "quant_branch_factor_mining_report.md").write_text(
        render_markdown_report(payload),
        encoding="utf-8",
    )


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# myQuant Quant Branch Factor Mining",
        "",
        f"- Run timestamp: {payload.get('run_timestamp')}",
        f"- Data root: `{payload.get('data_root')}`",
        f"- Fundamental mart: `{payload.get('fundamental_mart_root')}`",
        f"- Universes: {', '.join(payload.get('universes', []))}",
        f"- Horizon: {payload.get('horizon_days')} trading days",
        (
            "- Analysis start: "
            f"{payload.get('resolved_analysis_start_date') or 'full'}"
        ),
        f"- Registry write: {payload.get('registry_write')}",
        (
            "- Candidate maturity start: "
            f"{payload.get('candidate_maturity_start')} "
            f"(min signal coverage "
            f"{_safe_float(payload.get('min_candidate_signal_coverage')):.0%})"
        ),
        f"- Candidate count: {payload.get('candidate_count')}",
        f"- Qualified count: {payload.get('qualified_count')}",
        f"- Conclusion: **{payload.get('conclusion')}**",
        (
            "- Registry update: "
            f"{(payload.get('registry_update_manifest') or {}).get('status')}"
        ),
        "",
        "Passing here only opens manual production-factor review.",
        (
            "This runner can write zero-weight production_candidate records "
            "only when explicitly requested; it never promotes production_factor."
        ),
        "",
        "## Top Results",
        "",
        (
            "| Factor | Family | Start | Decision | Gates | Failed | "
            "Coverage | ICIR | RankIC | Corr | Delta | Blockers |"
        ),
        (
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | "
            "---: | ---: | ---: | --- |"
        ),
    ]
    for item in payload.get("results", [])[:40]:
        metrics = item.get("metrics", {})
        failed = (
            ",".join(str(gate) for gate in item.get("failed_gate_ids", []))
            or "-"
        )
        blockers = "; ".join(item.get("blockers", [])) or "-"
        lines.append(
            (
                "| {factor} | {family} | {start} | {decision} | "
                "{gates}/8 | {failed} | {coverage:.2%} | {icir:.3f} | "
                "{rankic:.4f} | {corr:.3f} | {delta:.4f} | "
                "{blockers} |"
            ).format(
                factor=item.get("name"),
                family=item.get("family"),
                start=item.get("effective_analysis_start_date") or "-",
                decision=item.get("decision"),
                gates=int(item.get("gates_passed", 0)),
                failed=failed,
                coverage=_safe_float(metrics.get("coverage_rate")),
                icir=_safe_float(metrics.get("icir")),
                rankic=_safe_float(metrics.get("mean_rankic")),
                corr=_safe_float(metrics.get("existing_factor_corr"), 1.0),
                delta=_safe_float(metrics.get("master_return_delta")),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Registry Update",
            "",
            "```json",
            json.dumps(
                payload.get("registry_update_manifest", {}),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            "```",
            "",
            "## PIT Evidence",
            "",
            "```json",
            json.dumps(
                {
                    "pit_coverage": payload.get("pit_coverage", {}),
                    "existing_composite_blocker": payload.get(
                        "existing_composite_blocker", ""
                    ),
                    "legacy_fundamental_fallback_allowed": payload.get(
                        "legacy_fundamental_fallback_allowed"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--fundamental-mart-root", default=str(DEFAULT_FUNDAMENTAL_MART_ROOT)
    )
    parser.add_argument("--universes", default=",".join(DEFAULT_UNIVERSES))
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=260)
    parser.add_argument("--analysis-start-date", default="auto")
    parser.add_argument(
        "--min-analysis-price-coverage", type=float, default=0.95
    )
    parser.add_argument(
        "--candidate-maturity-start",
        action="store_true",
        default=True,
        help=(
            "Start each candidate at the first rebalance date where signal "
            "coverage reaches --min-candidate-signal-coverage."
        ),
    )
    parser.add_argument(
        "--no-candidate-maturity-start",
        dest="candidate_maturity_start",
        action="store_false",
    )
    parser.add_argument(
        "--min-candidate-signal-coverage", type=float, default=0.60
    )
    parser.add_argument("--windows", default="5,10,15,20,25,30,40,60")
    parser.add_argument("--decision-cost-bps", type=float, default=1.0)
    parser.add_argument(
        "--incremental-sleeve-weight", type=float, default=0.03
    )
    parser.add_argument(
        "--include-price-volume", action="store_true", default=True
    )
    parser.add_argument(
        "--no-price-volume", dest="include_price_volume", action="store_false"
    )
    parser.add_argument(
        "--include-fundamental", action="store_true", default=True
    )
    parser.add_argument(
        "--no-fundamental", dest="include_fundamental", action="store_false"
    )
    parser.add_argument(
        "--include-formulaic", action="store_true", default=True
    )
    parser.add_argument(
        "--no-formulaic", dest="include_formulaic", action="store_false"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source-notes-json", default="")
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--registry-owner",
        default="myQuant daily factor mining automation",
    )
    parser.add_argument("--max-registry-candidates", type=int, default=5)
    parser.add_argument(
        "--write-production-candidates",
        action="store_true",
        help=(
            "Write qualified factors as zero-weight production_candidate "
            "records. Never promotes production_factor."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run_mining(parse_args(argv))
    print(payload["output_dir"])
    print(
        json.dumps(
            {
                "conclusion": payload["conclusion"],
                "candidate_count": payload["candidate_count"],
                "qualified_count": payload["qualified_count"],
                "qualified_factors": payload["qualified_factors"],
                "registry_update_status": payload[
                    "registry_update_manifest"
                ]["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
