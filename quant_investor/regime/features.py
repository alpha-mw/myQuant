from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.regime.types import RegimeFeatureSnapshot


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _clamp(value: Any, lower: float, upper: float, default: float = 0.0) -> float:
    numeric = _finite_float(value, default)
    return max(float(lower), min(float(upper), numeric))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _macro_metadata(macro_verdict: BranchVerdict | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(macro_verdict, BranchVerdict):
        return _mapping(macro_verdict.metadata)
    if isinstance(macro_verdict, Mapping):
        return _mapping(macro_verdict.get("metadata"))
    return {}


def _macro_score(macro_verdict: BranchVerdict | Mapping[str, Any] | None) -> float:
    if isinstance(macro_verdict, BranchVerdict):
        return _clamp(macro_verdict.final_score, -1.0, 1.0)
    if isinstance(macro_verdict, Mapping):
        return _clamp(macro_verdict.get("final_score", 0.0), -1.0, 1.0)
    return 0.0


def _share(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item) / len(values)


def _market_states(
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    states: list[Mapping[str, Any]] = []
    for payload in tradability_snapshot.values():
        if not isinstance(payload, Mapping):
            continue
        state = payload.get("market_state")
        if isinstance(state, Mapping):
            states.append(state)
    return states


def build_regime_feature_snapshot(
    *,
    market: str,
    universe_key: str,
    as_of: str,
    frames: Mapping[str, pd.DataFrame],
    tradability_snapshot: Mapping[str, Mapping[str, Any]],
    cross_section_quant: Mapping[str, Any],
    macro_verdict: BranchVerdict | Mapping[str, Any] | None,
) -> RegimeFeatureSnapshot:
    diagnostics: list[str] = []
    cross_section = _mapping(cross_section_quant)
    frame_count = len(frames or {})
    states = _market_states(tradability_snapshot or {})

    if not cross_section:
        diagnostics.append("cross_section_quant_missing")
    if frame_count <= 0:
        diagnostics.append("frames_empty")
    if not states:
        diagnostics.append("tradability_market_state_empty")

    sample_count = int(max(_finite_float(cross_section.get("sample_count", frame_count), frame_count), 0.0))
    average_return = _clamp(cross_section.get("average_return", 0.0), -1.0, 1.0)
    average_volatility = _clamp(cross_section.get("average_volatility", 0.0), 0.0, 1.0)
    breadth = _clamp(cross_section.get("breadth", 0.0), 0.0, 1.0)

    momentum_flags: list[bool] = []
    breakout_flags: list[bool] = []
    fake_breakout_flags: list[bool] = []
    drawdowns: list[float] = []
    liquidity_scores: list[float] = []
    volume_confirmations: list[float] = []
    for state in states:
        momentum_strength = _clamp(state.get("momentum_strength", 0.0), 0.0, 1.0)
        breakout_readiness = _clamp(state.get("breakout_readiness", 0.0), 0.0, 1.0)
        fake_breakout_risk = _clamp(state.get("fake_breakout_risk", 0.0), 0.0, 1.0)
        max_drawdown_pct = _clamp(state.get("max_drawdown_pct", 0.0), 0.0, 1.0)
        liquidity_score = _clamp(state.get("liquidity_score", 0.0), 0.0, 1.0)
        volume_confirmation = _clamp(state.get("volume_confirmation", 0.0), 0.0, 1.0)

        momentum_flags.append(momentum_strength >= 0.55)
        breakout_flags.append(
            breakout_readiness >= 0.50 and volume_confirmation >= 0.35
        )
        fake_breakout_flags.append(fake_breakout_risk >= 0.60)
        drawdowns.append(max_drawdown_pct)
        liquidity_scores.append(liquidity_score)
        volume_confirmations.append(volume_confirmation)

    macro_meta = _macro_metadata(macro_verdict)
    macro_target_gross_exposure = _clamp(
        macro_meta.get("target_gross_exposure", 0.55),
        0.0,
        1.0,
        default=0.55,
    )

    return RegimeFeatureSnapshot(
        as_of=str(as_of or ""),
        market=str(market or ""),
        universe_key=str(universe_key or ""),
        average_return=average_return,
        average_volatility=average_volatility,
        breadth=breadth,
        momentum_share=_share(momentum_flags),
        breakout_ready_share=_share(breakout_flags),
        fake_breakout_share=_share(fake_breakout_flags),
        median_drawdown=float(median(drawdowns)) if drawdowns else 0.0,
        average_liquidity=float(sum(liquidity_scores) / len(liquidity_scores)) if liquidity_scores else 0.50,
        average_volume_confirmation=(
            float(sum(volume_confirmations) / len(volume_confirmations))
            if volume_confirmations
            else 0.0
        ),
        macro_score=_macro_score(macro_verdict),
        macro_target_gross_exposure=macro_target_gross_exposure,
        sample_count=sample_count,
        diagnostics=diagnostics,
        metadata={
            "frame_count": frame_count,
            "tradability_market_state_count": len(states),
            "no_network": True,
            "no_llm": True,
        },
    )
