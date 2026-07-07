from __future__ import annotations

import math
from typing import Any

import pytest

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig


SYMBOL = "000001.SZ"


def _theme_rotation(
    *,
    symbol: str = SYMBOL,
    symbol_score: float = 0.95,
    symbol_smoothed_score: float | None = None,
    phase: str = "confirmed_rotation",
    risk_flags: list[str] | None = None,
    status: str = "success",
) -> dict[str, Any]:
    payload = {
        "status": status,
        "symbol_scores": {symbol: symbol_score},
        "symbol_primary_theme": {symbol: "industry::semiconductor"},
        "symbol_phase": {symbol: phase},
        "symbol_risk_flags": {symbol: list(risk_flags or [])},
        "theme_scores": {
            "industry::semiconductor": {
                "score": 0.88,
                "phase": phase,
                "member_count": 8,
            }
        },
    }
    if symbol_smoothed_score is not None:
        payload["symbol_smoothed_scores"] = {symbol: symbol_smoothed_score}
    return payload


def _market_state() -> dict[str, float]:
    return {
        "momentum_strength": 0.72,
        "breakout_readiness": 0.68,
        "volume_confirmation": 0.55,
        "trend_stability": 0.62,
        "distance_from_high_pct": 0.018,
        "fake_breakout_risk": 0.08,
        "max_drawdown_pct": 0.04,
        "return_20d": 0.09,
    }


def _global_context(metadata: dict[str, Any] | None = None) -> GlobalContext:
    payload = dict(metadata or {})
    payload.setdefault("symbol_market_state", {SYMBOL: _market_state()})
    return GlobalContext(
        market="CN",
        universe_key="full_a",
        universe_symbols=[SYMBOL],
        universe_tiers={"researchable": [SYMBOL]},
        industry_map={SYMBOL: "Semiconductor"},
        liquidity_filter={"liquidity_scores": {SYMBOL: 1.0}},
        data_quality_quarantine=[],
        metadata=payload,
    )


def _momentum_score(*, theme_boost_enabled: bool, metadata: dict[str, Any] | None) -> float:
    funnel = DeterministicFunnel(
        FunnelConfig(
            profile="momentum_leader",
            theme_boost_enabled=theme_boost_enabled,
            theme_boost_cap=0.05,
        )
    )
    return funnel._momentum_leader_score(
        symbol=SYMBOL,
        quant_scores={SYMBOL: 0.42},
        global_context=_global_context(metadata),
    )


def test_theme_boost_disabled_returns_zero():
    funnel = DeterministicFunnel(
        FunnelConfig(theme_boost_enabled=False, theme_boost_cap=0.05)
    )

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )

    assert boost == 0.0
    assert metadata["enabled"] is False
    assert metadata["reason"] == "disabled"


def test_theme_boost_confirmed_rotation_is_positive_and_capped():
    funnel = DeterministicFunnel(
        FunnelConfig(theme_boost_enabled=True, theme_boost_cap=0.05)
    )

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )

    assert boost > 0.0
    assert boost <= 0.05
    assert metadata["reason"] == "applied"
    assert metadata["score_source"] == "raw"
    assert metadata["phase"] == "confirmed_rotation"
    assert metadata["final_boost"] == boost


def test_theme_boost_missing_symbol_returns_zero():
    funnel = DeterministicFunnel(FunnelConfig(theme_boost_enabled=True))

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context(
            {"theme_rotation": _theme_rotation(symbol="000002.SZ")}
        ),
    )

    assert boost == 0.0
    assert metadata["reason"] == "symbol_theme_missing"


def test_theme_boost_overextended_penalizes():
    funnel = DeterministicFunnel(FunnelConfig(theme_boost_enabled=True))

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context(
            {
                "theme_rotation": _theme_rotation(
                    phase="overextended",
                    risk_flags=["theme_overextended"],
                )
            }
        ),
    )

    assert boost <= 0.0
    assert metadata["phase"] == "overextended"
    assert metadata["risk_penalty"] < 0.0


def test_theme_boost_distribution_penalizes():
    funnel = DeterministicFunnel(FunnelConfig(theme_boost_enabled=True))

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context(
            {
                "theme_rotation": _theme_rotation(
                    phase="distribution",
                    risk_flags=["theme_distribution_risk"],
                )
            }
        ),
    )

    assert boost < 0.0
    assert metadata["phase"] == "distribution"
    assert metadata["risk_penalty"] < 0.0


def test_theme_boost_crowding_flag_penalizes_inside_existing_gate():
    funnel = DeterministicFunnel(
        FunnelConfig(theme_boost_enabled=True, theme_boost_cap=0.10)
    )

    base_boost, _base_metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )
    crowded_boost, crowded_metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context(
            {
                "theme_rotation": _theme_rotation(
                    risk_flags=["theme_crowded"],
                )
            }
        ),
    )

    assert crowded_metadata["risk_penalty"] == -0.03
    assert crowded_boost == pytest.approx(base_boost - 0.03)


def test_theme_boost_smoothed_source_reads_symbol_smoothed_scores():
    funnel = DeterministicFunnel(
        FunnelConfig(
            theme_boost_enabled=True,
            theme_boost_cap=0.05,
            theme_boost_score_source="smoothed",
        )
    )

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context(
            {
                "theme_rotation": _theme_rotation(
                    symbol_score=0.95,
                    symbol_smoothed_score=0.60,
                )
            }
        ),
    )

    assert boost > 0.0
    assert metadata["reason"] == "applied"
    assert metadata["score_source"] == "smoothed"
    assert metadata["symbol_score"] == 0.60


def test_theme_boost_smoothed_source_missing_score_fails_closed():
    funnel = DeterministicFunnel(
        FunnelConfig(
            theme_boost_enabled=True,
            theme_boost_cap=0.05,
            theme_boost_score_source="smoothed",
        )
    )

    boost, metadata = funnel._theme_boost_for_symbol(
        symbol=SYMBOL,
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )

    assert boost == 0.0
    assert metadata["score_source"] == "smoothed"
    assert metadata["reason"] == "smoothed_theme_score_missing"


def test_momentum_leader_score_missing_theme_preserves_base_score():
    base_score = _momentum_score(theme_boost_enabled=False, metadata={})
    missing_theme_score = _momentum_score(theme_boost_enabled=True, metadata={})

    assert missing_theme_score == base_score


def test_momentum_leader_score_theme_boost_changes_score_when_enabled():
    base_score = _momentum_score(theme_boost_enabled=False, metadata={})
    score_with_theme = _momentum_score(
        theme_boost_enabled=True,
        metadata={"theme_rotation": _theme_rotation()},
    )

    assert score_with_theme > base_score
    assert (score_with_theme - base_score) <= 0.05 + 1e-9


def test_funnel_metadata_contains_theme_boost_flags():
    funnel = DeterministicFunnel(
        FunnelConfig(
            profile="momentum_leader",
            max_candidates=1,
            theme_boost_enabled=True,
            theme_boost_cap=0.05,
        )
    )

    output = funnel.run(
        quant_result=BranchResult(branch_name="quant", symbol_scores={SYMBOL: 0.42}),
        global_context=_global_context({"theme_rotation": _theme_rotation()}),
    )

    assert output.funnel_metadata["theme_boost_enabled"] is True
    assert math.isclose(output.funnel_metadata["theme_boost_cap"], 0.05)
    assert output.funnel_metadata["theme_boost_profile"] == "momentum_leader_only"
    assert output.funnel_metadata["theme_boost_available_count"] == 1
    assert output.funnel_metadata["theme_boost_applied_count"] == 1
