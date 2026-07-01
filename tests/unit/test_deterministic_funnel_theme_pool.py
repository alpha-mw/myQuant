from __future__ import annotations

import pytest

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import DeterministicFunnel, FunnelConfig


def _market_state(symbols: list[str]) -> dict[str, dict[str, float]]:
    return {
        symbol: {
            "momentum_strength": 0.72,
            "breakout_readiness": 0.66,
            "volume_confirmation": 0.60,
            "trend_stability": 0.64,
            "fake_breakout_risk": 0.05,
            "max_drawdown_pct": 0.03,
        }
        for symbol in symbols
    }


def _theme_score(theme_id: str, score: float, phase: str) -> dict[str, object]:
    return {
        "theme_id": theme_id,
        "theme_name": theme_id,
        "score": score,
        "phase": phase,
        "breadth": 0.70,
        "confidence": 0.80,
        "member_count": 4,
        "acceleration": 0.60,
        "volume_confirmation": 0.60,
        "overextension_risk": 0.05,
        "fake_breakout_risk": 0.05,
        "risk_flags": [],
    }


def _theme_rotation() -> dict[str, object]:
    return {
        "status": "success",
        "theme_scores": {
            "admitted": _theme_score("admitted", 0.88, "confirmed_rotation"),
            "low_score_theme": _theme_score("low_score_theme", 0.42, "accumulation"),
        },
        "symbol_scores": {
            "ADMIT1": 0.86,
            "LOW_SCORE": 0.35,
            "RISKY": 0.92,
        },
        "symbol_smoothed_scores": {
            "ADMIT1": 0.84,
            "LOW_SCORE": 0.34,
            "RISKY": 0.90,
        },
        "symbol_primary_theme": {
            "ADMIT1": "admitted",
            "LOW_SCORE": "low_score_theme",
            "RISKY": "admitted",
        },
        "symbol_phase": {
            "ADMIT1": "confirmed_rotation",
            "LOW_SCORE": "accumulation",
            "RISKY": "confirmed_rotation",
        },
        "symbol_risk_flags": {
            "ADMIT1": [],
            "LOW_SCORE": [],
            "RISKY": ["theme_fake_breakout_risk"],
        },
    }


def _context(theme_rotation: dict[str, object] | None = None) -> GlobalContext:
    symbols = ["ADMIT1", "LOW_SCORE", "RISKY", "UNTHEMED_HIGH"]
    metadata: dict[str, object] = {
        "symbol_market_state": _market_state(symbols),
        "markov_regime": {
            "production_eligible": True,
            "dominant_regime": "震荡低波",
            "transition_risk": 0.10,
            "confidence": 0.80,
        },
    }
    if theme_rotation is not None:
        metadata["theme_rotation"] = theme_rotation
    return GlobalContext(
        market="CN",
        universe_key="full_a",
        universe_symbols=symbols,
        universe_tiers={"researchable": symbols},
        liquidity_filter={"liquidity_scores": {symbol: 0.90 for symbol in symbols}},
        metadata=metadata,
    )


def _funnel(**overrides: object) -> DeterministicFunnel:
    values = {
        "max_candidates": 3,
        "profile": "classic",
        "theme_pool_enabled": True,
        "theme_pool_required": True,
        "theme_pool_use_markov_policy": True,
        "theme_pool_score_source": "smoothed",
        "theme_pool_fallback_to_raw_score": True,
        "theme_pool_min_theme_score": 0.58,
        "theme_pool_min_symbol_score": 0.55,
        "theme_pool_top_themes": 8,
        "theme_pool_max_symbols_per_theme": 30,
        "theme_pool_residual_ratio": 0.25,
        "theme_pool_min_residual_symbols": 0,
        "theme_pool_min_admitted_themes": 2,
        "theme_pool_allow_unthemed_residual": False,
        "theme_pool_include_risk_watch": True,
        "theme_pool_risk_watch_max_ratio": 0.50,
        "theme_pool_symbol_gate_mode": "classify",
    }
    values.update(overrides)
    return DeterministicFunnel(FunnelConfig(**values))


def test_theme_pool_generates_theme_member_candidates_before_ranking() -> None:
    quant = BranchResult(
        branch_name="quant",
        symbol_scores={
            "ADMIT1": 0.40,
            "LOW_SCORE": 0.95,
            "RISKY": 0.99,
            "UNTHEMED_HIGH": 1.00,
        },
    )

    output = _funnel().run(
        quant_result=quant,
        global_context=_context(_theme_rotation()),
    )

    assert output.funnel_metadata["theme_pool_status"] == "applied"
    assert "UNTHEMED_HIGH" not in output.candidates
    assert output.excluded_symbols["UNTHEMED_HIGH"] == "theme_pool_missing_theme_membership"
    assert set(output.candidates).issubset({"ADMIT1", "LOW_SCORE", "RISKY"})
    assert "RISKY" in output.candidates
    theme_pool = output.funnel_metadata["theme_pool"]
    assert theme_pool["core_symbol_count"] >= 1
    assert theme_pool["risk_watch_symbol_count"] == 1
    assert theme_pool["unthemed_exclusion_count"] == 1
    assert theme_pool["symbols"]["RISKY"]["bucket"] == "risk_watch_fake_breakout"
    assert theme_pool["symbols"]["RISKY"]["candidate_intent"] == "research_candidate_not_buy_signal"
    assert theme_pool["symbols"]["LOW_SCORE"]["bucket"] in {"extended_low_score", "extended"}


def test_theme_pool_required_does_not_silently_fallback_to_baseline() -> None:
    quant = BranchResult(
        branch_name="quant",
        symbol_scores={
            "ADMIT1": 0.40,
            "LOW_SCORE": 0.95,
            "RISKY": 0.99,
            "UNTHEMED_HIGH": 1.00,
        },
    )

    with pytest.raises(RuntimeError, match="theme_pool_required_but_theme_rotation_not_success"):
        _funnel().run(
            quant_result=quant,
            global_context=_context(theme_rotation=None),
        )


def test_theme_pool_disabled_preserves_legacy_ranking() -> None:
    quant = BranchResult(
        branch_name="quant",
        symbol_scores={
            "ADMIT1": 0.40,
            "LOW_SCORE": 0.95,
            "RISKY": 0.99,
            "UNTHEMED_HIGH": 1.00,
        },
    )

    output = _funnel(theme_pool_enabled=False).run(
        quant_result=quant,
        global_context=_context(theme_rotation=None),
    )

    assert output.candidates == ["UNTHEMED_HIGH", "RISKY", "LOW_SCORE"]
    assert output.funnel_metadata["theme_pool_status"] == "disabled"
