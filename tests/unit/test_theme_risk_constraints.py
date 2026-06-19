from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.market.dag.theme_context import build_theme_risk_constraints


SYMBOL = "000001.SZ"


def _context_with_theme(
    *,
    phase: str,
    risk_flags: list[str] | None = None,
    symbol_score: float = 0.78,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": {SYMBOL: symbol_score},
                "symbol_primary_theme": {SYMBOL: "industry::semiconductor"},
                "symbol_phase": {SYMBOL: phase},
                "symbol_risk_flags": {SYMBOL: list(risk_flags or [])},
                "theme_scores": {
                    "industry::semiconductor": {
                        "theme_name": "Semiconductor",
                        "score": 0.88,
                        "confidence": 0.7,
                        "member_count": 12,
                    }
                },
            }
        }
    )


def test_build_theme_risk_constraints_disabled():
    constraints = build_theme_risk_constraints(
        global_context=_context_with_theme(
            phase="overextended",
            risk_flags=["theme_overextended"],
        ),
        symbols=[SYMBOL],
        enabled=False,
    )

    assert constraints["theme_risk_guard_enabled"] is False
    assert constraints["theme_risk_by_symbol"] == {}
    assert constraints["theme_risk_flags"] == []
    assert constraints["theme_position_limits"] == {}
    assert constraints["diagnostic_notes"] == ["theme_risk_guard_disabled"]


def test_build_theme_risk_constraints_overextended():
    constraints = build_theme_risk_constraints(
        global_context=_context_with_theme(
            phase="overextended",
            risk_flags=["theme_overextended"],
        ),
        symbols=[SYMBOL],
        enabled=True,
        overextended_gross_cap=0.55,
        overextended_max_weight=0.09,
    )

    assert constraints["theme_action_cap"] == "hold"
    assert constraints["theme_position_limits"][SYMBOL] == pytest.approx(0.09)
    assert constraints["theme_gross_exposure_cap"] == pytest.approx(0.55)
    assert constraints["theme_risk_flags"] == ["theme_overextended"]
    assert constraints["theme_risk_by_symbol"][SYMBOL]["phase"] == "overextended"


def test_build_theme_risk_constraints_distribution_stricter_than_overextended():
    constraints = build_theme_risk_constraints(
        global_context=_context_with_theme(
            phase="distribution",
            risk_flags=["theme_distribution_risk", "theme_overextended"],
        ),
        symbols=[SYMBOL],
        enabled=True,
        overextended_gross_cap=0.60,
        overextended_max_weight=0.10,
        distribution_gross_cap=0.45,
        distribution_max_weight=0.08,
    )

    assert constraints["theme_action_cap"] == "hold"
    assert constraints["theme_position_limits"][SYMBOL] == pytest.approx(0.08)
    assert constraints["theme_gross_exposure_cap"] == pytest.approx(0.45)
    assert constraints["theme_risk_flags"] == [
        "theme_distribution_risk",
        "theme_overextended",
    ]


def test_build_theme_risk_constraints_fake_breakout_position_cap_only():
    constraints = build_theme_risk_constraints(
        global_context=_context_with_theme(
            phase="confirmed_rotation",
            risk_flags=["theme_fake_breakout_risk"],
        ),
        symbols=[SYMBOL],
        enabled=True,
        fake_breakout_max_weight=0.07,
    )

    assert constraints["theme_action_cap"] == ""
    assert constraints["theme_position_limits"][SYMBOL] == pytest.approx(0.07)
    assert constraints["theme_gross_exposure_cap"] is None
    assert constraints["theme_risk_flags"] == ["theme_fake_breakout_risk"]


def test_build_theme_risk_constraints_malformed_safe():
    context = SimpleNamespace(
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": None,
                "symbol_risk_flags": {SYMBOL: "not-a-list"},
            }
        }
    )

    constraints = build_theme_risk_constraints(
        global_context=context,
        symbols=[SYMBOL],
        enabled=True,
    )

    assert constraints["theme_risk_guard_enabled"] is True
    assert constraints["theme_risk_flags"] == []
    assert constraints["theme_position_limits"] == {}
    assert "theme_risk_guard_no_theme_risk" in constraints["diagnostic_notes"]
