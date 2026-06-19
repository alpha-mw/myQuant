from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.market.dag.theme_context import build_theme_portfolio_constraints


def _context_with_theme_rotation(
    *,
    symbol_scores: dict[str, float] | None = None,
    symbol_primary_theme: dict[str, str] | None = None,
    symbol_phase: dict[str, str] | None = None,
    symbol_risk_flags: dict[str, list[str]] | None = None,
    theme_scores: dict[str, dict[str, object]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "enabled": True,
                "status": "success",
                "symbol_scores": symbol_scores or {},
                "symbol_primary_theme": symbol_primary_theme or {},
                "symbol_phase": symbol_phase or {},
                "symbol_risk_flags": symbol_risk_flags or {},
                "theme_scores": theme_scores or {},
                "top_themes": [],
            }
        }
    )


def test_build_theme_portfolio_constraints_disabled() -> None:
    result = build_theme_portfolio_constraints(
        global_context=_context_with_theme_rotation(),
        symbols=["000001.SZ"],
        enabled=False,
    )

    assert result["theme_portfolio_cap_enabled"] is False
    assert result["theme_exposure_map"] == {}
    assert result["theme_caps"] == {}
    assert "theme_portfolio_cap_disabled" in result["diagnostic_notes"]


def test_build_theme_portfolio_constraints_success() -> None:
    result = build_theme_portfolio_constraints(
        global_context=_context_with_theme_rotation(
            symbol_scores={"000001.SZ": 0.92, "000002.SZ": 0.88},
            symbol_primary_theme={"000001.SZ": "industry::AI", "000002.SZ": "industry::AI"},
            symbol_phase={"000001.SZ": "confirmed_rotation", "000002.SZ": "confirmed_rotation"},
            symbol_risk_flags={"000001.SZ": [], "000002.SZ": []},
            theme_scores={
                "industry::AI": {
                    "theme_id": "industry::AI",
                    "theme_name": "AI",
                    "score": 0.9,
                }
            },
        ),
        symbols=["000001.SZ", "000002.SZ"],
        enabled=True,
        max_theme_exposure=0.35,
    )

    assert set(result["theme_exposure_map"]) == {"000001.SZ", "000002.SZ"}
    assert result["theme_caps"]["industry::AI"] == pytest.approx(0.35)
    assert result["theme_names"]["industry::AI"] == "AI"
    assert result["metadata"]["theme_count"] == 1


def test_build_theme_portfolio_constraints_overextended_cap() -> None:
    result = build_theme_portfolio_constraints(
        global_context=_context_with_theme_rotation(
            symbol_scores={"000001.SZ": 0.91},
            symbol_primary_theme={"000001.SZ": "industry::AI"},
            symbol_phase={"000001.SZ": "overextended"},
            symbol_risk_flags={"000001.SZ": ["theme_overextended"]},
            theme_scores={"industry::AI": {"theme_name": "AI"}},
        ),
        symbols=["000001.SZ"],
        enabled=True,
        max_theme_exposure=0.35,
        overextended_max_theme_exposure=0.25,
    )

    assert result["theme_caps"]["industry::AI"] == pytest.approx(0.25)
    assert result["theme_phases"]["industry::AI"] == "overextended"


def test_build_theme_portfolio_constraints_distribution_cap_strictest() -> None:
    result = build_theme_portfolio_constraints(
        global_context=_context_with_theme_rotation(
            symbol_scores={"000001.SZ": 0.91, "000002.SZ": 0.89},
            symbol_primary_theme={"000001.SZ": "industry::AI", "000002.SZ": "industry::AI"},
            symbol_phase={"000001.SZ": "distribution", "000002.SZ": "overextended"},
            symbol_risk_flags={
                "000001.SZ": ["theme_distribution_risk"],
                "000002.SZ": ["theme_overextended"],
            },
            theme_scores={"industry::AI": {"theme_name": "AI"}},
        ),
        symbols=["000001.SZ", "000002.SZ"],
        enabled=True,
        max_theme_exposure=0.35,
        overextended_max_theme_exposure=0.25,
        distribution_max_theme_exposure=0.15,
    )

    assert result["theme_caps"]["industry::AI"] == pytest.approx(0.15)
    assert result["theme_phases"]["industry::AI"] == "distribution"


def test_build_theme_portfolio_constraints_malformed_safe() -> None:
    malformed_context = SimpleNamespace(
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": None,
                "symbol_primary_theme": "not-a-map",
                "symbol_phase": None,
                "symbol_risk_flags": {"000001.SZ": "not-a-list"},
            }
        }
    )

    result = build_theme_portfolio_constraints(
        global_context=malformed_context,
        symbols=["000001.SZ"],
        enabled=True,
    )

    assert result["theme_portfolio_cap_enabled"] is True
    assert result["theme_exposure_map"] == {}
    assert result["theme_caps"] == {}
    assert "theme_portfolio_cap_no_theme_data" in result["diagnostic_notes"]
