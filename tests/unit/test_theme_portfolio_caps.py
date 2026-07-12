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
    symbol_theme_memberships: dict[str, list[str]] | None = None,
    protocol_v2: dict[str, object] | None = None,
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
                "symbol_theme_memberships": symbol_theme_memberships or {},
                "theme_scores": theme_scores or {},
                "protocol_v2": protocol_v2 or {},
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


def test_build_theme_portfolio_constraints_exposes_executable_tactical_lane() -> None:
    protocol = {
        "status": "formal",
        "formal_enabled": True,
        "formal_kill_switch": False,
        "protocol_hash": "a" * 64,
        "as_of": "2026-07-10",
        "prequalified_pool": ["tech::ai", "tactical::gold"],
        "tactical_lane_cap": {
            "regime": "震荡低波",
            "non_tech_nav_cap": 0.10,
            "non_tech_max_positions": 1,
            "enabled": True,
        },
        "states": {
            "tech::ai": {"mandate": "technology"},
            "tactical::gold": {"mandate": "tactical"},
        },
    }
    context = _context_with_theme_rotation(
        symbol_scores={"TECH": 0.9, "GOLD": 0.8},
        symbol_primary_theme={
            "TECH": "tech::ai",
            "GOLD": "tactical::gold",
        },
        symbol_theme_memberships={
            "TECH": ["tech::ai"],
            "GOLD": ["tactical::gold"],
        },
        symbol_phase={"TECH": "confirmed_rotation", "GOLD": "confirmed_rotation"},
        symbol_risk_flags={"TECH": [], "GOLD": []},
        theme_scores={
            "tech::ai": {"theme_name": "AI"},
            "tactical::gold": {"theme_name": "Gold"},
        },
        protocol_v2=protocol,
    )
    context.metadata["theme_rotation"]["symbol_theme_membership_details"] = {
        "TECH": [
            {
                "schema_version": "theme_membership.v2",
                "membership_id": "tech",
                "theme_id": "tech::ai",
                "theme_name": "AI",
                "theme_type": "technology",
                "symbol": "TECH",
                "effective_from": "2026-01-01",
                "available_at": "2026-07-01",
                "confidence": 0.9,
            }
        ],
        "GOLD": [
            {
                "schema_version": "theme_membership.v2",
                "membership_id": "gold",
                "theme_id": "tactical::gold",
                "theme_name": "Gold",
                "theme_type": "concept",
                "symbol": "GOLD",
                "effective_from": "2026-01-01",
                "available_at": "2026-07-01",
                "confidence": 0.9,
            }
        ],
    }

    result = build_theme_portfolio_constraints(
        global_context=context,
        symbols=["TECH", "GOLD"],
        enabled=True,
    )

    tactical = result["theme_tactical_lane"]
    assert tactical["enabled"] is True
    assert tactical["status"] == "active"
    assert tactical["regime"] == "震荡低波"
    assert tactical["non_tech_symbols"] == ["GOLD"]
    assert tactical["nav_cap"] == pytest.approx(0.10)
    assert tactical["max_positions"] == 1
    assert tactical["protocol_hash"] == "a" * 64


def test_tactical_lane_downtrend_remains_enforced_while_channel_is_closed() -> None:
    protocol = {
        "status": "formal",
        "formal_enabled": True,
        "formal_kill_switch": False,
        "protocol_hash": "b" * 64,
        "tactical_lane_cap": {
            "regime": "趋势下跌",
            "non_tech_nav_cap": 0.0,
            "non_tech_max_positions": 0,
            "enabled": False,
        },
        "states": {"tactical::gold": {"mandate": "tactical"}},
    }
    context = _context_with_theme_rotation(
        symbol_scores={"GOLD": 0.8},
        symbol_primary_theme={"GOLD": "tactical::gold"},
        symbol_theme_memberships={"GOLD": ["tactical::gold"]},
        symbol_phase={"GOLD": "confirmed_rotation"},
        symbol_risk_flags={"GOLD": []},
        theme_scores={"tactical::gold": {"theme_name": "Gold"}},
        protocol_v2=protocol,
    )

    tactical = build_theme_portfolio_constraints(
        global_context=context,
        symbols=["GOLD"],
        enabled=True,
    )["theme_tactical_lane"]

    assert tactical["enabled"] is True
    assert tactical["channel_open"] is False
    assert tactical["status"] == "closed_by_markov"
    assert tactical["nav_cap"] == 0.0
    assert tactical["max_positions"] == 0
    assert tactical["non_tech_symbols"] == ["GOLD"]


def test_prequalified_tactical_classification_ignores_unadmitted_tech_secondary() -> None:
    protocol = {
        "status": "prequalified",
        "formal_enabled": True,
        "formal_kill_switch": False,
        "protocol_hash": "c" * 64,
        "as_of": "2026-07-10",
        "prequalified_pool": ["tactical::gold"],
        "tactical_lane_cap": {
            "regime": "趋势上涨",
            "non_tech_nav_cap": 0.15,
            "non_tech_max_positions": 2,
            "enabled": True,
        },
        "states": {
            "tactical::gold": {"mandate": "tactical"},
            "tech::ai": {"mandate": "technology"},
        },
    }
    context = _context_with_theme_rotation(
        symbol_scores={"MIXED": 0.8},
        symbol_primary_theme={"MIXED": "tactical::gold"},
        symbol_theme_memberships={"MIXED": ["tactical::gold", "tech::ai"]},
        symbol_phase={"MIXED": "confirmed_rotation"},
        symbol_risk_flags={"MIXED": []},
        theme_scores={
            "tactical::gold": {"theme_name": "Gold"},
            "tech::ai": {"theme_name": "AI"},
        },
        protocol_v2=protocol,
    )
    context.metadata["theme_rotation"]["symbol_theme_membership_details"] = {
        "MIXED": [
            {
                "schema_version": "theme_membership.v2",
                "membership_id": "gold",
                "theme_id": "tactical::gold",
                "theme_name": "Gold",
                "theme_type": "concept",
                "symbol": "MIXED",
                "effective_from": "2026-01-01",
                "available_at": "2026-07-01",
                "confidence": 0.9,
            },
            {
                "schema_version": "theme_membership.v2",
                "membership_id": "ai",
                "theme_id": "tech::ai",
                "theme_name": "AI",
                "theme_type": "technology",
                "symbol": "MIXED",
                "effective_from": "2026-01-01",
                "available_at": "2026-07-01",
                "confidence": 0.9,
            },
        ]
    }

    tactical = build_theme_portfolio_constraints(
        global_context=context,
        symbols=["MIXED"],
        enabled=True,
    )["theme_tactical_lane"]

    assert tactical["enabled"] is True
    assert tactical["status"] == "active"
    assert tactical["non_tech_symbols"] == ["MIXED"]
    assert tactical["source"] == "prospective_prequalified_pit_memberships"
