from __future__ import annotations

import pytest

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, ICDecision
from quant_investor.agents.portfolio_constructor import PortfolioConstructor


def _ic_decision(symbol: str, *, score: float = 0.8, confidence: float = 0.9) -> ICDecision:
    return ICDecision(
        symbol=symbol,
        action=ActionLabel.BUY,
        final_score=score,
        final_confidence=confidence,
        selected_symbols=[symbol],
        metadata={
            "symbol_scores": {symbol: score},
            "symbol_confidences": {symbol: confidence},
            "symbol_calibrated_confidences": {symbol: confidence},
            "symbol_momentum_strengths": {symbol: score},
            "symbol_actions": {symbol: ActionLabel.BUY},
            "symbol_modes": {symbol: "target"},
        },
    )


def _payload(
    symbols: list[str],
    *,
    sectors: dict[str, str] | None = None,
    risk_limits: dict[str, object] | None = None,
) -> dict[str, object]:
    sector_map = sectors or {symbol: "Tech" for symbol in symbols}
    base_risk_limits: dict[str, object] = {
        "gross_exposure_cap": 1.0,
        "max_weight": 0.6,
        "position_limits": {symbol: 0.6 for symbol in symbols},
        "blocked_symbols": [],
        "sector_caps": {},
    }
    if risk_limits:
        base_risk_limits.update(risk_limits)
    return {
        "ic_decisions": [_ic_decision(symbol) for symbol in symbols],
        "macro_verdict": BranchVerdict(metadata={"target_gross_exposure": 1.0}),
        "risk_limits": base_risk_limits,
        "existing_portfolio": {"current_weights": {}},
        "tradability_snapshot": {
            symbol: {
                "is_tradable": True,
                "sector": sector_map[symbol],
                "liquidity_score": 1.0,
            }
            for symbol in symbols
        },
    }


def _theme_exposure_map(theme_by_symbol: dict[str, str]) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "primary_theme_id": theme_id,
            "primary_theme_name": theme_id.rsplit("::", maxsplit=1)[-1],
            "phase": "confirmed_rotation",
            "symbol_score": 0.9,
            "risk_flags": [],
        }
        for symbol, theme_id in theme_by_symbol.items()
    }


def _theme_total(plan, theme_id: str) -> float:
    exposure_map = plan.metadata["theme_exposure_map"]
    return sum(
        weight
        for symbol, weight in plan.target_weights.items()
        if exposure_map.get(symbol, {}).get("primary_theme_id") == theme_id
    )


def test_portfolio_constructor_theme_caps_disabled_preserves_baseline() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    constructor = PortfolioConstructor()

    baseline = constructor.run(_payload(symbols))
    disabled = constructor.run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": False,
                "theme_exposure_map": _theme_exposure_map(
                    {symbol: "industry::AI" for symbol in symbols}
                ),
                "theme_caps": {"industry::AI": 0.10},
            },
        )
    )

    assert disabled.target_weights == baseline.target_weights
    assert disabled.target_gross_exposure == baseline.target_gross_exposure
    assert disabled.position_limits == baseline.position_limits


def test_portfolio_constructor_applies_single_theme_cap() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]

    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(
                    {symbol: "industry::AI" for symbol in symbols}
                ),
                "theme_caps": {"industry::AI": 0.35},
            },
        )
    )

    assert _theme_total(plan, "industry::AI") <= 0.35 + 1e-6
    assert plan.concentration_metrics["theme_cap_applied_count"] == 1
    assert any("theme_portfolio_cap_applied" in note for note in plan.construction_notes)


def test_portfolio_constructor_does_not_increase_other_themes_after_theme_cap() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    sectors = {
        "000001.SZ": "Tech",
        "000002.SZ": "Tech",
        "000003.SZ": "Industrial",
    }
    theme_map = {
        "000001.SZ": "industry::AI",
        "000002.SZ": "industry::AI",
        "000003.SZ": "industry::Robotics",
    }
    constructor = PortfolioConstructor()

    baseline = constructor.run(_payload(symbols, sectors=sectors))
    capped = constructor.run(
        _payload(
            symbols,
            sectors=sectors,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(theme_map),
                "theme_caps": {"industry::AI": 0.25, "industry::Robotics": 0.60},
            },
        )
    )

    assert _theme_total(capped, "industry::AI") <= 0.25 + 1e-6
    assert capped.target_weights["000003.SZ"] <= baseline.target_weights["000003.SZ"] + 1e-6
    assert capped.target_gross_exposure <= baseline.target_gross_exposure + 1e-6


def test_portfolio_constructor_distribution_theme_stricter_cap() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]

    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(
                    {symbol: "industry::AI" for symbol in symbols}
                ),
                "theme_caps": {"industry::AI": 0.15},
            },
        )
    )

    assert _theme_total(plan, "industry::AI") <= 0.15 + 1e-6
    assert plan.concentration_metrics["theme_caps"]["industry::AI"] == pytest.approx(0.15)


def test_portfolio_constructor_theme_caps_malformed_safe() -> None:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]

    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": "not-a-map",
                "theme_caps": {"industry::AI": "not-a-number"},
            },
        )
    )

    assert plan.target_weights
    assert any("theme_portfolio_cap_malformed" in note for note in plan.construction_notes)


@pytest.mark.parametrize(
    ("regime", "nav_cap", "max_positions"),
    [
        ("趋势上涨", 0.15, 2),
        ("震荡低波", 0.10, 1),
        ("震荡高波", 0.05, 1),
    ],
)
def test_portfolio_constructor_enforces_markov_tactical_lane(
    regime: str,
    nav_cap: float,
    max_positions: int,
) -> None:
    symbols = ["TACTICAL_A", "TACTICAL_B", "TACTICAL_C", "TECH_A"]
    baseline = PortfolioConstructor().run(_payload(symbols))
    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(
                    {
                        "TACTICAL_A": "tactical::a",
                        "TACTICAL_B": "tactical::b",
                        "TACTICAL_C": "tactical::c",
                        "TECH_A": "tech::ai",
                    }
                ),
                "theme_caps": {},
                "theme_tactical_lane": {
                    "enabled": True,
                    "status": "active",
                    "regime": regime,
                    "non_tech_symbols": [
                        "TACTICAL_A",
                        "TACTICAL_B",
                        "TACTICAL_C",
                    ],
                    "nav_cap": nav_cap,
                    "max_positions": max_positions,
                    "protocol_hash": "a" * 64,
                    "formal_kill_switch": False,
                },
            },
        )
    )

    tactical_weights = [
        plan.target_weights.get(symbol, 0.0)
        for symbol in ("TACTICAL_A", "TACTICAL_B", "TACTICAL_C")
    ]
    assert sum(tactical_weights) <= nav_cap + 1e-6
    assert sum(weight > 1e-8 for weight in tactical_weights) <= max_positions
    assert plan.target_weights.get("TECH_A", 0.0) <= baseline.target_weights["TECH_A"]
    tactical = plan.concentration_metrics["theme_tactical_lane"]
    assert tactical["applied"] is True
    assert tactical["position_count_after"] <= max_positions


def test_portfolio_constructor_closes_non_tech_lane_in_downtrend() -> None:
    symbols = ["TACTICAL_A", "TACTICAL_B", "TECH_A"]
    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(
                    {
                        "TACTICAL_A": "tactical::a",
                        "TACTICAL_B": "tactical::b",
                        "TECH_A": "tech::ai",
                    }
                ),
                "theme_caps": {},
                "theme_tactical_lane": {
                    "enabled": False,
                    "status": "closed_by_markov",
                    "regime": "趋势下跌",
                    "non_tech_symbols": ["TACTICAL_A", "TACTICAL_B"],
                    "nav_cap": 0.0,
                    "max_positions": 0,
                    "protocol_hash": "a" * 64,
                    "formal_kill_switch": False,
                },
            },
        )
    )

    assert "TACTICAL_A" not in plan.target_weights
    assert "TACTICAL_B" not in plan.target_weights
    assert plan.target_weights["TECH_A"] > 0.0
    assert plan.concentration_metrics["theme_tactical_lane"]["applied"] is True


def test_portfolio_constructor_ignores_tactical_lane_under_theme_kill_switch() -> None:
    symbols = ["TACTICAL_A", "TECH_A"]
    baseline = PortfolioConstructor().run(_payload(symbols))
    plan = PortfolioConstructor().run(
        _payload(
            symbols,
            risk_limits={
                "theme_portfolio_cap_enabled": True,
                "theme_exposure_map": _theme_exposure_map(
                    {
                        "TACTICAL_A": "tactical::a",
                        "TECH_A": "tech::ai",
                    }
                ),
                "theme_caps": {},
                "theme_tactical_lane": {
                    "enabled": False,
                    "status": "formal_kill_switch_active",
                    "regime": "趋势下跌",
                    "non_tech_symbols": ["TACTICAL_A"],
                    "nav_cap": 0.0,
                    "max_positions": 0,
                    "protocol_hash": "a" * 64,
                    "formal_kill_switch": True,
                },
            },
        )
    )

    assert plan.target_weights == baseline.target_weights
    assert plan.concentration_metrics["theme_tactical_lane"]["applied"] is False
