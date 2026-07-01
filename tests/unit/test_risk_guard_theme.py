from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.market.dag.theme_context import build_theme_risk_constraints


SYMBOL = "000001.SZ"


def _payload(constraints: dict[str, object] | None = None) -> dict[str, object]:
    base_constraints: dict[str, object] = {
        "gross_exposure_cap": 0.90,
        "max_weight": 0.20,
        "risk_flags": [],
    }
    base_constraints.update(constraints or {})
    return {
        "branch_verdicts": {
            "quant": BranchVerdict(
                agent_name="quant",
                thesis="quant ok",
                symbol=SYMBOL,
                final_score=0.35,
                final_confidence=0.75,
            )
        },
        "macro_verdict": BranchVerdict(agent_name="macro", thesis="macro stable"),
        "portfolio_state": {
            "candidate_symbols": [SYMBOL],
            "current_weights": {},
        },
        "constraints": base_constraints,
    }


def test_risk_guard_theme_disabled_preserves_baseline():
    risk_guard = RiskGuard()

    baseline = risk_guard.run(_payload())
    with_disabled_theme = risk_guard.run(
        _payload(
            {
                "theme_risk_guard_enabled": False,
                "theme_risk_flags": ["theme_overextended"],
                "theme_action_cap": "hold",
                "theme_gross_exposure_cap": 0.60,
                "theme_position_limits": {SYMBOL: 0.10},
            }
        )
    )

    assert with_disabled_theme.action_cap == baseline.action_cap
    assert with_disabled_theme.gross_exposure_cap == pytest.approx(
        baseline.gross_exposure_cap
    )
    assert with_disabled_theme.max_weight == pytest.approx(baseline.max_weight)
    assert with_disabled_theme.position_limits == baseline.position_limits


def test_risk_guard_weak_macro_preserves_concentrated_single_name_cap():
    payload = _payload({"max_weight": 0.50})
    payload["macro_verdict"] = BranchVerdict(
        agent_name="macro",
        thesis="macro weak",
        final_score=-0.30,
        metadata={"target_gross_exposure": 0.90},
    )

    result = RiskGuard().run(payload)

    assert result.action_cap == ActionLabel.HOLD
    assert result.gross_exposure_cap == pytest.approx(0.50)
    assert result.max_weight == pytest.approx(0.50)
    assert result.position_limits[SYMBOL] == pytest.approx(0.50)


def test_risk_guard_multiple_risks_preserve_concentrated_single_name_cap():
    result = RiskGuard().run(
        _payload(
            {
                "max_weight": 0.50,
                "risk_flags": ["risk one", "risk two", "risk three"],
            }
        )
    )

    assert result.action_cap == ActionLabel.HOLD
    assert result.gross_exposure_cap == pytest.approx(0.60)
    assert result.max_weight == pytest.approx(0.50)
    assert result.position_limits[SYMBOL] == pytest.approx(0.50)


def test_risk_guard_applies_overextended_theme_overlay():
    result = RiskGuard().run(
        _payload(
            {
                "theme_risk_guard_enabled": True,
                "theme_risk_flags": ["theme_overextended"],
                "theme_action_cap": "hold",
                "theme_gross_exposure_cap": 0.60,
                "theme_position_limits": {SYMBOL: 0.10},
                "theme_risk_by_symbol": {
                    SYMBOL: {"phase": "overextended", "risk_flags": ["theme_overextended"]}
                },
            }
        )
    )

    assert result.action_cap == ActionLabel.HOLD
    assert result.gross_exposure_cap <= 0.60
    assert result.position_limits[SYMBOL] <= 0.10
    assert result.status in {AgentStatus.DEGRADED, AgentStatus.VETOED}
    assert result.metadata["theme_risk_guard_enabled"] is True
    assert result.metadata["theme_risk_flags"] == ["theme_overextended"]


def test_risk_guard_applies_distribution_stricter_cap():
    result = RiskGuard().run(
        _payload(
            {
                "theme_risk_guard_enabled": True,
                "theme_risk_flags": ["theme_distribution_risk"],
                "theme_action_cap": "hold",
                "theme_gross_exposure_cap": 0.45,
                "theme_position_limits": {SYMBOL: 0.08},
            }
        )
    )

    assert result.action_cap == ActionLabel.HOLD
    assert result.gross_exposure_cap <= 0.45
    assert result.position_limits[SYMBOL] <= 0.08


def test_risk_guard_theme_fake_breakout_position_cap_without_hard_veto():
    result = RiskGuard().run(
        _payload(
            {
                "theme_risk_guard_enabled": True,
                "theme_risk_flags": ["theme_fake_breakout_risk"],
                "theme_position_limits": {SYMBOL: 0.10},
            }
        )
    )

    assert result.position_limits[SYMBOL] <= 0.10
    assert result.hard_veto is False
    assert result.veto is False


def test_risk_guard_theme_malformed_constraints_safe():
    risk_guard = RiskGuard()

    baseline = risk_guard.run(_payload())
    result = risk_guard.run(
        _payload(
            {
                "theme_risk_guard_enabled": True,
                "theme_risk_flags": "theme_overextended",
                "theme_action_cap": "not-an-action",
                "theme_gross_exposure_cap": "not-a-float",
                "theme_position_limits": [("bad", "shape")],
                "theme_risk_by_symbol": "bad",
            }
        )
    )

    assert result.action_cap == baseline.action_cap
    assert result.gross_exposure_cap == pytest.approx(baseline.gross_exposure_cap)
    assert result.position_limits == baseline.position_limits


def test_theme_risk_helper_constraints_feed_risk_guard():
    global_context = SimpleNamespace(
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": {SYMBOL: 0.81},
                "symbol_primary_theme": {SYMBOL: "industry::semiconductor"},
                "symbol_phase": {SYMBOL: "overextended"},
                "symbol_risk_flags": {SYMBOL: ["theme_overextended_no_chase"]},
            }
        }
    )
    theme_constraints = build_theme_risk_constraints(
        global_context=global_context,
        symbols=[SYMBOL],
        enabled=True,
    )

    result = RiskGuard().run(_payload(theme_constraints))

    assert result.action_cap == ActionLabel.HOLD
    assert result.gross_exposure_cap <= 0.60
    assert result.position_limits[SYMBOL] <= 0.10
    assert result.hard_veto is False
