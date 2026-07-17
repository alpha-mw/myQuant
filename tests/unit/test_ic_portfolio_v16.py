from __future__ import annotations

import pytest

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, RiskLevel
from quant_investor.agents.ic_coordinator import V16ICCoordinator
from quant_investor.agents.portfolio_constructor import V16PortfolioConstructor
from quant_investor.market.dag.decision import (
    _run_v16_decision_phase,
    _run_v16_eligibility_phase,
)
from quant_investor.v16.candidate_pipeline import PosteriorMenuItem, Stage2Decision


def _decision(
    symbol: str,
    action: str,
    weight: float,
    *,
    selected: bool | None = None,
) -> Stage2Decision:
    return Stage2Decision(
        symbol=symbol,
        action=action,
        selected_for_portfolio=weight > 0.0 if selected is None else selected,
        target_weight=weight,
        rationale="IC已形成明确结论。",
        risk_acceptance_rationale="已明确接受研究风险。",
    )


def _menu(decisions: list[Stage2Decision]) -> list[PosteriorMenuItem]:
    return [
        PosteriorMenuItem(
            symbol=decision.symbol,
            posterior_win_rate=0.6,
            posterior_expected_alpha=0.01,
            posterior_edge_after_costs=0.005,
        )
        for decision in decisions
    ]


def _portfolio_payload(
    decisions: list[Stage2Decision],
    *,
    cash_ratio: float,
    existing_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "menu": _menu(decisions),
        "stage2_decisions": decisions,
        "cash_ratio": cash_ratio,
        "existing_weights": (
            {decision.symbol: 0.0 for decision in decisions}
            if existing_weights is None
            else existing_weights
        ),
        "total_capital": 10_000.0,
        "reference_price_by_symbol": {
            decision.symbol: 10.0 for decision in decisions if decision.action == "BUY"
        },
        "existing_shares_by_symbol": {
            decision.symbol: (
                decision.target_weight * 10_000.0 / 10.0 if decision.action == "HOLD" else 0.0
            )
            for decision in decisions
        },
    }


def test_v16_ic_maps_the_authoritative_stage2_contract() -> None:
    decision = V16ICCoordinator().run(
        {
            "stage2_decision": {
                "symbol": "AAA",
                "action": "BUY",
                "selected_for_portfolio": True,
                "target_weight": 0.2,
                "rationale": "净边际支持买入。",
                "risk_acceptance_rationale": "接受波动情景。",
            }
        }
    )
    assert decision == Stage2Decision(
        symbol="AAA",
        action="BUY",
        selected_for_portfolio=True,
        target_weight=0.2,
        rationale="净边际支持买入。",
        risk_acceptance_rationale="接受波动情景。",
    )

    with pytest.raises(ValueError, match="BUY, HOLD, AVOID, or SELL"):
        V16ICCoordinator().run(
            {
                "stage2_decision": {
                    "symbol": "AAA",
                    "action": "WATCH",
                    "selected_for_portfolio": False,
                    "target_weight": 0.0,
                    "rationale": "无有效动作。",
                    "risk_acceptance_rationale": None,
                }
            }
        )


def test_v16_portfolio_only_validates_and_maps_stage2_weights() -> None:
    decisions = [
        _decision("BUY", "BUY", 0.3),
        _decision("HOLD", "HOLD", 0.2),
        _decision("AVOID", "AVOID", 0.0),
        _decision("SELL", "SELL", 0.0),
    ]
    plan = V16PortfolioConstructor().run(
        _portfolio_payload(
            decisions,
            cash_ratio=0.5,
            existing_weights={
                "BUY": 0.0,
                "HOLD": 0.2,
                "AVOID": 0.0,
                "SELL": 0.1,
            },
        )
    )

    assert [target.symbol for target in plan.positions] == ["BUY", "HOLD"]
    assert plan.positions[0].target_capital == 3_000.0
    assert plan.positions[0].raw_target_shares == 300.0
    assert plan.positions[1].raw_target_shares == 200.0
    assert plan.cash_ratio == 0.5
    assert plan.cash_amount == 5_000.0


@pytest.mark.parametrize(
    ("decisions", "cash", "current", "message"),
    [
        ([_decision("A", "BUY", 0.0)], 1.0, {"A": 0.0}, "BUY requires"),
        ([_decision("A", "HOLD", 0.2)], 0.8, {"A": 0.0}, "HOLD must preserve"),
        ([_decision("A", "AVOID", 0.1)], 0.9, {"A": 0.0}, "AVOID requires"),
        ([_decision("A", "SELL", 0.1)], 0.9, {"A": 0.1}, "SELL requires"),
        ([_decision("A", "BUY", 0.2)], 0.7, {"A": 0.0}, "must equal 1"),
    ],
)
def test_v16_portfolio_fails_closed_without_silent_correction(
    decisions: list[Stage2Decision],
    cash: float,
    current: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        V16PortfolioConstructor().run(
            _portfolio_payload(
                decisions,
                cash_ratio=cash,
                existing_weights=current,
            )
        )


def test_v16_portfolio_rejects_more_than_twelve_positive_weights() -> None:
    decisions = [_decision(f"S{index:02d}", "BUY", 0.05) for index in range(13)]
    with pytest.raises(ValueError, match="exceed 12"):
        V16PortfolioConstructor().run(_portfolio_payload(decisions, cash_ratio=0.35))


def test_v16_portfolio_rejects_policy_inputs_it_does_not_own() -> None:
    decisions = [_decision("A", "BUY", 0.2)]
    payload = _portfolio_payload(decisions, cash_ratio=0.8)
    payload["risk_advisory"] = {"severity": "extreme"}
    with pytest.raises(ValueError, match="does not accept policy constraints"):
        V16PortfolioConstructor().run(payload)


def test_v16_portfolio_missing_buy_reference_price_fails_closed() -> None:
    decisions = [_decision("A", "BUY", 0.2)]
    payload = _portfolio_payload(decisions, cash_ratio=0.8)
    payload["reference_price_by_symbol"] = {}
    with pytest.raises(ValueError, match="BUY reference price missing"):
        V16PortfolioConstructor().run(payload)


def test_v16_phase_keeps_advisory_out_of_action_order_and_weights() -> None:
    menu = [
        PosteriorMenuItem(
            symbol="AAA",
            posterior_win_rate=0.6,
            posterior_expected_alpha=0.01,
            posterior_edge_after_costs=0.005,
        )
    ]
    stage2 = [
        {
            "symbol": "AAA",
            "action": "BUY",
            "selected_for_portfolio": True,
            "target_weight": 0.2,
            "rationale": "净边际支持买入。",
            "risk_acceptance_rationale": "接受所列风险情景。",
        }
    ]
    common = {
        "menu": menu,
        "stage2_responses": stage2,
        "branch_summaries": {"quant": BranchVerdict(agent_name="quant")},
        "macro_verdict": None,
        "cash_ratio": 0.8,
        "existing_weights": {"AAA": 0.0},
        "total_capital": 10_000.0,
        "reference_price_by_symbol": {"AAA": 10.0},
        "existing_shares_by_symbol": {"AAA": 0.0},
        "eligibility_by_symbol": _run_v16_eligibility_phase(
            symbols=["AAA"],
            readiness_by_symbol={
                "AAA": {
                    "pit_ready": True,
                    "data_ready": True,
                    "factor_ready": True,
                }
            },
        ),
        "execution_context_by_symbol": {
            "AAA": {
                "halted": False,
                "quote_fresh": True,
                "quote_price": 10.0,
                "lot_size": 100,
                "available_cash": 2_000.0,
                "human_authorized": False,
            }
        },
    }
    low = _run_v16_decision_phase(**common)
    extreme = _run_v16_decision_phase(
        **common,
        risk_advisory_context={"flags": ["fraud investigation"]},
    )

    assert low.risk_advisory.severity is RiskLevel.LOW
    assert extreme.risk_advisory.severity is RiskLevel.EXTREME
    assert low.ic_decisions == extreme.ic_decisions
    assert low.portfolio_plan == extreme.portfolio_plan
    assert extreme.ic_decisions[0].action == "BUY"
    assert extreme.portfolio_plan.positions[0].target_weight == 0.2
    assert extreme.portfolio_plan.positions[0].raw_target_shares == 200.0
    assert extreme.execution_decisions[0].research_action is ActionLabel.BUY
    assert extreme.execution_decisions[0].order_eligible is False
    assert "human_authorization_missing" in extreme.execution_decisions[0].blockers
