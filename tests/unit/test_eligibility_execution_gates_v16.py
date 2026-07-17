from __future__ import annotations

from quant_investor.agent_protocol import ActionLabel
from quant_investor.agents.eligibility_gate import EligibilityGate
from quant_investor.agents.execution_gate import ExecutionGate
from quant_investor.v16.candidate_pipeline import Stage2Decision


def _buy() -> Stage2Decision:
    return Stage2Decision(
        symbol="000001.SZ",
        action="BUY",
        selected_for_portfolio=True,
        target_weight=0.25,
        rationale="研究结论为买入。",
        risk_acceptance_rationale="接受已披露风险。",
    )


def test_eligibility_gate_runs_before_ic_and_owns_only_readiness() -> None:
    assert EligibilityGate.protocol_version == "v1"
    payload = {
        "symbol": "000001.SZ",
        "readiness": {
            "pit_ready": True,
            "data_ready": True,
            "factor_ready": False,
            "factor_ready_blockers": ["factor_governance_not_ready"],
            # Execution fields are deliberately irrelevant to this gate.
            "halted": False,
            "quote_fresh": True,
            "cash_sufficient": True,
            "human_authorized": True,
        },
    }
    decision = EligibilityGate().run(payload)

    assert not hasattr(decision, "research_action")
    assert decision.research_eligible is False
    assert decision.blockers == [
        "factor_ready_unconfirmed",
        "factor_governance_not_ready",
    ]


def test_blocked_execution_preserves_research_buy_and_target_weight() -> None:
    buy = _buy()
    eligibility = EligibilityGate().run(
        {
            "symbol": buy.symbol,
            "readiness": {
                "pit_ready": True,
                "data_ready": True,
                "factor_ready": True,
            },
        }
    )
    execution = ExecutionGate().run(
        {
            "ic_decision": buy,
            "eligibility_decision": eligibility,
            "execution_context": {
                "halted": True,
                "quote_fresh": False,
                "quote_price": 10.0,
                "lot_size": 100,
                "raw_target_shares": 250.0,
                "existing_shares": 0.0,
                "available_cash": 1_000.0,
                "human_authorized": False,
            },
        }
    )

    assert execution.research_action is ActionLabel.BUY
    assert execution.target_weight == 0.25
    assert execution.order_eligible is False
    assert execution.raw_order_shares == 250.0
    assert execution.executable_order_shares == 200.0
    assert execution.rounding_delta == 50.0
    assert {
        "symbol_suspended",
        "quote_not_fresh_or_unconfirmed",
        "cash_not_sufficient_or_unconfirmed",
        "human_authorization_missing",
    } <= set(execution.blockers)


def test_execution_gate_accepts_only_when_all_execution_checks_pass() -> None:
    assert ExecutionGate.protocol_version == "v1"
    buy = _buy()
    eligibility = EligibilityGate().run(
        {
            "symbol": buy.symbol,
            "readiness": {
                "pit_ready": True,
                "data_ready": True,
                "factor_ready": True,
            },
        }
    )
    execution = ExecutionGate().run(
        {
            "ic_decision": buy,
            "eligibility_decision": eligibility,
            "execution_context": {
                "halted": False,
                "suspended": False,
                "quote_fresh": True,
                "quote_price": 10.0,
                "lot_size": 100,
                "raw_target_shares": 300.0,
                "existing_shares": 0.0,
                "available_cash": 3_000.0,
                "human_authorized": True,
            },
        }
    )

    assert execution.order_eligible is True
    assert execution.raw_order_shares == 300.0
    assert execution.executable_order_shares == 300.0
    assert execution.rounding_delta == 0.0
    assert execution.blockers == []
    assert all(execution.checks.values())


def test_execution_gate_blocks_buy_smaller_than_one_lot_without_rewriting_buy() -> None:
    buy = _buy()
    eligibility = EligibilityGate().run(
        {
            "symbol": buy.symbol,
            "readiness": {
                "pit_ready": True,
                "data_ready": True,
                "factor_ready": True,
            },
        }
    )
    execution = ExecutionGate().run(
        {
            "ic_decision": buy,
            "eligibility_decision": eligibility,
            "execution_context": {
                "halted": False,
                "quote_fresh": True,
                "quote_price": 10.0,
                "lot_size": 100,
                "raw_target_shares": 80.0,
                "existing_shares": 0.0,
                "available_cash": 10_000.0,
                "human_authorized": True,
            },
        }
    )

    assert execution.research_action is ActionLabel.BUY
    assert execution.target_weight == 0.25
    assert execution.raw_order_shares == 80.0
    assert execution.executable_order_shares == 0.0
    assert execution.rounding_delta == 80.0
    assert execution.order_eligible is False
    assert "lot_not_valid_or_unconfirmed" in execution.blockers
