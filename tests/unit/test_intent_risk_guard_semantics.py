"""
设计意图 1：RiskGuard 拥有 hard veto 权，不能被 IC 覆盖。

当前预期：组件语义通过。
"""

from __future__ import annotations

from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict
from quant_investor.agents.ic_coordinator import ICCoordinator
from quant_investor.agents.risk_guard import RiskGuard


def _branch(agent_name: str, *, symbol: str, score: float, risks: list[str] | None = None) -> BranchVerdict:
    return BranchVerdict(
        agent_name=agent_name,
        thesis=f"{agent_name} 对 {symbol} 给出结构化判断。",
        symbol=symbol,
        final_score=score,
        final_confidence=0.78,
        action=ActionLabel.BUY,
        investment_risks=list(risks or []),
    )


def test_risk_guard_hard_veto_is_not_overridden_by_ic() -> None:
    branch_verdicts = {
        "kline": _branch("KlineAgent", symbol="AAA", score=0.62),
        "quant": _branch(
            "QuantAgent",
            symbol="AAA",
            score=0.58,
            risks=["fraud investigation ongoing"],
        ),
    }

    risk_decision = RiskGuard().run(
        {
            "branch_verdicts": branch_verdicts,
            "portfolio_state": {"candidate_symbols": ["AAA"], "current_weights": {}},
            "constraints": {
                "gross_exposure_cap": 0.80,
                "max_weight": 0.30,
                "veto_keywords": ["fraud"],
            },
        }
    )
    ic_decision = ICCoordinator().run(
        {
            "branch_verdicts": branch_verdicts,
            "risk_decision": risk_decision,
        }
    )

    assert risk_decision.veto is True
    assert risk_decision.hard_veto is True
    assert risk_decision.action_cap is ActionLabel.HOLD
    assert "AAA" in risk_decision.blocked_symbols
    assert ic_decision.status is AgentStatus.VETOED
    assert ic_decision.action is ActionLabel.HOLD
    assert ic_decision.metadata["risk_veto"] is True
    assert ic_decision.metadata["raw_action"] == ActionLabel.BUY.value

