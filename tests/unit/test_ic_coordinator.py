"""
ICCoordinator 单元测试。
"""

from __future__ import annotations

from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict, RiskDecision, RiskLevel
from quant_investor.agents.ic_coordinator import ICCoordinator


def test_ic_cannot_exceed_action_cap_after_veto():
    branch_verdicts = {
        "kline": BranchVerdict(
            agent_name="KlineAgent",
            thesis="趋势分支偏多。",
            final_score=0.7,
            final_confidence=0.8,
        ),
        "quant": BranchVerdict(
            agent_name="QuantAgent",
            thesis="量化分支偏多。",
            final_score=0.5,
            final_confidence=0.7,
        ),
    }
    risk_decision = RiskDecision(
        status=AgentStatus.VETOED,
        risk_level=RiskLevel.EXTREME,
        hard_veto=True,
        veto=True,
        action_cap=ActionLabel.HOLD,
        gross_exposure_cap=0.0,
        target_exposure_cap=0.0,
        max_weight=0.0,
        reasons=["出现硬性风控事件。"],
    )

    decision = ICCoordinator().run(
        {
            "branch_verdicts": branch_verdicts,
            "risk_decision": risk_decision,
        }
    )

    assert decision.status is AgentStatus.VETOED
    assert decision.action_suggestion is ActionLabel.HOLD
    assert decision.metadata["action_cap_applied"] is True


def test_ic_coordinator_outputs_non_empty_thesis():
    branch_verdicts = {
        "macro": BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观偏中性。",
            final_score=0.0,
            final_confidence=0.6,
        ),
        "fundamental": BranchVerdict(
            agent_name="FundamentalAgent",
            thesis="基本面偏多。",
            final_score=0.3,
            final_confidence=0.7,
        ),
    }
    risk_decision = RiskDecision(
        status=AgentStatus.SUCCESS,
        risk_level=RiskLevel.MEDIUM,
        action_cap=ActionLabel.BUY,
        gross_exposure_cap=0.7,
        target_exposure_cap=0.7,
        max_weight=0.12,
    )

    decision = ICCoordinator().run(
        {
            "branch_verdicts": branch_verdicts,
            "risk_decision": risk_decision,
        }
    )

    assert decision.ic_thesis.strip()
    assert decision.action_suggestion in {ActionLabel.BUY, ActionLabel.HOLD}
    assert decision.agreement_points
