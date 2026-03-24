"""
RiskGuard 与 MacroAgent 单元测试。
"""

from __future__ import annotations

from copy import deepcopy

from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.agents.risk_guard import RiskGuard


def test_macro_agent_returns_single_market_level_verdict():
    agent = MacroAgent()

    verdict = agent.run(
        {
            "market_snapshot": {
                "regime": "risk_on",
                "macro_score": 0.35,
                "liquidity_score": 0.25,
                "volatility_percentile": 32.0,
                "policy_signal": "supportive",
                "symbols": ["000001.SZ", "600519.SH"],
            }
        }
    )

    assert isinstance(verdict, BranchVerdict)
    assert verdict.symbol is None
    assert verdict.metadata["symbol"] is None
    assert verdict.metadata["regime"] == "risk_on"
    assert "target_gross_exposure" in verdict.metadata
    assert "style_bias" in verdict.metadata


def test_risk_guard_outputs_caps_without_mutating_branch_evidence():
    verdict = BranchVerdict(
        agent_name="QuantAgent",
        thesis="量化分支继续偏多。",
        symbol="000001.SZ",
        final_score=0.6,
        final_confidence=0.7,
        investment_risks=["fraud investigation ongoing"],
    )
    original = deepcopy(verdict)

    decision = RiskGuard().run(
        {
            "branch_verdicts": {"quant": verdict},
            "macro_verdict": None,
            "portfolio_state": {"candidate_symbols": ["000001.SZ"]},
            "constraints": {
                "force_veto": True,
                "action_cap": ActionLabel.BUY,
                "veto_action_cap": ActionLabel.HOLD,
                "gross_exposure_cap": 0.8,
                "max_weight": 0.2,
            },
        }
    )

    assert decision.status is AgentStatus.VETOED
    assert decision.veto is True
    assert decision.action_cap is ActionLabel.HOLD
    assert decision.gross_exposure_cap == 0.0
    assert decision.max_weight == 0.0
    assert verdict == original
