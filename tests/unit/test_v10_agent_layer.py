"""
V10 Agent 层默认行为测试。
"""

from __future__ import annotations

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BranchAgentOutput,
    MasterAgentOutput,
    RiskAgentOutput,
)
from quant_investor.pipeline.quant_investor_v10 import QuantInvestorV10


def test_v10_auto_selects_agent_models_from_available_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    investor = QuantInvestorV10(
        stock_pool=["000001.SZ"],
        verbose=False,
    )

    assert investor.agent_model == "deepseek-chat"
    assert investor.master_model == "deepseek-chat"


def test_v10_explicit_agent_model_is_shared_with_master_when_master_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    investor = QuantInvestorV10(
        stock_pool=["000001.SZ"],
        agent_model="gpt-5.4-mini",
        verbose=False,
    )

    assert investor.agent_model == "gpt-5.4-mini"
    assert investor.master_model == "gpt-5.4-mini"


def test_v10_agent_report_uses_named_subagents():
    agent_result = AgentEnhancedStrategy(
        agent_strategy=MasterAgentOutput(
            final_conviction="buy",
            final_score=0.22,
            confidence=0.66,
            portfolio_narrative="多分支共识偏正，风险可控。",
            risk_adjusted_exposure=0.58,
        ),
        branch_agent_outputs={
            "kline": BranchAgentOutput(
                branch_name="kline",
                conviction="buy",
                conviction_score=0.31,
                confidence=0.63,
                key_insights=["趋势延续性较强", "Chronos 预测可靠性稳定"],
            )
        },
        risk_agent_output=RiskAgentOutput(
            risk_assessment="elevated",
            max_recommended_exposure=0.55,
            risk_warnings=["VaR 上行，需控制仓位"],
        ),
        agent_layer_success=True,
    )

    section = QuantInvestorV10._build_agent_report_section(agent_result)

    assert "Master Agent IC" in section
    assert "KLine SubAgent" in section
    assert "Risk SubAgent" in section
