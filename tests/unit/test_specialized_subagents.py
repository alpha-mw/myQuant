"""
V10.1 专属 SubAgent 单元测试。

测试：
1. 专属合约的序列化/反序列化和继承关系
2. 每个专属 SubAgent 的输出验证逻辑
3. Orchestrator 的专属输入准备
4. Master Agent 的 IC 辩论增强
5. 向后兼容性
"""

from __future__ import annotations

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    BranchAgentInput,
    BranchAgentOutput,
    FundamentalAgentInput,
    FundamentalAgentOutput,
    ICDebateRound,
    IntelligenceAgentInput,
    IntelligenceAgentOutput,
    KLineAgentInput,
    KLineAgentOutput,
    MacroAgentInput,
    MacroAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    QuantAgentInput,
    QuantAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
)


# ---------------------------------------------------------------------------
# 1. Contract inheritance and backward compat
# ---------------------------------------------------------------------------

def test_backward_compat_aliases():
    """BranchAgentInput/Output are aliases for Base classes."""
    assert BranchAgentInput is BaseBranchAgentInput
    assert BranchAgentOutput is BaseBranchAgentOutput


def test_specialized_inputs_inherit_from_base():
    """All specialized inputs inherit from BaseBranchAgentInput."""
    for cls in (KLineAgentInput, QuantAgentInput, FundamentalAgentInput,
                IntelligenceAgentInput, MacroAgentInput):
        assert issubclass(cls, BaseBranchAgentInput), f"{cls.__name__} must inherit BaseBranchAgentInput"


def test_specialized_outputs_inherit_from_base():
    """All specialized outputs inherit from BaseBranchAgentOutput."""
    for cls in (KLineAgentOutput, QuantAgentOutput, FundamentalAgentOutput,
                IntelligenceAgentOutput, MacroAgentOutput):
        assert issubclass(cls, BaseBranchAgentOutput), f"{cls.__name__} must inherit BaseBranchAgentOutput"


def test_specialized_output_serialization_roundtrip():
    """Specialized outputs can serialize and deserialize correctly."""
    kline = KLineAgentOutput(
        branch_name="kline",
        conviction="buy",
        conviction_score=0.3,
        confidence=0.7,
        key_insights=["趋势强劲"],
        risk_flags=[],
        disagreements_with_algo=[],
        symbol_views={"000001.SZ": "看涨"},
        trend_assessment="strong_uptrend",
        model_reliability=0.8,
        pattern_signals=["双底"],
        timeframe_alignment="aligned_bullish",
        reversal_risk=0.1,
        predicted_return_assessment={"000001.SZ": "agrees"},
    )

    json_str = kline.model_dump_json()
    restored = KLineAgentOutput.model_validate_json(json_str)
    assert restored.trend_assessment == "strong_uptrend"
    assert restored.model_reliability == 0.8
    assert restored.reversal_risk == 0.1
    assert restored.timeframe_alignment == "aligned_bullish"


def test_specialized_output_as_base_output():
    """Specialized output can be used wherever BaseBranchAgentOutput is expected."""
    kline = KLineAgentOutput(
        branch_name="kline",
        conviction="buy",
        conviction_score=0.3,
        confidence=0.7,
        key_insights=["趋势强劲"],
    )

    # Can be assigned to a dict expecting BaseBranchAgentOutput
    outputs: dict[str, BaseBranchAgentOutput] = {"kline": kline}
    assert outputs["kline"].conviction == "buy"
    assert outputs["kline"].conviction_score == 0.3


# ---------------------------------------------------------------------------
# 2. Specialized contract fields
# ---------------------------------------------------------------------------

def test_kline_input_has_specialized_fields():
    inp = KLineAgentInput(
        branch_name="kline",
        base_score=0.2,
        final_score=0.25,
        confidence=0.7,
        predicted_returns={"000001.SZ": 0.05},
        kronos_confidence=0.82,
        chronos_confidence=0.75,
        model_agreement=0.6,
        volatility_percentile=35.0,
    )
    assert inp.predicted_returns == {"000001.SZ": 0.05}
    assert inp.kronos_confidence == 0.82
    assert inp.model_agreement == 0.6


def test_quant_output_has_specialized_fields():
    out = QuantAgentOutput(
        branch_name="quant",
        conviction="neutral",
        conviction_score=0.05,
        confidence=0.6,
        key_insights=["因子动量有效"],
        factor_quality_assessment={"momentum": "robust", "value": "decaying"},
        regime_suitability=0.7,
        overfitting_risk=0.2,
        factor_conflicts=["momentum vs value divergence"],
        recommended_factor_tilts={"momentum": "overweight"},
    )
    assert out.factor_quality_assessment["momentum"] == "robust"
    assert out.overfitting_risk == 0.2


def test_fundamental_output_has_specialized_fields():
    out = FundamentalAgentOutput(
        branch_name="fundamental",
        conviction="buy",
        conviction_score=0.4,
        confidence=0.65,
        key_insights=["ROE 高于行业均值"],
        earnings_quality_assessment="high",
        accounting_red_flags=[],
        valuation_stance="cheap",
        management_signal="positive",
    )
    assert out.earnings_quality_assessment == "high"
    assert out.valuation_stance == "cheap"


def test_intelligence_output_has_specialized_fields():
    out = IntelligenceAgentOutput(
        branch_name="intelligence",
        conviction="sell",
        conviction_score=-0.3,
        confidence=0.55,
        key_insights=["极度贪婪"],
        sentiment_regime="extreme_greed",
        contrarian_signal="sell_contrarian",
        information_asymmetry="distributing",
        event_timing_risk=0.7,
    )
    assert out.sentiment_regime == "extreme_greed"
    assert out.contrarian_signal == "sell_contrarian"
    assert out.event_timing_risk == 0.7


def test_macro_output_has_specialized_fields():
    out = MacroAgentOutput(
        branch_name="macro",
        conviction="neutral",
        conviction_score=0.0,
        confidence=0.5,
        key_insights=["流动性中性"],
        macro_regime_assessment="late_cycle",
        liquidity_outlook="restrictive",
        systemic_risk_level=0.4,
        regime_transition_risk=0.6,
        uniform_score_appropriateness="questionable",
    )
    assert out.liquidity_outlook == "restrictive"
    assert out.regime_transition_risk == 0.6


# ---------------------------------------------------------------------------
# 3. Enhanced Risk contracts
# ---------------------------------------------------------------------------

def test_risk_input_has_branch_disagreement():
    risk_in = RiskAgentInput(
        risk_metrics_summary={"var_95": 0.03},
        regime="趋势上涨",
        branch_disagreement_level=0.45,
        stop_loss_levels={"000001.SZ": 10.5},
    )
    assert risk_in.branch_disagreement_level == 0.45
    assert risk_in.stop_loss_levels["000001.SZ"] == 10.5


def test_risk_output_has_enhanced_fields():
    risk_out = RiskAgentOutput(
        risk_assessment="high",
        max_recommended_exposure=0.4,
        tail_risk_assessment="elevated",
        correlation_breakdown_risk=0.6,
        drawdown_scenario="市场可能因流动性收紧回撤 8-12%",
    )
    assert risk_out.tail_risk_assessment == "elevated"
    assert risk_out.correlation_breakdown_risk == 0.6


# ---------------------------------------------------------------------------
# 4. Enhanced Master Agent contracts
# ---------------------------------------------------------------------------

def test_ic_debate_round_serialization():
    rd = ICDebateRound(
        round_number=2,
        topic="技术面与基本面分歧",
        arguments=["K线看跌但基本面强劲", "短期回调不影响中期逻辑"],
        resolution="采纳基本面视角，适度降低仓位以应对短期波动",
    )
    data = rd.model_dump()
    assert data["round_number"] == 2
    assert len(data["arguments"]) == 2


def test_master_output_has_enhanced_fields():
    ic = MasterAgentOutput(
        final_conviction="buy",
        final_score=0.3,
        confidence=0.7,
        debate_rounds=[
            ICDebateRound(round_number=1, topic="共识确认", resolution="多数分支看涨"),
            ICDebateRound(round_number=2, topic="短期风险", resolution="降低仓位"),
            ICDebateRound(round_number=3, topic="风控审查", resolution="风险可接受"),
        ],
        conviction_drivers=["多分支共识偏正", "流动性环境支持"],
        time_horizon_weights={"short_term": 0.3, "medium_term": 0.4, "long_term": 0.3},
    )
    assert len(ic.debate_rounds) == 3
    assert ic.conviction_drivers[0] == "多分支共识偏正"
    assert sum(ic.time_horizon_weights.values()) == 1.0


def test_master_input_has_disagreement_matrix():
    mi = MasterAgentInput(
        market_regime="default",
        candidate_symbols=["000001.SZ"],
        branch_disagreement_matrix={
            "kline": {"quant": 0.4, "fundamental": 0.8},
            "quant": {"kline": 0.4, "fundamental": 0.5},
            "fundamental": {"kline": 0.8, "quant": 0.5},
        },
    )
    assert mi.branch_disagreement_matrix["kline"]["fundamental"] == 0.8


# ---------------------------------------------------------------------------
# 5. Agent Enhanced Strategy with specialized outputs
# ---------------------------------------------------------------------------

def test_agent_enhanced_strategy_accepts_specialized_outputs():
    """AgentEnhancedStrategy can hold specialized outputs via inheritance."""
    kline_out = KLineAgentOutput(
        branch_name="kline",
        conviction="buy",
        conviction_score=0.3,
        confidence=0.7,
        key_insights=["趋势强劲"],
        trend_assessment="strong_uptrend",
    )
    quant_out = QuantAgentOutput(
        branch_name="quant",
        conviction="neutral",
        conviction_score=0.05,
        confidence=0.6,
        key_insights=["因子信号中性"],
        factor_quality_assessment={"momentum": "robust"},
    )

    strategy = AgentEnhancedStrategy(
        agent_layer_success=True,
        branch_agent_outputs={"kline": kline_out, "quant": quant_out},
    )

    assert strategy.branch_agent_outputs["kline"] is not None
    assert strategy.branch_agent_outputs["kline"].conviction == "buy"
    # Can access base fields
    assert strategy.branch_agent_outputs["quant"] is not None
    assert strategy.branch_agent_outputs["quant"].conviction_score == 0.05


# ---------------------------------------------------------------------------
# 6. SubAgent class hierarchy
# ---------------------------------------------------------------------------

def test_specialized_subagents_are_importable():
    from quant_investor.agents.subagents import (
        FundamentalSubAgent,
        IntelligenceSubAgent,
        KLineSubAgent,
        MacroSubAgent,
        QuantSubAgent,
        SpecializedRiskSubAgent,
    )
    from quant_investor.agents.subagent import BaseSubAgent, BranchSubAgent

    assert issubclass(KLineSubAgent, BaseSubAgent)
    assert issubclass(QuantSubAgent, BaseSubAgent)
    assert issubclass(FundamentalSubAgent, BaseSubAgent)
    assert issubclass(IntelligenceSubAgent, BaseSubAgent)
    assert issubclass(MacroSubAgent, BaseSubAgent)
    assert issubclass(BranchSubAgent, BaseSubAgent)
    # SpecializedRiskSubAgent is standalone (not a branch agent)
    assert hasattr(SpecializedRiskSubAgent, "analyze")


def test_orchestrator_agent_registry():
    """Orchestrator uses specialized agents for known branches."""
    from quant_investor.agents.orchestrator import _AGENT_REGISTRY
    from quant_investor.agents.subagents import (
        FundamentalSubAgent,
        IntelligenceSubAgent,
        KLineSubAgent,
        MacroSubAgent,
        QuantSubAgent,
    )

    assert _AGENT_REGISTRY["kline"] is KLineSubAgent
    assert _AGENT_REGISTRY["quant"] is QuantSubAgent
    assert _AGENT_REGISTRY["fundamental"] is FundamentalSubAgent
    assert _AGENT_REGISTRY["intelligence"] is IntelligenceSubAgent
    assert _AGENT_REGISTRY["macro"] is MacroSubAgent
