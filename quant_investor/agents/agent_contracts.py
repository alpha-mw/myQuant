"""
Review-layer agent contracts.

These models stay advisory-only for the review layer. They must not be used to
let free text bypass the deterministic control chain.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _CompatModel(BaseModel):
    """Allow additive compatibility fields across partially migrated modules."""

    model_config = ConfigDict(extra="allow")


class BaseBranchAgentInput(_CompatModel):
    """Common input shared by all branch review agents."""

    branch_name: str
    base_score: float = Field(description="算法基础分 (-1.0 ~ 1.0)")
    final_score: float = Field(description="debate 调整后分数")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = ""
    bull_points: list[str] = Field(default_factory=list)
    bear_points: list[str] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)
    used_features: list[str] = Field(default_factory=list)
    symbol_scores: dict[str, float] = Field(default_factory=dict)
    market_regime: str = "default"
    calibrated_expected_return: float = 0.0
    branch_signals: dict[str, Any] = Field(default_factory=dict, description="分支关键信号摘要")
    recall_context: dict[str, Any] = Field(
        default_factory=dict,
        description="仅供 review layer 参考的历史回顾摘要，不可直接驱动最终仓位。",
    )


class BaseBranchAgentOutput(_CompatModel):
    """Common output shared by all branch review agents."""

    branch_name: str
    conviction: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"
    conviction_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_insights: list[str] = Field(default_factory=list, description="3-5 条关键洞察")
    risk_flags: list[str] = Field(default_factory=list)
    disagreements_with_algo: list[str] = Field(
        default_factory=list,
        description="agent 与量化模型结论的分歧点",
    )
    symbol_views: dict[str, str] = Field(default_factory=dict, description="个股简评")
    reasoning: str = ""


BranchAgentInput = BaseBranchAgentInput
BranchAgentOutput = BaseBranchAgentOutput


class KLineAgentInput(BaseBranchAgentInput):
    branch_mode: str = ""
    runtime_backend: str = ""
    focus_symbols: list[str] = Field(default_factory=list)
    predicted_returns: dict[str, float] = Field(default_factory=dict)
    kronos_confidence: float = 0.0
    chronos_confidence: float = 0.0
    model_agreement: float = 0.0
    detected_regimes: dict[str, str] = Field(default_factory=dict)
    trend_strength: dict[str, float] = Field(default_factory=dict)
    momentum_signals: dict[str, float] = Field(default_factory=dict)
    volatility_percentile: float = 0.0
    support_resistance: dict[str, dict[str, float]] = Field(default_factory=dict)


class KLineAgentOutput(BaseBranchAgentOutput):
    trend_assessment: str = "neutral"
    model_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    pattern_signals: list[str] = Field(default_factory=list)
    timeframe_alignment: str = "mixed"
    reversal_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_return_assessment: dict[str, str] = Field(default_factory=dict)


class FundamentalAgentInput(BaseBranchAgentInput):
    module_scores: dict[str, float] = Field(default_factory=dict)
    module_confidences: dict[str, float] = Field(default_factory=dict)
    module_coverages: dict[str, Any] = Field(default_factory=dict)
    governance_scores: dict[str, float] = Field(default_factory=dict)
    doc_sentiment: dict[str, float] = Field(default_factory=dict)
    data_staleness_days: dict[str, float] = Field(default_factory=dict)
    financial_quality: dict[str, dict[str, Any]] = Field(default_factory=dict)
    forecast_revisions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    valuation_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ownership_signals: dict[str, dict[str, Any]] = Field(default_factory=dict)


class FundamentalAgentOutput(BaseBranchAgentOutput):
    earnings_quality_assessment: str = "neutral"
    accounting_red_flags: list[str] = Field(default_factory=list)
    valuation_stance: str = "fair"
    management_signal: str = "neutral"
    data_quality_concerns: list[str] = Field(default_factory=list)
    module_override_reasons: dict[str, str] = Field(default_factory=dict)
    time_horizon_note: str = ""


class QuantAgentInput(BaseBranchAgentInput):
    factor_quality_summary: dict[str, Any] = Field(default_factory=dict)


class QuantAgentOutput(BaseBranchAgentOutput):
    factor_quality_assessment: dict[str, str] = Field(default_factory=dict)
    regime_suitability: float = Field(default=0.5, ge=0.0, le=1.0)
    overfitting_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    factor_conflicts: list[str] = Field(default_factory=list)
    recommended_factor_tilts: dict[str, str] = Field(default_factory=dict)


class IntelligenceAgentInput(BaseBranchAgentInput):
    catalyst_summary: dict[str, Any] = Field(default_factory=dict)


class IntelligenceAgentOutput(BaseBranchAgentOutput):
    sentiment_regime: str = "neutral"
    contrarian_signal: str = "none"
    catalyst_assessment: list[str] = Field(default_factory=list)
    information_asymmetry: str = "none"
    event_timing_risk: float = Field(default=0.0, ge=0.0, le=1.0)


class MacroAgentInput(BaseBranchAgentInput):
    macro_summary: dict[str, Any] = Field(default_factory=dict)


class MacroAgentOutput(BaseBranchAgentOutput):
    macro_regime_assessment: str = "neutral"
    liquidity_outlook: str = "neutral"
    systemic_risk_level: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_asset_implications: list[str] = Field(default_factory=list)
    uniform_score_appropriateness: str = "appropriate"
    regime_transition_risk: float = Field(default=0.0, ge=0.0, le=1.0)


class RiskAgentInput(_CompatModel):
    """Risk review input, still advisory-only."""

    risk_metrics_summary: dict[str, Any] = Field(default_factory=dict)
    regime: str = "default"
    position_sizing: dict[str, float] = Field(default_factory=dict)
    branch_agent_summaries: dict[str, BaseBranchAgentOutput] = Field(default_factory=dict)
    portfolio_level_risks: list[str] = Field(default_factory=list)


class RiskAgentOutput(_CompatModel):
    """Risk review output, never overrides deterministic hard-veto semantics."""

    risk_assessment: Literal["acceptable", "elevated", "high", "extreme"] = "elevated"
    max_recommended_exposure: float = Field(default=0.6, ge=0.0, le=1.0)
    position_adjustments: dict[str, float] = Field(default_factory=dict)
    risk_warnings: list[str] = Field(default_factory=list)
    hedging_suggestions: list[str] = Field(default_factory=list)
    tail_risk_assessment: str = "normal"
    correlation_breakdown_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    position_sizing_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    drawdown_scenario: str = ""
    reasoning: str = ""


class SymbolRecommendation(_CompatModel):
    """Review-layer recommendation for a single symbol."""

    symbol: str
    action: Literal["buy", "hold", "sell"] = "hold"
    conviction: str = "neutral"
    rationale: str = ""
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class TradeDecision(_CompatModel):
    """单个标的的交易决策。"""

    symbol: str
    action: Literal["buy", "hold", "sell"] = "hold"
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", description="该标的交易决策的具体依据")


class MasterAgentInput(_CompatModel):
    """IC master review input — 直接接收原始分支量化数据，不经过 SubAgent 处理。"""

    branch_results: dict[str, Any] = Field(
        default_factory=dict,
        description="5个分支的序列化 BranchResult（量化分数、signals、evidence）",
    )
    risk_result: dict[str, Any] = Field(
        default_factory=dict,
        description="风控层序列化输出（VaR、仓位建议、stop loss 等）",
    )
    ensemble_baseline: dict[str, Any] = Field(
        default_factory=dict,
        description="算法 EnsembleJudge 输出，作为参考基准",
    )
    market_regime: str = "default"
    candidate_symbols: list[str] = Field(default_factory=list)
    recall_context: dict[str, Any] = Field(
        default_factory=dict,
        description="过往交易记录、盈亏、历史投资逻辑和反思，仅供参考不可直接驱动仓位。",
    )


class MasterAgentOutput(_CompatModel):
    """IC master review output — 包含多轮多空辩论记录及完整投资决策。"""

    final_conviction: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"
    final_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    bull_case: str = Field(default="", description="多方核心论点汇总")
    bear_case: str = Field(default="", description="空方核心论点汇总")
    debate_rounds: list[str] = Field(default_factory=list, description="五轮多空辩论的逐轮摘要")
    consensus_areas: list[str] = Field(default_factory=list, description="多空双方共识点")
    disagreement_areas: list[str] = Field(default_factory=list, description="多空双方核心分歧")
    debate_resolution: list[str] = Field(default_factory=list, description="分歧如何最终裁决")
    conviction_drivers: list[str] = Field(default_factory=list, description="最终 conviction 的关键驱动")
    trade_decisions: list[TradeDecision] = Field(
        default_factory=list,
        description="每个候选标的的具体交易决策（action + target_weight + rationale）",
    )
    top_picks: list[SymbolRecommendation] = Field(default_factory=list)
    investment_thesis: str = Field(
        default="",
        description="本次决策的完整投资逻辑（含因果链和风险提示），供存档和学习回顾",
    )
    portfolio_narrative: str = Field(default="", description="3-5 句执行摘要")
    risk_adjusted_exposure: float = Field(default=0.5, ge=0.0, le=1.0)
    dissenting_views: list[str] = Field(default_factory=list, description="保留的少数派意见")


class AgentEnhancedStrategy(_CompatModel):
    """Agent-enhanced review bundle."""

    algorithmic_strategy: dict[str, Any] = Field(default_factory=dict)
    agent_strategy: MasterAgentOutput | None = None
    agent_layer_success: bool = False
    agent_layer_timings: dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = True
    branch_agent_outputs: dict[str, BaseBranchAgentOutput | None] = Field(default_factory=dict)
    risk_agent_output: RiskAgentOutput | None = None


__all__ = [
    "AgentEnhancedStrategy",
    "BaseBranchAgentInput",
    "BaseBranchAgentOutput",
    "BranchAgentInput",
    "BranchAgentOutput",
    "FundamentalAgentInput",
    "FundamentalAgentOutput",
    "IntelligenceAgentInput",
    "IntelligenceAgentOutput",
    "KLineAgentInput",
    "KLineAgentOutput",
    "MacroAgentInput",
    "MacroAgentOutput",
    "MasterAgentInput",
    "MasterAgentOutput",
    "QuantAgentInput",
    "QuantAgentOutput",
    "RiskAgentInput",
    "RiskAgentOutput",
    "SymbolRecommendation",
    "TradeDecision",
]
