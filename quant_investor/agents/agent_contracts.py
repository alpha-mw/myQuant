"""
V10.1 Multi-Agent 层的输入/输出 contracts。

Base contracts + 每个分支的专属合约，保证序列化安全和 schema 校验。
专属合约继承 Base，确保向后兼容。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base Branch SubAgent contracts (backward compatible)
# ---------------------------------------------------------------------------

class BaseBranchAgentInput(BaseModel):
    """所有分支 agent 的公共输入字段。"""

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


class BaseBranchAgentOutput(BaseModel):
    """所有分支 agent 的公共输出字段。"""

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


# Backward-compatible aliases
BranchAgentInput = BaseBranchAgentInput
BranchAgentOutput = BaseBranchAgentOutput


# ---------------------------------------------------------------------------
# KLine Branch specialized contracts
# ---------------------------------------------------------------------------

class KLineAgentInput(BaseBranchAgentInput):
    """K线技术分析分支专属输入。"""

    predicted_returns: dict[str, float] = Field(default_factory=dict, description="各标的预测收益率")
    kronos_confidence: float = Field(default=0.0, description="Kronos 模型置信度")
    chronos_confidence: float = Field(default=0.0, description="Chronos 模型置信度")
    model_agreement: float = Field(default=0.0, description="双模型一致性 (-1~1)")
    detected_regimes: dict[str, str] = Field(default_factory=dict, description="各标的市场状态")
    trend_strength: dict[str, float] = Field(default_factory=dict, description="趋势强度 (ADX)")
    momentum_signals: dict[str, float] = Field(default_factory=dict, description="动量信号 (RSI/MACD)")
    support_resistance: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的支撑/阻力位",
    )
    volatility_percentile: float = Field(default=50.0, description="当前波动率百分位")


class KLineAgentOutput(BaseBranchAgentOutput):
    """K线技术分析分支专属输出。"""

    trend_assessment: str = Field(default="neutral", description="趋势评估: strong_uptrend/weak_uptrend/neutral/weak_downtrend/strong_downtrend")
    model_reliability: float = Field(default=0.5, ge=0.0, le=1.0, description="agent 评估的模型可靠度")
    pattern_signals: list[str] = Field(default_factory=list, description="识别的技术形态")
    timeframe_alignment: str = Field(default="mixed", description="多时间框架一致性: aligned_bullish/aligned_bearish/divergent/mixed")
    reversal_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="趋势反转概率")
    predicted_return_assessment: dict[str, str] = Field(
        default_factory=dict, description="对各标的预测收益的判断: agrees/too_optimistic/too_pessimistic",
    )


# ---------------------------------------------------------------------------
# Quant Branch specialized contracts
# ---------------------------------------------------------------------------

class QuantAgentInput(BaseBranchAgentInput):
    """量化因子分支专属输入。"""

    factor_exposures: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的因子暴露 {symbol: {factor: value}}",
    )
    ic_metrics: dict[str, float] = Field(default_factory=dict, description="各因子 IC 值")
    ir_metrics: dict[str, float] = Field(default_factory=dict, description="各因子 IR 值")
    factor_decay_info: dict[str, int] = Field(default_factory=dict, description="因子半衰期 (天)")
    crowding_signals: dict[str, float] = Field(default_factory=dict, description="因子拥挤度")
    alpha_candidates: list[dict[str, Any]] = Field(default_factory=list, description="遗传算法 Alpha 候选")
    regime_factor_effectiveness: dict[str, float] = Field(
        default_factory=dict, description="当前 regime 下各因子有效性",
    )


class QuantAgentOutput(BaseBranchAgentOutput):
    """量化因子分支专属输出。"""

    factor_quality_assessment: dict[str, str] = Field(
        default_factory=dict, description="因子质量评估 {factor: robust/decaying/crowded}",
    )
    regime_suitability: float = Field(default=0.5, ge=0.0, le=1.0, description="因子对当前 regime 的适配度")
    overfitting_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="过拟合风险")
    factor_conflicts: list[str] = Field(default_factory=list, description="因子间矛盾信号")
    recommended_factor_tilts: dict[str, str] = Field(
        default_factory=dict, description="建议因子偏移 {factor: overweight/neutral/underweight}",
    )


# ---------------------------------------------------------------------------
# Fundamental Branch specialized contracts
# ---------------------------------------------------------------------------

class FundamentalAgentInput(BaseBranchAgentInput):
    """基本面分支专属输入。"""

    module_scores: dict[str, float] = Field(default_factory=dict, description="6 子模块评分")
    module_confidences: dict[str, float] = Field(default_factory=dict, description="子模块置信度")
    module_coverages: dict[str, str] = Field(default_factory=dict, description="子模块覆盖状态")
    financial_quality: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的财务质量 {symbol: {roe, margin, ...}}",
    )
    forecast_revisions: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的盈利预测修正 {symbol: {eps_growth, revision}}",
    )
    valuation_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的估值指标 {symbol: {pe, pb, ps}}",
    )
    governance_scores: dict[str, float] = Field(default_factory=dict, description="各标的治理评分")
    ownership_signals: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="各标的股权信号 {symbol: {concentration, institutional}}",
    )
    doc_sentiment: dict[str, float] = Field(default_factory=dict, description="各标的文档语义情绪")
    data_staleness_days: dict[str, int] = Field(default_factory=dict, description="数据时效性 (天)")


class FundamentalAgentOutput(BaseBranchAgentOutput):
    """基本面分支专属输出。"""

    earnings_quality_assessment: str = Field(default="neutral", description="盈利质量: high/neutral/low/red_flag")
    accounting_red_flags: list[str] = Field(default_factory=list, description="会计红旗信号")
    valuation_stance: str = Field(default="fair", description="估值立场: cheap/fair/expensive/bubble")
    management_signal: str = Field(default="neutral", description="管理层信号: positive/neutral/negative")
    data_quality_concerns: list[str] = Field(default_factory=list, description="数据质量问题")
    module_override_reasons: dict[str, str] = Field(
        default_factory=dict, description="对子模块评分的修正理由 {module: reason}",
    )
    time_horizon_note: str = Field(default="", description="时间视角说明")


# ---------------------------------------------------------------------------
# Intelligence Branch specialized contracts
# ---------------------------------------------------------------------------

class IntelligenceAgentInput(BaseBranchAgentInput):
    """智能融合分支专属输入。"""

    event_risk_score: float = Field(default=0.0, description="事件风险评分 (-1~1)")
    event_catalysts: list[str] = Field(default_factory=list, description="潜在催化事件")
    fear_greed_index: float = Field(default=50.0, ge=0.0, le=100.0, description="恐惧贪婪指数")
    sentiment_extremes: dict[str, float] = Field(default_factory=dict, description="情绪极端值指标")
    money_flow_signal: float = Field(default=0.0, description="资金流向信号 (-1~1)")
    smart_money_indicators: dict[str, float] = Field(default_factory=dict, description="聪明钱指标")
    market_breadth: dict[str, float] = Field(default_factory=dict, description="市场广度指标")
    sector_rotation_signal: str = Field(default="neutral", description="板块轮动信号")


class IntelligenceAgentOutput(BaseBranchAgentOutput):
    """智能融合分支专属输出。"""

    sentiment_regime: str = Field(
        default="neutral",
        description="情绪状态: extreme_fear/fear/neutral/greed/extreme_greed",
    )
    contrarian_signal: str = Field(
        default="none",
        description="逆向信号: strong_buy_contrarian/buy_contrarian/none/sell_contrarian/strong_sell_contrarian",
    )
    catalyst_assessment: list[str] = Field(default_factory=list, description="催化事件评估")
    information_asymmetry: str = Field(
        default="none",
        description="信息不对称: smart_money_accumulating/distributing/none",
    )
    event_timing_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="事件时间窗口风险")


# ---------------------------------------------------------------------------
# Macro Branch specialized contracts
# ---------------------------------------------------------------------------

class MacroAgentInput(BaseBranchAgentInput):
    """宏观分支专属输入。"""

    liquidity_score: float = Field(default=0.0, description="流动性评分 (-1~1)")
    monetary_policy_signal: str = Field(default="neutral", description="货币政策: easing/neutral/tightening")
    macro_volatility_percentile: float = Field(default=50.0, ge=0.0, le=100.0, description="波动率百分位")
    volatility_term_structure: str = Field(default="normal", description="波动率期限结构: contango/backwardation/flat")
    breadth_score: float = Field(default=0.0, description="市场广度 (-1~1)")
    momentum_structure: dict[str, float] = Field(default_factory=dict, description="多周期动量 {5d, 20d, 60d}")
    cross_asset_signals: dict[str, str] = Field(default_factory=dict, description="跨资产信号 {asset: signal}")
    yield_curve_signal: str = Field(default="normal", description="收益率曲线: steepening/flattening/inverted/normal")
    overall_risk_level: str = Field(default="normal", description="整体风险水平")


class MacroAgentOutput(BaseBranchAgentOutput):
    """宏观分支专属输出。"""

    macro_regime_assessment: str = Field(default="neutral", description="宏观 regime 判断")
    liquidity_outlook: str = Field(default="neutral", description="流动性前景: supportive/neutral/restrictive")
    systemic_risk_level: float = Field(default=0.0, ge=0.0, le=1.0, description="系统性风险水平")
    cross_asset_implications: list[str] = Field(default_factory=list, description="跨资产影响分析")
    uniform_score_appropriateness: str = Field(
        default="appropriate",
        description="统一打分合理性: appropriate/questionable/inappropriate",
    )
    regime_transition_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="regime 转换概率")


# ---------------------------------------------------------------------------
# Risk SubAgent contracts (enhanced)
# ---------------------------------------------------------------------------

class RiskAgentInput(BaseModel):
    """风控 SubAgent 的输入（增强版）。"""

    risk_metrics_summary: dict[str, Any] = Field(default_factory=dict)
    regime: str = "default"
    position_sizing: dict[str, float] = Field(default_factory=dict)
    branch_agent_summaries: dict[str, BaseBranchAgentOutput] = Field(default_factory=dict)
    portfolio_level_risks: list[str] = Field(default_factory=list)
    stop_loss_levels: dict[str, float] = Field(default_factory=dict, description="各标的止损价位")
    stress_test_results: dict[str, float] = Field(default_factory=dict, description="压力测试结果")
    branch_disagreement_level: float = Field(default=0.0, ge=0.0, le=1.0, description="分支间分歧度")
    correlation_matrix_digest: dict[str, float] = Field(
        default_factory=dict, description="关键相关性摘要",
    )


class RiskAgentOutput(BaseModel):
    """风控 SubAgent 的输出（增强版）。"""

    risk_assessment: Literal["acceptable", "elevated", "high", "extreme"] = "elevated"
    max_recommended_exposure: float = Field(default=0.6, ge=0.0, le=1.0)
    position_adjustments: dict[str, float] = Field(default_factory=dict)
    risk_warnings: list[str] = Field(default_factory=list)
    hedging_suggestions: list[str] = Field(default_factory=list)
    tail_risk_assessment: str = Field(default="normal", description="尾部风险: normal/elevated/critical")
    correlation_breakdown_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性崩溃风险")
    position_sizing_overrides: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="仓位覆盖 {symbol: {max_weight, reason_score}}",
    )
    drawdown_scenario: str = Field(default="", description="预期回撤情景")
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Master Agent (IC) contracts (enhanced)
# ---------------------------------------------------------------------------

class SymbolRecommendation(BaseModel):
    """单个标的推荐。"""

    symbol: str
    action: Literal["buy", "hold", "sell"] = "hold"
    conviction: str = "neutral"
    rationale: str = ""
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class ICDebateRound(BaseModel):
    """IC 辩论单轮记录。"""

    round_number: int = 1
    topic: str = ""
    arguments: list[str] = Field(default_factory=list, description="各方论点")
    resolution: str = Field(default="", description="裁决结论")


class MasterAgentInput(BaseModel):
    """IC Master Agent 的输入：所有 SubAgent 研报汇总（增强版）。"""

    branch_reports: dict[str, BaseBranchAgentOutput] = Field(default_factory=dict)
    risk_report: RiskAgentOutput | None = None
    ensemble_baseline: dict[str, Any] = Field(
        default_factory=dict,
        description="算法 EnsembleJudge 输出，作为参考基准",
    )
    market_regime: str = "default"
    candidate_symbols: list[str] = Field(default_factory=list)
    branch_disagreement_matrix: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="分支间 pairwise 分歧度",
    )


class MasterAgentOutput(BaseModel):
    """IC Master Agent 的输出：最终投资建议（增强版）。"""

    final_conviction: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"
    final_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    consensus_areas: list[str] = Field(default_factory=list, description="IC 共识点")
    disagreement_areas: list[str] = Field(default_factory=list, description="IC 分歧点")
    debate_rounds: list[ICDebateRound] = Field(default_factory=list, description="多轮辩论记录")
    debate_resolution: list[str] = Field(default_factory=list, description="分歧如何调解")
    top_picks: list[SymbolRecommendation] = Field(default_factory=list)
    portfolio_narrative: str = Field(default="", description="3-5 句投资论点")
    risk_adjusted_exposure: float = Field(default=0.5, ge=0.0, le=1.0)
    dissenting_views: list[str] = Field(default_factory=list, description="保留的少数派意见")
    conviction_drivers: list[str] = Field(default_factory=list, description="最终决策驱动因素")
    time_horizon_weights: dict[str, float] = Field(
        default_factory=dict, description="时间视角权重 {short_term, medium_term, long_term}",
    )


# ---------------------------------------------------------------------------
# Orchestrator output
# ---------------------------------------------------------------------------

class AgentEnhancedStrategy(BaseModel):
    """Agent 增强后的综合策略，包裹算法策略 + Agent 层输出。"""

    algorithmic_strategy: dict[str, Any] = Field(default_factory=dict)
    agent_strategy: MasterAgentOutput | None = None
    agent_layer_success: bool = False
    agent_layer_timings: dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = True
    branch_agent_outputs: dict[str, BaseBranchAgentOutput | None] = Field(default_factory=dict)
    risk_agent_output: RiskAgentOutput | None = None
