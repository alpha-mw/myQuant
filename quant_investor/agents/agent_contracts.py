"""
V10 Multi-Agent 层的输入/输出 contracts。

所有 agent 间通信使用 Pydantic v2 模型，保证序列化安全和 schema 校验。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Branch SubAgent contracts
# ---------------------------------------------------------------------------

class BranchAgentInput(BaseModel):
    """分支 SubAgent 的输入：量化分支结果 + 上下文。"""

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


class BranchAgentOutput(BaseModel):
    """分支 SubAgent 的输出：定性研判 + 关键洞察。"""

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


# ---------------------------------------------------------------------------
# Risk SubAgent contracts
# ---------------------------------------------------------------------------

class RiskAgentInput(BaseModel):
    """风控 SubAgent 的输入。"""

    risk_metrics_summary: dict[str, Any] = Field(default_factory=dict)
    regime: str = "default"
    position_sizing: dict[str, float] = Field(default_factory=dict)
    branch_agent_summaries: dict[str, BranchAgentOutput] = Field(default_factory=dict)
    portfolio_level_risks: list[str] = Field(default_factory=list)


class RiskAgentOutput(BaseModel):
    """风控 SubAgent 的输出。"""

    risk_assessment: Literal["acceptable", "elevated", "high", "extreme"] = "elevated"
    max_recommended_exposure: float = Field(default=0.6, ge=0.0, le=1.0)
    position_adjustments: dict[str, float] = Field(default_factory=dict)
    risk_warnings: list[str] = Field(default_factory=list)
    hedging_suggestions: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Master Agent (IC) contracts
# ---------------------------------------------------------------------------

class SymbolRecommendation(BaseModel):
    """单个标的推荐。"""

    symbol: str
    action: Literal["buy", "hold", "sell"] = "hold"
    conviction: str = "neutral"
    rationale: str = ""
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class MasterAgentInput(BaseModel):
    """IC Master Agent 的输入：所有 SubAgent 研报汇总。"""

    branch_reports: dict[str, BranchAgentOutput] = Field(default_factory=dict)
    risk_report: RiskAgentOutput | None = None
    ensemble_baseline: dict[str, Any] = Field(
        default_factory=dict,
        description="算法 EnsembleJudge 输出，作为参考基准",
    )
    market_regime: str = "default"
    candidate_symbols: list[str] = Field(default_factory=list)


class MasterAgentOutput(BaseModel):
    """IC Master Agent 的输出：最终投资建议。"""

    final_conviction: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"
    final_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    consensus_areas: list[str] = Field(default_factory=list, description="IC 共识点")
    disagreement_areas: list[str] = Field(default_factory=list, description="IC 分歧点")
    debate_resolution: list[str] = Field(default_factory=list, description="分歧如何调解")
    top_picks: list[SymbolRecommendation] = Field(default_factory=list)
    portfolio_narrative: str = Field(default="", description="3-5 句投资论点")
    risk_adjusted_exposure: float = Field(default=0.5, ge=0.0, le=1.0)
    dissenting_views: list[str] = Field(default_factory=list, description="保留的少数派意见")


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
    branch_agent_outputs: dict[str, BranchAgentOutput | None] = Field(default_factory=dict)
    risk_agent_output: RiskAgentOutput | None = None
