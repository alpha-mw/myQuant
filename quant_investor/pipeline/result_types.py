from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from quant_investor.agent_protocol import BranchVerdict, GlobalContext
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput, MasterAgentOutput, RiskAgentOutput
from quant_investor.branch_contracts import LLMUsageRecord, LLMUsageSummary, PortfolioStrategy, UnifiedDataBundle
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.versioning import (
    AGENT_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


@dataclass
class QuantInvestorPipelineResult:
    """Single-mainline pipeline result with baseline and review artifacts."""

    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    data_bundle: Optional[UnifiedDataBundle] = None
    branch_results: dict[str, Any] = field(default_factory=dict)
    calibrated_signals: dict[str, Any] = field(default_factory=dict)
    risk_results: Any = None
    final_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    final_report: str = ""
    execution_log: list[str] = field(default_factory=list)
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    agent_orchestration: Optional[dict[str, Any]] = None
    agent_portfolio_plan: Any = None
    agent_report_bundle: Any = None
    agent_ic_decisions: Any = field(default_factory=dict)
    agent_review_bundle: Any = None
    ic_hints_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_role_metadata: Any = None
    execution_trace: Any = None
    what_if_plan: Any = None
    llm_usage_records: list[LLMUsageRecord] = field(default_factory=list)
    llm_usage_summary: LLMUsageSummary = field(default_factory=LLMUsageSummary)
    llm_effective_records: list[LLMUsageRecord] = field(default_factory=list)
    llm_effective_summary: LLMUsageSummary = field(default_factory=LLMUsageSummary)
    llm_usage_session_id: str = ""
    data_snapshot: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)
    factor_data: dict[str, Any] = field(default_factory=dict)
    model_predictions: dict[str, Any] = field(default_factory=dict)
    macro_signal: str = "🟡"
    macro_summary: str = ""
    llm_ensemble_results: dict[str, Any] = field(default_factory=dict)
    baseline_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    baseline_risk_result: Any = None
    macro_verdict: BranchVerdict | None = None
    reviewed_research_by_symbol: dict[str, dict[str, BranchVerdict]] = field(default_factory=dict)
    reviewed_branch_summaries: dict[str, BranchVerdict] = field(default_factory=dict)
    branch_review_outputs: dict[str, BaseBranchAgentOutput | None] = field(default_factory=dict)
    master_review_output: MasterAgentOutput | None = None
    risk_review_output: RiskAgentOutput | None = None
    review_bundle: Any = None
    symbol_review_bundle: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_layer_enabled: bool = False
    agent_schema_version: str = AGENT_SCHEMA_VERSION
    pipeline_mode: str = "legacy"
    global_context: Optional[GlobalContext] = None
    funnel_output: Optional[FunnelOutput] = None
    bayesian_records: list[Any] = field(default_factory=list)
    shortlist_evidence: list[Any] = field(default_factory=list)
    bayesian_shortlist_symbols: list[str] = field(default_factory=list)


@dataclass
class ResearchCoreSnapshot:
    data_bundle: UnifiedDataBundle
    branch_results: dict[str, Any] = field(default_factory=dict)
    calibrated_signals: dict[str, Any] = field(default_factory=dict)
    risk_result: Any = None
    baseline_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    market_regime: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    execution_log: list[str] = field(default_factory=list)
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    llm_usage_session_id: str = ""
