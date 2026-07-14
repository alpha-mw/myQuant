from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from quant_investor.agent_protocol import (
    BayesianDecisionRecord,
    BranchVerdict,
    GlobalContext,
    ReportBundle,
    StockReviewBundle,
)
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput, MasterAgentOutput, RiskAgentOutput
from quant_investor.branch_contracts import (
    BranchResult,
    CalibratedBranchSignal,
    LLMUsageRecord,
    LLMUsageSummary,
    PortfolioStrategy,
    UnifiedDataBundle,
)
from quant_investor.funnel.deterministic_funnel import FunnelOutput
from quant_investor.versioning import (
    AGENT_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
    reject_retired_intelligence_keys,
)


@dataclass
class QuantInvestorPipelineResult:
    """Single-mainline pipeline result with baseline and review artifacts."""

    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    likelihood_schema_version: str = LIKELIHOOD_SCHEMA_VERSION
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

    def __post_init__(self) -> None:
        expected_versions = {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "ic_protocol_version": IC_PROTOCOL_VERSION,
            "report_protocol_version": REPORT_PROTOCOL_VERSION,
            "debate_template_version": DEBATE_TEMPLATE_VERSION,
            "agent_schema_version": AGENT_SCHEMA_VERSION,
        }
        for field_name, expected in expected_versions.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise ValueError(
                    f"QuantInvestorPipelineResult {field_name} mismatch: "
                    f"expected {expected!r}, got {actual!r}."
                )
        unexpected = sorted(set(self.branch_results) - {"quant", "fundamental", "macro"})
        if unexpected:
            raise ValueError(
                "QuantInvestorPipelineResult has noncanonical branch keys: "
                + ", ".join(unexpected)
            )
        unexpected_calibrated = sorted(
            set(self.calibrated_signals) - {"quant", "fundamental", "macro"}
        )
        if unexpected_calibrated:
            raise ValueError(
                "QuantInvestorPipelineResult has noncanonical calibrated signal keys: "
                + ", ".join(unexpected_calibrated)
            )
        for branch_name, result in self.branch_results.items():
            if not isinstance(result, BranchResult):
                raise ValueError(
                    "QuantInvestorPipelineResult branch results must be BranchResult objects."
                )
            result.validate()
            if result.branch_name != branch_name:
                raise ValueError(
                    "QuantInvestorPipelineResult branch result key/name mismatch."
                )
        for branch_name, signal in self.calibrated_signals.items():
            if not isinstance(signal, CalibratedBranchSignal):
                raise ValueError(
                    "QuantInvestorPipelineResult calibrated signals must be CalibratedBranchSignal objects."
                )
            signal.__post_init__()
            if signal.branch_name != branch_name:
                raise ValueError(
                    "QuantInvestorPipelineResult calibrated signal key/name mismatch."
                )
        self.final_strategy.__post_init__()
        self.baseline_strategy.__post_init__()
        if isinstance(self.agent_report_bundle, ReportBundle):
            self.agent_report_bundle.__post_init__()
        for review_bundle in (self.agent_review_bundle, self.review_bundle):
            if isinstance(review_bundle, StockReviewBundle):
                review_bundle.__post_init__()
        for branch_map in self.reviewed_research_by_symbol.values():
            unexpected_review = sorted(
                set(branch_map) - {"quant", "fundamental", "macro", "kline"}
            )
            if unexpected_review:
                raise ValueError(
                    "QuantInvestorPipelineResult reviewed research has non-v14 branches: "
                    + ", ".join(unexpected_review)
                )
            for verdict in branch_map.values():
                verdict.__post_init__()
        for verdict in self.reviewed_branch_summaries.values():
            verdict.__post_init__()
        for record in self.bayesian_records:
            if not isinstance(record, BayesianDecisionRecord):
                raise ValueError(
                    "QuantInvestorPipelineResult bayesian records must be BayesianDecisionRecord objects."
                )
            record.__post_init__()
        reject_retired_intelligence_keys(
            {
                "agent_orchestration": self.agent_orchestration or {},
                "ic_hints_by_symbol": self.ic_hints_by_symbol,
                "data_snapshot": self.data_snapshot,
                "raw_data": self.raw_data,
                "factor_data": self.factor_data,
                "model_predictions": self.model_predictions,
                "llm_ensemble_results": self.llm_ensemble_results,
                "symbol_review_bundle": self.symbol_review_bundle,
            },
            path="QuantInvestorPipelineResult",
        )


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
    likelihood_schema_version: str = LIKELIHOOD_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    llm_usage_session_id: str = ""

    def __post_init__(self) -> None:
        for field_name, actual, expected in (
            ("branch_schema_version", self.branch_schema_version, BRANCH_SCHEMA_VERSION),
            ("likelihood_schema_version", self.likelihood_schema_version, LIKELIHOOD_SCHEMA_VERSION),
            ("calibration_schema_version", self.calibration_schema_version, CALIBRATION_SCHEMA_VERSION),
        ):
            if actual != expected:
                raise ValueError(
                    f"ResearchCoreSnapshot {field_name} mismatch: "
                    f"expected {expected!r}, got {actual!r}."
                )
        unexpected = sorted(set(self.branch_results) - {"quant", "fundamental", "macro"})
        if unexpected:
            raise ValueError(
                "ResearchCoreSnapshot has noncanonical branch keys: "
                + ", ".join(unexpected)
            )
        unexpected_calibrated = sorted(
            set(self.calibrated_signals) - {"quant", "fundamental", "macro"}
        )
        if unexpected_calibrated:
            raise ValueError(
                "ResearchCoreSnapshot has noncanonical calibrated signal keys: "
                + ", ".join(unexpected_calibrated)
            )
        for branch_name, result in self.branch_results.items():
            if not isinstance(result, BranchResult):
                raise ValueError("ResearchCoreSnapshot branch results must be BranchResult objects.")
            result.validate()
            if result.branch_name != branch_name:
                raise ValueError("ResearchCoreSnapshot branch result key/name mismatch.")
        for branch_name, signal in self.calibrated_signals.items():
            if not isinstance(signal, CalibratedBranchSignal):
                raise ValueError(
                    "ResearchCoreSnapshot calibrated signals must be CalibratedBranchSignal objects."
                )
            signal.__post_init__()
            if signal.branch_name != branch_name:
                raise ValueError("ResearchCoreSnapshot calibrated signal key/name mismatch.")
        self.baseline_strategy.__post_init__()
        reject_retired_intelligence_keys(
            {"data_bundle_metadata": self.data_bundle.metadata, "timings": self.timings},
            path="ResearchCoreSnapshot",
        )
