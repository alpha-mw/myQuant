"""
当前公开 agent protocol。

主线不再依赖历史 sourceless shadow wrapper。
本文件直接声明当前稳定协议类型，供 Agent bridge、报告层和测试复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from quant_investor.artifact_validation import (
    require_finite_structure,
    validate_posterior_numeric_fields,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
    reject_retired_intelligence_keys,
)


class AgentStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    VETOED = "vetoed"


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ActionLabel(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    WATCH = "watch"
    AVOID = "avoid"


class ConfidenceLabel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class CoverageScope(str, Enum):
    BRANCH = "branch"
    SYMBOL = "symbol"
    MARKET = "market"
    PORTFOLIO = "portfolio"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


_ALLOWED_REVIEW_BRANCHES = frozenset((*CANONICAL_BRANCH_ORDER, "kline"))


def _require_current_version(actual: str, expected: str, field_name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{field_name} mismatch: expected {expected!r}, got {actual!r}."
        )


def _require_allowed_branch_keys(value: Mapping[str, Any], field_name: str) -> None:
    unexpected = sorted(set(value) - _ALLOWED_REVIEW_BRANCHES)
    if unexpected:
        raise ValueError(
            f"{field_name} contains non-v15 branch keys: {', '.join(unexpected)}."
        )


def _default_likelihood_payload() -> dict[str, Any]:
    return {
        "schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "quant_likelihood": 0.5,
        "fundamental_likelihood": 0.5,
        "correlation_matrix": {},
    }


@dataclass
class EventNote:
    title: str = ""
    message: str = ""
    scope: CoverageScope = CoverageScope.BRANCH
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityIssue:
    path: str = ""
    symbol: str = ""
    category: str = ""
    universe_key: str = ""
    issue_type: str = ""
    severity: str = "warning"
    message: str = ""
    resolver_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityDiagnostics:
    total_universe_count: int = 0
    researchable_universe_count: int = 0
    shortlistable_universe_count: int = 0
    final_selected_universe_count: int = 0
    quarantined_symbols: list[str] = field(default_factory=list)
    issue_count: int = 0
    blocking_issue_count: int = 0
    coverage_tiers: dict[str, Any] = field(default_factory=dict)
    issues: list[DataQualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    source: str = ""
    summary: str = ""
    direction: Direction = Direction.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    scope: CoverageScope = CoverageScope.BRANCH
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchVerdict:
    agent_name: str = ""
    thesis: str = ""
    symbol: Optional[str] = None
    status: AgentStatus = AgentStatus.SUCCESS
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)
    investment_risks: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    diagnostic_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_current_version(
            self.architecture_version,
            ARCHITECTURE_VERSION,
            "BranchVerdict architecture_version",
        )
        _require_current_version(
            self.branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "BranchVerdict branch_schema_version",
        )
        reject_retired_intelligence_keys(
            self.metadata,
            path="BranchVerdict.metadata",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="BranchVerdict")
        return payload


@dataclass
class RiskDecision:
    status: AgentStatus = AgentStatus.SUCCESS
    risk_level: RiskLevel = RiskLevel.LOW
    hard_veto: bool = False
    veto: bool = False
    action_cap: ActionLabel = ActionLabel.BUY
    max_weight: float = 1.0
    gross_exposure_cap: float = 1.0
    target_exposure_cap: float = 1.0
    blocked_symbols: list[str] = field(default_factory=list)
    position_limits: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    events: list[EventNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "RiskDecision architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "RiskDecision branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "RiskDecision ic_protocol_version")
        reject_retired_intelligence_keys(self.metadata, path="RiskDecision.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class ICDecision:
    status: AgentStatus = AgentStatus.SUCCESS
    thesis: str = ""
    symbol: Optional[str] = None
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    rationale_points: list[str] = field(default_factory=list)
    selected_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "ICDecision architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "ICDecision branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "ICDecision ic_protocol_version")
        reject_retired_intelligence_keys(self.metadata, path="ICDecision.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class PortfolioPlan:
    status: AgentStatus = AgentStatus.SUCCESS
    target_exposure: float = 0.0
    target_gross_exposure: float = 0.0
    target_net_exposure: float = 0.0
    cash_ratio: float = 1.0
    target_weights: dict[str, float] = field(default_factory=dict)
    target_positions: dict[str, float] = field(default_factory=dict)
    position_limits: dict[str, float] = field(default_factory=dict)
    blocked_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    concentration_metrics: dict[str, Any] = field(default_factory=dict)
    turnover_estimate: float = 0.0
    execution_notes: list[str] = field(default_factory=list)
    construction_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "PortfolioPlan architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "PortfolioPlan branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "PortfolioPlan ic_protocol_version")
        reject_retired_intelligence_keys(self.metadata, path="PortfolioPlan.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class ReportBundle:
    agent_name: str = "NarratorAgent"
    headline: str = ""
    summary: str = ""
    global_context: Optional["GlobalContext"] = None
    symbol_research_packets: dict[str, "SymbolResearchPacket"] = field(default_factory=dict)
    shortlist: list["ShortlistItem"] = field(default_factory=list)
    portfolio_decision: Optional["PortfolioDecision"] = None
    macro_verdict: Optional[BranchVerdict] = None
    branch_verdicts: dict[str, BranchVerdict] = field(default_factory=dict)
    risk_decision: Optional[RiskDecision] = None
    ic_decision: Optional[ICDecision] = None
    ic_decisions: list[ICDecision] = field(default_factory=list)
    review_bundle: Optional["StockReviewBundle"] = None
    ic_hints_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_role_metadata: Optional["ModelRoleMetadata"] = None
    execution_trace: Optional["ExecutionTrace"] = None
    what_if_plan: Optional["WhatIfPlan"] = None
    portfolio_plan: Optional[PortfolioPlan] = None
    markdown_report: str = ""
    executive_summary: list[str] = field(default_factory=list)
    market_view: list[str] = field(default_factory=list)
    branch_conclusions: list[str] = field(default_factory=list)
    stock_cards: list[str] = field(default_factory=list)
    coverage_summary: list[str] = field(default_factory=list)
    appendix_diagnostics: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[EventNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    likelihood_schema_version: str = LIKELIHOOD_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(
            self.architecture_version,
            ARCHITECTURE_VERSION,
            "ReportBundle architecture_version",
        )
        _require_current_version(
            self.branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "ReportBundle branch_schema_version",
        )
        _require_current_version(
            self.likelihood_schema_version,
            LIKELIHOOD_SCHEMA_VERSION,
            "ReportBundle likelihood_schema_version",
        )
        _require_current_version(
            self.report_protocol_version,
            REPORT_PROTOCOL_VERSION,
            "ReportBundle report_protocol_version",
        )
        _require_current_version(
            self.ic_protocol_version,
            IC_PROTOCOL_VERSION,
            "ReportBundle ic_protocol_version",
        )
        _require_allowed_branch_keys(self.branch_verdicts, "ReportBundle.branch_verdicts")
        for verdict in self.branch_verdicts.values():
            verdict.__post_init__()
        if self.macro_verdict is not None:
            self.macro_verdict.__post_init__()
        if self.risk_decision is not None:
            self.risk_decision.__post_init__()
        if self.ic_decision is not None:
            self.ic_decision.__post_init__()
        for decision in self.ic_decisions:
            decision.__post_init__()
        if self.portfolio_plan is not None:
            self.portfolio_plan.__post_init__()
        if self.portfolio_decision is not None:
            self.portfolio_decision.__post_init__()
        if self.review_bundle is not None:
            self.review_bundle.__post_init__()
        for packet in self.symbol_research_packets.values():
            packet.__post_init__()
        reject_retired_intelligence_keys(
            {
                "metadata": self.metadata,
                "ic_hints_by_symbol": self.ic_hints_by_symbol,
            },
            path="ReportBundle",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="ReportBundle")
        return payload


@dataclass
class ReviewTelemetry:
    stage: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    fallback: bool = False
    fallback_reason: str = ""
    score_delta: float = 0.0
    confidence_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchOverlayVerdict:
    symbol: str = ""
    branch_name: str = ""
    status: AgentStatus = AgentStatus.SUCCESS
    thesis: str = ""
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    base_score: float = 0.0
    adjusted_score: float = 0.0
    base_confidence: float = 0.0
    adjusted_confidence: float = 0.0
    score_delta: float = 0.0
    confidence_delta: float = 0.0
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    missing_risks: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    telemetry: ReviewTelemetry = field(default_factory=ReviewTelemetry)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "BranchOverlayVerdict architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "BranchOverlayVerdict branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "BranchOverlayVerdict ic_protocol_version")
        if self.branch_name and self.branch_name not in _ALLOWED_REVIEW_BRANCHES:
            raise ValueError(f"Non-v15 branch overlay: {self.branch_name!r}.")
        reject_retired_intelligence_keys(
            self.metadata,
            path="BranchOverlayVerdict.metadata",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class MasterICHint:
    symbol: str = ""
    status: AgentStatus = AgentStatus.SUCCESS
    thesis: str = ""
    action: ActionLabel = ActionLabel.HOLD
    direction: Direction = Direction.NEUTRAL
    score_hint: float = 0.0
    confidence_hint: float = 0.0
    score_delta: float = 0.0
    confidence_delta: float = 0.0
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    rationale_points: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    telemetry: ReviewTelemetry = field(default_factory=ReviewTelemetry)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "MasterICHint architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "MasterICHint branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "MasterICHint ic_protocol_version")
        reject_retired_intelligence_keys(self.metadata, path="MasterICHint.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class StockReviewBundle:
    agent_name: str = "StockReviewOrchestrator"
    branch_overlay_verdicts_by_symbol: dict[str, dict[str, BranchOverlayVerdict]] = field(default_factory=dict)
    master_hints_by_symbol: dict[str, MasterICHint] = field(default_factory=dict)
    ic_hints_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    branch_summaries: dict[str, BranchVerdict] = field(default_factory=dict)
    macro_verdict: Optional[BranchVerdict] = None
    risk_decision: Optional[RiskDecision] = None
    telemetry: list[ReviewTelemetry] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(
            self.architecture_version,
            ARCHITECTURE_VERSION,
            "StockReviewBundle architecture_version",
        )
        _require_current_version(
            self.branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "StockReviewBundle branch_schema_version",
        )
        _require_current_version(
            self.ic_protocol_version,
            IC_PROTOCOL_VERSION,
            "StockReviewBundle ic_protocol_version",
        )
        _require_current_version(
            self.report_protocol_version,
            REPORT_PROTOCOL_VERSION,
            "StockReviewBundle report_protocol_version",
        )
        _require_allowed_branch_keys(
            self.branch_summaries,
            "StockReviewBundle.branch_summaries",
        )
        for symbol, branch_map in self.branch_overlay_verdicts_by_symbol.items():
            _require_allowed_branch_keys(
                branch_map,
                f"StockReviewBundle.branch_overlay_verdicts_by_symbol[{symbol}]",
            )
            for verdict in branch_map.values():
                verdict.__post_init__()
        for verdict in self.branch_summaries.values():
            verdict.__post_init__()
        for hint in self.master_hints_by_symbol.values():
            hint.__post_init__()
        if self.macro_verdict is not None:
            self.macro_verdict.__post_init__()
        if self.risk_decision is not None:
            self.risk_decision.__post_init__()
        reject_retired_intelligence_keys(
            {
                "metadata": self.metadata,
                "ic_hints_by_symbol": self.ic_hints_by_symbol,
            },
            path="StockReviewBundle",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="StockReviewBundle")
        return payload


@dataclass
class ModelRoleMetadata:
    branch_model: str = ""
    master_model: str = ""
    agent_fallback_model: str = ""
    master_fallback_model: str = ""
    resolved_branch_model: str = ""
    resolved_master_model: str = ""
    master_reasoning_effort: str = ""
    branch_provider: str = ""
    master_provider: str = ""
    branch_timeout: float = 0.0
    master_timeout: float = 0.0
    agent_layer_enabled: bool = False
    branch_fallback_used: bool = False
    master_fallback_used: bool = False
    branch_fallback_reason: str = ""
    master_fallback_reason: str = ""
    universe_key: str = ""
    universe_size: int = 0
    universe_hash: str = ""
    branch_role: str = "per-stock analysis"
    master_role: str = "master synthesis / portfolio-level judgment before deterministic risk and sizing"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTraceStep:
    stage: str = ""
    role: str = ""
    model: str = ""
    success: bool = True
    conclusion: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""
    timeout_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    model_roles: ModelRoleMetadata = field(default_factory=ModelRoleMetadata)
    key_parameters: dict[str, Any] = field(default_factory=dict)
    resolver_directory_priority: list[str] = field(default_factory=list)
    physical_directories_used_for_full_a: list[str] = field(default_factory=list)
    local_union_fallback_used: bool = False
    resolved_file_paths_by_symbol: dict[str, str] = field(default_factory=dict)
    resolution_strategy: str = ""
    steps: list[ExecutionTraceStep] = field(default_factory=list)
    final_deterministic_outcome: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WhatIfScenario:
    scenario_name: str = ""
    trigger: str = ""
    monitoring_indicators: list[str] = field(default_factory=list)
    action: str = ""
    position_adjustment_rule: str = ""
    rerun_full_market_daily_path: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WhatIfPlan:
    scenarios: list[WhatIfScenario] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_by: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalContext:
    market: str = ""
    universe_key: str = "full_a"
    rebalance_date: str = ""
    latest_trade_date: str = ""
    universe_symbols: list[str] = field(default_factory=list)
    universe_hash: str = ""
    industry_map: dict[str, str] = field(default_factory=dict)
    liquidity_filter: dict[str, Any] = field(default_factory=dict)
    macro_regime: str = ""
    cross_section_quant: dict[str, Any] = field(default_factory=dict)
    style_exposures: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: dict[str, Any] = field(default_factory=dict)
    risk_budget: dict[str, Any] = field(default_factory=dict)
    data_quality_issues: list[DataQualityIssue] = field(default_factory=list)
    model_capability_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbol_name_map: dict[str, str] = field(default_factory=dict)
    data_quality_quarantine: list[str] = field(default_factory=list)
    regime_params: dict[str, Any] = field(default_factory=dict)
    freshness_mode: str = "stable"
    effective_target_trade_date: str = ""
    universe_tiers: dict[str, list[str]] = field(default_factory=dict)
    data_quality_diagnostics: DataQualityDiagnostics = field(default_factory=DataQualityDiagnostics)
    macro_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="GlobalContext")
        return payload


@dataclass
class SymbolResearchPacket:
    symbol: str = ""
    company_name: str = ""
    market: str = ""
    category: str = ""
    universe_key: str = "full_a"
    branch_verdicts: dict[str, BranchVerdict] = field(default_factory=dict)
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_confidences: dict[str, float] = field(default_factory=dict)
    branch_theses: dict[str, str] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    diagnostic_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("branch_verdicts", self.branch_verdicts),
            ("branch_scores", self.branch_scores),
            ("branch_confidences", self.branch_confidences),
            ("branch_theses", self.branch_theses),
        ):
            _require_allowed_branch_keys(
                value,
                f"SymbolResearchPacket.{field_name}",
            )
        for verdict in self.branch_verdicts.values():
            verdict.__post_init__()
        reject_retired_intelligence_keys(
            self.metadata,
            path="SymbolResearchPacket.metadata",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="SymbolResearchPacket")
        return payload


@dataclass
class ShortlistItem:
    symbol: str = ""
    company_name: str = ""
    category: str = ""
    rank_score: float = 0.0
    action: ActionLabel = ActionLabel.HOLD
    confidence: float = 0.0
    expected_upside: float = 0.0
    suggested_weight: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioDecision:
    status: AgentStatus = AgentStatus.SUCCESS
    shortlist: list[ShortlistItem] = field(default_factory=list)
    target_exposure: float = 0.0
    target_gross_exposure: float = 0.0
    target_net_exposure: float = 0.0
    cash_ratio: float = 1.0
    target_weights: dict[str, float] = field(default_factory=dict)
    target_positions: dict[str, float] = field(default_factory=dict)
    risk_constraints: dict[str, Any] = field(default_factory=dict)
    master_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    what_if_plan: Optional[WhatIfPlan] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_current_version(self.architecture_version, ARCHITECTURE_VERSION, "PortfolioDecision architecture_version")
        _require_current_version(self.branch_schema_version, BRANCH_SCHEMA_VERSION, "PortfolioDecision branch_schema_version")
        _require_current_version(self.ic_protocol_version, IC_PROTOCOL_VERSION, "PortfolioDecision ic_protocol_version")
        reject_retired_intelligence_keys(
            {
                "risk_constraints": self.risk_constraints,
                "master_hints": self.master_hints,
                "metadata": self.metadata,
            },
            path="PortfolioDecision",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)


@dataclass
class BayesianDecisionRecord:
    """Per-symbol Bayesian decision record with prior/likelihood/posterior."""

    symbol: str = ""
    company_name: str = ""
    prior: dict[str, float] = field(default_factory=dict)
    likelihoods: dict[str, Any] = field(default_factory=_default_likelihood_payload)
    posterior_win_rate: float = 0.0
    posterior_expected_alpha: float = 0.0
    posterior_confidence: float = 0.0
    posterior_action_score: float = 0.0
    posterior_edge_after_costs: float = 0.0
    posterior_capacity_penalty: float = 0.0
    rank: int = 0
    correlation_discount: float = 0.0
    coverage_discount: float = 0.0
    data_quality_penalty: float = 0.0
    fallback_penalty: float = 0.0
    regime_adjustment: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    action_threshold_used: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    likelihood_schema_version: str = LIKELIHOOD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_current_version(
            self.architecture_version,
            ARCHITECTURE_VERSION,
            "BayesianDecisionRecord architecture_version",
        )
        _require_current_version(
            self.branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "BayesianDecisionRecord branch_schema_version",
        )
        _require_current_version(
            self.likelihood_schema_version,
            LIKELIHOOD_SCHEMA_VERSION,
            "BayesianDecisionRecord likelihood_schema_version",
        )
        expected_keys = {
            "schema_version",
            "quant_likelihood",
            "fundamental_likelihood",
            "correlation_matrix",
        }
        actual_keys = set(self.likelihoods)
        if actual_keys != expected_keys:
            raise ValueError(
                "BayesianDecisionRecord likelihood fields must match v15; "
                f"unexpected={sorted(actual_keys - expected_keys)}, "
                f"missing={sorted(expected_keys - actual_keys)}."
            )
        if self.likelihoods.get("schema_version") != LIKELIHOOD_SCHEMA_VERSION:
            raise ValueError("BayesianDecisionRecord likelihood schema mismatch.")
        if self.likelihoods.get("correlation_matrix") != {}:
            raise ValueError(
                "BayesianDecisionRecord correlation_matrix must be empty."
            )
        for field_name in ("quant_likelihood", "fundamental_likelihood"):
            value = float(self.likelihoods[field_name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"BayesianDecisionRecord {field_name} must be in [0, 1]."
                )
        for field_name, raw_value in self.prior.items():
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                raise ValueError(
                    f"BayesianDecisionRecord prior.{field_name} must be numeric."
                )
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"BayesianDecisionRecord prior.{field_name} must be in [0, 1]."
                )
        validate_posterior_numeric_fields(self)
        require_finite_structure(self.metadata, path="BayesianDecisionRecord.metadata")
        unexpected_sources = sorted(
            set(self.evidence_sources) - {"quant", "fundamental"}
        )
        if unexpected_sources:
            raise ValueError(
                "BayesianDecisionRecord has unexpected evidence sources: "
                + ", ".join(unexpected_sources)
            )
        reject_retired_intelligence_keys(
            {"prior": self.prior, "likelihoods": self.likelihoods, "metadata": self.metadata},
            path="BayesianDecisionRecord",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        reject_retired_intelligence_keys(payload, path="BayesianDecisionRecord")
        return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_SCHEMA_VERSION",
    "IC_PROTOCOL_VERSION",
    "REPORT_PROTOCOL_VERSION",
    "ActionLabel",
    "AgentStatus",
    "BayesianDecisionRecord",
    "ConfidenceLabel",
    "CoverageScope",
    "Direction",
    "EventNote",
    "DataQualityIssue",
    "EvidenceItem",
    "RiskLevel",
    "BranchVerdict",
    "BranchOverlayVerdict",
    "RiskDecision",
    "ICDecision",
    "MasterICHint",
    "ModelRoleMetadata",
    "ExecutionTraceStep",
    "ExecutionTrace",
    "GlobalContext",
    "PortfolioPlan",
    "PortfolioDecision",
    "ReportBundle",
    "ShortlistItem",
    "ReviewTelemetry",
    "SymbolResearchPacket",
    "StockReviewBundle",
    "WhatIfScenario",
    "WhatIfPlan",
]
