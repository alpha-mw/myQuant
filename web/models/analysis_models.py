"""Pydantic models for analysis-related API endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)
from web.request_contract import reject_intelligence_named_keys


WebAnalysisBranch = Literal[
    "kline",
    "quant",
    "fundamental",
    "llm_debate",
    "macro",
]
_SUPPORTED_BRANCH_REQUEST_KEYS = {
    *CANONICAL_BRANCH_ORDER,
    "kline",
    "kronos",
    "llm_debate",
}
_CURRENT_WEB_RESULT_BRANCH_ORDER = (
    "kline",
    "quant",
    "fundamental",
    "llm_debate",
    "macro",
)


class AnalysisRiskConfig(BaseModel):
    capital: float = 1_000_000.0
    risk_level: str = "中等"
    max_single_position: float = 0.2
    max_drawdown_limit: float = 0.15
    default_stop_loss: float = 0.08
    keep_cash_buffer: bool = True


class AnalysisPortfolioConfig(BaseModel):
    candidate_limit: int = 10
    allocation_mode: str = "target_weight"
    allow_cash_buffer: bool = True


class AnalysisLlmDebateConfig(BaseModel):
    enabled: bool = True
    models: list[str] = Field(default_factory=list)
    rounds: int = 2
    assignment_mode: str = "random_balanced"
    judge_mode: str = "auto"
    judge_model: Optional[str] = None
    assignments: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "single"
    targets: list[str] = Field(default_factory=list)
    preset: str = "quick_scan"
    market: str = "CN"
    branches: dict[str, dict[str, Any]] = Field(default_factory=dict)
    risk: AnalysisRiskConfig = Field(default_factory=AnalysisRiskConfig)
    portfolio: AnalysisPortfolioConfig = Field(default_factory=AnalysisPortfolioConfig)
    llm_debate: AnalysisLlmDebateConfig = Field(default_factory=AnalysisLlmDebateConfig)

    # Supported non-branch compatibility fields.
    stocks: list[str] = Field(default_factory=list)
    capital: Optional[float] = None
    risk_level: Optional[str] = None
    enable_kline: Optional[bool] = None
    enable_kronos: Optional[bool] = None  # 向后兼容别名
    enable_llm_debate: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def validate_branch_contract(
        cls,
        value: Any,
    ) -> Any:
        if not isinstance(value, dict):
            return value
        reject_intelligence_named_keys(value)
        branches = value.get("branches", {}) or {}
        if not isinstance(branches, dict):
            raise ValueError("branches must be an object")
        unknown = sorted(set(branches) - _SUPPORTED_BRANCH_REQUEST_KEYS)
        if unknown:
            raise ValueError(
                "branches contains unsupported keys: " + ", ".join(unknown)
            )
        for branch_name, config in branches.items():
            if not isinstance(config, dict):
                raise ValueError(f"branches.{branch_name} must be an object")
            if branch_name in CANONICAL_BRANCH_ORDER and "enabled" in config:
                raise ValueError(
                    f"branches.{branch_name}.enabled is not supported; "
                    "v14 canonical branches always execute"
                )
            allowed_config_keys = {"settings"}
            if branch_name not in CANONICAL_BRANCH_ORDER:
                allowed_config_keys.add("enabled")
            unknown_config_keys = sorted(set(config) - allowed_config_keys)
            if unknown_config_keys:
                raise ValueError(
                    f"branches.{branch_name} contains unsupported keys: "
                    + ", ".join(unknown_config_keys)
                )
            settings = config.get("settings", {})
            if settings is not None and not isinstance(settings, dict):
                raise ValueError(f"branches.{branch_name}.settings must be an object")
        return value


class AnalysisPresetOption(BaseModel):
    id: str
    label: str
    description: str
    mode: str
    defaults: dict[str, Any] = Field(default_factory=dict)


class AnalysisModelOption(BaseModel):
    id: str
    label: str
    provider: str
    enabled: bool = True
    note: Optional[str] = None


class AnalysisOptionsResponse(BaseModel):
    presets: list[AnalysisPresetOption] = Field(default_factory=list)
    branch_defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    llm_models: list[AnalysisModelOption] = Field(default_factory=list)
    risk_templates: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    created_at: str
    source: str = "web"
    market: str = "CN"
    mode: str = "single"
    preset: str = "quick_scan"
    stock_count: int = 0
    stocks: list[str] = Field(default_factory=list)
    target_exposure: float = 0.0
    style_bias: str = "均衡"
    risk_level: str = "normal"
    candidate_symbols: list[str] = Field(default_factory=list)
    title: str = ""


class BranchDetailResult(BaseModel):
    branch_name: WebAnalysisBranch
    enabled: bool = True
    score: float = 0.0
    confidence: float = 0.0
    explanation: str = ""
    risks: list[str] = Field(default_factory=list)
    top_symbols: list[str] = Field(default_factory=list)
    branch_mode: Optional[str] = None
    settings: dict[str, Any] = Field(default_factory=dict)
    model_assignment: list[dict[str, Any]] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskReview(BaseModel):
    risk_level: str = "normal"
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    max_single_position: float = 0.2
    max_drawdown_limit: float = 0.15
    default_stop_loss: float = 0.08
    keep_cash_buffer: bool = True
    stress_test: str = ""


class SymbolDecision(BaseModel):
    symbol: str
    action: str = "watch"
    current_price: float = 0.0
    recommended_entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    suggested_weight: float = 0.0
    suggested_amount: float = 0.0
    suggested_shares: int = 0
    confidence: float = 0.0
    consensus_score: float = 0.0
    branch_positive_count: int = 0
    trend_regime: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExecutionPlan(BaseModel):
    capital: float = 0.0
    target_exposure: float = 0.0
    investable_capital: float = 0.0
    reserved_cash: float = 0.0
    symbol_decisions: list[SymbolDecision] = Field(default_factory=list)


class AnalysisSessionDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture_version: str
    branch_schema_version: str
    likelihood_schema_version: str
    report_protocol_version: str
    analysis_id: str
    created_at: str
    source: str = "web"
    request: AnalysisRunRequest
    total_time: float = 0.0
    research_mode: str = "production"
    final_decision: str = ""
    target_exposure: float = 0.0
    style_bias: str = "均衡"
    sector_preferences: list[str] = Field(default_factory=list)
    candidate_symbols: list[str] = Field(default_factory=list)
    data_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_notes: list[str] = Field(default_factory=list)
    branches: list[BranchDetailResult] = Field(default_factory=list)
    risk: RiskReview = Field(default_factory=RiskReview)
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    trade_recommendations: list[SymbolDecision] = Field(default_factory=list)
    report_markdown: str = ""
    execution_log: list[str] = Field(default_factory=list)
    llm_assignments: list[dict[str, Any]] = Field(default_factory=list)
    config_applied: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_retired_response_keys(cls, value: Any) -> Any:
        reject_intelligence_named_keys(value, path="analysis_session")
        return value

    @model_validator(mode="after")
    def validate_current_schema_envelope(self) -> "AnalysisSessionDetail":
        expected = {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
            "report_protocol_version": REPORT_PROTOCOL_VERSION,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ValueError(
                    f"{field_name} mismatch: expected {expected_value!r}, "
                    f"got {actual!r}"
                )
        branch_names = tuple(branch.branch_name for branch in self.branches)
        if branch_names != _CURRENT_WEB_RESULT_BRANCH_ORDER:
            raise ValueError(
                "branches mismatch: expected "
                f"{list(_CURRENT_WEB_RESULT_BRANCH_ORDER)!r}, "
                f"got {list(branch_names)!r}"
            )
        for branch in self.branches:
            if (
                branch.branch_name in CANONICAL_BRANCH_ORDER
                and branch.enabled is not True
            ):
                raise ValueError(
                    f"canonical branch {branch.branch_name!r} must be enabled"
                )
        return self


class AnalysisResult(AnalysisSessionDetail):
    """Backward-compatible alias for existing consumers."""


class AnalysisHistoryResponse(BaseModel):
    items: list[AnalysisHistoryItem] = Field(default_factory=list)
    total: int = 0


class AnalysisRunResponse(BaseModel):
    ok: bool = True
    job_id: Optional[str] = None
    status: str = "queued"
    result: Optional[AnalysisSessionDetail] = None
    error: Optional[str] = None


class AnalysisJobResponse(BaseModel):
    ok: bool = True
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[AnalysisSessionDetail] = None
    error: Optional[str] = None


class AnalysisDeleteResponse(BaseModel):
    ok: bool = True
    deleted_count: int = 0
    message: str = ""
