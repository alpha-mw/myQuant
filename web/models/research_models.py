"""Pydantic models for the research workspace API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from quant_investor.config import config


class ResearchRunRequest(BaseModel):
    """Maps 1:1 to QuantInvestor.__init__ parameters."""

    stock_pool: list[str] = Field(min_length=1)
    market: Literal["CN", "US"] = "CN"
    capital: float = 1_000_000.0
    risk_level: str = "中等"
    lookback_years: float = 1.0
    kline_backend: str = "hybrid"
    enable_macro: bool = True
    enable_quant: bool = True
    enable_kline: bool = True
    enable_fundamental: bool = True
    enable_intelligence: bool = True
    enable_agent_layer: bool = True
    review_model_priority: list[str] = Field(default_factory=list)
    agent_model: str = ""
    agent_fallback_model: str = ""
    master_model: str = ""
    master_fallback_model: str = ""
    agent_timeout: float = config.DEFAULT_AGENT_TIMEOUT_SECONDS
    master_timeout: float = config.DEFAULT_MASTER_TIMEOUT_SECONDS
    preset_id: Optional[str] = None
    # Stock-pool selection metadata (advisory, resolved server-side before run)
    stock_input_mode: Literal["custom", "universe", "multi"] = "custom"
    universe_keys: list[str] = Field(default_factory=list)
    universe_operation: Literal["replace", "merge"] = "replace"


class ResearchJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    created_at: str = ""
    progress_pct: float = 0.0
    error: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None


class ResearchReportResponse(BaseModel):
    markdown: str = ""


class ResearchHistoryItem(BaseModel):
    job_id: str
    created_at: str
    status: str
    market: str = "CN"
    stock_pool: list[str] = Field(default_factory=list)
    total_time: Optional[float] = None
    risk_level: str = "中等"
    preset_id: Optional[str] = None


class ResearchHistoryResponse(BaseModel):
    items: list[ResearchHistoryItem] = Field(default_factory=list)
    total: int = 0


class LLMModelOption(BaseModel):
    id: str
    provider: str
    label: str
    available: bool = False
    prompt_price: float = 0.0
    completion_price: float = 0.0


class LLMModelsResponse(BaseModel):
    models: list[LLMModelOption] = Field(default_factory=list)


class PresetCreateRequest(BaseModel):
    name: str
    description: str = ""
    config: ResearchRunRequest


class PresetUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[ResearchRunRequest] = None


class PresetResponse(BaseModel):
    preset_id: str
    name: str
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class PresetListResponse(BaseModel):
    presets: list[PresetResponse] = Field(default_factory=list)


class RecentRunSummary(BaseModel):
    job_id: str
    created_at: str
    market: str
    stock_pool: list[str] = Field(default_factory=list)
    status: str
    total_time: Optional[float] = None
    recall_context: dict[str, Any] = Field(default_factory=dict)
    selection_meta: dict[str, Any] = Field(default_factory=dict)


class StartupContextResponse(BaseModel):
    recent_runs: list[RecentRunSummary] = Field(default_factory=list)
    suggested_trades: list[dict[str, Any]] = Field(default_factory=list)
    recall_summary: dict[str, Any] = Field(default_factory=dict)
