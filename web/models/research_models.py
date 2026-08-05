"""Strict DTO for the public V17 research surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class V17ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: str
    relative_path: str
    byte_sha256: str


class V17AuthorityFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broker_calls: Literal[False]
    execution_calls: Literal[False]
    llm_control_calls: Literal[False]
    order_calls: Literal[False]
    provider_calls: Literal[False]
    selector_writes: Literal[False]
    trade_calls: Literal[False]


class V17Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    current_target: str
    final_target: str
    lane: Literal["SELECTION_POOL", "REVIEW_ONLY_HOLDING"]


class V17MainlinePublicRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["myquant.v17.v4.mainline-public-run.v1"]
    protocol: Literal["myquant.v17.v4"]
    canonical_strategy_id: str
    run_id: str
    state: Literal["ACTIVE"]
    market: Literal["CN_A_SHARE"]
    capability: Literal["RESEARCH_PORTFOLIO"]
    authority_source: Literal["FORMAL_V17_V4"]
    authority_flags: V17AuthorityFlags
    read_only: Literal[True]
    selector_used: Literal[False]
    fallback_used: Literal[False]
    active_pointer_ref: V17ArtifactRef
    mainline_run_ref: V17ArtifactRef
    formal_output_ref: V17ArtifactRef
    portfolio_output_ref: V17ArtifactRef
    source_closure_ref: V17ArtifactRef
    cash_weight: str
    gross_weight: str
    targets: list[V17Target]
    semantic_sha256: str


class LLMModelOption(BaseModel):
    id: str
    provider: str
    label: str
    available: bool = False
    prompt_price: float = 0.0
    completion_price: float = 0.0


class LLMModelsResponse(BaseModel):
    models: list[LLMModelOption]


__all__ = [
    "LLMModelOption",
    "LLMModelsResponse",
    "V17ArtifactRef",
    "V17AuthorityFlags",
    "V17MainlinePublicRun",
    "V17Target",
]
