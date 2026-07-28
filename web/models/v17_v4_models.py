"""Versioned, read-only DTOs for the V17 v4 canary Web surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class V17V4Authority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broker: Literal[False]
    execution: Literal[False]
    formal_research_publication: Literal[True]
    order: Literal[False]
    research_runtime_default: Literal[False]
    trade: Literal[False]


class V17V4SideEffects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broker_calls: Literal[False]
    execution_calls: Literal[False]
    llm_control_calls: Literal[False]
    order_calls: Literal[False]
    provider_calls: Literal[False]
    selector_writes: Literal[False]
    trade_calls: Literal[False]


class V17V4ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_version: str
    byte_sha256: str
    cutoff: str
    relative_path: str
    semantic_sha256: str
    strategy_id: str


class V17V4Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_target: str
    final_target: str
    lane: Literal["REVIEW_ONLY_HOLDING", "SELECTION_POOL"]
    symbol: str


class V17V4ResearchRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authority: V17V4Authority
    cash_weight: str
    cutoff: str
    formal_activation_receipt_ref: V17V4ArtifactRef
    formal_active_pointer_ref: V17V4ArtifactRef
    formal_output_ref: V17V4ArtifactRef
    gross_weight: str
    is_default: Literal[False]
    portfolio_output_ref: V17V4ArtifactRef
    protocol_version: Literal["myquant.v17.v4"]
    read_only: Literal[True]
    run_id: str
    semantic_sha256: str
    side_effects: V17V4SideEffects
    state: Literal["FORMAL_ACTIVE"]
    strategy_id: str
    surface: Literal["WEB"]
    targets: list[V17V4Target]
    version: Literal["myquant.v17.v4.public-run-dto.v1"]
    view_label: Literal["CANARY"]


__all__ = ["V17V4ResearchRunResponse"]
