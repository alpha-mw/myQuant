"""Diagnostic-only output used when the v16 Codex layer is disabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .protocol_matrix import ARCHITECTURE_VERSION, BRANCH_VERSION

DIAGNOSTIC_SCHEMA_VERSION = "v16.no-agent-diagnostic.v1"


@dataclass(frozen=True)
class V16NoAgentDiagnostic:
    run_id: str
    market: str
    eligible_symbol_count: int
    funnel_symbol_count: int
    data_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("diagnostic run_id is required")
        if str(self.market).strip().upper() != "CN":
            raise ValueError("v16 four-branch diagnostic currently supports CN only")
        if self.eligible_symbol_count < 0 or self.funnel_symbol_count < 0:
            raise ValueError("diagnostic counts must be non-negative")
        if self.funnel_symbol_count > min(500, self.eligible_symbol_count):
            raise ValueError("diagnostic Funnel count violates the sealed candidate boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_version": BRANCH_VERSION,
            "run_id": self.run_id,
            "market": "CN",
            "status": "diagnostic_only",
            "formal_shortlist_generated": False,
            "new_risk_authorized": False,
            "target_weights": {},
            "eligible_symbol_count": self.eligible_symbol_count,
            "funnel_symbol_count": self.funnel_symbol_count,
            "data_summary": dict(self.data_summary),
            "blockers": [
                "codex_stage1_disabled",
                "formal_llm_branch_missing",
                "formal_posterior_not_generated",
            ],
        }
