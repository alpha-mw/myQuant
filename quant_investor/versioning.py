"""
架构版本与 schema 常量。
"""

from __future__ import annotations

ARCHITECTURE_VERSION = "12.0.0-stable"

AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"

BRANCH_SCHEMA_VERSION = "branch-schema.v12.unified-mainline"
CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"
BRANCH_TRACKER_SCHEMA_VERSION = "2026-03-22.branch-tracker.v2"
DEBATE_TEMPLATE_VERSION = "2026-03-22.branch-debate.v2"
IC_PROTOCOL_VERSION = "ic-protocol.v12.mainline"
REPORT_PROTOCOL_VERSION = "report-protocol.v12.mainline"

PROMOTION_CANDIDATE_SCHEMA_VERSION = "learning.promotion_candidate.v1"
PROMOTION_DECISION_SCHEMA_VERSION = "learning.promotion_decision.v1"
RULE_PROPOSAL_SCHEMA_VERSION = "learning.rule_proposal.v1"
MEMORY_ITEM_SCHEMA_VERSION = "learning.memory_item.v1"
MEMORY_INDEX_SCHEMA_VERSION = "learning.memory_index.v1"
RECALL_HIT_SCHEMA_VERSION = "learning.recall_hit.v1"
RECALL_QUERY_SCHEMA_VERSION = "learning.recall_query.v1"
RECALL_PACKET_SCHEMA_VERSION = "learning.recall_packet.v1"
REFLECTION_EVIDENCE_SCHEMA_VERSION = "learning.reflection_evidence.v1"
REFLECTION_LESSON_DRAFT_SCHEMA_VERSION = "learning.reflection_lesson_draft.v1"
REFLECTION_REPORT_SCHEMA_VERSION = "learning.reflection_report.v1"
TRADE_CASE_SCHEMA_VERSION = "learning.trade_case.v1"

LEGACY_BRANCH_ORDER = [
    "kline",
    "quant",
    "llm_debate",
    "intelligence",
    "macro",
]

CURRENT_BRANCH_ORDER = (
    "kline",
    "quant",
    "fundamental",
    "intelligence",
    "macro",
)
BRANCH_ORDER = CURRENT_BRANCH_ORDER

LEGACY_BRANCH_WEIGHTS: dict[str, float] = {
    "kline": 0.22,
    "quant": 0.28,
    "llm_debate": 0.15,
    "intelligence": 0.20,
    "macro": 0.15,
}

CURRENT_BRANCH_WEIGHTS: dict[str, float] = {
    "kline": 0.22,
    "quant": 0.28,
    "fundamental": 0.15,
    "intelligence": 0.20,
    "macro": 0.15,
}


def output_version_payload(
    architecture_version: str = ARCHITECTURE_VERSION,
    branch_schema_version: str = BRANCH_SCHEMA_VERSION,
) -> dict[str, str]:
    return {
        "architecture_version": architecture_version,
        "branch_schema_version": branch_schema_version,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
    }
