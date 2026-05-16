"""
架构版本与 schema 常量。
"""

from __future__ import annotations

from quant_investor.branch_config import get_default_branch_weights

ARCHITECTURE_VERSION = "12.0.0-stable"

AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"

BRANCH_SCHEMA_VERSION = "branch-schema.v12.unified-mainline"
CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"
OUTCOME_LEDGER_SCHEMA_VERSION = "2026-04-26.outcome-ledger.v1"
CALIBRATION_V2_SCHEMA_VERSION = "2026-04-26.calibration-v2.v1"
POSTERIOR_OVERLAY_SCHEMA_VERSION = "2026-04-26.posterior-overlay.v1"
DATA_QUALITY_CONTRACT_SCHEMA_VERSION = "2026-04-26.data-quality-contract.v1"
RISK_TENSOR_SCHEMA_VERSION = "2026-04-26.risk-tensor.v1"
PORTFOLIO_OPTIMIZER_SCHEMA_VERSION = "2026-04-26.portfolio-optimizer.v1"
OBSERVABILITY_SCHEMA_VERSION = "2026-04-26.observability.v1"
AUDIT_BUNDLE_SCHEMA_VERSION = "2026-04-26.audit-bundle.v1"
FACTOR_GOVERNANCE_SCHEMA_VERSION = "2026-04-27.factor-governance.v1"
FACTOR_LIBRARY_SCHEMA_VERSION = "2026-04-27.factor-library.v1"
FACTOR_MATRIX_SCHEMA_VERSION = "2026-04-27.factor-matrix.v1"
FACTOR_EXPRESSION_SCHEMA_VERSION = "2026-04-27.factor-expression.v1"
FACTOR_BACKTEST_SCHEMA_VERSION = "2026-04-27.factor-backtest.v1"
FACTOR_ROBUSTNESS_SCHEMA_VERSION = "2026-04-27.factor-robustness.v1"
FACTOR_COST_CAPACITY_SCHEMA_VERSION = "2026-04-27.factor-cost-capacity.v1"
FACTOR_CORRELATION_SCHEMA_VERSION = "2026-04-27.factor-correlation.v1"
FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION = (
    "2026-04-27.factor-portfolio-contribution.v1"
)
FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION = "2026-04-27.factor-library-audit.v1"
FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION = (
    "2026-04-27.factor-production-guardrail.v1"
)
FACTOR_SHADOW_SCORING_SCHEMA_VERSION = "2026-04-27.factor-shadow-scoring.v1"
FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION = "2026-04-27.factor-shadow-comparison.v1"
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
    **get_default_branch_weights(),
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
