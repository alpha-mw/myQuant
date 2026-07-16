"""
架构版本与 schema 常量。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_investor.branch_config import get_default_branch_weights

ARCHITECTURE_VERSION = "15.0.0-stable"

AGENT_SCHEMA_VERSION = "2026-07-16.agent.v15.three-branch"

BRANCH_SCHEMA_VERSION = "branch-schema.v15.three-branch"
LIKELIHOOD_SCHEMA_VERSION = "likelihood-schema.v15.two-likelihood"
CALIBRATION_SCHEMA_VERSION = "2026-07-16.calibration.v15.three-branch"
OUTCOME_LEDGER_SCHEMA_VERSION = "2026-07-16.outcome-ledger.v15.three-branch"
CALIBRATION_V2_SCHEMA_VERSION = "2026-07-16.calibration-v2.v15.two-likelihood"
POSTERIOR_OVERLAY_SCHEMA_VERSION = (
    "2026-07-16.posterior-overlay.v15.two-likelihood.v3"
)
DATA_QUALITY_CONTRACT_SCHEMA_VERSION = "2026-04-26.data-quality-contract.v1"
RISK_TENSOR_SCHEMA_VERSION = "2026-04-26.risk-tensor.v1"
PORTFOLIO_OPTIMIZER_SCHEMA_VERSION = "2026-07-16.portfolio-optimizer.v15.v3"
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
FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION = "2026-04-27.factor-shadow-evidence.v1"
FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION = "2026-04-27.factor-evidence-dashboard.v1"
TUSHARE_DATA_CLEANING_SCHEMA_VERSION = "2026-04-27.tushare-data-cleaning.v1"
TUSHARE_FACTOR_READINESS_SCHEMA_VERSION = "2026-04-27.tushare-factor-readiness.v1"
TUSHARE_STORAGE_OPTIMIZATION_SCHEMA_VERSION = "2026-04-27.tushare-storage-optimization.v1"
TUSHARE_PARQUET_MIGRATION_SCHEMA_VERSION = "2026-04-27.tushare-parquet-migration.v1"
FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION = (
    "2026-04-27.factor-backtest-alignment-audit.v1"
)
FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION = "2026-04-27.factor-tradability-audit.v1"
FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION = (
    "2026-04-27.factor-execution-feasibility-audit.v1"
)
FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION = (
    "2026-04-27.factor-execution-cost-simulation.v1"
)
FACTOR_EXECUTION_PENALTY_SCHEMA_VERSION = "2026-04-27.factor-execution-penalty.v1"
BRANCH_TRACKER_SCHEMA_VERSION = "2026-07-16.branch-tracker.v15.three-branch"
DEBATE_TEMPLATE_VERSION = "2026-07-16.branch-debate.v15.three-branch"
IC_PROTOCOL_VERSION = "ic-protocol.v15.three-branch"
REPORT_PROTOCOL_VERSION = "report-protocol.v15.three-branch"

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

CURRENT_BRANCH_ORDER = (
    "quant",
    "fundamental",
    "macro",
)
BRANCH_ORDER = CURRENT_BRANCH_ORDER

CURRENT_BRANCH_WEIGHTS: dict[str, float] = {
    **get_default_branch_weights(),
}


def reject_retired_intelligence_keys(value: Any, *, path: str = "artifact") -> None:
    """Fail closed when a current artifact contains a retired structural key.

    Values are intentionally not inspected: prose can describe historical
    Intelligence behavior, while current machine-readable field names cannot
    resurrect the retired branch.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if "intelligence" in key_text.casefold():
                raise ValueError(
                    f"{child_path} contains a retired Intelligence key."
                )
            reject_retired_intelligence_keys(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_retired_intelligence_keys(child, path=f"{path}[{index}]")


def output_version_payload(
    architecture_version: str = ARCHITECTURE_VERSION,
    branch_schema_version: str = BRANCH_SCHEMA_VERSION,
) -> dict[str, str]:
    return {
        "architecture_version": architecture_version,
        "branch_schema_version": branch_schema_version,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
    }
