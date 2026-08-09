"""Five-state deterministic Decision v2 with no AI or portfolio authority."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any, Final

from .._core import (
    common_fields,
    content_ref,
    decimal_text,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .fusion import validate_fusion_projection_v2
from .graph import validate_evidence_graph_v2
from .models import (
    DecisionV2ContractError,
    decision_contract,
    validate_decision_policy_v2,
)

DECISION_RECEIPT_V2_VERSION: Final = "myquant.v17.research-intelligence-v2.decision-receipt.v1"
DECISION_STATES: Final = frozenset(
    {
        "THESIS_INVALIDATED",
        "INSUFFICIENT_EVIDENCE",
        "WATCHLIST",
        "RESEARCH_APPROVED",
        "PAPER_CANDIDATE",
    }
)
_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_RECEIPT_FIELDS: Final = _COMMON_FIELDS | {
    "bayesian_posterior",
    "blocker_codes",
    "company_code",
    "decision_id",
    "deterministic_percentile",
    "fusion_score",
    "graph_ref",
    "policy_ref",
    "projection_ref",
    "r22_hypothesis_status",
    "reason_codes",
    "risk_severity",
    "run_id",
    "semantic_sha256",
    "state",
    "version",
}


def _fail(message: str) -> None:
    raise DecisionV2ContractError(message)


def _projection_row(projection: Mapping[str, Any], *, company: str) -> tuple[dict[str, Any], str]:
    matches = [row for row in projection["projected_records"] if row["symbol"] == company]
    if len(matches) != 1:
        _fail("FusionProjectionV2 subject row is not unique")
    row = dict(matches[0])
    count = len(projection["projected_records"])
    with localcontext() as context:
        context.prec = 50
        percentile = Decimal(count - int(row["rank"]) + 1) / Decimal(count)
    return row, decimal_text(percentile)


def _policy_blockers(graph: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    blockers = list(graph["blocker_codes"])
    if graph["industry_state"] != policy["mandatory_industry_state"]:
        blockers.append(f"INDUSTRY_STATE_{graph['industry_state']}_NOT_ADMITTED")
    if graph["theme_state"] not in policy["mandatory_theme_states"]:
        blockers.append(f"THEME_STATE_{graph['theme_state']}_NOT_ADMITTED")
    if graph["fundamental_stale_sessions"] > policy["allowed_fundamental_stale_sessions"]:
        blockers.append("FUNDAMENTAL_STALENESS_EXCEEDED")
    return sorted(set(blockers), key=lambda value: value.encode("ascii"))


def _gate_reasons(
    *,
    graph: Mapping[str, Any],
    policy: Mapping[str, Any],
    fusion_score: str | None,
) -> tuple[list[str], bool]:
    reasons: list[str] = []
    passed = True
    vetoes = set(graph["policy_independent_hard_veto_codes"])
    allowed_vetoes = set(policy["hard_veto_codes"])
    if not vetoes.issubset(allowed_vetoes):
        _fail("risk closure contains an owner-unsealed hard veto code")
    if vetoes:
        reasons.append("HARD_RISK_VETO")
        passed = False
    if fusion_score is None or Decimal(fusion_score) < Decimal(policy["fusion_threshold"]):
        reasons.append("FUSION_SCORE_BELOW_MIN")
        passed = False
    if Decimal(graph["bayesian_posterior"]) < Decimal(policy["posterior_threshold"]):
        reasons.append("BAYESIAN_POSTERIOR_BELOW_MIN")
        passed = False
    if graph["overall_risk"] is not None and Decimal(graph["overall_risk"]) > Decimal(
        policy["max_risk"]
    ):
        reasons.append("RISK_ABOVE_MAX")
        passed = False
    if graph["r22_hypothesis_status"] != policy["required_r22_status"]:
        reasons.append("R22_REQUIRED_STATUS_NOT_MET")
        passed = False
    return reasons, passed


def _validated_projection(
    *,
    fusion_projection: Mapping[str, Any] | None,
    fusion_projection_validation_closure: Mapping[str, Any] | None,
    graph: Mapping[str, Any],
    issued_at: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if (fusion_projection is None) != (fusion_projection_validation_closure is None):
        _fail("fusion projection and closure must be provided together")
    if fusion_projection is None:
        if graph["fusion_ready"] is True:
            _fail("fusion-ready graph requires a same-closure projection")
        return None, None, None
    if type(fusion_projection_validation_closure) is not dict:
        _fail("fusion projection closure must be exact")
    projection = validate_fusion_projection_v2(
        fusion_projection,
        **dict(fusion_projection_validation_closure),
    )
    graph_ref = content_ref(graph, identity_field="graph_id")
    if (
        projection["timestamp"] != issued_at
        or projection["run_id"] != graph["run_id"]
        or graph_ref not in projection["graph_refs"]
    ):
        _fail("Decision v2 projection is outside the graph closure")
    projection_row, percentile = _projection_row(
        projection,
        company=graph["company_code"],
    )
    return projection, projection_row["effective_score"], percentile


def _decision_state(
    *,
    graph: Mapping[str, Any],
    blockers: list[str],
    deterministic_gates_passed: bool,
    reasons: list[str],
) -> str:
    preregistered_failed = (
        graph["r22_preregistered"] is True and graph["r22_hypothesis_status"] == "FAILED"
    )
    if preregistered_failed:
        reasons.append("PREREGISTERED_HYPOTHESIS_FAILED")
        return "THESIS_INVALIDATED"
    if blockers:
        return "INSUFFICIENT_EVIDENCE"
    if not deterministic_gates_passed:
        return "WATCHLIST"
    reasons.append("RESEARCH_GATES_PASSED")
    if graph["fundamental_stale_sessions"] > 0:
        reasons.append("FUNDAMENTAL_STALE_RESEARCH_ONLY")
        return "RESEARCH_APPROVED"
    reasons.append("PAPER_GATES_PASSED")
    return "PAPER_CANDIDATE"


@decision_contract
def make_decision_v2(
    *,
    evidence_graph: Mapping[str, Any],
    graph_validation_closure: Mapping[str, Any],
    fusion_projection: Mapping[str, Any] | None,
    fusion_projection_validation_closure: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Apply the fixed priority table while preserving all computed blockers."""

    issued_at = timestamp(as_of, label="as_of")
    if type(graph_validation_closure) is not dict:
        _fail("graph_validation_closure must be an exact mapping")
    graph = validate_evidence_graph_v2(evidence_graph, **dict(graph_validation_closure))
    policy_row = validate_decision_policy_v2(policy)
    if graph["timestamp"] != issued_at or policy_row["timestamp"] > issued_at:
        _fail("Decision v2 time closure mismatch")
    projection, fusion_score, percentile = _validated_projection(
        fusion_projection=fusion_projection,
        fusion_projection_validation_closure=fusion_projection_validation_closure,
        graph=graph,
        issued_at=issued_at,
    )
    blockers = _policy_blockers(graph, policy_row)
    reasons, deterministic_gates_passed = _gate_reasons(
        graph=graph,
        policy=policy_row,
        fusion_score=fusion_score,
    )
    state = _decision_state(
        graph=graph,
        blockers=blockers,
        deterministic_gates_passed=deterministic_gates_passed,
        reasons=reasons,
    )
    if state not in DECISION_STATES:
        _fail("Decision v2 state is invalid")
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "bayesian_posterior": graph["bayesian_posterior"],
            "blocker_codes": blockers,
            "company_code": graph["company_code"],
            "deterministic_percentile": percentile,
            "fusion_score": fusion_score,
            "graph_ref": content_ref(graph, identity_field="graph_id"),
            "policy_ref": content_ref(policy_row, identity_field="policy_id"),
            "projection_ref": (
                None
                if projection is None
                else content_ref(projection, identity_field="projection_id")
            ),
            "r22_hypothesis_status": graph["r22_hypothesis_status"],
            "reason_codes": sorted(set(reasons), key=lambda value: value.encode("ascii")),
            "risk_severity": graph["overall_risk"],
            "run_id": graph["run_id"],
            "state": state,
            "version": DECISION_RECEIPT_V2_VERSION,
        },
        identity_field="decision_id",
    )


@decision_contract
def validate_decision_receipt_v2(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="decision_id")
    require_exact_keys(row, _RECEIPT_FIELDS, label="DecisionReceiptV2")
    expected = make_decision_v2(**closure)
    if row != expected or row["version"] != DECISION_RECEIPT_V2_VERSION:
        _fail("DecisionReceiptV2 replay mismatch")
    return row


__all__ = [
    "DECISION_RECEIPT_V2_VERSION",
    "DECISION_STATES",
    "make_decision_v2",
    "validate_decision_receipt_v2",
]
