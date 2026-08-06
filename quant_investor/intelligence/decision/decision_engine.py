"""Exact five-state investment-decision policy for the I1 research layer."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Final

from .._core import content_ref
from .evidence_collector import (
    validate_context_replay_closure,
    validate_investment_decision_context,
)
from .models import (
    DECISION_RECEIPT_VERSION,
    MAX_BLOCKER_CODES,
    MAX_REASON_CODES,
    canonical_codes,
    canonical_content_ref,
    canonical_timestamp,
    ensure_artifact_size,
    fail,
    validate_decision_policy,
)
from .receipts import seal_artifact, validate_closed_artifact
from .risk_assessor import validate_risk_assessment_receipt

DECISION_STATE_ORDER: Final = (
    "THESIS_INVALIDATED",
    "INSUFFICIENT_EVIDENCE",
    "WATCHLIST",
    "RESEARCH_APPROVED",
    "PAPER_CANDIDATE",
)
REASON_CODES: Final = frozenset(
    {
        "HARD_RISK_VETO",
        "PAPER_CONFIDENCE_BELOW_MIN",
        "PAPER_GATES_PASSED",
        "PAPER_POSTERIOR_BELOW_MIN",
        "PAPER_REQUIRED_INPUT_UNAVAILABLE",
        "PAPER_R22_NOT_SUPPORTED",
        "PAPER_RISK_ABOVE_MAX",
        "PREREGISTERED_HYPOTHESIS_FAILED",
        "RESEARCH_CONFIDENCE_BELOW_MIN",
        "RESEARCH_GATES_PASSED",
        "RESEARCH_POSTERIOR_BELOW_MIN",
        "RESEARCH_R22_NOT_SUPPORTED",
        "RESEARCH_RISK_ABOVE_MAX",
    }
)
_DECISION_PAYLOAD_FIELDS: Final = {
    "bayesian_posterior",
    "blocker_codes",
    "context_ref",
    "policy_ref",
    "r22_hypothesis_status",
    "reason_codes",
    "research_confidence",
    "risk_ref",
    "state",
}


def _validated_context(
    context: Mapping[str, Any], context_replay_closure: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = validate_context_replay_closure(context_replay_closure)
    row = validate_investment_decision_context(
        context,
        i0_replay_inputs=replay["i0_replay_inputs"],
        policy=replay["policy"],
        context_notes=replay["context_notes"],
        ai_drafts=replay["ai_drafts"],
        r22_request_path=replay["r22_request_path"],
        r22_request_sha256=replay["r22_request_sha256"],
    )
    return row, replay


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value["artifact_id"]),
        str(value["artifact_version"]),
        str(value["byte_sha256"]),
        str(value["semantic_sha256"]),
    )


def _one_bound_component(
    values: Any,
    *,
    reference: Mapping[str, Any],
    identity_field: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(values, list):
        fail("I1_SHAPE_INVALID", f"{label} closure must be a list")
    expected_ref = canonical_content_ref(reference, label=f"context.{label}_ref")
    matches = [
        row
        for row in values
        if type(row) is dict and content_ref(row, identity_field=identity_field) == expected_ref
    ]
    if len(matches) != 1:
        fail("I1_REF_MISMATCH", f"decision requires exactly one bound {label}")
    return matches[0]


def _availability_status(context: Mapping[str, Any], requirement: str) -> str:
    availability = context["availability"]
    if type(availability) is not dict or type(availability.get(requirement)) is not dict:
        fail("I1_SHAPE_INVALID", "decision context availability is malformed")
    status = availability[requirement].get("status")
    if status not in {"AVAILABLE", "UNAVAILABLE"}:
        fail("I1_SHAPE_INVALID", "decision context availability status is invalid")
    return str(status)


def _risk_statuses(risk_receipt: Mapping[str, Any]) -> dict[str, str]:
    rows = risk_receipt.get("dimension_rows")
    if type(rows) is not list:
        fail("I1_SHAPE_INVALID", "risk receipt dimension rows are missing")
    result: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {
            "assessment_refs",
            "dimension",
            "dimension_severity",
            "status",
        }:
            fail("I1_SHAPE_INVALID", "risk dimension row shape is not closed")
        dimension = str(row["dimension"])
        if dimension in result or row["status"] not in {"AVAILABLE", "UNAVAILABLE"}:
            fail("I1_SHAPE_INVALID", "risk dimension rows are invalid")
        result[dimension] = str(row["status"])
    return result


def _missing_blockers(
    *,
    context: Mapping[str, Any],
    risk_statuses: Mapping[str, str],
    policy: Mapping[str, Any],
    tier: str,
) -> list[str]:
    required_classes = policy[f"{tier.lower()}_required_classes"]
    required_dimensions = policy[f"{tier.lower()}_required_risk_dimensions"]
    blockers = [
        f"{tier}_REQUIRED_{requirement}_UNAVAILABLE"
        for requirement in required_classes
        if _availability_status(context, str(requirement)) == "UNAVAILABLE"
    ]
    blockers.extend(
        f"{tier}_REQUIRED_RISK_{dimension}_UNAVAILABLE"
        for dimension in required_dimensions
        if risk_statuses.get(str(dimension)) == "UNAVAILABLE"
    )
    return blockers


def make_investment_decision(
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    risk_receipt: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Apply the fixed five-state precedence without creating trade authority."""

    cutoff = canonical_timestamp(as_of, label="as_of")
    context_row, replay = _validated_context(context, context_replay_closure)
    policy_row = validate_decision_policy(policy)
    if replay["policy"] != policy_row:
        fail("I1_REF_MISMATCH", "decision policy does not match the context replay closure")
    if context_row["timestamp"] != cutoff or context_row["as_of"] != cutoff:
        fail("I1_REF_MISMATCH", "decision as_of does not match its context")
    risk_row = validate_risk_assessment_receipt(
        risk_receipt,
        context=context_row,
        context_replay_closure=replay,
        policy=policy_row,
        assessments_by_dimension=assessments_by_dimension,
        as_of=cutoff,
    )

    i0 = replay["i0_replay_inputs"]
    fusion = i0["fusion_receipt"]
    expected_fusion_ref = canonical_content_ref(
        context_row["fusion_ref"], label="context.fusion_ref"
    )
    if content_ref(fusion, identity_field="receipt_id") != expected_fusion_ref:
        fail("I1_REF_MISMATCH", "decision context does not bind the replayed fusion receipt")
    bayesian = _one_bound_component(
        i0["bayesian_receipts"],
        reference=context_row["bayesian_ref"],
        identity_field="receipt_id",
        label="bayesian",
    )
    research_confidence = Decimal(str(fusion["research_confidence_score"]))
    posterior = Decimal(str(bayesian["posterior"]))
    overall_risk = (
        None if risk_row["overall_severity"] is None else Decimal(str(risk_row["overall_severity"]))
    )
    status = context_row["r22_hypothesis_status"]
    risk_statuses = _risk_statuses(risk_row)
    research_blockers = _missing_blockers(
        context=context_row,
        risk_statuses=risk_statuses,
        policy=policy_row,
        tier="RESEARCH",
    )
    paper_blockers = _missing_blockers(
        context=context_row,
        risk_statuses=risk_statuses,
        policy=policy_row,
        tier="PAPER",
    )

    reasons: list[str] = []
    if status == "FAILED":
        reasons.append("PREREGISTERED_HYPOTHESIS_FAILED")
    if risk_row["hard_veto_codes"]:
        reasons.append("HARD_RISK_VETO")
    if research_confidence < Decimal(str(policy_row["min_research_confidence"])):
        reasons.append("RESEARCH_CONFIDENCE_BELOW_MIN")
    if posterior < Decimal(str(policy_row["min_research_posterior"])):
        reasons.append("RESEARCH_POSTERIOR_BELOW_MIN")
    if overall_risk is not None and overall_risk > Decimal(str(policy_row["max_research_risk"])):
        reasons.append("RESEARCH_RISK_ABOVE_MAX")
    if policy_row["require_r22_supported_for_research"] and status != "SUPPORTED":
        reasons.append("RESEARCH_R22_NOT_SUPPORTED")
    if paper_blockers:
        reasons.append("PAPER_REQUIRED_INPUT_UNAVAILABLE")
    if research_confidence < Decimal(str(policy_row["min_paper_confidence"])):
        reasons.append("PAPER_CONFIDENCE_BELOW_MIN")
    if posterior < Decimal(str(policy_row["min_paper_posterior"])):
        reasons.append("PAPER_POSTERIOR_BELOW_MIN")
    if overall_risk is not None and overall_risk > Decimal(str(policy_row["max_paper_risk"])):
        reasons.append("PAPER_RISK_ABOVE_MAX")
    if policy_row["require_r22_supported_for_paper"] and status != "SUPPORTED":
        reasons.append("PAPER_R22_NOT_SUPPORTED")

    research_failure_reasons = {
        "HARD_RISK_VETO",
        "RESEARCH_CONFIDENCE_BELOW_MIN",
        "RESEARCH_POSTERIOR_BELOW_MIN",
        "RESEARCH_R22_NOT_SUPPORTED",
        "RESEARCH_RISK_ABOVE_MAX",
    }
    research_pass = all(
        (
            status != "FAILED",
            not research_blockers,
            not (set(reasons) & research_failure_reasons),
        )
    )
    if research_pass:
        reasons.append("RESEARCH_GATES_PASSED")
    paper_failure_reasons = {
        "PAPER_CONFIDENCE_BELOW_MIN",
        "PAPER_POSTERIOR_BELOW_MIN",
        "PAPER_R22_NOT_SUPPORTED",
        "PAPER_RISK_ABOVE_MAX",
    }
    paper_pass = all(
        (
            research_pass,
            not paper_blockers,
            not (set(reasons) & paper_failure_reasons),
        )
    )
    if paper_pass:
        reasons.append("PAPER_GATES_PASSED")

    if status == "FAILED":
        state = "THESIS_INVALIDATED"
    elif research_blockers:
        state = "INSUFFICIENT_EVIDENCE"
    elif not research_pass:
        state = "WATCHLIST"
    elif not paper_pass:
        state = "RESEARCH_APPROVED"
    else:
        state = "PAPER_CANDIDATE"

    reasons = canonical_codes(reasons, label="reason_codes", maximum=MAX_REASON_CODES)
    if any(reason not in REASON_CODES for reason in reasons):
        fail("I1_SHAPE_INVALID", "decision reason inventory is not closed")
    blockers = canonical_codes(
        [*research_blockers, *paper_blockers],
        label="blocker_codes",
        maximum=MAX_BLOCKER_CODES,
    )
    result = seal_artifact(
        version=DECISION_RECEIPT_VERSION,
        identity_field="decision_receipt_id",
        timestamp_value=cutoff,
        payload={
            "bayesian_posterior": str(bayesian["posterior"]),
            "blocker_codes": blockers,
            "context_ref": content_ref(context_row, identity_field="context_id"),
            "policy_ref": content_ref(policy_row, identity_field="policy_id"),
            "r22_hypothesis_status": status,
            "reason_codes": reasons,
            "research_confidence": str(fusion["research_confidence_score"]),
            "risk_ref": content_ref(risk_row, identity_field="risk_receipt_id"),
            "state": state,
        },
    )
    ensure_artifact_size(result)
    return result


def validate_investment_decision_receipt(
    document: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    risk_receipt: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Fully replay the decision so a resealed forged state is rejected."""

    row = validate_closed_artifact(
        document,
        version=DECISION_RECEIPT_VERSION,
        identity_field="decision_receipt_id",
        payload_fields=_DECISION_PAYLOAD_FIELDS,
    )
    expected = make_investment_decision(
        context=context,
        context_replay_closure=context_replay_closure,
        policy=policy,
        risk_receipt=risk_receipt,
        assessments_by_dimension=assessments_by_dimension,
        as_of=as_of,
    )
    if row != expected:
        fail("I1_REPLAY_MISMATCH", "decision receipt does not match its replay closure")
    return row


__all__ = [
    "DECISION_STATE_ORDER",
    "REASON_CODES",
    "make_investment_decision",
    "validate_decision_policy",
    "validate_investment_decision_receipt",
]
