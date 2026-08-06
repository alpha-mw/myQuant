"""Deterministic four-dimension risk assessment for the I1 decision layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import canonical_bytes

from .._core import content_ref
from .evidence_collector import (
    validate_context_replay_closure,
    validate_investment_decision_context,
)
from .models import (
    RISK_RECEIPT_VERSION,
    MAX_ASSESSMENTS_PER_DIMENSION,
    RISK_ASSESSMENT_KINDS,
    RISK_DIMENSIONS,
    bounded_text,
    canonical_decimal,
    canonical_content_ref,
    canonical_exact_ref,
    canonical_codes,
    canonical_timestamp,
    ensure_artifact_size,
    fail,
    sorted_content_refs,
    sorted_exact_source_refs,
    validate_decision_policy,
)
from .receipts import seal_artifact, validate_closed_artifact

RISK_DIMENSION_ORDER: Final = tuple(
    sorted(RISK_DIMENSIONS, key=lambda value: value.encode("ascii"))
)
ASSESSMENT_STATUSES: Final = {"AVAILABLE", "UNAVAILABLE"}
_ASSESSMENT_FIELDS: Final = {
    "assessment_id",
    "evidence_refs",
    "hard_veto_code",
    "kind",
    "reason",
    "severity",
    "source_refs",
}
_GROUP_FIELDS: Final = {"assessments", "status"}
_RISK_PAYLOAD_FIELDS: Final = {
    "context_ref",
    "dimension_rows",
    "hard_veto_codes",
    "overall_severity",
    "policy_ref",
    "unavailable_dimensions",
}


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value["artifact_id"]),
        str(value["artifact_version"]),
        str(value["byte_sha256"]),
        str(value["semantic_sha256"]),
    )


def _source_ref_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value[key]) for key in sorted(value))


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


def _canonical_source_ref(value: Any, *, label: str, as_of: str) -> dict[str, str]:
    row = canonical_exact_ref(value, label=label)
    if row["cutoff"] > as_of:
        fail("I1_FUTURE_INPUT", f"{label} postdates the risk receipt")
    return row


def _normalize_assessment(
    value: Any,
    *,
    label: str,
    as_of: str,
    admitted_evidence_keys: set[tuple[str, str, str, str]],
    authorized_source_keys: set[tuple[str, ...]],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _ASSESSMENT_FIELDS:
        fail("I1_SHAPE_INVALID", f"{label} shape is not closed")
    kind = value["kind"]
    if kind not in RISK_ASSESSMENT_KINDS:
        fail("I1_SHAPE_INVALID", f"{label}.kind is not allowlisted")
    severity_text = canonical_decimal(
        value["severity"],
        label=f"{label}.severity",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    severity = Decimal(severity_text)
    hard_veto_code = value["hard_veto_code"]
    if hard_veto_code is not None:
        hard_veto_code = canonical_codes([hard_veto_code], label=f"{label}.hard_veto_code")[0]
    if kind == "NO_MATERIAL_RISK_IDENTIFIED" and (
        severity != Decimal("0") or hard_veto_code is not None
    ):
        fail(
            "I1_SHAPE_INVALID",
            "NO_MATERIAL_RISK_IDENTIFIED requires zero severity and no hard veto",
        )

    raw_evidence_refs = value["evidence_refs"]
    if isinstance(raw_evidence_refs, (str, bytes)) or not isinstance(raw_evidence_refs, Sequence):
        fail("I1_SHAPE_INVALID", f"{label}.evidence_refs must be a sequence")
    evidence_refs = sorted_content_refs(
        raw_evidence_refs, label=f"{label}.evidence_refs", maximum=256
    )
    evidence_keys = [_ref_key(ref) for ref in evidence_refs]
    if any(key not in admitted_evidence_keys for key in evidence_keys):
        fail("I1_REF_MISMATCH", f"{label} cites evidence outside the decision context")

    raw_source_refs = value["source_refs"]
    if isinstance(raw_source_refs, (str, bytes)) or not isinstance(raw_source_refs, Sequence):
        fail("I1_SHAPE_INVALID", f"{label}.source_refs must be a sequence")
    source_refs = sorted_exact_source_refs(
        raw_source_refs, label=f"{label}.source_refs", maximum=256
    )
    for index, row in enumerate(source_refs):
        if row["cutoff"] > as_of:
            fail("I1_FUTURE_INPUT", f"{label}.source_refs[{index}] postdates the risk receipt")
    source_keys = [_source_ref_key(ref) for ref in source_refs]
    if any(key not in authorized_source_keys for key in source_keys):
        fail("I1_REF_MISMATCH", f"{label} cites a source outside the Observation closure")
    if not evidence_refs and not source_refs:
        fail("I1_SHAPE_INVALID", f"{label} requires admitted evidence or an authorized source")

    body = {
        "evidence_refs": evidence_refs,
        "hard_veto_code": hard_veto_code,
        "kind": kind,
        "reason": bounded_text(value["reason"], label=f"{label}.reason"),
        "severity": severity_text,
        "source_refs": source_refs,
    }
    expected_id = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if value["assessment_id"] != expected_id:
        fail("I1_REPLAY_MISMATCH", f"{label}.assessment_id does not match its content")
    return {"assessment_id": expected_id, **body}


def _normalize_assessments(
    assessments_by_dimension: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    replay: Mapping[str, Any],
    as_of: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if type(assessments_by_dimension) is not dict or set(assessments_by_dimension) != set(
        RISK_DIMENSIONS
    ):
        fail("I1_SHAPE_INVALID", "assessments_by_dimension must contain exactly four dimensions")
    admitted_evidence_keys = {
        _ref_key(canonical_content_ref(ref, label="context.evidence_refs"))
        for ref in context["evidence_refs"]
    }
    authorized = replay["i0_replay_inputs"]["observation_bundle"].get("authorized_evidence_refs")
    if type(authorized) is not list:
        fail("I1_SHAPE_INVALID", "Observation authorized source closure is malformed")
    authorized_source_keys = {
        _source_ref_key(_canonical_source_ref(ref, label="authorized_source_ref", as_of=as_of))
        for ref in authorized
    }

    normalized: dict[str, list[dict[str, Any]]] = {}
    dimension_rows: list[dict[str, Any]] = []
    for dimension in RISK_DIMENSION_ORDER:
        group = assessments_by_dimension[dimension]
        if type(group) is not dict or set(group) != _GROUP_FIELDS:
            fail("I1_SHAPE_INVALID", f"{dimension} assessment group shape is not closed")
        status = group["status"]
        if status not in ASSESSMENT_STATUSES:
            fail("I1_SHAPE_INVALID", f"{dimension} assessment status is invalid")
        raw_rows = group["assessments"]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            fail("I1_SHAPE_INVALID", f"{dimension}.assessments must be a sequence")
        if status == "UNAVAILABLE" and raw_rows:
            fail("I1_SHAPE_INVALID", f"{dimension} UNAVAILABLE must have no assessments")
        if status == "AVAILABLE" and not 1 <= len(raw_rows) <= MAX_ASSESSMENTS_PER_DIMENSION:
            fail("I1_SHAPE_INVALID", f"{dimension} AVAILABLE requires 1..16 assessments")
        rows = [
            _normalize_assessment(
                value,
                label=f"{dimension}.assessments[{index}]",
                as_of=as_of,
                admitted_evidence_keys=admitted_evidence_keys,
                authorized_source_keys=authorized_source_keys,
            )
            for index, value in enumerate(raw_rows)
        ]
        ids = [row["assessment_id"] for row in rows]
        if len(ids) != len(set(ids)):
            fail("I1_SHAPE_INVALID", f"{dimension} contains duplicate assessments")
        rows.sort(key=lambda row: row["assessment_id"].encode("ascii"))
        normalized[dimension] = rows
        severity = (
            None
            if status == "UNAVAILABLE"
            else canonical_decimal(
                max(Decimal(str(row["severity"])) for row in rows), label="dimension_severity"
            )
        )
        dimension_rows.append(
            {
                "assessment_refs": [row["assessment_id"] for row in rows],
                "dimension": dimension,
                "dimension_severity": severity,
                "status": status,
            }
        )
    return normalized, dimension_rows


def _triggers_veto(assessment: Mapping[str, Any], threshold: Decimal) -> bool:
    return all(
        (
            assessment["hard_veto_code"] is not None,
            Decimal(str(assessment["severity"])) >= threshold,
        )
    )


def assess_investment_risk(
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Normalize four risk dimensions and seal a read-only risk receipt."""

    cutoff = canonical_timestamp(as_of, label="as_of")
    context_row, replay = _validated_context(context, context_replay_closure)
    policy_row = validate_decision_policy(policy)
    if replay["policy"] != policy_row:
        fail("I1_REF_MISMATCH", "risk policy does not match the context replay closure")
    if context_row["timestamp"] != cutoff or context_row["as_of"] != cutoff:
        fail("I1_REF_MISMATCH", "risk as_of does not match its decision context")

    normalized, dimension_rows = _normalize_assessments(
        assessments_by_dimension,
        context=context_row,
        replay=replay,
        as_of=cutoff,
    )
    available_severities = [
        Decimal(str(row["dimension_severity"]))
        for row in dimension_rows
        if row["status"] == "AVAILABLE"
    ]
    overall_severity = (
        None
        if not available_severities
        else canonical_decimal(max(available_severities), label="overall_severity")
    )
    veto_threshold = Decimal(str(policy_row["hard_veto_severity"]))
    veto_codes = {
        str(assessment["hard_veto_code"])
        for rows in normalized.values()
        for assessment in rows
        if _triggers_veto(assessment, veto_threshold)
    }
    result = seal_artifact(
        version=RISK_RECEIPT_VERSION,
        identity_field="risk_receipt_id",
        timestamp_value=cutoff,
        payload={
            "context_ref": content_ref(context_row, identity_field="context_id"),
            "dimension_rows": dimension_rows,
            "hard_veto_codes": canonical_codes(
                list(veto_codes), label="hard_veto_codes", maximum=64
            ),
            "overall_severity": overall_severity,
            "policy_ref": content_ref(policy_row, identity_field="policy_id"),
            "unavailable_dimensions": [
                row["dimension"] for row in dimension_rows if row["status"] == "UNAVAILABLE"
            ],
        },
    )
    ensure_artifact_size(result)
    return result


def validate_risk_assessment_receipt(
    document: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Fully replay a risk receipt; a valid seal alone is never sufficient."""

    row = validate_closed_artifact(
        document,
        version=RISK_RECEIPT_VERSION,
        identity_field="risk_receipt_id",
        payload_fields=_RISK_PAYLOAD_FIELDS,
    )
    expected = assess_investment_risk(
        context=context,
        context_replay_closure=context_replay_closure,
        policy=policy,
        assessments_by_dimension=assessments_by_dimension,
        as_of=as_of,
    )
    if row != expected:
        fail("I1_REPLAY_MISMATCH", "risk receipt does not match its replay closure")
    return row


__all__ = [
    "RISK_DIMENSIONS",
    "assess_investment_risk",
    "validate_risk_assessment_receipt",
]
