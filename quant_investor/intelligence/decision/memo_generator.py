"""Deterministic investment-memo projection for the I1 decision layer.

This module deliberately contains no prose generation.  Every memo value is a
projection of an already validated hypothesis, evidence item, context note, risk
assessment, or decision receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from .._core import content_ref
from .decision_engine import validate_investment_decision_receipt
from .evidence_collector import (
    validate_context_replay_closure,
    validate_investment_decision_context,
)
from .models import (
    MEMO_VERSION,
    bounded_text,
    canonical_content_ref,
    canonical_decimal,
    canonical_timestamp,
    ensure_artifact_size,
    fail,
    validate_decision_policy,
)
from .receipts import seal_artifact, validate_closed_artifact
from .risk_assessor import validate_risk_assessment_receipt

_MEMO_PAYLOAD_FIELDS: Final = {
    "bayesian_posterior",
    "company_code",
    "company_display_name",
    "context_ref",
    "contrary_evidence",
    "date",
    "decision_ref",
    "decision_state",
    "falsification_conditions",
    "industry_context",
    "investment_thesis",
    "research_confidence",
    "review_due_at",
    "risk_factors",
    "risk_ref",
    "supporting_evidence",
    "valuation_context",
    "why_invest",
    "why_now",
}


def _replay_inputs(closure: Mapping[str, Any]) -> dict[str, Any]:
    return validate_context_replay_closure(closure)


def _validated_context(
    context: Mapping[str, Any], closure: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = _replay_inputs(closure)
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


def _validate_receipt_closure(
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    risk_receipt: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    as_of: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cutoff = canonical_timestamp(as_of, label="as_of")
    context_row, replay = _validated_context(context, context_replay_closure)
    policy_row = validate_decision_policy(policy)
    if replay["policy"] != policy_row:
        fail("I1_REF_MISMATCH", "memo policy does not match the context closure")
    risk_row = validate_risk_assessment_receipt(
        risk_receipt,
        context=context_row,
        context_replay_closure=replay,
        policy=policy_row,
        assessments_by_dimension=assessments_by_dimension,
        as_of=cutoff,
    )
    decision_row = validate_investment_decision_receipt(
        decision_receipt,
        context=context_row,
        context_replay_closure=replay,
        policy=policy_row,
        risk_receipt=risk_row,
        assessments_by_dimension=assessments_by_dimension,
        as_of=cutoff,
    )
    if context_row["timestamp"] != cutoff:
        fail("I1_REF_MISMATCH", "memo as_of does not match its decision context")
    return context_row, replay, risk_row, decision_row


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value["artifact_id"]),
        str(value["artifact_version"]),
        str(value["byte_sha256"]),
        str(value["semantic_sha256"]),
    )


def _project_evidence(
    refs: Sequence[Mapping[str, Any]],
    *,
    evidence_by_ref: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        fail("I1_SHAPE_INVALID", f"{label} must be non-empty")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, raw_ref in enumerate(refs):
        ref = canonical_content_ref(raw_ref, label=f"{label}[{index}]")
        key = _ref_key(ref)
        if key in seen:
            fail("I1_SHAPE_INVALID", f"{label} contains duplicate references")
        seen.add(key)
        evidence = evidence_by_ref.get(key)
        if evidence is None:
            fail("I1_REF_MISMATCH", f"{label} references evidence outside the context")
        rows.append(
            {
                "direction": str(evidence["direction"]),
                "evidence_ref": ref,
                "reason": bounded_text(evidence["reason"], label=f"{label}.reason"),
                "source_type": str(evidence["source_type"]),
                "strength": str(evidence["strength"]),
            }
        )
    return sorted(rows, key=lambda row: _ref_key(row["evidence_ref"]))


def _project_notes(
    *, context: Mapping[str, Any], closure: Mapping[str, Any]
) -> tuple[str | None, str | None, str | None, str | None]:
    admitted = {
        _ref_key(canonical_content_ref(ref, label="context.note_refs"))
        for ref in context["note_refs"]
    }
    selected: dict[str, list[tuple[tuple[str, str, str, str], str]]] = {}
    for note in closure["context_notes"]:
        ref = content_ref(note, identity_field="note_id")
        key = _ref_key(ref)
        if key not in admitted:
            continue
        selected.setdefault(str(note["kind"]), []).append(
            (key, bounded_text(note["text"], label="context_note.text"))
        )
    for rows in selected.values():
        rows.sort(key=lambda item: item[0])
    duplicate_kinds = sorted(
        (kind for kind, rows in selected.items() if len(rows) > 1),
        key=lambda value: value.encode("ascii"),
    )
    if duplicate_kinds:
        fail(
            "I1_SHAPE_INVALID",
            f"memo has multiple {duplicate_kinds[0]} context notes",
        )

    def one(kind: str) -> str | None:
        rows = selected.get(kind, [])
        return None if not rows else rows[0][1]

    return (
        one("COMPANY_DISPLAY_NAME"),
        one("WHY_NOW"),
        one("INDUSTRY_CONTEXT"),
        one("VALUATION_CONTEXT"),
    )


def _project_risks(
    assessments_by_dimension: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, group in assessments_by_dimension.items():
        assessments = group.get("assessments", []) if type(group) is dict else []
        for assessment in assessments:
            rows.append(
                {
                    "assessment_id": str(assessment["assessment_id"]),
                    "dimension": str(dimension),
                    "hard_veto_code": assessment["hard_veto_code"],
                    "kind": str(assessment["kind"]),
                    "reason": bounded_text(assessment["reason"], label="risk_assessment.reason"),
                    "severity": canonical_decimal(
                        assessment["severity"],
                        label="risk_assessment.severity",
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["assessment_id"].encode("ascii"),
            row["dimension"].encode("ascii"),
        ),
    )


def build_investment_memo(
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    risk_receipt: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Project a closed memo from validated decision inputs without generation."""

    context_row, replay, risk_row, decision_row = _validate_receipt_closure(
        context=context,
        context_replay_closure=context_replay_closure,
        policy=policy,
        risk_receipt=risk_receipt,
        assessments_by_dimension=assessments_by_dimension,
        decision_receipt=decision_receipt,
        as_of=as_of,
    )
    i0 = replay["i0_replay_inputs"]
    hypothesis_ref = canonical_content_ref(
        context_row["hypothesis_ref"], label="context.hypothesis_ref"
    )
    hypotheses = [
        row
        for row in i0["hypotheses"]
        if content_ref(row, identity_field="hypothesis_id") == hypothesis_ref
    ]
    if len(hypotheses) != 1:
        fail("I1_REF_MISMATCH", "memo requires exactly one bound hypothesis")
    hypothesis = hypotheses[0]

    admitted_evidence = {
        _ref_key(canonical_content_ref(ref, label="context.evidence_refs"))
        for ref in context_row["evidence_refs"]
    }
    evidence_by_ref: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for evidence in i0["evidence"]:
        ref = content_ref(evidence, identity_field="evidence_id")
        key = _ref_key(ref)
        if key in admitted_evidence:
            evidence_by_ref[key] = evidence

    supporting = _project_evidence(
        hypothesis["supporting_evidence_refs"],
        evidence_by_ref=evidence_by_ref,
        label="supporting_evidence",
    )
    contrary = _project_evidence(
        hypothesis["contrary_evidence_refs"],
        evidence_by_ref=evidence_by_ref,
        label="contrary_evidence",
    )
    display_name, why_now, industry, valuation = _project_notes(context=context_row, closure=replay)

    result = seal_artifact(
        version=MEMO_VERSION,
        identity_field="memo_id",
        timestamp_value=as_of,
        payload={
            "bayesian_posterior": str(decision_row["bayesian_posterior"]),
            "company_code": str(context_row["company_code"]),
            "company_display_name": display_name,
            "context_ref": content_ref(context_row, identity_field="context_id"),
            "contrary_evidence": contrary,
            "date": str(as_of)[:10],
            "decision_ref": content_ref(decision_row, identity_field="decision_receipt_id"),
            "decision_state": str(decision_row["state"]),
            "falsification_conditions": deepcopy(hypothesis["falsification_conditions"]),
            "industry_context": industry,
            "investment_thesis": bounded_text(hypothesis["thesis"], label="hypothesis.thesis"),
            "research_confidence": str(decision_row["research_confidence"]),
            "review_due_at": str(context_row["review_due_at"]),
            "risk_factors": _project_risks(assessments_by_dimension),
            "risk_ref": content_ref(risk_row, identity_field="risk_receipt_id"),
            "supporting_evidence": supporting,
            "valuation_context": valuation,
            "why_invest": bounded_text(
                hypothesis["why_it_may_be_true"],
                label="hypothesis.why_it_may_be_true",
            ),
            "why_now": why_now,
        },
    )
    ensure_artifact_size(result)
    return result


def validate_investment_memo(
    document: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    context_replay_closure: Mapping[str, Any],
    policy: Mapping[str, Any],
    risk_receipt: Mapping[str, Any],
    assessments_by_dimension: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Replay the full memo closure and reject even correctly resealed forgeries."""

    row = validate_closed_artifact(
        document,
        version=MEMO_VERSION,
        identity_field="memo_id",
        payload_fields=_MEMO_PAYLOAD_FIELDS,
    )
    expected = build_investment_memo(
        context=context,
        context_replay_closure=context_replay_closure,
        policy=policy,
        risk_receipt=risk_receipt,
        assessments_by_dimension=assessments_by_dimension,
        decision_receipt=decision_receipt,
        as_of=as_of,
    )
    if row != expected:
        fail("I1_REPLAY_MISMATCH", "investment memo does not match its replay closure")
    return row


__all__ = ["build_investment_memo", "validate_investment_memo"]
