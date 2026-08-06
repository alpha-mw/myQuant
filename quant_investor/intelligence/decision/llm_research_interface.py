"""Decision-facing validation for externally supplied, source-bound AI drafts.

This module never calls a model or provider.  It narrows the more general I0
AI-draft envelope to the three projections which may be copied into an I1
research context or memo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._core import (
    IntelligenceContractError,
    assert_no_authority,
    exact_ref,
    timestamp,
    validate_content_addressed,
)
from ..evidence.ai import AI_DRAFT_VERSION, build_ai_draft
from .models import DecisionContractError, bounded_text, fail

DECISION_AI_DRAFT_KINDS: Final = {
    "CONTRARY_EVIDENCE_DRAFT",
    "EXTRACTION",
    "SUMMARY",
}

_PAYLOAD_FIELDS: Final = {
    "CONTRARY_EVIDENCE_DRAFT": {"contrary_points"},
    "EXTRACTION": {"facts"},
    "SUMMARY": {"summary"},
}

_DRAFT_FIELDS: Final = {
    "authority",
    "confidence",
    "draft_id",
    "generated_at",
    "kind",
    "payload",
    "production",
    "research_only",
    "semantic_sha256",
    "source_refs",
    "version",
}


def _exact_ref_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value[key]) for key in sorted(value))


def _payload(kind: str, value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PAYLOAD_FIELDS[kind]:
        fail("I1_SHAPE_INVALID", f"AI draft {kind} payload shape is invalid")
    if kind == "SUMMARY":
        return {"summary": bounded_text(value["summary"], label="AI draft summary")}
    field = "facts" if kind == "EXTRACTION" else "contrary_points"
    rows = value[field]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        fail("I1_SHAPE_INVALID", f"AI draft {field} must be a sequence")
    if not 1 <= len(rows) <= 64:
        fail("I1_SHAPE_INVALID", f"AI draft {field} cardinality is invalid")
    normalized = [
        bounded_text(item, label=f"AI draft {field}[{index}]") for index, item in enumerate(rows)
    ]
    return {field: normalized}


def validate_decision_ai_draft(
    document: Mapping[str, Any],
    *,
    as_of: str,
    authorized_source_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay and validate one allowlisted decision-facing projection."""

    try:
        cutoff = timestamp(as_of, label="as_of")
        row = validate_content_addressed(document, identity_field="draft_id")
        if set(row) != _DRAFT_FIELDS or row.get("version") != AI_DRAFT_VERSION:
            fail(
                "I1_SHAPE_INVALID",
                "AI draft shape/version is not decision-facing",
            )
        assert_no_authority(row)
        kind = row.get("kind")
        if kind not in DECISION_AI_DRAFT_KINDS:
            fail("I1_SHAPE_INVALID", "AI draft kind is not decision-facing")
        payload = _payload(str(kind), row.get("payload"))
        generated_at = timestamp(row.get("generated_at"), label="AI draft.generated_at")
        if generated_at > cutoff:
            fail("I1_FUTURE_INPUT", "AI draft was generated after context as_of")
        source_refs = row.get("source_refs")
        if (
            isinstance(source_refs, (str, bytes))
            or not isinstance(source_refs, Sequence)
            or not source_refs
        ):
            fail("I1_SHAPE_INVALID", "AI draft source refs are required")
        normalized_refs = [
            exact_ref(value, label=f"AI draft.source_refs[{index}]")
            for index, value in enumerate(source_refs)
        ]
        keys = [_exact_ref_key(value) for value in normalized_refs]
        if len(keys) != len(set(keys)):
            fail("I1_SHAPE_INVALID", "AI draft source refs contain duplicates")
        authorized = {
            _exact_ref_key(exact_ref(value, label=f"authorized_source_refs[{index}]"))
            for index, value in enumerate(authorized_source_refs)
        }
        if any(key not in authorized for key in keys):
            fail(
                "I1_REF_MISMATCH",
                "AI draft source is outside Observation closure",
            )
        if any(ref["cutoff"] > generated_at for ref in normalized_refs):
            fail("I1_FUTURE_INPUT", "AI draft predates an admitted source")
        expected = build_ai_draft(
            kind=str(kind),
            payload=payload,
            source_refs=normalized_refs,
            generated_at=generated_at,
            confidence=row.get("confidence"),
        )
        if expected != row:
            fail("I1_REPLAY_MISMATCH", "AI draft replay mismatch")
        return row
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_SHAPE_INVALID", str(exc)) from exc


def project_decision_ai_draft(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the allowlisted narrative projection from a validated draft."""

    kind = document.get("kind")
    if kind not in DECISION_AI_DRAFT_KINDS:
        fail("I1_SHAPE_INVALID", "AI draft kind is not decision-facing")
    return {
        "kind": kind,
        "payload": _payload(str(kind), document.get("payload")),
    }


__all__ = [
    "DECISION_AI_DRAFT_KINDS",
    "project_decision_ai_draft",
    "validate_decision_ai_draft",
]
