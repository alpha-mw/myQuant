"""Immutable I1 decision-discipline chain.

The chain is separate from Investment Memory v1.  It records decision-process
milestones and evidence deltas, but performs no persistence and grants no
portfolio, broker, order, execution, or trade authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._core import (
    ZERO_SHA256,
    content_ref,
    sha256,
    sorted_exact_refs,
)
from .decision_engine import validate_investment_decision_receipt
from .models import (
    DISCIPLINE_ENTRY_VERSION,
    bounded_text,
    canonical_content_ref,
    canonical_timestamp,
    company_code,
    ensure_artifact_size,
    fail,
    sorted_content_refs,
)
from .receipts import seal_artifact, validate_closed_artifact

_ENTRY_PAYLOAD_FIELDS: Final = {
    "context_ref",
    "decision_ref",
    "event_type",
    "evidence_changes",
    "outcome_refs",
    "policy_ref",
    "previous_entry_sha256",
    "price_observation_refs",
    "stage",
    "status",
    "subject_id",
    "summary",
}
_DECISION_CLOSURE_FIELDS: Final = {
    "as_of",
    "assessments_by_dimension",
    "context",
    "context_replay_closure",
    "policy",
    "risk_receipt",
}
_HISTORY_VALUE_FIELDS: Final = {
    "decision_receipt",
    "decision_validation_closure",
}
_OUTCOME_VERSIONS: Final = {
    "myquant.v17.v4.forward-evaluation-receipt.v1",
    "myquant.v17.v4.forward-label.v1",
}
_FIRST: Final = ("BEFORE_DECISION", "DECISION_CREATED", "ACTIVE")
_AFTER_OUTCOME: Final = (
    "AFTER_OUTCOME",
    "DECISION_REVIEWED",
    "OUTCOME_AVAILABLE",
)
_CONFIRMED: Final = ("REVIEW", "THESIS_CONFIRMED", "CONFIRMED")
_FAILED: Final = ("REVIEW", "THESIS_FAILED", "FAILED")
_LEARNED: Final = ("REVIEW", "LESSON_LEARNED", "LEARNED")


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value["artifact_id"]),
        str(value["artifact_version"]),
        str(value["byte_sha256"]),
        str(value["semantic_sha256"]),
    )


def _normalize_content_refs(
    values: Sequence[Mapping[str, Any]], *, label: str
) -> list[dict[str, str]]:
    rows = sorted_content_refs(values, label=label)
    return [dict(row) for row in rows]


def _normalize_exact_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_versions: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    try:
        rows = sorted_exact_refs(
            values,
            label=label,
            expected_versions=expected_versions,
        )
    except Exception:
        fail("I1_SHAPE_INVALID", f"{label} is not a canonical exact-ref set")
    return rows


def _replay_decision(
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        type(decision_validation_closure) is not dict
        or set(decision_validation_closure) != _DECISION_CLOSURE_FIELDS
    ):
        fail("I1_SHAPE_INVALID", "decision_validation_closure shape is not closed")
    closure = dict(decision_validation_closure)
    decision = validate_investment_decision_receipt(
        decision_receipt,
        context=closure["context"],
        context_replay_closure=closure["context_replay_closure"],
        policy=closure["policy"],
        risk_receipt=closure["risk_receipt"],
        assessments_by_dimension=closure["assessments_by_dimension"],
        as_of=closure["as_of"],
    )
    return decision, closure


def _decision_binding(
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
) -> dict[str, Any]:
    decision, closure = _replay_decision(decision_receipt, decision_validation_closure)
    context = closure["context"]
    expected_context_ref = content_ref(context, identity_field="context_id")
    if decision["context_ref"] != expected_context_ref:
        fail("I1_REF_MISMATCH", "decision does not bind its replayed context")
    if decision["policy_ref"] != context["policy_ref"]:
        fail("I1_REF_MISMATCH", "decision and context policy refs differ")
    return {"closure": closure, "context": context, "decision": decision}


def _context_change_refs(context: Mapping[str, Any]) -> list[dict[str, str]]:
    values: list[Mapping[str, Any]] = list(context["evidence_refs"])
    r22_ref = context.get("r22_hypothesis_evaluation_ref")
    if r22_ref is not None:
        values.append(r22_ref)
    return _normalize_content_refs(values, label="context change refs")


def _evidence_changes(
    previous_context: Mapping[str, Any] | None,
    current_context: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    old_rows = [] if previous_context is None else _context_change_refs(previous_context)
    new_rows = _context_change_refs(current_context)
    old = {_ref_key(row): row for row in old_rows}
    new = {_ref_key(row): row for row in new_rows}
    return {
        "added_refs": [new[key] for key in sorted(set(new) - set(old))],
        "removed_refs": [old[key] for key in sorted(set(old) - set(new))],
    }


def _triple(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["stage"]), str(row["event_type"]), str(row["status"]))


def _assert_transition(
    previous_entry: Mapping[str, Any] | None,
    *,
    stage: Any,
    event_type: Any,
    status: Any,
) -> tuple[str, str, str]:
    if not all(type(value) is str for value in (stage, event_type, status)):
        fail("I1_DISCIPLINE_TRANSITION_INVALID", "discipline transition is malformed")
    current = (stage, event_type, status)
    if previous_entry is None:
        allowed = {_FIRST}
    elif _triple(previous_entry) == _FIRST:
        allowed = {_AFTER_OUTCOME}
    elif _triple(previous_entry) == _AFTER_OUTCOME:
        allowed = {_CONFIRMED, _FAILED}
    elif _triple(previous_entry) in {_CONFIRMED, _FAILED}:
        allowed = {_LEARNED}
    else:
        allowed = set()
    if current not in allowed:
        fail(
            "I1_DISCIPLINE_TRANSITION_INVALID",
            "decision-discipline transition is not allowlisted",
        )
    return current


def _assert_decision_continuity(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    changes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if previous is None:
        return
    previous_decision = previous["decision"]
    current_decision = current["decision"]
    previous_context = previous["context"]
    current_context = current["context"]
    if previous_context["company_code"] != current_context["company_code"]:
        fail("I1_DISCIPLINE_TRANSITION_INVALID", "discipline subject changed")
    if previous_decision["policy_ref"] != current_decision["policy_ref"]:
        fail("I1_DISCIPLINE_TRANSITION_INVALID", "discipline policy changed")

    has_changes = bool(changes["added_refs"] or changes["removed_refs"])
    if (
        content_ref(previous_context, identity_field="context_id")
        != content_ref(current_context, identity_field="context_id")
        and not has_changes
    ):
        fail(
            "I1_DISCIPLINE_TRANSITION_INVALID",
            "context changed without an evidence or R2.2 evaluation change",
        )
    guarded = ("risk_ref", "state", "research_confidence", "bayesian_posterior")
    if any(previous_decision[field] != current_decision[field] for field in guarded):
        if not has_changes:
            fail(
                "I1_DISCIPLINE_TRANSITION_INVALID",
                "decision changed without an evidence or R2.2 evaluation change",
            )


def _source_refs_at(
    *,
    outcome_refs: Sequence[Mapping[str, Any]],
    price_observation_refs: Sequence[Mapping[str, Any]],
    event_at: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    outcomes = _normalize_exact_refs(
        outcome_refs,
        label="outcome_refs",
        expected_versions=sorted(_OUTCOME_VERSIONS),
    )
    prices = _normalize_exact_refs(
        price_observation_refs,
        label="price_observation_refs",
    )
    if any(row["cutoff"] > event_at for row in outcomes + prices):
        fail("I1_FUTURE_INPUT", "discipline entry contains a future source ref")
    return outcomes, prices


def _build_entry(
    *,
    previous_entry_sha256: str,
    binding: Mapping[str, Any],
    previous_binding: Mapping[str, Any] | None,
    stage: str,
    event_type: str,
    status: str,
    summary: str,
    event_at: str,
    outcome_refs: Sequence[Mapping[str, Any]],
    price_observation_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decision = binding["decision"]
    context = binding["context"]
    event_timestamp = canonical_timestamp(event_at, label="event_at")
    if event_timestamp < decision["timestamp"] or event_timestamp < context["timestamp"]:
        fail("I1_FUTURE_INPUT", "discipline entry predates its decision or context")
    outcomes, prices = _source_refs_at(
        outcome_refs=outcome_refs,
        price_observation_refs=price_observation_refs,
        event_at=event_timestamp,
    )
    if (stage, event_type, status) == _AFTER_OUTCOME and not outcomes:
        fail(
            "I1_DISCIPLINE_TRANSITION_INVALID",
            "OUTCOME_AVAILABLE requires at least one exact outcome ref",
        )
    changes = _evidence_changes(
        None if previous_binding is None else previous_binding["context"],
        context,
    )
    _assert_decision_continuity(previous_binding, binding, changes=changes)
    result = seal_artifact(
        version=DISCIPLINE_ENTRY_VERSION,
        identity_field="entry_id",
        timestamp_value=event_timestamp,
        payload={
            "context_ref": content_ref(context, identity_field="context_id"),
            "decision_ref": content_ref(decision, identity_field="decision_receipt_id"),
            "event_type": event_type,
            "evidence_changes": changes,
            "outcome_refs": outcomes,
            "policy_ref": canonical_content_ref(
                decision["policy_ref"], label="decision.policy_ref"
            ),
            "previous_entry_sha256": previous_entry_sha256,
            "price_observation_refs": prices,
            "stage": stage,
            "status": status,
            "subject_id": str(context["company_code"]),
            "summary": bounded_text(summary, label="summary"),
        },
    )
    ensure_artifact_size(result)
    return result


def _validate_structural_chain(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        fail("I1_SHAPE_INVALID", "decision discipline chain must be a sequence")
    result: list[dict[str, Any]] = []
    previous_hash = ZERO_SHA256
    previous_timestamp: str | None = None
    previous_entry: dict[str, Any] | None = None
    subject: str | None = None
    policy_ref: dict[str, str] | None = None
    for index, document in enumerate(entries):
        row = validate_closed_artifact(
            document,
            version=DISCIPLINE_ENTRY_VERSION,
            identity_field="entry_id",
            payload_fields=_ENTRY_PAYLOAD_FIELDS,
        )
        try:
            previous_entry_sha256 = sha256(
                row["previous_entry_sha256"],
                label=f"entries[{index}].previous_entry_sha256",
            )
        except Exception:
            fail(
                "I1_SHAPE_INVALID",
                f"entries[{index}].previous_entry_sha256 is not canonical",
            )
        if row["previous_entry_sha256"] != previous_hash:
            fail("I1_REPLAY_MISMATCH", f"discipline entry {index} breaks the hash chain")
        event_at = canonical_timestamp(row["timestamp"], label=f"entries[{index}].timestamp")
        if previous_timestamp is not None and event_at <= previous_timestamp:
            fail(
                "I1_DISCIPLINE_TRANSITION_INVALID",
                "discipline timestamps must increase strictly",
            )
        _assert_transition(
            previous_entry,
            stage=row["stage"],
            event_type=row["event_type"],
            status=row["status"],
        )
        current_subject = str(row["subject_id"])
        company_code(current_subject, label=f"entries[{index}].subject_id")
        current_policy = canonical_content_ref(
            row["policy_ref"], label=f"entries[{index}].policy_ref"
        )
        current_context = canonical_content_ref(
            row["context_ref"], label=f"entries[{index}].context_ref"
        )
        current_decision = canonical_content_ref(
            row["decision_ref"], label=f"entries[{index}].decision_ref"
        )
        raw_changes = row["evidence_changes"]
        if type(raw_changes) is not dict or set(raw_changes) != {
            "added_refs",
            "removed_refs",
        }:
            fail("I1_SHAPE_INVALID", f"entries[{index}].evidence_changes is malformed")
        changes = {
            "added_refs": _normalize_content_refs(
                raw_changes["added_refs"],
                label=f"entries[{index}].evidence_changes.added_refs",
            ),
            "removed_refs": _normalize_content_refs(
                raw_changes["removed_refs"],
                label=f"entries[{index}].evidence_changes.removed_refs",
            ),
        }
        if set(map(_ref_key, changes["added_refs"])) & set(map(_ref_key, changes["removed_refs"])):
            fail("I1_SHAPE_INVALID", "a discipline ref cannot be both added and removed")
        outcomes, prices = _source_refs_at(
            outcome_refs=row["outcome_refs"],
            price_observation_refs=row["price_observation_refs"],
            event_at=event_at,
        )
        replayed = seal_artifact(
            version=DISCIPLINE_ENTRY_VERSION,
            identity_field="entry_id",
            timestamp_value=event_at,
            payload={
                "context_ref": current_context,
                "decision_ref": current_decision,
                "event_type": str(row["event_type"]),
                "evidence_changes": changes,
                "outcome_refs": outcomes,
                "policy_ref": current_policy,
                "previous_entry_sha256": previous_entry_sha256,
                "price_observation_refs": prices,
                "stage": str(row["stage"]),
                "status": str(row["status"]),
                "subject_id": current_subject,
                "summary": bounded_text(row["summary"], label=f"entries[{index}].summary"),
            },
        )
        if replayed != row:
            fail("I1_REPLAY_MISMATCH", f"discipline entry {index} is not canonical")
        if subject is not None and current_subject != subject:
            fail("I1_DISCIPLINE_TRANSITION_INVALID", "discipline subject changed")
        if policy_ref is not None and current_policy != policy_ref:
            fail("I1_DISCIPLINE_TRANSITION_INVALID", "discipline policy changed")
        subject = current_subject
        policy_ref = current_policy
        previous_hash = str(row["semantic_sha256"])
        previous_timestamp = event_at
        previous_entry = row
        result.append(row)
    return tuple(result)


def append_decision_discipline(
    entries: Sequence[Mapping[str, Any]],
    *,
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
    previous_decision_receipt: Mapping[str, Any] | None,
    previous_decision_validation_closure: Mapping[str, Any] | None,
    stage: str,
    event_type: str,
    status: str,
    summary: str,
    event_at: str,
    outcome_refs: Sequence[Mapping[str, Any]] = (),
    price_observation_refs: Sequence[Mapping[str, Any]] = (),
    expected_tip: str = ZERO_SHA256,
) -> tuple[dict[str, Any], ...]:
    """Return a newly extended discipline chain without mutating the input."""

    chain = _validate_structural_chain(entries)
    try:
        expected = sha256(expected_tip, label="expected_tip")
    except Exception:
        fail("I1_SHAPE_INVALID", "expected_tip is not a SHA-256")
    actual_tip = ZERO_SHA256 if not chain else str(chain[-1]["semantic_sha256"])
    if expected != actual_tip:
        fail("I1_REPLAY_MISMATCH", "discipline tail deletion or substitution detected")

    current_binding = _decision_binding(decision_receipt, decision_validation_closure)
    if chain:
        if previous_decision_receipt is None or previous_decision_validation_closure is None:
            fail("I1_SHAPE_INVALID", "non-root append requires the previous decision closure")
        previous_binding = _decision_binding(
            previous_decision_receipt,
            previous_decision_validation_closure,
        )
        if chain[-1]["decision_ref"] != content_ref(
            previous_binding["decision"], identity_field="decision_receipt_id"
        ):
            fail("I1_REF_MISMATCH", "previous decision does not bind the chain tip")
    else:
        if (
            previous_decision_receipt is not None
            or previous_decision_validation_closure is not None
        ):
            fail("I1_SHAPE_INVALID", "root append cannot provide a previous decision")
        previous_binding = None

    _assert_transition(
        None if not chain else chain[-1],
        stage=stage,
        event_type=event_type,
        status=status,
    )
    if chain and canonical_timestamp(event_at, label="event_at") <= chain[-1]["timestamp"]:
        fail(
            "I1_DISCIPLINE_TRANSITION_INVALID",
            "discipline timestamps must increase strictly",
        )
    entry = _build_entry(
        previous_entry_sha256=actual_tip,
        binding=current_binding,
        previous_binding=previous_binding,
        stage=stage,
        event_type=event_type,
        status=status,
        summary=summary,
        event_at=event_at,
        outcome_refs=outcome_refs,
        price_observation_refs=price_observation_refs,
    )
    return chain + (entry,)


def validate_decision_discipline_chain(
    entries: Sequence[Mapping[str, Any]],
    *,
    decision_history_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Fully replay every decision and every evidence delta in the chain."""

    chain = _validate_structural_chain(entries)
    if type(decision_history_by_id) is not dict:
        fail("I1_SHAPE_INVALID", "decision_history_by_id must be an object")
    referenced_ids = {str(row["decision_ref"]["artifact_id"]) for row in chain}
    if set(decision_history_by_id) != referenced_ids:
        fail("I1_SHAPE_INVALID", "decision history does not exactly cover the chain")

    bindings: dict[str, dict[str, Any]] = {}
    for decision_id, history in decision_history_by_id.items():
        try:
            sha256(decision_id, label="decision_history_by_id key")
        except Exception:
            fail("I1_SHAPE_INVALID", "decision history key is not canonical")
        if type(history) is not dict or set(history) != _HISTORY_VALUE_FIELDS:
            fail("I1_SHAPE_INVALID", "decision history value shape is not closed")
        binding = _decision_binding(
            history["decision_receipt"], history["decision_validation_closure"]
        )
        if binding["decision"]["decision_receipt_id"] != decision_id:
            fail("I1_REF_MISMATCH", "decision history key does not match its receipt")
        bindings[decision_id] = binding

    previous_binding: dict[str, Any] | None = None
    for index, row in enumerate(chain):
        binding = bindings[str(row["decision_ref"]["artifact_id"])]
        decision = binding["decision"]
        context = binding["context"]
        if row["decision_ref"] != content_ref(decision, identity_field="decision_receipt_id"):
            fail("I1_REF_MISMATCH", f"discipline entry {index} decision ref mismatches")
        if row["context_ref"] != content_ref(context, identity_field="context_id"):
            fail("I1_REF_MISMATCH", f"discipline entry {index} context ref mismatches")
        if row["subject_id"] != context["company_code"]:
            fail("I1_REF_MISMATCH", f"discipline entry {index} subject mismatches")
        if row["policy_ref"] != decision["policy_ref"]:
            fail("I1_REF_MISMATCH", f"discipline entry {index} policy ref mismatches")

        changes = _evidence_changes(
            None if previous_binding is None else previous_binding["context"], context
        )
        _assert_decision_continuity(previous_binding, binding, changes=changes)
        if row["evidence_changes"] != changes:
            fail(
                "I1_REPLAY_MISMATCH",
                f"discipline entry {index} evidence change-set mismatches",
            )
        if row["timestamp"] < decision["timestamp"] or row["timestamp"] < context["timestamp"]:
            fail("I1_FUTURE_INPUT", f"discipline entry {index} predates its decision")
        outcomes, prices = _source_refs_at(
            outcome_refs=row["outcome_refs"],
            price_observation_refs=row["price_observation_refs"],
            event_at=row["timestamp"],
        )
        expected = _build_entry(
            previous_entry_sha256=(
                ZERO_SHA256 if index == 0 else str(chain[index - 1]["semantic_sha256"])
            ),
            binding=binding,
            previous_binding=previous_binding,
            stage=str(row["stage"]),
            event_type=str(row["event_type"]),
            status=str(row["status"]),
            summary=str(row["summary"]),
            event_at=str(row["timestamp"]),
            outcome_refs=outcomes,
            price_observation_refs=prices,
        )
        if expected != row:
            fail("I1_REPLAY_MISMATCH", f"discipline entry {index} replay mismatch")
        previous_binding = binding
    return chain


__all__ = ["append_decision_discipline", "validate_decision_discipline_chain"]
