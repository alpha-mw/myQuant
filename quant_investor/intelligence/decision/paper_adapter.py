"""Closed external-review seam for I1 paper-intake proposals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol

from .._core import content_ref
from .decision_engine import validate_investment_decision_receipt
from .models import (
    PAPER_INTAKE_PROPOSAL_VERSION,
    canonical_timestamp,
    ensure_artifact_size,
    fail,
)
from .receipts import seal_artifact, validate_closed_artifact

_PAPER_PAYLOAD_FIELDS: Final = {"decision_ref", "status"}
_DECISION_CLOSURE_FIELDS: Final = {
    "as_of",
    "assessments_by_dimension",
    "context",
    "context_replay_closure",
    "policy",
    "risk_receipt",
}


class PaperPortfolioAdapter(Protocol):
    """Interface only; I1 never implements, instantiates, or calls this seam."""

    def submit_for_external_review(  # noqa: E704
        self,
        proposal: Mapping[str, Any],
        /,
    ) -> None: ...


def _replay_decision(
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(decision_validation_closure) is not dict
        or set(decision_validation_closure) != _DECISION_CLOSURE_FIELDS
    ):
        fail("I1_SHAPE_INVALID", "decision_validation_closure shape is not closed")
    closure = dict(decision_validation_closure)
    return validate_investment_decision_receipt(
        decision_receipt,
        context=closure["context"],
        context_replay_closure=closure["context_replay_closure"],
        policy=closure["policy"],
        risk_receipt=closure["risk_receipt"],
        assessments_by_dimension=closure["assessments_by_dimension"],
        as_of=closure["as_of"],
    )


def build_paper_intake_proposal(
    *,
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
    proposed_at: str,
) -> dict[str, Any]:
    """Build an external-review proposal, never a portfolio or order action."""

    decision = _replay_decision(decision_receipt, decision_validation_closure)
    proposed = canonical_timestamp(proposed_at, label="proposed_at")
    if proposed < decision["timestamp"]:
        fail("I1_FUTURE_INPUT", "paper proposal predates its decision")
    if decision["state"] != "PAPER_CANDIDATE":
        fail("I1_AUTHORITY_OPEN", "only PAPER_CANDIDATE may enter external paper review")
    result = seal_artifact(
        version=PAPER_INTAKE_PROPOSAL_VERSION,
        identity_field="proposal_id",
        timestamp_value=proposed,
        payload={
            "decision_ref": content_ref(decision, identity_field="decision_receipt_id"),
            "status": "PENDING_EXTERNAL_REVIEW",
        },
    )
    ensure_artifact_size(result)
    return result


def validate_paper_intake_proposal(
    document: Mapping[str, Any],
    *,
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
    proposed_at: str,
) -> dict[str, Any]:
    row = validate_closed_artifact(
        document,
        version=PAPER_INTAKE_PROPOSAL_VERSION,
        identity_field="proposal_id",
        payload_fields=_PAPER_PAYLOAD_FIELDS,
    )
    expected = build_paper_intake_proposal(
        decision_receipt=decision_receipt,
        decision_validation_closure=decision_validation_closure,
        proposed_at=proposed_at,
    )
    if row != expected:
        fail("I1_REPLAY_MISMATCH", "paper proposal does not match its replay closure")
    return row


__all__ = [
    "PaperPortfolioAdapter",
    "build_paper_intake_proposal",
    "validate_paper_intake_proposal",
]
