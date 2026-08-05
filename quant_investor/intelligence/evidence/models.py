"""Evidence records accepted by the deterministic intelligence engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._core import (
    IntelligenceContractError,
    decimal_text,
    decimal_value,
    exact_ref,
    require_no_future,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)

EVIDENCE_VERSION: Final = "myquant.v17.research-intelligence.evidence.v1"
EVIDENCE_SOURCES: Final = {
    "FUNDAMENTAL",
    "FORWARD_EVALUATION",
    "INDUSTRY",
    "QUANT",
    "REGIME",
    "THEME",
}
EVIDENCE_DIRECTIONS: Final = {"CONTRARY", "NEGATIVE", "NEUTRAL", "POSITIVE"}


def build_evidence(
    *,
    source_type: str,
    direction: str,
    likelihood_ratio: Any,
    strength: Any,
    reason: str,
    observed_at: str,
    available_at: str,
    source_ref: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one source-bound evidence item without inference authority."""

    if source_type not in EVIDENCE_SOURCES:
        raise IntelligenceContractError("evidence source type is not allowlisted")
    if direction not in EVIDENCE_DIRECTIONS:
        raise IntelligenceContractError("evidence direction is not allowlisted")
    ratio = decimal_value(
        likelihood_ratio,
        label="likelihood_ratio",
        minimum=Decimal("0"),
        minimum_exclusive=True,
    )
    evidence_strength = decimal_value(
        strength,
        label="strength",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if direction == "POSITIVE" and ratio <= 1:
        raise IntelligenceContractError("positive evidence requires likelihood_ratio > 1")
    if direction in {"NEGATIVE", "CONTRARY"} and ratio >= 1:
        raise IntelligenceContractError("negative evidence requires likelihood_ratio < 1")
    if direction == "NEUTRAL" and ratio != 1:
        raise IntelligenceContractError("neutral evidence requires likelihood_ratio = 1")
    if type(reason) is not str or not reason.strip():
        raise IntelligenceContractError("evidence reason is required")
    observed = timestamp(observed_at, label="observed_at")
    available = timestamp(available_at, label="available_at")
    if observed > available:
        raise IntelligenceContractError("evidence cannot be available before it is observed")
    reference = exact_ref(source_ref, label="source_ref")
    if reference["cutoff"] > available:
        raise IntelligenceContractError("evidence source is not available at available_at")
    if payload is not None and type(payload) is not dict:
        raise IntelligenceContractError("evidence payload must be an object")
    return seal_content_addressed(
        {
            "available_at": available,
            "direction": direction,
            "likelihood_ratio": decimal_text(ratio),
            "observed_at": observed,
            "payload": {} if payload is None else dict(payload),
            "reason": reason.strip(),
            "source_ref": reference,
            "source_type": source_type,
            "strength": decimal_text(evidence_strength),
            "version": EVIDENCE_VERSION,
        },
        identity_field="evidence_id",
    )


def validate_evidence(document: Mapping[str, Any], *, as_of: str) -> dict[str, Any]:
    normalized = validate_content_addressed(document, identity_field="evidence_id")
    expected = build_evidence(
        source_type=normalized.get("source_type"),
        direction=normalized.get("direction"),
        likelihood_ratio=normalized.get("likelihood_ratio"),
        strength=normalized.get("strength"),
        reason=normalized.get("reason"),
        observed_at=normalized.get("observed_at"),
        available_at=normalized.get("available_at"),
        source_ref=normalized.get("source_ref", {}),
        payload=normalized.get("payload"),
    )
    if expected != normalized or normalized.get("version") != EVIDENCE_VERSION:
        raise IntelligenceContractError("evidence replay mismatch")
    require_no_future(
        available_at=str(normalized["available_at"]),
        as_of=as_of,
        label="evidence",
    )
    return normalized


def validate_evidence_set(
    values: Sequence[Mapping[str, Any]], *, as_of: str
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise IntelligenceContractError("at least one evidence item is required")
    rows = [validate_evidence(value, as_of=as_of) for value in values]
    ids = [str(row["evidence_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise IntelligenceContractError("duplicate evidence cannot be counted twice")
    return sorted(rows, key=lambda row: str(row["evidence_id"]).encode("ascii"))


__all__ = [
    "EVIDENCE_DIRECTIONS",
    "EVIDENCE_SOURCES",
    "EVIDENCE_VERSION",
    "build_evidence",
    "validate_evidence",
    "validate_evidence_set",
]
