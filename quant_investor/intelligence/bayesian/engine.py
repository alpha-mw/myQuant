"""Evidence-bound Bayesian updates with no decision or execution authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    content_ref,
    decimal_text,
    decimal_value,
    identifier,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)
from ..evidence.models import validate_evidence_set

BAYESIAN_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.bayesian-evidence-receipt.v1"


def update_hypothesis(
    *,
    hypothesis_id: str,
    prior: Any,
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    """Apply independent evidence likelihood ratios in canonical evidence-id order."""

    hypothesis = identifier(hypothesis_id, label="hypothesis_id")
    prior_value = decimal_value(
        prior,
        label="prior",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    cutoff = timestamp(as_of, label="as_of")
    rows = validate_evidence_set(evidence, as_of=cutoff)

    with localcontext() as context:
        context.prec = 50
        prior_odds = prior_value / (Decimal("1") - prior_value)
        combined_likelihood = Decimal("1")
        residual_uncertainty = Decimal("1")
        direction_counts = {
            "CONTRARY": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0,
            "POSITIVE": 0,
        }
        reason_codes: list[str] = []
        for row in rows:
            strength = decimal_value(
                row["strength"],
                label="evidence.strength",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
            likelihood = decimal_value(
                row["likelihood_ratio"],
                label="evidence.likelihood_ratio",
                minimum=Decimal("0"),
                minimum_exclusive=True,
            )
            effective_likelihood = Decimal("1") + strength * (likelihood - Decimal("1"))
            combined_likelihood *= effective_likelihood
            residual_uncertainty *= Decimal("1") - strength
            direction_counts[str(row["direction"])] += 1
            reason_codes.append(str(row["reason"]))

        posterior_odds = prior_odds * combined_likelihood
        posterior = posterior_odds / (Decimal("1") + posterior_odds)
        confidence = Decimal("1") - residual_uncertainty

    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "confidence": decimal_text(confidence),
            "direction_counts": direction_counts,
            "evidence_refs": [content_ref(row, identity_field="evidence_id") for row in rows],
            "hypothesis_id": hypothesis,
            "likelihood": decimal_text(combined_likelihood),
            "posterior": decimal_text(posterior),
            "prior": decimal_text(prior_value),
            "production": False,
            "reason_codes": sorted(set(reason_codes), key=lambda item: item.encode()),
            "research_only": True,
            "timestamp": cutoff,
            "uncertainty": decimal_text(Decimal("1") - confidence),
            "version": BAYESIAN_RECEIPT_VERSION,
        },
        identity_field="receipt_id",
    )


def validate_bayesian_receipt(
    document: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    """Replay a receipt from its declared prior and admitted evidence closure."""

    normalized = validate_content_addressed(document, identity_field="receipt_id")
    if normalized.get("version") != BAYESIAN_RECEIPT_VERSION:
        raise IntelligenceContractError("Bayesian receipt version mismatch")
    expected_refs = normalized.get("evidence_refs")
    if type(expected_refs) is not list:
        raise IntelligenceContractError("Bayesian evidence refs are missing")
    admitted = [
        row
        for row in validate_evidence_set(evidence, as_of=as_of)
        if content_ref(row, identity_field="evidence_id") in expected_refs
    ]
    expected = update_hypothesis(
        hypothesis_id=normalized.get("hypothesis_id"),
        prior=normalized.get("prior"),
        evidence=admitted,
        as_of=as_of,
    )
    if expected != normalized:
        raise IntelligenceContractError("Bayesian receipt replay mismatch")
    return normalized


__all__ = [
    "BAYESIAN_RECEIPT_VERSION",
    "update_hypothesis",
    "validate_bayesian_receipt",
]
