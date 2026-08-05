"""Availability-aware branch fusion producing a Research Confidence Score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    assert_no_authority,
    decimal_text,
    decimal_value,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)
from .branches import BRANCH_VERSION

BRANCH_ORDER: Final = ("QUANT", "FUNDAMENTAL", "INDUSTRY", "THEME")
FUSION_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence.availability-aware-fusion-receipt.v1"
)


def _validate_branch(document: Mapping[str, Any], *, as_of: str) -> dict[str, Any]:
    row = validate_content_addressed(document, identity_field="branch_id")
    if row.get("version") != BRANCH_VERSION or row.get("branch_type") not in BRANCH_ORDER:
        raise IntelligenceContractError("branch type/version is not supported")
    if set(row) != {
        "authority",
        "availability",
        "branch_id",
        "branch_type",
        "confidence",
        "evidence_refs",
        "metrics",
        "production",
        "reliability",
        "research_only",
        "score",
        "semantic_sha256",
        "timestamp",
        "version",
    }:
        raise IntelligenceContractError("branch shape is not closed")
    if row.get("timestamp") > as_of:
        raise IntelligenceContractError("branch contains future evidence")
    assert_no_authority(row)
    for field in ("score", "confidence", "availability", "reliability"):
        decimal_value(
            row.get(field),
            label=f"branch.{field}",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    return row


def fuse_research_branches(*, branches: Sequence[Mapping[str, Any]], as_of: str) -> dict[str, Any]:
    """Fuse available branch evidence; Quant is required and weights are data-driven."""

    if isinstance(branches, (str, bytes)) or not isinstance(branches, Sequence):
        raise IntelligenceContractError("branches must be a sequence")
    cutoff = timestamp(as_of, label="as_of")
    rows = [_validate_branch(branch, as_of=cutoff) for branch in branches]
    types = [str(row["branch_type"]) for row in rows]
    if not rows or "QUANT" not in types:
        raise IntelligenceContractError("the Quant branch is required")
    if len(types) != len(set(types)):
        raise IntelligenceContractError("a branch type may appear only once")
    rows.sort(key=lambda row: BRANCH_ORDER.index(str(row["branch_type"])))

    with localcontext() as context:
        context.prec = 50
        masses = {
            str(row["branch_type"]): decimal_value(row["availability"], label="availability")
            * decimal_value(row["confidence"], label="confidence")
            * decimal_value(row["reliability"], label="reliability")
            for row in rows
        }
        total_mass = sum(masses.values())
        if total_mass <= 0:
            raise IntelligenceContractError("available branch mass must be positive")
        normalized_weights = {
            branch_type: mass / total_mass for branch_type, mass in masses.items()
        }
        raw_score = sum(
            decimal_value(row["score"], label="score") * normalized_weights[str(row["branch_type"])]
            for row in rows
        )
        branch_confidence = sum(
            decimal_value(row["confidence"], label="confidence")
            * normalized_weights[str(row["branch_type"])]
            for row in rows
        )
        expected_types = {"QUANT", "FUNDAMENTAL"} | (set(types) & {"INDUSTRY", "THEME"})
        availability_coverage = sum(
            decimal_value(row["availability"], label="availability")
            for row in rows
            if str(row["branch_type"]) in expected_types
        ) / Decimal(len(expected_types))
        research_confidence_score = raw_score * branch_confidence * availability_coverage

    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "availability_coverage": decimal_text(availability_coverage),
            "branch_refs": [
                {
                    "branch_id": row["branch_id"],
                    "semantic_sha256": row["semantic_sha256"],
                    "version": row["version"],
                }
                for row in rows
            ],
            "normalized_weights": {
                key: decimal_text(value) for key, value in normalized_weights.items()
            },
            "production": False,
            "research_confidence_score": decimal_text(research_confidence_score),
            "research_only": True,
            "timestamp": cutoff,
            "version": FUSION_RECEIPT_VERSION,
        },
        identity_field="receipt_id",
    )


def validate_fusion_receipt(
    document: Mapping[str, Any],
    *,
    branches: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    normalized = validate_content_addressed(document, identity_field="receipt_id")
    if normalized.get("version") != FUSION_RECEIPT_VERSION:
        raise IntelligenceContractError("fusion receipt version mismatch")
    expected = fuse_research_branches(branches=branches, as_of=as_of)
    if expected != normalized:
        raise IntelligenceContractError("fusion receipt replay mismatch")
    return normalized


__all__ = [
    "BRANCH_ORDER",
    "FUSION_RECEIPT_VERSION",
    "fuse_research_branches",
    "validate_fusion_receipt",
]
