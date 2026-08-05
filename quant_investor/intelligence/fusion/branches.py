"""Typed Quant and Fundamental research branch outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    content_ref,
    decimal_text,
    decimal_value,
    seal_content_addressed,
    timestamp,
    validate_content_addressed,
)
from ..evidence.models import validate_evidence_set

BRANCH_VERSION: Final = "myquant.v17.research-intelligence.branch-output.v1"


def _unit(value: Any, *, label: str) -> Decimal:
    return decimal_value(
        value,
        label=label,
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )


def _signed_unit(value: Any, *, label: str) -> Decimal:
    return decimal_value(
        value,
        label=label,
        minimum=Decimal("-1"),
        maximum=Decimal("1"),
    )


def _build(
    *,
    branch_type: str,
    score: Decimal,
    confidence: Decimal,
    availability: Decimal,
    reliability: Decimal,
    metrics: Mapping[str, str],
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    evidence_rows = validate_evidence_set(evidence, as_of=cutoff)
    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "availability": decimal_text(availability),
            "branch_type": branch_type,
            "confidence": decimal_text(confidence),
            "evidence_refs": [
                content_ref(row, identity_field="evidence_id") for row in evidence_rows
            ],
            "metrics": dict(metrics),
            "production": False,
            "reliability": decimal_text(reliability),
            "research_only": True,
            "score": decimal_text(score),
            "timestamp": cutoff,
            "version": BRANCH_VERSION,
        },
        identity_field="branch_id",
    )


def build_quant_branch(
    *,
    factor_score: Any,
    rank_ic: Any,
    icir: Any,
    exposure: Any,
    coverage: Any,
    confidence: Any,
    availability: Any,
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
    reliability: Any = "1",
) -> dict[str, Any]:
    """Build the mandatory Quant branch with explicit diagnostics."""

    factor = _unit(factor_score, label="factor_score")
    metrics = {
        "coverage": decimal_text(_unit(coverage, label="coverage")),
        "exposure": decimal_text(_signed_unit(exposure, label="exposure")),
        "factor_score": decimal_text(factor),
        "icir": decimal_text(_signed_unit(icir, label="icir")),
        "rank_ic": decimal_text(_signed_unit(rank_ic, label="rank_ic")),
    }
    return _build(
        branch_type="QUANT",
        score=factor,
        confidence=_unit(confidence, label="confidence"),
        availability=_unit(availability, label="availability"),
        reliability=_unit(reliability, label="reliability"),
        metrics=metrics,
        evidence=evidence,
        as_of=as_of,
    )


def build_fundamental_branch(
    *,
    quality: Any,
    earnings: Any,
    valuation: Any,
    industry_position: Any,
    confidence: Any,
    availability: Any,
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
    reliability: Any = "1",
) -> dict[str, Any]:
    """Build the Fundamental branch from four explicit research dimensions."""

    values = {
        "earnings": _unit(earnings, label="earnings"),
        "industry_position": _unit(industry_position, label="industry_position"),
        "quality": _unit(quality, label="quality"),
        "valuation": _unit(valuation, label="valuation"),
    }
    score = sum(values.values()) / Decimal(len(values))
    return _build(
        branch_type="FUNDAMENTAL",
        score=score,
        confidence=_unit(confidence, label="confidence"),
        availability=_unit(availability, label="availability"),
        reliability=_unit(reliability, label="reliability"),
        metrics={key: decimal_text(value) for key, value in values.items()},
        evidence=evidence,
        as_of=as_of,
    )


def validate_branch(
    document: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    """Replay a current I0 branch from its metrics and admitted evidence."""

    normalized = validate_content_addressed(document, identity_field="branch_id")
    if normalized.get("version") != BRANCH_VERSION:
        raise IntelligenceContractError("branch version mismatch")
    expected_refs = normalized.get("evidence_refs")
    metrics = normalized.get("metrics")
    if type(expected_refs) is not list or type(metrics) is not dict:
        raise IntelligenceContractError("branch evidence/metrics are missing")
    admitted = [
        row
        for row in validate_evidence_set(evidence, as_of=as_of)
        if content_ref(row, identity_field="evidence_id") in expected_refs
    ]
    common = {
        "availability": normalized.get("availability"),
        "confidence": normalized.get("confidence"),
        "evidence": admitted,
        "as_of": as_of,
        "reliability": normalized.get("reliability"),
    }
    if normalized.get("branch_type") == "QUANT":
        expected = build_quant_branch(
            factor_score=metrics.get("factor_score"),
            rank_ic=metrics.get("rank_ic"),
            icir=metrics.get("icir"),
            exposure=metrics.get("exposure"),
            coverage=metrics.get("coverage"),
            **common,
        )
    elif normalized.get("branch_type") == "FUNDAMENTAL":
        expected = build_fundamental_branch(
            quality=metrics.get("quality"),
            earnings=metrics.get("earnings"),
            valuation=metrics.get("valuation"),
            industry_position=metrics.get("industry_position"),
            **common,
        )
    else:
        raise IntelligenceContractError("branch type is reserved for a future sprint")
    if expected != normalized:
        raise IntelligenceContractError("branch replay mismatch")
    return normalized


__all__ = [
    "BRANCH_VERSION",
    "build_fundamental_branch",
    "build_quant_branch",
    "validate_branch",
]
