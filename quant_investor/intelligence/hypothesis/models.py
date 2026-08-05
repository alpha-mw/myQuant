"""Machine-checkable, evidence-bound hypothesis records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
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

HYPOTHESIS_VERSION: Final = "myquant.v17.research-intelligence.hypothesis.v1"
FALSIFICATION_OPERATORS: Final = {"EQ", "GT", "GTE", "LT", "LTE", "NEQ"}
ENTITY_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _bounded_text(value: Any, *, label: str, maximum: int = 4000) -> str:
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise IntelligenceContractError(f"{label} is required and bounded")
    return value.strip()


def _identifiers(values: Sequence[str], *, label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise IntelligenceContractError(f"{label} must be a non-empty sequence")
    rows = []
    for index, value in enumerate(values):
        if type(value) is not str or ENTITY_ID_RE.fullmatch(value) is None:
            raise IntelligenceContractError(f"{label}[{index}] is not a canonical entity id")
        rows.append(value)
    if len(rows) != len(set(rows)):
        raise IntelligenceContractError(f"{label} contains duplicates")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def _conditions(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise IntelligenceContractError("at least one falsification condition is required")
    result: list[dict[str, Any]] = []
    required = {"metric_id", "operator", "threshold", "window_sessions"}
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != required:
            raise IntelligenceContractError(
                f"falsification_conditions[{index}] has an invalid shape"
            )
        operator = value["operator"]
        if operator not in FALSIFICATION_OPERATORS:
            raise IntelligenceContractError("falsification operator is not allowlisted")
        window = value["window_sessions"]
        if type(window) is not int or type(window) is bool or not 1 <= window <= 252:
            raise IntelligenceContractError("falsification window must be 1..252 sessions")
        result.append(
            {
                "metric_id": identifier(value["metric_id"], label="metric_id"),
                "operator": str(operator),
                "threshold": decimal_text(decimal_value(value["threshold"], label="threshold")),
                "window_sessions": window,
            }
        )
    keys = [
        (row["metric_id"], row["operator"], row["threshold"], row["window_sessions"])
        for row in result
    ]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError("duplicate falsification conditions are rejected")
    return sorted(
        result,
        key=lambda row: (
            row["metric_id"].encode("ascii"),
            row["operator"].encode("ascii"),
            row["threshold"].encode("ascii"),
            row["window_sessions"],
        ),
    )


def build_hypothesis(
    *,
    thesis: str,
    why_it_may_be_true: str,
    what_would_make_it_fail: str,
    supporting_evidence: Sequence[Mapping[str, Any]],
    contrary_evidence: Sequence[Mapping[str, Any]],
    expected_window_start: str,
    expected_window_end: str,
    falsification_conditions: Sequence[Mapping[str, Any]],
    related_companies: Sequence[str],
    related_industries: Sequence[str],
    as_of: str,
) -> dict[str, Any]:
    """Build a hypothesis that states both its affirmative and failure cases."""

    cutoff = timestamp(as_of, label="as_of")
    start = timestamp(expected_window_start, label="expected_window_start")
    end = timestamp(expected_window_end, label="expected_window_end")
    if start < cutoff or end < start:
        raise IntelligenceContractError("expected window must be forward and ordered")
    support = validate_evidence_set(supporting_evidence, as_of=cutoff)
    contrary = validate_evidence_set(contrary_evidence, as_of=cutoff)
    if any(row["direction"] != "POSITIVE" for row in support):
        raise IntelligenceContractError("supporting evidence must be POSITIVE")
    if any(row["direction"] not in {"CONTRARY", "NEGATIVE"} for row in contrary):
        raise IntelligenceContractError("contrary evidence must be CONTRARY or NEGATIVE")
    support_ids = {str(row["evidence_id"]) for row in support}
    contrary_ids = {str(row["evidence_id"]) for row in contrary}
    if support_ids & contrary_ids:
        raise IntelligenceContractError("supporting and contrary evidence must be disjoint")
    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "contrary_evidence_refs": [
                content_ref(row, identity_field="evidence_id") for row in contrary
            ],
            "expected_window": {"end": end, "start": start},
            "falsification_conditions": _conditions(falsification_conditions),
            "production": False,
            "related_companies": _identifiers(related_companies, label="related_companies"),
            "related_industries": _identifiers(related_industries, label="related_industries"),
            "research_only": True,
            "supporting_evidence_refs": [
                content_ref(row, identity_field="evidence_id") for row in support
            ],
            "thesis": _bounded_text(thesis, label="thesis"),
            "timestamp": cutoff,
            "version": HYPOTHESIS_VERSION,
            "what_would_make_it_fail": _bounded_text(
                what_would_make_it_fail, label="what_would_make_it_fail"
            ),
            "why_it_may_be_true": _bounded_text(why_it_may_be_true, label="why_it_may_be_true"),
        },
        identity_field="hypothesis_id",
    )


def validate_hypothesis(
    document: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    row = validate_content_addressed(document, identity_field="hypothesis_id")
    if row.get("version") != HYPOTHESIS_VERSION:
        raise IntelligenceContractError("hypothesis version mismatch")
    if row.get("research_only") is not True or row.get("production") is not False:
        raise IntelligenceContractError("hypothesis authority boundary is open")
    if set(row) != {
        "authority",
        "contrary_evidence_refs",
        "expected_window",
        "falsification_conditions",
        "hypothesis_id",
        "production",
        "related_companies",
        "related_industries",
        "research_only",
        "semantic_sha256",
        "supporting_evidence_refs",
        "thesis",
        "timestamp",
        "version",
        "what_would_make_it_fail",
        "why_it_may_be_true",
    }:
        raise IntelligenceContractError("hypothesis shape is not closed")
    if evidence is None or as_of is None:
        raise IntelligenceContractError("hypothesis replay requires evidence and as_of")
    evidence_rows = validate_evidence_set(evidence, as_of=as_of)
    supporting_refs = row.get("supporting_evidence_refs")
    contrary_refs = row.get("contrary_evidence_refs")
    expected_window = row.get("expected_window")
    if (
        type(supporting_refs) is not list
        or type(contrary_refs) is not list
        or type(expected_window) is not dict
        or set(expected_window) != {"end", "start"}
    ):
        raise IntelligenceContractError("hypothesis evidence/window is malformed")
    supporting = [
        item
        for item in evidence_rows
        if content_ref(item, identity_field="evidence_id") in supporting_refs
    ]
    contrary = [
        item
        for item in evidence_rows
        if content_ref(item, identity_field="evidence_id") in contrary_refs
    ]
    expected = build_hypothesis(
        thesis=row.get("thesis"),
        why_it_may_be_true=row.get("why_it_may_be_true"),
        what_would_make_it_fail=row.get("what_would_make_it_fail"),
        supporting_evidence=supporting,
        contrary_evidence=contrary,
        expected_window_start=expected_window.get("start"),
        expected_window_end=expected_window.get("end"),
        falsification_conditions=row.get("falsification_conditions", []),
        related_companies=row.get("related_companies", []),
        related_industries=row.get("related_industries", []),
        as_of=as_of,
    )
    if expected != row:
        raise IntelligenceContractError("hypothesis replay mismatch")
    return row


__all__ = ["HYPOTHESIS_VERSION", "build_hypothesis", "validate_hypothesis"]
