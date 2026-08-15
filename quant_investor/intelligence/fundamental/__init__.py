"""Deterministic Fundamental assessment with explicit missingness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    decimal_text,
    decimal_value,
    require_no_future,
    timestamp,
    validate_artifact_ref,
)
from ..industry import validate_industry_assessment
from ..theme import validate_theme_assessment

FUNDAMENTAL_COMPONENTS: Final = (
    "business_quality",
    "earnings_quality",
    "growth_durability",
    "industry_cycle",
    "valuation",
)
FUNDAMENTAL_STATUSES: Final = frozenset({"COMPLETE", "PARTIAL", "MISSING", "BLOCKED"})


def _component_rows(
    scores: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Decimal, Decimal]:
    if type(scores) is not dict or set(scores) != set(FUNDAMENTAL_COMPONENTS):
        raise IntelligenceError("fundamental scores must contain exactly all five components")
    if type(weights) is not dict or set(weights) != set(FUNDAMENTAL_COMPONENTS):
        raise IntelligenceError("fundamental weights must contain exactly all five components")
    normalized_weights = {
        component: decimal_value(
            weights[component],
            label=f"weights.{component}",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        for component in FUNDAMENTAL_COMPONENTS
    }
    if sum(normalized_weights.values(), Decimal("0")) != Decimal("1"):
        raise IntelligenceError("fundamental weights must sum exactly to one")
    rows: list[dict[str, Any]] = []
    coverage = Decimal("0")
    weighted = Decimal("0")
    for component in sorted(FUNDAMENTAL_COMPONENTS, key=lambda item: item.encode("ascii")):
        value = scores[component]
        weight = normalized_weights[component]
        if value is None:
            rows.append(
                {
                    "component": component,
                    "score": None,
                    "status": "MISSING",
                    "weight": decimal_text(weight),
                }
            )
            continue
        score = decimal_value(
            value,
            label=f"scores.{component}",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        coverage += weight
        weighted += score * weight
        rows.append(
            {
                "component": component,
                "score": decimal_text(score),
                "status": "AVAILABLE",
                "weight": decimal_text(weight),
            }
        )
    return rows, coverage, weighted


def _component_binding(
    artifact: Mapping[str, Any] | bytes | None,
    *,
    kind: str,
    company: str,
    as_of: str,
) -> dict[str, str] | None:
    if artifact is None:
        return None
    validated = (
        validate_industry_assessment(artifact)
        if kind == "industry_assessment"
        else validate_theme_assessment(artifact)
    )
    require_no_future(validated, as_of=as_of, label=kind)
    payload = validated["payload"]
    if payload.get("company_code") != company:
        raise IntelligenceError(f"{kind} company closure differs")
    return artifact_ref(validated)


def assess_fundamental(
    *,
    company: str,
    component_scores: Mapping[str, Any],
    component_weights: Mapping[str, Any],
    minimum_coverage: Any,
    as_of: str,
    source_refs: Sequence[Mapping[str, Any]] = (),
    industry_assessment: Mapping[str, Any] | bytes | None = None,
    theme_assessment: Mapping[str, Any] | bytes | None = None,
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Build a five-component Fundamental assessment without imputation."""

    code = company_code(company)
    cutoff = timestamp(as_of, label="as_of")
    threshold = decimal_value(
        minimum_coverage,
        label="minimum_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    rows, coverage, weighted = _component_rows(component_scores, component_weights)
    if isinstance(source_refs, (str, bytes)) or not isinstance(source_refs, Sequence):
        raise IntelligenceError("source_refs must be a sequence")
    normalized_sources = [
        validate_artifact_ref(ref, label=f"source_refs[{index}]")
        for index, ref in enumerate(source_refs)
    ]
    normalized_sources.sort(
        key=lambda row: (
            str(row.get("kind", "")).encode("ascii"),
            str(row.get("artifact_id", "")).encode("ascii"),
        )
    )
    source_identities = [(row["kind"], row["artifact_id"]) for row in normalized_sources]
    if len(source_identities) != len(set(source_identities)):
        raise IntelligenceError("fundamental source closure is duplicated")
    industry_ref = _component_binding(
        industry_assessment,
        kind="industry_assessment",
        company=code,
        as_of=cutoff,
    )
    theme_ref = _component_binding(
        theme_assessment,
        kind="theme_assessment",
        company=code,
        as_of=cutoff,
    )
    if coverage == Decimal("1"):
        status = "COMPLETE"
    elif coverage == Decimal("0"):
        status = "MISSING"
    elif coverage < threshold:
        status = "BLOCKED"
    else:
        status = "PARTIAL"
    score_present = coverage > Decimal("0")
    raw_score = decimal_text(weighted) if score_present else None
    effective_score = decimal_text(weighted / coverage) if score_present else None
    blockers: list[str] = []
    if status == "BLOCKED":
        blockers.append("FUNDAMENTAL_COVERAGE_BELOW_POLICY")
    elif status == "MISSING":
        blockers.append("FUNDAMENTAL_COMPONENTS_MISSING")
    return build_artifact(
        kind="fundamental_assessment",
        identity_field="assessment_id",
        identity=assessment_id
        or business_identity(
            kind="fundamental_assessment",
            identity_inputs={"as_of": cutoff, "company_code": code},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": blockers,
            "company_code": code,
            "component_rows": rows,
            "coverage": decimal_text(coverage),
            "effective_score": effective_score,
            "industry_assessment_ref": industry_ref,
            "minimum_coverage": decimal_text(threshold),
            "raw_score": raw_score,
            "score_present": score_present,
            "source_refs": normalized_sources,
            "status": status,
            "theme_assessment_ref": theme_ref,
        },
    )


def validate_fundamental_assessment(  # noqa: C901 - five-component replay gate
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="fundamental_assessment")
    if payload.get("status") not in FUNDAMENTAL_STATUSES:
        raise IntelligenceError("fundamental assessment status is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("fundamental assessment authority is invalid")
    company_code(payload.get("company_code"))
    timestamp(payload.get("as_of"), label="fundamental.as_of")
    rows = payload.get("component_rows")
    if type(rows) is not list or len(rows) != len(FUNDAMENTAL_COMPONENTS):
        raise IntelligenceError("fundamental component closure is incomplete")
    components: set[str] = set()
    total_weight = Decimal("0")
    calculated_coverage = Decimal("0")
    calculated_weighted = Decimal("0")
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "component",
            "score",
            "status",
            "weight",
        }:
            raise IntelligenceError("fundamental component row shape is invalid")
        component = row["component"]
        if component not in FUNDAMENTAL_COMPONENTS or component in components:
            raise IntelligenceError("fundamental component identity is invalid")
        components.add(component)
        weight = decimal_value(
            row["weight"],
            label=f"component_rows[{index}].weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if row["weight"] != decimal_text(weight):
            raise IntelligenceError("fundamental component weight is not canonical")
        total_weight += weight
        if row["status"] == "AVAILABLE":
            score = decimal_value(
                row["score"],
                label=f"component_rows[{index}].score",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
            if row["score"] != decimal_text(score):
                raise IntelligenceError("fundamental component score is not canonical")
            calculated_coverage += weight
            calculated_weighted += score * weight
        elif row["status"] == "MISSING" and row["score"] is None:
            continue
        else:
            raise IntelligenceError("fundamental component status is invalid")
    if components != set(FUNDAMENTAL_COMPONENTS) or total_weight != Decimal("1"):
        raise IntelligenceError("fundamental component closure is incomplete")
    coverage = decimal_value(
        payload.get("coverage"),
        label="fundamental.coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if coverage != calculated_coverage or payload.get("coverage") != decimal_text(coverage):
        raise IntelligenceError("fundamental coverage does not replay")
    if coverage == Decimal("0") and payload.get("score_present") is not False:
        raise IntelligenceError("missing fundamental evidence cannot carry a score")
    if coverage > Decimal("0") and payload.get("score_present") is not True:
        raise IntelligenceError("fundamental score presence is inconsistent")
    expected_raw = decimal_text(calculated_weighted) if coverage > 0 else None
    expected_effective = decimal_text(calculated_weighted / coverage) if coverage > 0 else None
    if (
        payload.get("raw_score") != expected_raw
        or payload.get("effective_score") != expected_effective
    ):
        raise IntelligenceError("fundamental score does not replay")
    minimum_coverage = decimal_value(
        payload.get("minimum_coverage"),
        label="fundamental.minimum_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if payload.get("minimum_coverage") != decimal_text(minimum_coverage):
        raise IntelligenceError("fundamental minimum coverage is not canonical")
    expected_status = (
        "COMPLETE"
        if coverage == Decimal("1")
        else (
            "MISSING"
            if coverage == Decimal("0")
            else "BLOCKED" if coverage < minimum_coverage else "PARTIAL"
        )
    )
    if payload.get("status") != expected_status:
        raise IntelligenceError("fundamental status does not replay")
    expected_blockers = (
        ["FUNDAMENTAL_COVERAGE_BELOW_POLICY"]
        if expected_status == "BLOCKED"
        else ["FUNDAMENTAL_COMPONENTS_MISSING"] if expected_status == "MISSING" else []
    )
    if payload.get("blocker_codes") != expected_blockers:
        raise IntelligenceError("fundamental blockers do not replay")
    source_refs = payload.get("source_refs")
    if type(source_refs) is not list:
        raise IntelligenceError("fundamental source refs are invalid")
    normalized_sources = [
        validate_artifact_ref(ref, label=f"source_refs[{index}]")
        for index, ref in enumerate(source_refs)
    ]
    if normalized_sources != source_refs:
        raise IntelligenceError("fundamental source refs are not canonical")
    for field, expected_kind in (
        ("industry_assessment_ref", "industry_assessment"),
        ("theme_assessment_ref", "theme_assessment"),
    ):
        reference = payload.get(field)
        if reference is not None:
            normalized_ref = validate_artifact_ref(reference, label=field)
            if normalized_ref["kind"] != expected_kind:
                raise IntelligenceError(f"{field} kind is invalid")
    return normalized


__all__ = [
    "FUNDAMENTAL_COMPONENTS",
    "FUNDAMENTAL_STATUSES",
    "IntelligenceError",
    "assess_fundamental",
    "validate_fundamental_assessment",
]
