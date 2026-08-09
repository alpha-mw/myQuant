"""Owner-policy-bound industry evidence and deterministic numerical components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Any

from .._core import (
    content_ref,
    decimal_text,
    exact_ref,
    require_exact_keys,
    timestamp,
)
from .contracts import (
    artifact,
    closed_artifact,
    decimal,
    entity,
    exact_sequence,
    fail,
    no_future,
)
from .identity import (
    validate_industry_evaluation_receipt,
    validate_industry_taxonomy,
)
from .models import (
    COMPONENT_DIMENSIONS,
    COMPONENT_POLICY_VERSION,
    COMPONENT_RECEIPT_VERSION,
    DIRECTIONS,
    EVIDENCE_VERSION,
    MISSING_RULES,
)

_EVIDENCE_FIELDS = frozenset(
    {"cutoff", "dimension", "direction", "metric_id", "observations", "taxonomy_ref"}
)
_OBSERVATION_FIELDS = frozenset({"available_at", "industry_id", "source_refs", "value"})
_POLICY_FIELDS = frozenset({"dimensions"})
_DIMENSION_POLICY_FIELDS = frozenset(
    {
        "dimension",
        "dimension_weight",
        "metrics",
        "minimum_metric_coverage",
        "missing_rule",
        "winsor_lower",
        "winsor_upper",
    }
)
_METRIC_POLICY_FIELDS = frozenset({"direction", "metric_id", "weight"})
_COMPONENT_FIELDS = frozenset(
    {
        "as_of",
        "component_score",
        "dimension_rows",
        "evaluation_ref",
        "industry_id",
        "limitation_codes",
        "policy_ref",
        "status",
        "taxonomy_ref",
    }
)


def _observation(value: Mapping[str, Any], *, cutoff: str, index: int) -> dict[str, Any]:
    row = require_exact_keys(value, _OBSERVATION_FIELDS, label=f"observations[{index}]")
    available_at = timestamp(row["available_at"], label="observation.available_at")
    no_future(available_at=available_at, as_of=cutoff, label="observation")
    refs = [
        exact_ref(ref, label=f"observation.source_refs[{ref_index}]")
        for ref_index, ref in enumerate(
            exact_sequence(row["source_refs"], label="observation.source_refs")
        )
    ]
    refs.sort(
        key=lambda ref: (
            ref["relative_path"].encode("ascii"),
            ref["byte_sha256"].encode("ascii"),
        )
    )
    if not refs or len({(ref["relative_path"], ref["byte_sha256"]) for ref in refs}) != len(refs):
        fail("observation source refs are empty or duplicated")
    if any(ref["available_at"] > available_at for ref in refs):
        fail("observation predates its source")
    return {
        "available_at": available_at,
        "industry_id": entity(row["industry_id"], label="observation.industry_id"),
        "source_refs": refs,
        "value": decimal(row["value"], label="observation.value"),
    }


def build_industry_evidence(
    *,
    taxonomy: Mapping[str, Any],
    metric_id: str,
    dimension: str,
    direction: str,
    observations: Sequence[Mapping[str, Any]],
    cutoff: str,
    created_at: str,
) -> dict[str, Any]:
    taxonomy_row = validate_industry_taxonomy(taxonomy)
    exact_cutoff = timestamp(cutoff, label="cutoff")
    created = timestamp(created_at, label="created_at")
    if taxonomy_row["timestamp"] > exact_cutoff or created < exact_cutoff:
        fail("industry evidence chronology is invalid")
    if dimension not in COMPONENT_DIMENSIONS or direction not in DIRECTIONS:
        fail("industry evidence dimension or direction is invalid")
    rows = [
        _observation(value, cutoff=exact_cutoff, index=index)
        for index, value in enumerate(exact_sequence(observations, label="observations"))
    ]
    rows.sort(key=lambda row: row["industry_id"].encode("ascii"))
    valid_ids = {
        row["industry_id"]
        for row in taxonomy_row["rows"]
        if row["status"] == "ACTIVE"
        and row["available_at"] <= exact_cutoff
        and row["effective_from"] <= exact_cutoff
        and (row["effective_to"] is None or exact_cutoff <= row["effective_to"])
    }
    if (
        not rows
        or len({row["industry_id"] for row in rows}) != len(rows)
        or any(row["industry_id"] not in valid_ids for row in rows)
    ):
        fail("industry evidence observation identities are invalid")
    return artifact(
        version=EVIDENCE_VERSION,
        identity_field="evidence_id",
        timestamp_value=created,
        payload={
            "cutoff": exact_cutoff,
            "dimension": dimension,
            "direction": direction,
            "metric_id": entity(metric_id, label="metric_id"),
            "observations": rows,
            "taxonomy_ref": content_ref(taxonomy_row, identity_field="taxonomy_receipt_id"),
        },
    )


def validate_industry_evidence(
    value: Mapping[str, Any], *, taxonomy: Mapping[str, Any]
) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=EVIDENCE_VERSION,
        identity_field="evidence_id",
        payload_fields=_EVIDENCE_FIELDS,
    )
    rebuilt = build_industry_evidence(
        taxonomy=taxonomy,
        metric_id=row["metric_id"],
        dimension=row["dimension"],
        direction=row["direction"],
        observations=row["observations"],
        cutoff=row["cutoff"],
        created_at=row["timestamp"],
    )
    if rebuilt != row:
        fail("industry evidence replay mismatch")
    return row


def _metric_policy(value: Mapping[str, Any], *, index: int) -> dict[str, str]:
    row = require_exact_keys(value, _METRIC_POLICY_FIELDS, label=f"metrics[{index}]")
    direction = str(row["direction"])
    if direction not in DIRECTIONS:
        fail("component policy metric direction is invalid")
    return {
        "direction": direction,
        "metric_id": entity(row["metric_id"], label="policy.metric_id"),
        "weight": decimal(
            row["weight"], label="policy.metric.weight", minimum=Decimal(0), maximum=Decimal(1)
        ),
    }


def _dimension_policy(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    row = require_exact_keys(value, _DIMENSION_POLICY_FIELDS, label=f"dimensions[{index}]")
    dimension = str(row["dimension"])
    missing_rule = str(row["missing_rule"])
    if dimension not in COMPONENT_DIMENSIONS or missing_rule not in MISSING_RULES:
        fail("component dimension policy enum is invalid")
    metrics = [
        _metric_policy(metric, index=metric_index)
        for metric_index, metric in enumerate(
            exact_sequence(row["metrics"], label="policy.metrics")
        )
    ]
    if not metrics or len({metric["metric_id"] for metric in metrics}) != len(metrics):
        fail("component policy metrics are empty or duplicated")
    if sum(Decimal(metric["weight"]) for metric in metrics) != Decimal("1.000000000000"):
        fail("component metric weights must sum exactly to one")
    lower = decimal(
        row["winsor_lower"], label="winsor_lower", minimum=Decimal(0), maximum=Decimal(1)
    )
    upper = decimal(
        row["winsor_upper"], label="winsor_upper", minimum=Decimal(0), maximum=Decimal(1)
    )
    if Decimal(lower) >= Decimal(upper):
        fail("winsor bounds are invalid")
    return {
        "dimension": dimension,
        "dimension_weight": decimal(
            row["dimension_weight"],
            label="dimension_weight",
            minimum=Decimal(0),
            maximum=Decimal(1),
        ),
        "metrics": metrics,
        "minimum_metric_coverage": decimal(
            row["minimum_metric_coverage"],
            label="minimum_metric_coverage",
            minimum=Decimal(0),
            maximum=Decimal(1),
        ),
        "missing_rule": missing_rule,
        "winsor_lower": lower,
        "winsor_upper": upper,
    }


def build_industry_component_policy(
    *, dimensions: Sequence[Mapping[str, Any]], created_at: str
) -> dict[str, Any]:
    rows = [
        _dimension_policy(value, index=index)
        for index, value in enumerate(exact_sequence(dimensions, label="dimensions"))
    ]
    rows.sort(key=lambda row: COMPONENT_DIMENSIONS.index(row["dimension"]))
    if tuple(row["dimension"] for row in rows) != COMPONENT_DIMENSIONS:
        fail("component policy must explicitly define all six dimensions")
    if sum(Decimal(row["dimension_weight"]) for row in rows) != Decimal("1.000000000000"):
        fail("component dimension weights must sum exactly to one")
    return artifact(
        version=COMPONENT_POLICY_VERSION,
        identity_field="component_policy_id",
        timestamp_value=created_at,
        payload={"dimensions": rows},
    )


def validate_industry_component_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=COMPONENT_POLICY_VERSION,
        identity_field="component_policy_id",
        payload_fields=_POLICY_FIELDS,
    )
    rebuilt = build_industry_component_policy(
        dimensions=row["dimensions"], created_at=row["timestamp"]
    )
    if rebuilt != row:
        fail("industry component policy replay mismatch")
    return row


def _type7(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        fail("Type-7 percentile requires observations")
    if len(ordered) == 1:
        return ordered[0]
    with localcontext() as context:
        context.prec = 50
        position = Decimal(len(ordered) - 1) * probability
        lower_index = int(position.to_integral_value(rounding=ROUND_FLOOR))
        fraction = position - Decimal(lower_index)
        return ordered[lower_index] + fraction * (
            ordered[min(lower_index + 1, len(ordered) - 1)] - ordered[lower_index]
        )


def _percentile(values: Sequence[Decimal], target: Decimal) -> Decimal:
    ordered = sorted(values)
    if len(ordered) == 1:
        return Decimal("0.5")
    positions = [index for index, value in enumerate(ordered) if value == target]
    if not positions:
        fail("target is absent from its winsorized peer set")
    average_position = (Decimal(positions[0]) + Decimal(positions[-1])) / Decimal(2)
    return average_position / Decimal(len(ordered) - 1)


def _metric_component(
    *,
    dimension_policy: Mapping[str, Any],
    metric_policy: Mapping[str, Any],
    evidence_by_metric: Mapping[str, dict[str, Any]],
    industry_id: str,
) -> dict[str, Any] | None:
    evidence_row = evidence_by_metric.get(metric_policy["metric_id"])
    if evidence_row is None:
        return None
    if (
        evidence_row["dimension"] != dimension_policy["dimension"]
        or evidence_row["direction"] != metric_policy["direction"]
    ):
        fail("industry evidence does not match component policy")
    observations = {
        row["industry_id"]: Decimal(row["value"]) for row in evidence_row["observations"]
    }
    if industry_id not in observations:
        return None
    values = list(observations.values())
    lower = _type7(values, Decimal(dimension_policy["winsor_lower"]))
    upper = _type7(values, Decimal(dimension_policy["winsor_upper"]))
    winsorized = [min(max(value, lower), upper) for value in values]
    target = min(max(observations[industry_id], lower), upper)
    percentile = _percentile(winsorized, target)
    if metric_policy["direction"] == "LOWER_IS_BETTER":
        percentile = Decimal(1) - percentile
    return {
        "evidence_ref": content_ref(evidence_row, identity_field="evidence_id"),
        "metric_id": metric_policy["metric_id"],
        "percentile": decimal_text(percentile),
        "policy_weight": metric_policy["weight"],
        "winsorized_value": decimal_text(target),
    }


def _dimension_component(
    *,
    dimension_policy: Mapping[str, Any],
    evidence_by_metric: Mapping[str, dict[str, Any]],
    industry_id: str,
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    missing_metric_ids: list[str] = []
    for metric_policy in dimension_policy["metrics"]:
        metric_row = _metric_component(
            dimension_policy=dimension_policy,
            metric_policy=metric_policy,
            evidence_by_metric=evidence_by_metric,
            industry_id=industry_id,
        )
        if metric_row is None:
            missing_metric_ids.append(metric_policy["metric_id"])
        else:
            metric_rows.append(metric_row)
    coverage = Decimal(len(metric_rows)) / Decimal(len(dimension_policy["metrics"]))
    available = coverage >= Decimal(dimension_policy["minimum_metric_coverage"]) and not (
        dimension_policy["missing_rule"] == "BLOCK_COMPONENT" and missing_metric_ids
    )
    score: str | None = None
    if available:
        admitted_weights = sum(Decimal(row["policy_weight"]) for row in metric_rows)
        available = admitted_weights > 0
        if available:
            score = decimal_text(
                sum(
                    (
                        Decimal(row["percentile"]) * Decimal(row["policy_weight"])
                        for row in metric_rows
                    ),
                    Decimal("0"),
                )
                / admitted_weights
            )
    return {
        "dimension": dimension_policy["dimension"],
        "metric_coverage": decimal_text(coverage),
        "metric_rows": metric_rows,
        "missing_metric_ids": sorted(missing_metric_ids, key=lambda value: value.encode("ascii")),
        "score": score,
        "status": "AVAILABLE" if available else "MISSING",
    }


def _selected_taxonomy(
    *, evaluation: Mapping[str, Any], taxonomies: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    taxonomy_rows = [validate_industry_taxonomy(value) for value in taxonomies]
    selected_ref = evaluation["taxonomy_ref"]
    if selected_ref is None:
        fail("industry component requires a selected taxonomy")
    selected = [
        row
        for row in taxonomy_rows
        if content_ref(row, identity_field="taxonomy_receipt_id") == selected_ref
    ]
    if len(selected) != 1:
        fail("industry evaluation selected taxonomy is absent from closure")
    return selected[0]


def _evidence_by_metric(
    *,
    evidence: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    cutoff: str,
) -> dict[str, dict[str, Any]]:
    rows = [validate_industry_evidence(value, taxonomy=taxonomy) for value in evidence]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["timestamp"] > cutoff or row["cutoff"] > cutoff:
            fail("industry evidence is future-known")
        if row["metric_id"] in result:
            fail("industry component contains duplicate metric evidence")
        result[row["metric_id"]] = row
    return result


def build_industry_component_receipt(
    *,
    identity_evaluation: Mapping[str, Any],
    identity_policy: Mapping[str, Any],
    taxonomies: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
    component_policy: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    evaluation = validate_industry_evaluation_receipt(
        identity_evaluation,
        policy=identity_policy,
        taxonomies=taxonomies,
        catalogs=catalogs,
    )
    taxonomy_row = _selected_taxonomy(evaluation=evaluation, taxonomies=taxonomies)
    policy = validate_industry_component_policy(component_policy)
    cutoff = timestamp(as_of, label="as_of")
    if evaluation["timestamp"] > cutoff or policy["timestamp"] > cutoff:
        fail("component input is future-known")
    if evaluation["state"] != "AVAILABLE" or evaluation["primary_industry_id"] is None:
        fail("industry component requires AVAILABLE identity")
    industry_id = evaluation["primary_industry_id"]
    by_metric = _evidence_by_metric(evidence=evidence, taxonomy=taxonomy_row, cutoff=cutoff)
    dimension_rows: list[dict[str, Any]] = []
    limitations: list[str] = []
    for dimension_policy in policy["dimensions"]:
        dimension_row = _dimension_component(
            dimension_policy=dimension_policy,
            evidence_by_metric=by_metric,
            industry_id=industry_id,
        )
        if dimension_row["status"] == "MISSING":
            limitations.append(f"{dimension_policy['dimension']}_COMPONENT_MISSING")
        dimension_rows.append(dimension_row)
    component_score: str | None = None
    status = "MISSING"
    if all(row["status"] == "AVAILABLE" for row in dimension_rows):
        weights = {
            row["dimension"]: Decimal(row["dimension_weight"]) for row in policy["dimensions"]
        }
        component_score = decimal_text(
            sum(
                (Decimal(row["score"]) * weights[row["dimension"]] for row in dimension_rows),
                Decimal("0"),
            )
        )
        status = "AVAILABLE"
    return artifact(
        version=COMPONENT_RECEIPT_VERSION,
        identity_field="component_receipt_id",
        timestamp_value=cutoff,
        payload={
            "as_of": cutoff,
            "component_score": component_score,
            "dimension_rows": dimension_rows,
            "evaluation_ref": content_ref(evaluation, identity_field="evaluation_id"),
            "industry_id": industry_id,
            "limitation_codes": sorted(limitations, key=lambda value: value.encode("ascii")),
            "policy_ref": content_ref(policy, identity_field="component_policy_id"),
            "status": status,
            "taxonomy_ref": content_ref(taxonomy_row, identity_field="taxonomy_receipt_id"),
        },
    )


def validate_industry_component_receipt(
    value: Mapping[str, Any],
    *,
    identity_evaluation: Mapping[str, Any],
    identity_policy: Mapping[str, Any],
    taxonomies: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
    component_policy: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=COMPONENT_RECEIPT_VERSION,
        identity_field="component_receipt_id",
        payload_fields=_COMPONENT_FIELDS,
    )
    rebuilt = build_industry_component_receipt(
        identity_evaluation=identity_evaluation,
        identity_policy=identity_policy,
        taxonomies=taxonomies,
        catalogs=catalogs,
        component_policy=component_policy,
        evidence=evidence,
        as_of=row["as_of"],
    )
    if rebuilt != row:
        fail("industry component receipt replay mismatch")
    return row


__all__ = [
    "build_industry_component_policy",
    "build_industry_component_receipt",
    "build_industry_evidence",
    "validate_industry_component_policy",
    "validate_industry_component_receipt",
    "validate_industry_evidence",
]
