"""Owner-sealed policies for deterministic I4 Fundamental components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    canonical_bytes,
    code,
    common_fields,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    require_no_future,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .models import (
    COMPONENTS,
    COMPONENT_DIRECTIONS,
    COMPONENT_POLICY_VERSION,
    INDUSTRY_PROJECTION_METRIC,
    MISSING_RULES,
    PERCENTILE_METHOD,
    THEME_PROJECTION_METRIC,
    FundamentalContractError,
)

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_POLICY_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "policy_id",
    "semantic_sha256",
    "components",
    "owner_policy_ref",
}
_COMPONENT_FIELDS: Final = {
    "component",
    "implementation_sha256",
    "metric_rows",
    "minimum_coverage",
    "missing_rule",
    "percentile_method",
    "source_cutoff",
    "winsor_lower",
    "winsor_upper",
}
_METRIC_FIELDS: Final = {"metric_id", "direction", "weight"}


def _fail(message: str) -> None:
    raise FundamentalContractError(message)


def _metric_row(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    row = require_exact_keys(value, _METRIC_FIELDS, label=label)
    direction = code(row["direction"], label=f"{label}.direction")
    if direction not in COMPONENT_DIRECTIONS:
        _fail("Fundamental metric direction is invalid")
    weight = decimal_value(
        row["weight"],
        label=f"{label}.weight",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if weight <= 0:
        _fail("Fundamental metric weight must be positive")
    return {
        "metric_id": identifier(row["metric_id"], label=f"{label}.metric_id"),
        "direction": direction,
        "weight": decimal_text(weight),
    }


def _projection_metric_constraint(component: str, rows: Sequence[Mapping[str, str]]) -> None:
    expected = {
        "industry_cycle": INDUSTRY_PROJECTION_METRIC,
        "theme_narrative": THEME_PROJECTION_METRIC,
    }.get(component)
    if expected is None:
        return
    if (
        len(rows) != 1
        or rows[0]["metric_id"] != expected
        or rows[0]["direction"] != "HIGHER_IS_BETTER"
        or rows[0]["weight"] != decimal_text(Decimal("1"))
    ):
        _fail(f"{component} must be the exact I2/I3 projection metric")


def _metric_rows(value: Any, *, component: str, label: str) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("Fundamental metric rows must be a sequence")
    if not 1 <= len(value) <= 64:
        _fail("Fundamental metric row cardinality is invalid")
    metrics = [
        _metric_row(metric, label=f"{label}.metric_rows[{metric_index}]")
        for metric_index, metric in enumerate(value)
    ]
    metric_ids = [metric["metric_id"] for metric in metrics]
    if len(metric_ids) != len(set(metric_ids)):
        _fail("Fundamental component contains duplicate metric IDs")
    total = sum((Decimal(metric["weight"]) for metric in metrics), Decimal("0"))
    if total != Decimal("1.000000000000"):
        _fail("Fundamental metric weights must sum exactly to one")
    _projection_metric_constraint(component, metrics)
    return metrics


def _component_row(value: Mapping[str, Any], *, created_at: str, index: int) -> dict[str, Any]:
    label = f"components[{index}]"
    row = require_exact_keys(value, _COMPONENT_FIELDS, label=label)
    component = str(row["component"])
    if component not in COMPONENTS:
        _fail("Fundamental component is invalid")
    if row["percentile_method"] != PERCENTILE_METHOD:
        _fail("Fundamental percentile method must be Type-7 average-tie")
    metrics = _metric_rows(row["metric_rows"], component=component, label=label)
    lower = decimal_value(
        row["winsor_lower"],
        label=f"{label}.winsor_lower",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    upper = decimal_value(
        row["winsor_upper"],
        label=f"{label}.winsor_upper",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if lower >= upper:
        _fail("Fundamental winsorization bounds are invalid")
    minimum_coverage = decimal_value(
        row["minimum_coverage"],
        label=f"{label}.minimum_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if minimum_coverage <= 0:
        _fail("Fundamental minimum coverage must be positive")
    missing_rule = code(row["missing_rule"], label=f"{label}.missing_rule")
    if missing_rule not in MISSING_RULES:
        _fail("Fundamental missing rule is invalid")
    source_cutoff = timestamp(row["source_cutoff"], label=f"{label}.source_cutoff")
    if source_cutoff > created_at:
        _fail("Fundamental policy source cutoff is future-known")
    return {
        "component": component,
        "implementation_sha256": sha256(
            row["implementation_sha256"],
            label=f"{label}.implementation_sha256",
        ),
        "metric_rows": metrics,
        "minimum_coverage": decimal_text(minimum_coverage),
        "missing_rule": missing_rule,
        "percentile_method": PERCENTILE_METHOD,
        "source_cutoff": source_cutoff,
        "winsor_lower": decimal_text(lower),
        "winsor_upper": decimal_text(upper),
    }


def build_fundamental_component_policy(
    *,
    components: Sequence[Mapping[str, Any]],
    owner_policy_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Seal the exact five owner components without granting runtime authority."""

    created = timestamp(created_at, label="created_at")
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        _fail("Fundamental components must be a sequence")
    rows = [
        _component_row(value, created_at=created, index=index)
        for index, value in enumerate(components)
    ]
    rows.sort(key=lambda row: COMPONENTS.index(row["component"]))
    if tuple(row["component"] for row in rows) != COMPONENTS:
        _fail("Fundamental policy must define exactly all five owner components")
    owner_ref = exact_ref(owner_policy_ref, label="owner_policy_ref")
    require_no_future(
        available_at=owner_ref["available_at"],
        as_of=created,
        label="owner_policy_ref",
    )
    if owner_ref["cutoff"] > created:
        _fail("Fundamental owner policy cutoff is future-known")
    return seal(
        {
            **common_fields(timestamp_value=created),
            "components": rows,
            "owner_policy_ref": owner_ref,
            "version": COMPONENT_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_fundamental_component_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="policy_id")
    require_exact_keys(normalized, _POLICY_FIELDS, label="FundamentalComponentPolicy.v1")
    if (
        normalized["version"] != COMPONENT_POLICY_VERSION
        or normalized["authority"] != NO_AUTHORITY
        or normalized["research_only"] is not True
        or normalized["production"] is not False
    ):
        _fail("Fundamental component policy boundary is invalid")
    expected = build_fundamental_component_policy(
        components=normalized["components"],
        owner_policy_ref=normalized["owner_policy_ref"],
        created_at=normalized["timestamp"],
    )
    if canonical_bytes(normalized) != canonical_bytes(expected):
        _fail("Fundamental component policy differs from deterministic replay")
    return normalized


__all__ = [
    "build_fundamental_component_policy",
    "validate_fundamental_component_policy",
]
