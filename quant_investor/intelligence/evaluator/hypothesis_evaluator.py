"""Deterministic, research-only evaluation of sealed investment hypotheses.

The evaluator consumes already validated I0 hypothesis/evidence lineage and a
normalized lookup of forward metrics.  It reports scientific evidence only: it
does not mutate the hypothesis, Bayesian posterior, investment memory, factor
governance, or any production/runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import re
from typing import Any, Final, TypeAlias

from .._core import (
    IntelligenceContractError,
    content_ref,
    decimal_text,
    decimal_value,
    identifier,
    sha256,
    timestamp,
)

HYPOTHESIS_STATUSES: Final = {"FAILED", "SUPPORTED", "UNCERTAIN"}
FALSIFICATION_RESULTS: Final = {"INCONCLUSIVE", "NOT_TRIGGERED", "TRIGGERED"}
RULE_OPERATORS: Final = {"EQ", "GT", "GTE", "LT", "LTE", "NEQ"}
RULE_AGGREGATION: Final = "MEAN"
METRIC_STATUSES: Final = {"AVAILABLE", "BLOCKED", "UNAVAILABLE"}

CONTENT_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "semantic_sha256",
}
SPEC_FIELDS: Final = {
    "contrary_rules",
    "evidence_refs",
    "falsification_bindings",
    "hypothesis_ref",
    "min_coverage",
    "min_mature_origins",
    "spec_id",
    "support_rules",
}
RULE_FIELDS: Final = {
    "aggregation",
    "factor_id",
    "label_field",
    "metric_id",
    "operator",
    "threshold",
    "window_end",
    "window_start",
}
BINDING_FIELDS: Final = {
    "condition_index",
    "factor_id",
    "label_field",
    "metric_id",
    "window_end",
    "window_start",
}
METRIC_ROW_FIELDS: Final = {"input_origin_ids", "status", "value"}
DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MetricKey: TypeAlias = tuple[str, str, str, str, str]


def _content_reference(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != CONTENT_REF_FIELDS:
        raise IntelligenceContractError(f"{label} must be an exact content reference")
    artifact_version = value["artifact_version"]
    if type(artifact_version) is not str or not artifact_version.strip():
        raise IntelligenceContractError(f"{label}.artifact_version is required")
    return {
        "artifact_id": sha256(value["artifact_id"], label=f"{label}.artifact_id"),
        "artifact_version": artifact_version,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def _content_references(value: Any, *, label: str) -> list[dict[str, str]]:
    if type(value) is not list or not value:
        raise IntelligenceContractError(f"{label} must be a non-empty list")
    rows = [_content_reference(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    keys = [(row["artifact_id"], row["byte_sha256"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError(f"{label} contains duplicate references")
    return sorted(
        rows,
        key=lambda row: (
            row["artifact_id"].encode("ascii"),
            row["byte_sha256"].encode("ascii"),
        ),
    )


def _window_boundary(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise IntelligenceContractError(f"{label} must be a canonical date or timestamp")
    if DATE_RE.fullmatch(value) is not None:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError as exc:
            raise IntelligenceContractError(f"{label} must be a canonical date") from exc
        return value
    return timestamp(value, label=label)


def _window_key(start: Any, end: Any, *, label: str) -> tuple[str, str]:
    window_start = _window_boundary(start, label=f"{label}.window_start")
    window_end = _window_boundary(end, label=f"{label}.window_end")
    start_is_timestamp = DATE_RE.fullmatch(window_start) is None
    end_is_timestamp = DATE_RE.fullmatch(window_end) is None
    if start_is_timestamp != end_is_timestamp:
        raise IntelligenceContractError(f"{label} window boundary kinds must match")
    if window_end < window_start:
        raise IntelligenceContractError(f"{label} window is reversed")
    return window_start, window_end


def _normalize_rule(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != RULE_FIELDS:
        raise IntelligenceContractError(f"{label} has an invalid shape")
    if value["aggregation"] != RULE_AGGREGATION:
        raise IntelligenceContractError(f"{label} aggregation must be MEAN in v1")
    operator = value["operator"]
    if operator not in RULE_OPERATORS:
        raise IntelligenceContractError(f"{label} operator is not allowlisted")
    if value["label_field"] != "total_return":
        raise IntelligenceContractError(f"{label}.label_field is not supported in v1")
    window_start, window_end = _window_key(value["window_start"], value["window_end"], label=label)
    return {
        "aggregation": RULE_AGGREGATION,
        "factor_id": identifier(value["factor_id"], label=f"{label}.factor_id"),
        "label_field": identifier(value["label_field"], label=f"{label}.label_field"),
        "metric_id": identifier(value["metric_id"], label=f"{label}.metric_id"),
        "operator": operator,
        "threshold": decimal_text(decimal_value(value["threshold"], label=f"{label}.threshold")),
        "window_end": window_end,
        "window_start": window_start,
    }


def _rule_key(rule: Mapping[str, Any]) -> tuple[bytes, ...]:
    return tuple(
        str(rule[field]).encode("ascii")
        for field in (
            "factor_id",
            "metric_id",
            "window_start",
            "window_end",
            "label_field",
            "operator",
            "threshold",
        )
    )


def _normalize_rules(value: Any, *, label: str, required: bool) -> list[dict[str, Any]]:
    if type(value) is not list or (required and not value):
        qualifier = "non-empty " if required else ""
        raise IntelligenceContractError(f"{label} must be a {qualifier}list")
    rows = [_normalize_rule(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    keys = [_rule_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise IntelligenceContractError(f"{label} contains duplicate rules")
    return sorted(rows, key=_rule_key)


def _normalize_binding(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != BINDING_FIELDS:
        raise IntelligenceContractError(f"{label} has an invalid shape")
    condition_index = value["condition_index"]
    if type(condition_index) is not int or type(condition_index) is bool or condition_index < 0:
        raise IntelligenceContractError(f"{label}.condition_index must be a non-negative integer")
    if value["label_field"] != "total_return":
        raise IntelligenceContractError(f"{label}.label_field is not supported in v1")
    window_start, window_end = _window_key(value["window_start"], value["window_end"], label=label)
    return {
        "condition_index": condition_index,
        "factor_id": identifier(value["factor_id"], label=f"{label}.factor_id"),
        "label_field": identifier(value["label_field"], label=f"{label}.label_field"),
        "metric_id": identifier(value["metric_id"], label=f"{label}.metric_id"),
        "window_end": window_end,
        "window_start": window_start,
    }


def _normalize_bindings(
    value: Any, *, conditions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise IntelligenceContractError("falsification_bindings must be a non-empty list")
    rows = [
        _normalize_binding(item, label=f"falsification_bindings[{index}]")
        for index, item in enumerate(value)
    ]
    indexes = [row["condition_index"] for row in rows]
    if len(indexes) != len(set(indexes)):
        raise IntelligenceContractError("falsification bindings contain duplicate indexes")
    if set(indexes) != set(range(len(conditions))):
        raise IntelligenceContractError(
            "falsification bindings must cover every hypothesis condition exactly once"
        )
    result = sorted(rows, key=lambda row: row["condition_index"])
    for row in result:
        condition = conditions[row["condition_index"]]
        if row["metric_id"] != condition.get("metric_id"):
            raise IntelligenceContractError(
                "falsification binding metric_id does not match its hypothesis condition"
            )
    return result


def _normalize_origin_ids(value: Any, *, label: str, required: bool) -> list[str]:
    if type(value) is not list or (required and not value):
        qualifier = "non-empty " if required else ""
        raise IntelligenceContractError(f"{label} must be a {qualifier}list")
    rows: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or not item or len(item.encode("utf-8")) > 256:
            raise IntelligenceContractError(f"{label}[{index}] is invalid")
        rows.append(item)
    if len(rows) != len(set(rows)):
        raise IntelligenceContractError(f"{label} contains duplicates")
    return sorted(rows, key=lambda item: item.encode("utf-8"))


def _normalize_metric_row(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != METRIC_ROW_FIELDS:
        raise IntelligenceContractError(f"{label} has an invalid shape")
    status = value["status"]
    if status not in METRIC_STATUSES:
        raise IntelligenceContractError(f"{label}.status is not allowlisted")
    if status == "AVAILABLE":
        metric_value: str | None = decimal_text(
            decimal_value(value["value"], label=f"{label}.value")
        )
    else:
        if value["value"] is not None:
            raise IntelligenceContractError(f"{label}.value must be null when unavailable")
        metric_value = None
    return {
        "input_origin_ids": _normalize_origin_ids(
            value["input_origin_ids"],
            label=f"{label}.input_origin_ids",
            required=status == "AVAILABLE",
        ),
        "status": status,
        "value": metric_value,
    }


def _normalize_metric_lookup(value: Any) -> dict[MetricKey, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise IntelligenceContractError("metric_lookup must be a mapping")
    result: dict[MetricKey, dict[str, Any]] = {}
    for index, (raw_key, raw_row) in enumerate(value.items()):
        if type(raw_key) is not tuple or len(raw_key) != 5:
            raise IntelligenceContractError(f"metric_lookup key {index} has an invalid shape")
        factor_id = identifier(raw_key[0], label=f"metric_lookup key {index}.factor_id")
        metric_id = identifier(raw_key[1], label=f"metric_lookup key {index}.metric_id")
        window_start, window_end = _window_key(
            raw_key[2], raw_key[3], label=f"metric_lookup key {index}"
        )
        label_field = identifier(raw_key[4], label=f"metric_lookup key {index}.label_field")
        key = (factor_id, metric_id, window_start, window_end, label_field)
        if key in result:
            raise IntelligenceContractError("metric_lookup contains an ambiguous normalized key")
        result[key] = _normalize_metric_row(raw_row, label=f"metric_lookup[{key!r}]")
    return result


def _metric_key(value: Mapping[str, Any]) -> MetricKey:
    return (
        str(value["factor_id"]),
        str(value["metric_id"]),
        str(value["window_start"]),
        str(value["window_end"]),
        str(value["label_field"]),
    )


def _lookup_metric(
    lookup: Mapping[MetricKey, Mapping[str, Any]], locator: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    key = _metric_key(locator)
    try:
        return lookup[key]
    except KeyError as exc:
        raise IntelligenceContractError(f"{label} has no exact normalized metric") from exc


def _compare(value: Decimal, operator: str, threshold: Decimal) -> bool:
    if operator == "EQ":
        return value == threshold
    if operator == "NEQ":
        return value != threshold
    if operator == "GT":
        return value > threshold
    if operator == "GTE":
        return value >= threshold
    if operator == "LT":
        return value < threshold
    if operator == "LTE":
        return value <= threshold
    raise IntelligenceContractError("comparison operator is not allowlisted")


def _evaluate_rule(
    rule: Mapping[str, Any],
    *,
    lookup: Mapping[MetricKey, Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    metric = _lookup_metric(lookup, rule, label=f"{role.lower()} rule")
    value = metric["value"]
    if metric["status"] == "AVAILABLE":
        if value is None:
            raise IntelligenceContractError("available metric is missing its value")
        passed = _compare(
            decimal_value(value, label="metric.value"),
            str(rule["operator"]),
            decimal_value(rule["threshold"], label="rule.threshold"),
        )
        outcome = "PASS" if passed else "FAIL"
    else:
        outcome = "UNAVAILABLE"
    return {
        "factor_id": rule["factor_id"],
        "input_origin_ids": list(metric["input_origin_ids"]),
        "label_field": rule["label_field"],
        "metric_id": rule["metric_id"],
        "operator": rule["operator"],
        "outcome": outcome,
        "status": metric["status"],
        "threshold": rule["threshold"],
        "value": value,
        "window_end": rule["window_end"],
        "window_start": rule["window_start"],
    }


def _evaluate_falsification(
    binding: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    lookup: Mapping[MetricKey, Mapping[str, Any]],
) -> dict[str, Any]:
    metric = _lookup_metric(lookup, binding, label="falsification binding")
    value = metric["value"]
    if metric["status"] == "AVAILABLE":
        if value is None:
            raise IntelligenceContractError("available metric is missing its value")
        triggered = _compare(
            decimal_value(value, label="metric.value"),
            str(condition["operator"]),
            decimal_value(condition["threshold"], label="falsification threshold"),
        )
        outcome = "TRIGGERED" if triggered else "NOT_TRIGGERED"
    else:
        outcome = "INCONCLUSIVE"
    return {
        "condition_index": binding["condition_index"],
        "factor_id": binding["factor_id"],
        "input_origin_ids": list(metric["input_origin_ids"]),
        "label_field": binding["label_field"],
        "metric_id": binding["metric_id"],
        "operator": condition["operator"],
        "outcome": outcome,
        "status": metric["status"],
        "threshold": decimal_text(
            decimal_value(condition["threshold"], label="falsification threshold")
        ),
        "value": value,
        "window_end": binding["window_end"],
        "window_sessions": condition["window_sessions"],
        "window_start": binding["window_start"],
    }


def _metric_refs(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = {
        (
            str(row["factor_id"]),
            str(row["metric_id"]),
            str(row["window_start"]),
            str(row["window_end"]),
            str(row["label_field"]),
        )
        for group in groups
        for row in group
    }
    return [
        {
            "factor_id": factor_id,
            "label_field": label_field,
            "metric_id": metric_id,
            "window_end": window_end,
            "window_start": window_start,
        }
        for factor_id, metric_id, window_start, window_end, label_field in sorted(
            rows, key=lambda row: tuple(item.encode("ascii") for item in row)
        )
    ]


def evaluate_hypothesis(
    *,
    hypothesis: Mapping[str, Any],
    spec: Mapping[str, Any],
    metric_lookup: Mapping[MetricKey, Mapping[str, Any]],
    preregistered: bool,
    mature_origin_count: int,
    joint_coverage: Any,
) -> dict[str, Any]:
    """Evaluate one sealed hypothesis without granting decision authority.

    Numeric conditions are evaluated with the v1 ``MEAN`` aggregation policy.
    A preregistered falsification trigger is the only path to ``FAILED``.
    Post-hoc triggers remain visible as ``TRIGGERED`` evidence but cannot change
    the scientific status beyond ``UNCERTAIN``.
    """

    if type(spec) is not dict or set(spec) != SPEC_FIELDS:
        raise IntelligenceContractError("hypothesis evaluation spec shape is not closed")
    if type(preregistered) is not bool:
        raise IntelligenceContractError("preregistered must be boolean")
    if (
        type(mature_origin_count) is not int
        or type(mature_origin_count) is bool
        or mature_origin_count < 0
    ):
        raise IntelligenceContractError("mature_origin_count must be a non-negative integer")

    try:
        hypothesis_reference = content_ref(hypothesis, identity_field="hypothesis_id")
    except Exception as exc:
        if isinstance(exc, IntelligenceContractError):
            raise
        raise IntelligenceContractError("hypothesis is not content addressed") from exc
    normalized_hypothesis_ref = _content_reference(
        spec["hypothesis_ref"], label="spec.hypothesis_ref"
    )
    if normalized_hypothesis_ref != hypothesis_reference:
        raise IntelligenceContractError("spec hypothesis_ref does not bind the hypothesis")

    supporting_refs = _content_references(
        hypothesis.get("supporting_evidence_refs"), label="hypothesis.supporting_evidence_refs"
    )
    contrary_refs = _content_references(
        hypothesis.get("contrary_evidence_refs"), label="hypothesis.contrary_evidence_refs"
    )
    evidence_refs = _content_references(spec["evidence_refs"], label="spec.evidence_refs")
    expected_evidence_refs = sorted(
        supporting_refs + contrary_refs,
        key=lambda row: (
            row["artifact_id"].encode("ascii"),
            row["byte_sha256"].encode("ascii"),
        ),
    )
    if evidence_refs != expected_evidence_refs:
        raise IntelligenceContractError(
            "spec evidence_refs must exactly bind hypothesis support and contrary evidence"
        )

    conditions = hypothesis.get("falsification_conditions")
    if type(conditions) is not list or not conditions:
        raise IntelligenceContractError("hypothesis falsification conditions are missing")
    for index, condition in enumerate(conditions):
        if type(condition) is not dict or set(condition) != {
            "metric_id",
            "operator",
            "threshold",
            "window_sessions",
        }:
            raise IntelligenceContractError(
                f"hypothesis falsification condition {index} has an invalid shape"
            )
        identifier(condition["metric_id"], label=f"falsification condition {index}.metric_id")
        if condition["operator"] not in RULE_OPERATORS:
            raise IntelligenceContractError("hypothesis falsification operator is invalid")
        decimal_value(condition["threshold"], label=f"falsification condition {index}.threshold")
        window_sessions = condition["window_sessions"]
        if (
            type(window_sessions) is not int
            or type(window_sessions) is bool
            or not 1 <= window_sessions <= 252
        ):
            raise IntelligenceContractError("hypothesis falsification window is invalid")

    spec_id = identifier(spec["spec_id"], label="spec.spec_id")
    min_mature_origins = spec["min_mature_origins"]
    if (
        type(min_mature_origins) is not int
        or type(min_mature_origins) is bool
        or not 1 <= min_mature_origins <= 10000
    ):
        raise IntelligenceContractError("spec.min_mature_origins must be 1..10000")
    min_coverage = decimal_value(
        spec["min_coverage"],
        label="spec.min_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    coverage = decimal_value(
        joint_coverage,
        label="joint_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )

    support_rules = _normalize_rules(spec["support_rules"], label="support_rules", required=True)
    contrary_rules = _normalize_rules(
        spec["contrary_rules"], label="contrary_rules", required=False
    )
    bindings = _normalize_bindings(spec["falsification_bindings"], conditions=conditions)
    lookup = _normalize_metric_lookup(metric_lookup)

    support_results = [
        _evaluate_rule(rule, lookup=lookup, role="SUPPORT") for rule in support_rules
    ]
    contrary_results = [
        _evaluate_rule(rule, lookup=lookup, role="CONTRARY") for rule in contrary_rules
    ]
    falsification_results = [
        _evaluate_falsification(
            binding,
            condition=conditions[binding["condition_index"]],
            lookup=lookup,
        )
        for binding in bindings
    ]

    any_falsification_triggered = any(
        row["outcome"] == "TRIGGERED" for row in falsification_results
    )
    any_falsification_inconclusive = any(
        row["outcome"] == "INCONCLUSIVE" for row in falsification_results
    )
    all_support_pass = all(row["outcome"] == "PASS" for row in support_results)
    any_support_unavailable = any(row["outcome"] == "UNAVAILABLE" for row in support_results)
    any_contrary_met = any(row["outcome"] == "PASS" for row in contrary_results)
    any_contrary_unavailable = any(row["outcome"] == "UNAVAILABLE" for row in contrary_results)
    maturity_pass = mature_origin_count >= min_mature_origins
    coverage_pass = coverage >= min_coverage

    if any_falsification_triggered:
        falsification_result = "TRIGGERED"
    elif any_falsification_inconclusive:
        falsification_result = "INCONCLUSIVE"
    else:
        falsification_result = "NOT_TRIGGERED"

    limitations: set[str] = set()
    if any_falsification_inconclusive:
        limitations.add("FALSIFICATION_EVIDENCE_UNAVAILABLE")
    if any_support_unavailable:
        limitations.add("SUPPORT_EVIDENCE_UNAVAILABLE")
    if any_contrary_unavailable:
        limitations.add("CONTRARY_EVIDENCE_UNAVAILABLE")
    if not all_support_pass and not any_support_unavailable:
        limitations.add("SUPPORT_RULE_NOT_MET")
    if any_contrary_met:
        limitations.add("CONTRARY_RULE_MET")
    if not maturity_pass:
        limitations.add("INSUFFICIENT_MATURE_ORIGINS")
    if not coverage_pass:
        limitations.add("INSUFFICIENT_JOINT_COVERAGE")

    if preregistered and any_falsification_triggered:
        hypothesis_status = "FAILED"
        limitations.add("PREREGISTERED_FALSIFICATION_TRIGGERED")
    elif not preregistered and any_falsification_triggered:
        hypothesis_status = "UNCERTAIN"
        limitations.add("POSTHOC_FALSIFICATION_TRIGGER_NOT_CAUSAL")
    elif (
        preregistered
        and falsification_result == "NOT_TRIGGERED"
        and all_support_pass
        and not any_contrary_met
        and not any_contrary_unavailable
        and maturity_pass
        and coverage_pass
    ):
        hypothesis_status = "SUPPORTED"
    else:
        hypothesis_status = "UNCERTAIN"
        if not preregistered:
            limitations.add("POSTHOC_SPEC_NOT_SUPPORT_ELIGIBLE")

    return {
        "evidence_summary": {
            "contrary_results": contrary_results,
            "evidence_ref_count": len(evidence_refs),
            "falsification_results": falsification_results,
            "joint_coverage": decimal_text(coverage),
            "mature_origin_count": mature_origin_count,
            "preregistered": preregistered,
            "support_results": support_results,
        },
        "falsification_result": falsification_result,
        "hypothesis_ref": normalized_hypothesis_ref,
        "hypothesis_status": hypothesis_status,
        "limitations": sorted(limitations, key=lambda item: item.encode("ascii")),
        "metric_refs": _metric_refs(support_results, contrary_results, falsification_results),
        "spec_id": spec_id,
    }


__all__ = [
    "FALSIFICATION_RESULTS",
    "HYPOTHESIS_STATUSES",
    "METRIC_STATUSES",
    "MetricKey",
    "RULE_AGGREGATION",
    "RULE_OPERATORS",
    "evaluate_hypothesis",
]
