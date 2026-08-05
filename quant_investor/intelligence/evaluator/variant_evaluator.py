"""Pure, deterministic comparison of the three fixed V17 research variants.

The helpers in this module compare already-normalized aggregate research
metrics.  They do not select a variant, change factor weights, promote any
artifact, or write governance state.  Each comparison is paired by explicit
origin identity and fails closed when the two aggregates do not describe the
same origin set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from quant_investor.intelligence._core import (
    IntelligenceContractError,
    decimal_text,
    decimal_value,
    identifier,
)

CORE_VARIANT_ID: Final = "v17-quant-core"
INDUSTRY_VARIANT_ID: Final = "v17-quant-plus-industry"
INDUSTRY_THEME_VARIANT_ID: Final = "v17-quant-plus-industry-theme"
VARIANT_IDS: Final = (
    CORE_VARIANT_ID,
    INDUSTRY_VARIANT_ID,
    INDUSTRY_THEME_VARIANT_ID,
)

METRIC_DIRECTIONS: Final = {
    "long_short_spread": "HIGHER_IS_BETTER",
    "rank_ic": "HIGHER_IS_BETTER",
    "icir": "HIGHER_IS_BETTER",
    "turnover": "LOWER_IS_BETTER",
    "drawdown": "LOWER_IS_BETTER",
    "joint_coverage": "HIGHER_IS_BETTER",
    "cost_adjusted_return": "HIGHER_IS_BETTER",
}
METRIC_IDS: Final = tuple(METRIC_DIRECTIONS)

COMPARISON_VERSION: Final = "myquant.v17.research-intelligence.variant-comparison.v1"

_RULE_FIELDS: Final = {
    "degradation_threshold",
    "direction",
    "improvement_threshold",
    "metric_id",
    "tolerance",
}
_VARIANT_FIELDS: Final = {"available_origin_ids", "metrics", "status"}
_METRIC_FIELDS: Final = {"input_origin_ids", "status", "value"}
_EVALUABLE_STATUSES: Final = {"AVAILABLE", "COMPLETE", "PARTIAL"}
_INPUT_STATUSES: Final = _EVALUABLE_STATUSES | {"UNAVAILABLE"}


def _rules_by_metric(
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise IntelligenceContractError("variant rules must be a sequence")

    normalized: dict[str, dict[str, str]] = {}
    for index, raw_rule in enumerate(rules):
        if type(raw_rule) is not dict or set(raw_rule) != _RULE_FIELDS:
            raise IntelligenceContractError(
                f"variant rules[{index}] must contain the exact rule fields"
            )
        metric_id = raw_rule["metric_id"]
        if metric_id not in METRIC_DIRECTIONS:
            raise IntelligenceContractError(f"variant rules[{index}].metric_id is not allowlisted")
        if metric_id in normalized:
            raise IntelligenceContractError("variant rules contain duplicate metrics")
        direction = raw_rule["direction"]
        if direction != METRIC_DIRECTIONS[metric_id]:
            raise IntelligenceContractError(
                f"variant rules[{index}].direction does not match metric policy"
            )

        row = {"metric_id": metric_id, "direction": direction}
        for field in (
            "degradation_threshold",
            "improvement_threshold",
            "tolerance",
        ):
            value = decimal_value(
                raw_rule[field],
                label=f"variant rules[{index}].{field}",
                minimum=Decimal("0"),
            )
            row[field] = decimal_text(value)
        normalized[metric_id] = row

    if set(normalized) != set(METRIC_IDS):
        raise IntelligenceContractError("variant rules must cover all seven metrics")
    return normalized


def _origin_ids(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise IntelligenceContractError(f"{label} must be a sequence")
    normalized = tuple(
        identifier(origin_id, label=f"{label}[{index}]") for index, origin_id in enumerate(value)
    )
    if len(normalized) != len(set(normalized)):
        raise IntelligenceContractError(f"{label} contains duplicate origin IDs")
    return tuple(sorted(normalized, key=lambda item: item.encode("ascii")))


def _variant(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _VARIANT_FIELDS:
        raise IntelligenceContractError(
            f"{label} must contain status, metrics, and available_origin_ids"
        )
    status = value["status"]
    if status not in _INPUT_STATUSES:
        raise IntelligenceContractError(f"{label}.status is invalid")
    if type(value["metrics"]) is not dict:
        raise IntelligenceContractError(f"{label}.metrics must be a mapping")
    unknown_metrics = set(value["metrics"]) - set(METRIC_IDS)
    if unknown_metrics:
        raise IntelligenceContractError(f"{label}.metrics contains unknown metrics")

    metrics: dict[str, dict[str, Any]] = {}
    for metric_id in METRIC_IDS:
        raw_metric = value["metrics"].get(metric_id)
        if raw_metric is None:
            metrics[metric_id] = {
                "input_origin_ids": (),
                "status": "UNAVAILABLE",
                "value": None,
            }
            continue
        if type(raw_metric) is not dict or set(raw_metric) != _METRIC_FIELDS:
            raise IntelligenceContractError(
                f"{label}.metrics.{metric_id} must contain input_origin_ids, status, and value"
            )
        metric_status = raw_metric["status"]
        if metric_status not in {"AVAILABLE", "UNAVAILABLE"}:
            raise IntelligenceContractError(f"{label}.metrics.{metric_id}.status is invalid")
        if metric_status == "UNAVAILABLE":
            metric_origins = _origin_ids(
                raw_metric["input_origin_ids"],
                label=f"{label}.metrics.{metric_id}.input_origin_ids",
            )
            if raw_metric["value"] is not None:
                raise IntelligenceContractError(
                    f"{label}.metrics.{metric_id} cannot carry a value when unavailable"
                )
            metrics[metric_id] = {
                "input_origin_ids": metric_origins,
                "status": "UNAVAILABLE",
                "value": None,
            }
            continue
        metric_origins = _origin_ids(
            raw_metric["input_origin_ids"],
            label=f"{label}.metrics.{metric_id}.input_origin_ids",
        )
        if not metric_origins:
            raise IntelligenceContractError(
                f"{label}.metrics.{metric_id} must bind its contributing origins"
            )
        parsed = decimal_value(raw_metric["value"], label=f"{label}.metrics.{metric_id}.value")
        metrics[metric_id] = {
            "input_origin_ids": metric_origins,
            "status": "AVAILABLE",
            "value": decimal_text(parsed),
        }

    origins = _origin_ids(value["available_origin_ids"], label=f"{label}.available_origin_ids")
    if status in _EVALUABLE_STATUSES and not origins:
        raise IntelligenceContractError(
            f"{label} must bind at least one available origin when evaluable"
        )
    if status == "UNAVAILABLE" and (
        origins or any(row["status"] == "AVAILABLE" for row in metrics.values())
    ):
        raise IntelligenceContractError(f"{label} cannot carry available evidence when unavailable")
    return {"available_origin_ids": origins, "metrics": metrics, "status": status}


def _unavailable_comparison(
    *,
    baseline_variant_id: str,
    candidate_variant_id: str,
) -> dict[str, Any]:
    return {
        "version": COMPARISON_VERSION,
        "core_variant_id": CORE_VARIANT_ID,
        "baseline_variant_id": baseline_variant_id,
        "candidate_variant_id": candidate_variant_id,
        "status": "UNAVAILABLE",
        "conclusion": "UNAVAILABLE",
        "blockers": ["OPTIONAL_VARIANT_UNAVAILABLE"],
        "paired_origin_ids": [],
        "dropped_baseline_origin_ids": [],
        "dropped_candidate_origin_ids": [],
        "metric_comparisons": [],
    }


def compare_variant(
    candidate_variant_id: str,
    *,
    baseline_variant_id: str,
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one optional variant with its explicit adjacent baseline.

    ``improvement_delta`` is positive when the candidate is better, regardless
    of the metric's native direction.  A degradation beyond its rule threshold
    is an absolute negative veto.  Positive evidence requires every metric to
    remain within tolerance and at least one strictly positive improvement to
    meet its improvement threshold.
    """

    if baseline_variant_id not in VARIANT_IDS:
        raise IntelligenceContractError("baseline_variant_id is not allowlisted")
    if candidate_variant_id not in VARIANT_IDS or candidate_variant_id == CORE_VARIANT_ID:
        raise IntelligenceContractError("candidate_variant_id is not optional variant")
    if baseline_variant_id == candidate_variant_id:
        raise IntelligenceContractError("a variant cannot be compared with itself")
    normalized_rules = _rules_by_metric(rules)

    if baseline is None or candidate is None:
        return _unavailable_comparison(
            baseline_variant_id=baseline_variant_id,
            candidate_variant_id=candidate_variant_id,
        )
    normalized_baseline = _variant(baseline, label="baseline")
    normalized_candidate = _variant(candidate, label="candidate")
    if (
        normalized_baseline["status"] == "UNAVAILABLE"
        or normalized_candidate["status"] == "UNAVAILABLE"
    ):
        return _unavailable_comparison(
            baseline_variant_id=baseline_variant_id,
            candidate_variant_id=candidate_variant_id,
        )

    baseline_origins = set(normalized_baseline["available_origin_ids"])
    candidate_origins = set(normalized_candidate["available_origin_ids"])
    paired = tuple(
        sorted(baseline_origins & candidate_origins, key=lambda item: item.encode("ascii"))
    )
    dropped_baseline = tuple(
        sorted(baseline_origins - candidate_origins, key=lambda item: item.encode("ascii"))
    )
    dropped_candidate = tuple(
        sorted(candidate_origins - baseline_origins, key=lambda item: item.encode("ascii"))
    )

    blockers: set[str] = set()
    if dropped_baseline or dropped_candidate:
        blockers.add("PAIRED_ORIGIN_MISMATCH")

    comparisons: list[dict[str, Any]] = []
    any_veto = False
    any_improvement = False
    all_within_tolerance = True
    for metric_id in METRIC_IDS:
        rule = normalized_rules[metric_id]
        baseline_metric = normalized_baseline["metrics"][metric_id]
        candidate_metric = normalized_candidate["metrics"][metric_id]
        baseline_metric_origins = set(baseline_metric["input_origin_ids"])
        candidate_metric_origins = set(candidate_metric["input_origin_ids"])
        metric_paired_origins = sorted(
            baseline_metric_origins & candidate_metric_origins,
            key=lambda item: item.encode("ascii"),
        )
        metric_origin_mismatch = baseline_metric_origins != candidate_metric_origins
        if baseline_metric["status"] != "AVAILABLE" or candidate_metric["status"] != "AVAILABLE":
            blockers.add("REQUIRED_METRIC_UNAVAILABLE")
            all_within_tolerance = False
            comparisons.append(
                {
                    **rule,
                    "baseline_value": baseline_metric["value"],
                    "blocker_codes": ["REQUIRED_METRIC_UNAVAILABLE"],
                    "candidate_value": candidate_metric["value"],
                    "input_origin_ids": metric_paired_origins,
                    "improvement_delta": None,
                    "within_tolerance": False,
                    "improvement_threshold_met": False,
                    "degradation_veto": False,
                    "status": "UNAVAILABLE",
                }
            )
            continue
        if metric_origin_mismatch:
            blockers.add("PAIRED_ORIGIN_MISMATCH")
            all_within_tolerance = False
            comparisons.append(
                {
                    **rule,
                    "baseline_value": baseline_metric["value"],
                    "blocker_codes": ["PAIRED_ORIGIN_MISMATCH"],
                    "candidate_value": candidate_metric["value"],
                    "input_origin_ids": metric_paired_origins,
                    "improvement_delta": None,
                    "within_tolerance": False,
                    "improvement_threshold_met": False,
                    "degradation_veto": False,
                    "status": "UNAVAILABLE",
                }
            )
            continue

        baseline_value = Decimal(baseline_metric["value"])
        candidate_value = Decimal(candidate_metric["value"])
        if rule["direction"] == "HIGHER_IS_BETTER":
            improvement_delta = candidate_value - baseline_value
        else:
            improvement_delta = baseline_value - candidate_value
        degradation_threshold = Decimal(rule["degradation_threshold"])
        improvement_threshold = Decimal(rule["improvement_threshold"])
        tolerance = Decimal(rule["tolerance"])
        degradation_veto = improvement_delta < -degradation_threshold
        within_tolerance = improvement_delta >= -tolerance
        improvement_met = (
            improvement_delta > Decimal("0") and improvement_delta >= improvement_threshold
        )
        any_veto = any_veto or degradation_veto
        any_improvement = any_improvement or improvement_met
        all_within_tolerance = all_within_tolerance and within_tolerance
        comparisons.append(
            {
                **rule,
                "baseline_value": baseline_metric["value"],
                "blocker_codes": [],
                "candidate_value": candidate_metric["value"],
                "input_origin_ids": metric_paired_origins,
                "improvement_delta": decimal_text(improvement_delta),
                "within_tolerance": within_tolerance,
                "improvement_threshold_met": improvement_met,
                "degradation_veto": degradation_veto,
                "status": "AVAILABLE",
            }
        )

    if blockers:
        conclusion = "INCONCLUSIVE"
    elif any_veto:
        conclusion = "INCREMENTAL_NEGATIVE"
        blockers.add("VETO_METRIC_DEGRADED")
    elif all_within_tolerance and any_improvement:
        conclusion = "INCREMENTAL_POSITIVE"
    else:
        conclusion = "INCONCLUSIVE"
        blockers.add("NO_INCREMENTAL_EVIDENCE")

    return {
        "version": COMPARISON_VERSION,
        "core_variant_id": CORE_VARIANT_ID,
        "baseline_variant_id": baseline_variant_id,
        "candidate_variant_id": candidate_variant_id,
        "status": conclusion,
        "conclusion": conclusion,
        "blockers": sorted(blockers, key=lambda item: item.encode("ascii")),
        "paired_origin_ids": list(paired),
        "dropped_baseline_origin_ids": list(dropped_baseline),
        "dropped_candidate_origin_ids": list(dropped_candidate),
        "metric_comparisons": comparisons,
    }


def evaluate_variants(
    *,
    variants: Mapping[str, Mapping[str, Any] | None],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate both optional candidates against the required Quant Core.

    The industry-plus-theme result is deliberately cumulative versus core; it
    does not claim to isolate theme from industry.  Missing optional variants
    remain explicit and independent, so one optional candidate never prevents
    evaluation of the other.
    """

    if type(variants) is not dict:
        raise IntelligenceContractError("variants must be a mapping")
    unknown_variants = set(variants) - set(VARIANT_IDS)
    if unknown_variants:
        raise IntelligenceContractError("variants contains unknown variant IDs")
    if CORE_VARIANT_ID not in variants or variants[CORE_VARIANT_ID] is None:
        raise IntelligenceContractError("v17-quant-core is required")
    core = _variant(variants[CORE_VARIANT_ID], label=CORE_VARIANT_ID)
    if core["status"] not in _EVALUABLE_STATUSES:
        raise IntelligenceContractError("v17-quant-core must be available")

    industry = variants.get(INDUSTRY_VARIANT_ID)
    industry_theme = variants.get(INDUSTRY_THEME_VARIANT_ID)
    industry_comparison = compare_variant(
        INDUSTRY_VARIANT_ID,
        baseline_variant_id=CORE_VARIANT_ID,
        baseline=core,
        candidate=industry,
        rules=rules,
    )
    theme_comparison = compare_variant(
        INDUSTRY_THEME_VARIANT_ID,
        baseline_variant_id=CORE_VARIANT_ID,
        baseline=core,
        candidate=industry_theme,
        rules=rules,
    )
    return {
        "version": COMPARISON_VERSION,
        "core_variant_id": CORE_VARIANT_ID,
        "variant_ids": list(VARIANT_IDS),
        "comparisons": [industry_comparison, theme_comparison],
        "industry_incremental_conclusion": industry_comparison["conclusion"],
        "industry_theme_incremental_conclusion": theme_comparison["conclusion"],
        "limitations": ["THEME_INCREMENT_IS_CUMULATIVE_VS_CORE"],
    }


evaluate_variant_comparisons = evaluate_variants


__all__ = [
    "COMPARISON_VERSION",
    "CORE_VARIANT_ID",
    "INDUSTRY_THEME_VARIANT_ID",
    "INDUSTRY_VARIANT_ID",
    "METRIC_DIRECTIONS",
    "METRIC_IDS",
    "VARIANT_IDS",
    "compare_variant",
    "evaluate_variant_comparisons",
    "evaluate_variants",
]
