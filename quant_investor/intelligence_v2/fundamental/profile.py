"""Deterministic, PIT-bound Fundamental Intelligence Profile v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Any, Final

from quant_investor.v17_v4_runtime.forward_scoring_v3 import (
    FUNDAMENTAL_SCORING_V3_VERSION,
    score_fundamental_forward_v3,
)

from .._core import (
    NO_AUTHORITY,
    canonical_bytes,
    common_fields,
    content_ref,
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
from ..industry.component import validate_industry_component_receipt
from ..theme.engine import validate_theme_component_receipt
from .models import (
    COMPONENTS,
    FINANCIAL_QUALITY_METRICS,
    FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3,
    INDUSTRY_COMPONENT_VERSION,
    INDUSTRY_PROJECTION_METRIC,
    PROFILE_STATUSES,
    PROFILE_VERSION,
    THEME_COMPONENT_VERSION,
    THEME_PROJECTION_METRIC,
    FundamentalContractError,
)
from .policy import validate_fundamental_component_policy

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_PROFILE_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "profile_id",
    "semantic_sha256",
    "as_of",
    "company_code",
    "peer_symbols",
    "policy_ref",
    "scorer_implementation_sha256",
    "scorer_version",
    "financial_metric_rows",
    "component_metric_rows",
    "industry_component_refs",
    "theme_component_refs",
    "component_rows",
    "component_weights",
    "raw_float_audit",
    "subject_record",
    "status",
    "raw_score",
    "effective_score",
    "coverage",
    "score_present",
}
_FINANCIAL_INPUT_FIELDS: Final = {
    "available_at",
    "company_code",
    "metric_id",
    "source_ref",
    "value",
}
_COMPONENT_INPUT_FIELDS: Final = _FINANCIAL_INPUT_FIELDS | {"component"}
_INDUSTRY_CLOSURE_FIELDS: Final = {
    "catalogs",
    "component_policy",
    "evidence",
    "identity_evaluation",
    "identity_policy",
    "taxonomies",
}
_THEME_CLOSURE_FIELDS: Final = {
    "as_of",
    "component_policy",
    "exposure_closure",
    "exposure_receipt",
    "metric_rows",
}


def _fail(message: str) -> None:
    raise FundamentalContractError(message)


def _symbols(values: Sequence[str], *, subject: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("Fundamental peer symbols must be a sequence")
    if not 1 <= len(values) <= 500:
        _fail("Fundamental peer symbol cardinality is invalid")
    rows = [identifier(value, label=f"symbols[{index}]") for index, value in enumerate(values)]
    if len(rows) != len(set(rows)) or subject not in rows:
        _fail("Fundamental peer symbols are duplicated or omit the subject")
    return sorted(rows, key=lambda value: value.encode("ascii"))


def _source_metric_row(
    value: Mapping[str, Any],
    *,
    label: str,
    as_of: str,
    source_cutoff: str,
) -> dict[str, Any]:
    row = require_exact_keys(value, _FINANCIAL_INPUT_FIELDS, label=label)
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    if available_at > source_cutoff:
        _fail(f"{label} exceeds its owner-sealed source cutoff")
    source = exact_ref(row["source_ref"], label=f"{label}.source_ref")
    require_no_future(
        available_at=source["available_at"],
        as_of=as_of,
        label=f"{label}.source_ref",
    )
    if (
        source["available_at"] > available_at
        or source["cutoff"] > available_at
        or source["cutoff"] > source_cutoff
    ):
        _fail(f"{label} source chronology is invalid")
    return {
        "available_at": available_at,
        "company_code": identifier(row["company_code"], label=f"{label}.company_code"),
        "metric_id": identifier(row["metric_id"], label=f"{label}.metric_id"),
        "source_kind": "EXACT_PIT_SOURCE",
        "source_ref": source,
        "value": decimal_text(decimal_value(row["value"], label=f"{label}.value")),
    }


def _financial_rows(
    values: Sequence[Mapping[str, Any]], *, symbols: set[str], as_of: str
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("financial_metric_rows must be a sequence")
    if len(values) > 4096:
        _fail("financial_metric_rows exceeds maximum cardinality")
    rows = [
        _source_metric_row(
            value,
            label=f"financial_metric_rows[{index}]",
            as_of=as_of,
            source_cutoff=as_of,
        )
        for index, value in enumerate(values)
    ]
    if any(
        row["company_code"] not in symbols or row["metric_id"] not in FINANCIAL_QUALITY_METRICS
        for row in rows
    ):
        _fail("financial metric input is outside the frozen scorer closure")
    keys = [(row["company_code"], row["metric_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        _fail("financial metric input contains duplicate company/metric rows")
    rows.sort(key=lambda row: (row["company_code"].encode("ascii"), row["metric_id"]))
    return rows


def _policy_by_component(policy: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["component"]): row for row in policy["components"]}


def _scorer_implementation_sha256(value: str) -> str:
    implementation_sha = sha256(value, label="scorer_implementation_sha256")
    if implementation_sha != FUNDAMENTAL_SCORER_IMPLEMENTATION_SHA256_V3:
        _fail("Frozen Fundamental scorer implementation SHA is invalid")
    return implementation_sha


def _component_rows(
    values: Sequence[Mapping[str, Any]],
    *,
    symbols: set[str],
    policy_by_component: Mapping[str, Mapping[str, Any]],
    as_of: str,
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("component_metric_rows must be a sequence")
    if len(values) > 4096:
        _fail("component_metric_rows exceeds maximum cardinality")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        raw = require_exact_keys(
            value, _COMPONENT_INPUT_FIELDS, label=f"component_metric_rows[{index}]"
        )
        component = str(raw["component"])
        if component not in COMPONENTS or component in {
            "industry_cycle",
            "theme_narrative",
        }:
            _fail("I2/I3 projection components cannot be caller-supplied metrics")
        policy_row = policy_by_component[component]
        normalized = _source_metric_row(
            {key: raw[key] for key in _FINANCIAL_INPUT_FIELDS},
            label=f"component_metric_rows[{index}]",
            as_of=as_of,
            source_cutoff=str(policy_row["source_cutoff"]),
        )
        allowed_metrics = {row["metric_id"] for row in policy_row["metric_rows"]}
        if (
            normalized["company_code"] not in symbols
            or normalized["metric_id"] not in allowed_metrics
        ):
            _fail("component metric input is outside the subject/policy closure")
        rows.append({"component": component, **normalized})
    keys = [(row["company_code"], row["component"], row["metric_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        _fail("component metric input contains duplicate company/component/metric rows")
    rows.sort(
        key=lambda row: (
            row["company_code"].encode("ascii"),
            COMPONENTS.index(row["component"]),
            row["metric_id"].encode("ascii"),
        )
    )
    return rows


def _mapping_pair(
    receipts: Mapping[str, Any],
    closures: Mapping[str, Any],
    *,
    symbols: set[str],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(receipts) is not dict or type(closures) is not dict:
        _fail(f"{label} receipts and closures must be exact objects")
    if set(receipts) != set(closures) or not set(receipts) <= symbols:
        _fail(f"{label} receipt/closure domains do not match the peer set")
    return dict(receipts), dict(closures)


def _industry_projection_rows(
    *,
    receipts: Mapping[str, Any],
    closures: Mapping[str, Any],
    symbols: set[str],
    policy_row: Mapping[str, Any],
    as_of: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipt_map, closure_map = _mapping_pair(
        receipts, closures, symbols=symbols, label="industry component"
    )
    rows: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for symbol in sorted(receipt_map, key=lambda value: value.encode("ascii")):
        closure = require_exact_keys(
            closure_map[symbol],
            _INDUSTRY_CLOSURE_FIELDS,
            label=f"industry closure {symbol}",
        )
        receipt = validate_industry_component_receipt(receipt_map[symbol], **closure)
        if receipt["version"] != INDUSTRY_COMPONENT_VERSION:
            _fail("industry component version is invalid")
        if closure["identity_evaluation"].get("subject_id") != symbol:
            _fail("industry component subject binding is invalid")
        if receipt["timestamp"] > as_of or receipt["timestamp"] > policy_row["source_cutoff"]:
            _fail("industry component is future-known")
        reference = content_ref(receipt, identity_field="component_receipt_id")
        refs.append({"company_code": symbol, "receipt_ref": reference})
        if receipt["status"] == "AVAILABLE" and receipt["component_score"] is not None:
            rows.append(
                {
                    "available_at": receipt["timestamp"],
                    "company_code": symbol,
                    "component": "industry_cycle",
                    "metric_id": INDUSTRY_PROJECTION_METRIC,
                    "source_kind": "I2_COMPONENT_RECEIPT",
                    "source_ref": reference,
                    "value": decimal_text(
                        decimal_value(
                            receipt["component_score"],
                            label=f"industry component score {symbol}",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        )
                    ),
                }
            )
    return rows, refs


def _theme_projection_rows(
    *,
    receipts: Mapping[str, Any],
    closures: Mapping[str, Any],
    symbols: set[str],
    policy_row: Mapping[str, Any],
    as_of: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipt_map, closure_map = _mapping_pair(
        receipts, closures, symbols=symbols, label="theme component"
    )
    rows: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for symbol in sorted(receipt_map, key=lambda value: value.encode("ascii")):
        closure = require_exact_keys(
            closure_map[symbol], _THEME_CLOSURE_FIELDS, label=f"theme closure {symbol}"
        )
        receipt = validate_theme_component_receipt(receipt_map[symbol], **closure)
        if receipt["version"] != THEME_COMPONENT_VERSION:
            _fail("theme component version is invalid")
        if closure["exposure_receipt"].get("company_code") != symbol:
            _fail("theme component subject binding is invalid")
        if receipt["timestamp"] > as_of or receipt["timestamp"] > policy_row["source_cutoff"]:
            _fail("theme component is future-known")
        reference = content_ref(receipt, identity_field="component_receipt_id")
        refs.append({"company_code": symbol, "receipt_ref": reference})
        if receipt["status"] == "AVAILABLE" and receipt["component_score"] is not None:
            rows.append(
                {
                    "available_at": receipt["timestamp"],
                    "company_code": symbol,
                    "component": "theme_narrative",
                    "metric_id": THEME_PROJECTION_METRIC,
                    "source_kind": "I3_COMPONENT_RECEIPT",
                    "source_ref": reference,
                    "value": decimal_text(
                        decimal_value(
                            receipt["component_score"],
                            label=f"theme component score {symbol}",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        )
                    ),
                }
            )
    return rows, refs


def _type7(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        _fail("Type-7 percentile requires observations")
    if len(ordered) == 1:
        return ordered[0]
    with localcontext() as context:
        context.prec = 50
        position = Decimal(len(ordered) - 1) * probability
        lower = int(position.to_integral_value(rounding=ROUND_FLOOR))
        fraction = position - Decimal(lower)
        return ordered[lower] + fraction * (
            ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
        )


def _average_tie_percentiles(values: Mapping[str, Decimal]) -> dict[str, Decimal]:
    if not values:
        return {}
    count = Decimal(len(values))
    result: dict[str, Decimal] = {}
    for symbol, target in values.items():
        lower = sum(candidate < target for candidate in values.values())
        tied = sum(candidate == target for candidate in values.values())
        result[symbol] = (Decimal(lower) + (Decimal(tied) + Decimal(1)) / Decimal(2)) / count
    return result


def _metric_projection(
    rows: Sequence[Mapping[str, Any]],
    *,
    lower_probability: Decimal,
    upper_probability: Decimal,
    direction: str,
) -> dict[str, dict[str, Decimal]]:
    raw = {str(row["company_code"]): Decimal(str(row["value"])) for row in rows}
    if not raw:
        return {}
    lower = _type7(list(raw.values()), lower_probability)
    upper = _type7(list(raw.values()), upper_probability)
    winsorized = {symbol: min(max(value, lower), upper) for symbol, value in raw.items()}
    percentiles = _average_tie_percentiles(winsorized)
    if direction == "LOWER_IS_BETTER":
        percentiles = {symbol: Decimal(1) - value for symbol, value in percentiles.items()}
    return {
        symbol: {"projected": percentiles[symbol], "winsorized": winsorized[symbol]}
        for symbol in raw
    }


def _component_projection(
    *,
    policy_row: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    by_metric = {
        str(metric["metric_id"]): [
            row for row in input_rows if row["metric_id"] == metric["metric_id"]
        ]
        for metric in policy_row["metric_rows"]
    }
    projections = {
        str(metric["metric_id"]): _metric_projection(
            by_metric[str(metric["metric_id"])],
            lower_probability=Decimal(str(policy_row["winsor_lower"])),
            upper_probability=Decimal(str(policy_row["winsor_upper"])),
            direction=str(metric["direction"]),
        )
        for metric in policy_row["metric_rows"]
    }
    input_by_key = {(str(row["company_code"]), str(row["metric_id"])): row for row in input_rows}
    results: list[dict[str, Any]] = []
    for symbol in symbols:
        metric_rows: list[dict[str, Any]] = []
        coverage = Decimal("0")
        for metric in policy_row["metric_rows"]:
            metric_id = str(metric["metric_id"])
            projected = projections[metric_id].get(symbol)
            if projected is None:
                continue
            source = input_by_key[(symbol, metric_id)]
            coverage += Decimal(str(metric["weight"]))
            metric_rows.append(
                {
                    "available_at": source["available_at"],
                    "metric_id": metric_id,
                    "policy_weight": metric["weight"],
                    "projected_value": decimal_text(projected["projected"]),
                    "raw_value": source["value"],
                    "source_kind": source["source_kind"],
                    "source_ref": source["source_ref"],
                    "winsorized_value": decimal_text(projected["winsorized"]),
                }
            )
        all_present = len(metric_rows) == len(policy_row["metric_rows"])
        available = coverage >= Decimal(str(policy_row["minimum_coverage"]))
        if policy_row["missing_rule"] == "BLOCK_COMPONENT" and not all_present:
            available = False
        score = None
        if available and coverage > 0:
            score = decimal_text(
                sum(
                    (
                        Decimal(row["projected_value"]) * Decimal(row["policy_weight"])
                        for row in metric_rows
                    ),
                    Decimal("0"),
                )
                / coverage
            )
        results.append(
            {
                "company_code": symbol,
                "component": policy_row["component"],
                "coverage": decimal_text(coverage),
                "metric_rows": metric_rows,
                "score": score,
                "status": "AVAILABLE" if score is not None else "MISSING",
            }
        )
    return results


def _scorer_inputs(
    *,
    financial_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    financial: dict[str, dict[str, str]] = {metric: {} for metric in FINANCIAL_QUALITY_METRICS}
    for row in financial_rows:
        financial[str(row["metric_id"])][str(row["company_code"])] = str(row["value"])
    owner: dict[str, dict[str, dict[str, str]]] = {component: {} for component in COMPONENTS}
    for row in component_rows:
        if row["status"] != "AVAILABLE":
            continue
        source_rows = row["metric_rows"]
        available_at = max(str(source["available_at"]) for source in source_rows)
        owner[str(row["component"])][str(row["company_code"])] = {
            "available_at": available_at,
            "score": str(row["score"]),
        }
    return financial, owner


def _float_audit(value: Any) -> Any:
    if type(value) is float:
        return {"binary_float_repr": repr(value)}
    if type(value) is list:
        return [_float_audit(item) for item in value]
    if type(value) is dict:
        return {str(key): _float_audit(item) for key, item in value.items()}
    return value


def _decimal_projection(value: Any) -> Any:
    if type(value) is float:
        return decimal_text(Decimal(str(value)))
    if type(value) is list:
        return [_decimal_projection(item) for item in value]
    if type(value) is dict:
        return {str(key): _decimal_projection(item) for key, item in value.items()}
    return value


def build_fundamental_profile(
    *,
    company_code: str,
    symbols: Sequence[str],
    policy: Mapping[str, Any],
    financial_metric_rows: Sequence[Mapping[str, Any]],
    component_metric_rows: Sequence[Mapping[str, Any]],
    industry_component_receipts: Mapping[str, Any],
    industry_component_validation_closures: Mapping[str, Any],
    theme_component_receipts: Mapping[str, Any],
    theme_component_validation_closures: Mapping[str, Any],
    scorer_implementation_sha256: str,
    as_of: str,
) -> dict[str, Any]:
    """Build one subject profile from one exact peer-set scorer invocation."""

    issued_at = timestamp(as_of, label="as_of")
    subject = identifier(company_code, label="company_code")
    peers = _symbols(symbols, subject=subject)
    peer_set = set(peers)
    validated_policy = validate_fundamental_component_policy(policy)
    if validated_policy["timestamp"] > issued_at:
        _fail("Fundamental policy is future-known")
    policy_rows = _policy_by_component(validated_policy)
    financial = _financial_rows(financial_metric_rows, symbols=peer_set, as_of=issued_at)
    owner_inputs = _component_rows(
        component_metric_rows,
        symbols=peer_set,
        policy_by_component=policy_rows,
        as_of=issued_at,
    )
    industry_rows, industry_refs = _industry_projection_rows(
        receipts=industry_component_receipts,
        closures=industry_component_validation_closures,
        symbols=peer_set,
        policy_row=policy_rows["industry_cycle"],
        as_of=issued_at,
    )
    theme_rows, theme_refs = _theme_projection_rows(
        receipts=theme_component_receipts,
        closures=theme_component_validation_closures,
        symbols=peer_set,
        policy_row=policy_rows["theme_narrative"],
        as_of=issued_at,
    )
    all_component_inputs = sorted(
        [*owner_inputs, *industry_rows, *theme_rows],
        key=lambda row: (
            row["company_code"].encode("ascii"),
            COMPONENTS.index(row["component"]),
            row["metric_id"].encode("ascii"),
        ),
    )
    projected_components: list[dict[str, Any]] = []
    for component in COMPONENTS:
        projected_components.extend(
            _component_projection(
                policy_row=policy_rows[component],
                input_rows=[row for row in all_component_inputs if row["component"] == component],
                symbols=peers,
            )
        )
    projected_components.sort(
        key=lambda row: (row["company_code"].encode("ascii"), COMPONENTS.index(row["component"]))
    )
    financial_inputs, owner_scores = _scorer_inputs(
        financial_rows=financial,
        component_rows=projected_components,
    )
    raw_result = score_fundamental_forward_v3(
        symbols=peers,
        financial_quality_values=financial_inputs,
        owner_component_scores=owner_scores,
        cutoff=issued_at,
    )
    if raw_result.get("version") != FUNDAMENTAL_SCORING_V3_VERSION:
        _fail("Frozen Fundamental scorer version is invalid")
    matches = [row for row in raw_result.get("records", []) if row.get("symbol") == subject]
    if len(matches) != 1:
        _fail("Frozen Fundamental scorer omitted or duplicated the subject")
    raw_subject = matches[0]
    projected_subject = _decimal_projection(raw_subject)
    if projected_subject["status"] not in PROFILE_STATUSES:
        _fail("Frozen Fundamental subject status is invalid")
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "company_code": subject,
            "peer_symbols": peers,
            "policy_ref": content_ref(validated_policy, identity_field="policy_id"),
            "scorer_implementation_sha256": _scorer_implementation_sha256(
                scorer_implementation_sha256,
            ),
            "scorer_version": FUNDAMENTAL_SCORING_V3_VERSION,
            "financial_metric_rows": financial,
            "component_metric_rows": all_component_inputs,
            "industry_component_refs": industry_refs,
            "theme_component_refs": theme_refs,
            "component_rows": projected_components,
            "component_weights": _decimal_projection(raw_result["component_weights"]),
            "raw_float_audit": {
                "component_weights": _float_audit(raw_result["component_weights"]),
                "subject_record": _float_audit(raw_subject),
            },
            "subject_record": projected_subject,
            "status": projected_subject["status"],
            "raw_score": projected_subject["raw_score"],
            "effective_score": projected_subject["effective_score"],
            "coverage": projected_subject["coverage"],
            "score_present": projected_subject["score_present"],
            "version": PROFILE_VERSION,
        },
        identity_field="profile_id",
    )


def validate_fundamental_profile(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="profile_id")
    require_exact_keys(normalized, _PROFILE_FIELDS, label="FundamentalIntelligenceProfile.v1")
    if (
        normalized["version"] != PROFILE_VERSION
        or normalized["authority"] != NO_AUTHORITY
        or normalized["research_only"] is not True
        or normalized["production"] is not False
    ):
        _fail("Fundamental profile boundary is invalid")
    expected = build_fundamental_profile(**closure)
    if canonical_bytes(normalized) != canonical_bytes(expected):
        _fail("Fundamental profile differs from deterministic replay")
    return normalized


__all__ = ["build_fundamental_profile", "validate_fundamental_profile"]
