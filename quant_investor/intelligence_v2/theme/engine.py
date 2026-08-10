"""Deterministic Theme exposure, component, and risk receipts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    canonical_bytes,
    code,
    common_fields,
    content_ref,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    require_no_future,
    seal,
    timestamp,
    validate_seal,
)
from .contracts import (
    validate_theme_component_policy,
    validate_theme_lifecycle_policy,
    validate_theme_membership_catalog,
    validate_theme_registry,
    validate_theme_risk_policy,
)
from .models import (
    COMPONENT_RECEIPT_VERSION,
    COMPONENT_SOURCE_KINDS,
    EXPOSURE_RECEIPT_VERSION,
    RISK_RECEIPT_VERSION,
    ThemeContractError,
)

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_EXPOSURE_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "exposure_receipt_id",
    "semantic_sha256",
    "company_code",
    "as_of",
    "registry_ref",
    "catalog_ref",
    "lifecycle_policy_ref",
    "status",
    "exposure_rows",
    "cap_bucket_rows",
    "blocker_codes",
}
_METRIC_INPUT_FIELDS: Final = {
    "theme_id",
    "metric_id",
    "normalized_value",
    "available_at",
    "source_kind",
    "source_ref",
}
_COMPONENT_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "component_receipt_id",
    "semantic_sha256",
    "as_of",
    "exposure_ref",
    "component_policy_ref",
    "status",
    "component_score",
    "theme_rows",
    "blocker_codes",
}
_RISK_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "risk_receipt_id",
    "semantic_sha256",
    "as_of",
    "exposure_ref",
    "risk_policy_ref",
    "status",
    "overall_severity",
    "risk_rows",
    "hard_veto_codes",
    "blocker_codes",
}


def _assert_common(document: Mapping[str, Any]) -> None:
    if document.get("authority") != NO_AUTHORITY:
        raise ThemeContractError("theme receipt authority is open")
    if document.get("research_only") is not True or document.get("production") is not False:
        raise ThemeContractError("theme receipt research boundary is open")


def _assert_same(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ThemeContractError(f"{label} differs from deterministic replay")


def _ref(document: Mapping[str, Any], identity_field: str) -> dict[str, str]:
    return content_ref(document, identity_field=identity_field)


def _active(row: Mapping[str, Any], as_of_date: str) -> bool:
    return row["effective_from"] <= as_of_date and (
        row["effective_to"] is None or as_of_date <= row["effective_to"]
    )


def _resolve_precedence_row(
    rows: Sequence[Mapping[str, Any]], precedence: Sequence[str]
) -> tuple[dict[str, Any] | None, bool]:
    if not rows:
        return None, False
    ranks = {provider: index for index, provider in enumerate(precedence)}
    if any(row["provider_id"] not in ranks for row in rows):
        return None, True
    best_rank = min(ranks[row["provider_id"]] for row in rows)
    preferred = [row for row in rows if ranks[row["provider_id"]] == best_rank]
    latest = max(row["available_at"] for row in preferred)
    finalists = [row for row in preferred if row["available_at"] == latest]
    if len(finalists) != 1:
        return None, True
    return dict(finalists[0]), False


def _ancestor_at_level(
    theme_id: str, *, cap_level: int, registry_by_id: Mapping[str, Mapping[str, Any]]
) -> str | None:
    cursor = registry_by_id[theme_id]
    while cursor["level"] > cap_level:
        parent_id = cursor["parent_theme_id"]
        if parent_id is None or parent_id not in registry_by_id:
            return None
        cursor = registry_by_id[parent_id]
    if cursor["level"] != cap_level:
        return None
    return str(cursor["theme_id"])


def _terminal_coverage_projection(
    catalog: Mapping[str, Any], subject: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]] | None:
    coverage = {row["company_code"]: row for row in catalog["coverage_rows"]}.get(subject)
    if catalog["scope_status"] != "COMPLETE" or coverage is None:
        return "UNMAPPED", [], [], ["THEME_CATALOG_UNMAPPED"]
    if coverage["status"] == "UNMAPPED":
        return "UNMAPPED", [], [], ["THEME_CATALOG_UNMAPPED"]
    if coverage["status"] == "AMBIGUOUS":
        return "AMBIGUOUS", [], [], ["THEME_CATALOG_AMBIGUOUS"]
    return None


def _selected_memberships(
    rows: Sequence[Mapping[str, Any]], precedence: Sequence[str]
) -> dict[str, dict[str, Any]] | None:
    by_theme: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_theme[row["theme_id"]].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for theme_id, candidates in by_theme.items():
        winner, conflict = _resolve_precedence_row(candidates, precedence)
        if conflict or winner is None:
            return None
        selected[theme_id] = winner
    return selected


def _selected_lifecycle_rows(
    *,
    theme_ids: Sequence[str],
    registry_by_id: Mapping[str, Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    as_of_date: str,
) -> tuple[dict[str, dict[str, Any]] | None, bool]:
    selected: dict[str, dict[str, Any]] = {}
    retired = False
    for theme_id in theme_ids:
        candidates = [
            row
            for row in lifecycle["lifecycle_rows"]
            if row["theme_id"] == theme_id and _active(row, as_of_date)
        ]
        winner, conflict = _resolve_precedence_row(candidates, lifecycle["provider_precedence"])
        if conflict or winner is None:
            return None, False
        selected[theme_id] = winner
        theme = registry_by_id[theme_id]
        retired = retired or (
            theme["status"] == "RETIRED"
            or not _active(theme, as_of_date)
            or winner["status"] == "RETIRED"
        )
    return selected, retired


def _has_ancestor_overlap(
    theme_ids: Sequence[str], registry_by_id: Mapping[str, Mapping[str, Any]]
) -> bool:
    selected = set(theme_ids)
    for theme_id in theme_ids:
        cursor = registry_by_id[theme_id]["parent_theme_id"]
        while cursor is not None:
            if cursor in selected:
                return True
            cursor = registry_by_id[cursor]["parent_theme_id"]
    return False


def _materialize_exposure_rows(
    *,
    memberships: Mapping[str, Mapping[str, Any]],
    lifecycle_rows: Mapping[str, Mapping[str, Any]],
    registry_by_id: Mapping[str, Mapping[str, Any]],
    cap_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    exposures: list[dict[str, Any]] = []
    bucket_weights: dict[str, Decimal] = defaultdict(Decimal)
    bucket_themes: dict[str, list[str]] = defaultdict(list)
    for theme_id in sorted(memberships, key=lambda value: value.encode("ascii")):
        membership = memberships[theme_id]
        exposures.append(
            {
                "theme_id": theme_id,
                "exposure_basis": membership["exposure_basis"],
                "exposure_weight": membership["exposure_weight"],
                "membership_source_ref": membership["source_ref"],
                "lifecycle_source_ref": lifecycle_rows[theme_id]["source_ref"],
            }
        )
        bucket = _ancestor_at_level(theme_id, cap_level=cap_level, registry_by_id=registry_by_id)
        if bucket is None:
            return None
        bucket_weights[bucket] += Decimal(membership["exposure_weight"])
        bucket_themes[bucket].append(theme_id)
    buckets = [
        {
            "bucket_id": bucket,
            "exposure_weight": decimal_text(bucket_weights[bucket]),
            "theme_ids": sorted(bucket_themes[bucket], key=lambda value: value.encode("ascii")),
        }
        for bucket in sorted(bucket_weights, key=lambda value: value.encode("ascii"))
    ]
    return exposures, buckets


def _resolve_covered_projection(
    *,
    subject: str,
    as_of_date: str,
    catalog: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    registry_doc: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    active_rows = [
        row
        for row in catalog["membership_rows"]
        if row["company_code"] == subject and _active(row, as_of_date)
    ]
    if not active_rows:
        no_theme = {
            "bucket_id": "NO_THEME",
            "exposure_weight": decimal_text(Decimal("1")),
            "theme_ids": [],
        }
        return "NO_MEMBERSHIP", [], [no_theme], []
    memberships = _selected_memberships(active_rows, lifecycle["provider_precedence"])
    if memberships is None:
        return "AMBIGUOUS", [], [], ["THEME_MEMBERSHIP_AMBIGUOUS"]
    registry_by_id = {row["theme_id"]: row for row in registry_doc["themes"]}
    theme_ids = sorted(memberships, key=lambda value: value.encode("ascii"))
    selected_lifecycle, retired = _selected_lifecycle_rows(
        theme_ids=theme_ids,
        registry_by_id=registry_by_id,
        lifecycle=lifecycle,
        as_of_date=as_of_date,
    )
    if selected_lifecycle is None or _has_ancestor_overlap(theme_ids, registry_by_id):
        return "AMBIGUOUS", [], [], ["THEME_MEMBERSHIP_AMBIGUOUS"]
    if retired:
        return "RETIRED", [], [], ["THEME_MEMBERSHIP_RETIRED"]
    total = sum(
        (Decimal(row["exposure_weight"]) for row in memberships.values()),
        Decimal("0"),
    )
    if decimal_text(total) != decimal_text(Decimal("1")):
        return "AMBIGUOUS", [], [], ["THEME_EXPOSURE_WEIGHT_INVALID"]
    materialized = _materialize_exposure_rows(
        memberships=memberships,
        lifecycle_rows=selected_lifecycle,
        registry_by_id=registry_by_id,
        cap_level=lifecycle["cap_level"],
    )
    if materialized is None:
        return "AMBIGUOUS", [], [], ["THEME_CAP_LEVEL_UNRESOLVED"]
    return "AVAILABLE", materialized[0], materialized[1], []


def resolve_theme_exposure(
    *,
    company_code: str,
    registry: Mapping[str, Any],
    membership_catalog: Mapping[str, Any],
    lifecycle_policy: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    registry_doc = validate_theme_registry(registry)
    catalog = validate_theme_membership_catalog(membership_catalog, registry=registry_doc)
    lifecycle = validate_theme_lifecycle_policy(lifecycle_policy, registry=registry_doc)
    issued_at = timestamp(as_of, label="as_of")
    subject = identifier(company_code, label="company_code")
    if any(artifact["timestamp"] > issued_at for artifact in (registry_doc, catalog, lifecycle)):
        raise ThemeContractError("theme exposure contains future closure artifact")
    if catalog["registry_ref"] != _ref(registry_doc, "registry_id"):
        raise ThemeContractError("membership catalog registry binding mismatch")
    if lifecycle["registry_ref"] != _ref(registry_doc, "registry_id"):
        raise ThemeContractError("lifecycle policy registry binding mismatch")

    projection = _terminal_coverage_projection(catalog, subject)
    if projection is None:
        projection = _resolve_covered_projection(
            subject=subject,
            as_of_date=issued_at[:10].replace("-", ""),
            catalog=catalog,
            lifecycle=lifecycle,
            registry_doc=registry_doc,
        )
    status, exposure_rows, cap_rows, blockers = projection

    return seal(
        {
            "version": EXPOSURE_RECEIPT_VERSION,
            **common_fields(timestamp_value=issued_at),
            "company_code": subject,
            "as_of": issued_at,
            "registry_ref": _ref(registry_doc, "registry_id"),
            "catalog_ref": _ref(catalog, "catalog_id"),
            "lifecycle_policy_ref": _ref(lifecycle, "lifecycle_policy_id"),
            "status": status,
            "exposure_rows": exposure_rows,
            "cap_bucket_rows": cap_rows,
            "blocker_codes": sorted(blockers, key=lambda value: value.encode("ascii")),
        },
        identity_field="exposure_receipt_id",
    )


def validate_theme_exposure_receipt(
    document: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    membership_catalog: Mapping[str, Any],
    lifecycle_policy: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    require_exact_keys(document, _EXPOSURE_FIELDS, label="theme exposure receipt")
    validate_seal(document, identity_field="exposure_receipt_id")
    _assert_common(document)
    if document["version"] != EXPOSURE_RECEIPT_VERSION:
        raise ThemeContractError("theme exposure receipt version is invalid")
    expected = resolve_theme_exposure(
        company_code=document["company_code"],
        registry=registry,
        membership_catalog=membership_catalog,
        lifecycle_policy=lifecycle_policy,
        as_of=as_of,
    )
    _assert_same(document, expected, label="theme exposure receipt")
    return expected


def _validate_exposure_closure(
    exposure_receipt: Mapping[str, Any], exposure_closure: Mapping[str, Any]
) -> dict[str, Any]:
    require_exact_keys(
        exposure_closure,
        {"registry", "membership_catalog", "lifecycle_policy", "as_of"},
        label="exposure_closure",
    )
    return validate_theme_exposure_receipt(
        exposure_receipt,
        registry=exposure_closure["registry"],
        membership_catalog=exposure_closure["membership_catalog"],
        lifecycle_policy=exposure_closure["lifecycle_policy"],
        as_of=exposure_closure["as_of"],
    )


def _normalize_metric_row(value: Mapping[str, Any], *, index: int, as_of: str) -> dict[str, Any]:
    label = f"metric_rows[{index}]"
    row = require_exact_keys(value, _METRIC_INPUT_FIELDS, label=label)
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    source = exact_ref(row["source_ref"], label=f"{label}.source_ref")
    require_no_future(available_at=source["available_at"], as_of=as_of, label=f"{label}.source_ref")
    if source["available_at"] > available_at:
        raise ThemeContractError("metric source became available after metric row")
    source_kind = code(row["source_kind"], label=f"{label}.source_kind")
    if source_kind not in COMPONENT_SOURCE_KINDS:
        raise ThemeContractError("theme metric source kind is not allowed")
    return {
        "theme_id": identifier(row["theme_id"], label=f"{label}.theme_id"),
        "metric_id": identifier(row["metric_id"], label=f"{label}.metric_id"),
        "normalized_value": decimal_text(
            decimal_value(
                row["normalized_value"],
                label=f"{label}.normalized_value",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        ),
        "available_at": available_at,
        "source_kind": source_kind,
        "source_ref": source,
    }


def _normalize_metrics(
    metric_rows: Sequence[Mapping[str, Any]], *, as_of: str
) -> list[dict[str, Any]]:
    if isinstance(metric_rows, (str, bytes)) or not isinstance(metric_rows, Sequence):
        raise ThemeContractError("metric_rows must be a sequence")
    if len(metric_rows) > 4096:
        raise ThemeContractError("metric_rows exceeds maximum cardinality")
    rows = [
        _normalize_metric_row(value, index=index, as_of=as_of)
        for index, value in enumerate(metric_rows)
    ]
    keys = [(row["theme_id"], row["metric_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ThemeContractError("theme metric inputs contain duplicate theme/metric rows")
    return sorted(
        rows,
        key=lambda row: (row["theme_id"].encode("ascii"), row["metric_id"].encode("ascii")),
    )


def _project_one_theme(
    *,
    exposure_row: Mapping[str, Any],
    policy: Mapping[str, Any],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    theme_id = exposure_row["theme_id"]
    admitted = [
        (policy_row, metrics_by_key[(theme_id, policy_row["metric_id"])])
        for policy_row in policy["metric_rows"]
        if (theme_id, policy_row["metric_id"]) in metrics_by_key
    ]
    coverage = sum(
        (Decimal(policy_row["weight"]) for policy_row, _metric in admitted),
        Decimal("0"),
    )
    if coverage < Decimal(policy["minimum_coverage"]):
        return None, f"THEME_METRIC_COVERAGE_MISSING:{theme_id}"
    if policy["missing_rule"] == "BLOCK_COMPONENT" and len(admitted) != len(policy["metric_rows"]):
        return None, f"THEME_METRIC_REQUIRED_MISSING:{theme_id}"
    denominator = coverage if policy["missing_rule"] == "DROP_METRIC" else Decimal("1")
    score = Decimal("0")
    projected_rows: list[dict[str, Any]] = []
    for policy_row, metric in admitted:
        raw_value = Decimal(metric["normalized_value"])
        projected = (
            raw_value if policy_row["direction"] == "HIGHER_IS_BETTER" else Decimal("1") - raw_value
        )
        score += projected * Decimal(policy_row["weight"]) / denominator
        projected_rows.append(
            {
                "metric_id": policy_row["metric_id"],
                "normalized_value": metric["normalized_value"],
                "projected_value": decimal_text(projected),
                "policy_weight": policy_row["weight"],
                "source_kind": metric["source_kind"],
                "source_ref": metric["source_ref"],
            }
        )
    return (
        {
            "theme_id": theme_id,
            "exposure_weight": exposure_row["exposure_weight"],
            "metric_coverage": decimal_text(coverage),
            "theme_score": decimal_text(score),
            "metric_rows": projected_rows,
        },
        None,
    )


def _project_available_component(
    *,
    exposure: Mapping[str, Any],
    policy: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, list[dict[str, Any]], list[str]]:
    expected_themes = {row["theme_id"] for row in exposure["exposure_rows"]}
    expected_metrics = {row["metric_id"] for row in policy["metric_rows"]}
    if any(
        row["theme_id"] not in expected_themes or row["metric_id"] not in expected_metrics
        for row in metrics
    ):
        raise ThemeContractError("metric input is outside the exposure/policy closure")
    metrics_by_key = {(row["theme_id"], row["metric_id"]): row for row in metrics}
    rows: list[dict[str, Any]] = []
    total = Decimal("0")
    for exposure_row in exposure["exposure_rows"]:
        projection, blocker = _project_one_theme(
            exposure_row=exposure_row,
            policy=policy,
            metrics_by_key=metrics_by_key,
        )
        if projection is None:
            return "MISSING", None, [], [str(blocker)]
        rows.append(projection)
        total += Decimal(projection["theme_score"]) * Decimal(projection["exposure_weight"])
    if len(metrics) != sum(len(row["metric_rows"]) for row in rows):
        raise ThemeContractError("metric closure contains unused rows")
    return "AVAILABLE", decimal_text(total), rows, []


def _component_projection(
    *,
    exposure: Mapping[str, Any],
    policy: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, list[dict[str, Any]], list[str]]:
    if exposure["status"] == "NO_MEMBERSHIP":
        if metrics:
            raise ThemeContractError("NO_MEMBERSHIP component must not receive metrics")
        return "MISSING", None, [], ["NO_MEMBERSHIP_COMPONENT_MISSING"]
    if exposure["status"] != "AVAILABLE":
        if metrics:
            raise ThemeContractError("blocked theme exposure must not receive metrics")
        return "BLOCKED", None, [], [f"THEME_EXPOSURE_{exposure['status']}"]
    return _project_available_component(exposure=exposure, policy=policy, metrics=metrics)


def build_theme_component_receipt(
    *,
    exposure_receipt: Mapping[str, Any],
    exposure_closure: Mapping[str, Any],
    component_policy: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    exposure = _validate_exposure_closure(exposure_receipt, exposure_closure)
    policy = validate_theme_component_policy(component_policy)
    issued_at = timestamp(as_of, label="as_of")
    if exposure["timestamp"] > issued_at or policy["timestamp"] > issued_at:
        raise ThemeContractError("theme component contains future closure artifact")
    normalized = _normalize_metrics(metric_rows, as_of=issued_at)
    status, component_score, theme_rows, blockers = _component_projection(
        exposure=exposure, policy=policy, metrics=normalized
    )

    return seal(
        {
            "version": COMPONENT_RECEIPT_VERSION,
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "exposure_ref": _ref(exposure, "exposure_receipt_id"),
            "component_policy_ref": _ref(policy, "component_policy_id"),
            "status": status,
            "component_score": component_score,
            "theme_rows": theme_rows,
            "blocker_codes": sorted(blockers, key=lambda value: value.encode("ascii")),
        },
        identity_field="component_receipt_id",
    )


def validate_theme_component_receipt(
    document: Mapping[str, Any],
    *,
    exposure_receipt: Mapping[str, Any],
    exposure_closure: Mapping[str, Any],
    component_policy: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    require_exact_keys(document, _COMPONENT_FIELDS, label="theme component receipt")
    validate_seal(document, identity_field="component_receipt_id")
    _assert_common(document)
    if document["version"] != COMPONENT_RECEIPT_VERSION:
        raise ThemeContractError("theme component receipt version is invalid")
    expected = build_theme_component_receipt(
        exposure_receipt=exposure_receipt,
        exposure_closure=exposure_closure,
        component_policy=component_policy,
        metric_rows=metric_rows,
        as_of=as_of,
    )
    _assert_same(document, expected, label="theme component receipt")
    return expected


def build_theme_risk_receipt(
    *,
    exposure_receipt: Mapping[str, Any],
    exposure_closure: Mapping[str, Any],
    risk_policy: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    require_exact_keys(
        exposure_closure,
        {"registry", "membership_catalog", "lifecycle_policy", "as_of"},
        label="exposure_closure",
    )
    exposure = validate_theme_exposure_receipt(
        exposure_receipt,
        registry=exposure_closure["registry"],
        membership_catalog=exposure_closure["membership_catalog"],
        lifecycle_policy=exposure_closure["lifecycle_policy"],
        as_of=exposure_closure["as_of"],
    )
    policy = validate_theme_risk_policy(risk_policy)
    issued_at = timestamp(as_of, label="as_of")
    if exposure["timestamp"] > issued_at or policy["timestamp"] > issued_at:
        raise ThemeContractError("theme risk contains future closure artifact")

    status = "AVAILABLE"
    severity: str | None = decimal_text(Decimal("0"))
    risk_rows: list[dict[str, Any]] = []
    hard_vetoes: list[str] = []
    blockers: list[str] = []
    if exposure["status"] == "NO_MEMBERSHIP":
        status = "NO_MEMBERSHIP"
        severity = None
    elif exposure["status"] != "AVAILABLE":
        status = "BLOCKED"
        severity = None
        blockers.append(f"THEME_EXPOSURE_{exposure['status']}")
    else:
        max_exposure = Decimal(policy["max_single_theme_exposure"])
        prohibited = set(policy["prohibited_theme_ids"])
        observed_severity = Decimal("0")
        for row in exposure["exposure_rows"]:
            weight = Decimal(row["exposure_weight"])
            reasons: list[str] = []
            veto: str | None = None
            if weight > max_exposure:
                reasons.append("THEME_EXPOSURE_ABOVE_MAX")
            if row["theme_id"] in prohibited:
                reasons.append("PROHIBITED_THEME_EXPOSURE")
                veto = policy["hard_veto_codes_by_theme"][row["theme_id"]]
                hard_vetoes.append(veto)
            if reasons:
                observed_severity = max(observed_severity, weight)
                risk_rows.append(
                    {
                        "theme_id": row["theme_id"],
                        "exposure_weight": row["exposure_weight"],
                        "severity": decimal_text(weight),
                        "reason_codes": sorted(reasons, key=lambda value: value.encode("ascii")),
                        "hard_veto_code": veto,
                    }
                )
        severity = decimal_text(observed_severity)
    return seal(
        {
            "version": RISK_RECEIPT_VERSION,
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "exposure_ref": _ref(exposure, "exposure_receipt_id"),
            "risk_policy_ref": _ref(policy, "risk_policy_id"),
            "status": status,
            "overall_severity": severity,
            "risk_rows": risk_rows,
            "hard_veto_codes": sorted(set(hard_vetoes), key=lambda value: value.encode("ascii")),
            "blocker_codes": sorted(blockers, key=lambda value: value.encode("ascii")),
        },
        identity_field="risk_receipt_id",
    )


def validate_theme_risk_receipt(
    document: Mapping[str, Any],
    *,
    exposure_receipt: Mapping[str, Any],
    exposure_closure: Mapping[str, Any],
    risk_policy: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    require_exact_keys(document, _RISK_FIELDS, label="theme risk receipt")
    validate_seal(document, identity_field="risk_receipt_id")
    _assert_common(document)
    if document["version"] != RISK_RECEIPT_VERSION:
        raise ThemeContractError("theme risk receipt version is invalid")
    expected = build_theme_risk_receipt(
        exposure_receipt=exposure_receipt,
        exposure_closure=exposure_closure,
        risk_policy=risk_policy,
        as_of=as_of,
    )
    _assert_same(document, expected, label="theme risk receipt")
    return expected


__all__ = [
    "build_theme_component_receipt",
    "build_theme_risk_receipt",
    "resolve_theme_exposure",
    "validate_theme_component_receipt",
    "validate_theme_exposure_receipt",
    "validate_theme_risk_receipt",
]
