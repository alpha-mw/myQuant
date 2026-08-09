"""Canonical I3 Theme registries, catalogs, lifecycle, and owner policies."""

from __future__ import annotations

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
    session_date,
    sorted_unique,
    timestamp,
    validate_seal,
)
from .models import (
    CATALOG_SCOPE_STATES,
    COMPONENT_DIRECTIONS,
    COMPONENT_MISSING_RULES,
    COMPONENT_POLICY_VERSION,
    COVERAGE_STATES,
    EXPOSURE_BASES,
    LIFECYCLE_POLICY_VERSION,
    LIFECYCLE_STATES,
    MEMBERSHIP_CATALOG_VERSION,
    REGISTRY_VERSION,
    RISK_POLICY_VERSION,
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
_REGISTRY_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "registry_id",
    "semantic_sha256",
    "as_of",
    "themes",
}
_THEME_ROW_FIELDS: Final = {
    "theme_id",
    "display_name",
    "parent_theme_id",
    "level",
    "status",
    "effective_from",
    "effective_to",
    "available_at",
    "source_ref",
}
_CATALOG_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "catalog_id",
    "semantic_sha256",
    "as_of",
    "registry_ref",
    "scope_status",
    "scope_ref",
    "coverage_rows",
    "membership_rows",
}
_COVERAGE_ROW_FIELDS: Final = {
    "company_code",
    "status",
    "available_at",
    "source_ref",
}
_MEMBERSHIP_ROW_FIELDS: Final = {
    "company_code",
    "theme_id",
    "provider_id",
    "exposure_basis",
    "exposure_weight",
    "effective_from",
    "effective_to",
    "available_at",
    "source_ref",
}
_LIFECYCLE_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "lifecycle_policy_id",
    "semantic_sha256",
    "as_of",
    "registry_ref",
    "provider_precedence",
    "cap_level",
    "lifecycle_rows",
    "owner_policy_ref",
}
_LIFECYCLE_ROW_FIELDS: Final = {
    "theme_id",
    "provider_id",
    "status",
    "effective_from",
    "effective_to",
    "available_at",
    "source_ref",
}
_COMPONENT_POLICY_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "component_policy_id",
    "semantic_sha256",
    "metric_rows",
    "minimum_coverage",
    "missing_rule",
    "owner_policy_ref",
}
_COMPONENT_METRIC_POLICY_FIELDS: Final = {
    "metric_id",
    "direction",
    "weight",
}
_RISK_POLICY_FIELDS: Final = _COMMON_FIELDS | {
    "version",
    "risk_policy_id",
    "semantic_sha256",
    "max_single_theme_exposure",
    "prohibited_theme_ids",
    "hard_veto_codes_by_theme",
    "owner_policy_ref",
}


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ThemeContractError(f"{label} must be nonempty text")
    text = value.strip()
    if len(text.encode("utf-8")) > 4000:
        raise ThemeContractError(f"{label} exceeds 4000 UTF-8 bytes")
    canonical_bytes(text)
    return text


def _optional_session(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return session_date(value, label=label)


def _interval(*, effective_from: Any, effective_to: Any, label: str) -> tuple[str, str | None]:
    start = session_date(effective_from, label=f"{label}.effective_from")
    end = _optional_session(effective_to, label=f"{label}.effective_to")
    if end is not None and end < start:
        raise ThemeContractError(f"{label} effective interval is reversed")
    return start, end


def _source_ref(value: Mapping[str, Any], *, label: str, as_of: str) -> dict[str, str]:
    row = exact_ref(value, label=label)
    require_no_future(available_at=row["available_at"], as_of=as_of, label=label)
    return row


def _source_key(row: Mapping[str, str]) -> tuple[bytes, ...]:
    return tuple(
        row[field].encode("ascii")
        for field in (
            "artifact_id",
            "artifact_version",
            "relative_path",
            "byte_sha256",
            "semantic_sha256",
        )
    )


def _assert_common(document: Mapping[str, Any]) -> None:
    if document.get("authority") != NO_AUTHORITY:
        raise ThemeContractError("theme artifact authority is open")
    if document.get("research_only") is not True or document.get("production") is not False:
        raise ThemeContractError("theme artifact research boundary is open")


def _assert_same(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ThemeContractError(f"{label} differs from deterministic replay")


def _sequence(value: Any, *, label: str, maximum: int, allow_empty: bool) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ThemeContractError(f"{label} must be a sequence")
    if len(value) > maximum or (not allow_empty and not value):
        raise ThemeContractError(f"{label} cardinality is invalid")
    return value


def _ordered_unique_identifiers(values: Any, *, label: str, maximum: int) -> list[str]:
    sequence = _sequence(values, label=label, maximum=maximum, allow_empty=False)
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(sequence)]
    if len(rows) != len(set(rows)):
        raise ThemeContractError(f"{label} contains duplicate identities")
    return rows


def _normalize_theme_row(value: Mapping[str, Any], *, index: int, as_of: str) -> dict[str, Any]:
    label = f"themes[{index}]"
    row = require_exact_keys(value, _THEME_ROW_FIELDS, label=label)
    theme_id = identifier(row["theme_id"], label=f"{label}.theme_id")
    parent = row["parent_theme_id"]
    if parent is not None:
        parent = identifier(parent, label=f"{label}.parent_theme_id")
    if parent == theme_id:
        raise ThemeContractError("theme cannot parent itself")
    if type(row["level"]) is not int or not 0 <= row["level"] <= 32:
        raise ThemeContractError("theme level is invalid")
    status = code(row["status"], label=f"{label}.status")
    if status not in LIFECYCLE_STATES:
        raise ThemeContractError("theme registry status is invalid")
    start, end = _interval(
        effective_from=row["effective_from"],
        effective_to=row["effective_to"],
        label=label,
    )
    if status == "RETIRED" and end is None:
        raise ThemeContractError("retired theme must have effective_to")
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    source = _source_ref(row["source_ref"], label=f"{label}.source_ref", as_of=as_of)
    if source["available_at"] > available_at:
        raise ThemeContractError("theme source became available after theme row")
    return {
        "theme_id": theme_id,
        "display_name": _text(row["display_name"], label=f"{label}.display_name"),
        "parent_theme_id": parent,
        "level": row["level"],
        "status": status,
        "effective_from": start,
        "effective_to": end,
        "available_at": available_at,
        "source_ref": source,
    }


def _validate_theme_hierarchy(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [row["theme_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ThemeContractError("theme registry contains duplicate theme IDs")
    by_id = {row["theme_id"]: row for row in rows}
    for row in rows:
        parent_id = row["parent_theme_id"]
        if parent_id is None:
            if row["level"] != 0:
                raise ThemeContractError("root themes must be level zero")
            continue
        parent = by_id.get(parent_id)
        if parent is None or parent["level"] + 1 != row["level"]:
            raise ThemeContractError("theme parent hierarchy is incomplete")
        seen = {row["theme_id"]}
        while parent is not None:
            if parent["theme_id"] in seen:
                raise ThemeContractError("theme registry contains a hierarchy cycle")
            seen.add(parent["theme_id"])
            parent = by_id.get(parent["parent_theme_id"])


def _normalize_coverage_row(value: Mapping[str, Any], *, index: int, as_of: str) -> dict[str, Any]:
    label = f"coverage_rows[{index}]"
    row = require_exact_keys(value, _COVERAGE_ROW_FIELDS, label=label)
    status = code(row["status"], label=f"{label}.status")
    if status not in COVERAGE_STATES:
        raise ThemeContractError("coverage row status is invalid")
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    source = _source_ref(row["source_ref"], label=f"{label}.source_ref", as_of=as_of)
    if source["available_at"] > available_at:
        raise ThemeContractError("coverage source became available after coverage row")
    return {
        "company_code": identifier(row["company_code"], label=f"{label}.company_code"),
        "status": status,
        "available_at": available_at,
        "source_ref": source,
    }


def _normalize_membership_row(
    value: Mapping[str, Any],
    *,
    index: int,
    as_of: str,
    theme_ids: set[str],
) -> dict[str, Any]:
    label = f"membership_rows[{index}]"
    row = require_exact_keys(value, _MEMBERSHIP_ROW_FIELDS, label=label)
    theme_id = identifier(row["theme_id"], label=f"{label}.theme_id")
    if theme_id not in theme_ids:
        raise ThemeContractError("membership references unknown theme")
    basis = code(row["exposure_basis"], label=f"{label}.exposure_basis")
    if basis not in EXPOSURE_BASES:
        raise ThemeContractError("membership exposure basis is invalid")
    weight = decimal_value(
        row["exposure_weight"],
        label=f"{label}.exposure_weight",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if weight <= 0:
        raise ThemeContractError("membership exposure weight must be positive")
    start, end = _interval(
        effective_from=row["effective_from"], effective_to=row["effective_to"], label=label
    )
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    source = _source_ref(row["source_ref"], label=f"{label}.source_ref", as_of=as_of)
    if source["available_at"] > available_at:
        raise ThemeContractError("membership source became available after membership row")
    return {
        "company_code": identifier(row["company_code"], label=f"{label}.company_code"),
        "theme_id": theme_id,
        "provider_id": identifier(row["provider_id"], label=f"{label}.provider_id"),
        "exposure_basis": basis,
        "exposure_weight": decimal_text(weight),
        "effective_from": start,
        "effective_to": end,
        "available_at": available_at,
        "source_ref": source,
    }


def _normalize_lifecycle_row(
    value: Mapping[str, Any],
    *,
    index: int,
    as_of: str,
    theme_ids: set[str],
    providers: set[str],
) -> dict[str, Any]:
    label = f"lifecycle_rows[{index}]"
    row = require_exact_keys(value, _LIFECYCLE_ROW_FIELDS, label=label)
    theme_id = identifier(row["theme_id"], label=f"{label}.theme_id")
    provider_id = identifier(row["provider_id"], label=f"{label}.provider_id")
    if theme_id not in theme_ids or provider_id not in providers:
        raise ThemeContractError("lifecycle identity is outside policy closure")
    status = code(row["status"], label=f"{label}.status")
    if status not in LIFECYCLE_STATES:
        raise ThemeContractError("lifecycle status is invalid")
    start, end = _interval(
        effective_from=row["effective_from"], effective_to=row["effective_to"], label=label
    )
    available_at = timestamp(row["available_at"], label=f"{label}.available_at")
    require_no_future(available_at=available_at, as_of=as_of, label=label)
    source = _source_ref(row["source_ref"], label=f"{label}.source_ref", as_of=as_of)
    if source["available_at"] > available_at:
        raise ThemeContractError("lifecycle source became available after lifecycle row")
    return {
        "theme_id": theme_id,
        "provider_id": provider_id,
        "status": status,
        "effective_from": start,
        "effective_to": end,
        "available_at": available_at,
        "source_ref": source,
    }


def build_theme_registry(*, themes: Sequence[Mapping[str, Any]], as_of: str) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    rows = [
        _normalize_theme_row(value, index=index, as_of=cutoff)
        for index, value in enumerate(
            _sequence(themes, label="themes", maximum=1024, allow_empty=False)
        )
    ]
    _validate_theme_hierarchy(rows)
    rows.sort(key=lambda row: row["theme_id"].encode("ascii"))
    return seal(
        {
            "version": REGISTRY_VERSION,
            **common_fields(timestamp_value=cutoff),
            "as_of": cutoff,
            "themes": rows,
        },
        identity_field="registry_id",
    )


def validate_theme_registry(document: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(document, _REGISTRY_FIELDS, label="theme registry")
    validate_seal(document, identity_field="registry_id")
    _assert_common(document)
    if document["version"] != REGISTRY_VERSION or document["timestamp"] != document["as_of"]:
        raise ThemeContractError("theme registry version or timestamp is invalid")
    expected = build_theme_registry(themes=document["themes"], as_of=document["as_of"])
    _assert_same(document, expected, label="theme registry")
    return expected


def build_theme_membership_catalog(
    *,
    registry: Mapping[str, Any],
    scope_status: str,
    scope_ref: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    registry_doc = validate_theme_registry(registry)
    cutoff = timestamp(as_of, label="as_of")
    if cutoff < registry_doc["as_of"]:
        raise ThemeContractError("membership catalog predates registry")
    resolved_scope = code(scope_status, label="scope_status")
    if resolved_scope not in CATALOG_SCOPE_STATES:
        raise ThemeContractError("catalog scope status is invalid")
    scope_source = _source_ref(scope_ref, label="scope_ref", as_of=cutoff)
    coverage = [
        _normalize_coverage_row(value, index=index, as_of=cutoff)
        for index, value in enumerate(
            _sequence(coverage_rows, label="coverage_rows", maximum=10000, allow_empty=True)
        )
    ]
    coverage_ids = [row["company_code"] for row in coverage]
    if len(coverage_ids) != len(set(coverage_ids)):
        raise ThemeContractError("coverage rows contain duplicate companies")
    theme_ids = {row["theme_id"] for row in registry_doc["themes"]}
    memberships = [
        _normalize_membership_row(value, index=index, as_of=cutoff, theme_ids=theme_ids)
        for index, value in enumerate(
            _sequence(
                membership_rows,
                label="membership_rows",
                maximum=50000,
                allow_empty=True,
            )
        )
    ]
    coverage_by_company = {row["company_code"]: row["status"] for row in coverage}
    if any(coverage_by_company.get(row["company_code"]) != "COVERED" for row in memberships):
        raise ThemeContractError("membership rows require an exact COVERED subject row")
    membership_keys = [
        (
            row["company_code"],
            row["theme_id"],
            row["provider_id"],
            row["effective_from"],
            row["effective_to"],
            row["available_at"],
            _source_key(row["source_ref"]),
        )
        for row in memberships
    ]
    if len(membership_keys) != len(set(membership_keys)):
        raise ThemeContractError("membership catalog contains duplicate rows")
    coverage.sort(key=lambda row: row["company_code"].encode("ascii"))
    memberships.sort(
        key=lambda row: (
            row["company_code"].encode("ascii"),
            row["theme_id"].encode("ascii"),
            row["provider_id"].encode("ascii"),
            row["effective_from"].encode("ascii"),
            row["available_at"].encode("ascii"),
            _source_key(row["source_ref"]),
            canonical_bytes(row),
        )
    )
    return seal(
        {
            "version": MEMBERSHIP_CATALOG_VERSION,
            **common_fields(timestamp_value=cutoff),
            "as_of": cutoff,
            "registry_ref": _content_ref_of(registry_doc, "registry_id"),
            "scope_status": resolved_scope,
            "scope_ref": scope_source,
            "coverage_rows": coverage,
            "membership_rows": memberships,
        },
        identity_field="catalog_id",
    )


def validate_theme_membership_catalog(
    document: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    require_exact_keys(document, _CATALOG_FIELDS, label="theme membership catalog")
    validate_seal(document, identity_field="catalog_id")
    _assert_common(document)
    if (
        document["version"] != MEMBERSHIP_CATALOG_VERSION
        or document["timestamp"] != document["as_of"]
    ):
        raise ThemeContractError("theme membership catalog version or timestamp is invalid")
    expected = build_theme_membership_catalog(
        registry=registry,
        scope_status=document["scope_status"],
        scope_ref=document["scope_ref"],
        coverage_rows=document["coverage_rows"],
        membership_rows=document["membership_rows"],
        as_of=document["as_of"],
    )
    _assert_same(document, expected, label="theme membership catalog")
    return expected


def build_theme_lifecycle_policy(
    *,
    registry: Mapping[str, Any],
    provider_precedence: Sequence[str],
    cap_level: int,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    owner_policy_ref: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    registry_doc = validate_theme_registry(registry)
    cutoff = timestamp(as_of, label="as_of")
    precedence = _ordered_unique_identifiers(
        provider_precedence,
        label="provider_precedence",
        maximum=64,
    )
    if type(cap_level) is not int or not 0 <= cap_level <= 32:
        raise ThemeContractError("cap_level is invalid")
    theme_ids = {row["theme_id"] for row in registry_doc["themes"]}
    rows = [
        _normalize_lifecycle_row(
            value,
            index=index,
            as_of=cutoff,
            theme_ids=theme_ids,
            providers=set(precedence),
        )
        for index, value in enumerate(
            _sequence(
                lifecycle_rows,
                label="lifecycle_rows",
                maximum=4096,
                allow_empty=False,
            )
        )
    ]
    keys = [
        (
            row["theme_id"],
            row["provider_id"],
            row["effective_from"],
            row["effective_to"],
            row["available_at"],
            _source_key(row["source_ref"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ThemeContractError("lifecycle policy contains duplicate rows")
    rows.sort(
        key=lambda row: (
            row["theme_id"].encode("ascii"),
            row["provider_id"].encode("ascii"),
            row["effective_from"].encode("ascii"),
            row["available_at"].encode("ascii"),
            _source_key(row["source_ref"]),
            canonical_bytes(row),
        )
    )
    owner_ref = _source_ref(owner_policy_ref, label="owner_policy_ref", as_of=cutoff)
    return seal(
        {
            "version": LIFECYCLE_POLICY_VERSION,
            **common_fields(timestamp_value=cutoff),
            "as_of": cutoff,
            "registry_ref": _content_ref_of(registry_doc, "registry_id"),
            "provider_precedence": precedence,
            "cap_level": cap_level,
            "lifecycle_rows": rows,
            "owner_policy_ref": owner_ref,
        },
        identity_field="lifecycle_policy_id",
    )


def validate_theme_lifecycle_policy(
    document: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    require_exact_keys(document, _LIFECYCLE_FIELDS, label="theme lifecycle policy")
    validate_seal(document, identity_field="lifecycle_policy_id")
    _assert_common(document)
    if document["version"] != LIFECYCLE_POLICY_VERSION:
        raise ThemeContractError("theme lifecycle policy version is invalid")
    expected = build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=document["provider_precedence"],
        cap_level=document["cap_level"],
        lifecycle_rows=document["lifecycle_rows"],
        owner_policy_ref=document["owner_policy_ref"],
        as_of=document["as_of"],
    )
    _assert_same(document, expected, label="theme lifecycle policy")
    return expected


def build_theme_component_policy(
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    minimum_coverage: Any,
    missing_rule: str,
    owner_policy_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    created = timestamp(created_at, label="created_at")
    if isinstance(metric_rows, (str, bytes)) or not isinstance(metric_rows, Sequence):
        raise ThemeContractError("metric_rows must be a sequence")
    if not metric_rows or len(metric_rows) > 64:
        raise ThemeContractError("metric_rows cardinality is invalid")
    rows: list[dict[str, str]] = []
    for index, value in enumerate(metric_rows):
        row = require_exact_keys(
            value, _COMPONENT_METRIC_POLICY_FIELDS, label=f"metric_rows[{index}]"
        )
        direction = code(row["direction"], label=f"metric_rows[{index}].direction")
        if direction not in COMPONENT_DIRECTIONS:
            raise ThemeContractError("component metric direction is invalid")
        weight = decimal_value(
            row["weight"],
            label=f"metric_rows[{index}].weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if weight <= 0:
            raise ThemeContractError("component metric weight must be positive")
        rows.append(
            {
                "metric_id": identifier(row["metric_id"], label=f"metric_rows[{index}].metric_id"),
                "direction": direction,
                "weight": decimal_text(weight),
            }
        )
    metric_ids = [row["metric_id"] for row in rows]
    if len(metric_ids) != len(set(metric_ids)):
        raise ThemeContractError("component policy contains duplicate metric IDs")
    if rows != sorted(rows, key=lambda row: row["metric_id"].encode("ascii")):
        raise ThemeContractError("component metric policy rows must be ASCII sorted")
    total = sum((Decimal(row["weight"]) for row in rows), Decimal("0"))
    if decimal_text(total) != decimal_text(Decimal("1")):
        raise ThemeContractError("component metric weights must sum exactly to one")
    coverage = decimal_value(
        minimum_coverage,
        label="minimum_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    rule = code(missing_rule, label="missing_rule")
    if rule not in COMPONENT_MISSING_RULES:
        raise ThemeContractError("component missing rule is invalid")
    owner_ref = _source_ref(owner_policy_ref, label="owner_policy_ref", as_of=created)
    return seal(
        {
            "version": COMPONENT_POLICY_VERSION,
            **common_fields(timestamp_value=created),
            "metric_rows": rows,
            "minimum_coverage": decimal_text(coverage),
            "missing_rule": rule,
            "owner_policy_ref": owner_ref,
        },
        identity_field="component_policy_id",
    )


def validate_theme_component_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(document, _COMPONENT_POLICY_FIELDS, label="theme component policy")
    validate_seal(document, identity_field="component_policy_id")
    _assert_common(document)
    if document["version"] != COMPONENT_POLICY_VERSION:
        raise ThemeContractError("theme component policy version is invalid")
    expected = build_theme_component_policy(
        metric_rows=document["metric_rows"],
        minimum_coverage=document["minimum_coverage"],
        missing_rule=document["missing_rule"],
        owner_policy_ref=document["owner_policy_ref"],
        created_at=document["timestamp"],
    )
    _assert_same(document, expected, label="theme component policy")
    return expected


def build_theme_risk_policy(
    *,
    max_single_theme_exposure: Any,
    prohibited_theme_ids: Sequence[str],
    hard_veto_codes_by_theme: Mapping[str, str],
    owner_policy_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    created = timestamp(created_at, label="created_at")
    cap = decimal_value(
        max_single_theme_exposure,
        label="max_single_theme_exposure",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    prohibited = sorted_unique(
        prohibited_theme_ids,
        label="prohibited_theme_ids",
        maximum=256,
        allow_empty=True,
    )
    if type(hard_veto_codes_by_theme) is not dict:
        raise ThemeContractError("hard_veto_codes_by_theme must be a mapping")
    if set(hard_veto_codes_by_theme) != set(prohibited):
        raise ThemeContractError("every prohibited theme must have one hard veto code")
    vetoes = {
        theme_id: code(
            hard_veto_codes_by_theme[theme_id],
            label=f"hard_veto_codes_by_theme.{theme_id}",
        )
        for theme_id in prohibited
    }
    if len(set(vetoes.values())) != len(vetoes):
        raise ThemeContractError("hard veto codes must be unique")
    owner_ref = _source_ref(owner_policy_ref, label="owner_policy_ref", as_of=created)
    return seal(
        {
            "version": RISK_POLICY_VERSION,
            **common_fields(timestamp_value=created),
            "max_single_theme_exposure": decimal_text(cap),
            "prohibited_theme_ids": prohibited,
            "hard_veto_codes_by_theme": vetoes,
            "owner_policy_ref": owner_ref,
        },
        identity_field="risk_policy_id",
    )


def validate_theme_risk_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(document, _RISK_POLICY_FIELDS, label="theme risk policy")
    validate_seal(document, identity_field="risk_policy_id")
    _assert_common(document)
    if document["version"] != RISK_POLICY_VERSION:
        raise ThemeContractError("theme risk policy version is invalid")
    expected = build_theme_risk_policy(
        max_single_theme_exposure=document["max_single_theme_exposure"],
        prohibited_theme_ids=document["prohibited_theme_ids"],
        hard_veto_codes_by_theme=document["hard_veto_codes_by_theme"],
        owner_policy_ref=document["owner_policy_ref"],
        created_at=document["timestamp"],
    )
    _assert_same(document, expected, label="theme risk policy")
    return expected


def _content_ref_of(document: Mapping[str, Any], identity_field: str) -> dict[str, str]:
    return content_ref(document, identity_field=identity_field)


__all__ = [
    "build_theme_component_policy",
    "build_theme_lifecycle_policy",
    "build_theme_membership_catalog",
    "build_theme_registry",
    "build_theme_risk_policy",
    "validate_theme_component_policy",
    "validate_theme_lifecycle_policy",
    "validate_theme_membership_catalog",
    "validate_theme_registry",
    "validate_theme_risk_policy",
]
