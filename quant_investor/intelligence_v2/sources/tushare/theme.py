"""Deterministic DC-to-TDX Theme source compiler for the existing I3 contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Final

from ..._core import (
    NO_AUTHORITY,
    canonical_bytes,
    common_fields,
    content_ref,
    exact_ref,
    require_exact_keys,
    seal,
    session_date,
    timestamp,
    validate_seal,
)
from ...theme import (
    build_theme_lifecycle_policy,
    build_theme_membership_catalog,
    build_theme_registry,
    validate_theme_lifecycle_policy,
    validate_theme_membership_catalog,
    validate_theme_registry,
)
from ...theme.models import ThemeContractError

DC_PROVIDER: Final = "TUSHARE_DC"
TDX_PROVIDER: Final = "TUSHARE_TDX"
SOURCE_RECEIPT_VERSION_V1: Final = "myquant.v17.intelligence-v2.tushare-theme-source-receipt.v1"
SOURCE_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.tushare-theme-source-receipt.v2"
CATALOG_SHARD_MEMBERSHIP_LIMIT: Final = 9000
_QUANTUM: Final = Decimal("0.000000000001")
_DC_REGISTRY_FIELDS: Final = {"ts_code", "trade_date", "name", "idx_type", "level"}
_DC_MEMBER_FIELDS: Final = {"trade_date", "ts_code", "con_code", "name"}
_TDX_REGISTRY_FIELDS: Final = {"ts_code", "trade_date", "name", "idx_type", "idx_count"}
_TDX_MEMBER_FIELDS: Final = {"ts_code", "trade_date", "con_code", "con_name"}
_CAPTURE_FIELDS: Final = {"status", "rows", "source_ref"}
_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_RECEIPT_FIELDS_V1: Final = _COMMON_FIELDS | {
    "version",
    "source_receipt_id",
    "semantic_sha256",
    "trade_date",
    "company_keyset",
    "tdx_fallback_company_keyset",
    "company_rows",
    "registry_ref",
    "catalog_ref",
    "lifecycle_policy_ref",
}
_RECEIPT_FIELDS: Final = (_RECEIPT_FIELDS_V1 - {"catalog_ref"}) | {"catalog_rows"}
_COMPANY_RECEIPT_FIELDS: Final = {"company_code", "provider", "status", "theme_ids"}
_CATALOG_RECEIPT_FIELDS: Final = {
    "catalog_ref",
    "company_count",
    "first_company",
    "last_company",
}


def _fail(message: str) -> None:
    raise ThemeContractError(message)


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        _fail(f"{label} is invalid")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    try:
        return timestamp(value, label=label)
    except ThemeContractError:
        raise
    except Exception as exc:
        raise ThemeContractError(str(exc)) from exc


def _session(value: Any, *, label: str) -> str:
    try:
        return session_date(value, label=label)
    except ThemeContractError:
        raise
    except Exception as exc:
        raise ThemeContractError(str(exc)) from exc


def _source_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return exact_ref(value, label=label)
    except ThemeContractError:
        raise
    except Exception as exc:
        raise ThemeContractError(str(exc)) from exc


def _keyset(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        _fail("company_keyset must be a nonempty sequence")
    rows = [_identifier(item, label="company_keyset") for item in value]
    if len(rows) != len(set(rows)) or rows != sorted(rows, key=lambda item: item.encode("ascii")):
        _fail("company_keyset must be unique and ASCII sorted")
    return rows


def _captured_rows(
    value: Mapping[str, Any],
    *,
    fields: set[str],
    limit: int,
    label: str,
) -> tuple[bool, list[dict[str, Any]], dict[str, str]]:
    if type(value) is not dict or set(value) != _CAPTURE_FIELDS:
        _fail(f"{label} capture shape is invalid")
    if value["status"] not in {"COMPLETE", "INCOMPLETE"}:
        _fail(f"{label} capture status is invalid")
    raw_rows = value["rows"]
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        _fail(f"{label}.rows must be a sequence")
    rows: list[dict[str, Any]] = []
    valid_shape = True
    for raw in raw_rows:
        if type(raw) is not dict or set(raw) != fields:
            valid_shape = False
            continue
        rows.append(dict(raw))
    complete = value["status"] == "COMPLETE" and valid_shape and len(raw_rows) < limit
    return complete, rows, _source_ref(value["source_ref"], label=f"{label}.source_ref")


def _registry_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    trade_date: str,
    expected_fields: set[str],
    expected_type: str,
    source_ref: Mapping[str, Any],
    captured_at: str,
    limit: int,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    if len(rows) >= limit:
        _fail(f"{provider} registry row limit was reached")
    for raw in rows:
        if type(raw) is not dict or set(raw) != expected_fields:
            _fail(f"{provider} registry row shape is invalid")
        code = _identifier(raw["ts_code"], label=f"{provider}.ts_code")
        if raw["trade_date"] != trade_date or raw["idx_type"] != expected_type:
            _fail(f"{provider} registry scope mismatch")
        name = raw["name"]
        if type(name) is not str or not name:
            _fail(f"{provider} registry name is invalid")
        theme_id = f"{provider}:{code}"
        candidate = {
            "theme_id": theme_id,
            "display_name": name,
            "parent_theme_id": None,
            "level": 0,
            "status": "ACTIVE",
            "effective_from": trade_date,
            "effective_to": trade_date,
            "available_at": captured_at,
            "source_ref": dict(source_ref),
        }
        previous = by_id.get(theme_id)
        if previous is not None and canonical_bytes(previous) != canonical_bytes(candidate):
            conflicts.add(theme_id)
        else:
            by_id[theme_id] = candidate
    return by_id, conflicts


def _member_theme_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    company: str,
    trade_date: str,
    fields: set[str],
) -> tuple[list[str], set[str], bool]:
    by_code: dict[str, bytes] = {}
    conflicts: set[str] = set()
    for row in rows:
        if row.get("trade_date") != trade_date or row.get("con_code") != company:
            return [], set(), True
        code = _identifier(row.get("ts_code"), label=f"{provider}.membership.ts_code")
        identity = canonical_bytes(row)
        if code in by_code and by_code[code] != identity:
            conflicts.add(f"{provider}:{code}")
        by_code[code] = identity
    theme_ids = sorted(
        (f"{provider}:{code}" for code in by_code), key=lambda item: item.encode("ascii")
    )
    return theme_ids, conflicts, False


def _target_registry_ids(
    theme_ids: Sequence[str],
    *,
    registry_ids: set[str],
) -> list[str]:
    """Project an untyped member response onto the sealed target registry."""

    return [theme_id for theme_id in theme_ids if theme_id in registry_ids]


def _weights(theme_ids: Sequence[str]) -> dict[str, str]:
    if not theme_ids:
        return {}
    base = (Decimal(1) / Decimal(len(theme_ids))).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)
    values = [base for _ in theme_ids[:-1]]
    values.append(Decimal("1.000000000000") - sum(values, Decimal(0)))
    if any(value <= 0 for value in values):
        _fail("equal-membership exposure produced a nonpositive weight")
    return {theme_id: f"{value:.12f}" for theme_id, value in zip(theme_ids, values)}


def _select_company(
    *,
    company: str,
    trade_date: str,
    dc_capture: Mapping[str, Any],
    tdx_capture: Mapping[str, Any] | None,
    registry_ids: set[str],
    registry_conflicts: set[str],
) -> dict[str, Any]:
    dc_complete, dc_rows, dc_ref = _captured_rows(
        dc_capture,
        fields=_DC_MEMBER_FIELDS,
        limit=5000,
        label=f"dc_member.{company}",
    )
    dc_ids, dc_conflicts, dc_scope_mismatch = _member_theme_ids(
        dc_rows,
        provider=DC_PROVIDER,
        company=company,
        trade_date=trade_date,
        fields=_DC_MEMBER_FIELDS,
    )
    dc_target_ids = _target_registry_ids(dc_ids, registry_ids=registry_ids)
    dc_complete = dc_complete and not dc_scope_mismatch and not (dc_conflicts & set(dc_target_ids))
    if tdx_capture is None and dc_complete:
        outside_only = bool(dc_ids) and not dc_target_ids
        conflict = any(theme_id in registry_conflicts for theme_id in dc_target_ids)
        if outside_only or conflict:
            return {
                "provider": DC_PROVIDER,
                "source_ref": dc_ref,
                "status": "AMBIGUOUS",
                "themes": [],
            }
        return {
            "provider": DC_PROVIDER,
            "source_ref": dc_ref,
            "status": "COVERED",
            "themes": dc_target_ids,
        }
    if tdx_capture is None:
        return {"provider": None, "source_ref": dc_ref, "status": "UNMAPPED", "themes": []}
    tdx_complete, tdx_rows, tdx_ref = _captured_rows(
        tdx_capture,
        fields=_TDX_MEMBER_FIELDS,
        limit=3000,
        label=f"tdx_member.{company}",
    )
    tdx_ids, tdx_conflicts, tdx_scope_mismatch = _member_theme_ids(
        tdx_rows,
        provider=TDX_PROVIDER,
        company=company,
        trade_date=trade_date,
        fields=_TDX_MEMBER_FIELDS,
    )
    if not tdx_complete or tdx_scope_mismatch:
        return {"provider": TDX_PROVIDER, "source_ref": tdx_ref, "status": "UNMAPPED", "themes": []}
    tdx_target_ids = _target_registry_ids(tdx_ids, registry_ids=registry_ids)
    outside_only = bool(tdx_ids) and not tdx_target_ids
    conflict = bool(tdx_conflicts & set(tdx_target_ids)) or any(
        theme_id in registry_conflicts for theme_id in tdx_target_ids
    )
    if outside_only or conflict:
        return {
            "provider": TDX_PROVIDER,
            "source_ref": tdx_ref,
            "status": "AMBIGUOUS",
            "themes": [],
        }
    return {
        "provider": TDX_PROVIDER,
        "source_ref": tdx_ref,
        "status": "COVERED",
        "themes": tdx_target_ids,
    }


def _artifact_rows(
    *,
    company_rows: Sequence[Mapping[str, Any]],
    trade_date: str,
    captured_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    for row in company_rows:
        coverage.append(
            {
                "company_code": row["company_code"],
                "status": row["status"],
                "available_at": captured_at,
                "source_ref": row["source_ref"],
            }
        )
        for theme_id, weight in _weights(row["themes"]).items():
            memberships.append(
                {
                    "company_code": row["company_code"],
                    "theme_id": theme_id,
                    "provider_id": row["provider"],
                    "exposure_basis": "EQUAL_MEMBERSHIP",
                    "exposure_weight": weight,
                    "effective_from": trade_date,
                    "effective_to": trade_date,
                    "available_at": captured_at,
                    "source_ref": row["source_ref"],
                }
            )
    return coverage, memberships


def _catalog_shards(
    *,
    company_rows: Sequence[Mapping[str, Any]],
    trade_date: str,
    captured_at: str,
    registry: Mapping[str, Any],
    scope_ref: Mapping[str, Any],
    as_of: str,
) -> list[dict[str, Any]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    membership_count = 0
    for row in company_rows:
        row_count = len(row["themes"])
        if row_count > CATALOG_SHARD_MEMBERSHIP_LIMIT:
            _fail("one company exceeds the theme catalog shard limit")
        if current and membership_count + row_count > CATALOG_SHARD_MEMBERSHIP_LIMIT:
            groups.append(current)
            current = []
            membership_count = 0
        current.append(row)
        membership_count += row_count
    if current:
        groups.append(current)
    catalogs: list[dict[str, Any]] = []
    for group in groups:
        coverage_rows, membership_rows = _artifact_rows(
            company_rows=group,
            trade_date=trade_date,
            captured_at=captured_at,
        )
        catalog = build_theme_membership_catalog(
            registry=registry,
            scope_status="COMPLETE",
            scope_ref=scope_ref,
            coverage_rows=coverage_rows,
            membership_rows=membership_rows,
            as_of=as_of,
        )
        catalogs.append(validate_theme_membership_catalog(catalog, registry=registry))
    if not catalogs:
        _fail("theme catalog shard set is empty")
    return catalogs


def _source_receipt(
    *,
    trade_date: str,
    company_keyset: Sequence[str],
    fallback_keyset: Sequence[str],
    company_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    catalogs: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    return seal(
        {
            "version": SOURCE_RECEIPT_VERSION,
            **common_fields(timestamp_value=as_of),
            "trade_date": trade_date,
            "company_keyset": list(company_keyset),
            "tdx_fallback_company_keyset": list(fallback_keyset),
            "company_rows": [
                {
                    "company_code": row["company_code"],
                    "provider": row["provider"],
                    "status": row["status"],
                    "theme_ids": list(row["themes"]),
                }
                for row in company_rows
            ],
            "registry_ref": content_ref(registry, identity_field="registry_id"),
            "catalog_rows": [
                {
                    "catalog_ref": content_ref(catalog, identity_field="catalog_id"),
                    "company_count": len(catalog["coverage_rows"]),
                    "first_company": catalog["coverage_rows"][0]["company_code"],
                    "last_company": catalog["coverage_rows"][-1]["company_code"],
                }
                for catalog in catalogs
            ],
            "lifecycle_policy_ref": content_ref(lifecycle, identity_field="lifecycle_policy_id"),
        },
        identity_field="source_receipt_id",
    )


def _fallback_companies(
    *,
    companies: Sequence[str],
    captures: Mapping[str, Mapping[str, Any]],
    trade_date: str,
    registry: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    fallback: list[str] = []
    for company in companies:
        complete, rows, _ = _captured_rows(
            captures[company],
            fields=_DC_MEMBER_FIELDS,
            limit=5000,
            label=f"dc_member.{company}",
        )
        theme_ids, conflicts, scope_mismatch = _member_theme_ids(
            rows,
            provider=DC_PROVIDER,
            company=company,
            trade_date=trade_date,
            fields=_DC_MEMBER_FIELDS,
        )
        target_ids = _target_registry_ids(theme_ids, registry_ids=set(registry))
        if (
            not complete
            or scope_mismatch
            or bool(conflicts & set(target_ids))
            or (bool(theme_ids) and not target_ids)
        ):
            fallback.append(company)
    return sorted(fallback, key=lambda item: item.encode("ascii"))


def _tdx_registry_for_fallback(
    *,
    fallback: Sequence[str],
    registry_rows: Sequence[Mapping[str, Any]],
    registry_source_ref: Mapping[str, Any] | None,
    membership_captures: Mapping[str, Mapping[str, Any]],
    trade_date: str,
    captured_at: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not fallback:
        if registry_rows or registry_source_ref is not None or membership_captures:
            _fail("TDX fallback data is forbidden when DC coverage is complete")
        return {}, set()
    if registry_source_ref is None:
        _fail("TDX registry source is required for the sealed fallback keyset")
    if type(membership_captures) is not dict or set(membership_captures) != set(fallback):
        _fail("TDX membership capture keyset differs from fallback keyset")
    return _registry_rows(
        registry_rows,
        provider=TDX_PROVIDER,
        trade_date=trade_date,
        expected_fields=_TDX_REGISTRY_FIELDS,
        expected_type="概念板块",
        source_ref=_source_ref(registry_source_ref, label="tdx_registry_source_ref"),
        captured_at=captured_at,
        limit=1000,
    )


def compile_tushare_theme_source(
    *,
    trade_date: str,
    company_keyset: Sequence[str],
    dc_registry_rows: Sequence[Mapping[str, Any]],
    dc_registry_source_ref: Mapping[str, Any],
    dc_membership_captures: Mapping[str, Mapping[str, Any]],
    tdx_registry_rows: Sequence[Mapping[str, Any]],
    tdx_registry_source_ref: Mapping[str, Any] | None,
    tdx_membership_captures: Mapping[str, Mapping[str, Any]],
    scope_ref: Mapping[str, Any],
    owner_policy_ref: Mapping[str, Any],
    captured_at: str,
    as_of: str,
) -> dict[str, Any]:
    """Compile exact captured snapshots; this function performs no provider calls."""

    session = _session(trade_date, label="trade_date")
    cutoff = _timestamp(as_of, label="as_of")
    captured = _timestamp(captured_at, label="captured_at")
    if captured > cutoff:
        _fail("theme capture is from the future")
    companies = _keyset(company_keyset)
    if type(dc_membership_captures) is not dict or set(dc_membership_captures) != set(companies):
        _fail("DC membership capture keyset is incomplete")
    dc_source = _source_ref(dc_registry_source_ref, label="dc_registry_source_ref")
    dc_registry, dc_conflicts = _registry_rows(
        dc_registry_rows,
        provider=DC_PROVIDER,
        trade_date=session,
        expected_fields=_DC_REGISTRY_FIELDS,
        expected_type="概念板块",
        source_ref=dc_source,
        captured_at=captured,
        limit=5000,
    )
    fallback = _fallback_companies(
        companies=companies,
        captures=dc_membership_captures,
        trade_date=session,
        registry=dc_registry,
    )
    tdx_registry, tdx_conflicts = _tdx_registry_for_fallback(
        fallback=fallback,
        registry_rows=tdx_registry_rows,
        registry_source_ref=tdx_registry_source_ref,
        membership_captures=tdx_membership_captures,
        trade_date=session,
        captured_at=captured,
    )
    registry_rows = {**dc_registry, **tdx_registry}
    if not registry_rows:
        _fail("theme registry is empty")
    registry_ids = set(registry_rows)
    registry_conflicts = dc_conflicts | tdx_conflicts
    company_rows = []
    for company in companies:
        selected = _select_company(
            company=company,
            trade_date=session,
            dc_capture=dc_membership_captures[company],
            tdx_capture=tdx_membership_captures.get(company),
            registry_ids=registry_ids,
            registry_conflicts=registry_conflicts,
        )
        company_rows.append({"company_code": company, **selected})
    registry = build_theme_registry(themes=list(registry_rows.values()), as_of=cutoff)
    scope_source = _source_ref(scope_ref, label="scope_ref")
    catalogs = _catalog_shards(
        company_rows=company_rows,
        trade_date=session,
        captured_at=captured,
        registry=registry,
        scope_ref=scope_source,
        as_of=cutoff,
    )
    lifecycle_rows = [
        {
            "theme_id": row["theme_id"],
            "provider_id": row["theme_id"].split(":", 1)[0],
            "status": "ACTIVE",
            "effective_from": session,
            "effective_to": session,
            "available_at": captured,
            "source_ref": row["source_ref"],
        }
        for row in registry_rows.values()
    ]
    lifecycle = build_theme_lifecycle_policy(
        registry=registry,
        provider_precedence=[DC_PROVIDER, TDX_PROVIDER],
        cap_level=0,
        lifecycle_rows=lifecycle_rows,
        owner_policy_ref=_source_ref(owner_policy_ref, label="owner_policy_ref"),
        as_of=cutoff,
    )
    validate_theme_registry(registry)
    validate_theme_lifecycle_policy(lifecycle, registry=registry)
    receipt = _source_receipt(
        trade_date=session,
        company_keyset=companies,
        fallback_keyset=fallback,
        company_rows=company_rows,
        registry=registry,
        catalogs=catalogs,
        lifecycle=lifecycle,
        as_of=cutoff,
    )
    validate_tushare_theme_source_receipt(
        receipt,
        registry=registry,
        catalogs=catalogs,
        lifecycle_policy=lifecycle,
    )
    return {
        "registry": registry,
        "catalogs": catalogs,
        "lifecycle_policy": lifecycle,
        "source_receipt": receipt,
    }


def _receipt_keysets(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    company_keyset = row["company_keyset"]
    fallback = row["tdx_fallback_company_keyset"]
    for index, company_row in enumerate(row["company_rows"]):
        require_exact_keys(
            company_row,
            _COMPANY_RECEIPT_FIELDS,
            label=f"source_receipt.company_rows[{index}]",
        )
    if (
        not company_keyset
        or len(company_keyset) != len(set(company_keyset))
        or company_keyset != sorted(company_keyset, key=lambda item: item.encode("ascii"))
    ):
        _fail("theme source receipt company keyset is not ASCII sorted")
    if (
        len(fallback) != len(set(fallback))
        or fallback != sorted(fallback, key=lambda item: item.encode("ascii"))
        or not set(fallback).issubset(company_keyset)
    ):
        _fail("theme source receipt fallback keyset is outside company scope")
    return company_keyset, fallback


def _receipt_company_projection(
    *,
    catalogs: Sequence[Mapping[str, Any]],
    company_keyset: Sequence[str],
    fallback: Sequence[str],
) -> list[dict[str, Any]]:
    coverage = {
        item["company_code"]: item["status"]
        for catalog in catalogs
        for item in catalog["coverage_rows"]
    }
    if set(coverage) != set(company_keyset):
        _fail("theme source receipt company projection mismatch")
    memberships: dict[str, list[str]] = {company: [] for company in company_keyset}
    for catalog in catalogs:
        for item in catalog["membership_rows"]:
            memberships[item["company_code"]].append(item["theme_id"])
    return [
        {
            "company_code": company,
            "provider": TDX_PROVIDER if company in fallback else DC_PROVIDER,
            "status": coverage[company],
            "theme_ids": sorted(memberships[company], key=lambda item: item.encode("ascii")),
        }
        for company in company_keyset
    ]


def _validated_catalogs(
    *,
    catalogs: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    company_keyset: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(catalogs, (str, bytes)) or not isinstance(catalogs, Sequence):
        _fail("theme catalog shards must be a sequence")
    if not catalogs or len(catalogs) > 1024:
        _fail("theme catalog shard count is invalid")
    documents = [
        validate_theme_membership_catalog(catalog, registry=registry) for catalog in catalogs
    ]
    catalog_rows: list[dict[str, Any]] = []
    companies: list[str] = []
    refs: set[tuple[str, str, str, str]] = set()
    for index, catalog in enumerate(documents):
        coverage = [row["company_code"] for row in catalog["coverage_rows"]]
        if not coverage or coverage != sorted(coverage, key=lambda item: item.encode("ascii")):
            _fail("theme catalog shard company order is invalid")
        reference = content_ref(catalog, identity_field="catalog_id")
        reference_key = (
            reference["artifact_id"],
            reference["artifact_version"],
            reference["byte_sha256"],
            reference["semantic_sha256"],
        )
        if reference_key in refs:
            _fail("theme catalog shard reference is duplicated")
        refs.add(reference_key)
        catalog_rows.append(
            {
                "catalog_ref": reference,
                "company_count": len(coverage),
                "first_company": coverage[0],
                "last_company": coverage[-1],
            }
        )
        companies.extend(coverage)
        if index and catalog_rows[index - 1]["last_company"] >= coverage[0]:
            _fail("theme catalog shard ranges overlap")
    if companies != list(company_keyset):
        _fail("theme catalog shards do not close the company keyset")
    return documents, catalog_rows


def _receipt_catalog_documents(
    *,
    row: Mapping[str, Any],
    version: str,
    registry: Mapping[str, Any],
    company_keyset: Sequence[str],
    catalog: Mapping[str, Any] | None,
    catalogs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if version == SOURCE_RECEIPT_VERSION_V1:
        if catalog is None or catalogs:
            _fail("v1 theme source receipt requires exactly one catalog")
        documents = [validate_theme_membership_catalog(catalog, registry=registry)]
        expected_catalog = content_ref(documents[0], identity_field="catalog_id")
        return documents, row["catalog_ref"] == expected_catalog
    if catalog is not None:
        _fail("v2 theme source receipt forbids a singleton catalog")
    documents, expected_rows = _validated_catalogs(
        catalogs=catalogs,
        registry=registry,
        company_keyset=company_keyset,
    )
    for index, catalog_row in enumerate(row["catalog_rows"]):
        require_exact_keys(
            catalog_row,
            _CATALOG_RECEIPT_FIELDS,
            label=f"source_receipt.catalog_rows[{index}]",
        )
    return documents, row["catalog_rows"] == expected_rows


def validate_tushare_theme_source_receipt(
    document: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    lifecycle_policy: Mapping[str, Any],
    catalog: Mapping[str, Any] | None = None,
    catalogs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        version = document.get("version")
        fields = _RECEIPT_FIELDS_V1 if version == SOURCE_RECEIPT_VERSION_V1 else _RECEIPT_FIELDS
        row = require_exact_keys(document, fields, label="theme source receipt")
        validate_seal(row, identity_field="source_receipt_id")
        if version not in {SOURCE_RECEIPT_VERSION_V1, SOURCE_RECEIPT_VERSION}:
            _fail("theme source receipt version is invalid")
        if (
            row["authority"] != NO_AUTHORITY
            or row["research_only"] is not True
            or row["production"] is not False
        ):
            _fail("theme source receipt authority boundary is open")
        _timestamp(row["timestamp"], label="source_receipt.timestamp")
        _session(row["trade_date"], label="source_receipt.trade_date")
        registry_doc = validate_theme_registry(registry)
        lifecycle = validate_theme_lifecycle_policy(lifecycle_policy, registry=registry_doc)
        company_keyset, fallback = _receipt_keysets(row)
        catalog_documents, catalog_matches = _receipt_catalog_documents(
            row=row,
            version=version,
            registry=registry_doc,
            company_keyset=company_keyset,
            catalog=catalog,
            catalogs=catalogs,
        )
        if (
            row["registry_ref"] != content_ref(registry_doc, identity_field="registry_id")
            or row["lifecycle_policy_ref"]
            != content_ref(lifecycle, identity_field="lifecycle_policy_id")
            or not catalog_matches
        ):
            _fail("theme source receipt artifact binding mismatch")
        expected_company_rows = _receipt_company_projection(
            catalogs=catalog_documents,
            company_keyset=company_keyset,
            fallback=fallback,
        )
        if row["company_rows"] != expected_company_rows:
            _fail("theme source receipt company projection mismatch")
        return dict(row)
    except ThemeContractError:
        raise
    except Exception as exc:
        raise ThemeContractError(str(exc)) from exc


__all__ = [
    "compile_tushare_theme_source",
    "validate_tushare_theme_source_receipt",
]
