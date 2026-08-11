"""Deterministic SW2021 Industry source compiler for the existing I2 contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ...industry import (
    build_industry_membership_catalog,
    build_industry_taxonomy,
    validate_industry_membership_catalog,
    validate_industry_taxonomy,
)
from ...industry.models import IndustryContractError
from ..._core import exact_ref, timestamp

TAXONOMY_FIELDS = {
    "index_code",
    "industry_name",
    "parent_code",
    "level",
    "industry_code",
    "is_pub",
    "src",
}
MEMBERSHIP_FIELDS = {
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
}
TAXONOMY_ID = "TUSHARE_SW2021"
PROVIDER_ID = "TUSHARE_INDEX_MEMBER_ALL"


def _rows(value: Any, *, fields: set[str], label: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise IndustryContractError(f"{label} must be a sequence")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != fields:
            raise IndustryContractError(f"{label}[{index}] shape is invalid")
        result.append(dict(raw))
    return result


def _code(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise IndustryContractError(f"{label} is invalid")
    return value


def _date(value: Any, *, label: str, end: bool = False) -> str | None:
    if value in {None, ""}:
        return None
    if type(value) is not str or len(value) != 8 or not value.isdigit():
        raise IndustryContractError(f"{label} is invalid")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise IndustryContractError(f"{label} is invalid") from exc
    suffix = "23:59:59Z" if end else "00:00:00Z"
    return f"{parsed:%Y-%m-%d}T{suffix}"


def _industry_id(value: Any) -> str:
    return f"TUSHARE_SW2021:{_code(value, label='industry code')}"


def _timestamp(value: Any, *, label: str) -> str:
    try:
        return timestamp(value, label=label)
    except IndustryContractError:
        raise
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def _source_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return exact_ref(value, label=label)
    except IndustryContractError:
        raise
    except Exception as exc:
        raise IndustryContractError(str(exc)) from exc


def _taxonomy_rows(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    effective_from: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    if type(partitions) is not dict or set(partitions) != {"L1", "L2", "L3"}:
        raise IndustryContractError("industry taxonomy partition keyset is incomplete")
    result: list[dict[str, Any]] = []
    source_rows: list[tuple[str, dict[str, Any]]] = []
    index_codes: set[str] = set()
    index_code_by_industry_code: dict[str, str] = {}
    for expected_level in ("L1", "L2", "L3"):
        for row in _rows(
            partitions[expected_level],
            fields=TAXONOMY_FIELDS,
            label=f"taxonomy.{expected_level}",
        ):
            index_code = _code(row["index_code"], label="index_code")
            industry_code = _code(row["industry_code"], label="industry_code")
            if (
                row["level"] != expected_level
                or row["src"] != "SW2021"
                or index_code in index_codes
                or industry_code in index_code_by_industry_code
                or type(row["industry_name"]) is not str
                or not row["industry_name"]
            ):
                raise IndustryContractError("industry taxonomy identity is invalid")
            index_codes.add(index_code)
            index_code_by_industry_code[industry_code] = index_code
            source_rows.append((expected_level, row))
    for expected_level, row in source_rows:
        parent = row["parent_code"]
        if expected_level == "L1":
            if parent not in {None, "", "0"}:
                raise IndustryContractError("L1 taxonomy parent is invalid")
            parent_id = None
        else:
            if type(parent) is not str or parent not in index_code_by_industry_code:
                raise IndustryContractError("industry taxonomy parent is unresolved")
            parent_id = _industry_id(index_code_by_industry_code[parent])
        result.append(
            {
                "aliases": [],
                "available_at": captured_at,
                "effective_from": effective_from,
                "effective_to": None,
                "industry_id": _industry_id(row["index_code"]),
                "level": int(expected_level[1:]) - 1,
                "name": row["industry_name"],
                "parent_id": parent_id,
                "status": "ACTIVE",
            }
        )
    return result


def _membership_rows(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    l3_codes: Sequence[str],
) -> list[dict[str, Any]]:
    expected = {f"{code}|{flag}" for code in l3_codes for flag in ("Y", "N")}
    if type(partitions) is not dict or set(partitions) != expected:
        raise IndustryContractError("industry membership partition keyset is incomplete")
    union: dict[tuple[Any, ...], dict[str, Any]] = {}
    for partition_key in sorted(expected, key=lambda item: item.encode("ascii")):
        code, flag = partition_key.split("|", 1)
        rows = _rows(
            partitions[partition_key],
            fields=MEMBERSHIP_FIELDS,
            label=f"membership.{partition_key}",
        )
        if len(rows) >= 2000:
            raise IndustryContractError("industry membership row limit was reached")
        for row in rows:
            if row["l3_code"] != code or row["is_new"] != flag:
                raise IndustryContractError("industry membership scope mismatch")
            key = tuple(row[field] for field in sorted(MEMBERSHIP_FIELDS))
            union[key] = row
    return list(union.values())


def _subject_memberships(
    rows: Sequence[Mapping[str, Any]],
    *,
    listing_identity_by_company: Mapping[str, str],
    captured_at: str,
    cutoff: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        subject = _code(row["ts_code"], label="ts_code")
        if subject not in listing_identity_by_company:
            continue
        effective_from = _date(row["in_date"], label="in_date")
        effective_to = _date(row["out_date"], label="out_date", end=True)
        if effective_from is None or (effective_to is not None and effective_to < effective_from):
            raise IndustryContractError("industry membership interval is invalid")
        by_subject.setdefault(subject, []).append(
            {
                "available_at": captured_at,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "exposure": "1.000000000000",
                "industry_id": _industry_id(row["l1_code"]),
                "listing_identity": listing_identity_by_company[subject],
                "subject_id": subject,
            }
        )
    blocked: dict[str, str] = {}
    admitted: list[dict[str, Any]] = []
    for subject in sorted(listing_identity_by_company, key=lambda item: item.encode("ascii")):
        values = by_subject.get(subject, [])
        active = {
            row["industry_id"]
            for row in values
            if row["effective_from"] <= cutoff
            and (row["effective_to"] is None or cutoff <= row["effective_to"])
        }
        if len(active) > 1:
            blocked[subject] = "AMBIGUOUS"
            continue
        if not active:
            blocked[subject] = "UNMAPPED"
            continue
        unique = {
            (
                row["industry_id"],
                row["effective_from"],
                row["effective_to"],
            ): row
            for row in values
        }
        ordered = sorted(unique.values(), key=lambda row: row["effective_from"])
        if any(
            previous["effective_to"] is None
            or previous["effective_to"] >= current["effective_from"]
            for previous, current in zip(ordered, ordered[1:])
        ):
            blocked[subject] = "AMBIGUOUS"
            continue
        admitted.extend(ordered)
    return admitted, blocked


def compile_tushare_sw2021_industry_source(
    *,
    taxonomy_partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    membership_partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    listing_identity_by_company: Mapping[str, str],
    taxonomy_source_ref: Mapping[str, Any],
    membership_source_ref: Mapping[str, Any],
    taxonomy_effective_from: str,
    cutoff: str,
    captured_at: str,
) -> dict[str, Any]:
    """Compile exact captured rows; display-industry text is intentionally absent."""

    observed = _timestamp(captured_at, label="captured_at")
    exact_cutoff = _timestamp(cutoff, label="cutoff")
    effective = _timestamp(taxonomy_effective_from, label="taxonomy_effective_from")
    if observed < exact_cutoff or effective > exact_cutoff:
        raise IndustryContractError("industry compiler chronology is invalid")
    tax_ref = _source_ref(taxonomy_source_ref, label="taxonomy_source_ref")
    member_ref = _source_ref(membership_source_ref, label="membership_source_ref")
    taxonomy_rows = _taxonomy_rows(
        taxonomy_partitions,
        effective_from=effective,
        captured_at=observed,
    )
    l3_codes = sorted(row["index_code"] for row in taxonomy_partitions["L3"])
    memberships, blocked = _subject_memberships(
        _membership_rows(membership_partitions, l3_codes=l3_codes),
        listing_identity_by_company=listing_identity_by_company,
        captured_at=observed,
        cutoff=exact_cutoff,
    )
    taxonomy = build_industry_taxonomy(
        taxonomy_id=TAXONOMY_ID,
        rows=taxonomy_rows,
        source_ref=tax_ref,
        as_of=exact_cutoff,
    )
    catalog = build_industry_membership_catalog(
        provider_id=PROVIDER_ID,
        taxonomy_id=TAXONOMY_ID,
        memberships=memberships,
        source_ref=member_ref,
        cutoff=exact_cutoff,
        created_at=observed,
    )
    validate_industry_taxonomy(taxonomy)
    validate_industry_membership_catalog(catalog)
    return {
        "blocked_subjects": dict(sorted(blocked.items())),
        "catalog": catalog,
        "status": "AVAILABLE" if not blocked else "PARTIAL_BLOCKED",
        "taxonomy": taxonomy,
    }


__all__ = ["compile_tushare_sw2021_industry_source"]
