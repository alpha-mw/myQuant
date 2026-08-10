"""Deterministic industry identity, taxonomy, membership, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
from typing import Any

from .._core import (
    canonical_bytes,
    content_ref,
    sorted_unique,
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
    source_ref as canonical_source_ref,
)
from .models import (
    EVALUATION_RECEIPT_VERSION,
    IDENTITY_POLICY_VERSION,
    INDUSTRY_STATES,
    MEMBERSHIP_CATALOG_VERSION,
    TAXONOMY_STATUSES,
    TAXONOMY_VERSION,
)

_POLICY_FIELDS = frozenset(
    {
        "alias_collision_rule",
        "cap_taxonomy_level",
        "effective_available_rule",
        "exposure_tie_break",
        "merge_split_retirement_rule",
        "primary_industry_selection",
        "provider_precedence",
        "taxonomy_precedence",
    }
)
_TAXONOMY_FIELDS = frozenset({"source_ref", "taxonomy_id", "rows"})
_TAXONOMY_ROW_FIELDS = frozenset(
    {
        "aliases",
        "available_at",
        "effective_from",
        "effective_to",
        "industry_id",
        "level",
        "name",
        "parent_id",
        "status",
    }
)
_CATALOG_FIELDS = frozenset({"cutoff", "memberships", "provider_id", "source_ref", "taxonomy_id"})
_MEMBERSHIP_FIELDS = frozenset(
    {
        "available_at",
        "effective_from",
        "effective_to",
        "exposure",
        "industry_id",
        "listing_identity",
        "membership_id",
        "subject_id",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "as_of",
        "catalog_refs",
        "exposures",
        "listing_identity",
        "policy_ref",
        "primary_industry_id",
        "reason_codes",
        "state",
        "subject_id",
        "taxonomy_ref",
        "taxonomy_refs",
    }
)


def _ordered_unique(values: Sequence[Any], *, label: str) -> list[str]:
    rows = exact_sequence(values, label=label)
    normalized = [entity(value, label=f"{label}[{index}]") for index, value in enumerate(rows)]
    if not normalized or len(normalized) != len(set(normalized)):
        fail(f"{label} must be nonempty and unique")
    return normalized


def build_industry_identity_policy(
    *,
    created_at: str,
    provider_precedence: Sequence[str],
    taxonomy_precedence: Sequence[str],
    cap_taxonomy_level: str,
) -> dict[str, Any]:
    """Build the owner-sealed policy; no implicit precedence exists."""

    return artifact(
        version=IDENTITY_POLICY_VERSION,
        identity_field="policy_id",
        timestamp_value=created_at,
        payload={
            "alias_collision_rule": "AMBIGUOUS",
            "cap_taxonomy_level": entity(cap_taxonomy_level, label="cap_taxonomy_level"),
            "effective_available_rule": "EFFECTIVE_THEN_AVAILABLE",
            "exposure_tie_break": "INDUSTRY_ID_ASCII",
            "merge_split_retirement_rule": "SOURCE_CHRONOLOGY_REQUIRED",
            "primary_industry_selection": "HIGHEST_EXPOSURE",
            "provider_precedence": _ordered_unique(
                provider_precedence, label="provider_precedence"
            ),
            "taxonomy_precedence": _ordered_unique(
                taxonomy_precedence, label="taxonomy_precedence"
            ),
        },
    )


def validate_industry_identity_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=IDENTITY_POLICY_VERSION,
        identity_field="policy_id",
        payload_fields=_POLICY_FIELDS,
    )
    rebuilt = build_industry_identity_policy(
        created_at=row["timestamp"],
        provider_precedence=row["provider_precedence"],
        taxonomy_precedence=row["taxonomy_precedence"],
        cap_taxonomy_level=row["cap_taxonomy_level"],
    )
    if rebuilt != row:
        fail("industry identity policy replay mismatch")
    return row


def _taxonomy_row(value: Mapping[str, Any], *, as_of: str, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _TAXONOMY_ROW_FIELDS:
        fail(f"taxonomy.rows[{index}] shape is invalid")
    available_at = timestamp(value["available_at"], label="taxonomy.available_at")
    no_future(available_at=available_at, as_of=as_of, label="taxonomy row")
    effective_from = timestamp(value["effective_from"], label="taxonomy.effective_from")
    effective_to = value["effective_to"]
    if effective_to is not None:
        effective_to = timestamp(effective_to, label="taxonomy.effective_to")
        if effective_to < effective_from:
            fail("taxonomy interval is reversed")
    status = str(value["status"])
    if status not in TAXONOMY_STATUSES:
        fail("taxonomy status is invalid")
    if status == "RETIRED" and effective_to is None:
        fail("retired taxonomy row requires effective_to")
    name = value["name"]
    if type(name) is not str or not name.strip():
        fail("taxonomy name is required")
    parent_id = value["parent_id"]
    if parent_id is not None:
        parent_id = entity(parent_id, label="taxonomy.parent_id")
    level = value["level"]
    if type(level) is not int or type(level) is bool or level < 0 or level > 32:
        fail("taxonomy level is invalid")
    return {
        "aliases": sorted_unique(
            value["aliases"],
            label="taxonomy.aliases",
            maximum=64,
            allow_empty=True,
        ),
        "available_at": available_at,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "industry_id": entity(value["industry_id"], label="taxonomy.industry_id"),
        "level": level,
        "name": name.strip(),
        "parent_id": parent_id,
        "status": status,
    }


def _validate_taxonomy_chronology(rows: Sequence[dict[str, Any]], *, cutoff: str) -> None:
    histories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        histories.setdefault(row["industry_id"], []).append(row)
    for industry_id, history in histories.items():
        for previous, current in zip(history, history[1:]):
            if (
                previous["effective_to"] is None
                or previous["effective_to"] >= current["effective_from"]
            ):
                fail(f"taxonomy chronology overlaps for {industry_id}")

    eligible = [
        row
        for row in rows
        if row["available_at"] <= cutoff
        and row["effective_from"] <= cutoff
        and (row["effective_to"] is None or cutoff <= row["effective_to"])
    ]
    if len({row["industry_id"] for row in eligible}) != len(eligible):
        fail("taxonomy has multiple admissible rows for one industry")
    by_id = {row["industry_id"]: row for row in eligible}
    for row in eligible:
        parent = row["parent_id"]
        if parent is not None and (parent not in by_id or by_id[parent]["level"] >= row["level"]):
            fail("taxonomy parent hierarchy is invalid")
        seen: set[str] = set()
        while parent is not None:
            if parent in seen:
                fail("taxonomy hierarchy contains a cycle")
            seen.add(parent)
            parent = by_id[parent]["parent_id"]


def build_industry_taxonomy(
    *,
    taxonomy_id: str,
    rows: Sequence[Mapping[str, Any]],
    source_ref: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    normalized = [
        _taxonomy_row(value, as_of=cutoff, index=index)
        for index, value in enumerate(exact_sequence(rows, label="taxonomy.rows"))
    ]
    normalized.sort(
        key=lambda item: (
            item["industry_id"].encode("ascii"),
            item["effective_from"].encode("ascii"),
            item["available_at"].encode("ascii"),
        )
    )
    if not normalized:
        fail("taxonomy is empty")
    _validate_taxonomy_chronology(normalized, cutoff=cutoff)
    reference = canonical_source_ref(source_ref, label="taxonomy.source_ref")
    no_future(available_at=reference["available_at"], as_of=cutoff, label="taxonomy source")
    return artifact(
        version=TAXONOMY_VERSION,
        identity_field="taxonomy_receipt_id",
        timestamp_value=cutoff,
        payload={
            "rows": normalized,
            "source_ref": reference,
            "taxonomy_id": entity(taxonomy_id, label="taxonomy_id"),
        },
    )


def validate_industry_taxonomy(value: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=TAXONOMY_VERSION,
        identity_field="taxonomy_receipt_id",
        payload_fields=_TAXONOMY_FIELDS,
    )
    rebuilt = build_industry_taxonomy(
        taxonomy_id=row["taxonomy_id"],
        rows=row["rows"],
        source_ref=row["source_ref"],
        as_of=row["timestamp"],
    )
    if rebuilt != row:
        fail("industry taxonomy replay mismatch")
    return row


def _membership_row(value: Mapping[str, Any], *, cutoff: str, index: int) -> dict[str, Any]:
    required = set(_MEMBERSHIP_FIELDS) - {"membership_id"}
    actual = set(value) if type(value) is dict else set()
    if type(value) is not dict or (actual != required and actual != set(_MEMBERSHIP_FIELDS)):
        fail(f"memberships[{index}] shape is invalid")
    available_at = timestamp(value["available_at"], label="membership.available_at")
    no_future(available_at=available_at, as_of=cutoff, label="membership")
    effective_from = timestamp(value["effective_from"], label="membership.effective_from")
    effective_to = value["effective_to"]
    if effective_to is not None:
        effective_to = timestamp(effective_to, label="membership.effective_to")
        if effective_to < effective_from:
            fail("membership interval is reversed")
    body = {
        "available_at": available_at,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "exposure": decimal(
            value["exposure"], label="membership.exposure", minimum=Decimal(0), maximum=Decimal(1)
        ),
        "industry_id": entity(value["industry_id"], label="membership.industry_id"),
        "listing_identity": entity(value["listing_identity"], label="membership.listing_identity"),
        "subject_id": entity(value["subject_id"], label="membership.subject_id"),
    }
    membership_id = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if "membership_id" in value and value["membership_id"] != membership_id:
        fail("membership_id mismatch")
    return {**body, "membership_id": membership_id}


def _validate_membership_bundles(rows: Sequence[dict[str, Any]]) -> None:
    bundles: dict[tuple[str, str, str, str | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        bundle_key = (
            row["subject_id"],
            row["listing_identity"],
            row["effective_from"],
            row["effective_to"],
            row["available_at"],
        )
        bundles.setdefault(bundle_key, []).append(row)
    for bundle in bundles.values():
        if len({row["industry_id"] for row in bundle}) != len(bundle):
            fail("membership catalog repeats an industry")
        if sum(Decimal(row["exposure"]) for row in bundle) != Decimal("1.000000000000"):
            fail("membership exposures must sum exactly to one")
    histories: dict[tuple[str, str], list[tuple[str, str | None, str]]] = {}
    for subject, listing, effective_from, effective_to, available_at in bundles:
        histories.setdefault((subject, listing), []).append(
            (effective_from, effective_to, available_at)
        )
    for history in histories.values():
        ordered = sorted(history, key=lambda item: (item[0], item[2]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] is None or previous[1] >= current[0]:
                fail("membership catalog contains overlapping chronology bundles")


def build_industry_membership_catalog(
    *,
    provider_id: str,
    taxonomy_id: str,
    memberships: Sequence[Mapping[str, Any]],
    source_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
) -> dict[str, Any]:
    exact_cutoff = timestamp(cutoff, label="cutoff")
    created = timestamp(created_at, label="created_at")
    if created < exact_cutoff:
        fail("catalog created_at precedes cutoff")
    reference = canonical_source_ref(source_ref, label="catalog.source_ref")
    no_future(
        available_at=reference["available_at"],
        as_of=exact_cutoff,
        label="catalog source",
    )
    rows = [
        _membership_row(value, cutoff=exact_cutoff, index=index)
        for index, value in enumerate(exact_sequence(memberships, label="memberships"))
    ]
    rows.sort(key=lambda item: item["membership_id"].encode("ascii"))
    if not rows or len({row["membership_id"] for row in rows}) != len(rows):
        fail("membership catalog is empty or duplicated")
    _validate_membership_bundles(rows)
    return artifact(
        version=MEMBERSHIP_CATALOG_VERSION,
        identity_field="catalog_id",
        timestamp_value=created,
        payload={
            "cutoff": exact_cutoff,
            "memberships": rows,
            "provider_id": entity(provider_id, label="provider_id"),
            "source_ref": reference,
            "taxonomy_id": entity(taxonomy_id, label="taxonomy_id"),
        },
    )


def validate_industry_membership_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=MEMBERSHIP_CATALOG_VERSION,
        identity_field="catalog_id",
        payload_fields=_CATALOG_FIELDS,
    )
    rebuilt = build_industry_membership_catalog(
        provider_id=row["provider_id"],
        taxonomy_id=row["taxonomy_id"],
        memberships=row["memberships"],
        source_ref=row["source_ref"],
        cutoff=row["cutoff"],
        created_at=row["timestamp"],
    )
    if rebuilt != row:
        fail("industry membership catalog replay mismatch")
    return row


def _valid_industry_ids(taxonomy: Mapping[str, Any], *, cutoff: str) -> set[str]:
    return {
        row["industry_id"]
        for row in taxonomy["rows"]
        if row["available_at"] <= cutoff
        and row["effective_from"] <= cutoff
        and (row["effective_to"] is None or cutoff <= row["effective_to"])
        and row["status"] == "ACTIVE"
    }


def _validated_taxonomies(
    *, policy: Mapping[str, Any], values: Sequence[Mapping[str, Any]], cutoff: str
) -> list[dict[str, Any]]:
    rows = [
        validate_industry_taxonomy(value) for value in exact_sequence(values, label="taxonomies")
    ]
    taxonomy_ids = [row["taxonomy_id"] for row in rows]
    if not rows or len(taxonomy_ids) != len(set(taxonomy_ids)):
        fail("taxonomy closure is empty or duplicated")
    if set(taxonomy_ids) != set(policy["taxonomy_precedence"]):
        fail("taxonomy closure does not match policy precedence")
    if any(row["timestamp"] > cutoff for row in rows):
        fail("industry taxonomy is future-known")
    return rows


def _identity_candidates(
    *,
    policy: Mapping[str, Any],
    taxonomies: Mapping[str, dict[str, Any]],
    catalogs: Sequence[dict[str, Any]],
    subject: str,
    listing: str,
    cutoff: str,
) -> list[tuple[tuple[int, int, str], dict[str, str], dict[str, Any]]]:
    provider_rank = {value: index for index, value in enumerate(policy["provider_precedence"])}
    taxonomy_rank = {value: index for index, value in enumerate(policy["taxonomy_precedence"])}
    valid_industries = {
        taxonomy_id: _valid_industry_ids(taxonomy, cutoff=cutoff)
        for taxonomy_id, taxonomy in taxonomies.items()
    }
    candidates: list[tuple[tuple[int, int, str], dict[str, str], dict[str, Any]]] = []
    for catalog in catalogs:
        if catalog["timestamp"] > cutoff or catalog["cutoff"] > cutoff:
            fail("industry membership catalog is future-known")
        if catalog["taxonomy_id"] not in taxonomies:
            fail("membership catalog taxonomy binding mismatch")
        if (
            catalog["provider_id"] not in provider_rank
            or catalog["taxonomy_id"] not in taxonomy_rank
        ):
            continue
        active = [
            row
            for row in catalog["memberships"]
            if row["subject_id"] == subject
            and row["listing_identity"] == listing
            and row["available_at"] <= cutoff
            and row["effective_from"] <= cutoff
            and (row["effective_to"] is None or cutoff <= row["effective_to"])
        ]
        if not active:
            continue
        if any(
            row["industry_id"] not in valid_industries[catalog["taxonomy_id"]] for row in active
        ):
            fail("membership points to unavailable taxonomy identity")
        candidates.append(
            (
                (
                    provider_rank[catalog["provider_id"]],
                    taxonomy_rank[catalog["taxonomy_id"]],
                    max(
                        catalog["source_ref"]["available_at"],
                        max(row["available_at"] for row in active),
                    ),
                ),
                {row["industry_id"]: row["exposure"] for row in active},
                taxonomies[catalog["taxonomy_id"]],
            )
        )
    return candidates


def _resolve_identity(
    candidates: Sequence[tuple[tuple[int, int, str], dict[str, str], dict[str, Any]]],
) -> tuple[str, list[dict[str, str]], str | None, list[str], dict[str, Any] | None]:
    if not candidates:
        return "UNMAPPED", [], None, ["NO_ADMISSIBLE_MEMBERSHIP"], None
    best_precedence = min((key[0], key[1]) for key, _exposures, _taxonomy in candidates)
    precedence_rows = [item for item in candidates if item[0][:2] == best_precedence]
    latest = max(item[0][2] for item in precedence_rows)
    winners = [item for item in precedence_rows if item[0][2] == latest]
    if len({canonical_bytes(item[1]) for item in winners}) != 1:
        return (
            "AMBIGUOUS",
            [],
            None,
            ["SAME_PRECEDENCE_CLASSIFICATION_CONFLICT"],
            winners[0][2],
        )
    selected = winners[0][1]
    exposures = [
        {"exposure": selected[industry_id], "industry_id": industry_id}
        for industry_id in sorted(selected, key=lambda value: value.encode("ascii"))
    ]
    highest = max(Decimal(row["exposure"]) for row in exposures)
    primary = min(row["industry_id"] for row in exposures if Decimal(row["exposure"]) == highest)
    return "AVAILABLE", exposures, primary, ["IDENTITY_POLICY_RESOLVED"], winners[0][2]


def evaluate_industry_identity(
    *,
    policy: Mapping[str, Any],
    taxonomies: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
    subject_id: str,
    listing_identity: str,
    as_of: str,
) -> dict[str, Any]:
    policy_row = validate_industry_identity_policy(policy)
    cutoff = timestamp(as_of, label="as_of")
    if policy_row["timestamp"] > cutoff:
        fail("industry identity input is future-known")
    taxonomy_rows = _validated_taxonomies(policy=policy_row, values=taxonomies, cutoff=cutoff)
    taxonomy_by_id = {row["taxonomy_id"]: row for row in taxonomy_rows}
    subject = entity(subject_id, label="subject_id")
    listing = entity(listing_identity, label="listing_identity")
    catalog_rows = [validate_industry_membership_catalog(value) for value in catalogs]
    catalog_ids = [row["catalog_id"] for row in catalog_rows]
    if len(catalog_ids) != len(set(catalog_ids)):
        fail("industry membership catalog closure is duplicated")
    candidates = _identity_candidates(
        policy=policy_row,
        taxonomies=taxonomy_by_id,
        catalogs=catalog_rows,
        subject=subject,
        listing=listing,
        cutoff=cutoff,
    )
    state, exposures, primary, reasons, selected_taxonomy = _resolve_identity(candidates)
    if state not in INDUSTRY_STATES:
        fail("industry state is invalid")
    payload = {
        "as_of": cutoff,
        "catalog_refs": sorted(
            [content_ref(row, identity_field="catalog_id") for row in catalog_rows],
            key=lambda ref: (
                ref["artifact_id"].encode("ascii"),
                ref["byte_sha256"].encode("ascii"),
            ),
        ),
        "exposures": exposures,
        "listing_identity": listing,
        "policy_ref": content_ref(policy_row, identity_field="policy_id"),
        "primary_industry_id": primary,
        "reason_codes": reasons,
        "state": state,
        "subject_id": subject,
        "taxonomy_ref": (
            content_ref(selected_taxonomy, identity_field="taxonomy_receipt_id")
            if selected_taxonomy is not None
            else None
        ),
        "taxonomy_refs": sorted(
            [content_ref(row, identity_field="taxonomy_receipt_id") for row in taxonomy_rows],
            key=lambda ref: (
                ref["artifact_id"].encode("ascii"),
                ref["byte_sha256"].encode("ascii"),
            ),
        ),
    }
    return artifact(
        version=EVALUATION_RECEIPT_VERSION,
        identity_field="evaluation_id",
        timestamp_value=cutoff,
        payload=payload,
    )


def validate_industry_evaluation_receipt(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    taxonomies: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = closed_artifact(
        value,
        version=EVALUATION_RECEIPT_VERSION,
        identity_field="evaluation_id",
        payload_fields=_EVALUATION_FIELDS,
    )
    rebuilt = evaluate_industry_identity(
        policy=policy,
        taxonomies=taxonomies,
        catalogs=catalogs,
        subject_id=row["subject_id"],
        listing_identity=row["listing_identity"],
        as_of=row["as_of"],
    )
    if rebuilt != row:
        fail("industry evaluation replay mismatch")
    return row


__all__ = [
    "build_industry_identity_policy",
    "build_industry_membership_catalog",
    "build_industry_taxonomy",
    "evaluate_industry_identity",
    "validate_industry_evaluation_receipt",
    "validate_industry_identity_policy",
    "validate_industry_membership_catalog",
    "validate_industry_taxonomy",
]
