"""Closed inventory for durable Fundamental v4 provider evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

from ...._core import (
    canonical_bytes,
    common_fields,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .models import (
    PROVIDER_EVIDENCE_FILESET_V1,
    SOURCE_TABLES,
    FundamentalV4ContractError,
    fundamental_v4_contract,
)

REQUIRED_EVIDENCE_PATHS = tuple(
    sorted(
        {
            "comparison_outputs/coverage_diff.json",
            "comparison_outputs/derived_fingerprints.json",
            "comparison_outputs/duplicate_diff.json",
            "comparison_outputs/raw_row_diff.json",
            "comparison_outputs/raw_value_diff.json",
            "comparison_policy.json",
            "execution_plan.json",
            "logical_coverage.parquet",
            "reconciliation.json",
            "request_receipts.jsonl",
            *{f"baseline_raw/{table}.parquet" for table in SOURCE_TABLES},
            *{f"vip_raw/{table}.parquet" for table in SOURCE_TABLES},
        }
    )
)

_FIELDS = {
    "array_order_semantics",
    "authority",
    "created_at",
    "decision_protocol",
    "fileset_id",
    "fileset_sha256",
    "frozen_v1_manifest_sha256",
    "inventory",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
_ROW_FIELDS = {
    "byte_sha256",
    "mode",
    "relative_path",
    "semantic_sha256",
    "size_bytes",
}


def _inventory(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalV4ContractError("fileset inventory must be a sequence")
    rows: list[dict[str, Any]] = []
    for value in values:
        row = require_exact_keys(value, _ROW_FIELDS, label="fileset inventory row")
        relative_path = row.get("relative_path")
        size_bytes = row.get("size_bytes")
        if (
            type(relative_path) is not str
            or relative_path not in REQUIRED_EVIDENCE_PATHS
            or type(size_bytes) is not int
            or size_bytes < 0
            or row.get("mode") != "0600"
        ):
            raise FundamentalV4ContractError("fileset inventory row is invalid")
        rows.append(
            {
                "byte_sha256": sha256(
                    row.get("byte_sha256"),
                    label=f"{relative_path}.byte_sha256",
                ),
                "mode": "0600",
                "relative_path": relative_path,
                "semantic_sha256": sha256(
                    row.get("semantic_sha256"),
                    label=f"{relative_path}.semantic_sha256",
                ),
                "size_bytes": size_bytes,
            }
        )
    expected = sorted(rows, key=lambda row: row["relative_path"].encode("ascii"))
    if (
        rows != expected
        or len(rows) != len(REQUIRED_EVIDENCE_PATHS)
        or {row["relative_path"] for row in rows} != set(REQUIRED_EVIDENCE_PATHS)
    ):
        raise FundamentalV4ContractError("fileset inventory is not exact and sorted")
    return rows


@fundamental_v4_contract
def build_provider_evidence_fileset_manifest(
    *,
    inventory: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Seal the exact copy-set; the manifest intentionally excludes itself."""

    rows = _inventory(inventory)
    created = timestamp(created_at, label="created_at")
    body = {
        **common_fields(timestamp_value=created),
        "array_order_semantics": {"/inventory": "relative_path ASCII ascending"},
        "created_at": created,
        "fileset_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
        "inventory": rows,
        "version": PROVIDER_EVIDENCE_FILESET_V1,
    }
    return seal(body, identity_field="fileset_id")


@fundamental_v4_contract
def validate_provider_evidence_fileset_manifest(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="fileset_id")
    require_exact_keys(value, _FIELDS, label="provider evidence fileset manifest")
    if value.get("version") != PROVIDER_EVIDENCE_FILESET_V1:
        raise FundamentalV4ContractError("provider evidence fileset version mismatch")
    expected = build_provider_evidence_fileset_manifest(
        inventory=value["inventory"],
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalV4ContractError("provider evidence fileset replay mismatch")
    return value


__all__ = [
    "REQUIRED_EVIDENCE_PATHS",
    "build_provider_evidence_fileset_manifest",
    "validate_provider_evidence_fileset_manifest",
]
