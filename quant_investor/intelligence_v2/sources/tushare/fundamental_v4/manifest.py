"""Authoritative bulk-provider manifest v4."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

from ...._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    exact_ref,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .comparison import validate_fundamental_comparison_policy
from .fileset import validate_provider_evidence_fileset_manifest
from .models import (
    PROVIDER_MANIFEST_V4,
    SOURCE_TABLES,
    FundamentalV4ContractError,
    fundamental_v4_contract,
)
from .reconciliation import validate_fundamental_reconciliation_receipt
from .schedule import validate_fundamental_execution_closure_v4

_FIELDS = {
    "as_of",
    "authority",
    "authoritative_full_rebuild",
    "baseline_raw_evidence",
    "baseline_network_attempts",
    "comparison_policy_ref",
    "decision_protocol",
    "execution_closure_ref",
    "fileset_ref",
    "frozen_v1_manifest_sha256",
    "logical_coverage_ref",
    "manifest_id",
    "performance_gate_passed",
    "pit_cutoff",
    "production",
    "provider",
    "raw_table_fingerprints",
    "reconciliation_ref",
    "request_plan_ref",
    "request_receipts_ref",
    "research_only",
    "schema_version",
    "semantic_sha256",
    "source_provenance",
    "symbol_set_sha256",
    "symbols_requested",
    "tables",
    "timestamp",
    "version",
    "vip_network_attempts",
    "vip_raw_evidence",
}


def _inventory(fileset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["relative_path"]: row for row in fileset["inventory"]}


def _provider_relative(value: str) -> str:
    prefix = "provider_evidence/"
    if not value.startswith(prefix):
        raise FundamentalV4ContractError("provider evidence ref escaped its generation root")
    return value.removeprefix(prefix)


def _assert_inventory_binding(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    relative_path: str,
    byte_sha256: str,
) -> None:
    row = inventory.get(relative_path)
    if row is None or row["byte_sha256"] != byte_sha256:
        raise FundamentalV4ContractError("provider fileset byte binding mismatch")


@fundamental_v4_contract
def build_fundamental_provider_manifest_v4(
    *,
    execution_closure: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    reconciliation_closure: Mapping[str, Any],
    fileset: Mapping[str, Any],
    request_receipts_ref: Mapping[str, Any],
    logical_coverage_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Build v4 authority only from a PASSED full replay and durable fileset."""

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    validated_plan = execution["request_plan"]
    receipt = validate_fundamental_reconciliation_receipt(
        reconciliation,
        **dict(reconciliation_closure),
    )
    if receipt["plan_ref"] != content_ref(validated_plan, identity_field="plan_id"):
        raise FundamentalV4ContractError("provider manifest plan closure mismatch")
    if receipt["status"] != "PASSED" or receipt["performance_gate_passed"] is not True:
        raise FundamentalV4ContractError("provider manifest requires a passed reconciliation")
    evidence_fileset = validate_provider_evidence_fileset_manifest(fileset)
    request_ref = exact_ref(request_receipts_ref, label="request_receipts_ref")
    coverage_ref = exact_ref(logical_coverage_ref, label="logical_coverage_ref")
    created = timestamp(created_at, label="created_at")
    if request_ref["available_at"] > created or coverage_ref["available_at"] > created:
        raise FundamentalV4ContractError("provider manifest contains future files")
    inventory = _inventory(evidence_fileset)
    _assert_inventory_binding(
        inventory,
        relative_path="execution_plan.json",
        byte_sha256=hashlib.sha256(canonical_bytes(execution)).hexdigest(),
    )
    _assert_inventory_binding(
        inventory,
        relative_path="reconciliation.json",
        byte_sha256=hashlib.sha256(canonical_bytes(receipt)).hexdigest(),
    )
    _assert_inventory_binding(
        inventory,
        relative_path="request_receipts.jsonl",
        byte_sha256=request_ref["byte_sha256"],
    )
    _assert_inventory_binding(
        inventory,
        relative_path="logical_coverage.parquet",
        byte_sha256=coverage_ref["byte_sha256"],
    )
    policy = validate_fundamental_comparison_policy(reconciliation_closure["comparison_policy"])
    _assert_inventory_binding(
        inventory,
        relative_path="comparison_policy.json",
        byte_sha256=hashlib.sha256(canonical_bytes(policy)).hexdigest(),
    )
    for name, output_ref in receipt["comparison_output_refs"].items():
        _assert_inventory_binding(
            inventory,
            relative_path=_provider_relative(output_ref["relative_path"]),
            byte_sha256=output_ref["byte_sha256"],
        )
    raw_evidence = [
        *reconciliation_closure["baseline_raw_evidence"],
        *reconciliation_closure["vip_raw_evidence"],
    ]
    for row in raw_evidence:
        _assert_inventory_binding(
            inventory,
            relative_path=_provider_relative(row["file_ref"]["relative_path"]),
            byte_sha256=row["file_ref"]["byte_sha256"],
        )
    vip_rows = reconciliation_closure["vip_raw_evidence"]
    fingerprints = {row["table"]: row["canonical_multiset_sha256"] for row in vip_rows}
    if set(fingerprints) != set(SOURCE_TABLES):
        raise FundamentalV4ContractError("provider raw fingerprint set is invalid")
    body = {
        **common_fields(timestamp_value=created),
        "as_of": validated_plan["as_of"],
        "authoritative_full_rebuild": True,
        "baseline_raw_evidence": sorted(
            reconciliation_closure["baseline_raw_evidence"],
            key=lambda row: row["evidence_id"].encode("ascii"),
        ),
        "baseline_network_attempts": receipt["baseline_network_attempts"],
        "comparison_policy_ref": content_ref(policy, identity_field="policy_id"),
        "execution_closure_ref": content_ref(execution, identity_field="closure_id"),
        "fileset_ref": content_ref(evidence_fileset, identity_field="fileset_id"),
        "logical_coverage_ref": coverage_ref,
        "performance_gate_passed": True,
        "pit_cutoff": validated_plan["pit_cutoff"],
        "provider": "tushare",
        "raw_table_fingerprints": fingerprints,
        "reconciliation_ref": content_ref(receipt, identity_field="receipt_id"),
        "request_plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "request_receipts_ref": request_ref,
        "schema_version": PROVIDER_MANIFEST_V4,
        "source_provenance": "live_tushare_vip_explicit",
        "symbol_set_sha256": validated_plan["symbol_set_sha256"],
        "symbols_requested": len(validated_plan["symbols"]),
        "tables": list(SOURCE_TABLES),
        "version": PROVIDER_MANIFEST_V4,
        "vip_network_attempts": receipt["vip_network_attempts"],
        "vip_raw_evidence": sorted(
            reconciliation_closure["vip_raw_evidence"],
            key=lambda row: row["evidence_id"].encode("ascii"),
        ),
    }
    return seal(body, identity_field="manifest_id")


@fundamental_v4_contract
def validate_fundamental_provider_manifest_v4(
    document: Mapping[str, Any],
    **closure: Any,
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="manifest_id")
    require_exact_keys(value, _FIELDS, label="Fundamental provider manifest v4")
    if value.get("version") != PROVIDER_MANIFEST_V4:
        raise FundamentalV4ContractError("provider manifest v4 version mismatch")
    expected = build_fundamental_provider_manifest_v4(
        **closure,
        created_at=value["timestamp"],
    )
    if value != expected:
        raise FundamentalV4ContractError("provider manifest v4 replay mismatch")
    return value


__all__ = [
    "build_fundamental_provider_manifest_v4",
    "validate_fundamental_provider_manifest_v4",
]
