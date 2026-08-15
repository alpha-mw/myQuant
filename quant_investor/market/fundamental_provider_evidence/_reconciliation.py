"""Full-window Fundamental VIP reconciliation receipt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import pandas as pd

from ._codec import (
    canonical_bytes,
    common_fields,
    content_ref,
    exact_ref,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from ._comparison import (
    compare_fundamental_raw_tables,
    validate_fundamental_comparison_policy,
)
from ._evidence import (
    _plan_and_endpoints,
    _partition_ids,
    _validate_logical_coverage_validated,
    _validate_physical_request_receipt_validated,
    validate_raw_table_evidence,
)
from ._model import (
    RECONCILIATION_RECEIPT_SCHEMA,
    SOURCE_TABLES,
    FundamentalProviderEvidenceError,
    provider_evidence_contract,
)
from ._schedule import validate_fundamental_request_plan

_FIELDS = {
    "array_order_semantics",
    "authority",
    "baseline_network_attempts",
    "baseline_raw_evidence_refs",
    "blocker_codes",
    "comparison_output_refs",
    "comparison_output_sha256",
    "comparison_policy_ref",
    "coverage_passed",
    "decision_protocol",
    "derived_comparison_passed",
    "derived_fingerprints",
    "frozen_v1_manifest_sha256",
    "logical_coverage_refs",
    "performance_gate_passed",
    "physical_receipt_refs",
    "plan_ref",
    "production",
    "raw_comparison_passed",
    "receipt_id",
    "research_only",
    "semantic_sha256",
    "status",
    "timestamp",
    "version",
    "vip_network_attempts",
    "vip_raw_evidence_refs",
}
_OUTPUT_PATHS = {
    "coverage_diff": "provider_evidence/comparison_outputs/coverage_diff.json",
    "derived_fingerprints": ("provider_evidence/comparison_outputs/derived_fingerprints.json"),
    "duplicate_diff": "provider_evidence/comparison_outputs/duplicate_diff.json",
    "raw_row_diff": "provider_evidence/comparison_outputs/raw_row_diff.json",
    "raw_value_diff": "provider_evidence/comparison_outputs/raw_value_diff.json",
}
_DERIVED_TABLES = (
    "coverage",
    "fundamental_daily",
    "fundamental_period",
    "quarantine",
)


def _documents(value: Any, *, label: str, maximum: int) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalProviderEvidenceError(f"{label} must be a sequence")
    rows = list(value)
    if len(rows) > maximum:
        raise FundamentalProviderEvidenceError(f"{label} exceeds maximum cardinality")
    if any(type(row) is not dict for row in rows):
        raise FundamentalProviderEvidenceError(f"{label} contains a non-object")
    return rows


def _physical_receipts(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    plan_ref = content_ref(validated_plan, identity_field="plan_id")
    rows = [
        _validate_physical_request_receipt_validated(
            value,
            validated_plan=validated_plan,
            validated_plan_ref=plan_ref,
            endpoints=endpoints,
        )
        for value in _documents(values, label="physical_receipts", maximum=2_000)
    ]
    identities = [(row["table"], row["partition_id"]) for row in rows]
    expected = [(row["table"], row["partition_id"]) for row in plan["partition_rows"]]
    if sorted(identities) != sorted(expected) or len(identities) != len(set(identities)):
        raise FundamentalProviderEvidenceError("physical request keyset is not closed")
    return sorted(rows, key=lambda row: row["receipt_id"].encode("ascii"))


def _logical_coverages(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    physical_receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    del endpoint_plans
    validated_plan = plan
    physical_by_table = {
        table: [row for row in physical_receipts if row["table"] == table]
        for table in SOURCE_TABLES
    }
    cache: dict[
        tuple[str, str, str],
        tuple[list[Mapping[str, Any]], set[str]],
    ] = {}
    rows: list[dict[str, Any]] = []
    for value in _documents(values, label="logical_coverages", maximum=60_000):
        key = (value.get("table"), value.get("expected_start"), value.get("expected_end"))
        if any(type(item) is not str for item in key) or key[0] not in SOURCE_TABLES:
            raise FundamentalProviderEvidenceError("logical coverage interval identity is invalid")
        typed_key = (key[0], key[1], key[2])
        if typed_key not in cache:
            table, start, end = typed_key
            selected = [
                row
                for row in physical_by_table[table]
                if start <= row["partition_id"].split("=", 1)[1] <= end
            ]
            cache[typed_key] = (
                selected,
                _partition_ids(
                    validated_plan,
                    table=table,
                    expected_start=start,
                    expected_end=end,
                ),
            )
        receipts, partition_ids = cache[typed_key]
        rows.append(
            _validate_logical_coverage_validated(
                value,
                validated_plan=validated_plan,
                receipts=receipts,
                expected_partition_ids=partition_ids,
            )
        )
    identities = [(row["company_code"], row["table"]) for row in rows]
    expected = [(symbol, table) for symbol in plan["symbols"] for table in SOURCE_TABLES]
    if sorted(identities) != sorted(expected) or len(identities) != len(set(identities)):
        raise FundamentalProviderEvidenceError("logical coverage keyset is not closed")
    return sorted(rows, key=lambda row: row["coverage_id"].encode("ascii"))


def _raw_evidence(
    values: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        validate_raw_table_evidence(
            value,
            plan=plan,
            endpoint_plans=endpoint_plans,
        )
        for value in _documents(values, label=f"{lane.lower()}_raw_evidence", maximum=6)
    ]
    if (
        len(rows) != len(SOURCE_TABLES)
        or {row["table"] for row in rows} != set(SOURCE_TABLES)
        or any(row["lane"] != lane for row in rows)
    ):
        raise FundamentalProviderEvidenceError(f"{lane} raw evidence set is invalid")
    return sorted(rows, key=lambda row: row["evidence_id"].encode("ascii"))


def _output_refs(value: Mapping[str, Mapping[str, Any]], *, reconciled_at: str) -> dict:
    if type(value) is not dict or set(value) != set(_OUTPUT_PATHS):
        raise FundamentalProviderEvidenceError("comparison output ref set is invalid")
    result: dict[str, dict[str, str]] = {}
    for name, required_path in _OUTPUT_PATHS.items():
        item = exact_ref(value[name], label=f"comparison_output_refs.{name}")
        if item["relative_path"] != required_path or item["available_at"] > reconciled_at:
            raise FundamentalProviderEvidenceError("comparison output ref is not durable or timely")
        result[name] = item
    return result


def _derived(value: Mapping[str, Mapping[str, Any]]) -> tuple[dict, bool]:
    if type(value) is not dict or set(value) != set(_DERIVED_TABLES):
        raise FundamentalProviderEvidenceError("derived fingerprint set is invalid")
    result: dict[str, dict[str, str]] = {}
    for table in _DERIVED_TABLES:
        row = require_exact_keys(
            value[table],
            {"baseline_sha256", "vip_sha256"},
            label=f"derived_fingerprints.{table}",
        )
        baseline = sha256(row["baseline_sha256"], label=f"{table}.baseline_sha256")
        vip = sha256(row["vip_sha256"], label=f"{table}.vip_sha256")
        result[table] = {"baseline_sha256": baseline, "vip_sha256": vip}
    return result, all(row["baseline_sha256"] == row["vip_sha256"] for row in result.values())


def _comparison_hashes(
    comparison: Mapping[str, Any],
    *,
    coverages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], bool]:
    coverage_diff = [
        {"coverage_id": row["coverage_id"], "status": row["status"]}
        for row in coverages
        if row["status"] != "COMPLETE"
    ]
    values = {
        "coverage_diff": coverage_diff,
        "duplicate_diff": comparison["duplicate_diff"],
        "raw_row_diff": comparison["raw_row_diff"],
        "raw_value_diff": comparison["raw_value_diff"],
    }
    return (
        {
            name: hashlib.sha256(canonical_bytes(value)).hexdigest()
            for name, value in values.items()
        },
        not coverage_diff,
    )


def _content_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    identity_field: str,
) -> list[dict[str, str]]:
    rows = [content_ref(value, identity_field=identity_field) for value in values]
    return sorted(
        rows,
        key=lambda row: (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        ),
    )


@provider_evidence_contract
def build_fundamental_reconciliation_receipt(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    physical_receipts: Sequence[Mapping[str, Any]],
    logical_coverages: Sequence[Mapping[str, Any]],
    baseline_raw_evidence: Sequence[Mapping[str, Any]],
    vip_raw_evidence: Sequence[Mapping[str, Any]],
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
    comparison_policy: Mapping[str, Any],
    comparison_output_refs: Mapping[str, Mapping[str, Any]],
    derived_fingerprints: Mapping[str, Mapping[str, Any]],
    reconciled_at: str,
) -> dict[str, Any]:
    """Seal the only v4 reconciliation result eligible for promotion."""

    validated_plan = validate_fundamental_request_plan(
        plan,
        endpoint_plans=endpoint_plans,
    )
    reconciled = timestamp(reconciled_at, label="reconciled_at")
    if reconciled < validated_plan["created_at"]:
        raise FundamentalProviderEvidenceError("reconciliation predates its request plan")
    physical = _physical_receipts(
        physical_receipts,
        plan=validated_plan,
        endpoint_plans=endpoint_plans,
    )
    coverages = _logical_coverages(
        logical_coverages,
        plan=validated_plan,
        endpoint_plans=endpoint_plans,
        physical_receipts=physical,
    )
    baseline_evidence = _raw_evidence(
        baseline_raw_evidence,
        lane="BASELINE",
        plan=validated_plan,
        endpoint_plans=endpoint_plans,
    )
    vip_evidence = _raw_evidence(
        vip_raw_evidence,
        lane="VIP",
        plan=validated_plan,
        endpoint_plans=endpoint_plans,
    )
    policy = validate_fundamental_comparison_policy(comparison_policy)
    comparison = compare_fundamental_raw_tables(
        baseline_tables=baseline_tables,
        vip_tables=vip_tables,
        policy=policy,
    )
    outputs = _output_refs(comparison_output_refs, reconciled_at=reconciled)
    derived, derived_passed = _derived(derived_fingerprints)
    output_hashes, coverage_passed = _comparison_hashes(comparison, coverages=coverages)
    output_hashes["derived_fingerprints"] = hashlib.sha256(canonical_bytes(derived)).hexdigest()
    if any(outputs[name]["byte_sha256"] != output_hashes[name] for name in _OUTPUT_PATHS):
        raise FundamentalProviderEvidenceError("comparison output bytes do not match replay")
    vip_attempts = sum(row["attempts"] for row in physical)
    baseline_attempts = validated_plan["baseline_network_attempts"]
    performance_passed = vip_attempts * 10 <= baseline_attempts
    blockers: list[str] = []
    if not comparison["passed"]:
        blockers.append("RAW_RECONCILIATION_MISMATCH")
    if not coverage_passed:
        blockers.append("LOGICAL_COVERAGE_INCOMPLETE")
    if not derived_passed:
        blockers.append("DERIVED_FINGERPRINT_MISMATCH")
    if not performance_passed:
        blockers.append("VIP_NETWORK_ATTEMPT_LIMIT_EXCEEDED")
    body = {
        **common_fields(timestamp_value=reconciled),
        "array_order_semantics": {
            "/baseline_raw_evidence_refs": "content ref tuple ASCII ascending",
            "/blocker_codes": "ASCII ascending",
            "/logical_coverage_refs": "content ref tuple ASCII ascending",
            "/physical_receipt_refs": "content ref tuple ASCII ascending",
            "/vip_raw_evidence_refs": "content ref tuple ASCII ascending",
        },
        "baseline_network_attempts": baseline_attempts,
        "baseline_raw_evidence_refs": _content_refs(
            baseline_evidence,
            identity_field="evidence_id",
        ),
        "blocker_codes": sorted(blockers),
        "comparison_output_refs": outputs,
        "comparison_output_sha256": output_hashes,
        "comparison_policy_ref": content_ref(policy, identity_field="policy_id"),
        "coverage_passed": coverage_passed,
        "derived_comparison_passed": derived_passed,
        "derived_fingerprints": derived,
        "logical_coverage_refs": _content_refs(
            coverages,
            identity_field="coverage_id",
        ),
        "performance_gate_passed": performance_passed,
        "physical_receipt_refs": _content_refs(
            physical,
            identity_field="receipt_id",
        ),
        "plan_ref": content_ref(validated_plan, identity_field="plan_id"),
        "raw_comparison_passed": comparison["passed"],
        "status": "PASSED" if not blockers else "BLOCKED",
        "version": RECONCILIATION_RECEIPT_SCHEMA,
        "vip_network_attempts": vip_attempts,
        "vip_raw_evidence_refs": _content_refs(
            vip_evidence,
            identity_field="evidence_id",
        ),
    }
    return seal(body, identity_field="receipt_id")


@provider_evidence_contract
def validate_fundamental_reconciliation_receipt(
    document: Mapping[str, Any],
    **closure: Any,
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="receipt_id")
    require_exact_keys(value, _FIELDS, label="Fundamental reconciliation receipt")
    if value.get("version") != RECONCILIATION_RECEIPT_SCHEMA:
        raise FundamentalProviderEvidenceError("reconciliation receipt version mismatch")
    expected = build_fundamental_reconciliation_receipt(
        **closure,
        reconciled_at=value["timestamp"],
    )
    if value != expected:
        raise FundamentalProviderEvidenceError("reconciliation receipt replay mismatch")
    return value


__all__ = [
    "build_fundamental_reconciliation_receipt",
    "validate_fundamental_reconciliation_receipt",
]
