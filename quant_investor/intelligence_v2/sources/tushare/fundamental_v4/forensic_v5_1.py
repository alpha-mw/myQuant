"""Offline-only Fundamental reconciliation forensic and next-cycle epoch plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any, Final

from ...._core import (
    canonical_bytes,
    common_fields,
    identifier,
    require_exact_keys,
    seal,
    session_date,
    sha256,
    sorted_unique,
    validate_seal,
)
from .models import FundamentalV4ContractError, fundamental_v4_contract

FORENSIC_RECEIPT_VERSION: Final = "myquant.v17.fundamental-forensic-receipt.v5.1"
SAME_EPOCH_PLAN_VERSION: Final = "myquant.v17.fundamental-same-epoch-plan.v1"
FORENSIC_CLASSIFICATION: Final = "INSUFFICIENT_TO_DISAMBIGUATE"
NEXT_REQUIRED_EVIDENCE: Final = "SAME_SEALED_ACQUISITION_EPOCH_REQUIRED"
SOURCE_TABLES: Final = (
    "balancesheet",
    "cashflow",
    "daily_basic",
    "fina_indicator",
    "forecast",
    "income",
)

REQUIRED_EPOCH_BINDINGS: Final = (
    "baseline_checkpoint_ref",
    "baseline_provider_manifest_ref",
    "comparison_policy_ref",
    "epoch_close_evidence_ref",
    "epoch_open_evidence_ref",
    "official_partition_plan_ref",
    "pit_cutoff",
    "request_topology_sha256",
    "source_execution_closure_ref",
    "subject_scope_sha256",
    "symbol_set_sha256",
    "vip_checkpoint_ref",
)

STOP_CODES: Final = (
    "ACQUISITION_BLOCKED",
    "DERIVED_FINGERPRINT_MISMATCH",
    "FUNDAMENTAL_BASELINE_NOT_FRESH",
    "FUNDAMENTAL_BASELINE_REF_MISMATCH",
    "FUNDAMENTAL_CAPTURE_EPOCH_UNBOUND",
    "LOGICAL_COVERAGE_INCOMPLETE",
    "RAW_RECONCILIATION_MISMATCH",
    "RECONCILIATION_BLOCKED",
)

_SUMMARY_FIELDS: Final = {
    "checkpoint_execution_bundle_sha256",
    "diff_counts",
    "file_sha256",
    "implementation_commit",
    "package_sha256",
    "passed",
    "physical_receipt_count",
    "status",
    "transport_calls",
    "version",
}


def _table_object(value: Any, *, label: str) -> dict[str, Any]:
    document = require_exact_keys(value, set(SOURCE_TABLES), label=label)
    canonical_bytes(document)
    return document


def _file_sha(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _diff_counts(summary: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    rows = _table_object(summary.get("diff_counts"), label="summary.diff_counts")
    result: dict[str, dict[str, int]] = {}
    for table in SOURCE_TABLES:
        row = require_exact_keys(rows[table], {"row", "value"}, label=f"diff_counts.{table}")
        if type(row["row"]) is not int or type(row["value"]) is not int:
            raise FundamentalV4ContractError("diff counts must be integers")
        if row["row"] < 0 or row["value"] < 0:
            raise FundamentalV4ContractError("diff counts must be nonnegative")
        result[table] = {"row": row["row"], "value": row["value"]}
    return result


def _require_current_diff_shape(
    *,
    counts: Mapping[str, Mapping[str, int]],
    row_diff: Mapping[str, Any],
    value_diff: Mapping[str, Any],
    duplicate_diff: Mapping[str, Any],
) -> None:
    for table in SOURCE_TABLES:
        expected = (
            {"row": 2, "value": 2}
            if table == "fina_indicator"
            else {
                "row": 0,
                "value": 0,
            }
        )
        if dict(counts[table]) != expected:
            raise FundamentalV4ContractError("forensic diff shape is outside the sealed incident")
        if type(row_diff[table]) is not list or len(row_diff[table]) != expected["row"]:
            raise FundamentalV4ContractError("raw row diff count mismatch")
        if type(value_diff[table]) is not list or len(value_diff[table]) != expected["value"]:
            raise FundamentalV4ContractError("raw value diff count mismatch")
        duplicate = require_exact_keys(
            duplicate_diff[table],
            {"baseline_duplicate_row_count", "vip_duplicate_row_count"},
            label=f"duplicate_diff.{table}",
        )
        if duplicate != {
            "baseline_duplicate_row_count": 0,
            "vip_duplicate_row_count": 0,
        }:
            raise FundamentalV4ContractError("duplicate rows are outside the sealed incident")


def _diff_hashes(
    row_diff: Mapping[str, Any], value_diff: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    row_hashes: list[str] = []
    for index, raw in enumerate(row_diff["fina_indicator"]):
        row = require_exact_keys(
            raw,
            {"baseline_count", "row_sha256", "vip_count"},
            label=f"raw_row_diff.fina_indicator[{index}]",
        )
        if (row["baseline_count"], row["vip_count"]) not in {(1, 0), (0, 1)}:
            raise FundamentalV4ContractError("row diff direction is invalid")
        row_hashes.append(sha256(row["row_sha256"], label="row_sha256"))
    key_hashes: list[str] = []
    for index, raw in enumerate(value_diff["fina_indicator"]):
        row = require_exact_keys(
            raw,
            {"baseline_winner_sha256", "key_sha256", "vip_winner_sha256"},
            label=f"raw_value_diff.fina_indicator[{index}]",
        )
        baseline = row["baseline_winner_sha256"]
        vip = row["vip_winner_sha256"]
        if (baseline is None) == (vip is None):
            raise FundamentalV4ContractError("winner diff direction is invalid")
        if baseline is not None:
            sha256(baseline, label="baseline_winner_sha256")
        if vip is not None:
            sha256(vip, label="vip_winner_sha256")
        key_hashes.append(sha256(row["key_sha256"], label="key_sha256"))
    return sorted(row_hashes), sorted(key_hashes)


@fundamental_v4_contract
def build_fundamental_forensic_receipt_v5_1(
    *,
    produced_at: str,
    subject_id: str,
    period: str,
    baseline_ann_date: str,
    vip_ann_date: str,
    subject_binding_source_sha256: str,
    expected_row_sha256: Sequence[str],
    expected_key_sha256: Sequence[str],
    summary: Mapping[str, Any],
    raw_row_diff: Mapping[str, Any],
    raw_value_diff: Mapping[str, Any],
    duplicate_diff: Mapping[str, Any],
    table_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_summary = require_exact_keys(summary, _SUMMARY_FIELDS, label="summary")
    normalized_row = _table_object(raw_row_diff, label="raw_row_diff")
    normalized_value = _table_object(raw_value_diff, label="raw_value_diff")
    normalized_duplicate = _table_object(duplicate_diff, label="duplicate_diff")
    normalized_table = _table_object(table_evidence, label="table_evidence")
    counts = _diff_counts(normalized_summary)
    _require_current_diff_shape(
        counts=counts,
        row_diff=normalized_row,
        value_diff=normalized_value,
        duplicate_diff=normalized_duplicate,
    )
    declared_files = require_exact_keys(
        normalized_summary["file_sha256"],
        {
            "duplicate_diff.json",
            "raw_row_diff.json",
            "raw_value_diff.json",
            "table_evidence.json",
        },
        label="summary.file_sha256",
    )
    evidence = {
        "duplicate_diff.json": normalized_duplicate,
        "raw_row_diff.json": normalized_row,
        "raw_value_diff.json": normalized_value,
        "table_evidence.json": normalized_table,
    }
    evidence_shas = {name: _file_sha(document) for name, document in evidence.items()}
    if declared_files != evidence_shas:
        raise FundamentalV4ContractError("forensic source SHA mismatch")
    if (
        normalized_summary["passed"] is not False
        or normalized_summary["status"] != "RECONCILIATION_BLOCKED"
        or normalized_summary["transport_calls"] != 0
    ):
        raise FundamentalV4ContractError("forensic source is not a zero-network blocker")
    row_hashes, key_hashes = _diff_hashes(normalized_row, normalized_value)
    declared_rows = sorted_unique(expected_row_sha256, label="expected_row_sha256", maximum=2)
    declared_keys = sorted_unique(expected_key_sha256, label="expected_key_sha256", maximum=2)
    if row_hashes != declared_rows or key_hashes != declared_keys:
        raise FundamentalV4ContractError("subject binding hashes do not match raw diff")
    baseline_date = session_date(baseline_ann_date, label="baseline_ann_date")
    vip_date = session_date(vip_ann_date, label="vip_ann_date")
    if baseline_date == vip_date:
        raise FundamentalV4ContractError("forensic incident requires different announcement dates")
    return seal(
        {
            **common_fields(timestamp_value=produced_at),
            "classification": FORENSIC_CLASSIFICATION,
            "embedded_evidence": {
                "duplicate_diff": normalized_duplicate,
                "raw_row_diff": normalized_row,
                "raw_value_diff": normalized_value,
                "summary": normalized_summary,
                "table_evidence": normalized_table,
            },
            "evidence_file_sha256": evidence_shas,
            "next_required_evidence": NEXT_REQUIRED_EVIDENCE,
            "possible_causes": [
                "ACQUISITION_EPOCH_MISMATCH",
                "PROVIDER_REVISION_OR_RESTATEMENT_DRIFT",
            ],
            "promotion_authorized": False,
            "subject_binding": {
                "baseline_ann_date": baseline_date,
                "expected_key_sha256": declared_keys,
                "expected_row_sha256": declared_rows,
                "period": session_date(period, label="period"),
                "subject_binding_source_sha256": sha256(
                    subject_binding_source_sha256,
                    label="subject_binding_source_sha256",
                ),
                "subject_id": identifier(subject_id, label="subject_id"),
                "vip_ann_date": vip_date,
            },
            "tolerance_applied": False,
            "transport_calls": 0,
            "version": FORENSIC_RECEIPT_VERSION,
        },
        identity_field="forensic_receipt_id",
    )


@fundamental_v4_contract
def validate_fundamental_forensic_receipt_v5_1(
    document: Mapping[str, Any], *, subject_binding_source_sha256: str
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="forensic_receipt_id")
    binding = normalized.get("subject_binding", {})
    evidence = normalized.get("embedded_evidence", {})
    expected = build_fundamental_forensic_receipt_v5_1(
        produced_at=normalized.get("timestamp"),
        subject_id=binding.get("subject_id"),
        period=binding.get("period"),
        baseline_ann_date=binding.get("baseline_ann_date"),
        vip_ann_date=binding.get("vip_ann_date"),
        subject_binding_source_sha256=subject_binding_source_sha256,
        expected_row_sha256=binding.get("expected_row_sha256", ()),
        expected_key_sha256=binding.get("expected_key_sha256", ()),
        summary=evidence.get("summary", {}),
        raw_row_diff=evidence.get("raw_row_diff", {}),
        raw_value_diff=evidence.get("raw_value_diff", {}),
        duplicate_diff=evidence.get("duplicate_diff", {}),
        table_evidence=evidence.get("table_evidence", {}),
    )
    if normalized != expected:
        raise FundamentalV4ContractError("forensic receipt replay mismatch")
    return normalized


@fundamental_v4_contract
def build_inert_same_epoch_plan_v1(
    *, produced_at: str, forensic_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    sealed = validate_seal(forensic_receipt, identity_field="forensic_receipt_id")
    binding = sealed.get("subject_binding")
    if type(binding) is not dict:
        raise FundamentalV4ContractError("same-epoch plan requires a valid subject binding")
    normalized = validate_fundamental_forensic_receipt_v5_1(
        sealed,
        subject_binding_source_sha256=binding.get("subject_binding_source_sha256"),
    )
    return seal(
        {
            **common_fields(timestamp_value=produced_at),
            "campaign_state": "BLOCKED_PENDING_BOUND_INPUTS",
            "execution_authorized": False,
            "forensic_ref": {
                "artifact_id": normalized["forensic_receipt_id"],
                "artifact_version": normalized["version"],
                "byte_sha256": hashlib.sha256(canonical_bytes(normalized)).hexdigest(),
                "semantic_sha256": normalized["semantic_sha256"],
            },
            "missing_binding_fields": list(REQUIRED_EPOCH_BINDINGS),
            "network_attempts_executed": 0,
            "next_cycle_only": True,
            "pointer_mutation_authorized": False,
            "promotion_authorized": False,
            "required_binding_fields": list(REQUIRED_EPOCH_BINDINGS),
            "reuse_archived_baseline": False,
            "stop_codes": list(STOP_CODES),
            "terminal_success_condition": "EXACT_ZERO_DIFF_ALL_SIX_TABLES",
            "tolerance_permitted": False,
            "version": SAME_EPOCH_PLAN_VERSION,
        },
        identity_field="same_epoch_plan_id",
    )


@fundamental_v4_contract
def validate_inert_same_epoch_plan_v1(
    document: Mapping[str, Any], *, forensic_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="same_epoch_plan_id")
    expected = build_inert_same_epoch_plan_v1(
        produced_at=normalized.get("timestamp"), forensic_receipt=forensic_receipt
    )
    if normalized != expected:
        raise FundamentalV4ContractError("same-epoch plan replay mismatch")
    return normalized


__all__ = [
    "FORENSIC_CLASSIFICATION",
    "FORENSIC_RECEIPT_VERSION",
    "NEXT_REQUIRED_EVIDENCE",
    "REQUIRED_EPOCH_BINDINGS",
    "SAME_EPOCH_PLAN_VERSION",
    "STOP_CODES",
    "build_fundamental_forensic_receipt_v5_1",
    "build_inert_same_epoch_plan_v1",
    "validate_fundamental_forensic_receipt_v5_1",
    "validate_inert_same_epoch_plan_v1",
]
