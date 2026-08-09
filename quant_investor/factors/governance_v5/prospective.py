"""Prospective-only factor evaluation evidence for Governance v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from ._core import (
    FactorGovernanceV5Error,
    common_fields,
    decimal_text,
    decimal_value,
    identifier,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .contracts import validate_governance_policy, validate_preregistration

EVALUATION_VERSION: Final = "factor-prospective-evaluation-receipt.v5"
DIAGNOSTIC_VERSION: Final = "factor-diagnostic-scan-receipt.v5"


def build_diagnostic_scan_receipt(
    *,
    scanned_at: str,
    implementation_sha256: str,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    rows = [identifier(value, label="candidate_id") for value in candidate_ids]
    if not rows or len(rows) != len(set(rows)):
        raise FactorGovernanceV5Error("diagnostic candidate IDs are empty or duplicated")
    return seal(
        {
            **common_fields(timestamp_value=scanned_at),
            "version": DIAGNOSTIC_VERSION,
            "candidate_ids": sorted(rows, key=lambda value: value.encode("ascii")),
            "implementation_sha256": sha256(implementation_sha256, label="implementation_sha256"),
            "lane": "DIAGNOSTIC_ONLY",
            "promotion_eligible": False,
            "purpose": "NEXT_CYCLE_PREREGISTRATION_PROPOSAL_ONLY",
        },
        identity_field="diagnostic_receipt_id",
    )


def _path_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise FactorGovernanceV5Error("path rows must be a nonempty sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if type(raw) is not dict or set(raw) != {
            "path_id",
            "path_ic",
            "purge_proof_sha256",
            "split_sha256",
            "test_block_ids",
        }:
            raise FactorGovernanceV5Error(f"path_rows[{index}] shape is invalid")
        path_id = identifier(raw["path_id"], label="path_id")
        if path_id in seen:
            raise FactorGovernanceV5Error("duplicate prospective path")
        seen.add(path_id)
        block_ids = raw["test_block_ids"]
        if (
            isinstance(block_ids, (str, bytes))
            or not isinstance(block_ids, Sequence)
            or not block_ids
            or any(type(value) is not int or value < 0 for value in block_ids)
            or list(block_ids) != sorted(set(block_ids))
        ):
            raise FactorGovernanceV5Error("test_block_ids are invalid")
        normalized.append(
            {
                "path_id": path_id,
                "path_ic": decimal_text(decimal_value(raw["path_ic"], label="path_ic")),
                "purge_proof_sha256": sha256(raw["purge_proof_sha256"], label="purge_proof_sha256"),
                "split_sha256": sha256(raw["split_sha256"], label="split_sha256"),
                "test_block_ids": list(block_ids),
            }
        )
    return sorted(normalized, key=lambda row: row["path_id"].encode("ascii"))


def build_prospective_evaluation(
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    candidate_id: str,
    path_rows: Sequence[Mapping[str, Any]],
    evaluation_available_at: str,
    label_source_sha256: str,
    implementation_sha256: str,
    admitted: bool,
) -> dict[str, Any]:
    if type(admitted) is not bool:
        raise FactorGovernanceV5Error("admitted must be boolean")
    normalized_policy = validate_governance_policy(policy)
    normalized_prereg = validate_preregistration(preregistration, policy=policy)
    candidate = identifier(candidate_id, label="candidate_id")
    declared = {row["candidate_id"] for row in normalized_prereg["candidates"]}
    if candidate not in declared:
        raise FactorGovernanceV5Error("evaluation candidate is not preregistered")
    available = timestamp(evaluation_available_at, label="evaluation_available_at")
    if available < normalized_prereg["label_available_at"]:
        raise FactorGovernanceV5Error("prospective evaluation predates label availability")
    rows = _path_rows(path_rows)
    if len(rows) < normalized_policy["minimum_prospective_paths"]:
        raise FactorGovernanceV5Error("prospective path count is below policy")
    path_values = [decimal_value(row["path_ic"], label="path_ic") for row in rows]
    mean_path_ic = sum(path_values, Decimal("0")) / Decimal(len(path_values))
    candidate_row = next(
        row for row in normalized_prereg["candidates"] if row["candidate_id"] == candidate
    )
    parameter_status = (
        "NOT_APPLICABLE"
        if candidate_row["parameterization"] == "NONE"
        else "PREREGISTERED_PATHS_REQUIRED"
    )
    return seal(
        {
            **common_fields(timestamp_value=available),
            "version": EVALUATION_VERSION,
            "admitted": admitted,
            "candidate_id": candidate,
            "embargo_open_sessions": normalized_policy["embargo_open_sessions"],
            "implementation_sha256": sha256(implementation_sha256, label="implementation_sha256"),
            "label_source_sha256": sha256(label_source_sha256, label="label_source_sha256"),
            "lane": "PROSPECTIVE_ONLY",
            "mean_path_ic": decimal_text(mean_path_ic),
            "path_count": len(rows),
            "path_rows": rows,
            "parameter_stability_status": parameter_status,
            "policy_ref": normalized_policy["policy_id"],
            "preregistration_ref": normalized_prereg["preregistration_id"],
            "purge_open_sessions": normalized_policy["purge_open_sessions"],
        },
        identity_field="evaluation_receipt_id",
    )


def validate_prospective_evaluation(
    document: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = validate_seal(document, identity_field="evaluation_receipt_id")
    expected = build_prospective_evaluation(
        policy=policy,
        preregistration=preregistration,
        candidate_id=sealed["candidate_id"],
        path_rows=sealed.get("path_rows", ()),
        evaluation_available_at=sealed["timestamp"],
        label_source_sha256=sealed["label_source_sha256"],
        implementation_sha256=sealed["implementation_sha256"],
        admitted=sealed["admitted"],
    )
    if sealed != expected:
        raise FactorGovernanceV5Error("prospective evaluation replay mismatch")
    return sealed


def historical_support_projection(
    *,
    candidate_id: str,
    mean_path_ic: Any,
    path_count: int,
    produced_at: str,
    source_sha256: str,
) -> dict[str, Any]:
    if type(path_count) is not int or path_count < 1:
        raise FactorGovernanceV5Error("historical path_count must be positive")
    return seal(
        {
            **common_fields(timestamp_value=produced_at),
            "version": "factor-historical-support-receipt.v5",
            "admission_eligible": False,
            "candidate_id": identifier(candidate_id, label="candidate_id"),
            "lane": "BACKTEST_SUPPORT_ONLY",
            "mean_path_ic": decimal_text(decimal_value(mean_path_ic, label="mean_path_ic")),
            "path_count": path_count,
            "source_sha256": sha256(source_sha256, label="source_sha256"),
        },
        identity_field="support_receipt_id",
    )


__all__ = [
    "DIAGNOSTIC_VERSION",
    "EVALUATION_VERSION",
    "build_diagnostic_scan_receipt",
    "build_prospective_evaluation",
    "historical_support_projection",
    "validate_prospective_evaluation",
]
