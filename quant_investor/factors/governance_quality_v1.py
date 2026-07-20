"""Report-only quality qualification for FactorGovernanceProtocol v4 records.

This module deliberately excludes production state, allocation, activation
receipts, and risk authorization.  It qualifies an explicit research record
set for shadow observation while preserving the separate v4 production gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors.governance_protocol_v4 import (
    FACTOR_EVIDENCE_SCHEMA_VERSION,
    FDR_Q,
    MIN_MONTH_END_RANKIC_COUNT,
    MIN_NONOVERLAP_30D_COHORT_COUNT,
    PROTOCOL_VERSION,
    REQUIRED_GATE_IDS,
    TARGET_PRODUCTION_FACTOR_COUNT,
    _factor_maturity,
    _gate_status,
    _is_sha256,
    canonical_json_bytes,
    semantic_sha256,
)

QUALITY_INPUT_SCHEMA_VERSION = "factor-quality-input.v1"
QUALITY_POLICY_SCHEMA_VERSION = "factor-quality-policy.v1"
QUALITY_READINESS_SCHEMA_VERSION = "factor-quality-readiness.v1"
QUALITY_ROW_SCHEMA_VERSION = "factor-quality-assessment-row.v1"
RUNTIME_CONTRACT_SCHEMA_VERSION = "factor-production-runtime-contract.v4"
MIN_SHADOW_FACTOR_COUNT = 3
MIN_QUALITY_READY_FACTOR_COUNT = 5
MIN_QUALITY_FAMILY_COUNT = 3

SOURCE_RECORD_FIELDS = (
    "name",
    "family",
    "slot",
    "calendar_sha256",
    "gate_results",
    "maturity",
    "bh_q_value",
    "fdr_method",
    "runtime_contract",
    "runtime_contract_sha256",
    "runtime_contract_status",
    "evidence",
    "health",
)

ROW_FIELDS = {
    "schema_version",
    "name",
    "family",
    "slot",
    "calendar_sha256",
    "gate_status",
    "maturity",
    "bh_q_value",
    "fdr_method",
    "runtime_contract_schema_version",
    "runtime_contract_factor_name",
    "runtime_contract_set_identity_sha256",
    "runtime_contract_sha256",
    "replay_semantic_sha256",
    "health_status",
    "health_fresh",
    "data_blocked",
    "qualified",
    "blockers",
}

ASSESSMENT_FIELDS = {
    "schema_version",
    "protocol_version",
    "quality_policy_hash",
    "status",
    "input_valid",
    "input_error",
    "report_only",
    "production_authority",
    "quality_ready",
    "shadow_observation_eligible",
    "quality_factor_target",
    "quality_factor_count",
    "quality_family_count",
    "qualified_factor_count",
    "qualified_family_count",
    "quality_set_identity_sha256",
    "quality_set_sha256",
    "expected_quality_set_sha256",
    "source_records",
    "factors",
    "blockers",
    "assessment_sha256",
}


class FactorQualityV1Error(ValueError):
    """Raised when a persisted quality assessment fails exact validation."""


def factor_quality_policy() -> dict[str, Any]:
    """Return the independent, report-only quality policy."""

    return {
        "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
        "source_protocol_version": PROTOCOL_VERSION,
        "record_requirements": {
            "required_source_fields": list(SOURCE_RECORD_FIELDS),
            "required_gate_ids": list(REQUIRED_GATE_IDS),
            "maturity": {
                "month_end_rankic_count": MIN_MONTH_END_RANKIC_COUNT,
                "nonoverlap_30d_cohort_count": MIN_NONOVERLAP_30D_COHORT_COUNT,
            },
            "multiple_testing": {
                "method": "benjamini_hochberg_by_family",
                "q_lte": FDR_Q,
            },
            "runtime_contract_schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
            "runtime_set_identity_binding_required": True,
            "evidence_schema_version": FACTOR_EVIDENCE_SCHEMA_VERSION,
            "fresh_healthy_health_required": True,
        },
        "set_requirements": {
            "unique_names": True,
            "unique_slots": True,
            "shadow_factor_count_min": MIN_SHADOW_FACTOR_COUNT,
            "quality_ready_factor_count_min": MIN_QUALITY_READY_FACTOR_COUNT,
            "quality_ready_factor_count_max": TARGET_PRODUCTION_FACTOR_COUNT,
            "family_count_min": MIN_QUALITY_FAMILY_COUNT,
            "identity_hash_fields": ["name", "family", "slot"],
            "content_hash_fields": [
                "name",
                "family",
                "slot",
                "calendar_sha256",
                "runtime_contract_sha256",
                "replay_semantic_sha256",
            ],
        },
        "excluded_production_requirements": [
            "production_factor_state",
            "positive_weight",
            "factor_allocation_cap",
            "family_allocation_cap",
            "activation_receipt",
        ],
        "report_only": True,
        "production_authority": False,
        "legacy_evidence": {"v2": "reject", "v3": "reject"},
    }


def factor_quality_policy_hash() -> str:
    return semantic_sha256(factor_quality_policy())


QUALITY_POLICY_HASH = factor_quality_policy_hash()


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _normalize_source_records(value: Any) -> tuple[list[dict[str, Any]], str | None]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or isinstance(value, Mapping)
    ):
        return [], "quality_records_input_not_sequence"
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            return [], f"quality_record_not_mapping:index={index}"
        try:
            canonical = _canonical_copy(dict(raw))
        except (TypeError, ValueError):
            return [], f"quality_record_not_canonical_json:index={index}"
        normalized.append({field: canonical.get(field) for field in SOURCE_RECORD_FIELDS})
    return normalized, None


def _identity_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = [
        {
            "name": str(record.get("name") or "").strip(),
            "family": str(record.get("family") or "").strip(),
            "slot": str(record.get("slot") or "").strip(),
        }
        for record in records
    ]
    return sorted(rows, key=lambda row: (row["name"], row["family"], row["slot"]))


def factor_quality_set_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash only set identity, so runtime contracts can bind it without a cycle."""

    normalized, error = _normalize_source_records(records)
    if error is not None:
        raise FactorQualityV1Error(error)
    return semantic_sha256(_identity_rows(normalized))


def _assess_record(
    record: Mapping[str, Any],
    *,
    quality_set_identity_sha256: str,
) -> dict[str, Any]:
    name = str(record.get("name") or "").strip()
    family = str(record.get("family") or "").strip()
    slot = str(record.get("slot") or "").strip()
    label = name or "<missing>"
    blockers: list[str] = []
    if not name:
        blockers.append("quality_factor_name_missing")
    if not family:
        blockers.append("quality_factor_family_missing")
    if not slot:
        blockers.append("quality_factor_slot_missing")

    calendar_sha256 = record.get("calendar_sha256")
    if not _is_sha256(calendar_sha256):
        blockers.append("quality_calendar_sha256_invalid")

    gates = _gate_status(record.get("gate_results"))
    missing_gates = [gate_id for gate_id in REQUIRED_GATE_IDS if gate_id not in gates]
    failed_gates = [gate_id for gate_id in REQUIRED_GATE_IDS if gates.get(gate_id) is False]
    extra_gates = sorted(set(gates) - set(REQUIRED_GATE_IDS))
    if missing_gates:
        blockers.append("quality_eight_gates_missing:" + ",".join(map(str, missing_gates)))
    if failed_gates:
        blockers.append("quality_eight_gates_failed:" + ",".join(map(str, failed_gates)))
    if extra_gates:
        blockers.append("quality_gate_ids_unexpected:" + ",".join(map(str, extra_gates)))

    try:
        maturity = _factor_maturity(record)
    except (TypeError, ValueError):
        maturity = {}
        blockers.append("quality_maturity_input_invalid")
    if maturity.get("mature") is not True:
        blockers.append("quality_maturity_not_met")

    raw_q = record.get("bh_q_value")
    normalized_q: float | None
    if isinstance(raw_q, bool) or not isinstance(raw_q, (int, float)):
        normalized_q = None
        blockers.append("quality_bh_q_value_missing")
    else:
        normalized_q = float(raw_q)
        if not math.isfinite(normalized_q) or not 0.0 <= normalized_q <= FDR_Q:
            blockers.append("quality_bh_q_value_above_0.10")
            normalized_q = None
    fdr_method = str(record.get("fdr_method") or "")
    if fdr_method != "benjamini_hochberg_by_family":
        blockers.append("quality_bh_method_mismatch")

    runtime = record.get("runtime_contract")
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    runtime_schema = str(runtime_mapping.get("schema_version") or "")
    runtime_name = str(runtime_mapping.get("factor_name") or "")
    runtime_set_identity = str(runtime_mapping.get("quality_set_identity_sha256") or "")
    runtime_sha = record.get("runtime_contract_sha256")
    if not runtime_mapping:
        blockers.append("quality_runtime_contract_missing")
    else:
        if runtime_schema != RUNTIME_CONTRACT_SCHEMA_VERSION:
            blockers.append("quality_runtime_contract_schema_not_v4")
        if runtime_name != name:
            blockers.append("quality_runtime_contract_factor_identity_mismatch")
        if runtime_set_identity != quality_set_identity_sha256:
            blockers.append("quality_runtime_set_identity_mismatch")
        if not _is_sha256(runtime_sha):
            blockers.append("quality_runtime_contract_sha256_invalid")
        elif semantic_sha256(dict(runtime_mapping)) != runtime_sha:
            blockers.append("quality_runtime_contract_sha256_mismatch")
    if record.get("runtime_contract_status") != "verified":
        blockers.append("quality_runtime_contract_not_verified")

    evidence = record.get("evidence")
    evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
    replay_sha = evidence_mapping.get("replay_semantic_sha256")
    if not evidence_mapping:
        blockers.append("quality_v4_evidence_missing")
    else:
        if evidence_mapping.get("schema_version") != FACTOR_EVIDENCE_SCHEMA_VERSION:
            blockers.append("quality_v4_evidence_schema_mismatch")
        if evidence_mapping.get("status") != "verified":
            blockers.append("quality_v4_evidence_not_verified")
        if not _is_sha256(replay_sha):
            blockers.append("quality_v4_evidence_replay_sha_invalid")

    health = record.get("health")
    health_mapping = health if isinstance(health, Mapping) else {}
    health_status = str(health_mapping.get("status") or "")
    health_fresh = health_mapping.get("fresh") is True
    data_blocked = health_status == "data_blocked" or bool(
        health_mapping.get("data_blocked", False)
    )
    if not health_mapping:
        blockers.append("quality_fresh_health_missing")
    else:
        if data_blocked:
            blockers.append("quality_health_data_blocked")
        if health_status != "healthy":
            blockers.append("quality_health_not_healthy")
        if not health_fresh:
            blockers.append("quality_health_not_fresh")

    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": QUALITY_ROW_SCHEMA_VERSION,
        "name": label,
        "family": family,
        "slot": slot,
        "calendar_sha256": str(calendar_sha256) if _is_sha256(calendar_sha256) else "",
        "gate_status": {str(key): value for key, value in sorted(gates.items())},
        "maturity": maturity,
        "bh_q_value": normalized_q,
        "fdr_method": fdr_method,
        "runtime_contract_schema_version": runtime_schema,
        "runtime_contract_factor_name": runtime_name,
        "runtime_contract_set_identity_sha256": runtime_set_identity,
        "runtime_contract_sha256": str(runtime_sha) if _is_sha256(runtime_sha) else "",
        "replay_semantic_sha256": str(replay_sha) if _is_sha256(replay_sha) else "",
        "health_status": health_status,
        "health_fresh": health_fresh,
        "data_blocked": data_blocked,
        "qualified": not blockers,
        "blockers": blockers,
    }


def _quality_set_content_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_sha256: str,
) -> str:
    bindings = [
        {
            "name": str(row.get("name") or ""),
            "family": str(row.get("family") or ""),
            "slot": str(row.get("slot") or ""),
            "calendar_sha256": str(row.get("calendar_sha256") or ""),
            "runtime_contract_sha256": str(row.get("runtime_contract_sha256") or ""),
            "replay_semantic_sha256": str(row.get("replay_semantic_sha256") or ""),
        }
        for row in rows
    ]
    bindings.sort(key=lambda row: (row["name"], row["family"], row["slot"]))
    return semantic_sha256({"quality_set_identity_sha256": identity_sha256, "records": bindings})


def _derive_assessment(
    source_records: list[dict[str, Any]],
    *,
    expected_quality_set_sha256: Any = None,
    input_error: str | None = None,
) -> dict[str, Any]:
    identity_sha256 = semantic_sha256(_identity_rows(source_records))
    factors = [
        _assess_record(record, quality_set_identity_sha256=identity_sha256)
        for record in source_records
    ]
    quality_set_sha256 = _quality_set_content_sha256(factors, identity_sha256=identity_sha256)

    blockers: list[str] = []
    integrity_invalid = input_error is not None
    if input_error is not None:
        blockers.append(input_error)

    raw_names = [str(record.get("name") or "").strip() for record in source_records]
    if len(raw_names) != len(set(raw_names)):
        blockers.append("quality_factor_names_not_unique")
        integrity_invalid = True
    raw_slots = [
        str(record.get("slot") or "").strip()
        for record in source_records
        if str(record.get("slot") or "").strip()
    ]
    if len(raw_slots) != len(set(raw_slots)):
        blockers.append("quality_factor_slots_not_unique")
        integrity_invalid = True
    if any("quality_runtime_set_identity_mismatch" in row["blockers"] for row in factors):
        blockers.append("quality_runtime_set_identity_binding_invalid")
        integrity_invalid = True

    normalized_expected: str | None
    if expected_quality_set_sha256 is None:
        normalized_expected = None
    elif _is_sha256(expected_quality_set_sha256):
        normalized_expected = str(expected_quality_set_sha256)
        if normalized_expected != quality_set_sha256:
            blockers.append("quality_set_sha256_mismatch")
            integrity_invalid = True
    else:
        normalized_expected = (
            str(expected_quality_set_sha256)
            if isinstance(expected_quality_set_sha256, str)
            else None
        )
        blockers.append("expected_quality_set_sha256_invalid")
        integrity_invalid = True

    for index, row in enumerate(factors):
        label = row["name"] if row["name"] != "<missing>" else f"row_{index}"
        blockers.extend(f"{label}:{item}" for item in row["blockers"])

    factor_count = len(factors)
    families = {row["family"] for row in factors if row["family"]}
    qualified = [row for row in factors if row["qualified"]]
    qualified_families = {row["family"] for row in qualified if row["family"]}
    qualified_count = len(qualified)

    if integrity_invalid:
        status = "invalid"
    elif factor_count == 0 or qualified_count == 0:
        status = "blocked"
    elif qualified_count != factor_count:
        status = "partially_qualified"
    elif factor_count < MIN_SHADOW_FACTOR_COUNT:
        status = "insufficient_for_shadow"
    elif len(qualified_families) < MIN_QUALITY_FAMILY_COUNT:
        status = "blocked"
    elif factor_count < MIN_QUALITY_READY_FACTOR_COUNT:
        status = "shadow_observation"
    elif factor_count < TARGET_PRODUCTION_FACTOR_COUNT:
        status = "ready_underfilled"
    elif factor_count == TARGET_PRODUCTION_FACTOR_COUNT:
        status = "ready_target_10"
    else:
        status = "shadow_observation_above_target"

    if not integrity_invalid:
        if factor_count == 0:
            blockers.append("quality_records_empty")
        elif qualified_count == 0:
            blockers.append("quality_no_factors_qualified")
        elif qualified_count != factor_count:
            blockers.append("quality_records_not_all_qualified")
        elif factor_count < MIN_SHADOW_FACTOR_COUNT:
            blockers.append("quality_factor_count_below_shadow_3")
        elif len(qualified_families) < MIN_QUALITY_FAMILY_COUNT:
            blockers.append("quality_family_count_below_3")
        elif factor_count < MIN_QUALITY_READY_FACTOR_COUNT:
            blockers.append("quality_factor_count_below_readiness_5")
        elif factor_count > TARGET_PRODUCTION_FACTOR_COUNT:
            blockers.append("quality_factor_count_above_target_10")

    quality_ready = status in {"ready_underfilled", "ready_target_10"}
    shadow_eligible = status in {
        "shadow_observation",
        "ready_underfilled",
        "ready_target_10",
        "shadow_observation_above_target",
    }
    payload: dict[str, Any] = {
        "schema_version": QUALITY_READINESS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "quality_policy_hash": factor_quality_policy_hash(),
        "status": status,
        "input_valid": not integrity_invalid,
        "input_error": input_error,
        "report_only": True,
        "production_authority": False,
        "quality_ready": quality_ready,
        "shadow_observation_eligible": shadow_eligible,
        "quality_factor_target": TARGET_PRODUCTION_FACTOR_COUNT,
        "quality_factor_count": factor_count,
        "quality_family_count": len(families),
        "qualified_factor_count": qualified_count,
        "qualified_family_count": len(qualified_families),
        "quality_set_identity_sha256": identity_sha256,
        "quality_set_sha256": quality_set_sha256,
        "expected_quality_set_sha256": normalized_expected,
        "source_records": source_records,
        "factors": factors,
        "blockers": list(dict.fromkeys(blockers)),
    }
    payload["assessment_sha256"] = semantic_sha256(payload)
    return payload


def assess_factor_quality_readiness_v4(
    quality_records: Any,
    *,
    expected_quality_set_sha256: Any = None,
) -> dict[str, Any]:
    """Qualify an explicit v4 research set without granting production authority."""

    source_records, input_error = _normalize_source_records(quality_records)
    return _derive_assessment(
        source_records,
        expected_quality_set_sha256=expected_quality_set_sha256,
        input_error=input_error,
    )


def validate_factor_quality_readiness_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every persisted quality claim from normalized source evidence."""

    if not isinstance(value, Mapping) or set(value) != ASSESSMENT_FIELDS:
        raise FactorQualityV1Error("quality assessment fields invalid")
    source_records = value.get("source_records")
    if not isinstance(source_records, list):
        raise FactorQualityV1Error("quality source_records must be an array")
    for record in source_records:
        if not isinstance(record, Mapping) or set(record) != set(SOURCE_RECORD_FIELDS):
            raise FactorQualityV1Error("quality source record fields invalid")
    factors = value.get("factors")
    if not isinstance(factors, list) or any(
        not isinstance(row, Mapping) or set(row) != ROW_FIELDS for row in factors
    ):
        raise FactorQualityV1Error("quality factor assessment rows invalid")
    input_error = value.get("input_error")
    if input_error is not None and not isinstance(input_error, str):
        raise FactorQualityV1Error("quality input_error invalid")

    normalized, normalization_error = _normalize_source_records(source_records)
    if normalization_error is not None or normalized != source_records:
        raise FactorQualityV1Error("quality source_records are not normalized")
    expected = _derive_assessment(
        normalized,
        expected_quality_set_sha256=value.get("expected_quality_set_sha256"),
        input_error=input_error,
    )
    try:
        matches = canonical_json_bytes(dict(value)) == canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise FactorQualityV1Error("quality assessment is not canonical JSON") from exc
    if not matches:
        raise FactorQualityV1Error("quality assessment recomputation mismatch")
    return expected


__all__ = [
    "FactorQualityV1Error",
    "QUALITY_INPUT_SCHEMA_VERSION",
    "QUALITY_POLICY_HASH",
    "QUALITY_POLICY_SCHEMA_VERSION",
    "QUALITY_READINESS_SCHEMA_VERSION",
    "QUALITY_ROW_SCHEMA_VERSION",
    "assess_factor_quality_readiness_v4",
    "factor_quality_policy",
    "factor_quality_policy_hash",
    "factor_quality_set_identity_sha256",
    "validate_factor_quality_readiness_v1",
]
