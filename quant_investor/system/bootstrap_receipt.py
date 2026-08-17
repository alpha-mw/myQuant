"""Detached proof that an operational generation used the production assembler.

The receipt deliberately binds generation inputs rather than a generation ID.
The generation manifest may therefore include the receipt without creating a
content-hash cycle.  Activation must still replay the bound inputs; this
artifact is not an authorization merely because it is well formed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)

from .errors import SystemContractError
from .store import OBJECT_REF_SORT_FIELDS, validate_object_ref

PRODUCTION_BOOTSTRAP_RECEIPT_KIND: Final = "system.production_bootstrap_receipt"
PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256: Final = get_contract(
    PRODUCTION_BOOTSTRAP_RECEIPT_KIND
).contract_sha256
PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS: Final = frozenset(
    {
        "production_bootstrap_receipt_id",
        "state",
        "bootstrap_operator_request_ref",
        "source_root_id",
        "input_source_rows",
        "deployed_release_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_source_limitations",
        "release_code_manifest_sha256",
        "generation_created_at",
        "expected_assembly_id",
        "generation_intent_sha256",
        "mainline_ref",
        "source_refs",
        "factor_source_object_refs",
        "factor_policy_ref",
        "factor_evidence_refs",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "readiness_matrix_ref",
        "emergency_controller_sha256",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "source_blockers",
        "fundamental_machine_states",
        "signal_statistics",
        "signal_statistics_sha256",
        "assembler_module_path",
        "assembler_code_sha256",
    }
)
INPUT_SOURCE_ROW_FIELDS: Final = frozenset(
    {"field", "ordinal", "input_file_ref", "source_object_ref"}
)
ASSEMBLER_MODULE_PATH: Final = "quant_investor/factors/governance/production.py"
PRODUCTION_RESEARCH_ROLE: Final = "SOLE_PRODUCTION_BOOTSTRAP_RECEIPT"
FUNDAMENTAL_SOURCE_BLOCKERS: Final = (
    "FUNDAMENTAL_HISTORY_MIXED",
    "FUNDAMENTAL_HISTORY_NOT_HOMOGENEOUS",
    "FUNDAMENTAL_LEGACY_DIRECT_READER_PROVENANCE_LIMITED",
)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical identifier")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    return value


def _fundamental_machine_states(value: Any) -> dict[str, Any]:
    expected = {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }
    if type(value) is not dict or value != expected:
        raise SystemContractError("Fundamental machine states are not exact safe-successor state")
    return dict(value)


def production_generation_intent_sha256(
    *,
    generation_state: str,
    contract_catalog_sha256: str,
    release_manifest_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    factor_source_object_refs: Sequence[Mapping[str, Any]],
    factor_policy_ref: Mapping[str, Any],
    factor_evidence_refs: Sequence[Mapping[str, Any]],
    factor_active_set_ref: Mapping[str, Any],
    factor_validation_attestation_ref: Mapping[str, Any],
    skill_tree_sha256: str,
    automation_semantic_sha256: str,
    readiness_matrix_ref: Mapping[str, Any],
    emergency_controller_sha256: str,
    generation_created_at: str,
    expected_assembly_id: str,
) -> str:
    """Hash the full acyclic manifest intent, substituting one fixed receipt role."""

    body = {
        "generation_state": generation_state,
        "contract_catalog_sha256": contract_catalog_sha256,
        "release_manifest_ref": dict(release_manifest_ref),
        "source_refs": [dict(row) for row in source_refs],
        "factor_source_object_refs": [dict(row) for row in factor_source_object_refs],
        "factor_policy_ref": dict(factor_policy_ref),
        "factor_evidence_refs": [dict(row) for row in factor_evidence_refs],
        "factor_active_set_ref": dict(factor_active_set_ref),
        "factor_validation_attestation_ref": dict(factor_validation_attestation_ref),
        "mainline_ref": None,
        "research_role": PRODUCTION_RESEARCH_ROLE,
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": skill_tree_sha256,
        "automation_semantic_sha256": automation_semantic_sha256,
        "readiness_matrix_ref": dict(readiness_matrix_ref),
        "emergency_controller_sha256": emergency_controller_sha256,
        "created_at": generation_created_at,
        "assembly_id": expected_assembly_id,
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _refs(value: Any, *, label: str, minimum: int = 0) -> list[dict[str, str]]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} is not an exact reference list")
    refs = [validate_object_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    keys = [tuple(ref[field] for field in OBJECT_REF_SORT_FIELDS) for ref in refs]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} is not tuple-sorted and unique")
    return refs


def _input_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("input_source_rows must be nonempty")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != INPUT_SOURCE_ROW_FIELDS:
            raise SystemContractError("input source row fields are not exact")
        field = _identifier(row["field"], label=f"input_source_rows[{index}].field")
        ordinal = row["ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            raise SystemContractError("input source row ordinal is invalid")
        file_ref = row["input_file_ref"]
        if type(file_ref) is not dict or set(file_ref) != {"relative_path", "byte_sha256"}:
            raise SystemContractError("input source file ref fields are not exact")
        relative = file_ref["relative_path"]
        if type(relative) is not str or not relative:
            raise SystemContractError("input source relative path is invalid")
        normalized = {
            "field": field,
            "ordinal": ordinal,
            "input_file_ref": {
                "relative_path": relative,
                "byte_sha256": _sha(
                    file_ref["byte_sha256"],
                    label=f"input_source_rows[{index}].input_file_ref.byte_sha256",
                ),
            },
            "source_object_ref": validate_object_ref(
                row["source_object_ref"],
                label=f"input_source_rows[{index}].source_object_ref",
            ),
        }
        rows.append(normalized)
    keys = [(row["field"], row["ordinal"]) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError("input source rows are not sorted and unique")
    return rows


def _statistics(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2 or any(type(row) is not dict for row in value):
        raise SystemContractError("signal_statistics must contain exactly two objects")
    rows = [dict(row) for row in value]
    factor_ids = [row.get("factor_id") for row in rows]
    if any(type(value) is not str or not value for value in factor_ids) or len(
        set(factor_ids)
    ) != len(factor_ids):
        raise SystemContractError("signal_statistics factor identities are not exact")
    for index, row in enumerate(rows):
        if (
            type(row.get("finite_count")) is not int
            or row["finite_count"] <= 0
            or type(row.get("distinct_finite_count")) is not int
            or row["distinct_finite_count"] <= 1
        ):
            raise SystemContractError(
                f"signal_statistics[{index}] does not prove nonempty nonconstant output"
            )
    return rows


def validate_production_bootstrap_receipt(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate exact receipt structure and its deterministic self-identity."""

    try:
        receipt = validate_artifact(
            document,
            expected_kind=PRODUCTION_BOOTSTRAP_RECEIPT_KIND,
            expected_contract_sha256=PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("production bootstrap receipt contract failed") from exc
    payload = receipt["payload"]
    if set(payload) != PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS or payload["state"] != "VERIFIED":
        raise SystemContractError("production bootstrap receipt fields/state differ")
    _identifier(
        payload["production_bootstrap_receipt_id"],
        label="production_bootstrap_receipt_id",
    )
    validate_object_ref(
        payload["bootstrap_operator_request_ref"],
        label="bootstrap_operator_request_ref",
    )
    _identifier(payload["source_root_id"], label="source_root_id")
    validate_object_ref(payload["deployed_release_ref"], label="deployed_release_ref")
    validate_object_ref(
        payload["calendar_authority_policy_ref"],
        label="calendar_authority_policy_ref",
    )
    validate_object_ref(payload["calendar_compilation_ref"], label="calendar_compilation_ref")
    if payload["calendar_capability_ref"] is not None:
        validate_object_ref(
            payload["calendar_capability_ref"],
            label="calendar_capability_ref",
        )
    limitations = payload["calendar_source_limitations"]
    if (
        type(limitations) is not list
        or any(type(row) is not str or not row for row in limitations)
        or limitations != sorted(set(limitations))
    ):
        raise SystemContractError("calendar source limitations are not exact")
    degraded_limitations = [
        "BSE_CALENDAR_POLICY_PROJECTED_FROM_SSE_SZSE",
        "CALENDAR_AUTHORITY_DEGRADED",
    ]
    if not (
        (limitations == [] and payload["calendar_capability_ref"] is None)
        or (limitations == degraded_limitations and payload["calendar_capability_ref"] is not None)
    ):
        raise SystemContractError("calendar authority route/limitations differ")
    _sha(payload["release_code_manifest_sha256"], label="release_code_manifest_sha256")
    _timestamp(payload["generation_created_at"], label="generation_created_at")
    _sha(payload["expected_assembly_id"], label="expected_assembly_id")
    _sha(payload["generation_intent_sha256"], label="generation_intent_sha256")
    if payload["mainline_ref"] is not None:
        raise SystemContractError("production receipt mainline_ref must be null")
    _input_rows(payload["input_source_rows"])
    _refs(payload["source_refs"], label="source_refs", minimum=1)
    _refs(
        payload["factor_source_object_refs"],
        label="factor_source_object_refs",
        minimum=1,
    )
    validate_object_ref(payload["factor_policy_ref"], label="factor_policy_ref")
    _refs(payload["factor_evidence_refs"], label="factor_evidence_refs", minimum=1)
    validate_object_ref(payload["factor_active_set_ref"], label="factor_active_set_ref")
    validate_object_ref(
        payload["factor_validation_attestation_ref"],
        label="factor_validation_attestation_ref",
    )
    validate_object_ref(payload["readiness_matrix_ref"], label="readiness_matrix_ref")
    for field in (
        "emergency_controller_sha256",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "signal_statistics_sha256",
        "assembler_code_sha256",
    ):
        _sha(payload[field], label=field)
    blockers = payload["source_blockers"]
    if (
        type(blockers) is not list
        or any(type(row) is not str or not row for row in blockers)
        or blockers != sorted(FUNDAMENTAL_SOURCE_BLOCKERS)
    ):
        raise SystemContractError("production receipt source blockers are not machine-derived")
    _fundamental_machine_states(payload["fundamental_machine_states"])
    statistics = _statistics(payload["signal_statistics"])
    if (
        hashlib.sha256(canonical_json_bytes(statistics)).hexdigest()
        != payload["signal_statistics_sha256"]
    ):
        raise SystemContractError("production receipt signal statistics SHA differs")
    if payload["assembler_module_path"] != ASSEMBLER_MODULE_PATH:
        raise SystemContractError("production receipt assembler module path differs")
    identity_body = {
        key: payload[key]
        for key in sorted(PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS)
        if key != "production_bootstrap_receipt_id"
    }
    expected_id = (
        "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(identity_body)).hexdigest()
    )
    if payload["production_bootstrap_receipt_id"] != expected_id:
        raise SystemContractError("production bootstrap receipt identity does not replay")
    return receipt


def build_production_bootstrap_receipt(
    *,
    bootstrap_operator_request_ref: Mapping[str, Any],
    source_root_id: str,
    input_source_rows: Sequence[Mapping[str, Any]],
    deployed_release_ref: Mapping[str, Any],
    calendar_authority_policy_ref: Mapping[str, Any],
    calendar_compilation_ref: Mapping[str, Any],
    calendar_capability_ref: Mapping[str, Any] | None,
    calendar_source_limitations: Sequence[str],
    release_code_manifest_sha256: str,
    generation_created_at: str,
    expected_assembly_id: str,
    generation_intent_sha256: str,
    source_refs: Sequence[Mapping[str, Any]],
    factor_source_object_refs: Sequence[Mapping[str, Any]],
    factor_policy_ref: Mapping[str, Any],
    factor_evidence_refs: Sequence[Mapping[str, Any]],
    factor_active_set_ref: Mapping[str, Any],
    factor_validation_attestation_ref: Mapping[str, Any],
    readiness_matrix_ref: Mapping[str, Any],
    emergency_controller_sha256: str,
    skill_tree_sha256: str,
    automation_semantic_sha256: str,
    source_blockers: Sequence[str],
    fundamental_machine_states: Mapping[str, Any],
    signal_statistics: Sequence[Mapping[str, Any]],
    assembler_code_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal a receipt after all production-only validation has succeeded."""

    body = {
        "state": "VERIFIED",
        "bootstrap_operator_request_ref": dict(bootstrap_operator_request_ref),
        "source_root_id": source_root_id,
        "input_source_rows": [dict(row) for row in input_source_rows],
        "deployed_release_ref": dict(deployed_release_ref),
        "calendar_authority_policy_ref": dict(calendar_authority_policy_ref),
        "calendar_compilation_ref": dict(calendar_compilation_ref),
        "calendar_capability_ref": (
            None if calendar_capability_ref is None else dict(calendar_capability_ref)
        ),
        "calendar_source_limitations": list(calendar_source_limitations),
        "release_code_manifest_sha256": release_code_manifest_sha256,
        "generation_created_at": generation_created_at,
        "expected_assembly_id": expected_assembly_id,
        "generation_intent_sha256": generation_intent_sha256,
        "mainline_ref": None,
        "source_refs": [dict(row) for row in source_refs],
        "factor_source_object_refs": [dict(row) for row in factor_source_object_refs],
        "factor_policy_ref": dict(factor_policy_ref),
        "factor_evidence_refs": [dict(row) for row in factor_evidence_refs],
        "factor_active_set_ref": dict(factor_active_set_ref),
        "factor_validation_attestation_ref": dict(factor_validation_attestation_ref),
        "readiness_matrix_ref": dict(readiness_matrix_ref),
        "emergency_controller_sha256": emergency_controller_sha256,
        "skill_tree_sha256": skill_tree_sha256,
        "automation_semantic_sha256": automation_semantic_sha256,
        "source_blockers": list(source_blockers),
        "fundamental_machine_states": dict(fundamental_machine_states),
        "signal_statistics": [dict(row) for row in signal_statistics],
        "signal_statistics_sha256": hashlib.sha256(
            canonical_json_bytes([dict(row) for row in signal_statistics])
        ).hexdigest(),
        "assembler_module_path": ASSEMBLER_MODULE_PATH,
        "assembler_code_sha256": assembler_code_sha256,
    }
    identity = "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    artifact = seal_artifact(
        PRODUCTION_BOOTSTRAP_RECEIPT_KIND,
        {"production_bootstrap_receipt_id": identity, **body},
        created_at=created_at,
    )
    return validate_production_bootstrap_receipt(artifact)


__all__ = [
    "ASSEMBLER_MODULE_PATH",
    "FUNDAMENTAL_SOURCE_BLOCKERS",
    "INPUT_SOURCE_ROW_FIELDS",
    "PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256",
    "PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS",
    "PRODUCTION_BOOTSTRAP_RECEIPT_KIND",
    "PRODUCTION_RESEARCH_ROLE",
    "build_production_bootstrap_receipt",
    "production_generation_intent_sha256",
    "validate_production_bootstrap_receipt",
]
