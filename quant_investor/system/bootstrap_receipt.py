"""Detached proof that an operational generation used the production assembler.

The receipt deliberately binds generation inputs rather than a generation ID.
The generation manifest may therefore include the receipt without creating a
content-hash cycle.  Activation must still replay the bound inputs; this
artifact is not an authorization merely because it is well formed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        "input_source_rows",
        "deployed_release_ref",
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


def validate_production_bootstrap_receipt(
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
    validate_object_ref(payload["deployed_release_ref"], label="deployed_release_ref")
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
        or blockers != sorted(set(blockers))
    ):
        raise SystemContractError("production receipt source blockers are not canonical")
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
    input_source_rows: Sequence[Mapping[str, Any]],
    deployed_release_ref: Mapping[str, Any],
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
    signal_statistics: Sequence[Mapping[str, Any]],
    assembler_code_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal a receipt after all production-only validation has succeeded."""

    body = {
        "state": "VERIFIED",
        "bootstrap_operator_request_ref": dict(bootstrap_operator_request_ref),
        "input_source_rows": [dict(row) for row in input_source_rows],
        "deployed_release_ref": dict(deployed_release_ref),
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
    "INPUT_SOURCE_ROW_FIELDS",
    "PRODUCTION_BOOTSTRAP_RECEIPT_CONTRACT_SHA256",
    "PRODUCTION_BOOTSTRAP_RECEIPT_FIELDS",
    "PRODUCTION_BOOTSTRAP_RECEIPT_KIND",
    "build_production_bootstrap_receipt",
    "validate_production_bootstrap_receipt",
]
