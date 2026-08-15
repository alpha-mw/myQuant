"""Canonical assembly-request decoding for CLI and offline callers."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    SYSTEM_ASSEMBLY_REQUEST_FIELDS,
    validate_artifact,
)

from .errors import SystemContractError
from .store import GENERATION_STATES, OBJECT_REF_SORT_FIELDS, validate_object_ref

ASSEMBLY_REQUEST_KIND: Final = "system.assembly_request"
ASSEMBLY_REQUEST_FIELDS: Final = SYSTEM_ASSEMBLY_REQUEST_FIELDS
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} must be lowercase SHA-256")
    return value


def _ref(value: Any, *, label: str, nullable: bool = False) -> dict[str, str] | None:
    if value is None and nullable:
        return None
    return validate_object_ref(value, label=label)


def _refs(value: Any, *, label: str) -> list[dict[str, str]]:
    if type(value) is not list:
        raise SystemContractError(f"{label} must be a list of exact object refs")
    rows = [validate_object_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    keys = [tuple(row[field] for field in OBJECT_REF_SORT_FIELDS) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} refs must be tuple-sorted and unique")
    return rows


def decode_assembly_request(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate one sealed request and return exact ``assemble_generation`` kwargs."""

    try:
        envelope = validate_artifact(document, expected_kind=ASSEMBLY_REQUEST_KIND)
    except ContractError as exc:
        raise SystemContractError("assembly request contract closure failed") from exc
    payload = envelope["payload"]
    if set(payload) != set(ASSEMBLY_REQUEST_FIELDS):
        raise SystemContractError("assembly request payload fields are not exact")
    generation_state = payload["generation_state"]
    if type(generation_state) is not str or generation_state not in GENERATION_STATES:
        raise SystemContractError("assembly request generation_state is invalid")
    if payload["migration_receipt_ref"] is not None:
        raise SystemContractError("migration_receipt_ref must be the mandatory null tombstone")
    if payload["migration_marker_ref"] is not None:
        raise SystemContractError("migration_marker_ref must be the mandatory null tombstone")
    return {
        "generation_state": generation_state,
        "release_manifest_ref": _ref(payload["release_manifest_ref"], label="release_manifest_ref"),
        "source_refs": _refs(payload["source_refs"], label="source_refs"),
        "factor_source_object_refs": _refs(
            payload["factor_source_object_refs"], label="factor_source_object_refs"
        ),
        "factor_policy_ref": _ref(
            payload["factor_policy_ref"],
            label="factor_policy_ref",
            nullable=True,
        ),
        "factor_evidence_refs": _refs(
            payload["factor_evidence_refs"], label="factor_evidence_refs"
        ),
        "factor_active_set_ref": _ref(
            payload["factor_active_set_ref"],
            label="factor_active_set_ref",
            nullable=True,
        ),
        "factor_validation_attestation_ref": _ref(
            payload["factor_validation_attestation_ref"],
            label="factor_validation_attestation_ref",
            nullable=True,
        ),
        "mainline_ref": _ref(payload["mainline_ref"], label="mainline_ref", nullable=True),
        "research_refs": _refs(payload["research_refs"], label="research_refs"),
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": _sha256(payload["skill_tree_sha256"], label="skill_tree_sha256"),
        "automation_semantic_sha256": _sha256(
            payload["automation_semantic_sha256"],
            label="automation_semantic_sha256",
        ),
        "readiness_matrix_ref": _ref(payload["readiness_matrix_ref"], label="readiness_matrix_ref"),
        "emergency_controller_sha256": _sha256(
            payload["emergency_controller_sha256"],
            label="emergency_controller_sha256",
            nullable=True,
        ),
        "created_at": envelope["created_at"],
    }


__all__ = [
    "ASSEMBLY_REQUEST_FIELDS",
    "ASSEMBLY_REQUEST_KIND",
    "decode_assembly_request",
]
