"""Frozen, non-executing schema dispatch for the initial unified activation.

The permanent marker must remain readable after descendant releases change the
compiled contract catalog.  These validators authenticate the exact first-
activation schemas and deterministic identities without executing old code or
consulting the descendant registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import re
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes

from .errors import SystemContractError, SystemPreconditionError

_ENVELOPE_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "created_at", "payload", "semantic_sha256"}
)
_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE: Final = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")

INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256: Final = (
    "8242ab01dbe9bd3b939d388e198c77edeee4d3f0b74eba7d392e70d7343a48a0"
)
INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256: Final = (
    "621c315342a2906134a6a4e36185aa8fbac0669f2d6eb3533b1c2dd04baf7363"
)
INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256: Final = (
    "ad0949a37faaa4d6abadb465fe6b5d146fb3d4347df0919b731ac9fe870ab0e5"
)
INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256: Final = (
    "48d4ab79ab58a76444aa7df431943f8ac56e62d6d573bda27c760940105277d8"
)
INITIAL_PERMANENT_MARKER_CONTRACT_SHA256: Final = (
    "7ddfeaf9b7e675a25a2c1486f9607e5b68a13519a34ec52f60c9948695fc2b40"
)
INITIAL_MIGRATION_RECEIPT_CONTRACT_SHA256: Final = (
    "867c9113f66780d777b7474c8ed013d2fedbdd6e324fe62c2fd44e050470838e"
)

INITIAL_PRODUCTION_RECEIPT_FIELDS: Final = frozenset(
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
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
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
INITIAL_FINAL_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "final_authorization_id",
        "state",
        "accepted_baseline_commit",
        "historical_integration_commit",
        "historical_dirty_evidence_ref",
        "concurrent_task_handoff_ref",
        "main_checkout_adoption_ref",
        "legacy_disposition_ref",
        "deployed_release_ref",
        "production_generation_manifest_ref",
        "production_bootstrap_receipt_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
        "calendar_source_limitations",
        "calendar_policy_authorized",
        "release_commit",
        "release_tree",
        "final_integration_commit",
        "final_integration_tree",
        "ancestry_rows",
        "excluded_commit_rows",
        "final_worktree_inventory_sha256",
        "clean_checkout_readback_rows",
        "user_authorization_basis",
        "preflight_rows",
        "final_build_authorized",
        "cas_authorized",
    }
)
INITIAL_ACTIVATION_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "authorization_id",
        "state",
        "final_cutover_authorization_ref",
        "migration_receipt_ref",
        "target_generation_id",
        "target_generation_manifest_ref",
        "deployed_release_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
        "calendar_source_limitations",
        "target_active_pointer",
        "target_active_pointer_ref",
        "target_active_pointer_path",
        "permanent_marker_ref",
        "permanent_marker_path",
        "expected_active_pointer_sha256",
        "prepared_at",
        "activated_at",
        "actor_uid",
        "os_actor",
    }
)
INITIAL_ACTIVATION_PREPARED_FIELDS: Final = frozenset(
    {
        "transaction_id",
        "state",
        "activation_authorization_ref",
        "final_cutover_authorization_ref",
        "migration_receipt_ref",
        "target_active_pointer",
        "target_active_pointer_ref",
        "permanent_marker_ref",
        "expected_active_pointer_sha256",
        "prepared_at",
        "actor_uid",
    }
)
INITIAL_PERMANENT_MARKER_FIELDS: Final = frozenset(
    {
        "marker_id",
        "status",
        "cutover_id",
        "migration_receipt_ref",
        "inventory_ref",
        "archive_plan_ref",
        "active_pointer_ref",
        "generation_manifest_ref",
        "generation_id",
        "permanent_marker_path",
        "migration_replay_refused",
        "legacy_replay_refused",
        "blocker_codes",
    }
)
INITIAL_MIGRATION_RECEIPT_FIELDS: Final = frozenset(
    {
        "migration_receipt_id",
        "status",
        "cutover_id",
        "inventory_ref",
        "archive_plan_ref",
        "rules_ref",
        "source_to_target_rules_ref",
        "source_to_target",
        "target_generation_id",
        "target_generation_manifest_path",
        "target_generation_manifest_ref",
        "target_release_manifest_ref",
        "target_active_pointer_path",
        "target_active_pointer_ref",
        "expected_active_pointer_sha256",
        "permanent_marker_path",
        "write_performed",
        "cas_performed",
        "blocker_codes",
        "summary",
    }
)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC")
    return value


def validate_frozen_object_ref(
    value: Any,
    *,
    label: str,
    dispatch: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise SystemContractError(f"{label} fields are not exact")
    kind = value.get("kind")
    contract_sha = value.get("contract_sha256")
    artifact_id = value.get("artifact_id")
    if (
        type(kind) is not str
        or _KIND_RE.fullmatch(kind) is None
        or type(artifact_id) is not str
        or not artifact_id
    ):
        raise SystemContractError(f"{label} identity is invalid")
    normalized = {
        "kind": kind,
        "contract_sha256": _sha(contract_sha, label=f"{label}.contract_sha256"),
        "artifact_id": artifact_id,
        "semantic_sha256": _sha(value.get("semantic_sha256"), label=f"{label}.semantic"),
        "byte_sha256": _sha(value.get("byte_sha256"), label=f"{label}.bytes"),
    }
    if dispatch is not None and dispatch.get((kind, normalized["contract_sha256"])) is None:
        raise SystemContractError(f"{label} contract pair is not initial-catalog anchored")
    return normalized


def frozen_object_ref(artifact: Mapping[str, Any]) -> dict[str, str]:
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": hashlib.sha256(canonical_json_bytes(dict(artifact))).hexdigest(),
    }


def _artifact(
    document: Mapping[str, Any] | bytes,
    *,
    kind: str,
    contract_sha256: str,
    identity_field: str,
    payload_fields: frozenset[str],
) -> dict[str, Any]:
    if type(document) is bytes:
        value = parse_canonical_json_bytes(document, label=f"historical {kind}")
    elif type(document) is dict:
        canonical_json_bytes(document)
        value = dict(document)
    else:
        raise SystemContractError(f"historical {kind} is not an artifact")
    if (
        type(value) is not dict
        or set(value) != _ENVELOPE_FIELDS
        or value.get("kind") != kind
        or value.get("contract_sha256") != contract_sha256
        or type(value.get("payload")) is not dict
        or set(value["payload"]) != payload_fields
    ):
        raise SystemContractError(f"historical {kind} schema differs")
    payload = value["payload"]
    identity = payload.get(identity_field)
    if type(identity) is not str or not identity or value.get("artifact_id") != identity:
        raise SystemContractError(f"historical {kind} identity differs")
    created_at = _timestamp(value.get("created_at"), label=f"historical {kind}.created_at")
    semantic = _sha(value.get("semantic_sha256"), label=f"historical {kind}.semantic")
    preimage = {
        "domain": "myquant-artifact",
        "kind": kind,
        "contract_sha256": contract_sha256,
        "identity_field": identity_field,
        "artifact_id": identity,
        "created_at": created_at,
        "payload": payload,
    }
    if semantic != hashlib.sha256(canonical_json_bytes(preimage)).hexdigest():
        raise SystemContractError(f"historical {kind} semantic SHA differs")
    return value


def validate_initial_production_receipt(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.production_bootstrap_receipt",
        contract_sha256=INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256,
        identity_field="production_bootstrap_receipt_id",
        payload_fields=INITIAL_PRODUCTION_RECEIPT_FIELDS,
    )
    payload = artifact["payload"]
    body = {
        key: payload[key] for key in sorted(payload) if key != "production_bootstrap_receipt_id"
    }
    expected = "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload["state"] != "VERIFIED" or payload["production_bootstrap_receipt_id"] != expected:
        raise SystemPreconditionError("historical production receipt is not VERIFIED")
    return artifact


def validate_initial_final_authorization(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.final_cutover_authorization",
        contract_sha256=INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256,
        identity_field="final_authorization_id",
        payload_fields=INITIAL_FINAL_AUTHORIZATION_FIELDS,
    )
    payload = artifact["payload"]
    if (
        payload["state"] != "AUTHORIZED"
        or payload["calendar_policy_authorized"] is not True
        or payload["final_build_authorized"] is not True
        or payload["cas_authorized"] is not True
    ):
        raise SystemPreconditionError("historical final authorization is not authorized")
    return artifact


def _activation_identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-authorization-"
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-system-activation-authorization", "payload": dict(body)}
            )
        ).hexdigest()
    )


def _prepared_identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-transaction-"
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-system-activation-prepared", "payload": dict(body)}
            )
        ).hexdigest()
    )


def _migration_identity(kind: str, body: Mapping[str, Any], *, prefix: str) -> str:
    return (
        prefix
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-migration-identity", "kind": kind, "payload": dict(body)}
            )
        ).hexdigest()
    )


def validate_initial_activation_authorization(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.activation_authorization",
        contract_sha256=INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
        identity_field="authorization_id",
        payload_fields=INITIAL_ACTIVATION_AUTHORIZATION_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("authorization_id")
    if body["state"] != "AUTHORIZED" or identity != _activation_identity(body):
        raise SystemPreconditionError("historical activation authorization is invalid")
    return artifact


def validate_initial_activation_prepared(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.activation_prepared",
        contract_sha256=INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256,
        identity_field="transaction_id",
        payload_fields=INITIAL_ACTIVATION_PREPARED_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("transaction_id")
    if body["state"] != "PREPARED" or identity != _prepared_identity(body):
        raise SystemPreconditionError("historical prepared transaction is invalid")
    return artifact


def validate_initial_permanent_marker(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.migration.complete",
        contract_sha256=INITIAL_PERMANENT_MARKER_CONTRACT_SHA256,
        identity_field="marker_id",
        payload_fields=INITIAL_PERMANENT_MARKER_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("marker_id")
    if (
        body["status"] != "COMPLETE"
        or body["blocker_codes"] != []
        or body["migration_replay_refused"] is not True
        or body["legacy_replay_refused"] is not True
        or identity
        != _migration_identity("system.migration.complete", body, prefix="migration-marker-")
    ):
        raise SystemPreconditionError("historical permanent marker is invalid")
    return artifact


def validate_initial_migration_receipt(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.migration.receipt",
        contract_sha256=INITIAL_MIGRATION_RECEIPT_CONTRACT_SHA256,
        identity_field="migration_receipt_id",
        payload_fields=INITIAL_MIGRATION_RECEIPT_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("migration_receipt_id")
    if (
        body["status"] != "READY_FOR_CAS"
        or body["expected_active_pointer_sha256"] != "EMPTY"
        or body["write_performed"] is not False
        or body["cas_performed"] is not False
        or body["blocker_codes"] != []
        or identity
        != _migration_identity("system.migration.receipt", body, prefix="migration-receipt-")
    ):
        raise SystemPreconditionError("historical migration receipt is invalid")
    return artifact


def validate_initial_activation_bundle(
    *,
    final_authorization: Mapping[str, Any] | bytes,
    activation_authorization: Mapping[str, Any] | bytes,
    prepared_transaction: Mapping[str, Any] | bytes,
    migration_receipt: Mapping[str, Any] | bytes,
    permanent_marker: Mapping[str, Any] | bytes,
    active_pointer: Mapping[str, Any],
    generation_manifest: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    current_uid: int,
) -> dict[str, dict[str, Any]]:
    """Cross-bind every immutable first-activation artifact without current schemas."""

    final = validate_initial_final_authorization(final_authorization)
    authorization = validate_initial_activation_authorization(activation_authorization)
    prepared = validate_initial_activation_prepared(prepared_transaction)
    receipt = validate_initial_migration_receipt(migration_receipt)
    marker = validate_initial_permanent_marker(permanent_marker)
    pointer = dict(active_pointer)
    pointer_fields = {
        "generation_id",
        "manifest_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
    }
    if type(active_pointer) is not dict or set(pointer) != pointer_fields:
        raise SystemContractError("historical active pointer fields differ")
    pointer_raw = canonical_json_bytes(pointer)
    pointer_ref = {
        "generation_id": _sha(pointer["generation_id"], label="historical generation id"),
        "manifest_sha256": _sha(pointer["manifest_sha256"], label="historical manifest byte SHA"),
        "byte_sha256": hashlib.sha256(pointer_raw).hexdigest(),
    }
    manifest_ref = frozen_object_ref(generation_manifest)
    release_ref = validate_frozen_object_ref(
        deployed_release_ref,
        label="historical deployed release ref",
    )
    receipt_ref = frozen_object_ref(receipt)
    final_ref = frozen_object_ref(final)
    authorization_ref = frozen_object_ref(authorization)
    marker_ref = frozen_object_ref(marker)
    receipt_payload = receipt["payload"]
    if (
        receipt_payload["target_generation_id"] != pointer["generation_id"]
        or receipt_payload["target_generation_manifest_ref"] != manifest_ref
        or receipt_payload["target_release_manifest_ref"] != release_ref
        or receipt_payload["target_active_pointer_ref"] != pointer_ref
        or receipt_payload["target_active_pointer_path"] != "results/system/_active.json"
        or receipt_payload["permanent_marker_path"] != "results/system/_migration_complete.json"
    ):
        raise SystemPreconditionError("historical migration target binding differs")
    final_payload = final["payload"]
    auth_payload = authorization["payload"]
    expected_auth = {
        "final_cutover_authorization_ref": final_ref,
        "migration_receipt_ref": receipt_ref,
        "target_generation_id": pointer["generation_id"],
        "target_generation_manifest_ref": manifest_ref,
        "deployed_release_ref": release_ref,
        "calendar_authority_policy_ref": final_payload["calendar_authority_policy_ref"],
        "calendar_compilation_ref": final_payload["calendar_compilation_ref"],
        "calendar_capability_ref": final_payload["calendar_capability_ref"],
        "calendar_capture_execution_ref": final_payload["calendar_capture_execution_ref"],
        "calendar_authorization_basis": final_payload["calendar_authorization_basis"],
        "calendar_source_limitations": final_payload["calendar_source_limitations"],
        "target_active_pointer": pointer,
        "target_active_pointer_ref": pointer_ref,
        "target_active_pointer_path": "results/system/_active.json",
        "permanent_marker_ref": marker_ref,
        "permanent_marker_path": "results/system/_migration_complete.json",
        "expected_active_pointer_sha256": "EMPTY",
        "activated_at": pointer["activated_at"],
        "actor_uid": current_uid,
        "os_actor": f"uid:{current_uid}",
    }
    if any(auth_payload[field] != value for field, value in expected_auth.items()):
        raise SystemPreconditionError("historical activation authorization binding differs")
    prepared_at = _timestamp(auth_payload["prepared_at"], label="historical prepared_at")
    activated_at = _timestamp(auth_payload["activated_at"], label="historical activated_at")
    if prepared_at > activated_at or pointer["os_actor"] != f"uid:{current_uid}":
        raise SystemPreconditionError("historical activation actor/time binding differs")

    marker_payload = marker["payload"]
    expected_marker = {
        "migration_receipt_ref": receipt_ref,
        "active_pointer_ref": pointer_ref,
        "generation_manifest_ref": manifest_ref,
        "generation_id": pointer["generation_id"],
        "inventory_ref": receipt_payload["inventory_ref"],
        "archive_plan_ref": receipt_payload["archive_plan_ref"],
        "cutover_id": receipt_payload["cutover_id"],
        "permanent_marker_path": receipt_payload["permanent_marker_path"],
    }
    if any(marker_payload[field] != value for field, value in expected_marker.items()):
        raise SystemPreconditionError("historical permanent marker binding differs")

    prepared_payload = prepared["payload"]
    expected_prepared = {
        "state": "PREPARED",
        "activation_authorization_ref": authorization_ref,
        "final_cutover_authorization_ref": final_ref,
        "migration_receipt_ref": receipt_ref,
        "target_active_pointer": pointer,
        "target_active_pointer_ref": pointer_ref,
        "permanent_marker_ref": marker_ref,
        "expected_active_pointer_sha256": "EMPTY",
        "prepared_at": auth_payload["prepared_at"],
        "actor_uid": current_uid,
    }
    prepared_body = dict(prepared_payload)
    prepared_body.pop("transaction_id")
    if prepared_body != expected_prepared:
        raise SystemPreconditionError("historical prepared transaction binding differs")
    return {
        "final_authorization": final,
        "activation_authorization": authorization,
        "prepared_transaction": prepared,
        "migration_receipt": receipt,
        "permanent_marker": marker,
    }


__all__ = [
    "INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256",
    "INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256",
    "INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256",
    "INITIAL_PERMANENT_MARKER_CONTRACT_SHA256",
    "INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256",
    "frozen_object_ref",
    "validate_frozen_object_ref",
    "validate_initial_activation_authorization",
    "validate_initial_activation_bundle",
    "validate_initial_activation_prepared",
    "validate_initial_final_authorization",
    "validate_initial_migration_receipt",
    "validate_initial_permanent_marker",
    "validate_initial_production_receipt",
]
