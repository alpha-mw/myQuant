"""System-owned exact-byte authorization for the first unified activation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)
from quant_investor.migration.custody import artifact_exact_ref
from quant_investor.migration.errors import UnifiedCutoverError
from quant_investor.migration.migration import (
    build_permanent_marker_payload,
    validate_permanent_marker,
    validate_pre_cas_activation_target,
)

from .errors import SystemActivationAuthorizationError
from .storage import ACTIVE_POINTER_PATH, EMPTY_POINTER_SHA256, MIGRATION_MARKER_PATH
from .store import validate_object_ref

ACTIVATION_AUTHORIZATION_KIND: Final = "system.activation_authorization"
ACTIVATION_AUTHORIZATION_CONTRACT_SHA256: Final = get_contract(
    ACTIVATION_AUTHORIZATION_KIND
).contract_sha256
ACTIVATION_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "authorization_id",
        "state",
        "migration_receipt_ref",
        "target_generation_id",
        "target_generation_manifest_ref",
        "deployed_release_ref",
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
ACTIVATION_PREPARED_KIND: Final = "system.activation_prepared"
ACTIVATION_PREPARED_CONTRACT_SHA256: Final = get_contract(ACTIVATION_PREPARED_KIND).contract_sha256
ACTIVATION_PREPARED_FIELDS: Final = frozenset(
    {
        "transaction_id",
        "state",
        "activation_authorization_ref",
        "migration_receipt_ref",
        "target_active_pointer",
        "target_active_pointer_ref",
        "permanent_marker_ref",
        "expected_active_pointer_sha256",
        "prepared_at",
        "actor_uid",
    }
)
_POINTER_FIELDS: Final = frozenset(
    {
        "generation_id",
        "manifest_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
    }
)


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        raise SystemActivationAuthorizationError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemActivationAuthorizationError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemActivationAuthorizationError(f"{label} must be canonical UTC seconds")
    return parsed


def _identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-authorization-"
        + hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "myquant-system-activation-authorization",
                    "payload": dict(body),
                }
            )
        ).hexdigest()
    )


def _prepared_identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-transaction-"
        + hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "myquant-system-activation-prepared",
                    "payload": dict(body),
                }
            )
        ).hexdigest()
    )


def _pointer_ref(pointer: Mapping[str, Any]) -> dict[str, str]:
    raw = canonical_json_bytes(dict(pointer))
    return {
        "generation_id": pointer["generation_id"],
        "manifest_sha256": pointer["manifest_sha256"],
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_activation_authorization(
    *,
    migration_receipt: Mapping[str, Any] | bytes,
    target_active_pointer: Mapping[str, Any] | bytes,
    target_generation_manifest: Mapping[str, Any] | bytes,
    deployed_release_ref: Mapping[str, Any],
    prepared_at: str,
    actor_uid: int,
) -> dict[str, Any]:
    """Seal authorization for exact prebuilt pointer and deterministic marker bytes."""

    try:
        receipt = validate_pre_cas_activation_target(
            migration_receipt, target_active_pointer, target_generation_manifest
        )
        manifest = validate_artifact(
            target_generation_manifest, expected_kind="system.generation_manifest"
        )
    except (ContractError, UnifiedCutoverError) as exc:
        raise SystemActivationAuthorizationError("detached receipt target is invalid") from exc
    if isinstance(target_active_pointer, bytes):
        pointer = parse_canonical_json_bytes(
            target_active_pointer,
            label="target active pointer",
        )
    else:
        pointer = dict(target_active_pointer)
    if type(pointer) is not dict or set(pointer) != _POINTER_FIELDS:
        raise SystemActivationAuthorizationError("target pointer fields are not exact")
    normalized_release = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
    if normalized_release != manifest["payload"].get("release_manifest_ref"):
        raise SystemActivationAuthorizationError("deployed release binding mismatch")
    if type(actor_uid) is not int or actor_uid < 0:
        raise SystemActivationAuthorizationError("actor_uid is invalid")
    prepared = _timestamp(prepared_at, label="prepared_at")
    activated = _timestamp(pointer["activated_at"], label="activated_at")
    if activated < prepared:
        raise SystemActivationAuthorizationError("activated_at precedes prepared_at")
    expected_actor = f"uid:{actor_uid}"
    if pointer["os_actor"] != expected_actor:
        raise SystemActivationAuthorizationError("pointer actor does not bind actor_uid")
    marker = build_permanent_marker_payload(
        receipt,
        pointer,
        manifest,
        completed_at=pointer["activated_at"],
    )
    body = {
        "state": "AUTHORIZED",
        "migration_receipt_ref": artifact_exact_ref(receipt),
        "target_generation_id": manifest["semantic_sha256"],
        "target_generation_manifest_ref": artifact_exact_ref(manifest),
        "deployed_release_ref": normalized_release,
        "target_active_pointer": dict(pointer),
        "target_active_pointer_ref": _pointer_ref(pointer),
        "target_active_pointer_path": str(ACTIVE_POINTER_PATH),
        "permanent_marker_ref": artifact_exact_ref(marker),
        "permanent_marker_path": str(MIGRATION_MARKER_PATH),
        "expected_active_pointer_sha256": EMPTY_POINTER_SHA256,
        "prepared_at": prepared_at,
        "activated_at": pointer["activated_at"],
        "actor_uid": actor_uid,
        "os_actor": pointer["os_actor"],
    }
    return seal_artifact(
        ACTIVATION_AUTHORIZATION_KIND,
        {**body, "authorization_id": _identity(body)},
        created_at=prepared_at,
        contract_sha256=ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
    )


def validate_activation_authorization(  # noqa: C901
    authorization: Mapping[str, Any] | bytes,
    *,
    migration_receipt: Mapping[str, Any] | bytes,
    target_active_pointer: Mapping[str, Any] | bytes,
    target_generation_manifest: Mapping[str, Any] | bytes,
    deployed_release_ref: Mapping[str, Any],
    current_uid: int,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deep-validate all exact bytes and return authorization plus marker."""

    try:
        document = validate_artifact(
            authorization,
            expected_kind=ACTIVATION_AUTHORIZATION_KIND,
            expected_contract_sha256=ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemActivationAuthorizationError("authorization artifact is invalid") from exc
    payload = document["payload"]
    if set(payload) != ACTIVATION_AUTHORIZATION_FIELDS:
        raise SystemActivationAuthorizationError("authorization fields are not exact")
    body = dict(payload)
    identity = body.pop("authorization_id")
    if identity != _identity(body) or payload["state"] != "AUTHORIZED":
        raise SystemActivationAuthorizationError("authorization identity is invalid")
    try:
        receipt = validate_pre_cas_activation_target(
            migration_receipt, target_active_pointer, target_generation_manifest
        )
        manifest = validate_artifact(
            target_generation_manifest, expected_kind="system.generation_manifest"
        )
    except (ContractError, UnifiedCutoverError) as exc:
        raise SystemActivationAuthorizationError("authorization target is invalid") from exc
    pointer_raw = (
        target_active_pointer
        if isinstance(target_active_pointer, bytes)
        else canonical_json_bytes(dict(target_active_pointer))
    )
    try:
        pointer = parse_canonical_json_bytes(pointer_raw, label="target active pointer")
    except ContractError as exc:
        raise SystemActivationAuthorizationError("target pointer is not canonical") from exc
    if type(pointer) is not dict or set(pointer) != _POINTER_FIELDS:
        raise SystemActivationAuthorizationError("target pointer fields are not exact")
    normalized_release = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
    prepared = _timestamp(payload["prepared_at"], label="prepared_at")
    activated = _timestamp(payload["activated_at"], label="activated_at")
    observed_now = now or datetime.now(timezone.utc)
    if activated < prepared or activated > observed_now:
        raise SystemActivationAuthorizationError("authorization time bounds are invalid")
    if payload["actor_uid"] != current_uid or pointer["os_actor"] != f"uid:{current_uid}":
        raise SystemActivationAuthorizationError("activation UID changed after prepare")
    marker = build_permanent_marker_payload(
        receipt,
        pointer,
        manifest,
        completed_at=payload["activated_at"],
    )
    expected = {
        "migration_receipt_ref": artifact_exact_ref(receipt),
        "target_generation_id": manifest["semantic_sha256"],
        "target_generation_manifest_ref": artifact_exact_ref(manifest),
        "deployed_release_ref": normalized_release,
        "target_active_pointer": pointer,
        "target_active_pointer_ref": _pointer_ref(pointer),
        "target_active_pointer_path": str(ACTIVE_POINTER_PATH),
        "permanent_marker_ref": artifact_exact_ref(marker),
        "permanent_marker_path": str(MIGRATION_MARKER_PATH),
        "expected_active_pointer_sha256": EMPTY_POINTER_SHA256,
        "activated_at": pointer["activated_at"],
        "os_actor": pointer["os_actor"],
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise SystemActivationAuthorizationError("authorization exact binding mismatch")
    try:
        validate_permanent_marker(marker)
    except UnifiedCutoverError as exc:
        raise SystemActivationAuthorizationError("permanent marker is invalid") from exc
    return document, marker


def build_prepared_activation_transaction(
    authorization: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Seal the exact pointer/marker transaction before any pointer CAS."""

    try:
        document = validate_artifact(
            authorization,
            expected_kind=ACTIVATION_AUTHORIZATION_KIND,
            expected_contract_sha256=ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemActivationAuthorizationError("authorization artifact is invalid") from exc
    payload = document["payload"]
    if set(payload) != ACTIVATION_AUTHORIZATION_FIELDS or payload["state"] != "AUTHORIZED":
        raise SystemActivationAuthorizationError("authorization is not prepared-ready")
    body = {
        "state": "PREPARED",
        "activation_authorization_ref": artifact_exact_ref(document),
        "migration_receipt_ref": payload["migration_receipt_ref"],
        "target_active_pointer": payload["target_active_pointer"],
        "target_active_pointer_ref": payload["target_active_pointer_ref"],
        "permanent_marker_ref": payload["permanent_marker_ref"],
        "expected_active_pointer_sha256": payload["expected_active_pointer_sha256"],
        "prepared_at": payload["prepared_at"],
        "actor_uid": payload["actor_uid"],
    }
    return seal_artifact(
        ACTIVATION_PREPARED_KIND,
        {**body, "transaction_id": _prepared_identity(body)},
        created_at=payload["prepared_at"],
        contract_sha256=ACTIVATION_PREPARED_CONTRACT_SHA256,
    )


def validate_prepared_activation_transaction(
    prepared: Mapping[str, Any] | bytes,
    *,
    authorization: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        document = validate_artifact(
            prepared,
            expected_kind=ACTIVATION_PREPARED_KIND,
            expected_contract_sha256=ACTIVATION_PREPARED_CONTRACT_SHA256,
        )
        authorization_document = validate_artifact(
            authorization,
            expected_kind=ACTIVATION_AUTHORIZATION_KIND,
            expected_contract_sha256=ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemActivationAuthorizationError("prepared transaction is invalid") from exc
    payload = document["payload"]
    if set(payload) != ACTIVATION_PREPARED_FIELDS:
        raise SystemActivationAuthorizationError("prepared transaction fields are not exact")
    body = dict(payload)
    identity = body.pop("transaction_id")
    authorization_payload = authorization_document["payload"]
    expected = {
        "state": "PREPARED",
        "activation_authorization_ref": artifact_exact_ref(authorization_document),
        "migration_receipt_ref": authorization_payload["migration_receipt_ref"],
        "target_active_pointer": authorization_payload["target_active_pointer"],
        "target_active_pointer_ref": authorization_payload["target_active_pointer_ref"],
        "permanent_marker_ref": authorization_payload["permanent_marker_ref"],
        "expected_active_pointer_sha256": authorization_payload["expected_active_pointer_sha256"],
        "prepared_at": authorization_payload["prepared_at"],
        "actor_uid": authorization_payload["actor_uid"],
    }
    if identity != _prepared_identity(expected) or body != expected:
        raise SystemActivationAuthorizationError("prepared transaction binding mismatch")
    return document


__all__ = [
    "ACTIVATION_AUTHORIZATION_CONTRACT_SHA256",
    "ACTIVATION_AUTHORIZATION_FIELDS",
    "ACTIVATION_AUTHORIZATION_KIND",
    "ACTIVATION_PREPARED_CONTRACT_SHA256",
    "ACTIVATION_PREPARED_FIELDS",
    "ACTIVATION_PREPARED_KIND",
    "build_activation_authorization",
    "build_prepared_activation_transaction",
    "validate_activation_authorization",
    "validate_prepared_activation_transaction",
]
