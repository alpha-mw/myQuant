"""Owner-policy Ed25519 action permits; verification only, never signing."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from typing import Any, Final

from .._core import (
    canonical_bytes,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)
from .contracts import PublicationContractError, publication_common

OWNER_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.publication-owner-policy.v1"
ACTION_PERMIT_VERSION: Final = "myquant.v17.intelligence-v2.publication-permit.v1"
PERMIT_ACTIONS: Final = frozenset(
    {
        "ACTIVATE",
        "AUTHORIZE_PREACTIVATION",
        "QUARANTINE",
        "REVOKE_KEY",
        "ROLLBACK",
        "ROTATE_KEY",
    }
)

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "production",
    "publication_profile",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
OWNER_POLICY_FIELDS: Final = _COMMON_FIELDS | {
    "keys",
    "maximum_permit_lifetime_seconds",
    "policy_id",
}
PERMIT_FIELDS: Final = _COMMON_FIELDS | {
    "artifact_id",
    "claims",
    "permit_id",
    "signature_algorithm",
    "signature_base64",
    "signer_key_id",
}

KEY_FIELDS: Final = {
    "actions",
    "algorithm",
    "key_id",
    "not_after",
    "not_before",
    "public_key_base64",
    "revoked_at",
}

CLAIM_FIELDS: Final = {
    "action",
    "canonical_strategy_id",
    "expected_pointer_sha256",
    "expires_at",
    "issued_at",
    "nonce",
    "not_before",
    "subject_ref",
    "target_pointer_sha256",
}


def _decode_base64(value: Any, *, label: str, expected_length: int) -> bytes:
    if type(value) is not str:
        raise PublicationContractError(f"{label} must be canonical Base64")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise PublicationContractError(f"{label} must be canonical Base64") from exc
    if len(raw) != expected_length or base64.b64encode(raw).decode("ascii") != value:
        raise PublicationContractError(f"{label} length or canonical form is invalid")
    return raw


def _actions(values: Sequence[Any], *, label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PublicationContractError(f"{label} must be a sequence")
    rows = [str(value) for value in values]
    if (
        not rows
        or not set(rows).issubset(PERMIT_ACTIONS)
        or rows != sorted(rows)
        or len(rows) != len(set(rows))
    ):
        raise PublicationContractError(f"{label} must be sorted unique permit actions")
    return rows


def _key_row(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    row = require_exact_keys(value, KEY_FIELDS, label=f"keys[{index}]")
    if row["algorithm"] != "ED25519":
        raise PublicationContractError("publication key algorithm must be ED25519")
    public_key = _decode_base64(
        row["public_key_base64"],
        label=f"keys[{index}].public_key_base64",
        expected_length=32,
    )
    key_id = sha256(row["key_id"], label=f"keys[{index}].key_id")
    if key_id != hashlib.sha256(public_key).hexdigest():
        raise PublicationContractError("publication key_id does not bind its key bytes")
    not_before = timestamp(row["not_before"], label=f"keys[{index}].not_before")
    not_after = timestamp(row["not_after"], label=f"keys[{index}].not_after")
    revoked = row["revoked_at"]
    if revoked is not None:
        revoked = timestamp(revoked, label=f"keys[{index}].revoked_at")
    if not_before >= not_after or (revoked is not None and revoked < not_before):
        raise PublicationContractError("publication key chronology is invalid")
    return {
        "actions": _actions(row["actions"], label=f"keys[{index}].actions"),
        "algorithm": "ED25519",
        "key_id": key_id,
        "not_after": not_after,
        "not_before": not_before,
        "public_key_base64": row["public_key_base64"],
        "revoked_at": revoked,
    }


def build_publication_owner_policy(
    *,
    created_at: str,
    maximum_permit_lifetime_seconds: int,
    keys: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    if (
        type(maximum_permit_lifetime_seconds) is not int
        or not 1 <= maximum_permit_lifetime_seconds <= 86_400
    ):
        raise PublicationContractError("maximum permit lifetime is invalid")
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence) or not keys:
        raise PublicationContractError("owner policy requires publication keys")
    rows = [_key_row(value, index=index) for index, value in enumerate(keys)]
    key_ids = [row["key_id"] for row in rows]
    if key_ids != sorted(key_ids) or len(key_ids) != len(set(key_ids)):
        raise PublicationContractError("owner policy keys must be key_id sorted and unique")
    return seal(
        {
            **publication_common(at=issued_at),
            "keys": rows,
            "maximum_permit_lifetime_seconds": maximum_permit_lifetime_seconds,
            "version": OWNER_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_publication_owner_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="policy_id")
    require_exact_keys(normalized, OWNER_POLICY_FIELDS, label="publication owner policy")
    expected = build_publication_owner_policy(
        created_at=normalized.get("timestamp"),
        maximum_permit_lifetime_seconds=normalized.get("maximum_permit_lifetime_seconds"),
        keys=normalized.get("keys"),
    )
    if normalized != expected or normalized["version"] != OWNER_POLICY_VERSION:
        raise PublicationContractError("publication owner policy replay mismatch")
    return normalized


def _utc_seconds(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def permit_message(claims: Mapping[str, Any]) -> bytes:
    row = require_exact_keys(claims, CLAIM_FIELDS, label="permit claims")
    action = row["action"]
    if action not in PERMIT_ACTIONS:
        raise PublicationContractError("permit action is invalid")
    domain = f"{ACTION_PERMIT_VERSION}:{action}\x00".encode("ascii")
    return domain + canonical_bytes(dict(row))


def _claims(
    *,
    action: str,
    canonical_strategy_id: str,
    subject_ref: Mapping[str, Any],
    expected_pointer_sha256: str,
    target_pointer_sha256: str,
    issued_at: str,
    not_before: str,
    expires_at: str,
    nonce: str,
) -> dict[str, Any]:
    if action not in PERMIT_ACTIONS:
        raise PublicationContractError("permit action is invalid")
    issued = timestamp(issued_at, label="issued_at")
    starts = timestamp(not_before, label="not_before")
    expires = timestamp(expires_at, label="expires_at")
    if not issued <= starts <= expires:
        raise PublicationContractError("permit chronology is invalid")
    expected = (
        "EMPTY"
        if expected_pointer_sha256 == "EMPTY"
        else sha256(expected_pointer_sha256, label="expected_pointer_sha256")
    )
    return {
        "action": action,
        "canonical_strategy_id": identifier(canonical_strategy_id, label="canonical_strategy_id"),
        "expected_pointer_sha256": expected,
        "expires_at": expires,
        "issued_at": issued,
        "nonce": sha256(nonce, label="nonce"),
        "not_before": starts,
        "subject_ref": validate_content_ref(subject_ref, label="subject_ref"),
        "target_pointer_sha256": sha256(
            target_pointer_sha256,
            label="target_pointer_sha256",
        ),
    }


def build_action_permit(
    *,
    action: str,
    canonical_strategy_id: str,
    subject_ref: Mapping[str, Any],
    expected_pointer_sha256: str,
    target_pointer_sha256: str,
    issued_at: str,
    not_before: str,
    expires_at: str,
    nonce: str,
    signer_key_id: str,
    signature_base64: str,
) -> dict[str, Any]:
    claims = _claims(
        action=action,
        canonical_strategy_id=canonical_strategy_id,
        subject_ref=subject_ref,
        expected_pointer_sha256=expected_pointer_sha256,
        target_pointer_sha256=target_pointer_sha256,
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        nonce=nonce,
    )
    _decode_base64(signature_base64, label="signature_base64", expected_length=64)
    permit_id = hashlib.sha256(permit_message(claims)).hexdigest()
    return seal(
        {
            **publication_common(at=claims["issued_at"]),
            "claims": claims,
            "permit_id": permit_id,
            "signature_algorithm": "ED25519",
            "signature_base64": signature_base64,
            "signer_key_id": sha256(signer_key_id, label="signer_key_id"),
            "version": ACTION_PERMIT_VERSION,
        },
        identity_field="artifact_id",
    )


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise PublicationContractError("ED25519_VERIFIER_UNAVAILABLE") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except InvalidSignature as exc:
        raise PublicationContractError("Ed25519 publication signature is invalid") from exc


def validate_action_permit(
    document: Mapping[str, Any],
    *,
    owner_policy: Mapping[str, Any],
    expected_action: str,
    expected_subject_ref: Mapping[str, Any],
    expected_strategy_id: str,
    expected_pointer_sha256: str,
    target_pointer_sha256: str,
    verified_at: str,
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="artifact_id")
    require_exact_keys(normalized, PERMIT_FIELDS, label="publication action permit")
    claims = require_exact_keys(normalized.get("claims"), CLAIM_FIELDS, label="permit claims")
    if (
        normalized["version"] != ACTION_PERMIT_VERSION
        or normalized["signature_algorithm"] != "ED25519"
        or normalized["timestamp"] != normalized["claims"]["issued_at"]
        or {
            key: normalized[key]
            for key in (
                "authority",
                "decision_protocol",
                "production",
                "publication_profile",
                "research_only",
            )
        }
        != {
            key: publication_common(at=normalized["timestamp"])[key]
            for key in (
                "authority",
                "decision_protocol",
                "production",
                "publication_profile",
                "research_only",
            )
        }
    ):
        raise PublicationContractError("publication action permit envelope is invalid")
    policy = validate_publication_owner_policy(owner_policy)
    expected_claims = _claims(
        action=expected_action,
        canonical_strategy_id=expected_strategy_id,
        subject_ref=expected_subject_ref,
        expected_pointer_sha256=expected_pointer_sha256,
        target_pointer_sha256=target_pointer_sha256,
        issued_at=claims["issued_at"],
        not_before=claims["not_before"],
        expires_at=claims["expires_at"],
        nonce=claims["nonce"],
    )
    # _claims hashes nonce; compare other closed claims directly, then bind the supplied nonce hash.
    expected_claims["nonce"] = claims["nonce"]
    if claims != expected_claims:
        raise PublicationContractError("permit claims do not match the requested action")
    checked_at = timestamp(verified_at, label="verified_at")
    if not claims["not_before"] <= checked_at <= claims["expires_at"]:
        raise PublicationContractError("publication permit is outside its validity window")
    lifetime = int(
        (_utc_seconds(claims["expires_at"]) - _utc_seconds(claims["not_before"])).total_seconds()
    )
    if lifetime > policy["maximum_permit_lifetime_seconds"]:
        raise PublicationContractError("publication permit lifetime exceeds owner policy")
    key_rows = [row for row in policy["keys"] if row["key_id"] == normalized.get("signer_key_id")]
    if len(key_rows) != 1:
        raise PublicationContractError("publication permit signer is not trusted")
    key = key_rows[0]
    if expected_action not in key["actions"]:
        raise PublicationContractError("publication key lacks the requested action scope")
    if not key["not_before"] <= claims["issued_at"] <= key["not_after"]:
        raise PublicationContractError("publication key is not valid at permit issuance")
    if key["revoked_at"] is not None and checked_at >= key["revoked_at"]:
        raise PublicationContractError("publication key is revoked")
    message = permit_message(claims)
    if normalized.get("permit_id") != hashlib.sha256(message).hexdigest():
        raise PublicationContractError("publication permit_id mismatch")
    public_key = _decode_base64(
        key["public_key_base64"], label="public_key_base64", expected_length=32
    )
    signature = _decode_base64(
        normalized.get("signature_base64"), label="signature_base64", expected_length=64
    )
    _verify_ed25519(public_key, signature, message)
    return normalized


__all__ = [
    "ACTION_PERMIT_VERSION",
    "OWNER_POLICY_VERSION",
    "PERMIT_ACTIONS",
    "build_action_permit",
    "build_publication_owner_policy",
    "permit_message",
    "validate_action_permit",
    "validate_publication_owner_policy",
]
