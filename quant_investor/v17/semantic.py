"""Canonical JSON and semantic sealing for the v17 shadow lane.

The semantic digest is deliberately independent from presentation whitespace:
it is SHA-256 over compact, sorted-key UTF-8 JSON after removing the
``semantic_sha256`` field itself.  All helpers fail closed on values that JSON
cannot represent deterministically (for example NaN or non-string keys).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

SEMANTIC_SHA_FIELD = "semantic_sha256"
_SHA256_CHARS = frozenset("0123456789abcdef")


class SemanticContractError(ValueError):
    """Raised when a v17 payload is not canonically representable or sealed."""


def require_sha256(value: Any, *, label: str) -> str:
    """Return *value* as a validated lowercase SHA-256 hex string."""

    if not isinstance(value, str):
        raise SemanticContractError(f"{label} must be lowercase SHA-256")
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise SemanticContractError(f"{label} must be lowercase SHA-256")
    return value


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SemanticContractError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SemanticContractError(f"non-string JSON key at {path}")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise SemanticContractError(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode compact, sorted-key UTF-8 JSON without a trailing newline."""

    _validate_json_value(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SemanticContractError("payload is not canonical JSON") from exc


def semantic_sha256(value: Mapping[str, Any]) -> str:
    """Hash a mapping after removing its own semantic digest field."""

    if not isinstance(value, Mapping):
        raise SemanticContractError("semantic payload must be a mapping")
    payload = dict(value)
    payload.pop(SEMANTIC_SHA_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and seal an unsealed mapping.

    Supplying a digest to the sealer is rejected so a caller cannot
    accidentally preserve a stale digest.
    """

    if not isinstance(value, Mapping):
        raise SemanticContractError("semantic payload must be a mapping")
    if SEMANTIC_SHA_FIELD in value:
        raise SemanticContractError("semantic_sha256 must not be supplied")
    payload = dict(value)
    payload[SEMANTIC_SHA_FIELD] = semantic_sha256(payload)
    canonical_json_bytes(payload)
    return payload


def validate_semantic_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a defensive copy of a semantically sealed mapping."""

    if not isinstance(value, Mapping):
        raise SemanticContractError("semantic payload must be a mapping")
    payload = dict(value)
    declared = require_sha256(payload.get(SEMANTIC_SHA_FIELD), label=SEMANTIC_SHA_FIELD)
    expected = semantic_sha256(payload)
    if declared != expected:
        raise SemanticContractError("semantic_sha256 mismatch")
    canonical_json_bytes(payload)
    return payload


__all__ = [
    "SEMANTIC_SHA_FIELD",
    "SemanticContractError",
    "canonical_json_bytes",
    "require_sha256",
    "seal_semantic",
    "semantic_sha256",
    "validate_semantic_seal",
]
