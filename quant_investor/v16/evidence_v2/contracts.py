"""Canonical, detached-reference contracts for the v16 evidence-v2 lane.

This module is deliberately nonauthorizing.  It contains no discovery,
network, provider, registry, portfolio, or execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import posixpath
import unicodedata
from collections.abc import Mapping
from typing import Any

ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_ITEMS = 100_000
MAX_JSON_STRING_BYTES = 1024 * 1024
MAX_SAFE_INTEGER = (2**53) - 1
SEMANTIC_SHA_FIELD = "semantic_sha256"
EVIDENCE_REF_SCHEMA = "v16.evidence-ref.v2"
NONAUTHORIZING_PROJECTION_SCHEMA = "v16.evidence-contract-projection.v2"

_SHA256_CHARS = frozenset("0123456789abcdef")


class EvidenceV2Error(ValueError):
    """Raised when evidence-v2 input fails closed."""


def require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in _SHA256_CHARS for character in text)
    ):
        raise EvidenceV2Error(f"{label} must be lowercase SHA-256")
    return text


def encode_f64(value: Any) -> str:
    """Encode one finite binary64 value without JSON-number ambiguity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceV2Error("binary64 value must be numeric and not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceV2Error("binary64 value must be finite")
    if number == 0.0:
        number = 0.0
    return "f64:" + number.hex()


def decode_f64(value: Any, *, label: str = "binary64") -> float:
    if not isinstance(value, str) or not value.startswith("f64:"):
        raise EvidenceV2Error(f"{label} must use canonical f64 encoding")
    token = value[4:]
    try:
        number = float.fromhex(token)
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} is not valid binary64 hex") from exc
    if not math.isfinite(number) or encode_f64(number) != value:
        raise EvidenceV2Error(f"{label} is not canonical finite binary64")
    return number


def _validate_string(value: str, *, label: str) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EvidenceV2Error(f"{label} is not valid UTF-8") from exc
    if len(encoded) > MAX_JSON_STRING_BYTES:
        raise EvidenceV2Error(f"{label} exceeds the string byte limit")
    if unicodedata.normalize("NFC", value) != value:
        raise EvidenceV2Error(f"{label} must be Unicode NFC")


def validate_canonical_value(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    _item_counter: list[int] | None = None,
) -> None:
    """Validate the restricted JSON value domain used by evidence-v2."""

    item_counter = _item_counter if _item_counter is not None else [0]
    if depth > MAX_JSON_DEPTH:
        raise EvidenceV2Error(f"{path} exceeds maximum JSON depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if isinstance(value, bool) or abs(value) > MAX_SAFE_INTEGER:
            raise EvidenceV2Error(f"{path} integer exceeds the safe range")
        return
    if isinstance(value, float):
        raise EvidenceV2Error(f"{path} contains a native JSON float")
    if isinstance(value, str):
        _validate_string(value, label=path)
        return
    if isinstance(value, list):
        item_counter[0] += len(value)
        if item_counter[0] > MAX_JSON_ITEMS:
            raise EvidenceV2Error(f"{path} exceeds the aggregate item limit")
        for index, item in enumerate(value):
            validate_canonical_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                _item_counter=item_counter,
            )
        return
    if isinstance(value, dict):
        item_counter[0] += len(value)
        if item_counter[0] > MAX_JSON_ITEMS:
            raise EvidenceV2Error(f"{path} exceeds the aggregate item limit")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EvidenceV2Error(f"{path} object keys must be nonempty strings")
            try:
                key_bytes = key.encode("ascii")
            except UnicodeEncodeError as exc:
                raise EvidenceV2Error(f"{path} object keys must be ASCII") from exc
            if len(key_bytes) > 128:
                raise EvidenceV2Error(f"{path} object key exceeds 128 bytes")
            validate_canonical_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                _item_counter=item_counter,
            )
        return
    raise EvidenceV2Error(f"{path} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    validate_canonical_value(value)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > ARTIFACT_MAX_BYTES:
        raise EvidenceV2Error("canonical JSON exceeds the artifact byte limit")
    return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop(SEMANTIC_SHA_FIELD, None)
    return sha256_bytes(canonical_json_bytes(payload))


def seal_semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if SEMANTIC_SHA_FIELD in payload:
        raise EvidenceV2Error("semantic_sha256 must not be supplied to the sealer")
    payload[SEMANTIC_SHA_FIELD] = semantic_sha256(payload)
    canonical_json_bytes(payload)
    return payload


def validate_semantic_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    declared = require_sha256(payload.get(SEMANTIC_SHA_FIELD), label=SEMANTIC_SHA_FIELD)
    expected = semantic_sha256(payload)
    if declared != expected:
        raise EvidenceV2Error("semantic_sha256 mismatch")
    canonical_json_bytes(payload)
    return payload


def _reject_constant(value: str) -> None:
    raise EvidenceV2Error(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceV2Error(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def parse_canonical_json_bytes(
    payload: bytes,
    *,
    max_bytes: int = ARTIFACT_MAX_BYTES,
) -> Any:
    if not payload or len(payload) > max_bytes:
        raise EvidenceV2Error("canonical JSON byte length is invalid")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise EvidenceV2Error("canonical JSON must end with exactly one LF")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceV2Error("canonical JSON must be UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=lambda _value: (_ for _ in ()).throw(
                EvidenceV2Error("native JSON floats are forbidden")
            ),
        )
    except EvidenceV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceV2Error(f"invalid strict JSON: {exc}") from exc
    validate_canonical_value(value)
    if canonical_json_bytes(value) != payload:
        raise EvidenceV2Error("JSON bytes are not canonical evidence-v2 bytes")
    return value


@dataclass(frozen=True)
class EvidenceRef:
    schema_version: str
    artifact_schema: str
    absolute_path: str
    byte_sha256: str
    semantic_sha256: str
    root_policy: str

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_REF_SCHEMA:
            raise EvidenceV2Error("unsupported EvidenceRef schema")
        if not self.artifact_schema or not self.root_policy:
            raise EvidenceV2Error("EvidenceRef schema and root policy must be nonempty")
        if not self.absolute_path.startswith("/") or "\x00" in self.absolute_path:
            raise EvidenceV2Error("EvidenceRef path must be absolute and NUL-free")
        if (
            posixpath.normpath(self.absolute_path) != self.absolute_path
            or self.absolute_path.startswith("//")
            or self.absolute_path.endswith("/")
        ):
            raise EvidenceV2Error("EvidenceRef path must be lexically canonical")
        require_sha256(self.byte_sha256, label="EvidenceRef.byte_sha256")
        require_sha256(self.semantic_sha256, label="EvidenceRef.semantic_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "artifact_schema": self.artifact_schema,
            "absolute_path": self.absolute_path,
            "byte_sha256": self.byte_sha256,
            "semantic_sha256": self.semantic_sha256,
            "root_policy": self.root_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        expected = {
            "schema_version",
            "artifact_schema",
            "absolute_path",
            "byte_sha256",
            "semantic_sha256",
            "root_policy",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise EvidenceV2Error("EvidenceRef fields mismatch")
        return cls(**{key: str(value[key]) for key in expected})


@dataclass(frozen=True)
class BoundCanonicalArtifact:
    """Canonical bytes bound to both identities in an EvidenceRef."""

    reference: EvidenceRef
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceRef):
            raise EvidenceV2Error("bound canonical artifact requires an EvidenceRef")
        if not isinstance(self.payload, bytes):
            raise EvidenceV2Error("bound canonical artifact payload must be bytes")

    def read(self) -> dict[str, Any]:
        if sha256_bytes(self.payload) != self.reference.byte_sha256:
            raise EvidenceV2Error("bound canonical artifact byte SHA mismatch")
        value = parse_canonical_json_bytes(self.payload)
        if not isinstance(value, Mapping):
            raise EvidenceV2Error("bound canonical artifact must contain an object")
        normalized = validate_semantic_seal(value)
        if normalized.get("schema_version") != self.reference.artifact_schema:
            raise EvidenceV2Error("bound canonical artifact schema mismatch")
        if semantic_sha256(normalized) != self.reference.semantic_sha256:
            raise EvidenceV2Error("bound canonical artifact semantic SHA mismatch")
        return normalized


@dataclass(frozen=True)
class BoundRawArtifact:
    """Opaque bytes whose domain-specific semantic identity is checked by a parser."""

    reference: EvidenceRef
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceRef):
            raise EvidenceV2Error("bound raw artifact requires an EvidenceRef")
        if not isinstance(self.payload, bytes):
            raise EvidenceV2Error("bound raw artifact payload must be bytes")
        if not self.payload or len(self.payload) > ARTIFACT_MAX_BYTES * 4:
            raise EvidenceV2Error("bound raw artifact byte length is invalid")
        if sha256_bytes(self.payload) != self.reference.byte_sha256:
            raise EvidenceV2Error("bound raw artifact byte SHA mismatch")


def nonauthorizing_projection(*, blockers: list[str]) -> dict[str, Any]:
    normalized = sorted(set(str(item) for item in blockers if str(item)))
    if not normalized:
        normalized = ["evidence_v2_not_integrated"]
    return seal_semantic(
        {
            "schema_version": NONAUTHORIZING_PROJECTION_SCHEMA,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "readiness_status": "no_new_risk",
            "blockers": normalized,
        }
    )


__all__ = [
    "ARTIFACT_MAX_BYTES",
    "BoundCanonicalArtifact",
    "BoundRawArtifact",
    "EVIDENCE_REF_SCHEMA",
    "EvidenceRef",
    "EvidenceV2Error",
    "canonical_json_bytes",
    "decode_f64",
    "encode_f64",
    "nonauthorizing_projection",
    "parse_canonical_json_bytes",
    "require_sha256",
    "seal_semantic",
    "semantic_sha256",
    "sha256_bytes",
    "validate_canonical_value",
    "validate_semantic_seal",
]
