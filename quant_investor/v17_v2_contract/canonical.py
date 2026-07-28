"""Strict bounded JSON and deterministic set-like ordering for protocol v2."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, TypeVar

from .identities import IdentityContractError, require_sha256
from .limits import LIMITS, ContractLimitError, checked_add, require_nonnegative_int

T = TypeVar("T")
SEMANTIC_SHA_FIELD = "semantic_sha256"


class CanonicalContractError(ValueError):
    """Raised when JSON is unsafe, unbounded, or not canonically representable."""

    exit_code = 2


def _integer_from_token(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > LIMITS["max_integer_digits"]:
        raise CanonicalContractError("JSON integer exceeds the decimal digit limit")
    return int(token)


def _float_from_token(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise CanonicalContractError("non-finite JSON number rejected")
    return value


def _reject_constant(token: str) -> None:
    raise CanonicalContractError(f"non-finite JSON constant rejected: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalContractError(f"duplicate JSON key rejected: {key!r}")
        result[key] = value
    return result


def validate_json_limits(value: Any, *, label: str = "$") -> None:
    """Validate the materialized JSON tree with root depth defined as one."""

    nodes = 0

    def walk(current: Any, *, depth: int, path: str) -> None:
        nonlocal nodes
        if depth > LIMITS["max_depth"]:
            raise CanonicalContractError(f"JSON depth exceeds limit at {path}")
        try:
            nodes = checked_add(
                nodes,
                1,
                label="JSON total nodes",
                maximum=LIMITS["max_total_nodes"],
            )
        except ContractLimitError as exc:
            raise CanonicalContractError(str(exc)) from exc

        if current is None or type(current) is bool:
            return
        if type(current) is int:
            digits = str(abs(current))
            if len(digits) > LIMITS["max_integer_digits"]:
                raise CanonicalContractError(f"JSON integer digit limit exceeded at {path}")
            return
        if type(current) is float:
            if not math.isfinite(current):
                raise CanonicalContractError(f"non-finite JSON number at {path}")
            return
        if type(current) is str:
            try:
                encoded = current.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise CanonicalContractError(f"invalid Unicode string at {path}") from exc
            if len(encoded) > LIMITS["max_string_utf8_bytes"]:
                raise CanonicalContractError(f"JSON string byte limit exceeded at {path}")
            return
        if type(current) is list:
            if len(current) > LIMITS["max_container_members"]:
                raise CanonicalContractError(f"JSON array member limit exceeded at {path}")
            for index, item in enumerate(current):
                walk(item, depth=depth + 1, path=f"{path}[{index}]")
            return
        if type(current) is dict:
            if len(current) > LIMITS["max_container_members"]:
                raise CanonicalContractError(f"JSON object member limit exceeded at {path}")
            for key, item in current.items():
                if type(key) is not str:
                    raise CanonicalContractError(f"non-string JSON key at {path}")
                try:
                    key_bytes = key.encode("utf-8", errors="strict")
                except UnicodeError as exc:
                    raise CanonicalContractError(f"invalid Unicode key at {path}") from exc
                if len(key_bytes) > LIMITS["max_key_utf8_bytes"]:
                    raise CanonicalContractError(f"JSON key byte limit exceeded at {path}")
                walk(item, depth=depth + 1, path=f"{path}.{key}")
            return
        raise CanonicalContractError(f"unsupported JSON value at {path}: {type(current).__name__}")

    walk(value, depth=1, path=label)


def strict_json_loads(
    raw: bytes,
    *,
    label: str = "JSON",
    max_bytes: int | None = None,
) -> Any:
    """Parse strict UTF-8 JSON after enforcing its byte bound before decoding."""

    if type(raw) is not bytes:
        raise CanonicalContractError(f"{label} input must be bytes")
    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    try:
        require_nonnegative_int(bound, label=f"{label} byte limit")
    except ContractLimitError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if len(raw) > bound:
        raise CanonicalContractError(f"{label} exceeds the inclusive byte limit {bound}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CanonicalContractError(f"{label} UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CanonicalContractError(f"{label} is not strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_float_from_token,
            parse_int=_integer_from_token,
        )
    except CanonicalContractError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalContractError(f"{label} is invalid JSON") from exc
    validate_json_limits(value, label=label)
    return value


def canonical_json_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Encode compact sorted-key JSON without a trailing newline."""

    validate_json_limits(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanonicalContractError("value is not canonical JSON") from exc
    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    try:
        require_nonnegative_int(bound, label="canonical JSON byte limit")
    except ContractLimitError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if len(encoded) > bound:
        raise CanonicalContractError(f"canonical JSON exceeds the inclusive byte limit {bound}")
    return encoded


def canonical_resource_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Encode a package resource as compact sorted-key JSON plus one newline."""

    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    try:
        require_nonnegative_int(bound, label="canonical resource byte limit")
    except ContractLimitError as exc:
        raise CanonicalContractError(str(exc)) from exc
    payload = canonical_json_bytes(value, max_bytes=bound)
    if len(payload) >= bound:
        raise CanonicalContractError(f"canonical resource exceeds the inclusive byte limit {bound}")
    return payload + b"\n"


def load_canonical_resource(
    raw: bytes,
    *,
    label: str = "JSON resource",
    max_bytes: int | None = None,
) -> Any:
    """Parse a resource and require its exact compact sorted-key wire bytes."""

    value = strict_json_loads(raw, label=label, max_bytes=max_bytes)
    if raw != canonical_resource_bytes(value, max_bytes=max_bytes):
        raise CanonicalContractError(f"{label} is not canonical compact JSON plus newline")
    return value


def semantic_sha256(value: Any, *, max_bytes: int | None = None) -> str:
    """Hash compact canonical JSON after removing only the root seal field."""

    if type(value) is not dict:
        raise CanonicalContractError("semantic payload must be a JSON object")
    payload = dict(value)
    payload.pop(SEMANTIC_SHA_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(payload, max_bytes=max_bytes)).hexdigest()


def seal_semantic(value: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Return a sealed copy while refusing to preserve a caller-supplied seal."""

    if type(value) is not dict:
        raise CanonicalContractError("semantic payload must be a JSON object")
    if SEMANTIC_SHA_FIELD in value:
        raise CanonicalContractError("semantic_sha256 must not be supplied to the sealer")
    sealed = dict(value)
    sealed[SEMANTIC_SHA_FIELD] = semantic_sha256(sealed, max_bytes=max_bytes)
    canonical_json_bytes(sealed, max_bytes=max_bytes)
    return sealed


def validate_semantic_seal(value: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Validate a root semantic seal and return a defensive shallow copy."""

    if type(value) is not dict:
        raise CanonicalContractError("semantic payload must be a JSON object")
    try:
        declared = require_sha256(
            value.get(SEMANTIC_SHA_FIELD),
            label=SEMANTIC_SHA_FIELD,
        )
    except IdentityContractError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if declared != semantic_sha256(value, max_bytes=max_bytes):
        raise CanonicalContractError("semantic_sha256 mismatch")
    canonical_json_bytes(value, max_bytes=max_bytes)
    return dict(value)


def stored_byte_sha256(value: Any, *, max_bytes: int | None = None) -> str:
    """Hash the exact stored resource form, including its single final newline."""

    return hashlib.sha256(canonical_resource_bytes(value, max_bytes=max_bytes)).hexdigest()


def require_typed_json_scalar(
    value: Any,
    *,
    allow_null: bool,
    label: str = "JSON scalar",
) -> str | int | float | bool | None:
    """Validate a uniquely encoded scalar used in ordered dataset keys."""

    if type(allow_null) is not bool:
        raise CanonicalContractError("allow_null must be boolean")
    if value is None:
        if allow_null:
            return None
        raise CanonicalContractError(f"{label} cannot be null")
    if type(value) is bool:
        return value
    if type(value) is int:
        canonical_json_bytes(value)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalContractError(f"{label} must be finite")
        if value.is_integer():
            raise CanonicalContractError(f"{label} integral float is an alternate integer encoding")
        canonical_json_bytes(value)
        return value
    if type(value) is str:
        canonical_json_bytes(value)
        return value
    raise CanonicalContractError(f"{label} has unsupported type {type(value).__name__}")


def typed_scalar_total_order_key(
    value: Any,
    *,
    allow_null: bool,
    label: str = "JSON scalar",
) -> tuple[int, Any, bytes]:
    """Return the total-order key ``null < bool < numeric < string``.

    Integers and non-integral floats share one exact numeric domain.  The
    canonical JSON bytes are retained as the final deterministic tie-break.
    """

    scalar = require_typed_json_scalar(value, allow_null=allow_null, label=label)
    wire = canonical_json_bytes(scalar)
    if scalar is None:
        return (0, 0, wire)
    if type(scalar) is bool:
        return (1, scalar, wire)
    if type(scalar) is int:
        return (2, Fraction(scalar, 1), wire)
    if type(scalar) is float:
        return (2, Fraction.from_float(scalar), wire)
    return (3, scalar, wire)


def _set_identity_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def canonicalize_set_like(
    values: Sequence[T],
    *,
    identity_key: Callable[[T], Any],
    order_key: Callable[[T], Sequence[Any]],
    label: str,
) -> list[T]:
    """Reject duplicates first, then impose a complete deterministic total order."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise CanonicalContractError(f"{label} must be an array")
    identities: set[bytes] = set()
    decorated: list[tuple[tuple[bytes, ...], bytes, T]] = []
    for index, item in enumerate(values):
        identity = _set_identity_bytes(identity_key(item))
        if identity in identities:
            raise CanonicalContractError(f"{label} contains duplicate identity at index {index}")
        identities.add(identity)
        raw_order = order_key(item)
        if isinstance(raw_order, (str, bytes, bytearray)) or not isinstance(raw_order, Sequence):
            raise CanonicalContractError(f"{label} order key must be an array")
        order = tuple(canonical_json_bytes(component) for component in raw_order)
        decorated.append((order, canonical_json_bytes(item), item))
    decorated.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in decorated]


def require_canonical_set_like_wire(
    values: Sequence[T],
    *,
    identity_key: Callable[[T], Any],
    order_key: Callable[[T], Sequence[Any]],
    label: str,
) -> list[T]:
    """Return the values only when their wire order already equals canonical order."""

    canonical = canonicalize_set_like(
        values,
        identity_key=identity_key,
        order_key=order_key,
        label=label,
    )
    original_bytes = [canonical_json_bytes(item) for item in values]
    canonical_bytes = [canonical_json_bytes(item) for item in canonical]
    if original_bytes != canonical_bytes:
        raise CanonicalContractError(f"{label} is not in canonical wire order")
    return canonical


__all__ = [
    "CanonicalContractError",
    "canonical_json_bytes",
    "canonical_resource_bytes",
    "canonicalize_set_like",
    "load_canonical_resource",
    "require_canonical_set_like_wire",
    "seal_semantic",
    "semantic_sha256",
    "stored_byte_sha256",
    "strict_json_loads",
    "typed_scalar_total_order_key",
    "require_typed_json_scalar",
    "validate_semantic_seal",
    "validate_json_limits",
]
