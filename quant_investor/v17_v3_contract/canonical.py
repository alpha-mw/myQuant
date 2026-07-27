"""Strict canonical JSON and semantic SHA-256 helpers for protocol v3."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Final

from .identities import IdentityContractError, require_sha256
from .limits import LIMITS, ContractLimitError, checked_add

SEMANTIC_SHA_FIELD: Final = "semantic_sha256"


class CanonicalContractError(ValueError):
    """Raised when JSON is unsafe, unbounded, or noncanonical."""

    exit_code = 2


def _parse_integer(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > LIMITS["max_integer_digits"]:
        raise CanonicalContractError("JSON integer exceeds the decimal digit limit")
    return int(token)


def _parse_float(token: str) -> float:
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
            if len(str(abs(current))) > LIMITS["max_integer_digits"]:
                raise CanonicalContractError(f"JSON integer digit limit exceeded at {path}")
            return
        if type(current) is float:
            if not math.isfinite(current):
                raise CanonicalContractError(f"non-finite JSON number at {path}")
            return
        if type(current) is str:
            try:
                raw = current.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise CanonicalContractError(f"invalid Unicode string at {path}") from exc
            if len(raw) > LIMITS["max_string_utf8_bytes"]:
                raise CanonicalContractError(f"JSON string byte limit exceeded at {path}")
            return
        if type(current) is list:
            if len(current) > LIMITS["max_array_items"]:
                raise CanonicalContractError(f"JSON array item limit exceeded at {path}")
            for index, child in enumerate(current):
                walk(child, depth=depth + 1, path=f"{path}[{index}]")
            return
        if type(current) is dict:
            if len(current) > LIMITS["max_container_members"]:
                raise CanonicalContractError(f"JSON object member limit exceeded at {path}")
            for key, child in current.items():
                if type(key) is not str:
                    raise CanonicalContractError(f"non-string JSON key at {path}")
                if len(key.encode("utf-8", errors="strict")) > LIMITS["max_key_utf8_bytes"]:
                    raise CanonicalContractError(f"JSON key byte limit exceeded at {path}")
                walk(child, depth=depth + 1, path=f"{path}.{key}")
            return
        raise CanonicalContractError(f"unsupported JSON type at {path}: {type(current).__name__}")

    walk(value, depth=1, path=label)


def strict_json_loads(
    raw: bytes,
    *,
    label: str = "JSON",
    max_bytes: int | None = None,
) -> Any:
    if type(raw) is not bytes:
        raise CanonicalContractError(f"{label} input must be bytes")
    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    if type(bound) is not int or bound < 0:
        raise CanonicalContractError(f"{label} byte limit must be nonnegative")
    if len(raw) > bound:
        raise CanonicalContractError(f"{label} exceeds the inclusive byte limit {bound}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CanonicalContractError(f"{label} UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_integer,
        )
    except CanonicalContractError:
        raise
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalContractError(f"{label} is invalid strict JSON") from exc
    validate_json_limits(value, label=label)
    return value


def canonical_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Return compact sorted-key UTF-8 JSON without a trailing newline."""

    validate_json_limits(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanonicalContractError("value is not canonical JSON") from exc
    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    if type(bound) is not int or bound < 0 or len(raw) > bound:
        raise CanonicalContractError("canonical JSON exceeds its byte limit")
    return raw


canonical_json_bytes = canonical_bytes


def canonical_resource_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    bound = LIMITS["general_json_bytes"] if max_bytes is None else max_bytes
    raw = canonical_bytes(value, max_bytes=bound)
    if len(raw) >= bound:
        raise CanonicalContractError("canonical resource exceeds its byte limit")
    return raw + b"\n"


def load_canonical_resource(
    raw: bytes,
    *,
    label: str = "JSON resource",
    max_bytes: int | None = None,
) -> Any:
    value = strict_json_loads(raw, label=label, max_bytes=max_bytes)
    if raw != canonical_resource_bytes(value, max_bytes=max_bytes):
        raise CanonicalContractError(f"{label} is not canonical compact JSON plus newline")
    return value


def semantic_sha256(value: Any, *, max_bytes: int | None = None) -> str:
    if type(value) is not dict:
        raise CanonicalContractError("semantic payload must be an object")
    payload = dict(value)
    payload.pop(SEMANTIC_SHA_FIELD, None)
    return hashlib.sha256(canonical_bytes(payload, max_bytes=max_bytes)).hexdigest()


def seal_semantic(value: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    if type(value) is not dict or SEMANTIC_SHA_FIELD in value:
        raise CanonicalContractError("sealer requires an unsealed object")
    result = dict(value)
    result[SEMANTIC_SHA_FIELD] = semantic_sha256(result, max_bytes=max_bytes)
    canonical_bytes(result, max_bytes=max_bytes)
    return result


def validate_semantic_sha(value: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    if type(value) is not dict:
        raise CanonicalContractError("semantic payload must be an object")
    try:
        declared = require_sha256(value.get(SEMANTIC_SHA_FIELD), label=SEMANTIC_SHA_FIELD)
    except IdentityContractError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if declared != semantic_sha256(value, max_bytes=max_bytes):
        raise CanonicalContractError("semantic_sha256 mismatch")
    canonical_bytes(value, max_bytes=max_bytes)
    return dict(value)


validate_semantic_seal = validate_semantic_sha


__all__ = [
    "CanonicalContractError",
    "SEMANTIC_SHA_FIELD",
    "canonical_bytes",
    "canonical_json_bytes",
    "canonical_resource_bytes",
    "load_canonical_resource",
    "seal_semantic",
    "semantic_sha256",
    "strict_json_loads",
    "validate_json_limits",
    "validate_semantic_seal",
    "validate_semantic_sha",
]
