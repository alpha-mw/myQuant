"""Small local semantic sealer with no resource or filesystem dependency."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def _validate(value: Any) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("semantic payload contains non-finite float")
        return
    if type(value) is list:
        for item in value:
            _validate(item)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("semantic payload keys must be strings")
        for item in value.values():
            _validate(item)
        return
    raise ValueError(f"semantic payload type unsupported: {type(value).__name__}")


def _bytes(value: Mapping[str, Any]) -> bytes:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    _validate(payload)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def seal_semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    if "semantic_sha256" in value:
        raise ValueError("semantic_sha256 must not be supplied")
    result = dict(value)
    result["semantic_sha256"] = hashlib.sha256(_bytes(result)).hexdigest()
    return result


def validate_semantic_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    declared = result.get("semantic_sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
        or hashlib.sha256(_bytes(result)).hexdigest() != declared
    ):
        raise ValueError("semantic_sha256 mismatch")
    return result
