"""Canonical JSON helpers for the V17 v5 successor contract."""

from __future__ import annotations

from typing import Any

from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    seal_semantic,
    semantic_sha256,
    strict_json_loads,
    validate_json_limits,
    validate_semantic_sha,
)


def require_canonical_object(raw: bytes, *, label: str) -> dict[str, Any]:
    value = load_canonical_resource(raw, label=label)
    if type(value) is not dict:
        raise CanonicalContractError(f"{label} root must be an object")
    return dict(value)


__all__ = [
    "CanonicalContractError",
    "canonical_bytes",
    "canonical_resource_bytes",
    "load_canonical_resource",
    "require_canonical_object",
    "seal_semantic",
    "semantic_sha256",
    "strict_json_loads",
    "validate_json_limits",
    "validate_semantic_sha",
]
