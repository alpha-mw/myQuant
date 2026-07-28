"""Single-source numeric limits for the isolated v17 protocol-v2 contracts.

This module is deliberately independent from :mod:`quant_investor.v17`.  It
loads one package resource, performs no filesystem discovery, and exposes
strict helpers that reject booleans where JSON integer counts are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

LIMIT_PROFILE_VERSION: Final = "myquant.v17.v2.limits.v1"
PROTOCOL_VERSION: Final = "myquant.v17.v2"
_RESOURCE_PATH: Final = Path(__file__).with_name("resources") / "limits.v1.json"
_LIMIT_NAMES: Final = frozenset(
    {
        "fundamental_generation_manifest_bytes",
        "general_json_bytes",
        "max_batch_items",
        "max_candidates",
        "max_cell_utf8_bytes",
        "max_container_members",
        "max_dataset_bytes",
        "max_dataset_rows",
        "max_dataset_shards",
        "max_deep_reviews",
        "max_depth",
        "max_evidence_refs",
        "max_integer_digits",
        "max_key_utf8_bytes",
        "max_ledger_artifacts",
        "max_ledger_history",
        "max_shard_bytes",
        "max_sources",
        "max_string_utf8_bytes",
        "max_symbol_open_days",
        "max_total_nodes",
        "max_universe_symbols",
    }
)


class ContractLimitError(ValueError):
    """Raised when a protocol-v2 limit or bounded count is invalid."""

    exit_code = 2


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ContractLimitError(f"duplicate key in v17 v2 limits resource: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(token: str) -> None:
    raise ContractLimitError(f"non-finite value in v17 v2 limits resource: {token}")


def _load_limits() -> Mapping[str, int]:
    try:
        raw = _RESOURCE_PATH.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ContractLimitError("v17 v2 limits resource must not contain a UTF-8 BOM")
        text = raw.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        canonical = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except ContractLimitError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractLimitError("v17 v2 limits resource is unreadable") from exc
    if raw != canonical:
        raise ContractLimitError(
            "v17 v2 limits resource must be compact sorted-key UTF-8 plus one newline"
        )
    if type(payload) is not dict or set(payload) != {
        "limits",
        "protocol_version",
        "version",
    }:
        raise ContractLimitError("v17 v2 limits resource has an invalid shape")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ContractLimitError("v17 v2 limits protocol version mismatch")
    if payload["version"] != LIMIT_PROFILE_VERSION:
        raise ContractLimitError("v17 v2 limits resource version mismatch")
    raw_limits = payload["limits"]
    if type(raw_limits) is not dict or set(raw_limits) != _LIMIT_NAMES:
        raise ContractLimitError("v17 v2 limits table is incomplete")
    limits: dict[str, int] = {}
    for name in sorted(_LIMIT_NAMES):
        value = raw_limits[name]
        if type(value) is not int or value <= 0:
            raise ContractLimitError(f"v17 v2 limit {name} must be a positive integer")
        limits[name] = value
    return MappingProxyType(limits)


LIMITS: Final[Mapping[str, int]] = _load_limits()


def get_limit(name: str) -> int:
    """Return one named limit and fail closed for unknown names."""

    if type(name) is not str or name not in LIMITS:
        raise ContractLimitError(f"unknown v17 v2 limit: {name!r}")
    return LIMITS[name]


def require_nonnegative_int(
    value: Any,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    """Validate an exact JSON-style nonnegative integer, inclusively bounded."""

    if type(value) is not int or value < 0:
        raise ContractLimitError(f"{label} must be a nonnegative integer")
    if maximum is not None:
        if type(maximum) is not int or maximum < 0:
            raise ContractLimitError(f"{label} maximum must be a nonnegative integer")
        if value > maximum:
            raise ContractLimitError(f"{label} exceeds the inclusive maximum {maximum}")
    return value


def checked_add(
    total: Any,
    increment: Any,
    *,
    label: str,
    maximum: int,
) -> int:
    """Add two counts only after proving the inclusive aggregate cannot overflow."""

    current = require_nonnegative_int(total, label=f"{label} total")
    addition = require_nonnegative_int(increment, label=f"{label} increment")
    bound = require_nonnegative_int(maximum, label=f"{label} maximum")
    if addition > bound - current:
        raise ContractLimitError(f"{label} exceeds the inclusive maximum {bound}")
    return current + addition


__all__ = [
    "ContractLimitError",
    "LIMITS",
    "LIMIT_PROFILE_VERSION",
    "PROTOCOL_VERSION",
    "checked_add",
    "get_limit",
    "require_nonnegative_int",
]
