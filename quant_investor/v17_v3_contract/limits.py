"""Single-source bounded limits for protocol v3."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

PROTOCOL_VERSION: Final = "myquant.v17.v3"
LIMITS_VERSION: Final = "myquant.v17.v3.limits.v1"
_RESOURCE_PATH: Final = Path(__file__).with_name("resources") / "limits.v1.json"
_LIMIT_NAMES: Final = frozenset(
    {
        "general_json_bytes",
        "max_array_items",
        "max_candidates",
        "max_container_members",
        "max_depth",
        "max_evidence_refs",
        "max_integer_digits",
        "max_key_utf8_bytes",
        "max_ledger_artifacts",
        "max_ledger_events",
        "max_sources",
        "max_string_utf8_bytes",
        "max_total_nodes",
        "max_universe_symbols",
    }
)


class ContractLimitError(ValueError):
    """Raised when a v3 limit or count is invalid."""

    exit_code = 2


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractLimitError(f"duplicate limits key: {key!r}")
        result[key] = value
    return result


def _load_limits() -> Mapping[str, int]:
    try:
        raw = _RESOURCE_PATH.read_bytes()
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractLimitError(f"non-finite limit: {token}")
            ),
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
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ContractLimitError("v3 limits resource is unreadable") from exc
    if raw != canonical:
        raise ContractLimitError("v3 limits resource is not canonical JSON plus newline")
    if type(payload) is not dict or set(payload) != {
        "array_order_semantics",
        "authority",
        "limits",
        "protocol_version",
        "semantic_sha256",
        "version",
    }:
        raise ContractLimitError("v3 limits resource shape mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION or payload["version"] != LIMITS_VERSION:
        raise ContractLimitError("v3 limits identity mismatch")
    values = payload["limits"]
    if type(values) is not dict or set(values) != _LIMIT_NAMES:
        raise ContractLimitError("v3 limits table is incomplete")
    result: dict[str, int] = {}
    for name in sorted(_LIMIT_NAMES):
        value = values[name]
        if type(value) is not int or value <= 0:
            raise ContractLimitError(f"v3 limit {name} must be a positive integer")
        result[name] = value
    return MappingProxyType(result)


LIMITS: Final = _load_limits()


def get_limit(name: str) -> int:
    if type(name) is not str or name not in LIMITS:
        raise ContractLimitError(f"unknown v3 limit: {name!r}")
    return LIMITS[name]


def require_nonnegative_int(
    value: Any,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise ContractLimitError(f"{label} must be a nonnegative integer")
    if maximum is not None:
        if type(maximum) is not int or maximum < 0:
            raise ContractLimitError(f"{label} maximum must be nonnegative")
        if value > maximum:
            raise ContractLimitError(f"{label} exceeds the inclusive maximum {maximum}")
    return value


def checked_add(total: Any, increment: Any, *, label: str, maximum: int) -> int:
    current = require_nonnegative_int(total, label=f"{label} total")
    addition = require_nonnegative_int(increment, label=f"{label} increment")
    bound = require_nonnegative_int(maximum, label=f"{label} maximum")
    if addition > bound - current:
        raise ContractLimitError(f"{label} exceeds the inclusive maximum {bound}")
    return current + addition


__all__ = [
    "ContractLimitError",
    "LIMITS",
    "LIMITS_VERSION",
    "PROTOCOL_VERSION",
    "checked_add",
    "get_limit",
    "require_nonnegative_int",
]
