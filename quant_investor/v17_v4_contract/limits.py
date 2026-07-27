"""Hard JSON and collection limits for the V17 v4 contract surface."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Final, Mapping

PROTOCOL_VERSION: Final = "myquant.v17.v4"

LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "general_json_bytes": 8_388_608,
        "max_array_items": 10_000,
        "max_container_members": 512,
        "max_depth": 32,
        "max_evidence_refs": 256,
        "max_integer_digits": 64,
        "max_key_utf8_bytes": 256,
        "max_string_utf8_bytes": 1_048_576,
        "max_total_nodes": 250_000,
    }
)


class ContractLimitError(ValueError):
    """Raised when bounded v4 input limits are exceeded."""

    exit_code = 2


def require_nonnegative_int(
    value: Any,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise ContractLimitError(f"{label} must be a nonnegative integer")
    if maximum is not None and value > maximum:
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
    "PROTOCOL_VERSION",
    "checked_add",
    "require_nonnegative_int",
]
