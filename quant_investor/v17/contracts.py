"""Small, deterministic contracts shared by the v17 shadow modules."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
import math
import re
from typing import Any

V17_PACKAGE_VERSION = "17.0.0"
V17_SCHEMA_PREFIX = "myquant.v17"
AUTHORITY = False

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CN_SYMBOL_RE = re.compile(r"^(?:[0368]\d{5}\.(?:SZ|SH|BJ)|\d{6})$")


class V17ContractError(ValueError):
    """Fail-closed contract error for the v17 shadow-only lane."""

    exit_code = 2


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class QuantTiming(str, Enum):
    BUY_NOW = "BUY_NOW"
    WATCH = "WATCH"
    TRIM_TIMING = "TRIM_TIMING"


class FundamentalEligibility(str, Enum):
    F_ELIGIBLE = "F_ELIGIBLE"
    F_INELIGIBLE = "F_INELIGIBLE"
    UNAVAILABLE = "UNAVAILABLE"


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


def require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise V17ContractError(f"{label} must be an object")
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise V17ContractError(f"{label} shape mismatch; missing={missing}, extra={extra}")


def require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise V17ContractError(f"{label} must be a stable identifier")
    return value


def require_nonempty_string(value: Any, *, label: str, max_chars: int = 512) -> str:
    if not isinstance(value, str):
        raise V17ContractError(f"{label} must be a string")
    text = value.strip()
    if not text or text != value or len(text) > max_chars:
        raise V17ContractError(f"{label} must be a nonempty canonical string")
    return text


def require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise V17ContractError(f"{label} must be boolean")
    return value


def require_authority_false(value: Any) -> None:
    if value is not False:
        raise V17ContractError("v17 shadow authority must be false")


def require_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V17ContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise V17ContractError(f"{label} must be finite")
    if minimum is not None:
        if minimum_exclusive and number <= minimum:
            raise V17ContractError(f"{label} must be greater than {minimum}")
        if not minimum_exclusive and number < minimum:
            raise V17ContractError(f"{label} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise V17ContractError(f"{label} must be at most {maximum}")
    return number


def require_ratio(
    value: Any,
    *,
    label: str,
    allow_zero: bool = True,
) -> float:
    return require_number(
        value,
        label=label,
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=not allow_zero,
    )


def parse_iso_date(value: Any, *, label: str) -> date:
    if not isinstance(value, str):
        raise V17ContractError(f"{label} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise V17ContractError(f"{label} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise V17ContractError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise V17ContractError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V17ContractError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V17ContractError(f"{label} must be UTC")
    canonical = parsed.isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise V17ContractError(f"{label} must be a canonical UTC timestamp")
    return parsed


def require_symbol(value: Any, *, label: str = "symbol") -> str:
    if not isinstance(value, str) or not CN_SYMBOL_RE.fullmatch(value):
        raise V17ContractError(f"{label} must be a canonical CN security code")
    return value


def coerce_enum(value: Any, enum_type: type[Enum], *, label: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = sorted(str(item.value) for item in enum_type)
        raise V17ContractError(f"{label} must be one of {choices}") from exc


__all__ = [
    "AUTHORITY",
    "Availability",
    "FundamentalEligibility",
    "QuantTiming",
    "TradeSide",
    "V17ContractError",
    "V17_PACKAGE_VERSION",
    "V17_SCHEMA_PREFIX",
    "coerce_enum",
    "parse_iso_date",
    "parse_utc_timestamp",
    "require_authority_false",
    "require_bool",
    "require_exact_keys",
    "require_identifier",
    "require_nonempty_string",
    "require_number",
    "require_ratio",
    "require_symbol",
]
