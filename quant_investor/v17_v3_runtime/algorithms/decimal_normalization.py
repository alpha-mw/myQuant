"""Canonical Decimal handling shared by the v17 v3 pure algorithms."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from typing import TypeAlias

DECIMAL_QUANTUM = Decimal("0.000000000001")
DECIMAL_PRECISION = 34
DecimalInput: TypeAlias = Decimal | int | float | str


def _coerce_finite_decimal(value: DecimalInput, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float, str)):
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{label} must be decimal-compatible") from exc
    else:
        raise ValueError(f"{label} must be decimal-compatible")
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def normalize_decimal(value: DecimalInput, *, label: str = "value") -> Decimal:
    """Round a finite value to 1e-12 with banker rounding.

    The local context is widened for large fixed-point values while retaining
    the v17 calibration minimum precision of 34 significant digits.
    """

    result = _coerce_finite_decimal(value, label=label)
    digits = len(result.as_tuple().digits)
    raw_exponent = result.as_tuple().exponent
    if not isinstance(raw_exponent, int):
        raise ValueError(f"{label} must be finite")
    exponent = raw_exponent
    required_precision = max(
        DECIMAL_PRECISION,
        digits + max(0, -exponent) + max(0, exponent) + 16,
    )
    try:
        with localcontext() as context:
            context.prec = required_precision
            context.rounding = ROUND_HALF_EVEN
            normalized = result.quantize(DECIMAL_QUANTUM)
    except InvalidOperation as exc:
        raise ValueError(f"{label} cannot be normalized") from exc
    return Decimal("0").quantize(DECIMAL_QUANTUM) if normalized.is_zero() else normalized


def canonical_decimal_string(value: DecimalInput, *, label: str = "value") -> str:
    """Return the non-exponent canonical wire representation of ``value``."""

    normalized = normalize_decimal(value, label=label)
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


__all__ = [
    "DECIMAL_PRECISION",
    "DECIMAL_QUANTUM",
    "DecimalInput",
    "canonical_decimal_string",
    "normalize_decimal",
]
