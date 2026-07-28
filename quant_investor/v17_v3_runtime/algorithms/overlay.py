"""Monotonic shrink-only portfolio overlay validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

from .decimal_normalization import DECIMAL_PRECISION, DecimalInput, normalize_decimal


@dataclass(frozen=True)
class OverlayValidation:
    valid: bool
    baseline_targets: Mapping[str, Decimal]
    post_targets: Mapping[str, Decimal]
    baseline_gross: Decimal
    post_gross: Decimal
    cash_delta: Decimal
    blockers: tuple[str, ...] = ()


def _canonical_targets(
    values: Mapping[str, DecimalInput],
    *,
    label: str,
) -> Mapping[str, Decimal]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, Decimal] = {}
    for symbol, raw_value in values.items():
        if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol:
            raise ValueError(f"{label} symbol must be canonical")
        value = normalize_decimal(raw_value, label=f"{label}.{symbol}")
        if value < 0:
            raise ValueError(f"{label}.{symbol} must be nonnegative")
        result[symbol] = value
    return MappingProxyType(result)


def validate_monotonic_overlay(
    baseline_targets: Mapping[str, DecimalInput],
    post_targets: Mapping[str, DecimalInput],
) -> OverlayValidation:
    """Validate caller-supplied post targets without renormalizing them."""

    baseline = _canonical_targets(baseline_targets, label="baseline_targets")
    post = _canonical_targets(post_targets, label="post_targets")
    blockers: list[str] = []
    extra = sorted(set(post).difference(baseline))
    if extra:
        blockers.append(f"post_symbol_not_in_baseline:{','.join(extra)}")
    for symbol in sorted(set(post).intersection(baseline)):
        if post[symbol] > baseline[symbol]:
            blockers.append(f"post_target_exceeds_baseline:{symbol}")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        baseline_gross = normalize_decimal(
            sum(baseline.values(), Decimal("0")),
            label="baseline_gross",
        )
        post_gross = normalize_decimal(
            sum(post.values(), Decimal("0")),
            label="post_gross",
        )
    if post_gross > baseline_gross:
        blockers.append("post_gross_exceeds_baseline")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        cash_delta = normalize_decimal(
            baseline_gross - post_gross,
            label="cash_delta",
        )
    return OverlayValidation(
        valid=not blockers,
        baseline_targets=baseline,
        post_targets=post,
        baseline_gross=baseline_gross,
        post_gross=post_gross,
        cash_delta=cash_delta,
        blockers=tuple(blockers),
    )


__all__ = ["OverlayValidation", "validate_monotonic_overlay"]
