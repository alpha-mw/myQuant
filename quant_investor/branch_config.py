"""Canonical branch weight configuration for the research core."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

CANONICAL_BRANCH_ORDER: tuple[str, ...] = (
    "quant",
    "fundamental",
    "macro",
)

_RAW_THREE_BRANCH_WEIGHTS: dict[str, float] = {
    "quant": 0.28,
    "fundamental": 0.15,
    "macro": 0.15,
}


def _normalize_branch_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(weights[branch_name]) for branch_name in CANONICAL_BRANCH_ORDER)
    if total <= 0.0:
        raise ValueError("Raw branch weights must have a positive total.")
    return {
        branch_name: float(weights[branch_name]) / total
        for branch_name in CANONICAL_BRANCH_ORDER
    }


DEFAULT_BRANCH_WEIGHTS: dict[str, float] = _normalize_branch_weights(_RAW_THREE_BRANCH_WEIGHTS)

BRANCH_WEIGHT_VERSION = "branch-weights.v3.three-branch"

_WEIGHT_SUM_TOLERANCE = 1e-9


def validate_branch_weights(weights: Mapping[str, float]) -> None:
    """Validate branch weights without normalizing invalid input."""

    expected = set(CANONICAL_BRANCH_ORDER)
    actual = set(weights)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing branches: {', '.join(missing)}")
        if extra:
            details.append(f"extra branches: {', '.join(extra)}")
        expected_text = ", ".join(CANONICAL_BRANCH_ORDER)
        raise ValueError(
            "Branch weights must contain exactly the canonical branches "
            f"({expected_text}); {'; '.join(details)}."
        )

    total = 0.0
    for branch_name in CANONICAL_BRANCH_ORDER:
        value = weights[branch_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Branch weight for {branch_name!r} must be a finite number.")
        weight = float(value)
        if not math.isfinite(weight):
            raise ValueError(f"Branch weight for {branch_name!r} must be finite; got {value!r}.")
        if weight < 0.0:
            raise ValueError(f"Branch weight for {branch_name!r} must be non-negative; got {weight}.")
        total += weight

    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            "Branch weights must sum to 1.0 within "
            f"{_WEIGHT_SUM_TOLERANCE}; got {total:.12g}."
        )


def get_default_branch_weights() -> dict[str, float]:
    """Return a fresh copy of the validated default branch weights."""

    validate_branch_weights(DEFAULT_BRANCH_WEIGHTS)
    return dict(DEFAULT_BRANCH_WEIGHTS)


validate_branch_weights(DEFAULT_BRANCH_WEIGHTS)
