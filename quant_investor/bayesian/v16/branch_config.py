"""Unactivated v16 four-branch weight contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

CANONICAL_BRANCH_ORDER: tuple[str, ...] = (
    "quant",
    "fundamental",
    "macro",
    "llm",
)
DEFAULT_BRANCH_WEIGHTS: dict[str, float] = {
    branch_name: 0.25 for branch_name in CANONICAL_BRANCH_ORDER
}
BRANCH_WEIGHT_VERSION = "branch-weights.v16.four-branch.equal"


def validate_branch_weights(weights: Mapping[str, float]) -> None:
    if set(weights) != set(CANONICAL_BRANCH_ORDER):
        raise ValueError(
            "v16 branch weights must contain exactly " f"{list(CANONICAL_BRANCH_ORDER)!r}."
        )
    total = 0.0
    for branch_name in CANONICAL_BRANCH_ORDER:
        raw_value = weights[branch_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(f"v16 branch weight {branch_name!r} must be numeric.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"v16 branch weight {branch_name!r} must be finite.")
        if not math.isclose(value, 0.25, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Each v16 branch weight must equal exactly 0.25.")
        total += value
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("v16 branch weights must sum to 1.0.")


def get_default_branch_weights() -> dict[str, float]:
    validate_branch_weights(DEFAULT_BRANCH_WEIGHTS)
    return dict(DEFAULT_BRANCH_WEIGHTS)


validate_branch_weights(DEFAULT_BRANCH_WEIGHTS)


__all__ = [
    "BRANCH_WEIGHT_VERSION",
    "CANONICAL_BRANCH_ORDER",
    "DEFAULT_BRANCH_WEIGHTS",
    "get_default_branch_weights",
    "validate_branch_weights",
]
