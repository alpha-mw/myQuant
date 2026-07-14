"""Shared fail-closed validation for current structured artifacts."""

from __future__ import annotations

import math
from typing import Any


def _require_finite(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _require_bounded(
    value: float,
    field_name: str,
    lower: float,
    upper: float,
) -> float:
    number = _require_finite(value, field_name)
    if not lower <= number <= upper:
        raise ValueError(f"{field_name} must be in [{lower}, {upper}]; got {value!r}.")
    return number


def require_finite_structure(value: Any, *, path: str) -> None:
    """Reject NaN/Inf anywhere in a JSON-like current artifact."""

    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_structure(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite_structure(child, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _require_finite(value, path)


def validate_posterior_numeric_fields(value: Any) -> None:
    """Validate scalar semantics shared by posterior and decision artifacts."""

    for field_name in (
        "posterior_win_rate",
        "posterior_confidence",
        "action_threshold_used",
    ):
        setattr(
            value,
            field_name,
            _require_bounded(getattr(value, field_name), field_name, 0.0, 1.0),
        )
    value.posterior_action_score = _require_bounded(
        value.posterior_action_score,
        "posterior_action_score",
        -1.0,
        1.0,
    )
    value.regime_adjustment = _require_bounded(
        value.regime_adjustment,
        "regime_adjustment",
        -1.0,
        1.0,
    )
    for field_name in (
        "posterior_capacity_penalty",
        "correlation_discount",
        "coverage_discount",
        "data_quality_penalty",
        "fallback_penalty",
    ):
        setattr(
            value,
            field_name,
            _require_bounded(getattr(value, field_name), field_name, 0.0, 1.0),
        )
    for field_name in ("posterior_expected_alpha", "posterior_edge_after_costs"):
        setattr(value, field_name, _require_finite(getattr(value, field_name), field_name))
    if isinstance(value.rank, bool) or not isinstance(value.rank, int) or value.rank < 0:
        raise ValueError(f"rank must be a non-negative integer; got {value.rank!r}.")


__all__ = ["require_finite_structure", "validate_posterior_numeric_fields"]
