from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from quant_investor.regime.types import (
    REGIME_RANGE_HIGH_VOL,
    REGIME_RANGE_LOW_VOL,
    REGIME_STATES,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_UNKNOWN,
)


def _finite_non_negative(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric < 0.0:
        return 0.0
    return numeric


def _uniform(states: Sequence[str]) -> dict[str, float]:
    state_list = tuple(states) or REGIME_STATES
    value = 1.0 / len(state_list)
    return {state: value for state in state_list}


def normalize_probabilities(
    mapping: Mapping[str, Any] | None,
    states: Sequence[str] = REGIME_STATES,
) -> dict[str, float]:
    state_list = tuple(states) or REGIME_STATES
    if not isinstance(mapping, Mapping):
        return _uniform(state_list)
    values = {state: _finite_non_negative(mapping.get(state, 0.0)) for state in state_list}
    total = sum(values.values())
    if total <= 0.0:
        return _uniform(state_list)
    return {state: values[state] / total for state in state_list}


def default_transition_matrix(
    states: Sequence[str] = REGIME_STATES,
) -> dict[str, dict[str, float]]:
    base = {
        REGIME_TREND_UP: {
            REGIME_TREND_UP: 0.62,
            REGIME_RANGE_LOW_VOL: 0.22,
            REGIME_RANGE_HIGH_VOL: 0.10,
            REGIME_TREND_DOWN: 0.04,
            REGIME_UNKNOWN: 0.02,
        },
        REGIME_RANGE_LOW_VOL: {
            REGIME_TREND_UP: 0.22,
            REGIME_RANGE_LOW_VOL: 0.46,
            REGIME_RANGE_HIGH_VOL: 0.20,
            REGIME_TREND_DOWN: 0.07,
            REGIME_UNKNOWN: 0.05,
        },
        REGIME_RANGE_HIGH_VOL: {
            REGIME_TREND_UP: 0.12,
            REGIME_RANGE_LOW_VOL: 0.24,
            REGIME_RANGE_HIGH_VOL: 0.42,
            REGIME_TREND_DOWN: 0.17,
            REGIME_UNKNOWN: 0.05,
        },
        REGIME_TREND_DOWN: {
            REGIME_TREND_UP: 0.07,
            REGIME_RANGE_LOW_VOL: 0.18,
            REGIME_RANGE_HIGH_VOL: 0.25,
            REGIME_TREND_DOWN: 0.46,
            REGIME_UNKNOWN: 0.04,
        },
        REGIME_UNKNOWN: {
            REGIME_TREND_UP: 0.20,
            REGIME_RANGE_LOW_VOL: 0.25,
            REGIME_RANGE_HIGH_VOL: 0.25,
            REGIME_TREND_DOWN: 0.20,
            REGIME_UNKNOWN: 0.10,
        },
    }
    return {
        state: normalize_probabilities(base.get(state, {}), states=states)
        for state in tuple(states) or REGIME_STATES
    }


def _extract_regime(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("dominant_regime") or "")
    return str(getattr(item, "dominant_regime", "") or "")


def estimate_transition_matrix(
    history: Iterable[Any],
    states: Sequence[str] = REGIME_STATES,
    smoothing: float = 1.0,
) -> dict[str, dict[str, float]]:
    state_list = tuple(states) or REGIME_STATES
    smooth = _finite_non_negative(smoothing)
    if smooth <= 0.0:
        smooth = 1.0
    counts = {
        state: {target: smooth for target in state_list}
        for state in state_list
    }
    sequence = [
        regime
        for regime in (_extract_regime(item) for item in list(history or []))
        if regime in state_list
    ]
    for previous, current in zip(sequence, sequence[1:]):
        counts[previous][current] += 1.0
    return {
        state: normalize_probabilities(counts[state], states=state_list)
        for state in state_list
    }


def _sanitize_transition_matrix(
    transition_matrix: Mapping[str, Mapping[str, Any]] | None,
    states: Sequence[str],
) -> dict[str, dict[str, float]]:
    if not isinstance(transition_matrix, Mapping):
        return default_transition_matrix(states)
    return {
        state: normalize_probabilities(
            transition_matrix.get(state, {}) if isinstance(transition_matrix.get(state), Mapping) else {},
            states=states,
        )
        for state in tuple(states) or REGIME_STATES
    }


def markov_prior_update(
    previous_posterior: Mapping[str, Any] | None,
    transition_matrix: Mapping[str, Mapping[str, Any]] | None,
    states: Sequence[str] = REGIME_STATES,
) -> dict[str, float]:
    state_list = tuple(states) or REGIME_STATES
    previous = normalize_probabilities(previous_posterior, states=state_list)
    matrix = _sanitize_transition_matrix(transition_matrix, state_list)
    updated = {state: 0.0 for state in state_list}
    for previous_state in state_list:
        previous_weight = previous.get(previous_state, 0.0)
        row = matrix.get(previous_state, {})
        for target_state in state_list:
            updated[target_state] += previous_weight * float(row.get(target_state, 0.0))
    return normalize_probabilities(updated, states=state_list)


def bayesian_regime_update(
    previous_posterior: Mapping[str, Any] | None,
    transition_matrix: Mapping[str, Mapping[str, Any]] | None,
    likelihood: Mapping[str, Any] | None,
    states: Sequence[str] = REGIME_STATES,
) -> dict[str, float]:
    state_list = tuple(states) or REGIME_STATES
    prior = markov_prior_update(previous_posterior, transition_matrix, states=state_list)
    normalized_likelihood = normalize_probabilities(likelihood, states=state_list)
    posterior = {
        state: prior.get(state, 0.0) * normalized_likelihood.get(state, 0.0)
        for state in state_list
    }
    return normalize_probabilities(posterior, states=state_list)
