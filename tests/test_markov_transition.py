from __future__ import annotations

import pytest

from quant_investor.regime.transition import (
    bayesian_regime_update,
    default_transition_matrix,
    estimate_transition_matrix,
    markov_prior_update,
    normalize_probabilities,
)
from quant_investor.regime.types import (
    REGIME_RANGE_HIGH_VOL,
    REGIME_RANGE_LOW_VOL,
    REGIME_STATES,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_UNKNOWN,
)


def _assert_rows_sum_to_one(matrix: dict[str, dict[str, float]]) -> None:
    assert set(matrix) == set(REGIME_STATES)
    for state in REGIME_STATES:
        row = matrix[state]
        assert set(row) == set(REGIME_STATES)
        assert sum(row.values()) == pytest.approx(1.0)
        assert all(value >= 0.0 for value in row.values())


def test_default_transition_matrix_rows_sum_to_one() -> None:
    matrix = default_transition_matrix()

    _assert_rows_sum_to_one(matrix)
    assert matrix[REGIME_TREND_UP][REGIME_TREND_UP] == pytest.approx(0.62)
    assert matrix[REGIME_TREND_DOWN][REGIME_TREND_DOWN] == pytest.approx(0.46)


def test_estimate_transition_matrix_fills_missing_states_and_smooths() -> None:
    history = [
        {"dominant_regime": REGIME_TREND_UP},
        {"dominant_regime": REGIME_TREND_UP},
        {"dominant_regime": REGIME_RANGE_LOW_VOL},
        {"dominant_regime": REGIME_RANGE_HIGH_VOL},
    ]

    matrix = estimate_transition_matrix(history, smoothing=1.0)

    _assert_rows_sum_to_one(matrix)
    assert matrix[REGIME_TREND_UP][REGIME_TREND_UP] > matrix[REGIME_TREND_UP][REGIME_TREND_DOWN]
    assert matrix[REGIME_UNKNOWN][REGIME_UNKNOWN] == pytest.approx(0.2)


def test_probability_helpers_are_safe_for_missing_or_invalid_inputs() -> None:
    normalized = normalize_probabilities(
        {
            REGIME_TREND_UP: 2.0,
            REGIME_RANGE_LOW_VOL: -1.0,
            "not-a-state": 99.0,
        }
    )

    assert set(normalized) == set(REGIME_STATES)
    assert normalized[REGIME_TREND_UP] == pytest.approx(1.0)
    assert sum(normalized.values()) == pytest.approx(1.0)

    uniform = normalize_probabilities({})
    assert all(value == pytest.approx(1.0 / len(REGIME_STATES)) for value in uniform.values())


def test_markov_and_bayesian_updates_return_valid_posteriors() -> None:
    transition_matrix = default_transition_matrix()
    previous = {REGIME_TREND_UP: 0.8}
    likelihood = {
        REGIME_TREND_UP: 0.7,
        REGIME_RANGE_LOW_VOL: 0.2,
        REGIME_RANGE_HIGH_VOL: 0.05,
        REGIME_TREND_DOWN: 0.03,
        REGIME_UNKNOWN: 0.02,
    }

    prior = markov_prior_update(previous, transition_matrix)
    posterior = bayesian_regime_update(previous, transition_matrix, likelihood)

    assert set(prior) == set(REGIME_STATES)
    assert sum(prior.values()) == pytest.approx(1.0)
    assert set(posterior) == set(REGIME_STATES)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior[REGIME_TREND_UP] == max(posterior.values())
