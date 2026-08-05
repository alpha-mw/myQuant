"""Trial counting must count independent bets, not reparameterisations.

Seventy smoothing variants of one volume-stability idea are not seventy trials.
Treating them as such inflates the deflated-Sharpe bar far beyond what the
search actually explored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.trial_correction import (
    DEFAULT_TRIAL_CLUSTER_FLOOR,
    effective_trial_count,
)

DATES = pd.bdate_range("2021-01-04", periods=240)


def _series(seed: int, *, scale: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, scale, size=len(DATES)), index=DATES)


def _variants(base: pd.Series, count: int, *, noise: float) -> dict[str, pd.Series]:
    rng = np.random.default_rng(999)
    return {
        f"variant_{index}": base
        + pd.Series(
            rng.normal(0.0, noise, size=len(base)), index=base.index
        )
        for index in range(count)
    }


def test_independent_candidates_each_count_as_a_trial() -> None:
    series = {f"c{index}": _series(index) for index in range(8)}

    evidence = effective_trial_count(series)

    assert evidence["trial_count"] == 8
    assert evidence["effective_trial_count"] == 8
    assert evidence["cluster_count"] == 8


def test_reparameterisations_collapse_into_one_trial() -> None:
    base = _series(1)
    series = _variants(base, 20, noise=0.05)

    evidence = effective_trial_count(series)

    assert evidence["trial_count"] == 20
    assert evidence["effective_trial_count"] == 1


def test_a_mixed_set_counts_families_not_members() -> None:
    series: dict[str, pd.Series] = {}
    for family in range(4):
        series.update(
            {
                f"f{family}_{name}": value
                for name, value in _variants(
                    _series(100 + family), 10, noise=0.05
                ).items()
            }
        )

    evidence = effective_trial_count(series)

    assert evidence["trial_count"] == 40
    assert evidence["effective_trial_count"] == 4


def test_anticorrelated_candidates_are_the_same_bet() -> None:
    base = _series(3)
    series = {"long": base, "short": -base}

    evidence = effective_trial_count(series)

    # The sign of a factor is a convention, not an independent hypothesis.
    assert evidence["effective_trial_count"] == 1


def test_the_floor_controls_how_aggressively_trials_collapse() -> None:
    base = _series(7)
    series = _variants(base, 12, noise=1.0)

    loose = effective_trial_count(series, correlation_floor=0.99)
    tight = effective_trial_count(series, correlation_floor=0.10)

    assert loose["effective_trial_count"] > tight["effective_trial_count"]


def test_effective_count_never_exceeds_the_raw_count() -> None:
    series = {f"c{index}": _series(index) for index in range(5)}

    evidence = effective_trial_count(series)

    assert 1 <= evidence["effective_trial_count"] <= evidence["trial_count"]


def test_series_without_enough_overlap_stay_separate() -> None:
    base = _series(2)
    series = {
        "early": base.iloc[:100],
        "late": base.iloc[140:],
    }

    evidence = effective_trial_count(series)

    assert evidence["effective_trial_count"] == 2


def test_an_empty_set_fails_closed_to_one_trial() -> None:
    evidence = effective_trial_count({})

    assert evidence["trial_count"] == 0
    assert evidence["effective_trial_count"] == 1


def test_the_default_floor_matches_the_gate_redundancy_ceiling() -> None:
    assert DEFAULT_TRIAL_CLUSTER_FLOOR == pytest.approx(0.70)
