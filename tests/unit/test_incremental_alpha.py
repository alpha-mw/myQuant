from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.incremental_alpha import (
    NO_POOL_EVIDENCE_SOURCE,
    POOL_RESIDUAL_EVIDENCE_SOURCE,
    residualize_against_pool,
)

DATES = pd.bdate_range("2024-01-01", periods=5)
SYMBOLS = tuple(f"{index:06d}.SZ" for index in range(1, 25))


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=list(SYMBOLS))


def _pool() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))


def test_a_clone_of_the_pool_residualises_to_nothing() -> None:
    pool = _pool()

    residual = residualize_against_pool(pool * 3.0 + 1.0, pool, list(DATES))

    assert np.allclose(residual.to_numpy(), 0.0, atol=1e-9)


def test_an_orthogonal_signal_survives_residualisation() -> None:
    pool = _pool()
    rng = np.random.default_rng(77)
    signal = _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))

    residual = residualize_against_pool(signal, pool, list(DATES))

    # The residual keeps most of the cross-sectional dispersion.
    ranked = signal.rank(axis=1, pct=True)
    kept = residual.std(axis=1) / ranked.std(axis=1)
    assert float(kept.min()) > 0.5


def test_residual_is_orthogonal_to_the_pool_on_every_date() -> None:
    pool = _pool()
    rng = np.random.default_rng(5)
    signal = _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))

    residual = residualize_against_pool(signal, pool, list(DATES))

    for date in DATES:
        pool_ranked = pool.loc[date].rank(pct=True)
        assert float(residual.loc[date].corr(pool_ranked)) == pytest.approx(
            0.0, abs=1e-9
        )


def test_residualisation_without_a_pool_returns_the_ranked_signal() -> None:
    rng = np.random.default_rng(9)
    signal = _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))

    residual = residualize_against_pool(signal, None, list(DATES))

    ranked = signal.rank(axis=1, pct=True)
    centred = ranked.sub(ranked.mean(axis=1), axis=0)
    assert np.allclose(residual.to_numpy(), centred.to_numpy(), equal_nan=True)


def test_residualisation_survives_a_constant_pool() -> None:
    pool = _frame(np.ones((len(DATES), len(SYMBOLS))))
    rng = np.random.default_rng(13)
    signal = _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))

    residual = residualize_against_pool(signal, pool, list(DATES))

    # Nothing to project out, so only the cross-sectional mean is removed.
    assert float(residual.std(axis=1).min()) > 0.0


def test_residualisation_ignores_dates_outside_the_pool() -> None:
    pool = _pool().iloc[:2]
    rng = np.random.default_rng(21)
    signal = _frame(rng.normal(size=(len(DATES), len(SYMBOLS))))

    residual = residualize_against_pool(signal, pool, list(DATES))

    assert list(residual.index) == list(DATES)
    assert residual.loc[DATES[3]].notna().any()


def test_evidence_source_names_are_distinct() -> None:
    assert POOL_RESIDUAL_EVIDENCE_SOURCE != NO_POOL_EVIDENCE_SOURCE
