"""Candidate metrics must report what a candidate adds to the existing pool.

Standalone IC cannot distinguish a genuinely new factor from a reparameterised
copy of something already held.  These metrics measure the candidate after the
production pool has been projected out of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.incremental_alpha import (
    NO_POOL_EVIDENCE_SOURCE,
    POOL_RESIDUAL_EVIDENCE_SOURCE,
)
from scripts.retest_aquant_alpha_mix_8gate import RetestContext, candidate_metrics

SESSIONS = pd.bdate_range("2021-01-04", periods=900)
SYMBOLS = tuple(f"{index:06d}.SZ" for index in range(1, 31))


def _base(*, with_pool: bool) -> tuple[RetestContext, pd.DataFrame, pd.DataFrame]:
    """Two orthogonal edges, both genuinely priced into the forward return.

    ``alpha_pool`` is what production already holds; ``alpha_new`` is an equally
    strong but independent edge.  A candidate built on ``alpha_pool`` must
    residualise to nothing while one built on ``alpha_new`` must survive.
    """

    rng = np.random.default_rng(20260805)
    shape = (len(SESSIONS), len(SYMBOLS))

    def _frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=SESSIONS, columns=list(SYMBOLS))

    alpha_pool = _frame(rng.normal(size=shape))
    alpha_new = _frame(rng.normal(size=shape))
    forward = (
        alpha_pool.mul(0.01)
        .add(alpha_new.mul(0.01))
        .add(_frame(rng.normal(0.0, 0.02, size=shape)))
    )
    prices = _frame(
        100.0 + np.cumsum(rng.normal(0.0, 1.0, size=shape), axis=0)
    )
    pool_signal = alpha_pool.add(_frame(rng.normal(0.0, 0.05, size=shape)))
    new_signal = alpha_new.add(_frame(rng.normal(0.0, 0.05, size=shape)))
    month_end = list(
        pd.Series(SESSIONS, index=SESSIONS)
        .groupby(SESSIONS.to_period("M"))
        .tail(1)
    )
    context = RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in SYMBOLS},
        adj_close=prices,
        volume=prices,
        amount=prices * 1.0e6,
        forward_return=forward,
        rebalance_dates=[pd.Timestamp(date) for date in month_end],
        biweekly_dates=list(SESSIONS[::10]),
        existing_composite=pool_signal if with_pool else None,
        exposure_metadata={"status": "ready"},
    )
    return context, pool_signal, new_signal


def _metrics(context: RetestContext, signal: pd.DataFrame) -> dict:
    return candidate_metrics(
        signal=signal,
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )


def test_a_clone_of_the_pool_has_no_incremental_alpha() -> None:
    context, pool_signal, _new = _base(with_pool=True)

    metrics = _metrics(context, pool_signal * 2.0 + 5.0)

    assert metrics["incremental_alpha_evidence_source"] == (
        POOL_RESIDUAL_EVIDENCE_SOURCE
    )
    # Standalone it looks like a real factor; against the pool it is nothing.
    assert metrics["mean_rankic"] > 0.05
    assert abs(metrics["pool_residual_mean_rankic"]) < 0.01
    assert metrics["pool_residual_retention"] < 0.2


def test_an_independent_factor_keeps_its_alpha_after_residualisation() -> None:
    context, _pool_signal, new_signal = _base(with_pool=True)

    metrics = _metrics(context, new_signal)

    assert metrics["mean_rankic"] > 0.05
    assert metrics["pool_residual_mean_rankic"] > 0.05
    assert metrics["pool_residual_retention"] > 0.5


def test_incremental_alpha_separates_a_clone_from_a_new_factor() -> None:
    context, pool_signal, new_signal = _base(with_pool=True)

    clone = _metrics(context, pool_signal * 2.0 + 5.0)
    fresh = _metrics(context, new_signal)

    # Standalone IC cannot tell them apart; the residual can.
    assert abs(clone["mean_rankic"] - fresh["mean_rankic"]) < 0.05
    assert fresh["pool_residual_mean_rankic"] > clone["pool_residual_mean_rankic"]
    assert fresh["pool_residual_retention"] > clone["pool_residual_retention"]


def test_incremental_metrics_report_when_there_is_no_pool() -> None:
    context, pool_signal, _new = _base(with_pool=False)

    metrics = _metrics(context, pool_signal)

    assert metrics["incremental_alpha_evidence_source"] == NO_POOL_EVIDENCE_SOURCE
    # With nothing to residualise against, the standalone IC is all there is.
    assert metrics["pool_residual_mean_rankic"] == pytest.approx(
        metrics["mean_rankic"]
    )


def test_incremental_metrics_carry_cpcv_path_evidence() -> None:
    context, pool_signal, new_signal = _base(with_pool=True)

    fresh = _metrics(context, new_signal)
    clone = _metrics(context, pool_signal * 2.0 + 5.0)

    assert fresh["pool_residual_oos_positive_ratio"] > 0.55
    assert clone["pool_residual_oos_positive_ratio"] < 0.55
