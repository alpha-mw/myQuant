"""Gate 7 evidence must come from purged, embargoed CPCV paths.

``oos_positive_ratio`` used to be pinned to 0.0 with a contiguous-fold value
kept alongside as a diagnostic.  Contiguous folds leak across the 30-session
label window, so the replacement runs combinatorial purged cross-validation over
the session calendar and reports the share of paths with positive test IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.retest_aquant_alpha_mix_8gate import (
    CPCV_EMBARGO_DAYS,
    CPCV_PURGE_DAYS,
    RetestContext,
    candidate_metrics,
)

SESSIONS = pd.bdate_range("2021-01-04", periods=900)
SYMBOLS = tuple(f"{index:06d}.SZ" for index in range(1, 31))


def _context() -> RetestContext:
    rng = np.random.default_rng(20260805)
    prices = pd.DataFrame(
        100.0
        + np.cumsum(
            rng.normal(0.0, 1.0, size=(len(SESSIONS), len(SYMBOLS))), axis=0
        ),
        index=SESSIONS,
        columns=list(SYMBOLS),
    )
    forward = prices.pct_change(fill_method=None).shift(-1).fillna(0.0)
    month_end = list(
        pd.Series(SESSIONS, index=SESSIONS)
        .groupby(SESSIONS.to_period("M"))
        .tail(1)
    )
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in SYMBOLS},
        adj_close=prices,
        volume=prices,
        amount=prices * 1.0e6,
        forward_return=forward,
        rebalance_dates=[pd.Timestamp(date) for date in month_end],
        biweekly_dates=list(SESSIONS[::10]),
        existing_composite=None,
        exposure_metadata={"status": "ready"},
    )


def _metrics(signal: pd.DataFrame) -> dict:
    return candidate_metrics(
        signal=signal,
        context=_context(),
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )


def _predictive_signal(context: RetestContext) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(
        rng.normal(0.0, 0.02, size=context.forward_return.shape),
        index=context.forward_return.index,
        columns=context.forward_return.columns,
    )
    return context.forward_return + noise


def test_gate7_reports_purged_embargoed_cpcv_paths() -> None:
    context = _context()
    metrics = candidate_metrics(
        signal=_predictive_signal(context),
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )

    assert metrics["walk_forward_evidence_source"] == "cpcv_purged_embargoed"
    assert metrics["walk_forward_purged"] is True
    assert metrics["walk_forward_purge_days"] == float(CPCV_PURGE_DAYS)
    assert metrics["walk_forward_embargo_days"] == float(CPCV_EMBARGO_DAYS)
    # C(10, 2) = 45 combinatorial paths.
    assert metrics["walk_forward_fold_count"] == 45.0
    assert len(metrics["walk_forward_evidence_hash"]) == 64


def test_gate7_positive_ratio_tracks_a_real_edge() -> None:
    context = _context()
    metrics = candidate_metrics(
        signal=_predictive_signal(context),
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )

    assert metrics["oos_positive_ratio"] > 0.55
    assert metrics["date_range_robustness"] is True
    assert metrics["cpcv_mean_path_ic"] > 0.0


def test_gate7_does_not_systematically_favour_noise() -> None:
    """CPCV paths are heavily overlapping, so one draw proves nothing.

    Each block appears in nine of the forty-five paths, so ``positive_path_ratio``
    is far from a mean of independent trials: a single noise draw can and does
    land well above 0.55.  What must hold is that noise has no systematic edge,
    and that a real signal separates from it.  This is exactly why the path
    ratio is not sufficient on its own for admission - see the deflated Sharpe
    stage in docs/factor_mining_mechanism.md.
    """

    context = _context()
    ratios: list[float] = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(
            rng.normal(0.0, 1.0, size=context.forward_return.shape),
            index=context.forward_return.index,
            columns=context.forward_return.columns,
        )
        metrics = candidate_metrics(
            signal=noise,
            context=context,
            decision_cost_bps=1.0,
            incremental_sleeve=0.03,
        )
        ratios.append(float(metrics["oos_positive_ratio"]))

    assert 0.3 <= float(np.mean(ratios)) <= 0.7
    predictive = candidate_metrics(
        signal=_predictive_signal(context),
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )
    assert predictive["oos_positive_ratio"] > max(ratios)


def test_gate7_fails_closed_on_a_calendar_too_short_for_cpcv() -> None:
    context = _context()
    short = RetestContext(
        frames={},
        universe_by_symbol=context.universe_by_symbol,
        adj_close=context.adj_close.iloc[:40],
        volume=context.volume.iloc[:40],
        amount=context.amount.iloc[:40],
        forward_return=context.forward_return.iloc[:40],
        rebalance_dates=list(context.adj_close.index[:40:10]),
        biweekly_dates=list(context.adj_close.index[:40:10]),
        existing_composite=None,
        exposure_metadata={"status": "ready"},
    )

    metrics = candidate_metrics(
        signal=_predictive_signal(context).iloc[:40],
        context=short,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )

    assert metrics["walk_forward_evidence_source"] == "cpcv_unavailable"
    assert metrics["walk_forward_purged"] is False
    assert metrics["oos_positive_ratio"] == 0.0
    assert metrics["walk_forward_evidence_hash"] == ""
