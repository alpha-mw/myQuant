"""Gate 6 must read real neutralization evidence once exposure is governed.

The metric was pinned to ``neutralized_icir=0.0`` / ``style_exposure_only=True``
while the exposure maps were unavailable.  Now that the governed generation
supplies point-in-time sector and size buckets, the neutralized ICIR is real
evidence - but only when the exposure behind it says it is ready.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.retest_aquant_alpha_mix_8gate import RetestContext, candidate_metrics

DATES = pd.bdate_range("2024-01-01", periods=60)
# _rank_ic_series needs at least 20 names in the cross-section.
SYMBOLS = tuple(f"{index:06d}.SZ" for index in range(1, 31))


def _context(*, exposure_status: str) -> RetestContext:
    rng = np.random.default_rng(20260805)
    prices = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(0.0, 1.0, size=(len(DATES), len(SYMBOLS))), axis=0),
        index=DATES,
        columns=list(SYMBOLS),
    )
    forward = prices.pct_change(fill_method=None).shift(-1).fillna(0.0)
    sectors = {
        symbol: ("bank" if index % 2 else "tech")
        for index, symbol in enumerate(SYMBOLS)
    }
    sizes = {
        symbol: ("small", "mid", "large")[index % 3]
        for index, symbol in enumerate(SYMBOLS)
    }
    size_by_date = pd.DataFrame(
        [[sizes[symbol] for symbol in SYMBOLS]] * len(DATES),
        index=DATES,
        columns=list(SYMBOLS),
    )
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in SYMBOLS},
        adj_close=prices,
        volume=prices,
        amount=prices * 1.0e6,
        forward_return=forward,
        rebalance_dates=list(DATES[::5]),
        biweekly_dates=list(DATES[::10]),
        existing_composite=None,
        sector_by_symbol=sectors,
        size_bucket_by_symbol=sizes,
        size_bucket_by_date=size_by_date,
        exposure_metadata={"status": exposure_status},
    )


def _signal(context: RetestContext) -> pd.DataFrame:
    # A signal that survives sector/size demeaning: its edge is spread across
    # the cross-section rather than carried by a bucket constant, so removing
    # the bucket means barely moves the ranking.
    rng = np.random.default_rng(4242)
    noise = pd.DataFrame(
        rng.normal(0.0, 0.02, size=context.forward_return.shape),
        index=context.forward_return.index,
        columns=context.forward_return.columns,
    )
    return context.forward_return + noise


def _metrics(exposure_status: str) -> dict:
    context = _context(exposure_status=exposure_status)
    return candidate_metrics(
        signal=_signal(context),
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )


def test_gate6_uses_real_neutralized_icir_when_exposure_is_ready() -> None:
    metrics = _metrics("ready")

    assert metrics["neutralized_icir"] != 0.0
    assert metrics["neutralized_icir"] == metrics[
        "diagnostic_universe_neutralized_icir"
    ]
    assert metrics["neutralization_evidence_source"] == (
        "governed_exposure_sector_size_demean"
    )


def test_gate6_fails_closed_when_exposure_is_not_ready() -> None:
    metrics = _metrics("blocked")

    assert metrics["neutralized_icir"] == 0.0
    assert metrics["style_exposure_only"] is True
    assert metrics["neutralization_evidence_source"] == "exposure_not_ready"


def test_style_exposure_flag_survives_a_factor_that_is_not_pure_style() -> None:
    metrics = _metrics("ready")

    assert metrics["style_exposure_only"] is False


def test_style_exposure_flag_catches_a_pure_bucket_bet() -> None:
    context = _context(exposure_status="ready")
    # Every symbol takes its sector's constant, so demeaning within
    # sector x size buckets leaves exactly nothing behind.
    bucket_signal = pd.DataFrame(
        [
            [
                1.0 if context.sector_by_symbol[symbol] == "bank" else -1.0
                for symbol in SYMBOLS
            ]
        ]
        * len(DATES),
        index=DATES,
        columns=list(SYMBOLS),
    )

    metrics = candidate_metrics(
        signal=bucket_signal,
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )

    assert metrics["style_exposure_only"] is True
