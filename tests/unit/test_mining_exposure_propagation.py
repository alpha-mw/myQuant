"""Exposure maps must survive every context narrowing the miner performs.

The miner attaches sector/size exposure to the analysis-window context, then
re-derives a narrower context per candidate from its maturity start.  When that
narrowing dropped the exposure fields, every candidate was scored as if no
exposure existed: Gate 2 saw a single sector holding 100% of the cross-section
and Gate 6 saw a neutralized ICIR of zero, no matter how good the exposure
evidence was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.mine_quant_branch_factors import (
    _restrict_context_from_start,
    candidate_maturity_context,
)
from scripts.retest_aquant_alpha_mix_8gate import RetestContext

SESSIONS = pd.bdate_range("2024-01-01", periods=120)
SYMBOLS = tuple(f"{index:06d}.SZ" for index in range(1, 25))


def _context() -> RetestContext:
    prices = pd.DataFrame(
        100.0,
        index=SESSIONS,
        columns=list(SYMBOLS),
    )
    sizes = {
        symbol: ("small", "mid", "large")[index % 3]
        for index, symbol in enumerate(SYMBOLS)
    }
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in SYMBOLS},
        adj_close=prices,
        volume=prices,
        amount=prices,
        forward_return=prices.pct_change(fill_method=None).fillna(0.0),
        rebalance_dates=list(SESSIONS[::20]),
        biweekly_dates=list(SESSIONS[::10]),
        existing_composite=None,
        sector_by_symbol={
            symbol: ("bank" if index % 2 else "tech")
            for index, symbol in enumerate(SYMBOLS)
        },
        size_bucket_by_symbol=sizes,
        size_bucket_by_date=pd.DataFrame(
            [[sizes[symbol] for symbol in SYMBOLS]] * len(SESSIONS),
            index=SESSIONS,
            columns=list(SYMBOLS),
        ),
        exposure_metadata={"status": "ready", "source": "test"},
    )


def test_restricting_a_context_keeps_the_exposure_maps() -> None:
    start = SESSIONS[60]

    restricted = _restrict_context_from_start(_context(), start=start)

    assert restricted.sector_by_symbol == _context().sector_by_symbol
    assert restricted.size_bucket_by_symbol == _context().size_bucket_by_symbol
    assert restricted.exposure_metadata["status"] == "ready"
    assert not restricted.size_bucket_by_date.empty


def test_restricting_a_context_narrows_the_dynamic_size_buckets() -> None:
    start = SESSIONS[60]

    restricted = _restrict_context_from_start(_context(), start=start)

    assert restricted.size_bucket_by_date.index.min() >= start
    assert list(restricted.size_bucket_by_date.index) == list(
        restricted.adj_close.index
    )


def test_maturity_context_keeps_the_exposure_maps() -> None:
    context = _context()
    signal = pd.DataFrame(
        np.arange(len(SESSIONS) * len(SYMBOLS), dtype=float).reshape(
            len(SESSIONS), len(SYMBOLS)
        ),
        index=SESSIONS,
        columns=list(SYMBOLS),
    )

    matured, _start = candidate_maturity_context(
        context,
        signal,
        base_start=SESSIONS[20].strftime("%Y-%m-%d"),
        min_signal_coverage=0.6,
    )

    assert matured.exposure_metadata["status"] == "ready"
    assert matured.sector_by_symbol
    assert not matured.size_bucket_by_date.empty
