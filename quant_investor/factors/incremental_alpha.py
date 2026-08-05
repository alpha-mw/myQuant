"""Score a candidate on what it adds to the pool, not on what it holds alone.

The 2026-08-01 mining run made the case for this concretely: of 230 candidates,
exactly nine showed a positive portfolio increment and all nine sat at 0.81-1.00
correlation with factors already in production.  The miner had spent its whole
budget rediscovering `low_dollar_volume` and `amihud_illiquidity` at slightly
different windows, because every candidate was scored standalone and redundancy
was only checked at the end, as a rejection.

Projecting the existing pool out of a candidate before measuring its IC turns
redundancy into the thing the search optimises against rather than a late
veto - the set-level objective the alpha-mining literature settled on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

POOL_RESIDUAL_EVIDENCE_SOURCE = "cross_sectional_pool_residual"
NO_POOL_EVIDENCE_SOURCE = "no_production_pool"


def _ranked(row: pd.Series) -> pd.Series:
    return row.replace([np.inf, -np.inf], np.nan).rank(pct=True)


def residualize_against_pool(
    signal: pd.DataFrame,
    pool: pd.DataFrame | None,
    dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Return the part of ``signal`` the production pool does not explain.

    Both sides are ranked within each cross-section first, so the projection is
    scale-free and matches how RankIC reads the signal downstream.  Dates the
    pool does not cover fall back to a plain cross-sectional demean rather than
    being dropped: an uncovered date is not evidence of independence, but it is
    not evidence of redundancy either.
    """

    index = pd.DatetimeIndex(
        sorted({pd.Timestamp(date).normalize() for date in dates})
    )
    scoped = signal.reindex(index)
    residual = pd.DataFrame(
        np.nan,
        index=index,
        columns=scoped.columns,
        dtype=float,
    )
    pool_scoped = (
        pool.reindex(index=index, columns=scoped.columns)
        if pool is not None
        else None
    )
    for date in index:
        ranked = _ranked(scoped.loc[date])
        if ranked.notna().sum() < 2:
            continue
        centred = ranked.sub(ranked.mean())
        if pool_scoped is None:
            residual.loc[date] = centred
            continue
        pool_ranked = _ranked(pool_scoped.loc[date])
        common = centred.notna() & pool_ranked.notna()
        if common.sum() < 3:
            residual.loc[date] = centred
            continue
        pool_centred = pool_ranked[common].sub(pool_ranked[common].mean())
        variance = float(pool_centred.pow(2.0).sum())
        if variance <= 1e-12:
            # A pool with no cross-sectional dispersion explains nothing.
            residual.loc[date] = centred
            continue
        beta = float(centred[common].mul(pool_centred).sum() / variance)
        projected = centred.copy()
        projected[common] = centred[common].sub(pool_centred.mul(beta))
        residual.loc[date] = projected
    return residual
