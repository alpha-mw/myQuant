"""Correct a mining run's headline result for the fact that it picked a winner.

A run that evaluates 230 candidates and reports the best one is not running a
single test.  The maximum of many noisy Sharpe ratios is biased upward even when
every candidate is worthless, and the bias grows with the number of trials and
with how much the trial Sharpes vary.  Reporting the winner's uncorrected
statistic is how a factor zoo gets built.

Three corrections live here:

* the **deflated Sharpe ratio**, which subtracts the Sharpe you would expect the
  best of N trials to reach under the null and then accounts for skew, fat tails
  and sample length;
* the **probability of backtest overfitting**, which asks how often the
  in-sample winner lands in the bottom half out of sample;
* the **non-overlapping cohort t-statistic**, measured against the t > 3.0
  hurdle that Harvey, Liu and Zhu argue a newly discovered factor must clear
  once the size of the search is taken seriously.

In this codebase the "returns" series a Sharpe is taken over is the candidate's
per-rebalance RankIC series, so its Sharpe is exactly the ICIR already reported.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EULER_MASCHERONI = 0.577_215_664_901_532_9
# Bailey and Lopez de Prado treat a deflated Sharpe above 0.95 as evidence the
# result is not an artefact of the search.
DEFAULT_DSR_FLOOR = 0.95
# Harvey, Liu and Zhu (RFS 2016): with hundreds of factors already tested, the
# conventional t > 2.0 is not defensible for a newly discovered one.
HARVEY_LIU_ZHU_T_HURDLE = 3.0
# Above this, the in-sample winner is below the out-of-sample median more often
# than not, which makes the selection itself the finding.
DEFAULT_PBO_CEILING = 0.5
# The forward-return horizon in trading sessions, so cohorts spanning this much
# calendar do not share a label window.
DEFAULT_COHORT_SIZE = 30
# Matches the gate policy's ``max_existing_factor_corr``: above this, two
# candidates are not independent hypotheses about the cross-section.
DEFAULT_TRIAL_CLUSTER_FLOOR = 0.70
# Two IC series need at least this many shared observations before their
# correlation says anything about whether they are the same bet.
MIN_TRIAL_CLUSTER_OVERLAP = 12
TRIAL_CORRECTION_SCHEMA = "factor-trial-correction.v1"


def effective_trial_count(
    ic_series_by_name: Mapping[str, pd.Series],
    *,
    correlation_floor: float = DEFAULT_TRIAL_CLUSTER_FLOOR,
    min_overlap: int = MIN_TRIAL_CLUSTER_OVERLAP,
) -> dict[str, Any]:
    """Count independent bets rather than reparameterisations.

    Seventy smoothing variants of one idea explore one hypothesis, not seventy,
    and charging the deflated Sharpe for seventy makes the bar unreachable for
    reasons that have nothing to do with the data.  Candidates are clustered by
    the correlation of their IC series - the series already computed for the
    non-overlap test, so this costs no extra passes over the panel - and the
    cluster count is the effective trial count.

    Correlation is taken in absolute value: the sign of a factor is a
    convention, so a factor and its negation are the same bet.
    """

    names = [
        name
        for name, series in ic_series_by_name.items()
        if isinstance(series, pd.Series) and not series.dropna().empty
    ]
    empty = {
        "trial_count": len(ic_series_by_name),
        "effective_trial_count": 1,
        "cluster_count": 0,
        "correlation_floor": float(correlation_floor),
        "largest_cluster_size": 0,
    }
    if not names:
        return empty

    aligned = pd.DataFrame(
        {name: pd.Series(ic_series_by_name[name], dtype=float) for name in names}
    ).sort_index()

    parent = {name: name for name in names}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pair = aligned[[left, right]].dropna()
            if len(pair) < min_overlap:
                continue
            correlation = pair[left].corr(pair[right])
            if pd.isna(correlation):
                continue
            if abs(float(correlation)) >= float(correlation_floor):
                union(left, right)

    clusters: dict[str, int] = {}
    for name in names:
        root = find(name)
        clusters[root] = clusters.get(root, 0) + 1
    cluster_count = len(clusters)
    return {
        "trial_count": len(ic_series_by_name),
        "effective_trial_count": max(1, cluster_count),
        "cluster_count": cluster_count,
        "correlation_floor": float(correlation_floor),
        "largest_cluster_size": max(clusters.values()) if clusters else 0,
    }


def expected_max_sharpe_under_null(
    *,
    trial_sharpe_std: float,
    trial_count: int,
) -> float:
    """Sharpe the best of ``trial_count`` worthless trials is expected to reach.

    This is the selection bias the search introduces on its own: with no skill
    anywhere, the maximum of N draws still climbs with N and with how widely the
    draws are spread.
    """

    std = float(trial_sharpe_std)
    count = int(trial_count)
    if not math.isfinite(std) or std <= 0.0 or count <= 1:
        return 0.0
    upper = scipy_stats.norm.ppf(1.0 - 1.0 / count)
    lower = scipy_stats.norm.ppf(1.0 - 1.0 / (count * math.e))
    return float(
        std * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower)
    )


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    trial_sharpe_std: float,
    trial_count: int,
    sample_size: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability the observed Sharpe beats what the search alone would produce.

    ``kurtosis`` is the raw fourth moment, not the excess: a normal series is
    3.0.
    """

    sharpe = float(observed_sharpe)
    size = int(sample_size)
    if not math.isfinite(sharpe) or size < 2:
        return 0.0
    benchmark = expected_max_sharpe_under_null(
        trial_sharpe_std=trial_sharpe_std, trial_count=trial_count
    )
    skewness = float(skew) if math.isfinite(float(skew)) else 0.0
    kurt = float(kurtosis) if math.isfinite(float(kurtosis)) else 3.0
    variance = (
        1.0 - skewness * sharpe + ((kurt - 1.0) / 4.0) * sharpe * sharpe
    )
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0
    statistic = (sharpe - benchmark) * math.sqrt(size - 1) / math.sqrt(variance)
    if not math.isfinite(statistic):
        return 0.0
    return float(scipy_stats.norm.cdf(statistic))


def probability_of_backtest_overfitting(
    performance_by_block: pd.DataFrame,
) -> dict[str, Any]:
    """Run combinatorially symmetric cross-validation over the config set.

    ``performance_by_block`` holds one row per time block and one column per
    candidate configuration.  Every balanced split of the blocks into in-sample
    and out-of-sample halves nominates the in-sample winner and then asks where
    that winner ranks out of sample.  When the winner lands below the median
    more often than not, the selection is fitting noise.
    """

    frame = performance_by_block.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="any"
    )
    block_count, config_count = frame.shape
    empty = {
        "schema": TRIAL_CORRECTION_SCHEMA,
        "pbo": 1.0,
        "split_count": 0,
        "config_count": int(config_count),
        "block_count": int(block_count),
        "median_oos_rank": 0.0,
    }
    if config_count < 2 or block_count < 4 or block_count % 2:
        return empty

    values = frame.to_numpy(dtype=float)
    half = block_count // 2
    logits: list[float] = []
    oos_ranks: list[float] = []
    for in_sample in combinations(range(block_count), half):
        out_of_sample = [
            index for index in range(block_count) if index not in in_sample
        ]
        in_scores = values[list(in_sample), :].mean(axis=0)
        out_scores = values[out_of_sample, :].mean(axis=0)
        winner = int(np.argmax(in_scores))
        # Fractional rank of the winner out of sample, in (0, 1).
        rank = float(
            (out_scores < out_scores[winner]).sum()
            + 0.5 * (out_scores == out_scores[winner]).sum()
        ) / config_count
        rank = min(max(rank, 1.0 / (config_count + 1)), 1.0 - 1.0 / (config_count + 1))
        oos_ranks.append(rank)
        logits.append(math.log(rank / (1.0 - rank)))
    if not logits:
        return empty
    return {
        "schema": TRIAL_CORRECTION_SCHEMA,
        "pbo": float(np.mean([value <= 0.0 for value in logits])),
        "split_count": len(logits),
        "config_count": int(config_count),
        "block_count": int(block_count),
        "median_oos_rank": float(np.median(oos_ranks)),
    }


def infer_cohort_size(
    index: pd.DatetimeIndex,
    *,
    horizon_sessions: int = DEFAULT_COHORT_SIZE,
) -> int:
    """How many observations of ``index`` span one forward-return horizon.

    The horizon is counted in trading sessions but an IC series is sampled at
    whatever cadence the rebalance schedule uses.  A daily series needs about 30
    observations to span a 30-session horizon; a month-end series needs 2.
    Assuming 30 for both silently failed closed on every monthly series, which
    is the cadence the miner actually produces.
    """

    stamps = pd.DatetimeIndex(index).dropna().sort_values()
    if len(stamps) < 2:
        return 1
    # Gaps are counted in business days, the same unit the horizon is in.  A
    # calendar-day count would read a daily series' median gap as 1 and inflate
    # the cohort by the weekend ratio.
    days = stamps.values.astype("datetime64[D]")
    gaps = np.busday_count(days[:-1], days[1:]).astype(float)
    gaps = gaps[gaps > 0.0]
    if not len(gaps):
        return 1
    median_gap = float(np.median(gaps))
    if median_gap <= 0.0:
        return 1
    return max(1, int(math.ceil(float(horizon_sessions) / median_gap)))


def nonoverlap_t_statistic(
    ic_by_date: pd.Series,
    *,
    cohort_size: int | None = None,
) -> tuple[float, float, int]:
    """t-test on cohort means that genuinely do not share a forward window.

    RankIC against a 30-session forward return overlaps its neighbours, so an
    iid t-test on the raw series overstates significance.  Averaging inside
    disjoint cohorts of exactly one horizon removes the shared window instead of
    modelling it.  The cohort width defaults to whatever spans one horizon at
    this series' own sampling cadence.
    """

    series = pd.Series(ic_by_date, dtype=float).dropna().sort_index()
    size = (
        max(int(cohort_size), 1)
        if cohort_size is not None
        else infer_cohort_size(pd.DatetimeIndex(series.index))
    )
    if len(series) < 2 * size:
        return 0.0, 1.0, len(series) // size
    cohort_means = [
        float(series.iloc[start : start + size].mean())
        for start in range(0, len(series) - size + 1, size)
    ]
    cohort_means = [value for value in cohort_means if math.isfinite(value)]
    if len(cohort_means) < 2:
        return 0.0, 1.0, len(cohort_means)
    t_stat, p_value = scipy_stats.ttest_1samp(cohort_means, 0.0)
    if not math.isfinite(float(t_stat)):
        return 0.0, 1.0, len(cohort_means)
    return float(t_stat), float(p_value), len(cohort_means)
