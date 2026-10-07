"""Research replay of Factor lifecycle rules on historical IC series.

Everything here is research evidence for choosing lifecycle parameters; it
grants no admission, weight or activation.  Two questions are answered:

1. Which estimation half-life best predicts next-period IC?  This is the
   data-driven answer to "not too short, not too long".
2. Does a lifecycle rule (probation, watch, retire, capped quarterly
   reweighting) beat a static equal-weight set out of sample, after charging
   for the number of rule variants tried?

All decisions at period ``t`` use only ICs realized before ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .governance.statistics import deflated_sharpe_ratio

PREREGISTERED: Final = "PREREGISTERED"
PROBATION: Final = "PROBATION"
ACTIVE: Final = "ACTIVE"
WATCH: Final = "WATCH"
RETIRED: Final = "RETIRED"
STATE_WEIGHT_SCALE: Final = {PREREGISTERED: 0.0, PROBATION: None, ACTIVE: 1.0, WATCH: 0.5}
MIN_CROSS_SECTION: Final = 30


class LifecycleBacktestError(ValueError):
    """Raised when research input cannot support the requested replay."""


def forward_label(adj_close: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Return ``close[t+1+h] / close[t+1] - 1``: enter at the next close, hold ``h`` sessions."""

    if int(horizon) < 1:
        raise LifecycleBacktestError("horizon must be positive")
    close = adj_close.where(adj_close > 0.0)
    return close.shift(-(int(horizon) + 1)) / close.shift(-1) - 1.0


def month_end_origins(dates: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    """Last trading session of every calendar month present in ``dates``."""

    index = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates))))
    if index.empty:
        return []
    frame = pd.Series(index, index=index)
    return list(frame.groupby(index.to_period("M")).max())


def cross_sectional_rank_ic(signal: pd.Series, label: pd.Series) -> float:
    joined = pd.concat([signal, label], axis=1, join="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < MIN_CROSS_SECTION:
        return math.nan
    value = joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank())
    return float(value) if math.isfinite(value) else math.nan


def ic_panel(
    signals: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """``origin x factor`` RankIC; a factor is NaN where its cross-section is too thin."""

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin in origins:
        if origin not in labels.index:
            continue
        label = labels.loc[origin]
        rows[origin] = {
            name: cross_sectional_rank_ic(panel.loc[origin], label)
            for name, panel in signals.items()
            if origin in panel.index
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _ew_weights(count: int, half_life: float) -> np.ndarray:
    if math.isinf(half_life):
        return np.ones(count)
    ages = np.arange(count - 1, -1, -1, dtype=float)
    return np.power(0.5, ages / float(half_life))


def ew_estimate(values: np.ndarray, half_life: float) -> tuple[float, float, float]:
    """EW mean, EW standard deviation and Kish effective sample size of finite ``values``."""

    data = values[np.isfinite(values)]
    if data.size == 0:
        return math.nan, math.nan, 0.0
    weights = _ew_weights(data.size, half_life)
    mean = float(np.dot(weights, data) / weights.sum())
    effective = float(weights.sum() ** 2 / np.square(weights).sum())
    if data.size < 2:
        return mean, math.nan, effective
    variance = float(np.dot(weights, np.square(data - mean)) / weights.sum())
    correction = effective / (effective - 1.0) if effective > 1.0 else math.nan
    return mean, math.sqrt(max(variance * correction, 0.0)), effective


def select_half_life(
    ic: pd.DataFrame,
    *,
    half_lives: Sequence[float],
    min_history: int = 12,
) -> dict[str, Any]:
    """Score each half-life by one-step-ahead prediction of every factor's IC.

    For every period ``t`` and factor with at least ``min_history`` earlier
    observations, the prediction is the EW mean of those earlier ICs.  The
    score is the pooled mean squared error; the cross-sectional rank
    correlation between predictions and realizations is reported as a second
    view.  ``inf`` is the expanding mean; a half-life near 1 is "last value".
    """

    if ic.empty or not half_lives:
        raise LifecycleBacktestError("half-life selection needs IC history and candidates")
    values = ic.to_numpy(dtype=float)
    results: list[dict[str, Any]] = []
    for half_life in half_lives:
        squared: list[float] = []
        by_period_errors: list[tuple[int, float]] = []
        rank_corrs: list[float] = []
        for t in range(values.shape[0]):
            predictions = []
            realized = []
            for j in range(values.shape[1]):
                history = values[:t, j]
                if np.isfinite(history).sum() < min_history or not math.isfinite(values[t, j]):
                    continue
                mean, _sd, _n = ew_estimate(history, half_life)
                predictions.append(mean)
                realized.append(values[t, j])
                error = (mean - values[t, j]) ** 2
                squared.append(error)
                by_period_errors.append((t, error))
            if len(predictions) >= 3:
                corr = scipy_stats.spearmanr(predictions, realized).statistic
                if math.isfinite(corr):
                    rank_corrs.append(float(corr))
        if not squared:
            raise LifecycleBacktestError("no period has enough history to score a half-life")
        half = values.shape[0] // 2
        first = [e for t, e in by_period_errors if t < half]
        second = [e for t, e in by_period_errors if t >= half]
        results.append(
            {
                "half_life_periods": "inf" if math.isinf(half_life) else float(half_life),
                "mse": float(np.mean(squared)),
                "mse_first_half": float(np.mean(first)) if first else None,
                "mse_second_half": float(np.mean(second)) if second else None,
                "mean_cross_sectional_rank_corr": (
                    float(np.mean(rank_corrs)) if rank_corrs else None
                ),
                "prediction_count": len(squared),
            }
        )
    best = min(results, key=lambda row: row["mse"])
    halves = {}
    for label in ("mse_first_half", "mse_second_half"):
        scored = [row for row in results if row[label] is not None]
        if scored:
            halves[label.replace("mse_", "best_")] = min(scored, key=lambda row: row[label])[
                "half_life_periods"
            ]
    return {"results": results, "best_half_life_periods": best["half_life_periods"], **halves}


@dataclass(frozen=True)
class LifecycleRule:
    """One lifecycle rule; every field is a parameter charged as a trial."""

    half_life_periods: float = 6.0
    probation_periods: int = 6
    full_periods: int = 18
    probation_p: float = 0.80
    watch_p: float = 0.70
    retire_p: float = 0.50
    retire_periods: int = 3
    probation_cap: float = 0.5
    rebalance_every: int = 3
    max_step: float = 0.5
    prior_mean: float = 0.0
    prior_sd: float = 0.05
    observation_sd_floor: float = 0.01
    reentry_cooldown: int | None = 12
    max_factor_weight: float | None = None

    def __post_init__(self) -> None:
        if not (
            self.half_life_periods > 0
            and self.observation_sd_floor > 0
            and (self.max_factor_weight is None or 0.0 < self.max_factor_weight <= 1.0)
            and (self.reentry_cooldown is None or self.reentry_cooldown >= 1)
            and 1 <= self.probation_periods <= self.full_periods
            and 0.0 < self.retire_p < self.watch_p <= self.probation_p < 1.0
            and self.retire_periods >= 1
            and 0.0 < self.probation_cap <= 1.0
            and self.rebalance_every >= 1
            and 0.0 < self.max_step <= 2.0
            and self.prior_sd > 0
        ):
            raise LifecycleBacktestError("lifecycle rule is invalid")


def posterior_p_positive(
    history: np.ndarray,
    *,
    half_life: float,
    prior_mean: float,
    prior_sd: float,
    observation_sd_floor: float = 0.01,
) -> tuple[float, float, int]:
    """Posterior mean and ``P(IC > 0)`` from EW evidence with Kish effective size.

    The observation standard deviation is floored so a short or unusually
    smooth history cannot claim certainty.
    """

    count = int(np.isfinite(history).sum())
    mean, sd, effective = ew_estimate(history, half_life)
    prior_precision = 1.0 / prior_sd**2
    if count < 2 or not math.isfinite(sd):
        return prior_mean, float(scipy_stats.norm.sf(0.0, prior_mean, prior_sd)), count
    sd = max(sd, float(observation_sd_floor))
    data_precision = effective / sd**2
    precision = prior_precision + data_precision
    posterior_mean = (prior_precision * prior_mean + data_precision * mean) / precision
    posterior_sd = math.sqrt(1.0 / precision)
    return (
        float(posterior_mean),
        float(scipy_stats.norm.sf(0.0, posterior_mean, posterior_sd)),
        count,
    )


def _next_state(state: str, *, count: int, p: float, below_retire: int, rule: LifecycleRule) -> str:
    if state == RETIRED:
        return RETIRED
    if state == PREREGISTERED:
        return PROBATION if count >= rule.probation_periods and p >= rule.probation_p else state
    if state == PROBATION:
        if p < rule.retire_p and below_retire >= rule.retire_periods:
            return RETIRED
        if count >= rule.full_periods and p >= rule.probation_p:
            return ACTIVE
        return state
    if state == ACTIVE:
        return WATCH if p < rule.watch_p else state
    if state == WATCH:
        if below_retire >= rule.retire_periods:
            return RETIRED
        return ACTIVE if p >= rule.probation_p else state
    raise LifecycleBacktestError(f"unknown lifecycle state {state}")


def cap_weights(weights: np.ndarray, cap: float | None) -> np.ndarray:
    """Cap each weight at ``cap`` and redistribute the excess pro rata to uncapped names.

    When too few names are positive to reach full investment under the cap,
    the remainder stays uninvested rather than breaching the cap.
    """

    result = np.asarray(weights, dtype=float).copy()
    if cap is None or result.sum() <= 0:
        return result
    for _ in range(result.size):
        over = result > cap + 1e-12
        if not over.any():
            break
        excess = float((result[over] - cap).sum())
        result[over] = cap
        room = (result > 0) & (result < cap - 1e-12)
        if not room.any():
            break
        result[room] += excess * result[room] / result[room].sum()
    return result


def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - current
    distance = float(np.abs(delta).sum())
    if distance <= max_step or distance == 0.0:
        return target.copy()
    return current + delta * (max_step / distance)


def simulate_lifecycle(ic: pd.DataFrame, rule: LifecycleRule) -> dict[str, Any]:
    """Replay ``rule`` period by period; decisions at ``t`` see ICs before ``t`` only.

    A retired factor keeps being observed.  When ``reentry_cooldown`` is set it
    returns to ``PREREGISTERED`` after that many periods with its evidence
    window restarted, mirroring "re-entry only through a new preregistration".
    """

    if ic.empty:
        raise LifecycleBacktestError("lifecycle simulation needs an IC panel")
    values = ic.to_numpy(dtype=float)
    periods, factors = values.shape
    states = [PREREGISTERED] * factors
    below = [0] * factors
    evidence_start = [0] * factors
    retired_at: list[int | None] = [None] * factors
    weights = np.zeros(factors)
    portfolio: list[float] = []
    turnover = 0.0
    history_states: list[dict[str, int]] = []
    retirements: list[dict[str, Any]] = []
    reentries: list[dict[str, Any]] = []
    for t in range(periods):
        scores = np.zeros(factors)
        for j in range(factors):
            retired_period = retired_at[j]
            if (
                states[j] == RETIRED
                and rule.reentry_cooldown is not None
                and retired_period is not None
                and t - retired_period >= rule.reentry_cooldown
            ):
                states[j] = PREREGISTERED
                evidence_start[j] = t
                below[j] = 0
                retired_at[j] = None
                reentries.append({"factor": str(ic.columns[j]), "period": str(ic.index[t])})
            if states[j] == RETIRED or not np.isfinite(values[evidence_start[j] : t, j]).any():
                continue
            mean, p, count = posterior_p_positive(
                values[evidence_start[j] : t, j],
                half_life=rule.half_life_periods,
                prior_mean=rule.prior_mean,
                prior_sd=rule.prior_sd,
                observation_sd_floor=rule.observation_sd_floor,
            )
            below[j] = below[j] + 1 if p < rule.retire_p else 0
            new_state = _next_state(states[j], count=count, p=p, below_retire=below[j], rule=rule)
            if new_state == RETIRED and states[j] != RETIRED:
                retirements.append({"factor": str(ic.columns[j]), "period": str(ic.index[t])})
                retired_at[j] = t
            states[j] = new_state
            scale = STATE_WEIGHT_SCALE.get(new_state, 0.0)
            if new_state == PROBATION:
                scale = rule.probation_cap
            scores[j] = max(mean, 0.0) * float(scale or 0.0)
        if t % rule.rebalance_every == 0:
            total = scores.sum()
            target = scores / total if total > 0 else np.zeros(factors)
            target = cap_weights(target, rule.max_factor_weight)
            updated = _step_toward(weights, target, rule.max_step)
            turnover += float(np.abs(updated - weights).sum())
            weights = updated
        realized = np.where(np.isfinite(values[t]), values[t], 0.0)
        invested = weights[np.isfinite(values[t])].sum()
        portfolio.append(float(np.dot(weights, realized) / invested) if invested > 0 else math.nan)
        history_states.append({state: states.count(state) for state in set(states)})
    series = pd.Series(portfolio, index=ic.index, dtype=float)
    return {
        "rule": asdict(rule),
        "portfolio_ic": series,
        "turnover": turnover,
        "final_states": dict(zip(map(str, ic.columns), states)),
        "retirements": retirements,
        "reentries": reentries,
        "state_counts": history_states,
    }


def static_equal_weight(ic: pd.DataFrame, *, warmup_periods: int) -> pd.Series:
    """Equal weight over every factor with an IC, from ``warmup_periods`` onward."""

    series = ic.mean(axis=1, skipna=True)
    series.iloc[: int(warmup_periods)] = math.nan
    return series


def expanding_positive_weighted(
    ic: pd.DataFrame,
    *,
    warmup_periods: int,
    rebalance_every: int = 3,
    max_factor_weight: float | None = None,
) -> pd.Series:
    """Fair static baseline: weight by ``max(expanding mean IC, 0)``, no states.

    It knows the sign of each factor only from its own past, rebalances on the
    same cadence as the rules and never retires anything, so a rule must beat
    it by *timing*, not by excluding factors that were negative all along.
    """

    values = ic.to_numpy(dtype=float)
    weights = np.zeros(values.shape[1])
    portfolio = []
    for t in range(values.shape[0]):
        if t % int(rebalance_every) == 0 and t > 0:
            history = values[:t]
            counts = np.isfinite(history).sum(axis=0)
            means = np.where(counts > 0, np.nansum(history, axis=0) / np.maximum(counts, 1), 0.0)
            scores = np.maximum(means, 0.0)
            weights = scores / scores.sum() if scores.sum() > 0 else np.zeros_like(scores)
            weights = cap_weights(weights, max_factor_weight)
        finite = np.isfinite(values[t])
        invested = weights[finite].sum()
        portfolio.append(
            float(np.dot(weights[finite], values[t][finite]) / invested)
            if invested > 0
            else math.nan
        )
    series = pd.Series(portfolio, index=ic.index, dtype=float)
    series.iloc[: int(warmup_periods)] = math.nan
    return series


def summarize_ic_series(series: pd.Series) -> dict[str, Any]:
    data = pd.Series(series, dtype=float).dropna()
    if len(data) < 3:
        return {"count": int(len(data)), "mean": None, "ir": None}
    sd = float(data.std(ddof=1))
    cumulative = data.cumsum()
    drawdown = float((cumulative - cumulative.cummax()).min())
    rolling = data.rolling(12).sum().dropna()
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "sd": sd,
        "ir": float(data.mean() / sd) if sd > 0 else None,
        "t": float(data.mean() / sd * math.sqrt(len(data))) if sd > 0 else None,
        "hit_rate": float((data > 0).mean()),
        "max_drawdown_cumulative_ic": drawdown,
        "worst_12_period_sum": float(rolling.min()) if not rolling.empty else None,
        "skew": float(scipy_stats.skew(data)),
        "kurtosis": float(scipy_stats.kurtosis(data, fisher=False)),
    }


def rule_grid(**overrides: Sequence[Any]) -> list[LifecycleRule]:
    """Cartesian grid of rules; the grid size is the trial count charged by the DSR."""

    axes: dict[str, Sequence[Any]] = {
        "half_life_periods": (3.0, 6.0, 12.0, 24.0),
        "probation_periods": (6, 12),
        "watch_p": (0.70, 0.80),
        "retire_p": (0.40, 0.50),
        "reentry_cooldown": (None, 12),
    }
    axes.update(overrides)
    names = sorted(axes)
    rules = []
    for combination in itertools.product(*(axes[name] for name in names)):
        params = dict(zip(names, combination))
        params.setdefault("probation_p", max(0.80, float(params.get("watch_p", 0.7))))
        try:
            rules.append(LifecycleRule(**params))
        except LifecycleBacktestError:
            continue
    return rules


def deflated_best_rule(
    summaries: Sequence[Mapping[str, Any]], *, trial_count: int | None = None
) -> dict[str, Any]:
    """DSR of the best rule's IR, charging every rule tried."""

    usable = [row for row in summaries if row.get("ir") is not None]
    if len(usable) < 2:
        raise LifecycleBacktestError("deflation needs at least two scored rules")
    best = max(usable, key=lambda row: row["ir"])
    irs = [row["ir"] for row in usable]
    count = int(trial_count or len(usable))
    dsr = deflated_sharpe_ratio(
        observed_sharpe=float(best["ir"]),
        trial_sharpe_std=float(np.std(irs, ddof=1)),
        trial_count=count,
        sample_size=float(best["count"]),
        skew=float(best.get("skew") or 0.0),
        kurtosis=float(best.get("kurtosis") or 3.0),
    )
    return {"best_ir": float(best["ir"]), "trial_count": count, "dsr": dsr}


def regime_attribution(
    ic: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    events: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Correlate each factor's IC with regime series and report event-window ICs."""

    aligned = regimes.reindex(ic.index)
    correlations: dict[str, dict[str, float | None]] = {}
    for factor in ic.columns:
        correlations[str(factor)] = {}
        for regime in aligned.columns:
            pair = pd.concat([ic[factor], aligned[regime]], axis=1).dropna()
            value = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 12 else math.nan
            correlations[str(factor)][str(regime)] = float(value) if math.isfinite(value) else None
    windows: dict[str, dict[str, float | None]] = {}
    for name, (start, end) in events.items():
        scoped = ic.loc[(ic.index >= pd.Timestamp(start)) & (ic.index <= pd.Timestamp(end))]
        windows[name] = {
            str(factor): (float(scoped[factor].mean()) if scoped[factor].notna().any() else None)
            for factor in ic.columns
        }
    return {"regime_correlations": correlations, "event_window_mean_ic": windows}


__all__ = [
    "ACTIVE",
    "cap_weights",
    "LifecycleBacktestError",
    "LifecycleRule",
    "PREREGISTERED",
    "PROBATION",
    "RETIRED",
    "WATCH",
    "cross_sectional_rank_ic",
    "deflated_best_rule",
    "ew_estimate",
    "expanding_positive_weighted",
    "forward_label",
    "ic_panel",
    "month_end_origins",
    "posterior_p_positive",
    "regime_attribution",
    "rule_grid",
    "select_half_life",
    "simulate_lifecycle",
    "static_equal_weight",
    "summarize_ic_series",
]
