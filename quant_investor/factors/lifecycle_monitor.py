"""Non-authorizing decay monitoring for production and candidate Factor ICs.

The monitor answers one question per factor and horizon: is there evidence
that the effect has decayed?  It never changes a weight, pointer, admission or
activation.  Its statistics are deliberately split by what each can and cannot
say:

* the exponentially weighted RankIC is a *current* level estimate, and its
  half-life is the knob that trades reaction speed against noise;
* the non-overlapping t-statistic and the posterior ``P(IC > 0)`` are computed
  on disjoint cohort means, so a horizon-``h`` label never shares its window
  with a neighbour;
* the one-sided CUSUM accumulates evidence that cohort ICs sit below a
  reference level and raises an alarm without waiting for a fixed window.

Outcomes whose RankIC is unavailable are counted as source gaps, never as a
zero or as alpha failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .governance.statistics import nonoverlap_t_statistic
from .incremental_alpha import residualize_against_pool

MONITOR_SCHEMA: Final = "factor-lifecycle-monitor.v1"
MONITOR_AUTHORITY: Final = "NON_AUTHORIZING"

INSUFFICIENT_EVIDENCE: Final = "INSUFFICIENT_EVIDENCE"
SOURCE_BLOCKED: Final = "SOURCE_BLOCKED"
HEALTHY: Final = "HEALTHY"
WATCH: Final = "WATCH"
ALARM: Final = "ALARM"


class LifecycleMonitorError(ValueError):
    """Raised when monitor input cannot be read exactly."""


@dataclass(frozen=True)
class MonitorPolicy:
    """Thresholds of the monitor; none of them is an admission gate."""

    half_life_sessions: float = 120.0
    min_cohorts: int = 8
    min_origin_sessions: int = 60
    cusum_slack: float = 0.5
    cusum_threshold: float = 4.0
    watch_posterior: float = 0.70
    alarm_posterior: float = 0.50
    prior_mean: float = 0.0
    prior_sd: float = 0.05
    observation_sd_floor: float = 0.01

    def __post_init__(self) -> None:
        if not (
            self.half_life_sessions > 0
            and self.min_cohorts >= 2
            and self.min_origin_sessions >= 1
            and self.cusum_slack >= 0
            and self.cusum_threshold > 0
            and 0.0 < self.alarm_posterior < self.watch_posterior < 1.0
            and self.prior_sd > 0
            and self.observation_sd_floor > 0
        ):
            raise LifecycleMonitorError("monitor policy is invalid")


def _clean(values: pd.Series) -> pd.Series:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return series.sort_index()


def exponentially_weighted_mean(values: pd.Series, *, half_life: float) -> dict[str, Any]:
    """Weighted mean with weight ``0.5 ** (age / half_life)``; age 0 is the newest value."""

    if not half_life > 0:
        raise LifecycleMonitorError("half_life must be positive")
    series = _clean(values)
    if series.empty:
        return {"mean": None, "effective_n": 0.0, "count": 0}
    ages = np.arange(len(series) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / float(half_life))
    mean = float(np.dot(weights, series.to_numpy()) / weights.sum())
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    return {"mean": mean, "effective_n": effective_n, "count": int(len(series))}


def cohort_means(values: pd.Series, *, cohort_size: int) -> pd.Series:
    """Means of disjoint, consecutive cohorts; a trailing partial cohort is dropped."""

    size = int(cohort_size)
    if size < 1:
        raise LifecycleMonitorError("cohort_size must be positive")
    series = _clean(values)
    full = len(series) // size
    if full == 0:
        return pd.Series(dtype=float)
    trimmed = series.iloc[: full * size]
    means = trimmed.to_numpy().reshape(full, size).mean(axis=1)
    index = [trimmed.index[(block + 1) * size - 1] for block in range(full)]
    return pd.Series(means, index=index, dtype=float)


def lower_cusum(
    values: Sequence[float],
    *,
    reference: float,
    scale: float,
    slack: float,
    threshold: float,
) -> dict[str, Any]:
    """One-sided CUSUM for a downward shift below ``reference``.

    ``S_t = max(0, S_{t-1} + (reference - x_t) / scale - slack)``; the alarm
    fires the first time ``S_t`` exceeds ``threshold``.  With standardized iid
    normal input, slack 0.5 and threshold 4 give an in-control run length of
    roughly 170 observations and detect a one-sigma drop in about 9.
    """

    if not (math.isfinite(reference) and scale > 0 and slack >= 0 and threshold > 0):
        raise LifecycleMonitorError("CUSUM parameters are invalid")
    statistic = 0.0
    peak = 0.0
    first_alarm: int | None = None
    path: list[float] = []
    for index, value in enumerate(float(item) for item in values):
        if not math.isfinite(value):
            raise LifecycleMonitorError("CUSUM input must be finite")
        statistic = max(0.0, statistic + (reference - value) / scale - slack)
        peak = max(peak, statistic)
        path.append(statistic)
        if first_alarm is None and statistic > threshold:
            first_alarm = index
    return {
        "statistic": statistic,
        "max_statistic": peak,
        "alarm": statistic > threshold,
        "ever_alarmed": first_alarm is not None,
        "first_alarm_index": first_alarm,
        "path": path,
    }


def normal_posterior(
    observations: Sequence[float],
    *,
    prior_mean: float,
    prior_sd: float,
    observation_sd_floor: float,
) -> dict[str, Any]:
    """Normal-normal posterior for the mean IC of disjoint cohorts.

    The observation variance is the sample variance of the cohort means,
    floored so a short, accidentally smooth series cannot claim certainty.
    With fewer than two cohorts the prior is returned unchanged.
    """

    if not prior_sd > 0:
        raise LifecycleMonitorError("prior_sd must be positive")
    data = np.asarray([float(item) for item in observations], dtype=float)
    if data.size and not np.isfinite(data).all():
        raise LifecycleMonitorError("posterior input must be finite")
    prior_precision = 1.0 / prior_sd**2
    if data.size < 2:
        mean, sd = float(prior_mean), float(prior_sd)
        updated = False
    else:
        observation_sd = max(float(np.std(data, ddof=1)), float(observation_sd_floor))
        data_precision = data.size / observation_sd**2
        precision = prior_precision + data_precision
        mean = float((prior_precision * prior_mean + data_precision * data.mean()) / precision)
        sd = float(math.sqrt(1.0 / precision))
        updated = True
    return {
        "mean": mean,
        "sd": sd,
        "p_positive": float(scipy_stats.norm.sf(0.0, loc=mean, scale=sd)),
        "cohort_count": int(data.size),
        "updated_from_prior": updated,
    }


def classify_monitor_state(
    *,
    cohort_count: int,
    available_count: int,
    unavailable_count: int,
    posterior_p_positive: float,
    cusum_alarm: bool,
    policy: MonitorPolicy,
) -> tuple[str, list[str]]:
    """Map monitor statistics onto a non-authorizing state with explicit reasons."""

    if available_count == 0 and unavailable_count > 0:
        return SOURCE_BLOCKED, ["NO_AVAILABLE_RANK_IC"]
    shortfalls = []
    if available_count < policy.min_origin_sessions:
        shortfalls.append(f"ORIGIN_SESSIONS_BELOW_{policy.min_origin_sessions}")
    if cohort_count < policy.min_cohorts:
        shortfalls.append(f"COHORTS_BELOW_{policy.min_cohorts}")
    if shortfalls:
        return INSUFFICIENT_EVIDENCE, shortfalls
    reasons: list[str] = []
    if cusum_alarm:
        reasons.append("CUSUM_BELOW_REFERENCE")
    if posterior_p_positive < policy.alarm_posterior:
        reasons.append("POSTERIOR_BELOW_ALARM")
    elif posterior_p_positive < policy.watch_posterior:
        reasons.append("POSTERIOR_BELOW_WATCH")
    if cusum_alarm and posterior_p_positive < policy.alarm_posterior:
        return ALARM, reasons
    if reasons:
        return WATCH, reasons
    return HEALTHY, []


def evaluate_ic_series(
    ic_by_origin: pd.Series,
    *,
    horizon: int,
    policy: MonitorPolicy,
    unavailable_count: int = 0,
    reference_ic: float | None = None,
) -> dict[str, Any]:
    """All monitor statistics for one factor and label horizon."""

    if int(horizon) < 1:
        raise LifecycleMonitorError("horizon must be positive")
    series = _clean(ic_by_origin)
    cohorts = cohort_means(series, cohort_size=int(horizon))
    if len(series) >= 2 * int(horizon):
        t_stat, p_value, t_cohorts = nonoverlap_t_statistic(series, cohort_size=int(horizon))
    else:
        t_stat, p_value, t_cohorts = 0.0, 1.0, len(cohorts)
    posterior = normal_posterior(
        cohorts.tolist(),
        prior_mean=policy.prior_mean,
        prior_sd=policy.prior_sd,
        observation_sd_floor=policy.observation_sd_floor,
    )
    reference = policy.prior_mean if reference_ic is None else float(reference_ic)
    scale = max(
        float(np.std(cohorts.to_numpy(), ddof=1)) if len(cohorts) >= 2 else policy.prior_sd,
        policy.observation_sd_floor,
    )
    cusum = lower_cusum(
        cohorts.tolist(),
        reference=reference,
        scale=scale,
        slack=policy.cusum_slack,
        threshold=policy.cusum_threshold,
    )
    state, reasons = classify_monitor_state(
        cohort_count=len(cohorts),
        available_count=len(series),
        unavailable_count=int(unavailable_count),
        posterior_p_positive=posterior["p_positive"],
        cusum_alarm=bool(cusum["alarm"]),
        policy=policy,
    )
    return {
        "horizon": int(horizon),
        "available_origin_count": int(len(series)),
        "unavailable_origin_count": int(unavailable_count),
        "first_origin": str(series.index[0]) if len(series) else None,
        "last_origin": str(series.index[-1]) if len(series) else None,
        "mean_rank_ic": float(series.mean()) if len(series) else None,
        "ew_rank_ic": exponentially_weighted_mean(series, half_life=policy.half_life_sessions),
        "nonoverlap_t": {"t": t_stat, "p_value": p_value, "cohort_count": int(t_cohorts)},
        "cohort_count": int(len(cohorts)),
        "posterior": posterior,
        "cusum": {
            "reference": reference,
            "scale": scale,
            **{key: value for key, value in cusum.items() if key != "path"},
        },
        "state": state,
        "reasons": reasons,
    }


def _rank_ic(signal: pd.Series, returns: pd.Series) -> float | None:
    joined = pd.concat([signal, returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return None
    value = joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank())
    return float(value) if math.isfinite(value) else None


def residual_rank_ic(
    signal: pd.Series,
    pool_signal: pd.Series,
    returns: pd.Series,
) -> float | None:
    """RankIC of the part of ``signal`` that ``pool_signal`` does not explain."""

    date = pd.Timestamp("1970-01-01")
    residual = residualize_against_pool(
        pd.DataFrame([signal.astype(float)], index=[date]),
        pd.DataFrame([pool_signal.astype(float)], index=[date]),
        [date],
    ).iloc[0]
    return _rank_ic(residual, returns)


def size_exposure(signal: pd.Series, total_mv: pd.Series) -> float | None:
    """Rank correlation of the signal with log total market value; negative means small-cap."""

    sizes = total_mv.astype(float)
    sizes = np.log(sizes.where(sizes > 0.0))
    return _rank_ic(signal, sizes)


@dataclass(frozen=True)
class OutcomeRow:
    factor_id: str
    horizon: int
    origin_session: str
    rank_ic: float | None
    unavailable_reasons: tuple[str, ...]
    signals: pd.Series
    returns: pd.Series
    outcome_ref: Mapping[str, str]


def _decode_number(value: Any) -> float:
    """Decimal text, or the binary64 hex encoding production generations carry."""

    text = str(value)
    try:
        number = float.fromhex(text) if "0x" in text.lower() else float(text)
    except ValueError as exc:
        raise LifecycleMonitorError("outcome security value is not numeric") from exc
    if not math.isfinite(number):
        raise LifecycleMonitorError("outcome security value is not finite")
    return number


def outcome_row_from_payload(
    payload: Mapping[str, Any], *, factor_id: str, outcome_ref: Mapping[str, str]
) -> OutcomeRow:
    diagnostics = payload["diagnostics"]
    rank_ic = diagnostics["rank_ic"]
    value: float | None = None
    reasons: tuple[str, ...] = ()
    if rank_ic.get("state") == "AVAILABLE":
        value = float(rank_ic["value"])
        if not math.isfinite(value):
            raise LifecycleMonitorError("available RankIC is not finite")
    else:
        reasons = tuple(str(item) for item in rank_ic.get("reasons") or ["UNAVAILABLE"])
    signals: dict[str, float] = {}
    returns: dict[str, float] = {}
    for symbol, row in (diagnostics.get("securities") or {}).items():
        if row.get("signal") is not None:
            signals[symbol] = _decode_number(row["signal"])
        if row.get("raw_price_return") is not None:
            returns[symbol] = _decode_number(row["raw_price_return"])
    return OutcomeRow(
        factor_id=factor_id,
        horizon=int(payload["horizon"]),
        origin_session=str(payload["origin_session"]),
        rank_ic=value,
        unavailable_reasons=reasons,
        signals=pd.Series(signals, dtype=float),
        returns=pd.Series(returns, dtype=float),
        outcome_ref=dict(outcome_ref),
    )


def load_production_outcome_rows(workspace_root: str) -> tuple[list[OutcomeRow], dict[str, Any]]:
    """Read the unique revision head of every registered outcome series.

    Heads are resolved by the production outcome owner (exact predecessor
    chain, single root, no forks), and each head re-validates the observation,
    classification and source bytes it is bound to.
    """

    from . import production_outcomes as outcomes
    from .production_authority import FactorProductionStore

    store = FactorProductionStore(workspace_root)
    processing = store.read_optional("results/factors/outcome-processing.json")
    if processing is None:
        raise LifecycleMonitorError("outcome processing index is absent")
    processing_ref = {
        "path": "results/factors/outcome-processing.json",
        "sha256": processing.byte_sha256,
    }
    state = json.loads(processing.data)
    inventory = state["inventory_ref"]
    if store.read(inventory["path"]).byte_sha256 != inventory["sha256"]:
        raise LifecycleMonitorError("outcome inventory ref does not match its bytes")
    files = outcomes._registered_files(store, outcomes.ROOT)
    series_ids = sorted({PurePosixPath(path).parts[len(outcomes.ROOT.parts)] for path in files})
    rows: list[OutcomeRow] = []
    heads: list[dict[str, str]] = []
    for series_id in series_ids:
        head = outcomes._head(store, series_id)
        if head is None:
            continue
        payload = outcomes._read_outcome(store, head)["payload"]
        observation_ref = payload["observation_ref"]
        observation = store.read(observation_ref["path"])
        if observation.byte_sha256 != observation_ref["sha256"]:
            raise LifecycleMonitorError("observation bytes changed")
        factor_id = str(json.loads(observation.data)["payload"]["factor_id"])
        rows.append(outcome_row_from_payload(payload, factor_id=factor_id, outcome_ref=head))
        heads.append(dict(head))
    heads_digest = _sha_text(json.dumps(sorted(heads, key=lambda r: r["path"]), sort_keys=True))
    return rows, {
        "outcome_processing_ref": processing_ref,
        "inventory_ref": dict(inventory),
        "series_count": len(series_ids),
        "head_count": len(heads),
        "heads_sha256": heads_digest,
    }


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _residual_summary(rows: Iterable[OutcomeRow]) -> dict[str, dict[int, dict[str, Any]]]:
    by_key: dict[tuple[int, str], dict[str, OutcomeRow]] = {}
    for row in rows:
        if row.rank_ic is not None:
            by_key.setdefault((row.horizon, row.origin_session), {})[row.factor_id] = row
    values: dict[str, dict[int, list[float]]] = {}
    for (horizon, _origin), group in sorted(by_key.items()):
        if len(group) < 2:
            continue
        for factor_id, row in group.items():
            pool = [other.signals for name, other in group.items() if name != factor_id]
            pool_signal = pd.concat(pool, axis=1).mean(axis=1)
            value = residual_rank_ic(row.signals, pool_signal, row.returns)
            if value is not None:
                values.setdefault(factor_id, {}).setdefault(horizon, []).append(value)
    return {
        factor_id: {
            horizon: {
                "origin_count": len(items),
                "mean_residual_rank_ic": float(np.mean(items)),
                "pool": "OTHER_ACTIVE_FACTORS_EQUAL_RANK_MEAN",
            }
            for horizon, items in sorted(by_horizon.items())
        }
        for factor_id, by_horizon in sorted(values.items())
    }


def build_monitor_report(
    rows: Sequence[OutcomeRow],
    *,
    policy: MonitorPolicy,
    input_refs: Mapping[str, Any],
    size_by_origin: Mapping[str, pd.Series] | None = None,
    market_snapshot_ref: Mapping[str, str] | None = None,
    priors: Mapping[str, Mapping[str, float]] | None = None,
    prior_source: str = "POLICY_DEFAULT_WEAK_PRIOR",
) -> dict[str, Any]:
    """Assemble the non-authorizing monitor report."""

    factors: dict[str, Any] = {}
    grouped: dict[str, dict[int, list[OutcomeRow]]] = {}
    for row in rows:
        grouped.setdefault(row.factor_id, {}).setdefault(row.horizon, []).append(row)
    residuals = _residual_summary(rows)
    for factor_id in sorted(grouped, key=lambda value: value.encode("utf-8")):
        prior = dict((priors or {}).get(factor_id) or {})
        overrides = {key: float(prior[key]) for key in ("prior_mean", "prior_sd") if key in prior}
        factor_policy = MonitorPolicy(**{**asdict(policy), **overrides})
        horizons: dict[str, Any] = {}
        for horizon, items in sorted(grouped[factor_id].items()):
            available = {
                row.origin_session: row.rank_ic for row in items if row.rank_ic is not None
            }
            unavailable = [row for row in items if row.rank_ic is None]
            reasons: dict[str, int] = {}
            for row in unavailable:
                for reason in row.unavailable_reasons:
                    reasons[reason] = reasons.get(reason, 0) + 1
            series = pd.Series(available, dtype=float).sort_index()
            result = evaluate_ic_series(
                series,
                horizon=horizon,
                policy=factor_policy,
                unavailable_count=len(unavailable),
                reference_ic=prior.get("reference_ic"),
            )
            result["source_gap_reasons"] = dict(sorted(reasons.items()))
            horizons[str(horizon)] = result
        exposure: dict[str, Any] = {"state": "UNAVAILABLE", "reason": "MARKET_SIZE_NOT_BOUND"}
        if size_by_origin:
            values = []
            for row in grouped[factor_id].get(1, []):
                sizes = size_by_origin.get(row.origin_session)
                if sizes is not None:
                    value = size_exposure(row.signals, sizes)
                    if value is not None:
                        values.append(value)
            if values:
                exposure = {
                    "state": "AVAILABLE",
                    "measure": "RANK_CORR_SIGNAL_LOG_TOTAL_MV",
                    "origin_count": len(values),
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "interpretation": "NEGATIVE_MEANS_SMALL_CAP_TILT",
                }
        factors[factor_id] = {
            "horizons": horizons,
            "residual_rank_ic": residuals.get(factor_id, {}),
            "size_exposure": exposure,
            "prior": {
                "mean": factor_policy.prior_mean,
                "sd": factor_policy.prior_sd,
                "reference_ic": prior.get("reference_ic", factor_policy.prior_mean),
            },
        }
    return {
        "schema_version": MONITOR_SCHEMA,
        "authority": MONITOR_AUTHORITY,
        "grants_weight_change": False,
        "grants_admission": False,
        "policy": asdict(policy),
        "prior_source": prior_source,
        "inputs": dict(input_refs),
        "market_snapshot_ref": dict(market_snapshot_ref) if market_snapshot_ref else None,
        "factors": factors,
        "limitations": [
            "RAW_CLOSE_REFERENCE_RETURNS_NOT_EXECUTABLE",
            "PRODUCTION_OBSERVATIONS_ARE_NOT_FORMAL_PROSPECTIVE_CAPTURES",
            "SOURCE_GAPS_ARE_NOT_ALPHA_FAILURE",
            "STATES_ARE_DIAGNOSTIC_AND_GRANT_NO_WEIGHT_CHANGE",
        ],
    }


__all__ = [
    "ALARM",
    "HEALTHY",
    "INSUFFICIENT_EVIDENCE",
    "LifecycleMonitorError",
    "MONITOR_AUTHORITY",
    "MONITOR_SCHEMA",
    "MonitorPolicy",
    "OutcomeRow",
    "SOURCE_BLOCKED",
    "WATCH",
    "build_monitor_report",
    "classify_monitor_state",
    "cohort_means",
    "evaluate_ic_series",
    "exponentially_weighted_mean",
    "load_production_outcome_rows",
    "lower_cusum",
    "normal_posterior",
    "outcome_row_from_payload",
    "residual_rank_ic",
    "size_exposure",
]
