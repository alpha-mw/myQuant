"""Deterministic prospective evaluation and non-activating admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from itertools import combinations
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .bootstrap import PROSPECTIVE_LANE
from .common import (
    ANNUAL_OPEN_SESSIONS,
    BH_Q_CEILING,
    COST_BPS,
    CPCV_BLOCK_COUNT,
    CPCV_EMBARGO_OPEN_SESSIONS,
    CPCV_PATH_COUNT,
    CPCV_PURGE_OPEN_SESSIONS,
    CPCV_TEST_BLOCK_COUNT,
    DSR_FLOOR,
    LABEL_HORIZON_OPEN_SESSIONS,
    MIN_CLOSED_MONTH_ENDS,
    MIN_DAILY_RANKIC_SESSIONS,
    MIN_DISJOINT_COHORTS,
    PBO_CEILING,
    PBO_MIN_CONFIGURATIONS,
    PBO_SPLIT_COUNT,
    POSITIVE_PATH_RATIO_FLOOR,
    REDUNDANCY_CORRELATION_FLOOR,
    REDUNDANCY_MIN_OVERLAP,
    SIGNAL_OPEN_SESSIONS,
    SHRINKAGE_PSEUDO_COUNT,
    T_STAT_HURDLE,
    TURNOVER_CEILING,
    artifact_ref,
    business_identity,
    canonical_timestamp,
    decimal_text,
    exact_payload,
)
from .errors import FactorGovernanceError
from .execution import validate_execution_turnover_evidence
from .prospective import (
    _validate_observation_prevalidated,
    _validate_signal_capture_prevalidated,
    validate_configuration_selection,
    validate_preregistration,
)
from .statistics import (
    TRIAL_CORRECTION_KIND,
    benjamini_hochberg_by_family,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    redundancy_clusters,
)
from .weights import largest_remainder_weights

EVALUATION_KIND: Final = "factor.prospective_evaluation"
ADMITTED_SET_KIND: Final = "factor.admitted_set"
PROSPECTIVE_ADMISSION_ROUTE: Final = "PROSPECTIVE_ADMISSION"
MAX_ACTIVE_FACTORS: Final = 10
_MAX_EVALUATION_BYTES: Final = 1024 * 1024
_MAX_ADMITTED_SET_BYTES: Final = 128 * 1024

_EVALUATION_FIELDS: Final = {
    "evaluation_id",
    "preregistration_id",
    "selection_id",
    "lane",
    "observation_ids",
    "observation_count",
    "execution_turnover_evidence_ref",
    "candidate_rows",
    "trial_statistics",
    "redundancy_clusters",
    "admission_eligible",
    "blockers",
    "cost_bps",
}
_ADMITTED_SET_FIELDS: Final = {
    "admitted_set_id",
    "lane",
    "preregistration_id",
    "selection_id",
    "evaluation_id",
    "factor_rows",
    "weight_total",
    "weighting_method",
    "activation_authorized",
}
_CANDIDATE_ROW_FIELDS: Final = {
    "configuration_id",
    "factor_id",
    "family",
    "primitive",
    "normalized_slot",
    "valid_daily_rankic_sessions",
    "closed_calendar_month_end_observations",
    "disjoint_30_open_session_cohort_means",
    "maturity_passed",
    "mean_rank_ic",
    "mean_purged_oos_rank_ic",
    "shrunk_ic",
    "t_statistic",
    "t_p_value",
    "deflated_sharpe_ratio",
    "bh_q_value",
    "cpcv_path_count",
    "positive_path_ratio",
    "turnover",
    "total_estimated_cost",
    "gross_labeled_return_sum",
    "net_labeled_return_sum",
    "redundancy_cluster_id",
    "cluster_representative",
    "admission_eligible",
    "blockers",
}


def _selected_candidates(
    preregistration: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_configuration = {
        row["configuration_id"]: row for row in preregistration["payload"]["candidates"]
    }
    selected_ids = [
        row["selected_configuration_id"] for row in selection["payload"]["selected_configurations"]
    ]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise FactorGovernanceError("selection lacks distinct trial configurations")
    return [by_configuration[configuration_id] for configuration_id in selected_ids]


def _validated_capture_observation_chain(
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(signal_captures, (str, bytes)) or not isinstance(signal_captures, Sequence):
        raise FactorGovernanceError("signal captures must be a sequence")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise FactorGovernanceError("observations must be a sequence")
    if len(signal_captures) != SIGNAL_OPEN_SESSIONS or len(observations) != SIGNAL_OPEN_SESSIONS:
        raise FactorGovernanceError("evaluation requires exactly 360 captures and observations")
    captures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    previous_capture: dict[str, Any] | None = None
    previous_observation: dict[str, Any] | None = None
    for ordinal, (capture_value, observation_value) in enumerate(
        zip(signal_captures, observations, strict=True)
    ):
        capture = _validate_signal_capture_prevalidated(
            capture_value,
            preregistration=preregistration,
            selection=selection,
            previous_signal_capture=previous_capture,
        )
        observed = _validate_observation_prevalidated(
            observation_value,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
            previous_observation=previous_observation,
        )
        if capture["payload"]["ordinal"] != ordinal or observed["payload"]["ordinal"] != ordinal:
            raise FactorGovernanceError("capture and observation ordinals are not contiguous")
        captures.append(capture)
        rows.append(observed)
        previous_capture = capture
        previous_observation = observed
    terminal_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(
                candidate["coverage_gate"] == "FAILED"
                for candidate in row["payload"]["configuration_rows"]
            )
        ),
        None,
    )
    if terminal_index is not None and terminal_index != len(rows) - 1:
        raise FactorGovernanceError("observations continue after terminal coverage failure")
    return captures, rows


def _series_by_configuration(
    observations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    signal_sessions: Sequence[str],
) -> dict[str, pd.Series]:
    values: dict[str, dict[str, float]] = {row["configuration_id"]: {} for row in candidates}
    for observation in observations:
        payload = observation["payload"]
        session = payload["signal_session"]
        for row in payload["configuration_rows"]:
            rank_ic = row["rank_ic"] if row["valid_daily_rankic"] is True else None
            values[row["configuration_id"]][session] = (
                float(rank_ic) if rank_ic is not None else math.nan
            )
    index = pd.Index(list(signal_sessions), name="signal_session")
    return {
        configuration_id: pd.Series(by_session, dtype=float).reindex(index)
        for configuration_id, by_session in values.items()
    }


def _closed_month_end_count(series: pd.Series, open_sessions: Sequence[str]) -> int:
    month_ends = [
        open_sessions[index]
        for index in range(SIGNAL_OPEN_SESSIONS)
        if open_sessions[index][:7] != open_sessions[index + 1][:7]
    ]
    return int(series.reindex(month_ends).notna().sum())


def _cohort_means(series: pd.Series) -> list[float]:
    means: list[float] = []
    for start in range(0, SIGNAL_OPEN_SESSIONS, LABEL_HORIZON_OPEN_SESSIONS):
        cohort = series.iloc[slice(start, start + LABEL_HORIZON_OPEN_SESSIONS)]
        if len(cohort) != LABEL_HORIZON_OPEN_SESSIONS or cohort.isna().any():
            continue
        value = float(cohort.mean())
        if math.isfinite(value):
            means.append(value)
    return means


def _cohort_test(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 1.0
    statistic, p_value = scipy_stats.ttest_1samp(np.asarray(values, dtype=float), 0.0)
    if not math.isfinite(float(statistic)) or not math.isfinite(float(p_value)):
        return 0.0, 1.0
    return float(statistic), float(p_value)


def _block_performance(series_by_configuration: Mapping[str, pd.Series]) -> pd.DataFrame:
    block_size = SIGNAL_OPEN_SESSIONS // CPCV_BLOCK_COUNT
    rows: list[dict[str, float]] = []
    for block in range(CPCV_BLOCK_COUNT):
        start = block * block_size
        stop = start + block_size
        row = {}
        for configuration_id, series in series_by_configuration.items():
            observed = series.iloc[start:stop].dropna()
            row[configuration_id] = float(observed.mean()) if not observed.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(series_by_configuration))


def _cpcv_path_means(series: pd.Series) -> list[float]:
    block_size = SIGNAL_OPEN_SESSIONS // CPCV_BLOCK_COUNT
    path_means: list[float] = []
    for test_blocks in combinations(range(CPCV_BLOCK_COUNT), CPCV_TEST_BLOCK_COUNT):
        test_positions: set[int] = set()
        excluded: set[int] = set()
        for block in test_blocks:
            first = block * block_size
            last = first + block_size - 1
            test_positions.update(range(first, last + 1))
            excluded.update(
                range(
                    max(0, first - CPCV_PURGE_OPEN_SESSIONS),
                    min(SIGNAL_OPEN_SESSIONS, last + 1),
                )
            )
            excluded.update(
                range(
                    last + 1,
                    min(
                        SIGNAL_OPEN_SESSIONS,
                        last + 1 + CPCV_EMBARGO_OPEN_SESSIONS,
                    ),
                )
            )
        if not (set(range(SIGNAL_OPEN_SESSIONS)) - excluded):
            return []
        observed = series.iloc[sorted(test_positions)].dropna()
        if observed.empty:
            return []
        value = float(observed.mean())
        if not math.isfinite(value):
            return []
        path_means.append(value)
    return path_means


def _sharpe(series: pd.Series) -> float | None:
    observed = series.dropna().to_numpy(dtype=float)
    if len(observed) < 2:
        return None
    standard_deviation = float(np.std(observed, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return None
    result = float(np.mean(observed) / standard_deviation)
    return result if math.isfinite(result) else None


def _distribution_moments(series: pd.Series) -> tuple[float, float]:
    observed = series.dropna().to_numpy(dtype=float)
    if len(observed) < 4:
        return 0.0, 3.0
    skew = float(scipy_stats.skew(observed, bias=False))
    kurtosis = float(scipy_stats.kurtosis(observed, fisher=False, bias=False))
    return (
        skew if math.isfinite(skew) else 0.0,
        kurtosis if math.isfinite(kurtosis) else 3.0,
    )


def _maturity_passed(
    *, valid_daily_sessions: int, closed_month_ends: int, disjoint_cohorts: int
) -> bool:
    return (
        valid_daily_sessions >= MIN_DAILY_RANKIC_SESSIONS
        and closed_month_ends >= MIN_CLOSED_MONTH_ENDS
        and disjoint_cohorts >= MIN_DISJOINT_COHORTS
    )


def _maturity_blockers(
    valid_daily_sessions: int, closed_month_ends: int, disjoint_cohorts: int
) -> list[str]:
    blockers = []
    if valid_daily_sessions < MIN_DAILY_RANKIC_SESSIONS:
        blockers.append("DAILY_RANKIC_MATURITY_FAILED")
    if closed_month_ends < MIN_CLOSED_MONTH_ENDS:
        blockers.append("CLOSED_MONTH_END_MATURITY_FAILED")
    if disjoint_cohorts < MIN_DISJOINT_COHORTS:
        blockers.append("DISJOINT_COHORT_MATURITY_FAILED")
    return blockers


def _signal_metric_blockers(t_statistic: float, dsr: float) -> list[str]:
    blockers = []
    if not t_statistic > float(T_STAT_HURDLE):
        blockers.append("T_STATISTIC_FAILED")
    if dsr < float(DSR_FLOOR):
        blockers.append("DEFLATED_SHARPE_FAILED")
    return blockers


def _pbo_blockers(*, pbo_complete: bool, pbo_split_count: int, pbo: float) -> list[str]:
    blockers = []
    if pbo_complete is not True or pbo_split_count != PBO_SPLIT_COUNT:
        blockers.append("PBO_INCOMPLETE")
    elif pbo > float(PBO_CEILING):
        blockers.append("PBO_THRESHOLD_FAILED")
    return blockers


def _post_trial_blockers(
    *,
    bh_q_value: float,
    cpcv_path_count: int,
    positive_path_ratio: float,
    turnover: Decimal,
) -> list[str]:
    blockers = []
    if bh_q_value > float(BH_Q_CEILING):
        blockers.append("FAMILY_BH_FAILED")
    if cpcv_path_count != CPCV_PATH_COUNT:
        blockers.append("CPCV_INCOMPLETE")
    elif positive_path_ratio < float(POSITIVE_PATH_RATIO_FLOOR):
        blockers.append("POSITIVE_PATH_RATIO_FAILED")
    if turnover > TURNOVER_CEILING:
        blockers.append("TURNOVER_FAILED")
    return blockers


def _metric_blockers(
    *,
    valid_daily_sessions: int,
    closed_month_ends: int,
    disjoint_cohorts: int,
    t_statistic: float,
    dsr: float,
    pbo_complete: bool,
    pbo_split_count: int,
    pbo: float,
    bh_q_value: float,
    cpcv_path_count: int,
    positive_path_ratio: float,
    turnover: Decimal,
) -> list[str]:
    return (
        _maturity_blockers(valid_daily_sessions, closed_month_ends, disjoint_cohorts)
        + _signal_metric_blockers(t_statistic, dsr)
        + _pbo_blockers(
            pbo_complete=pbo_complete,
            pbo_split_count=pbo_split_count,
            pbo=pbo,
        )
        + _post_trial_blockers(
            bh_q_value=bh_q_value,
            cpcv_path_count=cpcv_path_count,
            positive_path_ratio=positive_path_ratio,
            turnover=turnover,
        )
    )


def _execution_metrics(
    execution_turnover_evidence: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    configuration_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = validate_execution_turnover_evidence(
        execution_turnover_evidence,
        preregistration=preregistration,
        selection=selection,
    )
    rows = evidence["payload"]["configuration_rows"]
    by_configuration = {row["configuration_id"]: row for row in rows}
    if set(by_configuration) != set(configuration_ids) or len(rows) != len(by_configuration):
        raise FactorGovernanceError("execution evidence must cover all configurations exactly")
    return evidence, by_configuration


def _redundancy_partition(
    candidates: Sequence[Mapping[str, Any]],
    configuration_ids: Sequence[str],
    series_by_configuration: Mapping[str, pd.Series],
) -> tuple[dict[str, str], list[tuple[str, ...]], dict[str, str]]:
    slots = {row["configuration_id"]: f"{row['family']}:{row['primitive']}" for row in candidates}
    components = list(
        redundancy_clusters(
            series_by_configuration,
            normalized_slots=slots,
            correlation_floor=float(REDUNDANCY_CORRELATION_FLOOR),
            min_overlap=REDUNDANCY_MIN_OVERLAP,
        )
    )
    clustered = {value for component in components for value in component}
    components.extend((value,) for value in configuration_ids if value not in clustered)
    components.sort(key=lambda values: values[0].encode("utf-8"))
    cluster_id_by_configuration: dict[str, str] = {}
    for component in components:
        cluster_id = business_identity("redundancy-cluster", {"configuration_ids": list(component)})
        for configuration_id in component:
            cluster_id_by_configuration[configuration_id] = cluster_id
    return slots, components, cluster_id_by_configuration


def _trial_metrics(
    series_by_configuration: Mapping[str, pd.Series],
    components: Sequence[tuple[str, ...]],
) -> tuple[dict[str, Any], dict[str, float | None], float, int, bool, int]:
    block_frame = _block_performance(series_by_configuration)
    pbo = probability_of_backtest_overfitting(block_frame)
    sharpe_by_configuration = {
        configuration_id: _sharpe(series)
        for configuration_id, series in series_by_configuration.items()
    }
    finite_sharpes = [
        value
        for value in sharpe_by_configuration.values()
        if value is not None and math.isfinite(value)
    ]
    trial_icir_complete = len(sharpe_by_configuration) >= PBO_MIN_CONFIGURATIONS and len(
        finite_sharpes
    ) == len(sharpe_by_configuration)
    trial_sharpe_std = (
        float(np.std(np.asarray(finite_sharpes, dtype=float), ddof=1))
        if trial_icir_complete
        else 0.0
    )
    if not math.isfinite(trial_sharpe_std):
        trial_icir_complete = False
        trial_sharpe_std = 0.0
    return (
        pbo,
        sharpe_by_configuration,
        trial_sharpe_std,
        max(1, len(components)),
        trial_icir_complete,
        len(finite_sharpes),
    )


def _preliminary_candidate_metrics(
    candidates: Sequence[Mapping[str, Any]],
    series_by_configuration: Mapping[str, pd.Series],
    open_sessions: Sequence[str],
    sharpe_by_configuration: Mapping[str, float | None],
    *,
    trial_sharpe_std: float,
    effective_trials: int,
    trial_icir_complete: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, str]]:
    preliminary: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    families: dict[str, str] = {}
    for candidate in candidates:
        configuration_id = candidate["configuration_id"]
        series = series_by_configuration[configuration_id]
        valid_count = int(series.notna().sum())
        month_end_count = _closed_month_end_count(series, open_sessions)
        cohort_means = _cohort_means(series)
        t_statistic, t_p_value = _cohort_test(cohort_means)
        mean_ic = float(series.dropna().mean()) if valid_count else 0.0
        path_means = _cpcv_path_means(series)
        mean_path_ic = float(np.mean(path_means)) if path_means else 0.0
        path_count = len(path_means)
        shrunk_ic = (
            max(0.0, mean_path_ic) * path_count / (path_count + float(SHRINKAGE_PSEUDO_COUNT))
        )
        skew, kurtosis = _distribution_moments(series)
        observed_icir = sharpe_by_configuration[configuration_id]
        dsr = (
            deflated_sharpe_ratio(
                observed_sharpe=observed_icir,
                trial_sharpe_std=trial_sharpe_std,
                trial_count=effective_trials,
                sample_size=valid_count,
                skew=skew,
                kurtosis=kurtosis,
            )
            if trial_icir_complete and observed_icir is not None
            else 0.0
        )
        path_ratio = float(np.mean(np.asarray(path_means) > 0.0)) if path_means else 0.0
        maturity_passed = _maturity_passed(
            valid_daily_sessions=valid_count,
            closed_month_ends=month_end_count,
            disjoint_cohorts=len(cohort_means),
        )
        preliminary[configuration_id] = {
            "candidate": candidate,
            "valid_count": valid_count,
            "month_end_count": month_end_count,
            "cohort_count": len(cohort_means),
            "maturity_passed": maturity_passed,
            "mean_ic": mean_ic,
            "mean_path_ic": mean_path_ic,
            "shrunk_ic": shrunk_ic,
            "t_statistic": t_statistic,
            "t_p_value": t_p_value,
            "dsr": dsr,
            "path_count": path_count,
            "path_ratio": path_ratio,
        }
        p_values[configuration_id] = t_p_value
        families[configuration_id] = candidate["family"]
    return preliminary, p_values, families


def _observation_completeness(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool]:
    window_complete = len(observations) == SIGNAL_OPEN_SESSIONS
    terminal_coverage_failure = any(
        row["coverage_gate"] == "FAILED"
        for observation in observations
        for row in observation["payload"]["configuration_rows"]
    )
    return window_complete, terminal_coverage_failure


def _candidate_eligibility(
    preliminary: Mapping[str, Mapping[str, Any]],
    pbo: Mapping[str, Any],
    q_values: Mapping[str, float],
    turnover: Mapping[str, Decimal],
    *,
    observation_window_complete: bool,
    terminal_coverage_failure: bool,
    trial_icir_complete: bool,
    execution_complete: bool,
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    observation_blockers = []
    if not observation_window_complete:
        observation_blockers.append("OBSERVATION_WINDOW_INCOMPLETE")
    if terminal_coverage_failure:
        observation_blockers.append("COVERAGE_TERMINAL_FAILURE")
    if not trial_icir_complete:
        observation_blockers.append("TRIAL_ICIR_INCOMPLETE")
    if not execution_complete:
        observation_blockers.append("EXECUTION_EVIDENCE_INCOMPLETE")
    base_eligible: dict[str, bool] = {}
    blocker_by_configuration: dict[str, list[str]] = {}
    for configuration_id, metrics in preliminary.items():
        blockers = list(observation_blockers)
        blockers.extend(
            _metric_blockers(
                valid_daily_sessions=metrics["valid_count"],
                closed_month_ends=metrics["month_end_count"],
                disjoint_cohorts=metrics["cohort_count"],
                t_statistic=metrics["t_statistic"],
                dsr=metrics["dsr"],
                pbo_complete=pbo["complete"],
                pbo_split_count=pbo["split_count"],
                pbo=pbo["pbo"],
                bh_q_value=q_values[configuration_id],
                cpcv_path_count=metrics["path_count"],
                positive_path_ratio=metrics["path_ratio"],
                turnover=turnover[configuration_id],
            )
        )
        blockers = sorted(set(blockers))
        blocker_by_configuration[configuration_id] = blockers
        base_eligible[configuration_id] = not blockers
    return base_eligible, blocker_by_configuration


def _cluster_representatives(
    components: Sequence[tuple[str, ...]],
    cluster_id_by_configuration: Mapping[str, str],
    base_eligible: Mapping[str, bool],
    preliminary: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    representatives: dict[str, str | None] = {}
    for component in components:
        cluster_id = cluster_id_by_configuration[component[0]]
        eligible = [value for value in component if base_eligible[value]]
        representatives[cluster_id] = (
            sorted(
                eligible,
                key=lambda value: _representative_rank(value, preliminary),
            )[0]
            if eligible
            else None
        )
    return representatives


def _representative_rank(
    configuration_id: str, preliminary: Mapping[str, Mapping[str, Any]]
) -> tuple[float, float, bytes]:
    metrics = preliminary[configuration_id]
    return (
        -float(metrics["dsr"]),
        -float(metrics["mean_path_ic"]),
        configuration_id.encode("utf-8"),
    )


def _limited_representatives(
    representative_by_cluster: Mapping[str, str | None],
    preliminary: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    representatives = [value for value in representative_by_cluster.values() if value is not None]
    ordered = sorted(
        set(representatives),
        key=lambda value: _representative_rank(value, preliminary),
    )
    return set(ordered[:MAX_ACTIVE_FACTORS])


def _evaluation_candidate_rows(
    configuration_ids: Sequence[str],
    preliminary: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, str],
    cluster_id_by_configuration: Mapping[str, str],
    representative_by_cluster: Mapping[str, str | None],
    blocker_by_configuration: Mapping[str, list[str]],
    base_eligible: Mapping[str, bool],
    admitted_representatives: set[str],
    q_values: Mapping[str, float],
    execution_metrics: Mapping[str, Mapping[str, Any]],
    annualized_turnover: Mapping[str, Decimal],
) -> list[dict[str, Any]]:
    rows = []
    for configuration_id in sorted(configuration_ids, key=lambda value: value.encode("utf-8")):
        metrics = preliminary[configuration_id]
        candidate = metrics["candidate"]
        execution = execution_metrics[configuration_id]
        cluster_id = cluster_id_by_configuration[configuration_id]
        representative = representative_by_cluster[cluster_id]
        blockers = list(blocker_by_configuration[configuration_id])
        if base_eligible[configuration_id] and representative != configuration_id:
            blockers.append("REDUNDANT_CONFIGURATION")
        if (
            base_eligible[configuration_id]
            and representative == configuration_id
            and configuration_id not in admitted_representatives
        ):
            blockers.append("ACTIVE_FACTOR_LIMIT_EXCEEDED")
        admitted = base_eligible[configuration_id] and configuration_id in admitted_representatives
        rows.append(
            {
                "configuration_id": configuration_id,
                "factor_id": candidate["factor_id"],
                "family": candidate["family"],
                "primitive": candidate["primitive"],
                "normalized_slot": slots[configuration_id],
                "valid_daily_rankic_sessions": metrics["valid_count"],
                "closed_calendar_month_end_observations": metrics["month_end_count"],
                "disjoint_30_open_session_cohort_means": metrics["cohort_count"],
                "maturity_passed": metrics["maturity_passed"],
                "mean_rank_ic": decimal_text(metrics["mean_ic"], label="mean_rank_ic"),
                "mean_purged_oos_rank_ic": decimal_text(
                    metrics["mean_path_ic"], label="mean_purged_oos_rank_ic"
                ),
                "shrunk_ic": decimal_text(metrics["shrunk_ic"], label="shrunk_ic"),
                "t_statistic": decimal_text(metrics["t_statistic"], label="t_statistic"),
                "t_p_value": decimal_text(metrics["t_p_value"], label="t_p_value"),
                "deflated_sharpe_ratio": decimal_text(metrics["dsr"], label="dsr"),
                "bh_q_value": decimal_text(q_values[configuration_id], label="bh_q_value"),
                "cpcv_path_count": metrics["path_count"],
                "positive_path_ratio": decimal_text(
                    metrics["path_ratio"], label="positive_path_ratio"
                ),
                "turnover": decimal_text(
                    annualized_turnover[configuration_id], label="annualized_turnover"
                ),
                "total_estimated_cost": execution["total_estimated_cost"],
                "gross_labeled_return_sum": execution["gross_labeled_return_sum"],
                "net_labeled_return_sum": execution["net_labeled_return_sum"],
                "redundancy_cluster_id": cluster_id,
                "cluster_representative": representative == configuration_id,
                "admission_eligible": admitted,
                "blockers": sorted(set(blockers)),
            }
        )
    return rows


def _evaluation_cluster_rows(
    components: Sequence[tuple[str, ...]],
    slots: Mapping[str, str],
    cluster_id_by_configuration: Mapping[str, str],
    representative_by_cluster: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    rows = []
    for component in components:
        cluster_id = cluster_id_by_configuration[component[0]]
        rows.append(
            {
                "cluster_id": cluster_id,
                "configuration_ids": list(component),
                "normalized_slots": sorted({slots[value] for value in component}),
                "representative_configuration_id": representative_by_cluster[cluster_id],
                "edge_rules": [
                    "SAME_NORMALIZED_PRIMITIVE_FAMILY_SLOT",
                    "ABS_RANKIC_CORRELATION_GTE_0.70_MIN_12_SHARED",
                ],
            }
        )
    return rows


def _global_evaluation_blockers(
    *,
    observation_window_complete: bool,
    terminal_coverage_failure: bool,
    trial_icir_complete: bool,
    execution_complete: bool,
    pbo: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    blockers = []
    if not observation_window_complete:
        blockers.append("OBSERVATION_WINDOW_INCOMPLETE")
    if terminal_coverage_failure:
        blockers.append("COVERAGE_TERMINAL_FAILURE")
    if not trial_icir_complete:
        blockers.append("TRIAL_ICIR_INCOMPLETE")
    if not execution_complete:
        blockers.append("EXECUTION_EVIDENCE_INCOMPLETE")
    if pbo["complete"] is not True or pbo["split_count"] != PBO_SPLIT_COUNT:
        blockers.append("PBO_INCOMPLETE")
    elif pbo["pbo"] > float(PBO_CEILING):
        blockers.append("PBO_THRESHOLD_FAILED")
    admission_eligible = any(row["admission_eligible"] for row in candidate_rows)
    if not admission_eligible:
        blockers.append("NO_ADMITTED_CONFIGURATION")
    return admission_eligible, sorted(set(blockers))


def _annualized_turnover(
    execution_metrics: Mapping[str, Mapping[str, Any]],
    configuration_ids: Sequence[str],
) -> dict[str, Decimal]:
    values: dict[str, Decimal] = {}
    for configuration_id in configuration_ids:
        row = execution_metrics[configuration_id]
        expected = (
            Decimal(row["total_turnover"])
            * Decimal(ANNUAL_OPEN_SESSIONS)
            / Decimal(SIGNAL_OPEN_SESSIONS)
        )
        actual = Decimal(row["annualized_turnover"])
        if decimal_text(expected, label="annualized_turnover") != row["annualized_turnover"]:
            raise FactorGovernanceError("execution annualized turnover differs")
        values[configuration_id] = actual
    return values


def _evaluation_payload(
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    signal_captures: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    execution_turnover_evidence: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    prereg_payload = preregistration["payload"]
    selection_payload = selection["payload"]
    candidates = _selected_candidates(preregistration, selection)
    configuration_ids = [row["configuration_id"] for row in candidates]
    execution, execution_by_configuration = _execution_metrics(
        execution_turnover_evidence,
        preregistration=preregistration,
        selection=selection,
        configuration_ids=configuration_ids,
    )
    if execution["payload"]["signal_capture_refs"] != [
        artifact_ref(value) for value in signal_captures
    ] or execution["payload"]["observation_refs"] != [
        artifact_ref(value) for value in observations
    ]:
        raise FactorGovernanceError("execution evidence does not bind the evaluated closure")
    turnover = _annualized_turnover(execution_by_configuration, configuration_ids)
    execution_complete = (
        execution["payload"]["execution_state"] == "COMPLETE"
        and execution["payload"]["blockers"] == []
    )
    series_by_configuration = _series_by_configuration(
        observations, candidates, prereg_payload["signal_sessions"]
    )
    slots, components, cluster_id_by_configuration = _redundancy_partition(
        candidates, configuration_ids, series_by_configuration
    )
    (
        pbo,
        sharpe_by_configuration,
        trial_sharpe_std,
        effective_trials,
        trial_icir_complete,
        trial_icir_count,
    ) = _trial_metrics(series_by_configuration, components)
    preliminary, p_values, families = _preliminary_candidate_metrics(
        candidates,
        series_by_configuration,
        prereg_payload["open_sessions"],
        sharpe_by_configuration,
        trial_sharpe_std=trial_sharpe_std,
        effective_trials=effective_trials,
        trial_icir_complete=trial_icir_complete,
    )
    q_values = benjamini_hochberg_by_family(p_values, families)

    observation_window_complete, terminal_coverage_failure = _observation_completeness(observations)
    base_eligible, blocker_by_configuration = _candidate_eligibility(
        preliminary,
        pbo,
        q_values,
        turnover,
        observation_window_complete=observation_window_complete,
        terminal_coverage_failure=terminal_coverage_failure,
        trial_icir_complete=trial_icir_complete,
        execution_complete=execution_complete,
    )
    representative_by_cluster = _cluster_representatives(
        components,
        cluster_id_by_configuration,
        base_eligible,
        preliminary,
    )
    admitted_representatives = _limited_representatives(
        representative_by_cluster,
        preliminary,
    )
    candidate_rows = _evaluation_candidate_rows(
        configuration_ids,
        preliminary,
        slots,
        cluster_id_by_configuration,
        representative_by_cluster,
        blocker_by_configuration,
        base_eligible,
        admitted_representatives,
        q_values,
        execution_by_configuration,
        turnover,
    )
    cluster_rows = _evaluation_cluster_rows(
        components,
        slots,
        cluster_id_by_configuration,
        representative_by_cluster,
    )
    admission_eligible, global_blockers = _global_evaluation_blockers(
        observation_window_complete=observation_window_complete,
        terminal_coverage_failure=terminal_coverage_failure,
        trial_icir_complete=trial_icir_complete,
        execution_complete=execution_complete,
        pbo=pbo,
        candidate_rows=candidate_rows,
    )

    observation_ids = [row["payload"]["observation_id"] for row in observations]
    identity_inputs = {
        "preregistration_id": prereg_payload["preregistration_id"],
        "selection_id": selection_payload["selection_id"],
        "observation_ids": observation_ids,
        "execution_turnover_evidence_ref": artifact_ref(execution),
    }
    return {
        "evaluation_id": business_identity("evaluation", identity_inputs),
        "preregistration_id": prereg_payload["preregistration_id"],
        "selection_id": selection_payload["selection_id"],
        "lane": PROSPECTIVE_LANE,
        "observation_ids": observation_ids,
        "observation_count": len(observations),
        "execution_turnover_evidence_ref": artifact_ref(execution),
        "candidate_rows": candidate_rows,
        "trial_statistics": {
            "trial_correction_kind": TRIAL_CORRECTION_KIND,
            "selected_configuration_count": len(configuration_ids),
            "effective_trial_count": effective_trials,
            "trial_icir_complete": trial_icir_complete,
            "trial_icir_count": trial_icir_count,
            "trial_sharpe_std": decimal_text(trial_sharpe_std, label="trial_sharpe_std"),
            "trial_sharpe_std_ddof": 1,
            "pbo": decimal_text(pbo["pbo"], label="pbo"),
            "pbo_complete": pbo["complete"],
            "pbo_block_count": pbo["block_count"],
            "pbo_split_count": pbo["split_count"],
            "cpcv_block_count": CPCV_BLOCK_COUNT,
            "cpcv_test_block_count": CPCV_TEST_BLOCK_COUNT,
            "cpcv_path_count": CPCV_PATH_COUNT,
            "cpcv_purge_open_sessions": CPCV_PURGE_OPEN_SESSIONS,
            "cpcv_embargo_open_sessions": CPCV_EMBARGO_OPEN_SESSIONS,
            "annual_open_sessions": ANNUAL_OPEN_SESSIONS,
            "turnover_observation_sessions": SIGNAL_OPEN_SESSIONS,
        },
        "redundancy_clusters": cluster_rows,
        "admission_eligible": admission_eligible,
        "blockers": global_blockers,
        "cost_bps": decimal_text(COST_BPS, label="cost_bps"),
    }


def _evaluate_preregistration(
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    execution_turnover_evidence: Mapping[str, Any] | bytes,
    created_at: str,
) -> dict[str, Any]:
    """Evaluate one immutable cycle after its final label-maturity session."""

    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    captures, rows = _validated_capture_observation_chain(
        signal_captures,
        observations,
        preregistration=prereg,
        selection=selected,
    )
    stamp = canonical_timestamp(created_at, label="created_at")
    execution = validate_execution_turnover_evidence(
        execution_turnover_evidence,
        preregistration=prereg,
        selection=selected,
    )
    if stamp < execution["created_at"]:
        raise FactorGovernanceError("evaluation predates its execution evidence")
    payload = _evaluation_payload(
        preregistration=prereg,
        selection=selected,
        signal_captures=captures,
        observations=rows,
        execution_turnover_evidence=execution,
    )
    artifact = seal_artifact(EVALUATION_KIND, payload, created_at=stamp)
    if len(canonical_json_bytes(artifact)) > _MAX_EVALUATION_BYTES:
        raise FactorGovernanceError("prospective evaluation exceeds its byte limit")
    return artifact


def validate_preregistration_evaluation(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    execution_turnover_evidence: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    envelope, payload = exact_payload(document, kind=EVALUATION_KIND, fields=_EVALUATION_FIELDS)
    if type(payload["candidate_rows"]) is not list or any(
        type(row) is not dict or set(row) != _CANDIDATE_ROW_FIELDS
        for row in payload["candidate_rows"]
    ):
        raise FactorGovernanceError("evaluation candidate row fields are not exact")
    if len(canonical_json_bytes(envelope)) > _MAX_EVALUATION_BYTES:
        raise FactorGovernanceError("prospective evaluation exceeds its byte limit")
    expected = _evaluate_preregistration(
        preregistration=preregistration,
        selection=selection,
        signal_captures=signal_captures,
        observations=observations,
        execution_turnover_evidence=execution_turnover_evidence,
        created_at=envelope["created_at"],
    )
    if envelope != expected:
        raise FactorGovernanceError("prospective evaluation does not replay exactly")
    return envelope


def _admitted_payload(
    *,
    evaluation: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_payload = evaluation["payload"]
    if evaluation_payload["admission_eligible"] is not True:
        raise FactorGovernanceError("evaluation has no prospectively admitted factor")
    admitted = [
        row
        for row in evaluation_payload["candidate_rows"]
        if row["admission_eligible"] is True and row["cluster_representative"] is True
    ]
    if not 1 <= len(admitted) <= MAX_ACTIVE_FACTORS:
        raise FactorGovernanceError("evaluation admitted set must contain one to ten factors")
    weights = largest_remainder_weights(
        {row["factor_id"]: Decimal(row["shrunk_ic"]) for row in admitted}
    )
    evaluation_reference = artifact_ref(evaluation)
    factor_rows = [
        {
            "factor_id": row["factor_id"],
            "configuration_id": row["configuration_id"],
            "family": row["family"],
            "primitive": row["primitive"],
            "evaluation_ref": evaluation_reference,
            "valid_daily_rankic_sessions": row["valid_daily_rankic_sessions"],
            "mean_rank_ic": row["mean_rank_ic"],
            "mean_purged_oos_rank_ic": row["mean_purged_oos_rank_ic"],
            "shrunk_ic": row["shrunk_ic"],
            "weight": weights[row["factor_id"]],
            "admission_route": PROSPECTIVE_ADMISSION_ROUTE,
        }
        for row in sorted(admitted, key=lambda item: item["factor_id"].encode("utf-8"))
    ]
    identity_inputs = {
        "evaluation_id": evaluation_payload["evaluation_id"],
        "factor_rows": factor_rows,
    }
    return {
        "admitted_set_id": business_identity("admitted-set", identity_inputs),
        "lane": PROSPECTIVE_LANE,
        "preregistration_id": preregistration["payload"]["preregistration_id"],
        "selection_id": selection["payload"]["selection_id"],
        "evaluation_id": evaluation_payload["evaluation_id"],
        "factor_rows": factor_rows,
        "weight_total": "1.000000000000",
        "weighting_method": "SHRUNK_IC_PLUS_10_LARGEST_REMAINDER",
        "activation_authorized": False,
    }


def _build_admitted_factor_set(
    *,
    evaluation: Mapping[str, Any] | bytes,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    execution_turnover_evidence: Mapping[str, Any] | bytes,
    created_at: str,
) -> dict[str, Any]:
    """Build a prospective set while keeping System activation separate."""

    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    evaluated = validate_preregistration_evaluation(
        evaluation,
        preregistration=prereg,
        selection=selected,
        signal_captures=signal_captures,
        observations=observations,
        execution_turnover_evidence=execution_turnover_evidence,
    )
    stamp = canonical_timestamp(created_at, label="created_at")
    if stamp < evaluated["created_at"]:
        raise FactorGovernanceError("admitted set predates its evaluation")
    payload = _admitted_payload(evaluation=evaluated, preregistration=prereg, selection=selected)
    artifact = seal_artifact(ADMITTED_SET_KIND, payload, created_at=stamp)
    if len(canonical_json_bytes(artifact)) > _MAX_ADMITTED_SET_BYTES:
        raise FactorGovernanceError("admitted Factor set exceeds its byte limit")
    return artifact


def validate_admitted_factor_set(
    document: Mapping[str, Any] | bytes,
    *,
    evaluation: Mapping[str, Any] | bytes,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    execution_turnover_evidence: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    envelope, _ = exact_payload(document, kind=ADMITTED_SET_KIND, fields=_ADMITTED_SET_FIELDS)
    if len(canonical_json_bytes(envelope)) > _MAX_ADMITTED_SET_BYTES:
        raise FactorGovernanceError("admitted Factor set exceeds its byte limit")
    expected = _build_admitted_factor_set(
        evaluation=evaluation,
        preregistration=preregistration,
        selection=selection,
        signal_captures=signal_captures,
        observations=observations,
        execution_turnover_evidence=execution_turnover_evidence,
        created_at=envelope["created_at"],
    )
    if envelope != expected:
        raise FactorGovernanceError("admitted factor set does not replay exactly")
    return envelope


__all__ = [
    "ADMITTED_SET_KIND",
    "EVALUATION_KIND",
    "PROSPECTIVE_ADMISSION_ROUTE",
    "validate_admitted_factor_set",
    "validate_preregistration_evaluation",
]
