"""Offline factor portfolio-contribution helpers.

The helpers in this module evaluate factor-return contribution against a local
baseline factor pool. They are not live portfolio optimizers and do not connect
research factors to stock selection, ``PortfolioConstructor``, or ``RiskGuard``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import SingleFactorBacktestRun
from quant_investor.factors.correlation import FactorRedundancyReport
from quant_investor.factors.metrics import (
    ReturnMetricSummary,
    TurnoverMetricSummary,
    build_return_metric_summary,
    build_turnover_metric_summary,
)
from quant_investor.versioning import FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION


CONTRIBUTION_VERDICT_IMPROVES = "improves"
CONTRIBUTION_VERDICT_NEUTRAL = "neutral"
CONTRIBUTION_VERDICT_DEGRADES = "degrades"
CONTRIBUTION_VERDICT_INSUFFICIENT_DATA = "insufficient_data"

CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE = "low_incremental_sharpe"
CONTRIBUTION_ISSUE_DRAWDOWN_DEGRADATION = "drawdown_degradation"
CONTRIBUTION_ISSUE_TURNOVER_INCREASE = "turnover_increase"
CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP = "insufficient_overlap"
CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN = "negative_incremental_return"

SUPPORTED_CONTRIBUTION_VERDICTS = {
    CONTRIBUTION_VERDICT_IMPROVES,
    CONTRIBUTION_VERDICT_NEUTRAL,
    CONTRIBUTION_VERDICT_DEGRADES,
    CONTRIBUTION_VERDICT_INSUFFICIENT_DATA,
}

_EPSILON = 1e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_json_serializable(value: Any, label: str) -> Any:
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_float(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(
        [_json_safe(part) for part in parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _to_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _metric_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


@dataclass
class FactorContributionConfig:
    schema_version: str = FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
    config_id: str = "default-factor-portfolio-contribution-v1"
    candidate_weight: float = 0.20
    baseline_weight: float = 0.80
    min_overlap_days: int = 60
    min_incremental_sharpe: float = 0.05
    min_incremental_annualized_return: float = 0.0
    max_drawdown_degradation: float = 0.05
    max_turnover_increase: float | None = None
    normalize_weights: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        )
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.candidate_weight = _non_negative_float(self.candidate_weight, "candidate_weight")
        self.baseline_weight = _non_negative_float(self.baseline_weight, "baseline_weight")
        if not self.normalize_weights and self.candidate_weight + self.baseline_weight <= 0.0:
            raise ValueError("candidate_weight + baseline_weight must be positive.")
        self.min_overlap_days = _non_negative_int(self.min_overlap_days, "min_overlap_days")
        self.min_incremental_sharpe = _finite_float(
            self.min_incremental_sharpe,
            "min_incremental_sharpe",
        )
        self.min_incremental_annualized_return = _finite_float(
            self.min_incremental_annualized_return,
            "min_incremental_annualized_return",
        )
        self.max_drawdown_degradation = _finite_float(
            self.max_drawdown_degradation,
            "max_drawdown_degradation",
        )
        self.max_turnover_increase = _optional_non_negative_float(
            self.max_turnover_increase,
            "max_turnover_increase",
        )
        self.normalize_weights = bool(self.normalize_weights)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorContributionConfig":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION)
            ),
            config_id=str(data.get("config_id", "default-factor-portfolio-contribution-v1")),
            candidate_weight=float(data.get("candidate_weight", 0.20)),
            baseline_weight=float(data.get("baseline_weight", 0.80)),
            min_overlap_days=int(data.get("min_overlap_days", 60)),
            min_incremental_sharpe=float(data.get("min_incremental_sharpe", 0.05)),
            min_incremental_annualized_return=float(
                data.get("min_incremental_annualized_return", 0.0)
            ),
            max_drawdown_degradation=float(data.get("max_drawdown_degradation", 0.05)),
            max_turnover_increase=data.get("max_turnover_increase"),
            normalize_weights=bool(data.get("normalize_weights", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorPoolReturnSeries:
    schema_version: str = FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
    series_id: str = ""
    name: str = ""
    returns_by_date: dict[str, float] = field(default_factory=dict)
    turnover_by_date: dict[str, float] = field(default_factory=dict)
    source_run_ids: list[str] = field(default_factory=list)
    weights_by_source: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        )
        self.series_id = _non_empty_str(self.series_id, "series_id")
        self.name = _non_empty_str(self.name, "name")
        self.returns_by_date = {
            str(date_key): _finite_float(value, "returns_by_date")
            for date_key, value in sorted(self.returns_by_date.items())
        }
        self.turnover_by_date = {
            str(date_key): _non_negative_float(value, "turnover_by_date")
            for date_key, value in sorted(self.turnover_by_date.items())
        }
        self.source_run_ids = _ordered_unique(self.source_run_ids)
        self.weights_by_source = {
            str(source): _finite_float(weight, "weights_by_source")
            for source, weight in sorted(self.weights_by_source.items())
            if str(source).strip()
        }
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorPoolReturnSeries":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION)
            ),
            series_id=str(data.get("series_id", "")),
            name=str(data.get("name", "")),
            returns_by_date=dict(data.get("returns_by_date", {}) or {}),
            turnover_by_date=dict(data.get("turnover_by_date", {}) or {}),
            source_run_ids=list(data.get("source_run_ids", []) or []),
            weights_by_source=dict(data.get("weights_by_source", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorPortfolioContributionReport:
    schema_version: str = FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
    report_id: str = ""
    candidate_factor_id: str | None = None
    candidate_factor_version: str | None = None
    candidate_run_id: str | None = None
    generated_at: str = ""
    config: FactorContributionConfig = field(default_factory=FactorContributionConfig)
    baseline_series: FactorPoolReturnSeries = field(default_factory=FactorPoolReturnSeries)
    candidate_series: FactorPoolReturnSeries = field(default_factory=FactorPoolReturnSeries)
    combined_series: FactorPoolReturnSeries = field(default_factory=FactorPoolReturnSeries)
    baseline_metrics: ReturnMetricSummary = field(default_factory=ReturnMetricSummary)
    candidate_metrics: ReturnMetricSummary = field(default_factory=ReturnMetricSummary)
    combined_metrics: ReturnMetricSummary = field(default_factory=ReturnMetricSummary)
    baseline_turnover_metrics: TurnoverMetricSummary = field(default_factory=TurnoverMetricSummary)
    combined_turnover_metrics: TurnoverMetricSummary = field(default_factory=TurnoverMetricSummary)
    overlap_days: int = 0
    incremental_annualized_return: float | None = None
    incremental_sharpe: float | None = None
    incremental_max_drawdown: float | None = None
    incremental_turnover: float | None = None
    issue_codes: list[str] = field(default_factory=list)
    verdict: str = CONTRIBUTION_VERDICT_INSUFFICIENT_DATA
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        )
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.candidate_factor_id = _optional_str(self.candidate_factor_id)
        self.candidate_factor_version = _optional_str(self.candidate_factor_version)
        self.candidate_run_id = _optional_str(self.candidate_run_id)
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        if not isinstance(self.config, FactorContributionConfig):
            self.config = FactorContributionConfig.from_dict(self.config)
        if not isinstance(self.baseline_series, FactorPoolReturnSeries):
            self.baseline_series = FactorPoolReturnSeries.from_dict(self.baseline_series)
        if not isinstance(self.candidate_series, FactorPoolReturnSeries):
            self.candidate_series = FactorPoolReturnSeries.from_dict(self.candidate_series)
        if not isinstance(self.combined_series, FactorPoolReturnSeries):
            self.combined_series = FactorPoolReturnSeries.from_dict(self.combined_series)
        if not isinstance(self.baseline_metrics, ReturnMetricSummary):
            self.baseline_metrics = ReturnMetricSummary.from_dict(self.baseline_metrics)
        if not isinstance(self.candidate_metrics, ReturnMetricSummary):
            self.candidate_metrics = ReturnMetricSummary.from_dict(self.candidate_metrics)
        if not isinstance(self.combined_metrics, ReturnMetricSummary):
            self.combined_metrics = ReturnMetricSummary.from_dict(self.combined_metrics)
        if not isinstance(self.baseline_turnover_metrics, TurnoverMetricSummary):
            self.baseline_turnover_metrics = TurnoverMetricSummary.from_dict(
                self.baseline_turnover_metrics
            )
        if not isinstance(self.combined_turnover_metrics, TurnoverMetricSummary):
            self.combined_turnover_metrics = TurnoverMetricSummary.from_dict(
                self.combined_turnover_metrics
            )
        self.overlap_days = _non_negative_int(self.overlap_days, "overlap_days")
        self.incremental_annualized_return = _optional_finite_float(
            self.incremental_annualized_return,
            "incremental_annualized_return",
        )
        self.incremental_sharpe = _optional_finite_float(
            self.incremental_sharpe,
            "incremental_sharpe",
        )
        self.incremental_max_drawdown = _optional_finite_float(
            self.incremental_max_drawdown,
            "incremental_max_drawdown",
        )
        self.incremental_turnover = _optional_finite_float(
            self.incremental_turnover,
            "incremental_turnover",
        )
        self.issue_codes = _ordered_unique(self.issue_codes)
        if self.verdict not in SUPPORTED_CONTRIBUTION_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(SUPPORTED_CONTRIBUTION_VERDICTS)}; "
                f"got {self.verdict!r}."
            )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "candidate_factor_id": self.candidate_factor_id,
            "candidate_factor_version": self.candidate_factor_version,
            "candidate_run_id": self.candidate_run_id,
            "generated_at": self.generated_at,
            "config": self.config.to_dict(),
            "baseline_series": self.baseline_series.to_dict(),
            "candidate_series": self.candidate_series.to_dict(),
            "combined_series": self.combined_series.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "combined_metrics": self.combined_metrics.to_dict(),
            "baseline_turnover_metrics": self.baseline_turnover_metrics.to_dict(),
            "combined_turnover_metrics": self.combined_turnover_metrics.to_dict(),
            "overlap_days": self.overlap_days,
            "incremental_annualized_return": self.incremental_annualized_return,
            "incremental_sharpe": self.incremental_sharpe,
            "incremental_max_drawdown": self.incremental_max_drawdown,
            "incremental_turnover": self.incremental_turnover,
            "issue_codes": list(self.issue_codes),
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorPortfolioContributionReport":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION)
            ),
            report_id=str(data.get("report_id", "")),
            candidate_factor_id=data.get("candidate_factor_id"),
            candidate_factor_version=data.get("candidate_factor_version"),
            candidate_run_id=data.get("candidate_run_id"),
            generated_at=str(data.get("generated_at", "")),
            config=FactorContributionConfig.from_dict(dict(data.get("config", {}) or {})),
            baseline_series=FactorPoolReturnSeries.from_dict(
                dict(data.get("baseline_series", {}) or {})
            ),
            candidate_series=FactorPoolReturnSeries.from_dict(
                dict(data.get("candidate_series", {}) or {})
            ),
            combined_series=FactorPoolReturnSeries.from_dict(
                dict(data.get("combined_series", {}) or {})
            ),
            baseline_metrics=ReturnMetricSummary.from_dict(
                dict(data.get("baseline_metrics", {}) or {})
            ),
            candidate_metrics=ReturnMetricSummary.from_dict(
                dict(data.get("candidate_metrics", {}) or {})
            ),
            combined_metrics=ReturnMetricSummary.from_dict(
                dict(data.get("combined_metrics", {}) or {})
            ),
            baseline_turnover_metrics=TurnoverMetricSummary.from_dict(
                dict(data.get("baseline_turnover_metrics", {}) or {})
            ),
            combined_turnover_metrics=TurnoverMetricSummary.from_dict(
                dict(data.get("combined_turnover_metrics", {}) or {})
            ),
            overlap_days=int(data.get("overlap_days", 0)),
            incremental_annualized_return=data.get("incremental_annualized_return"),
            incremental_sharpe=data.get("incremental_sharpe"),
            incremental_max_drawdown=data.get("incremental_max_drawdown"),
            incremental_turnover=data.get("incremental_turnover"),
            issue_codes=list(data.get("issue_codes", []) or []),
            verdict=str(data.get("verdict", CONTRIBUTION_VERDICT_INSUFFICIENT_DATA)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_contribution_config_id(config: FactorContributionConfig) -> str:
    parts = [
        config.schema_version,
        config.candidate_weight,
        config.baseline_weight,
        config.min_overlap_days,
        config.min_incremental_sharpe,
        config.min_incremental_annualized_return,
        config.max_drawdown_degradation,
        config.max_turnover_increase,
        config.normalize_weights,
        config.metadata,
    ]
    return f"factor-contribution-config-{_short_hash(parts)}"


def make_factor_pool_series_id(*, name: str, source_run_ids: Sequence[str]) -> str:
    resolved_name = _non_empty_str(name, "name")
    sources = _ordered_unique(source_run_ids)
    parts = [resolved_name, sources]
    return f"factor-pool-series-{_slug(resolved_name)}-{_short_hash(parts)}"


def make_portfolio_contribution_report_id(
    *,
    candidate_run_id: str | None,
    generated_at: str,
) -> str:
    parts = [candidate_run_id, generated_at]
    return f"factor-contribution-report-{_slug(candidate_run_id)}-{_short_hash(parts)}"


def _run_return_and_turnover_series(
    run: SingleFactorBacktestRun,
    *,
    use_after_cost: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    return_field = "after_cost_return" if use_after_cost else "long_short_return"
    returns: dict[str, float] = {}
    turnovers: dict[str, float] = {}
    for record in run.daily_records:
        date_key = str(record.date or record.execution_end_date)
        return_value = _to_finite_float(getattr(record, return_field))
        turnover_value = _to_finite_float(record.turnover)
        if return_value is not None:
            returns[date_key] = return_value
        if turnover_value is not None:
            turnovers[date_key] = max(0.0, turnover_value)
    return (
        {date_key: returns[date_key] for date_key in sorted(returns)},
        {date_key: turnovers[date_key] for date_key in sorted(turnovers)},
    )


def build_factor_pool_return_series(
    runs: Sequence[SingleFactorBacktestRun],
    *,
    name: str,
    weights_by_run_id: Mapping[str, float] | None = None,
    use_after_cost: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> FactorPoolReturnSeries:
    resolved_runs = list(runs)
    source_ids = _ordered_unique([run.run_id for run in resolved_runs])
    if weights_by_run_id is None:
        equal_weight = 0.0 if not source_ids else 1.0 / len(source_ids)
        weights = {run_id: equal_weight for run_id in source_ids}
    else:
        weights = {
            str(run_id): _non_negative_float(weight, "weights_by_run_id")
            for run_id, weight in sorted(weights_by_run_id.items())
        }

    return_by_run: dict[str, dict[str, float]] = {}
    turnover_by_run: dict[str, dict[str, float]] = {}
    all_dates: set[str] = set()
    for run in resolved_runs:
        returns, turnovers = _run_return_and_turnover_series(run, use_after_cost=use_after_cost)
        return_by_run[run.run_id] = returns
        turnover_by_run[run.run_id] = turnovers
        all_dates.update(returns)

    returns_by_date: dict[str, float] = {}
    turnover_by_date: dict[str, float] = {}
    for date_key in sorted(all_dates):
        available_returns: list[tuple[float, float]] = []
        available_turnovers: list[tuple[float, float]] = []
        for run_id in source_ids:
            if date_key not in return_by_run.get(run_id, {}):
                continue
            weight = weights.get(run_id, 0.0)
            if weight <= 0.0:
                continue
            available_returns.append((return_by_run[run_id][date_key], weight))
            if date_key in turnover_by_run.get(run_id, {}):
                available_turnovers.append((turnover_by_run[run_id][date_key], weight))
        total_weight = sum(weight for _value, weight in available_returns)
        if total_weight <= _EPSILON:
            continue
        returns_by_date[date_key] = sum(
            value * weight for value, weight in available_returns
        ) / total_weight
        turnover_weight = sum(weight for _value, weight in available_turnovers)
        if turnover_weight > _EPSILON:
            turnover_by_date[date_key] = sum(
                value * weight for value, weight in available_turnovers
            ) / turnover_weight

    series_metadata = {
        **_coerce_metadata(metadata),
        "factor_portfolio_contribution_schema_version": (
            FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        ),
        "offline_only": True,
        "use_after_cost": use_after_cost,
    }
    return FactorPoolReturnSeries(
        series_id=make_factor_pool_series_id(name=name, source_run_ids=source_ids),
        name=name,
        returns_by_date=returns_by_date,
        turnover_by_date=turnover_by_date,
        source_run_ids=source_ids,
        weights_by_source=weights,
        metadata=series_metadata,
    )


def _slice_pool_series(
    series: FactorPoolReturnSeries,
    *,
    dates: Sequence[str],
) -> FactorPoolReturnSeries:
    date_set = set(dates)
    return FactorPoolReturnSeries(
        series_id=series.series_id,
        name=series.name,
        returns_by_date={
            date_key: value
            for date_key, value in series.returns_by_date.items()
            if date_key in date_set
        },
        turnover_by_date={
            date_key: value
            for date_key, value in series.turnover_by_date.items()
            if date_key in date_set
        },
        source_run_ids=list(series.source_run_ids),
        weights_by_source=dict(series.weights_by_source),
        metadata=dict(series.metadata),
    )


def combine_candidate_with_baseline(
    baseline: FactorPoolReturnSeries,
    candidate: FactorPoolReturnSeries,
    config: FactorContributionConfig,
) -> FactorPoolReturnSeries:
    common_dates = sorted(set(baseline.returns_by_date) & set(candidate.returns_by_date))
    baseline_weight = config.baseline_weight
    candidate_weight = config.candidate_weight
    if config.normalize_weights:
        total_weight = baseline_weight + candidate_weight
        if total_weight > _EPSILON:
            baseline_weight = baseline_weight / total_weight
            candidate_weight = candidate_weight / total_weight
        else:
            baseline_weight = 0.0
            candidate_weight = 0.0

    returns_by_date = {
        date_key: (
            baseline_weight * baseline.returns_by_date[date_key]
            + candidate_weight * candidate.returns_by_date[date_key]
        )
        for date_key in common_dates
    }
    turnover_by_date = {
        date_key: (
            baseline_weight * baseline.turnover_by_date.get(date_key, 0.0)
            + candidate_weight * candidate.turnover_by_date.get(date_key, 0.0)
        )
        for date_key in common_dates
    }
    weights_by_source: dict[str, float] = {}
    for source, weight in baseline.weights_by_source.items():
        weights_by_source[source] = baseline_weight * weight
    for source, weight in candidate.weights_by_source.items():
        weights_by_source[source] = weights_by_source.get(source, 0.0) + candidate_weight * weight

    source_ids = _ordered_unique([*baseline.source_run_ids, *candidate.source_run_ids])
    metadata = {
        "factor_portfolio_contribution_schema_version": (
            FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        ),
        "offline_only": True,
        "analysis_type": "factor_return_contribution",
        "not_live_portfolio_optimizer": True,
        "baseline_series_id": baseline.series_id,
        "candidate_series_id": candidate.series_id,
    }
    return FactorPoolReturnSeries(
        series_id=make_factor_pool_series_id(
            name="baseline_plus_candidate",
            source_run_ids=source_ids,
        ),
        name="baseline_plus_candidate",
        returns_by_date=returns_by_date,
        turnover_by_date=turnover_by_date,
        source_run_ids=source_ids,
        weights_by_source=weights_by_source,
        metadata=metadata,
    )


def _returns_for_dates(series: FactorPoolReturnSeries, dates: Sequence[str]) -> list[float | None]:
    return [series.returns_by_date.get(date_key) for date_key in dates]


def _turnover_for_dates(series: FactorPoolReturnSeries, dates: Sequence[str]) -> list[float]:
    return [
        series.turnover_by_date[date_key]
        for date_key in dates
        if date_key in series.turnover_by_date
    ]


def build_factor_portfolio_contribution_report(
    *,
    candidate_run: SingleFactorBacktestRun,
    baseline_runs: Sequence[SingleFactorBacktestRun],
    config: FactorContributionConfig | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorPortfolioContributionReport:
    resolved_config = config or FactorContributionConfig()
    baseline_full = build_factor_pool_return_series(
        baseline_runs,
        name="baseline_factor_pool",
        use_after_cost=True,
        metadata={"role": "baseline"},
    )
    candidate_full = build_factor_pool_return_series(
        [candidate_run],
        name="candidate_factor",
        use_after_cost=True,
        metadata={"role": "candidate"},
    )
    common_dates = sorted(
        set(baseline_full.returns_by_date) & set(candidate_full.returns_by_date)
    )
    baseline_series = _slice_pool_series(baseline_full, dates=common_dates)
    candidate_series = _slice_pool_series(candidate_full, dates=common_dates)
    combined_series = combine_candidate_with_baseline(
        baseline_series,
        candidate_series,
        resolved_config,
    )

    baseline_metrics = build_return_metric_summary(
        "baseline_factor_pool",
        _returns_for_dates(baseline_series, common_dates),
        metadata={"analysis_type": "factor_return_contribution"},
    )
    candidate_metrics = build_return_metric_summary(
        "candidate_factor",
        _returns_for_dates(candidate_series, common_dates),
        metadata={"analysis_type": "factor_return_contribution"},
    )
    combined_metrics = build_return_metric_summary(
        "baseline_plus_candidate",
        _returns_for_dates(combined_series, common_dates),
        metadata={"analysis_type": "factor_return_contribution"},
    )
    baseline_turnover_metrics = build_turnover_metric_summary(
        "baseline_factor_pool",
        _turnover_for_dates(baseline_series, common_dates),
    )
    combined_turnover_metrics = build_turnover_metric_summary(
        "baseline_plus_candidate",
        _turnover_for_dates(combined_series, common_dates),
    )

    incremental_annualized_return = _metric_delta(
        combined_metrics.annualized_return,
        baseline_metrics.annualized_return,
    )
    incremental_sharpe = _metric_delta(combined_metrics.sharpe, baseline_metrics.sharpe)
    incremental_max_drawdown = _metric_delta(
        combined_metrics.max_drawdown,
        baseline_metrics.max_drawdown,
    )
    incremental_turnover = _metric_delta(
        combined_turnover_metrics.average_turnover,
        baseline_turnover_metrics.average_turnover,
    )

    issue_codes: list[str] = []
    overlap_days = len(common_dates)
    if overlap_days < resolved_config.min_overlap_days:
        issue_codes.append(CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP)
    if (
        incremental_sharpe is not None
        and incremental_sharpe < resolved_config.min_incremental_sharpe
    ):
        issue_codes.append(CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE)
    if (
        incremental_annualized_return is not None
        and incremental_annualized_return < resolved_config.min_incremental_annualized_return
    ):
        issue_codes.append(CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN)
    if (
        incremental_max_drawdown is not None
        and incremental_max_drawdown > resolved_config.max_drawdown_degradation
    ):
        issue_codes.append(CONTRIBUTION_ISSUE_DRAWDOWN_DEGRADATION)
    if (
        resolved_config.max_turnover_increase is not None
        and incremental_turnover is not None
        and incremental_turnover > resolved_config.max_turnover_increase
    ):
        issue_codes.append(CONTRIBUTION_ISSUE_TURNOVER_INCREASE)

    if CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP in issue_codes:
        verdict = CONTRIBUTION_VERDICT_INSUFFICIENT_DATA
    elif (
        CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN in issue_codes
        or CONTRIBUTION_ISSUE_DRAWDOWN_DEGRADATION in issue_codes
    ):
        verdict = CONTRIBUTION_VERDICT_DEGRADES
    elif (
        CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE in issue_codes
        or CONTRIBUTION_ISSUE_TURNOVER_INCREASE in issue_codes
    ):
        verdict = CONTRIBUTION_VERDICT_NEUTRAL
    else:
        verdict = CONTRIBUTION_VERDICT_IMPROVES

    report_metadata = {
        **_coerce_metadata(metadata),
        "factor_portfolio_contribution_schema_version": (
            FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION
        ),
        "analysis_type": "factor_return_contribution",
        "not_live_portfolio_optimizer": True,
        "offline_only": True,
    }
    return FactorPortfolioContributionReport(
        report_id=make_portfolio_contribution_report_id(
            candidate_run_id=candidate_run.run_id,
            generated_at=generated_at,
        ),
        candidate_factor_id=candidate_run.factor_id,
        candidate_factor_version=candidate_run.factor_version,
        candidate_run_id=candidate_run.run_id,
        generated_at=generated_at,
        config=resolved_config,
        baseline_series=baseline_series,
        candidate_series=candidate_series,
        combined_series=combined_series,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        combined_metrics=combined_metrics,
        baseline_turnover_metrics=baseline_turnover_metrics,
        combined_turnover_metrics=combined_turnover_metrics,
        overlap_days=overlap_days,
        incremental_annualized_return=incremental_annualized_return,
        incremental_sharpe=incremental_sharpe,
        incremental_max_drawdown=incremental_max_drawdown,
        incremental_turnover=incremental_turnover,
        issue_codes=issue_codes,
        verdict=verdict,
        metadata=report_metadata,
    )


def build_incremental_factor_validation_snapshot(
    *,
    redundancy_report: FactorRedundancyReport | None = None,
    contribution_report: FactorPortfolioContributionReport | None = None,
) -> dict[str, Any]:
    issue_codes = _ordered_unique([
        *(redundancy_report.issue_codes if redundancy_report is not None else []),
        *(contribution_report.issue_codes if contribution_report is not None else []),
    ])
    snapshot = {
        "max_abs_return_correlation": (
            redundancy_report.max_abs_return_correlation
            if redundancy_report is not None
            else None
        ),
        "max_abs_matrix_rank_correlation": (
            redundancy_report.max_abs_matrix_rank_correlation
            if redundancy_report is not None
            else None
        ),
        "max_abs_ic_correlation": (
            redundancy_report.max_abs_ic_correlation
            if redundancy_report is not None
            else None
        ),
        "redundant_factor_ids": (
            list(redundancy_report.redundant_factor_ids)
            if redundancy_report is not None
            else []
        ),
        "related_factor_ids": (
            list(redundancy_report.related_factor_ids)
            if redundancy_report is not None
            else []
        ),
        "redundancy_verdict": (
            redundancy_report.overall_verdict if redundancy_report is not None else None
        ),
        "contribution_verdict": (
            contribution_report.verdict if contribution_report is not None else None
        ),
        "incremental_annualized_return": (
            contribution_report.incremental_annualized_return
            if contribution_report is not None
            else None
        ),
        "incremental_sharpe": (
            contribution_report.incremental_sharpe if contribution_report is not None else None
        ),
        "incremental_max_drawdown": (
            contribution_report.incremental_max_drawdown
            if contribution_report is not None
            else None
        ),
        "incremental_turnover": (
            contribution_report.incremental_turnover if contribution_report is not None else None
        ),
        "issue_codes": issue_codes,
    }
    return dict(_ensure_json_serializable(snapshot, "incremental_factor_validation_snapshot"))


__all__ = [
    "CONTRIBUTION_VERDICT_IMPROVES",
    "CONTRIBUTION_VERDICT_NEUTRAL",
    "CONTRIBUTION_VERDICT_DEGRADES",
    "CONTRIBUTION_VERDICT_INSUFFICIENT_DATA",
    "CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE",
    "CONTRIBUTION_ISSUE_DRAWDOWN_DEGRADATION",
    "CONTRIBUTION_ISSUE_TURNOVER_INCREASE",
    "CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP",
    "CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN",
    "FactorContributionConfig",
    "FactorPoolReturnSeries",
    "FactorPortfolioContributionReport",
    "make_contribution_config_id",
    "make_factor_pool_series_id",
    "make_portfolio_contribution_report_id",
    "build_factor_pool_return_series",
    "combine_candidate_with_baseline",
    "build_factor_portfolio_contribution_report",
    "build_incremental_factor_validation_snapshot",
]
