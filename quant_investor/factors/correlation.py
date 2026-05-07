"""Offline factor correlation, redundancy, and residual contribution helpers.

This module compares already-produced research artifacts only. It does not fetch
market data, approve factors, or wire factors into stock selection, portfolio
construction, or risk controls.
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
from quant_investor.factors.matrix import FactorMatrix
from quant_investor.versioning import FACTOR_CORRELATION_SCHEMA_VERSION


CORRELATION_VERDICT_DISTINCT = "distinct"
CORRELATION_VERDICT_RELATED = "related"
CORRELATION_VERDICT_REDUNDANT = "redundant"
CORRELATION_VERDICT_INSUFFICIENT_DATA = "insufficient_data"

CORRELATION_ISSUE_HIGH_RETURN_CORRELATION = "high_return_correlation"
CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION = "high_matrix_correlation"
CORRELATION_ISSUE_HIGH_IC_CORRELATION = "high_ic_correlation"
CORRELATION_ISSUE_LOW_RESIDUAL_RETURN = "low_residual_return"
CORRELATION_ISSUE_INSUFFICIENT_OVERLAP = "insufficient_overlap"

SUPPORTED_CORRELATION_VERDICTS = {
    CORRELATION_VERDICT_DISTINCT,
    CORRELATION_VERDICT_RELATED,
    CORRELATION_VERDICT_REDUNDANT,
    CORRELATION_VERDICT_INSUFFICIENT_DATA,
}

HIGH_CORRELATION_ISSUES = {
    CORRELATION_ISSUE_HIGH_RETURN_CORRELATION,
    CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION,
    CORRELATION_ISSUE_HIGH_IC_CORRELATION,
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


def _unit_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, field_name)


def _correlation_or_none(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and not -1.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1]; got {value!r}.")
    return number


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


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _breaches_correlation_threshold(
    value: float | None,
    *,
    threshold: float,
    use_abs: bool,
) -> bool:
    if value is None:
        return False
    comparable = abs(value) if use_abs else value
    return comparable >= threshold


def _factor_id_from(
    run: SingleFactorBacktestRun | None,
    matrix: FactorMatrix | None,
) -> str | None:
    if run is not None and run.factor_id is not None:
        return run.factor_id
    if matrix is not None:
        return matrix.factor_id
    return None


def _factor_version_from(
    run: SingleFactorBacktestRun | None,
    matrix: FactorMatrix | None,
) -> str | None:
    if run is not None and run.factor_version is not None:
        return run.factor_version
    if matrix is not None:
        return matrix.factor_version
    return None


@dataclass
class FactorCorrelationConfig:
    schema_version: str = FACTOR_CORRELATION_SCHEMA_VERSION
    config_id: str = "default-factor-correlation-v1"
    max_return_correlation: float = 0.70
    max_matrix_rank_correlation: float = 0.70
    max_ic_correlation: float = 0.70
    min_overlap_days: int = 60
    min_residual_mean_return: float = 0.0
    use_abs_correlation: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_CORRELATION_SCHEMA_VERSION)
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.max_return_correlation = _unit_float(
            self.max_return_correlation,
            "max_return_correlation",
        )
        self.max_matrix_rank_correlation = _unit_float(
            self.max_matrix_rank_correlation,
            "max_matrix_rank_correlation",
        )
        self.max_ic_correlation = _unit_float(self.max_ic_correlation, "max_ic_correlation")
        self.min_overlap_days = _non_negative_int(self.min_overlap_days, "min_overlap_days")
        self.min_residual_mean_return = _finite_float(
            self.min_residual_mean_return,
            "min_residual_mean_return",
        )
        self.use_abs_correlation = bool(self.use_abs_correlation)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorCorrelationConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_CORRELATION_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "default-factor-correlation-v1")),
            max_return_correlation=float(data.get("max_return_correlation", 0.70)),
            max_matrix_rank_correlation=float(data.get("max_matrix_rank_correlation", 0.70)),
            max_ic_correlation=float(data.get("max_ic_correlation", 0.70)),
            min_overlap_days=int(data.get("min_overlap_days", 60)),
            min_residual_mean_return=float(data.get("min_residual_mean_return", 0.0)),
            use_abs_correlation=bool(data.get("use_abs_correlation", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorCorrelationPair:
    schema_version: str = FACTOR_CORRELATION_SCHEMA_VERSION
    pair_id: str = ""
    candidate_factor_id: str | None = None
    candidate_factor_version: str | None = None
    reference_factor_id: str | None = None
    reference_factor_version: str | None = None
    candidate_run_id: str | None = None
    reference_run_id: str | None = None
    candidate_matrix_id: str | None = None
    reference_matrix_id: str | None = None
    overlap_days: int = 0
    return_correlation: float | None = None
    rank_return_correlation: float | None = None
    matrix_rank_correlation_avg: float | None = None
    ic_correlation: float | None = None
    residual_mean_return: float | None = None
    issue_codes: list[str] = field(default_factory=list)
    verdict: str = CORRELATION_VERDICT_INSUFFICIENT_DATA
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_CORRELATION_SCHEMA_VERSION)
        self.pair_id = _non_empty_str(self.pair_id, "pair_id")
        self.candidate_factor_id = _optional_str(self.candidate_factor_id)
        self.candidate_factor_version = _optional_str(self.candidate_factor_version)
        self.reference_factor_id = _optional_str(self.reference_factor_id)
        self.reference_factor_version = _optional_str(self.reference_factor_version)
        self.candidate_run_id = _optional_str(self.candidate_run_id)
        self.reference_run_id = _optional_str(self.reference_run_id)
        self.candidate_matrix_id = _optional_str(self.candidate_matrix_id)
        self.reference_matrix_id = _optional_str(self.reference_matrix_id)
        self.overlap_days = _non_negative_int(self.overlap_days, "overlap_days")
        self.return_correlation = _correlation_or_none(
            self.return_correlation,
            "return_correlation",
        )
        self.rank_return_correlation = _correlation_or_none(
            self.rank_return_correlation,
            "rank_return_correlation",
        )
        self.matrix_rank_correlation_avg = _correlation_or_none(
            self.matrix_rank_correlation_avg,
            "matrix_rank_correlation_avg",
        )
        self.ic_correlation = _correlation_or_none(self.ic_correlation, "ic_correlation")
        self.residual_mean_return = _optional_finite_float(
            self.residual_mean_return,
            "residual_mean_return",
        )
        if self.verdict not in SUPPORTED_CORRELATION_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(SUPPORTED_CORRELATION_VERDICTS)}; "
                f"got {self.verdict!r}."
            )
        self.issue_codes = _ordered_unique(self.issue_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorCorrelationPair":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_CORRELATION_SCHEMA_VERSION)),
            pair_id=str(data.get("pair_id", "")),
            candidate_factor_id=data.get("candidate_factor_id"),
            candidate_factor_version=data.get("candidate_factor_version"),
            reference_factor_id=data.get("reference_factor_id"),
            reference_factor_version=data.get("reference_factor_version"),
            candidate_run_id=data.get("candidate_run_id"),
            reference_run_id=data.get("reference_run_id"),
            candidate_matrix_id=data.get("candidate_matrix_id"),
            reference_matrix_id=data.get("reference_matrix_id"),
            overlap_days=int(data.get("overlap_days", 0)),
            return_correlation=data.get("return_correlation"),
            rank_return_correlation=data.get("rank_return_correlation"),
            matrix_rank_correlation_avg=data.get("matrix_rank_correlation_avg"),
            ic_correlation=data.get("ic_correlation"),
            residual_mean_return=data.get("residual_mean_return"),
            issue_codes=list(data.get("issue_codes", []) or []),
            verdict=str(data.get("verdict", CORRELATION_VERDICT_INSUFFICIENT_DATA)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorRedundancyReport:
    schema_version: str = FACTOR_CORRELATION_SCHEMA_VERSION
    report_id: str = ""
    candidate_factor_id: str | None = None
    candidate_factor_version: str | None = None
    candidate_run_id: str | None = None
    generated_at: str = ""
    config: FactorCorrelationConfig = field(default_factory=FactorCorrelationConfig)
    pair_results: list[FactorCorrelationPair] = field(default_factory=list)
    max_abs_return_correlation: float | None = None
    max_abs_matrix_rank_correlation: float | None = None
    max_abs_ic_correlation: float | None = None
    redundant_factor_ids: list[str] = field(default_factory=list)
    related_factor_ids: list[str] = field(default_factory=list)
    overall_verdict: str = CORRELATION_VERDICT_INSUFFICIENT_DATA
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_CORRELATION_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.candidate_factor_id = _optional_str(self.candidate_factor_id)
        self.candidate_factor_version = _optional_str(self.candidate_factor_version)
        self.candidate_run_id = _optional_str(self.candidate_run_id)
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        if not isinstance(self.config, FactorCorrelationConfig):
            self.config = FactorCorrelationConfig.from_dict(self.config)
        self.pair_results = [
            pair
            if isinstance(pair, FactorCorrelationPair)
            else FactorCorrelationPair.from_dict(pair)
            for pair in self.pair_results
        ]
        self.pair_results = sorted(
            self.pair_results,
            key=lambda pair: (
                pair.reference_factor_id or "",
                pair.reference_factor_version or "",
                pair.reference_run_id or "",
                pair.reference_matrix_id or "",
            ),
        )
        self.max_abs_return_correlation = _unit_float_or_none(
            self.max_abs_return_correlation,
            "max_abs_return_correlation",
        )
        self.max_abs_matrix_rank_correlation = _unit_float_or_none(
            self.max_abs_matrix_rank_correlation,
            "max_abs_matrix_rank_correlation",
        )
        self.max_abs_ic_correlation = _unit_float_or_none(
            self.max_abs_ic_correlation,
            "max_abs_ic_correlation",
        )
        self.redundant_factor_ids = _ordered_unique(self.redundant_factor_ids)
        self.related_factor_ids = _ordered_unique(self.related_factor_ids)
        if self.overall_verdict not in SUPPORTED_CORRELATION_VERDICTS:
            raise ValueError(
                f"overall_verdict must be one of {sorted(SUPPORTED_CORRELATION_VERDICTS)}; "
                f"got {self.overall_verdict!r}."
            )
        self.issue_codes = _ordered_unique(self.issue_codes)
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
            "pair_results": [pair.to_dict() for pair in self.pair_results],
            "max_abs_return_correlation": self.max_abs_return_correlation,
            "max_abs_matrix_rank_correlation": self.max_abs_matrix_rank_correlation,
            "max_abs_ic_correlation": self.max_abs_ic_correlation,
            "redundant_factor_ids": list(self.redundant_factor_ids),
            "related_factor_ids": list(self.related_factor_ids),
            "overall_verdict": self.overall_verdict,
            "issue_codes": list(self.issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorRedundancyReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_CORRELATION_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            candidate_factor_id=data.get("candidate_factor_id"),
            candidate_factor_version=data.get("candidate_factor_version"),
            candidate_run_id=data.get("candidate_run_id"),
            generated_at=str(data.get("generated_at", "")),
            config=FactorCorrelationConfig.from_dict(dict(data.get("config", {}) or {})),
            pair_results=[
                FactorCorrelationPair.from_dict(pair)
                for pair in list(data.get("pair_results", []) or [])
                if isinstance(pair, Mapping)
            ],
            max_abs_return_correlation=data.get("max_abs_return_correlation"),
            max_abs_matrix_rank_correlation=data.get("max_abs_matrix_rank_correlation"),
            max_abs_ic_correlation=data.get("max_abs_ic_correlation"),
            redundant_factor_ids=list(data.get("redundant_factor_ids", []) or []),
            related_factor_ids=list(data.get("related_factor_ids", []) or []),
            overall_verdict=str(
                data.get("overall_verdict", CORRELATION_VERDICT_INSUFFICIENT_DATA)
            ),
            issue_codes=list(data.get("issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_correlation_config_id(config: FactorCorrelationConfig) -> str:
    parts = [
        config.schema_version,
        config.max_return_correlation,
        config.max_matrix_rank_correlation,
        config.max_ic_correlation,
        config.min_overlap_days,
        config.min_residual_mean_return,
        config.use_abs_correlation,
        config.metadata,
    ]
    return f"factor-correlation-config-{_short_hash(parts)}"


def make_correlation_pair_id(
    *,
    candidate_run_id: str | None,
    reference_run_id: str | None,
    candidate_matrix_id: str | None,
    reference_matrix_id: str | None,
) -> str:
    parts = [candidate_run_id, reference_run_id, candidate_matrix_id, reference_matrix_id]
    return f"factor-correlation-pair-{_short_hash(parts)}"


def make_redundancy_report_id(*, candidate_run_id: str | None, generated_at: str) -> str:
    parts = [candidate_run_id, generated_at]
    return f"factor-redundancy-report-{_slug(candidate_run_id)}-{_short_hash(parts)}"


def filter_paired_finite(
    x: Sequence[float | None],
    y: Sequence[float | None],
) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    for left_value, right_value in zip(x, y):
        left_number = _to_finite_float(left_value)
        right_number = _to_finite_float(right_value)
        if left_number is None or right_number is None:
            continue
        left.append(left_number)
        right.append(right_number)
    return left, right


def pearson_correlation(
    x: Sequence[float | None],
    y: Sequence[float | None],
) -> float | None:
    left, right = filter_paired_finite(x, y)
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= _EPSILON or right_ss <= _EPSILON:
        return None
    covariance = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    )
    return covariance / math.sqrt(left_ss * right_ss)


def rank_values(values: Sequence[float]) -> list[float]:
    numbers = [_finite_float(value, "values") for value in values]
    indexed = sorted(enumerate(numbers), key=lambda item: (item[1], item[0]))
    ranks = [0.0 for _value in numbers]
    index = 0
    while index < len(indexed):
        end_index = index + 1
        while end_index < len(indexed) and indexed[end_index][1] == indexed[index][1]:
            end_index += 1
        average_rank = (index + 1 + end_index) / 2.0
        for original_index, _value in indexed[index:end_index]:
            ranks[original_index] = average_rank
        index = end_index
    return ranks


def spearman_correlation(
    x: Sequence[float | None],
    y: Sequence[float | None],
) -> float | None:
    left, right = filter_paired_finite(x, y)
    if len(left) < 2:
        return None
    return pearson_correlation(rank_values(left), rank_values(right))


def simple_residual_series(
    y: Sequence[float | None],
    x: Sequence[float | None],
) -> list[float | None]:
    left, right = filter_paired_finite(y, x)
    if len(left) < 2:
        return [None for _value in y]
    y_mean = sum(left) / len(left)
    x_mean = sum(right) / len(right)
    x_centered = [value - x_mean for value in right]
    x_ss = sum(value * value for value in x_centered)
    if x_ss <= _EPSILON:
        return [None for _value in y]
    covariance = sum(
        (y_value - y_mean) * (x_value - x_mean)
        for y_value, x_value in zip(left, right)
    )
    beta = covariance / x_ss
    alpha = y_mean - beta * x_mean
    residuals: list[float | None] = []
    for y_value, x_value in zip(y, x):
        y_number = _to_finite_float(y_value)
        x_number = _to_finite_float(x_value)
        if y_number is None or x_number is None:
            residuals.append(None)
            continue
        residuals.append(y_number - (alpha + beta * x_number))
    return residuals


def _extract_record_value(run: SingleFactorBacktestRun, field_name: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for record in run.daily_records:
        value = _to_finite_float(getattr(record, field_name))
        if value is None:
            continue
        key = record.date or record.execution_end_date
        output[str(key)] = value
    return {date_key: output[date_key] for date_key in sorted(output)}


def extract_after_cost_return_series(run: SingleFactorBacktestRun) -> dict[str, float]:
    return _extract_record_value(run, "after_cost_return")


def extract_before_cost_return_series(run: SingleFactorBacktestRun) -> dict[str, float]:
    return _extract_record_value(run, "long_short_return")


def extract_ic_series(run: SingleFactorBacktestRun) -> dict[str, float]:
    output: dict[str, float] = {}
    for record in run.daily_records:
        raw_value = record.metadata.get("ic")
        if raw_value is None:
            raw_value = record.metadata.get("rank_ic")
        value = _to_finite_float(raw_value)
        if value is None:
            continue
        key = record.date or record.execution_end_date
        output[str(key)] = value
    return {date_key: output[date_key] for date_key in sorted(output)}


def align_series_by_date(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> tuple[list[str], list[float | None], list[float | None]]:
    dates = sorted(
        set(str(date_key) for date_key in left)
        & set(str(date_key) for date_key in right)
    )
    return dates, [left[date_key] for date_key in dates], [right[date_key] for date_key in dates]


def average_matrix_rank_correlation(
    candidate: FactorMatrix,
    reference: FactorMatrix,
    *,
    min_common_symbols: int = 2,
) -> float | None:
    common_dates = sorted(set(candidate.dates) & set(reference.dates))
    common_symbols = sorted(set(candidate.symbols) & set(reference.symbols))
    if not common_dates or len(common_symbols) < min_common_symbols:
        return None

    candidate_date_index = {date_key: index for index, date_key in enumerate(candidate.dates)}
    reference_date_index = {date_key: index for index, date_key in enumerate(reference.dates)}
    candidate_symbol_index = {symbol: index for index, symbol in enumerate(candidate.symbols)}
    reference_symbol_index = {symbol: index for index, symbol in enumerate(reference.symbols)}

    correlations: list[float] = []
    for date_key in common_dates:
        candidate_column = candidate_date_index[date_key]
        reference_column = reference_date_index[date_key]
        candidate_values = [
            candidate.values[candidate_symbol_index[symbol]][candidate_column]
            for symbol in common_symbols
        ]
        reference_values = [
            reference.values[reference_symbol_index[symbol]][reference_column]
            for symbol in common_symbols
        ]
        finite_candidate, finite_reference = filter_paired_finite(
            candidate_values,
            reference_values,
        )
        if len(finite_candidate) < min_common_symbols:
            continue
        correlation = spearman_correlation(finite_candidate, finite_reference)
        if correlation is not None:
            correlations.append(correlation)
    return _mean(correlations)


def evaluate_factor_correlation_pair(
    candidate_run: SingleFactorBacktestRun | None,
    reference_run: SingleFactorBacktestRun | None,
    *,
    candidate_matrix: FactorMatrix | None = None,
    reference_matrix: FactorMatrix | None = None,
    config: FactorCorrelationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorCorrelationPair:
    resolved_config = config or FactorCorrelationConfig()
    issue_codes: list[str] = []
    overlap_days = 0
    return_correlation: float | None = None
    rank_return_correlation: float | None = None
    matrix_rank_correlation_avg: float | None = None
    ic_correlation: float | None = None
    residual_mean_return: float | None = None

    if candidate_run is not None and reference_run is not None:
        candidate_returns = extract_after_cost_return_series(candidate_run)
        reference_returns = extract_after_cost_return_series(reference_run)
        _dates, candidate_values, reference_values = align_series_by_date(
            candidate_returns,
            reference_returns,
        )
        finite_candidate, _finite_reference = filter_paired_finite(
            candidate_values,
            reference_values,
        )
        overlap_days = len(finite_candidate)
        return_correlation = pearson_correlation(candidate_values, reference_values)
        rank_return_correlation = spearman_correlation(candidate_values, reference_values)
        residuals = simple_residual_series(candidate_values, reference_values)
        residual_mean_return = _mean([
            value for value in residuals if _to_finite_float(value) is not None
        ])
        if overlap_days < resolved_config.min_overlap_days:
            issue_codes.append(CORRELATION_ISSUE_INSUFFICIENT_OVERLAP)

        candidate_ic = extract_ic_series(candidate_run)
        reference_ic = extract_ic_series(reference_run)
        if candidate_ic and reference_ic:
            _ic_dates, candidate_ic_values, reference_ic_values = align_series_by_date(
                candidate_ic,
                reference_ic,
            )
            ic_correlation = pearson_correlation(candidate_ic_values, reference_ic_values)

    if candidate_matrix is not None and reference_matrix is not None:
        matrix_rank_correlation_avg = average_matrix_rank_correlation(
            candidate_matrix,
            reference_matrix,
        )

    if _breaches_correlation_threshold(
        return_correlation,
        threshold=resolved_config.max_return_correlation,
        use_abs=resolved_config.use_abs_correlation,
    ):
        issue_codes.append(CORRELATION_ISSUE_HIGH_RETURN_CORRELATION)
    if _breaches_correlation_threshold(
        matrix_rank_correlation_avg,
        threshold=resolved_config.max_matrix_rank_correlation,
        use_abs=resolved_config.use_abs_correlation,
    ):
        issue_codes.append(CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION)
    if _breaches_correlation_threshold(
        ic_correlation,
        threshold=resolved_config.max_ic_correlation,
        use_abs=resolved_config.use_abs_correlation,
    ):
        issue_codes.append(CORRELATION_ISSUE_HIGH_IC_CORRELATION)
    if (
        residual_mean_return is not None
        and residual_mean_return <= resolved_config.min_residual_mean_return
    ):
        issue_codes.append(CORRELATION_ISSUE_LOW_RESIDUAL_RETURN)

    high_return_or_matrix = any(
        issue in issue_codes
        for issue in (
            CORRELATION_ISSUE_HIGH_RETURN_CORRELATION,
            CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION,
        )
    )
    high_any = any(issue in HIGH_CORRELATION_ISSUES for issue in issue_codes)
    low_residual = CORRELATION_ISSUE_LOW_RESIDUAL_RETURN in issue_codes
    insufficient_overlap = CORRELATION_ISSUE_INSUFFICIENT_OVERLAP in issue_codes
    if insufficient_overlap and matrix_rank_correlation_avg is None:
        verdict = CORRELATION_VERDICT_INSUFFICIENT_DATA
    elif high_return_or_matrix and low_residual:
        verdict = CORRELATION_VERDICT_REDUNDANT
    elif high_any:
        verdict = CORRELATION_VERDICT_RELATED
    else:
        verdict = CORRELATION_VERDICT_DISTINCT

    pair_metadata = {
        **_coerce_metadata(metadata),
        "factor_correlation_schema_version": FACTOR_CORRELATION_SCHEMA_VERSION,
        "comparison_policy": "return_matrix_ic_residual",
        "offline_only": True,
    }
    candidate_run_id = candidate_run.run_id if candidate_run is not None else None
    reference_run_id = reference_run.run_id if reference_run is not None else None
    candidate_matrix_id = candidate_matrix.matrix_id if candidate_matrix is not None else None
    reference_matrix_id = reference_matrix.matrix_id if reference_matrix is not None else None
    return FactorCorrelationPair(
        pair_id=make_correlation_pair_id(
            candidate_run_id=candidate_run_id,
            reference_run_id=reference_run_id,
            candidate_matrix_id=candidate_matrix_id,
            reference_matrix_id=reference_matrix_id,
        ),
        candidate_factor_id=_factor_id_from(candidate_run, candidate_matrix),
        candidate_factor_version=_factor_version_from(candidate_run, candidate_matrix),
        reference_factor_id=_factor_id_from(reference_run, reference_matrix),
        reference_factor_version=_factor_version_from(reference_run, reference_matrix),
        candidate_run_id=candidate_run_id,
        reference_run_id=reference_run_id,
        candidate_matrix_id=candidate_matrix_id,
        reference_matrix_id=reference_matrix_id,
        overlap_days=overlap_days,
        return_correlation=return_correlation,
        rank_return_correlation=rank_return_correlation,
        matrix_rank_correlation_avg=matrix_rank_correlation_avg,
        ic_correlation=ic_correlation,
        residual_mean_return=residual_mean_return,
        issue_codes=issue_codes,
        verdict=verdict,
        metadata=pair_metadata,
    )


def _reference_key(
    run: SingleFactorBacktestRun | None,
    matrix: FactorMatrix | None,
) -> tuple[str, str, str, str]:
    return (
        _factor_id_from(run, matrix) or "",
        _factor_version_from(run, matrix) or "",
        run.run_id if run is not None else "",
        matrix.matrix_id if matrix is not None else "",
    )


def _find_matching_matrix_index(
    run: SingleFactorBacktestRun,
    matrices: Sequence[FactorMatrix],
    used_indexes: set[int],
    fallback_index: int,
) -> int | None:
    for index, matrix in enumerate(matrices):
        if index in used_indexes:
            continue
        if run.factor_id is not None and matrix.factor_id != run.factor_id:
            continue
        if run.factor_version is not None and matrix.factor_version != run.factor_version:
            continue
        return index
    if fallback_index < len(matrices) and fallback_index not in used_indexes:
        return fallback_index
    return None


def _max_abs(values: Sequence[float | None]) -> float | None:
    numbers = [abs(value) for value in values if value is not None]
    if not numbers:
        return None
    return max(numbers)


def build_factor_redundancy_report(
    *,
    candidate_run: SingleFactorBacktestRun | None,
    reference_runs: Sequence[SingleFactorBacktestRun] | None = None,
    candidate_matrix: FactorMatrix | None = None,
    reference_matrices: Sequence[FactorMatrix] | None = None,
    config: FactorCorrelationConfig | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorRedundancyReport:
    resolved_config = config or FactorCorrelationConfig()
    runs = list(reference_runs or [])
    matrices = list(reference_matrices or [])
    used_matrix_indexes: set[int] = set()
    seen_keys: set[tuple[str, str, str, str]] = set()
    pairs: list[FactorCorrelationPair] = []

    for index, reference_run in enumerate(runs):
        matrix_index = _find_matching_matrix_index(
            reference_run,
            matrices,
            used_matrix_indexes,
            index,
        )
        reference_matrix = matrices[matrix_index] if matrix_index is not None else None
        if matrix_index is not None:
            used_matrix_indexes.add(matrix_index)
        key = _reference_key(reference_run, reference_matrix)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pairs.append(
            evaluate_factor_correlation_pair(
                candidate_run,
                reference_run,
                candidate_matrix=candidate_matrix,
                reference_matrix=reference_matrix,
                config=resolved_config,
            )
        )

    for index, reference_matrix in enumerate(matrices):
        if index in used_matrix_indexes:
            continue
        key = _reference_key(None, reference_matrix)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pairs.append(
            evaluate_factor_correlation_pair(
                candidate_run,
                None,
                candidate_matrix=candidate_matrix,
                reference_matrix=reference_matrix,
                config=resolved_config,
            )
        )

    pairs = sorted(
        pairs,
        key=lambda pair: (
            pair.reference_factor_id or "",
            pair.reference_factor_version or "",
            pair.reference_run_id or "",
            pair.reference_matrix_id or "",
        ),
    )
    redundant_factor_ids = _ordered_unique([
        pair.reference_factor_id
        for pair in pairs
        if pair.verdict == CORRELATION_VERDICT_REDUNDANT and pair.reference_factor_id
    ])
    related_factor_ids = _ordered_unique([
        pair.reference_factor_id
        for pair in pairs
        if pair.verdict == CORRELATION_VERDICT_RELATED and pair.reference_factor_id
    ])
    issue_codes = _ordered_unique([
        issue
        for pair in pairs
        for issue in pair.issue_codes
    ])
    if any(pair.verdict == CORRELATION_VERDICT_REDUNDANT for pair in pairs):
        overall_verdict = CORRELATION_VERDICT_REDUNDANT
    elif any(pair.verdict == CORRELATION_VERDICT_RELATED for pair in pairs):
        overall_verdict = CORRELATION_VERDICT_RELATED
    elif not [pair for pair in pairs if pair.verdict != CORRELATION_VERDICT_INSUFFICIENT_DATA]:
        overall_verdict = CORRELATION_VERDICT_INSUFFICIENT_DATA
    else:
        overall_verdict = CORRELATION_VERDICT_DISTINCT

    report_metadata = {
        **_coerce_metadata(metadata),
        "factor_correlation_schema_version": FACTOR_CORRELATION_SCHEMA_VERSION,
        "comparison_policy": "return_matrix_ic_residual",
        "offline_only": True,
    }
    candidate_run_id = candidate_run.run_id if candidate_run is not None else None
    return FactorRedundancyReport(
        report_id=make_redundancy_report_id(
            candidate_run_id=candidate_run_id,
            generated_at=generated_at,
        ),
        candidate_factor_id=_factor_id_from(candidate_run, candidate_matrix),
        candidate_factor_version=_factor_version_from(candidate_run, candidate_matrix),
        candidate_run_id=candidate_run_id,
        generated_at=generated_at,
        config=resolved_config,
        pair_results=pairs,
        max_abs_return_correlation=_max_abs([pair.return_correlation for pair in pairs]),
        max_abs_matrix_rank_correlation=_max_abs([
            pair.matrix_rank_correlation_avg
            for pair in pairs
        ]),
        max_abs_ic_correlation=_max_abs([pair.ic_correlation for pair in pairs]),
        redundant_factor_ids=redundant_factor_ids,
        related_factor_ids=related_factor_ids,
        overall_verdict=overall_verdict,
        issue_codes=issue_codes,
        metadata=report_metadata,
    )


__all__ = [
    "CORRELATION_VERDICT_DISTINCT",
    "CORRELATION_VERDICT_RELATED",
    "CORRELATION_VERDICT_REDUNDANT",
    "CORRELATION_VERDICT_INSUFFICIENT_DATA",
    "CORRELATION_ISSUE_HIGH_RETURN_CORRELATION",
    "CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION",
    "CORRELATION_ISSUE_HIGH_IC_CORRELATION",
    "CORRELATION_ISSUE_LOW_RESIDUAL_RETURN",
    "CORRELATION_ISSUE_INSUFFICIENT_OVERLAP",
    "FactorCorrelationConfig",
    "FactorCorrelationPair",
    "FactorRedundancyReport",
    "make_correlation_config_id",
    "make_correlation_pair_id",
    "make_redundancy_report_id",
    "filter_paired_finite",
    "pearson_correlation",
    "rank_values",
    "spearman_correlation",
    "simple_residual_series",
    "extract_after_cost_return_series",
    "extract_before_cost_return_series",
    "extract_ic_series",
    "align_series_by_date",
    "average_matrix_rank_correlation",
    "evaluate_factor_correlation_pair",
    "build_factor_redundancy_report",
]
