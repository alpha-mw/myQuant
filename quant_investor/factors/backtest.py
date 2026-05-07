"""Offline single-factor backtest helpers.

This module consumes in-memory ``FactorMatrix`` and ``MatrixDataBundle``
artifacts. It does not fetch market data and does not connect research factors
to production stock selection or portfolio construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.matrix import (
    FIELD_BENCHMARK_CLOSE,
    FIELD_BENCHMARK_RET,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
    add_standard_derived_fields,
)
from quant_investor.factors.schema import (
    FactorBacktestConfig,
    FactorBacktestResult,
    make_backtest_result_id,
)
from quant_investor.versioning import FACTOR_BACKTEST_SCHEMA_VERSION


EXECUTION_PRICE_OPEN = "open"
EXECUTION_PRICE_VWAP = "vwap"
EXECUTION_PRICE_CLOSE = "close"

BACKTEST_MODE_LONG_SHORT = "long_short"
BACKTEST_MODE_LONG_ONLY = "long_only"

WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE = "equal_quantile_booksize"

DEFAULT_FACTOR_BACKTEST_DIR = Path("data/factor_library/backtest")
DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME = "factor_weight_matrices.jsonl"
DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME = "factor_backtest_runs.jsonl"
DEFAULT_FACTOR_DAILY_RECORDS_FILENAME = "factor_daily_records.jsonl"

SUPPORTED_EXECUTION_PRICES = {
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    EXECUTION_PRICE_CLOSE,
}
SUPPORTED_BACKTEST_MODES = {
    BACKTEST_MODE_LONG_SHORT,
    BACKTEST_MODE_LONG_ONLY,
}

_EPSILON = 1e-10


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
    text = str(value)
    return text if text else None


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1; got {value!r}.")
    return number


def _unit_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    return number


def _coerce_weight_value(value: Any, field_name: str) -> float | None:
    return _optional_finite_float(value, field_name)


def _coerce_weight_matrix(
    values: Sequence[Sequence[Any]],
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> list[list[float | None]]:
    output: list[list[float | None]] = []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a matrix.")
    if len(values) != rows:
        raise ValueError(f"{field_name} must have {rows} rows; got {len(values)}.")
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"{field_name} row {row_index} must be a sequence.")
        if len(row) != columns:
            raise ValueError(
                f"{field_name} row {row_index} must have {columns} columns; got {len(row)}."
            )
        output.append([
            _coerce_weight_value(value, f"{field_name}[{row_index}]")
            for value in row
        ])
    return output


def _number_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


def _resolve_execution_price(value: str | None) -> str:
    execution_price = str(value or EXECUTION_PRICE_VWAP).strip()
    _validate_supported(execution_price, "execution_price", SUPPORTED_EXECUTION_PRICES)
    return execution_price


def _resolve_mode(value: str | None) -> str:
    mode = str(value or "").strip()
    _validate_supported(mode, "mode", SUPPORTED_BACKTEST_MODES)
    return mode


def _coerce_dates(values: Sequence[str]) -> list[str]:
    dates = [str(value).strip() for value in values if str(value).strip()]
    if not dates:
        raise ValueError("dates must be non-empty.")
    parsed: list[date] = []
    for value in dates:
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dates must be ISO dates; got {value!r}.") from exc
        if parsed_value.isoformat() != value:
            raise ValueError(f"dates must be canonical ISO dates; got {value!r}.")
        parsed.append(parsed_value)
    if any(current >= next_value for current, next_value in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly ascending ISO dates.")
    return dates


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
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _to_positive_price(value: Any) -> float | None:
    number = _to_finite_float(value)
    if number is None or number <= 0.0:
        return None
    return number


def _safe_matrix_copy(values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
    return [[value for value in row] for row in values]


def _validate_weight_matrix_contract(matrix: "FactorWeightMatrix") -> None:
    columns = len(matrix.dates)
    for column_index in range(columns):
        long_values = [row[column_index] for row in matrix.long_weights]
        short_values = [row[column_index] for row in matrix.short_weights]
        net_values = [row[column_index] for row in matrix.net_weights]

        for value in long_values:
            if value is not None and value < -_EPSILON:
                raise ValueError("long_weights must be non-negative or None.")
        for value in short_values:
            if value is not None and value > _EPSILON:
                raise ValueError("short_weights must be non-positive or None.")

        long_sum = sum(_number_or_zero(value) for value in long_values)
        short_sum = sum(_number_or_zero(value) for value in short_values)
        long_book_exists = any(abs(_number_or_zero(value)) > _EPSILON for value in long_values)
        short_book_exists = any(abs(_number_or_zero(value)) > _EPSILON for value in short_values)

        if long_book_exists and abs(long_sum - 1.0) > 1e-8:
            raise ValueError("long_weights must sum to +1.0 when a long book exists.")
        if not long_book_exists and abs(long_sum) > 1e-8:
            raise ValueError("long_weights must sum to 0.0 when no long book exists.")
        if short_book_exists and abs(short_sum + 1.0) > 1e-8:
            raise ValueError("short_weights must sum to -1.0 when a short book exists.")
        if not short_book_exists and abs(short_sum) > 1e-8:
            raise ValueError("short_weights must sum to 0.0 when no short book exists.")

        for row_index, net_value in enumerate(net_values):
            expected = _number_or_zero(long_values[row_index]) + _number_or_zero(
                short_values[row_index]
            )
            if net_value is None:
                if abs(expected) > _EPSILON:
                    raise ValueError("net_weights must equal long_weights + short_weights.")
                continue
            if abs(net_value - expected) > 1e-8:
                raise ValueError("net_weights must equal long_weights + short_weights.")

    if matrix.metadata.get("mode") == BACKTEST_MODE_LONG_ONLY:
        for row_index, row in enumerate(matrix.short_weights):
            for column_index, value in enumerate(row):
                if value is not None and abs(value) > _EPSILON:
                    raise ValueError(
                        f"short_weights[{row_index}][{column_index}] must be zero in long-only mode."
                    )
        for row_index, row in enumerate(matrix.net_weights):
            for column_index, value in enumerate(row):
                long_value = matrix.long_weights[row_index][column_index]
                expected = _number_or_zero(long_value)
                if value is None:
                    if abs(expected) > _EPSILON:
                        raise ValueError("net_weights must equal long_weights in long-only mode.")
                    continue
                if abs(value - expected) > 1e-8:
                    raise ValueError("net_weights must equal long_weights in long-only mode.")


@dataclass
class FactorBacktestAlignment:
    schema_version: str = FACTOR_BACKTEST_SCHEMA_VERSION
    signal_date: str = ""
    execution_start_date: str = ""
    execution_end_date: str = ""
    signal_index: int = 0
    execution_start_index: int = 0
    execution_end_index: int = 0
    delay_days: int = 1
    holding_period_days: int = 1
    execution_price: str = EXECUTION_PRICE_VWAP
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_BACKTEST_SCHEMA_VERSION)
        self.signal_date = _non_empty_str(self.signal_date, "signal_date")
        self.execution_start_date = _non_empty_str(
            self.execution_start_date,
            "execution_start_date",
        )
        self.execution_end_date = _non_empty_str(self.execution_end_date, "execution_end_date")
        self.signal_index = _non_negative_int(self.signal_index, "signal_index")
        self.execution_start_index = _non_negative_int(
            self.execution_start_index,
            "execution_start_index",
        )
        self.execution_end_index = _non_negative_int(
            self.execution_end_index,
            "execution_end_index",
        )
        if self.execution_start_index <= self.signal_index:
            raise ValueError("execution_start_index must be greater than signal_index.")
        if self.execution_end_index <= self.execution_start_index:
            raise ValueError("execution_end_index must be greater than execution_start_index.")
        self.delay_days = _positive_int(self.delay_days, "delay_days")
        self.holding_period_days = _positive_int(
            self.holding_period_days,
            "holding_period_days",
        )
        self.execution_price = _resolve_execution_price(self.execution_price)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signal_date": self.signal_date,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "signal_index": self.signal_index,
            "execution_start_index": self.execution_start_index,
            "execution_end_index": self.execution_end_index,
            "delay_days": self.delay_days,
            "holding_period_days": self.holding_period_days,
            "execution_price": self.execution_price,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorBacktestAlignment":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_BACKTEST_SCHEMA_VERSION)),
            signal_date=str(data.get("signal_date", "")),
            execution_start_date=str(data.get("execution_start_date", "")),
            execution_end_date=str(data.get("execution_end_date", "")),
            signal_index=int(data.get("signal_index", 0)),
            execution_start_index=int(data.get("execution_start_index", 0)),
            execution_end_index=int(data.get("execution_end_index", 0)),
            delay_days=int(data.get("delay_days", 1)),
            holding_period_days=int(data.get("holding_period_days", 1)),
            execution_price=str(data.get("execution_price", EXECUTION_PRICE_VWAP)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorWeightMatrix:
    schema_version: str = FACTOR_BACKTEST_SCHEMA_VERSION
    weights_id: str = ""
    factor_matrix_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    config_id: str = ""
    symbols: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    long_weights: list[list[float | None]] = field(default_factory=list)
    short_weights: list[list[float | None]] = field(default_factory=list)
    net_weights: list[list[float | None]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_BACKTEST_SCHEMA_VERSION)
        self.weights_id = _non_empty_str(self.weights_id, "weights_id")
        self.factor_matrix_id = _non_empty_str(self.factor_matrix_id, "factor_matrix_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.symbols = [str(symbol).strip() for symbol in self.symbols if str(symbol).strip()]
        if not self.symbols:
            raise ValueError("symbols must be non-empty.")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must contain unique values.")
        self.dates = _coerce_dates(self.dates)
        rows = len(self.symbols)
        columns = len(self.dates)
        self.long_weights = _coerce_weight_matrix(
            self.long_weights,
            rows=rows,
            columns=columns,
            field_name="long_weights",
        )
        self.short_weights = _coerce_weight_matrix(
            self.short_weights,
            rows=rows,
            columns=columns,
            field_name="short_weights",
        )
        self.net_weights = _coerce_weight_matrix(
            self.net_weights,
            rows=rows,
            columns=columns,
            field_name="net_weights",
        )
        self.metadata = _coerce_metadata(self.metadata)
        _validate_weight_matrix_contract(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "weights_id": self.weights_id,
            "factor_matrix_id": self.factor_matrix_id,
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "config_id": self.config_id,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "long_weights": _json_safe(self.long_weights),
            "short_weights": _json_safe(self.short_weights),
            "net_weights": _json_safe(self.net_weights),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorWeightMatrix":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_BACKTEST_SCHEMA_VERSION)),
            weights_id=str(data.get("weights_id", "")),
            factor_matrix_id=str(data.get("factor_matrix_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            config_id=str(data.get("config_id", "")),
            symbols=list(data.get("symbols", []) or []),
            dates=list(data.get("dates", []) or []),
            long_weights=list(data.get("long_weights", []) or []),
            short_weights=list(data.get("short_weights", []) or []),
            net_weights=list(data.get("net_weights", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorDailyBacktestRecord:
    schema_version: str = FACTOR_BACKTEST_SCHEMA_VERSION
    date: str = ""
    signal_date: str = ""
    execution_start_date: str = ""
    execution_end_date: str = ""
    long_return: float | None = None
    short_return: float | None = None
    long_short_return: float | None = None
    after_cost_return: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    turnover: float = 0.0
    long_count: int = 0
    short_count: int = 0
    coverage_ratio: float = 0.0
    missing_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_BACKTEST_SCHEMA_VERSION)
        self.date = _non_empty_str(self.date, "date")
        self.signal_date = _non_empty_str(self.signal_date, "signal_date")
        self.execution_start_date = _non_empty_str(
            self.execution_start_date,
            "execution_start_date",
        )
        self.execution_end_date = _non_empty_str(self.execution_end_date, "execution_end_date")
        if self.date != self.execution_end_date:
            raise ValueError("date must equal execution_end_date.")
        for field_name in (
            "long_return",
            "short_return",
            "long_short_return",
            "after_cost_return",
            "benchmark_return",
            "excess_return",
        ):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        self.turnover = _optional_finite_float(self.turnover, "turnover") or 0.0
        if self.turnover < 0.0:
            raise ValueError("turnover must be non-negative.")
        self.long_count = _non_negative_int(self.long_count, "long_count")
        self.short_count = _non_negative_int(self.short_count, "short_count")
        self.coverage_ratio = _unit_float(self.coverage_ratio, "coverage_ratio")
        self.missing_ratio = _unit_float(self.missing_ratio, "missing_ratio")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "date": self.date,
            "signal_date": self.signal_date,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "long_return": self.long_return,
            "short_return": self.short_return,
            "long_short_return": self.long_short_return,
            "after_cost_return": self.after_cost_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "turnover": self.turnover,
            "long_count": self.long_count,
            "short_count": self.short_count,
            "coverage_ratio": self.coverage_ratio,
            "missing_ratio": self.missing_ratio,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorDailyBacktestRecord":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_BACKTEST_SCHEMA_VERSION)),
            date=str(data.get("date", "")),
            signal_date=str(data.get("signal_date", "")),
            execution_start_date=str(data.get("execution_start_date", "")),
            execution_end_date=str(data.get("execution_end_date", "")),
            long_return=data.get("long_return"),
            short_return=data.get("short_return"),
            long_short_return=data.get("long_short_return"),
            after_cost_return=data.get("after_cost_return"),
            benchmark_return=data.get("benchmark_return"),
            excess_return=data.get("excess_return"),
            turnover=float(data.get("turnover", 0.0)),
            long_count=int(data.get("long_count", 0)),
            short_count=int(data.get("short_count", 0)),
            coverage_ratio=float(data.get("coverage_ratio", 0.0)),
            missing_ratio=float(data.get("missing_ratio", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SingleFactorBacktestRun:
    schema_version: str = FACTOR_BACKTEST_SCHEMA_VERSION
    run_id: str = ""
    factor_matrix_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    config_id: str = ""
    start_date: str = ""
    end_date: str = ""
    mode: str = BACKTEST_MODE_LONG_SHORT
    alignment_policy: str = "signal_delay_execution_window"
    weighting_method: str = WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE
    weight_matrix: FactorWeightMatrix = field(default_factory=FactorWeightMatrix)
    daily_records: list[FactorDailyBacktestRecord] = field(default_factory=list)
    aggregate_result: FactorBacktestResult = field(default_factory=FactorBacktestResult)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_BACKTEST_SCHEMA_VERSION)
        self.run_id = _non_empty_str(self.run_id, "run_id")
        self.factor_matrix_id = _non_empty_str(self.factor_matrix_id, "factor_matrix_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.start_date = _non_empty_str(self.start_date, "start_date")
        self.end_date = _non_empty_str(self.end_date, "end_date")
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        self.mode = _resolve_mode(self.mode)
        self.alignment_policy = _non_empty_str(self.alignment_policy, "alignment_policy")
        self.weighting_method = _non_empty_str(self.weighting_method, "weighting_method")
        if not isinstance(self.weight_matrix, FactorWeightMatrix):
            self.weight_matrix = FactorWeightMatrix.from_dict(self.weight_matrix)
        self.daily_records = [
            record if isinstance(record, FactorDailyBacktestRecord)
            else FactorDailyBacktestRecord.from_dict(record)
            for record in self.daily_records
        ]
        self.daily_records = sorted(self.daily_records, key=lambda record: record.date)
        if not isinstance(self.aggregate_result, FactorBacktestResult):
            self.aggregate_result = FactorBacktestResult.from_dict(self.aggregate_result)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "factor_matrix_id": self.factor_matrix_id,
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "config_id": self.config_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mode": self.mode,
            "alignment_policy": self.alignment_policy,
            "weighting_method": self.weighting_method,
            "weight_matrix": self.weight_matrix.to_dict(),
            "daily_records": [record.to_dict() for record in self.daily_records],
            "aggregate_result": self.aggregate_result.to_dict(),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SingleFactorBacktestRun":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_BACKTEST_SCHEMA_VERSION)),
            run_id=str(data.get("run_id", "")),
            factor_matrix_id=str(data.get("factor_matrix_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            config_id=str(data.get("config_id", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            mode=str(data.get("mode", BACKTEST_MODE_LONG_SHORT)),
            alignment_policy=str(data.get("alignment_policy", "signal_delay_execution_window")),
            weighting_method=str(
                data.get("weighting_method", WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE)
            ),
            weight_matrix=FactorWeightMatrix.from_dict(dict(data.get("weight_matrix", {}) or {})),
            daily_records=[
                FactorDailyBacktestRecord.from_dict(record)
                for record in list(data.get("daily_records", []) or [])
                if isinstance(record, Mapping)
            ],
            aggregate_result=FactorBacktestResult.from_dict(
                dict(data.get("aggregate_result", {}) or {})
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_factor_weights_id(*, factor_matrix_id: str, config_id: str, mode: str) -> str:
    parts = [str(factor_matrix_id), str(config_id), str(mode)]
    return f"factor-weights-{_slug(factor_matrix_id)}-{_slug(mode)}-{_short_hash(parts)}"


def make_factor_backtest_run_id(
    *,
    factor_matrix_id: str,
    config_id: str,
    mode: str,
    start_date: str,
    end_date: str,
) -> str:
    parts = [str(factor_matrix_id), str(config_id), str(mode), str(start_date), str(end_date)]
    return (
        f"factor-bt-run-{_slug(factor_matrix_id)}-{_slug(mode)}-"
        f"{_slug(start_date)}-{_slug(end_date)}-{_short_hash(parts)}"
    )


def make_daily_record_id(*, run_id: str, date: str) -> str:
    parts = [str(run_id), str(date)]
    return f"factor-bt-daily-{_slug(run_id)}-{_slug(date)}-{_short_hash(parts)}"


def validate_finite_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number.")


def bps_to_decimal_return(value_bps: float) -> float:
    validate_finite_number(value_bps, field_name="value_bps")
    return float(value_bps) / 10000.0


def _usable_numbers(values: Sequence[float | None], *, field_name: str) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        validate_finite_number(value, field_name=field_name)
        output.append(float(value))
    return output


def safe_mean(values: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(values, field_name="values")
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def safe_std(values: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(values, field_name="values")
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return math.sqrt(variance)


def compound_returns(returns: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(returns, field_name="returns")
    if not numbers:
        return None
    total = 1.0
    for value in numbers:
        total *= 1.0 + value
    return total - 1.0


def max_drawdown_from_returns(returns: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(returns, field_name="returns")
    if not numbers:
        return None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in numbers:
        equity *= 1.0 + value
        if equity > peak:
            peak = equity
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max(0.0, max_drawdown)


def annualized_return_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    mean = safe_mean(returns)
    if mean is None:
        return None
    return mean * trading_days


def annualized_vol_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    std = safe_std(returns)
    if std is None:
        return None
    return std * math.sqrt(trading_days)


def sharpe_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    ann_ret = annualized_return_from_daily(returns, trading_days=trading_days)
    ann_vol = annualized_vol_from_daily(returns, trading_days=trading_days)
    if ann_ret is None or ann_vol is None or abs(ann_vol) <= _EPSILON:
        return None
    return ann_ret / ann_vol


def estimate_turnover(prev_weights: Mapping[str, float], next_weights: Mapping[str, float]) -> float:
    symbols = set(prev_weights) | set(next_weights)
    total = 0.0
    for symbol in symbols:
        previous = float(prev_weights.get(symbol, 0.0))
        next_value = float(next_weights.get(symbol, 0.0))
        validate_finite_number(previous, field_name=f"prev_weights.{symbol}")
        validate_finite_number(next_value, field_name=f"next_weights.{symbol}")
        total += abs(next_value - previous)
    return 0.5 * total


def build_backtest_alignments(
    dates: Sequence[str],
    *,
    delay_days: int,
    holding_period_days: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    execution_price: str = EXECUTION_PRICE_VWAP,
) -> list[FactorBacktestAlignment]:
    resolved_dates = _coerce_dates(dates)
    resolved_delay = _positive_int(delay_days, "delay_days")
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    resolved_execution_price = _resolve_execution_price(execution_price)
    if start_date is not None:
        date.fromisoformat(start_date)
    if end_date is not None:
        date.fromisoformat(end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date.")

    alignments: list[FactorBacktestAlignment] = []
    for signal_index, signal_date in enumerate(resolved_dates):
        if start_date is not None and signal_date < start_date:
            continue
        if end_date is not None and signal_date > end_date:
            continue
        execution_start_index = signal_index + resolved_delay
        execution_end_index = execution_start_index + resolved_holding_period
        if execution_end_index >= len(resolved_dates):
            continue
        alignments.append(
            FactorBacktestAlignment(
                signal_date=signal_date,
                execution_start_date=resolved_dates[execution_start_index],
                execution_end_date=resolved_dates[execution_end_index],
                signal_index=signal_index,
                execution_start_index=execution_start_index,
                execution_end_index=execution_end_index,
                delay_days=resolved_delay,
                holding_period_days=resolved_holding_period,
                execution_price=resolved_execution_price,
                metadata={"alignment_policy": "signal_delay_execution_window"},
            )
        )
    return alignments


def build_execution_return_matrix(
    bundle: MatrixDataBundle,
    *,
    execution_price: str,
    holding_period_days: int = 1,
) -> list[list[float | None]]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    field_by_execution_price = {
        EXECUTION_PRICE_OPEN: FIELD_OPEN,
        EXECUTION_PRICE_VWAP: FIELD_VWAP,
        EXECUTION_PRICE_CLOSE: FIELD_CLOSE,
    }
    field_name = field_by_execution_price[resolved_execution_price]
    source_bundle = bundle
    if field_name == FIELD_VWAP and not source_bundle.has_field(FIELD_VWAP):
        source_bundle = add_standard_derived_fields(source_bundle)
    if not source_bundle.has_field(field_name):
        raise ValueError(f"bundle does not contain execution price field {field_name!r}.")
    prices = source_bundle.get_field(field_name)
    output: list[list[float | None]] = []
    for row in prices:
        result_row: list[float | None] = []
        for column_index, value in enumerate(row):
            future_index = column_index + resolved_holding_period
            if future_index >= len(row):
                result_row.append(None)
                continue
            start_price = _to_positive_price(value)
            end_price = _to_positive_price(row[future_index])
            if start_price is None or end_price is None:
                result_row.append(None)
                continue
            result_row.append(end_price / start_price - 1.0)
        output.append(result_row)
    return output


def _factor_bundle_symbols_dates_match(factor_matrix: FactorMatrix, bundle: MatrixDataBundle) -> bool:
    return (
        list(factor_matrix.symbols) == list(bundle.contract.symbols)
        and list(factor_matrix.dates) == list(bundle.contract.dates)
    )


def _expected_direction(factor_matrix: FactorMatrix) -> float:
    value = factor_matrix.metadata.get("expected_direction", 1.0)
    number = float(value)
    if not math.isfinite(number) or number == 0.0:
        raise ValueError("expected_direction must be finite and non-zero.")
    return 1.0 if number > 0.0 else -1.0


def _is_universe_eligible(bundle: MatrixDataBundle, row_index: int, date_index: int) -> bool:
    if bundle.universe_mask is None:
        return True
    return bool(bundle.universe_mask[row_index][date_index])


def _is_tradable(bundle: MatrixDataBundle, row_index: int, date_index: int) -> bool:
    if bundle.tradability_mask is None:
        return True
    if date_index >= len(bundle.contract.dates):
        return False
    return bool(bundle.tradability_mask[row_index][date_index])


def _make_empty_weight_rows(symbol_count: int, date_count: int) -> list[list[float | None]]:
    return [[0.0 for _ in range(date_count)] for _ in range(symbol_count)]


def _quantile_for_rank(rank: int, sample_count: int, quantile_count: int) -> int:
    return min(quantile_count, int(rank * quantile_count / sample_count) + 1)


def _config_snapshot(config: FactorBacktestConfig) -> dict[str, Any]:
    return dict(_ensure_json_serializable(config.to_dict(), "config"))


def build_quantile_weight_matrix(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    *,
    mode: str = BACKTEST_MODE_LONG_SHORT,
    metadata: Mapping[str, Any] | None = None,
) -> FactorWeightMatrix:
    resolved_mode = _resolve_mode(mode)
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    execution_price = _resolve_execution_price(config.execution_price)
    expected_direction = _expected_direction(factor_matrix)
    symbol_count = len(factor_matrix.symbols)
    date_count = len(factor_matrix.dates)
    long_weights = _make_empty_weight_rows(symbol_count, date_count)
    short_weights = _make_empty_weight_rows(symbol_count, date_count)
    net_weights = _make_empty_weight_rows(symbol_count, date_count)
    execution_returns = build_execution_return_matrix(
        bundle,
        execution_price=execution_price,
        holding_period_days=1,
    )

    for date_index in range(date_count):
        execution_start_index = date_index + config.delay_days
        eligible: list[tuple[str, int, float]] = []
        for row_index, symbol in enumerate(factor_matrix.symbols):
            factor_value = _to_finite_float(factor_matrix.values[row_index][date_index])
            if factor_value is None:
                continue
            if not _is_universe_eligible(bundle, row_index, date_index):
                continue
            if not _is_tradable(bundle, row_index, execution_start_index):
                continue
            if execution_start_index >= date_count:
                continue
            if execution_returns[row_index][execution_start_index] is None:
                continue
            score = factor_value * expected_direction
            eligible.append((symbol, row_index, score))

        if not eligible:
            continue
        eligible = sorted(eligible, key=lambda item: (item[2], item[0]))
        long_rows: list[int] = []
        short_rows: list[int] = []
        for rank, (_symbol, row_index, _score) in enumerate(eligible):
            quantile_index = _quantile_for_rank(rank, len(eligible), config.quantile_count)
            if quantile_index == config.long_quantile:
                long_rows.append(row_index)
            if (
                resolved_mode == BACKTEST_MODE_LONG_SHORT
                and config.long_short
                and config.short_quantile is not None
                and quantile_index == config.short_quantile
            ):
                short_rows.append(row_index)

        if long_rows:
            long_weight = 1.0 / len(long_rows)
            for row_index in long_rows:
                long_weights[row_index][date_index] = long_weight
        if short_rows:
            short_weight = -1.0 / len(short_rows)
            for row_index in short_rows:
                short_weights[row_index][date_index] = short_weight
        for row_index in range(symbol_count):
            net_weights[row_index][date_index] = (
                _number_or_zero(long_weights[row_index][date_index])
                + _number_or_zero(short_weights[row_index][date_index])
            )

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "weighting_method": WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
            "mode": resolved_mode,
            "quantile_count": config.quantile_count,
            "long_quantile": config.long_quantile,
            "short_quantile": config.short_quantile,
            "expected_direction": expected_direction,
            "eligibility_policy": {
                "finite_factor_value": True,
                "universe_mask_signal_date": True,
                "tradability_mask_execution_start_date": True,
                "execution_return_required": True,
            },
            "execution_price": execution_price,
            "delay_days": config.delay_days,
            "config": _config_snapshot(config),
        }
    )

    return FactorWeightMatrix(
        weights_id=make_factor_weights_id(
            factor_matrix_id=factor_matrix.matrix_id,
            config_id=config.config_id,
            mode=resolved_mode,
        ),
        factor_matrix_id=factor_matrix.matrix_id,
        factor_id=factor_matrix.factor_id,
        factor_version=factor_matrix.factor_version,
        config_id=config.config_id,
        symbols=list(factor_matrix.symbols),
        dates=list(factor_matrix.dates),
        long_weights=long_weights,
        short_weights=short_weights,
        net_weights=net_weights,
        metadata=resolved_metadata,
    )


def _weights_for_column(
    symbols: Sequence[str],
    weights: Sequence[Sequence[float | None]],
    column_index: int,
) -> dict[str, float]:
    return {
        symbol: _number_or_zero(weights[row_index][column_index])
        for row_index, symbol in enumerate(symbols)
    }


def _nonzero_count(weights: Sequence[float | None]) -> int:
    return sum(1 for value in weights if abs(_number_or_zero(value)) > _EPSILON)


def _weighted_return(
    weights: Sequence[float | None],
    returns: Sequence[float | None],
) -> float | None:
    if _nonzero_count(weights) == 0:
        return None
    total = 0.0
    for weight, forward_return in zip(weights, returns):
        resolved_weight = _number_or_zero(weight)
        if abs(resolved_weight) <= _EPSILON:
            continue
        if forward_return is None:
            return None
        total += resolved_weight * forward_return
    return total


def _coverage_for_signal(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    signal_index: int,
) -> tuple[float, float]:
    denominator = 0
    usable = 0
    for row_index in range(len(factor_matrix.symbols)):
        if not _is_universe_eligible(bundle, row_index, signal_index):
            continue
        denominator += 1
        if _to_finite_float(factor_matrix.values[row_index][signal_index]) is not None:
            usable += 1
    if denominator == 0:
        return 0.0, 0.0
    coverage = usable / denominator
    return coverage, 1.0 - coverage


def _benchmark_return_at(
    bundle: MatrixDataBundle,
    execution_start_index: int,
    holding_period_days: int,
) -> float | None:
    if bundle.has_field(FIELD_BENCHMARK_RET):
        values = bundle.get_field(FIELD_BENCHMARK_RET)
        for row in values:
            if execution_start_index < len(row):
                value = _to_finite_float(row[execution_start_index])
                if value is not None:
                    return value
        return None
    if not bundle.has_field(FIELD_BENCHMARK_CLOSE):
        return None
    values = bundle.get_field(FIELD_BENCHMARK_CLOSE)
    if not values:
        return None
    row = values[0]
    end_index = execution_start_index + holding_period_days
    if execution_start_index >= len(row) or end_index >= len(row):
        return None
    start_price = _to_positive_price(row[execution_start_index])
    end_price = _to_positive_price(row[end_index])
    if start_price is None or end_price is None:
        return None
    return end_price / start_price - 1.0


def _rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(indexed):
        end_position = position + 1
        while end_position < len(indexed) and indexed[end_position][1] == indexed[position][1]:
            end_position += 1
        average_rank = (position + end_position - 1) / 2.0 + 1.0
        for indexed_position in range(position, end_position):
            original_index = indexed[indexed_position][0]
            ranks[original_index] = average_rank
        position = end_position
    return ranks


def _pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_var = sum((x_value - x_mean) ** 2 for x_value in x_values)
    y_var = sum((y_value - y_mean) ** 2 for y_value in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator <= _EPSILON:
        return None
    return covariance / denominator


def _ic_for_alignment(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    returns: Sequence[Sequence[float | None]],
    alignment: FactorBacktestAlignment,
) -> tuple[float | None, float | None, int]:
    expected_direction = _expected_direction(factor_matrix)
    scores: list[float] = []
    forward_returns: list[float] = []
    for row_index in range(len(factor_matrix.symbols)):
        if not _is_universe_eligible(bundle, row_index, alignment.signal_index):
            continue
        factor_value = _to_finite_float(factor_matrix.values[row_index][alignment.signal_index])
        forward_return = returns[row_index][alignment.execution_start_index]
        if factor_value is None or forward_return is None:
            continue
        scores.append(factor_value * expected_direction)
        forward_returns.append(forward_return)
    pearson = _pearson_correlation(scores, forward_returns)
    rank = _pearson_correlation(_rank_values(scores), _rank_values(forward_returns)) if len(scores) >= 2 else None
    return pearson, rank, len(scores)


def compute_daily_backtest_records(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    weight_matrix: FactorWeightMatrix,
    *,
    mode: str = BACKTEST_MODE_LONG_SHORT,
    holding_period_days: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> list[FactorDailyBacktestRecord]:
    resolved_mode = _resolve_mode(mode)
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    if list(weight_matrix.symbols) != list(factor_matrix.symbols):
        raise ValueError("weight_matrix symbols must match factor_matrix symbols.")
    if list(weight_matrix.dates) != list(factor_matrix.dates):
        raise ValueError("weight_matrix dates must match factor_matrix dates.")
    execution_price = _resolve_execution_price(config.execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    alignments = build_backtest_alignments(
        factor_matrix.dates,
        delay_days=config.delay_days,
        holding_period_days=resolved_holding_period,
        start_date=config.start_date or None,
        end_date=config.end_date or None,
        execution_price=execution_price,
    )
    execution_returns = build_execution_return_matrix(
        bundle,
        execution_price=execution_price,
        holding_period_days=resolved_holding_period,
    )
    daily_records: list[FactorDailyBacktestRecord] = []
    previous_net_weights = {symbol: 0.0 for symbol in factor_matrix.symbols}
    total_cost_bps = (
        config.transaction_cost_bps
        + config.slippage_bps
        + config.market_impact_bps
    )
    cost_per_turnover = bps_to_decimal_return(total_cost_bps)
    base_metadata = _coerce_metadata(metadata)

    for alignment in alignments:
        signal_index = alignment.signal_index
        execution_start_index = alignment.execution_start_index
        forward_returns = [
            execution_returns[row_index][execution_start_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        long_column = [
            weight_matrix.long_weights[row_index][signal_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        short_column = [
            weight_matrix.short_weights[row_index][signal_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        net_column = [
            weight_matrix.net_weights[row_index][signal_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        long_count = _nonzero_count(long_column)
        short_count = _nonzero_count(short_column)
        long_return = _weighted_return(long_column, forward_returns)
        short_return = _weighted_return(short_column, forward_returns)
        if resolved_mode == BACKTEST_MODE_LONG_ONLY:
            long_short_return = long_return
        elif long_count == 0 and short_count == 0:
            long_short_return = None
        elif (long_count > 0 and long_return is None) or (short_count > 0 and short_return is None):
            long_short_return = None
        else:
            long_short_return = (long_return or 0.0) + (short_return or 0.0)
        current_net_weights = _weights_for_column(
            factor_matrix.symbols,
            weight_matrix.net_weights,
            signal_index,
        )
        turnover = estimate_turnover(previous_net_weights, current_net_weights)
        previous_net_weights = current_net_weights
        after_cost_return = (
            None if long_short_return is None else long_short_return - turnover * cost_per_turnover
        )
        benchmark_return = _benchmark_return_at(bundle, execution_start_index, resolved_holding_period)
        excess_return = (
            None
            if after_cost_return is None or benchmark_return is None
            else after_cost_return - benchmark_return
        )
        coverage_ratio, missing_ratio = _coverage_for_signal(
            factor_matrix,
            bundle,
            signal_index,
        )
        pearson_ic, rank_ic, ic_sample_size = _ic_for_alignment(
            factor_matrix,
            bundle,
            execution_returns,
            alignment,
        )
        record_metadata = dict(base_metadata)
        record_metadata.update(
            {
                "alignment": alignment.to_dict(),
                "mode": resolved_mode,
                "holding_period_days": resolved_holding_period,
                "cost_bps": total_cost_bps,
                "ic": {
                    "pearson_ic": pearson_ic,
                    "rank_ic": rank_ic,
                    "sample_size": ic_sample_size,
                },
            }
        )
        daily_records.append(
            FactorDailyBacktestRecord(
                date=alignment.execution_end_date,
                signal_date=alignment.signal_date,
                execution_start_date=alignment.execution_start_date,
                execution_end_date=alignment.execution_end_date,
                long_return=long_return,
                short_return=short_return,
                long_short_return=long_short_return,
                after_cost_return=after_cost_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                turnover=turnover,
                long_count=long_count,
                short_count=short_count,
                coverage_ratio=coverage_ratio,
                missing_ratio=missing_ratio,
                metadata=record_metadata,
            )
        )
    return sorted(daily_records, key=lambda record: record.date)


def _factor_id_for_result(factor_matrix: FactorMatrix) -> str:
    return str(factor_matrix.factor_id or factor_matrix.matrix_id)


def _factor_version_for_result(factor_matrix: FactorMatrix) -> str:
    return str(factor_matrix.factor_version or "unversioned")


def _extract_ic_values(
    daily_records: Sequence[FactorDailyBacktestRecord],
    key: str,
) -> list[float | None]:
    values: list[float | None] = []
    for record in daily_records:
        ic_payload = record.metadata.get("ic", {})
        if isinstance(ic_payload, Mapping):
            values.append(ic_payload.get(key))  # type: ignore[arg-type]
    return values


def _average_int_field(
    daily_records: Sequence[FactorDailyBacktestRecord],
    field_name: str,
) -> float | None:
    values = [float(getattr(record, field_name)) for record in daily_records]
    return safe_mean(values)


def build_factor_backtest_result(
    *,
    factor_matrix: FactorMatrix,
    config: FactorBacktestConfig,
    daily_records: Sequence[FactorDailyBacktestRecord],
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestResult:
    sorted_records = sorted(daily_records, key=lambda record: record.date)
    after_cost_returns = [record.after_cost_return for record in sorted_records]
    before_cost_returns = [record.long_short_return for record in sorted_records]
    pearson_ic_values = _extract_ic_values(sorted_records, "pearson_ic")
    rank_ic_values = _extract_ic_values(sorted_records, "rank_ic")
    usable_ic_values = _usable_numbers(pearson_ic_values, field_name="ic_values")
    ic_mean = safe_mean(pearson_ic_values)
    ic_std = safe_std(pearson_ic_values)
    ic_count = len(usable_ic_values)
    icir = None if ic_mean is None or ic_std is None or abs(ic_std) <= _EPSILON else ic_mean / ic_std
    ic_t_stat = None
    if ic_mean is not None and ic_std is not None and abs(ic_std) > _EPSILON and ic_count > 1:
        ic_t_stat = ic_mean / ic_std * math.sqrt(ic_count)
    positive_ic_ratio = None
    if ic_count > 0:
        positive_ic_ratio = sum(1 for value in usable_ic_values if value > 0.0) / ic_count
    sample_days = sum(1 for value in after_cost_returns if value is not None)
    coverage_ratio = safe_mean([record.coverage_ratio for record in sorted_records]) or 0.0
    missing_ratio = safe_mean([record.missing_ratio for record in sorted_records]) or 0.0
    resolved_metadata = _coerce_metadata(metadata)
    holding_period_days = int(resolved_metadata.get("holding_period_days", 1) or 1)
    execution_price = _resolve_execution_price(
        str(resolved_metadata.get("execution_price") or config.execution_price or EXECUTION_PRICE_VWAP)
    )
    resolved_metadata.update(
        {
            "factor_backtest_schema_version": FACTOR_BACKTEST_SCHEMA_VERSION,
            "alignment_policy": "signal_delay_execution_window",
            "execution_price": execution_price,
            "delay_days": config.delay_days,
            "holding_period_days": holding_period_days,
            "costing_policy": {
                "transaction_cost_bps": config.transaction_cost_bps,
                "slippage_bps": config.slippage_bps,
                "market_impact_bps": config.market_impact_bps,
                "deduction": "turnover * decimal_total_bps",
            },
            "pass": "phase9_pass3",
        }
    )
    factor_id = _factor_id_for_result(factor_matrix)
    factor_version = _factor_version_for_result(factor_matrix)
    start_date = (
        config.start_date
        or (sorted_records[0].date if sorted_records else factor_matrix.dates[0])
    )
    end_date = (
        config.end_date
        or (sorted_records[-1].date if sorted_records else factor_matrix.dates[-1])
    )
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id=factor_id,
            factor_version=factor_version,
            config_id=config.config_id,
        ),
        factor_id=factor_id,
        factor_version=factor_version,
        config_id=config.config_id,
        start_date=start_date,
        end_date=end_date,
        sample_days=sample_days,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        ann_ret=annualized_return_from_daily(after_cost_returns),
        ann_vol=annualized_vol_from_daily(after_cost_returns),
        sharpe=sharpe_from_daily(after_cost_returns),
        max_drawdown=max_drawdown_from_returns(after_cost_returns),
        turnover_avg=safe_mean([record.turnover for record in sorted_records]),
        long_num_avg=_average_int_field(sorted_records, "long_count"),
        short_num_avg=_average_int_field(sorted_records, "short_count"),
        rank_ic_mean=safe_mean(rank_ic_values),
        ic_mean=ic_mean,
        icir=icir,
        ic_t_stat=ic_t_stat,
        positive_ic_ratio=positive_ic_ratio,
        top_bottom_spread=safe_mean(before_cost_returns),
        after_cost_top_bottom_spread=safe_mean(after_cost_returns),
        before_cost_sharpe=sharpe_from_daily(before_cost_returns),
        after_cost_sharpe=sharpe_from_daily(after_cost_returns),
        monotonicity_score=None,
        capacity_estimate=None,
        slice_metrics={},
        metadata=resolved_metadata,
    )


def _determine_mode(config: FactorBacktestConfig, mode: str | None) -> str:
    if mode is not None:
        return _resolve_mode(mode)
    if config.long_short:
        return BACKTEST_MODE_LONG_SHORT
    if config.long_only:
        return BACKTEST_MODE_LONG_ONLY
    raise ValueError("config must enable long_short or long_only.")


def run_single_factor_backtest(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    *,
    mode: str | None = None,
    holding_period_days: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> SingleFactorBacktestRun:
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    resolved_mode = _determine_mode(config, mode)
    execution_price = _resolve_execution_price(config.execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    base_metadata = _coerce_metadata(metadata)
    weight_matrix = build_quantile_weight_matrix(
        factor_matrix,
        bundle,
        config,
        mode=resolved_mode,
        metadata=base_metadata,
    )
    daily_records = compute_daily_backtest_records(
        factor_matrix,
        bundle,
        config,
        weight_matrix,
        mode=resolved_mode,
        holding_period_days=resolved_holding_period,
        metadata=base_metadata,
    )
    aggregate_result = build_factor_backtest_result(
        factor_matrix=factor_matrix,
        config=config,
        daily_records=daily_records,
        metadata={
            **base_metadata,
            "execution_price": execution_price,
            "holding_period_days": resolved_holding_period,
        },
    )
    run_id = make_factor_backtest_run_id(
        factor_matrix_id=factor_matrix.matrix_id,
        config_id=config.config_id,
        mode=resolved_mode,
        start_date=aggregate_result.start_date,
        end_date=aggregate_result.end_date,
    )
    weight_matrix.metadata = _coerce_metadata({
        **weight_matrix.metadata,
        "run_id": run_id,
    })
    for record in daily_records:
        record.metadata = _coerce_metadata({
            **record.metadata,
            "run_id": run_id,
            "daily_record_id": make_daily_record_id(run_id=run_id, date=record.date),
        })
    run_metadata = {
        **base_metadata,
        "factor_backtest_schema_version": FACTOR_BACKTEST_SCHEMA_VERSION,
        "offline_only": True,
        "pass": "phase9_pass3",
    }
    return SingleFactorBacktestRun(
        run_id=run_id,
        factor_matrix_id=factor_matrix.matrix_id,
        factor_id=factor_matrix.factor_id,
        factor_version=factor_matrix.factor_version,
        config_id=config.config_id,
        start_date=aggregate_result.start_date,
        end_date=aggregate_result.end_date,
        mode=resolved_mode,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=weight_matrix,
        daily_records=daily_records,
        aggregate_result=aggregate_result,
        metadata=run_metadata,
    )


__all__ = [
    "EXECUTION_PRICE_OPEN",
    "EXECUTION_PRICE_VWAP",
    "EXECUTION_PRICE_CLOSE",
    "BACKTEST_MODE_LONG_SHORT",
    "BACKTEST_MODE_LONG_ONLY",
    "WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE",
    "DEFAULT_FACTOR_BACKTEST_DIR",
    "DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME",
    "DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME",
    "DEFAULT_FACTOR_DAILY_RECORDS_FILENAME",
    "FactorBacktestAlignment",
    "FactorWeightMatrix",
    "FactorDailyBacktestRecord",
    "SingleFactorBacktestRun",
    "make_factor_weights_id",
    "make_factor_backtest_run_id",
    "make_daily_record_id",
    "validate_finite_number",
    "bps_to_decimal_return",
    "safe_mean",
    "safe_std",
    "compound_returns",
    "max_drawdown_from_returns",
    "annualized_return_from_daily",
    "annualized_vol_from_daily",
    "sharpe_from_daily",
    "estimate_turnover",
    "build_backtest_alignments",
    "build_execution_return_matrix",
    "build_quantile_weight_matrix",
    "compute_daily_backtest_records",
    "build_factor_backtest_result",
    "run_single_factor_backtest",
]
