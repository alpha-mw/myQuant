"""Contracts for offline single-factor backtests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.schema import FactorBacktestResult
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
    "SUPPORTED_EXECUTION_PRICES",
    "SUPPORTED_BACKTEST_MODES",
    "FactorBacktestAlignment",
    "FactorWeightMatrix",
    "FactorDailyBacktestRecord",
    "SingleFactorBacktestRun",
    "make_factor_weights_id",
    "make_factor_backtest_run_id",
    "make_daily_record_id",
]

