"""Offline factor governance schema contracts.

This module defines deterministic dataclass contracts only. It does not load
market data, evaluate factor expressions, calculate matrices, or alter runtime
selection behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import (
    FACTOR_GOVERNANCE_SCHEMA_VERSION,
    FACTOR_LIBRARY_SCHEMA_VERSION,
)


FACTOR_STATUS_DRAFT = "draft"
FACTOR_STATUS_RESEARCH_CANDIDATE = "research_candidate"
FACTOR_STATUS_BACKTESTED = "backtested"
FACTOR_STATUS_VALIDATED_RESEARCH = "validated_research"
FACTOR_STATUS_PAPER_TRADING = "paper_trading"
FACTOR_STATUS_PRODUCTION = "production"
FACTOR_STATUS_DEPRECATED = "deprecated"
FACTOR_STATUS_REJECTED = "rejected"
FACTOR_STATUS_DISABLED = "disabled"

FACTOR_FAMILY_PRICE = "price"
FACTOR_FAMILY_VOLUME = "volume"
FACTOR_FAMILY_MOMENTUM = "momentum"
FACTOR_FAMILY_REVERSAL = "reversal"
FACTOR_FAMILY_VOLATILITY = "volatility"
FACTOR_FAMILY_QUALITY = "quality"
FACTOR_FAMILY_VALUE = "value"
FACTOR_FAMILY_GROWTH = "growth"
FACTOR_FAMILY_SENTIMENT = "sentiment"
FACTOR_FAMILY_RISK = "risk"
FACTOR_FAMILY_CUSTOM = "custom"

ADMISSION_DECISION_APPROVE_PRODUCTION = "approve_production"
ADMISSION_DECISION_APPROVE_PAPER_TRADING = "approve_paper_trading"
ADMISSION_DECISION_REJECT = "reject"
ADMISSION_DECISION_NEEDS_RESEARCH = "needs_research"
ADMISSION_DECISION_DISABLE = "disable"

VALIDATION_VERDICT_PASS = "pass"
VALIDATION_VERDICT_WARN = "warn"
VALIDATION_VERDICT_FAIL = "fail"

DEFAULT_FACTOR_LIBRARY_DIR = Path("data/factor_library")
DEFAULT_FACTOR_DEFINITIONS_FILENAME = "factor_definitions.jsonl"
DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME = "factor_backtest_results.jsonl"
DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME = "factor_validation_reports.jsonl"
DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME = "factor_admission_decisions.jsonl"
DEFAULT_PRODUCTION_FACTORS_FILENAME = "production_factors.json"
DEFAULT_DEPRECATED_FACTORS_FILENAME = "deprecated_factors.json"

SUPPORTED_FACTOR_STATUSES = {
    FACTOR_STATUS_DRAFT,
    FACTOR_STATUS_RESEARCH_CANDIDATE,
    FACTOR_STATUS_BACKTESTED,
    FACTOR_STATUS_VALIDATED_RESEARCH,
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_DEPRECATED,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_DISABLED,
}

SUPPORTED_FACTOR_FAMILIES = {
    FACTOR_FAMILY_PRICE,
    FACTOR_FAMILY_VOLUME,
    FACTOR_FAMILY_MOMENTUM,
    FACTOR_FAMILY_REVERSAL,
    FACTOR_FAMILY_VOLATILITY,
    FACTOR_FAMILY_QUALITY,
    FACTOR_FAMILY_VALUE,
    FACTOR_FAMILY_GROWTH,
    FACTOR_FAMILY_SENTIMENT,
    FACTOR_FAMILY_RISK,
    FACTOR_FAMILY_CUSTOM,
}

SUPPORTED_ADMISSION_DECISIONS = {
    ADMISSION_DECISION_APPROVE_PRODUCTION,
    ADMISSION_DECISION_APPROVE_PAPER_TRADING,
    ADMISSION_DECISION_REJECT,
    ADMISSION_DECISION_NEEDS_RESEARCH,
    ADMISSION_DECISION_DISABLE,
}

SUPPORTED_VALIDATION_VERDICTS = {
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    VALIDATION_VERDICT_FAIL,
}

_NUMERIC_RESULT_FIELDS = (
    "ann_ret",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "turnover_avg",
    "long_num_avg",
    "short_num_avg",
    "rank_ic_mean",
    "ic_mean",
    "icir",
    "ic_t_stat",
    "positive_ic_ratio",
    "top_bottom_spread",
    "after_cost_top_bottom_spread",
    "before_cost_sharpe",
    "after_cost_sharpe",
    "monotonicity_score",
    "capacity_estimate",
)


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


def _coerce_json_dict(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, label))


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
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


def _unit_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, field_name)


def _positive_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _non_negative_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field_name)


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


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


@dataclass
class FactorDefinition:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    factor_id: str = ""
    factor_name: str = ""
    factor_family: str = FACTOR_FAMILY_CUSTOM
    status: str = FACTOR_STATUS_DRAFT
    version: str = "v1"
    expression: str = ""
    input_fields: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    universe: str = ""
    benchmark: str | None = None
    expected_direction: float = 1.0
    rebalance_frequency: str = ""
    lookback_window: int | None = None
    delay_days: int = 1
    execution_price: str = ""
    winsorization_rule: str | None = None
    standardization_rule: str | None = None
    neutralization_rule: str | None = None
    missing_value_rule: str | None = None
    point_in_time_required: bool = True
    st_filter: bool = True
    suspension_filter: bool = True
    limit_up_down_filter: bool = True
    new_listing_min_days: int | None = None
    adjustment_rule: str | None = None
    industry_neutral: bool = False
    size_neutral: bool = False
    economic_rationale: str = ""
    owner: str | None = None
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_name = _non_empty_str(self.factor_name, "factor_name")
        self.factor_family = str(self.factor_family)
        _validate_supported(self.factor_family, "factor_family", SUPPORTED_FACTOR_FAMILIES)
        self.status = str(self.status)
        _validate_supported(self.status, "status", SUPPORTED_FACTOR_STATUSES)
        self.version = _non_empty_str(self.version, "version")
        self.expression = _non_empty_str(self.expression, "expression")
        self.input_fields = _ordered_unique(self.input_fields)
        if not self.input_fields:
            raise ValueError("input_fields must be non-empty.")
        self.data_sources = _ordered_unique(self.data_sources)
        self.universe = str(self.universe)
        self.benchmark = _optional_str(self.benchmark)
        self.expected_direction = _finite_float(self.expected_direction, "expected_direction")
        if self.expected_direction not in (-1.0, 1.0):
            raise ValueError("expected_direction must be either 1.0 or -1.0.")
        self.rebalance_frequency = str(self.rebalance_frequency)
        self.lookback_window = _non_negative_int_or_none(self.lookback_window, "lookback_window")
        self.delay_days = int(self.delay_days)
        if self.delay_days < 1:
            raise ValueError("delay_days must be >= 1.")
        self.execution_price = str(self.execution_price)
        self.winsorization_rule = _optional_str(self.winsorization_rule)
        self.standardization_rule = _optional_str(self.standardization_rule)
        self.neutralization_rule = _optional_str(self.neutralization_rule)
        self.missing_value_rule = _optional_str(self.missing_value_rule)
        self.point_in_time_required = bool(self.point_in_time_required)
        self.st_filter = bool(self.st_filter)
        self.suspension_filter = bool(self.suspension_filter)
        self.limit_up_down_filter = bool(self.limit_up_down_filter)
        self.new_listing_min_days = _non_negative_int_or_none(
            self.new_listing_min_days,
            "new_listing_min_days",
        )
        self.adjustment_rule = _optional_str(self.adjustment_rule)
        self.industry_neutral = bool(self.industry_neutral)
        self.size_neutral = bool(self.size_neutral)
        self.economic_rationale = _non_empty_str(self.economic_rationale, "economic_rationale")
        self.owner = _optional_str(self.owner)
        self.created_at = str(self.created_at)
        self.updated_at = _optional_str(self.updated_at)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorDefinition":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            factor_id=str(data.get("factor_id", "")),
            factor_name=str(data.get("factor_name", "")),
            factor_family=str(data.get("factor_family", FACTOR_FAMILY_CUSTOM)),
            status=str(data.get("status", FACTOR_STATUS_DRAFT)),
            version=str(data.get("version", "v1")),
            expression=str(data.get("expression", "")),
            input_fields=list(data.get("input_fields", []) or []),
            data_sources=list(data.get("data_sources", []) or []),
            universe=str(data.get("universe", "")),
            benchmark=data.get("benchmark"),
            expected_direction=float(data.get("expected_direction", 1.0)),
            rebalance_frequency=str(data.get("rebalance_frequency", "")),
            lookback_window=data.get("lookback_window"),
            delay_days=int(data.get("delay_days", 1)),
            execution_price=str(data.get("execution_price", "")),
            winsorization_rule=data.get("winsorization_rule"),
            standardization_rule=data.get("standardization_rule"),
            neutralization_rule=data.get("neutralization_rule"),
            missing_value_rule=data.get("missing_value_rule"),
            point_in_time_required=bool(data.get("point_in_time_required", True)),
            st_filter=bool(data.get("st_filter", True)),
            suspension_filter=bool(data.get("suspension_filter", True)),
            limit_up_down_filter=bool(data.get("limit_up_down_filter", True)),
            new_listing_min_days=data.get("new_listing_min_days"),
            adjustment_rule=data.get("adjustment_rule"),
            industry_neutral=bool(data.get("industry_neutral", False)),
            size_neutral=bool(data.get("size_neutral", False)),
            economic_rationale=str(data.get("economic_rationale", "")),
            owner=data.get("owner"),
            created_at=str(data.get("created_at", "")),
            updated_at=data.get("updated_at"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorBacktestConfig:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    config_id: str = ""
    universe: str = ""
    benchmark: str | None = None
    start_date: str = ""
    end_date: str = ""
    rebalance_frequency: str = ""
    delay_days: int = 1
    execution_price: str = ""
    long_short: bool = False
    long_only: bool = True
    quantile_count: int = 5
    long_quantile: int = 5
    short_quantile: int | None = None
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    max_participation_rate: float | None = None
    min_coverage_ratio: float = 0.0
    neutralize_industry: bool = False
    neutralize_size: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.config_id = str(self.config_id)
        self.universe = str(self.universe)
        self.benchmark = _optional_str(self.benchmark)
        self.start_date = str(self.start_date)
        self.end_date = str(self.end_date)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        self.rebalance_frequency = str(self.rebalance_frequency)
        self.delay_days = int(self.delay_days)
        if self.delay_days < 1:
            raise ValueError("delay_days must be >= 1.")
        self.execution_price = str(self.execution_price)
        self.long_short = bool(self.long_short)
        self.long_only = bool(self.long_only)
        if not (self.long_short or self.long_only):
            raise ValueError("At least one of long_short or long_only must be true.")
        self.quantile_count = int(self.quantile_count)
        if self.quantile_count < 2:
            raise ValueError("quantile_count must be >= 2.")
        self.long_quantile = int(self.long_quantile)
        if not 1 <= self.long_quantile <= self.quantile_count:
            raise ValueError("long_quantile must be between 1 and quantile_count.")
        if self.short_quantile is not None:
            self.short_quantile = int(self.short_quantile)
            if not 1 <= self.short_quantile <= self.quantile_count:
                raise ValueError("short_quantile must be between 1 and quantile_count.")
        if self.long_short and self.short_quantile is None:
            raise ValueError("short_quantile is required when long_short=True.")
        self.transaction_cost_bps = _non_negative_float(
            self.transaction_cost_bps,
            "transaction_cost_bps",
        )
        self.slippage_bps = _non_negative_float(self.slippage_bps, "slippage_bps")
        self.market_impact_bps = _non_negative_float(self.market_impact_bps, "market_impact_bps")
        self.max_participation_rate = _unit_float_or_none(
            self.max_participation_rate,
            "max_participation_rate",
        )
        self.min_coverage_ratio = _unit_float(self.min_coverage_ratio, "min_coverage_ratio")
        self.neutralize_industry = bool(self.neutralize_industry)
        self.neutralize_size = bool(self.neutralize_size)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorBacktestConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "")),
            universe=str(data.get("universe", "")),
            benchmark=data.get("benchmark"),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            rebalance_frequency=str(data.get("rebalance_frequency", "")),
            delay_days=int(data.get("delay_days", 1)),
            execution_price=str(data.get("execution_price", "")),
            long_short=bool(data.get("long_short", False)),
            long_only=bool(data.get("long_only", True)),
            quantile_count=int(data.get("quantile_count", 5)),
            long_quantile=int(data.get("long_quantile", 5)),
            short_quantile=data.get("short_quantile"),
            transaction_cost_bps=float(data.get("transaction_cost_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            market_impact_bps=float(data.get("market_impact_bps", 0.0)),
            max_participation_rate=data.get("max_participation_rate"),
            min_coverage_ratio=float(data.get("min_coverage_ratio", 0.0)),
            neutralize_industry=bool(data.get("neutralize_industry", False)),
            neutralize_size=bool(data.get("neutralize_size", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorValidationThresholds:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    min_sample_days: int = 750
    min_coverage_ratio: float = 0.80
    min_rank_ic_mean: float = 0.02
    min_icir: float = 0.30
    min_ic_t_stat: float = 3.0
    min_after_cost_sharpe: float = 0.80
    max_drawdown: float | None = None
    max_turnover: float | None = None
    min_positive_ic_ratio: float = 0.55
    max_correlation_with_production: float = 0.70
    require_positive_after_cost_spread: bool = True
    require_monotonic_quantiles: bool = False
    require_point_in_time: bool = True
    production_revalidation_days: int = 90
    paper_trading_revalidation_days: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.min_sample_days = _positive_int(self.min_sample_days, "min_sample_days")
        self.min_coverage_ratio = _unit_float(self.min_coverage_ratio, "min_coverage_ratio")
        self.min_rank_ic_mean = _finite_float(self.min_rank_ic_mean, "min_rank_ic_mean")
        self.min_icir = _finite_float(self.min_icir, "min_icir")
        self.min_ic_t_stat = _finite_float(self.min_ic_t_stat, "min_ic_t_stat")
        self.min_after_cost_sharpe = _finite_float(
            self.min_after_cost_sharpe,
            "min_after_cost_sharpe",
        )
        self.max_drawdown = _unit_float_or_none(self.max_drawdown, "max_drawdown")
        self.max_turnover = _unit_float_or_none(self.max_turnover, "max_turnover")
        self.min_positive_ic_ratio = _unit_float(
            self.min_positive_ic_ratio,
            "min_positive_ic_ratio",
        )
        self.max_correlation_with_production = _unit_float(
            self.max_correlation_with_production,
            "max_correlation_with_production",
        )
        self.require_positive_after_cost_spread = bool(self.require_positive_after_cost_spread)
        self.require_monotonic_quantiles = bool(self.require_monotonic_quantiles)
        self.require_point_in_time = bool(self.require_point_in_time)
        self.production_revalidation_days = _positive_int(
            self.production_revalidation_days,
            "production_revalidation_days",
        )
        self.paper_trading_revalidation_days = _positive_int(
            self.paper_trading_revalidation_days,
            "paper_trading_revalidation_days",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorValidationThresholds":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            min_sample_days=int(data.get("min_sample_days", 750)),
            min_coverage_ratio=float(data.get("min_coverage_ratio", 0.80)),
            min_rank_ic_mean=float(data.get("min_rank_ic_mean", 0.02)),
            min_icir=float(data.get("min_icir", 0.30)),
            min_ic_t_stat=float(data.get("min_ic_t_stat", 3.0)),
            min_after_cost_sharpe=float(data.get("min_after_cost_sharpe", 0.80)),
            max_drawdown=data.get("max_drawdown"),
            max_turnover=data.get("max_turnover"),
            min_positive_ic_ratio=float(data.get("min_positive_ic_ratio", 0.55)),
            max_correlation_with_production=float(data.get("max_correlation_with_production", 0.70)),
            require_positive_after_cost_spread=bool(
                data.get("require_positive_after_cost_spread", True)
            ),
            require_monotonic_quantiles=bool(data.get("require_monotonic_quantiles", False)),
            require_point_in_time=bool(data.get("require_point_in_time", True)),
            production_revalidation_days=int(data.get("production_revalidation_days", 90)),
            paper_trading_revalidation_days=int(data.get("paper_trading_revalidation_days", 30)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorBacktestResult:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    result_id: str = ""
    factor_id: str = ""
    factor_version: str = ""
    config_id: str = ""
    start_date: str = ""
    end_date: str = ""
    sample_days: int = 0
    coverage_ratio: float = 0.0
    missing_ratio: float = 0.0
    ann_ret: float | None = None
    ann_vol: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    turnover_avg: float | None = None
    long_num_avg: float | None = None
    short_num_avg: float | None = None
    rank_ic_mean: float | None = None
    ic_mean: float | None = None
    icir: float | None = None
    ic_t_stat: float | None = None
    positive_ic_ratio: float | None = None
    top_bottom_spread: float | None = None
    after_cost_top_bottom_spread: float | None = None
    before_cost_sharpe: float | None = None
    after_cost_sharpe: float | None = None
    monotonicity_score: float | None = None
    capacity_estimate: float | None = None
    slice_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.result_id = _non_empty_str(self.result_id, "result_id")
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.start_date = str(self.start_date)
        self.end_date = str(self.end_date)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        self.sample_days = _non_negative_int(self.sample_days, "sample_days")
        self.coverage_ratio = _unit_float(self.coverage_ratio, "coverage_ratio")
        self.missing_ratio = _unit_float(self.missing_ratio, "missing_ratio")
        for field_name in _NUMERIC_RESULT_FIELDS:
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        self.slice_metrics = {
            str(key): dict(_ensure_json_serializable(value, f"slice_metrics.{key}"))
            for key, value in sorted(self.slice_metrics.items(), key=lambda item: str(item[0]))
        }
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorBacktestResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            result_id=str(data.get("result_id", "")),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            config_id=str(data.get("config_id", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            sample_days=int(data.get("sample_days", 0)),
            coverage_ratio=float(data.get("coverage_ratio", 0.0)),
            missing_ratio=float(data.get("missing_ratio", 0.0)),
            ann_ret=data.get("ann_ret"),
            ann_vol=data.get("ann_vol"),
            sharpe=data.get("sharpe"),
            max_drawdown=data.get("max_drawdown"),
            turnover_avg=data.get("turnover_avg"),
            long_num_avg=data.get("long_num_avg"),
            short_num_avg=data.get("short_num_avg"),
            rank_ic_mean=data.get("rank_ic_mean"),
            ic_mean=data.get("ic_mean"),
            icir=data.get("icir"),
            ic_t_stat=data.get("ic_t_stat"),
            positive_ic_ratio=data.get("positive_ic_ratio"),
            top_bottom_spread=data.get("top_bottom_spread"),
            after_cost_top_bottom_spread=data.get("after_cost_top_bottom_spread"),
            before_cost_sharpe=data.get("before_cost_sharpe"),
            after_cost_sharpe=data.get("after_cost_sharpe"),
            monotonicity_score=data.get("monotonicity_score"),
            capacity_estimate=data.get("capacity_estimate"),
            slice_metrics=dict(data.get("slice_metrics", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorValidationReport:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    report_id: str = ""
    factor_id: str = ""
    factor_version: str = ""
    generated_at: str = ""
    backtest_result_id: str = ""
    thresholds: FactorValidationThresholds = field(default_factory=FactorValidationThresholds)
    overall_verdict: str = VALIDATION_VERDICT_FAIL
    gate_results: dict[str, str] = field(default_factory=dict)
    failed_gates: list[str] = field(default_factory=list)
    warning_gates: list[str] = field(default_factory=list)
    metric_snapshot: dict[str, Any] = field(default_factory=dict)
    correlation_snapshot: dict[str, Any] = field(default_factory=dict)
    capacity_snapshot: dict[str, Any] = field(default_factory=dict)
    point_in_time_snapshot: dict[str, Any] = field(default_factory=dict)
    recommended_status: str = FACTOR_STATUS_REJECTED
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.generated_at = str(self.generated_at)
        self.backtest_result_id = _non_empty_str(self.backtest_result_id, "backtest_result_id")
        if not isinstance(self.thresholds, FactorValidationThresholds):
            self.thresholds = FactorValidationThresholds.from_dict(self.thresholds)
        self.overall_verdict = str(self.overall_verdict)
        _validate_supported(self.overall_verdict, "overall_verdict", SUPPORTED_VALIDATION_VERDICTS)
        self.gate_results = {
            str(key): str(value)
            for key, value in sorted(self.gate_results.items(), key=lambda item: str(item[0]))
        }
        for gate, verdict in self.gate_results.items():
            _validate_supported(verdict, f"gate_results.{gate}", SUPPORTED_VALIDATION_VERDICTS)
        self.failed_gates = _ordered_unique(self.failed_gates)
        self.warning_gates = _ordered_unique(self.warning_gates)
        self.metric_snapshot = _coerce_json_dict(self.metric_snapshot, "metric_snapshot")
        self.correlation_snapshot = _coerce_json_dict(
            self.correlation_snapshot,
            "correlation_snapshot",
        )
        self.capacity_snapshot = _coerce_json_dict(self.capacity_snapshot, "capacity_snapshot")
        self.point_in_time_snapshot = _coerce_json_dict(
            self.point_in_time_snapshot,
            "point_in_time_snapshot",
        )
        self.recommended_status = str(self.recommended_status)
        _validate_supported(self.recommended_status, "recommended_status", SUPPORTED_FACTOR_STATUSES)
        self.rationale = _non_empty_str(self.rationale, "rationale")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorValidationReport":
        data = dict(payload)
        threshold_payload = data.get("thresholds", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            generated_at=str(data.get("generated_at", "")),
            backtest_result_id=str(data.get("backtest_result_id", "")),
            thresholds=FactorValidationThresholds.from_dict(threshold_payload)
            if isinstance(threshold_payload, Mapping)
            else FactorValidationThresholds(),
            overall_verdict=str(data.get("overall_verdict", VALIDATION_VERDICT_FAIL)),
            gate_results=dict(data.get("gate_results", {}) or {}),
            failed_gates=list(data.get("failed_gates", []) or []),
            warning_gates=list(data.get("warning_gates", []) or []),
            metric_snapshot=dict(data.get("metric_snapshot", {}) or {}),
            correlation_snapshot=dict(data.get("correlation_snapshot", {}) or {}),
            capacity_snapshot=dict(data.get("capacity_snapshot", {}) or {}),
            point_in_time_snapshot=dict(data.get("point_in_time_snapshot", {}) or {}),
            recommended_status=str(data.get("recommended_status", FACTOR_STATUS_REJECTED)),
            rationale=str(data.get("rationale", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorAdmissionDecision:
    schema_version: str = FACTOR_GOVERNANCE_SCHEMA_VERSION
    decision_id: str = ""
    factor_id: str = ""
    factor_version: str = ""
    validation_report_id: str | None = None
    decision: str = ADMISSION_DECISION_NEEDS_RESEARCH
    target_status: str = FACTOR_STATUS_RESEARCH_CANDIDATE
    decided_at: str = ""
    decided_by: str = ""
    rationale: str = ""
    expires_at: str | None = None
    conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_GOVERNANCE_SCHEMA_VERSION)
        self.decision_id = _non_empty_str(self.decision_id, "decision_id")
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.validation_report_id = _optional_str(self.validation_report_id)
        self.decision = str(self.decision)
        _validate_supported(self.decision, "decision", SUPPORTED_ADMISSION_DECISIONS)
        self.target_status = str(self.target_status)
        _validate_supported(self.target_status, "target_status", SUPPORTED_FACTOR_STATUSES)
        self.decided_at = str(self.decided_at)
        self.decided_by = _non_empty_str(self.decided_by, "decided_by")
        self.rationale = _non_empty_str(self.rationale, "rationale")
        self.expires_at = _optional_str(self.expires_at)
        self.conditions = _ordered_unique(self.conditions)
        self.metadata = _coerce_metadata(self.metadata)
        if (
            self.decision == ADMISSION_DECISION_APPROVE_PRODUCTION
            and not self.validation_report_id
        ):
            raise ValueError("approve_production requires validation_report_id.")
        manual_override = bool(self.metadata.get("manual_override", False))
        if (
            self.decision == ADMISSION_DECISION_APPROVE_PAPER_TRADING
            and not self.validation_report_id
            and not manual_override
        ):
            raise ValueError(
                "approve_paper_trading requires validation_report_id unless manual_override=True."
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorAdmissionDecision":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_GOVERNANCE_SCHEMA_VERSION)),
            decision_id=str(data.get("decision_id", "")),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            validation_report_id=data.get("validation_report_id"),
            decision=str(data.get("decision", ADMISSION_DECISION_NEEDS_RESEARCH)),
            target_status=str(data.get("target_status", FACTOR_STATUS_RESEARCH_CANDIDATE)),
            decided_at=str(data.get("decided_at", "")),
            decided_by=str(data.get("decided_by", "")),
            rationale=str(data.get("rationale", "")),
            expires_at=data.get("expires_at"),
            conditions=list(data.get("conditions", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorLibraryEntry:
    schema_version: str = FACTOR_LIBRARY_SCHEMA_VERSION
    factor_id: str = ""
    factor_version: str = ""
    status: str = FACTOR_STATUS_PAPER_TRADING
    admission_decision_id: str = ""
    validation_report_id: str | None = None
    production_since: str | None = None
    paper_trading_since: str | None = None
    deprecated_at: str | None = None
    disabled_at: str | None = None
    expires_at: str | None = None
    last_revalidation_at: str | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_LIBRARY_SCHEMA_VERSION)
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.status = str(self.status)
        _validate_supported(self.status, "status", SUPPORTED_FACTOR_STATUSES)
        self.admission_decision_id = _non_empty_str(
            self.admission_decision_id,
            "admission_decision_id",
        )
        self.validation_report_id = _optional_str(self.validation_report_id)
        self.production_since = _optional_str(self.production_since)
        self.paper_trading_since = _optional_str(self.paper_trading_since)
        self.deprecated_at = _optional_str(self.deprecated_at)
        self.disabled_at = _optional_str(self.disabled_at)
        self.expires_at = _optional_str(self.expires_at)
        self.last_revalidation_at = _optional_str(self.last_revalidation_at)
        self.owner = _optional_str(self.owner)
        self.tags = _ordered_unique(self.tags)
        self.metadata = _coerce_metadata(self.metadata)
        if self.status == FACTOR_STATUS_PRODUCTION:
            if not self.production_since:
                raise ValueError("production status requires production_since.")
            if not self.validation_report_id:
                raise ValueError("production status requires validation_report_id.")
            if not self.admission_decision_id:
                raise ValueError("production status requires admission_decision_id.")

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorLibraryEntry":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_LIBRARY_SCHEMA_VERSION)),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            status=str(data.get("status", FACTOR_STATUS_PAPER_TRADING)),
            admission_decision_id=str(data.get("admission_decision_id", "")),
            validation_report_id=data.get("validation_report_id"),
            production_since=data.get("production_since"),
            paper_trading_since=data.get("paper_trading_since"),
            deprecated_at=data.get("deprecated_at"),
            disabled_at=data.get("disabled_at"),
            expires_at=data.get("expires_at"),
            last_revalidation_at=data.get("last_revalidation_at"),
            owner=data.get("owner"),
            tags=list(data.get("tags", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ProductionFactorLibrary:
    schema_version: str = FACTOR_LIBRARY_SCHEMA_VERSION
    library_id: str = ""
    generated_at: str = ""
    entries: list[FactorLibraryEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_LIBRARY_SCHEMA_VERSION)
        self.library_id = _non_empty_str(self.library_id, "library_id")
        self.generated_at = str(self.generated_at)
        self.entries = [
            entry if isinstance(entry, FactorLibraryEntry) else FactorLibraryEntry.from_dict(entry)
            for entry in self.entries
        ]
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            if entry.status != FACTOR_STATUS_PRODUCTION:
                raise ValueError("ProductionFactorLibrary entries must have status == production.")
            key = (entry.factor_id, entry.factor_version)
            if key in seen:
                raise ValueError(f"Duplicate production factor entry: {entry.factor_id} {entry.factor_version}")
            seen.add(key)
        self.entries = sorted(self.entries, key=lambda entry: (entry.factor_id, entry.factor_version))
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "library_id": self.library_id,
            "generated_at": self.generated_at,
            "entries": [entry.to_dict() for entry in self.entries],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProductionFactorLibrary":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_LIBRARY_SCHEMA_VERSION)),
            library_id=str(data.get("library_id", "")),
            generated_at=str(data.get("generated_at", "")),
            entries=[
                FactorLibraryEntry.from_dict(entry)
                for entry in list(data.get("entries", []) or [])
                if isinstance(entry, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_factor_id(*, factor_name: str, factor_family: str, expression: str) -> str:
    parts = [str(factor_name), str(factor_family), str(expression)]
    return f"factor-{_slug(factor_family)}-{_slug(factor_name)}-{_short_hash(parts)}"


def make_backtest_config_id(config: FactorBacktestConfig) -> str:
    payload = config.to_dict()
    payload["config_id"] = ""
    parts = [
        payload.get("universe", ""),
        payload.get("start_date", ""),
        payload.get("end_date", ""),
        payload,
    ]
    return (
        f"factor-bt-config-{_slug(config.universe)}-{_slug(config.start_date)}-"
        f"{_slug(config.end_date)}-{_short_hash(parts)}"
    )


def make_backtest_result_id(*, factor_id: str, factor_version: str, config_id: str) -> str:
    parts = [str(factor_id), str(factor_version), str(config_id)]
    return f"factor-bt-result-{_slug(factor_id)}-{_slug(factor_version)}-{_short_hash(parts)}"


def make_validation_report_id(*, factor_id: str, factor_version: str, backtest_result_id: str) -> str:
    parts = [str(factor_id), str(factor_version), str(backtest_result_id)]
    return f"factor-validation-{_slug(factor_id)}-{_slug(factor_version)}-{_short_hash(parts)}"


def make_admission_decision_id(
    *,
    factor_id: str,
    factor_version: str,
    decision: str,
    decided_at: str,
) -> str:
    parts = [str(factor_id), str(factor_version), str(decision), str(decided_at)]
    return f"factor-admission-{_slug(factor_id)}-{_slug(decision)}-{_short_hash(parts)}"


def make_production_library_id(entries: Sequence[FactorLibraryEntry]) -> str:
    ordered = sorted(entries, key=lambda entry: (entry.factor_id, entry.factor_version))
    parts = [entry.to_dict() for entry in ordered]
    return f"production-factor-library-{len(ordered)}-{_short_hash(parts)}"


__all__ = [
    "FACTOR_STATUS_DRAFT",
    "FACTOR_STATUS_RESEARCH_CANDIDATE",
    "FACTOR_STATUS_BACKTESTED",
    "FACTOR_STATUS_VALIDATED_RESEARCH",
    "FACTOR_STATUS_PAPER_TRADING",
    "FACTOR_STATUS_PRODUCTION",
    "FACTOR_STATUS_DEPRECATED",
    "FACTOR_STATUS_REJECTED",
    "FACTOR_STATUS_DISABLED",
    "FACTOR_FAMILY_PRICE",
    "FACTOR_FAMILY_VOLUME",
    "FACTOR_FAMILY_MOMENTUM",
    "FACTOR_FAMILY_REVERSAL",
    "FACTOR_FAMILY_VOLATILITY",
    "FACTOR_FAMILY_QUALITY",
    "FACTOR_FAMILY_VALUE",
    "FACTOR_FAMILY_GROWTH",
    "FACTOR_FAMILY_SENTIMENT",
    "FACTOR_FAMILY_RISK",
    "FACTOR_FAMILY_CUSTOM",
    "ADMISSION_DECISION_APPROVE_PRODUCTION",
    "ADMISSION_DECISION_APPROVE_PAPER_TRADING",
    "ADMISSION_DECISION_REJECT",
    "ADMISSION_DECISION_NEEDS_RESEARCH",
    "ADMISSION_DECISION_DISABLE",
    "VALIDATION_VERDICT_PASS",
    "VALIDATION_VERDICT_WARN",
    "VALIDATION_VERDICT_FAIL",
    "DEFAULT_FACTOR_LIBRARY_DIR",
    "DEFAULT_FACTOR_DEFINITIONS_FILENAME",
    "DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME",
    "DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME",
    "DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME",
    "DEFAULT_PRODUCTION_FACTORS_FILENAME",
    "DEFAULT_DEPRECATED_FACTORS_FILENAME",
    "SUPPORTED_FACTOR_STATUSES",
    "SUPPORTED_FACTOR_FAMILIES",
    "SUPPORTED_ADMISSION_DECISIONS",
    "SUPPORTED_VALIDATION_VERDICTS",
    "FactorDefinition",
    "FactorBacktestConfig",
    "FactorValidationThresholds",
    "FactorBacktestResult",
    "FactorValidationReport",
    "FactorAdmissionDecision",
    "FactorLibraryEntry",
    "ProductionFactorLibrary",
    "make_factor_id",
    "make_backtest_config_id",
    "make_backtest_result_id",
    "make_validation_report_id",
    "make_admission_decision_id",
    "make_production_library_id",
]
