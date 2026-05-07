"""Offline metric helpers for factor validation.

The helpers in this module operate on already-supplied in-memory return and
turnover series. They do not load provider data or alter factor admission
behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import FACTOR_ROBUSTNESS_SCHEMA_VERSION


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


def _non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric or None.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    number = _optional_finite_float(value, field_name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


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


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


@dataclass
class ReturnMetricSummary:
    schema_version: str = FACTOR_ROBUSTNESS_SCHEMA_VERSION
    metric_id: str = ""
    name: str = ""
    sample_count: int = 0
    mean_return: float | None = None
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    positive_return_ratio: float | None = None
    cumulative_return: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_ROBUSTNESS_SCHEMA_VERSION)
        self.metric_id = _non_empty_str(self.metric_id, "metric_id")
        self.name = _non_empty_str(self.name, "name")
        self.sample_count = _non_negative_int(self.sample_count, "sample_count")
        for field_name in (
            "mean_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "cumulative_return",
        ):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        if self.max_drawdown is not None and self.max_drawdown < 0.0:
            raise ValueError("max_drawdown must be non-negative.")
        self.positive_return_ratio = _unit_float_or_none(
            self.positive_return_ratio,
            "positive_return_ratio",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReturnMetricSummary":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_ROBUSTNESS_SCHEMA_VERSION)),
            metric_id=str(data.get("metric_id", "")),
            name=str(data.get("name", "")),
            sample_count=int(data.get("sample_count", 0)),
            mean_return=data.get("mean_return"),
            annualized_return=data.get("annualized_return"),
            annualized_volatility=data.get("annualized_volatility"),
            sharpe=data.get("sharpe"),
            max_drawdown=data.get("max_drawdown"),
            positive_return_ratio=data.get("positive_return_ratio"),
            cumulative_return=data.get("cumulative_return"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class TurnoverMetricSummary:
    schema_version: str = FACTOR_ROBUSTNESS_SCHEMA_VERSION
    metric_id: str = ""
    name: str = ""
    sample_count: int = 0
    average_turnover: float | None = None
    median_turnover: float | None = None
    max_turnover: float | None = None
    turnover_budget: float | None = None
    budget_breach_count: int = 0
    budget_breach_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_ROBUSTNESS_SCHEMA_VERSION)
        self.metric_id = _non_empty_str(self.metric_id, "metric_id")
        self.name = _non_empty_str(self.name, "name")
        self.sample_count = _non_negative_int(self.sample_count, "sample_count")
        for field_name in (
            "average_turnover",
            "median_turnover",
            "max_turnover",
            "turnover_budget",
        ):
            setattr(
                self,
                field_name,
                _optional_non_negative_float(getattr(self, field_name), field_name),
            )
        self.budget_breach_count = _non_negative_int(
            self.budget_breach_count,
            "budget_breach_count",
        )
        self.budget_breach_ratio = _unit_float_or_none(
            self.budget_breach_ratio,
            "budget_breach_ratio",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TurnoverMetricSummary":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_ROBUSTNESS_SCHEMA_VERSION)),
            metric_id=str(data.get("metric_id", "")),
            name=str(data.get("name", "")),
            sample_count=int(data.get("sample_count", 0)),
            average_turnover=data.get("average_turnover"),
            median_turnover=data.get("median_turnover"),
            max_turnover=data.get("max_turnover"),
            turnover_budget=data.get("turnover_budget"),
            budget_breach_count=int(data.get("budget_breach_count", 0)),
            budget_breach_ratio=data.get("budget_breach_ratio"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_metric_id(*, name: str, sample_count: int, metric_type: str) -> str:
    resolved_name = _non_empty_str(name, "name")
    resolved_type = _non_empty_str(metric_type, "metric_type")
    resolved_count = _non_negative_int(sample_count, "sample_count")
    parts = [resolved_name, resolved_count, resolved_type]
    return f"factor-metric-{_slug(resolved_type)}-{_slug(resolved_name)}-{_short_hash(parts)}"


def filter_none_finite(values: Sequence[float | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        output.append(_finite_float(value, "values"))
    return output


def cumulative_return(returns: Sequence[float | None]) -> float | None:
    numbers = filter_none_finite(returns)
    if not numbers:
        return None
    total = 1.0
    for value in numbers:
        total *= 1.0 + value
    return total - 1.0


def max_drawdown_from_returns(returns: Sequence[float | None]) -> float | None:
    numbers = filter_none_finite(returns)
    if not numbers:
        return None
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in numbers:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0.0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return max(0.0, drawdown)


def annualized_return_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    mean = _mean(filter_none_finite(returns))
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
    std = _std(filter_none_finite(returns))
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


def positive_ratio(values: Sequence[float | None]) -> float | None:
    numbers = filter_none_finite(values)
    if not numbers:
        return None
    return sum(1 for value in numbers if value > 0.0) / len(numbers)


def build_return_metric_summary(
    name: str,
    returns: Sequence[float | None],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ReturnMetricSummary:
    numbers = filter_none_finite(returns)
    sample_count = len(numbers)
    return ReturnMetricSummary(
        metric_id=make_metric_id(
            name=name,
            sample_count=sample_count,
            metric_type="return",
        ),
        name=name,
        sample_count=sample_count,
        mean_return=_mean(numbers),
        annualized_return=annualized_return_from_daily(numbers),
        annualized_volatility=annualized_vol_from_daily(numbers),
        sharpe=sharpe_from_daily(numbers),
        max_drawdown=max_drawdown_from_returns(numbers),
        positive_return_ratio=positive_ratio(numbers),
        cumulative_return=cumulative_return(numbers),
        metadata=_coerce_metadata(metadata),
    )


def build_turnover_metric_summary(
    name: str,
    turnovers: Sequence[float],
    *,
    turnover_budget: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TurnoverMetricSummary:
    numbers = filter_none_finite(turnovers)
    for value in numbers:
        if value < 0.0:
            raise ValueError("turnovers must be non-negative.")
    budget = _optional_non_negative_float(turnover_budget, "turnover_budget")
    breach_count = 0
    if budget is not None:
        breach_count = sum(1 for value in numbers if value > budget)
    breach_ratio = None if not numbers else breach_count / len(numbers)
    sample_count = len(numbers)
    return TurnoverMetricSummary(
        metric_id=make_metric_id(
            name=name,
            sample_count=sample_count,
            metric_type="turnover",
        ),
        name=name,
        sample_count=sample_count,
        average_turnover=_mean(numbers),
        median_turnover=_median(numbers),
        max_turnover=max(numbers) if numbers else None,
        turnover_budget=budget,
        budget_breach_count=breach_count,
        budget_breach_ratio=breach_ratio,
        metadata=_coerce_metadata(metadata),
    )


__all__ = [
    "ReturnMetricSummary",
    "TurnoverMetricSummary",
    "make_metric_id",
    "filter_none_finite",
    "cumulative_return",
    "max_drawdown_from_returns",
    "annualized_return_from_daily",
    "annualized_vol_from_daily",
    "sharpe_from_daily",
    "positive_ratio",
    "build_return_metric_summary",
    "build_turnover_metric_summary",
]
