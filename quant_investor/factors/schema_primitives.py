"""Primitive constants and validation helpers for factor governance schemas."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


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

NUMERIC_RESULT_FIELDS = (
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


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_json_serializable(value: Any, label: str) -> Any:
    safe = json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(ensure_json_serializable(value, "metadata"))


def coerce_json_dict(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(ensure_json_serializable(value, label))


def non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, field_name)


def non_negative_float(value: Any, field_name: str) -> float:
    number = finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def unit_float(value: Any, field_name: str) -> float:
    number = finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return unit_float(value, field_name)


def positive_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def non_negative_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def non_negative_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return non_negative_int(value, field_name)


def ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip().lower()
    resolved_slug = re.sub(r"[^a-z0-9._-]+", "-", resolved)
    return resolved_slug.strip("-") or "unknown"


def short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(
        [json_safe(part) for part in parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


__all__ = [
    "ADMISSION_DECISION_APPROVE_PAPER_TRADING",
    "ADMISSION_DECISION_APPROVE_PRODUCTION",
    "ADMISSION_DECISION_DISABLE",
    "ADMISSION_DECISION_NEEDS_RESEARCH",
    "ADMISSION_DECISION_REJECT",
    "DEFAULT_DEPRECATED_FACTORS_FILENAME",
    "DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME",
    "DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME",
    "DEFAULT_FACTOR_DEFINITIONS_FILENAME",
    "DEFAULT_FACTOR_LIBRARY_DIR",
    "DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME",
    "DEFAULT_PRODUCTION_FACTORS_FILENAME",
    "FACTOR_FAMILY_CUSTOM",
    "FACTOR_FAMILY_GROWTH",
    "FACTOR_FAMILY_MOMENTUM",
    "FACTOR_FAMILY_PRICE",
    "FACTOR_FAMILY_QUALITY",
    "FACTOR_FAMILY_REVERSAL",
    "FACTOR_FAMILY_RISK",
    "FACTOR_FAMILY_SENTIMENT",
    "FACTOR_FAMILY_VALUE",
    "FACTOR_FAMILY_VOLATILITY",
    "FACTOR_FAMILY_VOLUME",
    "FACTOR_STATUS_BACKTESTED",
    "FACTOR_STATUS_DEPRECATED",
    "FACTOR_STATUS_DISABLED",
    "FACTOR_STATUS_DRAFT",
    "FACTOR_STATUS_PAPER_TRADING",
    "FACTOR_STATUS_PRODUCTION",
    "FACTOR_STATUS_REJECTED",
    "FACTOR_STATUS_RESEARCH_CANDIDATE",
    "FACTOR_STATUS_VALIDATED_RESEARCH",
    "NUMERIC_RESULT_FIELDS",
    "SUPPORTED_ADMISSION_DECISIONS",
    "SUPPORTED_FACTOR_FAMILIES",
    "SUPPORTED_FACTOR_STATUSES",
    "SUPPORTED_VALIDATION_VERDICTS",
    "VALIDATION_VERDICT_FAIL",
    "VALIDATION_VERDICT_PASS",
    "VALIDATION_VERDICT_WARN",
    "coerce_json_dict",
    "coerce_metadata",
    "ensure_json_serializable",
    "finite_float",
    "json_safe",
    "non_empty_str",
    "non_negative_float",
    "non_negative_int",
    "non_negative_int_or_none",
    "optional_finite_float",
    "optional_str",
    "ordered_unique",
    "positive_int",
    "short_hash",
    "slug",
    "unit_float",
    "unit_float_or_none",
    "validate_supported",
]
