"""Offline A-share tradability and execution-feasibility audits.

This module diagnoses whether factor backtest weights would have been
executable under local A-share trading constraints. It is audit-only: helpers
here do not change stock selection, factor scores, RiskGuard, portfolio
construction, target weights, orders, providers, LLMs, or execution.
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

from quant_investor.factors.backtest import FactorWeightMatrix, SingleFactorBacktestRun
from quant_investor.factors.matrix import MatrixDataBundle
from quant_investor.versioning import (
    FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION,
    FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
)


TRADABILITY_AUDIT_PASS = "pass"
TRADABILITY_AUDIT_WARN = "warn"
TRADABILITY_AUDIT_FAIL = "fail"

TRADABILITY_ISSUE_INFO = "info"
TRADABILITY_ISSUE_WARNING = "warning"
TRADABILITY_ISSUE_BLOCKER = "blocker"

TRADABILITY_ISSUE_SUSPENDED = "suspended"
TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED = "limit_up_buy_blocked"
TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED = "limit_down_sell_blocked"
TRADABILITY_ISSUE_ST_FILTERED = "st_filtered"
TRADABILITY_ISSUE_DELISTED = "delisted"
TRADABILITY_ISSUE_NEW_LISTING = "new_listing"
TRADABILITY_ISSUE_NO_VALID_PRICE = "no_valid_price"
TRADABILITY_ISSUE_NO_VALID_VOLUME = "no_valid_volume"
TRADABILITY_ISSUE_LOW_AMOUNT = "low_amount"
TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION = "blocked_buy_transition"
TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION = "blocked_sell_transition"
TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD = "missing_tradability_field"
TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH = "mask_shape_mismatch"

EXECUTION_AUDIT_STATUS_FEASIBLE = "feasible"
EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE = "partially_feasible"
EXECUTION_AUDIT_STATUS_BLOCKED = "blocked"

TRADE_DIRECTION_BUY = "buy"
TRADE_DIRECTION_SELL = "sell"
TRADE_DIRECTION_HOLD = "hold"

FIELD_SUSPENDED = "suspended"
FIELD_LIMIT_UP = "limit_up"
FIELD_LIMIT_DOWN = "limit_down"
FIELD_IS_ST = "is_st"
FIELD_DELISTED = "delisted"
FIELD_LISTING_DAYS = "listing_days"
FIELD_LISTING_DATE = "listing_date"
FIELD_VALID_PRICE = "valid_price"
FIELD_VALID_VOLUME = "valid_volume"
FIELD_LOW_LIQUIDITY = "low_liquidity"
FIELD_AMOUNT = "amount"
FIELD_VOLUME = "volume"
FIELD_OPEN = "open"
FIELD_CLOSE = "close"
FIELD_VWAP = "vwap"

DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR = Path("data/factor_library/tradability_audit")
DEFAULT_TRADABILITY_MASKS_FILENAME = "tradability_masks.jsonl"
DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME = "tradability_audit_reports.jsonl"
DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME = "execution_feasibility_reports.jsonl"
DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME = "tradability_audit_report.md"
DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME = "execution_feasibility_report.md"

TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE = (
    "This tradability audit is offline-only and does not alter official scoring, "
    "stock selection, posterior, RiskGuard, PortfolioConstructor, target weights, "
    "orders, providers, LLMs, or execution."
)
EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE = (
    "This execution feasibility audit is offline-only and does not alter official "
    "scoring, stock selection, posterior, RiskGuard, PortfolioConstructor, target "
    "weights, orders, providers, LLMs, or execution."
)

SUPPORTED_TRADABILITY_AUDIT_VERDICTS = {
    TRADABILITY_AUDIT_PASS,
    TRADABILITY_AUDIT_WARN,
    TRADABILITY_AUDIT_FAIL,
}
SUPPORTED_TRADABILITY_ISSUE_SEVERITIES = {
    TRADABILITY_ISSUE_INFO,
    TRADABILITY_ISSUE_WARNING,
    TRADABILITY_ISSUE_BLOCKER,
}
SUPPORTED_TRADABILITY_ISSUE_CODES = {
    TRADABILITY_ISSUE_SUSPENDED,
    TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    TRADABILITY_ISSUE_ST_FILTERED,
    TRADABILITY_ISSUE_DELISTED,
    TRADABILITY_ISSUE_NEW_LISTING,
    TRADABILITY_ISSUE_NO_VALID_PRICE,
    TRADABILITY_ISSUE_NO_VALID_VOLUME,
    TRADABILITY_ISSUE_LOW_AMOUNT,
    TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
    TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
    TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD,
    TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH,
}
SUPPORTED_EXECUTION_AUDIT_STATUSES = {
    EXECUTION_AUDIT_STATUS_FEASIBLE,
    EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE,
    EXECUTION_AUDIT_STATUS_BLOCKED,
}
SUPPORTED_TRADE_DIRECTIONS = {
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_SELL,
    TRADE_DIRECTION_HOLD,
}

_FLOAT_TOLERANCE = 1e-12
_SEVERITY_ORDER = {
    TRADABILITY_ISSUE_BLOCKER: 0,
    TRADABILITY_ISSUE_WARNING: 1,
    TRADABILITY_ISSUE_INFO: 2,
}
_BLOCKER_ISSUE_CODES = {
    TRADABILITY_ISSUE_SUSPENDED,
    TRADABILITY_ISSUE_DELISTED,
    TRADABILITY_ISSUE_NO_VALID_PRICE,
    TRADABILITY_ISSUE_NO_VALID_VOLUME,
    TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
    TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
    TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH,
}
_WARNING_ISSUE_CODES = {
    TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    TRADABILITY_ISSUE_ST_FILTERED,
    TRADABILITY_ISSUE_NEW_LISTING,
    TRADABILITY_ISSUE_LOW_AMOUNT,
}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool; got {value!r}.")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative integer; got {value!r}.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative finite float or None.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be non-negative finite float or None.")
    return number


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite float; got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite float; got {value!r}.")
    return number


def _to_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    return number if math.isfinite(number) else None


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


def _coerce_dates(values: Sequence[Any]) -> list[str]:
    output = [str(value).strip() for value in values if str(value).strip()]
    if not output:
        raise ValueError("dates must be non-empty.")
    parsed: list[date] = []
    for value in output:
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dates must be ISO dates; got {value!r}.") from exc
        if parsed_value.isoformat() != value:
            raise ValueError(f"dates must be canonical ISO dates; got {value!r}.")
        parsed.append(parsed_value)
    if any(current >= next_value for current, next_value in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly ascending ISO dates.")
    return output


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


def _empty_bool_matrix(symbols: Sequence[str], dates: Sequence[str], value: bool) -> list[list[bool]]:
    return [[value for _ in dates] for _ in symbols]


def _empty_issue_tensor(symbols: Sequence[str], dates: Sequence[str]) -> list[list[list[str]]]:
    return [[[] for _ in dates] for _ in symbols]


def _validate_matrix_shape(
    values: Sequence[Sequence[Any]],
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
    field_name: str,
) -> None:
    if len(values) != len(symbols):
        raise ValueError(f"{field_name} must have {len(symbols)} rows; got {len(values)}.")
    for row_index, row in enumerate(values):
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise ValueError(f"{field_name} row {row_index} must be a sequence.")
        if len(row) != len(dates):
            raise ValueError(
                f"{field_name} row {row_index} must have {len(dates)} columns; got {len(row)}."
            )


def _coerce_bool_mask(
    values: Sequence[Sequence[Any]],
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
    field_name: str,
) -> list[list[bool]]:
    _validate_matrix_shape(values, symbols=symbols, dates=dates, field_name=field_name)
    output: list[list[bool]] = []
    for row_index, row in enumerate(values):
        output_row: list[bool] = []
        for column_index, value in enumerate(row):
            if not isinstance(value, bool):
                raise ValueError(
                    f"{field_name}[{row_index}][{column_index}] must be bool; got {value!r}."
                )
            output_row.append(value)
        output.append(output_row)
    return output


def _sorted_issue_codes(values: Sequence[Any]) -> list[str]:
    issue_codes = sorted({str(value).strip() for value in values if str(value).strip()})
    for issue_code in issue_codes:
        _validate_supported(issue_code, "issue_code", SUPPORTED_TRADABILITY_ISSUE_CODES)
    return issue_codes


def _coerce_issue_tensor(
    values: Sequence[Sequence[Sequence[Any]]],
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
) -> list[list[list[str]]]:
    if len(values) != len(symbols):
        raise ValueError(
            f"issue_codes_by_cell must have {len(symbols)} rows; got {len(values)}."
        )
    output: list[list[list[str]]] = []
    for row_index, row in enumerate(values):
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise ValueError(f"issue_codes_by_cell row {row_index} must be a sequence.")
        if len(row) != len(dates):
            raise ValueError(
                "issue_codes_by_cell row "
                f"{row_index} must have {len(dates)} columns; got {len(row)}."
            )
        output_row: list[list[str]] = []
        for column_index, codes in enumerate(row):
            if isinstance(codes, (str, bytes, bytearray)) or not isinstance(codes, Sequence):
                raise ValueError(
                    "issue_codes_by_cell"
                    f"[{row_index}][{column_index}] must be a sequence."
                )
            output_row.append(_sorted_issue_codes(codes))
        output.append(output_row)
    return output


def _issue_severity(issue_code: str) -> str:
    if issue_code in _BLOCKER_ISSUE_CODES:
        return TRADABILITY_ISSUE_BLOCKER
    if issue_code in _WARNING_ISSUE_CODES:
        return TRADABILITY_ISSUE_WARNING
    return TRADABILITY_ISSUE_INFO


def _issue_sort_key(issue: "FactorTradabilityIssue") -> tuple[int, str, str, str, str]:
    return (
        _SEVERITY_ORDER.get(issue.severity, 99),
        issue.symbol or "",
        issue.date or "",
        issue.issue_code,
        issue.issue_id,
    )


def _record_sort_key(
    record: "ExecutionTransitionAuditRecord",
) -> tuple[str, str, str, str]:
    return (
        record.execution_date,
        record.symbol,
        record.signal_date,
        record.record_id,
    )


def _as_bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return default
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        raise ValueError(f"Cannot coerce numeric value {value!r} to bool.")
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot coerce value {value!r} to bool.")


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    return parsed


def _matrix_values_equal(left: Sequence[Sequence[Any]], right: Sequence[Sequence[Any]]) -> bool:
    return json.dumps(_json_safe(left), sort_keys=True, allow_nan=False) == json.dumps(
        _json_safe(right),
        sort_keys=True,
        allow_nan=False,
    )


@dataclass
class AShareTradabilityConfig:
    schema_version: str = FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION
    config_id: str = ""
    market: str = "CN"
    min_listing_days: int = 60
    st_filter: bool = True
    suspension_filter: bool = True
    delisted_filter: bool = True
    limit_up_blocks_buy: bool = True
    limit_down_blocks_sell: bool = True
    require_valid_price: bool = True
    require_valid_volume: bool = True
    min_amount: float | None = None
    price_field: str = FIELD_VWAP
    volume_field: str = FIELD_VOLUME
    amount_field: str = FIELD_AMOUNT
    suspended_field: str = FIELD_SUSPENDED
    limit_up_field: str = FIELD_LIMIT_UP
    limit_down_field: str = FIELD_LIMIT_DOWN
    is_st_field: str = FIELD_IS_ST
    delisted_field: str = FIELD_DELISTED
    listing_days_field: str = FIELD_LISTING_DAYS
    listing_date_field: str = FIELD_LISTING_DATE
    valid_price_field: str = FIELD_VALID_PRICE
    valid_volume_field: str = FIELD_VALID_VOLUME
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.market = _non_empty_str(self.market, "market")
        self.min_listing_days = _non_negative_int(self.min_listing_days, "min_listing_days")
        self.st_filter = _require_bool(self.st_filter, "st_filter")
        self.suspension_filter = _require_bool(self.suspension_filter, "suspension_filter")
        self.delisted_filter = _require_bool(self.delisted_filter, "delisted_filter")
        self.limit_up_blocks_buy = _require_bool(
            self.limit_up_blocks_buy,
            "limit_up_blocks_buy",
        )
        self.limit_down_blocks_sell = _require_bool(
            self.limit_down_blocks_sell,
            "limit_down_blocks_sell",
        )
        self.require_valid_price = _require_bool(
            self.require_valid_price,
            "require_valid_price",
        )
        self.require_valid_volume = _require_bool(
            self.require_valid_volume,
            "require_valid_volume",
        )
        self.min_amount = _non_negative_float_or_none(self.min_amount, "min_amount")
        for field_name in (
            "price_field",
            "volume_field",
            "amount_field",
            "suspended_field",
            "limit_up_field",
            "limit_down_field",
            "is_st_field",
            "delisted_field",
            "listing_days_field",
            "listing_date_field",
            "valid_price_field",
            "valid_volume_field",
        ):
            setattr(self, field_name, _non_empty_str(getattr(self, field_name), field_name))
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_id": self.config_id,
            "market": self.market,
            "min_listing_days": self.min_listing_days,
            "st_filter": self.st_filter,
            "suspension_filter": self.suspension_filter,
            "delisted_filter": self.delisted_filter,
            "limit_up_blocks_buy": self.limit_up_blocks_buy,
            "limit_down_blocks_sell": self.limit_down_blocks_sell,
            "require_valid_price": self.require_valid_price,
            "require_valid_volume": self.require_valid_volume,
            "min_amount": self.min_amount,
            "price_field": self.price_field,
            "volume_field": self.volume_field,
            "amount_field": self.amount_field,
            "suspended_field": self.suspended_field,
            "limit_up_field": self.limit_up_field,
            "limit_down_field": self.limit_down_field,
            "is_st_field": self.is_st_field,
            "delisted_field": self.delisted_field,
            "listing_days_field": self.listing_days_field,
            "listing_date_field": self.listing_date_field,
            "valid_price_field": self.valid_price_field,
            "valid_volume_field": self.valid_volume_field,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AShareTradabilityConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "")),
            market=str(data.get("market", "CN")),
            min_listing_days=int(data.get("min_listing_days", 60)),
            st_filter=data.get("st_filter", True),
            suspension_filter=data.get("suspension_filter", True),
            delisted_filter=data.get("delisted_filter", True),
            limit_up_blocks_buy=data.get("limit_up_blocks_buy", True),
            limit_down_blocks_sell=data.get("limit_down_blocks_sell", True),
            require_valid_price=data.get("require_valid_price", True),
            require_valid_volume=data.get("require_valid_volume", True),
            min_amount=data.get("min_amount"),
            price_field=str(data.get("price_field", FIELD_VWAP)),
            volume_field=str(data.get("volume_field", FIELD_VOLUME)),
            amount_field=str(data.get("amount_field", FIELD_AMOUNT)),
            suspended_field=str(data.get("suspended_field", FIELD_SUSPENDED)),
            limit_up_field=str(data.get("limit_up_field", FIELD_LIMIT_UP)),
            limit_down_field=str(data.get("limit_down_field", FIELD_LIMIT_DOWN)),
            is_st_field=str(data.get("is_st_field", FIELD_IS_ST)),
            delisted_field=str(data.get("delisted_field", FIELD_DELISTED)),
            listing_days_field=str(data.get("listing_days_field", FIELD_LISTING_DAYS)),
            listing_date_field=str(data.get("listing_date_field", FIELD_LISTING_DATE)),
            valid_price_field=str(data.get("valid_price_field", FIELD_VALID_PRICE)),
            valid_volume_field=str(data.get("valid_volume_field", FIELD_VALID_VOLUME)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorTradabilityIssue:
    schema_version: str = FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION
    issue_id: str = ""
    symbol: str | None = None
    date: str | None = None
    issue_code: str = ""
    severity: str = TRADABILITY_ISSUE_WARNING
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)
        self.issue_id = _non_empty_str(self.issue_id, "issue_id")
        self.symbol = _optional_str(self.symbol)
        self.date = _optional_str(self.date)
        self.issue_code = _non_empty_str(self.issue_code, "issue_code")
        _validate_supported(self.issue_code, "issue_code", SUPPORTED_TRADABILITY_ISSUE_CODES)
        self.severity = _non_empty_str(self.severity, "severity")
        _validate_supported(
            self.severity,
            "severity",
            SUPPORTED_TRADABILITY_ISSUE_SEVERITIES,
        )
        self.message = _non_empty_str(self.message, "message")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issue_id": self.issue_id,
            "symbol": self.symbol,
            "date": self.date,
            "issue_code": self.issue_code,
            "severity": self.severity,
            "message": self.message,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorTradabilityIssue":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)),
            issue_id=str(data.get("issue_id", "")),
            symbol=data.get("symbol"),
            date=data.get("date"),
            issue_code=str(data.get("issue_code", "")),
            severity=str(data.get("severity", TRADABILITY_ISSUE_WARNING)),
            message=str(data.get("message", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class AShareTradabilityMask:
    schema_version: str = FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION
    mask_id: str = ""
    symbols: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    can_trade_mask: list[list[bool]] = field(default_factory=list)
    can_buy_mask: list[list[bool]] = field(default_factory=list)
    can_sell_mask: list[list[bool]] = field(default_factory=list)
    can_hold_mask: list[list[bool]] = field(default_factory=list)
    research_eligible_mask: list[list[bool]] = field(default_factory=list)
    issue_codes_by_cell: list[list[list[str]]] = field(default_factory=list)
    config: AShareTradabilityConfig = field(
        default_factory=lambda: AShareTradabilityConfig(config_id="default-tradability-config")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)
        self.mask_id = _non_empty_str(self.mask_id, "mask_id")
        self.symbols = [str(symbol).strip() for symbol in self.symbols if str(symbol).strip()]
        if not self.symbols:
            raise ValueError("symbols must be non-empty.")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must contain unique values.")
        self.dates = _coerce_dates(self.dates)
        if not isinstance(self.config, AShareTradabilityConfig):
            self.config = AShareTradabilityConfig.from_dict(self.config)
        for field_name in (
            "can_trade_mask",
            "can_buy_mask",
            "can_sell_mask",
            "can_hold_mask",
            "research_eligible_mask",
        ):
            setattr(
                self,
                field_name,
                _coerce_bool_mask(
                    getattr(self, field_name),
                    symbols=self.symbols,
                    dates=self.dates,
                    field_name=field_name,
                ),
            )
        self.issue_codes_by_cell = _coerce_issue_tensor(
            self.issue_codes_by_cell,
            symbols=self.symbols,
            dates=self.dates,
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mask_id": self.mask_id,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "can_trade_mask": _json_safe(self.can_trade_mask),
            "can_buy_mask": _json_safe(self.can_buy_mask),
            "can_sell_mask": _json_safe(self.can_sell_mask),
            "can_hold_mask": _json_safe(self.can_hold_mask),
            "research_eligible_mask": _json_safe(self.research_eligible_mask),
            "issue_codes_by_cell": _json_safe(self.issue_codes_by_cell),
            "config": self.config.to_dict(),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AShareTradabilityMask":
        data = dict(payload)
        config_payload = data.get("config", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)),
            mask_id=str(data.get("mask_id", "")),
            symbols=list(data.get("symbols", []) or []),
            dates=list(data.get("dates", []) or []),
            can_trade_mask=list(data.get("can_trade_mask", []) or []),
            can_buy_mask=list(data.get("can_buy_mask", []) or []),
            can_sell_mask=list(data.get("can_sell_mask", []) or []),
            can_hold_mask=list(data.get("can_hold_mask", []) or []),
            research_eligible_mask=list(data.get("research_eligible_mask", []) or []),
            issue_codes_by_cell=list(data.get("issue_codes_by_cell", []) or []),
            config=AShareTradabilityConfig.from_dict(config_payload)
            if isinstance(config_payload, Mapping)
            else config_payload,
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExecutionTransitionAuditRecord:
    schema_version: str = FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
    record_id: str = ""
    symbol: str = ""
    signal_date: str = ""
    execution_date: str = ""
    previous_weight: float = 0.0
    target_weight: float = 0.0
    trade_weight: float = 0.0
    trade_direction: str = TRADE_DIRECTION_HOLD
    can_buy: bool = True
    can_sell: bool = True
    can_trade: bool = True
    status: str = EXECUTION_AUDIT_STATUS_FEASIBLE
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
        )
        self.record_id = _non_empty_str(self.record_id, "record_id")
        self.symbol = _non_empty_str(self.symbol, "symbol")
        self.signal_date = _non_empty_str(self.signal_date, "signal_date")
        self.execution_date = _non_empty_str(self.execution_date, "execution_date")
        self.previous_weight = _finite_float(self.previous_weight, "previous_weight")
        self.target_weight = _finite_float(self.target_weight, "target_weight")
        self.trade_weight = _finite_float(self.trade_weight, "trade_weight")
        self.trade_direction = _non_empty_str(self.trade_direction, "trade_direction")
        _validate_supported(
            self.trade_direction,
            "trade_direction",
            SUPPORTED_TRADE_DIRECTIONS,
        )
        self.can_buy = _require_bool(self.can_buy, "can_buy")
        self.can_sell = _require_bool(self.can_sell, "can_sell")
        self.can_trade = _require_bool(self.can_trade, "can_trade")
        self.status = _non_empty_str(self.status, "status")
        _validate_supported(self.status, "status", SUPPORTED_EXECUTION_AUDIT_STATUSES)
        self.issue_codes = _sorted_issue_codes(self.issue_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "symbol": self.symbol,
            "signal_date": self.signal_date,
            "execution_date": self.execution_date,
            "previous_weight": self.previous_weight,
            "target_weight": self.target_weight,
            "trade_weight": self.trade_weight,
            "trade_direction": self.trade_direction,
            "can_buy": self.can_buy,
            "can_sell": self.can_sell,
            "can_trade": self.can_trade,
            "status": self.status,
            "issue_codes": list(self.issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionTransitionAuditRecord":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION,
                )
            ),
            record_id=str(data.get("record_id", "")),
            symbol=str(data.get("symbol", "")),
            signal_date=str(data.get("signal_date", "")),
            execution_date=str(data.get("execution_date", "")),
            previous_weight=float(data.get("previous_weight", 0.0)),
            target_weight=float(data.get("target_weight", 0.0)),
            trade_weight=float(data.get("trade_weight", 0.0)),
            trade_direction=str(data.get("trade_direction", TRADE_DIRECTION_HOLD)),
            can_buy=data.get("can_buy", True),
            can_sell=data.get("can_sell", True),
            can_trade=data.get("can_trade", True),
            status=str(data.get("status", EXECUTION_AUDIT_STATUS_FEASIBLE)),
            issue_codes=list(data.get("issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorExecutionFeasibilityReport:
    schema_version: str = FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    factor_matrix_id: str | None = None
    backtest_run_id: str | None = None
    weight_matrix_id: str | None = None
    mask_id: str | None = None
    total_transitions: int = 0
    feasible_transitions: int = 0
    blocked_transitions: int = 0
    partially_feasible_transitions: int = 0
    blocked_buy_count: int = 0
    blocked_sell_count: int = 0
    blocked_symbols: list[str] = field(default_factory=list)
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    transition_records: list[ExecutionTransitionAuditRecord] = field(default_factory=list)
    issues: list[FactorTradabilityIssue] = field(default_factory=list)
    verdict: str = TRADABILITY_AUDIT_PASS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
        )
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.factor_matrix_id = _optional_str(self.factor_matrix_id)
        self.backtest_run_id = _optional_str(self.backtest_run_id)
        self.weight_matrix_id = _optional_str(self.weight_matrix_id)
        self.mask_id = _optional_str(self.mask_id)
        self.transition_records = [
            record if isinstance(record, ExecutionTransitionAuditRecord)
            else ExecutionTransitionAuditRecord.from_dict(record)
            for record in self.transition_records
        ]
        self.transition_records = sorted(self.transition_records, key=_record_sort_key)
        self.issues = [
            issue if isinstance(issue, FactorTradabilityIssue)
            else FactorTradabilityIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = sorted(self.issues, key=_issue_sort_key)
        self.total_transitions = len(self.transition_records)
        self.feasible_transitions = sum(
            1 for record in self.transition_records
            if record.status == EXECUTION_AUDIT_STATUS_FEASIBLE
        )
        self.blocked_transitions = sum(
            1 for record in self.transition_records
            if record.status == EXECUTION_AUDIT_STATUS_BLOCKED
        )
        self.partially_feasible_transitions = sum(
            1 for record in self.transition_records
            if record.status == EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE
        )
        self.blocked_buy_count = sum(
            1 for record in self.transition_records
            if TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION in record.issue_codes
        )
        self.blocked_sell_count = sum(
            1 for record in self.transition_records
            if TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION in record.issue_codes
        )
        self.blocked_symbols = sorted(
            {
                record.symbol for record in self.transition_records
                if record.status != EXECUTION_AUDIT_STATUS_FEASIBLE
            }
        )
        self.issue_count = len(self.issues)
        self.blocker_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_BLOCKER
        )
        self.warning_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_WARNING
        )
        self.info_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_INFO
        )
        self.verdict = _non_empty_str(self.verdict, "verdict")
        _validate_supported(self.verdict, "verdict", SUPPORTED_TRADABILITY_AUDIT_VERDICTS)
        for field_name in (
            "total_transitions",
            "feasible_transitions",
            "blocked_transitions",
            "partially_feasible_transitions",
            "blocked_buy_count",
            "blocked_sell_count",
            "issue_count",
            "blocker_count",
            "warning_count",
            "info_count",
        ):
            _non_negative_int(getattr(self, field_name), field_name)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "factor_matrix_id": self.factor_matrix_id,
            "backtest_run_id": self.backtest_run_id,
            "weight_matrix_id": self.weight_matrix_id,
            "mask_id": self.mask_id,
            "total_transitions": self.total_transitions,
            "feasible_transitions": self.feasible_transitions,
            "blocked_transitions": self.blocked_transitions,
            "partially_feasible_transitions": self.partially_feasible_transitions,
            "blocked_buy_count": self.blocked_buy_count,
            "blocked_sell_count": self.blocked_sell_count,
            "blocked_symbols": list(self.blocked_symbols),
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "transition_records": [record.to_dict() for record in self.transition_records],
            "issues": [issue.to_dict() for issue in self.issues],
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorExecutionFeasibilityReport":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get(
                    "schema_version",
                    FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION,
                )
            ),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            factor_matrix_id=data.get("factor_matrix_id"),
            backtest_run_id=data.get("backtest_run_id"),
            weight_matrix_id=data.get("weight_matrix_id"),
            mask_id=data.get("mask_id"),
            total_transitions=int(data.get("total_transitions", 0)),
            feasible_transitions=int(data.get("feasible_transitions", 0)),
            blocked_transitions=int(data.get("blocked_transitions", 0)),
            partially_feasible_transitions=int(
                data.get("partially_feasible_transitions", 0)
            ),
            blocked_buy_count=int(data.get("blocked_buy_count", 0)),
            blocked_sell_count=int(data.get("blocked_sell_count", 0)),
            blocked_symbols=list(data.get("blocked_symbols", []) or []),
            issue_count=int(data.get("issue_count", 0)),
            blocker_count=int(data.get("blocker_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            info_count=int(data.get("info_count", 0)),
            transition_records=[
                ExecutionTransitionAuditRecord.from_dict(record)
                for record in list(data.get("transition_records", []) or [])
                if isinstance(record, Mapping)
            ],
            issues=[
                FactorTradabilityIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            verdict=str(data.get("verdict", TRADABILITY_AUDIT_PASS)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorTradabilityAuditReport:
    schema_version: str = FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    mask_id: str = ""
    symbols_count: int = 0
    dates_count: int = 0
    tradable_cell_count: int = 0
    blocked_cell_count: int = 0
    buy_blocked_cell_count: int = 0
    sell_blocked_cell_count: int = 0
    research_eligible_cell_count: int = 0
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issue_summary: dict[str, int] = field(default_factory=dict)
    issues: list[FactorTradabilityIssue] = field(default_factory=list)
    verdict: str = TRADABILITY_AUDIT_PASS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.mask_id = _non_empty_str(self.mask_id, "mask_id")
        self.issues = [
            issue if isinstance(issue, FactorTradabilityIssue)
            else FactorTradabilityIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = sorted(self.issues, key=_issue_sort_key)
        self.issue_count = len(self.issues)
        self.blocker_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_BLOCKER
        )
        self.warning_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_WARNING
        )
        self.info_count = sum(
            1 for issue in self.issues if issue.severity == TRADABILITY_ISSUE_INFO
        )
        self.issue_summary = {
            str(key): int(value)
            for key, value in sorted(self.issue_summary.items(), key=lambda item: str(item[0]))
        }
        _ensure_json_serializable(self.issue_summary, "issue_summary")
        self.verdict = _non_empty_str(self.verdict, "verdict")
        _validate_supported(self.verdict, "verdict", SUPPORTED_TRADABILITY_AUDIT_VERDICTS)
        for field_name in (
            "symbols_count",
            "dates_count",
            "tradable_cell_count",
            "blocked_cell_count",
            "buy_blocked_cell_count",
            "sell_blocked_cell_count",
            "research_eligible_cell_count",
            "issue_count",
            "blocker_count",
            "warning_count",
            "info_count",
        ):
            _non_negative_int(getattr(self, field_name), field_name)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "mask_id": self.mask_id,
            "symbols_count": self.symbols_count,
            "dates_count": self.dates_count,
            "tradable_cell_count": self.tradable_cell_count,
            "blocked_cell_count": self.blocked_cell_count,
            "buy_blocked_cell_count": self.buy_blocked_cell_count,
            "sell_blocked_cell_count": self.sell_blocked_cell_count,
            "research_eligible_cell_count": self.research_eligible_cell_count,
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issue_summary": dict(_json_safe(self.issue_summary)),
            "issues": [issue.to_dict() for issue in self.issues],
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorTradabilityAuditReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            mask_id=str(data.get("mask_id", "")),
            symbols_count=int(data.get("symbols_count", 0)),
            dates_count=int(data.get("dates_count", 0)),
            tradable_cell_count=int(data.get("tradable_cell_count", 0)),
            blocked_cell_count=int(data.get("blocked_cell_count", 0)),
            buy_blocked_cell_count=int(data.get("buy_blocked_cell_count", 0)),
            sell_blocked_cell_count=int(data.get("sell_blocked_cell_count", 0)),
            research_eligible_cell_count=int(data.get("research_eligible_cell_count", 0)),
            issue_count=int(data.get("issue_count", 0)),
            blocker_count=int(data.get("blocker_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            info_count=int(data.get("info_count", 0)),
            issue_summary=dict(data.get("issue_summary", {}) or {}),
            issues=[
                FactorTradabilityIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            verdict=str(data.get("verdict", TRADABILITY_AUDIT_PASS)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_tradability_config_id(config: AShareTradabilityConfig) -> str:
    payload = config.to_dict()
    payload.pop("config_id", None)
    return (
        f"tradability-config-{_slug(config.market)}-"
        f"ld{config.min_listing_days}-{_short_hash([payload])}"
    )


def make_tradability_issue_id(
    *,
    symbol: str | None,
    date: str | None,
    issue_code: str,
    message: str,
) -> str:
    parts = [symbol, date, str(issue_code), str(message)]
    return (
        f"tradability-issue-{_slug(issue_code)}-"
        f"{_slug(symbol)}-{_slug(date)}-{_short_hash(parts)}"
    )


def make_tradability_mask_id(
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
    config_id: str,
) -> str:
    parts = [list(symbols), list(dates), str(config_id)]
    return f"tradability-mask-{_slug(config_id)}-{_short_hash(parts)}"


def make_execution_transition_record_id(
    *,
    symbol: str,
    signal_date: str,
    execution_date: str,
    target_weight: float,
) -> str:
    parts = [str(symbol), str(signal_date), str(execution_date), float(target_weight)]
    return (
        f"execution-transition-{_slug(symbol)}-{_slug(signal_date)}-"
        f"{_slug(execution_date)}-{_short_hash(parts)}"
    )


def make_execution_feasibility_report_id(
    *,
    backtest_run_id: str | None,
    weight_matrix_id: str | None,
    generated_at: str,
) -> str:
    parts = [backtest_run_id, weight_matrix_id, str(generated_at)]
    return (
        f"execution-feasibility-report-{_slug(backtest_run_id)}-"
        f"{_slug(weight_matrix_id)}-{_slug(generated_at)}-{_short_hash(parts)}"
    )


def make_tradability_audit_report_id(
    *,
    mask_id: str,
    generated_at: str,
) -> str:
    parts = [str(mask_id), str(generated_at)]
    return (
        f"tradability-audit-report-{_slug(mask_id)}-"
        f"{_slug(generated_at)}-{_short_hash(parts)}"
    )


def get_matrix_field_optional(
    bundle: MatrixDataBundle,
    field_name: str,
) -> list[list[Any]] | None:
    if not bundle.has_field(field_name):
        return None
    return bundle.get_field(field_name)


def normalize_bool_matrix(
    values: Sequence[Sequence[Any]] | None,
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
    default: bool,
) -> list[list[bool]]:
    if values is None:
        return _empty_bool_matrix(symbols, dates, default)
    _validate_matrix_shape(values, symbols=symbols, dates=dates, field_name="bool_matrix")
    return [
        [_as_bool_value(value, default=default) for value in row]
        for row in values
    ]


def normalize_float_matrix(
    values: Sequence[Sequence[Any]] | None,
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
) -> list[list[float | None]]:
    if values is None:
        return [[None for _ in dates] for _ in symbols]
    _validate_matrix_shape(values, symbols=symbols, dates=dates, field_name="float_matrix")
    return [[_to_finite_float(value) for value in row] for row in values]


def build_valid_price_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[bool]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    if not config.require_valid_price:
        return _empty_bool_matrix(symbols, dates, True)
    explicit = get_matrix_field_optional(bundle, config.valid_price_field)
    if explicit is not None:
        return normalize_bool_matrix(explicit, symbols=symbols, dates=dates, default=False)
    price_matrix = normalize_float_matrix(
        get_matrix_field_optional(bundle, config.price_field),
        symbols=symbols,
        dates=dates,
    )
    if config.price_field == FIELD_VWAP and all(
        value is None for row in price_matrix for value in row
    ):
        amount_matrix = normalize_float_matrix(
            get_matrix_field_optional(bundle, config.amount_field),
            symbols=symbols,
            dates=dates,
        )
        volume_matrix = normalize_float_matrix(
            get_matrix_field_optional(bundle, config.volume_field),
            symbols=symbols,
            dates=dates,
        )
        return [
            [
                amount is not None
                and volume is not None
                and amount > 0.0
                and volume > 0.0
                and amount / volume > 0.0
                for amount, volume in zip(amount_row, volume_row)
            ]
            for amount_row, volume_row in zip(amount_matrix, volume_matrix)
        ]
    return [[value is not None and value > 0.0 for value in row] for row in price_matrix]


def build_valid_volume_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[bool]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    if not config.require_valid_volume:
        return _empty_bool_matrix(symbols, dates, True)
    explicit = get_matrix_field_optional(bundle, config.valid_volume_field)
    if explicit is not None:
        return normalize_bool_matrix(explicit, symbols=symbols, dates=dates, default=False)
    volume_matrix = normalize_float_matrix(
        get_matrix_field_optional(bundle, config.volume_field),
        symbols=symbols,
        dates=dates,
    )
    return [[value is not None and value > 0.0 for value in row] for row in volume_matrix]


def build_listing_days_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[int | None]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    listing_days_values = get_matrix_field_optional(bundle, config.listing_days_field)
    if listing_days_values is not None:
        numeric = normalize_float_matrix(listing_days_values, symbols=symbols, dates=dates)
        return [[int(value) if value is not None else None for value in row] for row in numeric]

    listing_date_values = get_matrix_field_optional(bundle, config.listing_date_field)
    if listing_date_values is not None:
        _validate_matrix_shape(
            listing_date_values,
            symbols=symbols,
            dates=dates,
            field_name=config.listing_date_field,
        )
        output: list[list[int | None]] = []
        for row in listing_date_values:
            output_row: list[int | None] = []
            for current_date_text, listing_date_value in zip(dates, row):
                current_date = _parse_iso_date(current_date_text)
                listed_date = _parse_iso_date(listing_date_value)
                if current_date is None or listed_date is None:
                    output_row.append(None)
                else:
                    output_row.append((current_date - listed_date).days)
            output.append(output_row)
        return output

    metadata_candidates = [
        bundle.metadata.get(config.listing_date_field),
        bundle.metadata.get("listing_dates"),
        bundle.metadata.get("listing_date_by_symbol"),
    ]
    for candidate in metadata_candidates:
        if isinstance(candidate, Mapping):
            output = []
            for symbol in symbols:
                listed_date = _parse_iso_date(candidate.get(symbol))
                symbol_listing_days: list[int | None] = []
                for current_date_text in dates:
                    current_date = _parse_iso_date(current_date_text)
                    if current_date is None or listed_date is None:
                        symbol_listing_days.append(None)
                    else:
                        symbol_listing_days.append((current_date - listed_date).days)
                output.append(symbol_listing_days)
            return output
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            if len(candidate) == len(symbols):
                output = []
                for listed_date_value in candidate:
                    listed_date = _parse_iso_date(listed_date_value)
                    row = []
                    for current_date_text in dates:
                        current_date = _parse_iso_date(current_date_text)
                        if current_date is None or listed_date is None:
                            row.append(None)
                        else:
                            row.append((current_date - listed_date).days)
                    output.append(row)
                return output
    return [[None for _ in dates] for _ in symbols]


def _default_tradability_config() -> AShareTradabilityConfig:
    config = AShareTradabilityConfig(config_id="placeholder")
    config.config_id = make_tradability_config_id(config)
    return config


def _missing_tradability_fields(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[str]:
    field_names = {
        config.suspended_field,
        config.limit_up_field,
        config.limit_down_field,
        config.is_st_field,
        config.delisted_field,
    }
    missing = {field for field in field_names if not bundle.has_field(field)}
    has_listing_metadata = any(
        key in bundle.metadata
        for key in (
            config.listing_date_field,
            "listing_dates",
            "listing_date_by_symbol",
        )
    )
    if (
        not bundle.has_field(config.listing_days_field)
        and not bundle.has_field(config.listing_date_field)
        and not has_listing_metadata
    ):
        missing.add(config.listing_days_field)
        missing.add(config.listing_date_field)
    if (
        config.min_amount is not None
        and not bundle.has_field(config.amount_field)
        and not bundle.has_field(FIELD_LOW_LIQUIDITY)
    ):
        missing.add(config.amount_field)
        missing.add(FIELD_LOW_LIQUIDITY)
    if config.require_valid_price:
        has_explicit_price_flag = bundle.has_field(config.valid_price_field)
        has_price = bundle.has_field(config.price_field)
        has_derivable_vwap = (
            config.price_field == FIELD_VWAP
            and bundle.has_field(config.amount_field)
            and bundle.has_field(config.volume_field)
        )
        if not (has_explicit_price_flag or has_price or has_derivable_vwap):
            missing.add(config.valid_price_field)
            missing.add(config.price_field)
    if (
        config.require_valid_volume
        and not bundle.has_field(config.valid_volume_field)
        and not bundle.has_field(config.volume_field)
    ):
        missing.add(config.valid_volume_field)
        missing.add(config.volume_field)
    return sorted(missing)


def _append_issue(codes: list[str], issue_code: str) -> None:
    if issue_code not in codes:
        codes.append(issue_code)


def build_ashare_tradability_mask(
    bundle: MatrixDataBundle,
    *,
    config: AShareTradabilityConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AShareTradabilityMask:
    resolved_config = config or _default_tradability_config()
    symbols = list(bundle.contract.symbols)
    dates = list(bundle.contract.dates)
    can_trade = _empty_bool_matrix(symbols, dates, True)
    can_buy = _empty_bool_matrix(symbols, dates, True)
    can_sell = _empty_bool_matrix(symbols, dates, True)
    can_hold = _empty_bool_matrix(symbols, dates, True)
    research_eligible = _empty_bool_matrix(symbols, dates, True)
    issue_codes = _empty_issue_tensor(symbols, dates)

    suspended = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.suspended_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    limit_up = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.limit_up_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    limit_down = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.limit_down_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    is_st = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.is_st_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    delisted = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.delisted_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    valid_price = build_valid_price_matrix(bundle, resolved_config)
    valid_volume = build_valid_volume_matrix(bundle, resolved_config)
    listing_days = build_listing_days_matrix(bundle, resolved_config)
    amount = normalize_float_matrix(
        get_matrix_field_optional(bundle, resolved_config.amount_field),
        symbols=symbols,
        dates=dates,
    )
    low_liquidity = normalize_bool_matrix(
        get_matrix_field_optional(bundle, FIELD_LOW_LIQUIDITY),
        symbols=symbols,
        dates=dates,
        default=False,
    )

    for row_index, _symbol in enumerate(symbols):
        for column_index, _date in enumerate(dates):
            cell_issues = issue_codes[row_index][column_index]
            if suspended[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                if resolved_config.suspension_filter:
                    research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_SUSPENDED)
            if delisted[row_index][column_index] and resolved_config.delisted_filter:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                can_hold[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_DELISTED)
            if is_st[row_index][column_index] and resolved_config.st_filter:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_ST_FILTERED)
            if limit_up[row_index][column_index] and resolved_config.limit_up_blocks_buy:
                can_buy[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED)
            if limit_down[row_index][column_index] and resolved_config.limit_down_blocks_sell:
                can_sell[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED)
            listed_days = listing_days[row_index][column_index]
            if listed_days is not None and listed_days < resolved_config.min_listing_days:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NEW_LISTING)
            if not valid_price[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NO_VALID_PRICE)
            if not valid_volume[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NO_VALID_VOLUME)
            amount_value = amount[row_index][column_index]
            low_amount = (
                low_liquidity[row_index][column_index]
                or (
                    resolved_config.min_amount is not None
                    and amount_value is not None
                    and amount_value < resolved_config.min_amount
                )
            )
            if low_amount:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LOW_AMOUNT)
            issue_codes[row_index][column_index] = sorted(cell_issues)

    resolved_metadata = _coerce_metadata(metadata)
    missing_fields = _missing_tradability_fields(bundle, resolved_config)
    resolved_metadata.update(
        {
            "factor_tradability_audit_schema_version": FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "missing_tradability_fields": missing_fields,
        }
    )
    return AShareTradabilityMask(
        mask_id=make_tradability_mask_id(
            symbols=symbols,
            dates=dates,
            config_id=resolved_config.config_id,
        ),
        symbols=symbols,
        dates=dates,
        can_trade_mask=can_trade,
        can_buy_mask=can_buy,
        can_sell_mask=can_sell,
        can_hold_mask=can_hold,
        research_eligible_mask=research_eligible,
        issue_codes_by_cell=issue_codes,
        config=resolved_config,
        metadata=resolved_metadata,
    )


def _make_issue(
    *,
    symbol: str | None,
    date_value: str | None,
    issue_code: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorTradabilityIssue:
    return FactorTradabilityIssue(
        issue_id=make_tradability_issue_id(
            symbol=symbol,
            date=date_value,
            issue_code=issue_code,
            message=message,
        ),
        symbol=symbol,
        date=date_value,
        issue_code=issue_code,
        severity=_issue_severity(issue_code),
        message=message,
        metadata=dict(metadata or {}),
    )


def _issues_from_mask(mask: AShareTradabilityMask) -> list[FactorTradabilityIssue]:
    issues: list[FactorTradabilityIssue] = []
    for row_index, symbol in enumerate(mask.symbols):
        for column_index, date_value in enumerate(mask.dates):
            for issue_code in mask.issue_codes_by_cell[row_index][column_index]:
                issues.append(
                    _make_issue(
                        symbol=symbol,
                        date_value=date_value,
                        issue_code=issue_code,
                        message=f"{issue_code} observed for {symbol} on {date_value}.",
                        metadata={
                            "mask_id": mask.mask_id,
                            "row_index": row_index,
                            "column_index": column_index,
                        },
                    )
                )
    for field_name in mask.metadata.get("missing_tradability_fields", []) or []:
        issues.append(
            _make_issue(
                symbol=None,
                date_value=None,
                issue_code=TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD,
                message=f"tradability field missing: {field_name}.",
                metadata={"field_name": field_name, "mask_id": mask.mask_id},
            )
        )
    by_id: dict[str, FactorTradabilityIssue] = {}
    for issue in issues:
        by_id.setdefault(issue.issue_id, issue)
    return sorted(by_id.values(), key=_issue_sort_key)


def _issue_summary(issues: Sequence[FactorTradabilityIssue]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for issue in issues:
        summary[issue.issue_code] = summary.get(issue.issue_code, 0) + 1
    return dict(sorted(summary.items()))


def _verdict_from_issues(issues: Sequence[FactorTradabilityIssue]) -> str:
    if any(issue.severity == TRADABILITY_ISSUE_BLOCKER for issue in issues):
        return TRADABILITY_AUDIT_FAIL
    if any(issue.severity == TRADABILITY_ISSUE_WARNING for issue in issues):
        return TRADABILITY_AUDIT_WARN
    return TRADABILITY_AUDIT_PASS


def build_tradability_audit_report(
    mask: AShareTradabilityMask,
    *,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorTradabilityAuditReport:
    issues = _issues_from_mask(mask)
    cell_count = len(mask.symbols) * len(mask.dates)
    tradable_count = sum(1 for row in mask.can_trade_mask for value in row if value)
    buy_blocked_count = sum(1 for row in mask.can_buy_mask for value in row if not value)
    sell_blocked_count = sum(1 for row in mask.can_sell_mask for value in row if not value)
    research_eligible_count = sum(
        1 for row in mask.research_eligible_mask for value in row if value
    )
    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "factor_tradability_audit_schema_version": FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "config": mask.config.to_dict(),
        }
    )
    return FactorTradabilityAuditReport(
        report_id=make_tradability_audit_report_id(
            mask_id=mask.mask_id,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        mask_id=mask.mask_id,
        symbols_count=len(mask.symbols),
        dates_count=len(mask.dates),
        tradable_cell_count=tradable_count,
        blocked_cell_count=cell_count - tradable_count,
        buy_blocked_cell_count=buy_blocked_count,
        sell_blocked_cell_count=sell_blocked_count,
        research_eligible_cell_count=research_eligible_count,
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_INFO),
        issue_summary=_issue_summary(issues),
        issues=issues,
        verdict=_verdict_from_issues(issues),
        metadata=resolved_metadata,
    )


def _alignment_value(alignment: Any, field_name: str) -> Any:
    if isinstance(alignment, Mapping):
        return alignment.get(field_name)
    return getattr(alignment, field_name, None)


def _execution_alignments(
    *,
    weight_matrix: FactorWeightMatrix,
    alignments: Sequence[Any] | None,
    run: SingleFactorBacktestRun | None,
) -> list[dict[str, str]]:
    if alignments is not None:
        output: list[dict[str, str]] = []
        for alignment in alignments:
            signal_date = _alignment_value(alignment, "signal_date")
            execution_date = (
                _alignment_value(alignment, "execution_start_date")
                or _alignment_value(alignment, "execution_date")
            )
            if signal_date is None or execution_date is None:
                continue
            output.append(
                {
                    "signal_date": str(signal_date),
                    "execution_date": str(execution_date),
                }
            )
        return sorted(output, key=lambda item: (item["execution_date"], item["signal_date"]))
    if run is not None:
        return [
            {
                "signal_date": record.signal_date,
                "execution_date": record.execution_start_date,
            }
            for record in sorted(run.daily_records, key=lambda record: record.execution_start_date)
        ]
    output = []
    for index, signal_date in enumerate(weight_matrix.dates[:-1]):
        output.append({"signal_date": signal_date, "execution_date": weight_matrix.dates[index + 1]})
    return output


def _weight_at(
    weight_matrix: FactorWeightMatrix,
    row_index: int,
    column_index: int,
) -> float:
    value = weight_matrix.net_weights[row_index][column_index]
    number = _to_finite_float(value)
    return number if number is not None else 0.0


def _trade_direction(trade_weight: float) -> str:
    if trade_weight > _FLOAT_TOLERANCE:
        return TRADE_DIRECTION_BUY
    if trade_weight < -_FLOAT_TOLERANCE:
        return TRADE_DIRECTION_SELL
    return TRADE_DIRECTION_HOLD


def _mask_cell(
    tradability_mask: AShareTradabilityMask,
    *,
    row_index: int,
    execution_date: str,
) -> tuple[bool, bool, bool, bool, list[str], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    try:
        column_index = tradability_mask.dates.index(execution_date)
    except ValueError:
        return (
            False,
            False,
            False,
            False,
            [TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH],
            {"execution_date_in_mask": False},
        )
    return (
        tradability_mask.can_buy_mask[row_index][column_index],
        tradability_mask.can_sell_mask[row_index][column_index],
        tradability_mask.can_trade_mask[row_index][column_index],
        tradability_mask.can_hold_mask[row_index][column_index],
        list(tradability_mask.issue_codes_by_cell[row_index][column_index]),
        {"execution_date_in_mask": True, "mask_column_index": column_index},
    )


def _record_issues(record: ExecutionTransitionAuditRecord) -> list[FactorTradabilityIssue]:
    issues: list[FactorTradabilityIssue] = []
    for issue_code in record.issue_codes:
        message = f"{issue_code} affects {record.symbol} on {record.execution_date}."
        if issue_code == TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION:
            message = "buy transition blocked on execution date."
        elif issue_code == TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION:
            message = "sell transition blocked on execution date."
        issues.append(
            _make_issue(
                symbol=record.symbol,
                date_value=record.execution_date,
                issue_code=issue_code,
                message=message,
                metadata={
                    "record_id": record.record_id,
                    "signal_date": record.signal_date,
                    "trade_direction": record.trade_direction,
                    "trade_weight": record.trade_weight,
                },
            )
        )
    return issues


def _mask_has_warning_issue(mask: AShareTradabilityMask) -> bool:
    for row in mask.issue_codes_by_cell:
        for codes in row:
            if any(_issue_severity(code) == TRADABILITY_ISSUE_WARNING for code in codes):
                return True
    return False


def _has_short_leg(weight_matrix: FactorWeightMatrix) -> bool:
    if str(weight_matrix.metadata.get("mode", "")).lower() == "long_short":
        return True
    for row in weight_matrix.short_weights:
        for value in row:
            number = _to_finite_float(value)
            if number is not None and abs(number) > _FLOAT_TOLERANCE:
                return True
    return False


def audit_factor_weight_execution_feasibility(
    *,
    weight_matrix: FactorWeightMatrix,
    tradability_mask: AShareTradabilityMask,
    alignments: Sequence[Any] | None = None,
    run: SingleFactorBacktestRun | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorExecutionFeasibilityReport:
    if list(weight_matrix.symbols) != list(tradability_mask.symbols):
        raise ValueError("weight_matrix symbols must match tradability_mask symbols.")
    date_to_index = {date_value: index for index, date_value in enumerate(weight_matrix.dates)}
    resolved_alignments = _execution_alignments(
        weight_matrix=weight_matrix,
        alignments=alignments,
        run=run,
    )
    previous_weights = {symbol: 0.0 for symbol in weight_matrix.symbols}
    records: list[ExecutionTransitionAuditRecord] = []

    for alignment in resolved_alignments:
        signal_date = alignment["signal_date"]
        execution_date = alignment["execution_date"]
        signal_index = date_to_index.get(signal_date)
        if signal_index is None:
            continue
        for row_index, symbol in enumerate(weight_matrix.symbols):
            previous_weight = previous_weights[symbol]
            target_weight = _weight_at(weight_matrix, row_index, signal_index)
            trade_weight = target_weight - previous_weight
            direction = _trade_direction(trade_weight)
            can_buy, can_sell, can_trade, can_hold, issue_codes, mask_metadata = _mask_cell(
                tradability_mask,
                row_index=row_index,
                execution_date=execution_date,
            )
            status = EXECUTION_AUDIT_STATUS_FEASIBLE
            if direction == TRADE_DIRECTION_BUY:
                if not can_buy:
                    _append_issue(issue_codes, TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION)
                if not can_buy or not can_trade:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            elif direction == TRADE_DIRECTION_SELL:
                if not can_sell:
                    _append_issue(issue_codes, TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION)
                if not can_sell or not can_trade:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            else:
                if not can_hold:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            issue_codes = sorted(set(issue_codes))
            records.append(
                ExecutionTransitionAuditRecord(
                    record_id=make_execution_transition_record_id(
                        symbol=symbol,
                        signal_date=signal_date,
                        execution_date=execution_date,
                        target_weight=target_weight,
                    ),
                    symbol=symbol,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    previous_weight=previous_weight,
                    target_weight=target_weight,
                    trade_weight=trade_weight,
                    trade_direction=direction,
                    can_buy=can_buy,
                    can_sell=can_sell,
                    can_trade=can_trade,
                    status=status,
                    issue_codes=issue_codes,
                    metadata={
                        "mask_id": tradability_mask.mask_id,
                        "signal_index": signal_index,
                        "row_index": row_index,
                        **mask_metadata,
                    },
                )
            )
            previous_weights[symbol] = target_weight

    records = sorted(records, key=_record_sort_key)
    issue_by_id: dict[str, FactorTradabilityIssue] = {}
    for record in records:
        for issue in _record_issues(record):
            issue_by_id.setdefault(issue.issue_id, issue)
    issues = sorted(issue_by_id.values(), key=_issue_sort_key)
    blocked_count = sum(
        1 for record in records if record.status == EXECUTION_AUDIT_STATUS_BLOCKED
    )
    verdict = TRADABILITY_AUDIT_PASS
    if blocked_count:
        verdict = TRADABILITY_AUDIT_FAIL
    elif _mask_has_warning_issue(tradability_mask):
        verdict = TRADABILITY_AUDIT_WARN

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "factor_execution_feasibility_audit_schema_version": (
                FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
            ),
            "non_runtime_impact": True,
            "no_pnl_adjustment": True,
        }
    )
    if _has_short_leg(weight_matrix):
        resolved_metadata["short_leg_is_research_analytic_not_cash_equity_short"] = True
    return FactorExecutionFeasibilityReport(
        report_id=make_execution_feasibility_report_id(
            backtest_run_id=run.run_id if run is not None else weight_matrix.metadata.get("run_id"),
            weight_matrix_id=weight_matrix.weights_id,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        factor_matrix_id=weight_matrix.factor_matrix_id,
        backtest_run_id=run.run_id if run is not None else weight_matrix.metadata.get("run_id"),
        weight_matrix_id=weight_matrix.weights_id,
        mask_id=tradability_mask.mask_id,
        total_transitions=len(records),
        feasible_transitions=sum(
            1 for record in records if record.status == EXECUTION_AUDIT_STATUS_FEASIBLE
        ),
        blocked_transitions=blocked_count,
        partially_feasible_transitions=sum(
            1 for record in records
            if record.status == EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE
        ),
        blocked_buy_count=sum(
            1 for record in records
            if TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION in record.issue_codes
        ),
        blocked_sell_count=sum(
            1 for record in records
            if TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION in record.issue_codes
        ),
        blocked_symbols=sorted(
            {
                record.symbol for record in records
                if record.status == EXECUTION_AUDIT_STATUS_BLOCKED
            }
        ),
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_INFO),
        transition_records=records,
        issues=issues,
        verdict=verdict,
        metadata=resolved_metadata,
    )


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_tradability_audit_markdown(report: FactorTradabilityAuditReport) -> str:
    lines = [
        "# A-share Tradability Audit",
        "",
        f"Generated at: {report.generated_at}",
        f"Verdict: {report.verdict}",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Symbols | {report.symbols_count} |",
        f"| Dates | {report.dates_count} |",
        f"| Tradable cells | {report.tradable_cell_count} |",
        f"| Blocked cells | {report.blocked_cell_count} |",
        f"| Buy-blocked cells | {report.buy_blocked_cell_count} |",
        f"| Sell-blocked cells | {report.sell_blocked_cell_count} |",
        f"| Research-eligible cells | {report.research_eligible_cell_count} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Issue Summary",
        "",
        "| Issue code | Count |",
        "| --- | ---: |",
    ]
    if report.issue_summary:
        for issue_code, count in report.issue_summary.items():
            lines.append(f"| {_escape_markdown_cell(issue_code)} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend([
        "",
        "## Issue Table",
        "",
        "| Severity | Symbol | Date | Issue code | Message |",
        "| --- | --- | --- | --- | --- |",
    ])
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"{_escape_markdown_cell(issue.severity)} | "
                f"{_escape_markdown_cell(issue.symbol or '')} | "
                f"{_escape_markdown_cell(issue.date or '')} | "
                f"{_escape_markdown_cell(issue.issue_code)} | "
                f"{_escape_markdown_cell(issue.message)} |"
            )
    else:
        lines.append("| none |  |  |  | No tradability issues. |")
    lines.extend([
        "",
        "## Non-runtime-impact Note",
        "",
        TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE,
        "",
    ])
    return "\n".join(lines)


def render_execution_feasibility_markdown(report: FactorExecutionFeasibilityReport) -> str:
    blocked_symbols = ", ".join(report.blocked_symbols) if report.blocked_symbols else "none"
    lines = [
        "# Factor Execution Feasibility Audit",
        "",
        f"Generated at: {report.generated_at}",
        f"Verdict: {report.verdict}",
        "",
        "## Transition Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total transitions | {report.total_transitions} |",
        f"| Feasible transitions | {report.feasible_transitions} |",
        f"| Blocked transitions | {report.blocked_transitions} |",
        f"| Partially feasible transitions | {report.partially_feasible_transitions} |",
        f"| Blocked buy transitions | {report.blocked_buy_count} |",
        f"| Blocked sell transitions | {report.blocked_sell_count} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Blocked Symbols",
        "",
        blocked_symbols,
        "",
        "## Issue Table",
        "",
        "| Severity | Symbol | Date | Issue code | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"{_escape_markdown_cell(issue.severity)} | "
                f"{_escape_markdown_cell(issue.symbol or '')} | "
                f"{_escape_markdown_cell(issue.date or '')} | "
                f"{_escape_markdown_cell(issue.issue_code)} | "
                f"{_escape_markdown_cell(issue.message)} |"
            )
    else:
        lines.append("| none |  |  |  | No execution feasibility issues. |")
    lines.extend([
        "",
        "## Transition Sample",
        "",
        "| Execution date | Symbol | Direction | Previous | Target | Trade | Status | Issues |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for record in report.transition_records[:20]:
        lines.append(
            "| "
            f"{_escape_markdown_cell(record.execution_date)} | "
            f"{_escape_markdown_cell(record.symbol)} | "
            f"{_escape_markdown_cell(record.trade_direction)} | "
            f"{record.previous_weight:.6g} | "
            f"{record.target_weight:.6g} | "
            f"{record.trade_weight:.6g} | "
            f"{_escape_markdown_cell(record.status)} | "
            f"{_escape_markdown_cell(','.join(record.issue_codes))} |"
        )
    if not report.transition_records:
        lines.append("|  |  |  | 0 | 0 | 0 | none |  |")
    lines.extend([
        "",
        "## Non-runtime-impact Note",
        "",
        EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE,
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "TRADABILITY_AUDIT_PASS",
    "TRADABILITY_AUDIT_WARN",
    "TRADABILITY_AUDIT_FAIL",
    "TRADABILITY_ISSUE_INFO",
    "TRADABILITY_ISSUE_WARNING",
    "TRADABILITY_ISSUE_BLOCKER",
    "TRADABILITY_ISSUE_SUSPENDED",
    "TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED",
    "TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED",
    "TRADABILITY_ISSUE_ST_FILTERED",
    "TRADABILITY_ISSUE_DELISTED",
    "TRADABILITY_ISSUE_NEW_LISTING",
    "TRADABILITY_ISSUE_NO_VALID_PRICE",
    "TRADABILITY_ISSUE_NO_VALID_VOLUME",
    "TRADABILITY_ISSUE_LOW_AMOUNT",
    "TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION",
    "TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION",
    "TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD",
    "TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH",
    "EXECUTION_AUDIT_STATUS_FEASIBLE",
    "EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE",
    "EXECUTION_AUDIT_STATUS_BLOCKED",
    "TRADE_DIRECTION_BUY",
    "TRADE_DIRECTION_SELL",
    "TRADE_DIRECTION_HOLD",
    "FIELD_SUSPENDED",
    "FIELD_LIMIT_UP",
    "FIELD_LIMIT_DOWN",
    "FIELD_IS_ST",
    "FIELD_DELISTED",
    "FIELD_LISTING_DAYS",
    "FIELD_LISTING_DATE",
    "FIELD_VALID_PRICE",
    "FIELD_VALID_VOLUME",
    "FIELD_LOW_LIQUIDITY",
    "FIELD_AMOUNT",
    "FIELD_VOLUME",
    "FIELD_OPEN",
    "FIELD_CLOSE",
    "FIELD_VWAP",
    "DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR",
    "DEFAULT_TRADABILITY_MASKS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME",
    "TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE",
    "EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE",
    "AShareTradabilityConfig",
    "FactorTradabilityIssue",
    "AShareTradabilityMask",
    "ExecutionTransitionAuditRecord",
    "FactorExecutionFeasibilityReport",
    "FactorTradabilityAuditReport",
    "make_tradability_config_id",
    "make_tradability_issue_id",
    "make_tradability_mask_id",
    "make_execution_transition_record_id",
    "make_execution_feasibility_report_id",
    "make_tradability_audit_report_id",
    "get_matrix_field_optional",
    "normalize_bool_matrix",
    "normalize_float_matrix",
    "build_valid_price_matrix",
    "build_valid_volume_matrix",
    "build_listing_days_matrix",
    "build_ashare_tradability_mask",
    "build_tradability_audit_report",
    "audit_factor_weight_execution_feasibility",
    "render_tradability_audit_markdown",
    "render_execution_feasibility_markdown",
]
