"""A-share tradability primitive constants, helpers, and masks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION


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
    record: Any,
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
    "make_tradability_config_id",
    "make_tradability_issue_id",
    "make_tradability_mask_id",
]
