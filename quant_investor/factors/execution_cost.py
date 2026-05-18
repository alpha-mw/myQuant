"""Offline execution-cost and execution-penalty simulation for factor backtests.

The helpers in this module consume already-built factor backtest artifacts,
weight matrices, local matrix bundles, and optional tradability masks. They
produce separate simulated return artifacts only. They do not alter stock
selection, factor admission, posterior scoring, RiskGuard, PortfolioConstructor,
target weights, orders, providers, LLMs, broker APIs, or live execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_SHORT,
    SingleFactorBacktestRun,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_VOLUME,
    FIELD_VWAP,
    MatrixDataBundle,
)
from quant_investor.factors.metrics import (
    annualized_return_from_daily,
    cumulative_return,
    max_drawdown_from_returns,
    sharpe_from_daily,
)
from quant_investor.factors.tradability import AShareTradabilityMask
from quant_investor.versioning import (
    FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION,
    FACTOR_EXECUTION_PENALTY_SCHEMA_VERSION,
)


EXECUTION_COST_SIMULATION_PASS = "pass"
EXECUTION_COST_SIMULATION_WARN = "warn"
EXECUTION_COST_SIMULATION_FAIL = "fail"

EXECUTION_SIMULATION_STATUS_OK = "ok"
EXECUTION_SIMULATION_STATUS_PARTIAL = "partial"
EXECUTION_SIMULATION_STATUS_BLOCKED = "blocked"
EXECUTION_SIMULATION_STATUS_MISSING_DATA = "missing_data"

EXECUTION_COST_ISSUE_INFO = "info"
EXECUTION_COST_ISSUE_WARNING = "warning"
EXECUTION_COST_ISSUE_BLOCKER = "blocker"

EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST = "high_turnover_cost"
EXECUTION_COST_ISSUE_HIGH_IMPACT_COST = "high_impact_cost"
EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST = "high_slippage_cost"
EXECUTION_COST_ISSUE_SPREAD_COST = "spread_cost"
EXECUTION_COST_ISSUE_STAMP_TAX_COST = "stamp_tax_cost"
EXECUTION_COST_ISSUE_BLOCKED_BUY = "blocked_buy"
EXECUTION_COST_ISSUE_BLOCKED_SELL = "blocked_sell"
EXECUTION_COST_ISSUE_PARTIAL_FILL = "partial_fill"
EXECUTION_COST_ISSUE_MISSING_AMOUNT = "missing_amount"
EXECUTION_COST_ISSUE_MISSING_VOLUME = "missing_volume"
EXECUTION_COST_ISSUE_MISSING_PRICE = "missing_price"
EXECUTION_COST_ISSUE_LOW_CAPACITY = "low_capacity"
EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG = "research_short_leg"

COST_MODEL_FIXED_BPS = "fixed_bps"
COST_MODEL_LINEAR_PARTICIPATION = "linear_participation"
COST_MODEL_SQRT_IMPACT = "sqrt_impact"

PENALTY_POLICY_BLOCK_TO_CASH = "block_to_cash"
PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT = "keep_previous_weight"
PENALTY_POLICY_MARK_UNEXECUTABLE_ONLY = "mark_unexecutable_only"

TRADE_DIRECTION_BUY = "buy"
TRADE_DIRECTION_SELL = "sell"
TRADE_DIRECTION_HOLD = "hold"

DEFAULT_FACTOR_EXECUTION_COST_DIR = Path("data/factor_library/execution_cost")
DEFAULT_EXECUTION_COST_REPORTS_FILENAME = "execution_cost_reports.jsonl"
DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME = (
    "execution_adjusted_daily_records.jsonl"
)
DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME = "execution_adjusted_runs.jsonl"
DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME = "execution_cost_report.md"
DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME = "execution_cost_dashboard.json"

EXECUTION_COST_NON_RUNTIME_IMPACT_NOTE = (
    "This execution-cost simulation is offline-only and does not alter official "
    "scoring, stock selection, posterior, RiskGuard, PortfolioConstructor, target "
    "weights, orders, providers, LLMs, or execution."
)

SUPPORTED_EXECUTION_COST_VERDICTS = {
    EXECUTION_COST_SIMULATION_PASS,
    EXECUTION_COST_SIMULATION_WARN,
    EXECUTION_COST_SIMULATION_FAIL,
}
SUPPORTED_EXECUTION_SIMULATION_STATUSES = {
    EXECUTION_SIMULATION_STATUS_OK,
    EXECUTION_SIMULATION_STATUS_PARTIAL,
    EXECUTION_SIMULATION_STATUS_BLOCKED,
    EXECUTION_SIMULATION_STATUS_MISSING_DATA,
}
SUPPORTED_EXECUTION_COST_ISSUE_SEVERITIES = {
    EXECUTION_COST_ISSUE_INFO,
    EXECUTION_COST_ISSUE_WARNING,
    EXECUTION_COST_ISSUE_BLOCKER,
}
SUPPORTED_EXECUTION_COST_ISSUE_CODES = {
    EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST,
    EXECUTION_COST_ISSUE_HIGH_IMPACT_COST,
    EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST,
    EXECUTION_COST_ISSUE_SPREAD_COST,
    EXECUTION_COST_ISSUE_STAMP_TAX_COST,
    EXECUTION_COST_ISSUE_BLOCKED_BUY,
    EXECUTION_COST_ISSUE_BLOCKED_SELL,
    EXECUTION_COST_ISSUE_PARTIAL_FILL,
    EXECUTION_COST_ISSUE_MISSING_AMOUNT,
    EXECUTION_COST_ISSUE_MISSING_VOLUME,
    EXECUTION_COST_ISSUE_MISSING_PRICE,
    EXECUTION_COST_ISSUE_LOW_CAPACITY,
    EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG,
}
SUPPORTED_COST_MODELS = {
    COST_MODEL_FIXED_BPS,
    COST_MODEL_LINEAR_PARTICIPATION,
    COST_MODEL_SQRT_IMPACT,
}
SUPPORTED_PENALTY_POLICIES = {
    PENALTY_POLICY_BLOCK_TO_CASH,
    PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT,
    PENALTY_POLICY_MARK_UNEXECUTABLE_ONLY,
}
SUPPORTED_TRADE_DIRECTIONS = {
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_SELL,
    TRADE_DIRECTION_HOLD,
}

_EPSILON = 1e-12
_SEVERITY_ORDER = {
    EXECUTION_COST_ISSUE_BLOCKER: 0,
    EXECUTION_COST_ISSUE_WARNING: 1,
    EXECUTION_COST_ISSUE_INFO: 2,
}
_BLOCKER_ISSUE_CODES = {
    EXECUTION_COST_ISSUE_BLOCKED_BUY,
    EXECUTION_COST_ISSUE_BLOCKED_SELL,
}
_WARNING_ISSUE_CODES = {
    EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST,
    EXECUTION_COST_ISSUE_HIGH_IMPACT_COST,
    EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST,
    EXECUTION_COST_ISSUE_PARTIAL_FILL,
    EXECUTION_COST_ISSUE_MISSING_AMOUNT,
    EXECUTION_COST_ISSUE_MISSING_VOLUME,
    EXECUTION_COST_ISSUE_MISSING_PRICE,
    EXECUTION_COST_ISSUE_LOW_CAPACITY,
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


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric; got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite numeric; got {value!r}.")
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
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative integer; got {value!r}.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


def _sorted_issue_codes(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = str(value).strip()
        if not code:
            continue
        _validate_supported(code, "issue_code", SUPPORTED_EXECUTION_COST_ISSUE_CODES)
        if code in seen:
            continue
        seen.add(code)
        result.append(code)
    return sorted(result)


def _issue_severity(issue_code: str) -> str:
    if issue_code in _BLOCKER_ISSUE_CODES:
        return EXECUTION_COST_ISSUE_BLOCKER
    if issue_code in _WARNING_ISSUE_CODES:
        return EXECUTION_COST_ISSUE_WARNING
    return EXECUTION_COST_ISSUE_INFO


def _issue_message(issue_code: str, symbol: str | None, date: str | None) -> str:
    subject = symbol or "portfolio"
    suffix = f" on {date}" if date else ""
    messages = {
        EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST: "high daily simulated cost return",
        EXECUTION_COST_ISSUE_HIGH_IMPACT_COST: "high market impact cost",
        EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST: "high slippage cost",
        EXECUTION_COST_ISSUE_SPREAD_COST: "spread cost applied",
        EXECUTION_COST_ISSUE_STAMP_TAX_COST: "stamp tax cost applied",
        EXECUTION_COST_ISSUE_BLOCKED_BUY: "buy transition blocked by tradability constraints",
        EXECUTION_COST_ISSUE_BLOCKED_SELL: "sell transition blocked by tradability constraints",
        EXECUTION_COST_ISSUE_PARTIAL_FILL: "participation exceeds configured capacity threshold",
        EXECUTION_COST_ISSUE_MISSING_AMOUNT: "amount data missing for traded symbol",
        EXECUTION_COST_ISSUE_MISSING_VOLUME: "volume data missing for traded symbol",
        EXECUTION_COST_ISSUE_MISSING_PRICE: "price data missing for traded symbol",
        EXECUTION_COST_ISSUE_LOW_CAPACITY: "participation exceeds max participation rate",
        EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG: "short leg is a research analytic caveat",
    }
    return f"{subject}: {messages[issue_code]}{suffix}."


def _record_sort_key(record: "SymbolExecutionCostRecord") -> tuple[str, str]:
    return (record.date, record.symbol)


def _daily_record_sort_key(record: "DailyExecutionCostRecord") -> str:
    return record.date


def _issue_sort_key(issue: "ExecutionCostIssue") -> tuple[int, str, str, str]:
    return (
        _SEVERITY_ORDER.get(issue.severity, 99),
        issue.date or "",
        issue.symbol or "",
        issue.issue_code,
    )


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


def _mean_optional(values: Sequence[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _matrix_value_by_symbol(
    symbols: Sequence[str],
    values: Sequence[Sequence[float | None]],
    column_index: int,
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for row_index, symbol in enumerate(symbols):
        if row_index >= len(values) or column_index >= len(values[row_index]):
            output[str(symbol)] = None
        else:
            output[str(symbol)] = values[row_index][column_index]
    return output


def _weights_by_symbol(
    symbols: Sequence[str],
    weights: Sequence[Sequence[float | None]],
    column_index: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for row_index, symbol in enumerate(symbols):
        value = None
        if row_index < len(weights) and column_index < len(weights[row_index]):
            value = weights[row_index][column_index]
        output[str(symbol)] = 0.0 if value is None else float(value)
    return output


def _is_long_short_research_run(run: SingleFactorBacktestRun) -> bool:
    if run.mode == BACKTEST_MODE_LONG_SHORT:
        return True
    return any(
        value is not None and float(value) < -_EPSILON
        for row in run.weight_matrix.net_weights
        for value in row
    )


@dataclass
class FactorExecutionCostConfig:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    config_id: str = ""
    market: str = "CN"
    commission_bps: float = 2.0
    stamp_tax_bps: float = 5.0
    exchange_fee_bps: float = 0.5
    slippage_bps: float = 2.0
    spread_bps: float = 0.0
    impact_model: str = COST_MODEL_LINEAR_PARTICIPATION
    impact_coefficient: float = 10.0
    max_participation_rate: float = 0.10
    partial_fill_allowed: bool = True
    penalty_policy: str = PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT
    apply_stamp_tax_on_sell_only: bool = True
    apply_cost_to_research_short_leg: bool = True
    high_cost_warning_threshold: float = 0.01
    high_impact_warning_bps: float = 20.0
    min_amount: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.config_id = _non_empty_str(self.config_id, "config_id")
        self.market = _non_empty_str(self.market, "market")
        for field_name in (
            "commission_bps",
            "stamp_tax_bps",
            "exchange_fee_bps",
            "slippage_bps",
            "spread_bps",
            "impact_coefficient",
            "high_cost_warning_threshold",
            "high_impact_warning_bps",
        ):
            setattr(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        self.max_participation_rate = _finite_float(
            self.max_participation_rate,
            "max_participation_rate",
        )
        if not 0.0 <= self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be in [0, 1].")
        self.partial_fill_allowed = _require_bool(
            self.partial_fill_allowed,
            "partial_fill_allowed",
        )
        self.impact_model = _non_empty_str(self.impact_model, "impact_model")
        _validate_supported(self.impact_model, "impact_model", SUPPORTED_COST_MODELS)
        self.penalty_policy = _non_empty_str(self.penalty_policy, "penalty_policy")
        _validate_supported(self.penalty_policy, "penalty_policy", SUPPORTED_PENALTY_POLICIES)
        self.apply_stamp_tax_on_sell_only = _require_bool(
            self.apply_stamp_tax_on_sell_only,
            "apply_stamp_tax_on_sell_only",
        )
        self.apply_cost_to_research_short_leg = _require_bool(
            self.apply_cost_to_research_short_leg,
            "apply_cost_to_research_short_leg",
        )
        self.min_amount = _optional_non_negative_float(self.min_amount, "min_amount")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_id": self.config_id,
            "market": self.market,
            "commission_bps": self.commission_bps,
            "stamp_tax_bps": self.stamp_tax_bps,
            "exchange_fee_bps": self.exchange_fee_bps,
            "slippage_bps": self.slippage_bps,
            "spread_bps": self.spread_bps,
            "impact_model": self.impact_model,
            "impact_coefficient": self.impact_coefficient,
            "max_participation_rate": self.max_participation_rate,
            "partial_fill_allowed": self.partial_fill_allowed,
            "penalty_policy": self.penalty_policy,
            "apply_stamp_tax_on_sell_only": self.apply_stamp_tax_on_sell_only,
            "apply_cost_to_research_short_leg": self.apply_cost_to_research_short_leg,
            "high_cost_warning_threshold": self.high_cost_warning_threshold,
            "high_impact_warning_bps": self.high_impact_warning_bps,
            "min_amount": self.min_amount,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorExecutionCostConfig":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            config_id=str(data.get("config_id", "")),
            market=str(data.get("market", "CN")),
            commission_bps=float(data.get("commission_bps", 2.0)),
            stamp_tax_bps=float(data.get("stamp_tax_bps", 5.0)),
            exchange_fee_bps=float(data.get("exchange_fee_bps", 0.5)),
            slippage_bps=float(data.get("slippage_bps", 2.0)),
            spread_bps=float(data.get("spread_bps", 0.0)),
            impact_model=str(data.get("impact_model", COST_MODEL_LINEAR_PARTICIPATION)),
            impact_coefficient=float(data.get("impact_coefficient", 10.0)),
            max_participation_rate=float(data.get("max_participation_rate", 0.10)),
            partial_fill_allowed=data.get("partial_fill_allowed", True),
            penalty_policy=str(data.get("penalty_policy", PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT)),
            apply_stamp_tax_on_sell_only=data.get("apply_stamp_tax_on_sell_only", True),
            apply_cost_to_research_short_leg=data.get("apply_cost_to_research_short_leg", True),
            high_cost_warning_threshold=float(data.get("high_cost_warning_threshold", 0.01)),
            high_impact_warning_bps=float(data.get("high_impact_warning_bps", 20.0)),
            min_amount=data.get("min_amount"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExecutionCostIssue:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    issue_id: str = ""
    symbol: str | None = None
    date: str | None = None
    issue_code: str = ""
    severity: str = EXECUTION_COST_ISSUE_WARNING
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.issue_id = _non_empty_str(self.issue_id, "issue_id")
        self.symbol = _optional_str(self.symbol)
        self.date = _optional_str(self.date)
        self.issue_code = _non_empty_str(self.issue_code, "issue_code")
        _validate_supported(
            self.issue_code,
            "issue_code",
            SUPPORTED_EXECUTION_COST_ISSUE_CODES,
        )
        self.severity = _non_empty_str(self.severity, "severity")
        _validate_supported(
            self.severity,
            "severity",
            SUPPORTED_EXECUTION_COST_ISSUE_SEVERITIES,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionCostIssue":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            issue_id=str(data.get("issue_id", "")),
            symbol=data.get("symbol"),
            date=data.get("date"),
            issue_code=str(data.get("issue_code", "")),
            severity=str(data.get("severity", EXECUTION_COST_ISSUE_WARNING)),
            message=str(data.get("message", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class DailyExecutionCostRecord:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    record_id: str = ""
    date: str = ""
    signal_date: str | None = None
    execution_date: str | None = None
    gross_return: float | None = None
    original_after_cost_return: float | None = None
    simulated_cost_return: float | None = None
    simulated_penalty_return: float | None = None
    simulated_after_cost_return: float | None = None
    turnover: float = 0.0
    buy_turnover: float = 0.0
    sell_turnover: float = 0.0
    commission_cost_return: float = 0.0
    stamp_tax_cost_return: float = 0.0
    exchange_fee_cost_return: float = 0.0
    slippage_cost_return: float = 0.0
    spread_cost_return: float = 0.0
    impact_cost_return: float = 0.0
    blocked_buy_count: int = 0
    blocked_sell_count: int = 0
    partial_fill_count: int = 0
    missing_data_count: int = 0
    status: str = EXECUTION_SIMULATION_STATUS_OK
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.record_id = _non_empty_str(self.record_id, "record_id")
        self.date = _non_empty_str(self.date, "date")
        self.signal_date = _optional_str(self.signal_date)
        self.execution_date = _optional_str(self.execution_date)
        for field_name in (
            "gross_return",
            "original_after_cost_return",
            "simulated_cost_return",
            "simulated_penalty_return",
            "simulated_after_cost_return",
        ):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        for field_name in (
            "turnover",
            "buy_turnover",
            "sell_turnover",
            "commission_cost_return",
            "stamp_tax_cost_return",
            "exchange_fee_cost_return",
            "slippage_cost_return",
            "spread_cost_return",
            "impact_cost_return",
        ):
            setattr(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        for field_name in (
            "blocked_buy_count",
            "blocked_sell_count",
            "partial_fill_count",
            "missing_data_count",
        ):
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.status = _non_empty_str(self.status, "status")
        _validate_supported(
            self.status,
            "status",
            SUPPORTED_EXECUTION_SIMULATION_STATUSES,
        )
        self.issue_codes = _sorted_issue_codes(self.issue_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "date": self.date,
            "signal_date": self.signal_date,
            "execution_date": self.execution_date,
            "gross_return": self.gross_return,
            "original_after_cost_return": self.original_after_cost_return,
            "simulated_cost_return": self.simulated_cost_return,
            "simulated_penalty_return": self.simulated_penalty_return,
            "simulated_after_cost_return": self.simulated_after_cost_return,
            "turnover": self.turnover,
            "buy_turnover": self.buy_turnover,
            "sell_turnover": self.sell_turnover,
            "commission_cost_return": self.commission_cost_return,
            "stamp_tax_cost_return": self.stamp_tax_cost_return,
            "exchange_fee_cost_return": self.exchange_fee_cost_return,
            "slippage_cost_return": self.slippage_cost_return,
            "spread_cost_return": self.spread_cost_return,
            "impact_cost_return": self.impact_cost_return,
            "blocked_buy_count": self.blocked_buy_count,
            "blocked_sell_count": self.blocked_sell_count,
            "partial_fill_count": self.partial_fill_count,
            "missing_data_count": self.missing_data_count,
            "status": self.status,
            "issue_codes": list(self.issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DailyExecutionCostRecord":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            record_id=str(data.get("record_id", "")),
            date=str(data.get("date", "")),
            signal_date=data.get("signal_date"),
            execution_date=data.get("execution_date"),
            gross_return=data.get("gross_return"),
            original_after_cost_return=data.get("original_after_cost_return"),
            simulated_cost_return=data.get("simulated_cost_return"),
            simulated_penalty_return=data.get("simulated_penalty_return"),
            simulated_after_cost_return=data.get("simulated_after_cost_return"),
            turnover=float(data.get("turnover", 0.0)),
            buy_turnover=float(data.get("buy_turnover", 0.0)),
            sell_turnover=float(data.get("sell_turnover", 0.0)),
            commission_cost_return=float(data.get("commission_cost_return", 0.0)),
            stamp_tax_cost_return=float(data.get("stamp_tax_cost_return", 0.0)),
            exchange_fee_cost_return=float(data.get("exchange_fee_cost_return", 0.0)),
            slippage_cost_return=float(data.get("slippage_cost_return", 0.0)),
            spread_cost_return=float(data.get("spread_cost_return", 0.0)),
            impact_cost_return=float(data.get("impact_cost_return", 0.0)),
            blocked_buy_count=int(data.get("blocked_buy_count", 0)),
            blocked_sell_count=int(data.get("blocked_sell_count", 0)),
            partial_fill_count=int(data.get("partial_fill_count", 0)),
            missing_data_count=int(data.get("missing_data_count", 0)),
            status=str(data.get("status", EXECUTION_SIMULATION_STATUS_OK)),
            issue_codes=list(data.get("issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SymbolExecutionCostRecord:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    record_id: str = ""
    symbol: str = ""
    date: str = ""
    previous_weight: float = 0.0
    target_weight: float = 0.0
    executable_weight: float = 0.0
    trade_weight: float = 0.0
    executed_trade_weight: float = 0.0
    trade_direction: str = TRADE_DIRECTION_HOLD
    amount: float | None = None
    volume: float | None = None
    price: float | None = None
    participation_rate: float | None = None
    fill_ratio: float | None = None
    commission_cost_return: float = 0.0
    stamp_tax_cost_return: float = 0.0
    exchange_fee_cost_return: float = 0.0
    slippage_cost_return: float = 0.0
    spread_cost_return: float = 0.0
    impact_cost_return: float = 0.0
    penalty_return: float = 0.0
    status: str = EXECUTION_SIMULATION_STATUS_OK
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.record_id = _non_empty_str(self.record_id, "record_id")
        self.symbol = _non_empty_str(self.symbol, "symbol")
        self.date = _non_empty_str(self.date, "date")
        for field_name in (
            "previous_weight",
            "target_weight",
            "executable_weight",
            "trade_weight",
            "executed_trade_weight",
        ):
            setattr(self, field_name, _finite_float(getattr(self, field_name), field_name))
        self.trade_direction = _non_empty_str(self.trade_direction, "trade_direction")
        _validate_supported(self.trade_direction, "trade_direction", SUPPORTED_TRADE_DIRECTIONS)
        for field_name in ("amount", "volume", "price", "participation_rate", "fill_ratio"):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        if self.participation_rate is not None and self.participation_rate < 0.0:
            raise ValueError("participation_rate must be non-negative when provided.")
        if self.fill_ratio is not None and not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be in [0, 1] when provided.")
        for field_name in (
            "commission_cost_return",
            "stamp_tax_cost_return",
            "exchange_fee_cost_return",
            "slippage_cost_return",
            "spread_cost_return",
            "impact_cost_return",
            "penalty_return",
        ):
            setattr(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        self.status = _non_empty_str(self.status, "status")
        _validate_supported(
            self.status,
            "status",
            SUPPORTED_EXECUTION_SIMULATION_STATUSES,
        )
        self.issue_codes = _sorted_issue_codes(self.issue_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "symbol": self.symbol,
            "date": self.date,
            "previous_weight": self.previous_weight,
            "target_weight": self.target_weight,
            "executable_weight": self.executable_weight,
            "trade_weight": self.trade_weight,
            "executed_trade_weight": self.executed_trade_weight,
            "trade_direction": self.trade_direction,
            "amount": self.amount,
            "volume": self.volume,
            "price": self.price,
            "participation_rate": self.participation_rate,
            "fill_ratio": self.fill_ratio,
            "commission_cost_return": self.commission_cost_return,
            "stamp_tax_cost_return": self.stamp_tax_cost_return,
            "exchange_fee_cost_return": self.exchange_fee_cost_return,
            "slippage_cost_return": self.slippage_cost_return,
            "spread_cost_return": self.spread_cost_return,
            "impact_cost_return": self.impact_cost_return,
            "penalty_return": self.penalty_return,
            "status": self.status,
            "issue_codes": list(self.issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolExecutionCostRecord":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            record_id=str(data.get("record_id", "")),
            symbol=str(data.get("symbol", "")),
            date=str(data.get("date", "")),
            previous_weight=float(data.get("previous_weight", 0.0)),
            target_weight=float(data.get("target_weight", 0.0)),
            executable_weight=float(data.get("executable_weight", 0.0)),
            trade_weight=float(data.get("trade_weight", 0.0)),
            executed_trade_weight=float(data.get("executed_trade_weight", 0.0)),
            trade_direction=str(data.get("trade_direction", TRADE_DIRECTION_HOLD)),
            amount=data.get("amount"),
            volume=data.get("volume"),
            price=data.get("price"),
            participation_rate=data.get("participation_rate"),
            fill_ratio=data.get("fill_ratio"),
            commission_cost_return=float(data.get("commission_cost_return", 0.0)),
            stamp_tax_cost_return=float(data.get("stamp_tax_cost_return", 0.0)),
            exchange_fee_cost_return=float(data.get("exchange_fee_cost_return", 0.0)),
            slippage_cost_return=float(data.get("slippage_cost_return", 0.0)),
            spread_cost_return=float(data.get("spread_cost_return", 0.0)),
            impact_cost_return=float(data.get("impact_cost_return", 0.0)),
            penalty_return=float(data.get("penalty_return", 0.0)),
            status=str(data.get("status", EXECUTION_SIMULATION_STATUS_OK)),
            issue_codes=list(data.get("issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorExecutionCostSimulationReport:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    factor_matrix_id: str | None = None
    backtest_run_id: str | None = None
    weight_matrix_id: str | None = None
    tradability_mask_id: str | None = None
    config: FactorExecutionCostConfig = field(
        default_factory=lambda: FactorExecutionCostConfig(config_id="default-execution-cost-config")
    )
    original_sample_days: int = 0
    simulated_sample_days: int = 0
    average_turnover: float | None = None
    average_cost_return: float | None = None
    average_penalty_return: float | None = None
    cumulative_original_return: float | None = None
    cumulative_simulated_return: float | None = None
    annualized_original_return: float | None = None
    annualized_simulated_return: float | None = None
    original_sharpe: float | None = None
    simulated_sharpe: float | None = None
    original_max_drawdown: float | None = None
    simulated_max_drawdown: float | None = None
    cost_drag_annualized_return: float | None = None
    cost_drag_sharpe: float | None = None
    blocked_buy_count: int = 0
    blocked_sell_count: int = 0
    partial_fill_count: int = 0
    missing_data_count: int = 0
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    daily_records: list[DailyExecutionCostRecord] = field(default_factory=list)
    symbol_records: list[SymbolExecutionCostRecord] = field(default_factory=list)
    issues: list[ExecutionCostIssue] = field(default_factory=list)
    verdict: str = EXECUTION_COST_SIMULATION_PASS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.factor_matrix_id = _optional_str(self.factor_matrix_id)
        self.backtest_run_id = _optional_str(self.backtest_run_id)
        self.weight_matrix_id = _optional_str(self.weight_matrix_id)
        self.tradability_mask_id = _optional_str(self.tradability_mask_id)
        if not isinstance(self.config, FactorExecutionCostConfig):
            self.config = FactorExecutionCostConfig.from_dict(self.config)
        self.daily_records = [
            record if isinstance(record, DailyExecutionCostRecord)
            else DailyExecutionCostRecord.from_dict(record)
            for record in self.daily_records
        ]
        self.daily_records = sorted(self.daily_records, key=_daily_record_sort_key)
        self.symbol_records = [
            record if isinstance(record, SymbolExecutionCostRecord)
            else SymbolExecutionCostRecord.from_dict(record)
            for record in self.symbol_records
        ]
        self.symbol_records = sorted(self.symbol_records, key=_record_sort_key)
        self.issues = [
            issue if isinstance(issue, ExecutionCostIssue)
            else ExecutionCostIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.issues = sorted(self.issues, key=_issue_sort_key)
        self.original_sample_days = _non_negative_int(
            self.original_sample_days,
            "original_sample_days",
        )
        self.simulated_sample_days = _non_negative_int(
            self.simulated_sample_days,
            "simulated_sample_days",
        )
        for field_name in (
            "average_turnover",
            "average_cost_return",
            "average_penalty_return",
            "cumulative_original_return",
            "cumulative_simulated_return",
            "annualized_original_return",
            "annualized_simulated_return",
            "original_sharpe",
            "simulated_sharpe",
            "original_max_drawdown",
            "simulated_max_drawdown",
            "cost_drag_annualized_return",
            "cost_drag_sharpe",
        ):
            setattr(self, field_name, _optional_finite_float(getattr(self, field_name), field_name))
        for field_name in (
            "blocked_buy_count",
            "blocked_sell_count",
            "partial_fill_count",
            "missing_data_count",
            "issue_count",
            "blocker_count",
            "warning_count",
            "info_count",
        ):
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.issue_count = len(self.issues)
        self.blocker_count = sum(
            1 for issue in self.issues if issue.severity == EXECUTION_COST_ISSUE_BLOCKER
        )
        self.warning_count = sum(
            1 for issue in self.issues if issue.severity == EXECUTION_COST_ISSUE_WARNING
        )
        self.info_count = sum(
            1 for issue in self.issues if issue.severity == EXECUTION_COST_ISSUE_INFO
        )
        self.verdict = _non_empty_str(self.verdict, "verdict")
        _validate_supported(self.verdict, "verdict", SUPPORTED_EXECUTION_COST_VERDICTS)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "factor_matrix_id": self.factor_matrix_id,
            "backtest_run_id": self.backtest_run_id,
            "weight_matrix_id": self.weight_matrix_id,
            "tradability_mask_id": self.tradability_mask_id,
            "config": self.config.to_dict(),
            "original_sample_days": self.original_sample_days,
            "simulated_sample_days": self.simulated_sample_days,
            "average_turnover": self.average_turnover,
            "average_cost_return": self.average_cost_return,
            "average_penalty_return": self.average_penalty_return,
            "cumulative_original_return": self.cumulative_original_return,
            "cumulative_simulated_return": self.cumulative_simulated_return,
            "annualized_original_return": self.annualized_original_return,
            "annualized_simulated_return": self.annualized_simulated_return,
            "original_sharpe": self.original_sharpe,
            "simulated_sharpe": self.simulated_sharpe,
            "original_max_drawdown": self.original_max_drawdown,
            "simulated_max_drawdown": self.simulated_max_drawdown,
            "cost_drag_annualized_return": self.cost_drag_annualized_return,
            "cost_drag_sharpe": self.cost_drag_sharpe,
            "blocked_buy_count": self.blocked_buy_count,
            "blocked_sell_count": self.blocked_sell_count,
            "partial_fill_count": self.partial_fill_count,
            "missing_data_count": self.missing_data_count,
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "daily_records": [record.to_dict() for record in self.daily_records],
            "symbol_records": [record.to_dict() for record in self.symbol_records],
            "issues": [issue.to_dict() for issue in self.issues],
            "verdict": self.verdict,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorExecutionCostSimulationReport":
        data = dict(payload)
        config_payload = data.get("config", {}) or {}
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            factor_matrix_id=data.get("factor_matrix_id"),
            backtest_run_id=data.get("backtest_run_id"),
            weight_matrix_id=data.get("weight_matrix_id"),
            tradability_mask_id=data.get("tradability_mask_id"),
            config=FactorExecutionCostConfig.from_dict(config_payload)
            if isinstance(config_payload, Mapping)
            else config_payload,
            original_sample_days=int(data.get("original_sample_days", 0)),
            simulated_sample_days=int(data.get("simulated_sample_days", 0)),
            average_turnover=data.get("average_turnover"),
            average_cost_return=data.get("average_cost_return"),
            average_penalty_return=data.get("average_penalty_return"),
            cumulative_original_return=data.get("cumulative_original_return"),
            cumulative_simulated_return=data.get("cumulative_simulated_return"),
            annualized_original_return=data.get("annualized_original_return"),
            annualized_simulated_return=data.get("annualized_simulated_return"),
            original_sharpe=data.get("original_sharpe"),
            simulated_sharpe=data.get("simulated_sharpe"),
            original_max_drawdown=data.get("original_max_drawdown"),
            simulated_max_drawdown=data.get("simulated_max_drawdown"),
            cost_drag_annualized_return=data.get("cost_drag_annualized_return"),
            cost_drag_sharpe=data.get("cost_drag_sharpe"),
            blocked_buy_count=int(data.get("blocked_buy_count", 0)),
            blocked_sell_count=int(data.get("blocked_sell_count", 0)),
            partial_fill_count=int(data.get("partial_fill_count", 0)),
            missing_data_count=int(data.get("missing_data_count", 0)),
            issue_count=int(data.get("issue_count", 0)),
            blocker_count=int(data.get("blocker_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            info_count=int(data.get("info_count", 0)),
            daily_records=[
                DailyExecutionCostRecord.from_dict(record)
                for record in list(data.get("daily_records", []) or [])
                if isinstance(record, Mapping)
            ],
            symbol_records=[
                SymbolExecutionCostRecord.from_dict(record)
                for record in list(data.get("symbol_records", []) or [])
                if isinstance(record, Mapping)
            ],
            issues=[
                ExecutionCostIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            verdict=str(data.get("verdict", EXECUTION_COST_SIMULATION_PASS)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExecutionAdjustedBacktestRun:
    schema_version: str = FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
    adjusted_run_id: str = ""
    source_backtest_run_id: str = ""
    cost_report_id: str = ""
    generated_at: str = ""
    original_daily_returns: dict[str, float | None] = field(default_factory=dict)
    simulated_daily_returns: dict[str, float | None] = field(default_factory=dict)
    cost_returns_by_date: dict[str, float | None] = field(default_factory=dict)
    penalty_returns_by_date: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(
            self.schema_version or FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        )
        self.adjusted_run_id = _non_empty_str(self.adjusted_run_id, "adjusted_run_id")
        self.source_backtest_run_id = _non_empty_str(
            self.source_backtest_run_id,
            "source_backtest_run_id",
        )
        self.cost_report_id = _non_empty_str(self.cost_report_id, "cost_report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.original_daily_returns = _coerce_return_series(
            self.original_daily_returns,
            "original_daily_returns",
        )
        self.simulated_daily_returns = _coerce_return_series(
            self.simulated_daily_returns,
            "simulated_daily_returns",
        )
        self.cost_returns_by_date = _coerce_return_series(
            self.cost_returns_by_date,
            "cost_returns_by_date",
        )
        self.penalty_returns_by_date = _coerce_return_series(
            self.penalty_returns_by_date,
            "penalty_returns_by_date",
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adjusted_run_id": self.adjusted_run_id,
            "source_backtest_run_id": self.source_backtest_run_id,
            "cost_report_id": self.cost_report_id,
            "generated_at": self.generated_at,
            "original_daily_returns": dict(_json_safe(self.original_daily_returns)),
            "simulated_daily_returns": dict(_json_safe(self.simulated_daily_returns)),
            "cost_returns_by_date": dict(_json_safe(self.cost_returns_by_date)),
            "penalty_returns_by_date": dict(_json_safe(self.penalty_returns_by_date)),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionAdjustedBacktestRun":
        data = dict(payload)
        return cls(
            schema_version=str(
                data.get("schema_version", FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION)
            ),
            adjusted_run_id=str(data.get("adjusted_run_id", "")),
            source_backtest_run_id=str(data.get("source_backtest_run_id", "")),
            cost_report_id=str(data.get("cost_report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            original_daily_returns=dict(data.get("original_daily_returns", {}) or {}),
            simulated_daily_returns=dict(data.get("simulated_daily_returns", {}) or {}),
            cost_returns_by_date=dict(data.get("cost_returns_by_date", {}) or {}),
            penalty_returns_by_date=dict(data.get("penalty_returns_by_date", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _coerce_return_series(values: Mapping[str, Any], field_name: str) -> dict[str, float | None]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {
        str(date_key): _optional_finite_float(value, f"{field_name}.{date_key}")
        for date_key, value in sorted(values.items(), key=lambda item: str(item[0]))
    }


def make_execution_cost_config_id(config: FactorExecutionCostConfig) -> str:
    payload = config.to_dict()
    payload.pop("config_id", None)
    return (
        f"execution-cost-config-{_slug(config.market)}-"
        f"{_slug(config.impact_model)}-{_short_hash([payload])}"
    )


def make_execution_cost_issue_id(
    *,
    symbol: str | None,
    date: str | None,
    issue_code: str,
    message: str,
) -> str:
    parts = [symbol, date, str(issue_code), str(message)]
    return (
        f"execution-cost-issue-{_slug(issue_code)}-"
        f"{_slug(symbol)}-{_slug(date)}-{_short_hash(parts)}"
    )


def make_daily_execution_cost_record_id(
    *,
    backtest_run_id: str | None,
    date: str,
) -> str:
    parts = [backtest_run_id, str(date)]
    return f"execution-cost-daily-{_slug(backtest_run_id)}-{_slug(date)}-{_short_hash(parts)}"


def make_symbol_execution_cost_record_id(
    *,
    symbol: str,
    date: str,
    target_weight: float,
) -> str:
    parts = [str(symbol), str(date), float(target_weight)]
    return (
        f"execution-cost-symbol-{_slug(symbol)}-{_slug(date)}-"
        f"{_short_hash(parts)}"
    )


def make_execution_cost_report_id(
    *,
    backtest_run_id: str | None,
    weight_matrix_id: str | None,
    generated_at: str,
) -> str:
    parts = [backtest_run_id, weight_matrix_id, str(generated_at)]
    return (
        f"execution-cost-report-{_slug(backtest_run_id)}-"
        f"{_slug(weight_matrix_id)}-{_slug(generated_at)}-{_short_hash(parts)}"
    )


def make_execution_adjusted_run_id(
    *,
    source_backtest_run_id: str,
    cost_report_id: str,
) -> str:
    parts = [str(source_backtest_run_id), str(cost_report_id)]
    return (
        f"execution-adjusted-run-{_slug(source_backtest_run_id)}-"
        f"{_slug(cost_report_id)}-{_short_hash(parts)}"
    )


def bps_to_decimal_return(value_bps: float) -> float:
    return _finite_float(value_bps, "value_bps") / 10000.0


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp_unit_interval(value: float) -> float:
    number = _finite_float(value, "value")
    return min(1.0, max(0.0, number))


def infer_trade_direction(trade_weight: float) -> str:
    number = _finite_float(trade_weight, "trade_weight")
    if number > _EPSILON:
        return TRADE_DIRECTION_BUY
    if number < -_EPSILON:
        return TRADE_DIRECTION_SELL
    return TRADE_DIRECTION_HOLD


def estimate_participation_rate(
    *,
    trade_weight: float,
    portfolio_value: float | None,
    amount: float | None,
) -> float | None:
    weight = _finite_float(trade_weight, "trade_weight")
    resolved_portfolio_value = safe_float(portfolio_value)
    resolved_amount = safe_float(amount)
    if (
        resolved_portfolio_value is None
        or resolved_amount is None
        or resolved_portfolio_value <= 0.0
        or resolved_amount <= 0.0
    ):
        return None
    return abs(weight) * resolved_portfolio_value / resolved_amount


def estimate_market_impact_bps(
    *,
    participation_rate: float | None,
    config: FactorExecutionCostConfig,
) -> float:
    participation = safe_float(participation_rate)
    if participation is None or participation < 0.0:
        return 0.0
    if config.impact_model == COST_MODEL_FIXED_BPS:
        return config.impact_coefficient
    if config.impact_model == COST_MODEL_SQRT_IMPACT:
        return config.impact_coefficient * math.sqrt(participation)
    if config.impact_model == COST_MODEL_LINEAR_PARTICIPATION:
        return config.impact_coefficient * participation
    raise ValueError(f"Unsupported impact_model: {config.impact_model!r}.")


def _extract_numeric_matrix(bundle: MatrixDataBundle, field_name: str) -> list[list[float | None]]:
    rows = len(bundle.contract.symbols)
    columns = len(bundle.contract.dates)
    if not bundle.has_field(field_name):
        return [[None for _ in range(columns)] for _ in range(rows)]
    values = bundle.get_field(field_name)
    bundle.validate_shape(values, field_name=field_name)
    return [[safe_float(item) for item in row] for row in values]


def extract_amount_matrix(
    bundle: MatrixDataBundle,
    amount_field: str = FIELD_AMOUNT,
) -> list[list[float | None]]:
    return _extract_numeric_matrix(bundle, amount_field)


def extract_volume_matrix(
    bundle: MatrixDataBundle,
    volume_field: str = FIELD_VOLUME,
) -> list[list[float | None]]:
    return _extract_numeric_matrix(bundle, volume_field)


def extract_price_matrix(
    bundle: MatrixDataBundle,
    price_field: str,
) -> list[list[float | None]]:
    if bundle.has_field(price_field):
        return _extract_numeric_matrix(bundle, price_field)
    if price_field != FIELD_VWAP:
        return _extract_numeric_matrix(bundle, price_field)
    amount_matrix = extract_amount_matrix(bundle)
    volume_matrix = extract_volume_matrix(bundle)
    output: list[list[float | None]] = []
    for amount_row, volume_row in zip(amount_matrix, volume_matrix):
        price_row: list[float | None] = []
        for amount, volume in zip(amount_row, volume_row):
            if amount is None or volume is None or amount <= 0.0 or volume <= 0.0:
                price_row.append(None)
            else:
                price_row.append(amount / volume)
        output.append(price_row)
    return output


def _default_execution_cost_config() -> FactorExecutionCostConfig:
    config = FactorExecutionCostConfig(config_id="placeholder")
    config.config_id = make_execution_cost_config_id(config)
    return config


def _mask_cell(
    tradability_mask: AShareTradabilityMask,
    *,
    symbol: str,
    date_index: int,
) -> tuple[bool, bool, bool, bool, list[str]]:
    try:
        row_index = tradability_mask.symbols.index(symbol)
    except ValueError:
        return False, False, False, False, ["symbol_missing_from_tradability_mask"]
    if date_index < 0 or date_index >= len(tradability_mask.dates):
        return False, False, False, False, ["date_missing_from_tradability_mask"]
    return (
        bool(tradability_mask.can_buy_mask[row_index][date_index]),
        bool(tradability_mask.can_sell_mask[row_index][date_index]),
        bool(tradability_mask.can_trade_mask[row_index][date_index]),
        bool(tradability_mask.can_hold_mask[row_index][date_index]),
        list(tradability_mask.issue_codes_by_cell[row_index][date_index]),
    )


def _build_issue(
    *,
    symbol: str | None,
    date: str | None,
    issue_code: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionCostIssue:
    message = _issue_message(issue_code, symbol, date)
    return ExecutionCostIssue(
        issue_id=make_execution_cost_issue_id(
            symbol=symbol,
            date=date,
            issue_code=issue_code,
            message=message,
        ),
        symbol=symbol,
        date=date,
        issue_code=issue_code,
        severity=_issue_severity(issue_code),
        message=message,
        metadata=dict(metadata or {}),
    )


def simulate_executable_weights_for_day(
    *,
    symbols: Sequence[str],
    date: str,
    previous_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    tradability_mask: AShareTradabilityMask | None,
    date_index: int,
    config: FactorExecutionCostConfig,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], list[SymbolExecutionCostRecord], list[ExecutionCostIssue]]:
    executable_weights: dict[str, float] = {}
    records: list[SymbolExecutionCostRecord] = []
    issues: list[ExecutionCostIssue] = []
    base_metadata = _coerce_metadata(metadata)

    for symbol in symbols:
        previous_weight = float(previous_weights.get(symbol, 0.0) or 0.0)
        target_weight = float(target_weights.get(symbol, 0.0) or 0.0)
        trade_weight = target_weight - previous_weight
        trade_direction = infer_trade_direction(trade_weight)
        status = EXECUTION_SIMULATION_STATUS_OK
        issue_codes: list[str] = []
        record_metadata = dict(base_metadata)
        can_buy = can_sell = can_trade = can_hold = True
        mask_issue_codes: list[str] = []
        if tradability_mask is None:
            record_metadata["no_tradability_mask_provided"] = True
        else:
            can_buy, can_sell, can_trade, can_hold, mask_issue_codes = _mask_cell(
                tradability_mask,
                symbol=str(symbol),
                date_index=date_index,
            )
            record_metadata["tradability_cell_issue_codes"] = list(mask_issue_codes)

        blocked_issue: str | None = None
        if trade_direction == TRADE_DIRECTION_BUY and (not can_trade or not can_buy):
            blocked_issue = EXECUTION_COST_ISSUE_BLOCKED_BUY
        elif trade_direction == TRADE_DIRECTION_SELL and (not can_trade or not can_sell):
            blocked_issue = EXECUTION_COST_ISSUE_BLOCKED_SELL
        elif trade_direction == TRADE_DIRECTION_HOLD and (not can_trade or not can_hold):
            blocked_issue = EXECUTION_COST_ISSUE_BLOCKED_SELL

        if blocked_issue is not None:
            status = EXECUTION_SIMULATION_STATUS_BLOCKED
            issue_codes.append(blocked_issue)
            issues.append(
                _build_issue(
                    symbol=str(symbol),
                    date=date,
                    issue_code=blocked_issue,
                    metadata={"tradability_cell_issue_codes": mask_issue_codes},
                )
            )
            if config.penalty_policy == PENALTY_POLICY_MARK_UNEXECUTABLE_ONLY:
                executable_weight = target_weight
                executed_trade_weight = trade_weight
            else:
                executable_weight = previous_weight
                executed_trade_weight = 0.0
            record_metadata["penalty_policy_simplification"] = (
                "Blocked transitions are retained at previous weight for "
                "keep_previous_weight and block_to_cash; this pass does not model cash mechanics."
            )
        else:
            executable_weight = target_weight if trade_direction != TRADE_DIRECTION_HOLD else previous_weight
            executed_trade_weight = executable_weight - previous_weight

        executable_weights[str(symbol)] = executable_weight
        records.append(
            SymbolExecutionCostRecord(
                record_id=make_symbol_execution_cost_record_id(
                    symbol=str(symbol),
                    date=date,
                    target_weight=target_weight,
                ),
                symbol=str(symbol),
                date=date,
                previous_weight=previous_weight,
                target_weight=target_weight,
                executable_weight=executable_weight,
                trade_weight=trade_weight,
                executed_trade_weight=executed_trade_weight,
                trade_direction=trade_direction,
                status=status,
                issue_codes=issue_codes,
                metadata=record_metadata,
            )
        )

    return executable_weights, sorted(records, key=_record_sort_key), sorted(issues, key=_issue_sort_key)


def _clone_symbol_record(
    record: SymbolExecutionCostRecord,
    **updates: Any,
) -> SymbolExecutionCostRecord:
    payload = record.to_dict()
    payload.update(updates)
    return SymbolExecutionCostRecord.from_dict(payload)


def compute_symbol_execution_costs_for_day(
    *,
    symbol_records: Sequence[SymbolExecutionCostRecord],
    amount_by_symbol: Mapping[str, float | None],
    volume_by_symbol: Mapping[str, float | None],
    price_by_symbol: Mapping[str, float | None],
    portfolio_value: float | None,
    config: FactorExecutionCostConfig,
) -> list[SymbolExecutionCostRecord]:
    output: list[SymbolExecutionCostRecord] = []
    for record in symbol_records:
        amount = safe_float(amount_by_symbol.get(record.symbol))
        volume = safe_float(volume_by_symbol.get(record.symbol))
        price = safe_float(price_by_symbol.get(record.symbol))
        participation_rate = estimate_participation_rate(
            trade_weight=record.executed_trade_weight,
            portfolio_value=portfolio_value,
            amount=amount,
        )
        impact_bps = estimate_market_impact_bps(
            participation_rate=participation_rate,
            config=config,
        )
        abs_trade_weight = abs(record.executed_trade_weight)
        issue_codes = list(record.issue_codes)
        status = record.status
        fill_ratio: float | None
        if abs(record.trade_weight) <= _EPSILON:
            fill_ratio = None
        elif status == EXECUTION_SIMULATION_STATUS_BLOCKED:
            fill_ratio = 0.0
        else:
            fill_ratio = 1.0

        if abs_trade_weight > _EPSILON:
            if amount is None:
                issue_codes.append(EXECUTION_COST_ISSUE_MISSING_AMOUNT)
            if volume is None:
                issue_codes.append(EXECUTION_COST_ISSUE_MISSING_VOLUME)
            if price is None:
                issue_codes.append(EXECUTION_COST_ISSUE_MISSING_PRICE)
            if participation_rate is not None and participation_rate > config.max_participation_rate:
                issue_codes.extend([
                    EXECUTION_COST_ISSUE_PARTIAL_FILL,
                    EXECUTION_COST_ISSUE_LOW_CAPACITY,
                ])
                if status != EXECUTION_SIMULATION_STATUS_BLOCKED:
                    status = EXECUTION_SIMULATION_STATUS_PARTIAL
                    if participation_rate > 0.0:
                        fill_ratio = clamp_unit_interval(
                            config.max_participation_rate / participation_rate
                        )
            if impact_bps >= config.high_impact_warning_bps and impact_bps > 0.0:
                issue_codes.append(EXECUTION_COST_ISSUE_HIGH_IMPACT_COST)
            if config.slippage_bps >= config.high_impact_warning_bps and config.slippage_bps > 0.0:
                issue_codes.append(EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST)
            if config.spread_bps > 0.0:
                issue_codes.append(EXECUTION_COST_ISSUE_SPREAD_COST)

        stamp_applies = (
            abs_trade_weight > _EPSILON
            and (
                not config.apply_stamp_tax_on_sell_only
                or record.trade_direction == TRADE_DIRECTION_SELL
            )
        )
        if stamp_applies and config.stamp_tax_bps > 0.0:
            issue_codes.append(EXECUTION_COST_ISSUE_STAMP_TAX_COST)

        output.append(
            _clone_symbol_record(
                record,
                amount=amount,
                volume=volume,
                price=price,
                participation_rate=participation_rate,
                fill_ratio=fill_ratio,
                commission_cost_return=abs_trade_weight
                * bps_to_decimal_return(config.commission_bps),
                stamp_tax_cost_return=(
                    abs_trade_weight * bps_to_decimal_return(config.stamp_tax_bps)
                    if stamp_applies
                    else 0.0
                ),
                exchange_fee_cost_return=abs_trade_weight
                * bps_to_decimal_return(config.exchange_fee_bps),
                slippage_cost_return=abs_trade_weight
                * bps_to_decimal_return(config.slippage_bps),
                spread_cost_return=abs_trade_weight
                * bps_to_decimal_return(config.spread_bps),
                impact_cost_return=abs_trade_weight * bps_to_decimal_return(impact_bps),
                status=status,
                issue_codes=issue_codes,
                metadata={
                    **record.metadata,
                    "impact_bps": impact_bps,
                    "cost_model": config.impact_model,
                    "max_participation_rate": config.max_participation_rate,
                },
            )
        )

    return sorted(output, key=_record_sort_key)


def _issues_from_symbol_records(
    records: Sequence[SymbolExecutionCostRecord],
) -> list[ExecutionCostIssue]:
    issues_by_id: dict[str, ExecutionCostIssue] = {}
    for record in records:
        for issue_code in record.issue_codes:
            issue = _build_issue(
                symbol=record.symbol,
                date=record.date,
                issue_code=issue_code,
                metadata={
                    "record_id": record.record_id,
                    "status": record.status,
                    "participation_rate": record.participation_rate,
                    "fill_ratio": record.fill_ratio,
                },
            )
            issues_by_id[issue.issue_id] = issue
    return sorted(issues_by_id.values(), key=_issue_sort_key)


def _dedupe_issues(issues: Sequence[ExecutionCostIssue]) -> list[ExecutionCostIssue]:
    by_id: dict[str, ExecutionCostIssue] = {}
    for issue in issues:
        by_id[issue.issue_id] = issue
    return sorted(by_id.values(), key=_issue_sort_key)


def _daily_status(issue_codes: Sequence[str]) -> str:
    if (
        EXECUTION_COST_ISSUE_BLOCKED_BUY in issue_codes
        or EXECUTION_COST_ISSUE_BLOCKED_SELL in issue_codes
    ):
        return EXECUTION_SIMULATION_STATUS_BLOCKED
    if EXECUTION_COST_ISSUE_PARTIAL_FILL in issue_codes:
        return EXECUTION_SIMULATION_STATUS_PARTIAL
    if (
        EXECUTION_COST_ISSUE_MISSING_AMOUNT in issue_codes
        or EXECUTION_COST_ISSUE_MISSING_VOLUME in issue_codes
        or EXECUTION_COST_ISSUE_MISSING_PRICE in issue_codes
    ):
        return EXECUTION_SIMULATION_STATUS_MISSING_DATA
    return EXECUTION_SIMULATION_STATUS_OK


def _set_symbol_penalty_returns(
    records: Sequence[SymbolExecutionCostRecord],
    *,
    gross_return: float | None,
) -> list[SymbolExecutionCostRecord]:
    gross_abs = abs(gross_return or 0.0)
    output: list[SymbolExecutionCostRecord] = []
    for record in records:
        if record.status != EXECUTION_SIMULATION_STATUS_BLOCKED:
            output.append(record)
            continue
        penalty_return = abs(record.target_weight - record.executable_weight) * gross_abs
        output.append(_clone_symbol_record(record, penalty_return=penalty_return))
    return sorted(output, key=_record_sort_key)


def build_daily_execution_cost_records(
    *,
    run: SingleFactorBacktestRun,
    bundle: MatrixDataBundle,
    tradability_mask: AShareTradabilityMask | None = None,
    config: FactorExecutionCostConfig | None = None,
    portfolio_value: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[DailyExecutionCostRecord], list[SymbolExecutionCostRecord], list[ExecutionCostIssue]]:
    resolved_config = config or _default_execution_cost_config()
    base_metadata = _coerce_metadata(metadata)
    symbols = list(run.weight_matrix.symbols)
    weight_dates = list(run.weight_matrix.dates)
    bundle_dates = list(bundle.contract.dates)
    price_matrix = extract_price_matrix(bundle, FIELD_VWAP)
    amount_matrix = extract_amount_matrix(bundle)
    volume_matrix = extract_volume_matrix(bundle)

    daily_records: list[DailyExecutionCostRecord] = []
    symbol_records: list[SymbolExecutionCostRecord] = []
    issues: list[ExecutionCostIssue] = []

    for source_record in run.daily_records:
        if source_record.signal_date not in weight_dates:
            continue
        signal_index = weight_dates.index(source_record.signal_date)
        if source_record.execution_start_date not in bundle_dates:
            continue
        execution_index = bundle_dates.index(source_record.execution_start_date)
        previous_index = signal_index - 1
        previous_weights = (
            {symbol: 0.0 for symbol in symbols}
            if previous_index < 0
            else _weights_by_symbol(symbols, run.weight_matrix.net_weights, previous_index)
        )
        target_weights = _weights_by_symbol(symbols, run.weight_matrix.net_weights, signal_index)
        _executable_weights, transition_records, transition_issues = (
            simulate_executable_weights_for_day(
                symbols=symbols,
                date=source_record.execution_start_date,
                previous_weights=previous_weights,
                target_weights=target_weights,
                tradability_mask=tradability_mask,
                date_index=execution_index,
                config=resolved_config,
                metadata=base_metadata,
            )
        )
        costed_records = compute_symbol_execution_costs_for_day(
            symbol_records=transition_records,
            amount_by_symbol=_matrix_value_by_symbol(symbols, amount_matrix, execution_index),
            volume_by_symbol=_matrix_value_by_symbol(symbols, volume_matrix, execution_index),
            price_by_symbol=_matrix_value_by_symbol(symbols, price_matrix, execution_index),
            portfolio_value=portfolio_value,
            config=resolved_config,
        )
        gross_return = source_record.long_short_return
        costed_records = _set_symbol_penalty_returns(
            costed_records,
            gross_return=gross_return,
        )
        symbol_records.extend(costed_records)
        issues.extend(transition_issues)
        issues.extend(_issues_from_symbol_records(costed_records))

        commission_cost = sum(record.commission_cost_return for record in costed_records)
        stamp_tax_cost = sum(record.stamp_tax_cost_return for record in costed_records)
        exchange_fee_cost = sum(record.exchange_fee_cost_return for record in costed_records)
        slippage_cost = sum(record.slippage_cost_return for record in costed_records)
        spread_cost = sum(record.spread_cost_return for record in costed_records)
        impact_cost = sum(record.impact_cost_return for record in costed_records)
        simulated_cost_return = (
            commission_cost
            + stamp_tax_cost
            + exchange_fee_cost
            + slippage_cost
            + spread_cost
            + impact_cost
        )
        simulated_penalty_return = sum(record.penalty_return for record in costed_records)
        simulated_after_cost_return = (
            None
            if gross_return is None
            else gross_return - simulated_cost_return - simulated_penalty_return
        )
        issue_codes = _sorted_issue_codes(
            [code for record in costed_records for code in record.issue_codes]
        )
        if simulated_cost_return >= resolved_config.high_cost_warning_threshold:
            issue_codes.append(EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST)
            issues.append(
                _build_issue(
                    symbol=None,
                    date=source_record.date,
                    issue_code=EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST,
                    metadata={
                        "simulated_cost_return": simulated_cost_return,
                        "threshold": resolved_config.high_cost_warning_threshold,
                    },
                )
            )
        issue_codes = _sorted_issue_codes(issue_codes)
        buy_turnover = sum(max(record.trade_weight, 0.0) for record in costed_records)
        sell_turnover = sum(abs(min(record.trade_weight, 0.0)) for record in costed_records)
        daily_records.append(
            DailyExecutionCostRecord(
                record_id=make_daily_execution_cost_record_id(
                    backtest_run_id=run.run_id,
                    date=source_record.date,
                ),
                date=source_record.date,
                signal_date=source_record.signal_date,
                execution_date=source_record.execution_start_date,
                gross_return=gross_return,
                original_after_cost_return=source_record.after_cost_return,
                simulated_cost_return=simulated_cost_return,
                simulated_penalty_return=simulated_penalty_return,
                simulated_after_cost_return=simulated_after_cost_return,
                turnover=source_record.turnover,
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                commission_cost_return=commission_cost,
                stamp_tax_cost_return=stamp_tax_cost,
                exchange_fee_cost_return=exchange_fee_cost,
                slippage_cost_return=slippage_cost,
                spread_cost_return=spread_cost,
                impact_cost_return=impact_cost,
                blocked_buy_count=sum(
                    1 for record in costed_records
                    if EXECUTION_COST_ISSUE_BLOCKED_BUY in record.issue_codes
                ),
                blocked_sell_count=sum(
                    1 for record in costed_records
                    if EXECUTION_COST_ISSUE_BLOCKED_SELL in record.issue_codes
                ),
                partial_fill_count=sum(
                    1 for record in costed_records
                    if EXECUTION_COST_ISSUE_PARTIAL_FILL in record.issue_codes
                ),
                missing_data_count=sum(
                    1 for record in costed_records
                    if (
                        EXECUTION_COST_ISSUE_MISSING_AMOUNT in record.issue_codes
                        or EXECUTION_COST_ISSUE_MISSING_VOLUME in record.issue_codes
                        or EXECUTION_COST_ISSUE_MISSING_PRICE in record.issue_codes
                    )
                ),
                status=_daily_status(issue_codes),
                issue_codes=issue_codes,
                metadata={
                    **base_metadata,
                    "penalty_return_rule": (
                        "blocked penalty = abs(target_weight - executable_weight) "
                        "* abs(gross_return); no cash or broker mechanics are modeled"
                    ),
                    "source_daily_record": source_record.to_dict(),
                },
            )
        )

    return (
        sorted(daily_records, key=_daily_record_sort_key),
        sorted(symbol_records, key=_record_sort_key),
        _dedupe_issues(issues),
    )


def build_execution_cost_simulation_report(
    *,
    run: SingleFactorBacktestRun,
    bundle: MatrixDataBundle,
    tradability_mask: AShareTradabilityMask | None = None,
    config: FactorExecutionCostConfig | None = None,
    portfolio_value: float | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorExecutionCostSimulationReport:
    resolved_config = config or _default_execution_cost_config()
    daily_records, symbol_records, issues = build_daily_execution_cost_records(
        run=run,
        bundle=bundle,
        tradability_mask=tradability_mask,
        config=resolved_config,
        portfolio_value=portfolio_value,
        metadata=metadata,
    )
    report_metadata = {
        **_coerce_metadata(metadata),
        "factor_execution_cost_simulation_schema_version": (
            FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION
        ),
        "factor_execution_penalty_schema_version": FACTOR_EXECUTION_PENALTY_SCHEMA_VERSION,
        "non_runtime_impact": True,
        "no_original_backtest_mutation": True,
        "no_admission_default_change": True,
    }
    if _is_long_short_research_run(run):
        report_metadata["short_leg_is_research_analytic_not_cash_equity_short"] = True
        issues.append(
            _build_issue(
                symbol=None,
                date=None,
                issue_code=EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG,
                metadata={"run_id": run.run_id, "mode": run.mode},
            )
        )

    issues = _dedupe_issues(issues)
    blocked_buy_count = sum(record.blocked_buy_count for record in daily_records)
    blocked_sell_count = sum(record.blocked_sell_count for record in daily_records)
    partial_fill_count = sum(record.partial_fill_count for record in daily_records)
    missing_data_count = sum(record.missing_data_count for record in daily_records)
    original_returns = [record.after_cost_return for record in run.daily_records]
    simulated_returns = [record.simulated_after_cost_return for record in daily_records]
    annualized_original = annualized_return_from_daily(original_returns)
    annualized_simulated = annualized_return_from_daily(simulated_returns)
    original_sharpe = sharpe_from_daily(original_returns)
    simulated_sharpe = sharpe_from_daily(simulated_returns)
    if blocked_buy_count > 0 or blocked_sell_count > 0:
        verdict = EXECUTION_COST_SIMULATION_FAIL
    elif any(issue.severity == EXECUTION_COST_ISSUE_WARNING for issue in issues):
        verdict = EXECUTION_COST_SIMULATION_WARN
    else:
        verdict = EXECUTION_COST_SIMULATION_PASS

    return FactorExecutionCostSimulationReport(
        report_id=make_execution_cost_report_id(
            backtest_run_id=run.run_id,
            weight_matrix_id=run.weight_matrix.weights_id,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        factor_matrix_id=run.factor_matrix_id,
        backtest_run_id=run.run_id,
        weight_matrix_id=run.weight_matrix.weights_id,
        tradability_mask_id=tradability_mask.mask_id if tradability_mask is not None else None,
        config=resolved_config,
        original_sample_days=len([value for value in original_returns if value is not None]),
        simulated_sample_days=len([value for value in simulated_returns if value is not None]),
        average_turnover=_mean_optional([record.turnover for record in daily_records]),
        average_cost_return=_mean_optional(
            [record.simulated_cost_return for record in daily_records]
        ),
        average_penalty_return=_mean_optional(
            [record.simulated_penalty_return for record in daily_records]
        ),
        cumulative_original_return=cumulative_return(original_returns),
        cumulative_simulated_return=cumulative_return(simulated_returns),
        annualized_original_return=annualized_original,
        annualized_simulated_return=annualized_simulated,
        original_sharpe=original_sharpe,
        simulated_sharpe=simulated_sharpe,
        original_max_drawdown=max_drawdown_from_returns(original_returns),
        simulated_max_drawdown=max_drawdown_from_returns(simulated_returns),
        cost_drag_annualized_return=(
            None
            if annualized_original is None or annualized_simulated is None
            else annualized_original - annualized_simulated
        ),
        cost_drag_sharpe=(
            None
            if original_sharpe is None or simulated_sharpe is None
            else original_sharpe - simulated_sharpe
        ),
        blocked_buy_count=blocked_buy_count,
        blocked_sell_count=blocked_sell_count,
        partial_fill_count=partial_fill_count,
        missing_data_count=missing_data_count,
        issue_count=len(issues),
        blocker_count=sum(
            1 for issue in issues if issue.severity == EXECUTION_COST_ISSUE_BLOCKER
        ),
        warning_count=sum(
            1 for issue in issues if issue.severity == EXECUTION_COST_ISSUE_WARNING
        ),
        info_count=sum(1 for issue in issues if issue.severity == EXECUTION_COST_ISSUE_INFO),
        daily_records=daily_records,
        symbol_records=symbol_records,
        issues=issues,
        verdict=verdict,
        metadata=report_metadata,
    )


def build_execution_adjusted_backtest_run(
    report: FactorExecutionCostSimulationReport,
    *,
    source_backtest_run_id: str,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionAdjustedBacktestRun:
    return ExecutionAdjustedBacktestRun(
        adjusted_run_id=make_execution_adjusted_run_id(
            source_backtest_run_id=source_backtest_run_id,
            cost_report_id=report.report_id,
        ),
        source_backtest_run_id=source_backtest_run_id,
        cost_report_id=report.report_id,
        generated_at=generated_at,
        original_daily_returns={
            record.date: record.original_after_cost_return for record in report.daily_records
        },
        simulated_daily_returns={
            record.date: record.simulated_after_cost_return for record in report.daily_records
        },
        cost_returns_by_date={
            record.date: record.simulated_cost_return for record in report.daily_records
        },
        penalty_returns_by_date={
            record.date: record.simulated_penalty_return for record in report.daily_records
        },
        metadata={
            **_coerce_metadata(metadata),
            "separate_execution_cost_artifact": True,
            "no_original_backtest_mutation": True,
        },
    )


def _format_optional(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_execution_cost_report_markdown(report: FactorExecutionCostSimulationReport) -> str:
    lines: list[str] = [
        "# Offline Execution Cost and Penalty Simulation",
        "",
        f"Generated at: `{report.generated_at}`",
        "",
        f"Verdict: `{report.verdict}`",
        "",
        "## Config summary",
        "",
        "| Field | Value |",
        "| --- | ---: |",
        f"| Market | `{report.config.market}` |",
        f"| Commission bps | {report.config.commission_bps:.4f} |",
        f"| Stamp tax bps | {report.config.stamp_tax_bps:.4f} |",
        f"| Exchange fee bps | {report.config.exchange_fee_bps:.4f} |",
        f"| Slippage bps | {report.config.slippage_bps:.4f} |",
        f"| Spread bps | {report.config.spread_bps:.4f} |",
        f"| Impact model | `{report.config.impact_model}` |",
        f"| Impact coefficient | {report.config.impact_coefficient:.4f} |",
        f"| Max participation rate | {report.config.max_participation_rate:.4f} |",
        f"| Penalty policy | `{report.config.penalty_policy}` |",
        "",
        "## Original vs simulated performance",
        "",
        "| Metric | Original | Simulated |",
        "| --- | ---: | ---: |",
        (
            f"| Cumulative return | {_format_optional(report.cumulative_original_return)} | "
            f"{_format_optional(report.cumulative_simulated_return)} |"
        ),
        (
            f"| Annualized return | {_format_optional(report.annualized_original_return)} | "
            f"{_format_optional(report.annualized_simulated_return)} |"
        ),
        (
            f"| Sharpe | {_format_optional(report.original_sharpe)} | "
            f"{_format_optional(report.simulated_sharpe)} |"
        ),
        (
            f"| Max drawdown | {_format_optional(report.original_max_drawdown)} | "
            f"{_format_optional(report.simulated_max_drawdown)} |"
        ),
        "",
        "## Cost breakdown",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average turnover | {_format_optional(report.average_turnover)} |",
        f"| Average simulated cost return | {_format_optional(report.average_cost_return)} |",
        f"| Average simulated penalty return | {_format_optional(report.average_penalty_return)} |",
        f"| Annualized return drag | {_format_optional(report.cost_drag_annualized_return)} |",
        f"| Sharpe drag | {_format_optional(report.cost_drag_sharpe)} |",
        "",
        "## Blocked / partial / missing data counts",
        "",
        "| Count | Value |",
        "| --- | ---: |",
        f"| Blocked buys | {report.blocked_buy_count} |",
        f"| Blocked sells | {report.blocked_sell_count} |",
        f"| Partial fills | {report.partial_fill_count} |",
        f"| Missing data | {report.missing_data_count} |",
        "",
        "## Issue table",
        "",
        "| Severity | Code | Symbol | Date | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    if report.issues:
        for issue in report.issues:
            lines.append(
                f"| {issue.severity} | `{issue.issue_code}` | "
                f"{issue.symbol or ''} | {issue.date or ''} | {issue.message} |"
            )
    else:
        lines.append("| none |  |  |  | No execution cost issues. |")
    lines.extend([
        "",
        "## Daily sample table",
        "",
        "| Date | Gross | Simulated cost | Penalty | Simulated after cost | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for record in report.daily_records[:10]:
        lines.append(
            f"| {record.date} | {_format_optional(record.gross_return)} | "
            f"{_format_optional(record.simulated_cost_return)} | "
            f"{_format_optional(record.simulated_penalty_return)} | "
            f"{_format_optional(record.simulated_after_cost_return)} | {record.status} |"
        )
    lines.extend([
        "",
        "## Non-runtime-impact note",
        "",
        EXECUTION_COST_NON_RUNTIME_IMPACT_NOTE,
        "",
    ])
    return "\n".join(lines)


def build_execution_cost_dashboard_payload(
    report: FactorExecutionCostSimulationReport,
) -> dict[str, Any]:
    payload = {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "verdict": report.verdict,
        "generated_at": report.generated_at,
        "metrics": {
            "cumulative_original_return": report.cumulative_original_return,
            "cumulative_simulated_return": report.cumulative_simulated_return,
            "annualized_original_return": report.annualized_original_return,
            "annualized_simulated_return": report.annualized_simulated_return,
            "original_sharpe": report.original_sharpe,
            "simulated_sharpe": report.simulated_sharpe,
            "original_max_drawdown": report.original_max_drawdown,
            "simulated_max_drawdown": report.simulated_max_drawdown,
            "cost_drag_annualized_return": report.cost_drag_annualized_return,
            "cost_drag_sharpe": report.cost_drag_sharpe,
        },
        "blocked_counts": {
            "blocked_buy_count": report.blocked_buy_count,
            "blocked_sell_count": report.blocked_sell_count,
            "partial_fill_count": report.partial_fill_count,
            "missing_data_count": report.missing_data_count,
        },
        "issue_counts": {
            "issue_count": report.issue_count,
            "blocker_count": report.blocker_count,
            "warning_count": report.warning_count,
            "info_count": report.info_count,
        },
        "config": report.config.to_dict(),
        "metadata": dict(_json_safe(report.metadata)),
    }
    return dict(_ensure_json_serializable(payload, "execution_cost_dashboard_payload"))


__all__ = [
    "EXECUTION_COST_SIMULATION_PASS",
    "EXECUTION_COST_SIMULATION_WARN",
    "EXECUTION_COST_SIMULATION_FAIL",
    "EXECUTION_SIMULATION_STATUS_OK",
    "EXECUTION_SIMULATION_STATUS_PARTIAL",
    "EXECUTION_SIMULATION_STATUS_BLOCKED",
    "EXECUTION_SIMULATION_STATUS_MISSING_DATA",
    "EXECUTION_COST_ISSUE_INFO",
    "EXECUTION_COST_ISSUE_WARNING",
    "EXECUTION_COST_ISSUE_BLOCKER",
    "EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST",
    "EXECUTION_COST_ISSUE_HIGH_IMPACT_COST",
    "EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST",
    "EXECUTION_COST_ISSUE_SPREAD_COST",
    "EXECUTION_COST_ISSUE_STAMP_TAX_COST",
    "EXECUTION_COST_ISSUE_BLOCKED_BUY",
    "EXECUTION_COST_ISSUE_BLOCKED_SELL",
    "EXECUTION_COST_ISSUE_PARTIAL_FILL",
    "EXECUTION_COST_ISSUE_MISSING_AMOUNT",
    "EXECUTION_COST_ISSUE_MISSING_VOLUME",
    "EXECUTION_COST_ISSUE_MISSING_PRICE",
    "EXECUTION_COST_ISSUE_LOW_CAPACITY",
    "EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG",
    "COST_MODEL_FIXED_BPS",
    "COST_MODEL_LINEAR_PARTICIPATION",
    "COST_MODEL_SQRT_IMPACT",
    "PENALTY_POLICY_BLOCK_TO_CASH",
    "PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT",
    "PENALTY_POLICY_MARK_UNEXECUTABLE_ONLY",
    "DEFAULT_FACTOR_EXECUTION_COST_DIR",
    "DEFAULT_EXECUTION_COST_REPORTS_FILENAME",
    "DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME",
    "DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME",
    "DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME",
    "DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME",
    "EXECUTION_COST_NON_RUNTIME_IMPACT_NOTE",
    "FactorExecutionCostConfig",
    "ExecutionCostIssue",
    "DailyExecutionCostRecord",
    "SymbolExecutionCostRecord",
    "FactorExecutionCostSimulationReport",
    "ExecutionAdjustedBacktestRun",
    "make_execution_cost_config_id",
    "make_execution_cost_issue_id",
    "make_daily_execution_cost_record_id",
    "make_symbol_execution_cost_record_id",
    "make_execution_cost_report_id",
    "make_execution_adjusted_run_id",
    "bps_to_decimal_return",
    "safe_float",
    "clamp_unit_interval",
    "infer_trade_direction",
    "estimate_participation_rate",
    "estimate_market_impact_bps",
    "extract_price_matrix",
    "extract_amount_matrix",
    "extract_volume_matrix",
    "simulate_executable_weights_for_day",
    "compute_symbol_execution_costs_for_day",
    "build_daily_execution_cost_records",
    "build_execution_cost_simulation_report",
    "build_execution_adjusted_backtest_run",
    "render_execution_cost_report_markdown",
    "build_execution_cost_dashboard_payload",
]
