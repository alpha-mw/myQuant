"""Primitive constants and helpers for execution-cost contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_SHORT,
    SingleFactorBacktestRun,
)

if TYPE_CHECKING:
    from quant_investor.factors.execution_cost_records import (
        DailyExecutionCostRecord,
        ExecutionCostIssue,
        SymbolExecutionCostRecord,
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
