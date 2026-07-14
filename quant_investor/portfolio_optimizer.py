"""Offline deterministic portfolio optimizer and walk-forward contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import PORTFOLIO_OPTIMIZER_SCHEMA_VERSION


DEFAULT_PORTFOLIO_OPTIMIZER_DIR = Path("data/portfolio_optimizer")
DEFAULT_OPTIMIZED_PLANS_FILENAME = "optimized_plans.jsonl"
DEFAULT_REBALANCE_RESULTS_FILENAME = "rebalance_results.jsonl"
DEFAULT_WALK_FORWARD_RESULTS_FILENAME = "walk_forward_results.jsonl"

PLAN_STATUS_OPTIMIZED = "optimized"
PLAN_STATUS_EMPTY = "empty"
PLAN_STATUS_INFEASIBLE = "infeasible"

CONSTRAINT_BLOCKED_SYMBOL = "blocked_symbol"
CONSTRAINT_MIN_EDGE = "min_edge"
CONSTRAINT_MAX_WEIGHT = "max_weight"
CONSTRAINT_GROSS_EXPOSURE = "gross_exposure"
CONSTRAINT_SECTOR_CAP = "sector_cap"
CONSTRAINT_TURNOVER_CAP = "turnover_cap"
CONSTRAINT_MAX_NAMES = "max_names"
CONSTRAINT_RISK_SCORE = "risk_score"

VIOLATION_INFO = "info"
VIOLATION_WARNING = "warning"
VIOLATION_BLOCKER = "blocker"

_VALID_PLAN_STATUSES = {
    PLAN_STATUS_OPTIMIZED,
    PLAN_STATUS_EMPTY,
    PLAN_STATUS_INFEASIBLE,
}
_VALID_VIOLATION_SEVERITIES = {
    VIOLATION_INFO,
    VIOLATION_WARNING,
    VIOLATION_BLOCKER,
}
_EPSILON = 1e-12
_UNKNOWN_SECTOR = "UNKNOWN"

def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
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
        json.dumps(safe, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _contains_overlay_provenance(value: Any) -> bool:
    if isinstance(value, Mapping):
        if (
            "calibrated_posterior_overlay" in value
            or "posterior_overlay_schema_version" in value
            or value.get("overlay_mode") == "shadow"
            or value.get("source_type") == "posterior_overlay"
        ):
            return True
        eligible = value.get(
            "production_eligible",
            value.get("eligible"),
        )
        if (
            value.get("report_only") is True
            and eligible is False
        ):
            return True
        return any(
            _contains_overlay_provenance(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_overlay_provenance(item) for item in value)
    return False


def _reject_overlay_provenance(value: Any, *, context: str) -> None:
    if _contains_overlay_provenance(value):
        raise ValueError(
            f"posterior overlay provenance is forbidden in {context}"
        )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values})


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps([_json_safe(part) for part in parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    validate_finite_number(number, field_name=field_name)
    return number


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    validate_finite_number(number, field_name=field_name)
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    number = _float_or_none(value, field_name)
    if number is not None and number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    number = _float_or_none(value, field_name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _finite_float_dict(value: Mapping[str, Any] | None, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    result: dict[str, float] = {}
    for key in sorted(value, key=str):
        result[str(key)] = _finite_float(value[key], f"{field_name}.{key}")
    return result


def _clean_weight_dict(value: Mapping[str, float], *, keep_zero: bool = False) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol in sorted(value):
        weight = _finite_float(value[symbol], f"weights.{symbol}")
        if keep_zero or abs(weight) > _EPSILON:
            result[str(symbol)] = 0.0 if abs(weight) <= _EPSILON else weight
    return result


def _candidate_adjusted_score(candidate: "OptimizationCandidate", config: "PortfolioOptimizerConfig") -> float:
    risk_multiplier = max(0.0, 1.0 - config.risk_score_penalty_weight * candidate.risk_score)
    return candidate.edge_after_costs * max(candidate.confidence, 0.0) * risk_multiplier


def _make_violation(
    *,
    symbol: str | None,
    constraint_type: str,
    severity: str,
    value: float | None,
    limit: float | None,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> "ConstraintViolation":
    return ConstraintViolation(
        violation_id=make_constraint_violation_id(
            symbol=symbol,
            constraint_type=constraint_type,
            message=message,
        ),
        symbol=symbol,
        constraint_type=constraint_type,
        severity=severity,
        value=value,
        limit=limit,
        message=message,
        metadata=_coerce_metadata(metadata),
    )


def _sort_violations(violations: Sequence["ConstraintViolation"]) -> list["ConstraintViolation"]:
    return sorted(
        violations,
        key=lambda violation: (
            violation.symbol or "",
            violation.constraint_type,
            violation.severity,
            violation.message,
            violation.violation_id,
        ),
    )


def clamp_unit_interval(value: float) -> float:
    validate_finite_number(value, field_name="value")
    return max(0.0, min(1.0, float(value)))


def validate_finite_number(value: float, *, field_name: str) -> None:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")


def bps_to_decimal_return(value_bps: float) -> float:
    bps = _non_negative_float(value_bps, "value_bps")
    return bps / 10000.0


def estimate_turnover(current_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> float:
    symbols = sorted(set(current_weights) | set(target_weights))
    turnover = 0.0
    for symbol in symbols:
        current = _finite_float(current_weights.get(symbol, 0.0), f"current_weights.{symbol}")
        target = _finite_float(target_weights.get(symbol, 0.0), f"target_weights.{symbol}")
        turnover += abs(target - current)
    return turnover


def compute_sector_weights(
    target_weights: Mapping[str, float],
    sector_by_symbol: Mapping[str, str | None],
) -> dict[str, float]:
    sector_weights: dict[str, float] = {}
    for symbol in sorted(target_weights):
        weight = _finite_float(target_weights[symbol], f"target_weights.{symbol}")
        sector = sector_by_symbol.get(symbol) or _UNKNOWN_SECTOR
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return {sector: sector_weights[sector] for sector in sorted(sector_weights)}


def compound_returns(returns: Sequence[float]) -> float:
    equity = 1.0
    for index, value in enumerate(returns):
        period_return = _finite_float(value, f"returns.{index}")
        equity *= 1.0 + period_return
    return equity - 1.0


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for index, value in enumerate(returns):
        period_return = _finite_float(value, f"returns.{index}")
        equity *= 1.0 + period_return
        if equity > peak:
            peak = equity
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max(0.0, max_drawdown)


def make_constraint_violation_id(*, symbol: str | None, constraint_type: str, message: str) -> str:
    return f"opt-violation-{_slug(symbol)}-{_slug(constraint_type)}-{_short_hash([symbol, constraint_type, message])}"


def make_plan_id(
    *,
    market: str,
    as_of: str,
    symbols: Sequence[str],
    config_hash: str | None = None,
) -> str:
    ordered_symbols = sorted(str(symbol) for symbol in symbols)
    parts = [market, as_of, ordered_symbols, config_hash or ""]
    return f"portfolio-plan-{_slug(market)}-{_slug(as_of)}-{len(ordered_symbols)}-{_short_hash(parts)}"


def make_rebalance_id(*, plan_id: str, evaluation_date: str) -> str:
    return f"rebalance-{_slug(evaluation_date)}-{_short_hash([plan_id, evaluation_date])}"


def make_walk_forward_run_id(
    *,
    market: str,
    start_date: str,
    end_date: str,
    rebalance_dates: Sequence[str],
) -> str:
    ordered_dates = [str(date) for date in rebalance_dates]
    return f"walk-forward-{_slug(market)}-{_slug(start_date)}-{_slug(end_date)}-{len(ordered_dates)}-{_short_hash([market, start_date, end_date, ordered_dates])}"


@dataclass
class PortfolioOptimizerConfig:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    max_weight: float = 0.10
    min_weight: float = 0.0
    gross_exposure_cap: float = 1.0
    sector_cap: float | None = 0.30
    turnover_cap: float | None = 0.30
    max_names: int | None = None
    min_edge_after_costs: float = 0.0
    max_risk_score: float | None = 0.80
    cash_buffer: float = 0.0
    allow_short: bool = False
    force_exit_blocked_symbols: bool = True
    risk_score_penalty_weight: float = 0.50
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_weight = _finite_float(self.max_weight, "max_weight")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")
        self.min_weight = _finite_float(self.min_weight, "min_weight")
        if not 0.0 <= self.min_weight <= self.max_weight:
            raise ValueError("min_weight must be in [0, max_weight].")
        self.gross_exposure_cap = _finite_float(self.gross_exposure_cap, "gross_exposure_cap")
        if not 0.0 < self.gross_exposure_cap <= 1.0:
            raise ValueError("gross_exposure_cap must be in (0, 1].")
        self.sector_cap = _unit_float_or_none(self.sector_cap, "sector_cap")
        if self.sector_cap is not None and self.sector_cap <= 0.0:
            raise ValueError("sector_cap must be in (0, 1] when provided.")
        self.turnover_cap = _non_negative_float_or_none(self.turnover_cap, "turnover_cap")
        if self.max_names is not None:
            self.max_names = int(self.max_names)
            if self.max_names <= 0:
                raise ValueError("max_names must be positive when provided.")
        self.min_edge_after_costs = _finite_float(self.min_edge_after_costs, "min_edge_after_costs")
        self.max_risk_score = _unit_float_or_none(self.max_risk_score, "max_risk_score")
        self.cash_buffer = _finite_float(self.cash_buffer, "cash_buffer")
        if not 0.0 <= self.cash_buffer < self.gross_exposure_cap:
            raise ValueError("cash_buffer must be in [0, gross_exposure_cap).")
        if self.allow_short:
            raise ValueError("allow_short=True is not implemented in Phase 7; only long-only mode is supported.")
        self.risk_score_penalty_weight = _non_negative_float(
            self.risk_score_penalty_weight,
            "risk_score_penalty_weight",
        )
        self.transaction_cost_bps = _non_negative_float(self.transaction_cost_bps, "transaction_cost_bps")
        self.slippage_bps = _non_negative_float(self.slippage_bps, "slippage_bps")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioOptimizerConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            max_weight=float(data.get("max_weight", 0.10)),
            min_weight=float(data.get("min_weight", 0.0)),
            gross_exposure_cap=float(data.get("gross_exposure_cap", 1.0)),
            sector_cap=data.get("sector_cap", 0.30),
            turnover_cap=data.get("turnover_cap", 0.30),
            max_names=data.get("max_names"),
            min_edge_after_costs=float(data.get("min_edge_after_costs", 0.0)),
            max_risk_score=data.get("max_risk_score", 0.80),
            cash_buffer=float(data.get("cash_buffer", 0.0)),
            allow_short=bool(data.get("allow_short", False)),
            force_exit_blocked_symbols=bool(data.get("force_exit_blocked_symbols", True)),
            risk_score_penalty_weight=float(data.get("risk_score_penalty_weight", 0.50)),
            transaction_cost_bps=float(data.get("transaction_cost_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class OptimizationCandidate:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    company_name: str = ""
    sector: str | None = None
    current_weight: float = 0.0
    max_weight: float | None = None
    expected_alpha: float = 0.0
    edge_after_costs: float = 0.0
    confidence: float = 0.0
    action_score: float = 0.0
    risk_score: float = 0.0
    liquidity_score: float | None = None
    estimated_transaction_cost_bps: float | None = None
    estimated_slippage_bps: float | None = None
    estimated_market_impact_bps: float | None = None
    is_blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol)
        self.market = str(self.market)
        self.as_of = str(self.as_of)
        self.company_name = str(self.company_name)
        self.sector = None if self.sector is None else str(self.sector)
        self.current_weight = _finite_float(self.current_weight, "current_weight")
        self.max_weight = _non_negative_float_or_none(self.max_weight, "max_weight")
        self.expected_alpha = _finite_float(self.expected_alpha, "expected_alpha")
        self.edge_after_costs = _finite_float(self.edge_after_costs, "edge_after_costs")
        self.confidence = clamp_unit_interval(float(self.confidence))
        self.action_score = _finite_float(self.action_score, "action_score")
        self.risk_score = clamp_unit_interval(float(self.risk_score))
        self.liquidity_score = _unit_float_or_none(self.liquidity_score, "liquidity_score")
        self.estimated_transaction_cost_bps = _non_negative_float_or_none(
            self.estimated_transaction_cost_bps,
            "estimated_transaction_cost_bps",
        )
        self.estimated_slippage_bps = _non_negative_float_or_none(
            self.estimated_slippage_bps,
            "estimated_slippage_bps",
        )
        self.estimated_market_impact_bps = _non_negative_float_or_none(
            self.estimated_market_impact_bps,
            "estimated_market_impact_bps",
        )
        self.block_reasons = _ordered_unique(self.block_reasons)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizationCandidate":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            company_name=str(data.get("company_name", "")),
            sector=data.get("sector"),
            current_weight=float(data.get("current_weight", 0.0)),
            max_weight=data.get("max_weight"),
            expected_alpha=float(data.get("expected_alpha", 0.0)),
            edge_after_costs=float(data.get("edge_after_costs", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            action_score=float(data.get("action_score", 0.0)),
            risk_score=float(data.get("risk_score", 0.0)),
            liquidity_score=data.get("liquidity_score"),
            estimated_transaction_cost_bps=data.get("estimated_transaction_cost_bps"),
            estimated_slippage_bps=data.get("estimated_slippage_bps"),
            estimated_market_impact_bps=data.get("estimated_market_impact_bps"),
            is_blocked=bool(data.get("is_blocked", False)),
            block_reasons=list(data.get("block_reasons", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ConstraintViolation:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    violation_id: str = ""
    symbol: str | None = None
    constraint_type: str = ""
    severity: str = VIOLATION_WARNING
    value: float | None = None
    limit: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_VIOLATION_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_VALID_VIOLATION_SEVERITIES)}; got {self.severity!r}.")
        self.symbol = None if self.symbol is None else str(self.symbol)
        self.value = _float_or_none(self.value, "value")
        self.limit = _float_or_none(self.limit, "limit")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstraintViolation":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            violation_id=str(data.get("violation_id", "")),
            symbol=data.get("symbol"),
            constraint_type=str(data.get("constraint_type", "")),
            severity=str(data.get("severity", VIOLATION_WARNING)),
            value=data.get("value"),
            limit=data.get("limit"),
            message=str(data.get("message", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class OptimizedPortfolioPlan:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    plan_id: str = ""
    as_of: str = ""
    market: str = ""
    status: str = PLAN_STATUS_EMPTY
    objective_value: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    trade_weights: dict[str, float] = field(default_factory=dict)
    selected_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    cash_weight: float = 1.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    turnover_estimate: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    violations: list[ConstraintViolation] = field(default_factory=list)
    candidate_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_PLAN_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_PLAN_STATUSES)}; got {self.status!r}.")
        self.objective_value = _finite_float(self.objective_value, "objective_value")
        self.target_weights = _clean_weight_dict(self.target_weights)
        self.current_weights = _clean_weight_dict(self.current_weights, keep_zero=True)
        self.trade_weights = _clean_weight_dict(self.trade_weights)
        if self.target_weights:
            self.selected_symbols = sorted(
                self.target_weights,
                key=lambda symbol: (-self.target_weights[symbol], symbol),
            )
        else:
            self.selected_symbols = sorted({str(symbol) for symbol in self.selected_symbols})
        self.blocked_symbols = _ordered_unique(self.blocked_symbols)
        self.rejected_symbols = _ordered_unique(self.rejected_symbols)
        self.cash_weight = _non_negative_float(self.cash_weight, "cash_weight")
        self.gross_exposure = _non_negative_float(self.gross_exposure, "gross_exposure")
        self.net_exposure = _finite_float(self.net_exposure, "net_exposure")
        self.long_exposure = _non_negative_float(self.long_exposure, "long_exposure")
        self.turnover_estimate = _non_negative_float(self.turnover_estimate, "turnover_estimate")
        self.sector_weights = _finite_float_dict(self.sector_weights, "sector_weights")
        self.violations = [
            violation if isinstance(violation, ConstraintViolation) else ConstraintViolation.from_dict(violation)
            for violation in self.violations
        ]
        self.violations = _sort_violations(self.violations)
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative.")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["violations"] = [violation.to_dict() for violation in self.violations]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizedPortfolioPlan":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            plan_id=str(data.get("plan_id", "")),
            as_of=str(data.get("as_of", "")),
            market=str(data.get("market", "")),
            status=str(data.get("status", PLAN_STATUS_EMPTY)),
            objective_value=float(data.get("objective_value", 0.0)),
            target_weights=dict(data.get("target_weights", {}) or {}),
            current_weights=dict(data.get("current_weights", {}) or {}),
            trade_weights=dict(data.get("trade_weights", {}) or {}),
            selected_symbols=list(data.get("selected_symbols", []) or []),
            blocked_symbols=list(data.get("blocked_symbols", []) or []),
            rejected_symbols=list(data.get("rejected_symbols", []) or []),
            cash_weight=float(data.get("cash_weight", 1.0)),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            long_exposure=float(data.get("long_exposure", 0.0)),
            turnover_estimate=float(data.get("turnover_estimate", 0.0)),
            sector_weights=dict(data.get("sector_weights", {}) or {}),
            violations=[
                ConstraintViolation.from_dict(violation)
                for violation in list(data.get("violations", []) or [])
                if isinstance(violation, Mapping)
            ],
            candidate_count=int(data.get("candidate_count", 0) or 0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class RebalanceInput:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    as_of: str = ""
    evaluation_date: str = ""
    market: str = ""
    candidates: list[OptimizationCandidate] = field(default_factory=list)
    forward_returns: dict[str, float] = field(default_factory=dict)
    benchmark_return: float | None = None
    current_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidates = [
            candidate if isinstance(candidate, OptimizationCandidate) else OptimizationCandidate.from_dict(candidate)
            for candidate in self.candidates
        ]
        self.forward_returns = _finite_float_dict(self.forward_returns, "forward_returns")
        self.benchmark_return = _float_or_none(self.benchmark_return, "benchmark_return")
        self.current_weights = _finite_float_dict(self.current_weights, "current_weights")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "evaluation_date": self.evaluation_date,
            "market": self.market,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "forward_returns": dict(self.forward_returns),
            "benchmark_return": self.benchmark_return,
            "current_weights": dict(self.current_weights),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RebalanceInput":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            as_of=str(data.get("as_of", "")),
            evaluation_date=str(data.get("evaluation_date", "")),
            market=str(data.get("market", "")),
            candidates=[
                OptimizationCandidate.from_dict(candidate)
                for candidate in list(data.get("candidates", []) or [])
                if isinstance(candidate, Mapping)
            ],
            forward_returns=dict(data.get("forward_returns", {}) or {}),
            benchmark_return=data.get("benchmark_return"),
            current_weights=dict(data.get("current_weights", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class RebalanceResult:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    rebalance_id: str = ""
    as_of: str = ""
    evaluation_date: str = ""
    market: str = ""
    plan_id: str = ""
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    realized_gross_return: float = 0.0
    estimated_cost_return: float = 0.0
    realized_net_return: float = 0.0
    benchmark_return: float | None = None
    excess_return: float | None = None
    turnover_estimate: float = 0.0
    selected_symbols: list[str] = field(default_factory=list)
    missing_return_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_weights = _clean_weight_dict(self.target_weights)
        self.current_weights = _clean_weight_dict(self.current_weights, keep_zero=True)
        self.realized_gross_return = _finite_float(self.realized_gross_return, "realized_gross_return")
        self.estimated_cost_return = _non_negative_float(self.estimated_cost_return, "estimated_cost_return")
        self.realized_net_return = _finite_float(self.realized_net_return, "realized_net_return")
        self.benchmark_return = _float_or_none(self.benchmark_return, "benchmark_return")
        self.excess_return = _float_or_none(self.excess_return, "excess_return")
        self.turnover_estimate = _non_negative_float(self.turnover_estimate, "turnover_estimate")
        self.selected_symbols = _ordered_unique(self.selected_symbols)
        self.missing_return_symbols = _ordered_unique(self.missing_return_symbols)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RebalanceResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            rebalance_id=str(data.get("rebalance_id", "")),
            as_of=str(data.get("as_of", "")),
            evaluation_date=str(data.get("evaluation_date", "")),
            market=str(data.get("market", "")),
            plan_id=str(data.get("plan_id", "")),
            target_weights=dict(data.get("target_weights", {}) or {}),
            current_weights=dict(data.get("current_weights", {}) or {}),
            realized_gross_return=float(data.get("realized_gross_return", 0.0)),
            estimated_cost_return=float(data.get("estimated_cost_return", 0.0)),
            realized_net_return=float(data.get("realized_net_return", 0.0)),
            benchmark_return=data.get("benchmark_return"),
            excess_return=data.get("excess_return"),
            turnover_estimate=float(data.get("turnover_estimate", 0.0)),
            selected_symbols=list(data.get("selected_symbols", []) or []),
            missing_return_symbols=list(data.get("missing_return_symbols", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class WalkForwardResult:
    schema_version: str = PORTFOLIO_OPTIMIZER_SCHEMA_VERSION
    run_id: str = ""
    market: str = ""
    start_date: str = ""
    end_date: str = ""
    rebalance_count: int = 0
    cumulative_gross_return: float = 0.0
    cumulative_net_return: float = 0.0
    cumulative_benchmark_return: float | None = None
    cumulative_excess_return: float | None = None
    annualized_net_return: float | None = None
    max_drawdown: float = 0.0
    average_turnover: float = 0.0
    total_estimated_cost_return: float = 0.0
    plans: list[OptimizedPortfolioPlan] = field(default_factory=list)
    rebalance_results: list[RebalanceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rebalance_count < 0:
            raise ValueError("rebalance_count must be non-negative.")
        self.cumulative_gross_return = _finite_float(self.cumulative_gross_return, "cumulative_gross_return")
        self.cumulative_net_return = _finite_float(self.cumulative_net_return, "cumulative_net_return")
        self.cumulative_benchmark_return = _float_or_none(
            self.cumulative_benchmark_return,
            "cumulative_benchmark_return",
        )
        self.cumulative_excess_return = _float_or_none(
            self.cumulative_excess_return,
            "cumulative_excess_return",
        )
        self.annualized_net_return = _float_or_none(self.annualized_net_return, "annualized_net_return")
        self.max_drawdown = _non_negative_float(self.max_drawdown, "max_drawdown")
        self.average_turnover = _non_negative_float(self.average_turnover, "average_turnover")
        self.total_estimated_cost_return = _non_negative_float(
            self.total_estimated_cost_return,
            "total_estimated_cost_return",
        )
        self.plans = [
            plan if isinstance(plan, OptimizedPortfolioPlan) else OptimizedPortfolioPlan.from_dict(plan)
            for plan in self.plans
        ]
        self.rebalance_results = [
            result if isinstance(result, RebalanceResult) else RebalanceResult.from_dict(result)
            for result in self.rebalance_results
        ]
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "market": self.market,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_count": self.rebalance_count,
            "cumulative_gross_return": self.cumulative_gross_return,
            "cumulative_net_return": self.cumulative_net_return,
            "cumulative_benchmark_return": self.cumulative_benchmark_return,
            "cumulative_excess_return": self.cumulative_excess_return,
            "annualized_net_return": self.annualized_net_return,
            "max_drawdown": self.max_drawdown,
            "average_turnover": self.average_turnover,
            "total_estimated_cost_return": self.total_estimated_cost_return,
            "plans": [plan.to_dict() for plan in self.plans],
            "rebalance_results": [result.to_dict() for result in self.rebalance_results],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WalkForwardResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", PORTFOLIO_OPTIMIZER_SCHEMA_VERSION)),
            run_id=str(data.get("run_id", "")),
            market=str(data.get("market", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            rebalance_count=int(data.get("rebalance_count", 0) or 0),
            cumulative_gross_return=float(data.get("cumulative_gross_return", 0.0)),
            cumulative_net_return=float(data.get("cumulative_net_return", 0.0)),
            cumulative_benchmark_return=data.get("cumulative_benchmark_return"),
            cumulative_excess_return=data.get("cumulative_excess_return"),
            annualized_net_return=data.get("annualized_net_return"),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            average_turnover=float(data.get("average_turnover", 0.0)),
            total_estimated_cost_return=float(data.get("total_estimated_cost_return", 0.0)),
            plans=[
                OptimizedPortfolioPlan.from_dict(plan)
                for plan in list(data.get("plans", []) or [])
                if isinstance(plan, Mapping)
            ],
            rebalance_results=[
                RebalanceResult.from_dict(result)
                for result in list(data.get("rebalance_results", []) or [])
                if isinstance(result, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def build_candidate_from_overlay(
    overlay: Any,
    *,
    risk_tensor: Any | None = None,
    current_weight: float = 0.0,
    default_max_weight: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizationCandidate:
    raise ValueError(
        "posterior overlay bridge is disabled: overlays are report-only"
    )


def build_candidates_from_overlays(
    overlays: Sequence[Any],
    *,
    risk_tensors_by_symbol: Mapping[str, Any] | None = None,
    current_weights: Mapping[str, float] | None = None,
    default_max_weight: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[OptimizationCandidate]:
    if len(overlays) == 0:
        return []
    raise ValueError(
        "posterior overlay bridge is disabled: overlays are report-only"
    )


def _allocate_greedy(
    eligible: Sequence[tuple[OptimizationCandidate, float]],
    *,
    config: PortfolioOptimizerConfig,
    gross_budget: float,
) -> tuple[dict[str, float], set[str], set[str]]:
    weights = {candidate.symbol: 0.0 for candidate, _score in eligible}
    sector_weights: dict[str, float] = {}
    remaining = [candidate for candidate, _score in eligible]
    score_by_symbol = {candidate.symbol: score for candidate, score in eligible}
    max_weight_capped: set[str] = set()
    sector_capped: set[str] = set()
    remaining_budget = gross_budget

    while remaining and remaining_budget > _EPSILON:
        total_score = sum(max(score_by_symbol[candidate.symbol], 0.0) for candidate in remaining)
        if total_score <= _EPSILON:
            break
        allocated_this_round = 0.0
        next_remaining: list[OptimizationCandidate] = []
        for candidate in remaining:
            desired = remaining_budget * max(score_by_symbol[candidate.symbol], 0.0) / total_score
            symbol_cap = min(config.max_weight, candidate.max_weight if candidate.max_weight is not None else config.max_weight)
            symbol_remaining = max(0.0, symbol_cap - weights[candidate.symbol])
            sector = candidate.sector or _UNKNOWN_SECTOR
            sector_remaining = math.inf
            if config.sector_cap is not None:
                sector_remaining = max(0.0, config.sector_cap - sector_weights.get(sector, 0.0))
            allowed = min(desired, symbol_remaining, sector_remaining, remaining_budget - allocated_this_round)
            if allowed > _EPSILON:
                weights[candidate.symbol] += allowed
                sector_weights[sector] = sector_weights.get(sector, 0.0) + allowed
                allocated_this_round += allowed
            if symbol_remaining <= desired + _EPSILON and symbol_remaining <= sector_remaining + _EPSILON:
                max_weight_capped.add(candidate.symbol)
                continue
            if sector_remaining <= desired + _EPSILON and sector_remaining <= symbol_remaining + _EPSILON:
                sector_capped.add(sector)
                continue
            if allowed > _EPSILON:
                next_remaining.append(candidate)
        remaining_budget -= allocated_this_round
        if allocated_this_round <= _EPSILON:
            break
        if len(next_remaining) == len(remaining) and remaining_budget <= _EPSILON:
            break
        remaining = next_remaining
    return _clean_weight_dict(weights), max_weight_capped, sector_capped


def optimize_portfolio(
    candidates: Sequence[OptimizationCandidate],
    *,
    config: PortfolioOptimizerConfig | None = None,
    market: str = "",
    as_of: str = "",
    current_weights: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizedPortfolioPlan:
    candidates = list(candidates)
    resolved_config = config or PortfolioOptimizerConfig()
    input_metadata = _coerce_metadata(metadata)
    _reject_overlay_provenance(
        resolved_config.metadata,
        context="optimizer config metadata",
    )
    _reject_overlay_provenance(
        input_metadata,
        context="optimizer input metadata",
    )
    for index, candidate in enumerate(candidates):
        if candidate.schema_version != PORTFOLIO_OPTIMIZER_SCHEMA_VERSION:
            raise ValueError(
                "candidate schema_version is not executable: "
                f"index={index}, schema={candidate.schema_version!r}"
            )
        _reject_overlay_provenance(
            candidate.metadata,
            context=f"candidate metadata at index {index}",
        )
    resolved_current_weights = _finite_float_dict(current_weights, "current_weights") if current_weights is not None else {}
    if not resolved_current_weights:
        resolved_current_weights = {
            candidate.symbol: candidate.current_weight
            for candidate in candidates
            if abs(candidate.current_weight) > _EPSILON
        }

    violations: list[ConstraintViolation] = []
    blocked_symbols: list[str] = []
    rejected_symbols: list[str] = []
    eligible: list[tuple[OptimizationCandidate, float]] = []
    sector_by_symbol = {candidate.symbol: candidate.sector for candidate in candidates}

    for candidate in candidates:
        if candidate.is_blocked:
            blocked_symbols.append(candidate.symbol)
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_BLOCKED_SYMBOL,
                    severity=VIOLATION_BLOCKER if resolved_config.force_exit_blocked_symbols else VIOLATION_INFO,
                    value=None,
                    limit=None,
                    message="Candidate blocked by execution or risk tensor constraints.",
                    metadata={"block_reasons": candidate.block_reasons},
                )
            )
            continue
        if candidate.edge_after_costs < resolved_config.min_edge_after_costs:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_MIN_EDGE,
                    severity=VIOLATION_INFO,
                    value=candidate.edge_after_costs,
                    limit=resolved_config.min_edge_after_costs,
                    message="Candidate edge after costs is below the configured minimum.",
                )
            )
            continue
        if resolved_config.max_risk_score is not None and candidate.risk_score > resolved_config.max_risk_score:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_RISK_SCORE,
                    severity=VIOLATION_WARNING,
                    value=candidate.risk_score,
                    limit=resolved_config.max_risk_score,
                    message="Candidate risk score is above the configured maximum.",
                )
            )
            continue
        score = _candidate_adjusted_score(candidate, resolved_config)
        eligible.append((candidate, score))

    eligible.sort(
        key=lambda item: (
            -item[1],
            -item[0].edge_after_costs,
            -item[0].action_score,
            item[0].symbol,
        )
    )
    if resolved_config.max_names is not None and len(eligible) > resolved_config.max_names:
        for candidate, _score in eligible[resolved_config.max_names :]:
            rejected_symbols.append(candidate.symbol)
            violations.append(
                _make_violation(
                    symbol=candidate.symbol,
                    constraint_type=CONSTRAINT_MAX_NAMES,
                    severity=VIOLATION_INFO,
                    value=float(len(eligible)),
                    limit=float(resolved_config.max_names),
                    message="Candidate excluded by max_names constraint.",
                )
            )
        eligible = eligible[: resolved_config.max_names]

    gross_budget = resolved_config.gross_exposure_cap - resolved_config.cash_buffer
    positive_scores = [score for _candidate, score in eligible if score > _EPSILON]
    forced_exit_symbols = {
        symbol
        for symbol in blocked_symbols
        if resolved_config.force_exit_blocked_symbols and abs(resolved_current_weights.get(symbol, 0.0)) > _EPSILON
    }

    raw_target_weights: dict[str, float] = {}
    max_weight_capped: set[str] = set()
    sector_capped: set[str] = set()
    if positive_scores:
        raw_target_weights, max_weight_capped, sector_capped = _allocate_greedy(
            [(candidate, score) for candidate, score in eligible if score > _EPSILON],
            config=resolved_config,
            gross_budget=gross_budget,
        )
    else:
        rejected_symbols.extend(candidate.symbol for candidate, _score in eligible)

    for symbol in max_weight_capped:
        limit = next(
            (
                min(resolved_config.max_weight, candidate.max_weight if candidate.max_weight is not None else resolved_config.max_weight)
                for candidate, _score in eligible
                if candidate.symbol == symbol
            ),
            resolved_config.max_weight,
        )
        violations.append(
            _make_violation(
                symbol=symbol,
                constraint_type=CONSTRAINT_MAX_WEIGHT,
                severity=VIOLATION_WARNING,
                value=raw_target_weights.get(symbol),
                limit=limit,
                message="Candidate allocation was clipped by max weight.",
            )
        )
    for sector in sector_capped:
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_SECTOR_CAP,
                severity=VIOLATION_WARNING,
                value=None,
                limit=resolved_config.sector_cap,
                message=f"Sector allocation was clipped by cap: {sector}.",
                metadata={"sector": sector},
            )
        )

    proposed_targets = dict(raw_target_weights)
    for symbol in forced_exit_symbols:
        proposed_targets[symbol] = 0.0

    proposed_turnover = estimate_turnover(resolved_current_weights, proposed_targets)
    final_targets = dict(proposed_targets)
    if resolved_config.turnover_cap is not None and proposed_turnover > resolved_config.turnover_cap + _EPSILON:
        non_forced_symbols = sorted((set(resolved_current_weights) | set(proposed_targets)) - forced_exit_symbols)
        forced_turnover = sum(abs(resolved_current_weights.get(symbol, 0.0)) for symbol in forced_exit_symbols)
        non_forced_turnover = sum(
            abs(proposed_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0))
            for symbol in non_forced_symbols
        )
        available_turnover = max(0.0, resolved_config.turnover_cap - forced_turnover)
        scale = 0.0 if non_forced_turnover <= _EPSILON else min(1.0, available_turnover / non_forced_turnover)
        final_targets = {}
        for symbol in non_forced_symbols:
            current = resolved_current_weights.get(symbol, 0.0)
            proposed = proposed_targets.get(symbol, 0.0)
            target = current + (proposed - current) * scale
            if abs(target) > _EPSILON:
                final_targets[symbol] = target
                if symbol not in sector_by_symbol:
                    sector_by_symbol[symbol] = None
        for symbol in forced_exit_symbols:
            final_targets[symbol] = 0.0
        severity = VIOLATION_WARNING
        final_turnover_for_violation = estimate_turnover(resolved_current_weights, final_targets)
        if final_turnover_for_violation > resolved_config.turnover_cap + _EPSILON:
            severity = VIOLATION_BLOCKER
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_TURNOVER_CAP,
                severity=severity,
                value=proposed_turnover,
                limit=resolved_config.turnover_cap,
                message="Proposed turnover exceeded the configured turnover cap and was scaled toward current weights.",
                metadata={
                    "scale": scale,
                    "final_turnover": final_turnover_for_violation,
                    "forced_exit_symbols": sorted(forced_exit_symbols),
                },
            )
        )

    clean_targets = _clean_weight_dict(final_targets)
    trade_weights = {
        symbol: final_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0)
        for symbol in sorted(set(resolved_current_weights) | set(final_targets))
        if abs(final_targets.get(symbol, 0.0) - resolved_current_weights.get(symbol, 0.0)) > _EPSILON
    }
    turnover_estimate = estimate_turnover(resolved_current_weights, clean_targets)
    gross_exposure = sum(abs(weight) for weight in clean_targets.values())
    net_exposure = sum(clean_targets.values())
    long_exposure = sum(weight for weight in clean_targets.values() if weight > 0.0)
    if gross_exposure > resolved_config.gross_exposure_cap + _EPSILON:
        violations.append(
            _make_violation(
                symbol=None,
                constraint_type=CONSTRAINT_GROSS_EXPOSURE,
                severity=VIOLATION_BLOCKER,
                value=gross_exposure,
                limit=resolved_config.gross_exposure_cap,
                message="Final gross exposure exceeds configured cap.",
            )
        )
    sector_weights = compute_sector_weights(clean_targets, sector_by_symbol)
    if resolved_config.sector_cap is not None:
        for sector, weight in sector_weights.items():
            if weight > resolved_config.sector_cap + _EPSILON:
                violations.append(
                    _make_violation(
                        symbol=None,
                        constraint_type=CONSTRAINT_SECTOR_CAP,
                        severity=VIOLATION_BLOCKER,
                        value=weight,
                        limit=resolved_config.sector_cap,
                        message=f"Final sector exposure exceeds cap: {sector}.",
                        metadata={"sector": sector},
                    )
                )

    score_by_symbol = {candidate.symbol: score for candidate, score in eligible}
    objective_value = sum(clean_targets.get(symbol, 0.0) * score_by_symbol.get(symbol, 0.0) for symbol in clean_targets)
    selected_symbols = sorted(clean_targets, key=lambda symbol: (-clean_targets[symbol], symbol))
    if candidates and len(blocked_symbols) == len(candidates):
        status = PLAN_STATUS_INFEASIBLE
    elif clean_targets:
        status = PLAN_STATUS_OPTIMIZED
    elif forced_exit_symbols:
        status = PLAN_STATUS_INFEASIBLE
    else:
        status = PLAN_STATUS_EMPTY

    config_snapshot = resolved_config.to_dict()
    config_hash = _short_hash([config_snapshot])
    plan_id = make_plan_id(market=market, as_of=as_of, symbols=selected_symbols, config_hash=config_hash)
    plan_metadata = {
        "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        "config": config_snapshot,
        "optimization_method": "deterministic_greedy_v1",
        "eligible_count": len(eligible),
        "rejected_count": len(_ordered_unique(rejected_symbols)),
        "input_metadata": input_metadata,
    }
    return OptimizedPortfolioPlan(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        plan_id=plan_id,
        as_of=as_of,
        market=market,
        status=status,
        objective_value=objective_value,
        target_weights=clean_targets,
        current_weights=resolved_current_weights,
        trade_weights=trade_weights,
        selected_symbols=selected_symbols,
        blocked_symbols=blocked_symbols,
        rejected_symbols=rejected_symbols,
        cash_weight=max(0.0, 1.0 - gross_exposure),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        turnover_estimate=turnover_estimate,
        sector_weights=sector_weights,
        violations=violations,
        candidate_count=len(candidates),
        metadata=plan_metadata,
    )


def evaluate_rebalance(
    plan: OptimizedPortfolioPlan,
    *,
    evaluation_date: str,
    forward_returns: Mapping[str, float],
    benchmark_return: float | None = None,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RebalanceResult:
    supplied_returns = _finite_float_dict(forward_returns, "forward_returns")
    missing_return_symbols = sorted(symbol for symbol in plan.target_weights if symbol not in supplied_returns)
    realized_gross_return = sum(
        plan.target_weights[symbol] * supplied_returns[symbol]
        for symbol in sorted(plan.target_weights)
        if symbol in supplied_returns
    )
    config_payload = plan.metadata.get("config", {}) if isinstance(plan.metadata.get("config"), Mapping) else {}
    resolved_transaction_cost_bps = (
        _non_negative_float(transaction_cost_bps, "transaction_cost_bps")
        if transaction_cost_bps is not None
        else _non_negative_float(config_payload.get("transaction_cost_bps", 0.0), "transaction_cost_bps")
    )
    resolved_slippage_bps = (
        _non_negative_float(slippage_bps, "slippage_bps")
        if slippage_bps is not None
        else _non_negative_float(config_payload.get("slippage_bps", 0.0), "slippage_bps")
    )
    estimated_cost_return = plan.turnover_estimate * bps_to_decimal_return(
        resolved_transaction_cost_bps + resolved_slippage_bps
    )
    realized_net_return = realized_gross_return - estimated_cost_return
    resolved_benchmark = _float_or_none(benchmark_return, "benchmark_return")
    excess_return = None if resolved_benchmark is None else realized_net_return - resolved_benchmark
    result_metadata = {
        "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        "input_metadata": _coerce_metadata(metadata),
        "cost_bps": {
            "transaction_cost_bps": resolved_transaction_cost_bps,
            "slippage_bps": resolved_slippage_bps,
        },
    }
    return RebalanceResult(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        rebalance_id=make_rebalance_id(plan_id=plan.plan_id, evaluation_date=evaluation_date),
        as_of=plan.as_of,
        evaluation_date=evaluation_date,
        market=plan.market,
        plan_id=plan.plan_id,
        target_weights=plan.target_weights,
        current_weights=plan.current_weights,
        realized_gross_return=realized_gross_return,
        estimated_cost_return=estimated_cost_return,
        realized_net_return=realized_net_return,
        benchmark_return=resolved_benchmark,
        excess_return=excess_return,
        turnover_estimate=plan.turnover_estimate,
        selected_symbols=plan.selected_symbols,
        missing_return_symbols=missing_return_symbols,
        metadata=result_metadata,
    )


def run_walk_forward_loop(
    rebalance_inputs: Sequence[RebalanceInput],
    *,
    config: PortfolioOptimizerConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    if not rebalance_inputs:
        return WalkForwardResult(
            schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            run_id=make_walk_forward_run_id(market="", start_date="", end_date="", rebalance_dates=[]),
            market="",
            start_date="",
            end_date="",
            rebalance_count=0,
            cumulative_gross_return=0.0,
            cumulative_net_return=0.0,
            cumulative_benchmark_return=None,
            cumulative_excess_return=None,
            annualized_net_return=None,
            max_drawdown=0.0,
            average_turnover=0.0,
            total_estimated_cost_return=0.0,
            plans=[],
            rebalance_results=[],
            metadata={
                "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
                "input_metadata": _coerce_metadata(metadata),
            },
        )

    resolved_inputs = sorted(
        [
            rebalance_input if isinstance(rebalance_input, RebalanceInput) else RebalanceInput.from_dict(rebalance_input)
            for rebalance_input in rebalance_inputs
        ],
        key=lambda item: (item.as_of, item.evaluation_date, item.market),
    )
    plans: list[OptimizedPortfolioPlan] = []
    results: list[RebalanceResult] = []
    rolling_current_weights: dict[str, float] = {}
    resolved_config = config or PortfolioOptimizerConfig()

    for rebalance_input in resolved_inputs:
        current_weights_for_period = (
            dict(rebalance_input.current_weights)
            if rebalance_input.current_weights
            else dict(rolling_current_weights)
        )
        plan = optimize_portfolio(
            rebalance_input.candidates,
            config=resolved_config,
            market=rebalance_input.market,
            as_of=rebalance_input.as_of,
            current_weights=current_weights_for_period,
            metadata=rebalance_input.metadata,
        )
        result = evaluate_rebalance(
            plan,
            evaluation_date=rebalance_input.evaluation_date,
            forward_returns=rebalance_input.forward_returns,
            benchmark_return=rebalance_input.benchmark_return,
        )
        plans.append(plan)
        results.append(result)
        rolling_current_weights = dict(plan.target_weights)

    gross_returns = [result.realized_gross_return for result in results]
    net_returns = [result.realized_net_return for result in results]
    benchmark_returns = [result.benchmark_return for result in results]
    cumulative_benchmark_return = (
        compound_returns([float(value) for value in benchmark_returns])
        if all(value is not None for value in benchmark_returns)
        else None
    )
    cumulative_net_return = compound_returns(net_returns)
    cumulative_excess_return = (
        None if cumulative_benchmark_return is None else cumulative_net_return - cumulative_benchmark_return
    )
    annualized_net_return = None
    if len(results) > 0 and cumulative_net_return > -1.0:
        annualized_net_return = (1.0 + cumulative_net_return) ** (252.0 / len(results)) - 1.0
    market = resolved_inputs[0].market
    start_date = resolved_inputs[0].as_of
    end_date = resolved_inputs[-1].evaluation_date or resolved_inputs[-1].as_of
    rebalance_dates = [rebalance_input.as_of for rebalance_input in resolved_inputs]
    return WalkForwardResult(
        schema_version=PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        run_id=make_walk_forward_run_id(
            market=market,
            start_date=start_date,
            end_date=end_date,
            rebalance_dates=rebalance_dates,
        ),
        market=market,
        start_date=start_date,
        end_date=end_date,
        rebalance_count=len(results),
        cumulative_gross_return=compound_returns(gross_returns),
        cumulative_net_return=cumulative_net_return,
        cumulative_benchmark_return=cumulative_benchmark_return,
        cumulative_excess_return=cumulative_excess_return,
        annualized_net_return=annualized_net_return,
        max_drawdown=max_drawdown_from_returns(net_returns),
        average_turnover=sum(result.turnover_estimate for result in results) / len(results),
        total_estimated_cost_return=sum(result.estimated_cost_return for result in results),
        plans=plans,
        rebalance_results=results,
        metadata={
            "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            "input_metadata": _coerce_metadata(metadata),
            "config": resolved_config.to_dict(),
        },
    )


def build_portfolio_constructor_patch(plan: OptimizedPortfolioPlan) -> dict[str, Any]:
    if plan.schema_version != PORTFOLIO_OPTIMIZER_SCHEMA_VERSION:
        raise ValueError(
            "plan schema_version is not executable for constructor patch: "
            f"{plan.schema_version!r}"
        )
    _reject_overlay_provenance(
        plan.metadata,
        context="portfolio plan metadata",
    )
    optimization_method = None
    if isinstance(plan.metadata, Mapping):
        optimization_method = plan.metadata.get("optimization_method")
    return {
        "target_weights": dict(sorted(plan.target_weights.items())),
        "blocked_symbols": list(plan.blocked_symbols),
        "rejected_symbols": list(plan.rejected_symbols),
        "turnover_estimate": plan.turnover_estimate,
        "sector_weights": dict(sorted(plan.sector_weights.items())),
        "gross_exposure": plan.gross_exposure,
        "net_exposure": plan.net_exposure,
        "violations": [violation.to_dict() for violation in plan.violations],
        "metadata": {
            "portfolio_optimizer_schema_version": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "status": plan.status,
            "optimization_method": optimization_method,
        },
    }


class PortfolioOptimizerStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_PORTFOLIO_OPTIMIZER_DIR
        self.optimized_plans_path = self.root_dir / DEFAULT_OPTIMIZED_PLANS_FILENAME
        self.rebalance_results_path = self.root_dir / DEFAULT_REBALANCE_RESULTS_FILENAME
        self.walk_forward_results_path = self.root_dir / DEFAULT_WALK_FORWARD_RESULTS_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}.") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}: expected object.")
                rows.append(payload)
        return rows

    def append_plan(self, plan: OptimizedPortfolioPlan) -> None:
        if plan.plan_id in self.get_plan_ids():
            raise ValueError(f"Duplicate plan_id: {plan.plan_id}")
        self._append_jsonl(self.optimized_plans_path, plan.to_dict())

    def append_rebalance_result(self, result: RebalanceResult) -> None:
        if result.rebalance_id in self.get_rebalance_ids():
            raise ValueError(f"Duplicate rebalance_id: {result.rebalance_id}")
        self._append_jsonl(self.rebalance_results_path, result.to_dict())

    def append_walk_forward_result(self, result: WalkForwardResult) -> None:
        if result.run_id in self.get_walk_forward_run_ids():
            raise ValueError(f"Duplicate run_id: {result.run_id}")
        self._append_jsonl(self.walk_forward_results_path, result.to_dict())

    def read_plans(self) -> list[OptimizedPortfolioPlan]:
        return [OptimizedPortfolioPlan.from_dict(row) for row in self._read_jsonl(self.optimized_plans_path)]

    def read_rebalance_results(self) -> list[RebalanceResult]:
        return [RebalanceResult.from_dict(row) for row in self._read_jsonl(self.rebalance_results_path)]

    def read_walk_forward_results(self) -> list[WalkForwardResult]:
        return [WalkForwardResult.from_dict(row) for row in self._read_jsonl(self.walk_forward_results_path)]

    def get_plan_ids(self) -> set[str]:
        return {plan.plan_id for plan in self.read_plans()}

    def get_rebalance_ids(self) -> set[str]:
        return {result.rebalance_id for result in self.read_rebalance_results()}

    def get_walk_forward_run_ids(self) -> set[str]:
        return {result.run_id for result in self.read_walk_forward_results()}


__all__ = [
    "DEFAULT_PORTFOLIO_OPTIMIZER_DIR",
    "DEFAULT_OPTIMIZED_PLANS_FILENAME",
    "DEFAULT_REBALANCE_RESULTS_FILENAME",
    "DEFAULT_WALK_FORWARD_RESULTS_FILENAME",
    "PLAN_STATUS_OPTIMIZED",
    "PLAN_STATUS_EMPTY",
    "PLAN_STATUS_INFEASIBLE",
    "CONSTRAINT_BLOCKED_SYMBOL",
    "CONSTRAINT_MIN_EDGE",
    "CONSTRAINT_MAX_WEIGHT",
    "CONSTRAINT_GROSS_EXPOSURE",
    "CONSTRAINT_SECTOR_CAP",
    "CONSTRAINT_TURNOVER_CAP",
    "CONSTRAINT_MAX_NAMES",
    "CONSTRAINT_RISK_SCORE",
    "VIOLATION_INFO",
    "VIOLATION_WARNING",
    "VIOLATION_BLOCKER",
    "PortfolioOptimizerConfig",
    "OptimizationCandidate",
    "ConstraintViolation",
    "OptimizedPortfolioPlan",
    "RebalanceInput",
    "RebalanceResult",
    "WalkForwardResult",
    "PortfolioOptimizerStore",
    "make_constraint_violation_id",
    "make_plan_id",
    "make_rebalance_id",
    "make_walk_forward_run_id",
    "clamp_unit_interval",
    "validate_finite_number",
    "bps_to_decimal_return",
    "estimate_turnover",
    "compute_sector_weights",
    "compound_returns",
    "max_drawdown_from_returns",
    "build_candidate_from_overlay",
    "build_candidates_from_overlays",
    "optimize_portfolio",
    "evaluate_rebalance",
    "run_walk_forward_loop",
    "build_portfolio_constructor_patch",
]
