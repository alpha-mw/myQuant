"""Dataclass records and IDs for execution-cost simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from quant_investor.factors.execution_cost_primitives import (
    COST_MODEL_FIXED_BPS,
    COST_MODEL_LINEAR_PARTICIPATION,
    COST_MODEL_SQRT_IMPACT,
    EXECUTION_COST_ISSUE_BLOCKER,
    EXECUTION_COST_ISSUE_INFO,
    EXECUTION_COST_ISSUE_WARNING,
    EXECUTION_COST_SIMULATION_PASS,
    EXECUTION_SIMULATION_STATUS_OK,
    PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT,
    SUPPORTED_COST_MODELS,
    SUPPORTED_EXECUTION_COST_ISSUE_CODES,
    SUPPORTED_EXECUTION_COST_ISSUE_SEVERITIES,
    SUPPORTED_EXECUTION_COST_VERDICTS,
    SUPPORTED_EXECUTION_SIMULATION_STATUSES,
    SUPPORTED_PENALTY_POLICIES,
    SUPPORTED_TRADE_DIRECTIONS,
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_HOLD,
    TRADE_DIRECTION_SELL,
    _EPSILON,
    _coerce_metadata,
    _daily_record_sort_key,
    _finite_float,
    _issue_sort_key,
    _json_safe,
    _non_empty_str,
    _non_negative_float,
    _non_negative_int,
    _optional_finite_float,
    _optional_non_negative_float,
    _optional_str,
    _record_sort_key,
    _require_bool,
    _short_hash,
    _slug,
    _sorted_issue_codes,
    _validate_supported,
)
from quant_investor.versioning import (
    FACTOR_EXECUTION_COST_SIMULATION_SCHEMA_VERSION,
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
