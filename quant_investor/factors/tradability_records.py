"""A-share tradability execution and audit report records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quant_investor.factors.tradability_primitives import (
    EXECUTION_AUDIT_STATUS_BLOCKED,
    EXECUTION_AUDIT_STATUS_FEASIBLE,
    EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE,
    SUPPORTED_EXECUTION_AUDIT_STATUSES,
    SUPPORTED_TRADE_DIRECTIONS,
    SUPPORTED_TRADABILITY_AUDIT_VERDICTS,
    TRADE_DIRECTION_HOLD,
    TRADABILITY_AUDIT_PASS,
    TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
    TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
    TRADABILITY_ISSUE_BLOCKER,
    TRADABILITY_ISSUE_INFO,
    TRADABILITY_ISSUE_WARNING,
    FactorTradabilityIssue,
    _coerce_metadata,
    _ensure_json_serializable,
    _finite_float,
    _issue_sort_key,
    _json_safe,
    _non_empty_str,
    _non_negative_int,
    _optional_str,
    _record_sort_key,
    _require_bool,
    _short_hash,
    _slug,
    _sorted_issue_codes,
    _validate_supported,
)
from quant_investor.versioning import (
    FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION,
    FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
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

__all__ = [
    "ExecutionTransitionAuditRecord",
    "FactorExecutionFeasibilityReport",
    "FactorTradabilityAuditReport",
    "make_execution_transition_record_id",
    "make_execution_feasibility_report_id",
    "make_tradability_audit_report_id",
]
