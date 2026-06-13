"""Offline execution-cost simulation contracts and helper types."""

from __future__ import annotations

from quant_investor.factors import execution_cost_primitives as _primitives
from quant_investor.factors import execution_cost_records as _records

__all__ = [  # noqa: F822
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
    "TRADE_DIRECTION_BUY",
    "TRADE_DIRECTION_SELL",
    "TRADE_DIRECTION_HOLD",
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
]

_COMPAT_PRIVATE_REEXPORTS = (
    '_EPSILON',
    '_coerce_metadata',
    '_daily_record_sort_key',
    '_ensure_json_serializable',
    '_is_long_short_research_run',
    '_issue_message',
    '_issue_severity',
    '_issue_sort_key',
    '_json_safe',
    '_matrix_value_by_symbol',
    '_mean_optional',
    '_record_sort_key',
    '_sorted_issue_codes',
    '_weights_by_symbol',
)

for _name in __all__:
    if hasattr(_records, _name):
        globals()[_name] = getattr(_records, _name)
    else:
        globals()[_name] = getattr(_primitives, _name)

for _name in _COMPAT_PRIVATE_REEXPORTS:
    if hasattr(_records, _name):
        globals()[_name] = getattr(_records, _name)
    else:
        globals()[_name] = getattr(_primitives, _name)

del _name
