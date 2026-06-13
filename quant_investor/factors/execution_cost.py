"""Offline execution-cost and execution-penalty simulation for factor backtests.
These helpers consume backtest artifacts, weight matrices, bundles, and masks.
produce separate simulated return artifacts only. They do not alter stock
selection, factor admission, posterior scoring, RiskGuard, PortfolioConstructor,
target weights, orders, providers, LLMs, broker APIs, or live execution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import SingleFactorBacktestRun
from quant_investor.factors.execution_cost_types import (
    COST_MODEL_FIXED_BPS,
    COST_MODEL_LINEAR_PARTICIPATION,
    COST_MODEL_SQRT_IMPACT,
    DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME,
    DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME,
    DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME,
    DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME,
    DEFAULT_EXECUTION_COST_REPORTS_FILENAME,
    DEFAULT_FACTOR_EXECUTION_COST_DIR,
    EXECUTION_COST_ISSUE_BLOCKED_BUY,
    EXECUTION_COST_ISSUE_BLOCKED_SELL,
    EXECUTION_COST_ISSUE_BLOCKER,
    EXECUTION_COST_ISSUE_HIGH_IMPACT_COST,
    EXECUTION_COST_ISSUE_HIGH_SLIPPAGE_COST,
    EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST,
    EXECUTION_COST_ISSUE_INFO,
    EXECUTION_COST_ISSUE_LOW_CAPACITY,
    EXECUTION_COST_ISSUE_MISSING_AMOUNT,
    EXECUTION_COST_ISSUE_MISSING_PRICE,
    EXECUTION_COST_ISSUE_MISSING_VOLUME,
    EXECUTION_COST_ISSUE_PARTIAL_FILL,
    EXECUTION_COST_ISSUE_RESEARCH_SHORT_LEG,
    EXECUTION_COST_ISSUE_SPREAD_COST,
    EXECUTION_COST_ISSUE_STAMP_TAX_COST,
    EXECUTION_COST_ISSUE_WARNING,
    EXECUTION_COST_NON_RUNTIME_IMPACT_NOTE,
    EXECUTION_COST_SIMULATION_FAIL,
    EXECUTION_COST_SIMULATION_PASS,
    EXECUTION_COST_SIMULATION_WARN,
    EXECUTION_SIMULATION_STATUS_BLOCKED,
    EXECUTION_SIMULATION_STATUS_MISSING_DATA,
    EXECUTION_SIMULATION_STATUS_OK,
    EXECUTION_SIMULATION_STATUS_PARTIAL,
    ExecutionAdjustedBacktestRun,
    ExecutionCostIssue,
    FactorExecutionCostConfig,
    FactorExecutionCostSimulationReport,
    DailyExecutionCostRecord,
    SymbolExecutionCostRecord,
    PENALTY_POLICY_BLOCK_TO_CASH,
    PENALTY_POLICY_KEEP_PREVIOUS_WEIGHT,
    PENALTY_POLICY_MARK_UNEXECUTABLE_ONLY,
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_HOLD,
    TRADE_DIRECTION_SELL,
    _EPSILON,
    _coerce_metadata,
    _daily_record_sort_key,
    _ensure_json_serializable,
    _is_long_short_research_run,
    _issue_message,
    _issue_severity,
    _issue_sort_key,
    _json_safe,
    _matrix_value_by_symbol,
    _mean_optional,
    _record_sort_key,
    _sorted_issue_codes,
    _weights_by_symbol,
    bps_to_decimal_return,
    clamp_unit_interval,
    estimate_market_impact_bps,
    estimate_participation_rate,
    infer_trade_direction,
    make_daily_execution_cost_record_id,
    make_execution_adjusted_run_id,
    make_execution_cost_config_id,
    make_execution_cost_issue_id,
    make_execution_cost_report_id,
    make_symbol_execution_cost_record_id,
    safe_float,
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
