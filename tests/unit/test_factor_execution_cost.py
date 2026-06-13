from __future__ import annotations

import importlib
import json
import math

import pytest

import quant_investor.factors.execution_cost as execution_cost
import quant_investor.factors.execution_cost_types as execution_cost_types
from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_ONLY,
    WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
)
from quant_investor.factors.execution_cost import (
    COST_MODEL_FIXED_BPS,
    COST_MODEL_LINEAR_PARTICIPATION,
    COST_MODEL_SQRT_IMPACT,
    EXECUTION_COST_ISSUE_BLOCKED_BUY,
    EXECUTION_COST_ISSUE_BLOCKED_SELL,
    EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST,
    EXECUTION_COST_ISSUE_LOW_CAPACITY,
    EXECUTION_COST_ISSUE_MISSING_AMOUNT,
    EXECUTION_COST_ISSUE_MISSING_PRICE,
    EXECUTION_COST_ISSUE_MISSING_VOLUME,
    EXECUTION_COST_ISSUE_PARTIAL_FILL,
    EXECUTION_COST_SIMULATION_FAIL,
    EXECUTION_COST_SIMULATION_WARN,
    EXECUTION_SIMULATION_STATUS_BLOCKED,
    EXECUTION_SIMULATION_STATUS_OK,
    EXECUTION_SIMULATION_STATUS_PARTIAL,
    DailyExecutionCostRecord,
    ExecutionAdjustedBacktestRun,
    FactorExecutionCostConfig,
    FactorExecutionCostSimulationReport,
    SymbolExecutionCostRecord,
    bps_to_decimal_return,
    build_daily_execution_cost_records,
    build_execution_adjusted_backtest_run,
    build_execution_cost_dashboard_payload,
    build_execution_cost_simulation_report,
    clamp_unit_interval,
    compute_symbol_execution_costs_for_day,
    estimate_market_impact_bps,
    estimate_participation_rate,
    extract_price_matrix,
    infer_trade_direction,
    make_execution_cost_config_id,
    make_execution_cost_report_id,
    render_execution_cost_report_markdown,
    simulate_executable_weights_for_day,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_VOLUME,
    FIELD_VWAP,
    MatrixDataBundle,
    MatrixDataContract,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id
from quant_investor.factors.tradability import AShareTradabilityConfig, AShareTradabilityMask


SYMBOLS = ["AAA", "BBB"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]


def _matrix(value):
    return [[value for _date in DATES] for _symbol in SYMBOLS]


def _config(**overrides) -> FactorExecutionCostConfig:
    config = FactorExecutionCostConfig(config_id="placeholder", **overrides)
    config.config_id = make_execution_cost_config_id(config)
    return config


def _bundle(*, missing_vwap: bool = False, missing_data: bool = False) -> MatrixDataBundle:
    fields = {
        FIELD_VOLUME: [[1000.0, 1000.0, 1000.0, 1000.0], [2000.0, 2000.0, 2000.0, 2000.0]],
        FIELD_AMOUNT: [[10000.0, 11000.0, 12000.0, 13000.0], [20000.0, 22000.0, 24000.0, 26000.0]],
    }
    if not missing_vwap:
        fields[FIELD_VWAP] = [[10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0, 13.0]]
    if missing_data:
        fields[FIELD_AMOUNT][0][1] = None
        fields[FIELD_VOLUME][0][1] = None
        if FIELD_VWAP in fields:
            fields[FIELD_VWAP][0][1] = None
    contract = MatrixDataContract(
        contract_id=make_matrix_contract_id(
            universe="CN",
            benchmark="CSI300",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=DATES,
        optional_fields=list(fields),
        field_sources={field_name: "fixture" for field_name in fields},
        point_in_time_flags={field_name: True for field_name in fields},
        metadata={"preserve_symbol_order": True},
    )
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(contract_id=contract.contract_id, field_names=fields),
        contract=contract,
        fields=fields,
        universe_mask=_matrix(True),
        tradability_mask=_matrix(True),
        metadata={"fixture": True},
    )


def _mask(*, block_buy: bool = False, block_sell: bool = False) -> AShareTradabilityMask:
    can_trade = _matrix(True)
    can_buy = _matrix(True)
    can_sell = _matrix(True)
    can_hold = _matrix(True)
    issue_codes = [[[] for _date in DATES] for _symbol in SYMBOLS]
    if block_buy:
        can_buy[0][1] = False
        issue_codes[0][1] = ["limit_up_buy_blocked"]
    if block_sell:
        can_sell[0][2] = False
        issue_codes[0][2] = ["limit_down_sell_blocked"]
    return AShareTradabilityMask(
        mask_id="mask-fixture",
        symbols=SYMBOLS,
        dates=DATES,
        can_trade_mask=can_trade,
        can_buy_mask=can_buy,
        can_sell_mask=can_sell,
        can_hold_mask=can_hold,
        research_eligible_mask=_matrix(True),
        issue_codes_by_cell=issue_codes,
        config=AShareTradabilityConfig(config_id="tradability-config-fixture"),
    )


def _weight_matrix() -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id="weights-fixture",
        factor_matrix_id="matrix-fixture",
        factor_id="factor-fixture",
        factor_version="v1",
        config_id="config-fixture",
        symbols=SYMBOLS,
        dates=DATES,
        long_weights=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        short_weights=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        net_weights=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        metadata={"mode": BACKTEST_MODE_LONG_ONLY},
    )


def _aggregate_result() -> FactorBacktestResult:
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id="factor-fixture",
            factor_version="v1",
            config_id="config-fixture",
        ),
        factor_id="factor-fixture",
        factor_version="v1",
        config_id="config-fixture",
        start_date=DATES[0],
        end_date=DATES[-1],
        sample_days=2,
        coverage_ratio=1.0,
        missing_ratio=0.0,
        ann_ret=0.10,
        ann_vol=0.20,
        sharpe=0.50,
        max_drawdown=0.01,
        turnover_avg=1.0,
        long_num_avg=1.0,
        short_num_avg=0.0,
        top_bottom_spread=0.03,
        after_cost_top_bottom_spread=0.029,
    )


def _daily_records() -> list[FactorDailyBacktestRecord]:
    return [
        FactorDailyBacktestRecord(
            date="2026-01-03",
            signal_date="2026-01-01",
            execution_start_date="2026-01-02",
            execution_end_date="2026-01-03",
            long_return=0.02,
            short_return=None,
            long_short_return=0.02,
            after_cost_return=0.019,
            benchmark_return=0.01,
            excess_return=0.009,
            turnover=1.0,
            long_count=1,
            short_count=0,
            coverage_ratio=1.0,
            missing_ratio=0.0,
        ),
        FactorDailyBacktestRecord(
            date="2026-01-04",
            signal_date="2026-01-02",
            execution_start_date="2026-01-03",
            execution_end_date="2026-01-04",
            long_return=-0.01,
            short_return=None,
            long_short_return=-0.01,
            after_cost_return=-0.011,
            benchmark_return=-0.005,
            excess_return=-0.006,
            turnover=1.0,
            long_count=1,
            short_count=0,
            coverage_ratio=1.0,
            missing_ratio=0.0,
        ),
    ]


def _run() -> SingleFactorBacktestRun:
    return SingleFactorBacktestRun(
        run_id="run-fixture",
        factor_matrix_id="matrix-fixture",
        factor_id="factor-fixture",
        factor_version="v1",
        config_id="config-fixture",
        start_date=DATES[0],
        end_date=DATES[-1],
        mode=BACKTEST_MODE_LONG_ONLY,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(),
        daily_records=_daily_records(),
        aggregate_result=_aggregate_result(),
        metadata={"offline_only": True},
    )


def test_execution_cost_contracts_are_split_and_reexported() -> None:
    primitives = importlib.import_module(
        "quant_investor.factors.execution_cost_primitives"
    )
    records = importlib.import_module(
        "quant_investor.factors.execution_cost_records"
    )

    assert (
        execution_cost.FactorExecutionCostConfig
        is execution_cost_types.FactorExecutionCostConfig
    )
    assert execution_cost_types.FactorExecutionCostConfig is records.FactorExecutionCostConfig
    assert execution_cost.ExecutionCostIssue is execution_cost_types.ExecutionCostIssue
    assert execution_cost_types.ExecutionCostIssue is records.ExecutionCostIssue
    assert (
        execution_cost.DailyExecutionCostRecord
        is execution_cost_types.DailyExecutionCostRecord
    )
    assert execution_cost_types.DailyExecutionCostRecord is records.DailyExecutionCostRecord
    assert (
        execution_cost.SymbolExecutionCostRecord
        is execution_cost_types.SymbolExecutionCostRecord
    )
    assert (
        execution_cost_types.SymbolExecutionCostRecord
        is records.SymbolExecutionCostRecord
    )
    assert (
        execution_cost.FactorExecutionCostSimulationReport
        is execution_cost_types.FactorExecutionCostSimulationReport
    )
    assert (
        execution_cost_types.FactorExecutionCostSimulationReport
        is records.FactorExecutionCostSimulationReport
    )
    assert (
        execution_cost.ExecutionAdjustedBacktestRun
        is execution_cost_types.ExecutionAdjustedBacktestRun
    )
    assert (
        execution_cost_types.ExecutionAdjustedBacktestRun
        is records.ExecutionAdjustedBacktestRun
    )
    assert (
        execution_cost.make_execution_cost_config_id
        is execution_cost_types.make_execution_cost_config_id
    )
    assert (
        execution_cost_types.make_execution_cost_config_id
        is records.make_execution_cost_config_id
    )
    assert (
        execution_cost.make_execution_cost_report_id
        is execution_cost_types.make_execution_cost_report_id
    )
    assert (
        execution_cost_types.make_execution_cost_report_id
        is records.make_execution_cost_report_id
    )
    assert execution_cost.safe_float is execution_cost_types.safe_float
    assert execution_cost_types.safe_float is records.safe_float
    assert execution_cost.infer_trade_direction is execution_cost_types.infer_trade_direction
    assert execution_cost_types._issue_message is primitives._issue_message
    assert execution_cost_types._matrix_value_by_symbol is primitives._matrix_value_by_symbol


def test_config_and_record_models_round_trip_and_validate() -> None:
    config = _config()
    assert FactorExecutionCostConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()

    for kwargs in (
        {"commission_bps": -1.0},
        {"max_participation_rate": 1.1},
        {"impact_model": "bad"},
        {"penalty_policy": "bad"},
    ):
        with pytest.raises(ValueError):
            _config(**kwargs)

    daily = DailyExecutionCostRecord(
        record_id="daily-1",
        date="2026-01-01",
        turnover=1.0,
        status=EXECUTION_SIMULATION_STATUS_OK,
    )
    symbol = SymbolExecutionCostRecord(
        record_id="symbol-1",
        symbol="AAA",
        date="2026-01-01",
        trade_direction="buy",
        fill_ratio=1.0,
        status=EXECUTION_SIMULATION_STATUS_OK,
    )
    report = FactorExecutionCostSimulationReport(
        report_id=make_execution_cost_report_id(
            backtest_run_id="run-fixture",
            weight_matrix_id="weights-fixture",
            generated_at="2026-04-27T00:00:00Z",
        ),
        generated_at="2026-04-27T00:00:00Z",
        config=config,
        daily_records=[daily],
        symbol_records=[symbol],
        issues=[],
    )
    adjusted = ExecutionAdjustedBacktestRun(
        adjusted_run_id="adjusted-1",
        source_backtest_run_id="run-fixture",
        cost_report_id=report.report_id,
        generated_at="2026-04-27T00:00:00Z",
        original_daily_returns={"2026-01-01": 0.01},
        simulated_daily_returns={"2026-01-01": 0.009},
        cost_returns_by_date={"2026-01-01": 0.001},
        penalty_returns_by_date={"2026-01-01": 0.0},
    )
    assert DailyExecutionCostRecord.from_dict(daily.to_dict()).to_dict() == daily.to_dict()
    assert SymbolExecutionCostRecord.from_dict(symbol.to_dict()).to_dict() == symbol.to_dict()
    assert FactorExecutionCostSimulationReport.from_dict(report.to_dict()).to_dict() == report.to_dict()
    assert ExecutionAdjustedBacktestRun.from_dict(adjusted.to_dict()).to_dict() == adjusted.to_dict()
    json.dumps(report.to_dict(), sort_keys=True)


def test_math_helpers_and_impact_models() -> None:
    assert bps_to_decimal_return(100.0) == 0.01
    assert infer_trade_direction(0.1) == "buy"
    assert infer_trade_direction(-0.1) == "sell"
    assert infer_trade_direction(0.0) == "hold"
    assert estimate_participation_rate(
        trade_weight=0.2,
        portfolio_value=1000.0,
        amount=10000.0,
    ) == 0.02
    assert clamp_unit_interval(-1.0) == 0.0
    assert clamp_unit_interval(2.0) == 1.0
    assert estimate_market_impact_bps(
        participation_rate=0.05,
        config=_config(impact_model=COST_MODEL_LINEAR_PARTICIPATION, impact_coefficient=10.0),
    ) == 0.5
    assert estimate_market_impact_bps(
        participation_rate=0.04,
        config=_config(impact_model=COST_MODEL_SQRT_IMPACT, impact_coefficient=10.0),
    ) == 2.0
    assert estimate_market_impact_bps(
        participation_rate=None,
        config=_config(impact_model=COST_MODEL_FIXED_BPS, impact_coefficient=10.0),
    ) == 0.0
    assert estimate_market_impact_bps(
        participation_rate=0.04,
        config=_config(impact_model=COST_MODEL_FIXED_BPS, impact_coefficient=10.0),
    ) == 10.0


def test_price_matrix_extracts_vwap_from_amount_volume_without_mutating_bundle() -> None:
    bundle = _bundle(missing_vwap=True)
    before = bundle.to_dict()
    price_matrix = extract_price_matrix(bundle, FIELD_VWAP)

    assert price_matrix[0][0] == 10.0
    assert bundle.to_dict() == before


def test_executable_weights_handle_clean_blocked_and_no_mask_cases() -> None:
    config = _config()
    previous = {"AAA": 0.0, "BBB": 0.0}
    target = {"AAA": 1.0, "BBB": 0.0}

    executable, records, issues = simulate_executable_weights_for_day(
        symbols=SYMBOLS,
        date="2026-01-02",
        previous_weights=previous,
        target_weights=target,
        tradability_mask=_mask(),
        date_index=1,
        config=config,
    )
    assert executable["AAA"] == 1.0
    assert records[0].status == EXECUTION_SIMULATION_STATUS_OK
    assert issues == []

    previous_snapshot = dict(previous)
    target_snapshot = dict(target)
    blocked_executable, blocked_records, blocked_issues = simulate_executable_weights_for_day(
        symbols=SYMBOLS,
        date="2026-01-02",
        previous_weights=previous,
        target_weights=target,
        tradability_mask=_mask(block_buy=True),
        date_index=1,
        config=config,
    )
    assert blocked_executable["AAA"] == 0.0
    assert blocked_records[0].status == EXECUTION_SIMULATION_STATUS_BLOCKED
    assert EXECUTION_COST_ISSUE_BLOCKED_BUY in blocked_records[0].issue_codes
    assert blocked_issues[0].issue_code == EXECUTION_COST_ISSUE_BLOCKED_BUY
    assert previous == previous_snapshot
    assert target == target_snapshot

    sell_executable, sell_records, _sell_issues = simulate_executable_weights_for_day(
        symbols=SYMBOLS,
        date="2026-01-03",
        previous_weights={"AAA": 1.0, "BBB": 0.0},
        target_weights={"AAA": 0.0, "BBB": 1.0},
        tradability_mask=_mask(block_sell=True),
        date_index=2,
        config=config,
    )
    assert sell_executable["AAA"] == 1.0
    assert EXECUTION_COST_ISSUE_BLOCKED_SELL in sell_records[0].issue_codes

    _no_mask_weights, no_mask_records, no_mask_issues = simulate_executable_weights_for_day(
        symbols=SYMBOLS,
        date="2026-01-02",
        previous_weights=previous,
        target_weights=target,
        tradability_mask=None,
        date_index=1,
        config=config,
    )
    assert no_mask_issues == []
    assert no_mask_records[0].metadata["no_tradability_mask_provided"] is True


def test_symbol_cost_computation_is_hand_calculated_and_flags_data_gaps() -> None:
    config = _config(
        commission_bps=10.0,
        exchange_fee_bps=5.0,
        slippage_bps=20.0,
        spread_bps=5.0,
        stamp_tax_bps=50.0,
        impact_model=COST_MODEL_FIXED_BPS,
        impact_coefficient=10.0,
        max_participation_rate=0.10,
    )
    record = SymbolExecutionCostRecord(
        record_id="symbol-sell",
        symbol="AAA",
        date="2026-01-02",
        previous_weight=1.0,
        target_weight=0.5,
        executable_weight=0.5,
        trade_weight=-0.5,
        executed_trade_weight=-0.5,
        trade_direction="sell",
    )

    [costed] = compute_symbol_execution_costs_for_day(
        symbol_records=[record],
        amount_by_symbol={"AAA": 1000.0},
        volume_by_symbol={"AAA": 100.0},
        price_by_symbol={"AAA": 10.0},
        portfolio_value=1000.0,
        config=config,
    )

    assert math.isclose(costed.commission_cost_return, 0.0005)
    assert math.isclose(costed.exchange_fee_cost_return, 0.00025)
    assert math.isclose(costed.slippage_cost_return, 0.001)
    assert math.isclose(costed.spread_cost_return, 0.00025)
    assert math.isclose(costed.stamp_tax_cost_return, 0.0025)
    assert math.isclose(costed.impact_cost_return, 0.0005)
    assert costed.status == EXECUTION_SIMULATION_STATUS_PARTIAL
    assert EXECUTION_COST_ISSUE_LOW_CAPACITY in costed.issue_codes
    assert EXECUTION_COST_ISSUE_PARTIAL_FILL in costed.issue_codes

    [buy_costed] = compute_symbol_execution_costs_for_day(
        symbol_records=[
            SymbolExecutionCostRecord(
                record_id="symbol-buy",
                symbol="BBB",
                date="2026-01-02",
                target_weight=0.5,
                executable_weight=0.5,
                trade_weight=0.5,
                executed_trade_weight=0.5,
                trade_direction="buy",
            )
        ],
        amount_by_symbol={"BBB": None},
        volume_by_symbol={"BBB": None},
        price_by_symbol={"BBB": None},
        portfolio_value=1000.0,
        config=config,
    )
    assert buy_costed.stamp_tax_cost_return == 0.0
    assert EXECUTION_COST_ISSUE_MISSING_AMOUNT in buy_costed.issue_codes
    assert EXECUTION_COST_ISSUE_MISSING_VOLUME in buy_costed.issue_codes
    assert EXECUTION_COST_ISSUE_MISSING_PRICE in buy_costed.issue_codes


def test_daily_simulation_report_markdown_dashboard_and_adjusted_run_are_separate() -> None:
    run = _run()
    before = run.to_dict()
    config = _config(
        commission_bps=2.0,
        exchange_fee_bps=0.5,
        slippage_bps=2.0,
        spread_bps=0.0,
        stamp_tax_bps=5.0,
        impact_model=COST_MODEL_LINEAR_PARTICIPATION,
        impact_coefficient=0.0,
        high_cost_warning_threshold=1.0,
    )

    daily_records, symbol_records, issues = build_daily_execution_cost_records(
        run=run,
        bundle=_bundle(),
        tradability_mask=_mask(),
        config=config,
        portfolio_value=100000.0,
    )
    assert daily_records[0].simulated_after_cost_return < daily_records[0].gross_return
    assert daily_records[1].buy_turnover == 1.0
    assert daily_records[1].sell_turnover == 1.0
    assert symbol_records
    assert issues
    assert run.to_dict() == before

    blocked_report = build_execution_cost_simulation_report(
        run=run,
        bundle=_bundle(),
        tradability_mask=_mask(block_buy=True),
        config=config,
        portfolio_value=100000.0,
        generated_at="2026-04-27T00:00:00Z",
    )
    assert blocked_report.verdict == EXECUTION_COST_SIMULATION_FAIL
    assert blocked_report.blocked_buy_count > 0
    assert blocked_report.daily_records[0].status == EXECUTION_SIMULATION_STATUS_BLOCKED
    assert blocked_report.daily_records[0].simulated_penalty_return is not None

    warning_report = build_execution_cost_simulation_report(
        run=run,
        bundle=_bundle(missing_data=True),
        tradability_mask=_mask(),
        config=_config(high_cost_warning_threshold=0.000001),
        portfolio_value=100000.0,
        generated_at="2026-04-27T00:00:01Z",
    )
    assert warning_report.verdict == EXECUTION_COST_SIMULATION_WARN
    assert warning_report.cost_drag_annualized_return is not None
    assert warning_report.cost_drag_sharpe is not None
    assert any(issue.issue_code == EXECUTION_COST_ISSUE_HIGH_TURNOVER_COST for issue in warning_report.issues)
    assert any(issue.issue_code == EXECUTION_COST_ISSUE_MISSING_AMOUNT for issue in warning_report.issues)

    markdown = render_execution_cost_report_markdown(warning_report)
    assert "offline-only and does not alter official scoring" in markdown
    dashboard = build_execution_cost_dashboard_payload(warning_report)
    json.dumps(dashboard, sort_keys=True)
    adjusted = build_execution_adjusted_backtest_run(
        warning_report,
        source_backtest_run_id=run.run_id,
        generated_at="2026-04-27T00:00:02Z",
    )
    assert adjusted.source_backtest_run_id == run.run_id
    assert adjusted.original_daily_returns["2026-01-03"] == 0.019
    assert adjusted.simulated_daily_returns["2026-01-03"] != 0.019
    assert run.to_dict() == before
