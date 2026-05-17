from __future__ import annotations

from quant_investor.factors.backtest import EXECUTION_PRICE_VWAP, run_single_factor_backtest
from quant_investor.factors.matrix import (
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
    compute_coverage,
    make_factor_matrix_id,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.schema import FactorBacktestConfig, make_backtest_config_id
from quant_investor.factors.tradability import (
    EXECUTION_AUDIT_STATUS_BLOCKED,
    EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE,
    FIELD_AMOUNT,
    FIELD_DELISTED,
    FIELD_IS_ST,
    FIELD_LIMIT_DOWN,
    FIELD_LIMIT_UP,
    FIELD_LISTING_DAYS,
    FIELD_LOW_LIQUIDITY,
    FIELD_SUSPENDED,
    FIELD_VALID_PRICE,
    FIELD_VALID_VOLUME,
    FIELD_VOLUME,
    FIELD_VWAP,
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_SELL,
    TRADABILITY_AUDIT_FAIL,
    TRADABILITY_AUDIT_PASS,
    TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
    TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
    TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    TRADABILITY_ISSUE_SUSPENDED,
    AShareTradabilityConfig,
    FactorExecutionFeasibilityReport,
    audit_factor_weight_execution_feasibility,
    build_ashare_tradability_mask,
    make_tradability_config_id,
    render_execution_feasibility_markdown,
)


SYMBOLS = ["AAA", "BBB", "CCC"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]


def _matrix(value):
    return [[value for _date in DATES] for _symbol in SYMBOLS]


def _config() -> AShareTradabilityConfig:
    config = AShareTradabilityConfig(config_id="placeholder")
    config.config_id = make_tradability_config_id(config)
    return config


def _fields() -> dict[str, list[list[object]]]:
    vwap = [
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [20.0, 19.0, 18.0, 17.0, 16.0],
        [30.0, 30.0, 33.0, 33.0, 36.0],
    ]
    volume = _matrix(1000.0)
    amount = [
        [price * volume[row_index][column_index] for column_index, price in enumerate(row)]
        for row_index, row in enumerate(vwap)
    ]
    return {
        FIELD_VWAP: vwap,
        FIELD_VOLUME: volume,
        FIELD_AMOUNT: amount,
        FIELD_SUSPENDED: _matrix(False),
        FIELD_LIMIT_UP: _matrix(False),
        FIELD_LIMIT_DOWN: _matrix(False),
        FIELD_IS_ST: _matrix(False),
        FIELD_DELISTED: _matrix(False),
        FIELD_LISTING_DAYS: _matrix(120),
        FIELD_VALID_PRICE: _matrix(True),
        FIELD_VALID_VOLUME: _matrix(True),
        FIELD_LOW_LIQUIDITY: _matrix(False),
    }


def _bundle(fields: dict[str, list[list[object]]] | None = None) -> MatrixDataBundle:
    resolved_fields = fields or _fields()
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
        optional_fields=list(resolved_fields),
        field_sources={field_name: "fixture" for field_name in resolved_fields},
        point_in_time_flags={field_name: True for field_name in resolved_fields},
        metadata={"preserve_symbol_order": True},
    )
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=resolved_fields,
        ),
        contract=contract,
        fields=resolved_fields,
        universe_mask=_matrix(True),
        tradability_mask=_matrix(True),
        metadata={"fixture": True},
    )


def _factor_matrix() -> FactorMatrix:
    values = [
        [3.0, 1.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0, 3.0, 2.0],
    ]
    coverage_ratio, missing_ratio = compute_coverage(values)
    return FactorMatrix(
        matrix_id=make_factor_matrix_id(
            expression="tradability_execution_fixture",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        factor_id="tradability-execution-fixture",
        factor_version="v1",
        expression="tradability_execution_fixture",
        symbols=SYMBOLS,
        dates=DATES,
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        metadata={"expected_direction": 1.0},
    )


def _backtest_config() -> FactorBacktestConfig:
    config = FactorBacktestConfig(
        config_id="placeholder",
        universe="CN",
        benchmark="CSI300",
        start_date="2026-01-01",
        end_date="2026-01-03",
        rebalance_frequency="daily",
        delay_days=1,
        execution_price=EXECUTION_PRICE_VWAP,
        long_short=True,
        long_only=False,
        quantile_count=3,
        long_quantile=3,
        short_quantile=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        market_impact_bps=0.0,
        min_coverage_ratio=0.0,
    )
    config.config_id = make_backtest_config_id(config)
    return config


def _run():
    return run_single_factor_backtest(_factor_matrix(), _bundle(), _backtest_config())


def _report(fields: dict[str, list[list[object]]] | None = None):
    run = _run()
    mask = build_ashare_tradability_mask(_bundle(fields or _fields()), config=_config())
    return audit_factor_weight_execution_feasibility(
        weight_matrix=run.weight_matrix,
        tradability_mask=mask,
        run=run,
        generated_at="2026-04-27T00:00:00Z",
    ), run


def test_clean_mask_execution_feasibility_passes_without_mutating_backtest() -> None:
    report, run = _report()
    run_payload = run.to_dict()

    rerun_report = audit_factor_weight_execution_feasibility(
        weight_matrix=run.weight_matrix,
        tradability_mask=build_ashare_tradability_mask(_bundle(), config=_config()),
        run=run,
        generated_at="2026-04-27T00:00:00Z",
    )

    assert report.verdict == TRADABILITY_AUDIT_PASS
    assert report.blocked_transitions == 0
    assert report.total_transitions == len(run.daily_records) * len(SYMBOLS)
    assert report.metadata["short_leg_is_research_analytic_not_cash_equity_short"] is True
    assert run.to_dict() == run_payload
    assert rerun_report.to_dict() == report.to_dict()


def test_limit_up_on_execution_date_blocks_buy_transition() -> None:
    fields = _fields()
    fields[FIELD_LIMIT_UP][0][1] = True
    report, _run_payload = _report(fields)

    blocked = [
        record for record in report.transition_records
        if record.symbol == "AAA" and record.execution_date == "2026-01-02"
    ][0]
    assert report.verdict == TRADABILITY_AUDIT_FAIL
    assert report.blocked_buy_count == 1
    assert report.blocked_symbols == ["AAA"]
    assert blocked.trade_direction == TRADE_DIRECTION_BUY
    assert blocked.status == EXECUTION_AUDIT_STATUS_BLOCKED
    assert blocked.issue_codes == [
        TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
        TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    ]


def test_limit_down_on_execution_date_blocks_sell_transition() -> None:
    fields = _fields()
    fields[FIELD_LIMIT_DOWN][1][1] = True
    report, _run_payload = _report(fields)

    blocked = [
        record for record in report.transition_records
        if record.symbol == "BBB" and record.execution_date == "2026-01-02"
    ][0]
    assert report.verdict == TRADABILITY_AUDIT_FAIL
    assert report.blocked_sell_count == 1
    assert report.blocked_symbols == ["BBB"]
    assert blocked.trade_direction == TRADE_DIRECTION_SELL
    assert blocked.status == EXECUTION_AUDIT_STATUS_BLOCKED
    assert blocked.issue_codes == [
        TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
        TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    ]


def test_suspension_on_execution_date_blocks_transition_and_orders_records() -> None:
    fields = _fields()
    fields[FIELD_SUSPENDED][0][1] = True
    report, _run_payload = _report(fields)

    assert report.verdict == TRADABILITY_AUDIT_FAIL
    assert report.blocked_symbols == ["AAA"]
    blocked = [
        record for record in report.transition_records
        if record.symbol == "AAA" and record.execution_date == "2026-01-02"
    ][0]
    assert blocked.status == EXECUTION_AUDIT_STATUS_BLOCKED
    assert blocked.issue_codes == [
        TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
        TRADABILITY_ISSUE_SUSPENDED,
    ]


def test_execution_report_round_trip_markdown_and_deterministic_ordering() -> None:
    fields = _fields()
    fields[FIELD_LIMIT_UP][0][1] = True
    fields[FIELD_LIMIT_DOWN][1][1] = True
    report, _run_payload = _report(fields)

    assert report.blocked_symbols == ["AAA", "BBB"]
    assert report.transition_records == sorted(
        report.transition_records,
        key=lambda record: (record.execution_date, record.symbol, record.signal_date, record.record_id),
    )
    assert FactorExecutionFeasibilityReport.from_dict(report.to_dict()).to_dict() == report.to_dict()
    markdown = render_execution_feasibility_markdown(report)
    assert EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE in markdown
