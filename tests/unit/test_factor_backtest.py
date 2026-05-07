from __future__ import annotations

import copy
import math

import pytest

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_ONLY,
    BACKTEST_MODE_LONG_SHORT,
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    FactorBacktestAlignment,
    SingleFactorBacktestRun,
    bps_to_decimal_return,
    build_backtest_alignments,
    build_execution_return_matrix,
    build_factor_backtest_result,
    build_quantile_weight_matrix,
    compute_daily_backtest_records,
    estimate_turnover,
    max_drawdown_from_returns,
    run_single_factor_backtest,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_BENCHMARK_RET,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VOLUME,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
    compute_coverage,
    make_factor_matrix_id,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.schema import (
    FactorBacktestConfig,
    FactorBacktestResult,
    make_backtest_config_id,
)
from quant_investor.versioning import FACTOR_BACKTEST_SCHEMA_VERSION


SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]


def _contract(required_fields: list[str] | None = None) -> MatrixDataContract:
    fields = required_fields or [
        FIELD_OPEN,
        FIELD_CLOSE,
        FIELD_AMOUNT,
        FIELD_VOLUME,
        FIELD_BENCHMARK_RET,
    ]
    return MatrixDataContract(
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
        required_fields=fields,
        field_sources={field_name: "fixture" for field_name in fields},
        point_in_time_flags={field_name: True for field_name in fields},
        metadata={"preserve_symbol_order": True},
    )


def _bundle(
    *,
    universe_mask: list[list[bool]] | None = None,
    tradability_mask: list[list[bool]] | None = None,
    include_vwap: bool = False,
) -> MatrixDataBundle:
    vwap = [
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [20.0, 19.0, 18.0, 17.0, 16.0],
        [30.0, 30.0, 33.0, 33.0, 36.0],
        [40.0, 42.0, 44.0, 46.0, 48.0],
    ]
    volume = [[100.0 for _date in DATES] for _symbol in SYMBOLS]
    amount = [
        [price * volume[row_index][column_index] for column_index, price in enumerate(row)]
        for row_index, row in enumerate(vwap)
    ]
    fields = {
        FIELD_OPEN: [
            [9.0, 10.0, 12.0, 15.0, 15.0],
            [18.0, 20.0, 18.0, 18.0, 17.0],
            [28.0, 30.0, 31.0, 31.0, 34.0],
            [38.0, 39.0, 41.0, 43.0, 45.0],
        ],
        FIELD_CLOSE: [
            [10.0, 12.0, 12.0, 13.0, 15.0],
            [20.0, 18.0, 18.0, 16.0, 16.0],
            [30.0, 31.0, 33.0, 34.0, 36.0],
            [40.0, 41.0, 42.0, 43.0, 44.0],
        ],
        FIELD_AMOUNT: amount,
        FIELD_VOLUME: volume,
        FIELD_BENCHMARK_RET: [
            [0.0, 0.02, -0.01, 0.01, 0.0],
            [0.0, 0.02, -0.01, 0.01, 0.0],
            [0.0, 0.02, -0.01, 0.01, 0.0],
            [0.0, 0.02, -0.01, 0.01, 0.0],
        ],
    }
    if include_vwap:
        fields[FIELD_VWAP] = vwap
    contract = _contract(required_fields=list(fields.keys()))
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=fields.keys(),
        ),
        contract=contract,
        fields=fields,
        universe_mask=universe_mask
        or [[True for _date in DATES] for _symbol in SYMBOLS],
        tradability_mask=tradability_mask
        or [[True for _date in DATES] for _symbol in SYMBOLS],
        metadata={"fixture": True},
    )


def _factor_matrix(*, expected_direction: float = 1.0) -> FactorMatrix:
    values = [
        [4.0, 1.0, 4.0, 2.0, 1.0],
        [1.0, 4.0, 1.0, 3.0, 4.0],
        [3.0, 2.0, 3.0, 4.0, 3.0],
        [2.0, 3.0, 2.0, 1.0, 2.0],
    ]
    coverage_ratio, missing_ratio = compute_coverage(values)
    return FactorMatrix(
        matrix_id=make_factor_matrix_id(
            expression="fixture_factor",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        factor_id="factor-fixture",
        factor_version="v1",
        expression="fixture_factor",
        symbols=SYMBOLS,
        dates=DATES,
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        metadata={"expected_direction": expected_direction},
    )


def _config(*, mode: str = BACKTEST_MODE_LONG_SHORT, execution_price: str = EXECUTION_PRICE_VWAP) -> FactorBacktestConfig:
    config = FactorBacktestConfig(
        config_id="placeholder",
        universe="CN",
        benchmark="CSI300",
        start_date="2026-01-01",
        end_date="2026-01-03",
        rebalance_frequency="daily",
        delay_days=1,
        execution_price=execution_price,
        long_short=mode == BACKTEST_MODE_LONG_SHORT,
        long_only=mode == BACKTEST_MODE_LONG_ONLY,
        quantile_count=2,
        long_quantile=2,
        short_quantile=1 if mode == BACKTEST_MODE_LONG_SHORT else None,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
        market_impact_bps=2.0,
        min_coverage_ratio=0.0,
    )
    config.config_id = make_backtest_config_id(config)
    return config


def test_alignment_delay_and_invalid_periods() -> None:
    alignments = build_backtest_alignments(DATES[:3], delay_days=1, holding_period_days=1)

    assert len(alignments) == 1
    assert alignments[0] == FactorBacktestAlignment(
        signal_date="2026-01-01",
        execution_start_date="2026-01-02",
        execution_end_date="2026-01-03",
        signal_index=0,
        execution_start_index=1,
        execution_end_index=2,
        delay_days=1,
        holding_period_days=1,
        execution_price=EXECUTION_PRICE_VWAP,
        metadata={"alignment_policy": "signal_delay_execution_window"},
    )
    assert build_backtest_alignments(DATES[:2], delay_days=1, holding_period_days=1) == []
    with pytest.raises(ValueError, match="delay_days"):
        build_backtest_alignments(DATES, delay_days=0)
    with pytest.raises(ValueError, match="holding_period_days"):
        build_backtest_alignments(DATES, delay_days=1, holding_period_days=0)


def test_execution_return_matrix_open_close_vwap_and_invalid_prices() -> None:
    bundle = _bundle()

    open_returns = build_execution_return_matrix(
        bundle,
        execution_price=EXECUTION_PRICE_OPEN,
    )
    close_returns = build_execution_return_matrix(
        bundle,
        execution_price=EXECUTION_PRICE_CLOSE,
    )
    vwap_returns = build_execution_return_matrix(
        bundle,
        execution_price=EXECUTION_PRICE_VWAP,
    )

    assert open_returns[0][0] == pytest.approx(10.0 / 9.0 - 1.0)
    assert close_returns[1][0] == pytest.approx(18.0 / 20.0 - 1.0)
    assert vwap_returns[0][1] == pytest.approx(12.0 / 11.0 - 1.0)
    assert not bundle.has_field(FIELD_VWAP)

    bad_fields = copy.deepcopy(bundle.fields)
    bad_fields[FIELD_CLOSE][0][0] = 0.0
    bad_bundle = MatrixDataBundle(
        bundle_id="bad-price-bundle",
        contract=bundle.contract,
        fields=bad_fields,
        universe_mask=bundle.universe_mask,
        tradability_mask=bundle.tradability_mask,
    )
    bad_returns = build_execution_return_matrix(
        bad_bundle,
        execution_price=EXECUTION_PRICE_CLOSE,
    )
    assert bad_returns[0][0] is None


def test_quantile_weight_matrix_long_short_equal_booksize_and_masks() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()

    weights = build_quantile_weight_matrix(matrix, bundle, config)

    assert weights.long_weights[0][0] == pytest.approx(0.5)
    assert weights.long_weights[2][0] == pytest.approx(0.5)
    assert weights.short_weights[1][0] == pytest.approx(-0.5)
    assert weights.short_weights[3][0] == pytest.approx(-0.5)
    assert sum(row[0] or 0.0 for row in weights.long_weights) == pytest.approx(1.0)
    assert sum(row[0] or 0.0 for row in weights.short_weights) == pytest.approx(-1.0)
    assert weights.net_weights[0][0] == pytest.approx(0.5)
    assert weights.net_weights[1][0] == pytest.approx(-0.5)

    universe_mask = [[True for _date in DATES] for _symbol in SYMBOLS]
    universe_mask[2][0] = False
    masked_weights = build_quantile_weight_matrix(matrix, _bundle(universe_mask=universe_mask), config)
    assert masked_weights.long_weights[2][0] == 0.0

    tradability_mask = [[True for _date in DATES] for _symbol in SYMBOLS]
    tradability_mask[0][1] = False
    tradability_weights = build_quantile_weight_matrix(
        matrix,
        _bundle(tradability_mask=tradability_mask),
        config,
    )
    assert tradability_weights.long_weights[0][0] == 0.0


def test_quantile_weight_matrix_long_only_expected_direction_and_input_immutability() -> None:
    matrix = _factor_matrix(expected_direction=-1.0)
    bundle = _bundle()
    config = _config(mode=BACKTEST_MODE_LONG_ONLY)
    matrix_before = matrix.to_dict()
    bundle_before = bundle.to_dict()

    weights = build_quantile_weight_matrix(
        matrix,
        bundle,
        config,
        mode=BACKTEST_MODE_LONG_ONLY,
    )

    assert weights.long_weights[1][0] == pytest.approx(0.5)
    assert weights.long_weights[3][0] == pytest.approx(0.5)
    assert all(row[0] == 0.0 for row in weights.short_weights)
    assert [row[0] for row in weights.net_weights] == [row[0] for row in weights.long_weights]
    assert matrix.to_dict() == matrix_before
    assert bundle.to_dict() == bundle_before


def test_daily_records_use_delayed_weights_returns_turnover_costs_and_benchmark() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    weights = build_quantile_weight_matrix(matrix, bundle, config)

    records = compute_daily_backtest_records(matrix, bundle, config, weights)

    assert len(records) == 3
    first = records[0]
    expected_long = 0.5 * (12.0 / 11.0 - 1.0) + 0.5 * (33.0 / 30.0 - 1.0)
    expected_short = -0.5 * (18.0 / 19.0 - 1.0) + -0.5 * (44.0 / 42.0 - 1.0)
    expected_long_short = expected_long + expected_short
    assert first.signal_date == "2026-01-01"
    assert first.execution_start_date == "2026-01-02"
    assert first.date == "2026-01-03"
    assert first.long_return == pytest.approx(expected_long)
    assert first.short_return == pytest.approx(expected_short)
    assert first.long_short_return == pytest.approx(expected_long_short)
    assert first.turnover == pytest.approx(1.0)
    assert bps_to_decimal_return(100.0) == pytest.approx(0.01)
    assert first.after_cost_return == pytest.approx(expected_long_short - 0.001)
    assert first.benchmark_return == pytest.approx(0.02)
    assert first.excess_return == pytest.approx(first.after_cost_return - 0.02)


def test_aggregate_result_metrics_metadata_and_ic_values() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    weights = build_quantile_weight_matrix(matrix, bundle, config)
    records = compute_daily_backtest_records(matrix, bundle, config, weights)

    result = build_factor_backtest_result(
        factor_matrix=matrix,
        config=config,
        daily_records=records,
        metadata={"holding_period_days": 1},
    )

    assert isinstance(result, FactorBacktestResult)
    assert result.sample_days == 3
    assert result.ann_ret is not None
    assert result.ann_vol is not None
    assert result.sharpe is not None
    assert result.max_drawdown is not None
    assert result.max_drawdown >= 0.0
    assert result.turnover_avg is not None
    assert result.long_num_avg == pytest.approx(2.0)
    assert result.short_num_avg == pytest.approx(2.0)
    assert result.top_bottom_spread is not None
    assert result.after_cost_top_bottom_spread is not None
    assert result.ic_mean is None or math.isfinite(result.ic_mean)
    assert result.rank_ic_mean is None or math.isfinite(result.rank_ic_mean)
    assert result.metadata["factor_backtest_schema_version"] == FACTOR_BACKTEST_SCHEMA_VERSION
    assert result.metadata["alignment_policy"] == "signal_delay_execution_window"
    assert max_drawdown_from_returns([0.1, -0.2, 0.05]) >= 0.0


def test_runner_returns_round_trippable_run_without_mutating_inputs() -> None:
    matrix = _factor_matrix()
    bundle = _bundle()
    config = _config()
    matrix_before = matrix.to_dict()
    bundle_before = bundle.to_dict()

    run = run_single_factor_backtest(matrix, bundle, config)
    round_trip = SingleFactorBacktestRun.from_dict(run.to_dict())

    assert isinstance(run, SingleFactorBacktestRun)
    assert isinstance(run.aggregate_result, FactorBacktestResult)
    assert round_trip.to_dict() == run.to_dict()
    assert run.metadata["offline_only"] is True
    assert matrix.to_dict() == matrix_before
    assert bundle.to_dict() == bundle_before


def test_numeric_turnover_helper_uses_half_abs_delta() -> None:
    assert estimate_turnover(
        {"AAA": 0.5, "BBB": -0.5},
        {"AAA": 0.0, "CCC": 1.0},
    ) == pytest.approx(1.0)
