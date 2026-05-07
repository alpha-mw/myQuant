from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_SHORT,
    WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
)
from quant_investor.factors.capacity import (
    CAPACITY_ISSUE_COST_DRAG,
    CAPACITY_ISSUE_HIGH_TURNOVER,
    CAPACITY_ISSUE_LOW_COVERAGE,
    CAPACITY_ISSUE_LOW_TRADABILITY,
    CAPACITY_ISSUE_PARTICIPATION_BREACH,
    CAPACITY_VERDICT_FAIL,
    CAPACITY_VERDICT_PASS,
    CAPACITY_VERDICT_WARN,
    FactorCostCapacityConfig,
    FactorCostCapacityReport,
    build_factor_cost_capacity_report,
    estimate_average_adv,
    estimate_factor_capacity,
)
from quant_investor.factors.matrix import FIELD_AMOUNT, MatrixDataBundle, MatrixDataContract
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id


SYMBOLS = ["AAA", "BBB"]


def _dates(count: int = 4) -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _bundle(
    dates: list[str],
    *,
    amount: float = 10_000_000.0,
    tradable: bool = True,
) -> MatrixDataBundle:
    contract = MatrixDataContract(
        contract_id="contract-capacity-fixture",
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=dates,
        required_fields=[FIELD_AMOUNT],
        field_sources={FIELD_AMOUNT: "fixture"},
        point_in_time_flags={FIELD_AMOUNT: True},
    )
    return MatrixDataBundle(
        bundle_id="bundle-capacity-fixture",
        contract=contract,
        fields={FIELD_AMOUNT: [[amount for _ in dates], [amount for _ in dates]]},
        tradability_mask=[[tradable for _ in dates], [tradable for _ in dates]],
    )


def _weight_matrix(dates: list[str]) -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id="weights-capacity-fixture",
        factor_matrix_id="matrix-capacity-fixture",
        factor_id="factor-capacity-fixture",
        factor_version="v1",
        config_id="config-capacity-fixture",
        symbols=SYMBOLS,
        dates=dates,
        long_weights=[[1.0 for _ in dates], [0.0 for _ in dates]],
        short_weights=[[0.0 for _ in dates], [-1.0 for _ in dates]],
        net_weights=[[1.0 for _ in dates], [-1.0 for _ in dates]],
    )


def _aggregate(
    dates: list[str],
    *,
    coverage_ratio: float = 0.95,
    before_cost_sharpe: float = 1.50,
    after_cost_sharpe: float = 1.00,
    turnover_avg: float = 0.20,
) -> FactorBacktestResult:
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id="factor-capacity-fixture",
            factor_version="v1",
            config_id="config-capacity-fixture",
        ),
        factor_id="factor-capacity-fixture",
        factor_version="v1",
        config_id="config-capacity-fixture",
        start_date=dates[0],
        end_date=dates[-1],
        sample_days=len(dates),
        coverage_ratio=coverage_ratio,
        missing_ratio=1.0 - coverage_ratio,
        ann_ret=0.20,
        ann_vol=0.10,
        sharpe=2.0,
        max_drawdown=0.05,
        turnover_avg=turnover_avg,
        long_num_avg=1.0,
        short_num_avg=1.0,
        rank_ic_mean=0.04,
        ic_mean=0.04,
        icir=0.60,
        ic_t_stat=4.0,
        positive_ic_ratio=0.65,
        top_bottom_spread=0.012,
        after_cost_top_bottom_spread=0.010,
        before_cost_sharpe=before_cost_sharpe,
        after_cost_sharpe=after_cost_sharpe,
        metadata={"point_in_time_passed": True},
    )


def _run(
    dates: list[str],
    *,
    turnover: float = 0.20,
    coverage_ratio: float = 0.95,
    before_cost_sharpe: float = 1.50,
    after_cost_sharpe: float = 1.00,
) -> SingleFactorBacktestRun:
    records = [
        FactorDailyBacktestRecord(
            date=current_date,
            signal_date=current_date,
            execution_start_date=current_date,
            execution_end_date=current_date,
            long_return=0.01,
            short_return=0.00,
            long_short_return=0.01,
            after_cost_return=0.009,
            benchmark_return=0.001,
            excess_return=0.008,
            turnover=turnover,
            long_count=1,
            short_count=1,
            coverage_ratio=coverage_ratio,
            missing_ratio=1.0 - coverage_ratio,
        )
        for current_date in dates
    ]
    return SingleFactorBacktestRun(
        run_id="run-capacity-fixture",
        factor_matrix_id="matrix-capacity-fixture",
        factor_id="factor-capacity-fixture",
        factor_version="v1",
        config_id="config-capacity-fixture",
        start_date=dates[0],
        end_date=dates[-1],
        mode=BACKTEST_MODE_LONG_SHORT,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(dates),
        daily_records=records,
        aggregate_result=_aggregate(
            dates,
            coverage_ratio=coverage_ratio,
            before_cost_sharpe=before_cost_sharpe,
            after_cost_sharpe=after_cost_sharpe,
            turnover_avg=turnover,
        ),
        metadata={"offline_only": True},
    )


def test_factor_cost_capacity_config_validates_inputs() -> None:
    with pytest.raises(ValueError, match="target_capital"):
        FactorCostCapacityConfig(config_id="bad", target_capital=0.0)
    with pytest.raises(ValueError, match="max_participation_rate"):
        FactorCostCapacityConfig(
            config_id="bad",
            target_capital=1_000_000.0,
            max_participation_rate=1.50,
        )
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        FactorCostCapacityConfig(
            config_id="bad",
            target_capital=1_000_000.0,
            transaction_cost_bps=-1.0,
        )


def test_estimate_average_adv_uses_amount_matrix() -> None:
    dates = _dates()
    bundle = _bundle(dates, amount=2_000_000.0)

    assert estimate_average_adv(bundle) == pytest.approx(2_000_000.0)


def test_build_factor_cost_capacity_report_computes_cost_drag() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates, turnover=0.20, before_cost_sharpe=1.50, after_cost_sharpe=1.00),
        _bundle(dates),
        config=FactorCostCapacityConfig(
            config_id="capacity-fixture",
            target_capital=100_000.0,
            transaction_cost_bps=10.0,
            slippage_bps=5.0,
            market_impact_bps=5.0,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert report.average_turnover == pytest.approx(0.20)
    assert report.total_cost_bps == pytest.approx(20.0)
    assert report.estimated_average_cost_return == pytest.approx(0.0004)
    assert report.cost_drag_ratio == pytest.approx(1 / 3)
    assert report.verdict == CAPACITY_VERDICT_PASS
    assert FactorCostCapacityReport.from_dict(report.to_dict()).to_dict() == report.to_dict()


def test_low_coverage_triggers_issue() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates, coverage_ratio=0.50),
        _bundle(dates),
        config=FactorCostCapacityConfig(config_id="low-coverage", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )

    assert CAPACITY_ISSUE_LOW_COVERAGE in report.issue_codes
    assert report.verdict == CAPACITY_VERDICT_FAIL


def test_low_tradability_triggers_issue() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates),
        _bundle(dates, tradable=False),
        config=FactorCostCapacityConfig(config_id="low-tradability", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )

    assert CAPACITY_ISSUE_LOW_TRADABILITY in report.issue_codes
    assert report.tradability_ratio == 0.0
    assert report.verdict == CAPACITY_VERDICT_FAIL


def test_high_turnover_triggers_warning_issue() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates, turnover=0.80),
        _bundle(dates),
        config=FactorCostCapacityConfig(
            config_id="high-turnover",
            target_capital=100_000.0,
            max_average_turnover=0.50,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert CAPACITY_ISSUE_HIGH_TURNOVER in report.issue_codes
    assert report.verdict == CAPACITY_VERDICT_WARN


def test_cost_drag_triggers_warning_issue() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates, before_cost_sharpe=1.00, after_cost_sharpe=0.40),
        _bundle(dates),
        config=FactorCostCapacityConfig(
            config_id="cost-drag",
            target_capital=100_000.0,
            max_cost_drag_ratio=0.50,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert CAPACITY_ISSUE_COST_DRAG in report.issue_codes
    assert report.verdict == CAPACITY_VERDICT_WARN


def test_participation_breach_detected_with_small_amount() -> None:
    dates = _dates()
    report = build_factor_cost_capacity_report(
        _run(dates, turnover=0.50),
        _bundle(dates, amount=100_000.0),
        config=FactorCostCapacityConfig(
            config_id="participation",
            target_capital=1_000_000.0,
            max_participation_rate=0.10,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert CAPACITY_ISSUE_PARTICIPATION_BREACH in report.issue_codes
    assert report.participation_breach_count == len(dates)
    assert report.participation_breach_ratio == 1.0
    assert report.verdict == CAPACITY_VERDICT_WARN


def test_estimated_capacity_is_deterministic() -> None:
    dates = _dates()
    config = FactorCostCapacityConfig(
        config_id="capacity-deterministic",
        target_capital=100_000.0,
        max_participation_rate=0.10,
    )

    capacity, metadata = estimate_factor_capacity(_run(dates, turnover=0.20), _bundle(dates), config)

    assert capacity == pytest.approx(5_000_000.0)
    assert metadata["participation_breach_count"] == 0
