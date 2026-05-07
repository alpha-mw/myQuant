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
    CAPACITY_VERDICT_PASS,
    CAPACITY_VERDICT_WARN,
    FactorCostCapacityConfig,
    build_factor_cost_capacity_report,
)
from quant_investor.factors.matrix import FIELD_AMOUNT, MatrixDataBundle, MatrixDataContract
from quant_investor.factors.robustness import (
    ROBUSTNESS_VERDICT_FAIL,
    ROBUSTNESS_VERDICT_PASS,
    ROBUSTNESS_VERDICT_WARN,
    SLICE_FULL_SAMPLE,
    FactorRobustnessReport,
    FactorSliceSpec,
    build_default_recent_slice_specs,
    build_enhanced_factor_validation_report,
    build_factor_robustness_report,
    build_regime_slice_specs,
    evaluate_slice_result,
)
from quant_investor.factors.schema import (
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_VALIDATED_RESEARCH,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorBacktestResult,
    FactorValidationThresholds,
    make_backtest_result_id,
)


SYMBOLS = ["AAA", "BBB"]


def _dates(count: int) -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _weight_matrix(dates: list[str]) -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id="weights-validation-fixture",
        factor_matrix_id="matrix-validation-fixture",
        factor_id="factor-validation-fixture",
        factor_version="v1",
        config_id="config-validation-fixture",
        symbols=SYMBOLS,
        dates=dates,
        long_weights=[[1.0 for _ in dates], [0.0 for _ in dates]],
        short_weights=[[0.0 for _ in dates], [-1.0 for _ in dates]],
        net_weights=[[1.0 for _ in dates], [-1.0 for _ in dates]],
    )


def _records(
    dates: list[str],
    *,
    before_returns: list[float] | None = None,
    after_returns: list[float] | None = None,
    turnover: float = 0.20,
    coverage_ratio: float = 0.95,
) -> list[FactorDailyBacktestRecord]:
    before = before_returns or [0.010 for _ in dates]
    after = after_returns or [0.009 for _ in dates]
    records: list[FactorDailyBacktestRecord] = []
    for index, current_date in enumerate(dates):
        records.append(
            FactorDailyBacktestRecord(
                date=current_date,
                signal_date=current_date,
                execution_start_date=current_date,
                execution_end_date=current_date,
                long_return=before[index] / 2.0,
                short_return=before[index] / 2.0,
                long_short_return=before[index],
                after_cost_return=after[index],
                benchmark_return=0.001,
                excess_return=after[index] - 0.001,
                turnover=turnover,
                long_count=1,
                short_count=1,
                coverage_ratio=coverage_ratio,
                missing_ratio=1.0 - coverage_ratio,
            )
        )
    return records


def _aggregate(
    dates: list[str],
    *,
    sample_days: int | None = None,
    coverage_ratio: float = 0.95,
    rank_ic_mean: float | None = 0.04,
    icir: float | None = 0.60,
    ic_t_stat: float | None = 4.0,
    after_cost_sharpe: float | None = 1.20,
    positive_ic_ratio: float | None = 0.65,
    after_cost_spread: float | None = 0.01,
    turnover_avg: float | None = 0.20,
    before_cost_sharpe: float | None = 1.40,
    max_drawdown: float | None = 0.05,
    metadata: dict[str, object] | None = None,
) -> FactorBacktestResult:
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id="factor-validation-fixture",
            factor_version="v1",
            config_id="config-validation-fixture",
        ),
        factor_id="factor-validation-fixture",
        factor_version="v1",
        config_id="config-validation-fixture",
        start_date=dates[0],
        end_date=dates[-1],
        sample_days=sample_days if sample_days is not None else len(dates),
        coverage_ratio=coverage_ratio,
        missing_ratio=1.0 - coverage_ratio,
        ann_ret=0.20,
        ann_vol=0.10,
        sharpe=2.0,
        max_drawdown=max_drawdown,
        turnover_avg=turnover_avg,
        long_num_avg=1.0,
        short_num_avg=1.0,
        rank_ic_mean=rank_ic_mean,
        ic_mean=0.04,
        icir=icir,
        ic_t_stat=ic_t_stat,
        positive_ic_ratio=positive_ic_ratio,
        top_bottom_spread=0.012,
        after_cost_top_bottom_spread=after_cost_spread,
        before_cost_sharpe=before_cost_sharpe,
        after_cost_sharpe=after_cost_sharpe,
        metadata=metadata or {"point_in_time_passed": True},
    )


def _run(
    *,
    count: int = 30,
    before_returns: list[float] | None = None,
    after_returns: list[float] | None = None,
    turnover: float = 0.20,
    coverage_ratio: float = 0.95,
    aggregate: FactorBacktestResult | None = None,
) -> SingleFactorBacktestRun:
    dates = _dates(count)
    return SingleFactorBacktestRun(
        run_id="run-validation-fixture",
        factor_matrix_id="matrix-validation-fixture",
        factor_id="factor-validation-fixture",
        factor_version="v1",
        config_id="config-validation-fixture",
        start_date=dates[0],
        end_date=dates[-1],
        mode=BACKTEST_MODE_LONG_SHORT,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(dates),
        daily_records=_records(
            dates,
            before_returns=before_returns,
            after_returns=after_returns,
            turnover=turnover,
            coverage_ratio=coverage_ratio,
        ),
        aggregate_result=aggregate or _aggregate(dates, coverage_ratio=coverage_ratio),
        metadata={"offline_only": True},
    )


def _bundle(dates: list[str]) -> MatrixDataBundle:
    contract = MatrixDataContract(
        contract_id="contract-validation-fixture",
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=dates,
        required_fields=[FIELD_AMOUNT],
        field_sources={FIELD_AMOUNT: "fixture"},
        point_in_time_flags={FIELD_AMOUNT: True},
    )
    return MatrixDataBundle(
        bundle_id="bundle-validation-fixture",
        contract=contract,
        fields={FIELD_AMOUNT: [[10_000_000.0 for _ in dates], [9_000_000.0 for _ in dates]]},
        tradability_mask=[[True for _ in dates], [True for _ in dates]],
    )


def test_factor_slice_spec_validates_date_ranges() -> None:
    with pytest.raises(ValueError, match="start_date"):
        FactorSliceSpec(
            name="bad",
            start_date="2026-01-03",
            end_date="2026-01-01",
        )


def test_build_default_recent_slice_specs_includes_full_and_recent() -> None:
    specs = build_default_recent_slice_specs(_dates(30), min_sample_days=20)

    assert [spec.name for spec in specs] == [
        SLICE_FULL_SAMPLE,
        "recent_1y",
        "recent_3y",
        "recent_5y",
    ]


def test_build_regime_slice_specs_creates_deterministic_slices() -> None:
    dates = _dates(4)
    regimes = {dates[0]: "bear", dates[1]: "bull", dates[2]: "bull", dates[3]: "bear"}

    specs = build_regime_slice_specs(dates, regimes, min_sample_days=1)

    assert [spec.regime_label for spec in specs] == ["bear", "bull"]
    assert specs[0].metadata["date_list"] == [dates[0], dates[3]]


def test_evaluate_slice_result_computes_metrics() -> None:
    run = _run(count=3, before_returns=[0.01, -0.02, 0.03], after_returns=[0.009, -0.021, 0.029])
    spec = FactorSliceSpec(name="fixture", start_date=_dates(3)[0], end_date=_dates(3)[-1], min_sample_days=1)

    result = evaluate_slice_result(run, spec)

    assert result.before_cost_metrics.sample_count == 3
    assert result.after_cost_metrics.sample_count == 3
    assert result.excess_metrics is not None
    assert result.turnover_metrics.average_turnover == pytest.approx(0.20)
    assert result.coverage_ratio == pytest.approx(0.95)
    assert result.verdict == ROBUSTNESS_VERDICT_PASS


def test_insufficient_sample_days_fails_slice() -> None:
    result = evaluate_slice_result(_run(count=3), FactorSliceSpec(name="short", min_sample_days=5))

    assert result.verdict == ROBUSTNESS_VERDICT_FAIL
    assert "insufficient_sample_days" in result.warnings


def test_threshold_breach_warns_slice() -> None:
    result = evaluate_slice_result(
        _run(count=30, turnover=0.90, coverage_ratio=0.70),
        FactorSliceSpec(name="warn", min_sample_days=5),
        max_turnover=0.50,
        min_coverage_ratio=0.80,
    )

    assert result.verdict == ROBUSTNESS_VERDICT_WARN
    assert result.warnings == ["coverage_ratio_below_threshold", "turnover_budget_breach"]


def test_build_factor_robustness_report_aggregates_deterministically() -> None:
    run = _run(count=30, turnover=0.90)
    report = build_factor_robustness_report(
        run,
        generated_at="2026-04-27T00:00:00",
        max_turnover=0.50,
        min_sample_days=20,
    )

    assert report.overall_verdict == ROBUSTNESS_VERDICT_WARN
    assert report.failed_slices == []
    assert report.warning_slices == sorted(report.warning_slices)
    assert FactorRobustnessReport.from_dict(report.to_dict()).to_dict() == report.to_dict()


def test_enhanced_validation_report_strong_run_recommends_validated_research() -> None:
    run = _run(count=30)
    robustness = build_factor_robustness_report(run, generated_at="2026-04-27T00:00:00", min_sample_days=5)
    capacity = build_factor_cost_capacity_report(
        run,
        _bundle(_dates(30)),
        config=FactorCostCapacityConfig(config_id="capacity-pass", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )

    report = build_enhanced_factor_validation_report(
        run=run,
        robustness_report=robustness,
        cost_capacity_report=capacity,
        thresholds=FactorValidationThresholds(min_sample_days=20),
        generated_at="2026-04-27T00:00:00",
    )

    assert report.overall_verdict == VALIDATION_VERDICT_PASS
    assert report.recommended_status == FACTOR_STATUS_VALIDATED_RESEARCH
    assert report.recommended_status != FACTOR_STATUS_PRODUCTION
    assert report.capacity_snapshot["cost_capacity_report"]["verdict"] == CAPACITY_VERDICT_PASS
    assert report.metric_snapshot["robustness"]["overall_verdict"] == ROBUSTNESS_VERDICT_PASS


def test_enhanced_validation_report_capacity_warning_recommends_paper_trading() -> None:
    run = _run(count=30)
    robustness = build_factor_robustness_report(run, generated_at="2026-04-27T00:00:00", min_sample_days=5)
    capacity = build_factor_cost_capacity_report(
        run,
        _bundle(_dates(30)),
        config=FactorCostCapacityConfig(
            config_id="capacity-warn",
            target_capital=1_000_000.0,
            max_average_turnover=0.10,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert capacity.verdict == CAPACITY_VERDICT_WARN
    report = build_enhanced_factor_validation_report(
        run=run,
        robustness_report=robustness,
        cost_capacity_report=capacity,
        thresholds=FactorValidationThresholds(min_sample_days=20),
        generated_at="2026-04-27T00:00:00",
    )

    assert report.overall_verdict == VALIDATION_VERDICT_WARN
    assert report.recommended_status == FACTOR_STATUS_PAPER_TRADING
    assert report.gate_results["cost_capacity"] == VALIDATION_VERDICT_WARN


def test_enhanced_validation_report_hard_metric_failure_rejects() -> None:
    dates = _dates(30)
    weak_run = _run(
        count=30,
        aggregate=_aggregate(dates, rank_ic_mean=-0.01),
    )
    robustness = build_factor_robustness_report(weak_run, generated_at="2026-04-27T00:00:00", min_sample_days=5)
    capacity = build_factor_cost_capacity_report(
        weak_run,
        _bundle(dates),
        config=FactorCostCapacityConfig(config_id="capacity-pass", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )

    report = build_enhanced_factor_validation_report(
        run=weak_run,
        robustness_report=robustness,
        cost_capacity_report=capacity,
        thresholds=FactorValidationThresholds(min_sample_days=20),
        generated_at="2026-04-27T00:00:00",
    )

    assert report.overall_verdict == VALIDATION_VERDICT_FAIL
    assert report.recommended_status == FACTOR_STATUS_REJECTED
    assert "rank_ic_mean" in report.failed_gates


def test_enhanced_validation_report_missing_required_metrics_fail() -> None:
    dates = _dates(30)
    missing_run = _run(
        count=30,
        aggregate=_aggregate(
            dates,
            rank_ic_mean=None,
            icir=None,
            ic_t_stat=None,
            after_cost_sharpe=None,
            positive_ic_ratio=None,
            after_cost_spread=None,
        ),
    )
    robustness = build_factor_robustness_report(missing_run, generated_at="2026-04-27T00:00:00", min_sample_days=5)
    capacity = build_factor_cost_capacity_report(
        missing_run,
        _bundle(dates),
        config=FactorCostCapacityConfig(config_id="capacity-pass", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )

    report = build_enhanced_factor_validation_report(
        run=missing_run,
        robustness_report=robustness,
        cost_capacity_report=capacity,
        thresholds=FactorValidationThresholds(min_sample_days=20),
        generated_at="2026-04-27T00:00:00",
    )

    assert {
        "rank_ic_mean",
        "icir",
        "ic_t_stat",
        "after_cost_sharpe",
        "positive_ic_ratio",
        "after_cost_spread",
    }.issubset(set(report.failed_gates))
