from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_SHORT,
    WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
)
from quant_investor.factors.contribution import (
    CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP,
    CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE,
    CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN,
    CONTRIBUTION_ISSUE_TURNOVER_INCREASE,
    CONTRIBUTION_VERDICT_DEGRADES,
    CONTRIBUTION_VERDICT_IMPROVES,
    CONTRIBUTION_VERDICT_INSUFFICIENT_DATA,
    CONTRIBUTION_VERDICT_NEUTRAL,
    FactorContributionConfig,
    FactorPortfolioContributionReport,
    build_factor_pool_return_series,
    build_factor_portfolio_contribution_report,
    build_incremental_factor_validation_snapshot,
)
from quant_investor.factors.correlation import (
    CORRELATION_VERDICT_DISTINCT,
    FactorCorrelationConfig,
    build_factor_redundancy_report,
)
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id


SYMBOLS = ["AAA", "BBB"]


def _dates(count: int) -> list[str]:
    start = date(2026, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _weight_matrix(*, run_id: str, factor_id: str, dates: list[str]) -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id=f"weights-{run_id}",
        factor_matrix_id=f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-contribution",
        symbols=SYMBOLS,
        dates=dates,
        long_weights=[[1.0 for _ in dates], [0.0 for _ in dates]],
        short_weights=[[0.0 for _ in dates], [-1.0 for _ in dates]],
        net_weights=[[1.0 for _ in dates], [-1.0 for _ in dates]],
        metadata={"run_id": run_id},
    )


def _aggregate(*, factor_id: str, dates: list[str]) -> FactorBacktestResult:
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id=factor_id,
            factor_version="v1",
            config_id="config-contribution",
        ),
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-contribution",
        start_date=dates[0],
        end_date=dates[-1],
        sample_days=len(dates),
        coverage_ratio=1.0,
        missing_ratio=0.0,
        ann_ret=0.10,
        ann_vol=0.10,
        sharpe=1.0,
        max_drawdown=0.02,
        turnover_avg=0.20,
        long_num_avg=1.0,
        short_num_avg=1.0,
        rank_ic_mean=0.03,
        ic_mean=0.03,
        icir=0.50,
        ic_t_stat=3.0,
        positive_ic_ratio=0.60,
        top_bottom_spread=0.01,
        after_cost_top_bottom_spread=0.009,
        before_cost_sharpe=1.2,
        after_cost_sharpe=1.0,
        metadata={"point_in_time_passed": True},
    )


def _run(
    *,
    run_id: str,
    factor_id: str,
    returns: list[float],
    turnover: float = 0.20,
) -> SingleFactorBacktestRun:
    dates = _dates(len(returns))
    records = [
        FactorDailyBacktestRecord(
            date=current_date,
            signal_date=current_date,
            execution_start_date=current_date,
            execution_end_date=current_date,
            long_return=value + 0.001,
            short_return=0.0,
            long_short_return=value + 0.001,
            after_cost_return=value,
            benchmark_return=0.0,
            excess_return=value,
            turnover=turnover,
            long_count=1,
            short_count=1,
            coverage_ratio=1.0,
            missing_ratio=0.0,
        )
        for current_date, value in zip(dates, returns)
    ]
    return SingleFactorBacktestRun(
        run_id=run_id,
        factor_matrix_id=f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-contribution",
        start_date=dates[0],
        end_date=dates[-1],
        mode=BACKTEST_MODE_LONG_SHORT,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(run_id=run_id, factor_id=factor_id, dates=dates),
        daily_records=records,
        aggregate_result=_aggregate(factor_id=factor_id, dates=dates),
        metadata={"offline_only": True},
    )


def test_factor_pool_return_series_equal_weights_missing_dates_and_turnover() -> None:
    run_a = _run(run_id="run-a", factor_id="a", returns=[0.01, 0.02, 0.03], turnover=0.10)
    run_b = _run(run_id="run-b", factor_id="b", returns=[0.03, 0.04], turnover=0.30)

    equal_weighted = build_factor_pool_return_series([run_a, run_b], name="baseline")

    assert equal_weighted.returns_by_date["2026-01-01"] == pytest.approx(0.02)
    assert equal_weighted.returns_by_date["2026-01-02"] == pytest.approx(0.03)
    assert equal_weighted.returns_by_date["2026-01-03"] == pytest.approx(0.03)
    assert equal_weighted.turnover_by_date["2026-01-01"] == pytest.approx(0.20)

    weighted = build_factor_pool_return_series(
        [run_a, run_b],
        name="weighted",
        weights_by_run_id={"run-a": 0.25, "run-b": 0.75},
    )
    assert weighted.returns_by_date["2026-01-01"] == pytest.approx(0.025)
    assert weighted.returns_by_date["2026-01-03"] == pytest.approx(0.03)


def test_contribution_report_improves_and_round_trips() -> None:
    baseline = _run(
        run_id="run-baseline",
        factor_id="baseline",
        returns=[0.01, -0.01, 0.01, -0.01],
    )
    candidate = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.02, 0.02, 0.02, 0.02],
    )

    report = build_factor_portfolio_contribution_report(
        candidate_run=candidate,
        baseline_runs=[baseline],
        config=FactorContributionConfig(config_id="improves", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )

    assert report.verdict == CONTRIBUTION_VERDICT_IMPROVES
    assert report.incremental_annualized_return is not None
    assert report.incremental_sharpe is not None
    assert report.combined_metrics.sample_count == 4
    assert (
        FactorPortfolioContributionReport.from_dict(report.to_dict()).to_dict()
        == report.to_dict()
    )


def test_contribution_report_degrades_on_negative_incremental_return() -> None:
    report = build_factor_portfolio_contribution_report(
        candidate_run=_run(
            run_id="run-candidate",
            factor_id="candidate",
            returns=[-0.05, -0.05, -0.05, -0.05],
        ),
        baseline_runs=[
            _run(run_id="run-baseline", factor_id="baseline", returns=[0.01, 0.01, 0.01, 0.01])
        ],
        config=FactorContributionConfig(config_id="degrades", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )

    assert CONTRIBUTION_ISSUE_NEGATIVE_INCREMENTAL_RETURN in report.issue_codes
    assert report.verdict == CONTRIBUTION_VERDICT_DEGRADES


def test_contribution_report_insufficient_overlap() -> None:
    report = build_factor_portfolio_contribution_report(
        candidate_run=_run(run_id="run-candidate", factor_id="candidate", returns=[0.01, 0.02]),
        baseline_runs=[
            _run(run_id="run-baseline", factor_id="baseline", returns=[0.01, 0.02])
        ],
        config=FactorContributionConfig(config_id="insufficient", min_overlap_days=5),
        generated_at="2026-04-27T00:00:00",
    )

    assert CONTRIBUTION_ISSUE_INSUFFICIENT_OVERLAP in report.issue_codes
    assert report.verdict == CONTRIBUTION_VERDICT_INSUFFICIENT_DATA


def test_contribution_report_low_sharpe_is_neutral() -> None:
    baseline = _run(
        run_id="run-baseline",
        factor_id="baseline",
        returns=[0.01, -0.01, 0.01, -0.01],
    )
    candidate = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.01, -0.01, 0.01, -0.01],
    )

    report = build_factor_portfolio_contribution_report(
        candidate_run=candidate,
        baseline_runs=[baseline],
        config=FactorContributionConfig(config_id="neutral", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )

    assert CONTRIBUTION_ISSUE_LOW_INCREMENTAL_SHARPE in report.issue_codes
    assert report.verdict == CONTRIBUTION_VERDICT_NEUTRAL


def test_contribution_report_turnover_increase_is_detected() -> None:
    baseline = _run(
        run_id="run-baseline",
        factor_id="baseline",
        returns=[0.01, -0.01, 0.01, -0.01],
        turnover=0.10,
    )
    candidate = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.01, -0.01, 0.01, -0.01],
        turnover=0.80,
    )

    report = build_factor_portfolio_contribution_report(
        candidate_run=candidate,
        baseline_runs=[baseline],
        config=FactorContributionConfig(
            config_id="turnover",
            min_overlap_days=2,
            min_incremental_sharpe=-1.0,
            max_turnover_increase=0.01,
        ),
        generated_at="2026-04-27T00:00:00",
    )

    assert CONTRIBUTION_ISSUE_TURNOVER_INCREASE in report.issue_codes
    assert report.verdict == CONTRIBUTION_VERDICT_NEUTRAL


def test_incremental_validation_snapshot_is_json_serializable() -> None:
    baseline = _run(
        run_id="run-baseline",
        factor_id="baseline",
        returns=[0.01, -0.01, 0.01, -0.01],
    )
    candidate = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.02, 0.02, 0.02, 0.02],
    )
    redundancy = build_factor_redundancy_report(
        candidate_run=candidate,
        reference_runs=[baseline],
        config=FactorCorrelationConfig(
            config_id="snapshot",
            min_overlap_days=2,
            max_return_correlation=1.0,
            min_residual_mean_return=-1.0,
        ),
        generated_at="2026-04-27T00:00:00",
    )
    contribution = build_factor_portfolio_contribution_report(
        candidate_run=candidate,
        baseline_runs=[baseline],
        config=FactorContributionConfig(config_id="snapshot", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )

    snapshot = build_incremental_factor_validation_snapshot(
        redundancy_report=redundancy,
        contribution_report=contribution,
    )

    assert snapshot["redundancy_verdict"] == CORRELATION_VERDICT_DISTINCT
    assert snapshot["contribution_verdict"] == CONTRIBUTION_VERDICT_IMPROVES
    assert "incremental_sharpe" in snapshot
    json.dumps(snapshot, sort_keys=True, allow_nan=False)
