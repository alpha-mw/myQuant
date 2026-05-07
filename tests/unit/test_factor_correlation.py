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
from quant_investor.factors.correlation import (
    CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION,
    CORRELATION_ISSUE_HIGH_RETURN_CORRELATION,
    CORRELATION_ISSUE_INSUFFICIENT_OVERLAP,
    CORRELATION_ISSUE_LOW_RESIDUAL_RETURN,
    CORRELATION_VERDICT_DISTINCT,
    CORRELATION_VERDICT_INSUFFICIENT_DATA,
    CORRELATION_VERDICT_REDUNDANT,
    CORRELATION_VERDICT_RELATED,
    FactorCorrelationConfig,
    FactorCorrelationPair,
    average_matrix_rank_correlation,
    build_factor_redundancy_report,
    evaluate_factor_correlation_pair,
    pearson_correlation,
    simple_residual_series,
    spearman_correlation,
    align_series_by_date,
)
from quant_investor.factors.matrix import FactorMatrix, compute_coverage
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id


SYMBOLS = ["AAA", "BBB", "CCC"]


def _dates(count: int) -> list[str]:
    start = date(2026, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _weight_matrix(*, run_id: str, factor_id: str, dates: list[str]) -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id=f"weights-{run_id}",
        factor_matrix_id=f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-correlation",
        symbols=SYMBOLS[:2],
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
            config_id="config-correlation",
        ),
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-correlation",
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
    ic: list[float] | None = None,
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
            turnover=0.20,
            long_count=1,
            short_count=1,
            coverage_ratio=1.0,
            missing_ratio=0.0,
            metadata={"rank_ic": ic[index]} if ic is not None else {},
        )
        for index, (current_date, value) in enumerate(zip(dates, returns))
    ]
    return SingleFactorBacktestRun(
        run_id=run_id,
        factor_matrix_id=f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-correlation",
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


def _matrix(
    *,
    matrix_id: str,
    factor_id: str,
    values: list[list[float | None]],
    symbols: list[str] | None = None,
) -> FactorMatrix:
    dates = _dates(len(values[0]))
    resolved_symbols = symbols or SYMBOLS
    coverage_ratio, missing_ratio = compute_coverage(values)
    return FactorMatrix(
        matrix_id=matrix_id,
        factor_id=factor_id,
        factor_version="v1",
        expression=factor_id,
        symbols=resolved_symbols,
        dates=dates,
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        metadata={"fixture": True},
    )


def test_correlation_numeric_helpers_known_cases() -> None:
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman_correlation([10.0, 20.0, 30.0], [1.0, 3.0, 2.0]) == pytest.approx(0.5)
    assert pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    residuals = simple_residual_series([1.0, 3.0, 5.0], [0.0, 1.0, 2.0])
    assert residuals == pytest.approx([0.0, 0.0, 0.0])

    dates, left, right = align_series_by_date(
        {"2026-01-02": 2.0, "2026-01-01": 1.0},
        {"2026-01-03": 3.0, "2026-01-01": 10.0},
    )
    assert dates == ["2026-01-01"]
    assert left == [1.0]
    assert right == [10.0]


def test_average_matrix_rank_correlation_and_insufficient_symbols() -> None:
    candidate = _matrix(
        matrix_id="matrix-candidate",
        factor_id="candidate",
        values=[[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]],
    )
    reference = _matrix(
        matrix_id="matrix-reference",
        factor_id="reference",
        values=[[10.0, 30.0], [20.0, 20.0], [30.0, 10.0]],
    )

    assert average_matrix_rank_correlation(candidate, reference) == pytest.approx(1.0)
    assert average_matrix_rank_correlation(
        candidate,
        _matrix(
            matrix_id="matrix-one-symbol",
            factor_id="one-symbol",
            values=[[10.0, 20.0]],
            symbols=["AAA"],
        ),
        min_common_symbols=2,
    ) is None


def test_correlation_pair_verdicts_and_round_trip() -> None:
    config = FactorCorrelationConfig(config_id="fixture", min_overlap_days=2)
    candidate = _run(run_id="run-candidate", factor_id="candidate", returns=[0.01, 0.02, 0.03])
    reference = _run(run_id="run-reference", factor_id="reference", returns=[0.02, 0.04, 0.06])

    redundant = evaluate_factor_correlation_pair(candidate, reference, config=config)

    assert CORRELATION_ISSUE_HIGH_RETURN_CORRELATION in redundant.issue_codes
    assert CORRELATION_ISSUE_LOW_RESIDUAL_RETURN in redundant.issue_codes
    assert redundant.verdict == CORRELATION_VERDICT_REDUNDANT
    assert FactorCorrelationPair.from_dict(redundant.to_dict()).to_dict() == redundant.to_dict()

    related = evaluate_factor_correlation_pair(
        None,
        None,
        candidate_matrix=_matrix(
            matrix_id="matrix-candidate",
            factor_id="candidate",
            values=[[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]],
        ),
        reference_matrix=_matrix(
            matrix_id="matrix-related",
            factor_id="related",
            values=[[10.0, 20.0], [20.0, 10.0], [30.0, 30.0]],
        ),
        config=config,
    )
    assert CORRELATION_ISSUE_HIGH_MATRIX_CORRELATION in related.issue_codes
    assert related.verdict == CORRELATION_VERDICT_RELATED

    insufficient = evaluate_factor_correlation_pair(
        candidate,
        reference,
        config=FactorCorrelationConfig(config_id="strict", min_overlap_days=5),
    )
    assert CORRELATION_ISSUE_INSUFFICIENT_OVERLAP in insufficient.issue_codes
    assert insufficient.verdict == CORRELATION_VERDICT_INSUFFICIENT_DATA

    distinct = evaluate_factor_correlation_pair(
        _run(run_id="run-distinct-candidate", factor_id="candidate", returns=[0.01, -0.01, 0.02]),
        _run(run_id="run-distinct-reference", factor_id="distinct", returns=[0.03, 0.01, -0.02]),
        config=FactorCorrelationConfig(
            config_id="distinct",
            min_overlap_days=2,
            max_return_correlation=1.0,
            min_residual_mean_return=-1.0,
        ),
    )
    assert distinct.verdict == CORRELATION_VERDICT_DISTINCT


def test_redundancy_report_aggregates_pairs_and_ids() -> None:
    candidate_run = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.01, 0.02, 0.03],
    )
    reference_run = _run(
        run_id="run-redundant",
        factor_id="ref-redundant",
        returns=[0.02, 0.04, 0.06],
    )
    candidate_matrix = _matrix(
        matrix_id="matrix-candidate",
        factor_id="candidate",
        values=[[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]],
    )
    matching_matrix = _matrix(
        matrix_id="matrix-ref-redundant",
        factor_id="ref-redundant",
        values=[[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]],
    )
    related_matrix = _matrix(
        matrix_id="matrix-ref-related",
        factor_id="ref-related",
        values=[[10.0, 20.0], [20.0, 10.0], [30.0, 30.0]],
    )
    config = FactorCorrelationConfig(config_id="report", min_overlap_days=2)

    report = build_factor_redundancy_report(
        candidate_run=candidate_run,
        reference_runs=[reference_run],
        candidate_matrix=candidate_matrix,
        reference_matrices=[matching_matrix, related_matrix],
        config=config,
        generated_at="2026-04-27T00:00:00",
    )
    same_id = build_factor_redundancy_report(
        candidate_run=candidate_run,
        reference_runs=[reference_run],
        candidate_matrix=candidate_matrix,
        reference_matrices=[matching_matrix, related_matrix],
        config=config,
        generated_at="2026-04-27T00:00:00",
    ).report_id

    assert report.report_id == same_id
    assert report.overall_verdict == CORRELATION_VERDICT_REDUNDANT
    assert report.max_abs_return_correlation == pytest.approx(1.0)
    assert report.max_abs_matrix_rank_correlation == pytest.approx(1.0)
    assert report.redundant_factor_ids == ["ref-redundant"]
    assert report.related_factor_ids == ["ref-related"]
    assert report.to_dict() == type(report).from_dict(report.to_dict()).to_dict()
