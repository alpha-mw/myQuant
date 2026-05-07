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
from quant_investor.factors.contribution import (
    FactorContributionConfig,
    build_factor_portfolio_contribution_report,
)
from quant_investor.factors.correlation import (
    FactorCorrelationConfig,
    build_factor_redundancy_report,
)
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id
from quant_investor.factors.store import FactorCorrelationContributionStore


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
        config_id="config-incremental-store",
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
            config_id="config-incremental-store",
        ),
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-incremental-store",
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


def _run(*, run_id: str, factor_id: str, returns: list[float]) -> SingleFactorBacktestRun:
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
        )
        for current_date, value in zip(dates, returns)
    ]
    return SingleFactorBacktestRun(
        run_id=run_id,
        factor_matrix_id=f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        config_id="config-incremental-store",
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


def _reports():
    candidate = _run(
        run_id="run-candidate",
        factor_id="candidate",
        returns=[0.01, 0.02, 0.03],
    )
    baseline = _run(
        run_id="run-baseline",
        factor_id="baseline",
        returns=[0.02, 0.04, 0.06],
    )
    redundancy = build_factor_redundancy_report(
        candidate_run=candidate,
        reference_runs=[baseline],
        config=FactorCorrelationConfig(config_id="store", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )
    contribution = build_factor_portfolio_contribution_report(
        candidate_run=candidate,
        baseline_runs=[baseline],
        config=FactorContributionConfig(config_id="store", min_overlap_days=2),
        generated_at="2026-04-27T00:00:00",
    )
    return redundancy, contribution


def test_append_and_read_incremental_reports(tmp_path) -> None:
    store = FactorCorrelationContributionStore(tmp_path / "incremental")
    redundancy, contribution = _reports()

    store.append_redundancy_report(redundancy)
    store.append_contribution_report(contribution)

    assert store.read_redundancy_reports()[0].to_dict() == redundancy.to_dict()
    assert store.read_contribution_reports()[0].to_dict() == contribution.to_dict()
    assert store.get_redundancy_report_ids() == {redundancy.report_id}
    assert store.get_contribution_report_ids() == {contribution.report_id}


def test_incremental_store_duplicate_ids_raise_value_error(tmp_path) -> None:
    store = FactorCorrelationContributionStore(tmp_path / "incremental")
    redundancy, contribution = _reports()

    store.append_redundancy_report(redundancy)
    store.append_contribution_report(contribution)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_redundancy_report(redundancy)
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_contribution_report(contribution)


def test_incremental_store_malformed_json_raises_value_error(tmp_path) -> None:
    store = FactorCorrelationContributionStore(tmp_path / "incremental")
    store.redundancy_reports_path.parent.mkdir(parents=True, exist_ok=True)
    store.redundancy_reports_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_redundancy_reports()


def test_incremental_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "incremental"
    store = FactorCorrelationContributionStore(root)
    redundancy, _contribution = _reports()

    assert not root.exists()
    store.append_redundancy_report(redundancy)

    assert root.exists()
    assert store.redundancy_reports_path.exists()
