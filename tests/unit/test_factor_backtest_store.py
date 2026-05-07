from __future__ import annotations

import pytest

from quant_investor.factors.backtest import (
    BACKTEST_MODE_LONG_SHORT,
    WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
)
from quant_investor.factors.schema import FactorBacktestResult, make_backtest_result_id
from quant_investor.factors.store import FactorBacktestArtifactStore


SYMBOLS = ["AAA", "BBB"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03"]


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
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        short_weights=[
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        net_weights=[
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        metadata={"mode": BACKTEST_MODE_LONG_SHORT},
    )


def _daily_record(run_id: str = "run-fixture", date: str = "2026-01-03") -> FactorDailyBacktestRecord:
    return FactorDailyBacktestRecord(
        date=date,
        signal_date="2026-01-01",
        execution_start_date="2026-01-02",
        execution_end_date=date,
        long_return=0.04,
        short_return=0.01,
        long_short_return=0.05,
        after_cost_return=0.049,
        benchmark_return=0.02,
        excess_return=0.029,
        turnover=1.0,
        long_count=1,
        short_count=1,
        coverage_ratio=1.0,
        missing_ratio=0.0,
        metadata={"run_id": run_id},
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
        start_date="2026-01-01",
        end_date="2026-01-03",
        sample_days=1,
        coverage_ratio=1.0,
        missing_ratio=0.0,
        ann_ret=0.10,
        ann_vol=0.20,
        sharpe=0.50,
        max_drawdown=0.0,
        turnover_avg=1.0,
        long_num_avg=1.0,
        short_num_avg=1.0,
        top_bottom_spread=0.05,
        after_cost_top_bottom_spread=0.049,
        metadata={"point_in_time_passed": True},
    )


def _run() -> SingleFactorBacktestRun:
    return SingleFactorBacktestRun(
        run_id="run-fixture",
        factor_matrix_id="matrix-fixture",
        factor_id="factor-fixture",
        factor_version="v1",
        config_id="config-fixture",
        start_date="2026-01-01",
        end_date="2026-01-03",
        mode=BACKTEST_MODE_LONG_SHORT,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(),
        daily_records=[_daily_record()],
        aggregate_result=_aggregate_result(),
        metadata={"offline_only": True},
    )


def test_append_and_read_weight_matrix(tmp_path) -> None:
    store = FactorBacktestArtifactStore(tmp_path / "backtest_store")
    matrix = _weight_matrix()

    store.append_weight_matrix(matrix)

    assert store.read_weight_matrices()[0].to_dict() == matrix.to_dict()
    assert store.get_weight_matrix_ids() == {matrix.weights_id}


def test_append_and_read_backtest_run(tmp_path) -> None:
    store = FactorBacktestArtifactStore(tmp_path / "backtest_store")
    run = _run()

    store.append_backtest_run(run)

    assert store.read_backtest_runs()[0].to_dict() == run.to_dict()
    assert store.get_backtest_run_ids() == {run.run_id}


def test_append_and_read_daily_records(tmp_path) -> None:
    store = FactorBacktestArtifactStore(tmp_path / "backtest_store")
    records = [_daily_record(), _daily_record(run_id="run-other")]

    count = store.append_daily_records(records)

    assert count == 2
    assert [record.to_dict() for record in store.read_daily_records()] == [
        record.to_dict() for record in records
    ]


def test_duplicate_ids_raise_value_error(tmp_path) -> None:
    store = FactorBacktestArtifactStore(tmp_path / "backtest_store")
    matrix = _weight_matrix()
    run = _run()
    record = _daily_record()

    store.append_weight_matrix(matrix)
    store.append_backtest_run(run)
    store.append_daily_records([record])

    with pytest.raises(ValueError, match="Duplicate weights_id"):
        store.append_weight_matrix(matrix)
    with pytest.raises(ValueError, match="Duplicate run_id"):
        store.append_backtest_run(run)
    with pytest.raises(ValueError, match="Duplicate factor daily backtest record"):
        store.append_daily_records([record])


def test_malformed_json_raises_clear_error(tmp_path) -> None:
    store = FactorBacktestArtifactStore(tmp_path / "backtest_store")
    store.weight_matrices_path.parent.mkdir(parents=True, exist_ok=True)
    store.weight_matrices_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_weight_matrices()


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "backtest_store"
    store = FactorBacktestArtifactStore(root)

    assert not root.exists()
    store.append_weight_matrix(_weight_matrix())

    assert root.exists()
    assert store.weight_matrices_path.exists()
