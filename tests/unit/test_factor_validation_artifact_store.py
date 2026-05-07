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
from quant_investor.factors.capacity import FactorCostCapacityConfig, build_factor_cost_capacity_report
from quant_investor.factors.matrix import FIELD_AMOUNT, MatrixDataBundle, MatrixDataContract
from quant_investor.factors.robustness import (
    build_enhanced_factor_validation_report,
    build_factor_robustness_report,
)
from quant_investor.factors.schema import (
    FactorBacktestResult,
    FactorValidationThresholds,
    make_backtest_result_id,
)
from quant_investor.factors.store import FactorValidationArtifactStore


SYMBOLS = ["AAA", "BBB"]


def _dates(count: int = 3) -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _weight_matrix(dates: list[str]) -> FactorWeightMatrix:
    return FactorWeightMatrix(
        weights_id="weights-store-validation",
        factor_matrix_id="matrix-store-validation",
        factor_id="factor-store-validation",
        factor_version="v1",
        config_id="config-store-validation",
        symbols=SYMBOLS,
        dates=dates,
        long_weights=[[1.0 for _ in dates], [0.0 for _ in dates]],
        short_weights=[[0.0 for _ in dates], [-1.0 for _ in dates]],
        net_weights=[[1.0 for _ in dates], [-1.0 for _ in dates]],
    )


def _run() -> SingleFactorBacktestRun:
    dates = _dates()
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
            turnover=0.20,
            long_count=1,
            short_count=1,
            coverage_ratio=0.95,
            missing_ratio=0.05,
        )
        for current_date in dates
    ]
    aggregate = FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id="factor-store-validation",
            factor_version="v1",
            config_id="config-store-validation",
        ),
        factor_id="factor-store-validation",
        factor_version="v1",
        config_id="config-store-validation",
        start_date=dates[0],
        end_date=dates[-1],
        sample_days=len(dates),
        coverage_ratio=0.95,
        missing_ratio=0.05,
        ann_ret=0.20,
        ann_vol=0.10,
        sharpe=2.0,
        max_drawdown=0.02,
        turnover_avg=0.20,
        long_num_avg=1.0,
        short_num_avg=1.0,
        rank_ic_mean=0.04,
        ic_mean=0.04,
        icir=0.60,
        ic_t_stat=4.0,
        positive_ic_ratio=0.65,
        top_bottom_spread=0.012,
        after_cost_top_bottom_spread=0.010,
        before_cost_sharpe=1.30,
        after_cost_sharpe=1.10,
        metadata={"point_in_time_passed": True},
    )
    return SingleFactorBacktestRun(
        run_id="run-store-validation",
        factor_matrix_id="matrix-store-validation",
        factor_id="factor-store-validation",
        factor_version="v1",
        config_id="config-store-validation",
        start_date=dates[0],
        end_date=dates[-1],
        mode=BACKTEST_MODE_LONG_SHORT,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=_weight_matrix(dates),
        daily_records=records,
        aggregate_result=aggregate,
        metadata={"offline_only": True},
    )


def _bundle() -> MatrixDataBundle:
    dates = _dates()
    contract = MatrixDataContract(
        contract_id="contract-store-validation",
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=dates,
        required_fields=[FIELD_AMOUNT],
        field_sources={FIELD_AMOUNT: "fixture"},
        point_in_time_flags={FIELD_AMOUNT: True},
    )
    return MatrixDataBundle(
        bundle_id="bundle-store-validation",
        contract=contract,
        fields={FIELD_AMOUNT: [[10_000_000.0 for _ in dates], [9_000_000.0 for _ in dates]]},
        tradability_mask=[[True for _ in dates], [True for _ in dates]],
    )


def _reports():
    run = _run()
    robustness = build_factor_robustness_report(
        run,
        generated_at="2026-04-27T00:00:00",
        min_sample_days=1,
    )
    capacity = build_factor_cost_capacity_report(
        run,
        _bundle(),
        config=FactorCostCapacityConfig(config_id="store-capacity", target_capital=100_000.0),
        generated_at="2026-04-27T00:00:00",
    )
    enhanced = build_enhanced_factor_validation_report(
        run=run,
        robustness_report=robustness,
        cost_capacity_report=capacity,
        thresholds=FactorValidationThresholds(min_sample_days=1),
        generated_at="2026-04-27T00:00:00",
    )
    return robustness, capacity, enhanced


def test_append_and_read_robustness_report(tmp_path) -> None:
    store = FactorValidationArtifactStore(tmp_path / "validation")
    robustness, _capacity, _enhanced = _reports()

    store.append_robustness_report(robustness)

    assert store.read_robustness_reports()[0].to_dict() == robustness.to_dict()
    assert store.get_robustness_report_ids() == {robustness.report_id}


def test_append_and_read_cost_capacity_report(tmp_path) -> None:
    store = FactorValidationArtifactStore(tmp_path / "validation")
    _robustness, capacity, _enhanced = _reports()

    store.append_cost_capacity_report(capacity)

    assert store.read_cost_capacity_reports()[0].to_dict() == capacity.to_dict()
    assert store.get_cost_capacity_report_ids() == {capacity.report_id}


def test_append_and_read_enhanced_validation_report(tmp_path) -> None:
    store = FactorValidationArtifactStore(tmp_path / "validation")
    _robustness, _capacity, enhanced = _reports()

    store.append_enhanced_validation_report(enhanced)

    assert store.read_enhanced_validation_reports()[0].to_dict() == enhanced.to_dict()
    assert store.get_enhanced_validation_report_ids() == {enhanced.report_id}


def test_duplicate_ids_raise_value_error(tmp_path) -> None:
    store = FactorValidationArtifactStore(tmp_path / "validation")
    robustness, capacity, enhanced = _reports()

    store.append_robustness_report(robustness)
    store.append_cost_capacity_report(capacity)
    store.append_enhanced_validation_report(enhanced)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_robustness_report(robustness)
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_cost_capacity_report(capacity)
    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_enhanced_validation_report(enhanced)


def test_malformed_json_raises_value_error(tmp_path) -> None:
    store = FactorValidationArtifactStore(tmp_path / "validation")
    store.robustness_reports_path.parent.mkdir(parents=True, exist_ok=True)
    store.robustness_reports_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_robustness_reports()


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "validation"
    store = FactorValidationArtifactStore(root)
    robustness, _capacity, _enhanced = _reports()

    assert not root.exists()
    store.append_robustness_report(robustness)

    assert root.exists()
    assert store.robustness_reports_path.exists()
