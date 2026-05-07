from __future__ import annotations

import pytest

from quant_investor.factors.admission import (
    build_production_factor_library,
    evaluate_backtest_against_thresholds,
    propose_admission_decision,
)
from quant_investor.factors.schema import (
    FACTOR_FAMILY_MOMENTUM,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_RESEARCH_CANDIDATE,
    FactorBacktestResult,
    FactorDefinition,
    FactorLibraryEntry,
    make_backtest_result_id,
    make_factor_id,
)
from quant_investor.factors.store import FactorGovernanceStore


def _definition() -> FactorDefinition:
    expression = "close / delay(close, 60) - 1"
    return FactorDefinition(
        factor_id=make_factor_id(
            factor_name="Sixty Day Momentum",
            factor_family=FACTOR_FAMILY_MOMENTUM,
            expression=expression,
        ),
        factor_name="Sixty Day Momentum",
        factor_family=FACTOR_FAMILY_MOMENTUM,
        status=FACTOR_STATUS_RESEARCH_CANDIDATE,
        version="v1",
        expression=expression,
        input_fields=["trade_date", "close"],
        data_sources=["local_csv"],
        universe="CN",
        benchmark="CSI300",
        expected_direction=1.0,
        rebalance_frequency="weekly",
        lookback_window=60,
        delay_days=1,
        execution_price="next_open",
        economic_rationale="Momentum captures persistent medium-horizon trend behavior.",
        created_at="2026-04-27",
    )


def _result() -> FactorBacktestResult:
    result_id = make_backtest_result_id(
        factor_id="factor-momentum-test",
        factor_version="v1",
        config_id="config-weekly",
    )
    return FactorBacktestResult(
        result_id=result_id,
        factor_id="factor-momentum-test",
        factor_version="v1",
        config_id="config-weekly",
        start_date="2021-01-01",
        end_date="2025-12-31",
        sample_days=1000,
        coverage_ratio=0.92,
        missing_ratio=0.08,
        rank_ic_mean=0.035,
        icir=0.50,
        ic_t_stat=4.0,
        positive_ic_ratio=0.62,
        after_cost_top_bottom_spread=0.06,
        after_cost_sharpe=0.95,
        metadata={"point_in_time_passed": True, "validation_generated_at": "2026-04-27"},
    )


def test_append_and_read_factor_definition(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    definition = _definition()

    store.append_factor_definition(definition)

    assert store.read_factor_definitions()[0].to_dict() == definition.to_dict()
    assert store.get_factor_definition_ids() == {definition.factor_id}


def test_append_and_read_backtest_result(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    result = _result()

    store.append_backtest_result(result)

    assert store.read_backtest_results()[0].to_dict() == result.to_dict()
    assert store.get_backtest_result_ids() == {result.result_id}


def test_append_and_read_validation_report(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    report = evaluate_backtest_against_thresholds(_result())

    store.append_validation_report(report)

    assert store.read_validation_reports()[0].to_dict() == report.to_dict()
    assert store.get_validation_report_ids() == {report.report_id}


def test_append_and_read_admission_decision(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    report = evaluate_backtest_against_thresholds(_result())
    decision = propose_admission_decision(report, decided_at="2026-04-27")

    store.append_admission_decision(decision)

    assert store.read_admission_decisions()[0].to_dict() == decision.to_dict()
    assert store.get_admission_decision_ids() == {decision.decision_id}


def test_duplicate_ids_raise_on_append(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    definition = _definition()
    store.append_factor_definition(definition)

    with pytest.raises(ValueError, match="Duplicate factor_id"):
        store.append_factor_definition(definition)


def test_malformed_json_raises_clear_error(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    store.factor_definitions_path.parent.mkdir(parents=True, exist_ok=True)
    store.factor_definitions_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_factor_definitions()


def test_save_and_load_production_library(tmp_path) -> None:
    store = FactorGovernanceStore(tmp_path / "factor_store")
    entry = FactorLibraryEntry(
        factor_id="factor-a",
        factor_version="v1",
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id="decision-a",
        validation_report_id="report-a",
        production_since="2026-04-27",
    )
    library = build_production_factor_library([entry], generated_at="2026-04-27")

    path = store.save_production_library(library)
    loaded = store.load_production_library()

    assert path == store.production_library_path
    assert loaded.to_dict() == library.to_dict()


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "factor_store"
    store = FactorGovernanceStore(root)

    assert not root.exists()
    store.append_factor_definition(_definition())

    assert root.exists()
    assert store.factor_definitions_path.exists()
