from __future__ import annotations

import pytest

from quant_investor.factors.execution_cost import (
    FactorExecutionCostConfig,
    build_execution_adjusted_backtest_run,
    build_execution_cost_dashboard_payload,
    build_execution_cost_simulation_report,
    make_execution_cost_config_id,
    render_execution_cost_report_markdown,
)
from quant_investor.factors.store import FactorExecutionCostSimulationStore

from tests.unit.test_factor_execution_cost import _bundle, _mask, _run


def _config() -> FactorExecutionCostConfig:
    config = FactorExecutionCostConfig(
        config_id="placeholder",
        impact_coefficient=0.0,
        high_cost_warning_threshold=1.0,
    )
    config.config_id = make_execution_cost_config_id(config)
    return config


def _report():
    return build_execution_cost_simulation_report(
        run=_run(),
        bundle=_bundle(),
        tradability_mask=_mask(),
        config=_config(),
        portfolio_value=100000.0,
        generated_at="2026-04-27T00:00:00Z",
    )


def test_append_and_read_execution_cost_report(tmp_path) -> None:
    store = FactorExecutionCostSimulationStore(tmp_path / "execution_cost")
    report = _report()

    store.append_execution_cost_report(report)

    assert store.read_execution_cost_reports()[0].to_dict() == report.to_dict()
    assert store.get_execution_cost_report_ids() == {report.report_id}


def test_append_and_read_adjusted_run_and_daily_records(tmp_path) -> None:
    store = FactorExecutionCostSimulationStore(tmp_path / "execution_cost")
    report = _report()
    adjusted = build_execution_adjusted_backtest_run(
        report,
        source_backtest_run_id="run-fixture",
        generated_at="2026-04-27T00:00:01Z",
    )

    store.append_execution_adjusted_run(adjusted)
    count = store.append_daily_execution_cost_records(report.daily_records)

    assert count == len(report.daily_records)
    assert store.read_execution_adjusted_runs()[0].to_dict() == adjusted.to_dict()
    assert [record.to_dict() for record in store.read_daily_execution_cost_records()] == [
        record.to_dict() for record in report.daily_records
    ]
    assert store.get_execution_adjusted_run_ids() == {adjusted.adjusted_run_id}


def test_duplicate_report_and_adjusted_run_ids_raise(tmp_path) -> None:
    store = FactorExecutionCostSimulationStore(tmp_path / "execution_cost")
    report = _report()
    adjusted = build_execution_adjusted_backtest_run(
        report,
        source_backtest_run_id="run-fixture",
        generated_at="2026-04-27T00:00:01Z",
    )

    store.append_execution_cost_report(report)
    store.append_execution_adjusted_run(adjusted)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_execution_cost_report(report)
    with pytest.raises(ValueError, match="Duplicate adjusted_run_id"):
        store.append_execution_adjusted_run(adjusted)


def test_malformed_json_raises_clear_value_error(tmp_path) -> None:
    store = FactorExecutionCostSimulationStore(tmp_path / "execution_cost")
    store.execution_cost_reports_path.parent.mkdir(parents=True, exist_ok=True)
    store.execution_cost_reports_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_execution_cost_reports()


def test_save_and_load_markdown_dashboard_and_create_directories(tmp_path) -> None:
    root = tmp_path / "missing" / "execution_cost"
    store = FactorExecutionCostSimulationStore(root)
    report = _report()
    markdown = render_execution_cost_report_markdown(report)
    dashboard = build_execution_cost_dashboard_payload(report)

    markdown_path = store.save_execution_cost_markdown(markdown)
    dashboard_path = store.save_execution_cost_dashboard(dashboard)

    assert markdown_path == store.execution_cost_markdown_path
    assert dashboard_path == store.execution_cost_dashboard_path
    assert store.load_execution_cost_markdown() == markdown
    assert store.load_execution_cost_dashboard() == dashboard
    assert root.exists()
