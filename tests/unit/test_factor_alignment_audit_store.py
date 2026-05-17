from __future__ import annotations

import pytest

from quant_investor.factors.alignment_audit import (
    ALIGNMENT_AUDIT_PASS,
    ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
    FactorBacktestAlignmentAuditConfig,
    FactorBacktestAlignmentAuditReport,
    make_alignment_audit_config_id,
    render_alignment_audit_markdown,
)
from quant_investor.factors.backtest import EXECUTION_PRICE_VWAP
from quant_investor.factors.store import FactorAlignmentAuditStore


def _report() -> FactorBacktestAlignmentAuditReport:
    config = FactorBacktestAlignmentAuditConfig(
        config_id="placeholder",
        expected_policy=ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
        expected_delay_days=1,
        expected_holding_period_days=1,
        execution_price=EXECUTION_PRICE_VWAP,
        metadata={"fixture": True},
    )
    config.config_id = make_alignment_audit_config_id(config)
    return FactorBacktestAlignmentAuditReport(
        report_id="alignment-report-store-fixture",
        generated_at="2026-04-27T00:00:00",
        factor_matrix_id="factor-matrix-fixture",
        backtest_run_id="backtest-run-fixture",
        config=config,
        total_records=0,
        passed_records=0,
        failed_records=0,
        issue_count=0,
        blocker_count=0,
        warning_count=0,
        info_count=0,
        records=[],
        issues=[],
        verdict=ALIGNMENT_AUDIT_PASS,
        metadata={"fixture": True, "non_runtime_impact": True},
    )


def test_append_and_read_alignment_audit_report(tmp_path) -> None:
    store = FactorAlignmentAuditStore(tmp_path / "alignment")
    report = _report()

    store.append_alignment_audit_report(report)

    assert store.read_alignment_audit_reports()[0].to_dict() == report.to_dict()
    assert store.get_alignment_audit_report_ids() == {report.report_id}


def test_duplicate_report_id_raises(tmp_path) -> None:
    store = FactorAlignmentAuditStore(tmp_path / "alignment")
    report = _report()
    store.append_alignment_audit_report(report)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_alignment_audit_report(report)


def test_malformed_json_raises_clear_value_error(tmp_path) -> None:
    store = FactorAlignmentAuditStore(tmp_path / "alignment")
    store.alignment_audit_reports_path.parent.mkdir(parents=True, exist_ok=True)
    store.alignment_audit_reports_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_alignment_audit_reports()


def test_save_and_load_markdown(tmp_path) -> None:
    store = FactorAlignmentAuditStore(tmp_path / "alignment")
    markdown = render_alignment_audit_markdown(_report())

    path = store.save_alignment_audit_markdown(markdown)

    assert path == store.alignment_audit_markdown_path
    assert store.load_alignment_audit_markdown() == markdown


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "alignment"
    store = FactorAlignmentAuditStore(root)

    assert not root.exists()
    store.append_alignment_audit_report(_report())

    assert root.exists()
    assert store.alignment_audit_reports_path.exists()

