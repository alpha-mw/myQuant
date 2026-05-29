import json

import pytest

from quant_investor.factors.evidence import (
    FactorAuditEvidenceSnapshot,
    FactorEvidenceCollectionConfig,
    FactorShadowEvidenceDateResult,
    MultiDateFactorEvidenceReport,
)
from quant_investor.factors.store import FactorEvidenceStore


def _date_result(result_id: str = "date-result-1") -> FactorShadowEvidenceDateResult:
    return FactorShadowEvidenceDateResult(
        result_id=result_id,
        as_of="2026-04-01",
        audit_snapshot=FactorAuditEvidenceSnapshot(as_of="2026-04-01"),
    )


def _report(report_id: str = "report-1") -> MultiDateFactorEvidenceReport:
    return MultiDateFactorEvidenceReport(
        report_id=report_id,
        generated_at="2026-04-03T00:00:00Z",
        config=FactorEvidenceCollectionConfig(as_of_dates=["2026-04-01"], min_observation_days=1),
        observation_days=1,
        start_date="2026-04-01",
        end_date="2026-04-01",
        date_results=[_date_result()],
    )


def test_factor_evidence_store_append_read_and_duplicates(tmp_path) -> None:
    store = FactorEvidenceStore(tmp_path / "evidence")
    result = _date_result()
    report = _report()
    store.append_date_result(result)
    store.append_multi_date_report(report)

    assert store.read_date_results()[0].result_id == result.result_id
    assert store.read_multi_date_reports()[0].report_id == report.report_id
    assert store.get_date_result_ids() == {result.result_id}
    assert store.get_multi_date_report_ids() == {report.report_id}
    with pytest.raises(ValueError):
        store.append_date_result(result)
    with pytest.raises(ValueError):
        store.append_multi_date_report(report)


def test_factor_evidence_store_rejects_malformed_json(tmp_path) -> None:
    store = FactorEvidenceStore(tmp_path)
    store.date_results_path.parent.mkdir(parents=True, exist_ok=True)
    store.date_results_path.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_date_results()


def test_factor_evidence_store_markdown_dashboard_and_directory_creation(tmp_path) -> None:
    store = FactorEvidenceStore(tmp_path / "nested" / "evidence")
    markdown_path = store.save_evidence_markdown("# Evidence\n")
    dashboard_path = store.save_evidence_dashboard({"status": "ok"})
    assert markdown_path.exists()
    assert dashboard_path.exists()
    assert store.load_evidence_markdown() == "# Evidence\n"
    assert store.load_evidence_dashboard() == {"status": "ok"}
    json.loads(dashboard_path.read_text(encoding="utf-8"))
