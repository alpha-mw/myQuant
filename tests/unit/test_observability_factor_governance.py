from __future__ import annotations

import json

from quant_investor.observability import (
    HEALTH_STATUS_FAIL,
    HEALTH_STATUS_WARN,
    build_observability_summary,
    discover_phase_artifacts,
    summarize_factor_governance_artifacts,
)


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_factor_governance_artifacts_are_discovered_and_counted(tmp_path) -> None:
    factor_root = tmp_path / "factor_library"
    _write_jsonl(factor_root / "factor_definitions.jsonl", [{"factor_id": "f1"}])
    _write_jsonl(factor_root / "factor_backtest_results.jsonl", [{"result_id": "r1"}])
    _write_jsonl(factor_root / "factor_validation_reports.jsonl", [{"report_id": "v1"}])
    _write_jsonl(factor_root / "factor_admission_decisions.jsonl", [{"decision_id": "d1"}])
    (factor_root / "production_factors.json").write_text(
        json.dumps({"entries": [{"factor_id": "f1"}]}),
        encoding="utf-8",
    )
    _write_jsonl(factor_root / "incremental" / "factor_redundancy_reports.jsonl", [{"report_id": "red1"}])
    _write_jsonl(factor_root / "incremental" / "factor_contribution_reports.jsonl", [{"report_id": "con1"}])
    _write_jsonl(factor_root / "audit" / "factor_library_audit_reports.jsonl", [{"report_id": "a1"}])

    refs = discover_phase_artifacts(factor_library_dir=factor_root)
    names = [ref.name for ref in refs]
    summary = summarize_factor_governance_artifacts(refs)
    system_summary = build_observability_summary(refs, generated_at="2026-04-28T00:00:00Z")

    assert "factor_definitions" in names
    assert summary.key_metrics["factor_definition_records"] == 1
    assert summary.key_metrics["backtest_result_records"] == 1
    assert summary.key_metrics["validation_report_records"] == 1
    assert summary.key_metrics["admission_decision_records"] == 1
    assert summary.key_metrics["production_factor_count"] == 1
    assert summary.key_metrics["redundancy_report_records"] == 1
    assert summary.key_metrics["contribution_report_records"] == 1
    assert summary.key_metrics["audit_report_records"] == 1
    assert any(item.module_name == "factor_governance" for item in system_summary.module_summaries)


def test_missing_factor_artifacts_warn_not_fail(tmp_path) -> None:
    refs = discover_phase_artifacts(factor_library_dir=tmp_path / "missing_factor_library")

    summary = summarize_factor_governance_artifacts(refs)

    assert summary.status == HEALTH_STATUS_WARN
    assert summary.failure_count == 0
    assert summary.warning_count > 0


def test_malformed_factor_json_fails(tmp_path) -> None:
    factor_root = tmp_path / "factor_library"
    factor_root.mkdir()
    (factor_root / "factor_definitions.jsonl").write_text("{bad json}\n", encoding="utf-8")

    summary = summarize_factor_governance_artifacts(
        discover_phase_artifacts(factor_library_dir=factor_root)
    )

    assert summary.status == HEALTH_STATUS_FAIL
    assert summary.failure_count == 1
