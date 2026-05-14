from __future__ import annotations

import json
from pathlib import Path

from quant_investor.observability import (
    HEALTH_STATUS_FAIL,
    HEALTH_STATUS_PASS,
    HEALTH_STATUS_WARN,
    build_dashboard_payload,
    build_observability_summary,
    build_run_manifest,
    discover_phase_artifacts,
    summarize_factor_governance_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "factor_library_shadow"


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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
    _write_jsonl(
        factor_root / "audit" / "factor_library_audit_reports.jsonl",
        [
            {
                "report_id": "a1",
                "verdict": "warn",
                "production_factor_count": 1,
                "blocked_factor_ids": ["f-blocked"],
                "shadow_only_factor_ids": ["f-shadow"],
                "expired_factor_count": 1,
                "warning_count": 2,
                "blocker_count": 1,
            }
        ],
    )
    _write_json(
        factor_root / "audit" / "factor_governance_dashboard.json",
        {
            "verdict": "warn",
            "counts": {
                "production_factor_count": 1,
                "blocked_factor_count": 1,
                "shadow_only_factor_count": 1,
                "expired_factor_count": 1,
                "warning_count": 2,
                "blocker_count": 1,
            },
            "blocked_factor_ids": ["f-blocked"],
            "shadow_only_factor_ids": ["f-shadow"],
        },
    )

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
    assert summary.key_metrics["blocked_factor_count"] == 1
    assert summary.key_metrics["shadow_only_factor_count"] == 1
    assert summary.key_metrics["expired_factor_count"] == 1
    assert summary.key_metrics["audit_verdict"] == "warn"
    assert summary.key_metrics["audit_warning_count"] == 2
    assert summary.key_metrics["audit_blocker_count"] == 1
    assert summary.key_metrics["redundancy_report_records"] == 1
    assert summary.key_metrics["contribution_report_records"] == 1
    assert summary.key_metrics["audit_report_records"] == 1
    assert any(item.module_name == "factor_governance" for item in system_summary.module_summaries)


def test_canonical_factor_fixture_is_discovered_and_dashboard_serializes() -> None:
    refs = discover_phase_artifacts(factor_library_dir=CANONICAL_FIXTURE_ROOT)
    names = [ref.name for ref in refs]
    summary = summarize_factor_governance_artifacts(refs)
    manifest = build_run_manifest(
        run_id="canonical-factor-shadow-fixture",
        artifact_refs=refs,
        generated_at="2026-05-01T00:00:00Z",
    )
    system_summary = build_observability_summary(
        refs,
        generated_at="2026-05-01T00:00:00Z",
    )
    dashboard = build_dashboard_payload(manifest, system_summary)

    assert "production_factors" in names
    assert "factor_library_audit_reports" in names
    assert "factor_governance_dashboard" in names
    assert summary.key_metrics["production_factor_count"] == 2
    assert summary.key_metrics["blocked_factor_count"] == 1
    assert summary.key_metrics["shadow_only_factor_count"] == 1
    assert summary.key_metrics["expired_factor_count"] == 1
    assert summary.key_metrics["audit_verdict"] == "fail"
    assert summary.key_metrics["audit_warning_count"] == 1
    assert summary.key_metrics["audit_blocker_count"] == 1
    json.dumps(dashboard, ensure_ascii=False, sort_keys=True)


def test_missing_factor_artifacts_warn_not_fail(tmp_path) -> None:
    refs = discover_phase_artifacts(factor_library_dir=tmp_path / "missing_factor_library")

    summary = summarize_factor_governance_artifacts(refs)

    assert summary.status == HEALTH_STATUS_WARN
    assert summary.failure_count == 0
    assert summary.warning_count > 0


def test_malformed_factor_json_fails_without_breaking_other_modules(tmp_path) -> None:
    outcome_dir = tmp_path / "outcome"
    _write_jsonl(outcome_dir / "predictions.jsonl", [{"prediction_id": "p1"}])
    _write_jsonl(outcome_dir / "outcomes.jsonl", [{"prediction_id": "p1"}])
    factor_root = tmp_path / "factor_library"
    factor_root.mkdir()
    (factor_root / "factor_definitions.jsonl").write_text("{bad json}\n", encoding="utf-8")

    refs = discover_phase_artifacts(
        outcome_ledger_dir=outcome_dir,
        factor_library_dir=factor_root,
    )
    factor_summary = summarize_factor_governance_artifacts(refs)
    system_summary = build_observability_summary(refs, generated_at="2026-05-01T00:00:00Z")
    outcome_summary = next(
        summary
        for summary in system_summary.module_summaries
        if summary.module_name == "outcome_ledger"
    )

    assert factor_summary.status == HEALTH_STATUS_FAIL
    assert factor_summary.failure_count == 1
    assert outcome_summary.status == HEALTH_STATUS_PASS
    assert outcome_summary.key_metrics["prediction_records"] == 1
