"""Project cleanup status report contract tests."""

from __future__ import annotations

import json

from scripts.project_cleanup_status import (
    build_project_cleanup_status,
    main as project_cleanup_status_main,
    write_project_cleanup_status,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _seed_cleanup_evidence(root):
    project_cleanup = root / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "data_duplicate_audit_20260101T000000Z"
        / "data_duplicate_audit.json",
        {
            "schema_version": "myquant.data_duplicate_audit.v1",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "summary": {
                "duplicate_group_count": 2,
                "duplicate_file_count": 5,
                "classification_summary": {
                    "duplicate_restore_source": 5,
                },
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_plan_20260101T000100Z"
        / "data_cleanup_plan.json",
        {
            "schema_version": "myquant.data_cleanup_plan.v1",
            "generated_at": "2026-01-01T00:01:00+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "candidate_group_count": 2,
                "candidate_file_count": 3,
                "potential_reclaim_bytes": 321,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_restore_policy_20260101T000150Z"
        / "data_cleanup_restore_policy.json",
        {
            "schema_version": "myquant.data_cleanup_restore_policy.v1",
            "generated_at": "2026-01-01T00:01:50+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "restore_source_group_count": 2,
                "restore_source_candidate_file_count": 3,
                "potential_reclaim_bytes": 777,
                "risk_level_summary": {
                    "high": 1,
                    "medium": 1,
                },
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_restore_reference_audit_20260101T000175Z"
        / "data_cleanup_restore_reference_audit.json",
        {
            "schema_version": "myquant.data_cleanup_restore_reference_audit.v1",
            "generated_at": "2026-01-01T00:01:55+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "candidate_path_count": 3,
                "referenced_candidate_path_count": 2,
                "unreferenced_candidate_path_count": 1,
                "referenced_group_count": 1,
                "unreferenced_group_count": 1,
                "scan_file_count": 12,
                "scan_mode": "candidate_owner_reports",
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_restore_readiness_20260101T000180Z"
        / "data_cleanup_restore_readiness.json",
        {
            "schema_version": "myquant.data_cleanup_restore_readiness.v1",
            "generated_at": "2026-01-01T00:01:58+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "group_count": 4,
                "reference_free_group_count": 3,
                "referenced_group_count": 1,
                "reference_free_candidate_path_count": 3,
                "reference_free_potential_reclaim_bytes": 688,
                "readiness_class_summary": {
                    "blocked_high_risk_policy": 1,
                    "blocked_referenced_candidate": 1,
                    "review_manifest_rewrite_required": 1,
                    "review_retained_copy_readback_required": 1,
                },
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_restore_readback_20260101T000190Z"
        / "data_cleanup_restore_readback.json",
        {
            "schema_version": "myquant.data_cleanup_restore_readback.v1",
            "generated_at": "2026-01-01T00:01:59+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "reviewed_group_count": 1,
                "skipped_by_filter_count": 3,
                "skipped_by_limit_count": 0,
                "retained_copy_readback_passed_count": 1,
                "blocked_count": 0,
                "verified_reclaim_bytes": 300,
                "status_summary": {
                    "retained_copy_readback_passed": 1,
                },
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_gate_20260101T000200Z"
        / "data_cleanup_gate.json",
        {
            "schema_version": "myquant.data_cleanup_gate.v1",
            "generated_at": "2026-01-01T00:02:00+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "clear_but_delete_disabled_count": 2,
                "blocked_count": 0,
                "runtime_candidate_reference_count": 0,
                "strategy_candidate_reference_count": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_readback_20260101T000300Z"
        / "data_cleanup_readback.json",
        {
            "schema_version": "myquant.data_cleanup_readback.v1",
            "generated_at": "2026-01-01T00:03:00+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "reviewed_candidate_count": 2,
                "hash_readback_passed_count": 2,
                "verified_reclaim_bytes": 321,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_whitelist_20260101T000400Z"
        / "data_cleanup_whitelist.json",
        {
            "schema_version": "myquant.data_cleanup_whitelist.v1",
            "generated_at": "2026-01-01T00:04:00+00:00",
            "delete_candidate_count": 0,
            "execute_allowed_count": 0,
            "summary": {
                "whitelist_item_count": 2,
                "potential_reclaim_bytes": 321,
                "manual_approval_required_count": 2,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_execute_20260101T000500Z"
        / "data_cleanup_execute.json",
        {
            "schema_version": "myquant.data_cleanup_execute.v1",
            "generated_at": "2026-01-01T00:05:00+00:00",
            "execution_performed": False,
            "summary": {
                "would_delete_count": 0,
                "deleted_count": 0,
                "deleted_reclaim_bytes": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "cleanup_baseline_20260101_000000"
        / "architecture_performance_audit.json",
        {
            "schema_version": "myquant_architecture_performance_audit.v1",
            "generated_at": "2026-01-01T00:00:00+08:00",
            "primary_findings": [
                {
                    "id": "arch-001",
                    "severity": "medium",
                    "title": "split a large market module",
                }
            ],
            "large_module_candidates": [
                {
                    "path": "quant_investor/market/analyze.py",
                    "lines": 2500,
                }
            ],
        },
    )
    _write_json(
        root / "results" / "cn_analysis_full" / "CN_Runtime_Profile_20260101.json",
        {
            "market": "CN",
            "universe": "full_a",
            "total_seconds": 9.5,
            "stages": [
                {
                    "name": "dag_batch_read",
                    "seconds": 1.8,
                    "metadata": {"per_symbol_fallback_count": 0},
                },
                {
                    "name": "dag_quant_branch_result",
                    "seconds": 3.1,
                    "metadata": {},
                },
            ],
        },
    )


def test_project_cleanup_status_maps_cleanup_objectives(tmp_path):
    _seed_cleanup_evidence(tmp_path)

    status = build_project_cleanup_status(tmp_path)
    objectives = status["objectives"]

    assert status["schema_version"] == "myquant.project_cleanup_status.v1"
    assert sorted(objectives) == [
        "duplicate_storage",
        "legacy_code_files",
        "redundant_code",
        "unnecessary_data",
        "unreasonable_structure",
    ]
    assert objectives["redundant_code"]["status"] == "clear"
    assert objectives["redundant_code"]["evidence"]["production_reference_count"] == 0
    assert objectives["legacy_code_files"]["evidence"]["existing_candidate_count"] == 0
    assert objectives["duplicate_storage"]["status"] == "review_only"
    assert objectives["duplicate_storage"]["evidence"]["duplicate_group_count"] == 2
    assert objectives["duplicate_storage"]["evidence"]["delete_candidate_count"] == 0
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_policy_group_count"
        ]
        == 2
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_policy_candidate_file_count"
        ]
        == 3
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_policy_potential_reclaim_bytes"
        ]
        == 777
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_policy_delete_candidate_count"
        ]
        == 0
    )
    assert objectives["duplicate_storage"]["evidence"]["restore_policy_high_risk_count"] == 1
    assert objectives["duplicate_storage"]["evidence"]["restore_policy_medium_risk_count"] == 1
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_candidate_path_count"
        ]
        == 3
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_referenced_candidate_path_count"
        ]
        == 2
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_unreferenced_candidate_path_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_referenced_group_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_scan_file_count"
        ]
        == 12
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_reference_scan_mode"
        ]
        == "candidate_owner_reports"
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_reference_free_group_count"
        ]
        == 3
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_referenced_group_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_reference_unknown_group_count"
        ]
        == 0
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_reference_free_candidate_path_count"
        ]
        == 3
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_reference_free_potential_reclaim_bytes"
        ]
        == 688
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_delete_candidate_count"
        ]
        == 0
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_retained_copy_readback_required_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_manifest_rewrite_required_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_blocked_high_risk_policy_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_blocked_referenced_candidate_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readiness_missing_reference_audit_group_count"
        ]
        == 0
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readback_reviewed_group_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readback_passed_group_count"
        ]
        == 1
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readback_blocked_group_count"
        ]
        == 0
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readback_verified_reclaim_bytes"
        ]
        == 300
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "restore_readback_delete_candidate_count"
        ]
        == 0
    )
    assert objectives["duplicate_storage"]["evidence"]["whitelist_item_count"] == 2
    assert (
        objectives["duplicate_storage"]["evidence"][
            "pending_manual_approval_count"
        ]
        == 2
    )
    assert (
        objectives["duplicate_storage"]["evidence"][
            "whitelist_potential_reclaim_bytes"
        ]
        == 321
    )
    assert objectives["unnecessary_data"]["evidence"]["active_runtime_delete_candidate_count"] == 0
    assert objectives["unreasonable_structure"]["status"] == "review_required"
    assert (
        objectives["unreasonable_structure"]["evidence"][
            "baseline_large_module_count"
        ]
        == 1
    )
    assert objectives["unreasonable_structure"]["evidence"]["large_module_count"] == 0
    assert status["performance"]["latest_profile"]["per_symbol_fallback_count"] == 0
    assert status["data_safety"]["active_runtime_delete_candidate_count"] == 0
    assert status["data_safety"]["data_cleanup_deleted_count"] == 0
    assert status["data_safety"]["restore_policy_delete_candidate_count"] == 0
    assert status["data_safety"]["restore_reference_delete_candidate_count"] == 0
    assert status["data_safety"]["restore_readiness_delete_candidate_count"] == 0
    assert status["data_safety"]["restore_readback_delete_candidate_count"] == 0
    assert status["sources"]["data_cleanup_restore_policy_json"].endswith(
        "data_cleanup_restore_policy_20260101T000150Z/data_cleanup_restore_policy.json"
    )
    assert status["sources"]["data_cleanup_restore_reference_audit_json"].endswith(
        "data_cleanup_restore_reference_audit_20260101T000175Z/data_cleanup_restore_reference_audit.json"
    )
    assert status["sources"]["data_cleanup_restore_readiness_json"].endswith(
        "data_cleanup_restore_readiness_20260101T000180Z/data_cleanup_restore_readiness.json"
    )
    assert status["sources"]["data_cleanup_restore_readback_json"].endswith(
        "data_cleanup_restore_readback_20260101T000190Z/data_cleanup_restore_readback.json"
    )


def test_project_cleanup_status_writes_reports_and_cli_output(tmp_path, capsys):
    _seed_cleanup_evidence(tmp_path)
    output_dir = tmp_path / "reports" / "project_cleanup" / "status"

    written = write_project_cleanup_status(tmp_path, output_dir=output_dir)

    payload = json.loads((output_dir / "project_cleanup_status.json").read_text())
    markdown = (output_dir / "project_cleanup_status.md").read_text()
    assert written["json"] == str(output_dir / "project_cleanup_status.json")
    assert payload["schema_version"] == "myquant.project_cleanup_status.v1"
    assert markdown.startswith("# Project Cleanup Status")
    assert "| duplicate_storage | review_only |" in markdown

    exit_code = project_cleanup_status_main(
        [
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "project cleanup status:" in stdout
    assert "duplicate_storage: review_only" in stdout


def test_project_cleanup_status_prefers_latest_code_retirement_audit(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    _write_json(
        tmp_path
        / "reports"
        / "project_cleanup"
        / "code_retirement_reference_audit_20260101T000600Z"
        / "code_retirement_reference_audit.json",
        {
            "schema_version": "myquant.code_retirement_reference_audit.v1",
            "generated_at": "2026-01-01T00:06:00+00:00",
            "candidate_count": 8,
            "production_reference_count": 1,
            "candidates": [
                {
                    "relative_path": "quant_investor/kronos_predictor.py",
                    "exists": True,
                    "production_reference_count": 1,
                }
            ],
        },
    )

    status = build_project_cleanup_status(tmp_path)

    assert status["objectives"]["redundant_code"]["status"] == "attention_required"
    assert (
        status["objectives"]["legacy_code_files"]["evidence"][
            "existing_candidate_count"
        ]
        == 1
    )
    assert status["sources"]["code_retirement_reference_audit_json"].endswith(
        "code_retirement_reference_audit.json"
    )


def test_project_cleanup_status_uses_current_large_module_evidence(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    analyze_path = tmp_path / "quant_investor" / "market" / "analyze.py"
    download_path = tmp_path / "quant_investor" / "market" / "download_cn.py"
    analyze_path.parent.mkdir(parents=True)
    analyze_path.write_text("print('small')\n", encoding="utf-8")
    download_path.write_text(("print('large')\n" * 1001), encoding="utf-8")

    status = build_project_cleanup_status(tmp_path)
    structure = status["objectives"]["unreasonable_structure"]["evidence"]

    assert structure["baseline_large_module_count"] == 1
    assert structure["current_large_module_count"] == 1
    assert structure["top_large_modules"] == [
        "quant_investor/market/download_cn.py"
    ]
    assert structure["current_large_modules"] == [
        {
            "path": "quant_investor/market/download_cn.py",
            "lines": 1001,
        }
    ]


def test_project_cleanup_status_marks_baseline_findings_resolved_by_current_evidence(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "cleanup_baseline_20260102_000000"
        / "architecture_performance_audit.json",
        {
            "schema_version": "myquant_architecture_performance_audit.v1",
            "generated_at": "2026-01-02T00:00:00+08:00",
            "primary_findings": [
                {"id": "perf-001", "title": "DAG context still reads symbols one by one"},
                {"id": "perf-002", "title": "Market run/analyze lacks DAG stage profile"},
                {"id": "perf-003", "title": "MarketDataReader repeats pointer lookups"},
                {"id": "arch-001", "title": "market/analyze.py mixes renderer and persistence"},
                {"id": "arch-002", "title": "strategy profiler should be generalized"},
            ],
            "large_module_candidates": [
                {"path": "quant_investor/market/analyze.py", "lines": 2500},
                {
                    "path": "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
                    "lines": 2600,
                },
            ],
        },
    )
    for path in [
        "quant_investor/market/analyze.py",
        "quant_investor/market/full_report.py",
        "quant_investor/market/full_report_helpers.py",
        "quant_investor/market/full_report_sections.py",
        "quant_investor/market/report_persistence.py",
        "quant_investor/market/runtime_profile.py",
        "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
        "quant_investor/monitoring/cn_aggressive_review_layer.py",
        "quant_investor/monitoring/cn_aggressive_review_runtime.py",
        "quant_investor/monitoring/cn_aggressive_rebalance.py",
        "quant_investor/monitoring/cn_aggressive_report_renderer.py",
    ]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("print('split')\n", encoding="utf-8")
    reader_path = tmp_path / "quant_investor" / "market" / "market_data_reader.py"
    reader_path.write_text(
        "\n".join(
            [
                "self._latest_payload = None",
                "self._snapshot_gate_cache = None",
                "self._serving_symbols_cache = None",
                "self._components_payload = None",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "results" / "cn_analysis_full" / "CN_Runtime_Profile_20260102.json",
        {
            "market": "CN",
            "universe": "full_a",
            "total_seconds": 12.3,
            "stages": [
                {"name": "dag_symbol_list", "seconds": 0.1, "metadata": {}},
                {
                    "name": "dag_batch_read",
                    "seconds": 2.1,
                    "metadata": {
                        "projected_column_count": 9,
                        "runtime_lookback_start_date": "20250416",
                        "batch_result_count": 5531,
                        "per_symbol_fallback_count": 0,
                    },
                },
                {"name": "dag_funnel", "seconds": 0.1, "metadata": {}},
                {"name": "dag_candidate_research", "seconds": 0.1, "metadata": {}},
                {"name": "dag_bayesian_selection", "seconds": 0.1, "metadata": {}},
                {"name": "dag_control_chain", "seconds": 0.1, "metadata": {}},
                {"name": "dag_reporting_artifacts", "seconds": 0.1, "metadata": {}},
                {
                    "name": "analysis_report_persistence",
                    "seconds": 0.1,
                    "metadata": {},
                },
            ],
        },
    )

    status = build_project_cleanup_status(tmp_path)
    structure = status["objectives"]["unreasonable_structure"]

    assert structure["status"] == "remediated_pending_rebaseline"
    assert structure["evidence"]["current_large_module_count"] == 0
    assert structure["evidence"]["remediated_baseline_finding_count"] == 5
    assert structure["evidence"]["unresolved_baseline_finding_count"] == 0
    assert structure["evidence"]["remediated_baseline_large_module_candidate_count"] == 2
    assert {
        item["id"]: item["status"]
        for item in structure["evidence"]["baseline_finding_remediation"]
    } == {
        "perf-001": "resolved_current_evidence",
        "perf-002": "resolved_current_evidence",
        "perf-003": "resolved_current_evidence",
        "arch-001": "resolved_current_evidence",
        "arch-002": "resolved_current_evidence",
    }


def test_project_cleanup_status_prefers_current_architecture_rebaseline(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "architecture_rebaseline_20260103T000000Z"
        / "architecture_performance_audit.json",
        {
            "schema_version": "myquant_architecture_performance_audit.v1",
            "audit_kind": "current_rebaseline",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "primary_findings": [],
            "large_module_candidates": [],
            "summary": {
                "primary_finding_count": 0,
                "large_module_candidate_count": 0,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    structure = status["objectives"]["unreasonable_structure"]

    assert structure["status"] == "clear"
    assert structure["evidence"]["primary_finding_count"] == 0
    assert structure["evidence"]["baseline_large_module_count"] == 0
    assert status["sources"]["architecture_performance_audit_json"].endswith(
        "architecture_rebaseline_20260103T000000Z/architecture_performance_audit.json"
    )


def test_project_cleanup_status_marks_duplicate_storage_approved_pending_execute(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    _write_json(
        tmp_path
        / "reports"
        / "project_cleanup"
        / "data_cleanup_whitelist_20260102T000000Z_approved"
        / "data_cleanup_whitelist.json",
        {
            "schema_version": "myquant.data_cleanup_whitelist.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "delete_candidate_count": 1,
            "execute_allowed_count": 1,
            "summary": {
                "whitelist_item_count": 2,
                "manual_approval_required_count": 1,
                "potential_reclaim_bytes": 321,
                "approved_for_delete_count": 1,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert duplicate["status"] == "approved_pending_execute"
    assert duplicate["evidence"]["whitelist_execute_allowed_count"] == 1
    assert duplicate["evidence"]["approved_for_delete_count"] == 1


def test_project_cleanup_status_accumulates_delete_execution_reports(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "data_cleanup_execute_20260102T000000Z"
        / "data_cleanup_execute.json",
        {
            "schema_version": "myquant.data_cleanup_execute.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "deleted_count": 3,
                "deleted_reclaim_bytes": 300,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_execute_20260103T000000Z"
        / "data_cleanup_execute.json",
        {
            "schema_version": "myquant.data_cleanup_execute.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "deleted_count": 4,
                "deleted_reclaim_bytes": 400,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert duplicate["status"] == "partial_cleanup_executed"
    assert duplicate["evidence"]["latest_deleted_count"] == 4
    assert duplicate["evidence"]["deleted_count"] == 7
    assert duplicate["evidence"]["deleted_reclaim_bytes"] == 700
    assert status["data_safety"]["data_cleanup_deleted_count"] == 7
    assert status["data_safety"]["data_cleanup_deleted_reclaim_bytes"] == 700


def test_project_cleanup_status_marks_duplicate_storage_clear_after_zero_audit(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "data_duplicate_audit_20260103T000000Z"
        / "data_duplicate_audit.json",
        {
            "schema_version": "myquant.data_duplicate_audit.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "summary": {
                "duplicate_group_count": 0,
                "duplicate_file_count": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_plan_20260103T000000Z"
        / "data_cleanup_plan.json",
        {
            "schema_version": "myquant.data_cleanup_plan.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "delete_candidate_count": 0,
            "summary": {
                "candidate_group_count": 0,
                "candidate_file_count": 0,
                "potential_reclaim_bytes": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_execute_20260103T000000Z"
        / "data_cleanup_execute.json",
        {
            "schema_version": "myquant.data_cleanup_execute.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "deleted_count": 4,
                "deleted_reclaim_bytes": 400,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert duplicate["status"] == "clear"
    assert duplicate["evidence"]["duplicate_group_count"] == 0
    assert duplicate["evidence"]["candidate_group_count"] == 0
    assert duplicate["evidence"]["deleted_count"] == 4
    assert status["data_safety"]["data_cleanup_deleted_count"] == 4


def test_project_cleanup_status_accumulates_reference_rewrite_reports(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "data_cleanup_reference_rewrite_20260102T000000Z"
        / "data_cleanup_reference_rewrite.json",
        {
            "schema_version": "myquant.data_cleanup_reference_rewrite.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": False,
            "execution_performed": True,
            "summary": {
                "selected_group_count": 2,
                "rewritten_deleted_count": 2,
                "rewritten_deleted_reclaim_bytes": 200,
                "references_rewritten_count": 4,
                "blocked_count": 0,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]["evidence"]

    assert duplicate["reference_rewrite_report_count"] == 1
    assert duplicate["reference_rewrite_rewritten_deleted_count"] == 2
    assert duplicate["reference_rewrite_rewritten_deleted_reclaim_bytes"] == 200
    assert duplicate["reference_rewrite_references_rewritten_count"] == 4
    assert status["data_safety"]["reference_rewrite_execution_performed"] is True
    assert status["data_safety"]["reference_rewrite_rewritten_deleted_count"] == 2


def test_project_cleanup_status_accumulates_empty_cell_flags_compaction_reports(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "empty_cell_flags_compaction_20260102T000000Z"
        / "empty_cell_flags_compaction.json",
        {
            "schema_version": "myquant.empty_cell_flags_compaction.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "compacted_count": 3,
                "compacted_reclaim_bytes": 3,
                "orphan_deleted_count": 2,
                "orphan_deleted_reclaim_bytes": 2,
                "references_rewritten_count": 6,
                "blocked_count": 1,
            },
        },
    )
    _write_json(
        project_cleanup
        / "empty_cell_flags_compaction_20260103T000000Z"
        / "empty_cell_flags_compaction.json",
        {
            "schema_version": "myquant.empty_cell_flags_compaction.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": False,
            "confirm_token_valid": False,
            "execution_performed": False,
            "summary": {
                "candidate_count": 1,
                "would_compact_count": 0,
                "compacted_count": 0,
                "compacted_reclaim_bytes": 0,
                "orphan_deleted_count": 0,
                "orphan_deleted_reclaim_bytes": 0,
                "references_rewritten_count": 0,
                "blocked_count": 1,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert status["sources"]["empty_cell_flags_compaction_json"].endswith(
        "empty_cell_flags_compaction_20260103T000000Z/empty_cell_flags_compaction.json"
    )
    assert duplicate["evidence"]["empty_cell_flags_compacted_count"] == 3
    assert duplicate["evidence"]["empty_cell_flags_compacted_reclaim_bytes"] == 3
    assert duplicate["evidence"]["empty_cell_flags_orphan_deleted_count"] == 2
    assert duplicate["evidence"]["empty_cell_flags_orphan_deleted_reclaim_bytes"] == 2
    assert duplicate["evidence"]["empty_cell_flags_latest_orphan_deleted_count"] == 0
    assert duplicate["evidence"]["empty_cell_flags_references_rewritten_count"] == 6
    assert duplicate["evidence"]["empty_cell_flags_latest_blocked_count"] == 1
    assert status["data_safety"]["empty_cell_flags_compaction_performed"] is True
    assert status["data_safety"]["empty_cell_flags_compacted_count"] == 3
    assert status["data_safety"]["empty_cell_flags_orphan_deleted_count"] == 2


def test_project_cleanup_status_accumulates_uniform_row_flags_compaction_reports(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "uniform_row_flags_compaction_20260102T000000Z"
        / "uniform_row_flags_compaction.json",
        {
            "schema_version": "myquant.uniform_row_flags_compaction.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "compacted_count": 4,
                "compacted_reclaim_bytes": 40,
                "references_rewritten_count": 4,
                "blocked_count": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "uniform_row_flags_compaction_20260103T000000Z"
        / "uniform_row_flags_compaction.json",
        {
            "schema_version": "myquant.uniform_row_flags_compaction.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": False,
            "confirm_token_valid": False,
            "execution_performed": False,
            "summary": {
                "candidate_count": 2,
                "would_compact_count": 1,
                "compacted_count": 0,
                "compacted_reclaim_bytes": 0,
                "references_rewritten_count": 0,
                "blocked_count": 1,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert status["sources"]["uniform_row_flags_compaction_json"].endswith(
        "uniform_row_flags_compaction_20260103T000000Z/"
        "uniform_row_flags_compaction.json"
    )
    assert duplicate["evidence"]["uniform_row_flags_report_count"] == 2
    assert duplicate["evidence"]["uniform_row_flags_compacted_count"] == 4
    assert duplicate["evidence"]["uniform_row_flags_compacted_reclaim_bytes"] == 40
    assert duplicate["evidence"]["uniform_row_flags_references_rewritten_count"] == 4
    assert duplicate["evidence"]["uniform_row_flags_latest_candidate_count"] == 2
    assert duplicate["evidence"]["uniform_row_flags_latest_would_compact_count"] == 1
    assert duplicate["evidence"]["uniform_row_flags_latest_blocked_count"] == 1
    assert status["data_safety"]["uniform_row_flags_compaction_performed"] is True
    assert status["data_safety"]["uniform_row_flags_compacted_count"] == 4
    assert status["data_safety"]["uniform_row_flags_latest_blocked_count"] == 1


def test_project_cleanup_status_accumulates_issue_cell_flags_compaction_reports(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "issue_cell_flags_compaction_20260102T000000Z"
        / "issue_cell_flags_compaction.json",
        {
            "schema_version": "myquant.issue_cell_flags_compaction.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "compacted_count": 4,
                "compacted_reclaim_bytes": 40,
                "references_rewritten_count": 8,
                "issue_row_count": 4,
                "blocked_count": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "issue_cell_flags_compaction_20260103T000000Z"
        / "issue_cell_flags_compaction.json",
        {
            "schema_version": "myquant.issue_cell_flags_compaction.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": False,
            "confirm_token_valid": False,
            "execution_performed": False,
            "summary": {
                "candidate_count": 2,
                "would_compact_count": 1,
                "compacted_count": 0,
                "compacted_reclaim_bytes": 0,
                "references_rewritten_count": 0,
                "issue_row_count": 0,
                "blocked_count": 1,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert status["sources"]["issue_cell_flags_compaction_json"].endswith(
        "issue_cell_flags_compaction_20260103T000000Z/"
        "issue_cell_flags_compaction.json"
    )
    assert duplicate["evidence"]["issue_cell_flags_report_count"] == 2
    assert duplicate["evidence"]["issue_cell_flags_compacted_count"] == 4
    assert duplicate["evidence"]["issue_cell_flags_compacted_reclaim_bytes"] == 40
    assert duplicate["evidence"]["issue_cell_flags_references_rewritten_count"] == 8
    assert duplicate["evidence"]["issue_cell_flags_issue_row_count"] == 4
    assert duplicate["evidence"]["issue_cell_flags_latest_candidate_count"] == 2
    assert duplicate["evidence"]["issue_cell_flags_latest_would_compact_count"] == 1
    assert duplicate["evidence"]["issue_cell_flags_latest_blocked_count"] == 1
    assert status["data_safety"]["issue_cell_flags_compaction_performed"] is True
    assert status["data_safety"]["issue_cell_flags_compacted_count"] == 4
    assert status["data_safety"]["issue_cell_flags_issue_row_count"] == 4
    assert status["data_safety"]["issue_cell_flags_latest_blocked_count"] == 1


def test_project_cleanup_status_accumulates_matrix_coverage_compaction_reports(
    tmp_path,
):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "matrix_coverage_compaction_20260102T000000Z"
        / "matrix_coverage_compaction.json",
        {
            "schema_version": "myquant.matrix_coverage_sidecar_compaction.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "compacted_count": 5,
                "compacted_reclaim_bytes": 50,
                "references_rewritten_count": 5,
                "blocked_count": 0,
            },
        },
    )
    _write_json(
        project_cleanup
        / "matrix_coverage_compaction_20260103T000000Z"
        / "matrix_coverage_compaction.json",
        {
            "schema_version": "myquant.matrix_coverage_sidecar_compaction.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": False,
            "confirm_token_valid": False,
            "execution_performed": False,
            "summary": {
                "candidate_count": 2,
                "would_compact_count": 1,
                "compacted_count": 0,
                "compacted_reclaim_bytes": 0,
                "references_rewritten_count": 0,
                "blocked_count": 1,
            },
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert status["sources"]["matrix_coverage_compaction_json"].endswith(
        "matrix_coverage_compaction_20260103T000000Z/"
        "matrix_coverage_compaction.json"
    )
    assert duplicate["evidence"]["matrix_coverage_report_count"] == 2
    assert duplicate["evidence"]["matrix_coverage_compacted_count"] == 5
    assert duplicate["evidence"]["matrix_coverage_compacted_reclaim_bytes"] == 50
    assert duplicate["evidence"]["matrix_coverage_references_rewritten_count"] == 5
    assert duplicate["evidence"]["matrix_coverage_latest_candidate_count"] == 2
    assert duplicate["evidence"]["matrix_coverage_latest_would_compact_count"] == 1
    assert duplicate["evidence"]["matrix_coverage_latest_blocked_count"] == 1
    assert status["data_safety"]["matrix_coverage_compaction_performed"] is True
    assert status["data_safety"]["matrix_coverage_compacted_count"] == 5
    assert status["data_safety"]["matrix_coverage_latest_blocked_count"] == 1


def test_project_cleanup_status_subtracts_executed_groups_from_pending(tmp_path):
    _seed_cleanup_evidence(tmp_path)
    project_cleanup = tmp_path / "reports" / "project_cleanup"
    _write_json(
        project_cleanup
        / "data_cleanup_whitelist_20260102T000000Z_approved"
        / "data_cleanup_whitelist.json",
        {
            "schema_version": "myquant.data_cleanup_whitelist.v1",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "delete_candidate_count": 1,
            "execute_allowed_count": 1,
            "summary": {
                "whitelist_item_count": 3,
                "manual_approval_required_count": 2,
                "potential_reclaim_bytes": 600,
                "approved_for_delete_count": 1,
            },
            "items": [
                {
                    "group_id": "dup-1",
                    "approval_status": "pending_manual_approval",
                    "execute_allowed": False,
                },
                {
                    "group_id": "dup-2",
                    "approval_status": "pending_manual_approval",
                    "execute_allowed": False,
                },
                {
                    "group_id": "dup-3",
                    "approval_status": "approved_for_delete",
                    "execute_allowed": True,
                },
            ],
        },
    )
    _write_json(
        project_cleanup
        / "data_cleanup_execute_20260103T000000Z"
        / "data_cleanup_execute.json",
        {
            "schema_version": "myquant.data_cleanup_execute.v1",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "apply_requested": True,
            "confirm_token_valid": True,
            "execution_performed": True,
            "summary": {
                "deleted_count": 2,
                "deleted_reclaim_bytes": 300,
            },
            "items": [
                {
                    "group_id": "dup-1",
                    "status": "deleted",
                    "reclaimable_bytes": 100,
                },
                {
                    "group_id": "dup-3",
                    "status": "deleted",
                    "reclaimable_bytes": 200,
                },
            ],
        },
    )

    status = build_project_cleanup_status(tmp_path)
    duplicate = status["objectives"]["duplicate_storage"]

    assert duplicate["evidence"]["pending_manual_approval_count"] == 1
    assert duplicate["evidence"]["approved_for_delete_count"] == 0
    assert duplicate["evidence"]["whitelist_execute_allowed_count"] == 0
    assert duplicate["evidence"]["deleted_group_count"] == 2
