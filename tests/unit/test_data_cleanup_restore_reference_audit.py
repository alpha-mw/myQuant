"""Restore-source duplicate reference audit contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.data_cleanup_restore_policy import build_restore_source_policy
from scripts.data_cleanup_restore_reference_audit import (
    build_restore_reference_audit,
    main as restore_reference_audit_main,
    write_restore_reference_audit,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _policy_fixture(tmp_path):
    plan = {
        "schema_version": "myquant.data_cleanup_plan.v1",
        "generated_at": "2026-06-12T10:00:00+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "summary": {
            "candidate_group_count": 3,
            "candidate_file_count": 6,
            "potential_reclaim_bytes": 688,
        },
        "candidates": [
            {
                "group_id": "raw-same-symbol",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 300,
                "candidate_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"
                ],
                "retained_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260602_raw.csv"
                ],
            },
            {
                "group_id": "row-flags-same-symbol",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 200,
                "candidate_paths": [
                    "data/cleaning_reports/tushare/daily/full_a_000002.SZ_20260601_row_flags.csv"
                ],
                "retained_paths": [
                    "data/cleaning_reports/tushare/daily/full_a_000002.SZ_20260602_row_flags.csv"
                ],
            },
            {
                "group_id": "matrix-cross-symbol",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 188,
                "candidate_paths": [
                    "data/factor_readiness/tushare/daily/full_a_000003.SZ_20260601_matrix_coverage.json"
                ],
                "retained_paths": [
                    "data/factor_readiness/tushare/daily/full_a_000004.SZ_20260601_matrix_coverage.json"
                ],
            },
        ],
    }
    return build_restore_source_policy(plan)


def _seed_reference_files(tmp_path):
    row_flags_path = (
        "data/cleaning_reports/tushare/daily/"
        "full_a_000002.SZ_20260601_row_flags.csv"
    )
    matrix_path = (
        "data/factor_readiness/tushare/daily/"
        "full_a_000003.SZ_20260601_matrix_coverage.json"
    )
    raw_path = "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"

    _write_text(
        tmp_path
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000002.SZ_20260601_cleaning_report.json",
        json.dumps({"row_flags_path": row_flags_path}),
    )
    _write_text(
        tmp_path
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000003.SZ_20260601_cleaning_report.json",
        json.dumps({"matrix_coverage_path": matrix_path}),
    )
    _write_text(
        tmp_path
        / "reports"
        / "project_cleanup"
        / "data_cleanup_restore_policy_20260101T000000Z"
        / "data_cleanup_restore_policy.json",
        json.dumps({"self_reference": raw_path}),
    )


def test_restore_reference_audit_scans_external_references_only(tmp_path):
    policy = _policy_fixture(tmp_path)
    _seed_reference_files(tmp_path)

    audit = build_restore_reference_audit(
        policy,
        root=tmp_path,
        scan_roots=[
            Path("data/cleaning_reports/tushare"),
            Path("data/factor_readiness/tushare"),
            Path("reports"),
        ],
    )

    assert audit["schema_version"] == "myquant.data_cleanup_restore_reference_audit.v1"
    assert audit["delete_candidate_count"] == 0
    assert audit["summary"]["candidate_path_count"] == 3
    assert audit["summary"]["referenced_candidate_path_count"] == 2
    assert audit["summary"]["unreferenced_candidate_path_count"] == 1
    assert audit["summary"]["referenced_group_count"] == 2
    assert audit["summary"]["unreferenced_group_count"] == 1

    items = {item["group_id"]: item for item in audit["groups"]}
    assert items["raw-same-symbol"]["reference_count"] == 0
    assert items["raw-same-symbol"]["unreferenced_candidate_paths"] == [
        "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"
    ]
    assert items["row-flags-same-symbol"]["reference_count"] == 1
    assert items["matrix-cross-symbol"]["referenced_candidate_paths"] == [
        "data/factor_readiness/tushare/daily/full_a_000003.SZ_20260601_matrix_coverage.json"
    ]
    assert (
        audit["summary"]["policy_class_reference_summary"][
            "same_symbol_raw_backup_duplicate"
        ]["unreferenced_group_count"]
        == 1
    )


def test_restore_reference_audit_skips_non_manifest_json_by_default(tmp_path):
    policy = _policy_fixture(tmp_path)
    raw_path = "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"
    _write_text(
        tmp_path
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "ad_hoc_debug.json",
        json.dumps({"debug_path": raw_path}),
    )

    default_audit = build_restore_reference_audit(
        policy,
        root=tmp_path,
        scan_roots=[Path("data/cleaning_reports/tushare")],
    )
    all_text_audit = build_restore_reference_audit(
        policy,
        root=tmp_path,
        scan_roots=[Path("data/cleaning_reports/tushare")],
        include_all_text_files=True,
    )

    assert default_audit["summary"]["scan_mode"] == "candidate_owner_reports"
    assert default_audit["summary"]["scan_file_count"] == 0
    assert default_audit["summary"]["referenced_candidate_path_count"] == 0
    assert all_text_audit["summary"]["scan_mode"] == "all_text_files"
    assert all_text_audit["summary"]["scan_file_count"] == 1
    assert all_text_audit["summary"]["referenced_candidate_path_count"] == 1


def test_restore_reference_audit_writes_reports_and_cli_output(tmp_path, capsys):
    policy = _policy_fixture(tmp_path)
    _seed_reference_files(tmp_path)
    policy_path = tmp_path / "data_cleanup_restore_policy.json"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "reports" / "project_cleanup" / "reference_audit"

    written = write_restore_reference_audit(
        policy_path,
        root=tmp_path,
        output_dir=output_dir,
        scan_roots=[
            Path("data/cleaning_reports/tushare"),
            Path("data/factor_readiness/tushare"),
        ],
        max_markdown_groups=2,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_restore_reference_audit.json").read_text()
    )
    markdown = (
        output_dir / "data_cleanup_restore_reference_audit.md"
    ).read_text()
    assert written["json"] == str(
        output_dir / "data_cleanup_restore_reference_audit.json"
    )
    assert payload["summary"]["referenced_candidate_path_count"] == 2
    assert markdown.startswith("# Data Cleanup Restore-Source Reference Audit")
    assert "Delete candidates: 0" in markdown

    exit_code = restore_reference_audit_main(
        [
            "--root",
            str(tmp_path),
            "--policy-json",
            str(policy_path),
            "--output-dir",
            str(output_dir / "cli"),
            "--scan-root",
            "data/cleaning_reports/tushare",
            "--scan-root",
            "data/factor_readiness/tushare",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup restore-source reference audit mode: dry-run" in stdout
    assert "referenced candidate paths: 2" in stdout
    assert "delete candidates: 0" in stdout
