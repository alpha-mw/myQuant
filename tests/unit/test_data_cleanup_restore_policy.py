"""Restore-source duplicate policy report contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_restore_policy import (
    build_restore_source_policy,
    main as restore_policy_main,
    write_restore_source_policy,
)


def _plan_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_plan.v1",
        "generated_at": "2026-06-12T10:00:00+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "summary": {
            "candidate_group_count": 4,
            "candidate_file_count": 8,
            "potential_reclaim_bytes": 888,
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
            {
                "group_id": "quarantine-skip",
                "candidate_type": "quarantine_restore_mirror",
                "reclaimable_bytes": 200,
                "candidate_paths": [
                    "reports/storage/csv_quarantine/cn_market_full/000005.SZ.csv"
                ],
                "retained_paths": [
                    "data/raw_backups/tushare/daily/full_a_000005.SZ_20260601_raw.csv"
                ],
            },
        ],
    }


def test_restore_source_policy_classifies_review_only_groups(tmp_path):
    policy = build_restore_source_policy(_plan_fixture(tmp_path))

    assert policy["schema_version"] == "myquant.data_cleanup_restore_policy.v1"
    assert policy["delete_candidate_count"] == 0
    assert policy["summary"]["restore_source_group_count"] == 3
    assert policy["summary"]["restore_source_candidate_file_count"] == 3
    assert policy["summary"]["potential_reclaim_bytes"] == 688
    assert policy["summary"]["policy_class_summary"] == {
        "cross_symbol_generated_artifact_duplicate": 1,
        "same_symbol_cleaning_artifact_duplicate": 1,
        "same_symbol_raw_backup_duplicate": 1,
    }
    assert policy["summary"]["risk_level_summary"] == {
        "high": 1,
        "medium": 2,
    }

    raw_group = policy["groups"][0]
    assert raw_group["group_id"] == "raw-same-symbol"
    assert raw_group["policy_class"] == "same_symbol_raw_backup_duplicate"
    assert raw_group["delete_allowed"] is False
    assert raw_group["symbols"] == ["000001.SZ"]
    assert "retained_raw_backup_policy_required" in raw_group["blockers"]
    assert "reference_scan_required" in raw_group["required_validations"]

    cross_symbol = policy["groups"][2]
    assert cross_symbol["policy_class"] == "cross_symbol_generated_artifact_duplicate"
    assert cross_symbol["risk_level"] == "high"
    assert cross_symbol["symbols"] == ["000003.SZ", "000004.SZ"]
    assert "cross_symbol_artifact_review_required" in cross_symbol["blockers"]


def test_restore_source_policy_writes_reports_and_cli_output(tmp_path, capsys):
    plan_path = tmp_path / "data_cleanup_plan.json"
    plan_path.write_text(
        json.dumps(_plan_fixture(tmp_path), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "policy"

    written = write_restore_source_policy(
        plan_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_groups=2,
    )

    payload = json.loads((output_dir / "data_cleanup_restore_policy.json").read_text())
    markdown = (output_dir / "data_cleanup_restore_policy.md").read_text()
    assert written["json"] == str(output_dir / "data_cleanup_restore_policy.json")
    assert payload["summary"]["restore_source_group_count"] == 3
    assert markdown.startswith("# Data Cleanup Restore-Source Policy")
    assert "Delete candidates: 0" in markdown
    assert "same_symbol_raw_backup_duplicate" in markdown

    exit_code = restore_policy_main(
        [
            "--root",
            str(tmp_path),
            "--plan-json",
            str(plan_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup restore-source policy mode: dry-run" in stdout
    assert "delete candidates: 0" in stdout
