"""Data cleanup plan contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_plan import (
    build_data_cleanup_plan,
    main as data_cleanup_plan_main,
    write_data_cleanup_plan,
)


def _audit_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_duplicate_audit.v1",
        "generated_at": "2026-06-12T09:10:12+00:00",
        "root": str(tmp_path),
        "duplicate_groups": [
            {
                "group_id": "dup-0001",
                "sha256": "abc",
                "size_bytes": 100,
                "files": [
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000001.SZ_raw.csv"
                    ),
                    (
                        "reports/storage/csv_quarantine/"
                        "cn_market_full/000001.SZ.csv"
                    ),
                ],
            },
            {
                "group_id": "dup-0002",
                "sha256": "def",
                "size_bytes": 50,
                "files": [
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000002.SZ_1_raw.csv"
                    ),
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000002.SZ_2_raw.csv"
                    ),
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000002.SZ_3_raw.csv"
                    ),
                ],
            },
        ],
    }


def test_cleanup_plan_marks_quarantine_mirror_as_review_only(tmp_path):
    plan = build_data_cleanup_plan(_audit_fixture(tmp_path))
    candidates = {
        item["group_id"]: item for item in plan["candidates"]
    }

    mirror = candidates["dup-0001"]
    assert mirror["candidate_type"] == "quarantine_restore_mirror"
    assert mirror["delete_allowed"] is False
    assert mirror["status"] == "review_required"
    assert mirror["candidate_paths"] == [
        "reports/storage/csv_quarantine/cn_market_full/000001.SZ.csv"
    ]
    assert mirror["retained_paths"] == [
        "data/raw_backups/tushare/daily/full_a_000001.SZ_raw.csv"
    ]
    assert mirror["reclaimable_bytes"] == 100
    assert "storage_validation_required" in mirror["blockers"]
    assert (
        "quant-investor market storage-validate-clean --market CN"
        in mirror["required_validations"]
    )
    assert (
        "quant-investor market storage-validate-clean --market CN"
        in plan["required_validations"]
    )


def test_cleanup_plan_summarizes_restore_source_duplicates(tmp_path):
    plan = build_data_cleanup_plan(_audit_fixture(tmp_path))
    candidates = {
        item["group_id"]: item for item in plan["candidates"]
    }
    restore_duplicate = candidates["dup-0002"]

    assert restore_duplicate["candidate_type"] == (
        "restore_source_duplicate_review"
    )
    assert restore_duplicate["delete_allowed"] is False
    assert len(restore_duplicate["candidate_paths"]) == 2
    assert restore_duplicate["reclaimable_bytes"] == 100
    assert plan["delete_candidate_count"] == 0
    assert plan["summary"]["candidate_group_count"] == 2
    assert plan["summary"]["candidate_file_count"] == 3
    assert plan["summary"]["potential_reclaim_bytes"] == 200


def test_cleanup_plan_writes_reports_and_cli_output(tmp_path, capsys):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(_audit_fixture(tmp_path)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "plan"

    written = write_data_cleanup_plan(
        audit_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_candidates=1,
    )

    payload = json.loads((output_dir / "data_cleanup_plan.json").read_text())
    assert written["json"] == str(output_dir / "data_cleanup_plan.json")
    assert payload["schema_version"] == "myquant.data_cleanup_plan.v1"
    assert payload["delete_candidate_count"] == 0
    markdown = (output_dir / "data_cleanup_plan.md").read_text()
    assert markdown.startswith("# Data Cleanup Plan")
    assert "Candidate table truncated to 1 of 2 rows" in markdown

    exit_code = data_cleanup_plan_main(
        [
            "--root",
            str(tmp_path),
            "--audit-json",
            str(audit_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup plan mode: dry-run" in stdout
    assert "delete candidates: 0" in stdout
