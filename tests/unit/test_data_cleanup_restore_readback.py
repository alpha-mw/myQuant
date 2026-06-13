"""Restore-source retained-copy readback contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_restore_policy import build_restore_source_policy
from scripts.data_cleanup_restore_readback import (
    build_restore_readback_report,
    main as restore_readback_main,
    write_restore_readback_report,
)
from scripts.data_cleanup_restore_readiness import build_restore_readiness
from tests.unit.test_data_cleanup_restore_readiness import (
    _plan_fixture,
    _reference_audit_fixture,
)


RAW_CANDIDATE = (
    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"
)
RAW_RETAINED = (
    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260602_raw.csv"
)


def _readiness_fixture(tmp_path):
    policy = build_restore_source_policy(_plan_fixture(tmp_path))
    return build_restore_readiness(policy, _reference_audit_fixture(policy))


def _write_file(root, relative_path, content):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_restore_readback_verifies_retained_copy_groups(tmp_path):
    readiness = _readiness_fixture(tmp_path)
    content = "ts_code,trade_date\n000001.SZ,20260610\n"
    _write_file(tmp_path, RAW_CANDIDATE, content)
    _write_file(tmp_path, RAW_RETAINED, content)

    report = build_restore_readback_report(readiness, repo_root=tmp_path)

    assert report["schema_version"] == "myquant.data_cleanup_restore_readback.v1"
    assert report["delete_candidate_count"] == 0
    assert report["summary"]["reviewed_group_count"] == 1
    assert report["summary"]["skipped_by_filter_count"] == 3
    assert report["summary"]["retained_copy_readback_passed_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert report["summary"]["verified_reclaim_bytes"] == 300
    assert report["summary"]["status_summary"] == {
        "retained_copy_readback_passed": 1,
    }

    item = report["groups"][0]
    assert item["group_id"] == "raw-unreferenced"
    assert item["delete_allowed"] is False
    assert item["readback_status"] == "retained_copy_readback_passed"
    assert item["candidate_files"][0]["sha256"] == item["retained_files"][0]["sha256"]
    assert "manual_delete_approval_required" in item["pending_checks"]


def test_restore_readback_blocks_hash_mismatches(tmp_path):
    readiness = _readiness_fixture(tmp_path)
    _write_file(tmp_path, RAW_CANDIDATE, "ts_code,trade_date\n000001.SZ,20260610\n")
    _write_file(tmp_path, RAW_RETAINED, "ts_code,trade_date\n000001.SZ,20260609\n")

    report = build_restore_readback_report(readiness, repo_root=tmp_path)

    assert report["summary"]["retained_copy_readback_passed_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    item = report["groups"][0]
    assert item["readback_status"] == "blocked"
    assert "hash_mismatch_or_unreadable" in item["blockers"]


def test_restore_readback_can_verify_manifest_rewrite_groups(tmp_path):
    readiness = _readiness_fixture(tmp_path)
    row_candidate = (
        "data/cleaning_reports/tushare/daily/"
        "full_a_000002.SZ_20260601_row_flags.csv"
    )
    row_retained = (
        "data/cleaning_reports/tushare/daily/"
        "full_a_000002.SZ_20260602_row_flags.csv"
    )
    content = "row,reason\n"
    _write_file(tmp_path, row_candidate, content)
    _write_file(tmp_path, row_retained, content)

    report = build_restore_readback_report(
        readiness,
        repo_root=tmp_path,
        readiness_class="review_manifest_rewrite_required",
    )

    assert report["summary"]["reviewed_group_count"] == 1
    assert report["summary"]["retained_copy_readback_passed_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    item = report["groups"][0]
    assert item["readiness_class"] == "review_manifest_rewrite_required"
    assert item["policy_class"] == "same_symbol_cleaning_artifact_duplicate"
    assert item["delete_allowed"] is False


def test_restore_readback_writes_reports_and_cli_output(tmp_path, capsys):
    readiness = _readiness_fixture(tmp_path)
    content = "ts_code,trade_date\n000001.SZ,20260610\n"
    _write_file(tmp_path, RAW_CANDIDATE, content)
    _write_file(tmp_path, RAW_RETAINED, content)
    readiness_path = tmp_path / "data_cleanup_restore_readiness.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    output_dir = tmp_path / "reports" / "restore_readback"

    written = write_restore_readback_report(
        readiness_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_groups=1,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_restore_readback.json").read_text()
    )
    markdown = (output_dir / "data_cleanup_restore_readback.md").read_text()
    assert written["json"] == str(output_dir / "data_cleanup_restore_readback.json")
    assert payload["summary"]["retained_copy_readback_passed_count"] == 1
    assert markdown.startswith("# Data Cleanup Restore-Source Readback")
    assert "Delete candidates: 0" in markdown

    exit_code = restore_readback_main(
        [
            "--root",
            str(tmp_path),
            "--readiness-json",
            str(readiness_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup restore-source readback mode: dry-run" in stdout
    assert "reviewed groups: 1" in stdout
    assert "delete candidates: 0" in stdout
