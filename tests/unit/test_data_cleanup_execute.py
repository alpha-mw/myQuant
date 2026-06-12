"""Data cleanup execution contract tests."""

from __future__ import annotations

import hashlib
import json

from scripts.data_cleanup_execute import (
    CONFIRM_TOKEN,
    build_data_cleanup_execution_report,
    main as data_cleanup_execute_main,
    write_data_cleanup_execution_report,
)


CANDIDATE_PATH = "reports/storage/csv_quarantine/cn_market_full/000001.SZ.csv"
RETAINED_PATH = "data/raw_backups/tushare/daily/full_a_000001.SZ_raw.csv"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _whitelist_item(
    *,
    approved: bool,
    content: str = "symbol,trade_date\n000001.SZ,20260610\n",
):
    return {
        "group_id": "dup-approved" if approved else "dup-pending",
        "candidate_type": "quarantine_restore_mirror",
        "approval_status": "approved_for_delete"
        if approved
        else "pending_manual_approval",
        "delete_allowed": approved,
        "execute_allowed": approved,
        "reclaimable_bytes": len(content.encode("utf-8")),
        "candidate_paths": [CANDIDATE_PATH],
        "retained_paths": [RETAINED_PATH],
        "candidate_sha256": [_sha256(content)],
        "retained_sha256": [_sha256(content)],
        "candidate_size_bytes": [len(content.encode("utf-8"))],
        "retained_size_bytes": [len(content.encode("utf-8"))],
        "required_pre_delete_gates": [],
        "rollback_source_paths": [RETAINED_PATH],
        "reason": "fixture",
    }


def _whitelist_fixture(tmp_path, *, approved: bool):
    return {
        "schema_version": "myquant.data_cleanup_whitelist.v1",
        "generated_at": "2026-06-12T10:52:50+00:00",
        "root": str(tmp_path),
        "items": [_whitelist_item(approved=approved)],
    }


def _write_fixture_files(tmp_path):
    content = "symbol,trade_date\n000001.SZ,20260610\n"
    for relative_path in (CANDIDATE_PATH, RETAINED_PATH):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_execution_skips_no_execute_whitelist_items(tmp_path):
    _write_fixture_files(tmp_path)
    report = build_data_cleanup_execution_report(
        _whitelist_fixture(tmp_path, approved=False),
        repo_root=tmp_path,
    )

    assert report["execution_performed"] is False
    assert report["summary"]["skipped_not_approved_count"] == 1
    assert report["summary"]["would_delete_count"] == 0
    assert (tmp_path / CANDIDATE_PATH).exists()


def test_execution_dry_run_for_approved_items_does_not_delete(tmp_path):
    _write_fixture_files(tmp_path)
    report = build_data_cleanup_execution_report(
        _whitelist_fixture(tmp_path, approved=True),
        repo_root=tmp_path,
    )

    assert report["execution_performed"] is False
    assert report["summary"]["would_delete_count"] == 1
    assert report["summary"]["planned_reclaim_bytes"] > 0
    assert (tmp_path / CANDIDATE_PATH).exists()


def test_execution_apply_deletes_only_approved_file(tmp_path):
    _write_fixture_files(tmp_path)
    candidate = tmp_path / CANDIDATE_PATH
    retained = tmp_path / RETAINED_PATH

    blocked = build_data_cleanup_execution_report(
        _whitelist_fixture(tmp_path, approved=True),
        repo_root=tmp_path,
        apply=True,
        confirm_token="WRONG",
    )

    assert blocked["summary"]["blocked_count"] == 1
    assert candidate.exists()

    applied = build_data_cleanup_execution_report(
        _whitelist_fixture(tmp_path, approved=True),
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
    )

    assert applied["execution_performed"] is True
    assert applied["summary"]["deleted_count"] == 1
    assert not candidate.exists()
    assert retained.exists()


def test_execution_writes_report_and_cli_output(tmp_path, capsys):
    _write_fixture_files(tmp_path)
    whitelist_path = tmp_path / "whitelist.json"
    whitelist_path.write_text(
        json.dumps(_whitelist_fixture(tmp_path, approved=False)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "execute"

    written = write_data_cleanup_execution_report(
        whitelist_path,
        root=tmp_path,
        output_dir=output_dir,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_execute.json").read_text()
    )
    assert written["json"] == str(output_dir / "data_cleanup_execute.json")
    assert payload["summary"]["skipped_not_approved_count"] == 1

    exit_code = data_cleanup_execute_main(
        [
            "--root",
            str(tmp_path),
            "--whitelist-json",
            str(whitelist_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup execution mode: dry-run" in stdout
    assert "skipped not approved: 1" in stdout
