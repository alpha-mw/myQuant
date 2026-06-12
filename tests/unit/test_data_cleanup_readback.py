"""Data cleanup readback contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_readback import (
    build_data_cleanup_readback_report,
    main as data_cleanup_readback_main,
    write_data_cleanup_readback_report,
)


def _gate_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_gate.v1",
        "generated_at": "2026-06-12T10:43:16+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "candidates": [
            {
                "group_id": "dup-pass",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
                "gate_status": "clear_but_delete_disabled",
                "reclaimable_bytes": 100,
                "candidate_paths": [
                    (
                        "reports/storage/csv_quarantine/"
                        "cn_market_full/000001.SZ.csv"
                    )
                ],
                "retained_paths": [
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000001.SZ_raw.csv"
                    )
                ],
            },
            {
                "group_id": "dup-mismatch",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
                "gate_status": "clear_but_delete_disabled",
                "reclaimable_bytes": 200,
                "candidate_paths": [
                    (
                        "reports/storage/csv_quarantine/"
                        "cn_market_full/000002.SZ.csv"
                    )
                ],
                "retained_paths": [
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000002.SZ_raw.csv"
                    )
                ],
            },
            {
                "group_id": "dup-blocked",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
                "gate_status": "blocked",
                "reclaimable_bytes": 300,
                "candidate_paths": [
                    (
                        "reports/storage/csv_quarantine/"
                        "cn_market_full/000003.SZ.csv"
                    )
                ],
                "retained_paths": [
                    (
                        "data/raw_backups/tushare/daily/"
                        "full_a_000003.SZ_raw.csv"
                    )
                ],
            },
        ],
    }


def _write_pair(tmp_path, candidate_path, retained_path, *, same=True):
    candidate = tmp_path / candidate_path
    retained = tmp_path / retained_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    retained.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("symbol,trade_date\n000001.SZ,20260610\n")
    retained.write_text(
        "symbol,trade_date\n000001.SZ,20260610\n"
        if same
        else "symbol,trade_date\n000002.SZ,20260610\n"
    )


def _write_fixture_files(tmp_path):
    _write_pair(
        tmp_path,
        "reports/storage/csv_quarantine/cn_market_full/000001.SZ.csv",
        "data/raw_backups/tushare/daily/full_a_000001.SZ_raw.csv",
    )
    _write_pair(
        tmp_path,
        "reports/storage/csv_quarantine/cn_market_full/000002.SZ.csv",
        "data/raw_backups/tushare/daily/full_a_000002.SZ_raw.csv",
        same=False,
    )


def test_cleanup_readback_passes_matching_hash(tmp_path):
    _write_fixture_files(tmp_path)
    report = build_data_cleanup_readback_report(
        _gate_fixture(tmp_path),
        repo_root=tmp_path,
        max_candidates=None,
    )
    candidates = {
        item["group_id"]: item for item in report["candidates"]
    }

    passed = candidates["dup-pass"]
    assert passed["readback_status"] == "hash_readback_passed"
    assert passed["delete_allowed"] is False
    assert "candidate_hash_matches_retained" in passed["passed_checks"]
    assert "manual_delete_approval_required" in passed["pending_checks"]


def test_cleanup_readback_blocks_mismatch_and_skips_gate_blocked(tmp_path):
    _write_fixture_files(tmp_path)
    report = build_data_cleanup_readback_report(
        _gate_fixture(tmp_path),
        repo_root=tmp_path,
        max_candidates=None,
    )
    candidates = {
        item["group_id"]: item for item in report["candidates"]
    }

    assert "dup-blocked" not in candidates
    mismatch = candidates["dup-mismatch"]
    assert mismatch["readback_status"] == "blocked"
    assert "candidate_hash_matches_retained" in mismatch["failed_checks"]
    assert report["summary"]["hash_readback_passed_count"] == 1
    assert report["summary"]["blocked_count"] == 1
    assert report["summary"]["verified_reclaim_bytes"] == 100


def test_cleanup_readback_writes_reports_and_cli_output(tmp_path, capsys):
    _write_fixture_files(tmp_path)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(_gate_fixture(tmp_path)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "readback"

    written = write_data_cleanup_readback_report(
        gate_path,
        root=tmp_path,
        output_dir=output_dir,
        max_candidates=1,
        max_markdown_candidates=1,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_readback.json").read_text()
    )
    assert written["json"] == str(output_dir / "data_cleanup_readback.json")
    assert payload["schema_version"] == "myquant.data_cleanup_readback.v1"
    assert payload["delete_candidate_count"] == 0
    assert payload["summary"]["reviewed_candidate_count"] == 1
    assert payload["summary"]["skipped_by_limit_count"] == 1
    markdown = (output_dir / "data_cleanup_readback.md").read_text()
    assert markdown.startswith("# Data Cleanup Readback Report")

    exit_code = data_cleanup_readback_main(
        [
            "--root",
            str(tmp_path),
            "--gate-json",
            str(gate_path),
            "--output-dir",
            str(output_dir / "cli"),
            "--max-candidates",
            "1",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup readback mode: dry-run" in stdout
    assert "delete candidates: 0" in stdout
