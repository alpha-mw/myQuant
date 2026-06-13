"""Data cleanup whitelist contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_whitelist import (
    build_data_cleanup_whitelist,
    main as data_cleanup_whitelist_main,
    write_data_cleanup_whitelist,
)


def _readback_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_readback.v1",
        "generated_at": "2026-06-12T10:50:30+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "candidates": [
            {
                "group_id": "dup-pass",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
                "readback_status": "hash_readback_passed",
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
                "candidate_files": [
                    {
                        "relative_path": (
                            "reports/storage/csv_quarantine/"
                            "cn_market_full/000001.SZ.csv"
                        ),
                        "size_bytes": 100,
                        "sha256": "abc",
                    }
                ],
                "retained_files": [
                    {
                        "relative_path": (
                            "data/raw_backups/tushare/daily/"
                            "full_a_000001.SZ_raw.csv"
                        ),
                        "size_bytes": 100,
                        "sha256": "abc",
                    }
                ],
            },
            {
                "group_id": "dup-blocked",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
                "readback_status": "blocked",
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
                "candidate_files": [],
                "retained_files": [],
            },
        ],
    }


def test_cleanup_whitelist_keeps_passed_items_no_execute(tmp_path):
    whitelist = build_data_cleanup_whitelist(_readback_fixture(tmp_path))

    assert whitelist["schema_version"] == "myquant.data_cleanup_whitelist.v1"
    assert whitelist["delete_candidate_count"] == 0
    assert whitelist["execute_allowed_count"] == 0
    assert whitelist["summary"]["whitelist_item_count"] == 1
    assert whitelist["summary"]["candidate_file_count"] == 1
    assert whitelist["summary"]["potential_reclaim_bytes"] == 100
    assert whitelist["summary"]["manual_approval_required_count"] == 1
    assert whitelist["approval_packet"]["approval_status_summary"] == {
        "pending_manual_approval": 1
    }
    assert whitelist["approval_packet"]["candidate_type_batches"] == [
        {
            "candidate_type": "quarantine_restore_mirror",
            "item_count": 1,
            "candidate_file_count": 1,
            "potential_reclaim_bytes": 100,
            "execute_allowed_count": 0,
        }
    ]
    assert whitelist["approval_packet"]["top_reclaim_items"] == [
        {
            "group_id": "dup-pass",
            "candidate_type": "quarantine_restore_mirror",
            "approval_status": "pending_manual_approval",
            "reclaimable_bytes": 100,
            "candidate_file_count": 1,
            "first_candidate_path": (
                "reports/storage/csv_quarantine/"
                "cn_market_full/000001.SZ.csv"
            ),
        }
    ]

    item = whitelist["items"][0]
    assert item["group_id"] == "dup-pass"
    assert item["approval_status"] == "pending_manual_approval"
    assert item["execute_allowed"] is False
    assert item["delete_allowed"] is False
    assert item["candidate_sha256"] == ["abc"]
    assert item["retained_sha256"] == ["abc"]
    assert item["rollback_source_paths"] == [
        "data/raw_backups/tushare/daily/full_a_000001.SZ_raw.csv"
    ]
    assert (
        "quant-investor market storage-validate-clean --market CN"
        in item["required_pre_delete_gates"]
    )
    assert (
        "quant-investor market storage-validate-clean --market CN"
        in whitelist["required_pre_delete_gates"]
    )


def test_cleanup_whitelist_accepts_restore_readback_groups(tmp_path):
    readback = {
        "schema_version": "myquant.data_cleanup_restore_readback.v1",
        "generated_at": "2026-06-12T11:52:04+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "groups": [
            {
                "group_id": "restore-pass",
                "candidate_type": "restore_source_duplicate_review",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "readback_status": "retained_copy_readback_passed",
                "delete_allowed": False,
                "reclaimable_bytes": 66,
                "candidate_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_old_raw.csv"
                ],
                "retained_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_new_raw.csv"
                ],
                "candidate_files": [
                    {
                        "relative_path": (
                            "data/raw_backups/tushare/daily/"
                            "full_a_000001.SZ_old_raw.csv"
                        ),
                        "size_bytes": 66,
                        "sha256": "abc",
                    }
                ],
                "retained_files": [
                    {
                        "relative_path": (
                            "data/raw_backups/tushare/daily/"
                            "full_a_000001.SZ_new_raw.csv"
                        ),
                        "size_bytes": 66,
                        "sha256": "abc",
                    }
                ],
            },
            {
                "group_id": "restore-blocked",
                "candidate_type": "restore_source_duplicate_review",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "readback_status": "blocked",
                "delete_allowed": False,
                "reclaimable_bytes": 99,
                "candidate_paths": [],
                "retained_paths": [],
                "candidate_files": [],
                "retained_files": [],
            },
        ],
    }

    whitelist = build_data_cleanup_whitelist(readback)

    assert whitelist["delete_candidate_count"] == 0
    assert whitelist["execute_allowed_count"] == 0
    assert whitelist["summary"]["whitelist_item_count"] == 1
    assert whitelist["summary"]["candidate_type_summary"] == {
        "restore_source_duplicate_review": 1,
    }
    item = whitelist["items"][0]
    assert item["group_id"] == "restore-pass"
    assert item["candidate_type"] == "restore_source_duplicate_review"
    assert item["candidate_sha256"] == ["abc"]
    assert item["retained_sha256"] == ["abc"]
    assert item["approval_status"] == "pending_manual_approval"
    assert item["execute_allowed"] is False


def test_cleanup_whitelist_writes_reports_and_cli_output(tmp_path, capsys):
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_fixture(tmp_path)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "whitelist"

    written = write_data_cleanup_whitelist(
        readback_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_items=1,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_whitelist.json").read_text()
    )
    assert written["json"] == str(output_dir / "data_cleanup_whitelist.json")
    assert payload["execute_allowed_count"] == 0
    assert payload["summary"]["manual_approval_required_count"] == 1
    markdown = (output_dir / "data_cleanup_whitelist.md").read_text()
    assert markdown.startswith("# Data Cleanup Approval Whitelist")
    assert "Manual approval required: 1" in markdown

    exit_code = data_cleanup_whitelist_main(
        [
            "--root",
            str(tmp_path),
            "--readback-json",
            str(readback_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup whitelist mode: no-execute" in stdout
    assert "execute allowed: 0" in stdout
