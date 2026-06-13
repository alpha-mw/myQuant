"""Data cleanup approval contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_approve import (
    APPROVAL_TOKEN,
    build_data_cleanup_approval,
    main as data_cleanup_approve_main,
    write_data_cleanup_approval,
)


def _item(group_id: str, *, approved: bool = False):
    approval_status = "approved_for_delete" if approved else "pending_manual_approval"
    return {
        "group_id": group_id,
        "candidate_type": "quarantine_restore_mirror",
        "approval_status": approval_status,
        "delete_allowed": approved,
        "execute_allowed": approved,
        "reclaimable_bytes": 100,
        "candidate_paths": [
            f"reports/storage/csv_quarantine/cn_market_full/{group_id}.csv"
        ],
        "retained_paths": [
            f"data/raw_backups/tushare/daily/{group_id}_raw.csv"
        ],
        "candidate_sha256": ["abc"],
        "retained_sha256": ["abc"],
        "candidate_size_bytes": [100],
        "retained_size_bytes": [100],
        "required_pre_delete_gates": [
            "manual_delete_approval_required",
        ],
        "rollback_source_paths": [
            f"data/raw_backups/tushare/daily/{group_id}_raw.csv"
        ],
        "reason": "fixture",
    }


def _whitelist_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_whitelist.v1",
        "generated_at": "2026-06-12T10:52:50+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "execute_allowed_count": 0,
        "summary": {
            "whitelist_item_count": 2,
            "manual_approval_required_count": 2,
            "potential_reclaim_bytes": 200,
        },
        "items": [_item("dup-0001"), _item("dup-0002")],
    }


def test_cleanup_approval_requires_token_and_preserves_whitelist(tmp_path):
    report = build_data_cleanup_approval(
        _whitelist_fixture(tmp_path),
        approve_group_ids=["dup-0001"],
        approval_token="WRONG",
    )

    assert report["execute_allowed_count"] == 0
    assert report["delete_candidate_count"] == 0
    assert report["approval_summary"]["approved_count"] == 0
    assert report["approval_summary"]["blocked_count"] == 1
    assert report["approval_summary"]["status_summary"] == {
        "blocked_approval_token_required": 1,
        "not_requested": 1,
    }
    assert [
        item["approval_status"] for item in report["items"]
    ] == ["pending_manual_approval", "pending_manual_approval"]


def test_cleanup_approval_approves_only_selected_groups(tmp_path):
    report = build_data_cleanup_approval(
        _whitelist_fixture(tmp_path),
        approve_group_ids=["dup-0002"],
        approval_token=APPROVAL_TOKEN,
    )

    assert report["execute_allowed_count"] == 1
    assert report["delete_candidate_count"] == 1
    assert report["summary"]["manual_approval_required_count"] == 1
    assert report["approval_summary"]["approved_count"] == 1
    assert report["approval_summary"]["status_summary"] == {
        "approved_for_delete": 1,
        "not_requested": 1,
    }
    items = {item["group_id"]: item for item in report["items"]}
    assert items["dup-0001"]["execute_allowed"] is False
    assert items["dup-0002"]["approval_status"] == "approved_for_delete"
    assert items["dup-0002"]["delete_allowed"] is True
    assert items["dup-0002"]["execute_allowed"] is True
    assert report["approval_packet"]["approval_status_summary"] == {
        "approved_for_delete": 1,
        "pending_manual_approval": 1,
    }


def test_cleanup_approval_writes_reports_and_cli_output(tmp_path, capsys):
    whitelist_path = tmp_path / "whitelist.json"
    whitelist_path.write_text(
        json.dumps(_whitelist_fixture(tmp_path)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "approval"

    written = write_data_cleanup_approval(
        whitelist_path,
        root=tmp_path,
        output_dir=output_dir,
        approve_group_ids=["dup-0001"],
        approval_token=APPROVAL_TOKEN,
    )

    payload = json.loads((output_dir / "data_cleanup_whitelist.json").read_text())
    markdown = (output_dir / "data_cleanup_approval.md").read_text()
    assert written["json"] == str(output_dir / "data_cleanup_whitelist.json")
    assert payload["execute_allowed_count"] == 1
    assert markdown.startswith("# Data Cleanup Approval Report")

    exit_code = data_cleanup_approve_main(
        [
            "--root",
            str(tmp_path),
            "--whitelist-json",
            str(whitelist_path),
            "--output-dir",
            str(output_dir / "cli"),
            "--approve-group",
            "dup-0002",
            "--approval-token",
            APPROVAL_TOKEN,
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup approval mode: token-gated" in stdout
    assert "approved: 1" in stdout
