"""Data cleanup inventory contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_inventory import (
    build_data_duplicate_audit,
    main as data_cleanup_main,
    write_data_duplicate_audit_manifest,
)


def test_data_duplicate_audit_groups_duplicates_without_delete_permission(
    tmp_path,
):
    raw_copy = (
        tmp_path / "data" / "raw_backups" / "tushare" / "daily_basic.csv"
    )
    quarantine_copy = (
        tmp_path
        / "reports"
        / "storage"
        / "csv_quarantine"
        / "daily_basic.csv"
    )
    factor_file = (
        tmp_path / "data" / "factor_readiness" / "tushare" / "unique.json"
    )

    raw_copy.parent.mkdir(parents=True)
    quarantine_copy.parent.mkdir(parents=True)
    factor_file.parent.mkdir(parents=True)
    raw_copy.write_text(
        "ts_code,trade_date\n000001.SZ,20260610\n",
        encoding="utf-8",
    )
    quarantine_copy.write_text(
        "ts_code,trade_date\n000001.SZ,20260610\n",
        encoding="utf-8",
    )
    factor_file.write_text('{"status":"ready"}\n', encoding="utf-8")

    manifest = build_data_duplicate_audit(tmp_path, max_file_bytes=1024)

    assert manifest["schema_version"] == "myquant.data_duplicate_audit.v1"
    assert manifest["delete_candidate_count"] == 0
    assert manifest["summary"]["duplicate_group_count"] == 1
    assert manifest["summary"]["duplicate_file_count"] == 2

    group = manifest["duplicate_groups"][0]
    assert group["delete_allowed"] is False
    assert sorted(group["files"]) == [
        "data/raw_backups/tushare/daily_basic.csv",
        "reports/storage/csv_quarantine/daily_basic.csv",
    ]

    files = {item["relative_path"]: item for item in manifest["files"]}
    assert files["data/raw_backups/tushare/daily_basic.csv"][
        "delete_allowed"
    ] is False
    assert files["reports/storage/csv_quarantine/daily_basic.csv"][
        "delete_allowed"
    ] is False
    assert files["data/factor_readiness/tushare/unique.json"][
        "duplicate_group_id"
    ] is None


def test_data_duplicate_audit_marks_oversized_files_without_hashing(tmp_path):
    large_file = (
        tmp_path / "data" / "cleaning_reports" / "tushare" / "large.json"
    )
    large_file.parent.mkdir(parents=True)
    large_file.write_text("abcdef", encoding="utf-8")

    manifest = build_data_duplicate_audit(tmp_path, max_file_bytes=3)
    files = {item["relative_path"]: item for item in manifest["files"]}
    item = files["data/cleaning_reports/tushare/large.json"]

    assert item["hash_status"] == "skipped_oversize"
    assert item["sha256"] is None
    assert item["delete_allowed"] is False
    assert manifest["summary"]["skipped_file_count"] == 1


def test_data_duplicate_audit_can_truncate_file_detail_rows(tmp_path):
    for index in range(3):
        file_path = (
            tmp_path
            / "data"
            / "raw_backups"
            / "tushare"
            / f"unique_{index}.csv"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"row,{index}\n", encoding="utf-8")

    manifest = build_data_duplicate_audit(
        tmp_path,
        max_file_bytes=1024,
        max_file_rows=1,
    )

    assert manifest["summary"]["scanned_file_count"] == 3
    assert manifest["summary"]["included_file_count"] == 1
    assert manifest["summary"]["files_truncated"] is True
    assert len(manifest["files"]) == 1


def test_data_duplicate_audit_writes_reports(tmp_path, capsys):
    cache_file = (
        tmp_path / "data" / "cn_market_full" / ".cache" / "symbols.json"
    )
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('["000001.SZ"]\n', encoding="utf-8")
    output_dir = (
        tmp_path / "reports" / "project_cleanup" / "data_audit_fixture"
    )

    written = write_data_duplicate_audit_manifest(
        tmp_path,
        output_dir=output_dir,
        max_file_bytes=1024,
    )

    payload = json.loads(
        (output_dir / "data_duplicate_audit.json").read_text()
    )
    assert written["json"] == str(output_dir / "data_duplicate_audit.json")
    assert payload["delete_candidate_count"] == 0
    assert payload["summary"]["hashed_file_count"] == 1
    assert payload["summary"]["files_truncated"] is False
    assert (output_dir / "data_duplicate_audit.md").read_text().startswith(
        "# Data Duplicate Audit"
    )

    exit_code = data_cleanup_main(
        [
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir / "cli"),
            "--max-file-mb",
            "1",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data duplicate audit mode: dry-run" in stdout
    assert "delete candidates: 0" in stdout
