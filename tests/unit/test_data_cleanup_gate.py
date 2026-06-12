"""Data cleanup gate contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_gate import (
    build_data_cleanup_gate_report,
    main as data_cleanup_gate_main,
    write_data_cleanup_gate_report,
)


def _plan_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_plan.v1",
        "generated_at": "2026-06-12T10:35:25+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "candidates": [
            {
                "group_id": "dup-clear",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
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
                "group_id": "dup-referenced",
                "candidate_type": "quarantine_restore_mirror",
                "delete_allowed": False,
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
        ],
    }


def _write_fixture_files(tmp_path):
    paths = [
        "reports/storage/csv_quarantine/cn_market_full/000001.SZ.csv",
        "reports/storage/csv_quarantine/cn_market_full/000002.SZ.csv",
        "data/raw_backups/tushare/daily/full_a_000001.SZ_raw.csv",
        "data/raw_backups/tushare/daily/full_a_000002.SZ_raw.csv",
    ]
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("symbol,trade_date\n", encoding="utf-8")


def test_cleanup_gate_keeps_clear_candidates_delete_disabled(tmp_path):
    _write_fixture_files(tmp_path)
    report = build_data_cleanup_gate_report(
        _plan_fixture(tmp_path),
        repo_root=tmp_path,
    )
    candidates = {
        item["group_id"]: item for item in report["candidates"]
    }

    clear = candidates["dup-clear"]
    assert clear["gate_status"] == "clear_but_delete_disabled"
    assert clear["delete_allowed"] is False
    assert clear["failed_checks"] == []
    assert "delete_disabled_by_policy" in clear["blockers"]
    assert report["delete_candidate_count"] == 0


def test_cleanup_gate_blocks_runtime_and_strategy_references(tmp_path):
    _write_fixture_files(tmp_path)
    referenced = "reports/storage/csv_quarantine/cn_market_full/000002.SZ.csv"
    strategy_note = (
        tmp_path
        / "results"
        / "strategy_records"
        / "CN"
        / "note.md"
    )
    strategy_note.parent.mkdir(parents=True)
    strategy_note.write_text(f"manual evidence path: {referenced}\n")
    latest = tmp_path / "data" / "parquet" / "cn" / "_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"manifest_path": "data/parquet/cn/manifest.json"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "data" / "parquet" / "cn" / "manifest.json"
    manifest.write_text(
        json.dumps({"quarantine_source": referenced}),
        encoding="utf-8",
    )

    report = build_data_cleanup_gate_report(
        _plan_fixture(tmp_path),
        repo_root=tmp_path,
    )
    candidates = {
        item["group_id"]: item for item in report["candidates"]
    }
    blocked = candidates["dup-referenced"]

    assert blocked["gate_status"] == "blocked"
    assert "runtime_reference_check" in blocked["failed_checks"]
    assert "strategy_record_reference_check" in blocked["failed_checks"]
    assert blocked["runtime_references"][referenced] == [
        "data/parquet/cn/manifest.json"
    ]
    assert blocked["strategy_references"][referenced] == [
        "results/strategy_records/CN/note.md"
    ]
    assert report["summary"]["blocked_count"] == 1
    assert report["summary"]["runtime_candidate_reference_count"] == 1
    assert report["summary"]["strategy_candidate_reference_count"] == 1


def test_cleanup_gate_writes_reports_and_cli_output(tmp_path, capsys):
    _write_fixture_files(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(_plan_fixture(tmp_path)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "gate"

    written = write_data_cleanup_gate_report(
        plan_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_candidates=1,
    )

    payload = json.loads((output_dir / "data_cleanup_gate.json").read_text())
    assert written["json"] == str(output_dir / "data_cleanup_gate.json")
    assert payload["schema_version"] == "myquant.data_cleanup_gate.v1"
    assert payload["delete_candidate_count"] == 0
    markdown = (output_dir / "data_cleanup_gate.md").read_text()
    assert markdown.startswith("# Data Cleanup Gate Report")
    assert "Candidate table truncated to 1 of 2 rows" in markdown

    exit_code = data_cleanup_gate_main(
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
    assert "data cleanup gate mode: dry-run" in stdout
    assert "delete candidates: 0" in stdout
