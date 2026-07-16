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


def test_cleanup_gate_blocks_retired_intelligence_evidence(tmp_path):
    candidate = "data/parquet/cn/intelligence_daily/part.parquet"
    retained = "data/parquet/cn/intelligence_daily/part.parquet.sha256"
    for relative_path in (candidate, retained):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable retirement evidence")
    plan = {
        "schema_version": "myquant.data_cleanup_plan.v1",
        "candidates": [
            {
                "group_id": "retired-intelligence-mart",
                "candidate_type": "retirement_evidence",
                "candidate_paths": [candidate],
                "retained_paths": [retained],
            }
        ],
    }

    report = build_data_cleanup_gate_report(plan, repo_root=tmp_path)
    result = report["candidates"][0]

    assert result["gate_status"] == "blocked"
    assert "retirement_evidence_protection_check" in result["failed_checks"]
    assert "candidate_is_protected_retirement_evidence" in result["blockers"]


def test_cleanup_gate_blocks_immutable_market_and_pit_generations(tmp_path):
    protected_paths = [
        "data/parquet/cn/_snapshots/snapshot-a/table/bars/part.parquet",
        (
            "data/parquet/cn/reference/_generations/"
            "pit-a/stock_basic_membership.parquet"
        ),
    ]
    retained = tmp_path / "retained"
    retained.mkdir()
    for relative_path in protected_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"active immutable generation")
    plan = {
        "candidates": [
            {
                "group_id": "active-generations",
                "candidate_paths": protected_paths,
                "retained_paths": ["retained"],
            }
        ]
    }

    result = build_data_cleanup_gate_report(
        plan,
        repo_root=tmp_path,
    )["candidates"][0]

    assert result["gate_status"] == "blocked"
    assert "active_runtime_path_check" in result["failed_checks"]


def test_cleanup_gate_blocks_parent_containing_retirement_evidence(tmp_path):
    candidate = tmp_path / "data" / "parquet" / "cn"
    candidate.mkdir(parents=True)
    retained = tmp_path / "retained"
    retained.mkdir()
    plan = {
        "candidates": [
            {
                "group_id": "broad-parent",
                "candidate_paths": ["data/parquet/cn"],
                "retained_paths": ["retained"],
            }
        ]
    }

    result = build_data_cleanup_gate_report(plan, repo_root=tmp_path)["candidates"][0]

    assert result["gate_status"] == "blocked"
    assert "candidate_is_protected_retirement_evidence" in result["blockers"]


def test_cleanup_gate_canonicalizes_retirement_evidence_paths(tmp_path):
    actual = tmp_path / "reports" / "daily" / "historical.md"
    actual.parent.mkdir(parents=True)
    actual.write_text("history", encoding="utf-8")
    retained = tmp_path / "retained.md"
    retained.write_text("history", encoding="utf-8")
    plan = {
        "candidates": [
            {
                "group_id": "traversal",
                "candidate_paths": ["reports/tmp/../daily/historical.md"],
                "retained_paths": ["retained.md"],
            }
        ]
    }

    result = build_data_cleanup_gate_report(plan, repo_root=tmp_path)["candidates"][0]

    assert "candidate_is_protected_retirement_evidence" in result["blockers"]
    assert "retirement_evidence_protection_check" in result["failed_checks"]


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
