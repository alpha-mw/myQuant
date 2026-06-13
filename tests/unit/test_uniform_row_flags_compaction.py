from __future__ import annotations

import json
from pathlib import Path

from scripts.compact_uniform_row_flags import (
    CONFIRM_TOKEN,
    build_uniform_row_flags_compaction_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_fixture(
    root: Path,
    *,
    uniform: bool = True,
) -> tuple[dict, dict, Path, str]:
    row_flags = (
        root
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_row_flags.csv"
    )
    row_flags.parent.mkdir(parents=True, exist_ok=True)
    if uniform:
        row_flags.write_text(
            "row_index,missing_required_column,quarantined,dropped\n"
            "0,True,False,False\n"
            "1,True,False,False\n",
            encoding="utf-8",
        )
    else:
        row_flags.write_text(
            "row_index,missing_required_column,quarantined,dropped\n"
            "0,True,False,False\n"
            "1,False,False,False\n",
            encoding="utf-8",
        )
    row_flags_rel = row_flags.relative_to(root).as_posix()
    cleaning_report = row_flags.with_name(
        "full_a_000001.SZ_20260312_cleaning_report.json"
    )
    _write_json(
        cleaning_report,
        {
            "row_flags_path": row_flags_rel,
            "metadata": {},
        },
    )
    policy = {
        "groups": [
            {
                "group_id": "dup-row",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [row_flags_rel],
                "retained_paths": [
                    "data/cleaning_reports/tushare/daily/"
                    "full_a_000002.SZ_20260312_row_flags.csv"
                ],
                "path_roles": ["cleaning_row_flags"],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-row",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [row_flags_rel],
                "referenced_candidate_paths": [row_flags_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }
    return policy, reference_audit, row_flags, cleaning_report.relative_to(root).as_posix()


def test_uniform_row_flags_compaction_dry_run_rewrites_nothing(tmp_path):
    policy, reference_audit, row_flags, cleaning_report_path = _write_fixture(tmp_path)

    report = build_uniform_row_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert row_flags.exists()
    payload = json.loads((tmp_path / cleaning_report_path).read_text())
    assert payload["row_flags_path"] == row_flags.relative_to(tmp_path).as_posix()


def test_uniform_row_flags_compaction_apply_rewrites_owner_then_deletes(tmp_path):
    policy, reference_audit, row_flags, cleaning_report_path = _write_fixture(tmp_path)
    row_flags_rel = row_flags.relative_to(tmp_path).as_posix()

    report = build_uniform_row_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
    )

    assert report["execution_performed"] is True
    assert report["summary"]["compacted_count"] == 1
    assert report["summary"]["compacted_reclaim_bytes"] > 0
    assert report["summary"]["references_rewritten_count"] == 1
    assert not row_flags.exists()
    payload = json.loads((tmp_path / cleaning_report_path).read_text())
    assert payload["row_flags_path"] is None
    assert payload["metadata"]["row_flags_compacted"] is True
    assert payload["metadata"]["row_flags_path_suppressed"] is True
    assert payload["metadata"]["row_flags_planned_path"] == row_flags_rel
    assert payload["metadata"]["row_flags_row_count"] == 2
    assert payload["metadata"]["row_flags_uniform_values"] == {
        "missing_required_column": True,
        "quarantined": False,
        "dropped": False,
    }


def test_uniform_row_flags_compaction_blocks_non_uniform_files(tmp_path):
    policy, reference_audit, row_flags, _cleaning_report_path = _write_fixture(
        tmp_path,
        uniform=False,
    )

    report = build_uniform_row_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "column is not uniform" in report["items"][0]["errors"][0]
    assert row_flags.exists()


def test_uniform_row_flags_compaction_blocks_external_references(tmp_path):
    policy, reference_audit, row_flags, _cleaning_report_path = _write_fixture(tmp_path)
    row_flags_rel = row_flags.relative_to(tmp_path).as_posix()
    extra = tmp_path / "results" / "strategy_records" / "manual.md"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(f"manual reference {row_flags_rel}\n", encoding="utf-8")

    report = build_uniform_row_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "unexpected references" in report["items"][0]["errors"][0]
    assert row_flags.exists()


def test_uniform_row_flags_compaction_ignores_derived_csv_inventory_refs(tmp_path):
    policy, reference_audit, row_flags, _cleaning_report_path = _write_fixture(
        tmp_path
    )
    row_flags_rel = row_flags.relative_to(tmp_path).as_posix()
    inventory = tmp_path / "reports" / "storage" / "csv_inventory_20260610.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(f"derived reference {row_flags_rel}\n", encoding="utf-8")

    report = build_uniform_row_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert report["summary"]["ignored_external_reference_count"] == 1
    assert report["items"][0]["ignored_external_reference_count"] == 1
    assert row_flags.exists()
