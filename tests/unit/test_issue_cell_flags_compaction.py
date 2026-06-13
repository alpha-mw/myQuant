from __future__ import annotations

import json
from pathlib import Path

from scripts.compact_issue_cell_flags import (
    CONFIRM_TOKEN,
    build_issue_cell_flags_compaction_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fixture(
    root: Path,
    *,
    matching_issue: bool = True,
) -> tuple[dict, dict, Path, str, str]:
    cell_flags = (
        root
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_cell_flags.csv"
    )
    cell_flags.parent.mkdir(parents=True, exist_ok=True)
    cell_flags.write_text(
        "row_index,primary_key_json,column,issue_code,severity,value,message\n"
        '3,"{""trade_date"":""2026-03-14"",""ts_code"":null}",'
        "ts_code,invalid_ts_code,warning,,invalid ts_code in ts_code\n",
        encoding="utf-8",
    )
    cell_flags_rel = cell_flags.relative_to(root).as_posix()
    cleaning_report = cell_flags.with_name(
        "full_a_000001.SZ_20260312_cleaning_report.json"
    )
    cleaning_report_rel = cleaning_report.relative_to(root).as_posix()
    factor_report = (
        root
        / "data"
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_factor_readiness_report.json"
    )
    factor_report_rel = factor_report.relative_to(root).as_posix()
    issue = {
        "column": "ts_code",
        "issue_code": "invalid_ts_code",
        "message": "invalid ts_code in ts_code"
        if matching_issue
        else "different message",
        "metadata": {},
        "primary_key": {"trade_date": "2026-03-14", "ts_code": None},
        "row_index": 3,
        "schema_version": "2026-04-27.tushare-data-cleaning.v1",
        "severity": "warning",
        "table_name": "daily",
    }
    payload = {
        "cell_flags_path": cell_flags_rel,
        "issues": [
            issue,
            {
                "column": None,
                "issue_code": "quarantined_row",
                "message": "1 rows quarantined",
                "metadata": {"quarantined_row_count": 1},
                "primary_key": {},
                "row_index": None,
                "severity": "warning",
                "table_name": "daily",
            },
        ],
        "metadata": {"factor_readiness_report_path": factor_report_rel},
    }
    _write_json(cleaning_report, payload)
    _write_json(
        factor_report,
        {
            "issues": [
                {
                    "issue_code": "missing_trade_cal",
                    "message": "trade calendar table is required",
                    "metadata": {},
                    "severity": "blocker",
                    "table_name": "trade_cal",
                }
            ],
            "table_reports": {"daily": dict(payload)},
        },
    )
    policy = {
        "groups": [
            {
                "group_id": "dup-cell",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [cell_flags_rel],
                "retained_paths": [
                    "data/cleaning_reports/tushare/daily/"
                    "full_a_000002.SZ_20260312_cell_flags.csv"
                ],
                "path_roles": ["cleaning_cell_flags"],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-cell",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [cell_flags_rel],
                "referenced_candidate_paths": [cell_flags_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }
    return policy, reference_audit, cell_flags, cleaning_report_rel, factor_report_rel


def test_issue_cell_flags_compaction_dry_run_rewrites_nothing(tmp_path):
    policy, reference_audit, cell_flags, cleaning_rel, _factor_rel = _fixture(
        tmp_path
    )

    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert cell_flags.exists()
    payload = json.loads((tmp_path / cleaning_rel).read_text())
    assert payload["cell_flags_path"] == cell_flags.relative_to(tmp_path).as_posix()


def test_issue_cell_flags_compaction_apply_rewrites_owner_then_deletes(tmp_path):
    policy, reference_audit, cell_flags, cleaning_rel, factor_rel = _fixture(
        tmp_path
    )
    cell_flags_rel = cell_flags.relative_to(tmp_path).as_posix()

    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
    )

    assert report["execution_performed"] is True
    assert report["summary"]["compacted_count"] == 1
    assert report["summary"]["references_rewritten_count"] == 2
    assert report["summary"]["issue_row_count"] == 1
    assert not cell_flags.exists()
    cleaning_report = json.loads((tmp_path / cleaning_rel).read_text())
    assert cleaning_report["cell_flags_path"] is None
    metadata = cleaning_report["metadata"]
    assert metadata["cell_flags_issue_backed_compacted"] is True
    assert metadata["cell_flags_path_suppressed"] is True
    assert metadata["cell_flags_planned_path"] == cell_flags_rel
    assert metadata["cell_flags_issue_row_count"] == 1
    factor_report = json.loads((tmp_path / factor_rel).read_text())
    table_report = factor_report["table_reports"]["daily"]
    assert table_report["cell_flags_path"] is None
    assert table_report["metadata"]["cell_flags_issue_backed_compacted"] is True


def test_issue_cell_flags_compaction_blocks_when_issue_rows_not_embedded(tmp_path):
    policy, reference_audit, cell_flags, _cleaning_rel, _factor_rel = _fixture(
        tmp_path,
        matching_issue=False,
    )

    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "not embedded" in report["items"][0]["errors"][0]
    assert cell_flags.exists()


def test_issue_cell_flags_compaction_blocks_external_references(tmp_path):
    policy, reference_audit, cell_flags, _cleaning_rel, _factor_rel = _fixture(
        tmp_path
    )
    cell_flags_rel = cell_flags.relative_to(tmp_path).as_posix()
    extra = tmp_path / "results" / "strategy_records" / "manual.md"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(f"manual reference {cell_flags_rel}\n", encoding="utf-8")

    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "unexpected references" in report["items"][0]["errors"][0]
    assert cell_flags.exists()


def test_issue_cell_flags_compaction_ignores_derived_csv_inventory_refs(tmp_path):
    policy, reference_audit, cell_flags, _cleaning_rel, _factor_rel = _fixture(
        tmp_path
    )
    cell_flags_rel = cell_flags.relative_to(tmp_path).as_posix()
    inventory = tmp_path / "reports" / "storage" / "csv_inventory_20260610.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(f"derived reference {cell_flags_rel}\n", encoding="utf-8")

    report = build_issue_cell_flags_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert report["summary"]["ignored_external_reference_count"] == 1
    assert report["items"][0]["ignored_external_reference_count"] == 1
    assert cell_flags.exists()
