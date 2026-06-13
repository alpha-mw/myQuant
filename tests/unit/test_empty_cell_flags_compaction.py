from __future__ import annotations

import json
from pathlib import Path

from scripts.compact_empty_cell_flags import (
    CONFIRM_TOKEN,
    ORPHAN_DELETE_CONFIRM_TOKEN,
    build_empty_cell_flags_compaction_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_fixture(root: Path, *, empty: bool = True) -> tuple[Path, str, str]:
    cell_flags = (
        root
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_cell_flags.csv"
    )
    cell_flags.parent.mkdir(parents=True, exist_ok=True)
    cell_flags.write_text("\n" if empty else "issue_code,column\nbad,close\n", encoding="utf-8")
    relative_cell_flags = cell_flags.relative_to(root).as_posix()

    cleaning_report = cell_flags.with_name(
        "full_a_000001.SZ_20260312_cleaning_report.json"
    )
    factor_report = (
        root
        / "data"
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_factor_readiness_report.json"
    )
    relative_factor_report = factor_report.relative_to(root).as_posix()
    _write_json(
        cleaning_report,
        {
            "cell_flags_path": relative_cell_flags,
            "metadata": {"factor_readiness_report_path": relative_factor_report},
        },
    )
    _write_json(
        factor_report,
        {
            "table_reports": {
                "daily": {
                    "cell_flags_path": relative_cell_flags,
                    "metadata": {},
                }
            }
        },
    )
    return cell_flags, cleaning_report.relative_to(root).as_posix(), relative_factor_report


def _write_orphan_cell_flags(root: Path) -> Path:
    cell_flags = (
        root
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000002.SZ_20260312_cell_flags.csv"
    )
    cell_flags.parent.mkdir(parents=True, exist_ok=True)
    cell_flags.write_text("\n", encoding="utf-8")
    return cell_flags


def test_empty_cell_flags_compaction_dry_run_rewrites_nothing(tmp_path):
    cell_flags, cleaning_report_path, factor_report_path = _write_fixture(tmp_path)

    report = build_empty_cell_flags_compaction_report(repo_root=tmp_path)

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    item = report["items"][0]
    assert item["cleaning_report_path"] == cleaning_report_path
    assert item["factor_readiness_report_path"] == factor_report_path
    assert cell_flags.exists()
    assert json.loads((tmp_path / cleaning_report_path).read_text())["cell_flags_path"] == (
        cell_flags.relative_to(tmp_path).as_posix()
    )


def test_empty_cell_flags_compaction_apply_rewrites_references_then_deletes(tmp_path):
    cell_flags, cleaning_report_path, factor_report_path = _write_fixture(tmp_path)
    relative_cell_flags = cell_flags.relative_to(tmp_path).as_posix()

    report = build_empty_cell_flags_compaction_report(
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
    )

    assert report["summary"]["compacted_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert not cell_flags.exists()
    cleaning_report = json.loads((tmp_path / cleaning_report_path).read_text())
    assert cleaning_report["cell_flags_path"] is None
    assert cleaning_report["metadata"]["cell_flags_empty"] is True
    assert cleaning_report["metadata"]["cell_flags_path_suppressed"] is True
    assert cleaning_report["metadata"]["cell_flags_planned_path"] == relative_cell_flags
    factor_report = json.loads((tmp_path / factor_report_path).read_text())
    table_report = factor_report["table_reports"]["daily"]
    assert table_report["cell_flags_path"] is None
    assert table_report["metadata"]["cell_flags_empty"] is True
    assert table_report["metadata"]["cell_flags_path_suppressed"] is True
    assert json.dumps(cleaning_report).count(relative_cell_flags) == 1
    assert json.dumps(factor_report).count(relative_cell_flags) == 1


def test_empty_cell_flags_compaction_ignores_non_empty_files(tmp_path):
    cell_flags, _cleaning_report_path, _factor_report_path = _write_fixture(
        tmp_path,
        empty=False,
    )

    report = build_empty_cell_flags_compaction_report(repo_root=tmp_path)

    assert report["summary"]["candidate_count"] == 0
    assert cell_flags.exists()


def test_empty_cell_flags_orphan_delete_requires_explicit_token(tmp_path):
    cell_flags = _write_orphan_cell_flags(tmp_path)

    dry_run = build_empty_cell_flags_compaction_report(
        repo_root=tmp_path,
        allow_orphan_delete=True,
    )

    assert dry_run["summary"]["would_delete_orphan_count"] == 1
    assert dry_run["summary"]["planned_reclaim_bytes"] == 1
    assert dry_run["summary"]["blocked_count"] == 0
    assert cell_flags.exists()

    blocked = build_empty_cell_flags_compaction_report(
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
        allow_orphan_delete=True,
    )

    assert blocked["summary"]["orphan_deleted_count"] == 0
    assert blocked["summary"]["blocked_count"] == 1
    assert blocked["items"][0]["status"] == "blocked_orphan_confirm_token_required"
    assert cell_flags.exists()

    applied = build_empty_cell_flags_compaction_report(
        repo_root=tmp_path,
        apply=True,
        confirm_token=ORPHAN_DELETE_CONFIRM_TOKEN,
        allow_orphan_delete=True,
    )

    assert applied["execution_performed"] is True
    assert applied["summary"]["orphan_deleted_count"] == 1
    assert applied["summary"]["orphan_deleted_reclaim_bytes"] == 1
    assert applied["summary"]["blocked_count"] == 0
    assert not cell_flags.exists()


def test_empty_cell_flags_orphan_delete_blocks_external_references(tmp_path):
    cell_flags = _write_orphan_cell_flags(tmp_path)
    relative_cell_flags = cell_flags.relative_to(tmp_path).as_posix()
    _write_json(
        tmp_path
        / "data"
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000002.SZ_20260312_factor_readiness_report.json",
        {"cell_flags_path": relative_cell_flags},
    )

    report = build_empty_cell_flags_compaction_report(
        repo_root=tmp_path,
        allow_orphan_delete=True,
    )

    assert report["summary"]["would_delete_orphan_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert report["summary"]["external_reference_match_count"] == 1
    assert report["items"][0]["status"] == "blocked_orphan_referenced"
    assert cell_flags.exists()
