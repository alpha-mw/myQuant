from __future__ import annotations

import json
from pathlib import Path

from scripts.data_cleanup_reference_rewrite import (
    CONFIRM_TOKEN,
    RESTORE_SOURCE_CONFIRM_TOKEN,
    build_reference_rewrite_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_reference_rewrite_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    candidate = (
        tmp_path
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_row_flags.csv"
    )
    retained = candidate.with_name("full_a_000001.SZ_20260311_row_flags.csv")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("issue_code,row\n", encoding="utf-8")
    retained.write_text("issue_code,row\n", encoding="utf-8")
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    retained_rel = retained.relative_to(tmp_path).as_posix()

    cleaning_report = candidate.with_name(
        "full_a_000001.SZ_20260312_cleaning_report.json"
    )
    factor_report = (
        tmp_path
        / "data"
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_factor_readiness_report.json"
    )
    factor_report_rel = factor_report.relative_to(tmp_path).as_posix()
    _write_json(
        cleaning_report,
        {
            "row_flags_path": candidate_rel,
            "metadata": {"factor_readiness_report_path": factor_report_rel},
        },
    )
    _write_json(
        factor_report,
        {"table_reports": {"daily": {"row_flags_path": candidate_rel}}},
    )
    _write_json(
        tmp_path / "reports" / "storage" / "csv_inventory_20260312.json",
        {"historical_path": candidate_rel},
    )
    policy = {
        "schema_version": "policy",
        "groups": [
            {
                "group_id": "dup-1",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": candidate.stat().st_size,
                "candidate_paths": [candidate_rel],
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "schema_version": "reference",
        "groups": [
            {
                "group_id": "dup-1",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "candidate_paths": [candidate_rel],
                "referenced_candidate_paths": [candidate_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }
    return policy, reference_audit, candidate, retained, cleaning_report


def test_reference_rewrite_dry_run_and_apply_rewrites_owner_reports(tmp_path):
    policy, reference_audit, candidate, retained, cleaning_report = (
        _seed_reference_rewrite_fixture(tmp_path)
    )

    dry_run = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert dry_run["summary"]["would_rewrite_delete_count"] == 1
    assert dry_run["summary"]["blocked_count"] == 0
    assert dry_run["summary"]["references_rewritten_count"] == 2
    assert dry_run["summary"]["scan_mode"] == "bounded_owner_external"
    assert dry_run["summary"]["scan_skipped_derived_reference_file_count"] == 1
    assert dry_run["items"][0]["ignored_reference_count"] == 0
    assert candidate.exists()

    blocked = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token="WRONG",
        max_groups=-1,
    )

    assert blocked["summary"]["rewritten_deleted_count"] == 0
    assert blocked["summary"]["blocked_count"] == 1
    assert candidate.exists()

    applied = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
        max_groups=-1,
    )

    assert applied["execution_performed"] is True
    assert applied["summary"]["rewritten_deleted_count"] == 1
    assert applied["summary"]["rewritten_deleted_reclaim_bytes"] == len(
        "issue_code,row\n"
    )
    assert not candidate.exists()
    retained_rel = retained.relative_to(tmp_path).as_posix()
    cleaning_payload = json.loads(cleaning_report.read_text(encoding="utf-8"))
    assert cleaning_payload["row_flags_path"] == retained_rel
    assert cleaning_payload["metadata"]["duplicate_reference_rewritten"] is True


def test_reference_rewrite_handles_multiple_candidates_in_one_group(tmp_path):
    base = tmp_path / "data" / "cleaning_reports" / "tushare" / "daily"
    base.mkdir(parents=True, exist_ok=True)
    retained = base / "full_a_000001.SZ_20260311_row_flags.csv"
    candidates = [
        base / "full_a_000001.SZ_20260312_row_flags.csv",
        base / "full_a_000001.SZ_20260313_row_flags.csv",
    ]
    for path in [retained, *candidates]:
        path.write_text("issue_code,row\n", encoding="utf-8")
    retained_rel = retained.relative_to(tmp_path).as_posix()

    candidate_rels = []
    cleaning_reports = []
    for candidate in candidates:
        candidate_rel = candidate.relative_to(tmp_path).as_posix()
        candidate_rels.append(candidate_rel)
        stem = candidate.name.removesuffix("_row_flags.csv")
        cleaning_report = base / f"{stem}_cleaning_report.json"
        factor_report = (
            tmp_path
            / "data"
            / "factor_readiness"
            / "tushare"
            / "daily"
            / f"{stem}_factor_readiness_report.json"
        )
        factor_report_rel = factor_report.relative_to(tmp_path).as_posix()
        _write_json(
            cleaning_report,
            {
                "row_flags_path": candidate_rel,
                "metadata": {"factor_readiness_report_path": factor_report_rel},
            },
        )
        _write_json(
            factor_report,
            {"table_reports": {"daily": {"row_flags_path": candidate_rel}}},
        )
        cleaning_reports.append(cleaning_report)

    policy = {
        "groups": [
            {
                "group_id": "dup-multi",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": len("issue_code,row\n") * len(candidates),
                "candidate_paths": candidate_rels,
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-multi",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "candidate_paths": candidate_rels,
                "referenced_candidate_paths": candidate_rels,
                "unreferenced_candidate_paths": [],
            }
        ],
    }

    dry_run = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert dry_run["summary"]["selected_group_count"] == 1
    assert dry_run["summary"]["selected_candidate_path_count"] == 2
    assert dry_run["summary"]["would_rewrite_delete_count"] == 2
    assert dry_run["summary"]["references_rewritten_count"] == 4
    assert dry_run["summary"]["planned_reclaim_bytes"] == len("issue_code,row\n") * 2

    applied = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
        max_groups=-1,
    )

    assert applied["summary"]["rewritten_deleted_count"] == 2
    assert applied["summary"]["references_rewritten_count"] == 4
    assert all(not candidate.exists() for candidate in candidates)
    for cleaning_report in cleaning_reports:
        payload = json.loads(cleaning_report.read_text(encoding="utf-8"))
        assert payload["row_flags_path"] == retained_rel


def test_reference_rewrite_allows_cell_flags_owner_key(tmp_path):
    base = tmp_path / "data" / "cleaning_reports" / "tushare" / "daily"
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / "full_a_000001.SZ_20260312_cell_flags.csv"
    retained = base / "full_a_000001.SZ_20260311_cell_flags.csv"
    candidate.write_text("cell\n", encoding="utf-8")
    retained.write_text("cell\n", encoding="utf-8")
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    retained_rel = retained.relative_to(tmp_path).as_posix()
    _write_json(
        base / "full_a_000001.SZ_20260312_cleaning_report.json",
        {"cell_flags_path": candidate_rel},
    )
    policy = {
        "groups": [
            {
                "group_id": "dup-cell",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": len("cell\n"),
                "candidate_paths": [candidate_rel],
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-cell",
                "policy_class": "same_symbol_cleaning_artifact_duplicate",
                "risk_level": "medium",
                "candidate_paths": [candidate_rel],
                "referenced_candidate_paths": [candidate_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }

    report = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert report["summary"]["would_rewrite_delete_count"] == 1
    assert report["summary"]["references_rewritten_count"] == 1


def test_reference_rewrite_raw_backup_requires_restore_token(tmp_path):
    raw_dir = tmp_path / "data" / "raw_backups" / "tushare" / "daily"
    report_dir = tmp_path / "data" / "cleaning_reports" / "tushare" / "daily"
    factor_dir = tmp_path / "data" / "factor_readiness" / "tushare" / "daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)
    candidate = raw_dir / "full_a_000001.SZ_20260312_raw.csv"
    retained = raw_dir / "full_a_000001.SZ_20260311_raw.csv"
    candidate.write_text("ts_code,close\n000001.SZ,10\n", encoding="utf-8")
    retained.write_text("ts_code,close\n000001.SZ,10\n", encoding="utf-8")
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    retained_rel = retained.relative_to(tmp_path).as_posix()
    cleaning_report = report_dir / "full_a_000001.SZ_20260312_cleaning_report.json"
    factor_report = factor_dir / "full_a_000001.SZ_20260312_factor_readiness_report.json"
    _write_json(cleaning_report, {"raw_backup_path": candidate_rel})
    _write_json(factor_report, {"raw_backup_path": candidate_rel})
    policy = {
        "groups": [
            {
                "group_id": "dup-raw",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": candidate.stat().st_size,
                "candidate_paths": [candidate_rel],
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-raw",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "risk_level": "medium",
                "candidate_paths": [candidate_rel],
                "referenced_candidate_paths": [candidate_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }

    dry_run = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert dry_run["summary"]["would_rewrite_delete_count"] == 1
    assert dry_run["summary"]["references_rewritten_count"] == 2

    blocked = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
        max_groups=-1,
    )

    assert blocked["summary"]["blocked_count"] == 1
    assert candidate.exists()

    applied = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=RESTORE_SOURCE_CONFIRM_TOKEN,
        max_groups=-1,
    )

    assert applied["summary"]["rewritten_deleted_count"] == 1
    assert applied["confirm_token_valid"] is True
    assert not candidate.exists()
    payload = json.loads(cleaning_report.read_text(encoding="utf-8"))
    assert payload["raw_backup_path"] == retained_rel
    assert payload["metadata"]["duplicate_reference_rewritten"] is True
    factor_payload = json.loads(factor_report.read_text(encoding="utf-8"))
    assert factor_payload["raw_backup_path"] == retained_rel
    assert factor_payload["metadata"]["duplicate_reference_rewritten"] is True


def test_reference_rewrite_skips_unreferenced_restore_source_candidates(tmp_path):
    raw_dir = tmp_path / "data" / "raw_backups" / "tushare" / "daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidate = raw_dir / "full_a_000001.SZ_20260312_raw.csv"
    retained = raw_dir / "full_a_000001.SZ_20260311_raw.csv"
    candidate.write_text("ts_code,close\n000001.SZ,10\n", encoding="utf-8")
    retained.write_text("ts_code,close\n000001.SZ,10\n", encoding="utf-8")
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    retained_rel = retained.relative_to(tmp_path).as_posix()
    policy = {
        "groups": [
            {
                "group_id": "dup-unreferenced",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": candidate.stat().st_size,
                "candidate_paths": [candidate_rel],
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-unreferenced",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "risk_level": "medium",
                "candidate_paths": [candidate_rel],
                "referenced_candidate_paths": [],
                "unreferenced_candidate_paths": [candidate_rel],
            }
        ],
    }

    report = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert report["summary"]["selected_group_count"] == 0
    assert report["summary"]["selected_candidate_path_count"] == 0
    assert report["summary"]["blocked_count"] == 0
    assert candidate.exists()


def test_reference_rewrite_factor_readiness_owner_paths(tmp_path):
    factor_dir = tmp_path / "data" / "factor_readiness" / "tushare" / "daily"
    report_dir = tmp_path / "data" / "cleaning_reports" / "tushare" / "daily"
    factor_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate = factor_dir / "hs300_000001.SZ_20260312_factor_ready_masks.json"
    retained = factor_dir / "full_a_000001.SZ_20260312_factor_ready_masks.json"
    _write_json(candidate, {"ready": True})
    _write_json(retained, {"ready": True})
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    retained_rel = retained.relative_to(tmp_path).as_posix()
    cleaning_report = report_dir / "hs300_000001.SZ_20260312_cleaning_report.json"
    factor_report = factor_dir / "hs300_000001.SZ_20260312_factor_readiness_report.json"
    _write_json(
        cleaning_report,
        {"metadata": {"factor_ready_masks_path": candidate_rel}},
    )
    _write_json(factor_report, {"status": "ok"})
    policy = {
        "groups": [
            {
                "group_id": "dup-factor",
                "policy_class": "same_symbol_factor_readiness_duplicate",
                "risk_level": "medium",
                "reclaimable_bytes": candidate.stat().st_size,
                "candidate_paths": [candidate_rel],
                "retained_paths": [retained_rel],
            }
        ],
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-factor",
                "policy_class": "same_symbol_factor_readiness_duplicate",
                "risk_level": "medium",
                "candidate_paths": [candidate_rel],
                "referenced_candidate_paths": [candidate_rel],
                "unreferenced_candidate_paths": [],
            }
        ],
    }

    applied = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=RESTORE_SOURCE_CONFIRM_TOKEN,
        max_groups=-1,
    )

    assert applied["summary"]["rewritten_deleted_count"] == 1
    assert applied["confirm_token_valid"] is True
    assert applied["summary"]["references_rewritten_count"] == 1
    assert not candidate.exists()
    payload = json.loads(cleaning_report.read_text(encoding="utf-8"))
    assert payload["metadata"]["factor_ready_masks_path"] == retained_rel


def test_reference_rewrite_blocks_unexpected_external_references(tmp_path):
    policy, reference_audit, candidate, _retained, _cleaning_report = (
        _seed_reference_rewrite_fixture(tmp_path)
    )
    candidate_rel = candidate.relative_to(tmp_path).as_posix()
    extra = tmp_path / "reports" / "daily" / "manual.md"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(f"still references {candidate_rel}\n", encoding="utf-8")

    report = build_reference_rewrite_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        max_groups=-1,
    )

    assert report["summary"]["would_rewrite_delete_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "unexpected_references" in report["items"][0]["errors"][0]
    assert candidate.exists()
