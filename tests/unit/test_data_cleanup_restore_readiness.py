"""Restore-source duplicate readiness report contract tests."""

from __future__ import annotations

import json

from scripts.data_cleanup_restore_policy import build_restore_source_policy
from scripts.data_cleanup_restore_readiness import (
    build_restore_readiness,
    main as restore_readiness_main,
    write_restore_readiness,
)


def _plan_fixture(tmp_path):
    return {
        "schema_version": "myquant.data_cleanup_plan.v1",
        "generated_at": "2026-06-12T10:00:00+00:00",
        "root": str(tmp_path),
        "delete_candidate_count": 0,
        "summary": {
            "candidate_group_count": 4,
            "candidate_file_count": 8,
            "potential_reclaim_bytes": 838,
        },
        "candidates": [
            {
                "group_id": "raw-unreferenced",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 300,
                "candidate_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260601_raw.csv"
                ],
                "retained_paths": [
                    "data/raw_backups/tushare/daily/full_a_000001.SZ_20260602_raw.csv"
                ],
            },
            {
                "group_id": "row-flags-unreferenced",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 200,
                "candidate_paths": [
                    "data/cleaning_reports/tushare/daily/full_a_000002.SZ_20260601_row_flags.csv"
                ],
                "retained_paths": [
                    "data/cleaning_reports/tushare/daily/full_a_000002.SZ_20260602_row_flags.csv"
                ],
            },
            {
                "group_id": "matrix-cross-symbol",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 188,
                "candidate_paths": [
                    "data/factor_readiness/tushare/daily/full_a_000003.SZ_20260601_matrix_coverage.json"
                ],
                "retained_paths": [
                    "data/factor_readiness/tushare/daily/full_a_000004.SZ_20260601_matrix_coverage.json"
                ],
            },
            {
                "group_id": "raw-referenced",
                "candidate_type": "restore_source_duplicate_review",
                "reclaimable_bytes": 150,
                "candidate_paths": [
                    "data/raw_backups/tushare/daily/full_a_000005.SZ_20260601_raw.csv"
                ],
                "retained_paths": [
                    "data/raw_backups/tushare/daily/full_a_000005.SZ_20260602_raw.csv"
                ],
            },
        ],
    }


def _reference_audit_fixture(policy):
    groups = []
    for group in policy["groups"]:
        candidate_paths = group["candidate_paths"]
        if group["group_id"] == "raw-referenced":
            groups.append(
                {
                    "group_id": group["group_id"],
                    "policy_class": group["policy_class"],
                    "risk_level": group["risk_level"],
                    "candidate_paths": candidate_paths,
                    "referenced_candidate_paths": candidate_paths,
                    "unreferenced_candidate_paths": [],
                    "reference_count": 1,
                }
            )
        else:
            groups.append(
                {
                    "group_id": group["group_id"],
                    "policy_class": group["policy_class"],
                    "risk_level": group["risk_level"],
                    "candidate_paths": candidate_paths,
                    "referenced_candidate_paths": [],
                    "unreferenced_candidate_paths": candidate_paths,
                    "reference_count": 0,
                }
            )
    return {
        "schema_version": "myquant.data_cleanup_restore_reference_audit.v1",
        "generated_at": "2026-06-12T11:00:00+00:00",
        "delete_candidate_count": 0,
        "summary": {
            "group_count": len(groups),
            "candidate_path_count": 4,
            "referenced_group_count": 1,
            "unreferenced_group_count": 3,
            "referenced_candidate_path_count": 1,
            "unreferenced_candidate_path_count": 3,
            "scan_mode": "candidate_owner_reports",
        },
        "groups": groups,
    }


def test_restore_readiness_classifies_reference_free_groups(tmp_path):
    policy = build_restore_source_policy(_plan_fixture(tmp_path))
    reference_audit = _reference_audit_fixture(policy)

    readiness = build_restore_readiness(policy, reference_audit)

    assert readiness["schema_version"] == "myquant.data_cleanup_restore_readiness.v1"
    assert readiness["delete_candidate_count"] == 0
    assert readiness["summary"]["group_count"] == 4
    assert readiness["summary"]["reference_free_group_count"] == 3
    assert readiness["summary"]["referenced_group_count"] == 1
    assert readiness["summary"]["reference_free_potential_reclaim_bytes"] == 688
    assert readiness["summary"]["readiness_class_summary"] == {
        "blocked_high_risk_policy": 1,
        "blocked_referenced_candidate": 1,
        "review_manifest_rewrite_required": 1,
        "review_retained_copy_readback_required": 1,
    }

    groups = {group["group_id"]: group for group in readiness["groups"]}
    assert groups["raw-unreferenced"]["readiness_class"] == (
        "review_retained_copy_readback_required"
    )
    assert groups["raw-unreferenced"]["delete_allowed"] is False
    assert "retained_copy_readback_required" in groups["raw-unreferenced"]["blockers"]
    assert groups["row-flags-unreferenced"]["readiness_class"] == (
        "review_manifest_rewrite_required"
    )
    assert "reference_rewrite_required" in groups["row-flags-unreferenced"]["blockers"]
    assert groups["matrix-cross-symbol"]["readiness_class"] == "blocked_high_risk_policy"
    assert "cross_symbol_artifact_review_required" in groups["matrix-cross-symbol"]["blockers"]
    assert groups["raw-referenced"]["readiness_class"] == "blocked_referenced_candidate"
    assert "candidate_path_still_referenced" in groups["raw-referenced"]["blockers"]


def test_restore_readiness_counts_only_fully_reference_free_candidate_paths(tmp_path):
    plan = _plan_fixture(tmp_path)
    plan["summary"]["candidate_group_count"] = 1
    plan["summary"]["candidate_file_count"] = 2
    plan["summary"]["potential_reclaim_bytes"] = 300
    plan["candidates"] = [
        {
            "group_id": "raw-partially-referenced",
            "candidate_type": "restore_source_duplicate_review",
            "reclaimable_bytes": 300,
            "candidate_paths": [
                "data/raw_backups/tushare/daily/full_a_000006.SZ_20260601_raw.csv",
                "data/raw_backups/tushare/daily/full_a_000006.SZ_20260602_raw.csv",
            ],
            "retained_paths": [
                "data/raw_backups/tushare/daily/full_a_000006.SZ_20260603_raw.csv"
            ],
        }
    ]
    policy = build_restore_source_policy(plan)
    reference_audit = {
        "schema_version": "myquant.data_cleanup_restore_reference_audit.v1",
        "generated_at": "2026-06-12T11:00:00+00:00",
        "delete_candidate_count": 0,
        "summary": {},
        "groups": [
            {
                "group_id": "raw-partially-referenced",
                "policy_class": "same_symbol_raw_backup_duplicate",
                "risk_level": "medium",
                "candidate_paths": policy["groups"][0]["candidate_paths"],
                "referenced_candidate_paths": [
                    policy["groups"][0]["candidate_paths"][0],
                ],
                "unreferenced_candidate_paths": [
                    policy["groups"][0]["candidate_paths"][1],
                ],
                "reference_count": 1,
            }
        ],
    }

    readiness = build_restore_readiness(policy, reference_audit)

    assert readiness["summary"]["reference_free_group_count"] == 0
    assert readiness["summary"]["referenced_group_count"] == 1
    assert readiness["summary"]["reference_free_candidate_path_count"] == 0
    assert readiness["summary"]["reference_free_potential_reclaim_bytes"] == 0


def test_restore_readiness_fails_closed_when_reference_group_is_missing(tmp_path):
    plan = _plan_fixture(tmp_path)
    plan["summary"]["candidate_group_count"] = 1
    plan["summary"]["candidate_file_count"] = 1
    plan["summary"]["potential_reclaim_bytes"] = 300
    plan["candidates"] = [
        {
            "group_id": "raw-missing-reference-group",
            "candidate_type": "restore_source_duplicate_review",
            "reclaimable_bytes": 300,
            "candidate_paths": [
                "data/raw_backups/tushare/daily/full_a_000007.SZ_20260601_raw.csv"
            ],
            "retained_paths": [
                "data/raw_backups/tushare/daily/full_a_000007.SZ_20260602_raw.csv"
            ],
        }
    ]
    policy = build_restore_source_policy(plan)
    reference_audit = {
        "schema_version": "myquant.data_cleanup_restore_reference_audit.v1",
        "generated_at": "2026-06-12T11:00:00+00:00",
        "delete_candidate_count": 0,
        "summary": {},
        "groups": [],
    }

    readiness = build_restore_readiness(policy, reference_audit)

    assert readiness["summary"]["reference_free_group_count"] == 0
    assert readiness["summary"]["referenced_group_count"] == 0
    assert readiness["summary"]["reference_unknown_group_count"] == 1
    assert readiness["summary"]["reference_free_candidate_path_count"] == 0
    assert readiness["summary"]["reference_free_potential_reclaim_bytes"] == 0
    assert readiness["summary"]["readiness_class_summary"] == {
        "blocked_missing_reference_audit_group": 1,
    }
    group = readiness["groups"][0]
    assert group["readiness_class"] == "blocked_missing_reference_audit_group"
    assert "reference_audit_group_missing" in group["blockers"]


def test_restore_readiness_writes_reports_and_cli_output(tmp_path, capsys):
    policy = build_restore_source_policy(_plan_fixture(tmp_path))
    reference_audit = _reference_audit_fixture(policy)
    policy_path = tmp_path / "data_cleanup_restore_policy.json"
    reference_path = tmp_path / "data_cleanup_restore_reference_audit.json"
    output_dir = tmp_path / "reports" / "project_cleanup" / "restore_readiness"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    reference_path.write_text(
        json.dumps(reference_audit, ensure_ascii=False),
        encoding="utf-8",
    )

    written = write_restore_readiness(
        policy_path,
        reference_path,
        root=tmp_path,
        output_dir=output_dir,
        max_markdown_groups=2,
    )

    payload = json.loads((output_dir / "data_cleanup_restore_readiness.json").read_text())
    markdown = (output_dir / "data_cleanup_restore_readiness.md").read_text()
    assert written["json"] == str(output_dir / "data_cleanup_restore_readiness.json")
    assert payload["summary"]["reference_free_group_count"] == 3
    assert markdown.startswith("# Data Cleanup Restore-Source Readiness")
    assert "Delete candidates: 0" in markdown

    exit_code = restore_readiness_main(
        [
            "--root",
            str(tmp_path),
            "--policy-json",
            str(policy_path),
            "--reference-audit-json",
            str(reference_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup restore-source readiness mode: dry-run" in stdout
    assert "reference-free groups: 3" in stdout
    assert "delete candidates: 0" in stdout
