from __future__ import annotations

import json
from pathlib import Path

from scripts.compact_matrix_coverage import (
    CONFIRM_TOKEN,
    build_matrix_coverage_compaction_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fixture(
    root: Path,
    *,
    matching_coverage: bool = True,
) -> tuple[dict, dict, Path, str, str]:
    matrix = (
        root
        / "data"
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_matrix_coverage.json"
    )
    matrix_rel = matrix.relative_to(root).as_posix()
    factor_report = matrix.with_name(
        "full_a_000001.SZ_20260312_factor_readiness_report.json"
    )
    factor_report_rel = factor_report.relative_to(root).as_posix()
    cleaning_report = (
        root
        / "data"
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_20260312_cleaning_report.json"
    )
    cleaning_report_rel = cleaning_report.relative_to(root).as_posix()
    summaries = [
        {
            "table_name": "daily",
            "symbol_count": 1,
            "date_count": 3,
            "field_coverage": {"close": 1.0},
        }
    ]
    _write_json(
        matrix,
        {
            "schema_version": "2026-04-27.tushare-factor-readiness.v1",
            "generated_at": "2026-03-12T00:00:00Z",
            "summaries": summaries,
        },
    )
    _write_json(
        factor_report,
        {
            "schema_version": "2026-04-27.tushare-factor-readiness.v1",
            "generated_at": "2026-03-12T00:00:00Z",
            "coverage_summaries": summaries
            if matching_coverage
            else [{**summaries[0], "date_count": 4}],
        },
    )
    _write_json(
        cleaning_report,
        {
            "metadata": {
                "matrix_coverage_path": matrix_rel,
                "factor_readiness_report_path": factor_report_rel,
            }
        },
    )
    policy = {
        "groups": [
            {
                "group_id": "dup-matrix",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [matrix_rel],
                "retained_paths": [
                    "data/factor_readiness/tushare/daily/"
                    "full_a_000002.SZ_20260312_matrix_coverage.json"
                ],
                "path_roles": ["factor_matrix_coverage"],
            }
        ]
    }
    reference_audit = {
        "groups": [
            {
                "group_id": "dup-matrix",
                "policy_class": "cross_symbol_generated_artifact_duplicate",
                "risk_level": "high",
                "candidate_paths": [matrix_rel],
                "referenced_candidate_paths": [matrix_rel],
                "unreferenced_candidate_paths": [],
            }
        ]
    }
    return policy, reference_audit, matrix, cleaning_report_rel, factor_report_rel


def test_matrix_coverage_compaction_dry_run_rewrites_nothing(tmp_path):
    policy, reference_audit, matrix, cleaning_report_rel, _factor_rel = _fixture(
        tmp_path
    )

    report = build_matrix_coverage_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert matrix.exists()
    cleaning_report = json.loads((tmp_path / cleaning_report_rel).read_text())
    assert cleaning_report["metadata"]["matrix_coverage_path"] == (
        matrix.relative_to(tmp_path).as_posix()
    )


def test_matrix_coverage_compaction_apply_rewrites_owner_then_deletes(tmp_path):
    policy, reference_audit, matrix, cleaning_report_rel, factor_rel = _fixture(
        tmp_path
    )
    matrix_rel = matrix.relative_to(tmp_path).as_posix()

    report = build_matrix_coverage_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
        apply=True,
        confirm_token=CONFIRM_TOKEN,
    )

    assert report["execution_performed"] is True
    assert report["summary"]["compacted_count"] == 1
    assert report["summary"]["references_rewritten_count"] == 1
    assert not matrix.exists()
    cleaning_report = json.loads((tmp_path / cleaning_report_rel).read_text())
    metadata = cleaning_report["metadata"]
    assert metadata["matrix_coverage_path"] is None
    assert metadata["matrix_coverage_path_suppressed"] is True
    assert metadata["matrix_coverage_planned_path"] == matrix_rel
    assert metadata["matrix_coverage_factor_readiness_report_path"] == factor_rel
    assert metadata["matrix_coverage_embedded_in_factor_readiness_report"] is True


def test_matrix_coverage_compaction_blocks_when_coverage_differs(tmp_path):
    policy, reference_audit, matrix, _cleaning_rel, _factor_rel = _fixture(
        tmp_path,
        matching_coverage=False,
    )

    report = build_matrix_coverage_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 0
    assert report["summary"]["blocked_count"] == 1
    assert "differs" in report["items"][0]["errors"][0]
    assert matrix.exists()


def test_matrix_coverage_compaction_ignores_derived_csv_inventory_refs(tmp_path):
    policy, reference_audit, matrix, _cleaning_rel, _factor_rel = _fixture(tmp_path)
    matrix_rel = matrix.relative_to(tmp_path).as_posix()
    inventory = tmp_path / "reports" / "storage" / "csv_inventory_20260610.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(f"derived reference {matrix_rel}\n", encoding="utf-8")

    report = build_matrix_coverage_compaction_report(
        policy,
        reference_audit,
        repo_root=tmp_path,
    )

    assert report["summary"]["would_compact_count"] == 1
    assert report["summary"]["blocked_count"] == 0
    assert report["summary"]["ignored_external_reference_count"] == 1
    assert report["items"][0]["ignored_external_reference_count"] == 1
    assert matrix.exists()
