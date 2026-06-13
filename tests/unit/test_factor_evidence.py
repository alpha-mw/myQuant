import json
import importlib

import pytest

from quant_investor.factors.evidence import (
    EVIDENCE_ISSUE_AUDIT_BLOCKER,
    EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS,
    EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES,
    EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY,
    EVIDENCE_STATUS_FAIL,
    EVIDENCE_STATUS_INSUFFICIENT_DATA,
    FactorAuditEvidenceSnapshot,
    FactorEvidenceCollectionConfig,
    FactorEvidenceDateInput,
    FactorShadowEvidenceDateResult,
    MultiDateFactorEvidenceReport,
    build_factor_evidence_dashboard_payload,
    build_multi_date_factor_evidence_report,
    collect_shadow_evidence_for_date,
    load_factor_matrices_from_paths,
    load_json_file_safe,
    load_jsonl_file_safe,
    load_production_library_safe,
    render_multi_date_evidence_markdown,
)
from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    FactorLibraryEntry,
    ProductionFactorLibrary,
)


def _library() -> ProductionFactorLibrary:
    return ProductionFactorLibrary(
        library_id="prod-lib",
        generated_at="2026-04-01T00:00:00Z",
        entries=[
            FactorLibraryEntry(
                factor_id="factor-a",
                factor_version="v1",
                status=FACTOR_STATUS_PRODUCTION,
                admission_decision_id="decision-a",
                validation_report_id="validation-a",
                production_since="2026-04-01",
            )
        ],
    )


def _matrix(as_of: str = "2026-04-01") -> FactorMatrix:
    return FactorMatrix(
        matrix_id=f"matrix-{as_of}",
        factor_id="factor-a",
        factor_version="v1",
        expression="close / open",
        symbols=["AAA", "BBB", "CCC"],
        dates=[as_of],
        values=[[3.0], [2.0], [1.0]],
        coverage_ratio=1.0,
        missing_ratio=0.0,
    )


def _candidates() -> list[dict[str, object]]:
    return [
        {"symbol": "AAA", "name": "A", "official_score": 0.9, "official_rank": 1},
        {"symbol": "BBB", "name": "B", "official_score": 0.8, "official_rank": 2},
        {"symbol": "CCC", "name": "C", "official_score": 0.7, "official_rank": 3},
    ]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _date_input(tmp_path, as_of="2026-04-01", candidates=None, library=True, matrices=True):
    library_path = _write_json(tmp_path / f"library-{as_of}.json", _library().to_dict()) if library else None
    matrix_path = tmp_path / f"matrices-{as_of}.jsonl"
    if matrices:
        matrix_path.write_text(json.dumps(_matrix(as_of).to_dict()) + "\n", encoding="utf-8")
    return FactorEvidenceDateInput(
        as_of=as_of,
        candidates=_candidates() if candidates is None else candidates,
        production_library_path=str(library_path) if library_path else None,
        factor_matrix_paths=[str(matrix_path)] if matrices else [],
    )


def test_evidence_contracts_are_split_and_reexported() -> None:
    evidence_module = importlib.import_module("quant_investor.factors.evidence")
    types_module = importlib.import_module("quant_investor.factors.evidence_types")
    exported_names = [
        "EVIDENCE_STATUS_OK",
        "EVIDENCE_STATUS_WARN",
        "EVIDENCE_STATUS_FAIL",
        "EVIDENCE_STATUS_INSUFFICIENT_DATA",
        "FactorEvidenceCollectionConfig",
        "FactorEvidenceDateInput",
        "FactorAuditEvidenceSnapshot",
        "FactorShadowEvidenceDateResult",
        "MultiDateFactorEvidenceReport",
        "make_evidence_collection_config_id",
        "make_evidence_date_result_id",
        "make_multi_date_evidence_report_id",
    ]
    for name in exported_names:
        assert getattr(evidence_module, name) is getattr(types_module, name)
        assert name in evidence_module.__all__


def test_evidence_dataclass_round_trips() -> None:
    config = FactorEvidenceCollectionConfig(as_of_dates=["2026-04-02", "2026-04-01"])
    assert config.as_of_dates == ["2026-04-01", "2026-04-02"]
    assert FactorEvidenceCollectionConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    with pytest.raises(ValueError):
        FactorEvidenceCollectionConfig(as_of_dates=["2026-04-01"], top_n=0)
    with pytest.raises(ValueError):
        FactorEvidenceCollectionConfig(as_of_dates=["2026-04-01"], min_observation_days=0)
    with pytest.raises(ValueError):
        FactorEvidenceCollectionConfig(as_of_dates=["2026-04-01"], min_top_n_overlap_ratio=1.5)

    date_input = FactorEvidenceDateInput(as_of="2026-04-01", candidates=_candidates())
    assert FactorEvidenceDateInput.from_dict(date_input.to_dict()).to_dict() == date_input.to_dict()
    snapshot = FactorAuditEvidenceSnapshot(as_of="2026-04-01", library_exists=True, production_factor_count=1)
    assert FactorAuditEvidenceSnapshot.from_dict(snapshot.to_dict()).to_dict() == snapshot.to_dict()
    result = FactorShadowEvidenceDateResult(
        result_id="result-1",
        as_of="2026-04-01",
        audit_snapshot=snapshot,
    )
    assert FactorShadowEvidenceDateResult.from_dict(result.to_dict()).to_dict() == result.to_dict()
    report = MultiDateFactorEvidenceReport(
        report_id="report-1",
        generated_at="2026-04-03T00:00:00Z",
        config=config,
        date_results=[result],
        observation_days=1,
    )
    assert MultiDateFactorEvidenceReport.from_dict(report.to_dict()).to_dict() == report.to_dict()


def test_artifact_loaders(tmp_path) -> None:
    assert load_json_file_safe(tmp_path / "missing.json")[0] is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert load_json_file_safe(bad_json)[0] is None
    valid_json = _write_json(tmp_path / "valid.json", {"a": 1})
    assert load_json_file_safe(valid_json) == ({"a": 1}, [])

    assert load_jsonl_file_safe(tmp_path / "missing.jsonl")[0] == []
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"a": 1}\nnot-json\n\n', encoding="utf-8")
    rows, warnings = load_jsonl_file_safe(bad_jsonl)
    assert rows == [{"a": 1}]
    assert warnings

    matrices_path = tmp_path / "matrices.jsonl"
    matrices_path.write_text(json.dumps(_matrix().to_dict()) + "\n", encoding="utf-8")
    matrices, warnings = load_factor_matrices_from_paths([matrices_path])
    assert warnings == []
    assert matrices[0].matrix_id == "matrix-2026-04-01"

    library_path = _write_json(tmp_path / "library.json", _library().to_dict())
    library, warnings = load_production_library_safe(library_path)
    assert warnings == []
    assert library is not None
    assert library.library_id == "prod-lib"


def test_single_date_evidence_collects_shadow_metrics_and_preserves_candidates(tmp_path) -> None:
    date_input = _date_input(tmp_path)
    original = [dict(row) for row in date_input.candidates]
    config = FactorEvidenceCollectionConfig(
        as_of_dates=["2026-04-01"],
        top_n=2,
        min_observation_days=1,
    )
    result = collect_shadow_evidence_for_date(
        date_input=date_input,
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert result.candidate_count == 3
    assert result.shadow_report_id
    assert result.average_factor_coverage_ratio == 1.0
    assert result.top_n_overlap_ratio == 1.0
    assert date_input.candidates == original


def test_single_date_warns_and_fails_for_missing_artifacts_and_audit_blocker(tmp_path) -> None:
    config = FactorEvidenceCollectionConfig(
        as_of_dates=["2026-04-01"],
        min_observation_days=1,
        require_library_audit_no_blocker=True,
    )
    missing_library = collect_shadow_evidence_for_date(
        date_input=_date_input(tmp_path, library=False),
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY in missing_library.warning_codes

    missing_matrices = collect_shadow_evidence_for_date(
        date_input=_date_input(tmp_path, matrices=False),
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES in missing_matrices.warning_codes

    no_candidates = collect_shadow_evidence_for_date(
        date_input=_date_input(tmp_path, candidates=[]),
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert no_candidates.status == EVIDENCE_STATUS_INSUFFICIENT_DATA

    blocker_input = _date_input(tmp_path)
    blocker_path = _write_json(tmp_path / "audit.json", {"verdict": "fail", "blocked_factor_ids": ["factor-a"]})
    blocker_input.library_audit_path = str(blocker_path)
    blocker = collect_shadow_evidence_for_date(
        date_input=blocker_input,
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert blocker.status == EVIDENCE_STATUS_FAIL
    assert EVIDENCE_ISSUE_AUDIT_BLOCKER in blocker.warning_codes


def test_multi_date_report_markdown_and_dashboard(tmp_path) -> None:
    config = FactorEvidenceCollectionConfig(
        as_of_dates=["2026-04-01", "2026-04-02"],
        top_n=2,
        min_observation_days=3,
    )
    report = build_multi_date_factor_evidence_report(
        date_inputs=[
            _date_input(tmp_path, "2026-04-01"),
            _date_input(tmp_path, "2026-04-02"),
        ],
        config=config,
        generated_at="2026-04-03T00:00:00Z",
    )
    assert report.observation_days == 2
    assert report.average_top_n_overlap_ratio == 1.0
    assert report.min_top_n_overlap_ratio == 1.0
    assert report.average_factor_coverage_ratio == 1.0
    assert report.average_abs_rank_delta == 0.0
    assert report.max_abs_rank_delta == 0
    assert EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS in report.warning_codes
    markdown = render_multi_date_evidence_markdown(report)
    assert "does not alter official scores" in markdown
    dashboard = build_factor_evidence_dashboard_payload(report)
    json.dumps(dashboard, sort_keys=True)
    assert dashboard["observation_days"] == 2
