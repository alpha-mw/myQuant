from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

import quant_investor.v16.evidence_v2.candidate_report_source_v2 as report_module
from quant_investor.v16.evidence_v2.candidate_report_source_v2 import (
    CandidateReportSourceEvidenceBundleV2,
    _f64,
    build_candidate_report_source_v2,
    validate_candidate_report_source_v2,
)
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.publication_plan_v2 import (
    PublicationPlanEvidenceBundleV2,
)
from tests.unit.test_v16_publication_plan_v2 import (
    _bound_at,
    _publication_inputs,
)


def _report_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CandidateReportSourceEvidenceBundleV2:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    bound_plan = _bound_at(str(plan["plan_absolute_path"]), plan)
    return CandidateReportSourceEvidenceBundleV2(
        publication_plan=PublicationPlanEvidenceBundleV2(
            plan=bound_plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )
    )


def test_candidate_report_projects_exact_recomputed_menu_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _report_evidence(tmp_path, monkeypatch)
    report = build_candidate_report_source_v2(evidence=evidence)

    assert [item["branch"] for item in report["formal_branches"]] == [
        "quant",
        "fundamental",
        "macro",
        "llm",
    ]
    assert all(item["weight"] == "0.25" for item in report["formal_branches"])
    assert report["menu"]
    assert all(
        [branch["branch"] for branch in item["branch_evidence"]]
        == ["quant", "fundamental", "macro", "llm"]
        for item in report["menu"]
    )
    forbidden_retrieval = {"score", "confidence", "probability", "weight"}
    assert all(
        not forbidden_retrieval.intersection(advisory)
        for item in report["menu"]
        for advisory in item["retrieval_advisory"]
    )
    assert all(item["posterior"]["win_rate"].startswith("f64:") for item in report["menu"])
    assert all(item["allocation"]["target_weight"].startswith("f64:") for item in report["menu"])
    assert report["cash_ratio"].startswith("f64:")
    assert report["projection_validation_complete"] is True
    assert report["authority_source_complete"] is False
    for field in (
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "production_pointer_switch_authorized",
        "dashboard_activation_authorized",
        "broker_side_effects",
    ):
        assert report[field] is False
    readiness = evidence.publication_plan.read()[1]
    assert set(readiness["blockers"]).issubset(report["blockers"])
    assert validate_candidate_report_source_v2(report, evidence=evidence) == report


def test_candidate_report_rejects_noncanonical_f64_and_native_float() -> None:
    for value in (
        "f64:-0x0.0p+0",
        "f64:nan",
        "f64:inf",
        "0.25",
    ):
        with pytest.raises(EvidenceV2Error):
            _f64(value, label="test")
    with pytest.raises(EvidenceV2Error, match="native JSON float"):
        seal_semantic(
            {
                "schema_version": "test.native-float.v1",
                "cash_ratio": 0.4,
            }
        )


def test_candidate_report_rejects_resealed_authority_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _report_evidence(tmp_path, monkeypatch)
    report = build_candidate_report_source_v2(evidence=evidence)
    tampered = deepcopy(report)
    tampered.pop("semantic_sha256")
    tampered["new_risk_authorized"] = True

    with pytest.raises(EvidenceV2Error, match="drifts from evidence"):
        validate_candidate_report_source_v2(
            seal_semantic(tampered),
            evidence=evidence,
        )


def test_candidate_report_module_does_not_import_legacy_report_or_writers() -> None:
    source = Path(report_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "quant_investor.reporting.v16_candidate_decision" not in imported
    assert "quant_investor.codex_review.workflow" not in imported
    assert "atomic_write_bytes" not in imported
    assert "write_exact_once" not in imported
    assert "CapitalMap" not in imported
    assert "HumanAuthorization" not in imported
