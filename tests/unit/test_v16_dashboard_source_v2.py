from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from quant_investor.v16.evidence_v2.candidate_report_source_v2 import (
    build_candidate_report_source_v2,
)
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.dashboard_source_v2 import (
    DASHBOARD_SOURCE_BLOCKERS,
    DashboardReportEvidenceBundleV2,
    DashboardSnapshotEvidenceBundleV2,
    build_dashboard_snapshot_v2,
    build_dashboard_source_status_v2,
    validate_dashboard_snapshot_v2,
    validate_dashboard_source_status_v2,
)
from tests.unit.test_v16_candidate_report_source_v2 import _report_evidence
from tests.unit.test_v16_publication_plan_v2 import _bound_at


def _dashboard_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DashboardReportEvidenceBundleV2, DashboardSnapshotEvidenceBundleV2]:
    report_evidence = _report_evidence(tmp_path, monkeypatch)
    report_payload = build_candidate_report_source_v2(evidence=report_evidence)
    report_path = report_evidence.publication_plan.read()[0]["planned_artifacts"][
        "candidate_report"
    ]["absolute_path"]
    report = _bound_at(report_path, report_payload)
    dashboard_report = DashboardReportEvidenceBundleV2(
        publication_plan=report_evidence.publication_plan,
        candidate_report=report,
        report_evidence=report_evidence,
    )
    snapshot_payload = build_dashboard_snapshot_v2(evidence=dashboard_report)
    snapshot_path = dashboard_report.publication_plan.read()[0]["planned_artifacts"][
        "dashboard_snapshot"
    ]["absolute_path"]
    snapshot = _bound_at(snapshot_path, snapshot_payload)
    return dashboard_report, DashboardSnapshotEvidenceBundleV2(
        report=dashboard_report,
        snapshot=snapshot,
    )


def test_dashboard_snapshot_and_status_are_exact_nonauthorizing_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_evidence, snapshot_evidence = _dashboard_evidence(tmp_path, monkeypatch)
    snapshot = snapshot_evidence.snapshot.read()
    status = build_dashboard_source_status_v2(evidence=snapshot_evidence)

    assert snapshot["schema_version"] == "dashboard_contract.v16.evidence-v2"
    assert snapshot["menu"] == report_evidence.candidate_report.read()["menu"]
    assert snapshot["readiness"]["status"] == "no_new_risk"
    assert snapshot["readiness"]["new_risk_authorized"] is False
    assert validate_dashboard_snapshot_v2(
        snapshot,
        evidence=report_evidence,
    ) == snapshot
    assert set(snapshot["blockers"]).issubset(status["blockers"])
    assert set(DASHBOARD_SOURCE_BLOCKERS).issubset(status["blockers"])
    assert status["publication_delivery_attested"] is False
    assert status["dashboard_app_integrated"] is False
    assert status["dashboard_activation_receipt_verified"] is False
    assert status["new_risk_authorized"] is False
    assert validate_dashboard_source_status_v2(
        status,
        evidence=snapshot_evidence,
    ) == status


def test_dashboard_snapshot_rejects_report_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_evidence, _ = _dashboard_evidence(tmp_path, monkeypatch)
    rebound = DashboardReportEvidenceBundleV2(
        publication_plan=report_evidence.publication_plan,
        candidate_report=_bound_at(
            str(tmp_path / "publication-run" / "wrong-report.json"),
            report_evidence.candidate_report.read(),
        ),
        report_evidence=report_evidence.report_evidence,
    )
    with pytest.raises(EvidenceV2Error, match="path drifts from plan"):
        build_dashboard_snapshot_v2(evidence=rebound)


def test_dashboard_status_rejects_resealed_activation_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence = _dashboard_evidence(tmp_path, monkeypatch)
    status = build_dashboard_source_status_v2(evidence=evidence)
    for field in (
        "dashboard_activation_receipt_verified",
        "dashboard_app_integrated",
        "dashboard_activation_authorized",
        "new_risk_authorized",
    ):
        tampered = deepcopy(status)
        tampered.pop("semantic_sha256")
        tampered[field] = True
        with pytest.raises(EvidenceV2Error, match="drifts from evidence"):
            validate_dashboard_source_status_v2(
                seal_semantic(tampered),
                evidence=evidence,
            )
