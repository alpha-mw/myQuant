from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import stat

import pytest

import quant_investor.v16.evidence_v2.publication_bundle_io_v2 as io_module
from quant_investor.v16.evidence_v2.candidate_report_source_v2 import (
    CandidateReportSourceEvidenceBundleV2,
)
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.dashboard_source_v2 import (
    DashboardReportEvidenceBundleV2,
    DashboardSnapshotEvidenceBundleV2,
)
from quant_investor.v16.evidence_v2.publication_aggregate_v2 import (
    AGGREGATE_ARTIFACT_ORDER,
    PublicationAggregateEvidenceBundleV2,
    validate_publication_aggregate_v2,
)
from quant_investor.v16.evidence_v2.publication_bundle_io_v2 import (
    PRIVATE_FILE_MODE,
    PUBLICATION_WRITE_ORDER,
    PublishedPublicationBundleV2,
    publish_publication_bundle_v2,
)
from quant_investor.v16.evidence_v2.publication_plan_v2 import (
    PublicationPlanEvidenceBundleV2,
)
from tests.unit.test_v16_publication_plan_v2 import _publication_inputs


def _aggregate_evidence(
    published: PublishedPublicationBundleV2,
    readiness,
    readiness_evidence,
) -> PublicationAggregateEvidenceBundleV2:
    plan_evidence = PublicationPlanEvidenceBundleV2(
        plan=published.publication_plan,
        readiness_v4=readiness,
        readiness_evidence=readiness_evidence,
    )
    report_evidence = CandidateReportSourceEvidenceBundleV2(
        publication_plan=plan_evidence
    )
    dashboard_report = DashboardReportEvidenceBundleV2(
        publication_plan=plan_evidence,
        candidate_report=published.candidate_report,
        report_evidence=report_evidence,
    )
    dashboard_evidence = DashboardSnapshotEvidenceBundleV2(
        report=dashboard_report,
        snapshot=published.dashboard_snapshot,
    )
    return PublicationAggregateEvidenceBundleV2(
        publication_plan=plan_evidence,
        candidate_report=published.candidate_report,
        dashboard_snapshot=published.dashboard_snapshot,
        dashboard_source_status=published.dashboard_source_status,
        dashboard_evidence=dashboard_evidence,
    )


def test_publication_bundle_writes_exclusively_and_aggregate_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    published = publish_publication_bundle_v2(
        plan_payload=plan,
        readiness_v4=readiness,
        readiness_evidence=readiness_evidence,
    )

    assert tuple(published.references()) == PUBLICATION_WRITE_ORDER
    for artifact in (
        published.publication_plan,
        published.candidate_report,
        published.dashboard_snapshot,
        published.dashboard_source_status,
        published.publication_aggregate,
    ):
        path = Path(artifact.reference.absolute_path)
        assert path.read_bytes() == artifact.payload
        assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_FILE_MODE
        assert path.stat().st_nlink == 1
    aggregate = published.publication_aggregate.read()
    assert aggregate["publication_artifact_set_complete"] is True
    assert aggregate["artifact_order"] == list(AGGREGATE_ARTIFACT_ORDER)
    assert [item["byte_size"] for item in aggregate["artifacts"]] == [
        len(published.publication_plan.payload),
        len(readiness.payload),
        len(published.candidate_report.payload),
        len(published.dashboard_snapshot.payload),
        len(published.dashboard_source_status.payload),
    ]
    for field in (
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "production_pointer_switch_authorized",
        "codex_activation_authorized",
        "dashboard_activation_authorized",
        "sealed_live_human_receipt_verified",
        "broker_side_effects",
    ):
        assert aggregate[field] is False
    evidence = _aggregate_evidence(published, readiness, readiness_evidence)
    assert validate_publication_aggregate_v2(
        aggregate,
        evidence=evidence,
    ) == aggregate


@pytest.mark.parametrize("target_kind", ["file", "symlink"])
def test_publication_bundle_preflight_rejects_existing_or_symlink_target_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    root = Path(plan["private_root"])
    target = Path(
        plan["planned_artifacts"]["dashboard_snapshot"]["absolute_path"]
    )
    if target_kind == "file":
        target.write_bytes(b"occupied")
    else:
        target.symlink_to(root / "missing-target")

    with pytest.raises(EvidenceV2Error, match="already exists"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )
    assert not Path(plan["plan_absolute_path"]).exists()
    assert not Path(
        plan["planned_artifacts"]["publication_aggregate"]["absolute_path"]
    ).exists()


def test_publication_bundle_partial_failure_is_terminal_and_not_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    original = io_module._write_exclusive
    calls = 0

    def fail_on_snapshot(root, artifact):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise EvidenceV2Error("injected publication failure")
        return original(root, artifact)

    monkeypatch.setattr(io_module, "_write_exclusive", fail_on_snapshot)
    with pytest.raises(EvidenceV2Error, match="injected publication failure"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )
    assert Path(plan["plan_absolute_path"]).is_file()
    assert Path(
        plan["planned_artifacts"]["candidate_report"]["absolute_path"]
    ).is_file()
    assert not Path(
        plan["planned_artifacts"]["dashboard_snapshot"]["absolute_path"]
    ).exists()
    assert not Path(
        plan["planned_artifacts"]["publication_aggregate"]["absolute_path"]
    ).exists()

    monkeypatch.setattr(io_module, "_write_exclusive", original)
    with pytest.raises(EvidenceV2Error, match="already exists"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )


def test_publication_aggregate_rejects_resealed_size_or_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    published = publish_publication_bundle_v2(
        plan_payload=plan,
        readiness_v4=readiness,
        readiness_evidence=readiness_evidence,
    )
    evidence = _aggregate_evidence(published, readiness, readiness_evidence)
    aggregate = published.publication_aggregate.read()
    for mutate in ("size", "authority"):
        tampered = deepcopy(aggregate)
        tampered.pop("semantic_sha256")
        if mutate == "size":
            tampered["artifacts"][0]["byte_size"] += 1
        else:
            tampered["new_risk_authorized"] = True
        with pytest.raises(EvidenceV2Error, match="drifts from evidence"):
            validate_publication_aggregate_v2(
                seal_semantic(tampered),
                evidence=evidence,
            )


def test_publication_bundle_rejects_nonprivate_root_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    os.chmod(plan["private_root"], 0o755)
    with pytest.raises(EvidenceV2Error, match="owner-owned mode 0700"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )


def test_publication_bundle_rejects_root_acl_before_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        io_module,
        "platform_acl_absent",
        lambda _descriptor, _label: False,
    )

    with pytest.raises(EvidenceV2Error, match="private root has an extended ACL"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )
    assert not Path(plan["plan_absolute_path"]).exists()


def test_publication_bundle_rejects_created_file_acl_before_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)

    def acl_absent(_descriptor, label):
        return label == "publication private root"

    monkeypatch.setattr(io_module, "platform_acl_absent", acl_absent)
    with pytest.raises(EvidenceV2Error, match="publication output .* extended ACL"):
        publish_publication_bundle_v2(
            plan_payload=plan,
            readiness_v4=readiness,
            readiness_evidence=readiness_evidence,
        )
    assert Path(plan["plan_absolute_path"]).is_file()
    assert not Path(
        plan["planned_artifacts"]["publication_aggregate"]["absolute_path"]
    ).exists()
