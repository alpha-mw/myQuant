from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

import quant_investor.codex_review.storage as review_storage
import quant_investor.codex_review.workflow as review_workflow
import quant_investor.v16.evidence_v2.readiness_v4 as readiness_module
from quant_investor.v16.evidence_v2.contracts import EvidenceV2Error, seal_semantic
from quant_investor.v16.evidence_v2.execution_handoff_source_v2 import (
    build_execution_source_status_v2,
    build_handoff_source_status_v2,
)
from quant_investor.v16.evidence_v2.readiness_v3 import ReadinessEvidenceBundleV3
from quant_investor.v16.evidence_v2.readiness_v4 import (
    READINESS_V4_FOUNDATION_BLOCKERS,
    ReadinessEvidenceBundleV4,
    build_v16_run_readiness_v4,
    validate_v16_run_readiness_v4,
)
from tests.unit.test_v16_codex_ic_source_v2 import _bound
from tests.unit.test_v16_execution_handoff_source_v2 import _source_evidence


def _readiness_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> ReadinessEvidenceBundleV4:
    source = _source_evidence(monkeypatch)
    monkeypatch.setattr(
        readiness_module,
        "validate_v16_run_readiness_v3",
        lambda value, **_kwargs: value,
    )
    execution = _bound(
        "future/execution_status.json",
        build_execution_source_status_v2(evidence=source),
    )
    handoff = _bound(
        "future/handoff_status.json",
        build_handoff_source_status_v2(evidence=source),
    )
    readiness_v3_evidence = ReadinessEvidenceBundleV3(
        factor_production_set=None,  # type: ignore[arg-type]
        schedule_lineage=None,  # type: ignore[arg-type]
        calibration_status=None,  # type: ignore[arg-type]
        calibration_evidence=None,  # type: ignore[arg-type]
    )
    return ReadinessEvidenceBundleV4(
        readiness_v3=source.plan.readiness_v3,
        readiness_v3_evidence=readiness_v3_evidence,
        plan=source.plan,
        ic_status=source.ic_status,
        ic_evidence=source.ic_evidence,
        execution_status=execution,
        handoff_status=handoff,
        execution_handoff_evidence=source,
    )


def test_readiness_v4_is_nonauthorizing_and_preserves_all_blocker_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _readiness_evidence(monkeypatch)
    readiness = build_v16_run_readiness_v4(evidence=evidence)

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
        assert readiness[field] is False
    assert readiness["readiness_status"] == "no_new_risk"
    assert set(READINESS_V4_FOUNDATION_BLOCKERS).issubset(readiness["blockers"])
    duplicate = "codex_authority_v2_disconnected_from_authorizing_consumers"
    duplicate_sources = [
        item["source"]
        for item in readiness["blocker_sources"]
        if item["blocker"] == duplicate
    ]
    assert len(duplicate_sources) >= 4
    assert len(duplicate_sources) == len(set(duplicate_sources))
    assert validate_v16_run_readiness_v4(readiness, evidence=evidence) == readiness


def test_readiness_v4_rejects_resealed_pointer_or_activation_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _readiness_evidence(monkeypatch)
    readiness = build_v16_run_readiness_v4(evidence=evidence)
    for field in (
        "production_pointer_switch_authorized",
        "codex_activation_authorized",
        "dashboard_activation_authorized",
        "sealed_live_human_receipt_verified",
    ):
        tampered = deepcopy(readiness)
        tampered.pop("semantic_sha256")
        tampered[field] = True
        with pytest.raises(EvidenceV2Error, match="drifts from reopened evidence"):
            validate_v16_run_readiness_v4(
                seal_semantic(tampered),
                evidence=evidence,
            )


def test_readiness_v4_rejects_status_path_outside_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _readiness_evidence(monkeypatch)
    rebound = ReadinessEvidenceBundleV4(
        readiness_v3=evidence.readiness_v3,
        readiness_v3_evidence=evidence.readiness_v3_evidence,
        plan=evidence.plan,
        ic_status=evidence.ic_status,
        ic_evidence=evidence.ic_evidence,
        execution_status=_bound(
            "future/not-execution-status.json",
            evidence.execution_status.read(),
        ),
        handoff_status=evidence.handoff_status,
        execution_handoff_evidence=evidence.execution_handoff_evidence,
    )
    with pytest.raises(EvidenceV2Error, match="drifts from plan: execution_status"):
        build_v16_run_readiness_v4(evidence=rebound)


def test_readiness_v4_does_not_call_authorizing_mutators_or_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _readiness_evidence(monkeypatch)

    def fail(*_args, **_kwargs):
        raise AssertionError("authorizing mutator/writer called")

    monkeypatch.setattr(review_workflow, "resume_review", fail)
    monkeypatch.setattr(review_workflow, "receive_review_response", fail)
    monkeypatch.setattr(review_workflow, "validate_review_response", fail)
    monkeypatch.setattr(review_storage, "atomic_write_bytes", fail)
    monkeypatch.setattr(review_storage, "write_exact_once", fail)

    assert build_v16_run_readiness_v4(evidence=evidence)[
        "readiness_status"
    ] == "no_new_risk"


def test_readiness_v4_public_builder_accepts_only_typed_evidence() -> None:
    assert set(inspect.signature(build_v16_run_readiness_v4).parameters) == {
        "evidence"
    }
    source = Path(readiness_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "quant_investor.codex_review.workflow" not in imported
    assert "CapitalMap" not in imported
    assert "HumanAuthorization" not in imported
    assert "ExecutionGate" not in imported
    assert "atomic_write_bytes" not in imported
    assert "write_exact_once" not in imported
