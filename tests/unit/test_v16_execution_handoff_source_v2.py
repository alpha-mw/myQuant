from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

import quant_investor.codex_review.storage as review_storage
import quant_investor.codex_review.workflow as review_workflow
import quant_investor.v16.evidence_v2.execution_handoff_source_v2 as source_module
from quant_investor.v16.evidence_v2.codex_ic_source_v2 import (
    build_codex_ic_source_status_v2,
)
from quant_investor.v16.evidence_v2.contracts import (
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
)
from quant_investor.v16.evidence_v2.execution_handoff_source_v2 import (
    EXECUTION_REQUIREMENTS,
    HANDOFF_REQUIREMENTS,
    PERMANENT_EXECUTION_HANDOFF_BLOCKERS,
    ExecutionHandoffSourceEvidenceBundleV2,
    build_execution_source_status_v2,
    build_handoff_source_status_v2,
    validate_execution_source_status_v2,
    validate_handoff_source_status_v2,
)
from tests.unit.test_v16_codex_ic_source_v2 import _bound, _evidence


def _source_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> ExecutionHandoffSourceEvidenceBundleV2:
    ic_evidence = _evidence(monkeypatch)
    ic_status = build_codex_ic_source_status_v2(evidence=ic_evidence)
    return ExecutionHandoffSourceEvidenceBundleV2(
        plan=ic_evidence.plan,
        ic_status=_bound("future/ic_status.json", ic_status),
        ic_evidence=ic_evidence,
    )


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "capital_map",
        "human_authorization",
        "human_authorized",
        "authorization_receipt",
        "execution_plan",
        "market_state",
        "shares",
        "prices",
        "orders",
        "order_eligible",
        "handoff_ready",
    }
    if isinstance(value, dict):
        return bool(forbidden.intersection(value)) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def test_execution_and_handoff_statuses_remain_source_only_and_nonauthorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _source_evidence(monkeypatch)
    execution = build_execution_source_status_v2(evidence=evidence)
    handoff = build_handoff_source_status_v2(evidence=evidence)

    assert execution["unsupported_requirement_ids"] == list(EXECUTION_REQUIREMENTS)
    assert handoff["unsupported_requirement_ids"] == list(HANDOFF_REQUIREMENTS)
    for status in (execution, handoff):
        assert status["artifact_role"] == "source_status_only"
        assert status["source_recomputation_complete"] is False
        assert status["readiness_status"] == "no_new_risk"
        assert status["activation_candidate"] is False
        assert status["new_risk_authorized"] is False
        assert status["production_apply_enabled"] is False
        assert _contains_forbidden_key(status) is False
        assert set(PERMANENT_EXECUTION_HANDOFF_BLOCKERS).issubset(
            status["blockers"]
        )
    assert validate_execution_source_status_v2(
        execution,
        evidence=evidence,
    ) == execution
    assert validate_handoff_source_status_v2(handoff, evidence=evidence) == handoff


def test_execution_handoff_rejects_ic_status_outside_predeclared_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _source_evidence(monkeypatch)
    rebound = ExecutionHandoffSourceEvidenceBundleV2(
        plan=evidence.plan,
        ic_status=_bound("future/not-the-ic-status.json", evidence.ic_status.read()),
        ic_evidence=evidence.ic_evidence,
    )

    with pytest.raises(EvidenceV2Error, match="drifts from source plan"):
        build_execution_source_status_v2(evidence=rebound)


def test_execution_handoff_rejects_resealed_legacy_authority_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _source_evidence(monkeypatch)
    status = build_handoff_source_status_v2(evidence=evidence)
    tampered = deepcopy(status)
    tampered.pop("semantic_sha256")
    tampered["human_authorized"] = True

    with pytest.raises(EvidenceV2Error, match="fields mismatch"):
        validate_handoff_source_status_v2(seal_semantic(tampered), evidence=evidence)
    with pytest.raises(TypeError):
        build_handoff_source_status_v2(  # type: ignore[call-arg]
            evidence=evidence,
            authorization_receipt={"decision": "AUTHORIZED"},
        )


def test_execution_handoff_does_not_call_workflow_mutators_or_storage_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _source_evidence(monkeypatch)

    def fail(*_args, **_kwargs):
        raise AssertionError("authorizing mutator/writer called")

    monkeypatch.setattr(review_workflow, "resume_review", fail)
    monkeypatch.setattr(review_workflow, "receive_review_response", fail)
    monkeypatch.setattr(review_workflow, "validate_review_response", fail)
    monkeypatch.setattr(review_storage, "atomic_write_bytes", fail)
    monkeypatch.setattr(review_storage, "write_exact_once", fail)

    assert build_execution_source_status_v2(evidence=evidence)[
        "new_risk_authorized"
    ] is False
    assert build_handoff_source_status_v2(evidence=evidence)[
        "new_risk_authorized"
    ] is False


def test_execution_handoff_module_has_only_typed_evidence_public_inputs() -> None:
    for builder in (
        build_execution_source_status_v2,
        build_handoff_source_status_v2,
    ):
        assert set(inspect.signature(builder).parameters) == {"evidence"}

    source = Path(source_module.__file__).read_text(encoding="utf-8")
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
    assert b'"human_authorized"' not in canonical_json_bytes(
        seal_semantic({"schema_version": "test-only.v1"})
    )
