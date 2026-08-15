from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from quant_investor.contracts import ArtifactValidationError, seal_artifact
from quant_investor.intelligence import (
    IntelligenceError,
    assess_readiness,
    compile_evidence,
    evaluate,
    forward,
    inspect,
    validate_readiness,
)
from quant_investor.intelligence._common import artifact_ref

NOW = "2026-08-14T00:00:00Z"


def source(source_id: str = "source-a") -> dict:
    return seal_artifact(
        "system.source_bundle",
        {"source_bundle_id": source_id, "sources": [], "state": "READY"},
        created_at=NOW,
    )


def assert_no_version_fields(value: object) -> None:
    if type(value) is list:
        for item in value:
            assert_no_version_fields(item)
    elif type(value) is dict:
        assert not {"version", "schema_version", "protocol_version"} & set(value)
        for item in value.values():
            assert_no_version_fields(item)


def test_forward_evaluate_compile_and_inspect_are_inactive_and_closed() -> None:
    input_artifact = source()
    request = forward(
        {
            "as_of": NOW,
            "input_refs": [artifact_ref(input_artifact)],
            "stages": ["industry", "fundamental"],
            "strategy_id": "research-strategy",
        },
        request_id="forward-request-001",
    )
    evaluation = evaluate(
        request,
        evaluated_at=NOW,
        stage_results={
            "industry": {
                "blocker_codes": [],
                "output_refs": [artifact_ref(input_artifact)],
                "status": "COMPLETE",
            },
            "fundamental": {
                "blocker_codes": [],
                "output_refs": [artifact_ref(input_artifact)],
                "status": "COMPLETE",
            },
        },
    )
    bundle = compile_evidence(evaluation, evidence=[input_artifact], compiled_at=NOW)
    inspection = inspect(bundle, inspected_at=NOW)

    assert request["artifact_id"] == "forward-request-001"
    assert evaluation["payload"]["status"] == "COMPLETE"
    assert bundle["payload"]["status"] == "READY"
    assert inspection["payload"]["status"] == "VALID"
    for artifact in (request, evaluation, bundle, inspection):
        assert artifact["payload"]["research_only"] is True
        assert artifact["payload"]["run_state"] == "INACTIVE"
        assert all(value is False for value in artifact["payload"]["authority"].values())
        assert_no_version_fields(artifact)


def test_business_identity_is_independent_from_nonidentity_payload() -> None:
    first_source = source("source-a")
    second_source = source("source-b")
    first = forward(
        {
            "as_of": NOW,
            "input_refs": [artifact_ref(first_source)],
            "stages": ["industry"],
            "strategy_id": "research-strategy",
        },
        request_id="owner-request-7",
    )
    changed_nonidentity_payload = forward(
        {
            "as_of": NOW,
            "input_refs": [artifact_ref(second_source)],
            "stages": ["industry"],
            "strategy_id": "research-strategy",
        },
        request_id="owner-request-7",
    )

    assert first["artifact_id"] == changed_nonidentity_payload["artifact_id"]
    assert first["semantic_sha256"] != changed_nonidentity_payload["semantic_sha256"]


def test_stage_closure_and_unknown_payload_fields_fail_closed() -> None:
    input_artifact = source()
    request = forward(
        {
            "as_of": NOW,
            "input_refs": [artifact_ref(input_artifact)],
            "stages": ["industry"],
            "strategy_id": "research-strategy",
        }
    )
    with pytest.raises(IntelligenceError, match="exactly close") as captured:
        evaluate(request, stage_results={}, evaluated_at=NOW)
    assert captured.value.code == "INTELLIGENCE_VALIDATION_FAILED"
    assert captured.value.exit_code == 2
    assert captured.value.public_fields == {}

    forged_payload = dict(request["payload"])
    forged_payload["unexpected"] = True
    with pytest.raises(ArtifactValidationError, match="fields are not exact"):
        seal_artifact("research_request", forged_payload, created_at=NOW)


def test_compile_evidence_preserves_blockers_and_never_fabricates_evidence() -> None:
    input_artifact = source()
    request = forward(
        {
            "as_of": NOW,
            "input_refs": [artifact_ref(input_artifact)],
            "stages": ["fundamental"],
            "strategy_id": "research-strategy",
        }
    )
    evaluation = evaluate(
        request,
        evaluated_at=NOW,
        stage_results={
            "fundamental": {
                "blocker_codes": ["FUNDAMENTAL_CUTOFF_STALE"],
                "output_refs": [],
                "status": "BLOCKED",
            }
        },
    )
    bundle = compile_evidence(evaluation, evidence=[], compiled_at=NOW)
    assert bundle["payload"]["status"] == "BLOCKED"
    assert bundle["payload"]["evidence_refs"] == []
    assert bundle["payload"]["blocker_codes"] == [
        "EVIDENCE_CLOSURE_EMPTY",
        "FUNDAMENTAL_CUTOFF_STALE",
    ]


def test_readiness_preserves_source_blockers_and_mainline_absence() -> None:
    readiness = assess_readiness(
        producer_identity="SYSTEM",
        assessed_at=NOW,
        factor_status=None,
        source_blockers=["FUNDAMENTAL_CUTOFF_STALE"],
        readiness_id="readiness-owner-1",
    )

    assert set(readiness["payload"]) == {
        "admission_route",
        "blockers",
        "factor_state",
        "factor_status_ref",
        "investment_state",
        "mainline_candidate_ref",
        "mainline_state",
        "producer_identity",
        "readiness_id",
    }
    assert readiness["payload"]["factor_state"] == "BLOCKED"
    assert readiness["payload"]["factor_status_ref"] is None
    assert readiness["payload"]["mainline_state"] == "UNINITIALIZED"
    assert readiness["payload"]["mainline_candidate_ref"] is None
    assert readiness["payload"]["investment_state"] == "BLOCKED"
    assert readiness["payload"]["blockers"] == [
        "FACTOR_STATUS_UNAVAILABLE",
        "FUNDAMENTAL_CUTOFF_STALE",
        "MAINLINE_CANDIDATE_ABSENT",
    ]
    assert validate_readiness(readiness) == readiness


def test_intelligence_import_and_readiness_validation_do_not_load_mainline() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys

class RejectMainline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "quant_investor.mainline" or fullname.startswith(
            "quant_investor.mainline."
        ):
            raise AssertionError(f"Mainline import attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectMainline())
from quant_investor.intelligence import assess_readiness, validate_readiness

readiness = assess_readiness(
    producer_identity="SYSTEM",
    assessed_at="2026-08-14T00:00:00Z",
    factor_status=None,
    source_blockers=["FUNDAMENTAL_CUTOFF_STALE"],
)
assert validate_readiness(readiness) == readiness
assert readiness["payload"]["mainline_candidate_ref"] is None
assert readiness["payload"]["mainline_state"] == "UNINITIALIZED"
assert not any(
    name == "quant_investor.mainline" or name.startswith("quant_investor.mainline.")
    for name in sys.modules
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
