from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    seal_semantic,
)
from quant_investor.v16.evidence_v2.readiness_v3 import (
    ARTIFACT_FILENAME,
    FOUNDATION_BLOCKERS,
    SCHEMA_VERSION,
    ReadinessEvidenceBundleV3,
    V16ReadinessV3Error,
    build_v16_run_readiness_v3,
    validate_v16_run_readiness_v3,
)


def _ref(name: str, schema: str) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"/private/evidence/{name}",
        byte_sha256=hashlib.sha256(f"{name}:bytes".encode()).hexdigest(),
        semantic_sha256=hashlib.sha256(f"{name}:semantic".encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


@pytest.fixture
def readiness_inputs(monkeypatch):
    schedule_ref = _ref(
        "schedule-v4.json",
        "v16.evidence-schedule-declaration.v4",
    )
    factor = {
        "schema_version": "factor-production-set-carrier-readback.v4.foundation",
        "blockers": [
            "factor_v4:production_factor_count_below_5",
            "shared-blocker",
        ],
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
    }
    schedule = {
        "schema_version": "v16.schedule-lineage-readback.v4",
        "protocol_attempt_id": "attempt-v16-readiness-v3-001",
        "epochs": ["A", "B"],
        "schedule_declaration_refs": [schedule_ref.to_dict()],
        "blockers": ["shared-blocker"],
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
        "readiness_status": "no_new_risk",
    }
    calibration = {
        "schema_version": "v16.prospective-calibration-source-status.v3",
        "protocol_attempt_id": schedule["protocol_attempt_id"],
        "schedule_ref": schedule_ref.to_dict(),
        "source_recomputation_complete": False,
        "blockers": ["shared-blocker", "calibration_source_recomputation_incomplete"],
        "blocker_sources": [
            {"blocker": "shared-blocker", "source": "branch_status:quant-sample"},
            {"blocker": "shared-blocker", "source": "target_status:quant-sample"},
            {
                "blocker": "calibration_source_recomputation_incomplete",
                "source": "calibration_source_status",
            },
        ],
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
    }
    evidence = ReadinessEvidenceBundleV3(
        factor_production_set=SimpleNamespace(
            carrier=SimpleNamespace(
                reference=_ref(
                    "factor-carrier.json",
                    "factor-production-set-carrier.v4.foundation",
                )
            )
        ),
        schedule_lineage=SimpleNamespace(),
        calibration_status=SimpleNamespace(
            reference=_ref(
                "calibration-source-status.json",
                "v16.prospective-calibration-source-status.v3",
            )
        ),
        calibration_evidence=SimpleNamespace(),
    )
    monkeypatch.setattr(
        ReadinessEvidenceBundleV3,
        "read",
        lambda _self: (factor, schedule, calibration),
    )
    return evidence, factor, schedule, calibration


def _payload(evidence: ReadinessEvidenceBundleV3) -> dict[str, object]:
    return build_v16_run_readiness_v3(
        run_id="v16-source-readiness-20260720",
        generated_at="2026-07-20T12:00:00Z",
        analysis_trade_date="2026-07-17",
        evidence=evidence,
    )


def test_readiness_v3_is_distinct_structurally_nonauthorizing_and_four_branch(
    readiness_inputs,
) -> None:
    evidence, _factor, _schedule, _calibration = readiness_inputs
    payload = _payload(evidence)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert ARTIFACT_FILENAME == "v16_run_readiness_v3.json"
    assert payload["formal_branches"] == [
        {"branch": branch, "weight": "0.25"}
        for branch in ("quant", "fundamental", "macro", "llm")
    ]
    assert payload["retrieval_role"] == "evidence_only_no_scoring_or_weight"
    assert payload["risk_advisor_role"] == "advisory_only"
    assert payload["activation_candidate"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["readiness_status"] == "no_new_risk"
    assert payload["broker_side_effects"] is False
    assert set(FOUNDATION_BLOCKERS).issubset(payload["blockers"])
    assert validate_v16_run_readiness_v3(payload, evidence=evidence) == payload


def test_readiness_v3_preserves_colliding_blocker_sources(readiness_inputs) -> None:
    evidence, _factor, _schedule, _calibration = readiness_inputs
    payload = _payload(evidence)

    assert payload["blockers"].count("shared-blocker") == 1
    shared_sources = [
        item["source"]
        for item in payload["blocker_sources"]
        if item["blocker"] == "shared-blocker"
    ]
    assert shared_sources == sorted(
        [
            "calibration_source:branch_status:quant-sample",
            "calibration_source:target_status:quant-sample",
            "factor_production_set",
            "schedule_lineage",
        ]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_candidate", True),
        ("new_risk_authorized", True),
        ("readiness_status", "ready"),
        ("broker_side_effects", True),
    ],
)
def test_readiness_v3_rejects_resealed_authorization_claims(
    readiness_inputs,
    field: str,
    value: object,
) -> None:
    evidence, _factor, _schedule, _calibration = readiness_inputs
    mutated = deepcopy(_payload(evidence))
    mutated.pop("semantic_sha256")
    mutated[field] = value

    with pytest.raises(V16ReadinessV3Error, match="must remain no_new_risk"):
        validate_v16_run_readiness_v3(seal_semantic(mutated), evidence=evidence)


@pytest.mark.parametrize("old_schema", ["v16_run_readiness.v1", "v16_run_readiness.v2"])
def test_readiness_v3_rejects_v1_v2_and_unknown_fields(
    readiness_inputs,
    old_schema: str,
) -> None:
    evidence, _factor, _schedule, _calibration = readiness_inputs
    mutated = deepcopy(_payload(evidence))
    mutated.pop("semantic_sha256")
    mutated["schema_version"] = old_schema
    with pytest.raises(V16ReadinessV3Error, match="identity mismatch"):
        validate_v16_run_readiness_v3(seal_semantic(mutated), evidence=evidence)

    injected = deepcopy(_payload(evidence))
    injected.pop("semantic_sha256")
    injected["production_apply_enabled"] = True
    with pytest.raises(V16ReadinessV3Error, match="fields mismatch"):
        validate_v16_run_readiness_v3(seal_semantic(injected), evidence=evidence)


def test_readiness_v3_validation_performs_no_path_io(
    readiness_inputs,
    monkeypatch,
) -> None:
    evidence, _factor, _schedule, _calibration = readiness_inputs
    payload = _payload(evidence)

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("readiness v3 attempted path I/O")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert validate_v16_run_readiness_v3(payload, evidence=evidence) == payload


def test_readiness_v3_requires_typed_evidence_bundle() -> None:
    with pytest.raises(V16ReadinessV3Error, match="ReadinessEvidenceBundleV3"):
        build_v16_run_readiness_v3(
            run_id="v16-source-readiness-20260720",
            generated_at="2026-07-20T12:00:00Z",
            analysis_trade_date="2026-07-17",
            evidence={"ready": True},  # type: ignore[arg-type]
        )
