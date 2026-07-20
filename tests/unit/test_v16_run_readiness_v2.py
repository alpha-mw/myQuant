from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.v16.evidence_v2.factor_carrier import (
    FactorProductionSetEvidenceBundleV4,
    bind_factor_production_set_carrier_v4,
    build_factor_production_set_carrier_v4,
)
from quant_investor.factors.registry_store import registry_payload_semantic_sha256
from quant_investor.v16.evidence_v2.readiness import (
    ARTIFACT_FILENAME,
    ReadinessEvidenceBundleV2,
    SCHEMA_VERSION,
    ScheduleLineageEvidenceBundleV3,
    V16ReadinessV2Error,
    build_v16_run_readiness_v2,
    validate_v16_run_readiness_v2,
)
from quant_investor.v16.evidence_v2.contracts import (
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    seal_semantic,
)
from quant_investor.v16.evidence_v2.schedule import ScheduleAnchorBinding

REGISTRY_PATH = Path("quant_investor/factor_registry/mined_factors.json").resolve()


def _evidence() -> ReadinessEvidenceBundleV2:
    raw = REGISTRY_PATH.read_bytes()
    registry_payload = json.loads(raw)
    registry = BoundRawArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema="mined-factor-registry.v1",
            absolute_path=str(REGISTRY_PATH),
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=registry_payload_semantic_sha256(registry_payload),
            root_policy="v16.governed-data-root.v2",
        ),
        payload=raw,
    )
    carrier_payload = build_factor_production_set_carrier_v4(
        as_of="2026-07-17",
        legacy_registry_ref=registry.reference,
    )
    factor = FactorProductionSetEvidenceBundleV4(
        carrier=bind_factor_production_set_carrier_v4(
            carrier_payload,
            absolute_path="/private/v16/factor-production-set-carrier.json",
        ),
        legacy_registry=registry,
    )
    return ReadinessEvidenceBundleV2(factor_production_set=factor)


def _payload(evidence: ReadinessEvidenceBundleV2) -> dict[str, object]:
    return build_v16_run_readiness_v2(
        run_id="v16-foundation-20260719",
        generated_at="2026-07-19T09:45:00Z",
        analysis_trade_date="2026-07-17",
        evidence=evidence,
    )


def test_readiness_v2_is_distinct_and_structurally_nonauthorizing() -> None:
    evidence = _evidence()
    payload = _payload(evidence)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_filename"] == ARTIFACT_FILENAME
    assert ARTIFACT_FILENAME == "v16_run_readiness_v2.json"
    assert payload["formal_branches"] == [
        {"branch": branch, "weight": "0.25"}
        for branch in ("quant", "fundamental", "macro", "llm")
    ]
    assert payload["retrieval_role"] == "evidence_only_no_scoring_or_weight"
    assert payload["risk_advisor_role"] == "advisory_only"
    assert payload["activation_candidate"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["readiness_status"] == "no_new_risk"
    assert "schedule_v3_lineage_missing" in payload["blockers"]
    assert "factor_v4:production_factor_count_below_5" in payload["blockers"]
    assert validate_v16_run_readiness_v2(payload, evidence=evidence) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_candidate", True),
        ("new_risk_authorized", True),
        ("readiness_status", "ready"),
        ("broker_side_effects", True),
    ],
)
def test_readiness_v2_rejects_authorization_claims_when_resealed(
    field: str,
    value: object,
) -> None:
    evidence = _evidence()
    payload = _payload(evidence)
    mutated = {key: item for key, item in payload.items() if key != "semantic_sha256"}
    mutated[field] = value

    with pytest.raises(V16ReadinessV2Error, match="must remain no_new_risk"):
        validate_v16_run_readiness_v2(seal_semantic(mutated), evidence=evidence)


def test_readiness_v2_rejects_nested_unknown_fields() -> None:
    evidence = _evidence()
    payload = _payload(evidence)
    mutated = {key: item for key, item in payload.items() if key != "semantic_sha256"}
    factor = dict(mutated["factor_production_set"])
    factor["caller_healthy"] = True
    mutated["factor_production_set"] = factor

    with pytest.raises(V16ReadinessV2Error, match="drifts from reopened evidence"):
        validate_v16_run_readiness_v2(seal_semantic(mutated), evidence=evidence)


def test_readiness_v2_rejects_cross_schema_and_mapping_inputs() -> None:
    evidence = _evidence()
    payload = _payload(evidence)
    mutated = {key: item for key, item in payload.items() if key != "semantic_sha256"}
    mutated["schema_version"] = "v16_run_readiness.v1"

    with pytest.raises(V16ReadinessV2Error, match="identity mismatch"):
        validate_v16_run_readiness_v2(seal_semantic(mutated), evidence=evidence)
    with pytest.raises(V16ReadinessV2Error, match="ReadinessEvidenceBundleV2"):
        build_v16_run_readiness_v2(
            run_id="v16-foundation-20260719",
            generated_at="2026-07-19T09:45:00Z",
            analysis_trade_date="2026-07-17",
            evidence={"factor_ready": True},  # type: ignore[arg-type]
        )


def test_readiness_v2_reopens_transitive_registry_bytes() -> None:
    evidence = _evidence()
    payload = _payload(evidence)
    registry = evidence.factor_production_set.legacy_registry
    object.__setattr__(registry, "payload", registry.payload + b"\n")

    with pytest.raises(V16ReadinessV2Error, match="byte SHA drift"):
        validate_v16_run_readiness_v2(payload, evidence=evidence)


def test_schedule_lineage_rejects_v2_anchor_type() -> None:
    v2_anchor = ScheduleAnchorBinding(schedule=None, timestamp=None)  # type: ignore[arg-type]
    bundle = ScheduleLineageEvidenceBundleV3(
        genesis=None,  # type: ignore[arg-type]
        schedule_anchors=(v2_anchor,),  # type: ignore[arg-type]
    )

    with pytest.raises(V16ReadinessV2Error, match="genesis has the wrong type"):
        bundle.read()


def test_readiness_v2_validation_performs_no_path_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    payload = _payload(evidence)

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("readiness validation attempted path I/O")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert validate_v16_run_readiness_v2(payload, evidence=evidence) == payload
