from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.factors.production_set_carrier_v4 import (
    FACTOR_PRODUCTION_SET_CARRIER_SCHEMA,
    FactorProductionSetCarrierV4Error,
    FactorProductionSetEvidenceBundleV4,
    bind_factor_production_set_carrier_v4,
    build_factor_production_set_carrier_v4,
    validate_factor_production_set_carrier_v4,
)
from quant_investor.factors.registry_store import (
    FactorRegistryMalformedError,
    FactorRegistryValidationError,
    parse_registry_snapshot_bytes_strict,
    registry_payload_semantic_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256
from quant_investor.factors.evidence_contracts import (
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    canonical_json_bytes,
    seal_semantic,
)

REGISTRY_PATH = Path("quant_investor/factor_registry/mined_factors.json").resolve()


def _registry_artifact(raw: bytes) -> BoundRawArtifact:
    payload = json.loads(raw)
    return BoundRawArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema="mined-factor-registry.v1",
            absolute_path=str(REGISTRY_PATH),
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=registry_payload_semantic_sha256(payload),
            root_policy="v16.governed-data-root.v2",
        ),
        payload=raw,
    )


def _bundle(raw: bytes) -> FactorProductionSetEvidenceBundleV4:
    registry = _registry_artifact(raw)
    carrier = build_factor_production_set_carrier_v4(
        as_of="2026-07-17",
        legacy_registry_ref=registry.reference,
    )
    return FactorProductionSetEvidenceBundleV4(
        carrier=bind_factor_production_set_carrier_v4(
            carrier,
            absolute_path="/private/v16/factor-production-set-carrier.json",
        ),
        legacy_registry=registry,
    )


def test_bound_current_registry_derives_one_factor_and_never_health() -> None:
    readback = _bundle(REGISTRY_PATH.read_bytes()).read()

    assert readback["production_factor_names"] == ["pv_low_dollar_volume_5d"]
    assert readback["production_factor_count"] == 1
    assert readback["production_family_count"] is None
    assert readback["healthy_factor_count"] == 0
    assert readback["factor_governance_ready"] is False
    assert readback["new_risk_authorized"] is False
    assert "production_factor_count_below_5" in readback["blockers"]
    assert "factor_abs_weight_above_0.20:pv_low_dollar_volume_5d" in readback["blockers"]
    assert "factor_v4_fresh_health_evidence_not_integrated" in readback["blockers"]


def test_carrier_rejects_caller_claimed_health_even_when_resealed() -> None:
    registry = _registry_artifact(REGISTRY_PATH.read_bytes())
    payload = build_factor_production_set_carrier_v4(
        as_of="2026-07-17",
        legacy_registry_ref=registry.reference,
    )
    mutated = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    mutated["healthy"] = True

    with pytest.raises(FactorProductionSetCarrierV4Error, match="fields mismatch"):
        validate_factor_production_set_carrier_v4(seal_semantic(mutated))


def test_bundle_detects_registry_byte_drift_after_binding() -> None:
    bundle = _bundle(REGISTRY_PATH.read_bytes())
    object.__setattr__(bundle.legacy_registry, "payload", bundle.legacy_registry.payload + b"\n")

    with pytest.raises(FactorProductionSetCarrierV4Error, match="byte SHA drift"):
        bundle.read()


def test_strict_byte_parser_rejects_duplicate_and_unknown_record_fields() -> None:
    with pytest.raises(FactorRegistryMalformedError, match="duplicate JSON key"):
        parse_registry_snapshot_bytes_strict(
            b'{"schema_version":"mined-factor-registry.v1",'
            b'"schema_version":"mined-factor-registry.v1",'
            b'"metadata":{},"factors":[]}'
        )

    payload = json.loads(REGISTRY_PATH.read_bytes())
    payload["factors"][0]["family"] = "caller-invented"
    with pytest.raises(FactorRegistryValidationError, match="unsupported fields"):
        parse_registry_snapshot_bytes_strict(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )


def test_overfilled_registry_is_derived_and_blocked() -> None:
    payload = json.loads(REGISTRY_PATH.read_bytes())
    selected = payload["factors"][:11]
    for row in selected:
        row["state"] = "production_factor"
        row["weight"] = 1.0
    names = sorted(str(row["name"]) for row in selected)
    payload["factors"] = selected
    payload["metadata"]["production_factor_count"] = len(names)
    payload["metadata"]["production_factor_names"] = names
    payload["metadata"]["production_factor_set_sha256"] = production_factor_set_sha256(
        names
    )
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    readback = _bundle(raw).read()

    assert readback["production_factor_count"] == 11
    assert "production_factor_count_above_target_10" in readback["blockers"]
    assert readback["factor_governance_ready"] is False


def test_readback_does_not_open_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(REGISTRY_PATH.read_bytes())

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("readback attempted filesystem I/O")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert bundle.read()["schema_version"].endswith(".foundation")


def test_carrier_schema_is_distinct_from_legacy_registry() -> None:
    assert FACTOR_PRODUCTION_SET_CARRIER_SCHEMA != "mined-factor-registry.v1"


def test_neutral_module_preserves_legacy_carrier_bytes_and_semantic_sha() -> None:
    registry_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema="mined-factor-registry.v1",
        absolute_path="/registry.json",
        byte_sha256="0" * 64,
        semantic_sha256="1" * 64,
        root_policy="v16.governed-data-root.v2",
    )
    payload = build_factor_production_set_carrier_v4(
        as_of="2026-07-17",
        legacy_registry_ref=registry_ref,
    )

    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == (
        "8c6d0346a6314e01b8b1de77d0438435e798ce487f2d7e545c3386374ced663e"
    )
    assert payload["semantic_sha256"] == (
        "13ee0e080098d09608d5fefa4a7cd6df75ec5702a579352a11c4982b2ef6e8ca"
    )
