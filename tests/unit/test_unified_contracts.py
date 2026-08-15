from __future__ import annotations

import hashlib
import math
from typing import Any

import pytest

from quant_investor.contracts import (
    ARTIFACT_ENVELOPE_FIELDS,
    LEGACY_CONTRACT_FIELDS,
    ArtifactValidationError,
    CanonicalJSONError,
    ContractDefinition,
    ContractRegistrationError,
    SYSTEM_ASSEMBLY_REQUEST_CONTRACT,
    SYSTEM_GENERATION_MANIFEST_CONTRACT,
    SYSTEM_INSTALLED_COMPONENT_MANIFEST_CONTRACT,
    SYSTEM_RELEASE_CONTRACT,
    SYSTEM_SOURCE_BUNDLE_CONTRACT,
    SYSTEM_SOURCE_OBJECT_CONTRACT,
    SYSTEM_VALIDATION_ATTESTATION_CONTRACT,
    SYSTEM_VALIDATION_RUN_REQUEST_CONTRACT,
    UnknownContractError,
    artifact_byte_sha256,
    artifact_semantic_preimage,
    canonical_json_bytes,
    contract_catalog_sha256,
    get_contract,
    parse_canonical_json_bytes,
    registered_contract_catalog,
    register_contract,
    seal_artifact,
    validate_artifact,
)

CREATED_AT = "2026-08-14T00:00:00Z"


def _release() -> dict[str, Any]:
    return seal_artifact(
        "system.release",
        {
            "release_id": "release-test",
            "state": "TEST",
            "code_sha256": "a" * 64,
            "wheel_sha256": "b" * 64,
            "code_manifest_sha256": "c" * 64,
        },
        created_at=CREATED_AT,
    )


def test_envelope_and_semantic_preimage_are_exact_and_acyclic() -> None:
    artifact = _release()

    assert set(artifact) == set(ARTIFACT_ENVELOPE_FIELDS)
    assert artifact["artifact_id"] == artifact["payload"]["release_id"]
    assert artifact_semantic_preimage(artifact) == {
        "domain": "myquant-artifact",
        "kind": "system.release",
        "contract_sha256": artifact["contract_sha256"],
        "identity_field": "release_id",
        "artifact_id": "release-test",
        "created_at": CREATED_AT,
        "payload": artifact["payload"],
    }
    assert "semantic_sha256" not in artifact_semantic_preimage(artifact)
    expected = hashlib.sha256(
        canonical_json_bytes(artifact_semantic_preimage(artifact))
    ).hexdigest()
    assert artifact["semantic_sha256"] == expected


def test_identity_must_be_supplied_and_match_the_envelope() -> None:
    with pytest.raises(ArtifactValidationError):
        seal_artifact(
            "system.release",
            {
                "state": "TEST",
                "code_sha256": "a" * 64,
                "wheel_sha256": "b" * 64,
                "code_manifest_sha256": "c" * 64,
            },
            created_at=CREATED_AT,
        )

    artifact = _release()
    artifact["artifact_id"] = "forged"
    with pytest.raises(ArtifactValidationError):
        validate_artifact(artifact)


def test_semantic_and_byte_hash_tamper_are_rejected() -> None:
    artifact = _release()
    canonical = canonical_json_bytes(artifact)
    assert artifact_byte_sha256(canonical) == hashlib.sha256(canonical).hexdigest()

    forged = dict(artifact)
    forged["semantic_sha256"] = "0" * 64
    with pytest.raises(ArtifactValidationError):
        validate_artifact(forged)

    with pytest.raises(CanonicalJSONError):
        artifact_byte_sha256(canonical + b"\n")


def test_unknown_contract_pair_and_legacy_payload_fields_fail_closed() -> None:
    artifact = _release()
    artifact["contract_sha256"] = "0" * 64
    with pytest.raises(UnknownContractError):
        validate_artifact(artifact)

    with pytest.raises(ArtifactValidationError):
        seal_artifact(
            "system.release",
            {
                "release_id": "legacy",
                "state": "TEST",
                "code_sha256": "a" * 64,
                "wheel_sha256": "b" * 64,
                "code_manifest_sha256": "c" * 64,
                "version": "1",
            },
            created_at=CREATED_AT,
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "missing", ["wheel_sha256", "code_manifest_sha256"]
)
def test_release_requires_explicit_wheel_and_code_manifest_identities(missing: str) -> None:
    assert SYSTEM_RELEASE_CONTRACT.required_payload_fields == frozenset(
        {
            "release_id",
            "state",
            "code_sha256",
            "wheel_sha256",
            "code_manifest_sha256",
        }
    )
    payload = dict(_release()["payload"])
    del payload[missing]

    with pytest.raises(ArtifactValidationError):
        seal_artifact("system.release", payload, created_at=CREATED_AT)


def test_release_identity_field_tamper_and_invalid_sha_are_rejected() -> None:
    artifact = _release()
    artifact["payload"]["wheel_sha256"] = "0" * 64
    with pytest.raises(ArtifactValidationError):
        validate_artifact(artifact)

    payload = dict(_release()["payload"])
    payload["code_manifest_sha256"] = "A" * 64
    with pytest.raises(ArtifactValidationError):
        seal_artifact("system.release", payload, created_at=CREATED_AT)


def test_factor_validation_receipt_contract_is_exact_and_versionless() -> None:
    definition = get_contract("factor.validation_receipt")
    assert definition.identity_field == "validation_receipt_id"
    assert definition.required_payload_fields == frozenset(
        {
            "validation_receipt_id",
            "policy_ref",
            "evidence_refs",
            "active_set_ref",
            "validated",
            "authority",
        }
    )
    payload = {
        "validation_receipt_id": "receipt",
        "policy_ref": None,
        "evidence_refs": [],
        "active_set_ref": None,
        "validated": True,
        "authority": "NON_AUTHORIZING",
    }
    assert (
        seal_artifact("factor.validation_receipt", payload, created_at=CREATED_AT)["payload"]
        == payload
    )
    for invalid in (
        {key: value for key, value in payload.items() if key != "validated"},
        {**payload, "version": 1},
    ):
        with pytest.raises(ArtifactValidationError):
            seal_artifact("factor.validation_receipt", invalid, created_at=CREATED_AT)


def test_factor_status_active_projection_requires_context_and_attestation_slots() -> None:
    payload: dict[str, Any] = {
        "status_id": "status-structural-test",
        "active": {
            "state": "ABSENT",
            "lane": "NONE",
            "admission_route": "NONE",
            "producer_identity": "NONE",
            "factor_set_ref": None,
            "factor_ids": [],
            "validation_receipt_ref": None,
            "contextual_result_ref": None,
            "validation_attestation_ref": None,
        },
        "observed": {},
        "readiness": "BLOCKED",
        "blockers": ["FACTOR_ABSENT"],
        "activation_mutation_authorized": False,
    }
    assert seal_artifact("factor.status", payload, created_at=CREATED_AT)["payload"] == payload

    for active in (
        {
            key: value
            for key, value in payload["active"].items()
            if key != "validation_attestation_ref"
        },
        {**payload["active"], "unexpected": None},
    ):
        with pytest.raises(ArtifactValidationError):
            seal_artifact(
                "factor.status",
                {**payload, "active": active},
                created_at=CREATED_AT,
            )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "raw",
    [
        b'{"a":1,"a":1}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1E-7}',
        b'{"a":-0}',
        b'{ "a":1}',
        b'{"a":1}\n',
    ],
)
def test_noncanonical_duplicate_and_nonfinite_json_is_rejected(raw: bytes) -> None:
    with pytest.raises(CanonicalJSONError):
        parse_canonical_json_bytes(raw)


def test_finite_floats_round_trip_in_the_single_canonical_spelling() -> None:
    value = {"negative_zero": -0.0, "small": 1e-07, "weight": 0.5}
    raw = canonical_json_bytes(value)

    assert raw == b'{"negative_zero":-0.0,"small":1e-07,"weight":0.5}'
    parsed = parse_canonical_json_bytes(raw)
    assert parsed["weight"] == 0.5
    assert math.copysign(1.0, parsed["negative_zero"]) == -1.0
    for number in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CanonicalJSONError):
            canonical_json_bytes({"value": number})


def test_optional_payload_fields_are_optional_but_unknown_fields_are_not() -> None:
    definition = ContractDefinition(
        kind="test.optional_contract",
        identity_field="item_id",
        required_payload_fields=frozenset({"item_id", "required"}),
        optional_payload_fields=frozenset({"optional"}),
    )

    assert definition.validate_payload({"item_id": "x", "required": True}) == {
        "item_id": "x",
        "required": True,
    }
    with pytest.raises(ArtifactValidationError):
        definition.validate_payload({"item_id": "x", "required": True, "unknown": True})


def test_catalog_entries_reconstruct_pair_dispatch_and_catalog_hash() -> None:
    catalog = list(registered_contract_catalog())
    assert catalog
    assert catalog == sorted(catalog, key=lambda row: (row["kind"], row["contract_sha256"]))
    assert all(
        set(row)
        == {
            "kind",
            "contract_sha256",
            "identity_field",
            "json_schema_sha256",
            "validator_code_sha256",
        }
        for row in catalog
    )
    for row in catalog:
        preimage = {
            "identity_field": row["identity_field"],
            "json_schema_sha256": row["json_schema_sha256"],
            "kind": row["kind"],
            "validator_code_sha256": row["validator_code_sha256"],
        }
        assert row["contract_sha256"] == hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
    assert (
        contract_catalog_sha256()
        == hashlib.sha256(canonical_json_bytes({"contracts": catalog})).hexdigest()
    )


def test_compiled_contract_registry_rejects_runtime_extension() -> None:
    definition = ContractDefinition(
        kind="test.runtime_extension",
        identity_field="extension_id",
        required_payload_fields=frozenset({"extension_id"}),
    )
    with pytest.raises(ContractRegistrationError, match="frozen"):
        register_contract(definition)

    compiled = get_contract("system.release")
    assert register_contract(compiled) is compiled


def test_system_generation_request_and_source_contracts_are_closed_and_versionless() -> None:
    definitions = (
        SYSTEM_RELEASE_CONTRACT,
        SYSTEM_GENERATION_MANIFEST_CONTRACT,
        SYSTEM_ASSEMBLY_REQUEST_CONTRACT,
        SYSTEM_SOURCE_BUNDLE_CONTRACT,
        SYSTEM_SOURCE_OBJECT_CONTRACT,
        SYSTEM_INSTALLED_COMPONENT_MANIFEST_CONTRACT,
        SYSTEM_VALIDATION_RUN_REQUEST_CONTRACT,
        SYSTEM_VALIDATION_ATTESTATION_CONTRACT,
    )

    for definition in definitions:
        public_fields = definition.required_payload_fields | definition.optional_payload_fields
        assert definition.allow_additional_payload_fields is False
        assert LEGACY_CONTRACT_FIELDS <= definition.forbidden_payload_fields
        assert not public_fields & LEGACY_CONTRACT_FIELDS
        assert all("version" not in field for field in public_fields)
