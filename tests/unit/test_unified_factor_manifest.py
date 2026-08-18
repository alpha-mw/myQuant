from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap

import pytest

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
)
from quant_investor.factors.governance import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorGovernanceError,
    validate_validator_manifest,
)
from quant_investor.factors.governance.bootstrap import BOOTSTRAP_REQUIRED_SOURCE_ROLES
from quant_investor.factors.governance.implementations import (
    implementation_code_sha256,
    installed_semantic_row,
)
from quant_investor.factors.governance.manifest import build_validator_manifest
import quant_investor.factors.governance.implementations as implementation_module

STAMP = "2026-08-14T00:00:00Z"
SHA = "1" * 64


def _release() -> dict:
    return seal_artifact(
        "system.release",
        {
            "release_id": "release-test",
            "state": "INSTALLED",
            "code_sha256": "2" * 64,
            "wheel_sha256": "3" * 64,
            "code_manifest_sha256": "4" * 64,
        },
        created_at=STAMP,
    )


def _component(release: dict, component_id: str, role: str) -> dict:
    from quant_investor.factors.governance.common import artifact_ref

    return seal_artifact(
        "system.installed_component_manifest",
        {
            "component_manifest_id": f"manifest-{component_id}",
            "component_id": component_id,
            "component_registry_sha256": SHA,
            "component_role": role,
            "package_name": "quant_investor.factors.governance",
            "module_names": ["quant_investor.factors.governance.implementations"],
            "entrypoints": [],
            "files": [],
            "release_manifest_ref": artifact_ref(release),
            "installed_code_manifest_sha256": "4" * 64,
            "allowed_source_formats": ["PARQUET"] if role == "SOURCE_DECODER" else [],
            "fallback_allowed": False,
            "component_sha256": hashlib.sha256(component_id.encode()).hexdigest(),
            "outcome": "VALIDATED",
            "authority": "NON_AUTHORIZING",
        },
        created_at=STAMP,
    )


def _contract_row(kind: str = "factor.bootstrap_set") -> dict[str, str]:
    definition = get_contract(kind)
    return {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "json_schema_sha256": definition.json_schema_sha256,
        "validator_code_sha256": definition.validator_code_sha256,
    }


def _manifest() -> dict:
    release = _release()
    return build_validator_manifest(
        release_manifest=release,
        contextual_validator_component=_component(
            release, "factor-contextual-validator", "CONTEXTUAL_VALIDATOR"
        ),
        source_decoder_component=_component(release, "factor-source-decoder", "SOURCE_DECODER"),
        implementation_components={
            LOW_DOLLAR_VOLUME: _component(
                release, "factor-low-dollar-volume", "SOURCE_IMPLEMENTATION"
            ),
            BLEND_W80: _component(release, "factor-blend-w80", "SOURCE_IMPLEMENTATION"),
        },
        validated_contracts=[_contract_row()],
        trusted_at=STAMP,
    )


def _reseal(payload: dict) -> dict:
    return seal_artifact("factor.validator_manifest", payload, created_at=STAMP)


def test_validator_manifest_replays_exact_installed_semantic_identity() -> None:
    manifest = validate_validator_manifest(_manifest())
    rows = manifest["payload"]["implementation_rows"]

    assert [row["factor_id"] for row in rows] == [
        BLEND_W80,
        LOW_DOLLAR_VOLUME,
    ]
    for row in rows:
        expected = installed_semantic_row(row["factor_id"])
        assert {field: row[field] for field in expected} == expected
        assert row["required_source_roles"] == list(BOOTSTRAP_REQUIRED_SOURCE_ROLES)
    assert manifest["payload"]["authority"] == "NON_AUTHORIZING"


@pytest.mark.parametrize("field", ["primitive", "formula", "direction"])
def test_validator_manifest_rejects_semantic_or_component_forgery(field: str) -> None:
    payload = copy.deepcopy(_manifest()["payload"])
    payload["implementation_rows"][0][field] = "forged"
    with pytest.raises(FactorGovernanceError):
        validate_validator_manifest(_reseal(payload))

    payload = copy.deepcopy(_manifest()["payload"])
    payload["implementation_rows"][0]["implementation_component_ref"] = copy.deepcopy(
        payload["contextual_validator_component_ref"]
    )
    with pytest.raises(FactorGovernanceError):
        validate_validator_manifest(_reseal(payload))


@pytest.mark.parametrize(
    "mutated_roles",
    [
        ["EXCHANGE_CALENDAR", "MARKET"],
        ["EXCHANGE_CALENDAR", "FUNDAMENTAL", "MARKET", "PIT_MEMBERSHIP"],
        ["MARKET", "EXCHANGE_CALENDAR", "PIT_MEMBERSHIP"],
        ["EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP", "UNKNOWN"],
    ],
)
def test_validator_manifest_rejects_required_source_role_mutation(
    mutated_roles: list[str],
) -> None:
    payload = copy.deepcopy(_manifest()["payload"])
    payload["implementation_rows"][0]["required_source_roles"] = mutated_roles
    with pytest.raises((ContractError, FactorGovernanceError)):
        validate_validator_manifest(_reseal(payload))


def test_validator_manifest_rejects_noncanonical_json_and_row_order() -> None:
    payload = copy.deepcopy(_manifest()["payload"])
    payload["implementation_rows"][0]["parameters_json"] = '{"x": 1}'
    with pytest.raises(FactorGovernanceError):
        validate_validator_manifest(_reseal(payload))

    payload = copy.deepcopy(_manifest()["payload"])
    payload["implementation_rows"].reverse()
    with pytest.raises(FactorGovernanceError):
        validate_validator_manifest(_reseal(payload))


@pytest.mark.parametrize("field", ["json_schema_sha256", "validator_code_sha256"])
def test_validator_manifest_rejects_compiled_contract_hash_forgery(field: str) -> None:
    payload = copy.deepcopy(_manifest()["payload"])
    payload["validated_contracts"][0][field] = "f" * 64
    with pytest.raises(FactorGovernanceError):
        validate_validator_manifest(_reseal(payload))


def test_w75_has_no_installed_implementation() -> None:
    with pytest.raises(FactorGovernanceError):
        installed_semantic_row(BLEND_W75_CONTROL)


@pytest.mark.parametrize(
    ("factor_id", "entrypoint_name"),
    [
        (LOW_DOLLAR_VOLUME, "_low_dollar_volume"),
        (BLEND_W80, "_blend_w80"),
    ],
)
def test_ast_identity_uses_module_qualname_and_attribute_free_node(
    factor_id: str, entrypoint_name: str
) -> None:
    entrypoint = getattr(implementation_module, entrypoint_name)
    parsed = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    node = parsed.body[0]
    preimage = {
        "domain": "myquant-python-ast-entrypoint",
        "module_name": entrypoint.__module__,
        "qualified_name": entrypoint.__qualname__,
        "node": ast.dump(node, annotate_fields=True, include_attributes=False),
    }
    assert (
        implementation_code_sha256(factor_id)
        == hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
    )
    assert not any(
        token in preimage["node"]
        for token in ("lineno=", "col_offset=", "end_lineno=", "end_col_offset=")
    )
