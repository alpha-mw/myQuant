"""Installed Factor validator manifest bound to the current System release.

The manifest is non-authorizing.  It closes the finite Factor implementation
set and the exact compiled contracts that the contextual validator will replay;
System remains responsible for component custody and validation attestation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
)

from .common import (
    artifact_ref,
    business_identity,
    canonical_timestamp,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .errors import FactorGovernanceError
from .implementations import installed_implementation_rows, installed_semantic_row

VALIDATOR_MANIFEST_KIND: Final = "factor.validator_manifest"

_MANIFEST_FIELDS: Final = {
    "validator_manifest_id",
    "release_manifest_ref",
    "contextual_validator_component_ref",
    "source_decoder_component_ref",
    "implementation_rows",
    "validated_contracts",
    "authority",
}
_IMPLEMENTATION_ROW_FIELDS: Final = {
    "factor_id",
    "implementation_id",
    "implementation_component_ref",
    "module_name",
    "qualified_name",
    "code_sha256",
    "family",
    "primitive",
    "direction",
    "formula",
    "normalized_expression",
    "parameters_json",
    "input_fields",
    "required_source_roles",
}
_VALIDATED_CONTRACT_ROW_FIELDS: Final = {
    "kind",
    "contract_sha256",
    "json_schema_sha256",
    "validator_code_sha256",
}
_MAX_IMPLEMENTATIONS: Final = 20
_MAX_VALIDATED_CONTRACTS: Final = 32


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8", errors="strict")


def _canonical_json_text(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise FactorGovernanceError(f"{label} must be canonical JSON text")
    try:
        parsed = parse_canonical_json_bytes(value.encode("utf-8"), label=label)
    except ContractError as exc:
        raise FactorGovernanceError(f"{label} must be canonical JSON text") from exc
    if not isinstance(parsed, (dict, list)):
        raise FactorGovernanceError(f"{label} must be a canonical JSON container")
    return value


def _normalize_validated_contracts(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError("validated contracts must be a sequence")
    if not 1 <= len(values) <= _MAX_VALIDATED_CONTRACTS:
        raise FactorGovernanceError("validated contract count is outside its bound")

    rows: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != _VALIDATED_CONTRACT_ROW_FIELDS:
            raise FactorGovernanceError(f"validated_contracts[{index}] fields are not exact")
        kind = value.get("kind")
        if type(kind) is not str or not kind.startswith("factor."):
            raise FactorGovernanceError(f"validated_contracts[{index}].kind is not a Factor kind")
        contract_sha256 = require_sha256(
            value.get("contract_sha256"),
            label=f"validated_contracts[{index}].contract_sha256",
        )
        json_schema_sha256 = require_sha256(
            value.get("json_schema_sha256"),
            label=f"validated_contracts[{index}].json_schema_sha256",
        )
        validator_code_sha256 = require_sha256(
            value.get("validator_code_sha256"),
            label=f"validated_contracts[{index}].validator_code_sha256",
        )
        try:
            definition = get_contract(kind, contract_sha256)
        except ContractError as exc:
            raise FactorGovernanceError(
                f"validated_contracts[{index}] is not a compiled contract pair"
            ) from exc
        expected = {
            "kind": definition.kind,
            "contract_sha256": definition.contract_sha256,
            "json_schema_sha256": definition.json_schema_sha256,
            "validator_code_sha256": definition.validator_code_sha256,
        }
        row = {
            "kind": kind,
            "contract_sha256": contract_sha256,
            "json_schema_sha256": json_schema_sha256,
            "validator_code_sha256": validator_code_sha256,
        }
        if row != expected:
            raise FactorGovernanceError(
                f"validated_contracts[{index}] differs from compiled contract code"
            )
        rows.append(row)

    rows.sort(key=lambda row: (_utf8_key(row["kind"]), row["contract_sha256"]))
    keys = [(row["kind"], row["contract_sha256"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise FactorGovernanceError("validated contracts are duplicated")
    return rows


def _normalize_implementation_rows(values: Any) -> list[dict[str, Any]]:
    if type(values) is not list or not 1 <= len(values) <= _MAX_IMPLEMENTATIONS:
        raise FactorGovernanceError("implementation row count is outside its bound")

    component_refs: dict[str, dict[str, str]] = {}
    factor_ids: list[str] = []
    seen_component_refs: set[tuple[str, ...]] = set()
    ref_fields = (
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    )
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != _IMPLEMENTATION_ROW_FIELDS:
            raise FactorGovernanceError(f"implementation_rows[{index}] fields are not exact")
        factor_id = value.get("factor_id")
        if type(factor_id) is not str or not factor_id:
            raise FactorGovernanceError(f"implementation_rows[{index}].factor_id is invalid")
        if factor_id in component_refs:
            raise FactorGovernanceError("implementation factor IDs are duplicated")
        component_ref = validate_artifact_ref(
            value.get("implementation_component_ref"),
            label=f"implementation_rows[{index}].implementation_component_ref",
            expected_kind="system.installed_component_manifest",
        )
        ref_key = tuple(component_ref[field] for field in ref_fields)
        if ref_key in seen_component_refs:
            raise FactorGovernanceError("implementation component refs are duplicated")
        seen_component_refs.add(ref_key)
        component_refs[factor_id] = component_ref
        factor_ids.append(factor_id)

        expected_semantic = installed_semantic_row(factor_id)
        semantic = {field: value[field] for field in expected_semantic}
        if semantic != expected_semantic:
            raise FactorGovernanceError(
                f"implementation_rows[{index}] differs from installed semantics"
            )
        require_sha256(
            value.get("code_sha256"),
            label=f"implementation_rows[{index}].code_sha256",
        )
        _canonical_json_text(
            value.get("normalized_expression"),
            label=f"implementation_rows[{index}].normalized_expression",
        )
        _canonical_json_text(
            value.get("parameters_json"),
            label=f"implementation_rows[{index}].parameters_json",
        )

    expected_rows = installed_implementation_rows(
        implementation_component_refs=component_refs,
        factor_ids=factor_ids,
    )
    if values != expected_rows:
        raise FactorGovernanceError("implementation rows are not in exact installed UTF-8 order")
    return expected_rows


def _manifest_payload(
    *,
    release_manifest_ref: Mapping[str, Any],
    contextual_validator_component_ref: Mapping[str, Any],
    source_decoder_component_ref: Mapping[str, Any],
    implementation_rows: list[dict[str, Any]],
    validated_contracts: list[dict[str, str]],
) -> dict[str, Any]:
    identity = {
        "release_manifest_ref": validate_artifact_ref(
            dict(release_manifest_ref),
            label="release_manifest_ref",
            expected_kind="system.release",
        ),
        "contextual_validator_component_ref": validate_artifact_ref(
            dict(contextual_validator_component_ref),
            label="contextual_validator_component_ref",
            expected_kind="system.installed_component_manifest",
        ),
        "source_decoder_component_ref": validate_artifact_ref(
            dict(source_decoder_component_ref),
            label="source_decoder_component_ref",
            expected_kind="system.installed_component_manifest",
        ),
        "implementation_rows": implementation_rows,
        "validated_contracts": validated_contracts,
        "authority": "NON_AUTHORIZING",
    }
    return {
        "validator_manifest_id": business_identity("factor-validator-manifest", identity),
        **identity,
    }


def build_validator_manifest(
    *,
    release_manifest: Mapping[str, Any] | bytes,
    contextual_validator_component: Mapping[str, Any] | bytes,
    source_decoder_component: Mapping[str, Any] | bytes,
    implementation_components: Mapping[str, Mapping[str, Any] | bytes],
    validated_contracts: Sequence[Mapping[str, Any]],
    trusted_at: str,
) -> dict[str, Any]:
    """Build the non-authorizing manifest through the trusted System seam."""

    release = validate_governance_artifact(release_manifest, expected_kind="system.release")
    contextual = validate_governance_artifact(
        contextual_validator_component,
        expected_kind="system.installed_component_manifest",
    )
    decoder = validate_governance_artifact(
        source_decoder_component,
        expected_kind="system.installed_component_manifest",
    )
    if type(implementation_components) is not dict:
        raise FactorGovernanceError("implementation components must be an exact mapping")
    component_refs: dict[str, dict[str, str]] = {}
    for factor_id, component in implementation_components.items():
        if type(factor_id) is not str or not factor_id:
            raise FactorGovernanceError("implementation component factor ID is invalid")
        normalized = validate_governance_artifact(
            component, expected_kind="system.installed_component_manifest"
        )
        component_refs[factor_id] = artifact_ref(normalized)

    implementation_rows = installed_implementation_rows(
        implementation_component_refs=component_refs
    )
    payload = _manifest_payload(
        release_manifest_ref=artifact_ref(release),
        contextual_validator_component_ref=artifact_ref(contextual),
        source_decoder_component_ref=artifact_ref(decoder),
        implementation_rows=implementation_rows,
        validated_contracts=_normalize_validated_contracts(validated_contracts),
    )
    return seal_artifact(
        VALIDATOR_MANIFEST_KIND,
        payload,
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )


def validate_validator_manifest(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate and replay the exact installed semantic manifest."""

    envelope, payload = exact_payload(
        document,
        kind=VALIDATOR_MANIFEST_KIND,
        fields=_MANIFEST_FIELDS,
    )
    expected = _manifest_payload(
        release_manifest_ref=payload["release_manifest_ref"],
        contextual_validator_component_ref=payload["contextual_validator_component_ref"],
        source_decoder_component_ref=payload["source_decoder_component_ref"],
        implementation_rows=_normalize_implementation_rows(payload["implementation_rows"]),
        validated_contracts=_normalize_validated_contracts(payload["validated_contracts"]),
    )
    if payload != expected:
        raise FactorGovernanceError("Factor validator manifest does not replay exactly")
    return envelope


__all__ = [
    "VALIDATOR_MANIFEST_KIND",
    "build_validator_manifest",
    "validate_validator_manifest",
]
