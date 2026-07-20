from __future__ import annotations

import builtins
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter
from quant_investor.factors.governance_screening_v4 import (
    build_candidate_catalog_v4,
    build_primitive_ontology_v4,
    canonical_semantic_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate(
    name: str,
    primitive_id: str,
    *,
    implementation: str,
) -> dict[str, Any]:
    input_field = primitive_id.replace("primitive_", "field_")
    return {
        "name": name,
        "implementation": implementation,
        "expression": input_field,
        "direction": 1,
        "params": {},
        "lookback": 2,
        "slot": f"slot:{primitive_id}",
        "input_fields": [input_field],
        "primitive_ids": [primitive_id],
    }


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[field] = adapter.semantic_sha256(result, exclude_fields=(field,))
    return result


def _mapping_row(candidate: dict[str, Any], source_index: int) -> dict[str, Any]:
    primitive_id = candidate["primitive_ids"][0]
    input_field = candidate["input_fields"][0]
    name = candidate["name"]
    return _seal(
        {
            "candidate_id": f"source:{source_index:03d}:{name}",
            "name": name,
            "source_definition_sha256": _digest(f"source-definition:{name}"),
            "catalog_definition_sha256": candidate["definition_sha256"],
            "implementation": adapter.CLASSIFICATION_IDENTITY_IMPLEMENTATION,
            "expression": candidate["expression"],
            "full_candidate_normalized_ast_sha256": _digest(f"ast:{name}"),
            "input_fields": list(candidate["input_fields"]),
            "primitive_ids": list(candidate["primitive_ids"]),
            "family": candidate["family"],
            "slot": candidate["slot"],
            "occurrence_count": 1,
            "occurrences": [
                {
                    "node_path": "body[0]",
                    "node_occurrence_index": 0,
                    "rule_occurrence_index": 0,
                    "subtree_sha256": _digest(f"subtree:{name}"),
                    "rule_id": f"rule:{primitive_id}",
                    "primitive_id": primitive_id,
                    "family": candidate["family"],
                    "match_cardinality": 1,
                    "consumed_identifiers": [
                        {
                            "identifier": input_field,
                            "node_path": "body[0].name",
                            "node_occurrence_index": 0,
                        }
                    ],
                }
            ],
            "mapping_status": "complete_unique_occurrence_accounting",
        },
        "mapping_semantic_sha256",
    )


def _mapping_proof(
    *,
    base_ontology: dict[str, Any],
    base_catalog: dict[str, Any],
    ontology: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    base_names = {candidate["name"] for candidate in base_catalog["candidates"]}
    new_candidates = [
        candidate
        for candidate in catalog["candidates"]
        if candidate["name"] not in base_names
    ]
    mappings = [
        _mapping_row(candidate, index)
        for index, candidate in enumerate(new_candidates)
    ]
    mappings.sort(key=lambda row: (row["candidate_id"], row["name"]))
    source_ids = sorted(row["candidate_id"] for row in mappings)
    return _seal(
        {
            "schema_version": adapter.PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
            "protocol_version": "v4.1",
            "cycle_id": "synthetic_formal_catalog_adapter",
            "source_idea_audit_sha256": _digest("source-audit"),
            "discovery_catalog_sha256": _digest("discovery-catalog"),
            "base_ontology_sha256": base_ontology["semantic_sha256"],
            "base_catalog_sha256": base_catalog["semantic_sha256"],
            "mapping_policy_sha256": _digest("mapping-policy"),
            "formal_ontology_sha256": ontology["semantic_sha256"],
            "formal_catalog_sha256": catalog["semantic_sha256"],
            "source_candidate_count": len(mappings),
            "new_candidate_count": len(mappings),
            "structural_alias_count": 0,
            "incompatible_count": 0,
            "catalog_base_candidate_count": len(base_catalog["candidates"]),
            "catalog_new_candidate_count": len(mappings),
            "catalog_total_candidate_count": len(catalog["candidates"]),
            "source_candidate_ids_sha256": adapter.semantic_sha256(source_ids),
            "new_candidate_mappings": mappings,
            "structural_aliases": [],
            "incompatible_candidates": [],
            "classification_only": True,
            "runtime_equivalence_claimed": False,
            "screening_eligible": False,
            "proposal_eligible": False,
            "registry_entry_created": False,
            "initial_weight_policy": "zero_only",
            "formal_admission_authority": False,
            "production_apply_enabled": False,
        },
        "proof_semantic_sha256",
    )


def _synthetic_inputs(
    *,
    base_count: int,
    new_count: int,
) -> dict[str, dict[str, Any]]:
    base_primitives = [
        {
            "primitive_id": f"primitive_base_{index:02d}",
            "family": f"family_base_{index:02d}",
        }
        for index in range(base_count)
    ]
    new_primitives = [
        {
            "primitive_id": f"primitive_new_{index:02d}",
            "family": f"family_new_{index:02d}",
        }
        for index in range(new_count)
    ]
    base_ontology = build_primitive_ontology_v4(base_primitives)
    ontology = build_primitive_ontology_v4([*base_primitives, *new_primitives])
    base_definitions = [
        _candidate(
            f"base_{index:02d}",
            f"primitive_base_{index:02d}",
            implementation=f"bound-base:{index:02d}",
        )
        for index in range(base_count)
    ]
    new_definitions = [
        _candidate(
            f"new_{index:02d}",
            f"primitive_new_{index:02d}",
            implementation=adapter.CLASSIFICATION_IDENTITY_IMPLEMENTATION,
        )
        for index in range(new_count)
    ]
    base_catalog = build_candidate_catalog_v4(
        ontology=base_ontology,
        candidates=base_definitions,
    )
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[*base_definitions, *new_definitions],
    )
    proof = _mapping_proof(
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=ontology,
        catalog=catalog,
    )
    return {
        "base_ontology": base_ontology,
        "base_catalog": base_catalog,
        "ontology": ontology,
        "catalog": catalog,
        "mapping_proof": proof,
    }


def _build(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return adapter.build_formal_catalog_adapter_validation_v4_1(**inputs)


def _validate(
    artifact: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return adapter.validate_formal_catalog_adapter_validation_v4_1(
        artifact, **inputs
    )


def _rehash_candidate(candidate: dict[str, Any]) -> None:
    candidate["definition_sha256"] = canonical_semantic_sha256(
        candidate,
        exclude_fields=("definition_sha256",),
    )


def _rehash_catalog(catalog: dict[str, Any]) -> None:
    catalog["semantic_sha256"] = canonical_semantic_sha256(
        catalog,
        exclude_fields=("semantic_sha256",),
    )


def _rehash_mapping(mapping: dict[str, Any]) -> None:
    mapping["mapping_semantic_sha256"] = adapter.semantic_sha256(
        mapping,
        exclude_fields=("mapping_semantic_sha256",),
    )


def _rehash_proof(proof: dict[str, Any]) -> None:
    proof["proof_semantic_sha256"] = adapter.semantic_sha256(
        proof,
        exclude_fields=("proof_semantic_sha256",),
    )


@pytest.mark.parametrize(("base_count", "new_count"), [(2, 1), (3, 2)])
def test_validate_only_adapter_is_generic_and_not_hardcoded_to_230_37(
    base_count: int,
    new_count: int,
) -> None:
    inputs = _synthetic_inputs(base_count=base_count, new_count=new_count)

    artifact = _build(inputs)

    assert artifact["schema_version"] == (
        "factor-governance-formal-catalog-adapter-validation.v4.1"
    )
    assert artifact["candidate_count"] == base_count + new_count
    assert artifact["base_candidate_count"] == base_count
    assert artifact["new_candidate_count"] == new_count
    assert artifact["descriptor_count"] == base_count + new_count
    assert _validate(artifact, inputs) == artifact
    assert artifact["validation_semantic_sha256"] == adapter.semantic_sha256(
        artifact,
        exclude_fields=("validation_semantic_sha256",),
    )


def test_all_descriptors_and_artifact_are_explicitly_non_executable() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=2)

    artifact = _build(inputs)

    assert artifact["catalog_schema_loader_readable"] is True
    assert artifact["classification_only"] is True
    assert artifact["source_authenticity_recomputed"] is False
    assert artifact["signal_computability_proven"] is False
    for field in (
        "data_loader_invoked",
        "statistics_invoked",
        "registry",
        "proposal",
        "apply",
        "registry_entry_created",
        "registry_mutation_performed",
        "proposal_eligible",
        "screening_eligible",
        "runtime_equivalence_verified",
        "production_apply_enabled",
    ):
        assert artifact[field] is False
    assert artifact["blockers"] == [
        "data_loader_not_invoked",
        "runtime_equivalence_not_verified",
        "screening_not_authorized",
        "signal_computability_not_proven",
        "source_authenticity_not_recomputed_by_adapter",
        "statistics_not_invoked",
    ]
    for descriptor in artifact["descriptors"]:
        assert adapter.validate_formal_catalog_candidate_descriptor_v4_1(
            descriptor
        ) == descriptor
        assert descriptor["classification_only"] is True
        assert descriptor["executable"] is False
        assert descriptor["screening_eligible"] is False
        assert descriptor["runtime_equivalence_verified"] is False
        assert descriptor["blockers"]


def test_base_ontology_and_candidate_definitions_must_be_preserved_exactly() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    drifted_ontology = copy.deepcopy(inputs)
    drifted_ontology["ontology"]["primitives"][0]["family"] = "drifted-family"
    drifted_ontology["ontology"]["semantic_sha256"] = canonical_semantic_sha256(
        drifted_ontology["ontology"], exclude_fields=("semantic_sha256",)
    )
    drifted_primitive = drifted_ontology["ontology"]["primitives"][0][
        "primitive_id"
    ]
    formal_candidate = next(
        row
        for row in drifted_ontology["catalog"]["candidates"]
        if drifted_primitive in row["primitive_ids"]
    )
    formal_candidate["family"] = "drifted-family"
    _rehash_candidate(formal_candidate)
    drifted_ontology["catalog"]["ontology_sha256"] = drifted_ontology[
        "ontology"
    ]["semantic_sha256"]
    _rehash_catalog(drifted_ontology["catalog"])
    drifted_ontology["mapping_proof"]["formal_ontology_sha256"] = (
        drifted_ontology["ontology"]["semantic_sha256"]
    )
    drifted_ontology["mapping_proof"]["formal_catalog_sha256"] = (
        drifted_ontology["catalog"]["semantic_sha256"]
    )
    _rehash_proof(drifted_ontology["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="base primitive mapping",
    ):
        _build(drifted_ontology)

    drifted_definition = copy.deepcopy(inputs)
    base_name = drifted_definition["base_catalog"]["candidates"][0]["name"]
    formal_base = next(
        row
        for row in drifted_definition["catalog"]["candidates"]
        if row["name"] == base_name
    )
    formal_base["expression"] = "different_bound_definition"
    _rehash_candidate(formal_base)
    _rehash_catalog(drifted_definition["catalog"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="base candidate definition drift",
    ):
        _build(drifted_definition)


def test_rejects_unknown_implementation_direction_and_primitive() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    new_name = inputs["mapping_proof"]["new_candidate_mappings"][0]["name"]

    unknown_implementation = copy.deepcopy(inputs)
    row = next(
        candidate
        for candidate in unknown_implementation["catalog"]["candidates"]
        if candidate["name"] == new_name
    )
    row["implementation"] = "unknown:runtime"
    _rehash_candidate(row)
    _rehash_catalog(unknown_implementation["catalog"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="unknown new-candidate implementation",
    ):
        _build(unknown_implementation)

    wrong_direction = copy.deepcopy(inputs)
    row = next(
        candidate
        for candidate in wrong_direction["catalog"]["candidates"]
        if candidate["name"] == new_name
    )
    row["direction"] = -1.0
    _rehash_candidate(row)
    _rehash_catalog(wrong_direction["catalog"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match=r"direction must be canonical \+1.0",
    ):
        _build(wrong_direction)

    unknown_primitive = copy.deepcopy(inputs)
    row = next(
        candidate
        for candidate in unknown_primitive["catalog"]["candidates"]
        if candidate["name"] == new_name
    )
    row["primitive_ids"] = ["unknown_primitive"]
    row["family"] = "unknown_family"
    _rehash_candidate(row)
    _rehash_catalog(unknown_primitive["catalog"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="unknown primitive_id",
    ):
        _build(unknown_primitive)


def test_rejects_duplicate_catalog_and_missing_or_extra_proof_names() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=2)

    duplicate = copy.deepcopy(inputs)
    duplicate["catalog"]["candidates"].append(
        copy.deepcopy(duplicate["catalog"]["candidates"][-1])
    )
    duplicate["catalog"]["candidates"].sort(key=lambda row: row["name"])
    _rehash_catalog(duplicate["catalog"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="sorted by unique name",
    ):
        _build(duplicate)

    missing = copy.deepcopy(inputs)
    removed = missing["mapping_proof"]["new_candidate_mappings"].pop()
    missing["mapping_proof"]["new_candidate_count"] -= 1
    missing["mapping_proof"]["source_candidate_count"] -= 1
    missing["mapping_proof"]["source_candidate_ids_sha256"] = (
        adapter.semantic_sha256(
            sorted(
                row["candidate_id"]
                for row in missing["mapping_proof"]["new_candidate_mappings"]
            )
        )
    )
    _rehash_proof(missing["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="new names drift",
    ):
        _build(missing)

    extra = copy.deepcopy(inputs)
    extra_row = copy.deepcopy(removed)
    extra_row["candidate_id"] = "source:999:extra_name"
    extra_row["name"] = "extra_name"
    _rehash_mapping(extra_row)
    extra["mapping_proof"]["new_candidate_mappings"].append(extra_row)
    extra["mapping_proof"]["new_candidate_mappings"].sort(
        key=lambda row: (row["candidate_id"], row["name"])
    )
    extra["mapping_proof"]["new_candidate_count"] += 1
    extra["mapping_proof"]["source_candidate_count"] += 1
    extra["mapping_proof"]["source_candidate_ids_sha256"] = (
        adapter.semantic_sha256(
            sorted(
                row["candidate_id"]
                for row in extra["mapping_proof"]["new_candidate_mappings"]
            )
        )
    )
    _rehash_proof(extra["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="new names drift",
    ):
        _build(extra)


def test_rejects_source_proof_drift_and_discovery_schema_substitution() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)

    source_drift = copy.deepcopy(inputs)
    source_drift["mapping_proof"]["new_candidate_mappings"][0][
        "source_definition_sha256"
    ] = _digest("drifted-source")
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="semantic SHA mismatch",
    ):
        _build(source_drift)

    proof_drift = copy.deepcopy(inputs)
    proof_drift["mapping_proof"]["formal_catalog_sha256"] = _digest(
        "other-formal-catalog"
    )
    _rehash_proof(proof_drift["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="binding drift: formal_catalog_sha256",
    ):
        _build(proof_drift)

    discovery_catalog = copy.deepcopy(inputs)
    discovery_catalog["catalog"] = {
        "schema_version": "factor-governance-discovery-catalog.v4.1",
        "members": [],
        "catalog_semantic_sha256": _digest("discovery"),
    }
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="schema validation failed",
    ):
        _build(discovery_catalog)

    discovery_proof = copy.deepcopy(inputs)
    discovery_proof["mapping_proof"] = {
        "schema_version": "factor-governance-discovery-catalog.v4.1",
        "members": [],
        "catalog_semantic_sha256": _digest("discovery"),
    }
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="mapping proof fields invalid",
    ):
        _build(discovery_proof)


def test_resealed_source_identity_is_explicitly_an_upstream_validation_scope() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    mapping = inputs["mapping_proof"]["new_candidate_mappings"][0]
    changed_source_sha = _digest("resealed-but-not-source-recomputed")
    mapping["source_definition_sha256"] = changed_source_sha
    _rehash_mapping(mapping)
    _rehash_proof(inputs["mapping_proof"])

    artifact = _build(inputs)

    new_descriptor = next(
        row for row in artifact["descriptors"] if row["catalog_role"] == "new_classification"
    )
    assert new_descriptor["source_definition_sha256"] == changed_source_sha
    assert artifact["source_authenticity_recomputed"] is False
    assert "source_authenticity_not_recomputed_by_adapter" in artifact["blockers"]
    assert "source_authenticity_not_recomputed_by_adapter" in new_descriptor[
        "blockers"
    ]


def test_occurrence_accounting_is_exact_and_supports_multiple_matches() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    mapping = inputs["mapping_proof"]["new_candidate_mappings"][0]
    second = copy.deepcopy(mapping["occurrences"][0])
    second["node_path"] = "body[1]"
    second["node_occurrence_index"] = 1
    second["rule_occurrence_index"] = 1
    second["subtree_sha256"] = _digest("second-subtree")
    second["consumed_identifiers"][0]["node_path"] = "body[1].name"
    second["consumed_identifiers"][0]["node_occurrence_index"] = 1
    mapping["occurrences"].append(second)
    mapping["occurrence_count"] = 2
    for occurrence in mapping["occurrences"]:
        occurrence["match_cardinality"] = 2
    _rehash_mapping(mapping)
    _rehash_proof(inputs["mapping_proof"])

    artifact = _build(inputs)

    assert artifact["candidate_count"] == 3
    duplicate_occurrence = copy.deepcopy(inputs)
    duplicate_occurrence["mapping_proof"]["new_candidate_mappings"][0][
        "occurrences"
    ][1]["node_occurrence_index"] = 0
    _rehash_mapping(
        duplicate_occurrence["mapping_proof"]["new_candidate_mappings"][0]
    )
    _rehash_proof(duplicate_occurrence["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="unique node occurrence index",
    ):
        _build(duplicate_occurrence)


def test_alias_and_incompatible_source_rows_remain_audited_and_excluded() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    proof = inputs["mapping_proof"]
    base_target = inputs["base_catalog"]["candidates"][0]
    alias = _seal(
        {
            "candidate_id": "source:900:alias_only",
            "name": "alias_only",
            "source_definition_sha256": _digest("alias-source-definition"),
            "structural_fingerprint_sha256": _digest("alias-fingerprint"),
            "target_candidate_id": "base:bound:target",
            "target_name": base_target["name"],
            "target_definition_sha256": base_target["definition_sha256"],
            "target_match_cardinality": 1,
            "excluded_from_catalog": True,
        },
        "alias_semantic_sha256",
    )
    incompatible = _seal(
        {
            "source_index": 2,
            "candidate_id": "source:901:incompatible_only",
            "name": "incompatible_only",
            "incompatibility_reasons": ["unsupported_call:rolling_std"],
            "reason_records": [
                {
                    "code": "unsupported_call",
                    "reason": "unsupported_call:rolling_std",
                }
            ],
            "excluded_from_catalog": True,
            "mapping_not_attempted": True,
        },
        "incompatible_semantic_sha256",
    )
    proof["structural_aliases"] = [alias]
    proof["incompatible_candidates"] = [incompatible]
    proof["structural_alias_count"] = 1
    proof["incompatible_count"] = 1
    proof["source_candidate_count"] = 3
    proof["source_candidate_ids_sha256"] = adapter.semantic_sha256(
        sorted(
            [
                proof["new_candidate_mappings"][0]["candidate_id"],
                alias["candidate_id"],
                incompatible["candidate_id"],
            ]
        )
    )
    _rehash_proof(proof)

    artifact = _build(inputs)

    assert artifact["candidate_count"] == 3
    assert {row["name"] for row in artifact["descriptors"]} == {
        "base_00",
        "base_01",
        "new_00",
    }

    reason_drift = copy.deepcopy(inputs)
    reason_row = reason_drift["mapping_proof"]["incompatible_candidates"][0]
    reason_row["reason_records"][0]["code"] = "different_code"
    reason_row["incompatible_semantic_sha256"] = adapter.semantic_sha256(
        reason_row, exclude_fields=("incompatible_semantic_sha256",)
    )
    _rehash_proof(reason_drift["mapping_proof"])
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="code prefix mismatch",
    ):
        _build(reason_drift)


def test_exact_fields_self_hash_and_input_rebinding_fail_closed() -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    artifact = _build(inputs)

    unknown = copy.deepcopy(artifact)
    unknown["unexpected"] = False
    unknown["validation_semantic_sha256"] = adapter.semantic_sha256(
        unknown, exclude_fields=("validation_semantic_sha256",)
    )
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="fields invalid",
    ):
        _validate(unknown, inputs)

    executable = copy.deepcopy(artifact)
    executable["descriptors"][0]["executable"] = True
    executable["descriptors"][0]["descriptor_semantic_sha256"] = (
        adapter.semantic_sha256(
            executable["descriptors"][0],
            exclude_fields=("descriptor_semantic_sha256",),
        )
    )
    executable["validation_semantic_sha256"] = adapter.semantic_sha256(
        executable, exclude_fields=("validation_semantic_sha256",)
    )
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="descriptor.executable must be false",
    ):
        _validate(executable, inputs)

    other_inputs = _synthetic_inputs(base_count=3, new_count=1)
    with pytest.raises(
        adapter.FactorGovernanceFormalCatalogAdapterV4_1Error,
        match="input binding drift",
    ):
        _validate(artifact, other_inputs)


def test_build_and_validate_do_not_import_or_invoke_guarded_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    inputs = _synthetic_inputs(base_count=2, new_count=1)
    forbidden_roots = {
        "pandas",
        "pyarrow",
        "statistics",
        "scripts",
        "tushare",
        "yfinance",
    }
    forbidden_fragments = (
        "market.data",
        "market.provider",
        "fundamental",
        "mine_quant_branch_factors",
        "build_factor_v4_pre_admission_report",
    )
    observed: list[str] = []
    real_import = builtins.__import__
    guarded_modules_before = {
        name
        for name in sys.modules
        if name.startswith("scripts.mine_quant_branch_factors")
    }

    def guarded_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        observed.append(name)
        root = name.split(".", 1)[0]
        if root in forbidden_roots or any(
            fragment in name for fragment in forbidden_fragments
        ):
            raise AssertionError(f"guarded import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())

    artifact = _build(inputs)
    assert _validate(artifact, inputs) == artifact

    assert list(tmp_path.iterdir()) == before
    assert not any(
        name.split(".", 1)[0] in forbidden_roots
        or any(fragment in name for fragment in forbidden_fragments)
        for name in observed
    )
    source = inspect.getsource(adapter)
    assert "MiningCandidate" not in source
    assert "from scripts" not in source
    assert "import pandas" not in source
    assert "import pyarrow" not in source
    assert "import statistics" not in source
    guarded_modules_after = {
        name
        for name in sys.modules
        if name.startswith("scripts.mine_quant_branch_factors")
    }
    assert guarded_modules_after == guarded_modules_before


def test_public_filename_and_api_are_stable() -> None:
    assert adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME == (
        "formal_catalog_adapter_validation.v4_1.json"
    )
    assert adapter.build_formal_catalog_adapter_v4_1 is (
        adapter.build_formal_catalog_adapter_validation_v4_1
    )
    assert adapter.validate_formal_catalog_adapter_v4_1 is (
        adapter.validate_formal_catalog_adapter_validation_v4_1
    )
    assert callable(adapter.build_formal_catalog_adapter_validation_v4_1)
    assert callable(adapter.validate_formal_catalog_adapter_validation_v4_1)
    assert callable(adapter.validate_formal_catalog_candidate_descriptor_v4_1)


def test_real_267_materializer_output_is_schema_readable_when_fixture_exists() -> None:
    repository = Path(__file__).resolve().parents[2]
    discovery_root = (
        repository
        / "reports/factor_governance/private/v4_1_cycle"
        / "factor_v4_1_discovery_20260718T170345Z"
    )
    base_root = (
        repository
        / "reports/factor_governance/private/v4_pre_admission"
        / "factor_v4_pre_admission_20260718_083224"
    )
    if not discovery_root.is_dir() or not base_root.is_dir():
        pytest.skip("owner-private real 267 materializer fixture is unavailable")

    from quant_investor.factors import (
        governance_discovery_v4_1 as discovery,
    )
    from quant_investor.factors import (
        governance_formal_catalog_materialization_v4_1 as materializer,
    )

    def read_object(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(value, dict)
        return value

    discovery_values = {
        filename: read_object(discovery_root / filename)
        for filename in discovery.CANONICAL_ARTIFACT_FILENAMES
    }
    base_ontology = read_object(base_root / "primitive_ontology.v4.json")
    base_catalog = read_object(base_root / "candidate_catalog.v4.json")
    source_bindings = materializer.build_formal_catalog_source_bindings_v4_1(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    code_bindings = []
    for suffix in materializer.REQUIRED_CODE_BINDING_SUFFIXES:
        path = repository / suffix
        raw = path.read_bytes()
        code_bindings.append(
            {
                "absolute_path": str(path),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    artifacts = materializer.build_formal_catalog_materialization_v4_1(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
    )
    formal_ontology = artifacts[materializer.FORMAL_ONTOLOGY_FILENAME]
    formal_catalog = artifacts[materializer.FORMAL_CATALOG_FILENAME]
    mapping_proof = artifacts[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]

    validation = adapter.build_formal_catalog_adapter_validation_v4_1(
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=formal_ontology,
        catalog=formal_catalog,
        mapping_proof=mapping_proof,
    )

    assert validation["candidate_count"] == 267
    assert validation["base_candidate_count"] == 230
    assert validation["new_candidate_count"] == 37
    assert validation["descriptor_count"] == 267
    assert all(row["executable"] is False for row in validation["descriptors"])
    assert adapter.validate_formal_catalog_adapter_validation_v4_1(
        validation,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=formal_ontology,
        catalog=formal_catalog,
        mapping_proof=mapping_proof,
    ) == validation

    rebound_artifacts = materializer.build_formal_catalog_materialization_v4_1(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=validation,
    )
    binding = rebound_artifacts[
        materializer.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME
    ]["adapter_validation_binding"]
    assert binding == {
        "schema_version": validation["schema_version"],
        "validation_semantic_sha256": validation["validation_semantic_sha256"],
    }
