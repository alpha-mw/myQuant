from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import governance_discovery_v4_1 as discovery
from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter
from quant_investor.factors import (
    governance_formal_catalog_materialization_v4_1 as materializer,
)
from quant_investor.factors.governance_screening_v4 import (
    build_candidate_catalog_v4,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = REPO_ROOT / (
    "reports/factor_governance/private/v4_1_cycle/"
    "factor_v4_1_discovery_20260718T170345Z"
)
BASE_ROOT = REPO_ROOT / (
    "reports/factor_governance/private/v4_pre_admission/"
    "factor_v4_pre_admission_20260718_083224"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_inputs() -> dict[str, Any]:
    required_private_paths = [
        *(DISCOVERY_ROOT / filename for filename in discovery.CANONICAL_ARTIFACT_FILENAMES),
        BASE_ROOT / "primitive_ontology.v4.json",
        BASE_ROOT / "candidate_catalog.v4.json",
    ]
    if not DISCOVERY_ROOT.is_dir() or not BASE_ROOT.is_dir() or any(
        not path.is_file() for path in required_private_paths
    ):
        pytest.skip("owner-private real formal-catalog fixture is unavailable")
    discovery_values = {
        filename: _read_json(DISCOVERY_ROOT / filename)
        for filename in discovery.CANONICAL_ARTIFACT_FILENAMES
    }
    base_ontology = _read_json(BASE_ROOT / "primitive_ontology.v4.json")
    base_catalog = _read_json(BASE_ROOT / "candidate_catalog.v4.json")
    source_bindings = materializer.build_formal_catalog_source_bindings_v4_1(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    code_bindings = []
    for suffix in materializer.REQUIRED_CODE_BINDING_SUFFIXES:
        path = REPO_ROOT / suffix
        assert path.is_file()
        raw = path.read_bytes()
        code_bindings.append(
            {
                "absolute_path": str(path),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return {
        "discovery_values": discovery_values,
        "base_ontology": base_ontology,
        "base_catalog": base_catalog,
        "source_bindings": source_bindings,
        "code_bindings": code_bindings,
    }


@pytest.fixture(scope="module")
def real_materialization(real_inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return materializer.build_formal_catalog_materialization_v4_1(**real_inputs)


def _reseal(payload: dict[str, Any], field: str) -> None:
    payload.pop(field, None)
    payload[field] = materializer.semantic_sha256_v4_1(payload)


def _candidate_inputs(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "name",
        "implementation",
        "expression",
        "direction",
        "params",
        "lookback",
        "slot",
        "input_fields",
        "primitive_ids",
    )
    return [{field: copy.deepcopy(row[field]) for field in fields} for row in catalog["candidates"]]


def _synthetic_source_accounting() -> tuple[dict[str, Any], dict[str, Any]]:
    ideas: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for index in range(37):
        candidate_id = f"synthetic-new-{index:03d}"
        ideas.append(
            {
                "candidate_id": candidate_id,
                "compatibility_status": "compatible",
                "catalog_role": "new_candidate",
                "selected": True,
            }
        )
        members.append({"candidate_id": candidate_id, "origin": "aquant"})
    for index in range(6):
        candidate_id = f"synthetic-alias-{index:03d}"
        ideas.append(
            {
                "candidate_id": candidate_id,
                "compatibility_status": "compatible",
                "catalog_role": "structural_alias",
                "selected": False,
            }
        )
        members.append({"candidate_id": candidate_id, "origin": "aquant"})
    for index in range(57):
        ideas.append(
            {
                "candidate_id": f"synthetic-incompatible-{index:03d}",
                "compatibility_status": "incompatible",
                "catalog_role": "incompatible",
                "selected": False,
            }
        )
    return {"ideas": ideas}, {"members": members}


def test_synthetic_source_accounting_is_exactly_100_equals_37_plus_6_plus_57() -> None:
    audit, catalog = _synthetic_source_accounting()

    new_rows, alias_rows, incompatible_rows = materializer._assert_exact_counts(
        audit=audit,
        discovery_catalog=catalog,
    )

    assert (len(new_rows), len(alias_rows), len(incompatible_rows)) == (37, 6, 57)
    accounted_ids = {
        row["candidate_id"]
        for row in [*new_rows, *alias_rows, *incompatible_rows]
    }
    assert len(accounted_ids) == 100


def test_synthetic_source_accounting_rejects_duplicate_candidate_id() -> None:
    audit, catalog = _synthetic_source_accounting()
    audit["ideas"][-1]["candidate_id"] = audit["ideas"][0]["candidate_id"]

    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="source candidate ids must be unique",
    ):
        materializer._assert_exact_counts(audit=audit, discovery_catalog=catalog)


def test_synthetic_source_accounting_rejects_misclassified_candidate() -> None:
    audit, catalog = _synthetic_source_accounting()
    audit["ideas"][0]["selected"] = False

    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match=r"exactly 100=37\+6\+57",
    ):
        materializer._assert_exact_counts(audit=audit, discovery_catalog=catalog)


def test_real_fixture_materializes_exact_100_13_230_to_18_267(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    assert tuple(real_materialization) == materializer.FORMAL_CATALOG_MATERIALIZATION_FILENAMES
    normalized = materializer.validate_formal_catalog_materialization_v4_1(
        real_materialization, **real_inputs
    )
    assert normalized == real_materialization
    assert tuple(row["binding_id"] for row in real_inputs["source_bindings"]) == (
        materializer.REQUIRED_SOURCE_BINDING_IDS
    )
    assert {
        row["absolute_path"].split(str(REPO_ROOT) + "/", 1)[-1]
        for row in real_inputs["code_bindings"]
    } == set(materializer.REQUIRED_CODE_BINDING_SUFFIXES)

    ontology = real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME]
    catalog = real_materialization[materializer.FORMAL_CATALOG_FILENAME]
    proof = real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    assert len(ontology["primitives"]) == 18
    assert len(catalog["candidates"]) == 267
    assert (
        proof["source_candidate_count"],
        proof["new_candidate_count"],
        proof["structural_alias_count"],
        proof["incompatible_count"],
    ) == (100, 37, 6, 57)
    assert (
        len(proof["new_candidate_mappings"]),
        len(proof["structural_aliases"]),
        len(proof["incompatible_candidates"]),
    ) == (37, 6, 57)

    base_primitives = {
        row["primitive_id"]: row for row in real_inputs["base_ontology"]["primitives"]
    }
    formal_primitives = {row["primitive_id"]: row for row in ontology["primitives"]}
    assert {name: formal_primitives[name] for name in base_primitives} == base_primitives
    assert set(formal_primitives) - set(base_primitives) == set(materializer.NEW_PRIMITIVES)
    assert all(formal_primitives[name]["family"] == name for name in materializer.NEW_PRIMITIVES)

    base_by_name = {
        row["name"]: row for row in real_inputs["base_catalog"]["candidates"]
    }
    formal_by_name = {row["name"]: row for row in catalog["candidates"]}
    assert {name: formal_by_name[name] for name in base_by_name} == base_by_name

    audit = real_inputs["discovery_values"][discovery.SOURCE_IDEA_AUDIT_FILENAME]
    selected = {
        row["name"]: row
        for row in audit["ideas"]
        if row["catalog_role"] == "new_candidate"
    }
    aliases = {
        row["name"]
        for row in audit["ideas"]
        if row["catalog_role"] == "structural_alias"
    }
    incompatible = {
        row["name"]
        for row in audit["ideas"]
        if row["catalog_role"] == "incompatible"
    }
    assert set(formal_by_name) - set(base_by_name) == set(selected)
    assert aliases.isdisjoint(formal_by_name)
    assert incompatible.isdisjoint(formal_by_name)
    for name, source in selected.items():
        row = formal_by_name[name]
        assert row["implementation"] == materializer.CLASSIFICATION_IMPLEMENTATION
        assert row["expression"] == source["expression"]
        assert row["input_fields"] == source["input_fields"]
        assert row["params"] == {}
        assert row["slot"] == "primitive:" + "+".join(row["primitive_ids"])


def test_real_proof_binds_occurrences_alias_definitions_and_exact_reasons(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    proof = real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    audit = real_inputs["discovery_values"][discovery.SOURCE_IDEA_AUDIT_FILENAME]
    audit_by_id = {row["candidate_id"]: row for row in audit["ideas"]}
    base_by_name = {
        row["name"]: row for row in real_inputs["base_catalog"]["candidates"]
    }

    all_source_ids = sorted(audit_by_id)
    assert proof["source_candidate_ids_sha256"] == materializer.semantic_sha256_v4_1(
        all_source_ids
    )
    mapping_ids = {row["candidate_id"] for row in proof["new_candidate_mappings"]}
    alias_ids = {row["candidate_id"] for row in proof["structural_aliases"]}
    incompatible_ids = {
        row["candidate_id"] for row in proof["incompatible_candidates"]
    }
    assert mapping_ids | alias_ids | incompatible_ids == set(all_source_ids)
    assert not (mapping_ids & alias_ids or mapping_ids & incompatible_ids or alias_ids & incompatible_ids)

    for row in proof["new_candidate_mappings"]:
        assert row["mapping_status"] == materializer.MAPPING_STATUS_COMPLETE
        assert row["occurrence_count"] == len(row["occurrences"])
        assert row["occurrences"]
        assert [item["node_occurrence_index"] for item in row["occurrences"]] == sorted(
            item["node_occurrence_index"] for item in row["occurrences"]
        )
        for occurrence in row["occurrences"]:
            assert occurrence["consumed_identifiers"]
            assert [
                item["node_occurrence_index"]
                for item in occurrence["consumed_identifiers"]
            ] == sorted(
                item["node_occurrence_index"]
                for item in occurrence["consumed_identifiers"]
            )

    targets = [row["target_candidate_id"] for row in proof["structural_aliases"]]
    assert len(targets) == len(set(targets)) == 6
    for row in proof["structural_aliases"]:
        assert row["target_match_cardinality"] == 1
        assert row["excluded_from_catalog"] is True
        assert row["target_definition_sha256"] == base_by_name[row["target_name"]][
            "definition_sha256"
        ]

    for row in proof["incompatible_candidates"]:
        source = audit_by_id[row["candidate_id"]]
        assert row["incompatibility_reasons"] == source["incompatibility_reasons"]
        assert [item["reason"] for item in row["reason_records"]] == source[
            "incompatibility_reasons"
        ]
        assert [item["code"] for item in row["reason_records"]] == [
            reason.split(":", 1)[0] for reason in source["incompatibility_reasons"]
        ]


def test_amount_rule_explicitly_disclaims_runtime_semantics_and_binds_base(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    policy = real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME]
    rule = next(
        row for row in policy["rules"] if row["rule_id"] == "identity.amount_to_traded_amount.v1"
    )
    assert rule["identifier"] == "amount"
    assert rule["primitive_id"] == rule["family"] == "traded_amount"
    assert rule["base_semantics_binding"] == {
        "base_ontology_sha256": real_inputs["base_ontology"]["semantic_sha256"],
        "primitive_id": "traded_amount",
        "family": "traded_amount",
        "primitive_semantic_sha256": materializer.semantic_sha256_v4_1(
            {"primitive_id": "traded_amount", "family": "traded_amount"}
        ),
    }
    for field in (
        "unit_equivalence_status",
        "scale_equivalence_status",
        "missing_value_equivalence_status",
        "zero_value_equivalence_status",
        "adjustment_equivalence_status",
    ):
        assert rule[field] == "not_runtime_verified"
    assert rule["classification_only"] is True
    assert rule["runtime_equivalence_claimed"] is False


def test_duplicate_identical_subtrees_are_preserved_per_occurrence(
    real_inputs: dict[str, Any],
) -> None:
    policy = materializer.build_primitive_mapping_policy_v4_1(
        base_ontology=real_inputs["base_ontology"]
    )
    result = materializer.build_candidate_primitive_proof_v4_1(
        expression="amount + amount",
        input_fields=["amount"],
        mapping_policy=policy,
    )
    assert result["primitive_ids"] == ["traded_amount"]
    assert result["occurrence_count"] == 2
    assert [row["node_path"] for row in result["occurrences"]] == [
        "root.left",
        "root.right",
    ]
    assert len({row["node_occurrence_index"] for row in result["occurrences"]}) == 2
    assert len({row["subtree_sha256"] for row in result["occurrences"]}) == 1
    assert [row["rule_occurrence_index"] for row in result["occurrences"]] == [0, 1]
    assert {row["match_cardinality"] for row in result["occurrences"]} == {2}


def test_ambiguous_same_node_rules_fail_the_candidate(
    real_inputs: dict[str, Any],
) -> None:
    policy = materializer.build_primitive_mapping_policy_v4_1(
        base_ontology=real_inputs["base_ontology"]
    )
    rules = copy.deepcopy(policy["rules"])
    duplicate = copy.deepcopy(
        next(row for row in rules if row["rule_id"] == "identity.amount_to_traded_amount.v1")
    )
    duplicate["rule_id"] = "synthetic.ambiguous.amount.v1"
    _reseal(duplicate, "rule_semantic_sha256")
    rules.append(duplicate)
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="ambiguous primitive mapping",
    ):
        materializer.classify_normalized_ast_occurrences_v4_1(
            normalized_ast=discovery.normalize_expression_ast_v4_1("amount"),
            input_fields=["amount"],
            rules=rules,
        )


def test_unconsumed_identifier_fails_without_partial_mapping(
    real_inputs: dict[str, Any],
) -> None:
    policy = materializer.build_primitive_mapping_policy_v4_1(
        base_ontology=real_inputs["base_ontology"]
    )
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="unconsumed data identifier occurrences",
    ):
        materializer.build_candidate_primitive_proof_v4_1(
            expression="amount + close",
            input_fields=["amount", "close"],
            mapping_policy=policy,
        )


def test_ancestor_descendant_rule_overlap_fails_closed(
    real_inputs: dict[str, Any],
) -> None:
    policy = materializer.build_primitive_mapping_policy_v4_1(
        base_ontology=real_inputs["base_ontology"]
    )
    rules = copy.deepcopy(policy["rules"])
    close_rule = copy.deepcopy(
        next(row for row in rules if row["rule_id"] == "identity.amount_to_traded_amount.v1")
    )
    close_rule.update(
        {
            "rule_id": "synthetic.overlap.close.v1",
            "identifier": "close",
            "primitive_id": "close_return",
            "family": "close_return",
            "consumed_identifier_multiset": ["close"],
            "base_semantics_binding": None,
        }
    )
    _reseal(close_rule, "rule_semantic_sha256")
    rules.append(close_rule)
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="overlapping primitive mapping",
    ):
        materializer.classify_normalized_ast_occurrences_v4_1(
            normalized_ast=discovery.normalize_expression_ast_v4_1(
                "(vwap - close) / close"
            ),
            input_fields=["close", "vwap"],
            rules=rules,
        )


def test_source_family_cannot_drive_primitive_or_formal_family(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    audit = real_inputs["discovery_values"][discovery.SOURCE_IDEA_AUDIT_FILENAME]
    source = next(row for row in audit["ideas"] if row["name"] == "alpha_amount_expansion_5_20")
    mutated = copy.deepcopy(source)
    mutated["source_family"] = "hostile_unreviewed_family"
    policy = real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME]
    before = materializer.build_candidate_primitive_proof_v4_1(
        expression=source["expression"],
        input_fields=source["input_fields"],
        mapping_policy=policy,
    )
    after = materializer.build_candidate_primitive_proof_v4_1(
        expression=mutated["expression"],
        input_fields=mutated["input_fields"],
        mapping_policy=policy,
    )
    assert before == after
    assert materializer.canonical_file_bytes_v4_1(before) == materializer.canonical_file_bytes_v4_1(
        after
    )
    assert materializer.semantic_sha256_v4_1(before) == materializer.semantic_sha256_v4_1(
        after
    )
    catalog_row = next(
        row
        for row in real_materialization[materializer.FORMAL_CATALOG_FILENAME]["candidates"]
        if row["name"] == source["name"]
    )
    assert catalog_row["primitive_ids"] == ["traded_amount"]
    assert catalog_row["family"] == "traded_amount"
    assert catalog_row["family"] not in {source["source_family"], mutated["source_family"]}


def test_alias_definition_drift_is_rejected_by_full_recomputation(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    proof = copy.deepcopy(
        real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    )
    proof["structural_aliases"][0]["target_definition_sha256"] = "0" * 64
    _reseal(proof["structural_aliases"][0], "alias_semantic_sha256")
    _reseal(proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="differs from exact recomputation",
    ):
        materializer.validate_primitive_mapping_proof_v4_1(
            proof,
            discovery_values=real_inputs["discovery_values"],
            base_ontology=real_inputs["base_ontology"],
            base_catalog=real_inputs["base_catalog"],
            formal_ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
            formal_catalog=real_materialization[materializer.FORMAL_CATALOG_FILENAME],
            mapping_policy=real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME],
        )


def test_incompatible_loss_is_rejected_even_if_proof_is_resealed(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    proof = copy.deepcopy(
        real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    )
    proof["incompatible_candidates"].pop()
    proof["incompatible_count"] = 56
    _reseal(proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="differs from exact recomputation",
    ):
        materializer.validate_primitive_mapping_proof_v4_1(
            proof,
            discovery_values=real_inputs["discovery_values"],
            base_ontology=real_inputs["base_ontology"],
            base_catalog=real_inputs["base_catalog"],
            formal_ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
            formal_catalog=real_materialization[materializer.FORMAL_CATALOG_FILENAME],
            mapping_policy=real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME],
        )


def test_valid_but_different_base_catalog_is_rejected_as_base_drift(
    real_inputs: dict[str, Any],
) -> None:
    definitions = _candidate_inputs(real_inputs["base_catalog"])
    definitions[0]["lookback"] += 1
    drifted = build_candidate_catalog_v4(
        ontology=real_inputs["base_ontology"], candidates=definitions
    )
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="source bundle validation failed",
    ):
        materializer.build_formal_catalog_materialization_v4_1(
            **{**real_inputs, "base_catalog": drifted}
        )


def test_standalone_proof_validator_rejects_remove_base_add_extra_catalog(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    formal_catalog = real_materialization[materializer.FORMAL_CATALOG_FILENAME]
    definitions = _candidate_inputs(formal_catalog)
    removed = next(
        row for row in definitions if row["name"] == "builtin_short_term_return_20d"
    )
    definitions.remove(removed)
    extra = copy.deepcopy(removed)
    extra["name"] = "zz_evil_extra"
    definitions.append(extra)
    drifted_catalog = build_candidate_catalog_v4(
        ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
        candidates=definitions,
    )
    assert len(drifted_catalog["candidates"]) == 267
    proof = copy.deepcopy(
        real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    )
    proof["formal_catalog_sha256"] = drifted_catalog["semantic_sha256"]
    _reseal(proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="base definition drifted",
    ):
        materializer.validate_primitive_mapping_proof_v4_1(
            proof,
            discovery_values=real_inputs["discovery_values"],
            base_ontology=real_inputs["base_ontology"],
            base_catalog=real_inputs["base_catalog"],
            formal_ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
            formal_catalog=drifted_catalog,
            mapping_policy=real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME],
        )


def test_unknown_proof_field_and_required_source_binding_drift_fail_closed(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    proof = copy.deepcopy(
        real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    )
    proof["unexpected"] = True
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="unknown=unexpected",
    ):
        materializer.validate_primitive_mapping_proof_v4_1(
            proof,
            discovery_values=real_inputs["discovery_values"],
            base_ontology=real_inputs["base_ontology"],
            base_catalog=real_inputs["base_catalog"],
            formal_ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
            formal_catalog=real_materialization[materializer.FORMAL_CATALOG_FILENAME],
            mapping_policy=real_materialization[materializer.PRIMITIVE_MAPPING_POLICY_FILENAME],
        )

    drifted_bindings = copy.deepcopy(real_inputs["source_bindings"])
    next(
        row for row in drifted_bindings if row["binding_id"] == "source_idea_audit"
    )["byte_sha256"] = "0" * 64
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="required materialization source binding mismatch",
    ):
        materializer.build_formal_catalog_materialization_v4_1(
            **{**real_inputs, "source_bindings": drifted_bindings}
        )

    missing_source = copy.deepcopy(real_inputs["source_bindings"][:-1])
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match=r"exact base2\+Discovery8 inventory",
    ):
        materializer.build_formal_catalog_materialization_v4_1(
            **{**real_inputs, "source_bindings": missing_source}
        )

    missing_code = copy.deepcopy(real_inputs["code_bindings"][:-1])
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="exact formal code set",
    ):
        materializer.build_formal_catalog_materialization_v4_1(
            **{**real_inputs, "code_bindings": missing_code}
        )

    nonnormalized_code = copy.deepcopy(real_inputs["code_bindings"])
    nonnormalized_code[0]["absolute_path"] = nonnormalized_code[0][
        "absolute_path"
    ].replace("/quant_investor/", "/./quant_investor/", 1)
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="normalized absolute paths",
    ):
        materializer.build_formal_catalog_materialization_v4_1(
            **{**real_inputs, "code_bindings": nonnormalized_code}
        )


def test_manifest_scope_is_pure_build_only_and_can_be_rebuilt_for_adapter(
    real_inputs: dict[str, Any],
    real_materialization: dict[str, dict[str, Any]],
) -> None:
    manifest = real_materialization[
        materializer.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME
    ]
    assert manifest["adapter_validation_status"] == "not_bound"
    assert manifest["adapter_validation_binding"] is None
    assert manifest["side_effect_scope"] == "pure_materializer_build_only"
    assert set(manifest["side_effects"].values()) == {False}
    for field in (
        "classification_only",
        "runtime_equivalence_claimed",
        "screening_eligible",
        "proposal_eligible",
        "registry_entry_created",
        "formal_admission_authority",
        "production_apply_enabled",
        "new_risk_authorized",
    ):
        assert manifest[field] is (field == "classification_only")
    assert manifest["initial_weight_policy"] == "zero_only"

    proof = real_materialization[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    adapter_validation = adapter.build_formal_catalog_adapter_validation_v4_1(
        base_ontology=real_inputs["base_ontology"],
        base_catalog=real_inputs["base_catalog"],
        ontology=real_materialization[materializer.FORMAL_ONTOLOGY_FILENAME],
        catalog=real_materialization[materializer.FORMAL_CATALOG_FILENAME],
        mapping_proof=proof,
    )
    core_four = {
        filename: real_materialization[filename]
        for filename in materializer.FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1]
    }
    rebuilt = materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
        artifacts=core_four,
        cycle_id=proof["cycle_id"],
        source_bindings=real_inputs["source_bindings"],
        code_bindings=real_inputs["code_bindings"],
        adapter_validation=adapter_validation,
    )
    assert rebuilt["adapter_validation_status"] == "bound"
    assert rebuilt["adapter_validation_binding"] == {
        "schema_version": adapter_validation["schema_version"],
        "validation_semantic_sha256": adapter_validation[
            "validation_semantic_sha256"
        ],
    }

    forged = copy.deepcopy(adapter_validation)
    forged["classification_only"] = False
    _reseal(forged, "validation_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="differs from exact validate-only recomputation",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=core_four,
            cycle_id=proof["cycle_id"],
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
            adapter_validation=forged,
        )

    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="cycle_id must equal",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=core_four,
            cycle_id="wrong_cycle",
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
            adapter_validation=adapter_validation,
        )
    wrong_cycle_manifest = copy.deepcopy(rebuilt)
    wrong_cycle_manifest["cycle_id"] = "wrong_cycle"
    _reseal(wrong_cycle_manifest, "manifest_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="cycle_id must equal",
    ):
        materializer.validate_formal_catalog_materialization_manifest_v4_1(
            wrong_cycle_manifest,
            artifacts=core_four,
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
            adapter_validation=adapter_validation,
        )

    stale_core = copy.deepcopy(core_four)
    stale_core[materializer.FORMAL_CATALOG_FILENAME]["candidates"].pop()
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="formal catalog validation failed",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=stale_core,
            cycle_id=proof["cycle_id"],
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
            adapter_validation=adapter_validation,
        )

    source_link_drift = copy.deepcopy(core_four)
    source_link_proof = source_link_drift[
        materializer.PRIMITIVE_MAPPING_PROOF_FILENAME
    ]
    source_link_proof["source_idea_audit_sha256"] = "0" * 64
    _reseal(source_link_proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="proof/source binding mismatch: source_idea_audit_sha256",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=source_link_drift,
            cycle_id=proof["cycle_id"],
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
        )

    occurrence_drift = copy.deepcopy(core_four)
    occurrence_drift_proof = occurrence_drift[
        materializer.PRIMITIVE_MAPPING_PROOF_FILENAME
    ]
    occurrence_drift_row = occurrence_drift_proof["new_candidate_mappings"][0]
    occurrence_drift_row["occurrences"][0]["node_path"] = "root.evil"
    _reseal(occurrence_drift_row, "mapping_semantic_sha256")
    _reseal(occurrence_drift_proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="mapping AST proof differs from exact recomputation",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=occurrence_drift,
            cycle_id=proof["cycle_id"],
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
        )

    base_drift = copy.deepcopy(core_four)
    drifted_definitions = _candidate_inputs(
        base_drift[materializer.FORMAL_CATALOG_FILENAME]
    )
    new_names = {row["name"] for row in proof["new_candidate_mappings"]}
    drifted_base_row = next(
        row for row in drifted_definitions if row["name"] not in new_names
    )
    drifted_base_row["lookback"] += 1
    drifted_formal_catalog = build_candidate_catalog_v4(
        ontology=base_drift[materializer.FORMAL_ONTOLOGY_FILENAME],
        candidates=drifted_definitions,
    )
    base_drift[materializer.FORMAL_CATALOG_FILENAME] = drifted_formal_catalog
    base_drift_proof = base_drift[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME]
    base_drift_proof["formal_catalog_sha256"] = drifted_formal_catalog[
        "semantic_sha256"
    ]
    _reseal(base_drift_proof, "proof_semantic_sha256")
    with pytest.raises(
        materializer.FactorGovernanceFormalCatalogMaterializationV4_1Error,
        match="does not preserve the proof-bound base catalog",
    ):
        materializer.rebuild_formal_catalog_materialization_manifest_v4_1(
            artifacts=base_drift,
            cycle_id=proof["cycle_id"],
            source_bindings=real_inputs["source_bindings"],
            code_bindings=real_inputs["code_bindings"],
        )


def test_materializer_uses_no_filesystem_during_pure_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    real_inputs: dict[str, Any],
) -> None:
    def forbidden_open(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("pure materializer attempted filesystem access")

    monkeypatch.setattr("builtins.open", forbidden_open)
    result = materializer.build_formal_catalog_materialization_v4_1(**real_inputs)
    assert len(result[materializer.FORMAL_CATALOG_FILENAME]["candidates"]) == 267
