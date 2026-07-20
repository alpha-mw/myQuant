"""Pure validate-only adapter for a FactorGovernance v4.1 formal catalog.

The adapter accepts only caller-supplied JSON-like mappings.  It proves that
the formal ontology/catalog remain readable by the v4 schema validators, that
the base ontology/catalog are preserved exactly, and that every added catalog
definition is bound to one classification-only mapping proof row.  Its proof
input is required to have been source-recomputed by the materializer bundle
validator: this module checks the proof's exact fields, hashes, accounting, and
catalog bindings but intentionally has no DISCOVERY or mapping-policy input and
does not independently authenticate source definitions.  It does not construct
a runtime factor object and grants no screening or execution right.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors.governance_screening_v4 import (
    FactorGovernanceScreeningV4Error,
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
)


PROTOCOL_VERSION = "v4.1"
PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION = (
    "factor-governance-primitive-mapping-proof.v4.1"
)
FORMAL_CATALOG_ADAPTER_VALIDATION_SCHEMA_VERSION = (
    "factor-governance-formal-catalog-adapter-validation.v4.1"
)
FORMAL_CATALOG_CANDIDATE_DESCRIPTOR_SCHEMA_VERSION = (
    "factor-governance-formal-catalog-candidate-descriptor.v4.1"
)
FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME = (
    "formal_catalog_adapter_validation.v4_1.json"
)

CLASSIFICATION_IDENTITY_IMPLEMENTATION = "aquant_expression_ast.v1"
NON_EXECUTABLE_DESCRIPTOR_IMPLEMENTATION = (
    "formal_catalog_definition_reference.validate_only.v4.1"
)

_SHA256_HEX = frozenset("0123456789abcdef")

_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "source_idea_audit_sha256",
        "discovery_catalog_sha256",
        "base_ontology_sha256",
        "base_catalog_sha256",
        "mapping_policy_sha256",
        "formal_ontology_sha256",
        "formal_catalog_sha256",
        "source_candidate_count",
        "new_candidate_count",
        "structural_alias_count",
        "incompatible_count",
        "catalog_base_candidate_count",
        "catalog_new_candidate_count",
        "catalog_total_candidate_count",
        "source_candidate_ids_sha256",
        "new_candidate_mappings",
        "structural_aliases",
        "incompatible_candidates",
        "classification_only",
        "runtime_equivalence_claimed",
        "screening_eligible",
        "proposal_eligible",
        "registry_entry_created",
        "initial_weight_policy",
        "formal_admission_authority",
        "production_apply_enabled",
        "proof_semantic_sha256",
    }
)
_NEW_MAPPING_FIELDS = frozenset(
    {
        "candidate_id",
        "name",
        "source_definition_sha256",
        "catalog_definition_sha256",
        "implementation",
        "expression",
        "full_candidate_normalized_ast_sha256",
        "input_fields",
        "primitive_ids",
        "family",
        "slot",
        "occurrence_count",
        "occurrences",
        "mapping_status",
        "mapping_semantic_sha256",
    }
)
_OCCURRENCE_FIELDS = frozenset(
    {
        "node_path",
        "node_occurrence_index",
        "rule_occurrence_index",
        "subtree_sha256",
        "rule_id",
        "primitive_id",
        "family",
        "match_cardinality",
        "consumed_identifiers",
    }
)
_CONSUMED_IDENTIFIER_FIELDS = frozenset(
    {"identifier", "node_path", "node_occurrence_index"}
)
_ALIAS_FIELDS = frozenset(
    {
        "candidate_id",
        "name",
        "source_definition_sha256",
        "structural_fingerprint_sha256",
        "target_candidate_id",
        "target_name",
        "target_definition_sha256",
        "target_match_cardinality",
        "excluded_from_catalog",
        "alias_semantic_sha256",
    }
)
_INCOMPATIBLE_FIELDS = frozenset(
    {
        "source_index",
        "candidate_id",
        "name",
        "incompatibility_reasons",
        "reason_records",
        "excluded_from_catalog",
        "mapping_not_attempted",
        "incompatible_semantic_sha256",
    }
)
_REASON_FIELDS = frozenset({"code", "reason"})
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "catalog_role",
        "catalog_definition_sha256",
        "catalog_implementation",
        "descriptor_implementation",
        "source_candidate_id",
        "source_definition_sha256",
        "mapping_proof_row_sha256",
        "classification_only",
        "executable",
        "screening_eligible",
        "runtime_equivalence_verified",
        "blockers",
        "descriptor_semantic_sha256",
    }
)
_VALIDATION_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "base_ontology_sha256",
        "base_catalog_sha256",
        "formal_ontology_sha256",
        "formal_catalog_sha256",
        "mapping_proof_schema_version",
        "mapping_proof_sha256",
        "candidate_count",
        "base_candidate_count",
        "new_candidate_count",
        "descriptor_count",
        "ordered_candidate_names_semantic_sha256",
        "ordered_definition_pairs_semantic_sha256",
        "base_primitive_mapping_preserved",
        "base_candidate_definitions_preserved",
        "catalog_schema_loader_readable",
        "classification_only",
        "source_authenticity_recomputed",
        "signal_computability_proven",
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
        "descriptors",
        "blockers",
        "validation_semantic_sha256",
    }
)

_BASE_DESCRIPTOR_BLOCKERS = (
    "data_loader_not_invoked",
    "runtime_equivalence_not_verified",
    "screening_not_authorized",
    "signal_computability_not_proven",
    "statistics_not_invoked",
    "validate_only_descriptor",
)
_NEW_DESCRIPTOR_BLOCKERS = tuple(
    sorted(
        (
            *_BASE_DESCRIPTOR_BLOCKERS,
            "classification_identity_non_executable",
            "source_authenticity_not_recomputed_by_adapter",
        )
    )
)
_VALIDATION_BLOCKERS = (
    "data_loader_not_invoked",
    "runtime_equivalence_not_verified",
    "screening_not_authorized",
    "signal_computability_not_proven",
    "source_authenticity_not_recomputed_by_adapter",
    "statistics_not_invoked",
)


class FactorGovernanceFormalCatalogAdapterV4_1Error(ValueError):
    """Raised when formal-catalog validation cannot be proven exactly."""


FactorGovernanceFormalCatalogAdapterV41Error = (
    FactorGovernanceFormalCatalogAdapterV4_1Error
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted, finite JSON bytes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def semantic_sha256(
    value: Any,
    *,
    exclude_fields: Sequence[str] = (),
) -> str:
    """Hash canonical JSON, excluding exact top-level self-hash fields."""

    normalized = copy.deepcopy(value)
    if exclude_fields:
        if not isinstance(normalized, Mapping):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                "exclude_fields requires a top-level object"
            )
        normalized = dict(normalized)
        seen: set[str] = set()
        for field in exclude_fields:
            if type(field) is not str or not field or field in seen:
                raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                    "exclude_fields must contain distinct non-empty strings"
                )
            seen.add(field)
            normalized.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return semantic_sha256(value, exclude_fields=(field,))


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result[field] = _self_hash(result, field)
    return result


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be an object"
        )
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} fields invalid: missing={missing}; unknown={unknown}"
        )
    return payload


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be an exact non-empty string"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be a positive integer"
        )
    return result


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be a list"
        )
    return list(value)


def _sorted_distinct_texts(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    items = [_text(item, f"{label}[]") for item in _list(value, label)]
    if not allow_empty and not items:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must not be empty"
        )
    if items != sorted(items) or len(items) != len(set(items)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be sorted and distinct"
        )
    return items


def _require_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} must be {str(expected).lower()}"
        )
    return expected


def _validate_self_hash(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    observed = _sha(payload[field], f"{label}.{field}")
    if observed != _self_hash(payload, field):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} semantic SHA mismatch"
        )
    return observed


def _validate_consumed_identifier(value: Any, label: str) -> dict[str, Any]:
    row = _exact(value, _CONSUMED_IDENTIFIER_FIELDS, label)
    return {
        "identifier": _text(row["identifier"], f"{label}.identifier"),
        "node_path": _text(row["node_path"], f"{label}.node_path"),
        "node_occurrence_index": _nonnegative_int(
            row["node_occurrence_index"], f"{label}.node_occurrence_index"
        ),
    }


def _validate_occurrence(
    value: Any,
    *,
    label: str,
    ontology_families: Mapping[str, str],
) -> dict[str, Any]:
    row = _exact(value, _OCCURRENCE_FIELDS, label)
    primitive_id = _text(row["primitive_id"], f"{label}.primitive_id")
    if primitive_id not in ontology_families:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} uses unknown primitive: {primitive_id}"
        )
    family = _text(row["family"], f"{label}.family")
    if family != ontology_families[primitive_id]:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} primitive family mismatch"
        )
    match_cardinality = _positive_int(
        row["match_cardinality"], f"{label}.match_cardinality"
    )
    identifiers = [
        _validate_consumed_identifier(item, f"{label}.consumed_identifiers[{index}]")
        for index, item in enumerate(
            _list(row["consumed_identifiers"], f"{label}.consumed_identifiers")
        )
    ]
    if not identifiers:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.consumed_identifiers must not be empty"
        )
    identifier_occurrence_indexes = [
        item["node_occurrence_index"] for item in identifiers
    ]
    if identifier_occurrence_indexes != sorted(identifier_occurrence_indexes) or len(
        identifier_occurrence_indexes
    ) != len(set(identifier_occurrence_indexes)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.consumed_identifiers must follow unique AST occurrence order"
        )
    return {
        "node_path": _text(row["node_path"], f"{label}.node_path"),
        "node_occurrence_index": _nonnegative_int(
            row["node_occurrence_index"], f"{label}.node_occurrence_index"
        ),
        "rule_occurrence_index": _nonnegative_int(
            row["rule_occurrence_index"], f"{label}.rule_occurrence_index"
        ),
        "subtree_sha256": _sha(row["subtree_sha256"], f"{label}.subtree_sha256"),
        "rule_id": _text(row["rule_id"], f"{label}.rule_id"),
        "primitive_id": primitive_id,
        "family": family,
        "match_cardinality": match_cardinality,
        "consumed_identifiers": identifiers,
    }


def _validate_new_mapping(
    value: Any,
    *,
    index: int,
    ontology_families: Mapping[str, str],
) -> dict[str, Any]:
    label = f"mapping_proof.new_candidate_mappings[{index}]"
    row = _exact(value, _NEW_MAPPING_FIELDS, label)
    occurrences = [
        _validate_occurrence(
            item,
            label=f"{label}.occurrences[{occurrence_index}]",
            ontology_families=ontology_families,
        )
        for occurrence_index, item in enumerate(
            _list(row["occurrences"], f"{label}.occurrences")
        )
    ]
    occurrence_count = _positive_int(
        row["occurrence_count"], f"{label}.occurrence_count"
    )
    if occurrence_count != len(occurrences):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.occurrence_count mismatch"
        )
    node_occurrence_indexes = [
        item["node_occurrence_index"] for item in occurrences
    ]
    if node_occurrence_indexes != sorted(node_occurrence_indexes) or len(
        node_occurrence_indexes
    ) != len(set(node_occurrence_indexes)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.occurrences must be ordered by unique node occurrence index"
        )
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        by_rule.setdefault(occurrence["rule_id"], []).append(occurrence)
    for rule_id, rule_occurrences in by_rule.items():
        expected_cardinality = len(rule_occurrences)
        if any(
            occurrence["match_cardinality"] != expected_cardinality
            for occurrence in rule_occurrences
        ):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"{label} rule match cardinality drift: {rule_id}"
            )
        if [
            occurrence["rule_occurrence_index"]
            for occurrence in rule_occurrences
        ] != list(range(expected_cardinality)):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"{label} rule occurrence indexes are not contiguous: {rule_id}"
            )
    primitive_ids = _sorted_distinct_texts(
        row["primitive_ids"], f"{label}.primitive_ids"
    )
    occurrence_primitive_ids = sorted(
        {item["primitive_id"] for item in occurrences}
    )
    if primitive_ids != occurrence_primitive_ids:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.primitive_ids do not match proved occurrences"
        )
    for primitive_id in primitive_ids:
        if primitive_id not in ontology_families:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"{label} uses unknown primitive: {primitive_id}"
            )
    implementation = _text(row["implementation"], f"{label}.implementation")
    if implementation != CLASSIFICATION_IDENTITY_IMPLEMENTATION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} uses unknown classification implementation"
        )
    mapping_status = _text(row["mapping_status"], f"{label}.mapping_status")
    if mapping_status != "complete_unique_occurrence_accounting":
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.mapping_status is not complete"
        )
    normalized = {
        "candidate_id": _text(row["candidate_id"], f"{label}.candidate_id"),
        "name": _text(row["name"], f"{label}.name"),
        "source_definition_sha256": _sha(
            row["source_definition_sha256"], f"{label}.source_definition_sha256"
        ),
        "catalog_definition_sha256": _sha(
            row["catalog_definition_sha256"],
            f"{label}.catalog_definition_sha256",
        ),
        "implementation": implementation,
        "expression": row["expression"],
        "full_candidate_normalized_ast_sha256": _sha(
            row["full_candidate_normalized_ast_sha256"],
            f"{label}.full_candidate_normalized_ast_sha256",
        ),
        "input_fields": _sorted_distinct_texts(
            row["input_fields"], f"{label}.input_fields"
        ),
        "primitive_ids": primitive_ids,
        "family": _text(row["family"], f"{label}.family"),
        "slot": _text(row["slot"], f"{label}.slot"),
        "occurrence_count": occurrence_count,
        "occurrences": occurrences,
        "mapping_status": mapping_status,
        "mapping_semantic_sha256": _sha(
            row["mapping_semantic_sha256"], f"{label}.mapping_semantic_sha256"
        ),
    }
    if type(normalized["expression"]) is not str:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.expression must be a string"
        )
    if normalized["mapping_semantic_sha256"] != _self_hash(
        normalized, "mapping_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} semantic SHA mismatch"
        )
    return normalized


def _validate_alias(value: Any, index: int) -> dict[str, Any]:
    label = f"mapping_proof.structural_aliases[{index}]"
    row = _exact(value, _ALIAS_FIELDS, label)
    normalized = {
        "candidate_id": _text(row["candidate_id"], f"{label}.candidate_id"),
        "name": _text(row["name"], f"{label}.name"),
        "source_definition_sha256": _sha(
            row["source_definition_sha256"], f"{label}.source_definition_sha256"
        ),
        "structural_fingerprint_sha256": _sha(
            row["structural_fingerprint_sha256"],
            f"{label}.structural_fingerprint_sha256",
        ),
        "target_candidate_id": _text(
            row["target_candidate_id"], f"{label}.target_candidate_id"
        ),
        "target_name": _text(row["target_name"], f"{label}.target_name"),
        "target_definition_sha256": _sha(
            row["target_definition_sha256"],
            f"{label}.target_definition_sha256",
        ),
        "target_match_cardinality": _positive_int(
            row["target_match_cardinality"], f"{label}.target_match_cardinality"
        ),
        "excluded_from_catalog": _require_bool(
            row["excluded_from_catalog"], True, f"{label}.excluded_from_catalog"
        ),
        "alias_semantic_sha256": _sha(
            row["alias_semantic_sha256"], f"{label}.alias_semantic_sha256"
        ),
    }
    if normalized["target_match_cardinality"] != 1:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.target_match_cardinality must be exactly 1"
        )
    if normalized["alias_semantic_sha256"] != _self_hash(
        normalized, "alias_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} semantic SHA mismatch"
        )
    return normalized


def _validate_incompatible(value: Any, index: int) -> dict[str, Any]:
    label = f"mapping_proof.incompatible_candidates[{index}]"
    row = _exact(value, _INCOMPATIBLE_FIELDS, label)
    reasons = _sorted_distinct_texts(
        row["incompatibility_reasons"], f"{label}.incompatibility_reasons"
    )
    reason_records = []
    for reason_index, raw_reason in enumerate(
        _list(row["reason_records"], f"{label}.reason_records")
    ):
        reason_label = f"{label}.reason_records[{reason_index}]"
        reason_row = _exact(raw_reason, _REASON_FIELDS, reason_label)
        reason_records.append(
            {
                "code": _text(reason_row["code"], f"{reason_label}.code"),
                "reason": _text(reason_row["reason"], f"{reason_label}.reason"),
            }
        )
    if not reason_records:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.reason_records must not be empty"
        )
    if reasons != [item["reason"] for item in reason_records]:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.reason_records do not preserve incompatibility reasons"
        )
    if any(
        item["code"] != item["reason"].split(":", 1)[0]
        for item in reason_records
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.reason_records code prefix mismatch"
        )
    reason_keys = [(item["code"], item["reason"]) for item in reason_records]
    if len(reason_keys) != len(set(reason_keys)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label}.reason_records must be distinct"
        )
    normalized = {
        "source_index": _nonnegative_int(
            row["source_index"], f"{label}.source_index"
        ),
        "candidate_id": _text(row["candidate_id"], f"{label}.candidate_id"),
        "name": _text(row["name"], f"{label}.name"),
        "incompatibility_reasons": reasons,
        "reason_records": reason_records,
        "excluded_from_catalog": _require_bool(
            row["excluded_from_catalog"], True, f"{label}.excluded_from_catalog"
        ),
        "mapping_not_attempted": _require_bool(
            row["mapping_not_attempted"], True, f"{label}.mapping_not_attempted"
        ),
        "incompatible_semantic_sha256": _sha(
            row["incompatible_semantic_sha256"],
            f"{label}.incompatible_semantic_sha256",
        ),
    }
    if normalized["incompatible_semantic_sha256"] != _self_hash(
        normalized, "incompatible_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"{label} semantic SHA mismatch"
        )
    return normalized


def _validate_mapping_proof(
    value: Mapping[str, Any],
    *,
    base_ontology_sha256: str,
    base_catalog_sha256: str,
    formal_ontology_sha256: str,
    formal_catalog_sha256: str,
    ontology_families: Mapping[str, str],
    base_candidates: Mapping[str, Mapping[str, Any]],
    formal_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _exact(value, _PROOF_FIELDS, "mapping proof")
    canonical_json_bytes(payload)
    if payload["schema_version"] != PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof schema mismatch; discovery artifacts are not accepted"
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof protocol mismatch"
        )
    bindings = {
        "base_ontology_sha256": base_ontology_sha256,
        "base_catalog_sha256": base_catalog_sha256,
        "formal_ontology_sha256": formal_ontology_sha256,
        "formal_catalog_sha256": formal_catalog_sha256,
    }
    for field, expected in bindings.items():
        if _sha(payload[field], f"mapping_proof.{field}") != expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"mapping proof binding drift: {field}"
            )
    new_rows = [
        _validate_new_mapping(
            row,
            index=index,
            ontology_families=ontology_families,
        )
        for index, row in enumerate(
            _list(payload["new_candidate_mappings"], "mapping proof new rows")
        )
    ]
    aliases = [
        _validate_alias(row, index)
        for index, row in enumerate(
            _list(payload["structural_aliases"], "mapping proof aliases")
        )
    ]
    incompatible = [
        _validate_incompatible(row, index)
        for index, row in enumerate(
            _list(
                payload["incompatible_candidates"],
                "mapping proof incompatible candidates",
            )
        )
    ]
    for label, rows in (
        ("new_candidate_mappings", new_rows),
        ("structural_aliases", aliases),
        ("incompatible_candidates", incompatible),
    ):
        keys = [(row["candidate_id"], row["name"]) for row in rows]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"mapping proof {label} must be canonically ordered and distinct"
            )
    source_ids = [
        row["candidate_id"] for row in (*new_rows, *aliases, *incompatible)
    ]
    if len(source_ids) != len(set(source_ids)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof source candidate IDs overlap across classifications"
        )
    source_names = [row["name"] for row in (*new_rows, *aliases, *incompatible)]
    if len(source_names) != len(set(source_names)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof source names overlap across classifications"
        )
    expected_source_ids_sha = semantic_sha256(sorted(source_ids))
    if _sha(
        payload["source_candidate_ids_sha256"],
        "mapping_proof.source_candidate_ids_sha256",
    ) != expected_source_ids_sha:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof source candidate ID accounting drift"
        )
    expected_new_names = sorted(set(formal_candidates) - set(base_candidates))
    mapped_names = [row["name"] for row in new_rows]
    if mapped_names != expected_new_names:
        missing = sorted(set(expected_new_names) - set(mapped_names))
        extra = sorted(set(mapped_names) - set(expected_new_names))
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"mapping proof new names drift: missing={missing}; extra={extra}"
        )
    by_name = {row["name"]: row for row in new_rows}
    for name in expected_new_names:
        candidate = formal_candidates[name]
        row = by_name[name]
        exact_links = {
            "catalog_definition_sha256": candidate["definition_sha256"],
            "implementation": candidate["implementation"],
            "expression": candidate["expression"],
            "input_fields": candidate["input_fields"],
            "primitive_ids": candidate["primitive_ids"],
            "family": candidate["family"],
            "slot": candidate["slot"],
        }
        for field, expected in exact_links.items():
            if row[field] != expected:
                raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                    f"mapping proof source/catalog drift for {name}: {field}"
                )
    for row in aliases:
        target = base_candidates.get(row["target_name"])
        if target is None:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                "mapping proof alias target is not one base candidate"
            )
        if row["target_definition_sha256"] != target["definition_sha256"]:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                "mapping proof alias target definition drift"
            )
    alias_target_ids = [row["target_candidate_id"] for row in aliases]
    if len(alias_target_ids) != len(set(alias_target_ids)):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof structural aliases do not bind distinct targets"
        )
    excluded_names = {row["name"] for row in (*aliases, *incompatible)}
    leaked_names = sorted(excluded_names & set(formal_candidates))
    if leaked_names:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"excluded source candidates leaked into formal catalog: {leaked_names}"
        )
    counts = {
        "source_candidate_count": len(source_ids),
        "new_candidate_count": len(new_rows),
        "structural_alias_count": len(aliases),
        "incompatible_count": len(incompatible),
        "catalog_base_candidate_count": len(base_candidates),
        "catalog_new_candidate_count": len(expected_new_names),
        "catalog_total_candidate_count": len(formal_candidates),
    }
    for field, expected in counts.items():
        observed = _nonnegative_int(payload[field], f"mapping_proof.{field}")
        if observed != expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"mapping proof count drift: {field}"
            )
    expected_constants = {
        "classification_only": True,
        "runtime_equivalence_claimed": False,
        "screening_eligible": False,
        "proposal_eligible": False,
        "registry_entry_created": False,
        "initial_weight_policy": "zero_only",
        "formal_admission_authority": False,
        "production_apply_enabled": False,
    }
    for field, expected in expected_constants.items():
        if payload[field] != expected or (
            type(expected) is bool and type(payload[field]) is not bool
        ):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"mapping proof non-executable contract drift: {field}"
            )
    normalized = {
        "schema_version": PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": _text(payload["cycle_id"], "mapping_proof.cycle_id"),
        "source_idea_audit_sha256": _sha(
            payload["source_idea_audit_sha256"],
            "mapping_proof.source_idea_audit_sha256",
        ),
        "discovery_catalog_sha256": _sha(
            payload["discovery_catalog_sha256"],
            "mapping_proof.discovery_catalog_sha256",
        ),
        **bindings,
        "mapping_policy_sha256": _sha(
            payload["mapping_policy_sha256"], "mapping_proof.mapping_policy_sha256"
        ),
        **counts,
        "source_candidate_ids_sha256": expected_source_ids_sha,
        "new_candidate_mappings": new_rows,
        "structural_aliases": aliases,
        "incompatible_candidates": incompatible,
        **expected_constants,
        "proof_semantic_sha256": _sha(
            payload["proof_semantic_sha256"],
            "mapping_proof.proof_semantic_sha256",
        ),
    }
    if normalized["proof_semantic_sha256"] != _self_hash(
        normalized, "proof_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "mapping proof semantic SHA mismatch"
        )
    return normalized


def _build_descriptor(
    candidate: Mapping[str, Any],
    *,
    role: str,
    mapping: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if role == "base_reference":
        source_candidate_id: str | None = None
        source_definition_sha256 = candidate["definition_sha256"]
        mapping_proof_row_sha256: str | None = None
        blockers = list(_BASE_DESCRIPTOR_BLOCKERS)
    else:
        if mapping is None:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                "new descriptor requires one mapping proof row"
            )
        source_candidate_id = mapping["candidate_id"]
        source_definition_sha256 = mapping["source_definition_sha256"]
        mapping_proof_row_sha256 = mapping["mapping_semantic_sha256"]
        blockers = list(_NEW_DESCRIPTOR_BLOCKERS)
    return _seal(
        {
            "schema_version": (
                FORMAL_CATALOG_CANDIDATE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "name": candidate["name"],
            "catalog_role": role,
            "catalog_definition_sha256": candidate["definition_sha256"],
            "catalog_implementation": candidate["implementation"],
            "descriptor_implementation": NON_EXECUTABLE_DESCRIPTOR_IMPLEMENTATION,
            "source_candidate_id": source_candidate_id,
            "source_definition_sha256": source_definition_sha256,
            "mapping_proof_row_sha256": mapping_proof_row_sha256,
            "classification_only": True,
            "executable": False,
            "screening_eligible": False,
            "runtime_equivalence_verified": False,
            "blockers": blockers,
        },
        "descriptor_semantic_sha256",
    )


def validate_formal_catalog_candidate_descriptor_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact non-executable candidate descriptor."""

    payload = _exact(value, _DESCRIPTOR_FIELDS, "candidate descriptor")
    if (
        payload["schema_version"]
        != FORMAL_CATALOG_CANDIDATE_DESCRIPTOR_SCHEMA_VERSION
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "candidate descriptor schema mismatch"
        )
    role = payload["catalog_role"]
    if role not in {"base_reference", "new_classification"}:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "candidate descriptor role mismatch"
        )
    expected_blockers = (
        _BASE_DESCRIPTOR_BLOCKERS
        if role == "base_reference"
        else _NEW_DESCRIPTOR_BLOCKERS
    )
    blockers = _sorted_distinct_texts(payload["blockers"], "descriptor.blockers")
    if tuple(blockers) != expected_blockers:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "candidate descriptor blockers mismatch"
        )
    source_candidate_id = payload["source_candidate_id"]
    mapping_row_sha = payload["mapping_proof_row_sha256"]
    if role == "base_reference":
        if source_candidate_id is not None or mapping_row_sha is not None:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                "base descriptor cannot claim mapping proof provenance"
            )
    else:
        source_candidate_id = _text(
            source_candidate_id, "descriptor.source_candidate_id"
        )
        mapping_row_sha = _sha(
            mapping_row_sha, "descriptor.mapping_proof_row_sha256"
        )
    normalized = {
        "schema_version": (
            FORMAL_CATALOG_CANDIDATE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "name": _text(payload["name"], "descriptor.name"),
        "catalog_role": role,
        "catalog_definition_sha256": _sha(
            payload["catalog_definition_sha256"],
            "descriptor.catalog_definition_sha256",
        ),
        "catalog_implementation": _text(
            payload["catalog_implementation"],
            "descriptor.catalog_implementation",
        ),
        "descriptor_implementation": _text(
            payload["descriptor_implementation"],
            "descriptor.descriptor_implementation",
        ),
        "source_candidate_id": source_candidate_id,
        "source_definition_sha256": _sha(
            payload["source_definition_sha256"],
            "descriptor.source_definition_sha256",
        ),
        "mapping_proof_row_sha256": mapping_row_sha,
        "classification_only": _require_bool(
            payload["classification_only"], True, "descriptor.classification_only"
        ),
        "executable": _require_bool(
            payload["executable"], False, "descriptor.executable"
        ),
        "screening_eligible": _require_bool(
            payload["screening_eligible"], False, "descriptor.screening_eligible"
        ),
        "runtime_equivalence_verified": _require_bool(
            payload["runtime_equivalence_verified"],
            False,
            "descriptor.runtime_equivalence_verified",
        ),
        "blockers": blockers,
        "descriptor_semantic_sha256": _sha(
            payload["descriptor_semantic_sha256"],
            "descriptor.descriptor_semantic_sha256",
        ),
    }
    if (
        normalized["descriptor_implementation"]
        != NON_EXECUTABLE_DESCRIPTOR_IMPLEMENTATION
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "candidate descriptor implementation is not validate-only"
        )
    if normalized["descriptor_semantic_sha256"] != _self_hash(
        normalized, "descriptor_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "candidate descriptor semantic SHA mismatch"
        )
    return normalized


def _prepare_inputs(
    *,
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    ontology: Mapping[str, Any],
    catalog: Mapping[str, Any],
    mapping_proof: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        normalized_base_ontology = validate_primitive_ontology_v4(base_ontology)
        normalized_base_catalog = validate_candidate_catalog_v4(
            base_catalog, ontology=normalized_base_ontology
        )
        normalized_ontology = validate_primitive_ontology_v4(ontology)
        normalized_catalog = validate_candidate_catalog_v4(
            catalog, ontology=normalized_ontology
        )
    except FactorGovernanceScreeningV4Error as exc:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"v4 ontology/catalog schema validation failed: {exc}"
        ) from exc

    base_primitive_map = {
        row["primitive_id"]: row["family"]
        for row in normalized_base_ontology["primitives"]
    }
    formal_primitive_map = {
        row["primitive_id"]: row["family"]
        for row in normalized_ontology["primitives"]
    }
    missing_primitives = sorted(set(base_primitive_map) - set(formal_primitive_map))
    drifted_primitives = sorted(
        primitive_id
        for primitive_id, family in base_primitive_map.items()
        if formal_primitive_map.get(primitive_id) != family
    )
    if missing_primitives or drifted_primitives:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "formal ontology does not preserve the exact base primitive mapping: "
            f"missing={missing_primitives}; drifted={drifted_primitives}"
        )

    base_candidates = {
        candidate["name"]: candidate
        for candidate in normalized_base_catalog["candidates"]
    }
    formal_candidates = {
        candidate["name"]: candidate
        for candidate in normalized_catalog["candidates"]
    }
    missing_base_names = sorted(set(base_candidates) - set(formal_candidates))
    if missing_base_names:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            f"formal catalog is missing base candidates: {missing_base_names}"
        )
    for name, base_candidate in base_candidates.items():
        if formal_candidates[name] != base_candidate:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"base candidate definition drift: {name}"
            )
    for candidate in formal_candidates.values():
        if type(candidate["direction"]) is not float or candidate["direction"] != 1.0:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"candidate direction must be canonical +1.0: {candidate['name']}"
            )
    new_names = sorted(set(formal_candidates) - set(base_candidates))
    for name in new_names:
        if (
            formal_candidates[name]["implementation"]
            != CLASSIFICATION_IDENTITY_IMPLEMENTATION
        ):
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"unknown new-candidate implementation: {name}"
            )

    normalized_proof = _validate_mapping_proof(
        mapping_proof,
        base_ontology_sha256=normalized_base_ontology["semantic_sha256"],
        base_catalog_sha256=normalized_base_catalog["semantic_sha256"],
        formal_ontology_sha256=normalized_ontology["semantic_sha256"],
        formal_catalog_sha256=normalized_catalog["semantic_sha256"],
        ontology_families=formal_primitive_map,
        base_candidates=base_candidates,
        formal_candidates=formal_candidates,
    )
    mappings_by_name = {
        row["name"]: row for row in normalized_proof["new_candidate_mappings"]
    }
    descriptors = [
        _build_descriptor(
            candidate,
            role=(
                "base_reference"
                if candidate["name"] in base_candidates
                else "new_classification"
            ),
            mapping=mappings_by_name.get(candidate["name"]),
        )
        for candidate in normalized_catalog["candidates"]
    ]
    return {
        "base_ontology": normalized_base_ontology,
        "base_catalog": normalized_base_catalog,
        "ontology": normalized_ontology,
        "catalog": normalized_catalog,
        "mapping_proof": normalized_proof,
        "base_candidates": base_candidates,
        "formal_candidates": formal_candidates,
        "new_names": new_names,
        "descriptors": descriptors,
    }


def build_formal_catalog_adapter_validation_v4_1(
    *,
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    ontology: Mapping[str, Any],
    catalog: Mapping[str, Any],
    mapping_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed, non-executable validation for an upstream-validated proof."""

    prepared = _prepare_inputs(
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=ontology,
        catalog=catalog,
        mapping_proof=mapping_proof,
    )
    normalized_catalog = prepared["catalog"]
    descriptors = prepared["descriptors"]
    payload = {
        "schema_version": FORMAL_CATALOG_ADAPTER_VALIDATION_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "base_ontology_sha256": prepared["base_ontology"]["semantic_sha256"],
        "base_catalog_sha256": prepared["base_catalog"]["semantic_sha256"],
        "formal_ontology_sha256": prepared["ontology"]["semantic_sha256"],
        "formal_catalog_sha256": normalized_catalog["semantic_sha256"],
        "mapping_proof_schema_version": PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
        "mapping_proof_sha256": prepared["mapping_proof"][
            "proof_semantic_sha256"
        ],
        "candidate_count": len(normalized_catalog["candidates"]),
        "base_candidate_count": len(prepared["base_candidates"]),
        "new_candidate_count": len(prepared["new_names"]),
        "descriptor_count": len(descriptors),
        "ordered_candidate_names_semantic_sha256": semantic_sha256(
            [candidate["name"] for candidate in normalized_catalog["candidates"]]
        ),
        "ordered_definition_pairs_semantic_sha256": semantic_sha256(
            [
                {
                    "name": candidate["name"],
                    "definition_sha256": candidate["definition_sha256"],
                }
                for candidate in normalized_catalog["candidates"]
            ]
        ),
        "base_primitive_mapping_preserved": True,
        "base_candidate_definitions_preserved": True,
        "catalog_schema_loader_readable": True,
        "classification_only": True,
        "source_authenticity_recomputed": False,
        "signal_computability_proven": False,
        "data_loader_invoked": False,
        "statistics_invoked": False,
        "registry": False,
        "proposal": False,
        "apply": False,
        "registry_entry_created": False,
        "registry_mutation_performed": False,
        "proposal_eligible": False,
        "screening_eligible": False,
        "runtime_equivalence_verified": False,
        "production_apply_enabled": False,
        "descriptors": descriptors,
        "blockers": list(_VALIDATION_BLOCKERS),
    }
    artifact = _seal(payload, "validation_semantic_sha256")
    return validate_formal_catalog_adapter_validation_v4_1(
        artifact,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=ontology,
        catalog=catalog,
        mapping_proof=mapping_proof,
    )


def validate_formal_catalog_adapter_validation_v4_1(
    value: Mapping[str, Any],
    *,
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    ontology: Mapping[str, Any],
    catalog: Mapping[str, Any],
    mapping_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind the artifact without claiming upstream source recomputation."""

    prepared = _prepare_inputs(
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=ontology,
        catalog=catalog,
        mapping_proof=mapping_proof,
    )
    payload = _exact(value, _VALIDATION_FIELDS, "adapter validation")
    canonical_json_bytes(payload)
    if payload["schema_version"] != FORMAL_CATALOG_ADAPTER_VALIDATION_SCHEMA_VERSION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation schema mismatch"
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation protocol mismatch"
        )
    expected_bindings = {
        "base_ontology_sha256": prepared["base_ontology"]["semantic_sha256"],
        "base_catalog_sha256": prepared["base_catalog"]["semantic_sha256"],
        "formal_ontology_sha256": prepared["ontology"]["semantic_sha256"],
        "formal_catalog_sha256": prepared["catalog"]["semantic_sha256"],
        "mapping_proof_sha256": prepared["mapping_proof"][
            "proof_semantic_sha256"
        ],
    }
    for field, expected in expected_bindings.items():
        if _sha(payload[field], f"adapter validation {field}") != expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"adapter validation input binding drift: {field}"
            )
    if payload["mapping_proof_schema_version"] != PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation mapping proof schema mismatch"
        )
    expected_counts = {
        "candidate_count": len(prepared["formal_candidates"]),
        "base_candidate_count": len(prepared["base_candidates"]),
        "new_candidate_count": len(prepared["new_names"]),
        "descriptor_count": len(prepared["descriptors"]),
    }
    for field, expected in expected_counts.items():
        if _nonnegative_int(payload[field], f"adapter validation {field}") != expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"adapter validation count drift: {field}"
            )
    expected_hashes = {
        "ordered_candidate_names_semantic_sha256": semantic_sha256(
            [candidate["name"] for candidate in prepared["catalog"]["candidates"]]
        ),
        "ordered_definition_pairs_semantic_sha256": semantic_sha256(
            [
                {
                    "name": candidate["name"],
                    "definition_sha256": candidate["definition_sha256"],
                }
                for candidate in prepared["catalog"]["candidates"]
            ]
        ),
    }
    for field, expected in expected_hashes.items():
        if _sha(payload[field], f"adapter validation {field}") != expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"adapter validation ordered membership drift: {field}"
            )
    expected_constants = {
        "base_primitive_mapping_preserved": True,
        "base_candidate_definitions_preserved": True,
        "catalog_schema_loader_readable": True,
        "classification_only": True,
        "source_authenticity_recomputed": False,
        "signal_computability_proven": False,
        "data_loader_invoked": False,
        "statistics_invoked": False,
        "registry": False,
        "proposal": False,
        "apply": False,
        "registry_entry_created": False,
        "registry_mutation_performed": False,
        "proposal_eligible": False,
        "screening_eligible": False,
        "runtime_equivalence_verified": False,
        "production_apply_enabled": False,
    }
    for field, expected in expected_constants.items():
        if type(payload[field]) is not bool or payload[field] is not expected:
            raise FactorGovernanceFormalCatalogAdapterV4_1Error(
                f"adapter validation non-executable contract drift: {field}"
            )
    descriptors = [
        validate_formal_catalog_candidate_descriptor_v4_1(row)
        for row in _list(payload["descriptors"], "adapter validation descriptors")
    ]
    if descriptors != prepared["descriptors"]:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation descriptor/source proof drift"
        )
    blockers = _sorted_distinct_texts(payload["blockers"], "adapter blockers")
    if tuple(blockers) != _VALIDATION_BLOCKERS:
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation blockers mismatch"
        )
    normalized = {
        "schema_version": FORMAL_CATALOG_ADAPTER_VALIDATION_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        **expected_bindings,
        "mapping_proof_schema_version": PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
        **expected_counts,
        **expected_hashes,
        **expected_constants,
        "descriptors": descriptors,
        "blockers": blockers,
        "validation_semantic_sha256": _sha(
            payload["validation_semantic_sha256"],
            "adapter validation validation_semantic_sha256",
        ),
    }
    if normalized["validation_semantic_sha256"] != _self_hash(
        normalized, "validation_semantic_sha256"
    ):
        raise FactorGovernanceFormalCatalogAdapterV4_1Error(
            "adapter validation semantic SHA mismatch"
        )
    return normalized


build_formal_catalog_adapter_v4_1 = (
    build_formal_catalog_adapter_validation_v4_1
)
validate_formal_catalog_adapter_v4_1 = (
    validate_formal_catalog_adapter_validation_v4_1
)


__all__ = [
    "CLASSIFICATION_IDENTITY_IMPLEMENTATION",
    "FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME",
    "FORMAL_CATALOG_ADAPTER_VALIDATION_SCHEMA_VERSION",
    "FORMAL_CATALOG_CANDIDATE_DESCRIPTOR_SCHEMA_VERSION",
    "FactorGovernanceFormalCatalogAdapterV4_1Error",
    "FactorGovernanceFormalCatalogAdapterV41Error",
    "NON_EXECUTABLE_DESCRIPTOR_IMPLEMENTATION",
    "PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION",
    "build_formal_catalog_adapter_v4_1",
    "build_formal_catalog_adapter_validation_v4_1",
    "canonical_json_bytes",
    "semantic_sha256",
    "validate_formal_catalog_adapter_v4_1",
    "validate_formal_catalog_adapter_validation_v4_1",
    "validate_formal_catalog_candidate_descriptor_v4_1",
]
