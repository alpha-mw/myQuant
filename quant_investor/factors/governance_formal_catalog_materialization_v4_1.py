"""Pure v4.1 DISCOVERY-to-formal-catalog materialization contracts.

This module deliberately has no filesystem, provider, market-data, statistics,
registry, or execution dependency.  It classifies exact normalized AST
occurrences, extends the v4 primitive ontology, and freezes a non-executable
267-candidate catalog from caller-supplied, already accepted artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors.governance_discovery_v4_1 import (
    AQUANT_SOURCE_RECEIPT_FILENAME,
    CANONICAL_ARTIFACT_FILENAMES,
    DISCOVERY_CYCLE_STATE_FILENAME,
    DISCOVERY_CATALOG_FILENAME,
    DISCOVERY_READBACK_REPORT_FILENAME,
    DISCOVERY_SOURCE_NODE_FILENAME,
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME,
    SELF_HASH_FIELD_BY_FILENAME,
    SOURCE_IDEA_AUDIT_FILENAME,
    STRUCTURAL_COLLISION_AUDIT_FILENAME,
    normalize_expression_ast_v4_1,
    validate_discovery_bundle_v4_1,
)
from quant_investor.factors.governance_screening_v4 import (
    build_candidate_catalog_v4,
    build_primitive_ontology_v4,
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
)


PROTOCOL_VERSION = "v4.1"

PRIMITIVE_MAPPING_POLICY_SCHEMA_VERSION = (
    "factor-governance-primitive-mapping-policy.v4.1"
)
PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION = (
    "factor-governance-primitive-mapping-proof.v4.1"
)
FORMAL_CATALOG_MATERIALIZATION_MANIFEST_SCHEMA_VERSION = (
    "factor-governance-formal-catalog-materialization-manifest.v4.1"
)

PRIMITIVE_MAPPING_POLICY_FILENAME = "primitive_mapping_policy.v4_1.json"
PRIMITIVE_MAPPING_PROOF_FILENAME = "primitive_mapping_proof.v4_1.json"
FORMAL_ONTOLOGY_FILENAME = "primitive_ontology.v4.json"
FORMAL_CATALOG_FILENAME = "candidate_catalog.v4.json"
FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME = (
    "formal_catalog_materialization_manifest.v4_1.json"
)
FORMAL_CATALOG_MATERIALIZATION_FILENAMES = (
    PRIMITIVE_MAPPING_POLICY_FILENAME,
    PRIMITIVE_MAPPING_PROOF_FILENAME,
    FORMAL_ONTOLOGY_FILENAME,
    FORMAL_CATALOG_FILENAME,
    FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME,
)

CLASSIFICATION_IMPLEMENTATION = "aquant_expression_ast.v1"
CLASSIFICATION_ONLY = True
RUNTIME_EQUIVALENCE_CLAIMED = False
SCREENING_ELIGIBLE = False
PROPOSAL_ELIGIBLE = False
REGISTRY_ENTRY_CREATED = False
INITIAL_WEIGHT_POLICY = "zero_only"
MAPPING_STATUS_COMPLETE = "complete_unique_occurrence_accounting"
NOT_RUNTIME_VERIFIED = "not_runtime_verified"

EXPECTED_SOURCE_CANDIDATE_COUNT = 100
EXPECTED_BASE_CANDIDATE_COUNT = 230
EXPECTED_NEW_CANDIDATE_COUNT = 37
EXPECTED_STRUCTURAL_ALIAS_COUNT = 6
EXPECTED_INCOMPATIBLE_COUNT = 57
EXPECTED_FORMAL_CANDIDATE_COUNT = 267
EXPECTED_BASE_PRIMITIVE_COUNT = 13
EXPECTED_FORMAL_PRIMITIVE_COUNT = 18

SOURCE_BINDING_ID_BY_DISCOVERY_FILENAME = {
    AQUANT_SOURCE_RECEIPT_FILENAME: f"discovery:{AQUANT_SOURCE_RECEIPT_FILENAME}",
    SOURCE_IDEA_AUDIT_FILENAME: "source_idea_audit",
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME: (
        f"discovery:{LOCAL_COMPATIBILITY_CONTRACT_FILENAME}"
    ),
    DISCOVERY_CATALOG_FILENAME: "discovery_catalog",
    STRUCTURAL_COLLISION_AUDIT_FILENAME: (
        f"discovery:{STRUCTURAL_COLLISION_AUDIT_FILENAME}"
    ),
    DISCOVERY_SOURCE_NODE_FILENAME: f"discovery:{DISCOVERY_SOURCE_NODE_FILENAME}",
    DISCOVERY_CYCLE_STATE_FILENAME: f"discovery:{DISCOVERY_CYCLE_STATE_FILENAME}",
    DISCOVERY_READBACK_REPORT_FILENAME: (
        f"discovery:{DISCOVERY_READBACK_REPORT_FILENAME}"
    ),
}
REQUIRED_SOURCE_BINDING_IDS = tuple(
    sorted(
        {
            "base_ontology",
            "base_catalog",
            *SOURCE_BINDING_ID_BY_DISCOVERY_FILENAME.values(),
        }
    )
)
REQUIRED_CODE_BINDING_SUFFIXES = (
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_discovery_v4_1.py",
    "quant_investor/factors/governance_discovery_readback_v4_1.py",
    "quant_investor/factors/governance_formal_catalog_adapter_v4_1.py",
    "quant_investor/factors/governance_formal_catalog_bundle_v4_1.py",
    "quant_investor/factors/governance_formal_catalog_materialization_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_source_v4_1.py",
    "scripts/build_factor_v4_1_formal_catalog.py",
)

NEW_PRIMITIVES = (
    "intraday_body_fraction",
    "lower_shadow_fraction",
    "turnover_rate",
    "upper_shadow_fraction",
    "vwap_close_gap",
)
FUNDAMENTAL_IDENTITY_PRIMITIVES = (
    "fcf_to_price",
    "fin_debt_to_assets",
    "fin_fcf_to_profit",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_roa",
    "fin_roe",
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_RULE_FIELDS = frozenset(
    {
        "rule_id",
        "match_kind",
        "normalized_ast_pattern",
        "identifier",
        "primitive_id",
        "family",
        "consumed_identifier_multiset",
        "base_semantics_binding",
        "classification_only",
        "runtime_equivalence_claimed",
        "unit_equivalence_status",
        "scale_equivalence_status",
        "missing_value_equivalence_status",
        "zero_value_equivalence_status",
        "adjustment_equivalence_status",
        "rule_semantic_sha256",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "ast_normalizer",
        "ast_normalization_contract",
        "traversal_order",
        "node_occurrence_index_base",
        "node_path_contract",
        "match_contract",
        "ancestor_match_contract",
        "overlap_contract",
        "identifier_consumption_contract",
        "base_ontology_sha256",
        "rule_count",
        "rules",
        "classification_only",
        "runtime_equivalence_claimed",
        "screening_eligible",
        "proposal_eligible",
        "registry_entry_created",
        "formal_admission_authority",
        "production_apply_enabled",
        "policy_semantic_sha256",
    }
)
_CONSUMED_IDENTIFIER_FIELDS = frozenset(
    {"identifier", "node_path", "node_occurrence_index"}
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
_REASON_RECORD_FIELDS = frozenset({"code", "reason"})
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
_SOURCE_BINDING_FIELDS = frozenset(
    {"binding_id", "byte_sha256", "semantic_sha256"}
)
_CODE_BINDING_FIELDS = frozenset(
    {"absolute_path", "raw_sha256", "size_bytes"}
)
_ADAPTER_BINDING_FIELDS = frozenset(
    {"schema_version", "validation_semantic_sha256"}
)
_ADAPTER_CANDIDATE_DESCRIPTOR_FIELDS = frozenset(
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
_ADAPTER_VALIDATION_ARTIFACT_FIELDS = frozenset(
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
_ADAPTER_BASE_DESCRIPTOR_BLOCKERS = (
    "data_loader_not_invoked",
    "runtime_equivalence_not_verified",
    "screening_not_authorized",
    "signal_computability_not_proven",
    "statistics_not_invoked",
    "validate_only_descriptor",
)
_ADAPTER_NEW_DESCRIPTOR_BLOCKERS = tuple(
    sorted(
        {
            *_ADAPTER_BASE_DESCRIPTOR_BLOCKERS,
            "classification_identity_non_executable",
            "source_authenticity_not_recomputed_by_adapter",
        }
    )
)
_ADAPTER_VALIDATION_BLOCKERS = (
    "data_loader_not_invoked",
    "runtime_equivalence_not_verified",
    "screening_not_authorized",
    "signal_computability_not_proven",
    "source_authenticity_not_recomputed_by_adapter",
    "statistics_not_invoked",
)
_ARTIFACT_BINDING_FIELDS = frozenset(
    {"filename", "byte_sha256", "semantic_sha256", "size_bytes"}
)
_SIDE_EFFECT_FIELDS = frozenset(
    {
        "filesystem_read_performed",
        "filesystem_write_performed",
        "provider_call_performed",
        "market_data_access_performed",
        "statistics_performed",
        "holdout_access_performed",
        "registry_write_performed",
        "wal_write_performed",
        "receipt_created",
        "pointer_mutation_performed",
        "production_apply_performed",
        "broker_call_performed",
        "order_created",
        "trade_created",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "artifact_bindings",
        "source_bindings",
        "source_bindings_semantic_sha256",
        "code_bindings",
        "code_bindings_semantic_sha256",
        "adapter_validation_binding",
        "adapter_validation_status",
        "source_candidate_count",
        "base_candidate_count",
        "new_candidate_count",
        "structural_alias_count",
        "incompatible_count",
        "formal_candidate_count",
        "base_primitive_count",
        "formal_primitive_count",
        "classification_only",
        "runtime_equivalence_claimed",
        "screening_eligible",
        "proposal_eligible",
        "registry_entry_created",
        "initial_weight_policy",
        "qualification",
        "formal_admission_authority",
        "production_apply_enabled",
        "new_risk_authorized",
        "side_effect_scope",
        "side_effects",
        "manifest_semantic_sha256",
    }
)


class FactorGovernanceFormalCatalogMaterializationV4_1Error(ValueError):
    """Raised when formal catalog materialization fails closed."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    """Return exact canonical semantic JSON bytes (without a trailing newline)."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"value is not canonical JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Any) -> bytes:
    """Return canonical private-artifact bytes with one trailing newline."""

    return canonical_json_bytes_v4_1(value) + b"\n"


def semantic_sha256_v4_1(
    value: Any,
    *,
    exclude_fields: Sequence[str] = (),
) -> str:
    normalized = copy.deepcopy(value)
    if exclude_fields:
        if not isinstance(normalized, Mapping):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "exclude_fields requires a top-level object"
            )
        normalized = dict(normalized)
        seen: set[str] = set()
        for field in exclude_fields:
            if type(field) is not str or not field or field in seen:
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    "exclude_fields must contain distinct non-empty strings"
                )
            seen.add(field)
            normalized.pop(field, None)
    return hashlib.sha256(canonical_json_bytes_v4_1(normalized)).hexdigest()


def artifact_sha256_v4_1(value: Any) -> str:
    """Return the SHA-256 of canonical artifact file bytes."""

    return hashlib.sha256(canonical_file_bytes_v4_1(value)).hexdigest()


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be an object"
        )
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} fields invalid: {';'.join(details)}"
        )
    return payload


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be an exact non-empty string"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_integer(value: Any, label: str) -> int:
    number = _nonnegative_integer(value, label)
    if number == 0:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be a positive integer"
        )
    return number


def _sorted_distinct_texts(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be a list"
        )
    rows = [_text(item, f"{label}[]") for item in value]
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} must be canonically sorted and distinct"
        )
    return rows


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    if field in result:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"self-hash field already present: {field}"
        )
    result[field] = semantic_sha256_v4_1(result)
    return result


def _assert_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    observed = _sha(payload[field], f"{label}.{field}")
    expected = semantic_sha256_v4_1(payload, exclude_fields=(field,))
    if observed != expected:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"{label} semantic SHA mismatch"
        )
    return observed


def _primitive_binding(
    *, ontology_sha256: str, primitive_id: str, family: str
) -> dict[str, str]:
    row = {"primitive_id": primitive_id, "family": family}
    return {
        "base_ontology_sha256": ontology_sha256,
        "primitive_id": primitive_id,
        "family": family,
        "primitive_semantic_sha256": semantic_sha256_v4_1(row),
    }


def _pattern(expression: str) -> dict[str, Any]:
    return normalize_expression_ast_v4_1(expression)


def _rule(
    *,
    rule_id: str,
    match_kind: str,
    primitive_id: str,
    family: str,
    normalized_ast_pattern: Mapping[str, Any] | None = None,
    identifier: str | None = None,
    base_semantics_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if match_kind == "exact_subtree":
        if normalized_ast_pattern is None or identifier is not None:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "exact_subtree rule shape is invalid"
            )
        identifiers: list[str] = []

        def collect(node: Mapping[str, Any]) -> None:
            kind = node["kind"]
            if kind == "name":
                identifiers.append(str(node["identifier"]))
            elif kind == "unary":
                collect(node["operand"])
            elif kind == "binary":
                collect(node["left"])
                collect(node["right"])
            elif kind == "call":
                for argument in node["arguments"]:
                    collect(argument)

        collect(normalized_ast_pattern)
    elif match_kind == "identity_identifier":
        if normalized_ast_pattern is not None or identifier is None:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "identity_identifier rule shape is invalid"
            )
        identifiers = [identifier]
    else:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"unsupported match kind: {match_kind}"
        )
    payload = {
        "rule_id": rule_id,
        "match_kind": match_kind,
        "normalized_ast_pattern": copy.deepcopy(normalized_ast_pattern),
        "identifier": identifier,
        "primitive_id": primitive_id,
        "family": family,
        "consumed_identifier_multiset": sorted(identifiers),
        "base_semantics_binding": copy.deepcopy(base_semantics_binding),
        "classification_only": True,
        "runtime_equivalence_claimed": False,
        "unit_equivalence_status": NOT_RUNTIME_VERIFIED,
        "scale_equivalence_status": NOT_RUNTIME_VERIFIED,
        "missing_value_equivalence_status": NOT_RUNTIME_VERIFIED,
        "zero_value_equivalence_status": NOT_RUNTIME_VERIFIED,
        "adjustment_equivalence_status": NOT_RUNTIME_VERIFIED,
    }
    return _seal(payload, "rule_semantic_sha256")


def build_primitive_mapping_policy_v4_1(
    *, base_ontology: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact classification-only normalized-AST mapping policy."""

    try:
        ontology = validate_primitive_ontology_v4(base_ontology)
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"base ontology is invalid: {exc}"
        ) from exc
    families = {
        row["primitive_id"]: row["family"] for row in ontology["primitives"]
    }
    if len(families) != EXPECTED_BASE_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "base ontology must contain exactly 13 primitives"
        )
    required_existing = {"traded_amount", *FUNDAMENTAL_IDENTITY_PRIMITIVES}
    if any(families.get(item) != item for item in required_existing):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "base ontology existing semantic bindings drifted"
        )

    rules = [
        _rule(
            rule_id="exact_ast.intraday_body_fraction.v1",
            match_kind="exact_subtree",
            normalized_ast_pattern=_pattern("(close - open) / (high - low)"),
            primitive_id="intraday_body_fraction",
            family="intraday_body_fraction",
        ),
        _rule(
            rule_id="exact_ast.lower_shadow_fraction.v1",
            match_kind="exact_subtree",
            normalized_ast_pattern=_pattern("(close - low) / (high - low)"),
            primitive_id="lower_shadow_fraction",
            family="lower_shadow_fraction",
        ),
        _rule(
            rule_id="exact_ast.upper_shadow_fraction.v1",
            match_kind="exact_subtree",
            normalized_ast_pattern=_pattern("(high - close) / (high - low)"),
            primitive_id="upper_shadow_fraction",
            family="upper_shadow_fraction",
        ),
        _rule(
            rule_id="exact_ast.vwap_close_gap.v1",
            match_kind="exact_subtree",
            normalized_ast_pattern=_pattern("(vwap - close) / close"),
            primitive_id="vwap_close_gap",
            family="vwap_close_gap",
        ),
        _rule(
            rule_id="identity.amount_to_traded_amount.v1",
            match_kind="identity_identifier",
            identifier="amount",
            primitive_id="traded_amount",
            family="traded_amount",
            base_semantics_binding=_primitive_binding(
                ontology_sha256=ontology["semantic_sha256"],
                primitive_id="traded_amount",
                family="traded_amount",
            ),
        ),
        _rule(
            rule_id="identity.turnover_rate.v1",
            match_kind="identity_identifier",
            identifier="turnover_rate",
            primitive_id="turnover_rate",
            family="turnover_rate",
        ),
    ]
    for primitive_id in FUNDAMENTAL_IDENTITY_PRIMITIVES:
        rules.append(
            _rule(
                rule_id=f"identity.fundamental.{primitive_id}.v1",
                match_kind="identity_identifier",
                identifier=primitive_id,
                primitive_id=primitive_id,
                family=primitive_id,
                base_semantics_binding=_primitive_binding(
                    ontology_sha256=ontology["semantic_sha256"],
                    primitive_id=primitive_id,
                    family=primitive_id,
                ),
            )
        )
    rules.sort(key=lambda row: row["rule_id"])
    payload = {
        "schema_version": PRIMITIVE_MAPPING_POLICY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ast_normalizer": "normalize_expression_ast_v4_1",
        "ast_normalization_contract": "syntax_only_no_algebraic_or_commutative_rewrite",
        "traversal_order": "normalized_ast_preorder_depth_first_left_to_right",
        "node_occurrence_index_base": 0,
        "node_path_contract": "root_dot_fields_arguments_bracket_index",
        "match_contract": "exact_normalized_ast_or_exact_identifier_occurrence",
        "ancestor_match_contract": "matched_ancestor_consumes_all_descendant_identifiers",
        "overlap_contract": "each_node_zero_or_one_rule_ambiguous_or_overlap_fails_batch",
        "identifier_consumption_contract": "every_data_identifier_occurrence_exactly_once",
        "base_ontology_sha256": ontology["semantic_sha256"],
        "rule_count": len(rules),
        "rules": rules,
        "classification_only": True,
        "runtime_equivalence_claimed": False,
        "screening_eligible": False,
        "proposal_eligible": False,
        "registry_entry_created": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
    }
    return _seal(payload, "policy_semantic_sha256")


def validate_primitive_mapping_policy_v4_1(
    value: Mapping[str, Any], *, base_ontology: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact policy fields, rule self-hashes, and fixed semantics."""

    payload = _exact(value, _POLICY_FIELDS, "primitive mapping policy")
    _assert_self_hash(payload, "policy_semantic_sha256", "primitive mapping policy")
    expected = build_primitive_mapping_policy_v4_1(base_ontology=base_ontology)
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "primitive mapping policy differs from the fixed v4.1 policy"
        )
    for index, raw in enumerate(payload["rules"]):
        row = _exact(raw, _RULE_FIELDS, f"rules[{index}]")
        _assert_self_hash(row, "rule_semantic_sha256", f"rules[{index}]")
    return copy.deepcopy(expected)


def _node_children(
    node: Mapping[str, Any], path: str
) -> list[tuple[Mapping[str, Any], str]]:
    kind = node.get("kind")
    if kind in {"name", "constant"}:
        return []
    if kind == "unary":
        return [(node["operand"], f"{path}.operand")]
    if kind == "binary":
        return [
            (node["left"], f"{path}.left"),
            (node["right"], f"{path}.right"),
        ]
    if kind == "call":
        return [
            (argument, f"{path}.arguments[{index}]")
            for index, argument in enumerate(node["arguments"])
        ]
    raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
        f"unknown normalized AST node kind at {path}: {kind}"
    )


def _indexed_nodes(
    tree: Mapping[str, Any]
) -> tuple[list[tuple[Mapping[str, Any], str, int]], dict[str, int]]:
    rows: list[tuple[Mapping[str, Any], str, int]] = []
    by_path: dict[str, int] = {}

    def visit(node: Mapping[str, Any], path: str) -> None:
        if not isinstance(node, Mapping):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"normalized AST node must be an object: {path}"
            )
        index = len(rows)
        rows.append((node, path, index))
        by_path[path] = index
        for child, child_path in _node_children(node, path):
            visit(child, child_path)

    visit(tree, "root")
    return rows, by_path


def _rule_matches(rule: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    if rule["match_kind"] == "exact_subtree":
        return node == rule["normalized_ast_pattern"]
    return node.get("kind") == "name" and node.get("identifier") == rule["identifier"]


def _descendant_identifiers(
    node: Mapping[str, Any], path: str, indexes: Mapping[str, int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(current: Mapping[str, Any], current_path: str) -> None:
        if current["kind"] == "name":
            result.append(
                {
                    "identifier": str(current["identifier"]),
                    "node_path": current_path,
                    "node_occurrence_index": indexes[current_path],
                }
            )
            return
        for child, child_path in _node_children(current, current_path):
            visit(child, child_path)

    visit(node, path)
    return result


def classify_normalized_ast_occurrences_v4_1(
    *,
    normalized_ast: Mapping[str, Any],
    input_fields: Sequence[str],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify exact AST occurrences and fail on ambiguity or non-consumption.

    ``rules`` is explicit to make duplicate-occurrence and ambiguity behavior
    independently testable.  Formal bundle construction always supplies the
    sealed fixed policy rules.
    """

    tree = copy.deepcopy(dict(normalized_ast))
    canonical_json_bytes_v4_1(tree)
    if not isinstance(input_fields, (list, tuple)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "input_fields must be a sequence"
        )
    normalized_fields = sorted(
        {_text(item, "input_fields[]") for item in input_fields}
    )
    if len(normalized_fields) != len(input_fields):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "input_fields must be distinct"
        )
    normalized_rules: list[dict[str, Any]] = []
    for index, raw in enumerate(rules):
        row = _exact(raw, _RULE_FIELDS, f"rules[{index}]")
        _assert_self_hash(row, "rule_semantic_sha256", f"rules[{index}]")
        normalized_rules.append(copy.deepcopy(row))
    rule_ids = [row["rule_id"] for row in normalized_rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "mapping rule ids must be distinct"
        )

    indexed, indexes = _indexed_nodes(tree)
    all_identifiers: list[dict[str, Any]] = [
        {
            "identifier": str(node["identifier"]),
            "node_path": path,
            "node_occurrence_index": index,
        }
        for node, path, index in indexed
        if node["kind"] == "name"
    ]
    if sorted({row["identifier"] for row in all_identifiers}) != normalized_fields:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "input_fields do not equal normalized AST data identifiers"
        )

    raw_matches: list[dict[str, Any]] = []

    def visit(node: Mapping[str, Any], path: str) -> None:
        matches = [rule for rule in normalized_rules if _rule_matches(rule, node)]
        if len(matches) > 1:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "ambiguous primitive mapping at "
                + path
                + ":"
                + ",".join(sorted(rule["rule_id"] for rule in matches))
            )
        if matches:
            rule = matches[0]
            descendant_rule_matches: list[tuple[str, str]] = []

            def inspect_consumed_descendants(
                current: Mapping[str, Any], current_path: str
            ) -> None:
                for child, child_path in _node_children(current, current_path):
                    for descendant_rule in normalized_rules:
                        if _rule_matches(descendant_rule, child):
                            descendant_rule_matches.append(
                                (child_path, descendant_rule["rule_id"])
                            )
                    inspect_consumed_descendants(child, child_path)

            inspect_consumed_descendants(node, path)
            if descendant_rule_matches:
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    "overlapping primitive mapping under matched ancestor "
                    + path
                    + ":"
                    + ",".join(
                        f"{child_path}@{child_rule_id}"
                        for child_path, child_rule_id in descendant_rule_matches
                    )
                )
            consumed = _descendant_identifiers(node, path, indexes)
            expected_multiset = sorted(rule["consumed_identifier_multiset"])
            if sorted(row["identifier"] for row in consumed) != expected_multiset:
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    f"rule consumed identifier multiset mismatch: {rule['rule_id']}"
                )
            raw_matches.append(
                {
                    "node_path": path,
                    "node_occurrence_index": indexes[path],
                    "subtree_sha256": semantic_sha256_v4_1(node),
                    "rule_id": rule["rule_id"],
                    "primitive_id": rule["primitive_id"],
                    "family": rule["family"],
                    "consumed_identifiers": consumed,
                }
            )
            return
        for child, child_path in _node_children(node, path):
            visit(child, child_path)

    visit(tree, "root")
    consumed_rows = [
        item for match in raw_matches for item in match["consumed_identifiers"]
    ]
    if consumed_rows != sorted(
        consumed_rows, key=lambda row: row["node_occurrence_index"]
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "consumed identifiers are not in stable occurrence order"
        )
    if consumed_rows != all_identifiers:
        consumed_indexes = {row["node_occurrence_index"] for row in consumed_rows}
        unconsumed = [
            row for row in all_identifiers if row["node_occurrence_index"] not in consumed_indexes
        ]
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "unconsumed data identifier occurrences: "
            + ",".join(
                f"{row['identifier']}@{row['node_path']}" for row in unconsumed
            )
        )
    if not raw_matches:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "candidate has zero primitive mapping matches"
        )
    cardinalities: dict[str, int] = {}
    for match in raw_matches:
        cardinalities[match["rule_id"]] = cardinalities.get(match["rule_id"], 0) + 1
    occurrence_indexes: dict[str, int] = {}
    occurrences: list[dict[str, Any]] = []
    for match in raw_matches:
        rule_id = match["rule_id"]
        rule_occurrence_index = occurrence_indexes.get(rule_id, 0)
        occurrence_indexes[rule_id] = rule_occurrence_index + 1
        occurrences.append(
            {
                "node_path": match["node_path"],
                "node_occurrence_index": match["node_occurrence_index"],
                "rule_occurrence_index": rule_occurrence_index,
                "subtree_sha256": match["subtree_sha256"],
                "rule_id": rule_id,
                "primitive_id": match["primitive_id"],
                "family": match["family"],
                "match_cardinality": cardinalities[rule_id],
                "consumed_identifiers": match["consumed_identifiers"],
            }
        )
    return {
        "full_candidate_normalized_ast_sha256": semantic_sha256_v4_1(tree),
        "primitive_ids": sorted({row["primitive_id"] for row in occurrences}),
        "occurrence_count": len(occurrences),
        "occurrences": occurrences,
        "mapping_status": MAPPING_STATUS_COMPLETE,
    }


def build_candidate_primitive_proof_v4_1(
    *,
    expression: str,
    input_fields: Sequence[str],
    mapping_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one expression and produce occurrence-complete proof data."""

    return classify_normalized_ast_occurrences_v4_1(
        normalized_ast=normalize_expression_ast_v4_1(expression),
        input_fields=input_fields,
        rules=mapping_policy["rules"],
    )


def _candidate_input_from_catalog_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(row[key])
        for key in (
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
    }


def _new_candidate_input(
    member: Mapping[str, Any], primitive_ids: Sequence[str]
) -> dict[str, Any]:
    ordered_primitives = sorted(primitive_ids)
    return {
        "name": member["name"],
        "implementation": CLASSIFICATION_IMPLEMENTATION,
        "expression": member["expression"],
        "direction": 1.0,
        "params": {},
        "lookback": member["lookback"],
        "slot": "primitive:" + "+".join(ordered_primitives),
        "input_fields": list(member["input_fields"]),
        "primitive_ids": ordered_primitives,
    }


def _normalize_source_bindings(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for binding_id, raw in value.items():
            normalized_id = _text(binding_id, "source binding id")
            if not isinstance(raw, Mapping):
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    "source binding descriptor must be an object"
                )
            rows.append(
                {
                    "binding_id": normalized_id,
                    "byte_sha256": _sha(
                        raw.get("byte_sha256"), f"source_bindings.{normalized_id}.byte_sha256"
                    ),
                    "semantic_sha256": _sha(
                        raw.get("semantic_sha256"),
                        f"source_bindings.{normalized_id}.semantic_sha256",
                    ),
                }
            )
    elif isinstance(value, list):
        for index, raw in enumerate(value):
            row = _exact(raw, _SOURCE_BINDING_FIELDS, f"source_bindings[{index}]")
            rows.append(
                {
                    "binding_id": _text(row["binding_id"], f"source_bindings[{index}].binding_id"),
                    "byte_sha256": _sha(row["byte_sha256"], f"source_bindings[{index}].byte_sha256"),
                    "semantic_sha256": _sha(
                        row["semantic_sha256"], f"source_bindings[{index}].semantic_sha256"
                    ),
                }
            )
    else:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source_bindings must be an object or list"
        )
    rows.sort(key=lambda row: row["binding_id"])
    ids = [row["binding_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source binding ids must be distinct"
        )
    if tuple(ids) != REQUIRED_SOURCE_BINDING_IDS:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source bindings must be the exact base2+Discovery8 inventory"
        )
    return rows


def _normalize_code_bindings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "code_bindings must be a non-empty list"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _CODE_BINDING_FIELDS, f"code_bindings[{index}]")
        absolute_path = _text(row["absolute_path"], f"code_bindings[{index}].absolute_path")
        if (
            not absolute_path.startswith("/")
            or posixpath.normpath(absolute_path) != absolute_path
            or "//" in absolute_path
        ):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "code binding paths must be normalized absolute paths"
            )
        rows.append(
            {
                "absolute_path": absolute_path,
                "raw_sha256": _sha(row["raw_sha256"], f"code_bindings[{index}].raw_sha256"),
                "size_bytes": _positive_integer(row["size_bytes"], f"code_bindings[{index}].size_bytes"),
            }
        )
    rows.sort(key=lambda row: row["absolute_path"])
    paths = [row["absolute_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "code binding paths must be distinct"
        )
    matched_suffixes: list[str] = []
    roots: set[str] = set()
    for path in paths:
        matches = [
            suffix
            for suffix in REQUIRED_CODE_BINDING_SUFFIXES
            if path.endswith("/" + suffix)
        ]
        if len(matches) != 1:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"code binding path is outside the exact formal allowlist: {path}"
            )
        suffix = matches[0]
        matched_suffixes.append(suffix)
        roots.add(path[: -(len(suffix) + 1)])
    if (
        set(matched_suffixes) != set(REQUIRED_CODE_BINDING_SUFFIXES)
        or len(matched_suffixes) != len(REQUIRED_CODE_BINDING_SUFFIXES)
        or len(roots) != 1
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "code bindings must be the exact formal code set under one repository root"
        )
    return rows


def _expected_source_bindings(
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows = [
        {
            "binding_id": "base_ontology",
            "byte_sha256": hashlib.sha256(
                canonical_json_bytes_v4_1(base_ontology)
            ).hexdigest(),
            "semantic_sha256": base_ontology["semantic_sha256"],
        },
        {
            "binding_id": "base_catalog",
            "byte_sha256": hashlib.sha256(
                canonical_json_bytes_v4_1(base_catalog)
            ).hexdigest(),
            "semantic_sha256": base_catalog["semantic_sha256"],
        },
    ]
    for filename in CANONICAL_ARTIFACT_FILENAMES:
        artifact = discovery_values[filename]
        semantic_field = SELF_HASH_FIELD_BY_FILENAME[filename]
        rows.append(
            {
                "binding_id": SOURCE_BINDING_ID_BY_DISCOVERY_FILENAME[filename],
                "byte_sha256": artifact_sha256_v4_1(artifact),
                "semantic_sha256": artifact[semantic_field],
            }
        )
    rows.sort(key=lambda row: row["binding_id"])
    return rows


def build_formal_catalog_source_bindings_v4_1(
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build the exact base2 plus accepted-Discovery8 provenance inventory."""

    try:
        ontology = validate_primitive_ontology_v4(base_ontology)
        catalog = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
        discovery = validate_discovery_bundle_v4_1(
            discovery_values,
            base_ontology=ontology,
            base_catalog=catalog,
        )
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"source binding dependency validation failed: {exc}"
        ) from exc
    return _expected_source_bindings(
        discovery_values=discovery,
        base_ontology=ontology,
        base_catalog=catalog,
    )


def _validate_required_source_bindings(
    value: Any,
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows = _normalize_source_bindings(value)
    expected = _expected_source_bindings(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    if rows != expected:
        mismatched = [
            expected_row["binding_id"]
            for row, expected_row in zip(rows, expected, strict=True)
            if row != expected_row
        ]
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "required materialization source binding mismatch: "
            + ",".join(mismatched)
        )
    return rows


def _expected_adapter_validation(
    artifacts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    normalized_artifacts = _validate_core_artifacts_for_manifest(artifacts)
    proof = normalized_artifacts[PRIMITIVE_MAPPING_PROOF_FILENAME]
    catalog = artifacts[FORMAL_CATALOG_FILENAME]
    mappings_by_name = {
        row["name"]: row for row in proof["new_candidate_mappings"]
    }
    descriptors: list[dict[str, Any]] = []
    for candidate in catalog["candidates"]:
        mapping = mappings_by_name.get(candidate["name"])
        if mapping is None:
            role = "base_reference"
            source_candidate_id = None
            source_definition_sha256 = candidate["definition_sha256"]
            mapping_proof_row_sha256 = None
            blockers = list(_ADAPTER_BASE_DESCRIPTOR_BLOCKERS)
        else:
            role = "new_classification"
            source_candidate_id = mapping["candidate_id"]
            source_definition_sha256 = mapping["source_definition_sha256"]
            mapping_proof_row_sha256 = mapping["mapping_semantic_sha256"]
            blockers = list(_ADAPTER_NEW_DESCRIPTOR_BLOCKERS)
        descriptor = {
            "schema_version": (
                "factor-governance-formal-catalog-candidate-descriptor.v4.1"
            ),
            "name": candidate["name"],
            "catalog_role": role,
            "catalog_definition_sha256": candidate["definition_sha256"],
            "catalog_implementation": candidate["implementation"],
            "descriptor_implementation": (
                "formal_catalog_definition_reference.validate_only.v4.1"
            ),
            "source_candidate_id": source_candidate_id,
            "source_definition_sha256": source_definition_sha256,
            "mapping_proof_row_sha256": mapping_proof_row_sha256,
            "classification_only": True,
            "executable": False,
            "screening_eligible": False,
            "runtime_equivalence_verified": False,
            "blockers": blockers,
        }
        descriptors.append(_seal(descriptor, "descriptor_semantic_sha256"))
    constants = {
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
    payload = {
        "schema_version": (
            "factor-governance-formal-catalog-adapter-validation.v4.1"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "base_ontology_sha256": proof["base_ontology_sha256"],
        "base_catalog_sha256": proof["base_catalog_sha256"],
        "formal_ontology_sha256": proof["formal_ontology_sha256"],
        "formal_catalog_sha256": proof["formal_catalog_sha256"],
        "mapping_proof_schema_version": PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
        "mapping_proof_sha256": proof["proof_semantic_sha256"],
        "candidate_count": len(catalog["candidates"]),
        "base_candidate_count": proof["catalog_base_candidate_count"],
        "new_candidate_count": proof["catalog_new_candidate_count"],
        "descriptor_count": len(descriptors),
        "ordered_candidate_names_semantic_sha256": semantic_sha256_v4_1(
            [candidate["name"] for candidate in catalog["candidates"]]
        ),
        "ordered_definition_pairs_semantic_sha256": semantic_sha256_v4_1(
            [
                {
                    "name": candidate["name"],
                    "definition_sha256": candidate["definition_sha256"],
                }
                for candidate in catalog["candidates"]
            ]
        ),
        **constants,
        "descriptors": descriptors,
        "blockers": list(_ADAPTER_VALIDATION_BLOCKERS),
    }
    return _seal(payload, "validation_semantic_sha256")


def _normalize_adapter_validation(
    value: Any,
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "adapter_validation must be an object or null"
        )
    payload = _exact(
        value, _ADAPTER_VALIDATION_ARTIFACT_FIELDS, "adapter validation"
    )
    schema = _text(payload["schema_version"], "adapter_validation.schema_version")
    if schema != "factor-governance-formal-catalog-adapter-validation.v4.1":
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "adapter validation schema mismatch"
        )
    semantic = _sha(
        payload["validation_semantic_sha256"],
        "adapter_validation.validation_semantic_sha256",
    )
    if semantic != semantic_sha256_v4_1(
        payload, exclude_fields=("validation_semantic_sha256",)
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "adapter validation semantic SHA mismatch"
        )
    if artifacts is None:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "adapter validation requires the exact four core artifacts"
        )
    expected = _expected_adapter_validation(artifacts)
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "adapter validation differs from exact validate-only recomputation"
        )
    for index, descriptor in enumerate(payload["descriptors"]):
        row = _exact(
            descriptor,
            _ADAPTER_CANDIDATE_DESCRIPTOR_FIELDS,
            f"adapter descriptors[{index}]",
        )
        _assert_self_hash(
            row,
            "descriptor_semantic_sha256",
            f"adapter descriptors[{index}]",
        )
    return {
        "schema_version": schema,
        "validation_semantic_sha256": semantic,
    }


def _semantic_field(filename: str, artifact: Mapping[str, Any]) -> str:
    fields = {
        PRIMITIVE_MAPPING_POLICY_FILENAME: "policy_semantic_sha256",
        PRIMITIVE_MAPPING_PROOF_FILENAME: "proof_semantic_sha256",
        FORMAL_ONTOLOGY_FILENAME: "semantic_sha256",
        FORMAL_CATALOG_FILENAME: "semantic_sha256",
    }
    return _sha(artifact[fields[filename]], f"{filename} semantic SHA")


def _artifact_bindings(
    artifacts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1]:
        artifact = artifacts[filename]
        rows.append(
            {
                "filename": filename,
                "byte_sha256": artifact_sha256_v4_1(artifact),
                "semantic_sha256": _semantic_field(filename, artifact),
                "size_bytes": len(canonical_file_bytes_v4_1(artifact)),
            }
        )
    return rows


def _authority_fields() -> dict[str, Any]:
    return {
        "classification_only": True,
        "runtime_equivalence_claimed": False,
        "screening_eligible": False,
        "proposal_eligible": False,
        "registry_entry_created": False,
        "initial_weight_policy": INITIAL_WEIGHT_POLICY,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
    }


def _validate_core_artifacts_for_manifest(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_names = set(FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1])
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest core input must contain exactly the four non-manifest artifacts"
        )
    try:
        formal_ontology = validate_primitive_ontology_v4(
            artifacts[FORMAL_ONTOLOGY_FILENAME]
        )
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"manifest formal ontology validation failed: {exc}"
        ) from exc
    if len(formal_ontology["primitives"]) != EXPECTED_FORMAL_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest formal ontology must contain exactly 18 primitives"
        )
    primitive_by_id = {
        row["primitive_id"]: row for row in formal_ontology["primitives"]
    }
    if any(
        primitive_by_id.get(primitive_id)
        != {"primitive_id": primitive_id, "family": primitive_id}
        for primitive_id in NEW_PRIMITIVES
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest formal ontology five-row extension drifted"
        )
    base_rows = [
        row
        for row in formal_ontology["primitives"]
        if row["primitive_id"] not in NEW_PRIMITIVES
    ]
    if len(base_rows) != EXPECTED_BASE_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest-derived base ontology must contain exactly 13 primitives"
        )
    base_ontology = build_primitive_ontology_v4(base_rows)
    policy = validate_primitive_mapping_policy_v4_1(
        artifacts[PRIMITIVE_MAPPING_POLICY_FILENAME],
        base_ontology=base_ontology,
    )
    try:
        formal_catalog = validate_candidate_catalog_v4(
            artifacts[FORMAL_CATALOG_FILENAME], ontology=formal_ontology
        )
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"manifest formal catalog validation failed: {exc}"
        ) from exc
    if len(formal_catalog["candidates"]) != EXPECTED_FORMAL_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest formal catalog must contain exactly 267 candidates"
        )

    proof = _exact(
        artifacts[PRIMITIVE_MAPPING_PROOF_FILENAME],
        _PROOF_FIELDS,
        "manifest primitive mapping proof",
    )
    _assert_self_hash(
        proof, "proof_semantic_sha256", "manifest primitive mapping proof"
    )
    if (
        proof["schema_version"] != PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION
        or proof["protocol_version"] != PROTOCOL_VERSION
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest primitive mapping proof schema/protocol mismatch"
        )
    _text(proof["cycle_id"], "manifest primitive mapping proof cycle_id")
    expected_counts = {
        "source_candidate_count": EXPECTED_SOURCE_CANDIDATE_COUNT,
        "new_candidate_count": EXPECTED_NEW_CANDIDATE_COUNT,
        "structural_alias_count": EXPECTED_STRUCTURAL_ALIAS_COUNT,
        "incompatible_count": EXPECTED_INCOMPATIBLE_COUNT,
        "catalog_base_candidate_count": EXPECTED_BASE_CANDIDATE_COUNT,
        "catalog_new_candidate_count": EXPECTED_NEW_CANDIDATE_COUNT,
        "catalog_total_candidate_count": EXPECTED_FORMAL_CANDIDATE_COUNT,
    }
    for field, expected in expected_counts.items():
        if type(proof[field]) is not int or proof[field] != expected:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest primitive mapping proof count mismatch: {field}"
            )
    expected_authority = _authority_fields()
    for field, expected in expected_authority.items():
        if proof[field] != expected or type(proof[field]) is not type(expected):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest primitive mapping proof authority drift: {field}"
            )
    expected_links = {
        "base_ontology_sha256": base_ontology["semantic_sha256"],
        "mapping_policy_sha256": policy["policy_semantic_sha256"],
        "formal_ontology_sha256": formal_ontology["semantic_sha256"],
        "formal_catalog_sha256": formal_catalog["semantic_sha256"],
    }
    for field, expected in expected_links.items():
        if proof[field] != expected:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest primitive mapping proof link mismatch: {field}"
            )

    mappings: list[dict[str, Any]] = []
    for index, raw in enumerate(proof["new_candidate_mappings"]):
        row = _exact(raw, _NEW_MAPPING_FIELDS, f"manifest mappings[{index}]")
        _assert_self_hash(
            row, "mapping_semantic_sha256", f"manifest mappings[{index}]"
        )
        if (
            row["implementation"] != CLASSIFICATION_IMPLEMENTATION
            or row["mapping_status"] != MAPPING_STATUS_COMPLETE
            or row["occurrence_count"] != len(row["occurrences"])
            or type(row["occurrence_count"]) is not int
            or row["occurrence_count"] <= 0
        ):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest mapping row contract drift: {row.get('name')}"
            )
        primitive_ids = _sorted_distinct_texts(
            row["primitive_ids"], f"manifest mappings[{index}].primitive_ids"
        )
        input_fields = _sorted_distinct_texts(
            row["input_fields"], f"manifest mappings[{index}].input_fields"
        )
        if not primitive_ids or not input_fields:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "manifest mapping primitive_ids/input_fields must be non-empty"
            )
        for occurrence_index, raw_occurrence in enumerate(row["occurrences"]):
            occurrence = _exact(
                raw_occurrence,
                _OCCURRENCE_FIELDS,
                f"manifest mappings[{index}].occurrences[{occurrence_index}]",
            )
            _text(occurrence["node_path"], "manifest occurrence node_path")
            _nonnegative_integer(
                occurrence["node_occurrence_index"],
                "manifest occurrence node_occurrence_index",
            )
            _nonnegative_integer(
                occurrence["rule_occurrence_index"],
                "manifest occurrence rule_occurrence_index",
            )
            _positive_integer(
                occurrence["match_cardinality"],
                "manifest occurrence match_cardinality",
            )
            _sha(occurrence["subtree_sha256"], "manifest occurrence subtree SHA")
            consumed = occurrence["consumed_identifiers"]
            if not isinstance(consumed, list) or not consumed:
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    "manifest occurrence consumed_identifiers must be non-empty"
                )
            for consumed_index, consumed_raw in enumerate(consumed):
                consumed_row = _exact(
                    consumed_raw,
                    _CONSUMED_IDENTIFIER_FIELDS,
                    f"manifest consumed identifiers[{consumed_index}]",
                )
                _text(consumed_row["identifier"], "manifest consumed identifier")
                _text(consumed_row["node_path"], "manifest consumed node_path")
                _nonnegative_integer(
                    consumed_row["node_occurrence_index"],
                    "manifest consumed node_occurrence_index",
                )
        recomputed = build_candidate_primitive_proof_v4_1(
            expression=_text(
                row["expression"], f"manifest mappings[{index}].expression"
            ),
            input_fields=input_fields,
            mapping_policy=policy,
        )
        expected_recomputed_fields = {
            "full_candidate_normalized_ast_sha256": row[
                "full_candidate_normalized_ast_sha256"
            ],
            "primitive_ids": primitive_ids,
            "occurrence_count": row["occurrence_count"],
            "occurrences": row["occurrences"],
            "mapping_status": row["mapping_status"],
        }
        if canonical_json_bytes_v4_1(recomputed) != canonical_json_bytes_v4_1(
            expected_recomputed_fields
        ):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest mapping AST proof differs from exact recomputation: {row['name']}"
            )
        mappings.append(copy.deepcopy(row))
    aliases: list[dict[str, Any]] = []
    for index, raw in enumerate(proof["structural_aliases"]):
        row = _exact(raw, _ALIAS_FIELDS, f"manifest aliases[{index}]")
        _assert_self_hash(row, "alias_semantic_sha256", f"manifest aliases[{index}]")
        if row["target_match_cardinality"] != 1 or row["excluded_from_catalog"] is not True:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "manifest alias exclusion/cardinality drift"
            )
        aliases.append(copy.deepcopy(row))
    incompatible: list[dict[str, Any]] = []
    for index, raw in enumerate(proof["incompatible_candidates"]):
        row = _exact(raw, _INCOMPATIBLE_FIELDS, f"manifest incompatible[{index}]")
        _assert_self_hash(
            row,
            "incompatible_semantic_sha256",
            f"manifest incompatible[{index}]",
        )
        reasons = row["incompatibility_reasons"]
        records = row["reason_records"]
        if not isinstance(reasons, list) or not reasons or not isinstance(records, list):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "manifest incompatible reasons must be non-empty lists"
            )
        normalized_records = [
            _exact(record, _REASON_RECORD_FIELDS, "manifest incompatible reason")
            for record in records
        ]
        if [record["reason"] for record in normalized_records] != reasons:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "manifest incompatible reason records drifted"
            )
        if row["excluded_from_catalog"] is not True or row["mapping_not_attempted"] is not True:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "manifest incompatible exclusion state drifted"
            )
        incompatible.append(copy.deepcopy(row))
    if (
        len(mappings) != EXPECTED_NEW_CANDIDATE_COUNT
        or len(aliases) != EXPECTED_STRUCTURAL_ALIAS_COUNT
        or len(incompatible) != EXPECTED_INCOMPATIBLE_COUNT
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest primitive mapping proof row counts drifted"
        )
    for rows, label in (
        (mappings, "mappings"),
        (aliases, "aliases"),
        (incompatible, "incompatible"),
    ):
        ids = [row["candidate_id"] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest proof {label} candidate ids must be sorted and unique"
            )
    source_ids = sorted(
        [row["candidate_id"] for row in mappings]
        + [row["candidate_id"] for row in aliases]
        + [row["candidate_id"] for row in incompatible]
    )
    if (
        len(source_ids) != EXPECTED_SOURCE_CANDIDATE_COUNT
        or len(source_ids) != len(set(source_ids))
        or proof["source_candidate_ids_sha256"]
        != semantic_sha256_v4_1(source_ids)
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest proof source candidate accounting drifted"
        )
    catalog_by_name = {
        row["name"]: row for row in formal_catalog["candidates"]
    }
    mapping_names = {row["name"] for row in mappings}
    classification_names = {
        row["name"]
        for row in formal_catalog["candidates"]
        if row["implementation"] == CLASSIFICATION_IMPLEMENTATION
    }
    if mapping_names != classification_names:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest catalog classification membership differs from proof mappings"
        )
    for mapping in mappings:
        candidate = catalog_by_name.get(mapping["name"])
        if (
            candidate is None
            or candidate["definition_sha256"]
            != mapping["catalog_definition_sha256"]
            or candidate["expression"] != mapping["expression"]
            or candidate["input_fields"] != mapping["input_fields"]
            or candidate["primitive_ids"] != mapping["primitive_ids"]
            or candidate["family"] != mapping["family"]
            or candidate["slot"] != mapping["slot"]
        ):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest catalog/mapping row drift: {mapping['name']}"
            )
    base_candidates = [
        _candidate_input_from_catalog_row(row)
        for row in formal_catalog["candidates"]
        if row["name"] not in mapping_names
    ]
    if len(base_candidates) != EXPECTED_BASE_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest-derived base catalog must contain exactly 230 candidates"
        )
    derived_base_catalog = build_candidate_catalog_v4(
        ontology=base_ontology,
        candidates=base_candidates,
    )
    if derived_base_catalog["semantic_sha256"] != proof["base_catalog_sha256"]:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest formal catalog does not preserve the proof-bound base catalog"
        )
    normalized_proof = copy.deepcopy(dict(proof))
    return {
        PRIMITIVE_MAPPING_POLICY_FILENAME: policy,
        PRIMITIVE_MAPPING_PROOF_FILENAME: normalized_proof,
        FORMAL_ONTOLOGY_FILENAME: formal_ontology,
        FORMAL_CATALOG_FILENAME: formal_catalog,
    }


def _assert_exact_counts(
    *, audit: Mapping[str, Any], discovery_catalog: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ideas = list(audit["ideas"])
    candidate_ids = [row["candidate_id"] for row in ideas]
    if len(ideas) != EXPECTED_SOURCE_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source audit must account for exactly 100 candidates"
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source candidate ids must be unique"
        )
    new_rows = [
        row
        for row in ideas
        if row["compatibility_status"] == "compatible"
        and row["catalog_role"] == "new_candidate"
        and row["selected"] is True
    ]
    alias_rows = [
        row
        for row in ideas
        if row["compatibility_status"] == "compatible"
        and row["catalog_role"] == "structural_alias"
        and row["selected"] is False
    ]
    incompatible_rows = [
        row
        for row in ideas
        if row["compatibility_status"] == "incompatible"
        and row["catalog_role"] == "incompatible"
        and row["selected"] is False
    ]
    classified_ids = [
        row["candidate_id"] for row in new_rows + alias_rows + incompatible_rows
    ]
    if (
        len(new_rows) != EXPECTED_NEW_CANDIDATE_COUNT
        or len(alias_rows) != EXPECTED_STRUCTURAL_ALIAS_COUNT
        or len(incompatible_rows) != EXPECTED_INCOMPATIBLE_COUNT
        or len(classified_ids) != EXPECTED_SOURCE_CANDIDATE_COUNT
        or len(classified_ids) != len(set(classified_ids))
        or set(classified_ids) != set(candidate_ids)
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "source candidate accounting must be exactly 100=37+6+57"
        )
    aquant_members = [
        row for row in discovery_catalog["members"] if row["origin"] == "aquant"
    ]
    expected_member_ids = {
        row["candidate_id"] for row in new_rows + alias_rows
    }
    if {row["candidate_id"] for row in aquant_members} != expected_member_ids:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "discovery catalog compatible membership differs from source accounting"
        )
    return new_rows, alias_rows, incompatible_rows


def _build_mapping_proof(
    *,
    audit: Mapping[str, Any],
    discovery_catalog: Mapping[str, Any],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    candidate_proofs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    new_rows, alias_rows, incompatible_rows = _assert_exact_counts(
        audit=audit, discovery_catalog=discovery_catalog
    )
    discovery_by_id = {
        row["candidate_id"]: row for row in discovery_catalog["members"]
    }
    formal_by_name = {row["name"]: row for row in formal_catalog["candidates"]}
    base_by_name = {row["name"]: row for row in base_catalog["candidates"]}
    base_member_by_id = {
        row["candidate_id"]: row
        for row in discovery_catalog["members"]
        if row["catalog_role"] == "base_reference"
    }

    mappings: list[dict[str, Any]] = []
    for idea in sorted(new_rows, key=lambda row: row["candidate_id"]):
        member = discovery_by_id[idea["candidate_id"]]
        catalog_row = formal_by_name[idea["name"]]
        classified = candidate_proofs[idea["candidate_id"]]
        row = {
            "candidate_id": idea["candidate_id"],
            "name": idea["name"],
            "source_definition_sha256": member["source_definition_sha256"],
            "catalog_definition_sha256": catalog_row["definition_sha256"],
            "implementation": CLASSIFICATION_IMPLEMENTATION,
            "expression": idea["expression"],
            "full_candidate_normalized_ast_sha256": classified[
                "full_candidate_normalized_ast_sha256"
            ],
            "input_fields": list(idea["input_fields"]),
            "primitive_ids": list(classified["primitive_ids"]),
            "family": catalog_row["family"],
            "slot": catalog_row["slot"],
            "occurrence_count": classified["occurrence_count"],
            "occurrences": copy.deepcopy(classified["occurrences"]),
            "mapping_status": MAPPING_STATUS_COMPLETE,
        }
        mappings.append(_seal(row, "mapping_semantic_sha256"))

    aliases: list[dict[str, Any]] = []
    alias_targets: list[str] = []
    for idea in sorted(alias_rows, key=lambda row: row["candidate_id"]):
        member = discovery_by_id[idea["candidate_id"]]
        target_id = idea["structural_alias_of"]
        matches = [
            row for candidate_id, row in base_member_by_id.items() if candidate_id == target_id
        ]
        if len(matches) != 1:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"structural alias target cardinality is not one: {idea['candidate_id']}"
            )
        target_member = matches[0]
        target = base_by_name.get(target_member["name"])
        if (
            target is None
            or target["definition_sha256"] != target_member["source_definition_sha256"]
        ):
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"structural alias target definition drifted: {idea['candidate_id']}"
            )
        alias_targets.append(target_id)
        row = {
            "candidate_id": idea["candidate_id"],
            "name": idea["name"],
            "source_definition_sha256": member["source_definition_sha256"],
            "structural_fingerprint_sha256": idea["structural_fingerprint_sha256"],
            "target_candidate_id": target_id,
            "target_name": target["name"],
            "target_definition_sha256": target["definition_sha256"],
            "target_match_cardinality": 1,
            "excluded_from_catalog": True,
        }
        aliases.append(_seal(row, "alias_semantic_sha256"))
    if len(alias_targets) != len(set(alias_targets)):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "structural aliases must bind distinct base targets"
        )

    incompatible: list[dict[str, Any]] = []
    for idea in sorted(incompatible_rows, key=lambda row: row["candidate_id"]):
        reasons = list(idea["incompatibility_reasons"])
        reason_records = [
            {"code": reason.split(":", 1)[0], "reason": reason}
            for reason in reasons
        ]
        row = {
            "source_index": idea["source_index"],
            "candidate_id": idea["candidate_id"],
            "name": idea["name"],
            "incompatibility_reasons": reasons,
            "reason_records": reason_records,
            "excluded_from_catalog": True,
            "mapping_not_attempted": True,
        }
        incompatible.append(_seal(row, "incompatible_semantic_sha256"))

    payload = {
        "schema_version": PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": discovery_catalog["cycle_id"],
        "source_idea_audit_sha256": audit["audit_semantic_sha256"],
        "discovery_catalog_sha256": discovery_catalog["catalog_semantic_sha256"],
        "base_ontology_sha256": base_ontology["semantic_sha256"],
        "base_catalog_sha256": base_catalog["semantic_sha256"],
        "mapping_policy_sha256": policy["policy_semantic_sha256"],
        "formal_ontology_sha256": formal_ontology["semantic_sha256"],
        "formal_catalog_sha256": formal_catalog["semantic_sha256"],
        "source_candidate_count": EXPECTED_SOURCE_CANDIDATE_COUNT,
        "new_candidate_count": EXPECTED_NEW_CANDIDATE_COUNT,
        "structural_alias_count": EXPECTED_STRUCTURAL_ALIAS_COUNT,
        "incompatible_count": EXPECTED_INCOMPATIBLE_COUNT,
        "catalog_base_candidate_count": EXPECTED_BASE_CANDIDATE_COUNT,
        "catalog_new_candidate_count": EXPECTED_NEW_CANDIDATE_COUNT,
        "catalog_total_candidate_count": EXPECTED_FORMAL_CANDIDATE_COUNT,
        "source_candidate_ids_sha256": semantic_sha256_v4_1(
            sorted(row["candidate_id"] for row in audit["ideas"])
        ),
        "new_candidate_mappings": mappings,
        "structural_aliases": aliases,
        "incompatible_candidates": incompatible,
        **_authority_fields(),
    }
    return _seal(payload, "proof_semantic_sha256")


def _build_manifest(
    *,
    cycle_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any,
) -> dict[str, Any]:
    normalized_artifacts = _validate_core_artifacts_for_manifest(artifacts)
    proof = normalized_artifacts[PRIMITIVE_MAPPING_PROOF_FILENAME]
    if cycle_id != proof.get("cycle_id"):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest cycle_id must equal the primitive mapping proof cycle_id"
        )
    expected_core_links = {
        "mapping_policy_sha256": normalized_artifacts[PRIMITIVE_MAPPING_POLICY_FILENAME].get(
            "policy_semantic_sha256"
        ),
        "formal_ontology_sha256": normalized_artifacts[FORMAL_ONTOLOGY_FILENAME].get(
            "semantic_sha256"
        ),
        "formal_catalog_sha256": normalized_artifacts[FORMAL_CATALOG_FILENAME].get(
            "semantic_sha256"
        ),
    }
    for field, expected in expected_core_links.items():
        if proof.get(field) != expected:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest core artifact binding mismatch: {field}"
            )
    normalized_sources = _normalize_source_bindings(source_bindings)
    source_semantics = {
        row["binding_id"]: row["semantic_sha256"] for row in normalized_sources
    }
    expected_source_links = {
        "base_ontology_sha256": source_semantics["base_ontology"],
        "base_catalog_sha256": source_semantics["base_catalog"],
        "source_idea_audit_sha256": source_semantics["source_idea_audit"],
        "discovery_catalog_sha256": source_semantics["discovery_catalog"],
    }
    for field, expected in expected_source_links.items():
        if proof.get(field) != expected:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"manifest proof/source binding mismatch: {field}"
            )
    normalized_code = _normalize_code_bindings(code_bindings)
    adapter_binding = _normalize_adapter_validation(
        adapter_validation, artifacts=normalized_artifacts
    )
    side_effects = {field: False for field in sorted(_SIDE_EFFECT_FIELDS)}
    payload = {
        "schema_version": FORMAL_CATALOG_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "artifact_bindings": _artifact_bindings(normalized_artifacts),
        "source_bindings": normalized_sources,
        "source_bindings_semantic_sha256": semantic_sha256_v4_1(normalized_sources),
        "code_bindings": normalized_code,
        "code_bindings_semantic_sha256": semantic_sha256_v4_1(normalized_code),
        "adapter_validation_binding": adapter_binding,
        "adapter_validation_status": "bound" if adapter_binding else "not_bound",
        "source_candidate_count": EXPECTED_SOURCE_CANDIDATE_COUNT,
        "base_candidate_count": EXPECTED_BASE_CANDIDATE_COUNT,
        "new_candidate_count": EXPECTED_NEW_CANDIDATE_COUNT,
        "structural_alias_count": EXPECTED_STRUCTURAL_ALIAS_COUNT,
        "incompatible_count": EXPECTED_INCOMPATIBLE_COUNT,
        "formal_candidate_count": EXPECTED_FORMAL_CANDIDATE_COUNT,
        "base_primitive_count": EXPECTED_BASE_PRIMITIVE_COUNT,
        "formal_primitive_count": EXPECTED_FORMAL_PRIMITIVE_COUNT,
        **_authority_fields(),
        "qualification": False,
        "new_risk_authorized": False,
        "side_effect_scope": "pure_materializer_build_only",
        "side_effects": side_effects,
    }
    return _seal(payload, "manifest_semantic_sha256")


def rebuild_formal_catalog_materialization_manifest_v4_1(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    cycle_id: str,
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any = None,
) -> dict[str, Any]:
    """Rebuild the manifest after a validate-only adapter artifact exists.

    ``artifacts`` must be exactly the four non-manifest core artifacts.  This
    helper remains pure; its all-false side-effect record is explicitly scoped
    only to the materializer build, not to a later private publisher.
    """

    expected_filenames = set(FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1])
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_filenames:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "manifest rebuild requires exactly the four non-manifest artifacts"
        )
    return _build_manifest(
        cycle_id=_text(cycle_id, "cycle_id"),
        artifacts=artifacts,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=adapter_validation,
    )


def _materialize(
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any,
) -> dict[str, dict[str, Any]]:
    try:
        ontology = validate_primitive_ontology_v4(base_ontology)
        catalog = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
        discovery = validate_discovery_bundle_v4_1(
            discovery_values,
            base_ontology=ontology,
            base_catalog=catalog,
        )
    except FactorGovernanceFormalCatalogMaterializationV4_1Error:
        raise
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"source bundle validation failed: {exc}"
        ) from exc
    if len(ontology["primitives"]) != EXPECTED_BASE_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "base ontology must contain exactly 13 primitives"
        )
    if len(catalog["candidates"]) != EXPECTED_BASE_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "base catalog must contain exactly 230 candidates"
        )
    audit = discovery[SOURCE_IDEA_AUDIT_FILENAME]
    discovery_catalog = discovery[DISCOVERY_CATALOG_FILENAME]
    _validate_required_source_bindings(
        source_bindings,
        discovery_values=discovery,
        base_ontology=ontology,
        base_catalog=catalog,
    )
    new_rows, _aliases, _incompatible = _assert_exact_counts(
        audit=audit, discovery_catalog=discovery_catalog
    )

    policy = build_primitive_mapping_policy_v4_1(base_ontology=ontology)
    member_by_id = {
        row["candidate_id"]: row for row in discovery_catalog["members"]
    }
    candidate_proofs: dict[str, dict[str, Any]] = {}
    new_candidate_inputs: list[dict[str, Any]] = []
    for idea in new_rows:
        member = member_by_id.get(idea["candidate_id"])
        if member is None:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"selected source candidate is absent from discovery catalog: {idea['candidate_id']}"
            )
        classified = build_candidate_primitive_proof_v4_1(
            expression=idea["expression"],
            input_fields=idea["input_fields"],
            mapping_policy=policy,
        )
        candidate_proofs[idea["candidate_id"]] = classified
        new_candidate_inputs.append(
            _new_candidate_input(member, classified["primitive_ids"])
        )

    formal_primitives = [copy.deepcopy(row) for row in ontology["primitives"]]
    formal_primitives.extend(
        {"primitive_id": primitive_id, "family": primitive_id}
        for primitive_id in NEW_PRIMITIVES
    )
    formal_ontology = build_primitive_ontology_v4(formal_primitives)
    if len(formal_ontology["primitives"]) != EXPECTED_FORMAL_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal ontology must contain exactly 18 primitives"
        )
    formal_primitive_by_id = {
        row["primitive_id"]: row for row in formal_ontology["primitives"]
    }
    if any(
        formal_primitive_by_id.get(row["primitive_id"]) != row
        for row in ontology["primitives"]
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal ontology does not preserve the exact base primitive rows"
        )

    base_inputs = [
        _candidate_input_from_catalog_row(row) for row in catalog["candidates"]
    ]
    formal_catalog = build_candidate_catalog_v4(
        ontology=formal_ontology,
        candidates=base_inputs + new_candidate_inputs,
    )
    if len(formal_catalog["candidates"]) != EXPECTED_FORMAL_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal catalog must contain exactly 267 candidates"
        )
    final_by_name = {row["name"]: row for row in formal_catalog["candidates"]}
    for base_row in catalog["candidates"]:
        if final_by_name.get(base_row["name"]) != base_row:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"base catalog definition drifted: {base_row['name']}"
            )
    expected_new_names = {row["name"] for row in new_rows}
    if set(final_by_name) - {row["name"] for row in catalog["candidates"]} != expected_new_names:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal catalog new membership differs from selected 37"
        )

    proof = _build_mapping_proof(
        audit=audit,
        discovery_catalog=discovery_catalog,
        base_ontology=ontology,
        base_catalog=catalog,
        policy=policy,
        formal_ontology=formal_ontology,
        formal_catalog=formal_catalog,
        candidate_proofs=candidate_proofs,
    )
    artifacts: dict[str, dict[str, Any]] = {
        PRIMITIVE_MAPPING_POLICY_FILENAME: policy,
        PRIMITIVE_MAPPING_PROOF_FILENAME: proof,
        FORMAL_ONTOLOGY_FILENAME: formal_ontology,
        FORMAL_CATALOG_FILENAME: formal_catalog,
    }
    artifacts[FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME] = _build_manifest(
        cycle_id=discovery_catalog["cycle_id"],
        artifacts=artifacts,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=adapter_validation,
    )
    return artifacts


def build_formal_catalog_materialization_v4_1(
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build all five formal materialization artifacts, or fail with no output."""

    return _materialize(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=adapter_validation,
    )


def validate_primitive_mapping_proof_v4_1(
    value: Mapping[str, Any],
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    mapping_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a proof by recomputing its complete source mapping."""

    payload = _exact(value, _PROOF_FIELDS, "primitive mapping proof")
    _assert_self_hash(payload, "proof_semantic_sha256", "primitive mapping proof")
    try:
        ontology = validate_primitive_ontology_v4(base_ontology)
        catalog = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
        normalized_discovery = validate_discovery_bundle_v4_1(
            discovery_values, base_ontology=ontology, base_catalog=catalog
        )
        normalized_formal_ontology = validate_formal_ontology_v4_1(
            formal_ontology, base_ontology=ontology
        )
        normalized_formal_catalog = validate_formal_catalog_v4_1(
            formal_catalog,
            formal_ontology=normalized_formal_ontology,
            base_ontology=ontology,
            base_catalog=catalog,
            discovery_values=normalized_discovery,
        )
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"mapping proof dependency validation failed: {exc}"
        ) from exc
    policy = validate_primitive_mapping_policy_v4_1(
        mapping_policy, base_ontology=ontology
    )
    audit = normalized_discovery[SOURCE_IDEA_AUDIT_FILENAME]
    discovery_catalog = normalized_discovery[DISCOVERY_CATALOG_FILENAME]
    new_rows, _aliases, _incompatible = _assert_exact_counts(
        audit=audit, discovery_catalog=discovery_catalog
    )
    candidate_proofs = {
        row["candidate_id"]: build_candidate_primitive_proof_v4_1(
            expression=row["expression"],
            input_fields=row["input_fields"],
            mapping_policy=policy,
        )
        for row in new_rows
    }
    expected = _build_mapping_proof(
        audit=audit,
        discovery_catalog=discovery_catalog,
        base_ontology=ontology,
        base_catalog=catalog,
        policy=policy,
        formal_ontology=normalized_formal_ontology,
        formal_catalog=normalized_formal_catalog,
        candidate_proofs=candidate_proofs,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "primitive mapping proof differs from exact recomputation"
        )
    return expected


def validate_formal_ontology_v4_1(
    value: Mapping[str, Any], *, base_ontology: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact 13-row base plus five-row primitive extension."""

    try:
        base = validate_primitive_ontology_v4(base_ontology)
        formal = validate_primitive_ontology_v4(value)
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"formal ontology validation failed: {exc}"
        ) from exc
    if len(base["primitives"]) != EXPECTED_BASE_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "base ontology must contain exactly 13 primitives"
        )
    expected = build_primitive_ontology_v4(
        [*base["primitives"]]
        + [
            {"primitive_id": primitive_id, "family": primitive_id}
            for primitive_id in NEW_PRIMITIVES
        ]
    )
    if formal != expected or len(formal["primitives"]) != EXPECTED_FORMAL_PRIMITIVE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal ontology is not the exact 13+5 extension"
        )
    return formal


def validate_formal_catalog_v4_1(
    value: Mapping[str, Any],
    *,
    formal_ontology: Mapping[str, Any],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    discovery_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate exact base preservation and classification-only new rows."""

    ontology = validate_formal_ontology_v4_1(
        formal_ontology, base_ontology=base_ontology
    )
    try:
        base = validate_candidate_catalog_v4(base_catalog, ontology=base_ontology)
        formal = validate_candidate_catalog_v4(value, ontology=ontology)
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            f"formal catalog validation failed: {exc}"
        ) from exc
    if len(base["candidates"]) != EXPECTED_BASE_CANDIDATE_COUNT or len(
        formal["candidates"]
    ) != EXPECTED_FORMAL_CANDIDATE_COUNT:
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal catalog count must be exact base230+new37=267"
        )
    base_by_name = {row["name"]: row for row in base["candidates"]}
    formal_by_name = {row["name"]: row for row in formal["candidates"]}
    for name, row in base_by_name.items():
        if formal_by_name.get(name) != row:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"formal catalog base definition drifted: {name}"
            )
    new_rows = [row for row in formal["candidates"] if row["name"] not in base_by_name]
    if len(new_rows) != EXPECTED_NEW_CANDIDATE_COUNT or any(
        row["implementation"] != CLASSIFICATION_IMPLEMENTATION
        or row["params"] != {}
        or row["direction"] != 1.0
        or row["slot"] != "primitive:" + "+".join(row["primitive_ids"])
        for row in new_rows
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal catalog new rows violate classification identity"
        )
    if discovery_values is not None:
        try:
            discovery = validate_discovery_bundle_v4_1(
                discovery_values,
                base_ontology=base_ontology,
                base_catalog=base,
            )
        except (TypeError, ValueError) as exc:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                f"formal catalog discovery validation failed: {exc}"
            ) from exc
        audit = discovery[SOURCE_IDEA_AUDIT_FILENAME]
        discovery_catalog = discovery[DISCOVERY_CATALOG_FILENAME]
        selected, _aliases, _incompatible = _assert_exact_counts(
            audit=audit, discovery_catalog=discovery_catalog
        )
        selected_by_name = {row["name"]: row for row in selected}
        if set(selected_by_name) != {row["name"] for row in new_rows}:
            raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                "formal catalog new membership differs from selected 37"
            )
        for row in new_rows:
            source = selected_by_name[row["name"]]
            if (
                row["expression"] != source["expression"]
                or row["lookback"] != source["lookback"]
                or row["input_fields"] != source["input_fields"]
            ):
                raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
                    f"formal catalog source identity drifted: {row['name']}"
                )
    return formal


def validate_formal_catalog_materialization_manifest_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any = None,
) -> dict[str, Any]:
    payload = _exact(value, _MANIFEST_FIELDS, "formal materialization manifest")
    _assert_self_hash(
        payload, "manifest_semantic_sha256", "formal materialization manifest"
    )
    for index, raw in enumerate(payload["artifact_bindings"]):
        _exact(raw, _ARTIFACT_BINDING_FIELDS, f"artifact_bindings[{index}]")
    for index, raw in enumerate(payload["source_bindings"]):
        _exact(raw, _SOURCE_BINDING_FIELDS, f"source_bindings[{index}]")
    for index, raw in enumerate(payload["code_bindings"]):
        _exact(raw, _CODE_BINDING_FIELDS, f"code_bindings[{index}]")
    if payload["adapter_validation_binding"] is not None:
        _exact(
            payload["adapter_validation_binding"],
            _ADAPTER_BINDING_FIELDS,
            "adapter_validation_binding",
        )
    side_effects = _exact(payload["side_effects"], _SIDE_EFFECT_FIELDS, "side_effects")
    if any(value is not False for value in side_effects.values()):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "all materialization side effects must be false"
        )
    expected = _build_manifest(
        cycle_id=payload["cycle_id"],
        artifacts=artifacts,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=adapter_validation,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal materialization manifest differs from exact recomputation"
        )
    return expected


def validate_formal_catalog_materialization_v4_1(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_bindings: Any,
    code_bindings: Any,
    adapter_validation: Any = None,
) -> dict[str, dict[str, Any]]:
    """Cross-validate exact five-file materialization by full recomputation."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        FORMAL_CATALOG_MATERIALIZATION_FILENAMES
    ):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal materialization must contain exactly the five canonical artifacts"
        )
    expected = _materialize(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        source_bindings=source_bindings,
        code_bindings=code_bindings,
        adapter_validation=adapter_validation,
    )
    if canonical_json_bytes_v4_1(artifacts) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogMaterializationV4_1Error(
            "formal materialization differs from exact all-or-nothing recomputation"
        )
    return expected


__all__ = [
    "CLASSIFICATION_IMPLEMENTATION",
    "FORMAL_CATALOG_FILENAME",
    "FORMAL_CATALOG_MATERIALIZATION_FILENAMES",
    "FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME",
    "FORMAL_CATALOG_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "FORMAL_ONTOLOGY_FILENAME",
    "FactorGovernanceFormalCatalogMaterializationV4_1Error",
    "INITIAL_WEIGHT_POLICY",
    "MAPPING_STATUS_COMPLETE",
    "NEW_PRIMITIVES",
    "PRIMITIVE_MAPPING_POLICY_FILENAME",
    "PRIMITIVE_MAPPING_POLICY_SCHEMA_VERSION",
    "PRIMITIVE_MAPPING_PROOF_FILENAME",
    "PRIMITIVE_MAPPING_PROOF_SCHEMA_VERSION",
    "REQUIRED_CODE_BINDING_SUFFIXES",
    "REQUIRED_SOURCE_BINDING_IDS",
    "SOURCE_BINDING_ID_BY_DISCOVERY_FILENAME",
    "artifact_sha256_v4_1",
    "build_candidate_primitive_proof_v4_1",
    "build_formal_catalog_materialization_v4_1",
    "build_formal_catalog_source_bindings_v4_1",
    "build_primitive_mapping_policy_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "classify_normalized_ast_occurrences_v4_1",
    "rebuild_formal_catalog_materialization_manifest_v4_1",
    "semantic_sha256_v4_1",
    "validate_formal_catalog_v4_1",
    "validate_formal_catalog_materialization_manifest_v4_1",
    "validate_formal_catalog_materialization_v4_1",
    "validate_formal_ontology_v4_1",
    "validate_primitive_mapping_policy_v4_1",
    "validate_primitive_mapping_proof_v4_1",
]
