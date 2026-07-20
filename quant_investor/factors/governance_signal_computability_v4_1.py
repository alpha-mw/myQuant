"""Fail-closed Factor v4.1 exact-37 signal-computability evidence.

This contract proves only that the pinned A_quant data objects can be
interpreted by the pinned MatrixDataset transformations and evaluated under
the accepted myQuant PIT envelope.  It deliberately cannot establish data
freshness, same-snapshot screening, Factor admission, production, portfolio,
or new-risk authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from typing import Any

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
from quant_investor.factors import governance_private_bundle_io as private_io


PROTOCOL_VERSION = "v4.1"
EXPECTED_CYCLE_ID = "cn_full_a_v4_1_20260717"
PINNED_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
CLAIM_SCOPE = "pinned_aquant_git_data_exact37_source_semantic_computability.v1"
READINESS = "EXPLORATORY_PINNED_SOURCE_SEMANTIC_COMPUTABILITY_ONLY"

SEMANTICS_SCHEMA_VERSION = "factor-governance-aquant-input-semantics-receipt.v4.1"
PROOF_SCHEMA_VERSION = "factor-governance-exact37-signal-computability.v4.1"
READBACK_SCHEMA_VERSION = "factor-governance-signal-computability-readback.v4.1"
SEMANTICS_FILENAME = "aquant_input_semantics_receipt.v4_1.json"
PROOF_FILENAME = "exact_37_signal_computability.v4_1.json"
READBACK_FILENAME = "signal_computability_readback.v4_1.json"
BUNDLE_INPUT_FILENAMES = (SEMANTICS_FILENAME, PROOF_FILENAME)
PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_signal_computability",
)

EXPECTED_CANDIDATE_COUNT = 37
EXPECTED_PREDECESSOR_PRESERVED_COUNT = 27
EXPECTED_NEWLY_COMPUTED_COUNT = 10
EXPECTED_SESSION_COUNT = 1227
EXPECTED_SYMBOL_COUNT = 5866
EXPECTED_ELIGIBLE_CELL_COUNT = 6346625
EXPECTED_MATRIX_SHAPE = [EXPECTED_SESSION_COUNT, EXPECTED_SYMBOL_COUNT]

EXPECTED_BARS_TREE_OID = "e87708d63d5cc53188f829252316035e657d72d4"
EXPECTED_FINANCIAL_TREE_OID = "4f6beebc55aad6cf2747b21c2f6ffb09a1de76eb"
EXPECTED_BARS_FULL_INVENTORY_SHA256 = (
    "eded23180cdea7e393ac3070c096298183fd6576419d5a0f8df9d967e7e1cc9b"
)
EXPECTED_BARS_SELECTED_INVENTORY_SHA256 = (
    "4f91b152521c5986f3372d2644493b70b07ab8cbf08b63bf861ddfa4ceddfdf2"
)
EXPECTED_FINANCIAL_INVENTORY_SHA256 = (
    "875dbafd8a95a92ed57482c33762583e14f54be864d11ebb4799f875f9ee9ca4"
)
EXPECTED_FINANCIAL_SCHEMA_MANIFEST_SHA256 = (
    "5fd4e31ee4d0b4ae8eddf432c0efbbe33b37b8c92b2b0e906d78a575fa1ae9c0"
)
EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256 = (
    "b18782ae3dde2ac8e2eb6717a115a62550e30ea84498811abe0388ae8f2959c6"
)
EXPECTED_BARS_IPC_SHA256 = (
    "0195883820950ab0dec3e2051f2f72cf5fcda920339e7a38f63b65c35a3086c6"
)
EXPECTED_FINANCIAL_IPC_SHA256 = (
    "67bbc6a1aac3bf37d1b02e736c7bcb08e0500ae606086d2e5d5dd8ef71719f80"
)

EXPECTED_CALENDAR_ACCOUNTING = {
    "aquant_date_count": 1000,
    "aquant_dates_sha256": (
        "16a417f1bc4cc27ebf752398dfb6fbadb778eff7e805b848d9f943de1629cfd7"
    ),
    "intersection_count": 950,
    "intersection_sha256": (
        "b5ac6c980f6be585d0f8b882a1877b616688055f29b2b83ec7a3341f407ecd6c"
    ),
    "max_observation_age_open_sessions": 15,
    "missing_myquant_through_max_observed_count": 262,
    "missing_myquant_through_max_observed_sha256": (
        "6ff59b2a03011ae12580eb1c75e18e9eed5ef0059f3ced6eaaaaffaaa323e49e"
    ),
    "myquant_date_count": 1227,
    "myquant_dates_sha256": (
        "46f64b931a482641f40d91b4a98059725c3463def6e76e56a0d05f6a783063f7"
    ),
    "off_myquant_calendar_count": 50,
    "off_myquant_calendar_sha256": (
        "18bf169b2ddb84248c7b010ad30b671d78354595f563aeef09a3f7d2a0b007bf"
    ),
    "tail_after_max_observed_count": 15,
    "tail_after_max_observed_sha256": (
        "ce22c71140d5ee5c4e5fb49fc8ee1abeb458c65a2cc4b67e56a5731379735edc"
    ),
}

EXPECTED_BARS_ACCOUNTING = {
    "full_entry_count": 492,
    "full_parquet_count": 246,
    "full_lock_count": 246,
    "full_byte_count": 222161239,
    "selected_parquet_count": 49,
    "selected_byte_count": 140728589,
    "selected_row_count": 2708771,
    "selected_date_count": 1000,
    "selected_symbol_count": 5579,
    "turnover_non_null_count": 2704741,
    "market_cap_non_null_count": 2705462,
    "duplicate_date_symbol_count": 0,
    "min_observed_bar_date": "2021-06-25",
    "max_observed_bar_date": "2026-06-26",
    "projected_arrow_buffer_bytes": 101188899,
    "projected_ipc_bytes": 100868184,
}

EXPECTED_FINANCIAL_ACCOUNTING = {
    "blob_count": 2147,
    "byte_count": 24753875,
    "row_count": 70459,
    "symbol_count": 2147,
    "logical_column_count": 17,
    "physical_schema_variant_count": 5,
    "path_symbol_mismatch_count": 0,
    "exact_duplicate_excess": 45453,
    "duplicate_report_period_excess": 48883,
    "post_report_period_selection_row_count": 21576,
    "max_availability_date": "2026-04-27",
    "logical_arrow_buffer_bytes": 9631797,
    "logical_ipc_bytes": 11639672,
}

CLAIM_NEGATIVES = {
    "bar_complete_through_proven": False,
    "coverage_sufficient_for_screening": False,
    "materialized_unit_lineage_proven": False,
    "producer_lineage_proven": False,
    "screening_dataset_match": False,
    "source_calendar_match": False,
    "source_data_quality_proven": False,
    "source_same_snapshot": False,
}

NONCOMPUTABILITY_AUTHORITY_FIELDS = {
    "admission_authority": False,
    "bh_authority": False,
    "factor_apply_authority": False,
    "family_bh_authoritative": False,
    "formal_admission_authority": False,
    "gate_1_8_authority": False,
    "maturity_authority": False,
    "new_risk_authorized": False,
    "new_risk_eligible": False,
    "portfolio_authority": False,
    "production_apply_enabled": False,
    "proposal_authority": False,
    "proposal_eligible": False,
    "qualification": False,
    "qualified": False,
    "registry_authority": False,
    "registry_entry_created": False,
    "same_snapshot_screening_verified": False,
    "screening_authority": False,
    "screening_eligible": False,
}

SIDE_EFFECT_FIELDS = {
    "apply": False,
    "broker": False,
    "budget": False,
    "candidate_generation": False,
    "llm": False,
    "maintenance": False,
    "network": False,
    "order": False,
    "portfolio": False,
    "production": False,
    "proposal": False,
    "provider": False,
    "registry": False,
    "trade": False,
    "transaction": False,
    "wal": False,
}

NEWLY_COMPUTED_NAMES = (
    "alpha_growth_quality_profit_roa",
    "alpha_quality_low_debt_assets",
    "alpha_quality_value_cash_fcf",
    "alpha_turnover_low_20d",
    "alpha_turnover_low_60d",
    "alpha_vwap_cash_quality_160",
    "alpha_vwap_growth_profit_160",
    "alpha_vwap_low_debt_160",
    "alpha_vwap_quality_roa_160",
    "alpha_vwap_quality_roe_160",
)

EXPRESSION_PRIMITIVE_NAMES = (
    "amount",
    "close",
    "fcf_to_price",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_roa",
    "fin_roe",
    "high",
    "low",
    "open",
    "turnover_rate",
    "vwap",
)

PRIMITIVE_NAMES = tuple(
    sorted((*EXPRESSION_PRIMITIVE_NAMES, "fin_free_cashflow", "market_cap", "volume"))
)

SOURCE_PARTITION = {
    "aquant_financial_fields": [
        "fcf_to_price",
        "fin_debt_to_assets",
        "fin_free_cashflow",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_roa",
        "fin_roe",
    ],
    "aquant_git_bar_fields": ["market_cap", "turnover_rate"],
    "myquant_frozen_fields": [
        "amount",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "vwap",
    ],
    "cross_source_fallback": False,
    "fcf_to_price_formula": "fin_free_cashflow_divided_by_aquant_total_mv",
}

EXPECTED_RESOURCE_LIMITS = {
    "max_git_blob_bytes": 16 * 1024 * 1024,
    "max_full_git_input_bytes": 512 * 1024 * 1024,
    "max_selected_bars_bytes": 256 * 1024 * 1024,
    "max_financial_bytes": 64 * 1024 * 1024,
    "max_selected_bar_rows": 3_000_000,
    "max_financial_rows": 100_000,
    "max_axis_cells": 8_000_000,
    "max_primitive_matrices": 16,
    "max_frame_bytes": 2 * 1024 * 1024,
    "max_message_bytes": 128 * 1024 * 1024,
    "max_child_output_bytes": 2 * 1024 * 1024,
    "max_child_seconds": 300,
    "max_total_wall_seconds": 900,
    "max_child_rss_bytes": 6 * 1024**3,
    "max_parent_rss_bytes": 8 * 1024**3,
    "child_address_space_bytes": 8 * 1024**3,
    "child_data_bytes": 6 * 1024**3,
    "darwin_memory_rlimit_supported": False,
    "memory_limit_enforcement": "post_child_peak_rss_abort_before_publication",
    "child_nofile": 32,
    "child_fsize_bytes": 0,
    "child_core_bytes": 0,
}

EXPECTED_PROTECTED_CONTEXTS = {
    "aquant_input_resolution_lane": "forbidden_context_only",
    "same_snapshot_screening_bundle": "protected_context_only_not_input_or_oracle",
    "provider_settings_sources": "co_committed_context_only_not_producer_lineage",
}


class FactorGovernanceSignalComputabilityV4_1Error(ValueError):
    """A signal-computability artifact failed closed."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(value) + b"\n"


def semantic_sha256_v4_1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_1(value)).hexdigest()


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} is not a SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} is not a SHA-256"
        ) from exc
    return value


def _git_oid(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 40:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} is not a SHA-1 Git object id"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} is not a SHA-1 Git object id"
        ) from exc
    return value


def _text(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} must be a non-empty string"
        )
    return value


def _exact(
    value: Mapping[str, Any], fields: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} field inventory mismatch"
        )
    return copy.deepcopy(dict(value))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if field in payload:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"payload already contains self-hash field: {field}"
        )
    payload[field] = semantic_sha256_v4_1(payload)
    return payload


def _validate_self_hash(value: Mapping[str, Any], field: str, context: str) -> None:
    stored = _sha(value.get(field), f"{context}.{field}")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if semantic_sha256_v4_1(payload) != stored:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} self-hash mismatch"
        )


def _validate_binding_rows(
    value: Any,
    *,
    context: str,
    expected_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} must be a non-empty list"
        )
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"{context} row must be an object"
            )
        row = copy.deepcopy(dict(raw))
        _text(row.get("binding_id"), f"{context} binding_id")
        _text(row.get("absolute_path"), f"{context} absolute_path")
        _sha(row.get("byte_sha256"), f"{context} byte_sha256")
        if "semantic_sha256" in row:
            _sha(row["semantic_sha256"], f"{context} semantic_sha256")
        if "ast_sha256" in row:
            _sha(row["ast_sha256"], f"{context} ast_sha256")
        rows.append(row)
    ids = [row["binding_id"] for row in rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} binding ids must be sorted and distinct"
        )
    if expected_ids is not None and ids != sorted(expected_ids):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} binding inventory mismatch"
        )
    return rows


def _validate_matrix_descriptor(value: Any, context: str) -> dict[str, Any]:
    fields = {
        "contract",
        "date_axis_sha256",
        "dtype",
        "matrix_sha256",
        "shape",
        "symbol_axis_sha256",
    }
    row = _exact(value, fields, context)
    if (
        row["contract"] != evaluator.MATRIX_HASH_CONTRACT_VERSION
        or row["dtype"] != "float64-little-endian"
        or row["shape"] != EXPECTED_MATRIX_SHAPE
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            f"{context} identity mismatch"
        )
    for field in ("date_axis_sha256", "symbol_axis_sha256", "matrix_sha256"):
        _sha(row[field], f"{context}.{field}")
    return row


def _validate_claim_isolation(payload: Mapping[str, Any], *, proven: bool) -> None:
    if payload.get("claim_negatives") != CLAIM_NEGATIVES:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability claim negatives mismatch"
        )
    if payload.get("side_effects") != SIDE_EFFECT_FIELDS:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability side effects must all remain false"
        )
    for field, expected in NONCOMPUTABILITY_AUTHORITY_FIELDS.items():
        if payload.get(field) is not expected:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"signal computability authority mismatch: {field}"
            )
    if payload.get("operator_runtime_equivalence_verified") is not True:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "operator runtime equivalence must remain verified"
        )
    if payload.get("signal_computability_proven") is not proven:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability result mismatch"
        )


def validate_input_semantics_receipt_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "claim_scope",
        "pinned_commit",
        "baseline_bindings",
        "code_bindings",
        "predecessor_bindings",
        "git_identity",
        "aquant_source_blobs",
        "bars_inventory",
        "calendar_accounting",
        "financial_inventory",
        "transformation_contract",
        "runtime_identity",
        "source_partition",
        "resource_limits",
        "protected_contexts",
        "claim_negatives",
        "side_effects",
        "receipt_semantic_sha256",
    }
    payload = _exact(value, fields, "input semantics receipt")
    _validate_self_hash(payload, "receipt_semantic_sha256", "input semantics receipt")
    if (
        payload["schema_version"] != SEMANTICS_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != EXPECTED_CYCLE_ID
        or payload["claim_scope"] != CLAIM_SCOPE
        or payload["pinned_commit"] != PINNED_COMMIT
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "input semantics receipt identity mismatch"
        )
    _validate_binding_rows(
        payload["baseline_bindings"],
        context="baseline bindings",
        expected_ids=("execution_baseline", "worktree_content_baseline"),
    )
    _validate_binding_rows(payload["code_bindings"], context="code bindings")
    _validate_binding_rows(
        payload["predecessor_bindings"], context="predecessor bindings"
    )
    git_identity = payload["git_identity"]
    if not isinstance(git_identity, Mapping):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "git identity must be an object"
        )
    _sha(git_identity.get("executable_sha256"), "git executable SHA")
    if git_identity.get("replacement_objects_disabled") is not True or git_identity.get(
        "lazy_fetch_disabled"
    ) is not True:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "git identity is not fail-closed"
        )
    source_blobs = payload["aquant_source_blobs"]
    if not isinstance(source_blobs, list) or len(source_blobs) != 3:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "A_quant source blob inventory mismatch"
        )
    for row in source_blobs:
        if row.get("mode") != "100644" or row.get("type") != "blob":
            raise FactorGovernanceSignalComputabilityV4_1Error(
                "A_quant source blob mode/type mismatch"
            )
        _git_oid(row.get("oid"), "A_quant source blob OID")
        _sha(row.get("sha256"), "A_quant source blob SHA")
    bars = payload["bars_inventory"]
    if (
        not isinstance(bars, Mapping)
        or bars.get("tree_oid") != EXPECTED_BARS_TREE_OID
        or bars.get("full_inventory_sha256")
        != EXPECTED_BARS_FULL_INVENTORY_SHA256
        or bars.get("selected_inventory_sha256")
        != EXPECTED_BARS_SELECTED_INVENTORY_SHA256
        or bars.get("projected_ipc_sha256") != EXPECTED_BARS_IPC_SHA256
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "A_quant bars inventory identity mismatch"
        )
    for key, expected in EXPECTED_BARS_ACCOUNTING.items():
        if bars.get(key) != expected:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"A_quant bars accounting mismatch: {key}"
            )
    calendar = payload["calendar_accounting"]
    if calendar != EXPECTED_CALENDAR_ACCOUNTING:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "A_quant/myQuant calendar accounting mismatch"
        )
    financial = payload["financial_inventory"]
    if (
        not isinstance(financial, Mapping)
        or financial.get("tree_oid") != EXPECTED_FINANCIAL_TREE_OID
        or financial.get("inventory_sha256")
        != EXPECTED_FINANCIAL_INVENTORY_SHA256
        or financial.get("physical_schema_manifest_sha256")
        != EXPECTED_FINANCIAL_SCHEMA_MANIFEST_SHA256
        or financial.get("logical_ipc_sha256") != EXPECTED_FINANCIAL_IPC_SHA256
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "A_quant financial inventory identity mismatch"
        )
    for key, expected in EXPECTED_FINANCIAL_ACCOUNTING.items():
        if financial.get(key) != expected:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"A_quant financial accounting mismatch: {key}"
            )
    transformation = payload["transformation_contract"]
    if (
        not isinstance(transformation, Mapping)
        or transformation.get("ast_manifest_sha256")
        != EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256
        or transformation.get("child_exec_event_count") != 1
        or transformation.get("child_filesystem_access_after_audit") is not False
        or transformation.get("child_network_access_after_audit") is not False
        or transformation.get("child_parent_all_match") is not True
        or transformation.get("descriptor_contract")
        != evaluator.MATRIX_HASH_CONTRACT_VERSION
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "pinned transformation contract mismatch"
        )
    if payload["source_partition"] != SOURCE_PARTITION:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "source partition mismatch"
        )
    if not isinstance(payload["runtime_identity"], Mapping):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "runtime identity missing"
        )
    _sha(
        payload["runtime_identity"].get("runtime_semantic_sha256"),
        "runtime identity semantic SHA",
    )
    if payload["resource_limits"] != EXPECTED_RESOURCE_LIMITS:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "resource limits mismatch"
        )
    if payload["protected_contexts"] != EXPECTED_PROTECTED_CONTEXTS:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "protected contexts mismatch"
        )
    if payload["claim_negatives"] != CLAIM_NEGATIVES:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "input semantics claim negatives mismatch"
        )
    if payload["side_effects"] != SIDE_EFFECT_FIELDS:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "input semantics side effects mismatch"
        )
    return payload


def build_input_semantics_receipt_v4_1(**values: Any) -> dict[str, Any]:
    payload = {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": EXPECTED_CYCLE_ID,
        "claim_scope": CLAIM_SCOPE,
        "pinned_commit": PINNED_COMMIT,
        **copy.deepcopy(values),
        "claim_negatives": dict(CLAIM_NEGATIVES),
        "side_effects": dict(SIDE_EFFECT_FIELDS),
    }
    return validate_input_semantics_receipt_v4_1(
        _seal(payload, "receipt_semantic_sha256")
    )


def _validate_primitive_rows(value: Any) -> list[dict[str, Any]]:
    fields = {
        "field",
        "source",
        "matrix",
        "finite_count",
        "nan_count",
        "positive_inf_count",
        "negative_inf_count",
        "outside_mask_non_nan_count",
    }
    if not isinstance(value, list) or len(value) != len(PRIMITIVE_NAMES):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "primitive matrix inventory mismatch"
        )
    rows = [_exact(row, fields, "primitive matrix row") for row in value]
    if [row["field"] for row in rows] != list(PRIMITIVE_NAMES):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "primitive matrix order mismatch"
        )
    for row in rows:
        _text(row["source"], "primitive source")
        _validate_matrix_descriptor(row["matrix"], f"primitive {row['field']}")
        for field in (
            "finite_count",
            "nan_count",
            "positive_inf_count",
            "negative_inf_count",
            "outside_mask_non_nan_count",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise FactorGovernanceSignalComputabilityV4_1Error(
                    f"primitive count invalid: {row['field']}:{field}"
                )
        if row["outside_mask_non_nan_count"] != 0:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"primitive leaked outside PIT mask: {row['field']}"
            )
    return rows


def _validate_candidate_rows(value: Any, *, proven: bool) -> list[dict[str, Any]]:
    fields = {
        "candidate_id",
        "name",
        "input_fields",
        "source_definition_sha256",
        "catalog_definition_sha256",
        "mapping_semantic_sha256",
        "normalized_ast_sha256",
        "predecessor_status",
        "predecessor_descriptor_preserved",
        "status",
        "eligible_cell_count",
        "finite_count",
        "nan_count",
        "positive_inf_count",
        "negative_inf_count",
        "outside_mask_non_nan_count",
        "signal_matrix",
        "row_semantic_sha256",
    }
    if not isinstance(value, list) or len(value) != EXPECTED_CANDIDATE_COUNT:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "candidate row inventory mismatch"
        )
    rows = [_exact(row, fields, "candidate row") for row in value]
    candidate_ids = [row["candidate_id"] for row in rows]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "candidate row order mismatch"
        )
    new_names = {row["name"] for row in rows if row["name"] in NEWLY_COMPUTED_NAMES}
    if new_names != set(NEWLY_COMPUTED_NAMES):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "newly computed candidate inventory mismatch"
        )
    preserved_count = 0
    for row in rows:
        _text(row["candidate_id"], "candidate id")
        _text(row["name"], "candidate name")
        if (
            not isinstance(row["input_fields"], list)
            or row["input_fields"] != sorted(set(row["input_fields"]))
            or any(field not in EXPRESSION_PRIMITIVE_NAMES for field in row["input_fields"])
        ):
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate input fields mismatch: {row['name']}"
            )
        for field in (
            "source_definition_sha256",
            "catalog_definition_sha256",
            "mapping_semantic_sha256",
            "normalized_ast_sha256",
            "row_semantic_sha256",
        ):
            _sha(row[field], f"candidate {field}")
        row_without_hash = {
            key: copy.deepcopy(item)
            for key, item in row.items()
            if key != "row_semantic_sha256"
        }
        if semantic_sha256_v4_1(row_without_hash) != row["row_semantic_sha256"]:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate row semantic SHA mismatch: {row['name']}"
            )
        _validate_matrix_descriptor(row["signal_matrix"], f"candidate {row['name']}")
        if row["eligible_cell_count"] != EXPECTED_ELIGIBLE_CELL_COUNT:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate eligible-cell count mismatch: {row['name']}"
            )
        counts = (
            row["finite_count"],
            row["nan_count"],
            row["positive_inf_count"],
            row["negative_inf_count"],
        )
        if any(type(item) is not int or item < 0 for item in counts) or sum(
            counts
        ) != EXPECTED_ELIGIBLE_CELL_COUNT:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate observation accounting mismatch: {row['name']}"
            )
        if row["outside_mask_non_nan_count"] != 0:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate leaked outside PIT mask: {row['name']}"
            )
        if row["name"] in NEWLY_COMPUTED_NAMES:
            if (
                row["predecessor_status"]
                not in {"turnover_data_blocked", "fundamental_semantic_blocked"}
                or row["predecessor_descriptor_preserved"] is not False
            ):
                raise FactorGovernanceSignalComputabilityV4_1Error(
                    f"newly computed predecessor accounting mismatch: {row['name']}"
                )
        else:
            preserved_count += 1
            if (
                row["predecessor_status"] != "no_label_signal_eval_diagnostic"
                or row["predecessor_descriptor_preserved"] is not True
            ):
                raise FactorGovernanceSignalComputabilityV4_1Error(
                    f"predecessor descriptor was not preserved: {row['name']}"
                )
        if proven and (
            row["status"] != "source_semantic_computability_verified"
            or row["finite_count"] <= 0
        ):
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"candidate is not computable: {row['name']}"
            )
    if preserved_count != EXPECTED_PREDECESSOR_PRESERVED_COUNT:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "predecessor preservation count mismatch"
        )
    return rows


def validate_signal_computability_proof_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "claim_scope",
        "pinned_commit",
        "readiness",
        "semantics_receipt_semantic_sha256",
        "predecessor_proof_bindings",
        "computation_passes",
        "primitive_matrices",
        "candidate_count",
        "predecessor_preserved_count",
        "newly_computed_count",
        "candidate_order_sha256",
        "result_manifest_sha256",
        "rows",
        "blockers",
        "claim_negatives",
        "side_effects",
        "operator_runtime_equivalence_verified",
        "signal_computability_proven",
        "proof_semantic_sha256",
        *NONCOMPUTABILITY_AUTHORITY_FIELDS,
    }
    payload = _exact(value, fields, "signal computability proof")
    _validate_self_hash(payload, "proof_semantic_sha256", "signal computability proof")
    proven = payload["signal_computability_proven"] is True
    if (
        payload["schema_version"] != PROOF_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != EXPECTED_CYCLE_ID
        or payload["claim_scope"] != CLAIM_SCOPE
        or payload["pinned_commit"] != PINNED_COMMIT
        or payload["readiness"] != READINESS
        or payload["candidate_count"] != EXPECTED_CANDIDATE_COUNT
        or payload["predecessor_preserved_count"]
        != EXPECTED_PREDECESSOR_PRESERVED_COUNT
        or payload["newly_computed_count"] != EXPECTED_NEWLY_COMPUTED_COUNT
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability proof identity mismatch"
        )
    _sha(
        payload["semantics_receipt_semantic_sha256"],
        "semantics receipt semantic SHA",
    )
    _sha(payload["candidate_order_sha256"], "candidate order SHA")
    _sha(payload["result_manifest_sha256"], "result manifest SHA")
    _validate_binding_rows(
        payload["predecessor_proof_bindings"], context="predecessor proof bindings"
    )
    passes = payload["computation_passes"]
    if not isinstance(passes, list) or len(passes) != 2:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "computability requires two fresh computation passes"
        )
    pass_ids = [row.get("pass_id") for row in passes]
    if pass_ids != ["first", "fresh_readback"]:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "computation pass identity mismatch"
        )
    pass_manifests = []
    for row in passes:
        if (
            row.get("child_parent_all_match") is not True
            or row.get("outside_mask_all_zero") is not True
            or row.get("candidate_count") != EXPECTED_CANDIDATE_COUNT
        ):
            raise FactorGovernanceSignalComputabilityV4_1Error(
                "computation pass failed equivalence or PIT checks"
            )
        pass_manifests.append(_sha(row.get("result_manifest_sha256"), "pass manifest"))
        _sha(row.get("runtime_semantic_sha256"), "pass runtime SHA")
    if len(set(pass_manifests)) != 1 or pass_manifests[0] != payload[
        "result_manifest_sha256"
    ]:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "fresh recomputation result manifest mismatch"
        )
    _validate_primitive_rows(payload["primitive_matrices"])
    rows = _validate_candidate_rows(payload["rows"], proven=proven)
    candidate_ids = [row["candidate_id"] for row in rows]
    if semantic_sha256_v4_1(candidate_ids) != payload["candidate_order_sha256"]:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "candidate order SHA mismatch"
        )
    blockers = payload["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or any(type(item) is not str or not item for item in blockers)
        or (proven and blockers)
        or (not proven and not blockers)
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability blocker accounting mismatch"
        )
    expected_manifest = semantic_sha256_v4_1(
        {
            "primitive_matrices": payload["primitive_matrices"],
            "rows": rows,
        }
    )
    if expected_manifest != payload["result_manifest_sha256"]:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability result manifest mismatch"
        )
    _validate_claim_isolation(payload, proven=proven)
    return payload


def build_signal_computability_proof_v4_1(
    *,
    semantics_receipt: Mapping[str, Any],
    predecessor_proof_bindings: Sequence[Mapping[str, Any]],
    computation_passes: Sequence[Mapping[str, Any]],
    primitive_matrices: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    receipt = validate_input_semantics_receipt_v4_1(semantics_receipt)
    normalized_rows = [copy.deepcopy(dict(row)) for row in rows]
    normalized_primitives = [copy.deepcopy(dict(row)) for row in primitive_matrices]
    normalized_blockers = sorted(set(blockers))
    proven = not normalized_blockers
    payload = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": EXPECTED_CYCLE_ID,
        "claim_scope": CLAIM_SCOPE,
        "pinned_commit": PINNED_COMMIT,
        "readiness": READINESS,
        "semantics_receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
        "predecessor_proof_bindings": [
            copy.deepcopy(dict(item)) for item in predecessor_proof_bindings
        ],
        "computation_passes": [copy.deepcopy(dict(item)) for item in computation_passes],
        "primitive_matrices": normalized_primitives,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "predecessor_preserved_count": EXPECTED_PREDECESSOR_PRESERVED_COUNT,
        "newly_computed_count": EXPECTED_NEWLY_COMPUTED_COUNT,
        "candidate_order_sha256": semantic_sha256_v4_1(
            [row["candidate_id"] for row in normalized_rows]
        ),
        "result_manifest_sha256": semantic_sha256_v4_1(
            {"primitive_matrices": normalized_primitives, "rows": normalized_rows}
        ),
        "rows": normalized_rows,
        "blockers": normalized_blockers,
        "claim_negatives": dict(CLAIM_NEGATIVES),
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        "operator_runtime_equivalence_verified": True,
        "signal_computability_proven": proven,
        **NONCOMPUTABILITY_AUTHORITY_FIELDS,
    }
    return validate_signal_computability_proof_v4_1(
        _seal(payload, "proof_semantic_sha256")
    )


def build_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(artifacts) != set(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability readback artifact inventory mismatch"
        )
    receipt = validate_input_semantics_receipt_v4_1(artifacts[SEMANTICS_FILENAME])
    proof = validate_signal_computability_proof_v4_1(artifacts[PROOF_FILENAME])
    if proof["semantics_receipt_semantic_sha256"] != receipt[
        "receipt_semantic_sha256"
    ]:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "proof does not bind the input semantics receipt"
        )
    bindings = [copy.deepcopy(dict(item)) for item in artifact_bindings]
    expected_bindings = []
    for filename, value in (
        (SEMANTICS_FILENAME, receipt),
        (PROOF_FILENAME, proof),
    ):
        raw = canonical_file_bytes_v4_1(value)
        expected_bindings.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    if canonical_json_bytes_v4_1(bindings) != canonical_json_bytes_v4_1(
        expected_bindings
    ):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability readback byte binding mismatch"
        )
    payload = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": EXPECTED_CYCLE_ID,
        "run_id": _text(run_id, "run_id"),
        "readiness": READINESS,
        "accepted": True,
        "artifact_bindings": bindings,
        "receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
        "proof_semantic_sha256": proof["proof_semantic_sha256"],
        "claim_negatives": dict(CLAIM_NEGATIVES),
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        "operator_runtime_equivalence_verified": True,
        "signal_computability_proven": proof["signal_computability_proven"],
        **NONCOMPUTABILITY_AUTHORITY_FIELDS,
    }
    return _seal(payload, "report_semantic_sha256")


def validate_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "accepted",
        "artifact_bindings",
        "receipt_semantic_sha256",
        "proof_semantic_sha256",
        "claim_negatives",
        "side_effects",
        "operator_runtime_equivalence_verified",
        "signal_computability_proven",
        "report_semantic_sha256",
        *NONCOMPUTABILITY_AUTHORITY_FIELDS,
    }
    payload = _exact(value, fields, "signal computability readback")
    _validate_self_hash(payload, "report_semantic_sha256", "signal computability readback")
    expected = build_readback_report_v4_1(
        run_id=payload["run_id"],
        artifacts=artifacts,
        artifact_bindings=artifact_bindings,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability readback differs from exact recomputation"
        )
    if payload["accepted"] is not True:
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "signal computability bundle was not accepted"
        )
    _validate_claim_isolation(
        payload, proven=payload["signal_computability_proven"] is True
    )
    return payload


def build_private_bundle_contract_v4_1(
    *, expected_artifacts: Mapping[str, Mapping[str, Any]]
) -> private_io.PrivateBundleContract:
    if set(expected_artifacts) != set(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceSignalComputabilityV4_1Error(
            "expected signal computability artifact inventory mismatch"
        )
    expected = {
        SEMANTICS_FILENAME: validate_input_semantics_receipt_v4_1(
            expected_artifacts[SEMANTICS_FILENAME]
        ),
        PROOF_FILENAME: validate_signal_computability_proof_v4_1(
            expected_artifacts[PROOF_FILENAME]
        ),
    }

    def validate_artifact(
        filename: str, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if filename == SEMANTICS_FILENAME:
            normalized = validate_input_semantics_receipt_v4_1(value)
        elif filename == PROOF_FILENAME:
            normalized = validate_signal_computability_proof_v4_1(value)
        elif filename == READBACK_FILENAME:
            normalized = copy.deepcopy(dict(value))
        else:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"unexpected signal computability artifact: {filename}"
            )
        if filename in expected and canonical_json_bytes_v4_1(
            normalized
        ) != canonical_json_bytes_v4_1(expected[filename]):
            raise FactorGovernanceSignalComputabilityV4_1Error(
                f"signal computability artifact differs from expected bytes: {filename}"
            )
        return normalized

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        if set(values) != {SEMANTICS_FILENAME, PROOF_FILENAME, READBACK_FILENAME}:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                "complete signal computability bundle inventory mismatch"
            )
        receipt = validate_input_semantics_receipt_v4_1(values[SEMANTICS_FILENAME])
        proof = validate_signal_computability_proof_v4_1(values[PROOF_FILENAME])
        if proof["semantics_receipt_semantic_sha256"] != receipt[
            "receipt_semantic_sha256"
        ]:
            raise FactorGovernanceSignalComputabilityV4_1Error(
                "complete signal computability bundle is not cross-bound"
            )
        return {
            SEMANTICS_FILENAME: receipt,
            PROOF_FILENAME: proof,
            READBACK_FILENAME: copy.deepcopy(dict(values[READBACK_FILENAME])),
        }

    def build_report(**kwargs: Any) -> Mapping[str, Any]:
        return build_readback_report_v4_1(**kwargs)

    def canonicalize(value: Mapping[str, Any]) -> bytes:
        return canonical_file_bytes_v4_1(value)

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=BUNDLE_INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonicalize,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_report,
        max_artifact_bytes=16 * 1024 * 1024,
        max_bundle_bytes=48 * 1024 * 1024,
    )


__all__ = [
    "CLAIM_NEGATIVES",
    "CLAIM_SCOPE",
    "EXPECTED_CALENDAR_ACCOUNTING",
    "EXPECTED_CYCLE_ID",
    "FactorGovernanceSignalComputabilityV4_1Error",
    "NEWLY_COMPUTED_NAMES",
    "NONCOMPUTABILITY_AUTHORITY_FIELDS",
    "PINNED_COMMIT",
    "PROOF_FILENAME",
    "PRIMITIVE_NAMES",
    "READBACK_FILENAME",
    "READINESS",
    "SEMANTICS_FILENAME",
    "SIDE_EFFECT_FIELDS",
    "SOURCE_PARTITION",
    "build_input_semantics_receipt_v4_1",
    "build_private_bundle_contract_v4_1",
    "build_signal_computability_proof_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "semantic_sha256_v4_1",
    "validate_input_semantics_receipt_v4_1",
    "validate_readback_report_v4_1",
    "validate_signal_computability_proof_v4_1",
]
