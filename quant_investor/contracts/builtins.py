"""Compiled stable contract allowlist for the unified runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import re
from typing import Any, Final

from .core import (
    LEGACY_CONTRACT_FIELDS,
    ArtifactValidationError,
    ContractDefinition,
    _freeze_contract_registry,
    register_contract,
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _exact_contract(
    kind: str,
    identity_field: str,
    fields: Iterable[str],
    *,
    validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> ContractDefinition:
    return register_contract(
        ContractDefinition(
            kind=kind,
            identity_field=identity_field,
            required_payload_fields=frozenset(fields),
            forbidden_payload_fields=LEGACY_CONTRACT_FIELDS,
            validator=validator,
        )
    )


def _exact_nested_object(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ArtifactValidationError(f"{label} fields are not exact")
    return value


def _exact_rows(value: Any, fields: set[str], *, label: str) -> list[Mapping[str, Any]]:
    if type(value) is not list:
        raise ArtifactValidationError(f"{label} must be a list")
    return [
        _exact_nested_object(row, fields, label=f"{label}[{index}]")
        for index, row in enumerate(value)
    ]


_OBJECT_REF_FIELDS: Final = {
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
}


def _exact_ref(value: Any, *, label: str, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    row = _exact_nested_object(value, _OBJECT_REF_FIELDS, label=label)
    for field in ("contract_sha256", "semantic_sha256", "byte_sha256"):
        if type(row.get(field)) is not str or _SHA256_RE.fullmatch(row[field]) is None:
            raise ArtifactValidationError(f"{label}.{field} must be lowercase SHA-256")
    for field in ("kind", "artifact_id"):
        text = row.get(field)
        if type(text) is not str or not text or text != text.strip():
            raise ArtifactValidationError(f"{label}.{field} must be canonical text")


def _validate_ref_rows(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if type(value) is not list:
        raise ArtifactValidationError(f"{label} must be a list")
    for index, ref in enumerate(value):
        _exact_ref(ref, label=f"{label}[{index}]")
    return value


def _validate_factor_validator_manifest(payload: Mapping[str, Any]) -> None:
    for field in (
        "release_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
    ):
        _exact_ref(payload.get(field), label=f"factor.validator_manifest.{field}")
    implementation_rows = _exact_rows(
        payload.get("implementation_rows"),
        {
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
        },
        label="factor.validator_manifest.implementation_rows",
    )
    for index, row in enumerate(implementation_rows):
        _exact_ref(
            row.get("implementation_component_ref"),
            label=f"factor.validator_manifest.implementation_rows[{index}].component_ref",
        )
        roles = row.get("required_source_roles")
        if (
            type(roles) is not list
            or roles != sorted(set(roles))
            or not roles
            or any(
                type(role) is not str
                or role
                not in {
                    "EXCHANGE_CALENDAR",
                    "FUNDAMENTAL",
                    "MARKET",
                    "PIT_MEMBERSHIP",
                }
                for role in roles
            )
        ):
            raise ArtifactValidationError(
                f"factor.validator_manifest.implementation_rows[{index}] "
                "required_source_roles are not exact"
            )
    _exact_rows(
        payload.get("validated_contracts"),
        {"kind", "contract_sha256", "json_schema_sha256", "validator_code_sha256"},
        label="factor.validator_manifest.validated_contracts",
    )


def _validate_factor_source_decode_attestation(payload: Mapping[str, Any]) -> None:
    decoder = _exact_nested_object(
        payload.get("decoder_contract"),
        {
            "decoder_id",
            "factor_validator_manifest_ref",
            "contextual_validator_component_ref",
            "source_decoder_component_ref",
            "decoder_code_sha256",
            "implementation_component_refs",
            "allowed_source_formats",
            "fallback_allowed",
        },
        label="factor.source_decode_attestation.decoder_contract",
    )
    _exact_ref(
        decoder.get("factor_validator_manifest_ref"),
        label="factor.source_decode_attestation.decoder_contract.factor_validator_manifest_ref",
    )
    for field in ("contextual_validator_component_ref", "source_decoder_component_ref"):
        _exact_ref(
            decoder.get(field),
            label=f"factor.source_decode_attestation.decoder_contract.{field}",
        )
    _validate_ref_rows(
        decoder.get("implementation_component_refs"),
        label="factor.source_decode_attestation.decoder_contract.implementation_component_refs",
    )
    _exact_rows(
        payload.get("source_bindings"),
        {
            "role",
            "source_object_ref",
            "source_root_id",
            "source_object_created_at",
            "media_type",
            "source_format",
            "source_byte_sha256",
            "source_byte_count",
            "decoded_schema_sha256",
            "normalized_sha256",
            "row_count",
            "column_count",
            "decoded_cell_count",
            "minimum_session",
            "maximum_session",
        },
        label="factor.source_decode_attestation.source_bindings",
    )
    for index, row in enumerate(payload["source_bindings"]):
        _exact_ref(
            row.get("source_object_ref"),
            label=f"factor.source_decode_attestation.source_bindings[{index}].source_object_ref",
        )


def _validate_factor_contextual_result(payload: Mapping[str, Any]) -> None:
    for field in (
        "intrinsic_receipt_ref",
        "policy_ref",
        "active_set_ref",
        "factor_validator_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
    ):
        _exact_ref(payload.get(field), label=f"factor.contextual_validation_result.{field}")
    _exact_ref(
        payload.get("composite_state_ref"),
        label="factor.contextual_validation_result.composite_state_ref",
        nullable=True,
    )
    _exact_ref(
        payload.get("custody_head_ref"),
        label="factor.contextual_validation_result.custody_head_ref",
        nullable=True,
    )
    for field in (
        "evidence_refs",
        "implementation_component_refs",
        "source_attestation_refs",
        "source_object_refs",
        "custody_record_refs",
    ):
        _validate_ref_rows(payload.get(field), label=f"factor.contextual_validation_result.{field}")


def _validate_factor_status(payload: Mapping[str, Any]) -> None:
    active = _exact_nested_object(
        payload.get("active"),
        {
            "state",
            "lane",
            "admission_route",
            "producer_identity",
            "factor_set_ref",
            "factor_ids",
            "validation_receipt_ref",
            "contextual_result_ref",
            "validation_attestation_ref",
        },
        label="factor.status.active",
    )
    for field in (
        "factor_set_ref",
        "validation_receipt_ref",
        "contextual_result_ref",
        "validation_attestation_ref",
    ):
        _exact_ref(
            active.get(field),
            label=f"factor.status.active.{field}",
            nullable=True,
        )


def _validate_installed_component(payload: Mapping[str, Any]) -> None:
    _exact_ref(
        payload.get("release_manifest_ref"),
        label="system.installed_component_manifest.release_manifest_ref",
    )
    _exact_rows(
        payload.get("entrypoints"),
        {"module_name", "qualified_name", "code_sha256"},
        label="system.installed_component_manifest.entrypoints",
    )
    _exact_rows(
        payload.get("files"),
        {"path", "byte_sha256", "size"},
        label="system.installed_component_manifest.files",
    )
    if payload.get("outcome") != "VALIDATED" or payload.get("authority") != "NON_AUTHORIZING":
        raise ArtifactValidationError("installed component must be validated and non-authorizing")


def _validate_validation_attestation(payload: Mapping[str, Any]) -> None:
    for field in (
        "validation_request_ref",
        "contextual_result_ref",
        "intrinsic_receipt_ref",
        "policy_ref",
        "active_set_ref",
        "release_manifest_ref",
        "factor_validator_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
    ):
        _exact_ref(payload.get(field), label=f"system.validation_attestation.{field}")
    _exact_ref(
        payload.get("candidate_state_ref"),
        label="system.validation_attestation.candidate_state_ref",
        nullable=True,
    )
    for field in (
        "evidence_refs",
        "source_object_refs",
        "implementation_component_refs",
        "source_attestation_refs",
        "custody_record_refs",
    ):
        _validate_ref_rows(payload.get(field), label=f"system.validation_attestation.{field}")
    _exact_ref(
        payload.get("custody_head_ref"),
        label="system.validation_attestation.custody_head_ref",
        nullable=True,
    )
    _exact_nested_object(
        payload.get("release_identity"),
        {"release_id", "code_sha256", "wheel_sha256", "code_manifest_sha256"},
        label="system.validation_attestation.release_identity",
    )
    _exact_rows(
        payload.get("compiled_contracts"),
        {"kind", "contract_sha256", "json_schema_sha256", "validator_code_sha256"},
        label="system.validation_attestation.compiled_contracts",
    )
    if payload.get("outcome") != "VALIDATED" or payload.get("authority") != "NON_AUTHORIZING":
        raise ArtifactValidationError(
            "validation attestation must be validated and non-authorizing"
        )


def _validate_system_release_payload(payload: Mapping[str, Any]) -> None:
    state = payload.get("state")
    if type(state) is not str or not state or state != state.strip():
        raise ArtifactValidationError("system.release state must be canonical text")
    for field in ("code_sha256", "wheel_sha256", "code_manifest_sha256"):
        value = payload.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            raise ArtifactValidationError(f"system.release {field} must be lowercase SHA-256")


_FACTOR_FIELD_SETS: Final = {
    "factor.bootstrap_set": (
        "bootstrap_set_id",
        "admission_route",
        "producer_identity",
        "input_contract",
        "factor_definitions",
        "factor_rows",
        "control_rows",
        "bootstrap_exception_evidence_ref",
        "factor_set_sha256",
        "weighting_method",
        "weight_total",
        "prospective_evidence_claimed",
        "activation_authorized",
    ),
    "factor.preregistration": (
        "preregistration_id",
        "lane",
        "stamp_source",
        "open_sessions",
        "signal_sessions",
        "maturity_sessions",
        "session_windows",
        "candidates",
        "exchange_calendar_ref",
        "implementation_manifest_ref",
        "source_decode_attestation_ref",
        "factor_validator_manifest_ref",
        "coverage_contract",
        "label_contract",
        "neutralization_contract",
        "maturity_contract",
        "validation_contract",
        "alternate_policy",
        "observation_policy",
        "authority",
    ),
    "factor.configuration_selection": (
        "selection_id",
        "preregistration_id",
        "first_signal_session",
        "source_decode_attestation_ref",
        "configuration_summary_rows",
        "selected_configurations",
        "selected_before_label",
        "label_inputs_used",
        "substitution_allowed",
        "selection_policy",
    ),
    "factor.prospective_observation": (
        "observation_id",
        "observation_lineage_id",
        "previous_observation_ref",
        "preregistration_id",
        "selection_id",
        "signal_capture_ref",
        "source_decode_attestation_ref",
        "ordinal",
        "signal_session",
        "label_start_session",
        "label_end_session",
        "label_formula",
        "neutralization_method",
        "coverage_minimum",
        "pit_universe_count",
        "pit_universe_sha256",
        "label_values_sha256",
        "label_finite_pair_count",
        "configuration_rows",
        "backfill",
        "substitution",
    ),
    "factor.prospective_evaluation": (
        "evaluation_id",
        "preregistration_id",
        "selection_id",
        "lane",
        "observation_ids",
        "observation_count",
        "execution_turnover_evidence_ref",
        "candidate_rows",
        "trial_statistics",
        "redundancy_clusters",
        "admission_eligible",
        "blockers",
        "cost_bps",
    ),
    "factor.admitted_set": (
        "admitted_set_id",
        "lane",
        "preregistration_id",
        "selection_id",
        "evaluation_id",
        "factor_rows",
        "weight_total",
        "weighting_method",
        "activation_authorized",
    ),
    "factor.status": (
        "status_id",
        "active",
        "observed",
        "readiness",
        "blockers",
        "activation_mutation_authorized",
    ),
    "factor.canonical_replay_evidence": (
        "replay_evidence_id",
        "full_control_chain_evaluated",
        "arm_sha256s",
        "evidence_sha256",
    ),
    "factor.bootstrap_exception_evidence": (
        "bootstrap_evidence_id",
        "admission_route",
        "producer_identity",
        "decision_source_id",
        "decision_source_sha256",
        "factor_rows",
        "reader_contract",
        "source_refs",
        "factor_set_sha256",
        "weight_total",
        "authorizes_readiness",
        "authorizes_selectability",
    ),
    "factor.validation_receipt": (
        "validation_receipt_id",
        "policy_ref",
        "evidence_refs",
        "active_set_ref",
        "validated",
        "authority",
    ),
    "factor.observation_head": (
        "observation_head_id",
        "observation_lineage_id",
        "preregistration_id",
        "selection_id",
        "observation_count",
        "previous_head_ref",
        "head_observation_ref",
        "authority",
    ),
    "factor.execution_turnover_evidence": (
        "execution_evidence_id",
        "preregistration_id",
        "selection_id",
        "lane",
        "signal_sessions_sha256",
        "signal_session_count",
        "signal_capture_refs",
        "observation_refs",
        "configuration_rows",
        "cost_contract",
        "execution_state",
        "blockers",
        "authority",
    ),
    "factor.validator_manifest": (
        "validator_manifest_id",
        "release_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
        "implementation_rows",
        "validated_contracts",
        "authority",
    ),
    "factor.source_decode_attestation": (
        "source_decode_attestation_id",
        "purpose",
        "preregistration_id",
        "selection_id",
        "ordinal",
        "signal_session",
        "maturity_session",
        "decoder_contract",
        "source_bindings",
        "normalized_inputs_sha256",
        "authority",
    ),
    "factor.signal_capture": (
        "signal_capture_id",
        "observation_lineage_id",
        "previous_signal_capture_ref",
        "preregistration_id",
        "selection_id",
        "ordinal",
        "signal_session",
        "source_decode_attestation_ref",
        "pit_universe_count",
        "pit_universe_sha256",
        "configuration_rows",
        "coverage_minimum",
        "label_inputs_used",
        "unlisted_universe_weight",
        "backfill",
        "authority",
    ),
    "factor.custody_record": (
        "custody_record_id",
        "custody_namespace_id",
        "preregistration_id",
        "sequence",
        "previous_custody_ref",
        "previous_composite_state_ref",
        "transaction_id",
        "transaction_sequence",
        "transaction_record_index",
        "transaction_record_count",
        "operation_request_sha256",
        "operation",
        "subject_refs",
        "source_attestation_refs",
        "stage_slot",
        "blockers",
        "stored_at",
        "clock_source",
        "authority",
    ),
    "factor.composite_state": (
        "composite_state_id",
        "custody_namespace_id",
        "preregistration_ref",
        "cycle_state",
        "transaction_sequence",
        "previous_composite_state_ref",
        "transaction_id",
        "custody_record_count",
        "custody_head_ref",
        "selection_ref",
        "signal_capture_count",
        "signal_capture_head_ref",
        "observation_count",
        "observation_head_ref",
        "execution_evidence_ref",
        "evaluation_ref",
        "admitted_set_ref",
        "intrinsic_receipt_ref",
        "resolved_signal_slot_count",
        "resolved_label_slot_count",
        "slot_tree_sha256",
        "terminal",
        "blockers",
        "last_stored_at",
        "authority",
    ),
    "factor.contextual_validation_result": (
        "contextual_result_id",
        "validation_namespace_id",
        "lane",
        "intrinsic_receipt_ref",
        "policy_ref",
        "evidence_refs",
        "active_set_ref",
        "composite_state_ref",
        "factor_validator_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
        "implementation_component_refs",
        "source_attestation_refs",
        "source_object_refs",
        "custody_record_refs",
        "custody_tree_sha256",
        "custody_head_ref",
        "validated",
        "blockers",
        "authority",
    ),
    "factor.production_market_pit_selection": (
        "market_pit_selection_id",
        "state",
        "selection_mode",
        "as_of",
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "market_bound_pit_pointer_file_ref",
        "pit_generation_id",
        "pit_generation_manifest_file_ref",
        "pit_membership_file_ref",
        "pit_generation_manifest_sha256",
        "pit_membership_sha256",
        "observed_current_pit_pointer_file_ref",
        "observed_current_pit_pointer_sha256",
        "observed_current_pit_generation_id",
        "pinned_as_of_disclosure",
        "user_authorization_basis",
        "selection_module_path",
        "selection_module_sha256",
    ),
    "factor.production_source_closure": (
        "factor_production_source_closure_id",
        "state",
        "activation_scope",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_verification",
        "release_install_input_source_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_recomputation_evidence": (
        "factor_production_recomputation_id",
        "state",
        "activation_scope",
        "source_closure_ref",
        "deployed_release_ref",
        "factor_active_set_ref",
        "as_of",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_values",
        "signal_statistics",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
    ),
    "factor.production_legacy_zero_call_certificate": (
        "factor_legacy_zero_call_id",
        "state",
        "activation_scope",
        "final_commit",
        "final_tree",
        "resolver_inventory_ref",
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
        "verification_module_path",
        "verification_module_sha256",
        "verification_command",
        "stdout_sha256",
        "stderr_sha256",
        "verified_at",
    ),
    "factor.production_market_input": (
        "factor_market_input_id",
        "state",
        "activation_scope",
        "as_of",
        "market_pit_selection_ref",
        "market_pointer_source_ref",
        "market_snapshot_manifest_source_ref",
        "market_scope_source_ref",
        "market_history_source_ref",
        "market_pointer_sha256",
        "market_snapshot_manifest_sha256",
        "market_history_sha256",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_generation_id",
        "pit_membership_sha256",
        "producer_module_path",
        "producer_module_sha256",
    ),
    "factor.production_calendar_capture_custody_attestation": (
        "calendar_capture_custody_attestation_id",
        "state",
        "activation_scope",
        "capture_root_name",
        "deployed_release_ref",
        "capture_transaction_ref",
        "capture_execution_ref",
        "capture_success_ref",
        "published_root_device",
        "published_root_inode",
        "published_leaf_manifest",
        "published_leaf_manifest_sha256",
        "verified_at",
    ),
    "factor.production_generation": (
        "factor_production_generation_id",
        "state",
        "activation_scope",
        "admission_route",
        "producer_identity",
        "as_of",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_verification",
        "release_install_input_source_ref",
        "source_closure_ref",
        "recomputation_evidence_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "market_input_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_values",
        "signal_statistics",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_pointer": (
        "factor_production_pointer_id",
        "factor_generation_id",
        "factor_generation_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
        "authority_scope",
        "pointer_raw_sha256",
    ),
    "factor.production_generation_receipt": (
        "factor_production_receipt_id",
        "state",
        "activation_scope",
        "source_closure_ref",
        "recomputation_evidence_ref",
        "factor_generation_ref",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "legacy_zero_call_ref",
        "market_input_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "active_factor_rows",
        "control_rows",
        "factor_readiness",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_activation_bundle": (
        "factor_production_activation_id",
        "state",
        "activation_scope",
        "factor_generation_receipt_ref",
        "target_factor_generation_id",
        "target_factor_generation_ref",
        "deployed_release_ref",
        "active_factor_rows",
        "control_rows",
        "low_signal_sha256",
        "w80_signal_sha256",
        "factor_readiness",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "target_factor_pointer_ref",
        "target_factor_pointer_path",
        "expected_factor_pointer_sha256",
        "prepared_at",
        "activated_at",
        "actor_uid",
        "os_actor",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_prepared": (
        "factor_production_prepared_id",
        "state",
        "activation_scope",
        "activation_bundle_ref",
        "factor_generation_receipt_ref",
        "target_factor_pointer_ref",
        "expected_factor_pointer_sha256",
        "prepared_at",
        "actor_uid",
    ),
    "factor.production_marker": (
        "factor_production_marker_id",
        "state",
        "activation_scope",
        "activation_bundle_ref",
        "prepared_transaction_ref",
        "factor_generation_receipt_ref",
        "factor_pointer_ref",
        "factor_generation_ref",
        "deployed_release_ref",
        "active_factor_rows",
        "control_rows",
        "factor_readiness",
        "factor_authority",
        "market_input_ref",
        "admission_route",
        "producer_identity",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_rollover_bundle": (
        "factor_production_rollover_bundle_id",
        "state",
        "activation_scope",
        "predecessor_pointer_ref",
        "target_pointer_ref",
        "previous_pointer_sha256",
        "target_pointer_sha256",
        "factor_generation_receipt_ref",
        "target_factor_generation_ref",
        "maintenance_receipt_path",
        "maintenance_receipt_sha256",
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
        "target_date",
        "prepared_at",
        "actor_uid",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
    "factor.production_rollover_prepared": (
        "factor_production_rollover_prepared_id",
        "state",
        "activation_scope",
        "rollover_bundle_ref",
        "expected_pointer_sha256",
        "target_pointer_sha256",
        "prepared_at",
        "actor_uid",
    ),
    "factor.production_rollover_commit": (
        "factor_production_rollover_commit_id",
        "state",
        "activation_scope",
        "rollover_bundle_ref",
        "rollover_prepared_ref",
        "previous_pointer_sha256",
        "target_pointer_sha256",
        "committed_at",
        "actor_uid",
        "cas_performed",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ),
}
_FACTOR_IDENTITIES: Final = {
    "factor.bootstrap_set": "bootstrap_set_id",
    "factor.preregistration": "preregistration_id",
    "factor.configuration_selection": "selection_id",
    "factor.prospective_observation": "observation_id",
    "factor.prospective_evaluation": "evaluation_id",
    "factor.admitted_set": "admitted_set_id",
    "factor.status": "status_id",
    "factor.canonical_replay_evidence": "replay_evidence_id",
    "factor.bootstrap_exception_evidence": "bootstrap_evidence_id",
    "factor.validation_receipt": "validation_receipt_id",
    "factor.observation_head": "observation_head_id",
    "factor.execution_turnover_evidence": "execution_evidence_id",
    "factor.validator_manifest": "validator_manifest_id",
    "factor.source_decode_attestation": "source_decode_attestation_id",
    "factor.signal_capture": "signal_capture_id",
    "factor.custody_record": "custody_record_id",
    "factor.composite_state": "composite_state_id",
    "factor.contextual_validation_result": "contextual_result_id",
    "factor.production_market_pit_selection": "market_pit_selection_id",
    "factor.production_source_closure": "factor_production_source_closure_id",
    "factor.production_recomputation_evidence": "factor_production_recomputation_id",
    "factor.production_legacy_zero_call_certificate": "factor_legacy_zero_call_id",
    "factor.production_market_input": "factor_market_input_id",
    "factor.production_calendar_capture_custody_attestation": (
        "calendar_capture_custody_attestation_id"
    ),
    "factor.production_generation": "factor_production_generation_id",
    "factor.production_pointer": "factor_production_pointer_id",
    "factor.production_generation_receipt": "factor_production_receipt_id",
    "factor.production_activation_bundle": "factor_production_activation_id",
    "factor.production_prepared": "factor_production_prepared_id",
    "factor.production_marker": "factor_production_marker_id",
    "factor.production_rollover_bundle": "factor_production_rollover_bundle_id",
    "factor.production_rollover_prepared": "factor_production_rollover_prepared_id",
    "factor.production_rollover_commit": "factor_production_rollover_commit_id",
}

FACTOR_CONTRACTS: Final = tuple(
    _exact_contract(
        kind,
        _FACTOR_IDENTITIES[kind],
        fields,
        validator={
            "factor.status": _validate_factor_status,
            "factor.validator_manifest": _validate_factor_validator_manifest,
            "factor.source_decode_attestation": _validate_factor_source_decode_attestation,
            "factor.contextual_validation_result": _validate_factor_contextual_result,
        }.get(kind),
    )
    for kind, fields in _FACTOR_FIELD_SETS.items()
)
FACTOR_CANONICAL_REPLAY_EVIDENCE_CONTRACT: Final = next(
    definition
    for definition in FACTOR_CONTRACTS
    if definition.kind == "factor.canonical_replay_evidence"
)


_RESEARCH_COMMON_FIELDS: Final = frozenset(
    {"authority", "production", "research_only", "run_state"}
)
_INTELLIGENCE_SPECS: Final = {
    "research_request": (
        "request_id",
        _RESEARCH_COMMON_FIELDS | {"as_of", "input_refs", "stages", "status", "strategy_id"},
    ),
    "research_evaluation": (
        "evaluation_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "blocker_codes",
            "evaluated_at",
            "request_ref",
            "stage_rows",
            "status",
            "strategy_id",
        },
    ),
    "evidence_bundle": (
        "bundle_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "blocker_codes",
            "compiled_at",
            "evaluation_ref",
            "evidence_refs",
            "status",
            "strategy_id",
        },
    ),
    "intelligence_inspection": (
        "inspection_id",
        _RESEARCH_COMMON_FIELDS
        | {"blocker_codes", "inspected_at", "status", "target_kind", "target_ref"},
    ),
    "decision_context": (
        "context_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "blocker_codes",
            "company_code",
            "component_refs",
            "evidence_refs",
            "hard_risk_codes",
            "hypothesis_status",
            "risk_status",
            "status",
        },
    ),
    "investment_decision": (
        "decision_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "blocker_codes",
            "company_code",
            "context_ref",
            "deterministic_percentile",
            "reason_codes",
            "state",
            "thresholds",
        },
    ),
    "industry_assessment": (
        "assessment_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "company_code",
            "component_score",
            "component_status",
            "exposures",
            "metric_rows",
            "primary_industry_id",
            "provider",
            "reason_codes",
            "status",
        },
    ),
    "theme_assessment": (
        "assessment_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "company_code",
            "component_score",
            "component_status",
            "exposures",
            "hard_veto_codes",
            "overall_severity",
            "provider",
            "reason_codes",
            "risk_rows",
            "status",
        },
    ),
    "fundamental_assessment": (
        "assessment_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "blocker_codes",
            "company_code",
            "component_rows",
            "coverage",
            "effective_score",
            "industry_assessment_ref",
            "minimum_coverage",
            "raw_score",
            "score_present",
            "source_refs",
            "status",
            "theme_assessment_ref",
        },
    ),
    "advisory_review": (
        "review_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "absolute_delta",
            "advisory_percentile",
            "as_of",
            "company_code",
            "decision_ref",
            "deterministic_decision_state",
            "deterministic_percentile",
            "reason_codes",
            "status",
            "validated_facts",
        },
    ),
    "research_portfolio": (
        "portfolio_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "blocker_codes",
            "cash_weight",
            "decision_refs",
            "gross_weight",
            "hard_veto_codes",
            "status",
            "strategy_id",
            "targets",
        },
    ),
    "paper_observation": (
        "observation_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "benchmark_return",
            "drawdown",
            "estimated_cost",
            "excess_return",
            "gross_return",
            "net_return",
            "portfolio_ref",
            "status",
            "strategy_id",
        },
    ),
    "graduation_assessment": (
        "assessment_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "assessed_at",
            "blocker_codes",
            "cumulative_excess_return",
            "observation_refs",
            "observation_count",
            "status",
            "strategy_id",
            "worst_drawdown",
        },
    ),
    "mainline_candidate": (
        "candidate_id",
        _RESEARCH_COMMON_FIELDS
        | {
            "as_of",
            "decision_ref",
            "evidence_bundle_ref",
            "investment_state",
            "portfolio_ref",
            "result",
            "status",
            "strategy_id",
        },
    ),
    "public_run": (
        "run_id",
        {
            "candidate_ref",
            "active_generation_id",
            "investment_state",
            "readiness_ref",
            "result",
            "status",
            "strategy_id",
        },
    ),
}

INTELLIGENCE_CONTRACTS: Final = tuple(
    _exact_contract(kind, identity_field, set(fields) | {identity_field})
    for kind, (identity_field, fields) in _INTELLIGENCE_SPECS.items()
)


READINESS_FIELDS: Final = frozenset(
    {
        "readiness_id",
        "factor_state",
        "factor_status_ref",
        "admission_route",
        "producer_identity",
        "mainline_state",
        "mainline_candidate_ref",
        "investment_state",
        "blockers",
    }
)
INTELLIGENCE_READINESS_CONTRACT: Final = _exact_contract(
    "intelligence_readiness", "readiness_id", READINESS_FIELDS
)


_MIGRATION_SPECS: Final = {
    "system.fundamental_veto_subject": (
        "veto_subject_id",
        {
            "veto_subject_id",
            "state",
            "bootstrap_admission_intent_sha256",
            "deployed_release_ref",
            "release_code_manifest_sha256",
            "system_as_of_date",
            "calendar_compilation_ref",
            "exchange_calendar_ref",
            "current_market_pointer_ref",
            "current_pit_pointer_ref",
            "current_pit_membership_ref",
            "fundamental_pointer_ref",
            "fundamental_manifest_ref",
            "fundamental_table_refs",
            "fundamental_evidence_refs",
            "fundamental_provenance_binding_sha256",
            "fundamental_target_bindings_sha256",
            "fundamental_snapshot_cutoff_date",
            "factor_set_sha256",
            "factor_dependency_rows",
            "factor_dependency_sha256",
        },
    ),
    "system.fundamental_operator_veto": (
        "veto_id",
        {
            "veto_id",
            "state",
            "veto_subject_ref",
            "reason_codes",
            "issued_at",
            "actor_uid",
            "os_actor",
            "human_signature_claimed",
            "system_activation_authorized",
            "factor_activation_authorized",
            "portfolio_authority",
            "strategy_record_authority",
            "broker_authority",
            "order_authority",
            "trade_authority",
            "funds_transfer_authority",
        },
    ),
    "system.fundamental_advisory_evidence": (
        "fundamental_advisory_id",
        {
            "fundamental_advisory_id",
            "state",
            "veto_subject_ref",
            "operator_veto_ref",
            "integrity_status",
            "required_by_active_factor_set",
            "system_as_of_date",
            "fundamental_snapshot_cutoff_date",
            "calendar_age_days",
            "open_session_age",
            "latest_admitted_available_at",
            "last_refresh_basis",
            "disclosure_check",
            "freshness_policy",
            "default_action",
            "operator_veto_present",
            "effective_action",
            "factor_dependency_rows",
            "factor_dependency_sha256",
            "fundamental_machine_states",
            "source_limitations",
            "generic_json_max_bytes",
            "predecessor_manifest_max_bytes",
            "fundamental_parquet_max_bytes",
            "generic_replay_max_cells",
            "daily_replay_max_cells",
            "fundamental_table_source_rows",
            "predecessor_manifest_source_ref",
            "ordinary_json_source_refs",
        },
    ),
    "system.migration.inventory": (
        "inventory_id",
        {
            "inventory_id",
            "status",
            "rules_ref",
            "dynamic_import_allowlist_ref",
            "legacy_seed_manifest_ref",
            "legacy_custody_scope_ref",
            "bootstrap_decision_ref",
            "replacement_test_map_ref",
            "tracked_roots",
            "runtime_roots",
            "files",
            "edges",
            "summary",
        },
    ),
    "system.migration.archive_plan": (
        "archive_plan_id",
        {
            "archive_plan_id",
            "archive_root",
            "blocker_codes",
            "cutover_id",
            "entries",
            "inventory_ref",
            "status",
            "summary",
        },
    ),
    "system.migration.receipt": (
        "migration_receipt_id",
        {
            "migration_receipt_id",
            "status",
            "cutover_id",
            "inventory_ref",
            "archive_plan_ref",
            "rules_ref",
            "source_to_target_rules_ref",
            "source_to_target",
            "target_generation_id",
            "target_generation_manifest_path",
            "target_generation_manifest_ref",
            "target_release_manifest_ref",
            "target_active_pointer_path",
            "target_active_pointer_ref",
            "expected_active_pointer_sha256",
            "permanent_marker_path",
            "write_performed",
            "cas_performed",
            "blocker_codes",
            "summary",
        },
    ),
    "system.migration.complete": (
        "marker_id",
        {
            "marker_id",
            "status",
            "cutover_id",
            "migration_receipt_ref",
            "inventory_ref",
            "archive_plan_ref",
            "active_pointer_ref",
            "generation_manifest_ref",
            "generation_id",
            "permanent_marker_path",
            "migration_replay_refused",
            "legacy_replay_refused",
            "blocker_codes",
        },
    ),
    "system.activation_authorization": (
        "authorization_id",
        {
            "authorization_id",
            "state",
            "final_cutover_authorization_ref",
            "migration_receipt_ref",
            "target_generation_id",
            "target_generation_manifest_ref",
            "deployed_release_ref",
            "calendar_authority_policy_ref",
            "calendar_compilation_ref",
            "calendar_capability_ref",
            "calendar_capture_execution_ref",
            "calendar_authorization_basis",
            "calendar_source_limitations",
            "bootstrap_admission_intent_sha256",
            "factor_dependency_sha256",
            "fundamental_veto_subject_ref",
            "fundamental_operator_veto_ref",
            "fundamental_advisory_ref",
            "fundamental_advisory_authorized",
            "target_active_pointer",
            "target_active_pointer_ref",
            "target_active_pointer_path",
            "permanent_marker_ref",
            "permanent_marker_path",
            "expected_active_pointer_sha256",
            "prepared_at",
            "activated_at",
            "actor_uid",
            "os_actor",
        },
    ),
    "system.activation_prepared": (
        "transaction_id",
        {
            "transaction_id",
            "state",
            "activation_authorization_ref",
            "final_cutover_authorization_ref",
            "migration_receipt_ref",
            "bootstrap_admission_intent_sha256",
            "factor_dependency_sha256",
            "fundamental_veto_subject_ref",
            "fundamental_operator_veto_ref",
            "fundamental_advisory_ref",
            "target_active_pointer",
            "target_active_pointer_ref",
            "permanent_marker_ref",
            "expected_active_pointer_sha256",
            "prepared_at",
            "actor_uid",
        },
    ),
    "system.concurrent_task_handoff": (
        "handoff_id",
        {
            "handoff_id",
            "state",
            "task_name",
            "thread_id",
            "accepted_baseline_commit",
            "handoff_type",
            "task_commit",
            "task_tree",
            "path_rows",
            "focused_test_rows",
            "writer_ended",
            "main_clean",
            "readback_rows",
        },
    ),
    "system.main_checkout_adoption": (
        "adoption_id",
        {
            "adoption_id",
            "state",
            "task_name",
            "thread_id",
            "source_task_outcome",
            "handoff_type",
            "accepted_baseline_commit",
            "accepted_baseline_tree",
            "adoption_commit",
            "adoption_tree",
            "adoption_parent",
            "path_rows",
            "task_origin_paths",
            "orphan_paths",
            "disposition_rows",
            "focused_test_rows",
            "full_gate_refs",
            "source_task_completion",
            "writer_ended",
            "main_clean",
            "readback_rows",
            "user_authorization_basis",
            "task_authorship_claimed",
            "human_signature_claimed",
            "history_rewritten",
        },
    ),
    "system.legacy_source_disposition": (
        "disposition_id",
        {
            "disposition_id",
            "state",
            "source_commit",
            "rows",
            "blocked_unresolved_count",
        },
    ),
    "system.final_cutover_authorization": (
        "final_authorization_id",
        {
            "final_authorization_id",
            "state",
            "accepted_baseline_commit",
            "historical_integration_commit",
            "historical_dirty_evidence_ref",
            "concurrent_task_handoff_ref",
            "main_checkout_adoption_ref",
            "legacy_disposition_ref",
            "deployed_release_ref",
            "production_generation_manifest_ref",
            "production_bootstrap_receipt_ref",
            "calendar_authority_policy_ref",
            "calendar_compilation_ref",
            "calendar_capability_ref",
            "calendar_capture_execution_ref",
            "calendar_authorization_basis",
            "calendar_source_limitations",
            "calendar_policy_authorized",
            "bootstrap_admission_intent_sha256",
            "factor_dependency_sha256",
            "fundamental_veto_subject_ref",
            "fundamental_operator_veto_ref",
            "fundamental_advisory_ref",
            "fundamental_advisory_authorized",
            "release_commit",
            "release_tree",
            "final_integration_commit",
            "final_integration_tree",
            "ancestry_rows",
            "excluded_commit_rows",
            "final_worktree_inventory_sha256",
            "clean_checkout_readback_rows",
            "user_authorization_basis",
            "preflight_rows",
            "final_build_authorized",
            "cas_authorized",
        },
    ),
    "system.cutover_gate_evidence": (
        "evidence_id",
        {
            "evidence_id",
            "state",
            "gate_id",
            "runner_id",
            "runner_spec_sha256",
            "runner_code_sha256",
            "final_commit",
            "final_tree",
            "environment_sha256",
            "batch_results",
            "subject_ref",
            "started_at",
            "finished_at",
        },
    ),
    "system.release_install_evidence": (
        "release_install_id",
        {
            "release_install_id",
            "state",
            "final_commit",
            "final_tree",
            "code_tree_sha256",
            "git_code_manifest_sha256",
            "release_ref",
            "source_archive",
            "wheel",
            "install_root",
            "python_executable",
            "python_executable_sha256",
            "import_origin",
            "installed_code_manifest_sha256",
            "contract_catalog_sha256",
            "lockfile_sha256",
            "dependency_install_mode",
            "editable_install",
            "source_tree_import",
        },
    ),
    "system.production_bootstrap_receipt": (
        "production_bootstrap_receipt_id",
        {
            "production_bootstrap_receipt_id",
            "state",
            "bootstrap_operator_request_ref",
            "bootstrap_admission_intent_sha256",
            "source_root_id",
            "input_source_rows",
            "deployed_release_ref",
            "calendar_authority_policy_ref",
            "calendar_compilation_ref",
            "calendar_capability_ref",
            "calendar_capture_execution_ref",
            "calendar_authorization_basis",
            "calendar_source_limitations",
            "release_code_manifest_sha256",
            "generation_created_at",
            "expected_assembly_id",
            "generation_intent_sha256",
            "mainline_ref",
            "source_refs",
            "factor_source_object_refs",
            "factor_policy_ref",
            "factor_evidence_refs",
            "factor_active_set_ref",
            "factor_validation_attestation_ref",
            "readiness_matrix_ref",
            "emergency_controller_sha256",
            "skill_tree_sha256",
            "automation_semantic_sha256",
            "source_blockers",
            "fundamental_machine_states",
            "factor_dependency_rows",
            "factor_dependency_sha256",
            "fundamental_veto_subject_ref",
            "fundamental_operator_veto_ref",
            "fundamental_advisory_ref",
            "signal_statistics",
            "signal_statistics_sha256",
            "assembler_module_path",
            "assembler_code_sha256",
        },
    ),
}
MIGRATION_CONTRACTS: Final = tuple(
    _exact_contract(kind, identity_field, fields)
    for kind, (identity_field, fields) in _MIGRATION_SPECS.items()
)


SYSTEM_RELEASE_CONTRACT: Final = register_contract(
    ContractDefinition(
        kind="system.release",
        identity_field="release_id",
        required_payload_fields=frozenset(
            {
                "release_id",
                "state",
                "code_sha256",
                "wheel_sha256",
                "code_manifest_sha256",
            }
        ),
        forbidden_payload_fields=LEGACY_CONTRACT_FIELDS,
        validator=_validate_system_release_payload,
    )
)
SYSTEM_SOURCE_BUNDLE_CONTRACT: Final = _exact_contract(
    "system.source_bundle",
    "source_bundle_id",
    {"source_bundle_id", "state", "sources"},
)
SYSTEM_SOURCE_OBJECT_CONTRACT: Final = _exact_contract(
    "system.source_object",
    "source_object_id",
    {
        "source_object_id",
        "source_root_id",
        "relative_path",
        "media_type",
        "source_format",
        "byte_sha256",
    },
)
SYSTEM_INSTALLED_COMPONENT_MANIFEST_FIELDS: Final = frozenset(
    {
        "component_manifest_id",
        "component_id",
        "component_registry_sha256",
        "component_role",
        "package_name",
        "module_names",
        "entrypoints",
        "files",
        "release_manifest_ref",
        "installed_code_manifest_sha256",
        "allowed_source_formats",
        "fallback_allowed",
        "component_sha256",
        "outcome",
        "authority",
    }
)
SYSTEM_INSTALLED_COMPONENT_MANIFEST_CONTRACT: Final = _exact_contract(
    "system.installed_component_manifest",
    "component_manifest_id",
    SYSTEM_INSTALLED_COMPONENT_MANIFEST_FIELDS,
    validator=_validate_installed_component,
)
SYSTEM_VALIDATION_RUN_REQUEST_FIELDS: Final = frozenset(
    {
        "validation_request_id",
        "validation_profile_id",
        "component_registry_sha256",
        "validation_namespace_id",
        "release_manifest_ref",
        "factor_validator_manifest_ref",
        "intrinsic_receipt_ref",
        "candidate_state_ref",
    }
)
SYSTEM_VALIDATION_RUN_REQUEST_CONTRACT: Final = _exact_contract(
    "system.validation_run_request",
    "validation_request_id",
    SYSTEM_VALIDATION_RUN_REQUEST_FIELDS,
)
SYSTEM_VALIDATION_ATTESTATION_FIELDS: Final = frozenset(
    {
        "attestation_id",
        "validation_request_ref",
        "validation_profile_id",
        "component_registry_sha256",
        "validation_namespace_id",
        "validation_lane",
        "validation_intent_sha256",
        "validation_plan_sha256",
        "candidate_state_ref",
        "candidate_state_pointer_sha256",
        "contextual_result_ref",
        "intrinsic_receipt_ref",
        "policy_ref",
        "evidence_refs",
        "active_set_ref",
        "source_object_refs",
        "release_manifest_ref",
        "release_identity",
        "installed_code_manifest_sha256",
        "compiled_contracts",
        "factor_validator_manifest_ref",
        "contextual_validator_component_ref",
        "source_decoder_component_ref",
        "implementation_component_refs",
        "source_attestation_refs",
        "custody_record_refs",
        "custody_head_ref",
        "custody_tree_sha256",
        "factor_source_stat_tree_sha256",
        "factor_source_total_bytes",
        "maximum_total_factor_source_bytes",
        "validated_at",
        "clock_source",
        "outcome",
        "authority",
    }
)
SYSTEM_VALIDATION_ATTESTATION_CONTRACT: Final = _exact_contract(
    "system.validation_attestation",
    "attestation_id",
    SYSTEM_VALIDATION_ATTESTATION_FIELDS,
    validator=_validate_validation_attestation,
)
SYSTEM_ASSEMBLY_REQUEST_FIELDS: Final = frozenset(
    {
        "assembly_request_id",
        "generation_state",
        "release_manifest_ref",
        "source_refs",
        "factor_source_object_refs",
        "factor_policy_ref",
        "factor_evidence_refs",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "mainline_ref",
        "research_refs",
        "migration_receipt_ref",
        "migration_marker_ref",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "readiness_matrix_ref",
        "emergency_controller_sha256",
    }
)
SYSTEM_ASSEMBLY_REQUEST_CONTRACT: Final = _exact_contract(
    "system.assembly_request",
    "assembly_request_id",
    SYSTEM_ASSEMBLY_REQUEST_FIELDS,
)
SYSTEM_BOOTSTRAP_OPERATOR_REQUEST_FIELDS: Final = frozenset(
    {
        "bootstrap_operation_id",
        "bootstrap_admission_intent_sha256",
        "state",
        "source_root_id",
        "release_manifest_ref",
        "exchange_calendar_file_ref",
        "market_scope_file_ref",
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "market_table_file_refs",
        "pit_pointer_file_ref",
        "pit_generation_manifest_file_ref",
        "pit_membership_file_ref",
        "calendar_runtime_json_file_ref",
        "calendar_compilation_file_ref",
        "calendar_authority_policy_file_ref",
        "official_calendar_raw_file_refs",
        "official_calendar_capture_file_refs",
        "official_calendar_decoder_admission_file_refs",
        "official_calendar_index_closure_file_refs",
        "trusted_provider_calendar_raw_file_refs",
        "trusted_provider_calendar_capture_file_refs",
        "trusted_provider_calendar_capability_file_ref",
        "trusted_provider_calendar_capture_transaction_file_ref",
        "trusted_provider_calendar_capture_execution_file_ref",
        "trusted_provider_calendar_capture_success_file_ref",
        "trusted_provider_release_install_input_file_ref",
        "fundamental_pointer_file_ref",
        "fundamental_generation_manifest_file_ref",
        "fundamental_table_file_refs",
        "fundamental_evidence_file_refs",
        "fundamental_operator_veto_file_ref",
        "bootstrap_decision_file_ref",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "source_blockers",
        "trusted_at",
    }
)
SYSTEM_BOOTSTRAP_OPERATOR_REQUEST_CONTRACT: Final = _exact_contract(
    "system.bootstrap_operator_request",
    "bootstrap_operation_id",
    SYSTEM_BOOTSTRAP_OPERATOR_REQUEST_FIELDS,
)
SYSTEM_EXCHANGE_CALENDAR_MANIFEST_FIELDS: Final = frozenset(
    {
        "calendar_manifest_id",
        "state",
        "coverage_start_date",
        "cutoff_date",
        "timezone",
        "calendar_file_ref",
        "transform_code_sha256",
        "exchange_rows",
    }
)
SYSTEM_EXCHANGE_CALENDAR_MANIFEST_CONTRACT: Final = _exact_contract(
    "system.exchange_calendar_manifest",
    "calendar_manifest_id",
    SYSTEM_EXCHANGE_CALENDAR_MANIFEST_FIELDS,
)
SYSTEM_EXCHANGE_CALENDAR_CAPTURE_FIELDS: Final = frozenset(
    {
        "calendar_capture_id",
        "state",
        "evidence_role",
        "exchange_id",
        "issuer",
        "request_url",
        "effective_url",
        "redirect_chain",
        "request_headers",
        "response_headers",
        "http_status",
        "tls_verified",
        "captured_at",
        "raw_file_ref",
        "raw_sha256",
        "raw_byte_length",
        "raw_media_type",
        "decoder_admission_ref",
        "decoder_id",
        "decoder_sha256",
        "projection_sha256",
    }
)
SYSTEM_EXCHANGE_CALENDAR_CAPTURE_CONTRACT: Final = _exact_contract(
    "system.exchange_calendar_capture",
    "calendar_capture_id",
    SYSTEM_EXCHANGE_CALENDAR_CAPTURE_FIELDS,
)
SYSTEM_EXCHANGE_CALENDAR_DECODER_ADMISSION_FIELDS: Final = frozenset(
    {
        "decoder_admission_id",
        "state",
        "exchange_id",
        "evidence_role",
        "issuer",
        "endpoint_scheme",
        "endpoint_host",
        "endpoint_path_query_template",
        "issuer_category_id",
        "category_scope",
        "category_completeness_policy",
        "query_window_semantics",
        "required_query_parameters",
        "page_parameter",
        "cursor_parameter",
        "required_category_set_id",
        "discovery_start_date",
        "fixture_request_url",
        "fixture_effective_url",
        "fixture_redirect_chain",
        "fixture_tls_verified",
        "redirect_policy",
        "http_status",
        "raw_media_type",
        "response_headers",
        "fixture_raw_file_ref",
        "fixture_raw_sha256",
        "fixture_captured_at",
        "decoder_id",
        "decoder_sha256",
        "fixture_projection_sha256",
        "review_basis",
    }
)
SYSTEM_EXCHANGE_CALENDAR_DECODER_ADMISSION_CONTRACT: Final = _exact_contract(
    "system.exchange_calendar_decoder_admission",
    "decoder_admission_id",
    SYSTEM_EXCHANGE_CALENDAR_DECODER_ADMISSION_FIELDS,
)
SYSTEM_EXCHANGE_CALENDAR_INDEX_CLOSURE_FIELDS: Final = frozenset(
    {
        "index_closure_id",
        "state",
        "exchange_id",
        "issuer",
        "issuer_category_id",
        "required_category_set_id",
        "category_scope",
        "category_completeness_policy",
        "query_window_semantics",
        "root_capture_ref",
        "page_capture_refs",
        "reported_page_count",
        "reported_item_count",
        "observed_item_count",
        "discovery_publish_start_date",
        "discovery_publish_end_date",
        "calendar_effective_coverage_start_date",
        "calendar_effective_coverage_end_date",
        "entry_rows",
        "body_capture_refs",
        "pagination_complete",
        "discovery_window_complete",
        "calendar_coverage_bound",
        "unknown_relevant_count",
    }
)
SYSTEM_EXCHANGE_CALENDAR_INDEX_CLOSURE_CONTRACT: Final = _exact_contract(
    "system.exchange_calendar_index_closure",
    "index_closure_id",
    SYSTEM_EXCHANGE_CALENDAR_INDEX_CLOSURE_FIELDS,
)
SYSTEM_EXCHANGE_CALENDAR_COMPILATION_FIELDS: Final = frozenset(
    {
        "compilation_id",
        "state",
        "authority_route",
        "policy_ref",
        "coverage_start_date",
        "cutoff_date",
        "timezone",
        "compiler_relative_path",
        "compiler_code_sha256",
        "compiler_ast_sha256",
        "release_ref",
        "pit_exchange_ids",
        "market_session_dates_sha256",
        "source_exchange_rows",
        "source_capture_refs",
        "decoder_admission_refs",
        "index_closure_refs",
        "precedence_rules",
        "exchange_rows",
        "runtime_projection",
        "calendar_json_file_ref",
        "calendar_parquet_file_ref",
        "contradiction_rows",
    }
)
SYSTEM_EXCHANGE_CALENDAR_COMPILATION_CONTRACT: Final = _exact_contract(
    "system.exchange_calendar_compilation",
    "compilation_id",
    SYSTEM_EXCHANGE_CALENDAR_COMPILATION_FIELDS,
)
SYSTEM_CALENDAR_AUTHORITY_POLICY_FIELDS: Final = frozenset(
    {
        "calendar_authority_policy_id",
        "state",
        "authority_route",
        "requested_scope",
        "authority_tier",
        "confidence",
        "expected_compilation_kind",
        "direct_exchange_official_calendar_exchange_ids",
        "direct_provider_calendar_exchange_ids",
        "unsupported_or_undocumented_probe_exchange_ids",
        "policy_projected_calendar_exchange_ids",
        "provider_capability_ref",
        "source_limitations",
        "requires_explicit_final_cutover_authorization",
        "user_authorization_basis",
        "human_signature_claimed",
        "time_semantics",
        "envelope_source",
        "timezone",
        "processing_open_local",
        "processing_close_local",
        "full_exchange_session_authority_available",
    }
)
SYSTEM_CALENDAR_AUTHORITY_POLICY_CONTRACT: Final = _exact_contract(
    "system.calendar_authority_policy",
    "calendar_authority_policy_id",
    SYSTEM_CALENDAR_AUTHORITY_POLICY_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FIELDS: Final = frozenset(
    {
        "calendar_capture_id",
        "state",
        "evidence_role",
        "provider",
        "api_name",
        "exchange_id",
        "endpoint_url",
        "request_parameters_sanitized",
        "request_parameters_sha256",
        "expected_fields",
        "captured_at",
        "http_status",
        "tls_verified",
        "redirect_chain",
        "response_headers",
        "raw_file_ref",
        "raw_sha256",
        "raw_byte_length",
        "request_id_sha256",
        "provider_reported_count",
        "item_count",
        "normalized_count",
        "has_more",
        "capture_start_date",
        "cutoff_date",
        "projection_sha256",
        "calendar_authority_conferred",
        "capability_ref",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capture",
    "calendar_capture_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPABILITY_FIELDS: Final = frozenset(
    {
        "calendar_capability_id",
        "state",
        "provider",
        "api_name",
        "docs_url",
        "docs_captured_at",
        "docs_http_status",
        "docs_tls_verified",
        "docs_redirect_chain",
        "docs_response_headers",
        "docs_raw_file_ref",
        "docs_raw_sha256",
        "docs_raw_byte_length",
        "decoder_id",
        "decoder_relative_path",
        "decoder_code_sha256",
        "decoder_ast_sha256",
        "decoder_projection_sha256",
        "documented_input_exchange_ids",
        "documented_stock_output_exchange_ids",
        "documented_fields",
        "bse_documentation_state",
        "conclusion",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPABILITY_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capability",
    "calendar_capability_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPABILITY_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_COMPILATION_FIELDS: Final = frozenset(
    {
        "compilation_id",
        "state",
        "authority_route",
        "authority_tier",
        "confidence",
        "policy_ref",
        "provider_capability_ref",
        "release_ref",
        "coverage_start_date",
        "capture_start_date",
        "cutoff_date",
        "timezone",
        "pit_exchange_ids",
        "direct_provider_calendar_exchange_ids",
        "unsupported_or_undocumented_probe_exchange_ids",
        "policy_projected_calendar_exchange_ids",
        "provider_capture_refs",
        "source_limitations",
        "time_semantics",
        "envelope_source",
        "processing_open_local",
        "processing_close_local",
        "full_exchange_session_authority_available",
        "projection_source_exchange_ids",
        "anchor_open_date",
        "predecessor_open_date",
        "capture_projection_sha256",
        "market_session_dates_sha256",
        "exchange_rows",
        "runtime_projection",
        "calendar_json_file_ref",
        "calendar_parquet_file_ref",
        "contradiction_rows",
        "compiler_relative_path",
        "compiler_code_sha256",
        "compiler_ast_sha256",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_COMPILATION_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_compilation",
    "compilation_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_COMPILATION_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_TRANSACTION_FIELDS: Final = frozenset(
    {
        "capture_transaction_id",
        "state",
        "capture_root_name",
        "capture_start_date",
        "cutoff_date",
        "captured_at",
        "documentation_raw_file_ref",
        "capability_file_ref",
        "policy_file_ref",
        "provider_raw_file_refs",
        "provider_capture_file_refs",
        "network_call_count",
        "source_limitations",
        "all_leaves_sha256",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_TRANSACTION_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capture_transaction",
    "capture_transaction_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_TRANSACTION_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_EXECUTION_FIELDS: Final = frozenset(
    {
        "capture_execution_id",
        "state",
        "capture_root_name",
        "deployed_release_ref",
        "release_install_input_file_ref",
        "release_install_evidence_ref",
        "release_install_verification_sha256",
        "release_repository_root",
        "final_commit",
        "final_tree",
        "wheel_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
        "installed_import_origin",
        "operator_relative_path",
        "operator_code_sha256",
        "operator_ast_sha256",
        "documentation_raw_file_ref",
        "capability_file_ref",
        "policy_file_ref",
        "provider_raw_file_refs",
        "provider_capture_file_refs",
        "capture_transaction_file_ref",
        "network_call_count",
        "operation_spec",
        "observed_started_at",
        "observed_completed_at",
        "source_limitations",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_EXECUTION_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capture_execution",
    "capture_execution_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_EXECUTION_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_SUCCESS_FIELDS: Final = frozenset(
    {
        "capture_success_id",
        "state",
        "capture_root_name",
        "capture_transaction_file_ref",
        "capture_execution_file_ref",
        "published_leaf_file_refs",
        "published_leaves_sha256",
        "published_root_device",
        "published_root_inode",
        "observed_completed_at",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_SUCCESS_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capture_success",
    "capture_success_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_SUCCESS_FIELDS,
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FAILURE_FIELDS: Final = frozenset(
    {
        "capture_failure_id",
        "state",
        "capture_root_name",
        "failed_at",
        "error_code",
        "success_root_published",
        "published_root_device",
        "published_root_inode",
    }
)
SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FAILURE_CONTRACT: Final = _exact_contract(
    "system.trusted_provider_calendar_capture_failure",
    "capture_failure_id",
    SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FAILURE_FIELDS,
)
SYSTEM_READINESS_CONTRACT: Final = _exact_contract(
    "system.readiness", "readiness_id", READINESS_FIELDS
)
SYSTEM_GENERATION_MANIFEST_FIELDS: Final = frozenset(
    {
        "assembly_id",
        "generation_state",
        "contract_catalog_sha256",
        "release_manifest_ref",
        "source_refs",
        "factor_source_object_refs",
        "factor_policy_ref",
        "factor_evidence_refs",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "mainline_ref",
        "research_refs",
        "migration_receipt_ref",
        "migration_marker_ref",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "readiness_matrix_ref",
        "emergency_controller_sha256",
    }
)
SYSTEM_GENERATION_MANIFEST_CONTRACT: Final = _exact_contract(
    "system.generation_manifest",
    "assembly_id",
    SYSTEM_GENERATION_MANIFEST_FIELDS,
)

_freeze_contract_registry()


__all__ = [
    "FACTOR_CANONICAL_REPLAY_EVIDENCE_CONTRACT",
    "FACTOR_CONTRACTS",
    "INTELLIGENCE_CONTRACTS",
    "INTELLIGENCE_READINESS_CONTRACT",
    "MIGRATION_CONTRACTS",
    "READINESS_FIELDS",
    "SYSTEM_ASSEMBLY_REQUEST_CONTRACT",
    "SYSTEM_ASSEMBLY_REQUEST_FIELDS",
    "SYSTEM_BOOTSTRAP_OPERATOR_REQUEST_CONTRACT",
    "SYSTEM_BOOTSTRAP_OPERATOR_REQUEST_FIELDS",
    "SYSTEM_CALENDAR_AUTHORITY_POLICY_CONTRACT",
    "SYSTEM_CALENDAR_AUTHORITY_POLICY_FIELDS",
    "SYSTEM_EXCHANGE_CALENDAR_MANIFEST_CONTRACT",
    "SYSTEM_EXCHANGE_CALENDAR_MANIFEST_FIELDS",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPABILITY_CONTRACT",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPABILITY_FIELDS",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_CONTRACT",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_CAPTURE_FIELDS",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_COMPILATION_CONTRACT",
    "SYSTEM_TRUSTED_PROVIDER_CALENDAR_COMPILATION_FIELDS",
    "SYSTEM_EXCHANGE_CALENDAR_DECODER_ADMISSION_CONTRACT",
    "SYSTEM_EXCHANGE_CALENDAR_DECODER_ADMISSION_FIELDS",
    "SYSTEM_GENERATION_MANIFEST_CONTRACT",
    "SYSTEM_GENERATION_MANIFEST_FIELDS",
    "SYSTEM_INSTALLED_COMPONENT_MANIFEST_CONTRACT",
    "SYSTEM_INSTALLED_COMPONENT_MANIFEST_FIELDS",
    "SYSTEM_READINESS_CONTRACT",
    "SYSTEM_RELEASE_CONTRACT",
    "SYSTEM_SOURCE_BUNDLE_CONTRACT",
    "SYSTEM_SOURCE_OBJECT_CONTRACT",
    "SYSTEM_VALIDATION_ATTESTATION_CONTRACT",
    "SYSTEM_VALIDATION_ATTESTATION_FIELDS",
    "SYSTEM_VALIDATION_RUN_REQUEST_CONTRACT",
    "SYSTEM_VALIDATION_RUN_REQUEST_FIELDS",
]
