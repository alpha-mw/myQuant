"""Stable CLI handlers for the unified runtime.

The handlers are intentionally thin.  They either read verified active state or
seal inactive candidate artifacts from one exact local request.  None of the
Factor or Research handlers can call System activation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError


def _exact_fields(value: Any, fields: set[str], *, code: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise CommandError(code)
    return dict(value)


def _request(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> tuple[bytes, dict[str, Any]]:
    return read_exact_request(workspace_root, request_path, expected_request_sha256)


def _optional_mapping(
    *,
    workspace_root: str,
    request_path: str | None,
    expected_request_sha256: str | None,
    code: str,
) -> dict[str, Any] | None:
    if (request_path is None) != (expected_request_sha256 is None):
        raise CommandError(code)
    if request_path is None:
        return None
    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256 or "",
    )
    return document


def system_status(
    *,
    workspace_root: str,
    deployed_release_ref_path: str | None = None,
    expected_deployed_release_ref_sha256: str | None = None,
    external_routing_path: str | None = None,
    expected_external_routing_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a compact status projection even when blocked or suspended."""

    from quant_investor.system import SystemStore, validate_object_ref

    deployed_document = _optional_mapping(
        workspace_root=workspace_root,
        request_path=deployed_release_ref_path,
        expected_request_sha256=expected_deployed_release_ref_sha256,
        code="DEPLOYED_RELEASE_REF_ARGUMENTS_INVALID",
    )
    deployed_ref = (
        None
        if deployed_document is None
        else validate_object_ref(deployed_document, label="deployed_release_ref")
    )
    external_routing = _optional_mapping(
        workspace_root=workspace_root,
        request_path=external_routing_path,
        expected_request_sha256=expected_external_routing_sha256,
        code="EXTERNAL_ROUTING_ARGUMENTS_INVALID",
    )
    observed = SystemStore(workspace_root).status(
        deployed_release_ref=deployed_ref,
        external_routing=external_routing,
    )
    readiness = observed.get("readiness")
    capabilities = {
        "factor": "UNINITIALIZED",
        "investment": "BLOCKED",
        "mainline": "UNINITIALIZED",
        "system": observed.get("state", "BLOCKED"),
    }
    if type(readiness) is dict:
        capabilities.update(
            {
                "factor": readiness.get("factor_state", "BLOCKED"),
                "investment": readiness.get("investment_state", "BLOCKED"),
                "mainline": readiness.get("mainline_state", "BLOCKED"),
            }
        )
    return {
        "status": "OK",
        "active_generation_id": observed.get("generation_id"),
        "capabilities": capabilities,
        "calendar_authority_route": observed.get("calendar_authority_route"),
        "calendar_authority_confidence": observed.get("calendar_authority_confidence"),
        "calendar_source_limitations": list(observed.get("calendar_source_limitations", [])),
        "external_routing_state": observed.get("external_routing_state", "BLOCKED"),
        "blockers": list(observed.get("blockers", [])),
    }


def system_verify(
    *,
    workspace_root: str,
    generation_id: str | None,
    deployed_release_ref_path: str | None = None,
    expected_deployed_release_ref_sha256: str | None = None,
) -> dict[str, Any]:
    from quant_investor.system import SystemStore, validate_object_ref

    deployed_document = _optional_mapping(
        workspace_root=workspace_root,
        request_path=deployed_release_ref_path,
        expected_request_sha256=expected_deployed_release_ref_sha256,
        code="DEPLOYED_RELEASE_REF_ARGUMENTS_INVALID",
    )
    deployed_ref = (
        None
        if deployed_document is None
        else validate_object_ref(deployed_document, label="deployed_release_ref")
    )
    verified = SystemStore(workspace_root).verify(
        generation_id or None, deployed_release_ref=deployed_ref
    )
    if verified.get("generation_id") is None:
        return {
            "status": "UNINITIALIZED",
            "active_generation_id": None,
            "generation_state": "UNINITIALIZED",
            "verified": False,
            "blockers": list(verified.get("blockers", [])),
        }
    if (
        generation_id is None
        and verified.get("generation_state") == "OPERATIONAL"
        and verified.get("deployed_release_verified") is not True
    ):
        return {
            "status": "BLOCKED",
            "active_generation_id": verified["generation_id"],
            "generation_state": verified["generation_state"],
            "verified": False,
            "blockers": ["SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"],
        }
    return {
        "status": "VERIFIED",
        "active_generation_id": verified["generation_id"],
        "generation_state": verified["generation_state"],
        "verified": verified["verified"],
        "calendar_authority_route": verified.get("calendar_authority_route"),
        "calendar_authority_confidence": verified.get("calendar_authority_confidence"),
        "calendar_source_limitations": list(verified.get("calendar_source_limitations", [])),
        "blockers": [],
    }


def system_assemble(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.system import SystemStore

    raw, _ = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    generation = SystemStore(workspace_root).assemble_from_request(raw)
    return {
        "status": "ASSEMBLED",
        "generation_id": generation["generation_id"],
        "generation_state": generation["generation_state"],
        "manifest_sha256": generation["manifest_sha256"],
    }


def system_bootstrap_assemble(
    *,
    workspace_root: str,
    input_root: str,
    request_path: str,
    expected_request_sha256: str,
) -> dict[str, Any]:
    """Run the production bootstrap assembler without activating System."""

    from quant_investor.factors.governance.production import (
        assemble_production_bootstrap,
    )

    raw, _ = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    return assemble_production_bootstrap(
        workspace_root=workspace_root,
        input_root=Path(workspace_root) / input_root,
        request_raw=raw,
    )


def system_calendar_capture(
    *,
    workspace_root: str,
    capture_parent: str,
    capture_root_name: str,
    cutoff_date: str,
    release_repository_root: str,
    release_install_input_path: str,
    expected_release_install_input_sha256: str,
) -> dict[str, Any]:
    """Capture one immutable Tushare Tier-1 calendar transaction."""

    from quant_investor.market.tushare_calendar_authority import (
        capture_trusted_provider_calendar_evidence,
    )

    release_input_raw, _ = _request(
        workspace_root=workspace_root,
        request_path=release_install_input_path,
        expected_request_sha256=expected_release_install_input_sha256,
    )
    return capture_trusted_provider_calendar_evidence(
        capture_parent=capture_parent,
        capture_root_name=capture_root_name,
        cutoff_date=cutoff_date,
        release_install_input_raw=release_input_raw,
        expected_release_install_input_sha256=expected_release_install_input_sha256,
        release_repository_root=release_repository_root,
    )


def system_activate(
    *,
    workspace_root: str,
    generation_id: str,
    expected_pointer_sha256: str,
    migration_receipt_path: str,
    expected_migration_receipt_sha256: str,
    final_cutover_authorization_path: str,
    expected_final_cutover_authorization_sha256: str,
    activation_authorization_path: str,
    expected_activation_authorization_sha256: str,
    target_active_pointer_path: str,
    expected_target_active_pointer_sha256: str,
    deployed_release_ref_path: str,
    expected_deployed_release_ref_sha256: str,
) -> dict[str, Any]:
    from quant_investor.system import SystemStore, validate_object_ref

    receipt_raw, _ = _request(
        workspace_root=workspace_root,
        request_path=migration_receipt_path,
        expected_request_sha256=expected_migration_receipt_sha256,
    )
    final_authorization_raw, _ = _request(
        workspace_root=workspace_root,
        request_path=final_cutover_authorization_path,
        expected_request_sha256=expected_final_cutover_authorization_sha256,
    )
    authorization_raw, _ = _request(
        workspace_root=workspace_root,
        request_path=activation_authorization_path,
        expected_request_sha256=expected_activation_authorization_sha256,
    )
    pointer_raw, pointer_document = _request(
        workspace_root=workspace_root,
        request_path=target_active_pointer_path,
        expected_request_sha256=expected_target_active_pointer_sha256,
    )
    deployed_document = _optional_mapping(
        workspace_root=workspace_root,
        request_path=deployed_release_ref_path,
        expected_request_sha256=expected_deployed_release_ref_sha256,
        code="DEPLOYED_RELEASE_REF_ARGUMENTS_INVALID",
    )
    if deployed_document is None:
        raise CommandError("DEPLOYED_RELEASE_REF_REQUIRED")
    deployed_ref = validate_object_ref(deployed_document, label="deployed_release_ref")
    if expected_pointer_sha256 != "EMPTY":
        raise CommandError("INITIAL_ACTIVATION_EXPECTED_EMPTY_REQUIRED")
    if (
        type(pointer_document) is not dict
        or pointer_document.get("generation_id") != generation_id
        or pointer_document.get("previous_pointer_sha256") != expected_pointer_sha256
    ):
        raise CommandError("ACTIVATION_POINTER_ARGUMENT_MISMATCH")
    active = SystemStore(workspace_root).activate_initial_generation(
        target_active_pointer_raw=pointer_raw,
        migration_receipt_raw=receipt_raw,
        final_cutover_authorization_raw=final_authorization_raw,
        activation_authorization_raw=authorization_raw,
        deployed_release_ref=deployed_ref,
    )
    activation = active["activation"]
    return {
        "status": "ACTIVATED",
        "active_generation_id": active["generation_id"],
        "generation_state": active["generation_state"],
        "pointer_byte_sha256": active["pointer_byte_sha256"],
        "pointer_semantic_sha256": active["pointer_byte_sha256"],
        "migration_receipt_ref": activation["migration_receipt_ref"],
        "authorization_ref": activation["authorization_ref"],
        "final_cutover_authorization_ref": activation["final_cutover_authorization_ref"],
        "marker_byte_sha256": activation["marker_byte_sha256"],
        "marker_semantic_sha256": activation["marker_semantic_sha256"],
        "cas_performed": activation["cas_performed"],
        "marker_only_recovery": not activation["cas_performed"],
    }


def system_suspend(
    *,
    workspace_root: str,
    generation_id: str,
    expected_pointer_sha256: str,
    target_active_pointer_path: str,
    expected_target_active_pointer_sha256: str,
) -> dict[str, Any]:
    """Emergency-only CAS to an already-built minimal suspended generation."""

    from quant_investor.system import SystemStore

    store = SystemStore(workspace_root)
    if expected_pointer_sha256 == "EMPTY":
        raise CommandError("SUSPEND_EXPECTED_NONEMPTY_REQUIRED")
    pointer_raw, pointer = _request(
        workspace_root=workspace_root,
        request_path=target_active_pointer_path,
        expected_request_sha256=expected_target_active_pointer_sha256,
    )
    if type(pointer) is not dict or pointer.get("generation_id") != generation_id:
        raise CommandError("SUSPEND_POINTER_ARGUMENT_MISMATCH")
    active = store.activate_suspended_generation(
        target_active_pointer_raw=pointer_raw,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    return {
        "status": "SUSPENDED",
        "active_generation_id": active["generation_id"],
        "generation_state": active["generation_state"],
        "pointer_sha256": active["pointer_byte_sha256"],
        "cas_performed": True,
    }


def factor_status(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    """Build one Factor status from an exact, already-stored validation closure."""

    from quant_investor.factors.governance import (
        FactorValidationStore,
        validate_factor_status,
    )
    from quant_investor.system import SystemStore, object_ref_for_artifact

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {
            "active_contextual_result_ref",
            "active_factor_set_ref",
            "active_validation_attestation_ref",
            "active_validation_receipt_ref",
            "observed_composite_state_ref",
        },
        code="FACTOR_STATUS_REQUEST_INVALID",
    )
    store = FactorValidationStore(system_store=SystemStore(workspace_root))
    status = validate_factor_status(store.build_status(**values))
    payload = status["payload"]
    return {
        "blockers": list(payload["blockers"]),
        "readiness": payload["readiness"],
        "status_ref": object_ref_for_artifact(status),
    }


def factor_mine(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.factors.governance import FactorValidationStore
    from quant_investor.system import SystemStore

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {
            "exchange_calendar_ref",
            "expected_composite_state_ref",
            "factor_validator_manifest_ref",
            "implementation_manifest_ref",
        },
        code="FACTOR_MINE_REQUEST_INVALID",
    )
    store = FactorValidationStore(system_store=SystemStore(workspace_root))
    return _factor_composite_projection(store.mine(**values))


def _factor_composite_projection(artifact: Any) -> dict[str, Any]:
    from quant_investor.factors.governance import validate_composite_state
    from quant_investor.system import object_ref_for_artifact

    normalized = validate_composite_state(artifact)
    payload = normalized["payload"]
    return {
        "blockers": list(payload["blockers"]),
        "composite_state_ref": object_ref_for_artifact(normalized),
        "cycle_state": payload["cycle_state"],
        "terminal": payload["terminal"],
    }


def factor_observe(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.factors.governance import FactorValidationStore
    from quant_investor.system import SystemStore

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    action = document.get("action")
    if action == "SIGNAL":
        values = _exact_fields(
            document,
            {
                "action",
                "expected_composite_state_ref",
                "market_history_ref",
                "pit_universe_ref",
                "preregistration_ref",
                "selection_ref",
                "sparse_weights_ref",
            },
            code="FACTOR_OBSERVE_REQUEST_INVALID",
        )
        values.pop("action")
        store = FactorValidationStore(system_store=SystemStore(workspace_root))
        return _factor_composite_projection(store.observe_signal(**values))
    if action != "LABEL":
        raise CommandError("FACTOR_OBSERVE_ACTION_INVALID")
    values = _exact_fields(
        document,
        {
            "action",
            "expected_composite_state_ref",
            "matured_label_prices_ref",
            "preregistration_ref",
            "selection_ref",
            "signal_capture_ref",
        },
        code="FACTOR_OBSERVE_REQUEST_INVALID",
    )
    values.pop("action")
    store = FactorValidationStore(system_store=SystemStore(workspace_root))
    return _factor_composite_projection(store.observe_label(**values))


def factor_evaluate(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.system import SystemStore

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    action = document.get("action")
    if action == "REQUEST_CONTEXTUAL_VALIDATION":
        return _factor_contextual_evaluate(
            workspace_root=workspace_root,
            document=document,
        )
    allowed_actions = {
        "BUILD_ADMITTED_SET",
        "BUILD_INTRINSIC_RECEIPT",
        "EVALUATE_PREREGISTRATION",
        "FINALIZE_EXECUTION",
    }
    if action not in allowed_actions:
        raise CommandError("FACTOR_EVALUATE_ACTION_INVALID")
    values = _exact_fields(
        document,
        {
            "action",
            "expected_composite_state_ref",
            "preregistration_ref",
            "selection_ref",
        },
        code="FACTOR_EVALUATE_REQUEST_INVALID",
    )
    from quant_investor.factors.governance import FactorValidationStore

    store = FactorValidationStore(system_store=SystemStore(workspace_root))
    return _factor_composite_projection(store.evaluate(**values))


def _factor_contextual_evaluate(*, workspace_root: str, document: Any) -> dict[str, Any]:
    from quant_investor.system import SystemStore, validate_object_ref

    values = _exact_fields(
        document,
        {
            "action",
            "expected_composite_state_ref",
            "validation_run_request_ref",
        },
        code="FACTOR_EVALUATE_REQUEST_INVALID",
    )
    try:
        request_ref = validate_object_ref(
            values["validation_run_request_ref"],
            label="validation_run_request_ref",
        )
        expected_state_ref = values["expected_composite_state_ref"]
        if expected_state_ref is not None:
            expected_state_ref = validate_object_ref(
                expected_state_ref,
                label="expected_composite_state_ref",
            )
    except Exception as exc:
        raise CommandError("FACTOR_EVALUATE_REQUEST_INVALID") from exc
    if request_ref["kind"] != "system.validation_run_request" or (
        expected_state_ref is not None and expected_state_ref["kind"] != "factor.composite_state"
    ):
        raise CommandError("FACTOR_EVALUATE_REQUEST_INVALID")
    system_store = SystemStore(workspace_root)
    request_artifact = system_store.get_object(request_ref)
    request_payload = request_artifact.get("payload")
    if (
        request_artifact.get("kind") != "system.validation_run_request"
        or type(request_payload) is not dict
        or request_payload.get("candidate_state_ref") != expected_state_ref
    ):
        raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_STATE_MISMATCH")
    projected = _factor_contextual_validation_projection(system_store.run_validation(request_ref))
    if projected["validation_request_ref"] != request_ref:
        raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID")
    return projected


def _factor_contextual_validation_projection(result: Any) -> dict[str, Any]:
    from quant_investor.system import validate_object_ref

    fields = {
        "completion_sha256",
        "contextual_result_ref",
        "outcome",
        "validation_attestation_ref",
        "validation_request_ref",
    }
    if type(result) is not dict or not fields.issubset(result):
        raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID")
    projected = {field: result[field] for field in fields}
    expected_kinds = {
        "contextual_result_ref": "factor.contextual_validation_result",
        "validation_attestation_ref": "system.validation_attestation",
        "validation_request_ref": "system.validation_run_request",
    }
    for field, expected_kind in expected_kinds.items():
        try:
            reference = validate_object_ref(projected[field], label=field)
        except Exception as exc:
            raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID") from exc
        if reference["kind"] != expected_kind:
            raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID")
        projected[field] = reference
    completion_sha256 = projected["completion_sha256"]
    if (
        type(completion_sha256) is not str
        or len(completion_sha256) != 64
        or any(character not in "0123456789abcdef" for character in completion_sha256)
        or projected["outcome"] != "VALIDATED"
    ):
        raise CommandError("FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID")
    return projected


def factor_history(*, workspace_root: str) -> dict[str, Any]:
    """Read the verified current Factor lineage without consulting legacy stores."""

    from quant_investor.system import SystemStore

    store = SystemStore(workspace_root)
    active = store.read_active()
    if active is None:
        return {
            "status": "OK",
            "active_generation_id": None,
            "entries": [],
            "blockers": ["SYSTEM_ACTIVE_POINTER_ABSENT"],
        }
    if (
        active.get("generation_state") == "OPERATIONAL"
        and active.get("deployed_release_verified") is not True
    ):
        return {
            "status": "BLOCKED",
            "active_generation_id": active.get("generation_id"),
            "entries": [],
            "blockers": ["SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"],
        }
    factor_set = active.get("factor_active_set")
    factor_status_artifact = active.get("factor_status")
    entry = {
        "generation_id": active["generation_id"],
        "factor_set_ref": active["manifest"]["payload"]["factor_active_set_ref"],
        "factor_status_ref": active.get("factor_status_ref"),
        "observed_candidate_state": (
            factor_status_artifact.get("payload", {}).get("observed", {}).get("cycle_state")
            if type(factor_status_artifact) is dict
            else None
        ),
        "factor_ids": sorted(
            [
                row["factor_id"]
                for row in factor_set.get("payload", {}).get("factor_rows", [])
                if type(row) is dict and type(row.get("factor_id")) is str
            ]
            if type(factor_set) is dict
            else []
        ),
    }
    return {
        "status": "OK",
        "active_generation_id": active["generation_id"],
        "entries": [entry],
        "blockers": [],
    }


def research_forward(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.intelligence import forward

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {"created_at", "request", "request_id"},
        code="RESEARCH_FORWARD_REQUEST_INVALID",
    )
    return forward(
        values["request"],
        created_at=values["created_at"],
        request_id=values["request_id"],
    )


def research_evaluate(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.intelligence import evaluate

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {"evaluated_at", "evaluation_id", "request", "stage_results"},
        code="RESEARCH_EVALUATE_REQUEST_INVALID",
    )
    return evaluate(**values)


def research_compile_evidence(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.intelligence import compile_evidence

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {"bundle_id", "compiled_at", "evaluation", "evidence"},
        code="RESEARCH_COMPILE_EVIDENCE_REQUEST_INVALID",
    )
    return compile_evidence(**values)


def research_readiness(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.intelligence import assess_readiness

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {
            "assessed_at",
            "factor_status",
            "producer_identity",
            "readiness_id",
            "source_blockers",
        },
        code="RESEARCH_READINESS_REQUEST_INVALID",
    )
    return assess_readiness(**values)


def research_inspect(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    from quant_investor.intelligence import inspect

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {"artifact", "inspected_at", "inspection_id"},
        code="RESEARCH_INSPECT_REQUEST_INVALID",
    )
    return inspect(**values)


__all__ = [
    "factor_evaluate",
    "factor_history",
    "factor_mine",
    "factor_observe",
    "factor_status",
    "research_compile_evidence",
    "research_evaluate",
    "research_forward",
    "research_inspect",
    "research_readiness",
    "system_activate",
    "system_assemble",
    "system_status",
    "system_suspend",
    "system_verify",
]
