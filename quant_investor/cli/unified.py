"""Stable CLI handlers for the unified runtime.

The handlers are intentionally thin.  They either read verified active state or
seal inactive candidate artifacts from one exact local request.  None of the
Factor or Research handlers can call System activation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError
from quant_investor.contracts import canonical_json_bytes


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
        "fundamental_advisory": observed.get("fundamental_advisory"),
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
    fundamental_artifact = verified.get("fundamental_advisory")
    fundamental_advisory = (
        fundamental_artifact.get("payload")
        if type(fundamental_artifact) is dict and type(fundamental_artifact.get("payload")) is dict
        else None
    )
    if verified.get("generation_state") == "OPERATIONAL" and (
        fundamental_advisory is None or fundamental_advisory.get("effective_action") != "PROCEED"
    ):
        return {
            "status": "BLOCKED",
            "active_generation_id": verified["generation_id"],
            "generation_state": verified["generation_state"],
            "verified": False,
            "fundamental_advisory": fundamental_advisory,
            "blockers": ["SYSTEM_FUNDAMENTAL_ADVISORY_INVALID"],
        }
    return {
        "status": "VERIFIED",
        "active_generation_id": verified["generation_id"],
        "generation_state": verified["generation_state"],
        "verified": verified["verified"],
        "calendar_authority_route": verified.get("calendar_authority_route"),
        "calendar_authority_confidence": verified.get("calendar_authority_confidence"),
        "calendar_source_limitations": list(verified.get("calendar_source_limitations", [])),
        "fundamental_advisory": fundamental_advisory,
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


def system_bootstrap_admission_preflight(
    *,
    workspace_root: str,
    input_root: str,
    request_path: str,
    expected_request_sha256: str,
) -> dict[str, Any]:
    """Derive the immutable veto subject without building a generation."""

    from quant_investor.factors.governance.production import (
        prepare_production_bootstrap_admission,
    )

    raw, _ = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    return prepare_production_bootstrap_admission(
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


def system_release_prepare(
    *,
    workspace_root: str,
    release_root: str,
    release_repository_root: str,
    final_commit: str,
    final_tree: str,
) -> dict[str, Any]:
    """Build and publish installed-release evidence without runtime authority."""

    from quant_investor.system import (
        prepare_operational_release,
        publish_release_install_input,
    )

    prepared = prepare_operational_release(
        repository_root=release_repository_root,
        release_root=release_root,
        final_commit=final_commit,
        final_tree=final_tree,
        created_at=None,
    )
    return publish_release_install_input(
        workspace_root=workspace_root,
        release_root=release_root,
        release_install_evidence=prepared["release_install_evidence"],
        deployed_release=prepared["release"],
        repository_root=release_repository_root,
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


def factor_production_status(*, workspace_root: str) -> dict[str, Any]:
    """Return the isolated Factor production authority state without System access."""

    from quant_investor.factors.production_authority import verify_factor_production

    observed = verify_factor_production(workspace_root)
    return {
        "command_status": "COMPLETED",
        "authority_domain": "FACTOR_PRODUCTION_ONLY",
        "activation_scope": observed["activation_scope"],
        "factor_readiness": observed["factor_readiness"],
        "factor_authority": observed["factor_authority"],
        "factor_generation_id": observed.get("factor_generation_id"),
        "factor_generation_sha256": observed.get("factor_generation_sha256"),
        "factor_pointer_byte_sha256": observed.get("factor_pointer_byte_sha256"),
        "factor_pointer_semantic_sha256": observed.get("factor_pointer_semantic_sha256"),
        "marker_byte_sha256": observed.get("marker_byte_sha256"),
        "marker_semantic_sha256": observed.get("marker_semantic_sha256"),
        "as_of": observed.get("as_of"),
        "active_factors": list(observed.get("active_factors", [])),
        "control_factors": list(observed.get("control_factors", [])),
        "fundamental_dependency_state": observed.get("fundamental_dependency_state"),
        "fundamental_freshness_policy": observed.get("fundamental_freshness_policy"),
        "system_authority": observed.get("system_authority", "NONE"),
        "mainline_authority": observed.get("mainline_authority", "NONE"),
        "investment_authority": observed.get("investment_authority", "NONE"),
        "portfolio_authority": observed.get("portfolio_authority", "NONE"),
        "broker_authority": observed.get("broker_authority", "NONE"),
        "order_authority": observed.get("order_authority", "NONE"),
        "trade_authority": observed.get("trade_authority", "NONE"),
        "funds_transfer_authority": observed.get("funds_transfer_authority", "NONE"),
        "system_runtime_state": "NOT_EVALUATED",
        "grants_system_authority": False,
        "grants_trading_authority": False,
        "blockers": list(observed.get("blockers", [])),
    }


def factor_production_verify(*, workspace_root: str) -> dict[str, Any]:
    """Revalidate pointer, marker, generation, sources, and sealed signals."""

    status = factor_production_status(workspace_root=workspace_root)
    verified = status["factor_authority"] == "ACTIVE" and not status["blockers"]
    return {
        **status,
        "command_status": "VERIFIED" if verified else "BLOCKED",
        "verified": verified,
    }


def factor_production_signal(*, workspace_root: str, factor_id: str) -> dict[str, Any]:
    """Read one deterministic signal from the verified active generation."""

    from quant_investor.factors.production_authority import read_factor_production_signal

    return {
        "command_status": "VERIFIED_ACTIVE_SIGNAL",
        "authority_domain": "FACTOR_PRODUCTION_ONLY",
        "system_runtime_state": "NOT_EVALUATED",
        "grants_system_authority": False,
        "grants_trading_authority": False,
        **read_factor_production_signal(workspace_root, factor_id=factor_id),
    }


def factor_production_observe(*, workspace_root: str) -> dict[str, Any]:
    """Register immutable LOW/W80 observations for the verified active head."""

    from quant_investor.factors.production_observation import (
        register_factor_production_observations,
    )

    return register_factor_production_observations(workspace_root)


def _factor_activation_projection(
    activated: dict[str, Any],
    *,
    command_status: str,
    prepared_sources: dict[str, Any] | None,
) -> dict[str, Any]:
    activation = activated["activation"]
    return {
        "command_status": command_status,
        "authority_domain": "FACTOR_PRODUCTION_ONLY",
        "factor_authority": activated["factor_authority"],
        "factor_readiness": activated["factor_readiness"],
        "factor_generation_id": activated["factor_generation_id"],
        "factor_generation_sha256": activated["factor_generation_sha256"],
        "factor_pointer_byte_sha256": activated["factor_pointer_byte_sha256"],
        "factor_pointer_semantic_sha256": activated["factor_pointer_semantic_sha256"],
        "marker_byte_sha256": activated["marker_byte_sha256"],
        "marker_semantic_sha256": activated["marker_semantic_sha256"],
        "active_factors": activated["active_factors"],
        "control_factors": activated["control_factors"],
        "as_of": activated["as_of"],
        "operation_id": (None if prepared_sources is None else prepared_sources["operation_id"]),
        "operation_inputs_sha256": (
            None if prepared_sources is None else prepared_sources["operation_inputs_sha256"]
        ),
        "operation_inputs_ref": (
            None if prepared_sources is None else prepared_sources["operation_inputs_ref"]
        ),
        "cas_performed": activation["cas_performed"],
        "marker_only_recovery": activation["marker_only_recovery"],
        "system_runtime_state": "NOT_EVALUATED",
        "grants_system_authority": False,
        "grants_trading_authority": False,
        "broker_order_trade_fund_writes": 0,
    }


def factor_production_activate(  # noqa: C901 - one atomic public operator boundary
    *,
    workspace_root: str,
    market_data_root: str,
    calendar_capture_root: str,
    expected_calendar_success_sha256: str,
    expected_empty: bool,
) -> dict[str, Any]:
    """Prepare and perform the sole expected-EMPTY Factor production cutover."""

    from quant_investor.factors.governance.factor_production_prepare import (
        prepare_factor_production,
    )
    from quant_investor.factors.production_authority import (
        FACTOR_AUTHORITY_ACTIVE,
        FactorProductionStore,
        verify_factor_production,
    )
    from quant_investor.system import SystemStore

    if expected_empty is not True:
        raise CommandError("FACTOR_EXPECTED_EMPTY_REQUIRED")
    initial = verify_factor_production(workspace_root)
    if initial.get("factor_authority") == "BLOCKED" and initial.get("blockers") == [
        "FACTOR_PRODUCTION_MARKER_ABSENT"
    ]:
        recovered = FactorProductionStore(
            workspace_root
        ).recover_initial_marker_from_active_pointer()
        if recovered.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE:
            raise CommandError("FACTOR_PRODUCTION_FINAL_VERIFY_FAILED")
        return _factor_activation_projection(
            recovered,
            command_status="MARKER_RECOVERED",
            prepared_sources=None,
        )
    if initial.get("factor_authority") != "INACTIVE" or initial.get("blockers") != [
        "FACTOR_ACTIVE_POINTER_ABSENT"
    ]:
        raise CommandError("FACTOR_EXPECTED_EMPTY_FAILED")
    prepared_sources = prepare_factor_production(
        workspace_root=workspace_root,
        market_data_root=market_data_root,
        calendar_capture_root=calendar_capture_root,
        expected_calendar_success_sha256=expected_calendar_success_sha256,
    )
    workspace = Path(workspace_root).resolve(strict=True)
    current_release_root = Path(prepared_sources["release_repository_root"]).resolve(strict=True)
    source_root = (workspace / prepared_sources["source_root"]).resolve(strict=True)
    if workspace not in source_root.parents:
        raise CommandError("FACTOR_PREPARED_SOURCE_ROOT_INVALID")
    source_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id=prepared_sources["source_root_id"],
    )
    generation = source_store.get_object(prepared_sources["factor_production_generation_ref"])
    source_closure = source_store.get_object(
        prepared_sources["factor_production_source_closure_ref"]
    )
    recomputation = source_store.get_object(prepared_sources["factor_production_recomputation_ref"])
    source_payload = source_closure["payload"]
    legacy = source_store.get_object(source_payload["legacy_zero_call_ref"])
    market_input = source_store.get_object(source_payload["market_input_ref"])
    factor_store = FactorProductionStore.from_system_source_custody(
        workspace,
        source_root=source_root,
        source_root_id=prepared_sources["source_root_id"],
        release_repository_root=current_release_root,
    )
    activated_at = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    prepared_activation = factor_store.prepare_initial_activation(
        factor_generation=generation,
        source_closure=source_closure,
        recomputation_evidence=recomputation,
        legacy_zero_call_certificate=legacy,
        market_input=market_input,
        prepared_at=activated_at,
        activated_at=activated_at,
    )
    activated = factor_store.activate_initial_generation(
        target_factor_pointer_raw=prepared_activation["target_factor_pointer_raw"],
        factor_generation_receipt_raw=prepared_activation["factor_generation_receipt_raw"],
        activation_bundle_raw=prepared_activation["activation_bundle_raw"],
        prepared_transaction_raw=prepared_activation["prepared_transaction_raw"],
        permanent_marker_raw=prepared_activation["permanent_marker_raw"],
    )
    if activated.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE:
        raise CommandError("FACTOR_PRODUCTION_FINAL_VERIFY_FAILED")
    return _factor_activation_projection(
        activated,
        command_status="ACTIVATED",
        prepared_sources=prepared_sources,
    )


def factor_production_rollover(  # noqa: C901 - one guarded successor boundary
    *,
    workspace_root: str,
    market_data_root: str,
    calendar_capture_root: str,
    expected_calendar_success_sha256: str,
    maintenance_receipt: str,
    expected_maintenance_receipt_sha256: str,
    expected_current_pointer_sha256: str,
) -> dict[str, Any]:
    """Prepare and commit one Factor-only successor after exact daily maintenance."""

    from quant_investor.factors.governance.factor_production_prepare import (
        prepare_factor_production,
    )
    from quant_investor.factors.production_authority import FactorProductionStore
    from quant_investor.factors.production_rollover import (
        canonical_input_closure,
        validate_daily_maintenance_receipt,
    )
    from quant_investor.system import SystemStore

    maintenance = validate_daily_maintenance_receipt(
        workspace_root=workspace_root,
        receipt_path=maintenance_receipt,
        expected_receipt_sha256=expected_maintenance_receipt_sha256,
    )
    factor_store = FactorProductionStore(workspace_root)
    current = factor_store.read("results/factors/_active.json")
    if current.byte_sha256 != expected_current_pointer_sha256:
        recovered = factor_store.recover_rollover_for_inputs(
            expected_pointer_sha256=expected_current_pointer_sha256,
            maintenance_sha256=expected_maintenance_receipt_sha256,
        )
        rollover = recovered["rollover"]
        return {
            **_factor_activation_projection(
                {**recovered, "activation": rollover},
                command_status=(
                    "ROLLOVER_COMMIT_RECOVERED"
                    if rollover["commit_recovered"]
                    else "ROLLOVER_IDEMPOTENT"
                ),
                prepared_sources=None,
            ),
            "previous_pointer_sha256": rollover["previous_pointer_sha256"],
            "rollover_commit_ref": rollover["rollover_commit_ref"],
            "prospective_evidence_status": "NOT_CONFIGURED",
            "prospective_write_performed": False,
            "upstream_maintenance_status": maintenance["upstream_maintenance_status"],
            "macro_status": maintenance["macro_status"],
            "macro_blockers": maintenance["macro_blockers"],
            "macro_used_by_factor": False,
        }
    current_generation = factor_store._read_generation_for_pointer(  # noqa: SLF001
        current.data, label="Factor rollover current"
    )
    current_as_of = current_generation["payload"]["as_of"]
    target_date = maintenance["target_date"]
    if target_date < current_as_of:
        raise CommandError("FACTOR_ROLLOVER_TARGET_PRECEDES_ACTIVE")
    canonical_inputs = canonical_input_closure(
        workspace_root=workspace_root, market_data_root=market_data_root
    )
    core_closure = maintenance["core_closure"]
    for canonical_key, core_key in (
        ("market_pointer_sha256", "market_pointer_sha256"),
        ("market_manifest_sha256", "market_manifest_sha256"),
        ("pit_pointer_sha256", "pit_pointer_sha256"),
        ("pit_manifest_sha256", "pit_manifest_sha256"),
    ):
        if canonical_inputs[canonical_key] != core_closure[core_key]:
            raise CommandError("FACTOR_ROLLOVER_MAINTENANCE_BINDING_MISMATCH")
    active_market = factor_store._read_artifact_ref(  # noqa: SLF001
        current_generation["payload"]["market_input_ref"],
        label="Factor rollover active Market input",
    )
    active_market_payload = active_market["payload"]
    same_input_binding = (
        active_market_payload["market_pointer_sha256"] == canonical_inputs["market_pointer_sha256"]
        and active_market_payload["market_snapshot_manifest_sha256"]
        == canonical_inputs["market_manifest_sha256"]
        and active_market_payload["pit_membership_sha256"] == core_closure["pit_membership_sha256"]
    )
    if target_date == current_as_of and same_input_binding:
        verified = factor_store.verify_active()
        if verified.get("factor_authority") != "ACTIVE" or verified.get("blockers"):
            raise CommandError("FACTOR_PRODUCTION_FINAL_VERIFY_FAILED")
        return {
            "command_status": "FACTOR_PRODUCTION_NO_ACTION",
            "authority_domain": "FACTOR_PRODUCTION_ONLY",
            "factor_authority": "ACTIVE",
            "factor_readiness": "READY",
            "factor_generation_id": verified["factor_generation_id"],
            "as_of": current_as_of,
            "factor_pointer_byte_sha256": current.byte_sha256,
            "cas_performed": False,
            "prospective_evidence_status": "NOT_CONFIGURED",
            "prospective_write_performed": False,
            "grants_system_authority": False,
            "grants_trading_authority": False,
            "broker_order_trade_fund_writes": 0,
            "upstream_maintenance_status": maintenance["upstream_maintenance_status"],
            "macro_status": maintenance["macro_status"],
            "macro_blockers": maintenance["macro_blockers"],
            "macro_used_by_factor": False,
        }
    prepared_sources = prepare_factor_production(
        workspace_root=workspace_root,
        market_data_root=market_data_root,
        calendar_capture_root=calendar_capture_root,
        expected_calendar_success_sha256=expected_calendar_success_sha256,
    )
    if prepared_sources.get("as_of") != target_date:
        raise CommandError("FACTOR_ROLLOVER_PREPARED_DATE_MISMATCH")
    workspace = Path(workspace_root).resolve(strict=True)
    current_release_root = Path(prepared_sources["release_repository_root"]).resolve(strict=True)
    source_root = (workspace / prepared_sources["source_root"]).resolve(strict=True)
    source_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id=prepared_sources["source_root_id"],
    )
    generation = source_store.get_object(prepared_sources["factor_production_generation_ref"])
    source_closure = source_store.get_object(
        prepared_sources["factor_production_source_closure_ref"]
    )
    recomputation = source_store.get_object(prepared_sources["factor_production_recomputation_ref"])
    source_payload = source_closure["payload"]
    legacy = source_store.get_object(source_payload["legacy_zero_call_ref"])
    market_input = source_store.get_object(source_payload["market_input_ref"])
    live_store = FactorProductionStore.from_system_source_custody(
        workspace,
        source_root=source_root,
        source_root_id=prepared_sources["source_root_id"],
        release_repository_root=current_release_root,
    )
    activated_at = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    prepared = live_store.prepare_rollover_activation(
        factor_generation=generation,
        source_closure=source_closure,
        recomputation_evidence=recomputation,
        legacy_zero_call_certificate=legacy,
        market_input=market_input,
        expected_pointer_sha256=expected_current_pointer_sha256,
        maintenance=maintenance,
        canonical_inputs=canonical_inputs,
        prepared_at=activated_at,
        activated_at=activated_at,
    )
    activated = live_store.activate_rollover_generation(
        target_factor_pointer_raw=prepared["target_factor_pointer_raw"],
        previous_factor_pointer_raw=prepared["previous_factor_pointer_raw"],
        factor_generation_receipt_raw=prepared["factor_generation_receipt_raw"],
        rollover_bundle_raw=prepared["rollover_bundle_raw"],
        rollover_prepared_raw=prepared["rollover_prepared_raw"],
        rollover_commit_raw=prepared["rollover_commit_raw"],
        canonical_paths=prepared["canonical_paths"],
    )
    rollover = activated["rollover"]
    return {
        **_factor_activation_projection(
            {**activated, "activation": rollover},
            command_status=(
                "ROLLOVER_ACTIVATED"
                if rollover["cas_performed"]
                else (
                    "ROLLOVER_COMMIT_RECOVERED"
                    if rollover["commit_recovered"]
                    else "ROLLOVER_IDEMPOTENT"
                )
            ),
            prepared_sources=prepared_sources,
        ),
        "previous_pointer_sha256": rollover["previous_pointer_sha256"],
        "rollover_commit_ref": rollover["rollover_commit_ref"],
        "prospective_evidence_status": "NOT_CONFIGURED",
        "prospective_write_performed": False,
        "upstream_maintenance_status": maintenance["upstream_maintenance_status"],
        "macro_status": maintenance["macro_status"],
        "macro_blockers": maintenance["macro_blockers"],
        "macro_used_by_factor": False,
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


def research_compile_daily(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    """Compile one exact, offline, inactive daily Intelligence closure."""

    from quant_investor.factors.production_authority import (
        assert_factor_production_pointer,
        read_factor_production_research_inputs,
    )
    from quant_investor.intelligence import (
        build_factor_research_rank,
        compile_daily_intelligence,
    )

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {
            "as_of",
            "expected_factor_pointer_sha256",
            "industry_source",
            "low_observation_path",
            "low_observation_sha256",
            "policy",
            "strategy_id",
            "theme_source",
            "w80_observation_path",
            "w80_observation_sha256",
        },
        code="RESEARCH_COMPILE_DAILY_REQUEST_INVALID",
    )
    _, low_observation = _request(
        workspace_root=workspace_root,
        request_path=values.pop("low_observation_path"),
        expected_request_sha256=values.pop("low_observation_sha256"),
    )
    _, w80_observation = _request(
        workspace_root=workspace_root,
        request_path=values.pop("w80_observation_path"),
        expected_request_sha256=values.pop("w80_observation_sha256"),
    )
    expected_pointer = values.pop("expected_factor_pointer_sha256")
    snapshot = read_factor_production_research_inputs(
        workspace_root,
        expected_pointer_sha256=expected_pointer,
    )
    rank = build_factor_research_rank(
        snapshot=snapshot,
        observations=[low_observation, w80_observation],
        policy=values["policy"],
        as_of=values["as_of"],
    )

    def source_document(reference: Any, *, code: str) -> dict[str, Any]:
        row = _exact_fields(reference, {"path", "sha256"}, code=code)
        _, loaded = _request(
            workspace_root=workspace_root,
            request_path=row["path"],
            expected_request_sha256=row["sha256"],
        )
        return loaded

    companies = [row["symbol"] for row in rank["payload"]["pool_rows"]]
    industry_source = values.pop("industry_source")
    if industry_source is None:
        industry_projection = None
    else:
        industry_values = _exact_fields(
            industry_source,
            {
                "membership_capture",
                "membership_partitions",
                "membership_plan",
                "taxonomy_capture",
                "taxonomy_plan",
            },
            code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
        )
        partitions = industry_values["membership_partitions"]
        if type(partitions) is not list or not partitions:
            raise CommandError("RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID")
        from quant_investor.intelligence import project_tushare_industry_source

        industry_projection = project_tushare_industry_source(
            taxonomy_plan=source_document(
                industry_values["taxonomy_plan"],
                code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
            ),
            taxonomy_capture=source_document(
                industry_values["taxonomy_capture"],
                code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
            ),
            membership_plan=source_document(
                industry_values["membership_plan"],
                code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
            ),
            membership_capture=source_document(
                industry_values["membership_capture"],
                code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
            ),
            partition_documents=[
                source_document(
                    reference,
                    code="RESEARCH_DAILY_INDUSTRY_SOURCE_INVALID",
                )
                for reference in partitions
            ],
            companies=sorted(companies, key=lambda item: item.encode("ascii")),
            as_of=values["as_of"],
        )

    theme_source = values.pop("theme_source")
    if theme_source is None:
        theme_projection = None
    else:
        theme_values = _exact_fields(
            theme_source,
            {
                "dc_capture",
                "dc_partitions",
                "dc_plan",
                "tdx_capture",
                "tdx_partitions",
                "tdx_plan",
            },
            code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
        )
        dc_partitions = theme_values["dc_partitions"]
        tdx_partitions = theme_values["tdx_partitions"]
        if type(dc_partitions) is not list or not dc_partitions or type(tdx_partitions) is not list:
            raise CommandError("RESEARCH_DAILY_THEME_SOURCE_INVALID")
        from quant_investor.intelligence import project_tushare_theme_source

        theme_projection = project_tushare_theme_source(
            dc_plan=source_document(
                theme_values["dc_plan"],
                code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
            ),
            dc_capture=source_document(
                theme_values["dc_capture"],
                code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
            ),
            dc_partitions=[
                source_document(
                    reference,
                    code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
                )
                for reference in dc_partitions
            ],
            tdx_plan=(
                None
                if theme_values["tdx_plan"] is None
                else source_document(
                    theme_values["tdx_plan"],
                    code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
                )
            ),
            tdx_capture=(
                None
                if theme_values["tdx_capture"] is None
                else source_document(
                    theme_values["tdx_capture"],
                    code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
                )
            ),
            tdx_partitions=[
                source_document(
                    reference,
                    code="RESEARCH_DAILY_THEME_SOURCE_INVALID",
                )
                for reference in tdx_partitions
            ],
            policy=values["policy"],
            as_of=values["as_of"],
        )
        if (
            theme_projection["payload"]["company_set_sha256"]
            != hashlib.sha256(
                canonical_json_bytes(sorted(companies, key=lambda item: item.encode("ascii")))
            ).hexdigest()
        ):
            raise CommandError("RESEARCH_DAILY_THEME_COMPANY_SET_MISMATCH")
    result = compile_daily_intelligence(
        rank=rank,
        industry_projection=industry_projection,
        theme_projection=theme_projection,
        **values,
    )
    assert_factor_production_pointer(
        workspace_root,
        expected_pointer_sha256=expected_pointer,
    )
    return result


def research_publish_policy(*, workspace_root: str) -> dict[str, Any]:
    """Publish the exact owner-approved immutable Phase A policy."""

    from quant_investor.intelligence import publish_phase_a_policy

    return publish_phase_a_policy(workspace_root)


def research_morning_strategy(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    """Validate or seal one exact 09:45 morning-strategy closure."""

    from quant_investor.intelligence import run_morning_strategy

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    return run_morning_strategy(workspace_root=workspace_root, request=document)


def research_morning_cutover(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    """Seal one exact 20:20 morning cutover/rollback decision."""

    from quant_investor.intelligence import evaluate_morning_cutover

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    return evaluate_morning_cutover(workspace_root=workspace_root, request=document)


def research_publish_theme_policy(*, workspace_root: str) -> dict[str, Any]:
    """Publish the exact owner-approved ACTIVE Theme v2 policy bundle."""

    from quant_investor.intelligence import publish_theme_policy_v2

    return publish_theme_policy_v2(workspace_root)


def _factor_signal_close_cutoff(signal_date: Any) -> str:
    if type(signal_date) is not str:
        raise CommandError("RESEARCH_POOL_SIGNAL_DATE_INVALID")
    try:
        parsed = datetime.strptime(signal_date, "%Y%m%d")
    except ValueError as exc:
        raise CommandError("RESEARCH_POOL_SIGNAL_DATE_INVALID") from exc
    return parsed.replace(hour=7, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def research_publish_pool(
    *, workspace_root: str, request_path: str, expected_request_sha256: str
) -> dict[str, Any]:
    """Derive and atomically publish one immutable Top100 Factor pool."""

    from quant_investor.factors.production_authority import (
        assert_factor_production_pointer,
        read_factor_production_research_inputs,
    )
    from quant_investor.intelligence import (
        IntelligenceError,
        build_factor_research_rank,
    )
    from quant_investor.intelligence.storage import (
        DailyResearchPoolStore,
        approved_pool_policy,
    )

    _, document = _request(
        workspace_root=workspace_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
    )
    values = _exact_fields(
        document,
        {
            "expected_factor_pointer_sha256",
            "expected_policy_sha256",
            "low_observation_path",
            "low_observation_sha256",
            "policy_path",
            "w80_observation_path",
            "w80_observation_sha256",
        },
        code="RESEARCH_POOL_PUBLISH_REQUEST_INVALID",
    )
    try:
        expected_policy = approved_pool_policy(values["policy_path"])
    except IntelligenceError as exc:
        raise CommandError("RESEARCH_POOL_POLICY_PATH_INVALID") from exc
    _, policy = _request(
        workspace_root=workspace_root,
        request_path=values["policy_path"],
        expected_request_sha256=values["expected_policy_sha256"],
    )
    if policy != expected_policy:
        raise CommandError("RESEARCH_POOL_POLICY_BYTES_INVALID")
    _, low = _request(
        workspace_root=workspace_root,
        request_path=values["low_observation_path"],
        expected_request_sha256=values["low_observation_sha256"],
    )
    _, w80 = _request(
        workspace_root=workspace_root,
        request_path=values["w80_observation_path"],
        expected_request_sha256=values["w80_observation_sha256"],
    )
    expected_pointer = values["expected_factor_pointer_sha256"]
    snapshot = read_factor_production_research_inputs(
        workspace_root,
        expected_pointer_sha256=expected_pointer,
    )
    source_sealed_at = max(
        str(snapshot["factor_generation"].get("created_at") or ""),
        str(low.get("created_at") or ""),
        str(w80.get("created_at") or ""),
    )
    rank = build_factor_research_rank(
        snapshot=snapshot,
        observations=[low, w80],
        policy=policy,
        as_of=_factor_signal_close_cutoff(snapshot["signal_date"]),
        created_at=source_sealed_at,
    )
    store = DailyResearchPoolStore(workspace_root)
    return store.publish(
        rank=rank,
        expected_policy_sha256=values["expected_policy_sha256"],
        policy_path=values["policy_path"],
        before_publish=lambda: assert_factor_production_pointer(
            workspace_root,
            expected_pointer_sha256=expected_pointer,
        ),
    )


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
    "factor_production_activate",
    "factor_production_rollover",
    "factor_production_observe",
    "factor_production_signal",
    "factor_production_status",
    "factor_production_verify",
    "factor_status",
    "research_compile_evidence",
    "research_compile_daily",
    "research_publish_policy",
    "research_publish_theme_policy",
    "research_publish_pool",
    "research_evaluate",
    "research_forward",
    "research_inspect",
    "research_morning_cutover",
    "research_morning_strategy",
    "research_readiness",
    "system_activate",
    "system_assemble",
    "system_bootstrap_admission_preflight",
    "system_bootstrap_assemble",
    "system_calendar_capture",
    "system_status",
    "system_suspend",
    "system_verify",
]
