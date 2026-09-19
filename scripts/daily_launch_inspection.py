"""Read-only installed launcher inspection, sharing native catch-up and serving gates."""

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.automatic_catchup_contract import (
    AutomaticCatchupError,
    run_path,
    validate_automatic_request,
)
from quant_investor.operations.automatic_catchup_resolution import (
    resolve_automatic_request,
    read_automatic_resolution,
)
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.automatic_catchup_closure import completion_state
from quant_investor.operations.catchup_binding import BindingSources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.daily_launch_contract import (
    SCHEMA,
    REGISTERED_SCHEMA,
    validate_launch_inspection,
)
from quant_investor.operations.execution_controls import verify_recipe_static_controls
from scripts.daily_automatic_catchup import (
    _current_expired,
    _wrapper,
    _registered_recovery_selection,
)
from quant_investor.operations.automatic_registered_recovery import (
    registered_resolution,
    expired_completion_error,
)
from scripts.daily_catchup import _serving_gate, run_upstream_catchup


def _read_context(workspace, request_ref, release_install_ref, synthetic, storage):
    path = run_path(request_ref, "resolution.v1.json")
    saved = storage.read(path)
    if saved is None:
        return (
            resolve_automatic_request(
                workspace=workspace,
                request_ref=request_ref,
                release_install_ref=release_install_ref,
                synthetic=synthetic,
            ),
            None,
        )
    ref = {"path": path, "sha256": saved.byte_sha256}
    return (
        read_automatic_resolution(
            workspace=workspace,
            resolution_ref=ref,
            release_install_ref=release_install_ref,
            synthetic=synthetic,
        ),
        ref,
    )


def _producer_controls(context, unresolved):
    source, resolution = context["sources"], context["resolution"]
    for day in unresolved:
        template = resolution["derived_collection"]["recipes"].get(day)
        if template is not None:
            from quant_investor.operations.catchup_binding import check_template_sources

            check_template_sources(source, template, profile="FRESH")
            verify_recipe_static_controls(
                workspace=source.workspace, recipe=template, document=source.document
            )
            acquisition = template["theme_acquisition_ref"]
            if acquisition is not None:
                from quant_investor.operations.theme_acquisition import (
                    validate_theme_acquisition_policy,
                )

                validate_theme_acquisition_policy(source.document(acquisition))
        else:
            from scripts.daily_native_inputs import load_native_inputs, verify_loaded_native_inputs

            native_ref = resolution["derived_request"]["day_input_refs"][day]
            loaded_day, inputs = load_native_inputs(
                workspace=source.workspace, input_ref=native_ref
            )
            verify_loaded_native_inputs(
                workspace=source.workspace,
                input_ref=native_ref,
                trade_date=loaded_day,
                inputs=inputs,
            )
            installed = source.document(resolution["release_install_ref"])
            if (
                loaded_day != day
                or source.document(inputs.release_ref) != installed["deployed_release"]
            ):
                raise ContractError("AUTO_LAUNCH_NATIVE_RELEASE_MISMATCH")
    source.recheck()


def _serving_complete(context, target_ref):
    resolution = context["resolution"]
    day = resolution["target_trade_date"]
    if resolution["day_scopes"][day]["dashboard_mode"] != "CURRENT_LATEST_EOD":
        return True
    return _serving_gate(
        context["sources"].workspace,
        {
            "trade_date": day,
            "execution_state": "NO_ACTION",
            "business_state": "COMPLETE",
            "completion_ref": target_ref,
        },
        read_only=True,
    )


def inspect_daily_launch(*, workspace, request_ref, release_install_ref, synthetic=False):
    source = BindingSources(workspace)
    document = source.document(request_ref)
    if document.get("schema_version") == "cn-daily-production-request.v1":
        from scripts.daily_bootstrap_launch import inspect_bootstrap_launch

        return inspect_bootstrap_launch(
            workspace=workspace,
            request_ref=request_ref,
            release_install_ref=release_install_ref,
            synthetic=synthetic,
        )
    request = validate_automatic_request(document, release_install_ref=release_install_ref)
    if request["action"] != "CATCH_UP":
        raise ContractError("AUTO_LAUNCH_CATCHUP_REQUIRED")
    storage = AutomaticRunStorage(workspace)
    pending = storage.pending()
    if (
        pending is not None
        and pending["state"] == "ACTIVE"
        and pending["auto_request_ref"] != request_ref
    ):
        raise AutomaticCatchupError(
            "AUTO_PENDING_REQUEST_CONFLICT", pending_request_ref=pending["auto_request_ref"]
        )
    if storage.read(run_path(request_ref, "closure.v1.json")) is not None:
        raise AutomaticCatchupError(
            "AUTO_RESOLUTION_EXPIRED_UNSTARTED", pending_request_ref=request_ref
        )
    context, resolution_ref = _read_context(
        workspace, request_ref, release_install_ref, synthetic, storage
    )
    if context["missing_input_dates"]:
        raise AutomaticCatchupError(
            "AUTO_INPUTS_MISSING", missing_input_dates=context["missing_input_dates"]
        )
    resolution = context["resolution"]
    registered = registered_resolution(resolution)
    recovery_scope = "NONE"
    _, unresolved, target_ref = completion_state(context, synthetic=synthetic)
    result = None
    if unresolved:
        if (
            pending is not None
            and pending["auto_request_ref"] == request_ref
            and pending["state"] == "IDLE"
        ):
            raise ContractError("AUTO_IDLE_RUN_NOT_COMPLETED")
        selected = None
        if registered and pending is not None and pending["state"] == "ACTIVE":
            selected = _registered_recovery_selection(context, resolution_ref, synthetic)
        if selected:
            mode = "LOCAL_REPAIR"
            recovery_scope = "COMMITTED_DAG_RECOVERY"
        elif pending is not None and pending["state"] == "ACTIVE" and _current_expired(resolution):
            mode = "LOCAL_REPAIR"
        else:
            _producer_controls(context, unresolved)
            mode = "PRODUCER_REQUIRED"
    else:
        try:
            serving = _serving_complete(context, target_ref)
        except ContractError as exc:
            if str(exc) != "AUTO_PUBLICATION_EXPIRED" or not registered:
                raise
            error = expired_completion_error(context, synthetic=synthetic)
            if (
                pending is None
                or pending["state"] != "ACTIVE"
                or pending["resolution_ref"] != resolution_ref
            ):
                raise error
            serving = False
        complete_files = resolution_ref is not None
        for key in ("derived_request_ref", "derived_collection_ref"):
            file = storage.read(resolution[key]["path"])
            complete_files = complete_files and file is not None
        matching_idle = (
            pending is not None
            and pending["state"] == "IDLE"
            and pending["resolution_ref"] == resolution_ref
        )
        if serving and complete_files and matching_idle:
            replay = run_upstream_catchup(
                workspace=workspace,
                request_ref=resolution["derived_request_ref"],
                synthetic=synthetic,
                no_producers=True,
                read_only=True,
            )
            result = _wrapper(
                context, request_ref=request_ref, resolution_ref=resolution_ref, result=replay
            )
            mode = "COMPLETE_READ_ONLY"
        else:
            mode = "LOCAL_REPAIR"
            if registered:
                recovery_scope = "SERVING_ONLY"
    source.recheck()
    context["sources"].recheck()
    if storage.pending() != pending:
        raise ContractError("AUTO_LAUNCH_PENDING_CHANGED")
    value = {
        "schema_version": REGISTERED_SCHEMA if registered else SCHEMA,
        "request_ref": request_ref,
        "release_install_ref": release_install_ref,
        "mode": mode,
        "result": result,
        "authority": dict(FALSE_AUTHORITY),
        **({"recovery_scope": recovery_scope} if registered else {}),
    }
    # Ensure the inspection wire is canonical/serializable before leaving installed code.
    canonical_json_bytes(value)
    return validate_launch_inspection(
        value, request_ref=request_ref, release_install_ref=release_install_ref
    )
