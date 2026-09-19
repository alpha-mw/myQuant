"""Automatic daily-close resolves once, then delegates to the existing catch-up controller."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.errors import SystemStorageError
from quant_investor.operations.automatic_catchup_contract import (
    AutomaticCatchupError,
    RESULT_SCHEMA,
    document_ref,
    run_path,
    validate_automatic_request,
    validate_automatic_result,
)
from quant_investor.operations.automatic_catchup_resolution import (
    resolve_automatic_request,
    read_automatic_resolution,
)
from quant_investor.operations.automatic_catchup_storage import (
    AutomaticRunStorage,
    PENDING_SCHEMA,
    automatic_execution,
)
from quant_investor.operations.automatic_catchup_closure import (
    completion_state,
    expire_unstarted,
    prime_day_locks,
)
from quant_investor.operations.catchup_binding import BindingSources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.automatic_registered_recovery import (
    registered_resolution,
    inspect_registered_batch,
    expired_completion_error,
)


def _wrapper(context, *, request_ref, resolution_ref=None, result=None):
    resolution, request = context["resolution"], context["request"]
    plan = request["action"] == "PLAN"
    missing = context["missing_input_dates"] if plan else []
    value = {
        "schema_version": RESULT_SCHEMA,
        "action": request["action"],
        "auto_request_ref": request_ref,
        **{
            key: resolution[key]
            for key in (
                "target_trade_date",
                "anchor_ref",
                "adopted_completion_refs",
                "ordered_trade_dates",
                "day_scopes",
            )
        },
        "missing_input_dates": missing,
        "resolution_ref": resolution_ref,
        "status": ("BLOCKED" if missing else "PLANNED") if plan else result["execution_state"],
        "result": result,
        "authority": dict(FALSE_AUTHORITY),
    }
    return validate_automatic_result(
        value, request_ref=request_ref, request=request, resolution=resolution
    )


def _current_expired(resolution):
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    return any(
        day < today and scope["maintenance_mode"] == "CURRENT"
        for day, scope in resolution["day_scopes"].items()
    )


def _finish_pending(storage, pending, *, release_install_ref, synthetic):
    """Retire only native-complete work or a proven expired-unstarted lease."""
    try:
        old = read_automatic_resolution(
            workspace=storage.workspace,
            resolution_ref=pending["resolution_ref"],
            release_install_ref=release_install_ref,
            synthetic=synthetic,
        )
        _, unresolved, _ = completion_state(old, synthetic=synthetic)
        if not unresolved:
            if registered_resolution(old["resolution"]):
                from scripts.daily_catchup import run_upstream_catchup

                try:
                    run_upstream_catchup(
                        workspace=storage.workspace,
                        request_ref=old["resolution"]["derived_request_ref"],
                        synthetic=synthetic,
                        no_producers=True,
                        read_only=True,
                    )
                except ContractError as exc:
                    if str(exc) != "AUTO_PUBLICATION_EXPIRED":
                        raise
                    _retire_expired_registered(storage, old, pending["resolution_ref"], synthetic)
            storage.set_pending({**pending, "state": "IDLE"})
            return
        if _current_expired(old["resolution"]):
            expire_unstarted(
                storage, old, resolution_ref=pending["resolution_ref"], synthetic=synthetic
            )
            return
    except (ContractError, SystemStorageError, OSError, ValueError) as exc:
        if isinstance(exc, AutomaticCatchupError) and exc.code == "AUTO_PUBLICATION_EXPIRED":
            raise
        raise AutomaticCatchupError(
            "AUTO_PENDING_RECOVERY_UNCONFIRMED", pending_request_ref=pending["auto_request_ref"]
        ) from exc
    raise AutomaticCatchupError(
        "AUTO_PENDING_REQUEST_CONFLICT", pending_request_ref=pending["auto_request_ref"]
    )


def _persist(storage, context, request_ref):
    resolution = context["resolution"]
    if context["missing_input_dates"]:
        raise AutomaticCatchupError(
            "AUTO_INPUTS_MISSING", missing_input_dates=context["missing_input_dates"]
        )
    context["sources"].recheck()
    path = run_path(request_ref, "resolution.v1.json")
    rows = [(path, canonical_json_bytes(resolution))]
    for ref_key, body_key in (
        ("derived_collection_ref", "derived_collection"),
        ("derived_request_ref", "derived_request"),
    ):
        rows.append((resolution[ref_key]["path"], canonical_json_bytes(resolution[body_key])))
    for destination, raw in rows:
        stored = storage.read(destination)
        if stored is not None and stored.data != raw:
            raise ContractError("AUTO_DERIVED_FILE_CONFLICT")
    for destination, raw in rows:
        storage.write(destination, raw)
    context["sources"].recheck()
    return document_ref(path, resolution)


def run_automatic_catchup(
    *,
    workspace,
    request_ref,
    release_install_ref,
    synthetic=False,
    no_producers=False,
    committed_recovery_only=False,
):
    source = BindingSources(workspace)
    request = validate_automatic_request(
        source.document(request_ref), release_install_ref=release_install_ref
    )
    if type(no_producers) is not bool or (no_producers and request["action"] != "CATCH_UP"):
        raise ContractError("AUTO_NO_PRODUCERS_FLAG_INVALID")
    if type(committed_recovery_only) is not bool or (
        committed_recovery_only and (no_producers or request["action"] != "CATCH_UP")
    ):
        raise ContractError("AUTO_COMMITTED_RECOVERY_MODE_INVALID")
    if request["action"] == "PLAN":
        context = resolve_automatic_request(
            workspace=workspace,
            request_ref=request_ref,
            release_install_ref=release_install_ref,
            synthetic=synthetic,
        )
        return _wrapper(context, request_ref=request_ref)
    storage = AutomaticRunStorage(workspace)
    with storage.locked():
        pending = storage.pending()
        recovery = committed_recovery_only
        if recovery and (
            pending is None
            or pending["state"] != "ACTIVE"
            or pending["auto_request_ref"] != request_ref
        ):
            raise AutomaticCatchupError("AUTO_COMMITTED_RECOVERY_REQUIRED")
        if pending is not None and pending["state"] == "ACTIVE":
            if pending["auto_request_ref"] != request_ref:
                _finish_pending(
                    storage, pending, release_install_ref=release_install_ref, synthetic=synthetic
                )
            else:
                previous = read_automatic_resolution(
                    workspace=workspace,
                    resolution_ref=pending["resolution_ref"],
                    release_install_ref=release_install_ref,
                    synthetic=synthetic,
                )
                _, unresolved, _ = completion_state(previous, synthetic=synthetic)
                if unresolved and registered_resolution(previous["resolution"]):
                    selected = _registered_recovery_selection(
                        previous, pending["resolution_ref"], synthetic, required=recovery
                    )
                    recovery = recovery or bool(selected)
                elif recovery and unresolved:
                    raise AutomaticCatchupError("AUTO_COMMITTED_RECOVERY_REQUIRED")
                if unresolved and not recovery and _current_expired(previous["resolution"]):
                    _finish_pending(
                        storage,
                        pending,
                        release_install_ref=release_install_ref,
                        synthetic=synthetic,
                    )
        path = run_path(request_ref, "resolution.v1.json")
        if storage.read(run_path(request_ref, "closure.v1.json")) is not None:
            raise AutomaticCatchupError(
                "AUTO_RESOLUTION_EXPIRED_UNSTARTED", pending_request_ref=request_ref
            )
        saved = storage.read(path)
        context = (
            resolve_automatic_request(
                workspace=workspace,
                request_ref=request_ref,
                release_install_ref=release_install_ref,
                synthetic=synthetic,
            )
            if saved is None
            else read_automatic_resolution(
                workspace=workspace,
                resolution_ref={"path": path, "sha256": saved.byte_sha256},
                release_install_ref=release_install_ref,
                synthetic=synthetic,
            )
        )
        if recovery and not registered_resolution(context["resolution"]):
            raise AutomaticCatchupError("AUTO_COMMITTED_RECOVERY_REQUIRED")
        if no_producers:
            _, unresolved, _ = completion_state(context, synthetic=synthetic)
            if unresolved:
                raise AutomaticCatchupError(
                    "AUTO_PRODUCERS_REQUIRED", missing_input_dates=unresolved
                )
        resolution_ref = _persist(storage, context, request_ref)
        resolution = context["resolution"]
        pending = storage.pending()
        same_idle = (
            pending is not None
            and pending["state"] == "IDLE"
            and pending["resolution_ref"] == resolution_ref
        )
        if not same_idle:
            prime_day_locks(storage, resolution)
            storage.set_pending(
                {
                    "schema_version": PENDING_SCHEMA,
                    "state": "ACTIVE",
                    "auto_request_ref": request_ref,
                    "resolution_ref": resolution_ref,
                }
            )
        try:
            with automatic_execution(
                storage,
                resolution_ref=resolution_ref,
                resolution=resolution,
                completed_context=context if same_idle else None,
                synthetic=synthetic,
            ):
                from scripts.daily_production import dispatch_daily_request

                result = dispatch_daily_request(
                    workspace=workspace,
                    request_ref=resolution["derived_request_ref"],
                    release_install_ref=release_install_ref,
                    synthetic=synthetic,
                    no_producers=no_producers,
                    **({"committed_recovery_only": True} if recovery else {}),
                )
        except ContractError as exc:
            if str(exc) != "AUTO_PUBLICATION_EXPIRED" or not registered_resolution(resolution):
                raise
            _retire_expired_registered(storage, context, resolution_ref, synthetic)
        _, unresolved, _ = completion_state(context, synthetic=synthetic)
        if not unresolved:
            pending = storage.pending()
            storage.set_pending({**pending, "state": "IDLE"})
        return _wrapper(
            context, request_ref=request_ref, resolution_ref=resolution_ref, result=result
        )


def _registered_recovery_selection(context, resolution_ref, synthetic, *, required=False):
    try:
        return inspect_registered_batch(
            context, resolution_ref=resolution_ref, synthetic=synthetic, required=required
        )
    except Exception as exc:
        raise AutomaticCatchupError(
            "AUTO_PENDING_RECOVERY_UNCONFIRMED",
            pending_request_ref=context["resolution"]["auto_request_ref"],
        ) from exc


def _retire_expired_registered(storage, context, resolution_ref, synthetic):
    storage.require_lock()
    pending = storage.pending()
    if (
        pending is None
        or pending["auto_request_ref"] != context["resolution"]["auto_request_ref"]
        or pending["resolution_ref"] != resolution_ref
    ):
        raise ContractError("AUTO_PUBLICATION_PENDING_MISMATCH")
    error = expired_completion_error(context, synthetic=synthetic)
    storage.require_lock()
    if storage.pending() != pending:
        raise ContractError("AUTO_PUBLICATION_PENDING_CHANGED")
    if pending["state"] == "ACTIVE":
        idle = {**pending, "state": "IDLE"}
        storage.set_pending(idle)
        if storage.pending() != idle:
            raise ContractError("AUTO_PUBLICATION_IDLE_READBACK_FAILED")
    raise error


def inspect_automatic_resolution(
    *, workspace, resolution_ref, release_install_ref, synthetic=False, request_ref=None
):
    """Fixed installed read-only boundary; no lock/metadata creation or mutable-head read."""
    if resolution_ref is None:
        if request_ref is None:
            raise ContractError("AUTO_PLAN_REQUEST_REQUIRED")
        context = resolve_automatic_request(
            workspace=workspace,
            request_ref=request_ref,
            release_install_ref=release_install_ref,
            synthetic=synthetic,
        )
        if context["request"]["action"] != "PLAN":
            raise ContractError("AUTO_PLAN_REQUEST_REQUIRED")
        return context["resolution"]
    if request_ref is not None:
        raise ContractError("AUTO_RESOLUTION_READER_FIELDS_INVALID")
    context = read_automatic_resolution(
        workspace=workspace,
        resolution_ref=resolution_ref,
        release_install_ref=release_install_ref,
        synthetic=synthetic,
    )
    return context["resolution"]
