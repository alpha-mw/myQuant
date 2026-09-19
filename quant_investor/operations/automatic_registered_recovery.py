"""Exact original registered batch recovery selection and expired EOD disclosure."""

from .automatic_catchup_contract import AutomaticCatchupError, completion_day
from .automatic_catchup_closure import completion_state
from .automatic_catchup_storage import AutomaticRunStorage
from .automatic_origin import origin_path, read_automatic_origin, recheck_origin
from .catchup_binding import binding_path, read_catchup_binding
from .daily_contract import ContractError
from .journal_storage import JournalStorage


def registered_resolution(resolution):
    return any(
        recipe["schema_version"] == "cn-daily-execute-recipe.v6"
        for recipe in resolution["derived_collection"]["recipes"].values()
    )


def require_matching_active(storage, *, resolution, resolution_ref):
    pending = storage.pending()
    if (
        pending is None
        or pending["state"] != "ACTIVE"
        or pending["auto_request_ref"] != resolution["auto_request_ref"]
        or pending["resolution_ref"] != resolution_ref
    ):
        raise AutomaticCatchupError(
            "AUTO_PENDING_RECOVERY_UNCONFIRMED", pending_request_ref=resolution["auto_request_ref"]
        )
    return pending


def inspect_registered_batch(context, *, resolution_ref, synthetic, required=False):
    """Read fixed binding/origin paths; never create a replacement source package."""
    from scripts.daily_registered_recovery import (
        inspect_registered_recovery,
        registered_commit_required,
    )

    resolution, workspace = context["resolution"], context["sources"].workspace
    if required and not registered_resolution(resolution):
        raise AutomaticCatchupError("AUTO_COMMITTED_RECOVERY_REQUIRED")
    _, unresolved, _ = completion_state(context, synthetic=synthetic)
    if not unresolved:
        return {}
    if not registered_resolution(resolution):
        if required:
            raise AutomaticCatchupError("AUTO_COMMITTED_RECOVERY_REQUIRED")
        return None
    storage = AutomaticRunStorage(workspace)
    pending = require_matching_active(storage, resolution=resolution, resolution_ref=resolution_ref)
    selected = {}
    journal = JournalStorage(workspace)
    for day in unresolved:
        template = resolution["derived_collection"]["recipes"].get(day)
        if (
            template is None
            or template["schema_version"] != "cn-daily-execute-recipe.v6"
            or resolution["day_scopes"][day]["maintenance_mode"] != "CURRENT"
        ):
            if required:
                raise AutomaticCatchupError(
                    "AUTO_COMMITTED_RECOVERY_REQUIRED", missing_input_dates=unresolved
                )
            return None
        stored = journal.read(binding_path(day, resolution["derived_request_ref"], "v3"))
        if stored is None:
            if required:
                raise AutomaticCatchupError(
                    "AUTO_COMMITTED_RECOVERY_REQUIRED", missing_input_dates=[day]
                )
            return None
        reference = {"path": stored.relative_path, "sha256": stored.byte_sha256}
        bound = read_catchup_binding(workspace=workspace, binding_ref=reference)
        if not registered_commit_required(workspace=workspace, recipe=bound["recipe"]):
            if required:
                raise AutomaticCatchupError(
                    "AUTO_COMMITTED_RECOVERY_REQUIRED", missing_input_dates=[day]
                )
            return None
        original = journal.read(origin_path(day, bound["binding"]["execution_request_ref"]))
        if original is None:
            raise AutomaticCatchupError(
                "AUTO_PENDING_RECOVERY_UNCONFIRMED",
                pending_request_ref=resolution["auto_request_ref"],
            )
        origin_ref = {"path": original.relative_path, "sha256": original.byte_sha256}
        origin = read_automatic_origin(
            workspace=workspace, reference=origin_ref, synthetic=synthetic
        )
        if (
            origin["origin"]["resolution_ref"] != resolution_ref
            or origin["resolution"]["resolution"] != resolution
            or origin["origin"]["catchup_binding_ref"] != reference
        ):
            raise ContractError("AUTO_RECOVERY_ORIGIN_RESOLUTION_MISMATCH")
        scope = inspect_registered_recovery(
            workspace=workspace,
            request_ref=bound["binding"]["execution_request_ref"],
            release_install_ref=resolution["release_install_ref"],
            synthetic=synthetic,
            automatic_origin_ref=origin_ref,
        )
        bound["sources"].recheck()
        recheck_origin(origin)
        selected[day] = {
            "binding_ref": reference,
            "bound": bound,
            "origin_ref": origin_ref,
            "scope": scope,
        }
    context["sources"].recheck()
    if (
        require_matching_active(storage, resolution=resolution, resolution_ref=resolution_ref)
        != pending
    ):
        raise ContractError("AUTO_RECOVERY_PENDING_CHANGED")
    return selected


def expired_completion_error(context, *, synthetic):
    """Independently prove the whole completed range and exact publication expiry."""
    from scripts.daily_catchup import run_upstream_catchup
    from .completion_readback import inspect_recorded_completion

    completed, unresolved, target = completion_state(context, synthetic=synthetic)
    if unresolved:
        raise AutomaticCatchupError(
            "AUTO_PENDING_RECOVERY_UNCONFIRMED", missing_input_dates=unresolved
        )
    day = context["resolution"]["target_trade_date"]
    if context["resolution"]["day_scopes"][day]["dashboard_mode"] != "CURRENT_LATEST_EOD":
        raise ContractError("AUTO_PUBLICATION_EXPIRY_SCOPE_INVALID")
    resolution = context["resolution"]
    workspace = context["sources"].workspace
    journal = JournalStorage(workspace)
    stored = journal.read(binding_path(day, resolution["derived_request_ref"], "v3"))
    if stored is None:
        raise ContractError("AUTO_PUBLICATION_ORIGINAL_BINDING_MISSING")
    bound = read_catchup_binding(
        workspace=workspace,
        binding_ref={"path": stored.relative_path, "sha256": stored.byte_sha256},
    )
    original = journal.read(origin_path(day, bound["binding"]["execution_request_ref"]))
    if original is None:
        raise ContractError("AUTO_PUBLICATION_ORIGINAL_ORIGIN_MISSING")
    origin_ref = {"path": original.relative_path, "sha256": original.byte_sha256}
    origin = read_automatic_origin(workspace=workspace, reference=origin_ref, synthetic=synthetic)
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=day, completion_ref=target
    )
    snapshot = recorded.get("completed_handoff_snapshot")
    if (
        snapshot is None
        or snapshot.document("handoff").get("automatic_origin_ref") != origin_ref
        or origin["resolution"]["resolution"] != resolution
    ):
        raise ContractError("AUTO_PUBLICATION_ORIGINAL_COMPLETION_MISMATCH")
    try:
        run_upstream_catchup(
            workspace=workspace,
            request_ref=resolution["derived_request_ref"],
            synthetic=synthetic,
            no_producers=True,
            read_only=True,
        )
    except ContractError as exc:
        if str(exc) != "AUTO_PUBLICATION_EXPIRED":
            raise
    else:
        raise ContractError("AUTO_PUBLICATION_EXPIRY_UNCONFIRMED")
    snapshot.recheck()
    bound["sources"].recheck()
    recheck_origin(origin)
    context["sources"].recheck()
    return AutomaticCatchupError(
        "AUTO_PUBLICATION_EXPIRED",
        completed_eod_refs=[
            {"trade_date": completion_day(ref), "completion_ref": ref} for ref in completed
        ],
    )
