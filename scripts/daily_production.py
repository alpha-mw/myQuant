"""Fixed action dispatch over existing native daily producers and readers."""

from pathlib import Path
import os
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS, validate_ref
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.production_request import validate_production_request
from quant_investor.operations.production_result import SCHEMA, validate_production_result
from quant_investor.operations.catchup import plan_catchup, catchup_result_dates
from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
from scripts.daily_materialization import (
    plan_daily_recipe,
    execute_daily_recipe,
    materialize_daily_inputs,
)
from scripts.daily_completion import run_materialized_native_input
from scripts.daily_completion_replay import replay_native_completion
from scripts.daily_catchup import run_native_catchup, run_upstream_catchup


def _checked_result(value, *, action, target_trade_date, expected_dates):
    try:
        return validate_production_result(
            value, action=action, target_trade_date=target_trade_date, expected_dates=expected_dates
        )
    except ContractError as exc:
        raise RuntimeError("native daily production result is invalid") from exc


def dispatch_daily_request(
    *,
    workspace: str,
    request_ref: dict,
    release_install_ref: dict,
    synthetic: bool = False,
    no_producers: bool = False,
    committed_recovery_only: bool = False,
) -> dict:
    if any(type(value) is not bool for value in (synthetic, no_producers, committed_recovery_only)):
        raise ContractError("DAILY_PRODUCTION_PROVENANCE_INVALID")
    if committed_recovery_only and no_producers:
        raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_PROFILE_INVALID")
    reference = validate_ref(request_ref)
    storage = SecureSystemStorage(workspace)
    stored = storage.read_workspace_file_bytes(reference["path"], maximum_bytes=8 * 1024 * 1024)
    if stored.byte_sha256 != reference["sha256"]:
        raise ContractError("DAILY_PRODUCTION_REQUEST_SHA_MISMATCH")
    document = parse_canonical_json_bytes(stored.data)
    from quant_investor.operations.automatic_catchup_contract import (
        REQUEST_SCHEMA as AUTO_SCHEMA,
        REQUEST_SCHEMA_V2 as AUTO_SCHEMA_V2,
    )

    if (
        type(document) is dict
        and type(document.get("schema_version")) is str
        and document["schema_version"] in {AUTO_SCHEMA, AUTO_SCHEMA_V2}
    ):
        from scripts.daily_automatic_catchup import run_automatic_catchup

        if (
            not synthetic
            and document.get("action") != "PLAN"
            and os.path.lexists(Path(workspace) / "SYNTHETIC-DATA-CLOCK.json")
        ):
            raise ContractError("SYNTHETIC_WORKSPACE_REQUIRES_INTERNAL_RUN")
        return run_automatic_catchup(
            workspace=workspace,
            request_ref=reference,
            release_install_ref=release_install_ref,
            synthetic=synthetic,
            no_producers=no_producers,
            **({"committed_recovery_only": True} if committed_recovery_only else {}),
        )
    request = validate_production_request(document, release_install_ref=release_install_ref)
    if canonical_json_bytes(request) != stored.data:
        raise ContractError("DAILY_PRODUCTION_REQUEST_CANONICAL_REQUIRED")
    action, day = request["action"], request["target_trade_date"]
    if committed_recovery_only and action not in {"EXECUTE", "CATCH_UP"}:
        raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_PROFILE_INVALID")
    if no_producers and (action != "CATCH_UP" or request["recipe_ref"] is None):
        raise ContractError("AUTO_NO_PRODUCERS_FLAG_INVALID")
    if committed_recovery_only and action == "CATCH_UP" and request["recipe_ref"] is None:
        raise ContractError("AUTO_COMMITTED_RECOVERY_CAPABILITY_REQUIRED")
    if action != "PLAN":
        from quant_investor.operations.automatic_catchup_storage import require_dispatch_ownership

        require_dispatch_ownership(workspace=workspace, request_ref=reference, request=request)
    if action == "PLAN":
        return plan_daily_recipe(workspace=workspace, request_ref=reference)
    if not synthetic and os.path.lexists(Path(workspace) / "SYNTHETIC-DATA-CLOCK.json"):
        raise ContractError("SYNTHETIC_WORKSPACE_REQUIRES_INTERNAL_RUN")
    if action == "CATCH_UP":
        previous = Path(request["previous_completion_ref"]["path"]).parent.name
        planned = plan_catchup(
            workspace=workspace,
            calendar_ref=request["calendar_ref"],
            raw_calendar_ref=request["raw_calendar_ref"],
            previous_trade_date=previous,
            target_trade_date=day,
            day_input_refs=request["day_input_refs"],
        )
        dates = catchup_result_dates(planned, previous_trade_date=previous, target_trade_date=day)
        if request["recipe_ref"] is not None:
            result = run_upstream_catchup(
                workspace=workspace,
                request_ref=reference,
                synthetic=synthetic,
                no_producers=no_producers,
                **({"committed_recovery_only": True} if committed_recovery_only else {}),
            )
        else:
            result = run_native_catchup(
                workspace=workspace,
                calendar_ref=request["calendar_ref"],
                raw_calendar_ref=request["raw_calendar_ref"],
                previous_completion_ref=request["previous_completion_ref"],
                target_trade_date=day,
                day_input_refs=request["day_input_refs"],
                synthetic=synthetic,
            )
        return _checked_result(result, action=action, target_trade_date=day, expected_dates=dates)
    journal = DailyJournal(workspace, day)
    already_completed = journal.storage.read(str(journal.root / "completion.v1.json")) is not None
    if action == "EXECUTE":
        result = _execute_with_bootstrap_lock(
            workspace, reference, request, storage, synthetic, committed_recovery_only
        )
    else:
        recovered = read_maintenance_handoff(
            workspace=workspace, handoff_ref=request["maintenance_handoff_ref"]
        )
        if (
            recovered["handoff"]["trade_date"] != day
            or recovered["handoff"]["release_install_ref"] != release_install_ref
        ):
            raise ContractError("DAILY_PRODUCTION_RESUME_BINDING_INVALID")
        result = _resume_registered_or_materialized(workspace, recovered, synthetic)
    if result.get("schema_version") == SCHEMA:
        # Native requested-session guard alone may return this explicit no-session result.
        if action != "EXECUTE" or result.get("business_state") != "NON_TRADING_DAY":
            raise RuntimeError("native daily production result is invalid")
        return _checked_result(result, action=action, target_trade_date=day, expected_dates=[])
    status = result.get("status")
    completion = result.get("completion_ref")
    if status == "COMPLETE":
        if completion is None:
            raise RuntimeError("native daily completion ref is missing")
        replay = replay_native_completion(
            workspace=workspace, trade_date=day, completion_ref=completion
        )
        if (
            replay.get("native_replay_validated") is not True
            or replay.get("completion_ref") != completion
            or replay.get("trade_date") != day
            or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
            or replay.get("synthetic") is not synthetic
        ):
            raise RuntimeError("native daily completion replay result is invalid")
        state, business = ("NO_ACTION" if already_completed else "SUCCEEDED"), "COMPLETE"
    else:
        if (
            status not in {"PARTIAL", "BLOCKED", "FAILED", "PENDING", "RUNNING"}
            or completion is not None
        ):
            raise RuntimeError("native daily production result is invalid")
        state, business = (
            status if status in {"PARTIAL", "BLOCKED", "FAILED"} else "PARTIAL"
        ), "INCOMPLETE"
    public = {
        "schema_version": SCHEMA,
        "action": action,
        "target_trade_date": day,
        "execution_state": state,
        "business_state": business,
        "days": [
            {
                "trade_date": day,
                "execution_state": state,
                "business_state": business,
                "completion_ref": completion,
            }
        ],
        "authority": dict(FALSE_AUTHORITY),
    }
    return _checked_result(public, action=action, target_trade_date=day, expected_dates=[day])


def _resume_registered_or_materialized(workspace, recovered, synthetic):
    if recovered["recipe"]["schema_version"] == "cn-daily-execute-recipe.v6":
        from scripts.daily_registered_recovery import (
            registered_commit_required,
            recover_registered_handoff,
        )

        if registered_commit_required(workspace=workspace, recipe=recovered["recipe"]):
            return recover_registered_handoff(
                workspace=workspace, recovered=recovered, synthetic=synthetic
            )
    materialized = materialize_daily_inputs(
        workspace=workspace, handoff_ref=recovered["handoff_ref"]
    )
    return run_materialized_native_input(
        workspace=workspace,
        input_ref=materialized["native_inputs_ref"],
        resume=True,
        synthetic=synthetic,
    )


def _execute_with_bootstrap_lock(workspace, reference, request, storage, synthetic, restricted):
    from quant_investor.operations.execution_recipe import validate_execution_recipe
    from quant_investor.operations.bootstrap_launch_contract import BootstrapLaunchInputs
    from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
    from scripts.daily_bootstrap_launch import require_no_active_automatic

    ref = request["recipe_ref"]
    stored = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=8 * 1024 * 1024)
    if stored.byte_sha256 != ref["sha256"]:
        raise ContractError("EXECUTION_CONTROL_SHA_INVALID")
    recipe = validate_execution_recipe(parse_canonical_json_bytes(stored.data), request=request)
    if restricted and recipe["schema_version"] != "cn-daily-execute-recipe.v6":
        BootstrapLaunchInputs(
            workspace=workspace,
            request_ref=reference,
            release_install_ref=request["release_install_ref"],
        ).recheck()
    kwargs = {"workspace": workspace, "request_ref": reference, "synthetic": synthetic}
    if restricted:
        kwargs["committed_recovery_only"] = True
    if recipe["bootstrap_ref"] is None:
        return execute_daily_recipe(**kwargs)
    control = AutomaticRunStorage(workspace)
    with control.locked():
        require_no_active_automatic(control)
        result = execute_daily_recipe(**kwargs)
        control.require_lock()
        return result
