"""Original-package recovery after a registered Store CAS; no financial writer."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.automatic_origin import (
    read_automatic_origin,
    require_origin_execution,
    require_live_origin,
    recheck_origin,
)
from quant_investor.operations.daily_preparation import Sources
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.execution_recipe import validate_execution_recipe
from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
from quant_investor.operations.materialization_contract import (
    read_selected_materialization,
    validate_materialization_shape,
)
from quant_investor.operations.production_request import validate_production_request
from quant_investor.operations.research_cutoff import read_cutoff_inputs
from quant_investor.operations.research_timing import CURRENT
from quant_investor.strategy_records.corporate_contracts import instant
from scripts.daily_production_store_adapter import (
    StoreCloseAdapter,
    RECORD_ROOT,
    native,
    _operation_lock,
)


def _now():
    return datetime.now(timezone.utc)


def validate_recovery_chronology(*, handoff, plan, cutoff, materialization, completion=None):
    """Original custody precedes observed CAS; no reconstruction of the CAS time."""
    stamps = [
        handoff["sealed_at"],
        plan["transaction_planned_at"],
        cutoff["physical_reads_completed_at"],
        cutoff["portfolio_created_at"],
        cutoff["as_of"],
        cutoff["sealed_at"],
        materialization["sealed_at"],
    ]
    times = [
        instant(value) if index == 2 else utc_stamp(value) for index, value in enumerate(stamps)
    ]
    if times != sorted(times) or times[-1] > _now():
        raise ContractError("REGISTERED_RECOVERY_CUSTODY_TIME_INVALID")
    if (
        completion is not None
        and not times[-1] <= utc_stamp(completion["cas_observed_at"]) <= _now()
    ):
        raise ContractError("REGISTERED_RECOVERY_CAS_OBSERVATION_INVALID")
    return stamps[-1]


def _original_inputs(workspace, request_ref, release_install_ref):
    # Preserve the native request/source mode contract, including checked-in
    # read-only 0644 inputs, while rechecking both bytes and file identities.
    source = Sources(str(Path(workspace).resolve(strict=True)))
    request_ref = validate_ref(request_ref)
    request = validate_production_request(
        source.document(request_ref), release_install_ref=release_install_ref
    )
    if (
        request["schema_version"] != "cn-daily-production-request.v1"
        or request["action"] != "EXECUTE"
    ):
        raise ContractError("REGISTERED_RECOVERY_EXECUTE_REQUIRED")
    recipe = validate_execution_recipe(source.document(request["recipe_ref"]), request=request)
    if (
        recipe["schema_version"] != "cn-daily-execute-recipe.v6"
        or recipe["research_timing"]["mode"] != CURRENT
    ):
        raise ContractError("REGISTERED_RECOVERY_PROFILE_REQUIRED")
    return source, request, recipe


def registered_commit_required(*, workspace, recipe):
    """A changed/missing current writer cannot take the fresh financial branch."""
    pointer = Path(workspace) / RECORD_ROOT / "_record_store/current.v1.json"
    return native._pointer_sha(pointer) != recipe["store_preimages"]["store_pointer_ref"]["sha256"]


def recover_registered_handoff(*, workspace, recovered, synthetic=False):
    """RESUME derives the original identity from exact retained custody."""
    handoff = recovered["handoff"]
    origin_ref = handoff.get("automatic_origin_ref")
    request_ref = handoff["request_ref"]
    if origin_ref is not None:
        context = read_automatic_origin(
            workspace=workspace, reference=origin_ref, synthetic=synthetic
        )
        request_ref = context["origin"]["execution_request_ref"]
    return recover_registered_committed(
        workspace=workspace,
        request_ref=request_ref,
        release_install_ref=handoff["release_install_ref"],
        synthetic=synthetic,
        automatic_origin_ref=origin_ref,
    )


def inspect_registered_recovery(
    *, workspace, request_ref, release_install_ref, synthetic=False, automatic_origin_ref=None
):
    """Read-only inspection of one exact execution, never a scan or new cutoff."""
    from scripts.daily_materialization import MaterializedInputs, verify_materialized_inputs
    from scripts.daily_native_inputs import load_native_inputs

    if type(synthetic) is not bool:
        raise ContractError("REGISTERED_RECOVERY_PROVENANCE_INVALID")
    source, request, recipe = _original_inputs(workspace, request_ref, release_install_ref)
    day = request["target_trade_date"]
    journal = DailyJournal(source.workspace, day)
    execution = journal.root / "executions" / request_ref["sha256"]
    handoff_path = str(execution / "maintenance-handoff.v1.json")
    stored = journal.storage.read(handoff_path)
    if stored is None:
        raise ContractError("REGISTERED_RECOVERY_HANDOFF_MISSING")
    handoff_ref = {"path": handoff_path, "sha256": stored.byte_sha256}
    handoff = source.document(handoff_ref)
    recovered = read_maintenance_handoff(workspace=source.workspace, handoff_ref=handoff_ref)
    if handoff["schema_version"] not in {
        "cn-daily-maintenance-handoff.v2",
        "cn-daily-maintenance-handoff.v4",
    }:
        raise ContractError("REGISTERED_RECOVERY_HANDOFF_PROFILE_INVALID")
    for field, prefix, original in (
        ("request_ref", "request", request_ref),
        ("recipe_ref", "recipe", request["recipe_ref"]),
    ):
        expected = {
            "path": str(execution / "inputs" / f"{prefix}-{original['sha256']}.json"),
            "sha256": original["sha256"],
        }
        if handoff[field] != expected or source.raw(expected) != source.raw(original):
            raise ContractError("REGISTERED_RECOVERY_ORIGINAL_INPUT_MISMATCH")
    if (
        recovered["handoff"] != handoff
        or recovered["request"] != request
        or recovered["recipe"] != recipe
    ):
        raise ContractError("REGISTERED_RECOVERY_RETAINED_INPUT_MISMATCH")
    origin = None
    if handoff.get("automatic_origin_ref") != automatic_origin_ref:
        raise ContractError("REGISTERED_RECOVERY_AUTOMATIC_ORIGIN_MISMATCH")
    if automatic_origin_ref is not None:
        origin = read_automatic_origin(
            workspace=source.workspace, reference=automatic_origin_ref, synthetic=synthetic
        )
        require_origin_execution(
            origin,
            request_ref=request_ref,
            request=request,
            recipe=recipe,
            sealed_at=handoff["sealed_at"],
        )
    materialization = read_selected_materialization(
        journal=journal, execution=execution, recipe=recipe
    )
    if materialization is None:
        raise ContractError("REGISTERED_RECOVERY_MATERIALIZATION_MISSING")
    materialization_ref = {
        "path": materialization.relative_path,
        "sha256": materialization.byte_sha256,
    }
    record = source.document(materialization_ref)
    validate_materialization_shape(record)
    cutoff_ref = validate_ref(record["cutoff_ref"])
    if cutoff_ref["path"] != str(execution / "research-cutoff.v2.json"):
        raise ContractError("REGISTERED_RECOVERY_CUTOFF_PATH_INVALID")
    cutoff = read_cutoff_inputs(journal=journal, cutoff_ref=cutoff_ref, repair=False)
    source.raw(cutoff_ref)
    for ref in cutoff["receipt"]["source_refs"]:
        source.raw(ref)
    for document in (record, cutoff["receipt"]):
        for name, ref in document.items():
            if name.endswith("_ref") and ref is not None:
                source.raw(ref)
    loaded_day, inputs = load_native_inputs(
        workspace=source.workspace, input_ref=record["native_inputs_ref"]
    )
    native_input = source.document(record["native_inputs_ref"])
    for name in ("decision_recipe_ref", "store_policy_ref"):
        source.raw(native_input[name])
    if loaded_day != day or native_input["schema_version"] != "cn-daily-native-inputs.v7":
        raise ContractError("REGISTERED_RECOVERY_NATIVE_PROFILE_INVALID")
    materialized = MaterializedInputs(
        materialization_ref, record["native_inputs_ref"], inputs, "NO_ACTION"
    )
    verify_materialized_inputs(
        journal=journal, recovered=recovered, materialized=materialized, readonly=True
    )
    adapter = StoreCloseAdapter(
        arguments=inputs.store_arguments,
        trade_date=day,
        plan_ref=record["store_plan_ref"],
        release_ref=recipe["release_ref"],
    )
    source.raw(record["store_plan_ref"])
    # C1 owns its native JSON serialization; preserve its exact bytes and use
    # the adapter's native decoder instead of imposing DAG canonical encoding.
    plan = adapter.plan
    from quant_investor.operations.registered_recovery_source import verify_configured_origin

    configured = verify_configured_origin(source=source, origin=origin, native_plan=plan)
    if (
        adapter.version != 2
        or plan != adapter.plan
        or plan != cutoff["native_plan"]
        or plan["preimages"]["calendar_receipt_sha256"] != handoff["calendar_ref"]["sha256"]
        or plan["registered_event_declaration_ref"] != recipe["registered_event_declaration_ref"]
        or record["cutoff_ref"] != native_input["cutoff_ref"]
    ):
        raise ContractError("REGISTERED_RECOVERY_PLAN_BINDING_INVALID")
    completion_path = native._completion_path(
        adapter.args["record_root"], plan["transaction_id"], 2
    )
    complete_metadata = (
        completion_path.exists() and completion_path.with_name("committed-pointer.v1.json").exists()
    )
    proof = (
        native.inspect_frozen_close_commit if complete_metadata else native.inspect_close_commit
    )(**adapter._proof_args())
    validate_recovery_chronology(
        handoff=handoff,
        plan=plan,
        cutoff=cutoff["receipt"],
        materialization=record,
        completion=proof["completion"],
    )
    source.recheck()
    if origin is not None:
        recheck_origin(origin)
    return {
        "source": source,
        "request": request,
        "recipe": recipe,
        "recovered": recovered,
        "materialized": materialized,
        "record": record,
        "cutoff": cutoff,
        "plan": plan,
        "adapter": adapter,
        "proof": proof,
        "origin": origin,
        "metadata_complete": complete_metadata,
        "configured": configured,
    }


@contextmanager
def _strategy_lock(workspace, origin):
    if origin is not None:
        require_live_origin(origin, workspace=workspace)
        yield lambda: require_live_origin(origin, workspace=workspace)
        require_live_origin(origin, workspace=workspace)
    else:
        storage = AutomaticRunStorage(workspace)
        with storage.locked():

            def recheck():
                storage.require_lock()
                pending = storage.pending()
                if pending is not None and pending["state"] == "ACTIVE":
                    raise ContractError("REGISTERED_RECOVERY_AUTOMATIC_RUN_ACTIVE")

            recheck()
            yield recheck
            recheck()


def recover_registered_committed(
    *, workspace, request_ref, release_install_ref, synthetic=False, automatic_origin_ref=None
):
    """Recheck original package under all locks, repair metadata, then resume EOD."""
    from scripts.daily_completion import run_materialized_native_input

    source, request, _ = _original_inputs(workspace, request_ref, release_install_ref)
    origin = (
        None
        if automatic_origin_ref is None
        else read_automatic_origin(
            workspace=source.workspace, reference=automatic_origin_ref, synthetic=synthetic
        )
    )
    journal = DailyJournal(source.workspace, request["target_trade_date"])
    with _strategy_lock(source.workspace, origin) as recheck_strategy:
        with journal.locked(), _operation_lock(str(Path(source.workspace) / RECORD_ROOT)):
            scope = inspect_registered_recovery(
                workspace=source.workspace,
                request_ref=request_ref,
                release_install_ref=release_install_ref,
                synthetic=synthetic,
                automatic_origin_ref=automatic_origin_ref,
            )

            def guard(*, proof, completion):
                journal._require_lock()
                recheck_strategy()
                source.recheck()
                scope["source"].recheck()
                if scope["configured"] is not None:
                    scope["configured"]["source_context"].recheck()
                again = read_maintenance_handoff(
                    workspace=source.workspace, handoff_ref=scope["recovered"]["handoff_ref"]
                )
                if any(
                    again[name] != scope["recovered"][name]
                    for name in ("handoff", "request", "recipe")
                ):
                    raise ContractError("REGISTERED_RECOVERY_HANDOFF_CHANGED")
                if (
                    proof["plan"] != scope["plan"]
                    or proof["pointer_sha256"] != scope["proof"]["pointer_sha256"]
                ):
                    raise ContractError("REGISTERED_RECOVERY_NATIVE_COMMIT_CHANGED")
                validate_recovery_chronology(
                    handoff=scope["recovered"]["handoff"],
                    plan=scope["plan"],
                    cutoff=scope["cutoff"]["receipt"],
                    materialization=scope["record"],
                    completion=completion,
                )

            if scope["metadata_complete"]:
                guard(proof=scope["proof"], completion=scope["proof"]["completion"])
            else:
                native.recover_close_completion(
                    **scope["adapter"]._proof_args(), _metadata_guard=guard
                )
            final = native.inspect_frozen_close_commit(**scope["adapter"]._proof_args())
            guard(proof=final, completion=final["completion"])
        recheck_strategy()
        return run_materialized_native_input(
            workspace=source.workspace,
            input_ref=scope["materialized"].native_inputs_ref,
            resume=True,
            synthetic=synthetic,
        )
