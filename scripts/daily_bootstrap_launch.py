"""Bootstrap inspection and cutoff-committed recovery through existing native owners."""

from pathlib import PurePosixPath
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.operations.bootstrap_launch_contract import BootstrapLaunchInputs
from quant_investor.operations.bootstrap import verify_initial_bootstrap_baseline
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.daily_launch_contract import (
    BOOTSTRAP_SCHEMA,
    validate_launch_inspection,
)
from quant_investor.operations.execution_controls import (
    read_execution_controls,
    verify_execution_install_and_research_policies,
    verify_recipe_static_controls,
)
from quant_investor.operations.maintenance_handoff import (
    read_maintenance_handoff,
    read_recorded_maintenance_handoff,
)
from quant_investor.operations.maintenance_readback import (
    locked_finalized_maintenance_replay,
    read_auxiliary_stage_records,
)
from quant_investor.operations.materialization_contract import read_selected_materialization
from quant_investor.operations.research_cutoff import read_cutoff_inputs
from quant_investor.operations.dependency_diagnostics import DependencyInputError
from quant_investor.operations.research_timing import assert_fresh_acquisition_open
from quant_investor.operations.production_result import validate_production_result
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.system.errors import SystemNotFound, SystemStorageError
from scripts.daily_completion_replay import replay_native_completion
from scripts.daily_store_materialization import verify_initial_store_controls


def require_no_active_automatic(storage):
    pending = storage.pending()
    if pending is not None and pending["state"] == "ACTIVE":
        raise ContractError("BOOTSTRAP_AUTOMATIC_RUN_ACTIVE")


def _result(day, *, completion_ref=None, closed=False):
    value = {
        "schema_version": "cn-daily-production-result.v1",
        "action": "EXECUTE",
        "target_trade_date": day,
        "execution_state": "NO_ACTION",
        "business_state": "NON_TRADING_DAY" if closed else "COMPLETE",
        "days": (
            []
            if closed
            else [
                {
                    "trade_date": day,
                    "execution_state": "NO_ACTION",
                    "business_state": "COMPLETE",
                    "completion_ref": completion_ref,
                }
            ]
        ),
        "authority": dict(FALSE_AUTHORITY),
    }
    return validate_production_result(
        value, action="EXECUTE", target_trade_date=day, expected_dates=[] if closed else [day]
    )


def _complete(inputs, journal, *, synthetic):
    from quant_investor.operations.completion_readback import inspect_recorded_completion
    from scripts.daily_catchup import _serving_gate

    stored = journal.storage.read(str(journal.root / "completion.v1.json"))
    if stored is None:
        return None
    ref = {"path": stored.relative_path, "sha256": stored.byte_sha256}
    inspected = inspect_recorded_completion(
        workspace=inputs.workspace, trade_date=inputs.day, completion_ref=ref
    )
    snapshot = inspected.get("completed_handoff_snapshot")
    if snapshot is None:
        raise ContractError("BOOTSTRAP_COMPLETED_SNAPSHOT_REQUIRED")
    recovered = read_recorded_maintenance_handoff(snapshot)
    inputs.bind(recovered)
    replay = replay_native_completion(
        workspace=inputs.workspace, trade_date=inputs.day, completion_ref=ref
    )
    if (
        replay.get("native_replay_validated") is not True
        or replay.get("completion_ref") != ref
        or replay.get("trade_date") != inputs.day
        or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
        or replay.get("synthetic") is not synthetic
    ):
        raise ContractError("BOOTSTRAP_COMPLETED_NATIVE_REPLAY_INVALID")
    result = _result(inputs.day, completion_ref=ref)
    serving = not inputs.recipe["publish_current_dashboard"] or _serving_gate(
        inputs.workspace, dict(result["days"][0]), read_only=True
    )
    inputs.recheck()
    snapshot.recheck()
    return {
        "mode": "COMPLETE_READ_ONLY" if serving else "LOCAL_REPAIR",
        "recovery_scope": "NONE" if serving else "SERVING_ONLY",
        "result": result if serving else None,
        "completion_ref": ref,
    }


def _claim_exists(inputs):
    path = f"data/private/cn_daily_maintenance/logical_tasks/{inputs.day}-2020-execute/claim.json"
    try:
        inputs.source.storage.read_workspace_file_bytes(path, maximum_bytes=1024 * 1024)
    except FileNotFoundError:
        return False
    except SystemStorageError as exc:
        if type(exc) is SystemNotFound or (
            type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError)
        ):
            return False
        raise
    return True


def _finalized(inputs):
    if not _claim_exists(inputs):
        return None, None
    with locked_finalized_maintenance_replay(
        workspace=inputs.workspace,
        run_root="data/private/cn_daily_maintenance",
        run_date=inputs.day,
    ) as result:
        closed = (
            result is not None
            and result.get("requested_session_result", {}).get("classification")
            == "CONFIRMED_CLOSED"
        )
        if closed and (
            result.get("request_target_trade_date") != inputs.day
            or result.get("target_date") != inputs.day
            or result.get("core_completion_ref") is not None
        ):
            raise ContractError("BOOTSTRAP_CLOSED_SESSION_BINDING_INVALID")
        inputs.recheck()
        return result, _result(inputs.day, closed=True) if closed else None


def _handoff(inputs, journal):
    stored = journal.storage.read(inputs.paths["handoff"])
    if stored is None:
        return None
    ref = {"path": inputs.paths["handoff"], "sha256": stored.byte_sha256}
    recovered = read_maintenance_handoff(workspace=inputs.workspace, handoff_ref=ref)
    inputs.bind(recovered)
    return recovered


def _materialization(inputs, journal, recovered, cutoff_ref):
    from scripts.daily_materialization import MaterializedInputs, verify_materialized_inputs
    from scripts.daily_native_inputs import load_native_inputs

    stored = read_selected_materialization(
        journal=journal, execution=PurePosixPath(inputs.paths["execution"]), recipe=inputs.recipe
    )
    if stored is None:
        return
    record = parse_canonical_json_bytes(stored.data)
    if record.get("cutoff_ref") != cutoff_ref:
        raise ContractError("BOOTSTRAP_MATERIALIZATION_CUTOFF_MISMATCH")
    day, native = load_native_inputs(
        workspace=inputs.workspace, input_ref=record["native_inputs_ref"]
    )
    if day != inputs.day:
        raise ContractError("BOOTSTRAP_MATERIALIZATION_DATE_MISMATCH")
    value = MaterializedInputs(
        {"path": stored.relative_path, "sha256": stored.byte_sha256},
        record["native_inputs_ref"],
        native,
        "NO_ACTION",
    )
    verify_materialized_inputs(
        journal=journal, recovered=recovered, materialized=value, readonly=True
    )


def committed_bootstrap_scope(inputs, journal, recovered):
    stored = journal.storage.read(inputs.paths["cutoff"])
    if stored is None:
        return None
    ref = {"path": inputs.paths["cutoff"], "sha256": stored.byte_sha256}
    receipt = inputs.source.document(ref)
    if (
        receipt["request_ref"] != inputs.paths["request_ref"]
        or receipt["maintenance_handoff_ref"] != recovered["handoff_ref"]
        or receipt["core_handoff_ref"] != recovered["handoff"]["core_handoff_ref"]
        or receipt["trade_date"] != inputs.day
    ):
        raise ContractError("BOOTSTRAP_CUTOFF_BINDING_MISMATCH")
    try:
        read_cutoff_inputs(journal=journal, cutoff_ref=ref, repair=False)
    except DependencyInputError as exc:
        if exc.reason_code != "CUTOFF_COMMITTED_OBJECT_MISSING":
            raise
    _materialization(inputs, journal, recovered, ref)
    inputs.recheck()
    return ref


def inspect_bootstrap_launch(*, workspace, request_ref, release_install_ref, synthetic=False):
    inputs = BootstrapLaunchInputs(
        workspace=workspace, request_ref=request_ref, release_install_ref=release_install_ref
    )
    require_no_active_automatic(AutomaticRunStorage(workspace))
    journal = DailyJournal(workspace, inputs.day)
    value = _complete(inputs, journal, synthetic=synthetic)
    if value is None:
        finalized, closed = _finalized(inputs)
        if closed is not None:
            value = {"mode": "COMPLETE_READ_ONLY", "recovery_scope": "NONE", "result": closed}
        else:
            recovered = _handoff(inputs, journal)
            cutoff = (
                None if recovered is None else committed_bootstrap_scope(inputs, journal, recovered)
            )
            if cutoff is not None:
                value = {
                    "mode": "LOCAL_REPAIR",
                    "recovery_scope": "COMMITTED_DAG_RECOVERY",
                    "result": None,
                }
            else:
                _fresh_controls(inputs, warm=recovered is not None or finalized is not None)
                value = {"mode": "PRODUCER_REQUIRED", "recovery_scope": "NONE", "result": None}
    inputs.recheck()
    require_no_active_automatic(AutomaticRunStorage(workspace))
    result = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "request_ref": inputs.request_ref,
        "release_install_ref": release_install_ref,
        "target_trade_date": inputs.day,
        **{key: value[key] for key in ("mode", "recovery_scope", "result")},
        "authority": dict(FALSE_AUTHORITY),
    }
    return validate_launch_inspection(
        result,
        request_ref=request_ref,
        release_install_ref=release_install_ref,
        target_trade_date=inputs.day,
    )


def _fresh_controls(inputs, *, warm):
    verify_recipe_static_controls(
        workspace=inputs.workspace, recipe=inputs.recipe, document=inputs.source.document
    )
    assert_fresh_acquisition_open(inputs.recipe)
    if (
        datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        != inputs.day
    ):
        raise ContractError("BOOTSTRAP_FRESH_TARGET_NOT_CURRENT_DAY")
    if not warm:
        controls = read_execution_controls(
            workspace=inputs.workspace, request_ref=inputs.request_ref
        )
        verify_execution_install_and_research_policies(controls)
        verify_initial_store_controls(workspace=inputs.workspace, recipe=inputs.recipe)
        verify_initial_bootstrap_baseline(workspace=inputs.workspace, recipe=inputs.recipe)
        controls.recheck()


def recover_bootstrap_committed(*, workspace, request_ref, release_install_ref, synthetic=False):
    """Caller holds automatic strategy lock. This can run governed downstream writers."""
    from scripts.daily_dashboard_publication import complete_serving_result
    from scripts.daily_materialization import materialize_locked
    from scripts.daily_completion import run_materialized_native_input

    inputs = BootstrapLaunchInputs(
        workspace=workspace, request_ref=request_ref, release_install_ref=release_install_ref
    )
    journal = DailyJournal(workspace, inputs.day)
    complete = _complete(inputs, journal, synthetic=synthetic)
    if complete is not None:
        inputs.recheck()
        return complete_serving_result(
            workspace=workspace,
            completion_ref=complete["completion_ref"],
            base={"execution_state": "NO_ACTION"},
        )
    _, closed = _finalized(inputs)
    if closed is not None:
        return closed
    recovered = _handoff(inputs, journal)
    if recovered is None:
        raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_REQUIRED")
    if committed_bootstrap_scope(inputs, journal, recovered) is None:
        raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_REQUIRED")
    run_root = str(PurePosixPath(recovered["handoff"]["logical_claim_ref"]["path"]).parents[2])
    auxiliary = read_auxiliary_stage_records(
        workspace=workspace,
        run_root=run_root,
        core_ref=recovered["handoff"]["maintenance_core_ref"],
    )
    with journal.locked():
        recovered = _handoff(inputs, journal)
        if recovered is None:
            raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_REQUIRED")
        cutoff = committed_bootstrap_scope(inputs, journal, recovered)
        if cutoff is None:
            raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_REQUIRED")
        materialized = materialize_locked(
            journal=journal,
            recovered=recovered,
            auxiliary=auxiliary,
            _execute_theme=False,
            _committed_cutoff_ref=cutoff,
        )
        inputs.recheck()
    return run_materialized_native_input(
        workspace=workspace,
        input_ref=materialized.native_inputs_ref,
        resume=True,
        synthetic=synthetic,
    )
