"""Materialize exact downstream inputs from the early native maintenance handoff."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import (
    ContractError,
    EOD_NODE_IDS,
    GRAPH_SHA256,
    validate_ref,
    utc_stamp,
)
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
from quant_investor.operations.maintenance_readback import read_auxiliary_stage_records
from quant_investor.operations.research_materialization import publish_research_input
from quant_investor.operations import bootstrap as bootstrap_evidence
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_completion_replay import replay_native_completion
from scripts.daily_native_inputs import load_native_inputs, verify_loaded_native_inputs
from scripts.daily_native_registry import NativeDailyInputs
from scripts.daily_store_materialization import (
    prepare_materialized_store_plan,
    materialized_adjustment_refs,
    verify_materialized_adjustment_refs,
)

from quant_investor.operations.materialization_contract import (
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V4,
    SCHEMA_V5,
    SCHEMA_V6,
    layout_for_recipe,
    read_selected_materialization,
    validate_materialization_shape,
    validate_bound_materialization,
)


def execute_daily_recipe(
    *,
    workspace: str,
    request_ref: dict,
    synthetic: bool = False,
    _catchup_binding_ref=None,
    _automatic_origin_ref=None,
    committed_recovery_only: bool = False,
) -> dict:
    """Native EXECUTE; historical requests require the fixed batch derivation."""
    from pathlib import Path
    from zoneinfo import ZoneInfo
    from quant_investor.operations.production_request import validate_production_request
    from quant_investor.operations.execution_controls import (
        read_execution_controls,
        verify_execution_install_and_research_policies,
    )
    from quant_investor.operations.maintenance_handoff import publish_maintenance_handoff
    from quant_investor.operations.completion_readback import inspect_recorded_completion
    from quant_investor.market.daily_factor_loop import DailyFactorLoop
    from quant_investor.market.daily_maintenance import run_cn_daily_maintenance
    from scripts.daily_store_materialization import verify_initial_store_controls
    from scripts.daily_completion_replay import verify_initial_previous_completion
    from scripts.daily_completion import run_materialized_native_input

    if type(synthetic) is not bool or type(committed_recovery_only) is not bool:
        raise ContractError("EXECUTION_SYNTHETIC_FLAG_INVALID")
    root = Path(workspace).resolve(strict=True)
    source = SecureSystemStorage(str(root))
    request = parse_canonical_json_bytes(_read(source, request_ref))
    request = validate_production_request(
        request, release_install_ref=request["release_install_ref"]
    )
    if request["action"] != "EXECUTE":
        raise ContractError("EXECUTION_ACTION_REQUIRED")
    if committed_recovery_only:
        if _catchup_binding_ref is not None:
            raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_PROFILE_INVALID")
        original_recipe = parse_canonical_json_bytes(_read(source, request["recipe_ref"]))
        if original_recipe.get("schema_version") == "cn-daily-execute-recipe.v6":
            from scripts.daily_registered_recovery import recover_registered_committed

            return recover_registered_committed(
                workspace=str(root),
                request_ref=request_ref,
                release_install_ref=request["release_install_ref"],
                synthetic=synthetic,
                automatic_origin_ref=_automatic_origin_ref,
            )
        if _automatic_origin_ref is not None:
            raise ContractError("BOOTSTRAP_COMMITTED_RECOVERY_PROFILE_INVALID")
        from scripts.daily_bootstrap_launch import recover_bootstrap_committed

        return recover_bootstrap_committed(
            workspace=str(root),
            request_ref=request_ref,
            release_install_ref=request["release_install_ref"],
            synthetic=synthetic,
        )
    from quant_investor.operations.production_request import HISTORICAL_SCHEMA
    from quant_investor.operations.catchup_binding import read_catchup_binding, historical_input

    historical = request["schema_version"] == HISTORICAL_SCHEMA
    if historical != (_catchup_binding_ref is not None):
        raise ContractError("EXECUTION_CATCHUP_BINDING_REQUIRED")
    origin = None
    if _automatic_origin_ref is not None:
        from quant_investor.operations.automatic_origin import (
            read_automatic_origin,
            require_origin_execution,
            require_live_origin,
        )

        if historical:
            raise ContractError("EXECUTION_AUTOMATIC_ORIGIN_PROFILE_INVALID")
        origin = read_automatic_origin(
            workspace=str(root),
            reference=_automatic_origin_ref,
            synthetic=synthetic,
        )
        require_origin_execution(origin, request_ref=request_ref, request=request)
        require_live_origin(origin, workspace=str(root))
    bound = None
    if historical:
        bound = read_catchup_binding(workspace=str(root), binding_ref=_catchup_binding_ref)
        if bound["binding"]["execution_request_ref"] != request_ref or bound["request"] != request:
            raise ContractError("EXECUTION_CATCHUP_REQUEST_MISMATCH")
    day = request["target_trade_date"]
    journal = DailyJournal(str(root), day)
    completed = journal.storage.read(str(journal.root / "completion.v1.json"))
    if completed is not None:
        ref = {"path": completed.relative_path, "sha256": completed.byte_sha256}
        inspected = inspect_recorded_completion(
            workspace=str(root), trade_date=day, completion_ref=ref
        )
        snapshot = inspected.get("completed_handoff_snapshot")
        if (
            snapshot is None
            or snapshot.document("handoff")["request_ref"]["sha256"] != request_ref["sha256"]
        ):
            raise ContractError("EXECUTION_COMPLETED_REQUEST_MISMATCH")
        replay_native_completion(workspace=str(root), trade_date=day, completion_ref=ref)
        from scripts.daily_dashboard_publication import complete_serving_result

        return complete_serving_result(
            workspace=str(root), completion_ref=ref, base={"execution_state": "NO_ACTION"}
        )
    path = str(journal.root / "executions" / request_ref["sha256"] / "maintenance-handoff.v1.json")

    def resume(ref):
        recovered = read_maintenance_handoff(workspace=str(root), handoff_ref=ref)
        if recovered["handoff"]["request_ref"]["sha256"] != request_ref["sha256"]:
            raise ContractError("EXECUTION_HANDOFF_REQUEST_MISMATCH")
        if (
            _automatic_origin_ref is not None
            and recovered["handoff"].get("automatic_origin_ref") != _automatic_origin_ref
        ):
            raise ContractError("EXECUTION_HANDOFF_AUTOMATIC_ORIGIN_MISMATCH")
        if recovered["recipe"]["schema_version"] == "cn-daily-execute-recipe.v6":
            from scripts.daily_registered_recovery import (
                registered_commit_required,
                recover_registered_committed,
            )

            if registered_commit_required(workspace=str(root), recipe=recovered["recipe"]):
                return recover_registered_committed(
                    workspace=str(root),
                    request_ref=request_ref,
                    release_install_ref=request["release_install_ref"],
                    synthetic=synthetic,
                    automatic_origin_ref=_automatic_origin_ref,
                )
        materialized = materialize_daily_inputs(
            workspace=str(root), handoff_ref=ref, _execute_theme=True
        )
        return run_materialized_native_input(
            workspace=str(root),
            input_ref=materialized["native_inputs_ref"],
            resume=True,
            synthetic=synthetic,
        )

    existing = journal.storage.read(path)
    if existing is not None:
        return resume({"path": path, "sha256": existing.byte_sha256})
    controls = read_execution_controls(workspace=str(root), request_ref=request_ref)
    verify_execution_install_and_research_policies(controls)
    recipe = controls.document(controls.recipe_ref)
    from quant_investor.operations.research_timing import assert_fresh_acquisition_open

    assert_fresh_acquisition_open(recipe)
    verify_initial_store_controls(workspace=str(root), recipe=recipe)
    published = []

    def early_handoff(state):
        controls.recheck()
        kwargs = {}
        if bound is not None:
            bound["sources"].recheck()
            kwargs["_catchup_binding_ref"] = _catchup_binding_ref
        if origin is not None:
            require_live_origin(origin, workspace=str(root))
            kwargs["_automatic_origin_ref"] = _automatic_origin_ref
        ref = publish_maintenance_handoff(
            workspace=str(root), request_ref=request_ref, state=dict(state), **kwargs
        )
        published.append(ref)
        return ref

    context_ref = recipe["factor_loop_context_ref"]
    run_root = root / "data/private/cn_daily_maintenance"

    def make_loop():
        controls.recheck()
        return DailyFactorLoop(
            workspace_root=str(root),
            run_root=str(run_root),
            context_path=str(root / context_ref["path"]),
            context_sha256=context_ref["sha256"],
            core_handoff_completed=early_handoff,
        )

    def finish():
        if not published:
            raise ContractError("EXECUTION_EARLY_HANDOFF_UNAVAILABLE")
        if any(ref != published[0] for ref in published):
            raise ContractError("EXECUTION_EARLY_HANDOFF_CONFLICT")
        return resume(published[0])

    def closed_result():
        from quant_investor.operations.production_result import validate_production_result

        result = {
            "schema_version": "cn-daily-production-result.v1",
            "action": "EXECUTE",
            "target_trade_date": day,
            "execution_state": "NO_ACTION",
            "business_state": "NON_TRADING_DAY",
            "days": [],
            "authority": FALSE_AUTHORITY,
        }
        return validate_production_result(
            result, action="EXECUTE", target_trade_date=day, expected_dates=[]
        )

    import os
    from quant_investor.operations.maintenance_readback import locked_finalized_maintenance_replay

    claim = run_root / "logical_tasks" / (day + "-2020-execute") / "claim.json"
    if os.path.lexists(claim):
        recovery_kwargs = {"_catchup_binding_ref": _catchup_binding_ref} if historical else {}
        if not historical and recipe["previous_completion_ref"] is not None:
            recovery_kwargs["expected_previous_trade_date"] = Path(
                recipe["previous_completion_ref"]["path"]
            ).parent.name
        with locked_finalized_maintenance_replay(
            workspace=str(root),
            run_root=str(run_root.relative_to(root)),
            run_date=day,
            **recovery_kwargs,
        ) as recovered:
            if recovered is not None:
                if (
                    recovered.get("requested_session_result", {}).get("classification")
                    == "CONFIRMED_CLOSED"
                ):
                    return closed_result()
                core = recovered.get("core_completion_ref")
                if core is None:
                    raise ContractError("EXECUTION_FINALIZED_CORE_MISSING")
                loop = make_loop()
                loop._replay_core_completed(core)
                loop.report(maintenance=recovered)
        if recovered is not None:
            return finish()
    if (
        not historical
        and datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        != day
    ):
        raise ContractError("EXECUTION_FRESH_TARGET_NOT_CURRENT_SESSION_DATE")
    if recipe["bootstrap_ref"] is not None:
        bootstrap_evidence.verify_initial_bootstrap_baseline(workspace=str(root), recipe=recipe)
    else:
        verify_initial_previous_completion(workspace=str(root), recipe=recipe)
    loop = make_loop()
    controls.recheck()
    maintenance_kwargs = {"_expected_target_trade_date": day}
    if not historical and recipe["previous_completion_ref"] is not None:
        maintenance_kwargs["_expected_previous_trade_date"] = Path(
            recipe["previous_completion_ref"]["path"]
        ).parent.name
    if bound is not None:
        bound["sources"].recheck()
        maintenance_kwargs = {"_historical_calendar_input": historical_input(bound["binding"])}
    maintenance = run_cn_daily_maintenance(
        workspace_root=str(root),
        run_root=str(run_root),
        mode="execute",
        attempt_slot="2020",
        core_completed=loop.core_completed,
        _core_replay_completed=loop._replay_core_completed,
        **maintenance_kwargs,
    )
    if (
        maintenance.get("execution_disposition") == "NON_TRADING_DAY"
        or maintenance.get("requested_session_result", {}).get("classification")
        == "CONFIRMED_CLOSED"
    ):
        return closed_result()
    loop.report(maintenance=maintenance)
    return finish()


@dataclass(frozen=True)
class MaterializedInputs:
    materialization_ref: dict
    native_inputs_ref: dict
    inputs: NativeDailyInputs
    status: str


def _read(storage, ref):
    validate_ref(ref)
    raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
    if raw.byte_sha256 != ref["sha256"]:
        raise ContractError("MATERIALIZATION_REF_SHA_MISMATCH")
    return raw.data


def _recorded(journal, recovered, auxiliary, path, stored, *, loaded_inputs=None):
    value = parse_canonical_json_bytes(stored.data)
    execution = PurePosixPath(recovered["handoff_ref"]["path"]).parent
    validate_bound_materialization(
        workspace=str(journal.storage._io.workspace_root),
        value=value,
        recipe=recovered["recipe"],
        execution=execution,
        path=path,
    )
    if (
        read_selected_materialization(
            journal=journal, execution=execution, recipe=recovered["recipe"]
        )
        != stored
    ):
        raise ContractError("MATERIALIZATION_RECEIPT_CHANGED")
    if (
        value["trade_date"] != journal.trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or value["maintenance_handoff_ref"] != recovered["handoff_ref"]
        or not _false_authority(value["authority"])
    ):
        raise ContractError("MATERIALIZATION_RECEIPT_INVALID")
    if (
        not utc_stamp(recovered["handoff"]["sealed_at"])
        <= utc_stamp(value["sealed_at"])
        <= datetime.now(timezone.utc)
    ):
        raise ContractError("MATERIALIZATION_RECEIPT_TIME_INVALID")
    if (
        value["schema_version"] in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}
        and value["theme_source_handoff_ref"] is not None
    ):
        from quant_investor.operations.theme_handoff_readback import read_theme_handoff

        theme = read_theme_handoff(
            journal=journal,
            request_ref=recovered["handoff"]["request_ref"],
            core_handoff_ref=recovered["handoff"]["core_handoff_ref"],
            handoff_ref=value["theme_source_handoff_ref"],
        )
        if utc_stamp(theme["handoff"]["sealed_at"]) > utc_stamp(value["sealed_at"]):
            raise ContractError("MATERIALIZATION_THEME_SEAL_CHRONOLOGY_INVALID")
    execution = PurePosixPath(path).parent
    for key, prefix in (("native_inputs_ref", "native"), ("research_request_ref", "research")):
        ref = validate_ref(value[key])
        expected_path = str(execution / "inputs" / f"{prefix}-{ref['sha256']}.json")
        if value["schema_version"] in {SCHEMA_V5, SCHEMA_V6} and key == "research_request_ref":
            expected_path = str(
                execution
                / (
                    "inputs/research.v3.json"
                    if value["schema_version"] == SCHEMA_V6
                    else "inputs/research.v2.json"
                )
            )
        if ref["path"] != expected_path:
            raise ContractError("MATERIALIZATION_INPUT_PATH_INVALID")
    expected_aux = {
        name: (
            None
            if recovered["recipe"]["research_sources"][name]["mode"] == "PINNED"
            else auxiliary["stages"][name]["ref"]
        )
        for name in ("fundamental", "macro")
    }
    if value["auxiliary_stage_refs"] != expected_aux:
        raise ContractError("MATERIALIZATION_AUXILIARY_BINDING_CHANGED")
    storage = SecureSystemStorage(str(journal.storage._io.workspace_root))
    for ref in [
        value["research_request_ref"],
        value["store_plan_ref"],
        *[r for r in expected_aux.values() if r is not None],
    ]:
        _read(storage, ref)
    native_document = parse_canonical_json_bytes(_read(storage, value["native_inputs_ref"]))
    if native_document.get("schema_version") in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        from quant_investor.operations.decision_recipe import read_decision_recipe

        bound = read_decision_recipe(
            workspace=journal.storage._io.workspace_root,
            trade_date=journal.trade_date,
            recipe_ref=native_document["decision_recipe_ref"],
            research_request_ref=value["research_request_ref"],
            store_plan_ref=value["store_plan_ref"],
        )
        if utc_stamp(bound["portfolio"]["created_at"]) > utc_stamp(value["sealed_at"]):
            raise ContractError("MATERIALIZATION_PORTFOLIO_CUSTODY_AFTER_SEAL")
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    if value["schema_version"] in {SCHEMA_V5, SCHEMA_V6}:
        from quant_investor.operations.research_cutoff import read_cutoff_inputs

        cutoff = read_cutoff_inputs(journal=journal, cutoff_ref=value["cutoff_ref"])
        if value["schema_version"] == SCHEMA_V6 and any(
            reference != recipe["registered_event_declaration_ref"]
            for reference in (
                value["registered_event_declaration_ref"],
                native_document.get("registered_event_declaration_ref"),
                cutoff["receipt"].get("registered_event_declaration_ref"),
            )
        ):
            raise ContractError("MATERIALIZATION_REGISTERED_BINDING_MISMATCH")
        if (
            cutoff["research_request_ref"] != value["research_request_ref"]
            or cutoff["store_plan_ref"] != value["store_plan_ref"]
            or cutoff["corporate_action_context_ref"] != value["corporate_action_context_ref"]
            or cutoff["corporate_action_template_ref"] != value["corporate_action_template_ref"]
            or cutoff["receipt"]["maintenance_handoff_ref"] != recovered["handoff_ref"]
            or native_document.get("schema_version")
            != (
                "cn-daily-native-inputs.v7"
                if value["schema_version"] == SCHEMA_V6
                else "cn-daily-native-inputs.v6"
            )
            or native_document.get("cutoff_ref") != value["cutoff_ref"]
            or native_document.get("corporate_action_context_ref")
            != value["corporate_action_context_ref"]
            or native_document.get("dashboard_publication_policy")
            != recipe["dashboard_publication_policy"]
            or utc_stamp(cutoff["receipt"]["sealed_at"]) > utc_stamp(value["sealed_at"])
        ):
            raise ContractError("MATERIALIZATION_CUTOFF_CLOSURE_MISMATCH")
    elif value["schema_version"] in {SCHEMA_V3, SCHEMA_V4}:
        expected_profile = (
            "cn-daily-native-inputs.v5"
            if value["schema_version"] == SCHEMA_V4
            else "cn-daily-native-inputs.v4"
        )
        if (
            native_document.get("schema_version") != expected_profile
            or native_document.get("corporate_action_context_ref")
            != recipe["corporate_action_context_ref"]
        ):
            raise ContractError("MATERIALIZATION_CORPORATE_CONTEXT_MISMATCH")
        if (
            value["schema_version"] == SCHEMA_V4
            and native_document.get("dashboard_publication_policy")
            != recipe["dashboard_publication_policy"]
        ):
            raise ContractError("MATERIALIZATION_DASHBOARD_POLICY_MISMATCH")
    elif native_document.get("schema_version") in {
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        raise ContractError("MATERIALIZATION_UNEXPECTED_CORPORATE_PROFILE")
    from scripts.daily_store_adoption import verify_store_plan_binding
    from scripts.daily_store_materialization import store_materialization_arguments
    from scripts.daily_production_store_adapter import native

    bound_plan = native._load_json(
        journal.storage._io.workspace_root / value["store_plan_ref"]["path"],
        expected_sha=value["store_plan_ref"]["sha256"],
        label="materialization Store binding plan",
    )
    verify_store_plan_binding(
        arguments=store_materialization_arguments(
            journal.storage._io.workspace_root, handoff, recipe
        ),
        plan_ref=value["store_plan_ref"],
        plan=bound_plan,
        custody_at=value["sealed_at"],
    )
    previous = recipe["previous_completion_ref"]
    if previous is None:
        from scripts.daily_production_store_adapter import native

        proof = bootstrap_evidence.verify_bootstrap_native(
            workspace=str(journal.storage._io.workspace_root), recovered=recovered
        )
        plan = native._load_json(
            journal.storage._io.workspace_root / value["store_plan_ref"]["path"],
            expected_sha=value["store_plan_ref"]["sha256"],
            label="bootstrap retained Store plan",
        )
        bootstrap_evidence.verify_bootstrap_store_plan(
            proof=proof, plan=plan, calendar_ref=handoff["calendar_ref"]
        )
        previous_day = proof["previous_trade_date"]
    else:
        previous_day = PurePosixPath(previous["path"]).parent.name
    expected = {
        "factor_pointer_sha256": handoff["factor_pointer_ref"]["sha256"],
        "release_ref": handoff["release_ref"],
        "calendar_ref": handoff["calendar_ref"],
        "market_snapshot_ref": handoff["market_snapshot_ref"],
        "store_policy_ref": recipe["policy_refs"]["store"],
        "retrospective_ref": recipe["retrospective_ref"],
        "benchmark_ref": recipe["dashboard_sources"]["benchmark_ref"],
        "risk_free_ref": recipe["dashboard_sources"]["risk_free_ref"],
        "publish_current_dashboard": recipe["publish_current_dashboard"],
        "previous_trade_date": previous_day,
    }
    core = parse_canonical_json_bytes(_read(storage, handoff["core_handoff_ref"]))
    calendar = parse_canonical_json_bytes(_read(storage, core["node_refs"]["calendar"]))
    expected.update(
        next_session_calendar_proof_ref=calendar["output_refs"].get("next_session_calendar_proof"),
        next_session_calendar_failure_ref=calendar["output_refs"].get(
            "next_session_calendar_failure"
        ),
    )
    if any(native_document[key] != item for key, item in expected.items()):
        raise ContractError("MATERIALIZATION_RECIPE_BINDING_CHANGED")
    verify_materialized_adjustment_refs(
        workspace=journal.storage._io.workspace_root,
        snapshot_ref=handoff["market_snapshot_ref"],
        refs=native_document["adjustment_market_refs"],
    )
    if (
        native_document["research_request_ref"] != value["research_request_ref"]
        or native_document["store_plan_ref"] != value["store_plan_ref"]
    ):
        raise ContractError("MATERIALIZATION_NATIVE_BINDING_INVALID")
    if loaded_inputs is None:
        day, inputs = load_native_inputs(
            workspace=str(journal.storage._io.workspace_root), input_ref=value["native_inputs_ref"]
        )
    else:
        verify_loaded_native_inputs(
            workspace=str(journal.storage._io.workspace_root),
            input_ref=value["native_inputs_ref"],
            trade_date=journal.trade_date,
            inputs=loaded_inputs,
        )
        day, inputs = journal.trade_date, loaded_inputs
    if day != journal.trade_date:
        raise ContractError("MATERIALIZATION_NATIVE_DATE_INVALID")
    if journal.storage.read(path) != stored:
        raise ContractError("MATERIALIZATION_RECEIPT_CHANGED")
    return MaterializedInputs(
        {"path": path, "sha256": stored.byte_sha256},
        value["native_inputs_ref"],
        inputs,
        "NO_ACTION",
    )


def verify_materialized_inputs(*, journal, recovered, materialized, readonly=False):
    """Read-only replay from sealed auxiliary refs, preserving the loaded context."""
    from quant_investor.operations import research_materialization

    if type(readonly) is not bool:
        raise ContractError("MATERIALIZATION_VERIFY_MODE_INVALID")
    if not readonly:
        journal._require_lock()
    ref = validate_ref(materialized.materialization_ref)
    stored = journal.storage.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise ContractError("MATERIALIZATION_RECEIPT_SHA_MISMATCH")
    record = parse_canonical_json_bytes(stored.data)
    validate_materialization_shape(record)
    if type(record["auxiliary_stage_refs"]) is not dict or set(record["auxiliary_stage_refs"]) != {
        "fundamental",
        "macro",
    }:
        raise ContractError("MATERIALIZATION_AUXILIARY_BINDING_CHANGED")
    storage = SecureSystemStorage(str(journal.storage._io.workspace_root))
    stages = {}
    for name in ("fundamental", "macro"):
        stage_ref = record["auxiliary_stage_refs"][name]
        document = (
            None if stage_ref is None else parse_canonical_json_bytes(_read(storage, stage_ref))
        )
        if document is not None and (
            type(document) is not dict
            or set(document) != {"state", "result"}
            or document["state"] != "STAGE_COMPLETED"
            or type(document["result"]) is not dict
        ):
            raise ContractError("MATERIALIZATION_AUXILIARY_STAGE_INVALID")
        stages[name] = {
            "state": "MISSING" if stage_ref is None else "RECORDED",
            "ref": stage_ref,
            "document": document,
        }
    auxiliary = {"stages": stages}
    verified = _recorded(
        journal, recovered, auxiliary, ref["path"], stored, loaded_inputs=materialized.inputs
    )
    if verified.native_inputs_ref != materialized.native_inputs_ref:
        raise ContractError("MATERIALIZATION_NATIVE_BINDING_INVALID")
    if record["schema_version"] in {SCHEMA_V5, SCHEMA_V6}:
        from quant_investor.operations.research_cutoff import read_cutoff_inputs

        cutoff = read_cutoff_inputs(journal=journal, cutoff_ref=record["cutoff_ref"])
        research = {
            "research_request_ref": cutoff["research_request_ref"],
            "theme_source_handoff_ref": cutoff["source_bundle"]["theme_source_handoff_ref"],
            "auxiliary_stage_refs": cutoff["source_bundle"]["auxiliary_stage_refs"],
        }
    else:
        research = research_materialization.publish_research_input(
            journal=journal, recovered=recovered, auxiliary=auxiliary, verify_only=True
        )
    if (
        record["schema_version"] in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}
        and research["theme_source_handoff_ref"] != record["theme_source_handoff_ref"]
    ):
        raise ContractError("MATERIALIZATION_THEME_HANDOFF_BINDING_CHANGED")
    if (
        research["research_request_ref"] != record["research_request_ref"]
        or research["auxiliary_stage_refs"] != record["auxiliary_stage_refs"]
    ):
        raise ContractError("MATERIALIZATION_RESEARCH_BINDING_INVALID")
    return record


def _cutoff_path(execution, schema):
    if schema not in {SCHEMA_V5, SCHEMA_V6}:
        raise ContractError("MATERIALIZATION_CUTOFF_PROFILE_INVALID")
    return str(
        execution
        / ("research-cutoff.v2.json" if schema == SCHEMA_V6 else "research-cutoff.v1.json")
    )


def materialize_locked(
    *,
    journal: DailyJournal,
    recovered: dict,
    auxiliary: dict,
    _execute_theme: bool = False,
    _committed_cutoff_ref=None,
) -> MaterializedInputs:
    """Existing receipt bypasses plan creation/current financial preimages on recovery."""
    journal._require_lock()
    if type(_execute_theme) is not bool:
        raise ContractError("MATERIALIZATION_EXECUTE_THEME_MODE_INVALID")
    workspace = str(journal.storage._io.workspace_root)
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    if handoff["trade_date"] != journal.trade_date:
        raise ContractError("MATERIALIZATION_DAY_MISMATCH")
    execution = PurePosixPath(recovered["handoff_ref"]["path"]).parent
    if execution != journal.root / "executions" / handoff["request_ref"]["sha256"]:
        raise ContractError("MATERIALIZATION_EXECUTION_PATH_INVALID")
    schema, filename, _ = layout_for_recipe(recipe)
    path = str(execution / filename)
    if _committed_cutoff_ref is not None:
        _require_committed_cutoff(journal, execution, schema, _execute_theme, _committed_cutoff_ref)
    old = read_selected_materialization(journal=journal, execution=execution, recipe=recipe)
    if old is not None:
        if (
            _committed_cutoff_ref is not None
            and parse_canonical_json_bytes(old.data).get("cutoff_ref") != _committed_cutoff_ref
        ):
            raise ContractError("BOOTSTRAP_COMMITTED_CUTOFF_MISMATCH")
        return _recorded(journal, recovered, auxiliary, path, old)
    if (
        schema in {SCHEMA_V5, SCHEMA_V6}
        and journal.storage.read(_cutoff_path(execution, schema)) is None
    ):
        from quant_investor.operations.research_timing import assert_fresh_acquisition_open

        assert_fresh_acquisition_open(recipe)
    if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
        raise ContractError("MATERIALIZATION_EOD_ALREADY_COMPLETED")
    bootstrap = None
    if recipe["previous_completion_ref"] is None:
        bootstrap = bootstrap_evidence.verify_bootstrap_native(
            workspace=workspace, recovered=recovered
        )
        bootstrap_evidence.verify_bootstrap_initial_absence(journal=journal, proof=bootstrap)
    storage = SecureSystemStorage(workspace)
    if _execute_theme and recipe.get("theme_acquisition_ref") is not None:
        from quant_investor.operations.theme_handoff_publish import publish_theme_handoff

        publish_theme_handoff(
            journal=journal,
            request_ref=handoff["request_ref"],
            core_handoff_ref=handoff["core_handoff_ref"],
        )
    cutoff = None
    if schema in {SCHEMA_V5, SCHEMA_V6}:
        from quant_investor.operations.research_cutoff import (
            prepare_cutoff_inputs,
            read_cutoff_inputs,
        )
        from scripts.daily_store_materialization import store_materialization_arguments

        cutoff_path = _cutoff_path(execution, schema)
        retained = (
            _require_committed_cutoff(
                journal, execution, schema, _execute_theme, _committed_cutoff_ref
            )
            if _committed_cutoff_ref is not None
            else journal.storage.read(cutoff_path)
        )
        if retained is not None:
            cutoff = read_cutoff_inputs(
                journal=journal,
                cutoff_ref={"path": cutoff_path, "sha256": retained.byte_sha256},
                repair=True,
            )
            arguments = store_materialization_arguments(workspace, handoff, recipe)
            arguments["expected_store_pointer_sha"] = cutoff["native_plan"]["preimages"][
                "store_pointer_sha256"
            ]
            prepared = {
                "store_plan_ref": cutoff["store_plan_ref"],
                "native_plan": cutoff["native_plan"],
                "store_arguments": arguments,
                "retained_source_pointer_ref": None,
            }
        else:
            prepared = prepare_materialized_store_plan(journal=journal, recovered=recovered)
            cutoff = prepare_cutoff_inputs(
                journal=journal, recovered=recovered, auxiliary=auxiliary, prepared=prepared
            )
        research = {
            "research_request_ref": cutoff["research_request_ref"],
            "theme_source_handoff_ref": cutoff["source_bundle"]["theme_source_handoff_ref"],
            "auxiliary_stage_refs": cutoff["source_bundle"]["auxiliary_stage_refs"],
        }
    else:
        research = publish_research_input(journal=journal, recovered=recovered, auxiliary=auxiliary)
        prepared = prepare_materialized_store_plan(journal=journal, recovered=recovered)
    from quant_investor.operations.decision_recipe import (
        publish_decision_recipe,
        read_decision_recipe,
    )

    decision_recipe_ref = publish_decision_recipe(
        journal=journal,
        research_request_ref=research["research_request_ref"],
        store_plan_ref=prepared["store_plan_ref"],
        retained_pointer_ref=prepared.get("retained_source_pointer_ref"),
    )
    if bootstrap is not None:
        bootstrap_evidence.verify_bootstrap_store_plan(
            proof=bootstrap, plan=prepared["native_plan"], calendar_ref=handoff["calendar_ref"]
        )
    source_book = read_decision_recipe(
        workspace=workspace,
        trade_date=journal.trade_date,
        recipe_ref=decision_recipe_ref,
        research_request_ref=research["research_request_ref"],
        store_plan_ref=prepared["store_plan_ref"],
    )
    held_refs = materialized_adjustment_refs(
        journal=journal,
        recovered=recovered,
        prepared=prepared,
        portfolio_state=source_book["portfolio"],
    )
    core = parse_canonical_json_bytes(_read(storage, handoff["core_handoff_ref"]))
    calendar = parse_canonical_json_bytes(_read(storage, core["node_refs"]["calendar"]))
    previous = recipe["previous_completion_ref"]
    previous_day = (
        bootstrap["previous_trade_date"]
        if bootstrap is not None
        else PurePosixPath(previous["path"]).parent.name
    )
    document = {
        "schema_version": "cn-daily-native-inputs.v3",
        "trade_date": journal.trade_date,
        "factor_pointer_sha256": handoff["factor_pointer_ref"]["sha256"],
        "release_ref": handoff["release_ref"],
        "research_request_ref": research["research_request_ref"],
        "store_plan_ref": prepared["store_plan_ref"],
        "decision_recipe_ref": decision_recipe_ref,
        "store_policy_ref": recipe["policy_refs"]["store"],
        "retrospective_ref": recipe["retrospective_ref"],
        "calendar_ref": handoff["calendar_ref"],
        "market_snapshot_ref": handoff["market_snapshot_ref"],
        "benchmark_ref": recipe["dashboard_sources"]["benchmark_ref"],
        "risk_free_ref": recipe["dashboard_sources"]["risk_free_ref"],
        "previous_trade_date": previous_day,
        "adjustment_market_refs": held_refs,
        "publish_current_dashboard": recipe["publish_current_dashboard"],
        "next_session_calendar_proof_ref": calendar["output_refs"].get(
            "next_session_calendar_proof"
        ),
        "next_session_calendar_failure_ref": calendar["output_refs"].get(
            "next_session_calendar_failure"
        ),
    }
    if schema in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        document["schema_version"] = "cn-daily-native-inputs.v4"
        document["corporate_action_context_ref"] = (
            cutoff["corporate_action_context_ref"]
            if cutoff is not None
            else recipe["corporate_action_context_ref"]
        )
    if schema in {SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        document["schema_version"] = "cn-daily-native-inputs.v5"
        document["dashboard_publication_policy"] = recipe["dashboard_publication_policy"]
    if cutoff is not None:
        document["schema_version"] = "cn-daily-native-inputs.v6"
        document["cutoff_ref"] = cutoff["cutoff_ref"]
    if schema == SCHEMA_V6:
        document["schema_version"] = "cn-daily-native-inputs.v7"
        document["registered_event_declaration_ref"] = recipe["registered_event_declaration_ref"]
    raw = canonical_json_bytes(document)
    digest = hashlib.sha256(raw).hexdigest()
    native_path = str(execution / "inputs" / f"native-{digest}.json")
    native_input = journal.storage.write(native_path, raw)
    native_ref = {"path": native_path, "sha256": native_input.byte_sha256}
    day, inputs = load_native_inputs(workspace=workspace, input_ref=native_ref)
    if day != journal.trade_date:
        raise ContractError("MATERIALIZATION_NATIVE_DATE_INVALID")
    value = {
        "schema_version": schema,
        "trade_date": day,
        "graph_sha256": GRAPH_SHA256,
        "maintenance_handoff_ref": recovered["handoff_ref"],
        "research_request_ref": research["research_request_ref"],
        "store_plan_ref": prepared["store_plan_ref"],
        "native_inputs_ref": native_ref,
        "auxiliary_stage_refs": research["auxiliary_stage_refs"],
        "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": FALSE_AUTHORITY,
    }
    if schema in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        value["corporate_action_context_ref"] = document["corporate_action_context_ref"]
    if schema in {SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        value["dashboard_publication_policy"] = recipe["dashboard_publication_policy"]
    if cutoff is not None:
        value["cutoff_ref"] = cutoff["cutoff_ref"]
        value["corporate_action_template_ref"] = recipe["corporate_action_template_ref"]
    if schema == SCHEMA_V6:
        value["registered_event_declaration_ref"] = recipe["registered_event_declaration_ref"]
    if schema in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        value["theme_source_handoff_ref"] = research["theme_source_handoff_ref"]
    validate_bound_materialization(
        workspace=workspace, value=value, recipe=recipe, execution=execution, path=path
    )
    if utc_stamp(value["sealed_at"]) < utc_stamp(handoff["sealed_at"]):
        raise ContractError("MATERIALIZATION_CLOCK_BEFORE_HANDOFF")
    if bootstrap is not None:
        bootstrap_evidence.verify_bootstrap_initial_absence(journal=journal, proof=bootstrap)
    recorded = journal.storage.write(path, canonical_json_bytes(value))
    return MaterializedInputs(
        {"path": path, "sha256": recorded.byte_sha256}, native_ref, inputs, "MATERIALIZED"
    )


def _require_committed_cutoff(journal, execution, schema, execute_theme, reference):
    ref = validate_ref(reference)
    if (
        schema not in {SCHEMA_V5, SCHEMA_V6}
        or execute_theme
        or ref["path"] != _cutoff_path(execution, schema)
    ):
        raise ContractError("BOOTSTRAP_COMMITTED_CUTOFF_SCOPE_INVALID")
    stored = journal.storage.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise ContractError("BOOTSTRAP_COMMITTED_CUTOFF_MISMATCH")
    return stored


def materialize_daily_inputs(
    *, workspace: str, handoff_ref: dict, _execute_theme: bool = False
) -> dict:
    """Compose guarded input preparation only; no maintenance or native business run."""
    if type(_execute_theme) is not bool:
        raise ContractError("MATERIALIZATION_EXECUTE_THEME_MODE_INVALID")
    recovered = read_maintenance_handoff(workspace=workspace, handoff_ref=handoff_ref)
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    previous = recipe["previous_completion_ref"]
    if previous is None:
        bootstrap_evidence.verify_bootstrap_native(workspace=workspace, recovered=recovered)
    else:
        previous_day = PurePosixPath(previous["path"]).parent.name
        parent = replay_native_completion(
            workspace=workspace, trade_date=previous_day, completion_ref=previous
        )
        if (
            parent.get("native_replay_validated") is not True
            or parent.get("completion_ref") != previous
            or parent.get("trade_date") != previous_day
            or parent.get("validated_nodes") != sorted(EOD_NODE_IDS)
        ):
            raise ContractError("MATERIALIZATION_PREVIOUS_EOD_INVALID")
    run_root = str(PurePosixPath(handoff["logical_claim_ref"]["path"]).parents[2])
    auxiliary = read_auxiliary_stage_records(
        workspace=workspace, run_root=run_root, core_ref=handoff["maintenance_core_ref"]
    )
    journal = DailyJournal(workspace, handoff["trade_date"])
    with journal.locked():
        result = materialize_locked(
            journal=journal, recovered=recovered, auxiliary=auxiliary, _execute_theme=_execute_theme
        )
    return {
        "status": result.status,
        "trade_date": journal.trade_date,
        "materialization_ref": result.materialization_ref,
        "native_inputs_ref": result.native_inputs_ref,
        "execution_authorized": False,
    }


def plan_daily_recipe(*, workspace: str, request_ref: dict) -> dict:
    """Read declared inputs only; the target is planned, not Calendar/open-day admitted."""
    from quant_investor.operations.execution_controls import read_execution_controls
    from quant_investor.operations.production_result import SCHEMA, validate_production_result

    controls = read_execution_controls(workspace=workspace, request_ref=request_ref)
    request = controls.document(controls.request_ref)
    if request["action"] != "PLAN":
        raise ContractError("PRODUCTION_PLAN_ACTION_REQUIRED")
    day = request["target_trade_date"]
    controls.recheck()
    result = {
        "schema_version": SCHEMA,
        "action": "PLAN",
        "target_trade_date": day,
        "execution_state": "PLANNED",
        "business_state": "NOT_EVALUATED",
        "days": [
            {
                "trade_date": day,
                "execution_state": "PLANNED",
                "business_state": "NOT_EVALUATED",
                "completion_ref": None,
            }
        ],
        "authority": dict(FALSE_AUTHORITY),
    }
    return validate_production_result(
        result, action="PLAN", target_trade_date=day, expected_dates=[day]
    )
