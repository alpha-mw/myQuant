"""Native EOD convergence seal. This file cannot create business outputs.

All sixteen native adapters must replay the exact selected journal terminals before
publication. Availability uses the actual verification clock, never the trade date.
Prospective eligibility is deliberately not inferred by this closure.
"""

from datetime import datetime, timezone

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import (
    ContractError,
    EOD_NODE_IDS,
    GRAPH,
    GRAPH_SHA256,
    NodeState,
    utc_stamp,
    validate_ref,
)
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from scripts.daily_native_inputs import load_native_inputs


def _replay(registry) -> dict:
    journal = registry.runner.journal
    journal._require_lock()
    if set(registry.adapters) != EOD_NODE_IDS or set(registry.templates) != EOD_NODE_IDS:
        raise ContractError("EOD_NATIVE_NODE_SET_INCOMPLETE")
    completed = {}
    for spec in GRAPH:
        if not spec.eod_required:
            continue
        request = registry.runner._request(
            registry.templates[spec.node_id], spec.node_id, completed
        )
        if request["release_ref"] != dict(registry.inputs.release_ref):
            raise ContractError("EOD_RELEASE_BINDING_MISMATCH")
        selected = journal.inspect(request)
        if selected["state"] != "SUCCEEDED":
            raise ContractError("EOD_NODE_NOT_SUCCEEDED:" + spec.node_id)
        probe = registry.adapters[spec.node_id].probe(request)
        outcome = probe.outcome
        if (
            probe.recovery_only
            or outcome is None
            or outcome.state != NodeState.SUCCEEDED
            or outcome.failure_code is not None
            or outcome.output_refs != selected["terminal"]["output_refs"]
        ):
            raise ContractError("EOD_NATIVE_REPLAY_FAILED:" + spec.node_id)
        if journal.inspect(request) != selected:
            raise ContractError("EOD_JOURNAL_CHANGED_DURING_REPLAY")
        completed[spec.node_id] = selected
    refs = {node: row["terminal_ref"] for node, row in completed.items()}
    from quant_investor.operations.daily_timing import recorded_daily_timing

    recorded_daily_timing(
        trade_date=registry.trade_date,
        nodes=completed,
        terminal_refs=refs,
        verified_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return refs


def _validate_native_context(registry, *, native_inputs_ref: dict, synthetic: bool) -> None:
    """Require the existing native adapter set and caller-owned day lock."""
    from scripts.daily_native_registry import NativeDailyRegistry

    if type(registry) is not NativeDailyRegistry or type(synthetic) is not bool:
        raise ContractError("EOD_CONTEXT_INVALID")
    registry.runner.journal._require_lock()
    validate_ref(native_inputs_ref)
    # Fail early before loading context or touching any completion file.
    if set(registry.adapters) != EOD_NODE_IDS or set(registry.templates) != EOD_NODE_IDS:
        raise ContractError("EOD_NATIVE_NODE_SET_INCOMPLETE")
    from quant_investor.operations.core_pool import (
        CORE_NODES,
        CoreEvidenceAdapter,
        NativePoolAdapter,
    )
    from quant_investor.operations.research_sources import SOURCE_NODES, ResearchSourceAdapter
    from quant_investor.operations.research_decision import ResearchDecisionAdapter
    from quant_investor.operations.corporate_actions import CorporateActionAdapter
    from quant_investor.operations.corporate_adapter import CorporateReconciliationAdapter
    from scripts.daily_production_store_adapter import StoreCloseAdapter
    from scripts.daily_dashboard_adapter import CurrentDashboardAdapter, HistoricalDashboardAdapter
    from scripts.daily_dashboard_sealed import SealedDashboardAdapter

    expected: dict[str, type] = {node: CoreEvidenceAdapter for node in CORE_NODES}
    expected["top100"] = NativePoolAdapter
    expected.update({node: ResearchSourceAdapter for node in SOURCE_NODES})
    expected.update(
        {
            "decision": ResearchDecisionAdapter,
            "corporate_action_recon": (
                CorporateReconciliationAdapter
                if getattr(registry.inputs, "corporate_action_context_ref", None) is not None
                else CorporateActionAdapter
            ),
            "store": StoreCloseAdapter,
            "dashboard": (
                SealedDashboardAdapter
                if getattr(registry.inputs, "dashboard_publication_policy", None) is not None
                else (
                    CurrentDashboardAdapter
                    if registry.inputs.publish_current_dashboard
                    else HistoricalDashboardAdapter
                )
            ),
        }
    )
    if any(type(registry.adapters[node]) is not expected[node] for node in EOD_NODE_IDS):
        raise ContractError("EOD_NATIVE_ADAPTER_TYPE_INVALID")


def seal_native_completion(registry, *, native_inputs_ref: dict, synthetic: bool) -> dict:
    """Standalone sealer loads and validates one native context under caller's lock."""
    _validate_native_context(registry, native_inputs_ref=native_inputs_ref, synthetic=synthetic)
    day, inputs = load_native_inputs(workspace=registry.workspace, input_ref=native_inputs_ref)
    if day != registry.trade_date or inputs != registry.inputs:
        raise ContractError("EOD_INPUT_CONTEXT_MISMATCH")
    return _seal_loaded_completion(
        registry, native_inputs_ref=native_inputs_ref, synthetic=synthetic
    )


def _seal_loaded_completion(registry, *, native_inputs_ref: dict, synthetic: bool) -> dict:
    """Historical completion replay only; new closures require materialized v2."""
    from scripts.daily_completion_replay import replay_native_completion
    from scripts.daily_native_inputs import verify_loaded_native_inputs

    journal = registry.runner.journal
    journal._require_lock()
    path = str(journal.root / "completion.v1.json")
    stored = journal.storage.read(path)
    if stored is None:
        raise ContractError("EOD_MATERIALIZATION_REQUIRED")
    verify_loaded_native_inputs(
        workspace=registry.workspace,
        input_ref=native_inputs_ref,
        trade_date=registry.trade_date,
        inputs=registry.inputs,
    )
    value = parse_canonical_json_bytes(stored.data)
    if (
        value.get("native_inputs_ref") != native_inputs_ref
        or type(synthetic) is not bool
        or value.get("synthetic") is not synthetic
    ):
        raise ContractError("EOD_EXISTING_COMPLETION_CONFLICT")
    ref = {"path": path, "sha256": stored.byte_sha256}
    result = replay_native_completion(
        workspace=registry.workspace,
        trade_date=registry.trade_date,
        completion_ref=ref,
        loaded_inputs=registry.inputs,
    )
    if (
        result.get("native_replay_validated") is not True
        or result.get("completion_ref") != ref
        or result.get("trade_date") != registry.trade_date
        or result.get("validated_nodes") != sorted(EOD_NODE_IDS)
    ):
        raise ContractError("EOD_NATIVE_REPLAY_FAILED")
    return ref


def run_and_seal_native_input(
    *, workspace: str, input_ref: dict, resume: bool = False, synthetic: bool
) -> dict:
    """Compatibility Python entry routed exclusively through materialized v2 work."""
    return run_materialized_native_input(
        workspace=workspace, input_ref=input_ref, resume=resume, synthetic=synthetic
    )


def run_and_seal_loaded_native_input(
    registry, *, input_ref: dict, resume: bool, synthetic: bool
) -> dict:
    """Legacy loaded entry is historical replay only and cannot run new business work."""
    from quant_investor.operations.daily_status import read_daily_status

    if type(resume) is not bool or type(synthetic) is not bool:
        raise ContractError("EOD_EXECUTION_CONTEXT_INVALID")
    ref = _seal_loaded_completion(registry, native_inputs_ref=input_ref, synthetic=synthetic)
    return {
        **read_daily_status(registry.workspace, registry.trade_date),
        "status": "COMPLETE",
        "completion_ref": ref,
        "completion_status": "NATIVE_COMPLETION_VALIDATED",
    }


def seal_materialized_completion(registry, *, materialized, synthetic: bool) -> dict:
    """Seal v2 after native ledger publication, or fully replay an existing v2.

    Caller retains its day lock. Existing legacy completion bytes are never upgraded.
    """
    from scripts.daily_ledger import publish_native_ledger
    from scripts.daily_completion_replay import replay_native_completion
    from scripts.daily_native_inputs import verify_loaded_native_inputs

    journal = registry.runner.journal
    journal._require_lock()
    if type(synthetic) is not bool:
        raise ContractError("EOD_EXECUTION_CONTEXT_INVALID")
    verify_loaded_native_inputs(
        workspace=registry.workspace,
        input_ref=materialized.native_inputs_ref,
        trade_date=registry.trade_date,
        inputs=registry.inputs,
    )
    if registry.inputs != materialized.inputs:
        raise ContractError("EOD_INPUT_CONTEXT_MISMATCH")
    path = str(journal.root / "completion.v1.json")
    old = journal.storage.read(path)
    if old is not None:
        recorded = parse_canonical_json_bytes(old.data)
        if recorded.get("schema_version") == "cn-daily-eod-completion.v1":
            raise ContractError("EOD_LEGACY_PATH_OCCUPIED")
        if (
            recorded.get("schema_version") != "cn-daily-eod-completion.v2"
            or recorded.get("native_inputs_ref") != materialized.native_inputs_ref
            or recorded.get("materialization_ref") != materialized.materialization_ref
            or type(recorded.get("synthetic")) is not bool
            or recorded["synthetic"] is not synthetic
        ):
            raise ContractError("EOD_EXISTING_COMPLETION_CONFLICT")
        ref = {"path": path, "sha256": old.byte_sha256}
        ledger_ref = recorded["prospective_ledger_ref"]
    else:
        published = publish_native_ledger(registry, materialized=materialized, synthetic=synthetic)
        ledger = published["ledger"]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if utc_stamp(now) < utc_stamp(ledger["published_at"]):
            raise ContractError("EOD_CLOCK_BEFORE_LEDGER")
        document = {
            "schema_version": "cn-daily-eod-completion.v2",
            "status": "SUCCEEDED",
            "market": "CN",
            "strategy_id": "aggressive_tech_manufacturing",
            "trade_date": registry.trade_date,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": dict(registry.inputs.release_ref),
            "native_inputs_ref": materialized.native_inputs_ref,
            "node_terminal_refs": ledger["node_terminal_refs"],
            "synthetic": synthetic,
            "prospective_admission_state": (
                "LEDGER_ELIGIBLE" if ledger["prospective"] else "LEDGER_INELIGIBLE"
            ),
            "authority": FALSE_AUTHORITY,
            "native_validation_completed_at": now,
            "materialization_ref": materialized.materialization_ref,
            "prospective_ledger_ref": published["ledger_ref"],
        }
        stored = journal.storage.write(path, canonical_json_bytes(document))
        ref = {"path": path, "sha256": stored.byte_sha256}
        ledger_ref = published["ledger_ref"]
    replay = replay_native_completion(
        workspace=registry.workspace,
        trade_date=registry.trade_date,
        completion_ref=ref,
        loaded_inputs=registry.inputs,
    )
    if (
        replay.get("native_replay_validated") is not True
        or replay.get("completion_ref") != ref
        or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
        or replay.get("trade_date") != registry.trade_date
        or replay.get("synthetic") is not synthetic
        or replay.get("ledger", {}).get("ledger_ref") != ledger_ref
    ):
        raise ContractError("EOD_V2_NATIVE_REPLAY_FAILED")
    return ref


def run_and_seal_materialized_input(
    registry, *, materialized, resume: bool, synthetic: bool
) -> dict:
    """Continue the materializer's loaded context through native execution and v2 seal."""
    from scripts.daily_native_inputs import run_loaded_native_input
    from scripts.daily_dashboard_publication import complete_serving_result
    from quant_investor.operations.daily_status import read_daily_status

    journal = registry.runner.journal
    journal._require_lock()
    if type(resume) is not bool or type(synthetic) is not bool:
        raise ContractError("EOD_EXECUTION_CONTEXT_INVALID")
    if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
        ref = seal_materialized_completion(registry, materialized=materialized, synthetic=synthetic)
        return complete_serving_result(
            workspace=registry.workspace,
            completion_ref=ref,
            journal=journal,
            base=read_daily_status(registry.workspace, registry.trade_date),
        )
    result = run_loaded_native_input(
        registry, input_ref=materialized.native_inputs_ref, resume=resume
    )
    if set(result["nodes"]) != EOD_NODE_IDS or any(
        row["state"] != "SUCCEEDED" for row in result["nodes"].values()
    ):
        return result
    ref = seal_materialized_completion(registry, materialized=materialized, synthetic=synthetic)
    sealed = {
        **result,
        "status": "COMPLETE",
        "completion_ref": ref,
        "completion_status": "NATIVE_COMPLETION_VALIDATED",
    }
    sealed = complete_serving_result(
        workspace=registry.workspace, completion_ref=ref, journal=journal, base=sealed
    )
    journal.storage.write(
        str(journal.root / "dag-status.v1.json"), canonical_json_bytes(sealed), projection=True
    )
    return sealed


def run_materialized_native_input(
    *, workspace: str, input_ref: dict, resume: bool = False, synthetic: bool
) -> dict:
    """Exact-input installed/catch-up entry; new work requires a sealed materialization."""
    from pathlib import PurePosixPath
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
    from scripts.daily_materialization import MaterializedInputs, verify_materialized_inputs
    from scripts.daily_native_registry import NativeDailyRegistry
    from scripts.daily_completion_replay import replay_native_completion

    if type(resume) is not bool or type(synthetic) is not bool:
        raise ContractError("EOD_EXECUTION_CONTEXT_INVALID")
    ref = validate_ref(input_ref)
    day, inputs = load_native_inputs(workspace=workspace, input_ref=ref)
    journal = DailyJournal(workspace, day)
    completion_path = str(journal.root / "completion.v1.json")
    completed = journal.storage.read(completion_path)
    if completed is not None:
        value = parse_canonical_json_bytes(completed.data)
        if value.get("native_inputs_ref") != ref or value.get("synthetic") is not synthetic:
            raise ContractError("EOD_EXISTING_COMPLETION_CONFLICT")
        completion_ref = {"path": completion_path, "sha256": completed.byte_sha256}
        checked = replay_native_completion(
            workspace=workspace, trade_date=day, completion_ref=completion_ref, loaded_inputs=inputs
        )
        if (
            checked.get("native_replay_validated") is not True
            or checked.get("validated_nodes") != sorted(EOD_NODE_IDS)
            or checked.get("completion_ref") != completion_ref
            or checked.get("trade_date") != day
            or checked.get("synthetic") is not synthetic
        ):
            raise ContractError("EOD_NATIVE_REPLAY_FAILED")
        return {
            **read_daily_status(workspace, day),
            "status": "COMPLETE",
            "completion_ref": completion_ref,
            "completion_status": "NATIVE_COMPLETION_VALIDATED",
        }
    native_path = PurePosixPath(ref["path"])
    execution = native_path.parent.parent
    if (
        native_path.name != f"native-{ref['sha256']}.json"
        or native_path.parent.name != "inputs"
        or execution.parent != journal.root / "executions"
        or len(execution.name) != 64
        or any(c not in "0123456789abcdef" for c in execution.name)
    ):
        raise ContractError("EOD_MATERIALIZATION_REQUIRED")
    from quant_investor.operations.materialization_contract import (
        layout_for_recipe,
        read_selected_materialization,
        validate_bound_materialization,
    )

    handoff_path = str(execution / "maintenance-handoff.v1.json")
    retained_handoff = journal.storage.read(handoff_path)
    if retained_handoff is None:
        raise ContractError("EOD_MATERIALIZATION_REQUIRED")
    recovered = read_maintenance_handoff(
        workspace=workspace,
        handoff_ref={"path": handoff_path, "sha256": retained_handoff.byte_sha256},
    )
    _, filename, _ = layout_for_recipe(recovered["recipe"])
    path = str(execution / filename)
    stored = read_selected_materialization(
        journal=journal, execution=execution, recipe=recovered["recipe"]
    )
    if stored is None:
        raise ContractError("EOD_MATERIALIZATION_REQUIRED")
    record = parse_canonical_json_bytes(stored.data)
    validate_bound_materialization(
        workspace=workspace,
        value=record,
        recipe=recovered["recipe"],
        execution=execution,
        path=path,
    )
    if record["maintenance_handoff_ref"] != recovered["handoff_ref"]:
        raise ContractError("EOD_MATERIALIZATION_HANDOFF_MISMATCH")
    previous = recovered["recipe"]["previous_completion_ref"]
    if previous is None:
        from quant_investor.operations.bootstrap import verify_bootstrap_native

        verify_bootstrap_native(workspace=workspace, recovered=recovered)
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
    materialized = MaterializedInputs(
        {"path": path, "sha256": stored.byte_sha256}, ref, inputs, "NO_ACTION"
    )
    with journal.locked():
        verify_materialized_inputs(journal=journal, recovered=recovered, materialized=materialized)
        registry = NativeDailyRegistry(workspace, day, inputs, journal=journal)
        return run_and_seal_materialized_input(
            registry, materialized=materialized, resume=resume, synthetic=synthetic
        )
