"""Code-owned full native replay of one EOD; no Morning admission or publication."""

from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.completion_decision import replay_completed_decision
from quant_investor.operations.completion_corporate import replay_completed_corporate_actions
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from scripts.daily_completion_dashboard import replay_completed_dashboard


def replay_native_completion(
    *, workspace: str, trade_date: str, completion_ref: dict, loaded_inputs=None
) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    # Decision composes core7, four research sources, Macro and Decision (13).
    decision = replay_completed_decision(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    corporate = replay_completed_corporate_actions(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    # Dashboard composes the native committed Store proof and native pair (2).
    dashboard = replay_completed_dashboard(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_NATIVE_COMPLETION_CHANGED_DURING_REPLAY")
    ledger = None
    if recorded["schema_version"] == "cn-daily-eod-completion.v2":
        from scripts.daily_ledger_replay import replay_completed_ledger

        ledger = replay_completed_ledger(
            workspace=workspace,
            trade_date=trade_date,
            completion_ref=completion_ref,
            loaded_inputs=loaded_inputs,
        )
    result = {
        "schema_version": "cn-daily-eod-native-readback.v1",
        "completion_ref": dict(completion_ref),
        "trade_date": trade_date,
        "native_replay_validated": True,
        "validated_nodes": sorted(EOD_NODE_IDS),
        "synthetic": recorded["synthetic"],
        "decision": decision["result"],
        "corporate_action": corporate["projection"],
        "dashboard": {"v1": dashboard["v1"], "v2": dashboard["v2"]},
        "consumer_admission": False,
        "authority": recorded["authority"],
    }
    if ledger is not None:
        result["ledger"] = ledger
    if "daily_evidence" in dashboard:
        result["dashboard"]["daily_evidence"] = dashboard["daily_evidence"]
    return result


def verify_initial_previous_completion(*, workspace: str, recipe: dict) -> dict:
    """Replay the exact preceding EOD before maintenance; Calendar adjacency follows later."""
    from pathlib import Path, PurePosixPath
    from quant_investor.cli.unified import _daily_source_file
    from quant_investor.contracts import parse_canonical_json_bytes
    from quant_investor.operations.daily_contract import validate_ref
    from quant_investor.operations.daily_journal import _validate_day
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.factors.production_authority import (
        FactorProductionStore,
        FACTOR_ACTIVE_POINTER_PATH,
        FACTOR_PRODUCTION_MARKER_PATH,
        FACTOR_AUTHORITY_ACTIVE,
    )

    ref = validate_ref(recipe["previous_completion_ref"])
    previous = PurePosixPath(ref["path"]).parent.name
    _validate_day(previous)
    if previous >= recipe["target_trade_date"] or recipe["bootstrap_ref"] is not None:
        raise ContractError("EXECUTION_PREVIOUS_ANCHOR_INVALID")
    replay = replay_native_completion(workspace=workspace, trade_date=previous, completion_ref=ref)
    if (
        replay["native_replay_validated"] is not True
        or set(replay["validated_nodes"]) != EOD_NODE_IDS
    ):
        raise ContractError("EXECUTION_PREVIOUS_NATIVE_REPLAY_INCOMPLETE")
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=previous, completion_ref=ref
    )["recorded_completion"]
    _, raw, _ = _daily_source_file(
        Path(workspace).resolve(strict=True),
        recorded["native_inputs_ref"],
        code="EXECUTION_PREVIOUS_INPUTS_CHANGED",
    )
    inputs = parse_canonical_json_bytes(raw)
    store = FactorProductionStore(workspace)
    pointer = store.read(FACTOR_ACTIVE_POINTER_PATH)
    marker = store.read(FACTOR_PRODUCTION_MARKER_PATH)
    verified = store.verify_active()
    if (
        pointer is None
        or marker is None
        or pointer.byte_sha256 != inputs["factor_pointer_sha256"]
        or verified.get("factor_pointer_byte_sha256") != pointer.byte_sha256
        or verified.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE
        or verified.get("as_of") != previous
    ):
        raise ContractError("EXECUTION_PREVIOUS_FACTOR_BASELINE_MISMATCH")
    status = read_daily_status(workspace, previous)
    output = status["nodes"]["store"]["terminal"]["output_refs"]["pointer"]
    if recipe.get("schema_version") == "cn-daily-execute-recipe.v6":
        from scripts.registered_daily_event_sources import read_recipe_source

        proof = read_recipe_source(workspace=workspace, recipe=recipe)
        matches = output == proof["declaration"]["baseline_store_pointer_ref"]
    else:
        matches = output["sha256"] == recipe["store_preimages"]["store_pointer_ref"]["sha256"]
    if not matches:
        raise ContractError("EXECUTION_PREVIOUS_STORE_BASELINE_MISMATCH")
    if (
        store.read(FACTOR_ACTIVE_POINTER_PATH) != pointer
        or store.read(FACTOR_PRODUCTION_MARKER_PATH) != marker
    ):
        raise ContractError("EXECUTION_PREVIOUS_FACTOR_CHANGED")
    return {"previous_trade_date": previous, "completion_ref": ref, "execution_authorized": False}
