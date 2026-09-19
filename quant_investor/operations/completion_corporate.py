"""Read-only completed corporate-action evidence replay."""

from pathlib import PurePosixPath
import json
from quant_investor.strategy_records.store import canonical_json_bytes as store_json_bytes
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from .completion_readback import inspect_recorded_completion
from .corporate_actions import CorporateActionEvidence
from .daily_contract import ContractError, validate_ref
from .daily_journal import request_identity


def replay_completed_corporate_actions(
    *, workspace: str, trade_date: str, completion_ref: dict
) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    storage = SecureSystemStorage(workspace)
    seen = {}

    def read(ref, *, native_store=False):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=8 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_CORPORATE_SOURCE_SHA_MISMATCH")
        seen[ref["path"]] = raw.data
        if native_store:
            value = json.loads(raw.data)
            if store_json_bytes(value) != raw.data:
                raise ContractError("EOD_CORPORATE_NATIVE_STORE_ENCODING_INVALID")
            return value
        return parse_canonical_json_bytes(raw.data)

    inputs = read(recorded["native_inputs_ref"])
    plan = read(inputs["store_plan_ref"], native_store=True)
    terminal_ref = recorded["node_terminal_refs"]["corporate_action_recon"]
    terminal = read(terminal_ref)
    path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
    raw = storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
    seen[path] = raw.data
    request = parse_canonical_json_bytes(raw.data)
    if request_identity(request)[1] != terminal["request_key"]:
        raise ContractError("EOD_CORPORATE_REQUEST_INVALID")
    recipe = read(request["input_refs"]["recipe"])
    expected_recipe = {
        "event_pointer_ref": recipe["event_pointer_ref"],
        "previous_trade_date": inputs["previous_trade_date"],
        "market_refs": inputs["adjustment_market_refs"],
        "calendar_ref": inputs["calendar_ref"],
        "market_snapshot_ref": inputs["market_snapshot_ref"],
    }
    versioned = inputs["schema_version"] in {
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }
    if versioned:
        from .corporate_adapter import SCHEMA, SCHEMA_V3, derive_corporate_projection
        from .daily_contract import utc_stamp

        expected_recipe.update(
            schema_version=(
                SCHEMA_V3 if inputs["schema_version"] == "cn-daily-native-inputs.v7" else SCHEMA
            ),
            custody_at=recipe.get("custody_at"),
            **{
                k: inputs[k]
                for k in (
                    "corporate_action_context_ref",
                    "decision_recipe_ref",
                    "research_request_ref",
                    "store_plan_ref",
                )
            },
        )
        if inputs["schema_version"] == "cn-daily-native-inputs.v7":
            expected_recipe["registered_event_declaration_ref"] = inputs[
                "registered_event_declaration_ref"
            ]
        if utc_stamp(recipe.get("custody_at")) > utc_stamp(terminal["finished_at"]):
            raise ContractError("EOD_CORPORATE_CUSTODY_AFTER_TERMINAL")
    if (
        recipe != expected_recipe
        or recipe["event_pointer_ref"]["sha256"] != plan["preimages"]["event_pointer_sha256"]
    ):
        raise ContractError("EOD_CORPORATE_RECIPE_BINDING_INVALID")
    read(recipe["event_pointer_ref"], native_store=True)
    evidence = CorporateActionEvidence(workspace=workspace, trade_date=trade_date, recipe=recipe)
    extra_outputs = {}
    if versioned:
        derived = derive_corporate_projection(evidence=evidence, recipe=recipe)
        if derived is None:
            raise ContractError("EOD_CORPORATE_EVENT_MISSING")
        rebuilt, report, report_ref = derived[:3]
        if read(report_ref) != report:
            raise ContractError("EOD_CORPORATE_RECONCILIATION_DIFFERS")
        extra_outputs["reconciliation"] = report_ref
        if inputs["schema_version"] == "cn-daily-native-inputs.v7":
            transition, transition_ref = derived[3:]
            if read(transition_ref) != transition:
                raise ContractError("EOD_REGISTERED_TRANSITION_DIFFERS")
            extra_outputs["registered_transition"] = transition_ref
    else:
        rebuilt = evidence.project()
    if rebuilt is None or canonical_json_bytes(rebuilt) != canonical_json_bytes(
        read(terminal["output_refs"]["financial_events"])
    ):
        raise ContractError("EOD_CORPORATE_NATIVE_REPLAY_DIFFERS")
    if terminal["output_refs"] != {
        "financial_events": terminal["output_refs"]["financial_events"],
        "event_generation": rebuilt["event_generation_ref"],
        **extra_outputs,
    }:
        raise ContractError("EOD_CORPORATE_OUTPUT_BINDING_INVALID")
    for path, raw in seen.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024).data != raw:
            raise ContractError("EOD_CORPORATE_SOURCE_CHANGED_DURING_REPLAY")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_CORPORATE_COMPLETION_CHANGED_DURING_REPLAY")
    return {
        "completion_ref": dict(completion_ref),
        "projection": rebuilt,
        "validation_scope": "COMPLETED_CORPORATE_ACTION_NATIVE_REPLAY",
        "consumer_admission": False,
    }
