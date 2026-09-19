"""Native core replay from one completed EOD without current-head substitution."""

from pathlib import PurePosixPath

from quant_investor.contracts import parse_canonical_json_bytes, validate_artifact
from quant_investor.system.store import object_ref_for_artifact
from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.intelligence import build_factor_research_rank
from quant_investor.intelligence.storage import DailyResearchPoolStore
from quant_investor.system.storage import SecureSystemStorage
from .completion_readback import inspect_recorded_completion
from .core_pool import CORE_NODES, validate_core_observation
from .daily_contract import ContractError, validate_ref
from .daily_journal import request_identity


def replay_completed_core(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    storage = SecureSystemStorage(workspace)
    observed = {}

    def read_bytes(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_CORE_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = raw.data
        return raw.data

    def read(ref):
        return parse_canonical_json_bytes(read_bytes(ref), label="recorded core")

    terminals, requests = {}, {}
    pointer_ref = None
    for node in CORE_NODES:
        terminal_ref = recorded["node_terminal_refs"][node]
        terminal = read(terminal_ref)
        path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
        raw = storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
        observed[path] = raw.data
        req = parse_canonical_json_bytes(raw.data)
        if request_identity(req)[1] != terminal["request_key"]:
            raise ContractError("EOD_CORE_REQUEST_KEY_MISMATCH")
        selected = req["input_refs"]["factor_pointer"]
        if pointer_ref is not None and pointer_ref != selected:
            raise ContractError("EOD_CORE_POINTER_BINDING_MISMATCH")
        pointer_ref = selected
        terminals[node], requests[node] = terminal, req
    if pointer_ref is None:
        raise ContractError("EOD_CORE_POINTER_MISSING")
    inputs = read(recorded["native_inputs_ref"])
    from .native_input_contract import validate_native_input_shape

    validate_native_input_shape(inputs)
    if inputs["factor_pointer_sha256"] != pointer_ref["sha256"]:
        raise ContractError("EOD_CORE_INPUT_POINTER_MISMATCH")
    pointer = read(pointer_ref)
    snapshot = FactorProductionStore(workspace).inspect_recorded_research_inputs(
        pointer_raw=observed[pointer_ref["path"]],
        expected_pointer_sha256=pointer_ref["sha256"],
        expected_trade_date=trade_date,
    )
    generation = snapshot["factor_generation"]["payload"]
    if (
        object_ref_for_artifact(validate_artifact(read(recorded["release_ref"])))
        != generation["deployed_release_ref"]
    ):
        raise ContractError("EOD_CORE_RELEASE_BINDING_MISMATCH")
    expected = {}
    for node, keys in {
        "calendar": ("calendar_compilation_ref", "calendar_capture_custody_attestation_ref"),
        "pit": ("market_pit_selection_ref",),
        "market": ("market_input_ref",),
    }.items():
        expected[node] = {
            key: {
                "path": (
                    f"results/factors/objects/{generation[key]['kind']}/"
                    f"{generation[key]['byte_sha256']}.json"
                ),
                "sha256": generation[key]["byte_sha256"],
            }
            for key in keys
        }
    from .future_calendar_binding import future_calendar_outputs

    future = future_calendar_outputs(
        workspace=workspace,
        trade_date=trade_date,
        proof_ref=inputs.get("next_session_calendar_proof_ref"),
        failure_ref=inputs.get("next_session_calendar_failure_ref"),
        factor_snapshot=snapshot,
        finished_at=terminals["calendar"]["finished_at"],
    )
    declared_future = {
        k: v
        for k, v in requests["calendar"]["input_refs"].items()
        if k.startswith("next_session_calendar_")
    }
    if declared_future != future:
        raise ContractError("EOD_CORE_FUTURE_CALENDAR_BINDING_MISMATCH")
    expected["calendar"].update(future)
    expected["factor"] = {
        "generation": {
            "path": (
                f"results/factors/generations/{snapshot['factor_generation_id']}/" "generation.json"
            ),
            "sha256": snapshot["factor_generation_sha256"],
        }
    }
    observations = []
    for node, alias in [("low_observation", "LOW"), ("w80_observation", "W80")]:
        ref = terminals[node]["output_refs"][alias]
        if ref["path"] != (
            f"results/factors/observations/{trade_date[:4]}/{trade_date[4:6]}/"
            f"{trade_date[6:]}/{alias}.json"
        ):
            raise ContractError("EOD_CORE_OBSERVATION_PATH_INVALID")
        document = read(ref)
        validate_core_observation(document, alias=alias, snapshot=snapshot)
        observations.append(document)
        expected[node] = {alias: ref}
    policy_ref = requests["top100"]["policy_refs"]["research"]
    policy = read(policy_ref)
    rank = build_factor_research_rank(
        snapshot=snapshot,
        observations=observations,
        policy=policy,
        as_of=f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T07:00:00Z",
        created_at=max(
            snapshot["factor_generation"]["created_at"], *(o["created_at"] for o in observations)
        ),
    )
    expected["top100"] = DailyResearchPoolStore(workspace).verify(
        rank=rank,
        observations=observations,
        expected_policy_sha256=policy_ref["sha256"],
        policy_path=policy_ref["path"],
    )
    for node, outputs in expected.items():
        if terminals[node]["output_refs"] != outputs:
            raise ContractError("EOD_CORE_OUTPUT_BINDING_MISMATCH:" + node)
        for name, ref in outputs.items():
            if node == "top100" and name == "top100.parquet":
                read_bytes(ref)
            else:
                read(ref)
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
            raise ContractError("EOD_CORE_SOURCE_CHANGED_DURING_REPLAY")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_CORE_COMPLETION_CHANGED_DURING_REPLAY")
    from .core_timing import recorded_core_timing

    timing = recorded_core_timing(
        pointer=pointer,
        pointer_ref=pointer_ref,
        generation=snapshot["factor_generation"],
        observations=observations,
        observation_refs={
            "LOW": expected["low_observation"]["LOW"],
            "W80": expected["w80_observation"]["W80"],
        },
        terminal_refs={node: recorded["node_terminal_refs"][node] for node in CORE_NODES},
        terminals=terminals,
    )
    return {
        "completion_ref": dict(completion_ref),
        "recorded_timing": timing,
        "snapshot": snapshot,
        "rank": rank,
        "observations": observations,
        "validation_scope": "COMPLETED_CORE_NATIVE_REPLAY",
        "consumer_admission": False,
    }
