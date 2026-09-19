"""Deterministic replay of an existing Decision from completed immutable evidence."""

from pathlib import Path
from pathlib import PurePosixPath
from quant_investor.cli import unified
from quant_investor.contracts import parse_canonical_json_bytes, canonical_json_bytes
from quant_investor.intelligence import build_factor_research_rank
from quant_investor.intelligence.daily import compile_daily_intelligence
from quant_investor.system.storage import SecureSystemStorage
from .completion_readback import inspect_recorded_completion
from .completion_core import replay_completed_core
from .completion_macro import replay_completed_macro
from .completion_research import replay_completed_research_sources
from .daily_contract import ContractError, validate_ref
from .daily_journal import DailyJournal, request_identity
from .research_capture import ResearchCapture
from quant_investor.intelligence.theme_sources import split_theme_source
from quant_investor.intelligence.pcb_ai_hardware import partition_exposure_evidence
from .native_input_contract import validate_native_input_shape


def replay_completed_decision(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    storage = SecureSystemStorage(workspace)
    observed = {}

    from .research_file_readback import ResearchFileReadback

    files = ResearchFileReadback(workspace)
    source_file = files.source_file

    def read(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=64 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_DECISION_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = raw.data
        return parse_canonical_json_bytes(raw.data)

    inputs = read(recorded["native_inputs_ref"])
    validate_native_input_shape(inputs)
    request_ref = inputs["research_request_ref"]
    read(request_ref)
    from .research_request import load_research_request

    loaded = load_research_request(workspace=workspace, reference=request_ref)
    request = loaded["document"]
    read(loaded["native_request_ref"])
    core = replay_completed_core(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    sources = replay_completed_research_sources(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["artifacts"]
    macro = replay_completed_macro(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    rank = build_factor_research_rank(
        snapshot=core["snapshot"],
        observations=core["observations"],
        policy=request["policy"],
        as_of=request["as_of"],
    )
    frame, fundamental = unified._daily_fundamental_source(
        request["company_evidence"],
        source_file,
        workspace=Path(workspace),
        decision_as_of=request["as_of"],
    )
    exposure = sources["exposure"][1:]
    if split_theme_source(request["theme_source"])[2]:
        exposure, _ = partition_exposure_evidence(
            [value for value in exposure if value["kind"] == "company_source_evidence"],
            [row["symbol"] for row in rank["payload"]["pool_rows"]],
        )
    rebuilt = compile_daily_intelligence(
        as_of=request["as_of"],
        strategy_id=request["strategy_id"],
        policy=request["policy"],
        rank=rank,
        industry_projection=sources["industry"][0],
        theme_projection=sources["theme"][0],
        exposure_evidence=exposure,
        fundamental_frame=frame,
        fundamental_source=fundamental,
        market_risk_evidence=macro["macro_artifact"],
    )
    captured = ResearchCapture(DailyJournal(workspace, trade_date)).read(request_ref)
    if captured is None or canonical_json_bytes(captured[1]) != canonical_json_bytes(rebuilt):
        raise ContractError("EOD_DECISION_NATIVE_RECOMPILE_DIFFERS")
    terminal = read(recorded["node_terminal_refs"]["decision"])
    if terminal["output_refs"]["result"] != captured[0]["result_ref"]:
        raise ContractError("EOD_DECISION_RESULT_BINDING_INVALID")
    report = None
    node_path = str(
        PurePosixPath(recorded["node_terminal_refs"]["decision"]["path"]).parent.parent
        / "request.json"
    )
    node_raw = storage.read_workspace_file_bytes(node_path, maximum_bytes=8 * 1024 * 1024)
    node_request = read({"path": node_path, "sha256": node_raw.byte_sha256})
    capture_path = (
        ResearchCapture(DailyJournal(workspace, trade_date))._root(request_ref) + "/capture.v1.json"
    )
    capture_bytes = storage.read_workspace_file_bytes(capture_path, maximum_bytes=32 * 1024 * 1024)
    if request_identity(node_request)[1] != terminal["request_key"] or terminal["output_refs"][
        "capture"
    ] != {"path": capture_path, "sha256": capture_bytes.byte_sha256}:
        raise ContractError("EOD_DECISION_CAPTURE_BINDING_INVALID")
    if inputs["schema_version"] in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        from .decision_publication import DecisionReportPublication

        publication = DecisionReportPublication(
            journal=DailyJournal(workspace, trade_date),
            recipe_ref=inputs["decision_recipe_ref"],
            research_request_ref=request_ref,
            store_plan_ref=inputs["store_plan_ref"],
        )
        output = publication.probe(node_request, captured)
        if (
            output is None
            or set(terminal["output_refs"]) != {"capture", "result", *output}
            or any(terminal["output_refs"][key] != ref for key, ref in output.items())
        ):
            raise ContractError("EOD_DECISION_REPORT_OUTPUT_MISMATCH")
        report = read(output["decision.v2.json"])
    elif (
        set(terminal["output_refs"]) != {"capture", "result"}
        or "decision_recipe" in node_request["input_refs"]
    ):
        raise ContractError("EOD_DECISION_LEGACY_PROFILE_MISMATCH")
    files.recheck()
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=64 * 1024 * 1024).data != raw:
            raise ContractError("EOD_DECISION_SOURCE_CHANGED_DURING_REPLAY")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_DECISION_COMPLETION_CHANGED_DURING_REPLAY")
    return {
        "completion_ref": dict(completion_ref),
        "result": rebuilt,
        "validation_scope": "COMPLETED_DECISION_NATIVE_RECOMPILE",
        "consumer_admission": False,
        **({"decision_report": report} if report is not None else {}),
    }
