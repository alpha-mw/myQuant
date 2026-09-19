"""Pure native replay of four captured research sources in a recorded completion."""

from pathlib import PurePosixPath

from quant_investor.contracts import (
    parse_canonical_json_bytes,
    canonical_json_bytes,
    validate_artifact,
)
from quant_investor.intelligence.storage import DailyResearchPoolStore, approved_theme_policy_v2
from quant_investor.system.storage import SecureSystemStorage
from .completion_readback import inspect_recorded_completion
from .daily_contract import ContractError, NodeState, validate_ref, utc_stamp
from .daily_journal import request_identity, _false_authority
from .research_projection import project_research_source
from .research_recipes import derive_focus_context, source_recipe, artifact_output_names
from quant_investor.intelligence.pcb_ai_hardware import MEMBERSHIP_KIND
from quant_investor.intelligence.low_frequency import freshness_profile


def replay_completed_research_sources(
    *, workspace: str, trade_date: str, completion_ref: dict
) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    storage = SecureSystemStorage(workspace)
    observed = {}
    from .research_file_readback import ResearchFileReadback

    files = ResearchFileReadback(workspace)

    def read(ref):
        validate_ref(ref)
        item = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if item.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_RESEARCH_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = item.data
        return parse_canonical_json_bytes(item.data, label="completed research source")

    def source_document(reference, *, code):
        return read(reference)

    source_file = files.source_file

    native_inputs = read(recorded["native_inputs_ref"])
    read(native_inputs["research_request_ref"])
    from .research_request import load_research_request

    loaded = load_research_request(
        workspace=workspace, reference=native_inputs["research_request_ref"]
    )
    original = loaded["document"]
    read(loaded["native_request_ref"])
    top = read(recorded["node_terminal_refs"]["top100"])
    pool_ref = top["output_refs"]["manifest.json"]
    pool = validate_artifact(read(pool_ref))
    prefix = str(PurePosixPath(pool_ref["path"]).parent)
    rank = read(
        {
            "path": prefix + "/factor_research_rank.json",
            "sha256": pool["payload"]["rank_byte_sha256"],
        }
    )
    DailyResearchPoolStore(workspace).verify(
        rank=rank,
        expected_policy_sha256=pool["payload"]["policy_byte_sha256"],
        policy_path=pool["payload"]["policy_path"],
    )
    if (
        pool["payload"]["signal_date"] != trade_date
        or original["policy"] != approved_theme_policy_v2()
        or original["expected_factor_pointer_sha256"] != pool["payload"]["factor_pointer_sha256"]
    ):
        raise ContractError("EOD_RESEARCH_POOL_BINDING_INVALID")
    companies = [row["symbol"] for row in rank["payload"]["pool_rows"]]
    focus_context = derive_focus_context(
        workspace=workspace,
        request=original,
        rank=rank,
        pool_store=DailyResearchPoolStore(workspace),
    )
    projected = {}
    for node, field in [
        ("theme", "theme_source"),
        ("industry", "industry_source"),
        ("exposure", "exposure_rows"),
        ("fundamental", "fundamental_source"),
    ]:
        terminal_ref = recorded["node_terminal_refs"][node]
        terminal = read(terminal_ref)
        path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
        raw = storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
        observed[path] = raw.data
        request = parse_canonical_json_bytes(raw.data)
        if (
            request_identity(request)[1] != terminal["request_key"]
            or request["input_refs"]["pool"] != pool_ref
        ):
            raise ContractError("EOD_RESEARCH_REQUEST_BINDING_INVALID")
        recipe = read(request["input_refs"]["recipe"])
        if recipe != source_recipe(
            node=node,
            field=field,
            request=original,
            pool_ref=pool_ref,
            focus_context=focus_context,
            freshness_contract=freshness_profile(recipe),
            completion_policy=loaded["source_completion_policy"],
        ):
            raise ContractError("EOD_RESEARCH_RECIPE_BINDING_INVALID")
        result = project_research_source(
            node=node,
            recipe=recipe,
            companies=companies,
            source_document=source_document,
            source_file=source_file,
            workspace=workspace,
            theme=projected.get("theme", [None])[0],
            focus_membership=next(
                (v for v in projected.get("theme", []) if v["kind"] == MEMBERSHIP_KIND), None
            ),
        )
        if result is None or result.state != NodeState.SUCCEEDED:
            raise ContractError("EOD_RESEARCH_NATIVE_REPLAY_INCOMPLETE")
        cap = read(terminal["output_refs"]["capture"])
        names = artifact_output_names(result.artifacts)
        if set(terminal["output_refs"]) != {"capture", *names}:
            raise ContractError("EOD_RESEARCH_CAPTURE_INVALID")
        refs = [terminal["output_refs"][name] for name in names]
        if (
            set(terminal["output_refs"]) != {"capture", *names}
            or cap.get("schema_version") != "cn-daily-research-node-capture.v1"
            or cap.get("request_key") != terminal["request_key"]
            or cap.get("state") != "SUCCEEDED"
            or cap.get("artifact_refs") != refs
            or not _false_authority(cap.get("authority"))
            or cap.get("research_cutoff") != original["as_of"]
            or not utc_stamp(original["as_of"])
            <= utc_stamp(cap["captured_at"])
            <= utc_stamp(terminal["finished_at"])
        ):
            raise ContractError("EOD_RESEARCH_CAPTURE_INVALID")
        for artifact, ref in zip(result.artifacts, refs):
            if canonical_json_bytes(artifact) != canonical_json_bytes(read(ref)):
                raise ContractError("EOD_RESEARCH_ARTIFACT_DOES_NOT_REPLAY")
        projected[node] = result.artifacts
    files.recheck()
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=64 * 1024 * 1024).data != raw:
            raise ContractError("EOD_RESEARCH_SOURCE_CHANGED_DURING_REPLAY")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_RESEARCH_COMPLETION_CHANGED_DURING_REPLAY")
    return {
        "completion_ref": dict(completion_ref),
        "artifacts": projected,
        "validation_scope": "COMPLETED_RESEARCH_SOURCES_NATIVE_REPLAY",
        "consumer_admission": False,
    }
