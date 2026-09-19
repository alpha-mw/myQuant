"""Read-only Macro replay scoped to a previously sealed EOD, never new admission.

This is one validator in the full Morning replay matrix. It cannot authorize
Morning or fresh Decision compilation, and does not inspect or clear live vetoes.
"""

from pathlib import PurePosixPath, Path
from datetime import datetime

from quant_investor.contracts import parse_canonical_json_bytes, canonical_json_bytes
from quant_investor.intelligence.daily_evidence import build_market_risk_evidence
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND, freshness_profile
from quant_investor.intelligence.macro_freshness import macro_freshness_from_closure
from quant_investor.macro.readiness_closure import validate_macro_readiness_closure
from quant_investor.system.storage import SecureSystemStorage
from .completion_readback import inspect_recorded_completion
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal, request_identity, _false_authority
from .research_capture import ResearchCapture
from .research_recipes import source_recipe


def _original_availability(closure: dict, *, trade_date: str, cutoff: str) -> None:
    # The native closure validator owns timestamp grammar, including microseconds.
    available = datetime.fromisoformat(closure["available_at"].replace("Z", "+00:00"))
    if closure["target_date"] != trade_date or available > utc_stamp(cutoff):
        raise ContractError("EOD_MACRO_UNAVAILABLE_AT_ORIGINAL_DECISION")


def replay_completed_macro(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    storage = SecureSystemStorage(workspace)
    observed = {}

    def source_file(ref, *, code):
        validate_ref(ref)
        item = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if item.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_MACRO_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = item.data
        return Path(workspace) / ref["path"], item.data, dict(ref)

    def read(ref):
        _, raw, _ = source_file(ref, code="EOD_MACRO_SOURCE_SHA_MISMATCH")
        return parse_canonical_json_bytes(raw, label="completed Macro source")

    def node(name):
        terminal_ref = recorded["node_terminal_refs"][name]
        terminal = read(terminal_ref)
        path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
        item = storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
        observed[path] = item.data
        request = parse_canonical_json_bytes(item.data, label="completed node request")
        if request_identity(request)[1] != terminal["request_key"]:
            raise ContractError("EOD_MACRO_NODE_REQUEST_MISMATCH")
        return terminal, request

    macro, macro_request = node("macro")
    decision, decision_request = node("decision")
    inputs = read(recorded["native_inputs_ref"])
    request_ref = inputs["research_request_ref"]
    if decision_request["input_refs"]["native_request"] != request_ref:
        raise ContractError("EOD_MACRO_DECISION_REQUEST_MISMATCH")
    read(request_ref)
    from .research_request import load_research_request

    loaded = load_research_request(workspace=workspace, reference=request_ref)
    original = loaded["document"]
    read(loaded["native_request_ref"])
    recipe = read(macro_request["input_refs"]["recipe"])
    profile = freshness_profile(recipe)
    names = ["artifact"] + ([FRESHNESS_KIND] if profile is not None else [])
    if set(macro["output_refs"]) != {"capture", *names}:
        raise ContractError("EOD_MACRO_CAPTURE_BINDING_INVALID")
    if profile is not None and recipe != source_recipe(
        node="macro",
        field="macro_risk",
        request=original,
        pool_ref=macro_request["input_refs"]["pool"],
        focus_context=None,
        freshness_contract=profile,
    ):
        raise ContractError("EOD_MACRO_RECIPE_BINDING_INVALID")
    capture = read(macro["output_refs"]["capture"])
    decision_capture = read(decision["output_refs"]["capture"])
    result = read(decision["output_refs"]["result"])
    captured = ResearchCapture(DailyJournal(workspace, trade_date)).read(request_ref)
    if captured is None or captured != (decision_capture, result):
        raise ContractError("EOD_MACRO_DECISION_CAPTURE_MISMATCH")
    risk = (original.get("company_evidence") or {}).get("macro_risk")
    if (
        type(risk) is not dict
        or set(risk) != {"classification", "source"}
        or risk["classification"] != "CANONICAL_MACRO_READY"
        or recipe.get("macro_risk") != risk
        or decision_capture["result_ref"] != decision["output_refs"]["result"]
        or capture.get("schema_version") != "cn-daily-research-node-capture.v1"
        or capture.get("state") != "SUCCEEDED"
        or capture.get("request_key") != macro["request_key"]
        or capture.get("artifact_refs") != [macro["output_refs"][name] for name in names]
        or not _false_authority(capture.get("authority"))
    ):
        raise ContractError("EOD_MACRO_CAPTURE_BINDING_INVALID")
    cutoff = original["as_of"]
    if any(
        instant != cutoff
        for instant in (
            recipe.get("as_of"),
            capture.get("research_cutoff"),
            decision_capture.get("research_cutoff"),
            result.get("as_of"),
        )
    ):
        raise ContractError("EOD_MACRO_CUTOFF_MISMATCH")
    if (
        utc_stamp(cutoff).strftime("%Y%m%d") != trade_date
        or utc_stamp(capture["captured_at"]) < utc_stamp(cutoff)
        or utc_stamp(capture["captured_at"]) > utc_stamp(macro["finished_at"])
        or utc_stamp(decision_capture["captured_at"]) > utc_stamp(decision["finished_at"])
    ):
        raise ContractError("EOD_MACRO_CUSTODY_INVALID")
    closure = validate_macro_readiness_closure(
        workspace_root=workspace, closure=read(risk["source"])
    )
    _original_availability(closure, trade_date=trade_date, cutoff=cutoff)
    rebuilt = build_market_risk_evidence(
        source_path=risk["source"]["path"],
        source_sha256=risk["source"]["sha256"],
        blocker_codes=[],
        classification="CANONICAL_MACRO_READY",
        as_of=cutoff,
    )
    artifact = read(macro["output_refs"]["artifact"])
    decision_risks = [a for a in result["artifacts"] if a["kind"] == "market_risk_evidence"]
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact) or decision_risks != [
        rebuilt
    ]:
        raise ContractError("EOD_MACRO_ARTIFACT_DOES_NOT_REPLAY")
    if profile is not None:
        report = macro_freshness_from_closure(
            workspace=workspace,
            as_of=cutoff,
            closure_ref=risk["source"],
            source_file=source_file,
            closure=closure,
        )
        if report["payload"]["critical_missing_codes"] or report != read(
            macro["output_refs"][FRESHNESS_KIND]
        ):
            raise ContractError("EOD_MACRO_FRESHNESS_DOES_NOT_REPLAY")
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
            raise ContractError("EOD_MACRO_SOURCE_CHANGED_DURING_REPLAY")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_MACRO_COMPLETION_CHANGED_DURING_REPLAY")
    return {
        "completion_ref": dict(completion_ref),
        "macro_artifact": rebuilt,
        "macro_readiness_ref": risk["source"],
        "synthetic": recorded["synthetic"],
        "validation_scope": "COMPLETED_MACRO_NATIVE_REPLAY",
        "consumer_admission": False,
    }
