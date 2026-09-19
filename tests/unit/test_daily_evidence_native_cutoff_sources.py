"""Native research/book sources with an explicit controlled core/PIT fixture boundary."""

import pytest
from datetime import datetime, timedelta, timezone
import json
import hashlib

from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.output import CommandError
from quant_investor.operations.research_source_bundle import SourceBundle, build_source_bundle
from quant_investor.operations.research_corporate_inputs import derive_corporate_inputs
from quant_investor.intelligence.pcb_ai_hardware import FOCUS_COMPANIES, EVIDENCE_KIND
from _native_cutoff_sources_fixture import build


def test_native_five_domain_projection_and_corporate_sources(tmp_path, monkeypatch):
    case = build(tmp_path, monkeypatch)
    journal, workspace = case["journal"], case["workspace"]
    with journal.locked():
        bundle = build_source_bundle(
            journal=journal, recovered=case["recovered"], auxiliary={"stages": {}}
        )
    sources = SourceBundle(journal=journal, document=bundle)
    result = sources.project("2026-08-28T13:30:00Z", current_macro=True)
    print("NATIVE_SOURCE_FIXTURE all five source domains projected", flush=True)
    assert set(result["projection_sha256s"]) == {
        "industry",
        "theme",
        "exposure",
        "fundamental",
        "macro",
    }
    assert all(result["artifacts"].values())
    exposure = result["artifacts"]["exposure"][0]
    assert exposure["payload"]["status"] == "BLOCKED"
    focus = next(a for a in result["artifacts"]["exposure"] if a["kind"] == EVIDENCE_KIND)
    assert focus["payload"]["completion_state"] == "PARTIAL_WITH_EXPLICIT_MISSING"
    assert [row["company_code"] for row in focus["payload"]["company_rows"]] == list(
        FOCUS_COMPANIES
    )
    corporate = derive_corporate_inputs(sources=sources, cutoff="2026-08-28T13:30:00Z")
    assert corporate["corporate_event_list_ref"] is None
    assert any(
        row["time_semantics"] == "OWNER_EFFECTIVE" and row["original_time"].endswith("+08:00")
        for row in corporate["source_times"]
    )
    assert not (workspace / sources.reference["path"]).exists()
    assert not (workspace / corporate["corporate_context_ref"]["path"]).exists()
    sources.files.recheck()
    payload, _, _ = sources.payload("2026-08-28T13:30:00Z")
    pointer = workspace / "data/parquet/cn/macro_observations/_latest.json"
    original = pointer.read_bytes()
    try:
        pointer.write_bytes(original + b"\n")
        frozen, admission = sources._macro(payload, case["ctx"].rank, current=False)
        assert canonical_json_bytes(frozen) == canonical_json_bytes(result["artifacts"]["macro"])
        assert admission == result["macro_admission"]
        with pytest.raises(CommandError, match="MACRO_RISK_INVALID"):
            sources._macro(payload, case["ctx"].rank, current=True)
    finally:
        pointer.write_bytes(original)


@pytest.mark.parametrize("historical", [False, True])
def test_native_sources_materialize_v6_and_replay_without_writes(tmp_path, monkeypatch, historical):
    from quant_investor.operations import research_cutoff, research_timing
    from quant_investor.operations.research_request import load_research_request
    from quant_investor.operations.research_sources import ResearchSources
    from scripts import daily_materialization
    from scripts.daily_native_inputs import load_native_inputs

    case = build(
        tmp_path,
        monkeypatch,
        mode=research_timing.HISTORICAL if historical else research_timing.CURRENT,
    )
    journal, workspace = case["journal"], case["workspace"]
    clock = {
        "now": (
            datetime(2026, 9, 1, 13, 30, 0, 250000, tzinfo=timezone.utc)
            if historical
            else datetime(2026, 8, 28, 13, 30, 0, 250000, tzinfo=timezone.utc)
        )
    }

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"]

    monkeypatch.setattr(research_timing, "datetime", Clock)
    monkeypatch.setattr(daily_materialization, "datetime", Clock)
    monkeypatch.setattr(research_cutoff, "_clock", lambda: clock["now"])

    def align(seconds):
        assert 0 < seconds <= 1
        clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(research_cutoff.time, "sleep", align)
    with journal.locked():
        materialized = daily_materialization.materialize_locked(
            journal=journal, recovered=case["recovered"], auxiliary={"stages": {}}
        )
    print("NATIVE_SOURCE_FIXTURE materialization5/native6 written", flush=True)
    native = json.loads((workspace / materialized.native_inputs_ref["path"]).read_bytes())
    assert native["schema_version"] == "cn-daily-native-inputs.v6"
    request = load_research_request(workspace=workspace, reference=native["research_request_ref"])
    assert request["document"]["as_of"] == (
        "2026-08-28T13:30:00Z" if historical else "2026-08-28T13:30:01Z"
    )
    assert request["cutoff"]["portfolio_state"]["payload"]["timing_status"] == (
        "LATE_RECORDED" if historical else "ON_TIME"
    )
    # Factor activation/rank production is the explicit outer fixture boundary.
    # Model the public rank builder's cutoff-specific artifact, preserving this
    # fixture's frozen values and refs. No production rank or pointer is changed.
    from quant_investor.cli import unified
    from quant_investor.intelligence import compile_daily_intelligence
    from quant_investor.contracts import seal_artifact

    sources = SourceBundle(journal=journal, document=request["cutoff"]["source_bundle"])
    payload = request["document"]
    projections = sources.project(payload["as_of"], current_macro=True)["artifacts"]
    macro_pointer = workspace / "data/parquet/cn/macro_observations/_latest.json"
    original_macro = macro_pointer.read_bytes()
    try:
        macro_pointer.write_bytes(original_macro + b"\n")
        frozen = load_research_request(
            workspace=workspace, reference=native["research_request_ref"]
        )
        assert frozen["cutoff"]["receipt"] == request["cutoff"]["receipt"]
        with pytest.raises(CommandError, match="MACRO_RISK_INVALID"):
            unified._daily_macro_evidence(
                payload["company_evidence"]["macro_risk"],
                sources.files.source_file,
                workspace,
                case["ctx"].rank,
                payload,
            )
    finally:
        macro_pointer.write_bytes(original_macro)
    frame, fundamental = unified._daily_fundamental_source(
        payload["company_evidence"],
        sources.files.source_file,
        workspace=workspace,
        decision_as_of=payload["as_of"],
    )
    rank = seal_artifact(
        case["ctx"].rank["kind"],
        {**case["ctx"].rank["payload"], "as_of": payload["as_of"]},
        created_at=payload["as_of"],
    )
    assert rank["payload"]["pool_rows"] == case["ctx"].rank["payload"]["pool_rows"]
    assert (
        rank["payload"]["factor_generation_ref"]
        == case["ctx"].rank["payload"]["factor_generation_ref"]
    )
    compiled = compile_daily_intelligence(
        as_of=payload["as_of"],
        strategy_id=payload["strategy_id"],
        policy=payload["policy"],
        rank=rank,
        industry_projection=projections["industry"][0],
        theme_projection=projections["theme"][0],
        exposure_evidence=[
            a for a in projections["exposure"] if a["kind"] == "company_source_evidence"
        ],
        fundamental_frame=frame,
        fundamental_source=fundamental,
        market_risk_evidence=projections["macro"][0],
    )
    assert compiled["as_of"] == request["document"]["as_of"]
    assert compiled["status"] == "PARTIAL"
    assert all(value is False for value in compiled["authority"].values())
    print("NATIVE_SOURCE_FIXTURE native compiler consumed cutoff-bound sources", flush=True)
    context = ResearchSources(
        workspace=str(workspace),
        journal=journal,
        native_request_ref=native["research_request_ref"],
        pool_manifest_ref=case["ctx"].pool_ref,
        release_ref=case["ctx"].release_ref,
    )
    with journal.locked():
        context.prepare()
    exposure_recipe = context._read(context.recipe_refs["exposure"])
    assert exposure_recipe["source_completion_policy"] == request["source_completion_policy"]
    before = {
        str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in workspace.rglob("*")
        if p.is_file()
    }
    clock["now"] = datetime(2026, 9, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(
        research_cutoff.time, "sleep", lambda *a: pytest.fail("replay aligned a new clock")
    )
    with journal.locked():
        replay = daily_materialization.materialize_locked(
            journal=journal, recovered=case["recovered"], auxiliary={"stages": {}}
        )
    day, inputs = load_native_inputs(workspace=str(workspace), input_ref=replay.native_inputs_ref)
    assert day == journal.trade_date and inputs.cutoff_ref == native["cutoff_ref"]
    assert (
        replay.status == "NO_ACTION" and replay.native_inputs_ref == materialized.native_inputs_ref
    )
    assert before == {
        str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in workspace.rglob("*")
        if p.is_file()
    }
