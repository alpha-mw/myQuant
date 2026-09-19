"""Native compiler, Store portfolio and report publication with explicit source seams.

The outer Factor/CLI admission and eight-domain collector are isolated here;
collector/native whole-DAG acceptance is separate. No fabricated Factor proof.
"""

from copy import deepcopy
from pathlib import Path
import json

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.intelligence import compile_daily_intelligence
from quant_investor.intelligence.decision_report import DOMAINS, REPORT_KIND, build_decision_report
from quant_investor.operations import research_decision
from quant_investor.operations.decision_recipe import publish_decision_recipe, read_decision_recipe
from quant_investor.operations.decision_sources import DecisionSources
from quant_investor.operations import decision_publication
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_research_sources import context, put
from _native_daily_store_fixture import NativeStoreFixture, DAYS
from scripts.daily_production_store_adapter import prepare_store_plan


def setup(root, monkeypatch):
    ctx, journal = context(root)
    fixture = NativeStoreFixture(root)
    for day in DAYS:
        args = fixture.advance(day)
    plan = prepare_store_plan(args)
    plan_ref = {"path": plan["plan_path"], "sha256": plan["plan_sha256"]}
    anchor = put(
        root, "source-collector-seam.json", {"synthetic": True, "native_source_admission": False}
    )
    source_inputs = {
        "domain_physical_refs": {d: [anchor] for d in DOMAINS},
        "freshness_reports": {"fundamental": None, "macro": None},
    }
    monkeypatch.setattr(DecisionSources, "collect", lambda self: deepcopy(source_inputs))
    with journal.locked():
        ctx.prepare()
        theme = ctx.project("theme", ctx.template("theme")).artifacts[0]
        result = compile_daily_intelligence(
            as_of=ctx.request["as_of"],
            strategy_id=ctx.request["strategy_id"],
            rank=ctx.rank,
            policy=ctx.request["policy"],
            industry_projection=None,
            theme_projection=theme,
        )
        recipe_ref = publish_decision_recipe(
            journal=journal, research_request_ref=ctx.request_ref, store_plan_ref=plan_ref
        )
    bound = read_decision_recipe(
        workspace=root,
        trade_date=journal.trade_date,
        recipe_ref=recipe_ref,
        research_request_ref=ctx.request_ref,
        store_plan_ref=plan_ref,
    )
    monkeypatch.setattr(
        research_decision, "research_compile_daily", lambda **kwargs: deepcopy(result)
    )
    source_inputs["domain_physical_refs"]["portfolio"] = [
        bound["recipe"]["portfolio_source_ref"],
        *bound["portfolio"]["payload"]["source_refs"],
    ]
    adapter = research_decision.ResearchDecisionAdapter(
        ctx, decision_recipe_ref=recipe_ref, store_plan_ref=plan_ref
    )
    return ctx, adapter, bound, result, source_inputs


def inventory(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_v2_report_preserves_native_decisions_and_has_no_trading_authority(tmp_path, monkeypatch):
    ctx, adapter, bound, native, _ = setup(tmp_path, monkeypatch)
    with ctx.journal.locked():
        request = adapter.template()
        ctx.journal.begin(request)
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        assert outcome.state.value == "SUCCEEDED"
        assert set(outcome.output_refs) == {
            "capture",
            "result",
            "decision.v2.json",
            "decision_report_capture",
        }
        ctx.journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)
    report = json.loads((tmp_path / outcome.output_refs["decision.v2.json"]["path"]).read_bytes())
    assert report["kind"] == REPORT_KIND
    assert report["payload"]["prospective"] is False
    assert [r["decision"] for r in report["payload"]["company_rows"]] == [
        r["state"] for r in native["decisions"]
    ]
    assert set(report["payload"]["source_bindings"]) == set(DOMAINS)
    assert report["payload"]["source_bindings"]["portfolio"]["status"] == "PARTIAL"
    assert report["payload"]["portfolio_state_ref"] == bound["recipe"]["portfolio_source_ref"]
    assert all(v is False for v in report["payload"]["authority"].values())
    assert adapter.capture.read(ctx.request_ref)[1] == native
    before = inventory(tmp_path)
    assert adapter.probe(request).outcome == outcome
    assert inventory(tmp_path) == before


def test_report_without_side_capture_adopts_original_creation_time(tmp_path, monkeypatch):
    ctx, adapter, _, _, _ = setup(tmp_path, monkeypatch)
    request = adapter.template()
    original = ctx.journal.storage.write

    def fail_capture(path, raw, **kwargs):
        if str(path) == adapter.report.capture_path(request):
            raise RuntimeError("after report before capture")
        return original(path, raw, **kwargs)

    with ctx.journal.locked():
        ctx.journal.begin(request)
        with monkeypatch.context() as fault:
            fault.setattr(ctx.journal.storage, "write", fail_capture)
            with pytest.raises(RuntimeError, match="after report before capture"):
                adapter.execute(request)
        path = tmp_path / adapter.report.path
        raw, mtime = path.read_bytes(), path.stat().st_mtime_ns
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        assert outcome is not None
        assert path.read_bytes() == raw and path.stat().st_mtime_ns == mtime
        ctx.journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)


def test_capture_without_terminal_adopts_without_sampling_another_clock(tmp_path, monkeypatch):
    ctx, adapter, _, _, _ = setup(tmp_path, monkeypatch)
    with ctx.journal.locked():
        request = adapter.template()
        ctx.journal.begin(request)
        adapter.execute(request)
        before = inventory(tmp_path)

        def no_clock():
            pytest.fail("adoption sampled a replacement report/capture clock")

        monkeypatch.setattr(decision_publication, "_now", no_clock)
        adapter.execute(request)
        assert adapter.probe(request).outcome is not None
        assert inventory(tmp_path) == before


@pytest.mark.parametrize("field", ["decision", "confidence"])
def test_forged_report_state_or_confidence_rejects_even_with_new_envelope(
    tmp_path, monkeypatch, field
):
    ctx, adapter, _, _, _ = setup(tmp_path, monkeypatch)
    with ctx.journal.locked():
        request = adapter.template()
        ctx.journal.begin(request)
        adapter.execute(request)
        path = tmp_path / adapter.report.path
        report = json.loads(path.read_bytes())
        report["payload"]["company_rows"][0][field] = "forged"
        forged = seal_artifact(report["kind"], report["payload"], created_at=report["created_at"])
        path.write_bytes(canonical_json_bytes(forged))
        before = path.read_bytes()
        with pytest.raises(ContractError, match="IMMUTABLE_CONFLICT"):
            adapter.probe(request)
        assert path.read_bytes() == before


def test_missing_or_case_variant_domains_reject(tmp_path, monkeypatch):
    ctx, _, bound, native, source_inputs = setup(tmp_path, monkeypatch)
    source_inputs["domain_physical_refs"]["Factor"] = source_inputs["domain_physical_refs"].pop(
        "factor"
    )
    with pytest.raises(ValueError, match="DOMAIN_SET_INVALID"):
        build_decision_report(
            result=native,
            result_ref=ctx.request_ref,
            portfolio_state=bound["portfolio"],
            portfolio_state_ref=bound["recipe"]["portfolio_source_ref"],
            created_at=bound["portfolio"]["created_at"],
            **source_inputs,
        )
