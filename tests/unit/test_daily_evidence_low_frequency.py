"""Real source adapters/captures, native scoring and legacy synthetic pool admission."""

import hashlib
import json

import pandas as pd
import pytest

from quant_investor.intelligence.daily_evidence import FUNDAMENTAL_METRICS
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND, FRESHNESS_CONTRACT
from quant_investor.operations.research_sources import ResearchSourceAdapter
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.journal_revisions import append_revision
from test_daily_evidence_research_sources import context, put
from test_daily_evidence_research_auxiliary import changed


def fundamental_source(root, companies, day, *, nulls=()):
    rows = [
        {
            "ts_code": company,
            "trade_date": day,
            **{key: None if key in nulls else 0.1 + index / 1000 for key in FUNDAMENTAL_METRICS},
        }
        for index, company in enumerate(companies)
    ]
    path = root / "fundamental/g1/daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    path.chmod(0o600)
    return {
        "available_at": "2026-08-27T08:00:00.000001Z",
        "daily_parquet": {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "pointer": put(
            root,
            "fundamental/pointer.json",
            {
                "generation_id": "g1",
                "status": "OK",
                "metadata": {"binding_aware_research_ready": True, "gate2_passed": True},
            },
        ),
    }


@pytest.mark.parametrize(
    "day,state,nulls",
    [
        ("20260828", "FRESH", ()),
        ("20260814", "ACCEPTABLE_LAG", ()),
        ("20251201", "STALE_WARNING", ()),
        ("20260829", "MISSING", ()),
        ("20260828", "FRESH", ("fin_roe", "fin_roa", "fin_debt_to_assets")),
        ("20260828", "MISSING", FUNDAMENTAL_METRICS),
    ],
)
def test_four_states_capture_and_status_are_readonly(tmp_path, day, state, nulls):
    ctx, journal = context(tmp_path)
    descriptor = fundamental_source(tmp_path, ctx.companies, day, nulls=nulls)
    ctx = changed(ctx, tmp_path, {"fundamental_source": descriptor}, "fundamental-request.json")
    with journal.locked():
        ctx.prepare()
        request = ctx.template("fundamental")
        recipe = ctx._read(request["input_refs"]["recipe"])
        assert recipe["freshness_contract"] == FRESHNESS_CONTRACT
        adapter = ResearchSourceAdapter(ctx, "fundamental")
        journal.begin(request)
        assert adapter.probe(request).safe_to_execute
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        report = ctx._read(outcome.output_refs[FRESHNESS_KIND])
        body = report["payload"]
        assert body["freshness_state"] == state
        assert len(body["entries"]) == 100
        assert outcome.state.value == ("PARTIAL" if state == "MISSING" else "SUCCEEDED")
        assert bool(body["critical_missing_codes"]) == (state == "MISSING")
        assert body["policy"]["blocking_after_days"] is None
        assert all(value is False for value in body["authority"].values())
        if state == "STALE_WARNING" or nulls:
            assert body["warning_codes"]
        if day != "20260829":
            assert body["entries"][0]["known_at"] == descriptor["available_at"]
        journal.finish(
            request,
            state=outcome.state,
            output_refs=outcome.output_refs,
            failure_code=outcome.failure_code,
        )
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    status = read_daily_status(str(tmp_path), "20260828")
    assert status["nodes"]["fundamental"][FRESHNESS_KIND]["freshness_state"] == state
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("node", ["fundamental", "macro"])
def test_missing_report_emitted_before_input_missing_and_can_revise(tmp_path, node):
    ctx, journal = context(tmp_path)
    with journal.locked():
        ctx.prepare()
        request = ctx.template(node)
        adapter = ResearchSourceAdapter(ctx, node)
        journal.begin(request)
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        assert set(outcome.output_refs) == {"capture", FRESHNESS_KIND}
        assert outcome.failure_code == "INPUT_MISSING"
        body = ctx._read(outcome.output_refs[FRESHNESS_KIND])["payload"]
        assert body["freshness_state"] == "MISSING" and body["critical_missing_codes"]
        terminal = journal.finish(
            request,
            state=outcome.state,
            output_refs=outcome.output_refs,
            failure_code=outcome.failure_code,
        )
        new = json.loads(json.dumps(request))
        new["input_refs"]["recipe"] = put(tmp_path, "arrived-recipe.json", {"source_arrived": True})
        append_revision(journal, new, expected_request_key=terminal["request_key"])


def test_legacy_projection_shape_and_unknown_selector(tmp_path):
    ctx, journal = context(tmp_path)
    with journal.locked():
        ctx.prepare()
    from quant_investor.operations.research_projection import project_research_source

    recipe = ctx._read(ctx.template("fundamental")["input_refs"]["recipe"])
    recipe.pop("freshness_contract")
    kwargs = dict(
        node="fundamental",
        recipe=recipe,
        companies=ctx.companies,
        source_document=None,
        source_file=None,
        workspace=tmp_path,
    )
    assert project_research_source(**kwargs) is None
    for value in (None, False, "unknown"):
        recipe["freshness_contract"] = value
        with pytest.raises(Exception, match="LOW_FREQUENCY_CONTRACT_INVALID"):
            project_research_source(**kwargs)


def test_stale_fundamental_keeps_native_decision_scores_and_capture(tmp_path, monkeypatch):
    from functools import partial
    from quant_investor.cli.unified import _daily_fundamental_source, _daily_source_file
    from quant_investor.intelligence import compile_daily_intelligence
    from quant_investor.operations import research_decision

    ctx, journal = context(tmp_path)
    compiled = []
    with journal.locked():
        ctx.prepare()
        theme = ctx.project("theme", ctx.template("theme")).artifacts[0]
        for day in ("20260828", "20251201"):
            descriptor = fundamental_source(tmp_path, ctx.companies, day)
            frame, source = _daily_fundamental_source(
                {"fundamental_source": descriptor},
                partial(_daily_source_file, tmp_path),
                workspace=tmp_path,
                decision_as_of=ctx.request["as_of"],
            )
            result = compile_daily_intelligence(
                as_of=ctx.request["as_of"],
                strategy_id=ctx.request["strategy_id"],
                rank=ctx.rank,
                policy=ctx.request["policy"],
                industry_projection=None,
                theme_projection=theme,
                fundamental_frame=frame,
                fundamental_source=source,
            )
            compiled.append(result)
        for result in compiled:
            assessments = [
                a["payload"] for a in result["artifacts"] if a["kind"] == "fundamental_assessment"
            ]
            assert len(assessments) == 100
            assert not any(a["blocker_codes"] for a in assessments)
            assert FRESHNESS_KIND not in {a["kind"] for a in result["artifacts"]}
        assert [row["state"] for row in compiled[0]["decisions"]] == [
            row["state"] for row in compiled[1]["decisions"]
        ]
        # Same outer CLI/Factor fixture seam as the existing Decision adapter test.
        monkeypatch.setattr(research_decision, "research_compile_daily", lambda **kw: compiled[1])
        adapter = research_decision.ResearchDecisionAdapter(ctx)
        request = adapter.template()
        adapter.execute(request)
        assert adapter.probe(request).outcome.state.value == "SUCCEEDED"
