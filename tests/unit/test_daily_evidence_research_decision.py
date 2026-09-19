"""Decision adapter with real native compiler output; CLI Factor loading is a fixture seam."""

from quant_investor.intelligence import compile_daily_intelligence
from quant_investor.operations import research_decision
from test_daily_evidence_research_sources import context


def test_native_insufficient_evidence_decision_is_captured_without_granting_admission(
    tmp_path, monkeypatch
):
    ctx, journal = context(tmp_path)
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
        assert result["status"] == "PARTIAL"
        monkeypatch.setattr(research_decision, "research_compile_daily", lambda **kwargs: result)
        adapter = research_decision.ResearchDecisionAdapter(ctx)
        request = adapter.template()
        assert adapter.probe(request).safe_to_execute
        adapter.execute(request)
        assert adapter.probe(request).outcome.state.value == "SUCCEEDED"
        captured = adapter.capture.read(ctx.request_ref)
        assert captured[0]["research_status"] == "PARTIAL"
        assert all(row["state"] == "INSUFFICIENT_EVIDENCE" for row in captured[1]["decisions"])
        assert all(value is False for value in captured[1]["authority"].values())
