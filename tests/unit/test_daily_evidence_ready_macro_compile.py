"""Ready Macro data must not trigger allocation without candidate liquidity inputs."""

import pytest
from test_daily_evidence_research_sources import context
from quant_investor.intelligence import compile_daily_intelligence
from quant_investor.intelligence.daily_evidence import build_market_risk_evidence


@pytest.mark.parametrize("veto", [False, True])
def test_daily_macro_cash_projection_requires_hard_veto(tmp_path, veto):
    ctx, journal = context(tmp_path)
    risk = build_market_risk_evidence(
        source_path=(
            "data/veto.json"
            if veto
            else "results/intelligence/macro_readiness/20260828/" + "d" * 64 + ".json"
        ),
        source_sha256="d" * 64,
        blocker_codes=["MACRO_RELEASE_CONTRACT_BLOCKED"] if veto else [],
        classification="PIPELINE_DATA_VETO" if veto else "CANONICAL_MACRO_READY",
        as_of=ctx.request["as_of"],
    )
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
            market_risk_evidence=risk,
        )
    if veto:
        assert result["research_portfolio"] is not None
    else:
        assert result["research_portfolio"] is None
    assert all(value is False for value in result["authority"].values())
