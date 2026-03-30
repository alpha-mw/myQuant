from __future__ import annotations

from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan, ReportBundle, RiskDecision

from tests.integration._helpers import assert_protocol_bundle, run_stubbed_quant_path


def test_shortlist_second_pass_end_to_end(monkeypatch):
    run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ", "600519.SH", "000858.SZ"],
        thesis_prefix="shortlist-second-pass",
        veto=False,
        kline_backend="chronos",
    )

    result = run.result
    bundle = result.agent_report_bundle

    assert run.investor.kline_backend == "chronos"
    assert isinstance(bundle.macro_verdict, BranchVerdict)
    assert isinstance(bundle.risk_decision, RiskDecision)
    assert isinstance(bundle.ic_decision, ICDecision)
    assert isinstance(bundle.portfolio_plan, PortfolioPlan)
    assert isinstance(bundle, ReportBundle)
    assert_protocol_bundle(
        bundle,
        risk_decision=run.artifacts["risk_decision"],
        ic_decision=run.artifacts["ic_decision"],
        portfolio_plan=run.artifacts["portfolio_plan"],
    )
    assert bundle.risk_decision.hard_veto is False
    assert bundle.ic_decision.selected_symbols
    assert bundle.portfolio_plan.target_weights
    assert sum(bundle.portfolio_plan.target_weights.values()) > 0
    assert set(bundle.portfolio_plan.target_weights) == set(run.artifacts["portfolio_plan"].target_weights)
    assert set(bundle.branch_verdicts) == {"kline", "quant", "fundamental", "intelligence", "macro"}
