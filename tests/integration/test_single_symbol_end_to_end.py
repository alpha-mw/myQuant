from __future__ import annotations

from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan, ReportBundle, RiskDecision

from tests.integration._helpers import assert_protocol_bundle, run_stubbed_quant_path


def test_single_symbol_end_to_end(monkeypatch):
    run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ"],
        thesis_prefix="single-symbol",
        veto=True,
        kline_backend="hybrid",
    )

    result = run.result
    bundle = result.agent_report_bundle

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
    assert bundle.macro_verdict.thesis
    assert bundle.risk_decision.hard_veto is True
    assert bundle.risk_decision.veto is True
    assert bundle.portfolio_plan.target_weights == {}
    assert bundle.portfolio_plan.cash_ratio == 1.0
    assert bundle.ic_decision.rejected_symbols == ["000001.SZ"]
    assert result.agent_report_bundle.portfolio_plan.target_weights == run.artifacts["portfolio_plan"].target_weights
    assert result.agent_report_bundle.ic_decision.action == run.artifacts["ic_decision"].action
