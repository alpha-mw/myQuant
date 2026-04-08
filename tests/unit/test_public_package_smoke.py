"""
公共入口与构建面烟测。
"""

from __future__ import annotations

from pathlib import Path

import quant_investor
import quant_investor.cli.main as cli_main


def test_public_package_exports():
    assert hasattr(quant_investor, "QuantInvestor")
    assert hasattr(quant_investor, "QuantInvestorPipelineResult")
    assert hasattr(quant_investor, "BranchResult")
    assert {
        name
        for name in dir(quant_investor)
        if name.startswith("QuantInvestor")
    } == {"QuantInvestor", "QuantInvestorPipelineResult"}


def test_cli_market_download_dispatches(monkeypatch):
    captured = {}

    def _fake_run_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_download", _fake_run_download)
    cli_main.main(["market", "download", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_cli_market_analyze_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_analysis(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_analysis", _fake_run_market_analysis)
    cli_main.main(["market", "analyze", "--market", "US", "--mode", "sample"])

    assert captured["market"] == "US"
    assert captured["mode"] == "sample"


def test_cli_market_backtest_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_backtest(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_backtest", _fake_run_market_backtest)
    cli_main.main(["market", "backtest", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_pyproject_only_packages_quant_investor():
    pyproject_text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'packages = ["quant_investor"]' in pyproject_text
    assert 'quant-investor = "quant_investor.cli.main:main"' in pyproject_text


def test_cli_research_dispatches_single_mainline(monkeypatch):
    captured = {}

    class _FakeInvestor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

        def print_report(self):
            return None

    monkeypatch.setattr(cli_main, "QuantInvestor", _FakeInvestor)
    cli_main.main(["research", "run", "--stocks", "000001.SZ"])

    assert captured["stock_pool"] == ["000001.SZ"]


def test_single_mainline_one_symbol_mock_run_includes_version_fields(monkeypatch):
    from types import SimpleNamespace

    from quant_investor.agent_protocol import (
        ActionLabel,
        AgentStatus,
        BranchVerdict,
        ExecutionTrace,
        GlobalContext,
        PortfolioDecision,
        ShortlistItem,
        WhatIfPlan,
    )
    from quant_investor.branch_contracts import BranchResult
    import quant_investor.pipeline.mainline as mainline_module

    shortlist = [
        ShortlistItem(
            symbol="000001.SZ",
            company_name="平安银行",
            category="full_a",
            rank_score=0.91,
            action=ActionLabel.BUY,
            confidence=0.78,
            expected_upside=0.12,
            suggested_weight=0.25,
            rationale=["posterior top rank"],
        )
    ]
    portfolio_decision = PortfolioDecision(
        status=AgentStatus.SUCCESS,
        shortlist=shortlist,
        target_exposure=0.45,
        target_gross_exposure=0.45,
        target_net_exposure=0.45,
        cash_ratio=0.55,
        target_weights={"000001.SZ": 0.25},
        target_positions={"000001.SZ": 250000.0},
        metadata={"selected_count": 1},
    )
    report_bundle = SimpleNamespace(
        markdown_report="# DAG Report",
        headline="DAG headline",
        summary="DAG summary",
        executive_summary=["summary line"],
        market_view=["market line"],
        branch_verdicts={"macro": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7)},
        macro_verdict=BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7),
        risk_decision=None,
        ic_decision=SimpleNamespace(action=ActionLabel.BUY, thesis="ic ok"),
        ic_decisions=[],
        model_role_metadata=SimpleNamespace(to_dict=lambda: {"branch_model": "deepseek-reasoner", "master_model": "moonshot-v1-128k"}),
        execution_trace=ExecutionTrace(),
        what_if_plan=WhatIfPlan(),
        portfolio_plan=SimpleNamespace(
            target_weights={"000001.SZ": 0.25},
            target_positions={"000001.SZ": 250000.0},
            position_limits={"000001.SZ": 0.25},
            blocked_symbols=[],
            rejected_symbols=[],
            target_exposure=0.45,
            target_gross_exposure=0.45,
            target_net_exposure=0.45,
            cash_ratio=0.55,
            execution_notes=["note"],
            construction_notes=[],
            status=AgentStatus.SUCCESS,
        ),
        coverage_summary=[],
        warnings=[],
        appendix_diagnostics=[],
        ic_hints_by_symbol={"000001.SZ": {"action": "buy", "score": 0.8}},
    )
    dag_artifacts = {
        "global_context": GlobalContext(
            market="CN",
            universe_key="full_a",
            universe_symbols=["000001.SZ"],
            universe_tiers={"total": ["000001.SZ"], "researchable": ["000001.SZ"], "shortlistable": ["000001.SZ"], "final_selected": ["000001.SZ"]},
        ),
        "symbol_research_packets": {},
        "branch_verdicts_by_symbol": {"000001.SZ": {"macro": BranchVerdict(agent_name="macro", thesis="ok", symbol="000001.SZ", final_score=0.2, final_confidence=0.7)}},
        "branch_summaries": {"macro": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7)},
        "macro_verdict": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7),
        "risk_decision": None,
        "ic_decisions": [],
        "shortlist": shortlist,
        "portfolio_plan": report_bundle.portfolio_plan,
        "portfolio_decision": portfolio_decision,
        "review_bundle": SimpleNamespace(branch_summaries={}, ic_hints_by_symbol={"000001.SZ": {"action": "buy"}}, fallback_reasons=[]),
        "model_role_metadata": report_bundle.model_role_metadata,
        "what_if_plan": report_bundle.what_if_plan,
        "execution_trace": report_bundle.execution_trace,
        "tradability_snapshot": {"000001.SZ": {"tradable": True}},
        "data_quality_issues": [],
        "data_quality_summary": {"researchable_count": 1},
        "resolver": {"resolution_strategy": "logical_full_a"},
        "report_bundle": report_bundle,
        "portfolio_master_output": SimpleNamespace(final_score=0.7, confidence=0.8),
        "portfolio_master_meta": {"confidence": 0.8},
        "branch_results": {"macro": BranchResult(branch_name="macro", final_score=0.2, final_confidence=0.7, symbol_scores={"000001.SZ": 0.2})},
        "bayesian_records": [],
        "funnel_output": SimpleNamespace(candidates=["000001.SZ"], excluded_symbols={}, funnel_metadata={}),
    }
    monkeypatch.setattr(mainline_module, "_execute_market_dag", lambda **kwargs: dag_artifacts)

    result = quant_investor.QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()

    assert result.architecture_version == "12.0.0-stable"
    assert result.branch_schema_version == "branch-schema.v12.unified-mainline"
    assert result.calibration_schema_version
    assert result.debate_template_version
    assert result.final_strategy.architecture_version == result.architecture_version
