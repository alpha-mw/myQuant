from __future__ import annotations

from types import SimpleNamespace

import pytest

import quant_investor.pipeline.mainline as mainline_module
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
from quant_investor.pipeline.mainline import QuantInvestor


def _fake_dag_artifacts() -> dict[str, object]:
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
        branch_verdicts={
            "macro": BranchVerdict(
                agent_name="macro",
                thesis="macro ok",
                final_score=0.2,
                final_confidence=0.7,
            )
        },
        macro_verdict=BranchVerdict(
            agent_name="macro",
            thesis="macro ok",
            final_score=0.2,
            final_confidence=0.7,
        ),
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
        ic_hints_by_symbol={"000001.SZ": {"action": "buy", "score": 0.8, "confidence": 0.7}},
    )
    return {
        "global_context": GlobalContext(
            market="CN",
            universe_key="full_a",
            universe_symbols=["000001.SZ"],
            universe_tiers={
                "total": ["000001.SZ"],
                "researchable": ["000001.SZ"],
                "shortlistable": ["000001.SZ"],
                "final_selected": ["000001.SZ"],
            },
        ),
        "symbol_research_packets": {},
        "branch_verdicts_by_symbol": {
            "000001.SZ": {
                "macro": BranchVerdict(
                    agent_name="macro",
                    thesis="macro ok",
                    symbol="000001.SZ",
                    final_score=0.2,
                    final_confidence=0.7,
                )
            }
        },
        "branch_summaries": {
            "macro": BranchVerdict(
                agent_name="macro",
                thesis="macro ok",
                final_score=0.2,
                final_confidence=0.7,
            )
        },
        "macro_verdict": BranchVerdict(
            agent_name="macro",
            thesis="macro ok",
            final_score=0.2,
            final_confidence=0.7,
        ),
        "risk_decision": None,
        "ic_decisions": [],
        "shortlist": shortlist,
        "portfolio_plan": report_bundle.portfolio_plan,
        "portfolio_decision": portfolio_decision,
        "review_bundle": SimpleNamespace(
            branch_summaries={},
            ic_hints_by_symbol={"000001.SZ": {"action": "buy", "score": 0.8, "confidence": 0.7}},
            fallback_reasons=[],
        ),
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
        "branch_results": {
            "macro": BranchResult(
                branch_name="macro",
                final_score=0.2,
                final_confidence=0.7,
                symbol_scores={"000001.SZ": 0.2},
            )
        },
        "bayesian_records": [],
        "funnel_output": SimpleNamespace(candidates=["000001.SZ"], excluded_symbols={}, funnel_metadata={}),
    }


def test_quant_investor_run_uses_market_dag_not_legacy_research_core(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_execute_market_dag(**kwargs):
        captured.update(kwargs)
        return _fake_dag_artifacts()

    def _legacy_path_should_not_run(*_args, **_kwargs):
        raise AssertionError("legacy research core should not run")

    monkeypatch.setattr(mainline_module, "_execute_market_dag", _fake_execute_market_dag, raising=False)
    monkeypatch.setattr(QuantInvestor, "_run_research_core", _legacy_path_should_not_run)

    investor = QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        total_capital=1_000_000,
        enable_agent_layer=True,
        verbose=False,
    )
    result = investor.run()

    assert captured["symbols"] == ["000001.SZ"]
    assert captured["market"] == "CN"
    assert result.pipeline_mode == "bayesian"
    assert result.final_strategy.target_weights == {"000001.SZ": 0.25}
    assert result.final_strategy.candidate_symbols == ["000001.SZ"]
    assert result.agent_report_bundle.markdown_report == "# DAG Report"
    assert result.master_review_output is not None
