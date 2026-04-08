from __future__ import annotations

import asyncio
from types import SimpleNamespace

import quant_investor.pipeline.mainline as mainline_module
from quant_investor.agent_protocol import ActionLabel, AgentStatus, ExecutionTrace, GlobalContext, PortfolioDecision, ShortlistItem, WhatIfPlan
from quant_investor.agents.agent_contracts import MasterAgentOutput
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.pipeline.mainline import QuantInvestor


def test_timeout_budget_helpers_stay_available():
    assert AgentOrchestrator.compute_outer_timeout(
        30.0,
        max_retries=2,
        cushion_seconds=10.0,
    ) == 71.0
    assert AgentOrchestrator.branch_request_timeout("kline", 30.0) == 45.0
    assert AgentOrchestrator.branch_max_tokens("kline", 1000) == 600
    assert AgentOrchestrator.compute_recommended_total_timeout(
        timeout_per_agent=30.0,
        master_timeout=60.0,
        existing_total_timeout=120.0,
    ) >= 120.0


def test_mainline_forwards_recall_context_to_unified_dag(monkeypatch):
    captured: dict[str, object] = {}
    recall_context = {
        "recent_markets": ["CN"],
        "recent_symbols": ["000001.SZ"],
        "top_picks": [{"symbol": "000001.SZ", "action": "buy"}],
    }

    def _fake_execute_market_dag(**kwargs):
        captured.update(kwargs)
        shortlist = [
            ShortlistItem(
                symbol="000001.SZ",
                company_name="平安银行",
                action=ActionLabel.BUY,
                confidence=0.8,
                suggested_weight=0.2,
            )
        ]
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
            "portfolio_decision": PortfolioDecision(
                status=AgentStatus.SUCCESS,
                shortlist=shortlist,
                target_exposure=0.2,
                target_gross_exposure=0.2,
                target_net_exposure=0.2,
                cash_ratio=0.8,
                target_weights={"000001.SZ": 0.2},
                target_positions={"000001.SZ": 200000.0},
            ),
            "portfolio_plan": SimpleNamespace(
                target_weights={"000001.SZ": 0.2},
                target_positions={"000001.SZ": 200000.0},
                position_limits={"000001.SZ": 0.2},
                blocked_symbols=[],
                rejected_symbols=[],
                execution_notes=[],
                target_exposure=0.2,
                target_gross_exposure=0.2,
                target_net_exposure=0.2,
                cash_ratio=0.8,
            ),
            "report_bundle": SimpleNamespace(
                markdown_report="# report",
                headline="headline",
                summary="summary",
                executive_summary=[],
                market_view=[],
                branch_verdicts={},
                macro_verdict=None,
                portfolio_plan=SimpleNamespace(
                    target_weights={"000001.SZ": 0.2},
                    target_positions={"000001.SZ": 200000.0},
                    position_limits={"000001.SZ": 0.2},
                    blocked_symbols=[],
                    rejected_symbols=[],
                    execution_notes=[],
                ),
                execution_trace=ExecutionTrace(),
                what_if_plan=WhatIfPlan(),
            ),
            "branch_results": {},
            "branch_summaries": {},
            "branch_verdicts_by_symbol": {},
            "shortlist": shortlist,
            "review_bundle": SimpleNamespace(ic_hints_by_symbol={}, fallback_reasons=[]),
            "bayesian_records": [],
            "funnel_output": SimpleNamespace(candidates=["000001.SZ"], excluded_symbols={}),
            "execution_trace": ExecutionTrace(),
            "what_if_plan": WhatIfPlan(),
        }

    monkeypatch.setattr(mainline_module, "_execute_market_dag", _fake_execute_market_dag, raising=False)

    investor = QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        enable_agent_layer=True,
        agent_model="deepseek-chat",
        master_model="deepseek-chat",
        recall_context=recall_context,
        verbose=False,
    )
    investor.run()

    assert captured["recall_context"] == recall_context
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["master_model"] == "deepseek-chat"


def test_extract_common_fields_keeps_scalar_branch_signals_and_recall_context():
    orchestrator = AgentOrchestrator(branch_model="deepseek-chat", master_model="deepseek-chat")
    branch_result = BranchResult(
        branch_name="kline",
        final_score=0.2,
        final_confidence=0.7,
        explanation="test",
        symbol_scores={"000001.SZ": 0.2},
        signals={
            "model_mode": "chronos",
            "volatility_percentile": 42.0,
            "predicted_return": {"000001.SZ": 0.03},
        },
    )

    payload = orchestrator._build_branch_overlay_packet(
        symbol="000001.SZ",
        branch_name="kline",
        branch_result=branch_result,
        calibrated_signal={"expected_return": 0.01},
        macro_summary={"market_regime": "neutral"},
        risk_summary={"risk_level": "medium"},
        recall_context={"recent_symbols": ["000001.SZ"]},
    )

    assert payload.branch_signals == {
        "model_mode": "chronos",
        "volatility_percentile": 42.0,
        "predicted_return": {"000001.SZ": 0.03},
        "expected_return": 0.01,
    }
    assert payload.metadata["recall_context"] == {"recent_symbols": ["000001.SZ"]}


def test_master_agent_input_receives_recall_context(monkeypatch):
    captured_input: dict[str, object] = {}

    class FakeMasterAgent:
        def __init__(self, **_kwargs):
            pass

        async def deliberate(self, agent_input):
            captured_input["payload"] = agent_input
            return MasterAgentOutput(final_conviction="buy", final_score=0.2, confidence=0.7)

    monkeypatch.setattr(mainline_module, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr("quant_investor.agents.orchestrator.MasterAgent", FakeMasterAgent)

    orchestrator = AgentOrchestrator(branch_model="deepseek-chat", master_model="deepseek-chat")
    branch_results = {
        "kline": BranchResult(
            branch_name="kline",
            final_score=0.2,
            final_confidence=0.7,
            symbol_scores={"000001.SZ": 0.2},
            explanation="kline thesis",
        )
    }

    result = asyncio.run(
        orchestrator._run_master_agent(
            branch_results=branch_results,
            risk_result=None,
            ensemble_output={"aggregate_score": 0.1},
            market_regime="neutral",
            candidate_symbols=["000001.SZ"],
            llm_client=SimpleNamespace(),
            recall_context={"top_picks": [{"symbol": "000001.SZ", "action": "buy"}]},
        )
    )

    assert result.final_conviction == "buy"
    assert captured_input["payload"].recall_context == {
        "top_picks": [{"symbol": "000001.SZ", "action": "buy"}]
    }
