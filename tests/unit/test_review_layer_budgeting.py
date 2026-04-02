from __future__ import annotations

import asyncio
from types import SimpleNamespace

import quant_investor.pipeline.mainline as mainline_module
from quant_investor.agents.agent_contracts import BaseBranchAgentOutput, MasterAgentOutput
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.pipeline.mainline import QuantInvestor


def test_timeout_budget_helpers_stay_available():
    orchestrator = AgentOrchestrator(branch_model="deepseek-chat", master_model="deepseek-chat")
    assert orchestrator.max_tokens_master == 3000
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


def test_mainline_passes_recall_context_to_review_layer(monkeypatch):
    captured_init: dict[str, float] = {}
    captured_enhance: dict[str, object] = {}

    class FakeReviewAgentOrchestrator:
        @classmethod
        def compute_recommended_total_timeout(cls, **kwargs):
            return 999.0

        def __init__(self, **kwargs):
            captured_init.update(kwargs)

        def enhance_sync(self, **kwargs):
            captured_enhance.update(kwargs)
            return SimpleNamespace(
                branch_agent_outputs={},
                agent_strategy=None,
                risk_agent_output=None,
                agent_layer_timings={},
            )

    monkeypatch.setattr(mainline_module, "ReviewAgentOrchestrator", FakeReviewAgentOrchestrator)
    monkeypatch.setattr(mainline_module, "has_provider_for_model", lambda _model: True)

    recall_context = {
        "recent_markets": ["CN"],
        "recent_symbols": ["000001.SZ"],
        "top_picks": [{"symbol": "000001.SZ", "action": "buy"}],
    }
    investor = QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        enable_agent_layer=True,
        agent_model="deepseek-chat",
        master_model="deepseek-chat",
        recall_context=recall_context,
        verbose=False,
    )
    monkeypatch.setattr(investor, "_build_ensemble_output", lambda _baseline: {})

    snapshot = SimpleNamespace(
        market_regime="neutral",
        branch_results={},
        calibrated_signals={},
        risk_result={},
        baseline_strategy=None,
        data_bundle=UnifiedDataBundle(market="CN", symbols=["000001.SZ"], symbol_data={"000001.SZ": {}}),
    )

    investor._run_review_layer(snapshot)

    assert captured_init["total_timeout"] == 999.0
    assert captured_enhance["recall_context"] == recall_context


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

    payload = orchestrator._extract_common_fields(
        branch_result,
        {"expected_return": 0.01},
        "neutral",
        recall_context={"recent_symbols": ["000001.SZ"]},
    )

    assert payload["branch_signals"] == {
        "model_mode": "chronos",
        "volatility_percentile": 42.0,
        "predicted_return": {"000001.SZ": 0.03},
    }
    assert payload["recall_context"] == {"recent_symbols": ["000001.SZ"]}


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
    branch_outputs = {
        "kline": BaseBranchAgentOutput(
            branch_name="kline",
            conviction="buy",
            conviction_score=0.2,
            confidence=0.7,
        )
    }

    result = asyncio.run(
        orchestrator._run_master_agent(
            branch_outputs=branch_outputs,
            risk_output=None,
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
