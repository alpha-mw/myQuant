from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import quant_investor.agents.stock_reviewers as stock_reviewers
import quant_investor.pipeline.mainline as mainline_module
from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict, Direction, ExecutionTrace, GlobalContext, PortfolioDecision, ShortlistItem, WhatIfPlan
from quant_investor.agents.agent_contracts import MasterAgentInput, MasterAgentOutput
from quant_investor.agents.master_agent import MasterAgent
import quant_investor.agents.llm_client as legacy_llm_client
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    BranchOverlayReviewer,
    MasterICAgent,
    MasterSymbolPacket,
)
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.pipeline.mainline import QuantInvestor


def _make_branch_result(branch_name: str, score: float, confidence: float) -> BranchResult:
    return BranchResult(
        branch_name=branch_name,
        final_score=score,
        final_confidence=confidence,
        conclusion=f"{branch_name} conclusion",
        explanation=f"{branch_name} explanation",
        investment_risks=[f"{branch_name} risk"],
        coverage_notes=[f"{branch_name} coverage"],
        diagnostic_notes=[f"{branch_name} diagnostic"],
        support_drivers=[f"{branch_name} support"],
        drag_drivers=[f"{branch_name} drag"],
        symbol_scores={"000001.SZ": score},
        signals={"signal": 1},
    )


def test_branch_overlay_reviewer_applies_bounded_adjustment(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "thesis": "overlay thesis",
            "score_delta": 0.5,
            "confidence_delta": -0.5,
            "agreement_points": ["agreement"],
            "conflict_points": ["conflict"],
            "missing_risks": ["missing risk"],
            "contradictions": ["contradiction"],
            "risk_flags": ["risk flag"],
        }

    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(stock_reviewers.LLMClient, "complete", _fake_complete)

    packet = BranchOverlayPacket(
        symbol="000001.SZ",
        branch_name="quant",
        base_score=0.2,
        base_confidence=0.6,
        thesis="base thesis",
        direction="neutral",
        action="hold",
        agreement_points=["agreement"],
        conflict_points=["conflict"],
        risk_points=["risk"],
        branch_signals={"signal": 1},
        macro_summary={"regime": "neutral"},
        risk_summary={"risk_level": "medium"},
    )
    reviewer = BranchOverlayReviewer(
        branch_name="quant",
        llm_client=stock_reviewers.LLMClient(),
        model="deepseek-chat",
        timeout=1.0,
    )

    verdict = asyncio.run(reviewer.review(packet))

    assert verdict.status == AgentStatus.SUCCESS
    assert verdict.score_delta == 0.10
    assert verdict.adjusted_score == pytest.approx(0.30)
    assert verdict.confidence_delta == -0.10
    assert verdict.adjusted_confidence == 0.50
    assert captured["stage"] == "review_branch_overlay"
    assert captured["actor_name"] == "000001.SZ:quant"
    assert verdict.telemetry.stage == "review_branch_overlay"
    assert verdict.telemetry.success is True
    assert verdict.telemetry.fallback is False


def test_branch_overlay_reviewer_forwards_fallback_model(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "thesis": "overlay thesis",
            "score_delta": 0.0,
            "confidence_delta": 0.0,
            "agreement_points": [],
            "conflict_points": [],
            "missing_risks": [],
            "contradictions": [],
            "risk_flags": [],
        }

    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(stock_reviewers.LLMClient, "complete", _fake_complete)

    reviewer = BranchOverlayReviewer(
        branch_name="kline",
        llm_client=stock_reviewers.LLMClient(),
        model="qwen-plus",
        fallback_model="moonshot-v1-128k",
        timeout=1.0,
    )
    asyncio.run(
        reviewer.review(
            BranchOverlayPacket(
                symbol="000001.SZ",
                branch_name="kline",
                base_score=0.1,
                base_confidence=0.4,
                thesis="base thesis",
                direction="neutral",
                action="hold",
            )
        )
    )

    assert captured["fallback_model"] == "moonshot-v1-128k"


def test_branch_overlay_reviewer_falls_back_when_provider_missing(monkeypatch):
    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: False)

    packet = BranchOverlayPacket(
        symbol="000001.SZ",
        branch_name="kline",
        base_score=0.1,
        base_confidence=0.4,
        thesis="base thesis",
        direction="neutral",
        action="hold",
        agreement_points=["agreement"],
        conflict_points=["conflict"],
        risk_points=["risk"],
        branch_signals={"signal": 1},
        macro_summary={"regime": "neutral"},
        risk_summary={"risk_level": "medium"},
    )
    reviewer = BranchOverlayReviewer(
        branch_name="kline",
        llm_client=stock_reviewers.LLMClient(),
        model="deepseek-chat",
        timeout=1.0,
    )

    verdict = asyncio.run(reviewer.review(packet))

    assert verdict.status == AgentStatus.DEGRADED
    assert verdict.telemetry.fallback is True
    assert verdict.telemetry.success is False
    assert verdict.telemetry.fallback_reason
    assert verdict.adjusted_score <= 0.2


def test_master_ic_agent_respects_bounded_adjustment_and_hard_veto(monkeypatch):
    async def _fake_complete(self, *args, **kwargs):
        return {
            "thesis": "master thesis",
            "score_delta": 0.9,
            "confidence_delta": 0.9,
            "agreement_points": ["agreement"],
            "conflict_points": ["conflict"],
            "rationale_points": ["rationale"],
            "risk_flags": ["risk flag"],
        }

    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(stock_reviewers.LLMClient, "complete", _fake_complete)

    packet = MasterSymbolPacket(
        symbol="000001.SZ",
        branch_overlay_summaries=[
            {
                "adjusted_score": 0.2,
                "adjusted_confidence": 0.6,
                "risk_flags": ["risk"],
            }
        ],
        macro_summary={"regime": "neutral"},
        risk_summary={"risk_flags": ["risk"], "conflicts": ["conflict"]},
        baseline_score=0.2,
        baseline_confidence=0.6,
        hard_veto=True,
    )
    agent = MasterICAgent(
        llm_client=stock_reviewers.LLMClient(),
        model="deepseek-chat",
        timeout=1.0,
    )

    hint = asyncio.run(agent.deliberate(packet))

    assert hint.status == AgentStatus.VETOED
    assert hint.score_delta == 0.18
    assert hint.confidence_delta == 0.12
    assert hint.score_hint <= 0.0
    assert hint.action in {ActionLabel.HOLD, ActionLabel.AVOID}
    assert hint.telemetry.stage == "review_master_symbol"
    assert hint.telemetry.success is True


def test_master_ic_agent_forwards_reasoning_effort(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "thesis": "master thesis",
            "score_delta": 0.0,
            "confidence_delta": 0.0,
            "agreement_points": [],
            "conflict_points": [],
            "rationale_points": [],
            "risk_flags": [],
        }

    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(stock_reviewers.LLMClient, "complete", _fake_complete)

    packet = MasterSymbolPacket(
        symbol="000001.SZ",
        branch_overlay_summaries=[
            {
                "adjusted_score": 0.2,
                "adjusted_confidence": 0.6,
                "risk_flags": [],
            }
        ],
        macro_summary={"regime": "neutral"},
        risk_summary={"risk_flags": [], "conflicts": []},
        baseline_score=0.2,
        baseline_confidence=0.6,
        hard_veto=False,
    )
    agent = MasterICAgent(
        llm_client=stock_reviewers.LLMClient(),
        model="moonshot-v1-128k",
        reasoning_effort="high",
        timeout=1.0,
    )

    hint = asyncio.run(agent.deliberate(packet))

    assert hint.status == AgentStatus.SUCCESS
    assert captured["reasoning_effort"] == "high"
    assert captured["stage"] == "review_master_symbol"
    assert captured["actor_name"] == "IC:000001.SZ"


def test_master_ic_agent_forwards_fallback_model(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "thesis": "master thesis",
            "score_delta": 0.0,
            "confidence_delta": 0.0,
            "agreement_points": [],
            "conflict_points": [],
            "rationale_points": [],
            "risk_flags": [],
        }

    monkeypatch.setattr(stock_reviewers, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(stock_reviewers.LLMClient, "complete", _fake_complete)

    agent = MasterICAgent(
        llm_client=stock_reviewers.LLMClient(),
        model="qwen-plus",
        fallback_model="moonshot-v1-128k",
        timeout=1.0,
    )
    asyncio.run(
        agent.deliberate(
            MasterSymbolPacket(
                symbol="000001.SZ",
                branch_overlay_summaries=[],
                macro_summary={},
                risk_summary={},
                baseline_score=0.0,
                baseline_confidence=0.5,
            )
        )
    )

    assert captured["fallback_model"] == "moonshot-v1-128k"


def test_legacy_master_agent_forwards_reasoning_effort(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_complete(self, *args, **kwargs):
        captured.update(kwargs)
        return {
            "final_conviction": "buy",
            "final_score": 0.2,
            "confidence": 0.7,
            "consensus_areas": [],
            "disagreement_areas": [],
            "debate_resolution": [],
            "top_picks": [],
            "portfolio_narrative": "narrative",
            "risk_adjusted_exposure": 0.5,
            "dissenting_views": [],
        }

    monkeypatch.setattr(legacy_llm_client.LLMClient, "complete", _fake_complete)

    agent = MasterAgent(
        llm_client=legacy_llm_client.LLMClient(),
        model="moonshot-v1-128k",
        reasoning_effort="high",
        timeout=1.0,
    )

    output = asyncio.run(
        agent.deliberate(
            MasterAgentInput(
                branch_reports={},
                risk_report=None,
                ensemble_baseline={"aggregate_score": 0.0},
                market_regime="neutral",
                candidate_symbols=[],
            )
        )
    )

    assert output.final_conviction == "buy"
    assert captured["reasoning_effort"] == "high"


def test_unified_dag_preserves_symbol_ic_hints_and_review_bundle(monkeypatch):
    investor = QuantInvestor(stock_pool=["000001.SZ"], enable_agent_layer=True, verbose=False)
    hints = {
        "000001.SZ": {
            "score": 0.42,
            "confidence": 0.88,
            "action": "buy",
            "rationale_points": ["symbol hint"],
        }
    }
    review_bundle = SimpleNamespace(ic_hints_by_symbol=hints, fallback_reasons=[])

    monkeypatch.setattr(
        mainline_module,
        "_execute_market_dag",
        lambda **_kwargs: {
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
                shortlist=[
                    ShortlistItem(
                        symbol="000001.SZ",
                        company_name="平安银行",
                        action=ActionLabel.BUY,
                        confidence=0.88,
                        suggested_weight=0.35,
                    )
                ],
                target_exposure=0.35,
                target_gross_exposure=0.35,
                target_net_exposure=0.35,
                cash_ratio=0.65,
                target_weights={"000001.SZ": 0.35},
                target_positions={"000001.SZ": 0.35},
            ),
            "portfolio_plan": SimpleNamespace(
                target_weights={"000001.SZ": 0.35},
                target_positions={"000001.SZ": 0.35},
                position_limits={"000001.SZ": 0.35},
                blocked_symbols=[],
                rejected_symbols=[],
                execution_notes=[],
                target_exposure=0.35,
                target_gross_exposure=0.35,
                target_net_exposure=0.35,
                cash_ratio=0.65,
            ),
            "report_bundle": SimpleNamespace(
                markdown_report="# report",
                headline="headline",
                summary="summary",
                executive_summary=["summary"],
                market_view=[],
                branch_verdicts={},
                macro_verdict=BranchVerdict(thesis="macro thesis"),
                coverage_summary=[],
                appendix_diagnostics=[],
                warnings=[],
                portfolio_plan=SimpleNamespace(
                    target_weights={"000001.SZ": 0.35},
                    target_positions={"000001.SZ": 0.35},
                    position_limits={"000001.SZ": 0.35},
                    blocked_symbols=[],
                    rejected_symbols=[],
                    execution_notes=[],
                ),
                execution_trace=ExecutionTrace(),
                what_if_plan=WhatIfPlan(),
                review_bundle=review_bundle,
                ic_hints_by_symbol=hints,
            ),
            "review_bundle": review_bundle,
            "branch_results": {"kline": _make_branch_result("kline", 0.4, 0.7)},
            "branch_summaries": {},
            "branch_verdicts_by_symbol": {},
            "shortlist": [],
            "bayesian_records": [],
            "funnel_output": SimpleNamespace(candidates=["000001.SZ"], excluded_symbols={}),
            "execution_trace": ExecutionTrace(),
            "what_if_plan": WhatIfPlan(),
            "portfolio_master_output": MasterAgentOutput(final_conviction="buy", final_score=0.2, confidence=0.7),
        },
        raising=False,
    )

    result = investor.run()

    assert result.ic_hints_by_symbol == hints
    assert result.review_bundle is review_bundle
    assert result.agent_report_bundle.review_bundle is review_bundle
    assert result.final_strategy.target_weights == {"000001.SZ": 0.35}
