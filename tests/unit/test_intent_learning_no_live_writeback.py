"""
设计意图 9：learning 模块只能生成 recall / proposal，不能自动改 live 规则。

当前预期：通过。
"""

from __future__ import annotations

from datetime import datetime

from quant_investor.learning.learning_orchestrator import LearningOrchestrator
from quant_investor.learning.trade_case_store import (
    AttributionSnapshot,
    ExecutionSnapshot,
    HumanDecisionSnapshot,
    OutcomeSnapshot,
    PreTradeSnapshot,
    TradeCase,
)


def _build_trade_case(
    *,
    case_id: str = "case-learning-intent",
    symbol: str = "000001.SZ",
    market_regime: str = "risk_on",
    sector: str = "financials",
) -> TradeCase:
    return TradeCase(
        case_id=case_id,
        symbol=symbol,
        decision_time=datetime.fromisoformat("2026-03-24T09:30:00+08:00"),
        pretrade_snapshot=PreTradeSnapshot(
            market_regime=market_regime,
            target_gross_exposure=0.55,
            branch_scores={"kline": 0.6, "quant": 0.58},
            branch_confidences={"kline": 0.72, "quant": 0.7},
            ic_action="buy",
            ic_thesis="量价共振，建议中等强度参与。",
            support_agents=["KlineAgent", "QuantAgent"],
            dissenting_agents=[],
        ),
        human_decision=HumanDecisionSnapshot(
            human_action="executed",
            human_reason=["desk_review"],
            manual_override=False,
            override_direction=None,
            override_notes="",
        ),
        execution_snapshot=ExecutionSnapshot(
            executed=True,
            entry_price=12.34,
            entry_time=datetime.fromisoformat("2026-03-24T09:35:00+08:00"),
            quantity=1_000,
            manual_notes="execution notes",
        ),
        outcomes=OutcomeSnapshot(
            t1_return=0.01,
            t5_return=0.05,
            t10_return=0.07,
            t20_return=0.09,
            mfe=0.12,
            mae=-0.02,
            stop_loss_hit=False,
            take_profit_hit=True,
            outcome_status="closed",
        ),
        attribution=AttributionSnapshot(
            helpful_agents=["QuantAgent"],
            misleading_agents=[],
            correct_risk_controls=["gross_exposure_cap"],
            missed_risks=[],
            summary="learning intent 测试样本。",
        ),
        error_tags=[],
        lesson_draft=[],
        memory_status="raw",
        metadata={
            "style_bias": "balanced_quality",
            "sector": sector,
            "volatility_regime": "medium",
            "holding_horizon": "t20",
        },
    )


def test_learning_pipeline_only_emits_recall_and_pending_proposals(tmp_path) -> None:
    orchestrator = LearningOrchestrator(tmp_path)
    trade_case = _build_trade_case()

    result = orchestrator.run_closed_loop(
        trade_case,
        recall_context={
            "symbol": trade_case.symbol,
            "as_of": "2026-03-25T09:30:00+08:00",
            "market_regime": trade_case.pretrade_snapshot.market_regime,
            "sector": trade_case.metadata["sector"],
            "support_agents": trade_case.pretrade_snapshot.support_agents,
            "candidate_action": trade_case.pretrade_snapshot.ic_action,
            "volatility_regime": trade_case.metadata["volatility_regime"],
            "error_tags": [],
        },
        top_k=5,
    )

    packet = result["recall_packet"]
    assert result["live_updates"] == []
    assert all(proposal.approval_status == "pending" for proposal in result["rule_proposals"])
    assert packet is not None
    assert packet.summary_for_ic
    assert packet.summary_for_risk_guard
    assert not hasattr(packet, "target_weight")
    assert not hasattr(packet, "action")
