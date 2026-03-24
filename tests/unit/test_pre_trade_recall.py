"""
Pre-trade recall 单元测试。
"""

from __future__ import annotations

from datetime import datetime

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.memory_promoter import MemoryPromoter, PromotionDecision
from quant_investor.learning.post_trade_reflector import PostTradeReflector
from quant_investor.learning.pre_trade_recall import PreTradeRecall
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
    case_id: str,
    symbol: str = "000001.SZ",
    market_regime: str = "risk_on",
    sector: str = "financials",
    support_agents: list[str] | None = None,
    t1_return: float | None = 0.01,
    t5_return: float | None = 0.05,
    t10_return: float | None = 0.07,
    t20_return: float | None = 0.09,
    mae: float | None = -0.02,
    missed_risks: list[str] | None = None,
    error_tags: list[str] | None = None,
) -> TradeCase:
    support = support_agents or ["KlineAgent", "QuantAgent"]
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
            support_agents=support,
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
            t1_return=t1_return,
            t5_return=t5_return,
            t10_return=t10_return,
            t20_return=t20_return,
            mfe=0.12 if t20_return is not None and t20_return > 0 else 0.02,
            mae=mae,
            stop_loss_hit=False,
            take_profit_hit=True if t20_return is not None and t20_return > 0.08 else False,
            outcome_status="closed" if t5_return is not None else "pending",
        ),
        attribution=AttributionSnapshot(
            helpful_agents=["QuantAgent"] if (t5_return or 0) > 0 else [],
            misleading_agents=["KlineAgent"] if (t5_return or 0) < 0 else [],
            correct_risk_controls=[] if missed_risks else ["gross_exposure_cap"],
            missed_risks=missed_risks or [],
            summary="pre-trade recall 测试样本。",
        ),
        error_tags=error_tags or [],
        lesson_draft=[],
        memory_status="raw",
        metadata={
            "style_bias": "balanced_quality",
            "sector": sector,
            "volatility_regime": "medium",
            "holding_horizon": "t20",
        },
    )


def _build_recall_fixture():
    cases = [
        _build_trade_case(case_id="case-similar-risk", missed_risks=["liquidity_gap"], t5_return=-0.08, t10_return=-0.09, t20_return=-0.1, mae=-0.12),
        _build_trade_case(case_id="case-similar-win", t5_return=0.05, t10_return=0.07, t20_return=0.09, mae=-0.02),
        _build_trade_case(case_id="case-different", market_regime="risk_off", support_agents=["MacroAgent"], sector="energy", t5_return=0.03, t10_return=0.04, t20_return=0.05),
    ]
    indexer = MemoryIndexer()
    memory_items = [indexer.build_memory_item_from_case(case) for case in cases]

    reflector = PostTradeReflector()
    reports = [reflector.reflect_case(case) for case in cases]
    promoter = MemoryPromoter()
    candidates = promoter.collect_candidates(cases, reports)
    updated_candidates = []
    for candidate in candidates:
        if candidate.lesson_type == "risk_rule_candidate":
            updated_candidates.append(
                promoter.promote(
                    candidate,
                    PromotionDecision(
                        candidate_id=candidate.candidate_id,
                        target_status="approved_rule_candidate",
                        decision="approve_rule_candidate",
                        reason="测试中手动标记为 pending rule candidate。",
                        support_count=max(candidate.support_count, 2),
                        counter_count=max(candidate.counter_count, 1),
                        confidence=max(candidate.confidence, 0.8),
                        missing_requirements=[],
                        rule_proposal_recommended=True,
                    ),
                )
            )
        elif candidate.lesson_type == "semantic_candidate":
            updated_candidates.append(
                promoter.promote(
                    candidate,
                    PromotionDecision(
                        candidate_id=candidate.candidate_id,
                        target_status="validated_pattern",
                        decision="validate_pattern",
                        reason="测试中手动标记为 validated pattern。",
                        support_count=candidate.support_count,
                        counter_count=candidate.counter_count,
                        confidence=candidate.confidence,
                        missing_requirements=[],
                        rule_proposal_recommended=False,
                    ),
                )
            )
        else:
            updated_candidates.append(candidate)

    recall = PreTradeRecall(
        memory_items=memory_items,
        promotion_candidates=updated_candidates,
    )
    query = recall.build_query(
        {
            "symbol": "000001.SZ",
            "as_of": "2026-03-25T09:30:00+08:00",
            "market_regime": "risk_on",
            "sector": "financials",
            "support_agents": ["QuantAgent", "KlineAgent"],
            "candidate_action": "buy",
            "volatility_regime": "medium",
            "error_tags": [],
        }
    )
    return recall, query


def test_pre_trade_recall_can_retrieve_similar_cases_by_regime_and_support_pattern():
    recall, query = _build_recall_fixture()

    packet = recall.retrieve(query, top_k=5)

    assert packet.similar_cases
    assert packet.similar_cases[0].source_case_ids[0] in {
        "case-similar-risk",
        "case-similar-win",
    }
    assert all(hit.tags["market_regime"] == "risk_on" for hit in packet.similar_cases[:2])
    assert all(
        hit.tags["branch_support_pattern"] == "klineagent+quantagent"
        for hit in packet.similar_cases[:2]
    )


def test_pre_trade_recall_prioritizes_risk_memory_over_opportunity_memory():
    recall, query = _build_recall_fixture()

    packet = recall.retrieve(query, top_k=5)

    assert packet.similar_cases[0].memory_type == "risk_case"
    assert packet.similar_cases[0].cautionary is True


def test_recall_packet_contains_required_summaries_and_no_direct_action_or_weight_output():
    recall, query = _build_recall_fixture()

    packet = recall.retrieve(query, top_k=5)

    assert packet.summary_for_ic
    assert packet.summary_for_risk_guard
    assert not hasattr(packet, "target_weight")
    assert not hasattr(packet, "action")
    assert "target_weight" not in packet.summary_for_ic
    assert "action=" not in packet.summary_for_ic
    assert "target_weight" not in packet.summary_for_risk_guard


def test_pre_trade_recall_exposes_validated_and_pending_patterns():
    recall, query = _build_recall_fixture()

    packet = recall.retrieve(query, top_k=5)

    assert packet.validated_patterns
    assert packet.pending_rule_candidates
    assert all(hit.status == "validated_pattern" for hit in packet.validated_patterns)
    assert all(hit.status == "approved_rule_candidate" for hit in packet.pending_rule_candidates)
