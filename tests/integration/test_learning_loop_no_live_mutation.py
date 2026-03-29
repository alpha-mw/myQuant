from __future__ import annotations

from dataclasses import dataclass

from quant_investor.learning.learning_orchestrator import LearningOrchestrator
from quant_investor.learning.memory_promoter import PromotionCandidate, PromotionDecision, RuleProposal
from quant_investor.learning.pre_trade_recall import RecallPacket, RecallQuery


@dataclass
class _TradeCaseStub:
    case_id: str


def test_learning_loop_only_emits_recall_and_proposal(monkeypatch, tmp_path):
    orchestrator = LearningOrchestrator(base_dir=tmp_path)
    live_decision = {
        "action": "buy",
        "target_weights": {"000001.SZ": 0.35},
    }
    recall_query = RecallQuery(
        symbol="000001.SZ",
        as_of=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        market_regime="neutral",
        sector="bank",
        branch_support_pattern="mixed",
        consensus_count=1,
        candidate_action="watch",
        volatility_regime="normal",
    )
    recall_packet = RecallPacket(symbol="000001.SZ", query=recall_query)

    monkeypatch.setattr(orchestrator, "save_trade_case", lambda trade_case: tmp_path / f"{trade_case.case_id}.json")
    monkeypatch.setattr(orchestrator, "_case_has_observed_outcome", lambda _trade_case: True)
    monkeypatch.setattr(orchestrator, "index_case", lambda _case_id: {"memory_item_id": "memory-1"})
    monkeypatch.setattr(orchestrator, "reflect_case", lambda _case_id: {"reflection_report_id": "reflection-1"})
    monkeypatch.setattr(
        orchestrator,
        "run_promotion_cycle",
        lambda: (
            [
                PromotionCandidate(
                    candidate_id="candidate-1",
                    source_case_ids=["case-1"],
                    lesson_statement="respect veto",
                    lesson_type="risk",
                    support_count=2,
                    counter_count=0,
                    regimes_seen=["neutral"],
                    sectors_seen=["bank"],
                    confidence=0.8,
                    status="candidate_lesson",
                    evidence_summary="stable",
                )
            ],
            [
                PromotionDecision(
                    candidate_id="candidate-1",
                    target_status="validated_pattern",
                    decision="validate_pattern",
                    reason="sufficient support",
                    support_count=2,
                    counter_count=0,
                    confidence=0.8,
                    missing_requirements=[],
                    rule_proposal_recommended=True,
                )
            ],
            [
                RuleProposal(
                    proposal_id="proposal-1",
                    proposal_type="risk_guard_update",
                    suggestion="tighten veto recall",
                    evidence="case evidence",
                    expected_effect="safer live review",
                )
            ],
        ),
    )
    monkeypatch.setattr(orchestrator, "build_recall_packet", lambda _context, top_k=5: recall_packet)

    result = orchestrator.run_closed_loop(
        _TradeCaseStub(case_id="case-1"),
        recall_context={"symbol": "000001.SZ", "live_decision": live_decision},
        top_k=3,
    )

    assert isinstance(result["recall_packet"], RecallPacket)
    assert isinstance(result["rule_proposals"][0], RuleProposal)
    assert result["live_updates"] == []
    assert live_decision == {
        "action": "buy",
        "target_weights": {"000001.SZ": 0.35},
    }
