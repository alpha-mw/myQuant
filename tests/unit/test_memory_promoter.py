"""
Memory promoter 单元测试。
"""

from __future__ import annotations

from datetime import datetime

from quant_investor.learning.memory_promoter import MemoryPromoter
from quant_investor.learning.post_trade_reflector import PostTradeReflector
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
    sector: str = "financials",
    market_regime: str = "risk_on",
    t1_return: float | None = 0.01,
    t5_return: float | None = 0.05,
    t10_return: float | None = 0.07,
    t20_return: float | None = 0.09,
    mae: float | None = -0.02,
    stop_loss_hit: bool | None = False,
    correct_risk_controls: list[str] | None = None,
    missed_risks: list[str] | None = None,
    error_tags: list[str] | None = None,
) -> TradeCase:
    return TradeCase(
        case_id=case_id,
        symbol="000001.SZ",
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
            t1_return=t1_return,
            t5_return=t5_return,
            t10_return=t10_return,
            t20_return=t20_return,
            mfe=0.12 if t20_return is not None and t20_return > 0 else 0.02,
            mae=mae,
            stop_loss_hit=stop_loss_hit,
            take_profit_hit=True if t20_return is not None and t20_return > 0.08 else False,
            outcome_status="closed" if t5_return is not None else "pending",
        ),
        attribution=AttributionSnapshot(
            helpful_agents=["QuantAgent"] if (t5_return or 0) > 0 else [],
            misleading_agents=["KlineAgent"] if (t5_return or 0) < 0 else [],
            correct_risk_controls=correct_risk_controls or ["gross_exposure_cap"],
            missed_risks=missed_risks or [],
            summary="memory promoter 测试样本。",
        ),
        error_tags=error_tags or [],
        lesson_draft=[],
        memory_status="raw",
        metadata={
            "style_bias": "balanced_quality",
            "sector": sector,
            "volatility_regime": "medium",
        },
    )


def _reflect_cases(cases: list[TradeCase]):
    reflector = PostTradeReflector()
    return [reflector.reflect_case(case) for case in cases]


def test_single_case_lesson_will_not_become_approved_rule_candidate():
    promoter = MemoryPromoter()
    cases = [
        _build_trade_case(
            case_id="case-single-risk",
            t1_return=-0.03,
            t5_return=-0.08,
            t10_return=-0.09,
            t20_return=-0.1,
            mae=-0.12,
            correct_risk_controls=[],
            missed_risks=["liquidity_gap"],
        )
    ]
    reports = _reflect_cases(cases)

    candidates = promoter.collect_candidates(cases, reports)
    risk_candidate = next(item for item in candidates if item.lesson_type == "risk_rule_candidate")
    decision = promoter.validate_candidate(risk_candidate, cases)
    promoted = promoter.promote(risk_candidate, decision)

    assert risk_candidate.support_count == 1
    assert promoted.status != "approved_rule_candidate"
    assert decision.target_status in {"candidate_lesson", "raw_case"}


def test_multi_case_support_with_counter_analysis_can_become_validated_pattern():
    promoter = MemoryPromoter()
    cases = [
        _build_trade_case(
            case_id="case-risk-1",
            sector="financials",
            t1_return=-0.03,
            t5_return=-0.08,
            t10_return=-0.09,
            t20_return=-0.1,
            mae=-0.12,
            correct_risk_controls=[],
            missed_risks=["liquidity_gap"],
        ),
        _build_trade_case(
            case_id="case-risk-2",
            sector="financials",
            t1_return=-0.02,
            t5_return=-0.06,
            t10_return=-0.07,
            t20_return=-0.09,
            mae=-0.09,
            correct_risk_controls=[],
            missed_risks=["event_gap"],
        ),
        _build_trade_case(
            case_id="case-counter-1",
            sector="financials",
            t1_return=0.01,
            t5_return=0.04,
            t10_return=0.05,
            t20_return=0.06,
            mae=-0.02,
            correct_risk_controls=["gross_exposure_cap"],
            missed_risks=[],
        ),
    ]
    reports = _reflect_cases(cases)

    candidates = promoter.collect_candidates(cases, reports)
    risk_candidate = next(item for item in candidates if item.lesson_type == "risk_rule_candidate")
    decision = promoter.validate_candidate(risk_candidate, cases)
    promoted = promoter.promote(risk_candidate, decision)

    assert risk_candidate.support_count == 2
    assert risk_candidate.counter_count >= 1
    assert decision.target_status == "validated_pattern"
    assert promoted.status == "validated_pattern"


def test_candidate_without_counter_analysis_cannot_be_promoted_to_high_level():
    promoter = MemoryPromoter()
    cases = [
        _build_trade_case(
            case_id="case-no-counter-1",
            sector="industrials",
            t1_return=-0.03,
            t5_return=-0.08,
            t10_return=-0.09,
            t20_return=-0.1,
            mae=-0.12,
            correct_risk_controls=[],
            missed_risks=["liquidity_gap"],
        ),
        _build_trade_case(
            case_id="case-no-counter-2",
            sector="industrials",
            t1_return=-0.02,
            t5_return=-0.06,
            t10_return=-0.07,
            t20_return=-0.09,
            mae=-0.09,
            correct_risk_controls=[],
            missed_risks=["event_gap"],
        ),
    ]
    reports = _reflect_cases(cases)

    candidates = promoter.collect_candidates(cases, reports)
    risk_candidate = next(item for item in candidates if item.lesson_type == "risk_rule_candidate")
    decision = promoter.validate_candidate(risk_candidate, cases)

    assert risk_candidate.support_count == 2
    assert risk_candidate.counter_count == 0
    assert decision.target_status == "candidate_lesson"
    assert "counter_count<1" in decision.missing_requirements


def test_rule_proposal_is_generated_and_defaults_to_pending():
    promoter = MemoryPromoter()
    cases = [
        _build_trade_case(
            case_id="case-proposal-1",
            sector="financials",
            t1_return=-0.03,
            t5_return=-0.08,
            t10_return=-0.09,
            t20_return=-0.1,
            mae=-0.12,
            correct_risk_controls=[],
            missed_risks=["liquidity_gap"],
        ),
        _build_trade_case(
            case_id="case-proposal-2",
            sector="financials",
            t1_return=-0.02,
            t5_return=-0.06,
            t10_return=-0.07,
            t20_return=-0.09,
            mae=-0.09,
            correct_risk_controls=[],
            missed_risks=["event_gap"],
        ),
        _build_trade_case(
            case_id="case-proposal-counter",
            sector="financials",
            t1_return=0.01,
            t5_return=0.04,
            t10_return=0.05,
            t20_return=0.06,
            mae=-0.02,
            correct_risk_controls=["gross_exposure_cap"],
            missed_risks=[],
        ),
    ]
    reports = _reflect_cases(cases)

    candidates = promoter.collect_candidates(cases, reports)
    risk_candidate = next(item for item in candidates if item.lesson_type == "risk_rule_candidate")
    decision = promoter.validate_candidate(risk_candidate, cases)
    promoted = promoter.promote(risk_candidate, decision)
    proposal = promoter.build_rule_proposal(promoted)

    assert proposal.proposal_type == "risk_guard_update"
    assert proposal.approval_status == "pending"
    assert "RiskGuard" in proposal.suggestion
    assert proposal.evidence
