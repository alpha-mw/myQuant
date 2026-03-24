"""
Post-trade reflector 单元测试。
"""

from __future__ import annotations

from datetime import datetime

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
    human_action: str = "executed",
    t1_return: float | None = 0.01,
    t5_return: float | None = 0.05,
    t10_return: float | None = 0.07,
    t20_return: float | None = 0.09,
    mae: float | None = -0.02,
    stop_loss_hit: bool | None = False,
    correct_risk_controls: list[str] | None = None,
    missed_risks: list[str] | None = None,
    helpful_agents: list[str] | None = None,
    misleading_agents: list[str] | None = None,
    error_tags: list[str] | None = None,
) -> TradeCase:
    return TradeCase(
        case_id=case_id,
        symbol="000001.SZ",
        decision_time=datetime.fromisoformat("2026-03-24T09:30:00+08:00"),
        pretrade_snapshot=PreTradeSnapshot(
            market_regime="risk_on",
            target_gross_exposure=0.55,
            branch_scores={"kline": 0.6, "quant": 0.58},
            branch_confidences={"kline": 0.72, "quant": 0.7},
            ic_action="buy",
            ic_thesis="量价共振，建议中等强度参与。",
            support_agents=["KlineAgent", "QuantAgent"],
            dissenting_agents=[],
        ),
        human_decision=HumanDecisionSnapshot(
            human_action=human_action,
            human_reason=["desk_review"],
            manual_override=human_action == "overridden",
            override_direction="sell" if human_action == "overridden" else None,
            override_notes="人工逆向处理。" if human_action == "overridden" else "",
        ),
        execution_snapshot=ExecutionSnapshot(
            executed=human_action == "executed",
            entry_price=12.34 if human_action == "executed" else None,
            entry_time=(
                datetime.fromisoformat("2026-03-24T09:35:00+08:00")
                if human_action == "executed"
                else None
            ),
            quantity=1_000 if human_action == "executed" else None,
            manual_notes="execution notes",
        ),
        outcomes=OutcomeSnapshot(
            t1_return=t1_return,
            t5_return=t5_return,
            t10_return=t10_return,
            t20_return=t20_return,
            mfe=0.12 if t20_return is not None and t20_return > 0 else 0.01,
            mae=mae,
            stop_loss_hit=stop_loss_hit,
            take_profit_hit=True if t20_return is not None and t20_return > 0.08 else False,
            outcome_status="closed" if t5_return is not None else "pending",
        ),
        attribution=AttributionSnapshot(
            helpful_agents=helpful_agents or ["QuantAgent"],
            misleading_agents=misleading_agents or [],
            correct_risk_controls=correct_risk_controls or ["gross_exposure_cap"],
            missed_risks=missed_risks or [],
            summary="本案用于 deterministic 复盘测试。",
        ),
        error_tags=error_tags or [],
        lesson_draft=[],
        memory_status="raw",
        metadata={"style_bias": "balanced_quality"},
    )


def test_profitable_case_generates_non_empty_reflection_report():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(case_id="case-win")

    report = reflector.reflect_case(trade_case)

    assert report.case_id == "case-win"
    assert report.symbol == "000001.SZ"
    assert report.thesis_validation == "correct"
    assert report.timing_assessment in {"good", "acceptable"}
    assert report.risk_control_assessment == "good"
    assert report.summary
    assert report.evidence


def test_loss_case_generates_failure_factors_and_error_tag_suggestions():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(
        case_id="case-loss",
        t1_return=-0.03,
        t5_return=-0.08,
        t10_return=-0.09,
        t20_return=-0.1,
        mae=-0.12,
        stop_loss_hit=False,
        correct_risk_controls=[],
        missed_risks=["liquidity_gap"],
        misleading_agents=["KlineAgent"],
    )

    report = reflector.reflect_case(trade_case)

    assert report.thesis_validation == "incorrect"
    assert report.key_failure_factors
    assert any("missed_risks" in item or "风险控制" in item for item in report.key_failure_factors)
    assert report.suggested_error_tags
    assert "risk_control_gap" in report.suggested_error_tags


def test_overridden_case_can_be_assessed_as_harmful():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(
        case_id="case-override-harm",
        human_action="overridden",
        t1_return=0.02,
        t5_return=0.06,
        t10_return=0.08,
        t20_return=0.1,
        mae=-0.03,
    )

    report = reflector.reflect_case(trade_case)

    assert report.human_override_assessment == "harmful"
    assert any("人工偏离系统建议" in item for item in report.key_failure_factors)


def test_overridden_case_can_be_assessed_as_helpful():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(
        case_id="case-override-help",
        human_action="overridden",
        t1_return=-0.02,
        t5_return=-0.06,
        t10_return=-0.08,
        t20_return=-0.09,
        mae=-0.1,
        correct_risk_controls=[],
        missed_risks=["event_gap"],
    )

    report = reflector.reflect_case(trade_case)

    assert report.human_override_assessment == "helpful"
    assert any("避免了潜在损失" in item for item in report.key_success_factors)


def test_reflection_generates_non_empty_lesson_drafts():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(
        case_id="case-lessons",
        t1_return=0.0,
        t5_return=0.02,
        t10_return=0.03,
        t20_return=0.04,
        mae=-0.09,
        error_tags=["entry_window_blur"],
    )

    report = reflector.reflect_case(trade_case)

    assert report.lesson_drafts
    assert report.lesson_drafts[0].lesson_type == "case_lesson"
    assert all(item.statement for item in report.lesson_drafts)


def test_skipped_case_is_supported_without_rule_writeback():
    reflector = PostTradeReflector()
    trade_case = _build_trade_case(
        case_id="case-skipped",
        human_action="skipped",
        t1_return=0.01,
        t5_return=0.04,
        t10_return=0.05,
        t20_return=0.06,
        mae=-0.02,
    )

    report = reflector.reflect_case(trade_case)

    assert report.human_override_assessment == "harmful"
    assert report.lesson_drafts
    assert isinstance(report.suggested_error_tags, list)
