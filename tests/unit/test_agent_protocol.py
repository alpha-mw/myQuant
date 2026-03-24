"""
统一 Agent 协议层单元测试。
"""

from __future__ import annotations

import pytest

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    CoverageScope,
    Direction,
    EvidenceItem,
    EventNote,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
    RiskLevel,
)


def test_agent_protocol_dataclasses_can_instantiate():
    evidence = EvidenceItem(
        source="QuantAgent",
        summary="Alpha 与动量因子同时转强。",
        direction=Direction.BULLISH,
        score=0.4,
        confidence=0.7,
        scope=CoverageScope.SYMBOL,
        symbols=["000001.SZ"],
    )
    event = EventNote(
        title="macro_cached",
        message="本轮宏观上下文复用缓存快照。",
        scope=CoverageScope.GLOBAL,
        risk_level=RiskLevel.LOW,
    )
    verdict = BranchVerdict(
        agent_name="QuantAgent",
        thesis="量价共振仍然支持偏多判断。",
        status=AgentStatus.SUCCESS,
        direction=Direction.BULLISH,
        action=ActionLabel.BUY,
        confidence_label=ConfidenceLabel.HIGH,
        final_score=0.55,
        final_confidence=0.8,
        evidence=[evidence],
        events=[event],
        investment_risks=["拥挤度回升需要继续监控。"],
        coverage_notes=["基本面覆盖不足未进入本分支结论。"],
        diagnostic_notes=["未触发降级路径。"],
    )
    risk = RiskDecision(
        status=AgentStatus.DEGRADED,
        risk_level=RiskLevel.HIGH,
        hard_veto=False,
        target_exposure_cap=0.6,
        blocked_symbols=["300750.SZ"],
        position_limits={"000001.SZ": 0.15},
        reasons=["宏观波动率抬升，先收缩总仓位。"],
        events=[event],
    )
    ic = ICDecision(
        status=AgentStatus.SUCCESS,
        thesis="保留高质量龙头，收缩高波动边际标的。",
        direction=Direction.BULLISH,
        action=ActionLabel.BUY,
        confidence_label=ConfidenceLabel.MEDIUM,
        final_score=0.3,
        final_confidence=0.65,
        selected_symbols=["000001.SZ"],
        rejected_symbols=["300750.SZ"],
        rationale_points=["风险调整后胜率更高。"],
    )
    plan = PortfolioPlan(
        status=AgentStatus.SUCCESS,
        target_exposure=0.6,
        cash_ratio=0.4,
        target_weights={"000001.SZ": 0.15, "600519.SH": 0.12},
        position_limits={"000001.SZ": 0.15, "600519.SH": 0.12},
        blocked_symbols=["300750.SZ"],
        execution_notes=["按稳定排序生成目标仓位。"],
    )
    bundle = ReportBundle(
        headline="组合维持偏多但总仓位受限",
        summary="研究共识偏多，但风险上限压低了目标暴露。",
        branch_verdicts={"quant": verdict},
        risk_decision=risk,
        ic_decision=ic,
        portfolio_plan=plan,
        highlights=["偏多结论来自量化与 K 线一致性。"],
        warnings=["高波动成长股暂不纳入。"],
        diagnostics=[event],
    )

    assert verdict.thesis
    assert risk.target_exposure_cap == 0.6
    assert ic.selected_symbols == ["000001.SZ"]
    assert plan.target_weights["000001.SZ"] == 0.15
    assert bundle.branch_verdicts["quant"].action is ActionLabel.BUY


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_branch_verdict_accepts_score_and_confidence_boundaries(score: float, confidence: float):
    verdict = BranchVerdict(
        agent_name="KlineAgent",
        thesis="边界值测试。",
        status=AgentStatus.SUCCESS,
        final_score=score,
        final_confidence=confidence,
    )

    assert verdict.final_score == score
    assert verdict.final_confidence == confidence


@pytest.mark.parametrize("score", [-1.01, 1.01])
def test_branch_verdict_rejects_score_out_of_range(score: float):
    with pytest.raises(ValueError, match=r"final_score"):
        BranchVerdict(
            agent_name="KlineAgent",
            thesis="非法分数测试。",
            status=AgentStatus.SUCCESS,
            final_score=score,
            final_confidence=0.5,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_branch_verdict_rejects_confidence_out_of_range(confidence: float):
    with pytest.raises(ValueError, match=r"final_confidence"):
        BranchVerdict(
            agent_name="KlineAgent",
            thesis="非法置信度测试。",
            status=AgentStatus.SUCCESS,
            final_score=0.1,
            final_confidence=confidence,
        )


@pytest.mark.parametrize("thesis", ["", "   "])
def test_branch_verdict_requires_non_empty_thesis(thesis: str):
    with pytest.raises(ValueError, match=r"thesis"):
        BranchVerdict(
            agent_name="FundamentalAgent",
            thesis=thesis,
            status=AgentStatus.SUCCESS,
            final_score=0.2,
            final_confidence=0.6,
        )


def test_status_unknown_is_not_allowed():
    with pytest.raises(ValueError, match=r"status"):
        RiskDecision(status="unknown", risk_level=RiskLevel.MEDIUM)
