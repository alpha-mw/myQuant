"""
NarratorAgent 报告层测试。
"""

from __future__ import annotations

from copy import deepcopy
import re

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    Direction,
    ICDecision,
    PortfolioPlan,
)
from quant_investor.agents.narrator_agent import NarratorAgent


def _macro_verdict() -> BranchVerdict:
    return BranchVerdict(
        agent_name="MacroAgent",
        thesis="宏观环境中性偏稳，适合控制总暴露。",
        final_score=0.08,
        final_confidence=0.82,
        metadata={
            "regime": "balanced",
            "target_gross_exposure": 0.45,
            "style_bias": "balanced_quality",
        },
    )


def _branch_summaries() -> dict[str, BranchVerdict]:
    return {
        "kline": BranchVerdict(
            agent_name="KlineAgent",
            thesis="趋势仍偏正，但高频深模型已回退到基础结果。",
            final_score=0.24,
            final_confidence=0.58,
            investment_risks=["短线波动放大。"],
            diagnostic_notes=["Could not infer frequency"],
        ),
        "quant": BranchVerdict(
            agent_name="QuantAgent",
            thesis="量化因子与预期收益维持正向。",
            final_score=0.31,
            final_confidence=0.74,
            investment_risks=["拥挤度回升。"],
        ),
        "fundamental": BranchVerdict(
            agent_name="FundamentalAgent",
            thesis="基本面质量稳定，但部分子模块存在覆盖缺口。",
            final_score=0.18,
            final_confidence=0.62,
            investment_risks=["provider_missing"],
            coverage_notes=["provider_missing", "盈利预测 18/30 标的已覆盖。"],
            diagnostic_notes=["ValueError: hidden stack should not surface"],
        ),
        "intelligence": BranchVerdict(
            agent_name="IntelligenceAgent",
            thesis="情绪与资金流中性偏正。",
            final_score=0.12,
            final_confidence=0.57,
            investment_risks=["事件催化兑现不及预期。"],
        ),
    }


def _ic_decisions() -> list[ICDecision]:
    return [
        ICDecision(
            thesis="消费与银行方向进入执行名单。",
            final_score=0.28,
            final_confidence=0.70,
            action=ActionLabel.BUY,
            selected_symbols=["600519.SH", "000001.SZ"],
            metadata={
                "symbol_candidates": [
                    {
                        "symbol": "600519.SH",
                        "action": "strong_buy",
                        "confidence": 0.82,
                        "one_line_conclusion": "600519.SH 当前更适合观察，等待更清晰确认。",
                    },
                    {
                        "symbol": "000001.SZ",
                        "action": "buy",
                        "confidence": 0.68,
                        "one_line_conclusion": "000001.SZ 当前进入轻仓试错区间。",
                    },
                ]
            },
        )
    ]


def _portfolio_plan() -> PortfolioPlan:
    return PortfolioPlan(
        target_exposure=0.20,
        target_gross_exposure=0.20,
        target_net_exposure=0.20,
        cash_ratio=0.80,
        target_positions={"600519.SH": 0.12, "000001.SZ": 0.08},
        rejected_symbols=["300750.SZ"],
        concentration_metrics={"top1_weight": 0.12, "hhi": 0.0208},
        turnover_estimate=0.11,
        construction_notes=["target_weight 仅由规则决定。"],
        metadata={"action_cap": "buy"},
    )


def test_narrator_agent_builds_structured_report_without_mutating_decisions():
    macro = _macro_verdict()
    branches = _branch_summaries()
    ic_decisions = _ic_decisions()
    portfolio_plan = _portfolio_plan()
    diagnostics = [
        "[INFO] batch finished",
        "Traceback: hidden stack",
    ]

    ic_before = deepcopy(ic_decisions)
    plan_before = deepcopy(portfolio_plan)

    bundle = NarratorAgent().run(
        {
            "macro_verdict": macro,
            "branch_summaries": branches,
            "ic_decisions": ic_decisions,
            "portfolio_plan": portfolio_plan,
            "run_diagnostics": diagnostics,
        }
    )

    assert ic_decisions == ic_before
    assert portfolio_plan == plan_before
    assert len(bundle.executive_summary) == 3
    assert bundle.markdown_report.startswith("# 投资研究执行报告")
    assert "## 三句话执行摘要" in bundle.markdown_report
    assert bundle.markdown_report.count("- 一句话结论:") == 2
    assert all(card["one_line_conclusion"].strip() for card in bundle.stock_cards)
    assert re.search(r"\d+ 条覆盖说明", bundle.markdown_report)
    assert re.search(r"\d+/\d+ 个分支", bundle.markdown_report)
    assert "Could not infer frequency" not in bundle.markdown_report
    assert "Traceback" not in bundle.markdown_report
    assert "ValueError:" not in bundle.markdown_report
    assert "[INFO]" not in bundle.markdown_report
    assert "provider_missing" not in bundle.markdown_report
    assert "snapshot_missing" not in bundle.markdown_report
    assert "投资风险包括 部分模块当前缺少覆盖" not in bundle.markdown_report


def test_narrator_agent_keeps_actions_and_weights_read_only_but_fixes_display_copy():
    bundle = NarratorAgent().run(
        {
            "macro_verdict": _macro_verdict(),
            "branch_summaries": _branch_summaries(),
            "ic_decisions": _ic_decisions(),
            "portfolio_plan": _portfolio_plan(),
            "run_diagnostics": [],
        }
    )

    cards = {card["symbol"]: card for card in bundle.stock_cards}
    assert cards["600519.SH"]["target_weight"] == 0.12
    assert cards["600519.SH"]["display_action"] == "买入"
    assert "观察" not in cards["600519.SH"]["one_line_conclusion"]
    assert cards["000001.SZ"]["display_action"] == "买入"
    assert "轻仓试错" not in cards["000001.SZ"]["one_line_conclusion"]


def test_narrator_agent_accepts_mapping_branch_summary_with_enum_values():
    bundle = NarratorAgent().run(
        {
            "macro_verdict": _macro_verdict(),
            "branch_summaries": {
                "quant": {
                    "agent_name": "QuantAgent",
                    "thesis": "量化分支结构化结论保持有效。",
                    "status": AgentStatus.SUCCESS,
                    "direction": Direction.BULLISH,
                    "action": ActionLabel.BUY,
                    "confidence_label": ConfidenceLabel.HIGH,
                    "final_score": 0.3,
                    "final_confidence": 0.7,
                }
            },
            "ic_decisions": _ic_decisions(),
            "portfolio_plan": _portfolio_plan(),
            "run_diagnostics": [],
        }
    )

    assert bundle.branch_verdicts["quant"].status is AgentStatus.SUCCESS
    assert bundle.branch_verdicts["quant"].direction is Direction.BULLISH
    assert bundle.branch_verdicts["quant"].action is ActionLabel.BUY
