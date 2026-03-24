"""
设计意图 3：NarratorAgent 不参与交易决策。

当前预期：组件语义通过。
"""

from __future__ import annotations

from copy import deepcopy

from quant_investor.agent_protocol import ActionLabel, BranchVerdict, ICDecision, PortfolioPlan
from quant_investor.agents.narrator_agent import NarratorAgent


def _macro_verdict() -> BranchVerdict:
    return BranchVerdict(
        agent_name="MacroAgent",
        thesis="宏观环境中性偏稳。",
        final_score=0.08,
        final_confidence=0.82,
        metadata={"regime": "balanced", "target_gross_exposure": 0.45, "style_bias": "balanced_quality"},
    )


def _branch_summaries() -> dict[str, BranchVerdict]:
    return {
        "kline": BranchVerdict(
            agent_name="KlineAgent",
            thesis="趋势仍偏正。",
            final_score=0.24,
            final_confidence=0.58,
            investment_risks=["短线波动放大。"],
        ),
        "quant": BranchVerdict(
            agent_name="QuantAgent",
            thesis="量化因子保持正向。",
            final_score=0.31,
            final_confidence=0.74,
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
                        "action": "buy",
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


def test_narrator_agent_is_read_only_and_does_not_rewrite_decisions() -> None:
    macro = _macro_verdict()
    branches = _branch_summaries()
    ic_decisions = _ic_decisions()
    portfolio_plan = _portfolio_plan()

    ic_before = deepcopy(ic_decisions)
    plan_before = deepcopy(portfolio_plan)

    bundle = NarratorAgent().run(
        {
            "macro_verdict": macro,
            "branch_summaries": branches,
            "ic_decisions": ic_decisions,
            "portfolio_plan": portfolio_plan,
            "run_diagnostics": [],
        }
    )

    assert ic_decisions == ic_before
    assert portfolio_plan == plan_before
    assert bundle.metadata["narrator_read_only"] is True
    assert bundle.portfolio_plan == plan_before
    cards = {card["symbol"]: card for card in bundle.stock_cards}
    assert cards["600519.SH"]["target_weight"] == 0.12
    assert cards["000001.SZ"]["target_weight"] == 0.08
    assert bundle.ic_decisions[0].action is ActionLabel.BUY

