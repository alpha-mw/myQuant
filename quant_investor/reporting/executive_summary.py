"""
执行摘要构建器。
"""

from __future__ import annotations

from typing import Mapping, Sequence

from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan


def confidence_label(confidence: float) -> str:
    """把连续置信度映射成中文标签。"""

    if confidence >= 0.75:
        return "高"
    if confidence >= 0.45:
        return "中"
    return "低"


class ExecutiveSummaryBuilder:
    """生成固定 3 条执行摘要。"""

    def __init__(
        self,
        macro_verdict: BranchVerdict,
        branch_summaries: Mapping[str, BranchVerdict | Mapping[str, object]],
        ic_decisions: Sequence[ICDecision],
        portfolio_plan: PortfolioPlan,
    ) -> None:
        self.macro_verdict = macro_verdict
        self.branch_summaries = dict(branch_summaries)
        self.ic_decisions = list(ic_decisions)
        self.portfolio_plan = portfolio_plan

    def build(self) -> list[str]:
        regime = str(self.macro_verdict.metadata.get("regime", "neutral"))
        style_bias = str(self.macro_verdict.metadata.get("style_bias", "balanced"))
        gross = float(self.portfolio_plan.target_gross_exposure)
        net = float(self.portfolio_plan.target_net_exposure)
        position_count = len(self.portfolio_plan.target_positions)
        turnover = float(self.portfolio_plan.turnover_estimate)
        avg_confidence = self._average_confidence()
        ic_action = self._aggregate_ic_action()

        return [
            (
                f"当前市场 regime 为 {regime}，目标总暴露控制在 {gross:.1%}，"
                f"净暴露约 {net:.1%}。"
            ),
            (
                f"IC 当前建议动作以 {ic_action} 为主，组合风格偏 {style_bias}，"
                f"本轮计划配置 {position_count} 只标的。"
            ),
            (
                f"跨分支平均可信度为{confidence_label(avg_confidence)}，"
                f"预估换手约 {turnover:.1%}，执行应保持纪律化。"
            ),
        ]

    def _average_confidence(self) -> float:
        values: list[float] = []
        for branch in self.branch_summaries.values():
            if isinstance(branch, BranchVerdict):
                values.append(float(branch.final_confidence))
                continue
            if isinstance(branch, Mapping):
                values.append(float(branch.get("final_confidence", branch.get("confidence", 0.0))))
        if self.macro_verdict is not None:
            values.append(float(self.macro_verdict.final_confidence))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _aggregate_ic_action(self) -> str:
        if not self.ic_decisions:
            return "观察"
        priorities = {
            "avoid": 0,
            "sell": 1,
            "watch": 2,
            "hold": 3,
            "buy": 4,
        }
        action = min(
            (decision.action.value for decision in self.ic_decisions),
            key=lambda item: priorities.get(item, 2),
        )
        return {
            "avoid": "回避",
            "sell": "回避",
            "watch": "观察",
            "hold": "持有",
            "buy": "买入",
        }.get(action, "观察")
