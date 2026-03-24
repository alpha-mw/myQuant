"""
规则化 ICCoordinator。
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ICDecision,
    RiskDecision,
)
from quant_investor.agents.base import BaseAgent


class ICCoordinator(BaseAgent):
    """只基于结构化 branch verdicts 和 risk decision 做一致性协调。"""

    agent_name = "ICCoordinator"
    _FORBIDDEN_KEYS = {
        "market_snapshot",
        "symbol_data",
        "data_bundle",
        "ohlcv",
        "fundamentals",
        "event_data",
        "sentiment_data",
    }

    def run(self, payload: Mapping[str, Any]) -> ICDecision:
        envelope = self.ensure_payload(payload)
        self.require_keys(envelope, "branch_verdicts", "risk_decision")
        forbidden = sorted(key for key in self._FORBIDDEN_KEYS if key in envelope and envelope[key] is not None)
        if forbidden:
            raise ValueError(f"ICCoordinator 不允许直接读取原始市场大数据: {', '.join(forbidden)}")

        branch_verdicts = self._normalize_branch_verdicts(envelope["branch_verdicts"])
        risk_decision = envelope["risk_decision"]
        if not isinstance(risk_decision, RiskDecision):
            raise TypeError("risk_decision 必须是 RiskDecision")

        if not branch_verdicts:
            raise ValueError("ICCoordinator 至少需要一个 BranchVerdict")

        agreement_points = self._build_agreement_points(branch_verdicts)
        conflict_points = self._build_conflict_points(branch_verdicts)

        scores = [verdict.final_score for verdict in branch_verdicts.values()]
        confidences = [verdict.final_confidence for verdict in branch_verdicts.values()]
        raw_score = self.clamp(fmean(scores), -1.0, 1.0)
        raw_confidence = self.clamp(fmean(confidences), 0.0, 1.0)
        raw_action = self.score_to_action(raw_score)
        action = self.clamp_action_to_cap(raw_action, risk_decision.action_cap)

        if risk_decision.veto:
            status = AgentStatus.VETOED
        elif conflict_points:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.SUCCESS

        thesis = self._build_thesis(
            agreement_points=agreement_points,
            conflict_points=conflict_points,
            action=action,
            risk_decision=risk_decision,
        )

        rationale_points = []
        rationale_points.extend(agreement_points[:2])
        rationale_points.extend(conflict_points[:2])
        rationale_points.extend(risk_decision.reasons[:2])

        return ICDecision(
            status=status,
            thesis=thesis,
            direction=self.score_to_direction(raw_score),
            action=action,
            confidence_label=self.confidence_to_label(raw_confidence),
            final_score=raw_score,
            final_confidence=raw_confidence,
            agreement_points=agreement_points,
            conflict_points=conflict_points,
            rationale_points=rationale_points,
            metadata={
                "raw_action": raw_action.value,
                "action_cap": risk_decision.action_cap.value,
                "action_cap_applied": action != raw_action,
                "risk_veto": risk_decision.veto,
            },
        )

    @staticmethod
    def _normalize_branch_verdicts(payload: Any) -> dict[str, BranchVerdict]:
        if isinstance(payload, Mapping):
            return {
                str(name): verdict
                for name, verdict in payload.items()
                if isinstance(verdict, BranchVerdict)
            }
        raise TypeError("branch_verdicts 必须是 Mapping[str, BranchVerdict]")

    @staticmethod
    def _build_agreement_points(branch_verdicts: Mapping[str, BranchVerdict]) -> list[str]:
        bullish = [name for name, verdict in branch_verdicts.items() if verdict.direction.value == "bullish"]
        bearish = [name for name, verdict in branch_verdicts.items() if verdict.direction.value == "bearish"]
        neutral = [name for name, verdict in branch_verdicts.items() if verdict.direction.value == "neutral"]

        points: list[str] = []
        if len(bullish) >= 2:
            points.append(f"一致偏多分支: {', '.join(sorted(bullish))}")
        if len(bearish) >= 2:
            points.append(f"一致偏空分支: {', '.join(sorted(bearish))}")
        if len(neutral) == len(branch_verdicts):
            points.append("全部分支暂未形成明显方向性结论。")
        if not points:
            points.append("至少存在局部共识，但尚未形成全局单边一致。")
        return points

    @staticmethod
    def _build_conflict_points(branch_verdicts: Mapping[str, BranchVerdict]) -> list[str]:
        directions = {verdict.direction.value for verdict in branch_verdicts.values()}
        conflicts: list[str] = []
        if "bullish" in directions and "bearish" in directions:
            conflicts.append("分支方向出现多空对冲，需服从更保守的动作约束。")
        degraded = [name for name, verdict in branch_verdicts.items() if verdict.status != AgentStatus.SUCCESS]
        if degraded:
            conflicts.append(f"部分分支为非成功状态: {', '.join(sorted(degraded))}")
        return conflicts

    @staticmethod
    def _build_thesis(
        agreement_points: list[str],
        conflict_points: list[str],
        action: ActionLabel,
        risk_decision: RiskDecision,
    ) -> str:
        head = agreement_points[0] if agreement_points else "研究层未形成强共识"
        risk_tail = (
            f"RiskGuard 已触发硬约束，动作上限收敛到 {risk_decision.action_cap.value}"
            if risk_decision.veto
            else f"动作需服从 RiskGuard 的上限 {risk_decision.action_cap.value}"
        )
        if conflict_points:
            return f"{head}；但存在分支冲突，因此 {risk_tail}，IC 建议当前执行 {action.value}。"
        return f"{head}；{risk_tail}，IC 建议当前执行 {action.value}。"
