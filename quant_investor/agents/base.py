"""
控制层 Agent 的极简基类。

只提供统一 `run(payload)` 约定和少量公共工具，不引入复杂继承体系。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    CoverageScope,
    Direction,
    EvidenceItem,
)
from quant_investor.branch_contracts import BranchResult


class BaseAgent(ABC):
    """控制层与新协议层的最小公共基类。"""

    agent_name: str = "BaseAgent"

    def __init__(self, agent_name: str | None = None) -> None:
        if agent_name:
            self.agent_name = agent_name

    @abstractmethod
    def run(self, payload: Mapping[str, Any]) -> Any:
        """执行单次 agent 推理。"""

    @staticmethod
    def ensure_payload(payload: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload 必须是 Mapping")
        return dict(payload)

    @staticmethod
    def require_keys(payload: Mapping[str, Any], *keys: str) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise KeyError(f"payload 缺少必要字段: {', '.join(missing)}")

    @staticmethod
    def copy_value(value: Any) -> Any:
        return deepcopy(value)

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @classmethod
    def score_to_direction(cls, score: float) -> Direction:
        score = cls.clamp(score, -1.0, 1.0)
        if score >= 0.15:
            return Direction.BULLISH
        if score <= -0.15:
            return Direction.BEARISH
        return Direction.NEUTRAL

    @classmethod
    def score_to_action(cls, score: float) -> ActionLabel:
        score = cls.clamp(score, -1.0, 1.0)
        if score >= 0.25:
            return ActionLabel.BUY
        if score <= -0.35:
            return ActionLabel.SELL
        return ActionLabel.HOLD

    @staticmethod
    def confidence_to_label(confidence: float) -> ConfidenceLabel:
        confidence = max(0.0, min(1.0, float(confidence)))
        if confidence >= 0.85:
            return ConfidenceLabel.VERY_HIGH
        if confidence >= 0.65:
            return ConfidenceLabel.HIGH
        if confidence >= 0.4:
            return ConfidenceLabel.MEDIUM
        if confidence >= 0.2:
            return ConfidenceLabel.LOW
        return ConfidenceLabel.VERY_LOW

    @staticmethod
    def action_priority(action: ActionLabel | str) -> int:
        label = action if isinstance(action, ActionLabel) else ActionLabel(str(action).strip().lower())
        return {
            ActionLabel.AVOID: 0,
            ActionLabel.SELL: 1,
            ActionLabel.HOLD: 2,
            ActionLabel.WATCH: 2,
            ActionLabel.BUY: 3,
        }[label]

    @classmethod
    def more_restrictive_action(
        cls,
        left: ActionLabel | str,
        right: ActionLabel | str,
    ) -> ActionLabel:
        left_label = left if isinstance(left, ActionLabel) else ActionLabel(str(left).strip().lower())
        right_label = right if isinstance(right, ActionLabel) else ActionLabel(str(right).strip().lower())
        if cls.action_priority(left_label) <= cls.action_priority(right_label):
            return left_label
        return right_label

    @classmethod
    def clamp_action_to_cap(
        cls,
        action: ActionLabel | str,
        action_cap: ActionLabel | str,
    ) -> ActionLabel:
        action_label = action if isinstance(action, ActionLabel) else ActionLabel(str(action).strip().lower())
        cap_label = (
            action_cap if isinstance(action_cap, ActionLabel)
            else ActionLabel(str(action_cap).strip().lower())
        )
        if cls.action_priority(action_label) <= cls.action_priority(cap_label):
            return action_label
        return cap_label

    @staticmethod
    def top_symbols(symbol_scores: Mapping[str, float], limit: int = 5) -> list[str]:
        ranked = sorted(
            ((str(symbol), float(score)) for symbol, score in symbol_scores.items()),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        return [symbol for symbol, _ in ranked[:limit]]

    @staticmethod
    def dedupe_texts(items: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def is_coverage_like_note(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        keywords = (
            "provider_missing",
            "snapshot_missing",
            "provider_error",
            "coverage",
            "missing_scope",
            "forecast_provider_missing",
            "document_snapshot_missing",
            "缺少覆盖",
            "覆盖",
            "数据覆盖",
            "数据接口本轮不可用",
            "接口不可用",
        )
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def is_diagnostic_like_note(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        keywords = (
            "timeout",
            "timed out",
            "failed",
            "fallback",
            "traceback",
            "valueerror",
            "runtimeerror",
            "typeerror",
            "keyerror",
            "exception",
            "degraded",
            "llm_provider_missing",
            "scheduler_gate_not_met",
            "超时",
            "回退",
            "失败",
            "异常",
            "日志",
        )
        return any(keyword in lowered for keyword in keywords)

    @classmethod
    def partition_bucket_notes(
        cls,
        items: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        investment_risks: list[str] = []
        coverage_notes: list[str] = []
        diagnostic_notes: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            if cls.is_coverage_like_note(text):
                coverage_notes.append(text)
                continue
            if cls.is_diagnostic_like_note(text):
                diagnostic_notes.append(text)
                continue
            investment_risks.append(text)
        return (
            cls.dedupe_texts(investment_risks),
            cls.dedupe_texts(coverage_notes),
            cls.dedupe_texts(diagnostic_notes),
        )

    def branch_result_to_verdict(
        self,
        branch_result: BranchResult,
        thesis: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        scope: CoverageScope = CoverageScope.BRANCH,
    ) -> BranchVerdict:
        """将旧 BranchResult 映射为新协议层 BranchVerdict。"""

        final_score = float(branch_result.final_score if branch_result.final_score is not None else branch_result.score)
        final_confidence = float(
            branch_result.final_confidence
            if branch_result.final_confidence is not None
            else branch_result.confidence
        )
        thesis_text = (
            str(thesis or "").strip()
            or str(branch_result.conclusion or "").strip()
            or next((str(item).strip() for item in branch_result.thesis_points if str(item).strip()), "")
            or str(branch_result.explanation or "").strip()
            or f"{branch_result.branch_name} 分支已生成结构化判断。"
        )
        source_risks = [
            str(item)
            for item in (branch_result.investment_risks or branch_result.risks)
            if str(item).strip()
        ]
        investment_risks, rerouted_coverage, rerouted_diagnostic = self.partition_bucket_notes(
            source_risks
        )
        coverage_notes = self.dedupe_texts(
            [str(item) for item in branch_result.coverage_notes if str(item).strip()]
            + rerouted_coverage
        )
        diagnostic_notes = self.dedupe_texts(
            [str(item) for item in branch_result.diagnostic_notes if str(item).strip()]
            + rerouted_diagnostic
        )
        degraded_by_diagnostic = any(
            token in note.lower()
            for note in diagnostic_notes
            for token in ("timeout", "timed out", "failed", "fallback", "回退", "超时", "失败")
        )
        status = AgentStatus.SUCCESS
        if (
            not branch_result.success
            or branch_result.metadata.get("degraded_reason")
            or degraded_by_diagnostic
        ):
            status = AgentStatus.DEGRADED

        symbols = self.top_symbols(branch_result.symbol_scores)

        summary = (
            str(branch_result.explanation or "").strip()
            or str(branch_result.conclusion or "").strip()
            or thesis_text
        )

        merged_metadata = dict(branch_result.metadata or {})
        merged_metadata.update(
            {
                "legacy_branch_name": branch_result.branch_name,
                "legacy_success": branch_result.success,
                "legacy_symbol_scores": dict(branch_result.symbol_scores),
            }
        )
        if metadata:
            merged_metadata.update(dict(metadata))

        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=thesis_text,
            symbol=None,
            status=status,
            direction=self.score_to_direction(final_score),
            action=self.score_to_action(final_score),
            confidence_label=self.confidence_to_label(final_confidence),
            final_score=final_score,
            final_confidence=final_confidence,
            evidence=[
                EvidenceItem(
                    source=self.agent_name,
                    summary=summary,
                    direction=self.score_to_direction(final_score),
                    score=final_score,
                    confidence=final_confidence,
                    scope=scope,
                    symbols=symbols,
                    metadata={"top_symbols": symbols},
                )
            ],
            investment_risks=investment_risks,
            coverage_notes=coverage_notes,
            diagnostic_notes=diagnostic_notes,
            metadata=merged_metadata,
        )
