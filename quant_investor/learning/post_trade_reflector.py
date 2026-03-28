"""
Learning 反思模块。

提供可读源码实现，统一 reflection 相关 dataclass 的 `schema_version` 命名，
并保持闭环只输出结构化复盘结论，不修改 live 决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.trade_case_store import TradeCase as BaseTradeCase
from quant_investor.versioning import (
    REFLECTION_EVIDENCE_SCHEMA_VERSION,
    REFLECTION_LESSON_DRAFT_SCHEMA_VERSION,
    REFLECTION_REPORT_SCHEMA_VERSION,
    TRADE_CASE_SCHEMA_VERSION,
)


@dataclass
class ReflectionEvidence:
    evidence_type: str
    observation: str
    implication: str
    metric_value: Any | None = None
    schema_version: str = REFLECTION_EVIDENCE_SCHEMA_VERSION


@dataclass
class ReflectionLessonDraft:
    lesson_type: str
    statement: str
    rationale: str
    confidence: float
    promotion_recommendation: str
    schema_version: str = REFLECTION_LESSON_DRAFT_SCHEMA_VERSION


@dataclass
class ReflectionReport:
    case_id: str
    symbol: str
    thesis_validation: str
    timing_assessment: str
    risk_control_assessment: str
    human_override_assessment: str
    key_success_factors: list[str] = field(default_factory=list)
    key_failure_factors: list[str] = field(default_factory=list)
    lesson_drafts: list[ReflectionLessonDraft] = field(default_factory=list)
    suggested_error_tags: list[str] = field(default_factory=list)
    summary: str = ""
    evidence: list[ReflectionEvidence] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    schema_version: str = REFLECTION_REPORT_SCHEMA_VERSION


@dataclass
class TradeCase(BaseTradeCase):
    schema_version: str = TRADE_CASE_SCHEMA_VERSION


class PostTradeReflector:
    """交易后反思器，只生成结构化复盘报告。"""

    def __init__(self, memory_indexer: MemoryIndexer | None = None) -> None:
        self.memory_indexer = memory_indexer

    def reflect_case(self, trade_case: TradeCase) -> ReflectionReport:
        realized_return = self._first_realized_return(trade_case)
        thesis_validation = self._classify_return(realized_return)
        timing_assessment = "good" if realized_return >= 0 else "needs_improvement"
        risk_control_assessment = "good" if not trade_case.attribution.missed_risks else "needs_improvement"
        human_override_assessment = "override_used" if trade_case.human_decision.manual_override else "neutral"

        evidence = [
            ReflectionEvidence(
                evidence_type="return",
                observation=f"realized_return={realized_return:.4f}",
                implication="positive follow-through" if realized_return >= 0 else "negative follow-through",
                metric_value=realized_return,
            ),
            ReflectionEvidence(
                evidence_type="risk",
                observation=trade_case.attribution.summary or "risk attribution unavailable",
                implication="review risk controls when missed risks are present"
                if trade_case.attribution.missed_risks
                else "risk controls broadly aligned",
            ),
        ]

        lesson_drafts = [
            ReflectionLessonDraft(
                lesson_type="case_lesson",
                statement=self._lesson_statement(trade_case, thesis_validation),
                rationale="derived from realized outcome and attribution snapshot",
                confidence=self._lesson_confidence(realized_return),
                promotion_recommendation="candidate_only",
            )
        ]

        key_success = list(trade_case.attribution.helpful_agents)
        key_failure = list(trade_case.attribution.misleading_agents)
        suggested_error_tags = list(trade_case.error_tags)
        summary = self._build_summary(trade_case, thesis_validation, realized_return)

        return ReflectionReport(
            case_id=trade_case.case_id,
            symbol=trade_case.symbol,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
            key_success_factors=key_success,
            key_failure_factors=key_failure,
            lesson_drafts=lesson_drafts,
            suggested_error_tags=suggested_error_tags,
            summary=summary,
            evidence=evidence,
        )

    @staticmethod
    def _first_realized_return(trade_case: TradeCase) -> float:
        for value in (
            trade_case.outcomes.t5_return,
            trade_case.outcomes.t10_return,
            trade_case.outcomes.t20_return,
            trade_case.outcomes.t1_return,
        ):
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _classify_return(realized_return: float) -> str:
        if realized_return > 0.02:
            return "correct"
        if realized_return < -0.02:
            return "incorrect"
        return "mixed"

    @staticmethod
    def _lesson_confidence(realized_return: float) -> float:
        return min(0.95, max(0.55, 0.55 + abs(realized_return) * 2.0))

    @staticmethod
    def _lesson_statement(trade_case: TradeCase, thesis_validation: str) -> str:
        if thesis_validation == "correct":
            return f"{trade_case.symbol} 的执行与原始研究方向一致，可保留同类 setup。"
        if thesis_validation == "incorrect":
            return f"{trade_case.symbol} 的研究假设未兑现，应收紧同类 setup 的风险预算。"
        return f"{trade_case.symbol} 的研究结果分化，需要结合更多上下文再复用经验。"

    @staticmethod
    def _build_summary(trade_case: TradeCase, thesis_validation: str, realized_return: float) -> str:
        return (
            f"{trade_case.symbol} case={trade_case.case_id} "
            f"validation={thesis_validation} realized_return={realized_return:.4f}"
        )

__all__ = [
    "MemoryIndexer",
    "PostTradeReflector",
    "ReflectionEvidence",
    "ReflectionLessonDraft",
    "ReflectionReport",
    "TradeCase",
]
