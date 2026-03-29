"""
Learning promotion 模块。

提供可读源码实现，统一 promotion 相关 dataclass 的 `schema_version` 命名，
并保持闭环只输出 candidate / decision / proposal，不直接改写 live 规则。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.post_trade_reflector import ReflectionReport, TradeCase
from quant_investor.versioning import (
    PROMOTION_CANDIDATE_SCHEMA_VERSION,
    PROMOTION_DECISION_SCHEMA_VERSION,
    RULE_PROPOSAL_SCHEMA_VERSION,
)


@dataclass
class PromotionCandidate:
    candidate_id: str
    source_case_ids: list[str]
    lesson_statement: str
    lesson_type: str
    support_count: int
    counter_count: int
    regimes_seen: list[str]
    sectors_seen: list[str]
    confidence: float
    status: str
    evidence_summary: str
    counter_case_ids: list[str] | None = None
    schema_version: str = PROMOTION_CANDIDATE_SCHEMA_VERSION


@dataclass
class PromotionDecision:
    candidate_id: str
    target_status: str
    decision: str
    reason: str
    support_count: int
    counter_count: int
    confidence: float
    missing_requirements: list[str]
    rule_proposal_recommended: bool
    schema_version: str = PROMOTION_DECISION_SCHEMA_VERSION


@dataclass
class RuleProposal:
    proposal_id: str
    proposal_type: str
    suggestion: str
    evidence: str
    expected_effect: str
    schema_version: str = RULE_PROPOSAL_SCHEMA_VERSION


class MemoryPromoter:
    """根据反思报告生成可回顾的经验候选与规则建议。"""

    def collect_candidates(
        self,
        cases: Iterable[TradeCase],
        reflection_reports: Iterable[ReflectionReport],
    ) -> list[PromotionCandidate]:
        case_map = {case.case_id: case for case in cases}
        candidates: list[PromotionCandidate] = []

        for report in reflection_reports:
            trade_case = case_map.get(report.case_id)
            regime = "unknown"
            sector = "unknown"
            if trade_case is not None:
                regime = str(trade_case.pretrade_snapshot.market_regime or "unknown")
                sector = str(trade_case.metadata.get("sector", "unknown"))

            for index, lesson in enumerate(report.lesson_drafts, start=1):
                candidates.append(
                    PromotionCandidate(
                        candidate_id=f"{report.case_id}:{index}",
                        source_case_ids=[report.case_id],
                        lesson_statement=lesson.statement,
                        lesson_type=lesson.lesson_type,
                        support_count=1 + len(report.key_success_factors),
                        counter_count=len(report.key_failure_factors),
                        regimes_seen=[regime],
                        sectors_seen=[sector],
                        confidence=float(lesson.confidence),
                        status="candidate_lesson",
                        evidence_summary=report.summary or lesson.rationale,
                        counter_case_ids=[],
                    )
                )

        return candidates

    def validate_candidate(
        self,
        candidate: PromotionCandidate,
        cases: Iterable[TradeCase],
    ) -> PromotionDecision:
        del cases
        support_gap = candidate.support_count - candidate.counter_count
        if support_gap >= 2 and candidate.confidence >= 0.7:
            target_status = "validated_pattern"
            decision = "validate_pattern"
            reason = "support materially exceeds counter evidence"
        elif candidate.counter_count > candidate.support_count:
            target_status = "rejected"
            decision = "reject_candidate"
            reason = "counter evidence dominates current lesson"
        else:
            target_status = "candidate_lesson"
            decision = "keep_candidate"
            reason = "insufficient support for promotion"

        return PromotionDecision(
            candidate_id=candidate.candidate_id,
            target_status=target_status,
            decision=decision,
            reason=reason,
            support_count=candidate.support_count,
            counter_count=candidate.counter_count,
            confidence=candidate.confidence,
            missing_requirements=[] if target_status != "candidate_lesson" else ["more_support_cases"],
            rule_proposal_recommended=target_status in {"validated_pattern", "approved_rule_candidate"},
        )

    @staticmethod
    def archive_rejected(candidate: PromotionCandidate) -> PromotionCandidate:
        return replace(candidate, status="rejected")

    @staticmethod
    def promote(candidate: PromotionCandidate, decision: PromotionDecision) -> PromotionCandidate:
        return replace(candidate, status=decision.target_status)

    @staticmethod
    def build_rule_proposal(candidate: PromotionCandidate) -> RuleProposal:
        proposal_type = "risk_guard_update" if candidate.lesson_type == "risk" else "research_process_update"
        return RuleProposal(
            proposal_id=f"proposal:{candidate.candidate_id}",
            proposal_type=proposal_type,
            suggestion=candidate.lesson_statement,
            evidence=candidate.evidence_summary,
            expected_effect="improve future structured review and recall quality",
        )


__all__ = [
    "MemoryIndexer",
    "MemoryPromoter",
    "PromotionCandidate",
    "PromotionDecision",
    "ReflectionReport",
    "RuleProposal",
    "TradeCase",
]
