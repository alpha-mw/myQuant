"""
Post-trade learning 第四阶段的 memory promoter。

该模块只负责把 lesson draft 与历史案件整理为晋升候选和规则提案，
不会自动修改 live 系统。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Iterable, Literal

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.post_trade_reflector import ReflectionReport
from quant_investor.learning.trade_case_store import TradeCase


_ALLOWED_CANDIDATE_STATUS = {
    "raw_case",
    "candidate_lesson",
    "validated_pattern",
    "approved_rule_candidate",
    "rejected",
}
_LESSON_PRIORITY = {
    "risk_rule_candidate": 0,
    "semantic_candidate": 1,
    "report_improvement": 2,
    "case_lesson": 3,
}
_SPACE_PATTERN = re.compile(r"\s+")


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
    status: Literal[
        "raw_case",
        "candidate_lesson",
        "validated_pattern",
        "approved_rule_candidate",
        "rejected",
    ]
    evidence_summary: str
    counter_case_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_CANDIDATE_STATUS:
            raise ValueError(f"status 必须是 {_ALLOWED_CANDIDATE_STATUS} 之一")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 区间")
        if self.counter_case_ids is None:
            self.counter_case_ids = []


@dataclass
class PromotionDecision:
    candidate_id: str
    target_status: Literal[
        "raw_case",
        "candidate_lesson",
        "validated_pattern",
        "approved_rule_candidate",
        "rejected",
    ]
    decision: Literal["keep_candidate", "validate_pattern", "approve_rule_candidate", "reject"]
    reason: str
    support_count: int
    counter_count: int
    confidence: float
    missing_requirements: list[str]
    rule_proposal_recommended: bool


@dataclass
class RuleProposal:
    proposal_id: str
    proposal_type: Literal[
        "risk_guard_update",
        "prompt_update",
        "action_mapping_update",
        "report_update",
    ]
    suggestion: str
    evidence: str
    expected_effect: str
    approval_status: Literal["pending", "approved", "rejected"] = "pending"


class MemoryPromoter:
    """保守阈值下的经验晋升器。"""

    def __init__(
        self,
        *,
        min_support_for_validated: int = 2,
        min_counter_for_validated: int = 1,
        min_support_for_approved: int = 4,
        min_counter_for_approved: int = 2,
        min_regimes_for_approved: int = 2,
        min_confidence_for_validated: float = 0.65,
        min_confidence_for_approved: float = 0.78,
    ) -> None:
        self.min_support_for_validated = min_support_for_validated
        self.min_counter_for_validated = min_counter_for_validated
        self.min_support_for_approved = min_support_for_approved
        self.min_counter_for_approved = min_counter_for_approved
        self.min_regimes_for_approved = min_regimes_for_approved
        self.min_confidence_for_validated = min_confidence_for_validated
        self.min_confidence_for_approved = min_confidence_for_approved
        self.memory_indexer = MemoryIndexer()

    def collect_candidates(
        self,
        cases: Iterable[TradeCase],
        reflection_reports: Iterable[ReflectionReport],
    ) -> list[PromotionCandidate]:
        case_list = list(cases)
        report_list = list(reflection_reports)
        case_by_id = {case.case_id: case for case in case_list}
        tags_by_case = {
            case.case_id: self.memory_indexer.extract_tags(case)
            for case in case_list
        }
        report_by_case = {report.case_id: report for report in report_list}

        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for report in report_list:
            if report.case_id not in case_by_id:
                continue
            for lesson in report.lesson_drafts:
                if lesson.promotion_recommendation == "discard":
                    continue
                normalized_statement = self._normalize_statement(lesson.statement)
                key = (lesson.lesson_type, normalized_statement)
                bucket = grouped.setdefault(
                    key,
                    {
                        "lesson_statement": lesson.statement,
                        "lesson_type": lesson.lesson_type,
                        "source_case_ids": [],
                        "confidences": [],
                    },
                )
                bucket["source_case_ids"].append(report.case_id)
                bucket["confidences"].append(float(lesson.confidence))

        candidates: list[PromotionCandidate] = []
        for (lesson_type, normalized_statement), bucket in grouped.items():
            support_case_ids = sorted(set(bucket["source_case_ids"]))
            regimes_seen = sorted(
                {
                    tags_by_case[case_id].market_regime
                    for case_id in support_case_ids
                    if case_id in tags_by_case
                }
            )
            sectors_seen = sorted(
                {
                    tags_by_case[case_id].sector
                    for case_id in support_case_ids
                    if case_id in tags_by_case
                }
            )
            counter_case_ids = self._collect_counter_cases(
                support_case_ids=support_case_ids,
                cases=case_list,
                tags_by_case=tags_by_case,
                report_by_case=report_by_case,
                lesson_type=lesson_type,
                normalized_statement=normalized_statement,
            )
            base_confidence = (
                sum(float(item) for item in bucket["confidences"]) / len(bucket["confidences"])
                if bucket["confidences"]
                else 0.5
            )
            confidence = self._aggregate_confidence(
                lesson_type=lesson_type,
                base_confidence=base_confidence,
                support_count=len(support_case_ids),
                counter_count=len(counter_case_ids),
                regime_count=len(regimes_seen),
            )
            initial_status = self._initial_status(
                lesson_type=lesson_type,
                support_count=len(support_case_ids),
            )
            evidence_summary = self._build_evidence_summary(
                lesson_statement=str(bucket["lesson_statement"]),
                lesson_type=lesson_type,
                support_case_ids=support_case_ids,
                counter_case_ids=counter_case_ids,
                regimes_seen=regimes_seen,
                sectors_seen=sectors_seen,
                confidence=confidence,
            )
            candidates.append(
                PromotionCandidate(
                    candidate_id=self._candidate_id(lesson_type, normalized_statement),
                    source_case_ids=support_case_ids,
                    lesson_statement=str(bucket["lesson_statement"]),
                    lesson_type=lesson_type,
                    support_count=len(support_case_ids),
                    counter_count=len(counter_case_ids),
                    regimes_seen=regimes_seen,
                    sectors_seen=sectors_seen,
                    confidence=confidence,
                    status=initial_status,
                    evidence_summary=evidence_summary,
                    counter_case_ids=counter_case_ids,
                )
            )

        return sorted(
            candidates,
            key=lambda item: (
                _LESSON_PRIORITY.get(item.lesson_type, 99),
                -item.support_count,
                -item.confidence,
                item.candidate_id,
            ),
        )

    def validate_candidate(
        self,
        candidate: PromotionCandidate,
        cases: Iterable[TradeCase],
    ) -> PromotionDecision:
        case_ids = {case.case_id for case in cases}
        support_count = len([case_id for case_id in candidate.source_case_ids if case_id in case_ids])
        counter_count = len(
            [case_id for case_id in (candidate.counter_case_ids or []) if case_id in case_ids]
        )
        missing_requirements: list[str] = []
        if support_count < self.min_support_for_validated:
            missing_requirements.append(
                f"support_count<{self.min_support_for_validated}"
            )
        if counter_count < self.min_counter_for_validated:
            missing_requirements.append(
                f"counter_count<{self.min_counter_for_validated}"
            )
        if candidate.confidence < self.min_confidence_for_validated:
            missing_requirements.append(
                f"confidence<{self.min_confidence_for_validated:.2f}"
            )

        is_risk_candidate = candidate.lesson_type == "risk_rule_candidate"
        can_validate = not missing_requirements
        can_approve = (
            is_risk_candidate
            and support_count >= self.min_support_for_approved
            and counter_count >= self.min_counter_for_approved
            and len(candidate.regimes_seen) >= self.min_regimes_for_approved
            and candidate.confidence >= self.min_confidence_for_approved
        )

        if support_count == 0:
            return PromotionDecision(
                candidate_id=candidate.candidate_id,
                target_status="rejected",
                decision="reject",
                reason="候选缺少有效 support cases。",
                support_count=support_count,
                counter_count=counter_count,
                confidence=candidate.confidence,
                missing_requirements=["support_cases_missing"],
                rule_proposal_recommended=False,
            )

        if can_approve:
            return PromotionDecision(
                candidate_id=candidate.candidate_id,
                target_status="approved_rule_candidate",
                decision="approve_rule_candidate",
                reason="风险类候选已满足多案例、反例分析、跨 regime 和高置信度门槛。",
                support_count=support_count,
                counter_count=counter_count,
                confidence=candidate.confidence,
                missing_requirements=[],
                rule_proposal_recommended=True,
            )

        if can_validate:
            return PromotionDecision(
                candidate_id=candidate.candidate_id,
                target_status="validated_pattern",
                decision="validate_pattern",
                reason="候选已满足多案例支持和 counter analysis，可以提升为 validated_pattern。",
                support_count=support_count,
                counter_count=counter_count,
                confidence=candidate.confidence,
                missing_requirements=[],
                rule_proposal_recommended=is_risk_candidate,
            )

        return PromotionDecision(
            candidate_id=candidate.candidate_id,
            target_status="candidate_lesson" if candidate.status != "raw_case" else "raw_case",
            decision="keep_candidate",
            reason="证据尚不足以进一步晋升，继续保留为候选经验。",
            support_count=support_count,
            counter_count=counter_count,
            confidence=candidate.confidence,
            missing_requirements=missing_requirements,
            rule_proposal_recommended=False,
        )

    def build_rule_proposal(self, candidate: PromotionCandidate) -> RuleProposal:
        proposal_type = self._proposal_type_for_lesson_type(candidate.lesson_type)
        suggestion = self._build_suggestion(candidate, proposal_type)
        expected_effect = self._expected_effect(proposal_type)
        return RuleProposal(
            proposal_id=self._proposal_id(candidate.candidate_id, proposal_type),
            proposal_type=proposal_type,
            suggestion=suggestion,
            evidence=candidate.evidence_summary,
            expected_effect=expected_effect,
            approval_status="pending",
        )

    def promote(
        self,
        candidate: PromotionCandidate,
        decision: PromotionDecision,
    ) -> PromotionCandidate:
        return replace(
            candidate,
            status=decision.target_status,
            confidence=max(candidate.confidence, decision.confidence),
            evidence_summary=(
                f"{candidate.evidence_summary} | promotion_decision={decision.decision}"
                f" | reason={decision.reason}"
            ),
        )

    def archive_rejected(self, candidate: PromotionCandidate) -> PromotionCandidate:
        return replace(
            candidate,
            status="rejected",
            evidence_summary=f"{candidate.evidence_summary} | archived=rejected",
        )

    def _collect_counter_cases(
        self,
        *,
        support_case_ids: list[str],
        cases: list[TradeCase],
        tags_by_case: dict[str, object],
        report_by_case: dict[str, ReflectionReport],
        lesson_type: str,
        normalized_statement: str,
    ) -> list[str]:
        support_regimes = {
            tags_by_case[case_id].market_regime
            for case_id in support_case_ids
            if case_id in tags_by_case
        }
        support_sectors = {
            tags_by_case[case_id].sector
            for case_id in support_case_ids
            if case_id in tags_by_case
        }
        support_patterns = {
            tags_by_case[case_id].branch_support_pattern
            for case_id in support_case_ids
            if case_id in tags_by_case
        }

        counter_case_ids: list[str] = []
        support_set = set(support_case_ids)
        for case in cases:
            if case.case_id in support_set or case.case_id not in tags_by_case:
                continue
            tags = tags_by_case[case.case_id]
            same_context = (
                tags.market_regime in support_regimes
                or tags.sector in support_sectors
                or tags.branch_support_pattern in support_patterns
            )
            if not same_context:
                continue

            report = report_by_case.get(case.case_id)
            if report is None:
                counter_case_ids.append(case.case_id)
                continue

            has_same_lesson = any(
                lesson.lesson_type == lesson_type
                and self._normalize_statement(lesson.statement) == normalized_statement
                for lesson in report.lesson_drafts
                if lesson.promotion_recommendation != "discard"
            )
            if not has_same_lesson:
                counter_case_ids.append(case.case_id)

        return sorted(set(counter_case_ids))

    @staticmethod
    def _normalize_statement(statement: str) -> str:
        return _SPACE_PATTERN.sub(" ", statement.strip().lower())

    @staticmethod
    def _candidate_id(lesson_type: str, normalized_statement: str) -> str:
        digest = hashlib.sha1(f"{lesson_type}|{normalized_statement}".encode("utf-8")).hexdigest()[:12]
        return f"candidate::{lesson_type}::{digest}"

    @staticmethod
    def _proposal_id(candidate_id: str, proposal_type: str) -> str:
        digest = hashlib.sha1(f"{candidate_id}|{proposal_type}".encode("utf-8")).hexdigest()[:12]
        return f"proposal::{proposal_type}::{digest}"

    @staticmethod
    def _initial_status(*, lesson_type: str, support_count: int) -> str:
        if lesson_type == "case_lesson" and support_count <= 1:
            return "raw_case"
        return "candidate_lesson"

    @staticmethod
    def _aggregate_confidence(
        *,
        lesson_type: str,
        base_confidence: float,
        support_count: int,
        counter_count: int,
        regime_count: int,
    ) -> float:
        support_bonus = min(support_count, 5) * 0.05
        counter_bonus = 0.04 if counter_count > 0 else -0.06
        regime_bonus = min(regime_count, 3) * 0.03
        risk_bonus = 0.05 if lesson_type == "risk_rule_candidate" else 0.0
        confidence = base_confidence + support_bonus + counter_bonus + regime_bonus + risk_bonus
        return round(min(0.95, max(0.1, confidence)), 4)

    @staticmethod
    def _build_evidence_summary(
        *,
        lesson_statement: str,
        lesson_type: str,
        support_case_ids: list[str],
        counter_case_ids: list[str],
        regimes_seen: list[str],
        sectors_seen: list[str],
        confidence: float,
    ) -> str:
        return (
            f"lesson_type={lesson_type} | lesson_statement={lesson_statement} | "
            f"support_cases={support_case_ids} | counter_cases={counter_case_ids} | "
            f"regimes_seen={regimes_seen} | sectors_seen={sectors_seen} | "
            f"confidence={confidence:.4f}"
        )

    @staticmethod
    def _proposal_type_for_lesson_type(lesson_type: str) -> str:
        mapping = {
            "risk_rule_candidate": "risk_guard_update",
            "report_improvement": "report_update",
            "semantic_candidate": "prompt_update",
            "case_lesson": "action_mapping_update",
        }
        return mapping.get(lesson_type, "prompt_update")

    @staticmethod
    def _build_suggestion(candidate: PromotionCandidate, proposal_type: str) -> str:
        if proposal_type == "risk_guard_update":
            return (
                "为 RiskGuard 增加候选显式约束，仅作为待审批提案："
                f"{candidate.lesson_statement}"
            )
        if proposal_type == "report_update":
            return f"在复盘/报告说明中补充候选改进项：{candidate.lesson_statement}"
        if proposal_type == "action_mapping_update":
            return f"评估是否需要调整 action mapping 的候选逻辑：{candidate.lesson_statement}"
        return f"在 prompt/经验说明层补充候选模式：{candidate.lesson_statement}"

    @staticmethod
    def _expected_effect(proposal_type: str) -> str:
        effects = {
            "risk_guard_update": "优先减少已知风险暴露，并把经验保持在待审批状态。",
            "report_update": "提升复盘和说明的可解释性，但不影响 live 决策。",
            "action_mapping_update": "为后续映射逻辑评审提供候选证据，不直接改生产规则。",
            "prompt_update": "在研究/复盘提示层补充候选经验，不直接驱动仓位或规则。",
        }
        return effects[proposal_type]
