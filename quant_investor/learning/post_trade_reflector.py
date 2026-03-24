"""
Post-trade learning 第三阶段的 deterministic 单笔交易复盘器。

该模块只输出结构化复盘建议，不自动修改系统规则、仓位或权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from quant_investor.learning.memory_indexer import MemoryIndexer
from quant_investor.learning.trade_case_store import TradeCase


@dataclass
class ReflectionEvidence:
    evidence_type: str
    observation: str
    implication: str
    metric_value: float | str | None = None


@dataclass
class ReflectionLessonDraft:
    lesson_type: Literal[
        "case_lesson",
        "semantic_candidate",
        "risk_rule_candidate",
        "report_improvement",
    ]
    statement: str
    rationale: str
    confidence: float
    promotion_recommendation: Literal["keep_as_case", "candidate_only", "discard"]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 区间")


@dataclass
class ReflectionReport:
    case_id: str
    symbol: str
    thesis_validation: Literal["correct", "partially_correct", "incorrect", "unresolved"]
    timing_assessment: Literal["good", "acceptable", "poor", "unresolved"]
    risk_control_assessment: Literal["good", "poor", "unresolved"]
    human_override_assessment: Literal["helpful", "harmful", "neutral", "not_applicable"]
    key_success_factors: list[str] = field(default_factory=list)
    key_failure_factors: list[str] = field(default_factory=list)
    lesson_drafts: list[ReflectionLessonDraft] = field(default_factory=list)
    suggested_error_tags: list[str] = field(default_factory=list)
    summary: str = ""
    evidence: list[ReflectionEvidence] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class PostTradeReflector:
    """基于确定性规则生成单笔交易复盘。"""

    def __init__(self, memory_indexer: MemoryIndexer | None = None):
        self.memory_indexer = memory_indexer

    def reflect_case(self, trade_case: TradeCase) -> ReflectionReport:
        thesis_validation = self.evaluate_thesis(trade_case)
        timing_assessment = self.evaluate_timing(trade_case)
        risk_control_assessment = self.evaluate_risk_controls(trade_case)
        human_override_assessment = self.evaluate_human_override(trade_case)
        evidence = self._build_evidence(
            trade_case,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
        )
        success_factors, failure_factors = self._collect_factors(
            trade_case,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
        )
        lesson_drafts = self.generate_lesson_drafts(
            trade_case,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
            success_factors=success_factors,
            failure_factors=failure_factors,
        )
        suggested_error_tags = self.suggest_error_tags(
            trade_case,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
        )

        summary_parts = [
            f"thesis={thesis_validation}",
            f"timing={timing_assessment}",
            f"risk={risk_control_assessment}",
            f"human={human_override_assessment}",
        ]
        if success_factors:
            summary_parts.append(f"success={'; '.join(success_factors[:2])}")
        if failure_factors:
            summary_parts.append(f"failure={'; '.join(failure_factors[:2])}")

        return ReflectionReport(
            case_id=trade_case.case_id,
            symbol=trade_case.symbol,
            thesis_validation=thesis_validation,
            timing_assessment=timing_assessment,
            risk_control_assessment=risk_control_assessment,
            human_override_assessment=human_override_assessment,
            key_success_factors=success_factors,
            key_failure_factors=failure_factors,
            lesson_drafts=lesson_drafts,
            suggested_error_tags=suggested_error_tags,
            summary=" | ".join(summary_parts),
            evidence=evidence,
            generated_at=trade_case.decision_time,
        )

    def evaluate_thesis(self, trade_case: TradeCase) -> str:
        thesis_return = self._thesis_return(trade_case)
        mae_abs = self._mae_abs(trade_case)

        if thesis_return is None:
            return "unresolved"
        if thesis_return >= 0.02 and (mae_abs is None or mae_abs <= 0.05):
            return "correct"
        if thesis_return > 0 or abs(thesis_return) < 0.01:
            return "partially_correct"
        if trade_case.outcomes.stop_loss_hit and thesis_return > -0.03:
            return "partially_correct"
        return "incorrect"

    def evaluate_timing(self, trade_case: TradeCase) -> str:
        thesis_return = self._thesis_return(trade_case)
        t1_return = trade_case.outcomes.t1_return
        mae_abs = self._mae_abs(trade_case)

        if thesis_return is None:
            return "unresolved"
        if thesis_return > 0:
            if (t1_return is None or t1_return >= 0.0) and (mae_abs is None or mae_abs <= 0.03):
                return "good"
            if (mae_abs is None or mae_abs <= 0.08) and (t1_return is None or t1_return > -0.02):
                return "acceptable"
            return "poor"
        if abs(thesis_return) < 0.01 and (mae_abs is None or mae_abs <= 0.03):
            return "acceptable"
        return "poor"

    def evaluate_risk_controls(self, trade_case: TradeCase) -> str:
        thesis_return = self._thesis_return(trade_case)
        has_controls = bool(trade_case.attribution.correct_risk_controls)
        has_missed_risks = bool(trade_case.attribution.missed_risks)

        if thesis_return is None and not has_controls and not has_missed_risks:
            return "unresolved"
        if has_missed_risks:
            return "poor"
        if trade_case.outcomes.stop_loss_hit and (thesis_return is None or thesis_return > -0.05):
            return "good"
        if has_controls and (thesis_return is None or thesis_return > -0.05):
            return "good"
        if thesis_return is not None and thesis_return <= -0.05:
            return "poor"
        return "unresolved"

    def evaluate_human_override(self, trade_case: TradeCase) -> str:
        thesis_return = self._thesis_return(trade_case)
        human_action = trade_case.human_decision.human_action

        if human_action == "executed" and not trade_case.human_decision.manual_override:
            return "not_applicable"
        if thesis_return is None:
            return "neutral"
        if human_action in {"skipped", "overridden"}:
            if thesis_return >= 0.02:
                return "harmful"
            if thesis_return <= -0.02:
                return "helpful"
        return "neutral"

    def generate_lesson_drafts(
        self,
        trade_case: TradeCase,
        *,
        thesis_validation: str,
        timing_assessment: str,
        risk_control_assessment: str,
        human_override_assessment: str,
        success_factors: list[str],
        failure_factors: list[str],
    ) -> list[ReflectionLessonDraft]:
        lesson_drafts: list[ReflectionLessonDraft] = []
        case_statement = (
            f"{trade_case.symbol} 本案复盘结论: thesis={thesis_validation}, "
            f"timing={timing_assessment}, risk={risk_control_assessment}."
        )
        case_rationale = (
            f"成功因素={'; '.join(success_factors) if success_factors else '无显著正向因素'}；"
            f"失败因素={'; '.join(failure_factors) if failure_factors else '无显著负向因素'}。"
        )
        lesson_drafts.append(
            ReflectionLessonDraft(
                lesson_type="case_lesson",
                statement=case_statement,
                rationale=case_rationale,
                confidence=self._lesson_confidence(trade_case, base=0.75),
                promotion_recommendation="keep_as_case",
            )
        )

        if thesis_validation in {"correct", "partially_correct"}:
            lesson_drafts.append(
                ReflectionLessonDraft(
                    lesson_type="semantic_candidate",
                    statement=(
                        f"{trade_case.pretrade_snapshot.market_regime} 下，"
                        f"{', '.join(trade_case.pretrade_snapshot.support_agents) or '低共识'}"
                        " 的支持模式可继续作为候选经验观察。"
                    ),
                    rationale=(
                        "本案至少部分验证了原始 thesis，但当前阶段只保留为 candidate，"
                        "不自动晋升为系统记忆或规则。"
                    ),
                    confidence=self._lesson_confidence(trade_case, base=0.62),
                    promotion_recommendation="candidate_only",
                )
            )

        if risk_control_assessment == "poor":
            lesson_drafts.append(
                ReflectionLessonDraft(
                    lesson_type="risk_rule_candidate",
                    statement="该案例暴露了需要进一步显式化的风险控制条件。",
                    rationale=(
                        f"missed_risks={trade_case.attribution.missed_risks or ['none']}，"
                        f"stop_loss_hit={trade_case.outcomes.stop_loss_hit}。"
                    ),
                    confidence=self._lesson_confidence(trade_case, base=0.68),
                    promotion_recommendation="candidate_only",
                )
            )

        if trade_case.error_tags or timing_assessment == "poor":
            lesson_drafts.append(
                ReflectionLessonDraft(
                    lesson_type="report_improvement",
                    statement="需要在案件记录或说明层补充更明确的执行窗口与风险提示。",
                    rationale=(
                        f"existing_error_tags={trade_case.error_tags or ['none']}，"
                        f"timing_assessment={timing_assessment}，"
                        f"human_override_assessment={human_override_assessment}。"
                    ),
                    confidence=self._lesson_confidence(trade_case, base=0.58),
                    promotion_recommendation="candidate_only",
                )
            )

        return lesson_drafts

    def suggest_error_tags(
        self,
        trade_case: TradeCase,
        *,
        thesis_validation: str,
        timing_assessment: str,
        risk_control_assessment: str,
        human_override_assessment: str,
    ) -> list[str]:
        suggested: list[str] = []
        if thesis_validation == "incorrect":
            suggested.append("thesis_miss")
        if timing_assessment == "poor":
            suggested.append("timing_mismatch")
        if risk_control_assessment == "poor":
            suggested.append("risk_control_gap")
        if trade_case.attribution.missed_risks:
            suggested.append("missed_risk_mapping")
        if human_override_assessment == "harmful":
            suggested.append("harmful_override")
        elif human_override_assessment == "helpful":
            suggested.append("helpful_override")
        if self._mae_abs(trade_case) is not None and self._mae_abs(trade_case) > 0.08:
            suggested.append("deep_drawdown")
        if trade_case.human_decision.human_action == "skipped" and self._thesis_return(trade_case) not in (None,):
            if self._thesis_return(trade_case) > 0.02:
                suggested.append("missed_execution")

        existing = {tag.strip().lower() for tag in trade_case.error_tags if tag.strip()}
        deduped: list[str] = []
        for tag in suggested:
            normalized = tag.strip().lower()
            if normalized and normalized not in existing and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @staticmethod
    def _thesis_return(trade_case: TradeCase) -> float | None:
        for attr in ("t5_return", "t10_return", "t20_return", "t1_return"):
            value = getattr(trade_case.outcomes, attr)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def _mae_abs(trade_case: TradeCase) -> float | None:
        if trade_case.outcomes.mae is None:
            return None
        return abs(float(trade_case.outcomes.mae))

    @staticmethod
    def _lesson_confidence(trade_case: TradeCase, *, base: float) -> float:
        confidence_values = list(trade_case.pretrade_snapshot.branch_confidences.values())
        avg_conf = (
            sum(float(value) for value in confidence_values) / len(confidence_values)
            if confidence_values
            else 0.5
        )
        if PostTradeReflector._thesis_return(trade_case) is None:
            return round(max(0.3, base - 0.2), 4)
        return round(min(0.95, max(0.3, base * 0.6 + avg_conf * 0.4)), 4)

    def _collect_factors(
        self,
        trade_case: TradeCase,
        *,
        thesis_validation: str,
        timing_assessment: str,
        risk_control_assessment: str,
        human_override_assessment: str,
    ) -> tuple[list[str], list[str]]:
        success_factors: list[str] = []
        failure_factors: list[str] = []

        if thesis_validation == "correct":
            success_factors.append("原始 thesis 与后续收益方向一致")
        elif thesis_validation == "partially_correct":
            success_factors.append("原始 thesis 只被部分验证")
        else:
            failure_factors.append("原始 thesis 未被后续价格路径验证")

        if trade_case.pretrade_snapshot.support_agents:
            success_factors.append(
                f"support_agents={','.join(trade_case.pretrade_snapshot.support_agents)}"
            )
        if timing_assessment == "poor":
            failure_factors.append("进入时机或执行窗口较差")
        elif timing_assessment == "good":
            success_factors.append("进入时机较优且回撤可控")

        if risk_control_assessment == "good":
            success_factors.append("风险控制在本案中起到保护作用")
        elif risk_control_assessment == "poor":
            failure_factors.append("风险控制未充分覆盖实际风险暴露")

        if trade_case.attribution.missed_risks:
            failure_factors.append(
                f"missed_risks={','.join(trade_case.attribution.missed_risks)}"
            )
        if trade_case.attribution.helpful_agents:
            success_factors.append(
                f"helpful_agents={','.join(trade_case.attribution.helpful_agents)}"
            )
        if trade_case.attribution.misleading_agents:
            failure_factors.append(
                f"misleading_agents={','.join(trade_case.attribution.misleading_agents)}"
            )

        if human_override_assessment == "helpful":
            success_factors.append("人工偏离系统建议后避免了潜在损失")
        elif human_override_assessment == "harmful":
            failure_factors.append("人工偏离系统建议后损害了本案表现")

        if not success_factors and self._thesis_return(trade_case) is None:
            success_factors.append("结果尚未完全落地，先保留原始案件")
        if not failure_factors and trade_case.error_tags:
            failure_factors.append(f"existing_error_tags={','.join(trade_case.error_tags)}")

        return success_factors, failure_factors

    def _build_evidence(
        self,
        trade_case: TradeCase,
        *,
        thesis_validation: str,
        timing_assessment: str,
        risk_control_assessment: str,
        human_override_assessment: str,
    ) -> list[ReflectionEvidence]:
        evidence = [
            ReflectionEvidence(
                evidence_type="thesis_return",
                observation="优先使用 t5/t10/t20/t1 收益序列判断 thesis 是否被验证。",
                implication=f"thesis_validation={thesis_validation}",
                metric_value=self._thesis_return(trade_case),
            ),
            ReflectionEvidence(
                evidence_type="drawdown",
                observation="使用 MAE 衡量中途 adverse excursion 是否可控。",
                implication=f"timing_assessment={timing_assessment}",
                metric_value=self._mae_abs(trade_case),
            ),
            ReflectionEvidence(
                evidence_type="risk_controls",
                observation="结合 correct_risk_controls、missed_risks 与 stop_loss_hit 评估风控。",
                implication=f"risk_control_assessment={risk_control_assessment}",
                metric_value=",".join(trade_case.attribution.correct_risk_controls)
                if trade_case.attribution.correct_risk_controls
                else None,
            ),
            ReflectionEvidence(
                evidence_type="human_action",
                observation="根据 skipped/overridden 与后续表现评估人工干预影响。",
                implication=f"human_override_assessment={human_override_assessment}",
                metric_value=trade_case.human_decision.human_action,
            ),
        ]

        if self.memory_indexer is not None:
            related = self.memory_indexer.get_related_cases(trade_case.case_id, top_k=3)
            if related:
                evidence.append(
                    ReflectionEvidence(
                        evidence_type="related_cases",
                        observation="本地 memory index 中存在结构相近的历史案件。",
                        implication=f"related_cases={','.join(item.source_case_id for item in related)}",
                        metric_value=len(related),
                    )
                )
        return evidence
