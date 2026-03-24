"""
Post-trade learning 第五阶段的 pre-trade recall。

该模块只做参考检索与摘要，不直接修改 live action、branch score 或 target weight。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from quant_investor.learning.memory_indexer import MemoryItem
from quant_investor.learning.memory_promoter import PromotionCandidate


@dataclass
class RecallQuery:
    symbol: str
    as_of: datetime
    market_regime: str
    sector: str
    branch_support_pattern: str
    consensus_count: int
    candidate_action: str
    volatility_regime: str
    error_tags: list[str] = field(default_factory=list)


@dataclass
class RecallHit:
    memory_id: str
    source_case_ids: list[str]
    title: str
    statement: str
    relevance_score: float
    memory_type: str
    status: str
    cautionary: bool
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallPacket:
    symbol: str
    query: RecallQuery
    similar_cases: list[RecallHit] = field(default_factory=list)
    cautionary_patterns: list[RecallHit] = field(default_factory=list)
    validated_patterns: list[RecallHit] = field(default_factory=list)
    pending_rule_candidates: list[RecallHit] = field(default_factory=list)
    summary_for_ic: str = ""
    summary_for_risk_guard: str = ""


class PreTradeRecall:
    """在 pre-trade 阶段召回历史案例与候选经验。"""

    def __init__(
        self,
        *,
        memory_items: Iterable[MemoryItem] | None = None,
        promotion_candidates: Iterable[PromotionCandidate] | None = None,
    ) -> None:
        self.memory_items = list(memory_items or [])
        self.promotion_candidates = list(promotion_candidates or [])

    def build_query(self, current_symbol_context: Mapping[str, Any]) -> RecallQuery:
        support_agents = current_symbol_context.get("support_agents", [])
        branch_support_pattern = current_symbol_context.get("branch_support_pattern")
        if branch_support_pattern:
            normalized_pattern = self._normalize_branch_support_pattern(branch_support_pattern)
        else:
            normalized_pattern = self._normalize_branch_support_pattern(support_agents)

        consensus_count = current_symbol_context.get("consensus_count")
        if consensus_count is None:
            consensus_count = len(self._support_pattern_parts(normalized_pattern))

        as_of = current_symbol_context.get("as_of", datetime.now())
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of)

        return RecallQuery(
            symbol=str(current_symbol_context.get("symbol", "")).strip(),
            as_of=as_of,
            market_regime=self._normalize_string(current_symbol_context.get("market_regime")),
            sector=self._normalize_string(current_symbol_context.get("sector")),
            branch_support_pattern=normalized_pattern,
            consensus_count=int(consensus_count or 0),
            candidate_action=self._normalize_string(
                current_symbol_context.get("candidate_action"),
                fallback="unknown",
            ),
            volatility_regime=self._normalize_string(
                current_symbol_context.get("volatility_regime")
            ),
            error_tags=self._normalize_string_list(current_symbol_context.get("error_tags", [])),
        )

    def retrieve(self, query: RecallQuery, top_k: int = 10) -> RecallPacket:
        memory_hits = [
            self._memory_hit_from_item(item, query)
            for item in self.memory_items
        ]
        ranked_memory_hits = [
            hit
            for hit in sorted(
                memory_hits,
                key=lambda item: (-item.relevance_score, not item.cautionary, item.memory_id),
            )
            if hit.relevance_score > 0
        ][:top_k]

        cautionary_patterns = [hit for hit in ranked_memory_hits if hit.cautionary][:top_k]
        validated_patterns = self._candidate_hits(query, status="validated_pattern", top_k=top_k)
        pending_rule_candidates = self._candidate_hits(
            query,
            status="approved_rule_candidate",
            top_k=top_k,
        )

        packet = RecallPacket(
            symbol=query.symbol,
            query=query,
            similar_cases=ranked_memory_hits,
            cautionary_patterns=cautionary_patterns,
            validated_patterns=validated_patterns,
            pending_rule_candidates=pending_rule_candidates,
        )
        packet.summary_for_ic = self.build_summary_for_ic(packet)
        packet.summary_for_risk_guard = self.build_summary_for_risk_guard(packet)
        return packet

    def score_relevance(self, memory_item: MemoryItem, query: RecallQuery) -> float:
        tags = memory_item.tags
        score = 0.0

        if tags.market_regime == query.market_regime:
            score += 6.0
        if tags.branch_support_pattern == query.branch_support_pattern:
            score += 5.0
        if tags.consensus_count == query.consensus_count:
            score += 2.0
        elif abs(tags.consensus_count - query.consensus_count) == 1:
            score += 1.0
        if tags.sector == query.sector:
            score += 1.5
        if tags.volatility_regime == query.volatility_regime:
            score += 1.5
        if query.symbol and str(memory_item.metadata.get("symbol", "")) == query.symbol:
            score += 0.5
        if query.candidate_action != "unknown" and str(memory_item.metadata.get("ic_action", "")) == query.candidate_action:
            score += 1.0

        error_overlap = len(set(query.error_tags).intersection(tags.error_tags))
        score += float(error_overlap) * 1.25

        if memory_item.memory_type == "risk_case":
            score += 3.0
        elif memory_item.memory_type == "counterfactual_case":
            score += 2.0

        if tags.outcome_bucket in {"big_loss", "small_loss"}:
            score += 2.0
        elif tags.outcome_bucket in {"big_win", "small_win"}:
            score += 0.75

        if memory_item.status == "indexed_pending_outcome":
            score -= 0.5

        score += float(memory_item.confidence)
        return round(max(0.0, score), 4)

    def build_summary_for_ic(self, recall_packet: RecallPacket) -> str:
        lines: list[str] = []
        if recall_packet.similar_cases:
            top_cases = ", ".join(
                f"{hit.title}({hit.relevance_score:.2f})"
                for hit in recall_packet.similar_cases[:3]
            )
            lines.append(f"相似 setup 参考: {top_cases}")
        if recall_packet.validated_patterns:
            top_patterns = ", ".join(
                hit.statement for hit in recall_packet.validated_patterns[:2]
            )
            lines.append(f"已验证模式参考: {top_patterns}")
        if recall_packet.pending_rule_candidates:
            pending_text = ", ".join(
                hit.statement for hit in recall_packet.pending_rule_candidates[:2]
            )
            lines.append(
                f"待审批规则候选仅供参考: {pending_text}"
            )
        if not lines:
            lines.append("未检索到高相关历史参考，保持当前研究结论独立评估。")
        return " | ".join(lines)

    def build_summary_for_risk_guard(self, recall_packet: RecallPacket) -> str:
        lines: list[str] = []
        if recall_packet.cautionary_patterns:
            caution_text = ", ".join(
                f"{hit.statement}({hit.relevance_score:.2f})"
                for hit in recall_packet.cautionary_patterns[:3]
            )
            lines.append(f"历史失败/风险模式提醒: {caution_text}")
        if recall_packet.pending_rule_candidates:
            pending_text = ", ".join(
                hit.statement for hit in recall_packet.pending_rule_candidates[:2]
            )
            lines.append(
                f"存在待审批风险候选，不能视为已生效规则: {pending_text}"
            )
        if not lines:
            lines.append("未发现高相关风险记忆，仍需按当前风控规则独立审查。")
        return " | ".join(lines)

    def _memory_hit_from_item(self, memory_item: MemoryItem, query: RecallQuery) -> RecallHit:
        score = self.score_relevance(memory_item, query)
        cautionary = self._is_cautionary_memory(memory_item)
        return RecallHit(
            memory_id=memory_item.memory_id,
            source_case_ids=list(memory_item.support_cases) or [memory_item.source_case_id],
            title=memory_item.title,
            statement=memory_item.statement,
            relevance_score=score,
            memory_type=memory_item.memory_type,
            status=memory_item.status,
            cautionary=cautionary,
            tags=asdict(memory_item.tags),
        )

    def _candidate_hits(
        self,
        query: RecallQuery,
        *,
        status: str,
        top_k: int,
    ) -> list[RecallHit]:
        hits: list[RecallHit] = []
        for candidate in self.promotion_candidates:
            if candidate.status != status:
                continue
            score = self._score_candidate(candidate, query)
            if score <= 0:
                continue
            hits.append(
                RecallHit(
                    memory_id=candidate.candidate_id,
                    source_case_ids=list(candidate.source_case_ids),
                    title=f"{candidate.lesson_type} | {candidate.status}",
                    statement=candidate.lesson_statement,
                    relevance_score=score,
                    memory_type=candidate.lesson_type,
                    status=candidate.status,
                    cautionary=candidate.lesson_type == "risk_rule_candidate",
                    tags={
                        "regimes_seen": list(candidate.regimes_seen),
                        "sectors_seen": list(candidate.sectors_seen),
                        "support_count": candidate.support_count,
                        "counter_count": candidate.counter_count,
                    },
                )
            )
        return sorted(
            hits,
            key=lambda item: (-item.relevance_score, not item.cautionary, item.memory_id),
        )[:top_k]

    @staticmethod
    def _score_candidate(candidate: PromotionCandidate, query: RecallQuery) -> float:
        score = 0.0
        if query.market_regime in candidate.regimes_seen:
            score += 5.0
        if query.sector in candidate.sectors_seen:
            score += 1.5
        if candidate.lesson_type == "risk_rule_candidate":
            score += 3.0
        if candidate.status == "validated_pattern":
            score += 1.5
        elif candidate.status == "approved_rule_candidate":
            score += 2.0
        score += min(candidate.support_count, 5) * 0.5
        score += min(candidate.counter_count, 3) * 0.3
        score += float(candidate.confidence)
        return round(score, 4)

    @staticmethod
    def _is_cautionary_memory(memory_item: MemoryItem) -> bool:
        return (
            memory_item.memory_type in {"risk_case", "counterfactual_case"}
            or memory_item.tags.outcome_bucket in {"small_loss", "big_loss"}
        )

    @staticmethod
    def _normalize_string(value: Any, fallback: str = "unknown") -> str:
        text = str(value).strip() if value is not None else ""
        return text if text else fallback

    @staticmethod
    def _normalize_string_list(values: Iterable[Any]) -> list[str]:
        normalized = {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }
        return sorted(normalized)

    @classmethod
    def _normalize_branch_support_pattern(cls, value: Any) -> str:
        if isinstance(value, str):
            if "+" in value:
                parts = value.split("+")
            elif "," in value:
                parts = value.split(",")
            else:
                parts = [value]
        else:
            parts = list(value or [])
        normalized = sorted(
            {
                str(item).strip().lower()
                for item in parts
                if str(item).strip()
            }
        )
        return "+".join(normalized) if normalized else "none"

    @staticmethod
    def _support_pattern_parts(branch_support_pattern: str) -> list[str]:
        if not branch_support_pattern or branch_support_pattern == "none":
            return []
        return [item for item in branch_support_pattern.split("+") if item]
