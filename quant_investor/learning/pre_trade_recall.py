"""
Learning recall 模块。

提供可读源码实现，统一 recall 相关 dataclass 的 `schema_version` 命名，
并保持召回层只返回结构化提示，不直接改写 live 决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from quant_investor.learning.memory_indexer import MemoryItem as BaseMemoryItem
from quant_investor.learning.memory_promoter import PromotionCandidate
from quant_investor.versioning import (
    MEMORY_ITEM_SCHEMA_VERSION,
    PROMOTION_CANDIDATE_SCHEMA_VERSION,
    RECALL_HIT_SCHEMA_VERSION,
    RECALL_PACKET_SCHEMA_VERSION,
    RECALL_QUERY_SCHEMA_VERSION,
)


@dataclass
class RecallHit:
    memory_id: str
    source_case_ids: list[str]
    title: str
    statement: str
    relevance_score: float
    memory_type: str
    status: str
    cautionary: bool = False
    schema_version: str = RECALL_HIT_SCHEMA_VERSION


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
    schema_version: str = RECALL_QUERY_SCHEMA_VERSION


@dataclass
class RecallPacket:
    symbol: str
    query: RecallQuery
    hits: list[RecallHit] = field(default_factory=list)
    candidate_lessons: list[PromotionCandidate] = field(default_factory=list)
    summary: str = ""
    schema_version: str = RECALL_PACKET_SCHEMA_VERSION


@dataclass
class MemoryItem(BaseMemoryItem):
    schema_version: str = MEMORY_ITEM_SCHEMA_VERSION


class PreTradeRecall:
    """在当前 symbol 上下文下检索历史案例与经验候选。"""

    def __init__(
        self,
        *,
        memory_items: list[MemoryItem] | None = None,
        promotion_candidates: list[PromotionCandidate] | None = None,
    ) -> None:
        self.memory_items = list(memory_items or [])
        self.promotion_candidates = list(promotion_candidates or [])

    def build_query(self, current_symbol_context: dict[str, Any]) -> RecallQuery:
        as_of = current_symbol_context.get("as_of")
        if not isinstance(as_of, datetime):
            as_of = datetime.now(timezone.utc)
        return RecallQuery(
            symbol=str(current_symbol_context.get("symbol", "")),
            as_of=as_of,
            market_regime=str(current_symbol_context.get("market_regime", "unknown")),
            sector=str(current_symbol_context.get("sector", "unknown")),
            branch_support_pattern=str(current_symbol_context.get("branch_support_pattern", "unknown")),
            consensus_count=int(current_symbol_context.get("consensus_count", 0)),
            candidate_action=str(current_symbol_context.get("candidate_action", "watch")),
            volatility_regime=str(current_symbol_context.get("volatility_regime", "unknown")),
        )

    def retrieve(self, query: RecallQuery, *, top_k: int = 10) -> RecallPacket:
        ranked_hits = sorted(
            (self._to_recall_hit(item, query) for item in self.memory_items),
            key=lambda hit: (-hit.relevance_score, hit.memory_id),
        )[:top_k]

        lessons = [
            candidate
            for candidate in self.promotion_candidates
            if candidate.status in {"candidate_lesson", "validated_pattern", "approved_rule_candidate"}
        ][:top_k]

        summary = (
            f"recall_hits={len(ranked_hits)} candidate_lessons={len(lessons)} "
            f"symbol={query.symbol} regime={query.market_regime}"
        )
        return RecallPacket(symbol=query.symbol, query=query, hits=ranked_hits, candidate_lessons=lessons, summary=summary)

    @staticmethod
    def _to_recall_hit(item: MemoryItem, query: RecallQuery) -> RecallHit:
        score = 0.2
        if item.tags.market_regime == query.market_regime:
            score += 0.25
        if item.tags.sector == query.sector:
            score += 0.20
        if item.tags.branch_support_pattern == query.branch_support_pattern:
            score += 0.15
        score += min(item.confidence, 1.0) * 0.2
        cautionary = item.memory_type in {"risk_case", "counterfactual_case"} or item.tags.outcome_bucket.endswith("loss")
        return RecallHit(
            memory_id=item.memory_id,
            source_case_ids=list(item.support_cases or [item.source_case_id]),
            title=item.title,
            statement=item.statement,
            relevance_score=min(score, 1.0),
            memory_type=item.memory_type,
            status=item.status,
            cautionary=cautionary,
        )

PromotionCandidate.__dataclass_fields__["schema_version"].default = PROMOTION_CANDIDATE_SCHEMA_VERSION

__all__ = [
    "MemoryItem",
    "PromotionCandidate",
    "RecallHit",
    "RecallPacket",
    "RecallQuery",
    "PreTradeRecall",
]
