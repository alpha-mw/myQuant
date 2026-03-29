"""
Post-trade learning 轻量 orchestrator。

负责把 trade_case_store -> memory_indexer -> post_trade_reflector ->
memory_promoter -> pre_trade_recall 串成闭环，但不会自动修改 live 规则。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from quant_investor.learning.memory_indexer import MemoryIndexer, MemoryItem, MemoryTags
from quant_investor.learning.memory_promoter import (
    MemoryPromoter,
    PromotionCandidate,
    PromotionDecision,
    RuleProposal,
)
from quant_investor.learning.post_trade_reflector import (
    PostTradeReflector,
    ReflectionEvidence,
    ReflectionLessonDraft,
    ReflectionReport,
)
from quant_investor.learning.pre_trade_recall import (
    PreTradeRecall,
    RecallHit,
    RecallPacket,
    RecallQuery,
)
from quant_investor.learning.trade_case_store import TradeCase, TradeCaseStore
from quant_investor.versioning import MEMORY_ITEM_SCHEMA_VERSION


class LearningOrchestrator:
    """学习闭环的本地协调器。"""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        promoter: MemoryPromoter | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.trade_case_store = TradeCaseStore(self.base_dir)
        self.memory_indexer = MemoryIndexer(self.base_dir)
        self.post_trade_reflector = PostTradeReflector(memory_indexer=self.memory_indexer)
        self.memory_promoter = promoter or MemoryPromoter()
        self.reflections_dir = self.base_dir / "reflections"
        self.promotion_candidates_dir = self.base_dir / "promotion_candidates"
        self.promotion_decisions_dir = self.base_dir / "promotion_decisions"
        self.rule_proposals_dir = self.base_dir / "rule_proposals"
        self.recall_packets_dir = self.base_dir / "recall_packets"

    def save_trade_case(self, trade_case: TradeCase) -> Path:
        return self.trade_case_store.save_case(trade_case)

    def index_case(self, case_id: str) -> MemoryItem | None:
        trade_case = self.trade_case_store.load_case(case_id)
        if not self._case_has_observed_outcome(trade_case):
            return None
        self.memory_indexer.index_case(trade_case)
        return self.memory_indexer.build_memory_item_from_case(trade_case)

    def reflect_case(self, case_id: str) -> ReflectionReport | None:
        trade_case = self.trade_case_store.load_case(case_id)
        if not self._case_has_observed_outcome(trade_case):
            return None
        report = self.post_trade_reflector.reflect_case(trade_case)
        self._persist_reflection(report)
        return report

    def run_promotion_cycle(
        self,
    ) -> tuple[list[PromotionCandidate], list[PromotionDecision], list[RuleProposal]]:
        cases = self.trade_case_store.list_cases()
        reflection_reports = self.load_reflection_reports()
        raw_candidates = self.memory_promoter.collect_candidates(cases, reflection_reports)

        promoted_candidates: list[PromotionCandidate] = []
        decisions: list[PromotionDecision] = []
        proposals: list[RuleProposal] = []

        self._clear_json_dir(self.promotion_candidates_dir)
        self._clear_json_dir(self.promotion_decisions_dir)
        self._clear_json_dir(self.rule_proposals_dir)

        for candidate in raw_candidates:
            decision = self.memory_promoter.validate_candidate(candidate, cases)
            decisions.append(decision)
            if decision.target_status == "rejected":
                updated_candidate = self.memory_promoter.archive_rejected(candidate)
            else:
                updated_candidate = self.memory_promoter.promote(candidate, decision)

            promoted_candidates.append(updated_candidate)
            self._persist_promotion_candidate(updated_candidate)
            self._persist_promotion_decision(decision)

            if updated_candidate.status in {
                "candidate_lesson",
                "validated_pattern",
                "approved_rule_candidate",
            }:
                proposal = self.memory_promoter.build_rule_proposal(updated_candidate)
                proposals.append(proposal)
                self._persist_rule_proposal(proposal)

        return promoted_candidates, decisions, proposals

    def build_recall_packet(
        self,
        current_symbol_context: Mapping[str, Any],
        *,
        top_k: int = 10,
    ) -> RecallPacket:
        recall = PreTradeRecall(
            memory_items=self.load_memory_items(),
            promotion_candidates=self.load_promotion_candidates(),
        )
        query = recall.build_query(current_symbol_context)
        packet = recall.retrieve(query, top_k=top_k)
        self._persist_recall_packet(packet)
        return packet

    def run_closed_loop(
        self,
        trade_case: TradeCase,
        *,
        recall_context: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        case_path = self.save_trade_case(trade_case)
        indexed_memory = None
        reflection_report = None
        promotion_candidates: list[PromotionCandidate] = []
        promotion_decisions: list[PromotionDecision] = []
        rule_proposals: list[RuleProposal] = []

        if self._case_has_observed_outcome(trade_case):
            indexed_memory = self.index_case(trade_case.case_id)
            reflection_report = self.reflect_case(trade_case.case_id)
            promotion_candidates, promotion_decisions, rule_proposals = self.run_promotion_cycle()

        recall_packet = None
        if recall_context is not None:
            recall_packet = self.build_recall_packet(recall_context, top_k=top_k)

        return {
            "case_path": case_path,
            "memory_item": indexed_memory,
            "reflection_report": reflection_report,
            "promotion_candidates": promotion_candidates,
            "promotion_decisions": promotion_decisions,
            "rule_proposals": rule_proposals,
            "recall_packet": recall_packet,
            "live_updates": [],
        }

    def load_reflection_reports(self) -> list[ReflectionReport]:
        reports: list[ReflectionReport] = []
        for path in sorted(self.reflections_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            reports.append(self._reflection_report_from_payload(payload))
        return reports

    def load_promotion_candidates(self) -> list[PromotionCandidate]:
        candidates: list[PromotionCandidate] = []
        for path in sorted(self.promotion_candidates_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates.append(
                PromotionCandidate(
                    candidate_id=str(payload["candidate_id"]),
                    source_case_ids=[str(item) for item in payload.get("source_case_ids", [])],
                    lesson_statement=str(payload["lesson_statement"]),
                    lesson_type=str(payload["lesson_type"]),
                    support_count=int(payload["support_count"]),
                    counter_count=int(payload["counter_count"]),
                    regimes_seen=[str(item) for item in payload.get("regimes_seen", [])],
                    sectors_seen=[str(item) for item in payload.get("sectors_seen", [])],
                    confidence=float(payload["confidence"]),
                    status=str(payload["status"]),
                    evidence_summary=str(payload["evidence_summary"]),
                    counter_case_ids=[str(item) for item in payload.get("counter_case_ids", [])],
                )
            )
        return candidates

    def load_memory_items(self) -> list[MemoryItem]:
        reloaded_indexer = MemoryIndexer(self.base_dir)
        return list(reloaded_indexer._memory_items.values())

    @staticmethod
    def _case_has_observed_outcome(trade_case: TradeCase) -> bool:
        if str(trade_case.outcomes.outcome_status).strip().lower() not in {"", "pending", "unresolved"}:
            return True
        return any(
            getattr(trade_case.outcomes, attr) is not None
            for attr in ("t1_return", "t5_return", "t10_return", "t20_return", "mfe", "mae")
        )

    def _persist_reflection(self, report: ReflectionReport) -> Path:
        path = self.reflections_dir / f"{quote(report.case_id, safe='')}.json"
        self._write_json(path, self._serialize(report))
        return path

    def _persist_promotion_candidate(self, candidate: PromotionCandidate) -> Path:
        path = self.promotion_candidates_dir / f"{quote(candidate.candidate_id, safe='')}.json"
        self._write_json(path, self._serialize(candidate))
        return path

    def _persist_promotion_decision(self, decision: PromotionDecision) -> Path:
        path = self.promotion_decisions_dir / f"{quote(decision.candidate_id, safe='')}.json"
        self._write_json(path, self._serialize(decision))
        return path

    def _persist_rule_proposal(self, proposal: RuleProposal) -> Path:
        path = self.rule_proposals_dir / f"{quote(proposal.proposal_id, safe='')}.json"
        self._write_json(path, self._serialize(proposal))
        return path

    def _persist_recall_packet(self, packet: RecallPacket) -> Path:
        timestamp = packet.query.as_of.isoformat().replace(":", "_")
        path = self.recall_packets_dir / f"{quote(packet.symbol, safe='')}_{timestamp}.json"
        self._write_json(path, self._serialize(packet))
        return path

    @staticmethod
    def _serialize(value: Any) -> Any:
        if is_dataclass(value):
            return {
                key: LearningOrchestrator._serialize(item)
                for key, item in asdict(value).items()
            }
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): LearningOrchestrator._serialize(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [LearningOrchestrator._serialize(item) for item in value]
        return value

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            tmp_path = Path(handle.name)
        tmp_path.replace(path)

    @staticmethod
    def _clear_json_dir(path: Path) -> None:
        if not path.exists():
            return
        for file_path in path.glob("*.json"):
            file_path.unlink()

    @staticmethod
    def _reflection_report_from_payload(payload: dict[str, Any]) -> ReflectionReport:
        return ReflectionReport(
            case_id=str(payload["case_id"]),
            symbol=str(payload["symbol"]),
            thesis_validation=str(payload["thesis_validation"]),
            timing_assessment=str(payload["timing_assessment"]),
            risk_control_assessment=str(payload["risk_control_assessment"]),
            human_override_assessment=str(payload["human_override_assessment"]),
            key_success_factors=[str(item) for item in payload.get("key_success_factors", [])],
            key_failure_factors=[str(item) for item in payload.get("key_failure_factors", [])],
            lesson_drafts=[
                ReflectionLessonDraft(
                    lesson_type=str(item["lesson_type"]),
                    statement=str(item["statement"]),
                    rationale=str(item["rationale"]),
                    confidence=float(item["confidence"]),
                    promotion_recommendation=str(item["promotion_recommendation"]),
                )
                for item in payload.get("lesson_drafts", [])
            ],
            suggested_error_tags=[str(item) for item in payload.get("suggested_error_tags", [])],
            summary=str(payload.get("summary", "")),
            evidence=[
                ReflectionEvidence(
                    evidence_type=str(item["evidence_type"]),
                    observation=str(item["observation"]),
                    implication=str(item["implication"]),
                    metric_value=item.get("metric_value"),
                )
                for item in payload.get("evidence", [])
            ],
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
        )


def _memory_item_from_payload(payload: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        memory_id=str(payload["memory_id"]),
        source_case_id=str(payload["source_case_id"]),
        memory_type=str(payload["memory_type"]),
        title=str(payload["title"]),
        statement=str(payload["statement"]),
        tags=MemoryTags(**dict(payload["tags"])),
        support_cases=[str(item) for item in payload.get("support_cases", [])],
        counter_cases=[str(item) for item in payload.get("counter_cases", [])],
        confidence=float(payload.get("confidence", 0.0)),
        status=str(payload.get("status", "indexed_only")),
        ttl_days=payload.get("ttl_days"),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        metadata=dict(payload.get("metadata", {})),
        schema_version=str(payload.get("schema_version", MEMORY_ITEM_SCHEMA_VERSION)),
    )


def _recall_query_from_payload(payload: dict[str, Any]) -> RecallQuery:
    return RecallQuery(
        symbol=str(payload["symbol"]),
        as_of=datetime.fromisoformat(str(payload["as_of"])),
        market_regime=str(payload["market_regime"]),
        sector=str(payload["sector"]),
        branch_support_pattern=str(payload["branch_support_pattern"]),
        consensus_count=int(payload["consensus_count"]),
        candidate_action=str(payload["candidate_action"]),
        volatility_regime=str(payload["volatility_regime"]),
        error_tags=[str(item) for item in payload.get("error_tags", [])],
    )


def _recall_hit_from_payload(payload: dict[str, Any]) -> RecallHit:
    return RecallHit(
        memory_id=str(payload["memory_id"]),
        source_case_ids=[str(item) for item in payload.get("source_case_ids", [])],
        title=str(payload["title"]),
        statement=str(payload["statement"]),
        relevance_score=float(payload["relevance_score"]),
        memory_type=str(payload["memory_type"]),
        status=str(payload["status"]),
        cautionary=bool(payload["cautionary"]),
        tags=dict(payload.get("tags", {})),
    )


def _recall_packet_from_payload(payload: dict[str, Any]) -> RecallPacket:
    return RecallPacket(
        symbol=str(payload["symbol"]),
        query=_recall_query_from_payload(dict(payload["query"])),
        similar_cases=[_recall_hit_from_payload(dict(item)) for item in payload.get("similar_cases", [])],
        cautionary_patterns=[
            _recall_hit_from_payload(dict(item))
            for item in payload.get("cautionary_patterns", [])
        ],
        validated_patterns=[
            _recall_hit_from_payload(dict(item))
            for item in payload.get("validated_patterns", [])
        ],
        pending_rule_candidates=[
            _recall_hit_from_payload(dict(item))
            for item in payload.get("pending_rule_candidates", [])
        ],
        summary_for_ic=str(payload.get("summary_for_ic", "")),
        summary_for_risk_guard=str(payload.get("summary_for_risk_guard", "")),
    )
