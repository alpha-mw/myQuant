"""
Post-trade learning 第二阶段的本地 memory indexer。

该模块只负责把原始 TradeCase 转成可检索的 MemoryItem 并建立结构化索引，
不接入外部向量数据库，不做经验晋升，也不修改任何 live 决策规则。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Literal
from urllib.parse import quote

from quant_investor.learning.trade_case_store import TradeCase
from quant_investor.versioning import MEMORY_INDEX_SCHEMA_VERSION, MEMORY_ITEM_SCHEMA_VERSION
_ALLOWED_MEMORY_TYPES = {
    "episodic",
    "semantic_candidate",
    "risk_case",
    "preference_case",
    "counterfactual_case",
}
_OUTCOME_BUCKETS = {
    "big_win",
    "small_win",
    "flat",
    "small_loss",
    "big_loss",
    "unresolved",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_SEARCH_PRIORITY_FIELDS = (
    "market_regime",
    "branch_support_pattern",
    "consensus_count",
    "human_action",
    "outcome_bucket",
)


@dataclass
class MemoryTags:
    market_regime: str
    style_bias: str
    sector: str
    branch_support_pattern: str
    consensus_count: int
    human_action: str
    outcome_bucket: str
    error_tags: list[str] = field(default_factory=list)
    holding_horizon: str = "unknown"
    volatility_regime: str = "unknown"

    def __post_init__(self) -> None:
        if self.outcome_bucket not in _OUTCOME_BUCKETS:
            raise ValueError(f"outcome_bucket 必须是 {_OUTCOME_BUCKETS} 之一")


@dataclass
class MemoryItem:
    memory_id: str
    source_case_id: str
    memory_type: Literal[
        "episodic",
        "semantic_candidate",
        "risk_case",
        "preference_case",
        "counterfactual_case",
    ]
    title: str
    statement: str
    tags: MemoryTags
    support_cases: list[str] = field(default_factory=list)
    counter_cases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "indexed_only"
    ttl_days: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_ITEM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.memory_type not in _ALLOWED_MEMORY_TYPES:
            raise ValueError(f"memory_type 必须是 {_ALLOWED_MEMORY_TYPES} 之一")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 区间")
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")


@dataclass
class MemoryIndexRecord:
    memory_id: str
    source_case_id: str
    symbol: str
    memory_type: str
    title: str
    statement: str
    tags: MemoryTags
    confidence: float
    status: str
    normalized_text: str
    search_terms: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")


def _normalize_text_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _normalize_string(value: Any, fallback: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def _unique_sorted_strings(values: Iterable[Any]) -> list[str]:
    normalized = {_normalize_string(value, fallback="").lower() for value in values}
    normalized.discard("")
    return sorted(normalized)


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


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
        created_at=_parse_datetime(payload["created_at"]),
        metadata=dict(payload.get("metadata", {})),
        schema_version=str(payload.get("schema_version", MEMORY_ITEM_SCHEMA_VERSION)),
    )


def _memory_index_record_from_payload(payload: dict[str, Any]) -> MemoryIndexRecord:
    return MemoryIndexRecord(
        memory_id=str(payload["memory_id"]),
        source_case_id=str(payload["source_case_id"]),
        symbol=str(payload["symbol"]),
        memory_type=str(payload["memory_type"]),
        title=str(payload["title"]),
        statement=str(payload["statement"]),
        tags=MemoryTags(**dict(payload["tags"])),
        confidence=float(payload.get("confidence", 0.0)),
        status=str(payload.get("status", "indexed_only")),
        normalized_text=str(payload.get("normalized_text", "")),
        search_terms=[str(item) for item in payload.get("search_terms", [])],
        created_at=_parse_datetime(payload["created_at"]),
        metadata=dict(payload.get("metadata", {})),
        schema_version=str(payload.get("schema_version", MEMORY_INDEX_SCHEMA_VERSION)),
    )


class MemoryIndexer:
    """TradeCase 到 MemoryItem/MemoryIndexRecord 的本地索引器。"""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.memories_dir = self.base_dir / "memory_items" if self.base_dir else None
        self.index_path = self.base_dir / "memory_index.json" if self.base_dir else None
        self._memory_items: dict[str, MemoryItem] = {}
        self._index_records: dict[str, MemoryIndexRecord] = {}
        self._case_to_memory: dict[str, str] = {}
        if self.base_dir is not None:
            self._load_from_disk()

    def build_memory_item_from_case(self, trade_case: TradeCase) -> MemoryItem:
        tags = self.extract_tags(trade_case)
        memory_type = self._infer_memory_type(trade_case, tags)
        created_at = trade_case.decision_time
        confidence = self._estimate_confidence(trade_case, tags)
        return MemoryItem(
            memory_id=self._memory_id(trade_case.case_id),
            source_case_id=trade_case.case_id,
            memory_type=memory_type,
            title=self._build_title(trade_case, memory_type, tags),
            statement=self._build_statement(trade_case, tags),
            tags=tags,
            support_cases=[trade_case.case_id],
            counter_cases=[],
            confidence=confidence,
            status="indexed_pending_outcome" if tags.outcome_bucket == "unresolved" else "indexed_only",
            ttl_days=self._ttl_days_for_memory_type(memory_type),
            created_at=created_at,
            metadata={
                "symbol": trade_case.symbol,
                "decision_time": trade_case.decision_time.isoformat(),
                "ic_action": trade_case.pretrade_snapshot.ic_action,
                "target_gross_exposure": trade_case.pretrade_snapshot.target_gross_exposure,
                "manual_override": trade_case.human_decision.manual_override,
                "override_direction": trade_case.human_decision.override_direction,
                "outcome_status": trade_case.outcomes.outcome_status,
                "attribution_summary": trade_case.attribution.summary,
                **dict(trade_case.metadata),
            },
        )

    def extract_tags(self, trade_case: TradeCase) -> MemoryTags:
        support_agents = _unique_sorted_strings(trade_case.pretrade_snapshot.support_agents)
        horizon = _normalize_string(
            trade_case.metadata.get("holding_horizon"),
            fallback=self._infer_holding_horizon(trade_case),
        )
        return MemoryTags(
            market_regime=_normalize_string(trade_case.pretrade_snapshot.market_regime),
            style_bias=_normalize_string(trade_case.metadata.get("style_bias")),
            sector=_normalize_string(trade_case.metadata.get("sector")),
            branch_support_pattern="+".join(support_agents) if support_agents else "none",
            consensus_count=len(support_agents),
            human_action=trade_case.human_decision.human_action,
            outcome_bucket=self._bucketize_outcome(trade_case),
            error_tags=_unique_sorted_strings(trade_case.error_tags),
            holding_horizon=horizon,
            volatility_regime=_normalize_string(trade_case.metadata.get("volatility_regime")),
        )

    def index_case(self, trade_case: TradeCase) -> MemoryIndexRecord:
        memory_item = self.build_memory_item_from_case(trade_case)
        record = self._build_index_record(memory_item)
        self._memory_items[memory_item.memory_id] = memory_item
        self._index_records[memory_item.memory_id] = record
        self._case_to_memory[trade_case.case_id] = memory_item.memory_id
        self._persist_memory_item(memory_item)
        self._persist_index()
        return record

    def rebuild_index(self, cases: Iterable[TradeCase]) -> list[MemoryIndexRecord]:
        self._memory_items.clear()
        self._index_records.clear()
        self._case_to_memory.clear()
        if self.memories_dir is not None and self.memories_dir.exists():
            for path in self.memories_dir.glob("*.json"):
                path.unlink()

        for trade_case in cases:
            memory_item = self.build_memory_item_from_case(trade_case)
            record = self._build_index_record(memory_item)
            self._memory_items[memory_item.memory_id] = memory_item
            self._index_records[memory_item.memory_id] = record
            self._case_to_memory[trade_case.case_id] = memory_item.memory_id
            self._persist_memory_item(memory_item)

        self._persist_index()
        return self._sorted_records(self._index_records.values())

    def search_by_tags(
        self,
        *,
        market_regime: str | None = None,
        branch_support_pattern: str | None = None,
        consensus_count: int | None = None,
        human_action: str | None = None,
        outcome_bucket: str | None = None,
        error_tags: list[str] | None = None,
        text_query: str | None = None,
        top_k: int | None = None,
    ) -> list[MemoryIndexRecord]:
        normalized_error_tags = _unique_sorted_strings(error_tags or [])
        query_terms = set(_normalize_text_tokens(text_query or ""))
        scored: list[tuple[float, MemoryIndexRecord]] = []

        for record in self._index_records.values():
            if market_regime is not None and record.tags.market_regime != market_regime:
                continue
            if (
                branch_support_pattern is not None
                and record.tags.branch_support_pattern != branch_support_pattern
            ):
                continue
            if consensus_count is not None and record.tags.consensus_count != consensus_count:
                continue
            if human_action is not None and record.tags.human_action != human_action:
                continue
            if outcome_bucket is not None and record.tags.outcome_bucket != outcome_bucket:
                continue
            if normalized_error_tags and not set(normalized_error_tags).issubset(record.tags.error_tags):
                continue

            overlap = query_terms.intersection(record.search_terms)
            if query_terms and not overlap:
                continue

            score = self._search_score(
                record,
                market_regime=market_regime,
                branch_support_pattern=branch_support_pattern,
                consensus_count=consensus_count,
                human_action=human_action,
                outcome_bucket=outcome_bucket,
                error_tag_matches=len(normalized_error_tags),
                text_matches=len(overlap),
            )
            scored.append((score, record))

        ranked = [
            record
            for _, record in sorted(
                scored,
                key=lambda item: (-item[0], -item[1].confidence, item[1].source_case_id),
            )
        ]
        return ranked[:top_k] if top_k is not None else ranked

    def get_related_cases(self, case_id: str, top_k: int = 5) -> list[MemoryIndexRecord]:
        memory_id = self._case_to_memory.get(case_id)
        if memory_id is None:
            return []

        anchor = self._index_records[memory_id]
        scored: list[tuple[float, MemoryIndexRecord]] = []
        anchor_terms = set(anchor.search_terms)
        anchor_errors = set(anchor.tags.error_tags)

        for candidate in self._index_records.values():
            if candidate.source_case_id == case_id:
                continue
            score = 0.0
            for field_name in _SEARCH_PRIORITY_FIELDS:
                if getattr(anchor.tags, field_name) == getattr(candidate.tags, field_name):
                    score += {
                        "market_regime": 4.0,
                        "branch_support_pattern": 4.0,
                        "consensus_count": 2.0,
                        "human_action": 2.0,
                        "outcome_bucket": 4.0,
                    }[field_name]
            score += float(len(anchor_errors.intersection(candidate.tags.error_tags))) * 1.5
            score += float(len(anchor_terms.intersection(candidate.search_terms))) * 0.25
            if anchor.tags.style_bias == candidate.tags.style_bias:
                score += 1.0
            if anchor.tags.sector == candidate.tags.sector:
                score += 1.0
            if score > 0:
                scored.append((score, candidate))

        ranked = [
            record
            for _, record in sorted(
                scored,
                key=lambda item: (-item[0], -item[1].confidence, item[1].source_case_id),
            )
        ]
        return ranked[:top_k]

    @staticmethod
    def _memory_id(case_id: str) -> str:
        return f"memory::{quote(case_id, safe='')}"

    @staticmethod
    def _ttl_days_for_memory_type(memory_type: str) -> int | None:
        return {
            "episodic": 365,
            "semantic_candidate": 180,
            "risk_case": 730,
            "preference_case": 365,
            "counterfactual_case": 180,
        }.get(memory_type, 365)

    @staticmethod
    def _infer_holding_horizon(trade_case: TradeCase) -> str:
        if trade_case.outcomes.t20_return is not None:
            return "t20"
        if trade_case.outcomes.t10_return is not None:
            return "t10"
        if trade_case.outcomes.t5_return is not None:
            return "t5"
        if trade_case.outcomes.t1_return is not None:
            return "t1"
        return "unknown"

    @staticmethod
    def _bucketize_outcome(trade_case: TradeCase) -> str:
        if str(trade_case.outcomes.outcome_status).strip().lower() in {"pending", "unresolved", ""}:
            return "unresolved"

        for attr in ("t20_return", "t10_return", "t5_return", "t1_return"):
            value = getattr(trade_case.outcomes, attr)
            if value is None:
                continue
            if value >= 0.08:
                return "big_win"
            if value >= 0.01:
                return "small_win"
            if value > -0.01:
                return "flat"
            if value > -0.05:
                return "small_loss"
            return "big_loss"
        return "unresolved"

    @staticmethod
    def _infer_memory_type(trade_case: TradeCase, tags: MemoryTags) -> str:
        if trade_case.human_decision.human_action == "skipped":
            return "counterfactual_case"
        if trade_case.human_decision.human_action == "overridden":
            return "preference_case"
        if (
            tags.outcome_bucket in {"small_loss", "big_loss"}
            or bool(trade_case.attribution.missed_risks)
            or bool(trade_case.outcomes.stop_loss_hit)
        ):
            return "risk_case"
        if trade_case.lesson_draft:
            return "semantic_candidate"
        return "episodic"

    @staticmethod
    def _estimate_confidence(trade_case: TradeCase, tags: MemoryTags) -> float:
        branch_confidences = list(trade_case.pretrade_snapshot.branch_confidences.values())
        avg_branch_conf = (
            sum(float(value) for value in branch_confidences) / len(branch_confidences)
            if branch_confidences
            else 0.5
        )
        consensus_bonus = min(tags.consensus_count, 4) * 0.08
        outcome_bonus = 0.0 if tags.outcome_bucket == "unresolved" else 0.08
        human_bonus = 0.05 if tags.human_action == "executed" else 0.0
        confidence = 0.25 + avg_branch_conf * 0.4 + consensus_bonus + outcome_bonus + human_bonus
        return round(min(0.95, max(0.1, confidence)), 4)

    @staticmethod
    def _build_title(trade_case: TradeCase, memory_type: str, tags: MemoryTags) -> str:
        return (
            f"{trade_case.symbol} {memory_type} "
            f"{trade_case.human_decision.human_action} {tags.outcome_bucket}"
        )

    @staticmethod
    def _build_statement(trade_case: TradeCase, tags: MemoryTags) -> str:
        components = [
            f"regime={tags.market_regime}",
            f"ic_action={trade_case.pretrade_snapshot.ic_action or 'unknown'}",
            f"support={tags.branch_support_pattern}",
            f"human_action={tags.human_action}",
            f"outcome_bucket={tags.outcome_bucket}",
        ]
        if trade_case.attribution.summary.strip():
            components.append(trade_case.attribution.summary.strip())
        elif trade_case.lesson_draft:
            components.append(trade_case.lesson_draft[-1].strip())
        if trade_case.error_tags:
            components.append(f"error_tags={','.join(_unique_sorted_strings(trade_case.error_tags))}")
        return " | ".join(components)

    def _build_index_record(self, memory_item: MemoryItem) -> MemoryIndexRecord:
        search_terms = sorted(
            set(
                _normalize_text_tokens(
                    " ".join(
                        [
                            memory_item.title,
                            memory_item.statement,
                            memory_item.tags.market_regime,
                            memory_item.tags.branch_support_pattern,
                            memory_item.tags.human_action,
                            memory_item.tags.outcome_bucket,
                            " ".join(memory_item.tags.error_tags),
                        ]
                    )
                )
            )
        )
        normalized_text = " ".join(search_terms)
        return MemoryIndexRecord(
            memory_id=memory_item.memory_id,
            source_case_id=memory_item.source_case_id,
            symbol=str(memory_item.metadata.get("symbol", "unknown")),
            memory_type=memory_item.memory_type,
            title=memory_item.title,
            statement=memory_item.statement,
            tags=memory_item.tags,
            confidence=memory_item.confidence,
            status=memory_item.status,
            normalized_text=normalized_text,
            search_terms=search_terms,
            created_at=memory_item.created_at,
            metadata=dict(memory_item.metadata),
        )

    @staticmethod
    def _search_score(
        record: MemoryIndexRecord,
        *,
        market_regime: str | None,
        branch_support_pattern: str | None,
        consensus_count: int | None,
        human_action: str | None,
        outcome_bucket: str | None,
        error_tag_matches: int,
        text_matches: int,
    ) -> float:
        score = 0.0
        if market_regime is not None:
            score += 4.0
        if branch_support_pattern is not None:
            score += 4.0
        if consensus_count is not None:
            score += 2.0
        if human_action is not None:
            score += 2.0
        if outcome_bucket is not None:
            score += 4.0
        score += float(error_tag_matches) * 1.5
        score += float(text_matches)
        score += record.confidence
        return score

    @staticmethod
    def _sorted_records(records: Iterable[MemoryIndexRecord]) -> list[MemoryIndexRecord]:
        return sorted(records, key=lambda item: (item.created_at, item.source_case_id))

    def _load_from_disk(self) -> None:
        if self.memories_dir is not None and self.memories_dir.exists():
            for path in sorted(self.memories_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                item = _memory_item_from_payload(payload)
                self._memory_items[item.memory_id] = item
                self._case_to_memory[item.source_case_id] = item.memory_id

        if self.index_path is not None and self.index_path.exists():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            for raw_record in payload.get("records", []):
                record = _memory_index_record_from_payload(dict(raw_record))
                if record.memory_id in self._memory_items:
                    self._index_records[record.memory_id] = record

    def _persist_memory_item(self, memory_item: MemoryItem) -> None:
        if self.memories_dir is None:
            return
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        path = self.memories_dir / f"{quote(memory_item.memory_id, safe='')}.json"
        self._write_json(path, _serialize_value(memory_item))

    def _persist_index(self) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEMORY_INDEX_SCHEMA_VERSION,
            "records": [_serialize_value(record) for record in self._sorted_records(self._index_records.values())],
        }
        self._write_json(self.index_path, payload)

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
