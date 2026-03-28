"""
Post-trade learning 第一阶段的原始交易案件存储层。

该模块只负责结构化案件的持久化与更新，不参与 agent 编排、
报告生成或 live 决策逻辑，也不做任何自动晋升。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Literal
from urllib.parse import quote


TRADE_CASE_SCHEMA_VERSION = "trade_case.v1"
_ALLOWED_HUMAN_ACTIONS = {"executed", "skipped", "overridden"}


@dataclass
class PreTradeSnapshot:
    market_regime: str
    target_gross_exposure: float | None
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_confidences: dict[str, float] = field(default_factory=dict)
    ic_action: str = ""
    ic_thesis: str = ""
    support_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)


@dataclass
class HumanDecisionSnapshot:
    human_action: Literal["executed", "skipped", "overridden"]
    human_reason: list[str] = field(default_factory=list)
    manual_override: bool = False
    override_direction: str | None = None
    override_notes: str = ""

    def __post_init__(self) -> None:
        if self.human_action not in _ALLOWED_HUMAN_ACTIONS:
            raise ValueError(f"human_action 必须是 {_ALLOWED_HUMAN_ACTIONS} 之一")


@dataclass
class ExecutionSnapshot:
    executed: bool
    entry_price: float | None = None
    entry_time: datetime | None = None
    quantity: float | None = None
    manual_notes: str = ""


@dataclass
class OutcomeSnapshot:
    t1_return: float | None = None
    t5_return: float | None = None
    t10_return: float | None = None
    t20_return: float | None = None
    mfe: float | None = None
    mae: float | None = None
    stop_loss_hit: bool | None = None
    take_profit_hit: bool | None = None
    outcome_status: str = "pending"


@dataclass
class AttributionSnapshot:
    helpful_agents: list[str] = field(default_factory=list)
    misleading_agents: list[str] = field(default_factory=list)
    correct_risk_controls: list[str] = field(default_factory=list)
    missed_risks: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class TradeCase:
    case_id: str
    symbol: str
    decision_time: datetime
    pretrade_snapshot: PreTradeSnapshot
    human_decision: HumanDecisionSnapshot
    execution_snapshot: ExecutionSnapshot
    outcomes: OutcomeSnapshot = field(default_factory=OutcomeSnapshot)
    attribution: AttributionSnapshot = field(default_factory=AttributionSnapshot)
    error_tags: list[str] = field(default_factory=list)
    lesson_draft: list[str] = field(default_factory=list)
    memory_status: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRADE_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id 不能为空")
        if not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not isinstance(self.decision_time, datetime):
            raise TypeError("decision_time 必须是 datetime")
        if not self.schema_version.strip():
            raise ValueError("schema_version 不能为空")


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _parse_filter_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value).date()


def _trade_case_from_payload(payload: dict[str, Any]) -> TradeCase:
    schema_version = str(payload.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError("TradeCase payload 缺少 schema_version")
    if schema_version != TRADE_CASE_SCHEMA_VERSION:
        raise ValueError(f"不支持的 schema_version: {schema_version}")

    return TradeCase(
        case_id=str(payload["case_id"]),
        symbol=str(payload["symbol"]),
        decision_time=_parse_datetime(payload["decision_time"]) or datetime.min,
        pretrade_snapshot=PreTradeSnapshot(**dict(payload["pretrade_snapshot"])),
        human_decision=HumanDecisionSnapshot(**dict(payload["human_decision"])),
        execution_snapshot=ExecutionSnapshot(
            **{
                **dict(payload["execution_snapshot"]),
                "entry_time": _parse_datetime(dict(payload["execution_snapshot"]).get("entry_time")),
            }
        ),
        outcomes=OutcomeSnapshot(**dict(payload.get("outcomes", {}))),
        attribution=AttributionSnapshot(**dict(payload.get("attribution", {}))),
        error_tags=[str(tag) for tag in payload.get("error_tags", [])],
        lesson_draft=[str(item) for item in payload.get("lesson_draft", [])],
        memory_status=str(payload.get("memory_status", "raw")),
        metadata=dict(payload.get("metadata", {})),
        schema_version=schema_version,
    )


class TradeCaseStore:
    """原始 Trade Case 持久化目录。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.cases_dir = self.base_dir / "cases"

    def save_case(self, trade_case: TradeCase) -> Path:
        if trade_case.schema_version != TRADE_CASE_SCHEMA_VERSION:
            raise ValueError(
                f"TradeCase schema_version 必须为 {TRADE_CASE_SCHEMA_VERSION}"
            )
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        path = self._case_path(trade_case.case_id)
        self._write_json(path, _serialize_value(trade_case))
        return path

    def load_case(self, case_id: str) -> TradeCase:
        path = self._case_path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"未找到 TradeCase: {case_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _trade_case_from_payload(payload)

    def list_cases(
        self,
        symbol: str | None = None,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> list[TradeCase]:
        start = _parse_filter_date(start_date)
        end = _parse_filter_date(end_date)
        cases: list[TradeCase] = []
        for path in sorted(self.cases_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case = _trade_case_from_payload(payload)
            if symbol is not None and case.symbol != symbol:
                continue
            decision_date = case.decision_time.date()
            if start is not None and decision_date < start:
                continue
            if end is not None and decision_date > end:
                continue
            cases.append(case)
        return sorted(cases, key=lambda item: (item.decision_time, item.case_id))

    def update_outcomes(self, case_id: str, outcome_snapshot: OutcomeSnapshot) -> TradeCase:
        trade_case = self.load_case(case_id)
        trade_case.outcomes = outcome_snapshot
        self.save_case(trade_case)
        return trade_case

    def append_lesson_draft(self, case_id: str, lesson: str) -> TradeCase:
        trade_case = self.load_case(case_id)
        normalized_lesson = lesson.strip()
        if normalized_lesson:
            trade_case.lesson_draft.append(normalized_lesson)
            self.save_case(trade_case)
        return trade_case

    def add_error_tags(self, case_id: str, tags: list[str]) -> TradeCase:
        trade_case = self.load_case(case_id)
        existing = list(trade_case.error_tags)
        for tag in tags:
            normalized = tag.strip()
            if normalized and normalized not in existing:
                existing.append(normalized)
        trade_case.error_tags = existing
        self.save_case(trade_case)
        return trade_case

    def _case_path(self, case_id: str) -> Path:
        encoded_id = quote(case_id, safe="")
        return self.cases_dir / f"{encoded_id}.json"

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
