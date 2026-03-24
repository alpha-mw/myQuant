#!/usr/bin/env python3
"""
统一 Agent 协议层。

本模块只定义新的协议类型与最小校验规则，不改写旧 pipeline 的业务逻辑。
后续新架构可逐步收敛到这里，旧的 `contracts.py` / `branch_contracts.py`
 继续保持兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, TypeVar

from quant_investor.versioning import (
    ARCHITECTURE_VERSION_CURRENT,
    BRANCH_SCHEMA_VERSION_V9,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


class Direction(str, Enum):
    """方向标签。"""

    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class ActionLabel(str, Enum):
    """动作标签。"""

    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    WATCH = "watch"
    AVOID = "avoid"


class ConfidenceLabel(str, Enum):
    """离散化置信度标签。"""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AgentStatus(str, Enum):
    """Agent 执行状态。"""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"
    VETOED = "vetoed"


class RiskLevel(str, Enum):
    """风险等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class CoverageScope(str, Enum):
    """覆盖范围标签。"""

    GLOBAL = "global"
    MARKET = "market"
    PORTFOLIO = "portfolio"
    BRANCH = "branch"
    SYMBOL = "symbol"
    MODULE = "module"


_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_enum(value: _EnumT | str, enum_cls: type[_EnumT], field_name: str) -> _EnumT:
    """兼容字符串入参并收敛为枚举。"""

    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 {enum_cls.__name__} 或字符串")

    raw = value.strip()
    if not raw:
        raise ValueError(f"{field_name} 不得为空")

    try:
        return enum_cls(raw.lower())
    except ValueError:
        try:
            return enum_cls[raw.upper()]
        except KeyError as exc:
            raise ValueError(f"{field_name} 不支持取值: {value}") from exc


def _validate_score(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")
    if value < -1.0 or value > 1.0:
        raise ValueError(f"{field_name} 必须落在 [-1, 1] 区间内")
    return value


def _validate_confidence(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} 必须落在 [0, 1] 区间内")
    return value


def _validate_non_negative(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")
    if value < 0.0:
        raise ValueError(f"{field_name} 不得小于 0")
    return value


def _require_non_empty_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} 不得为空")
    return text


def _dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


@dataclass
class EvidenceItem:
    """标准化证据条目。"""

    source: str
    summary: str
    direction: Direction = Direction.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    scope: CoverageScope = CoverageScope.SYMBOL
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.direction = _coerce_enum(self.direction, Direction, "direction")
        self.scope = _coerce_enum(self.scope, CoverageScope, "scope")
        self.score = _validate_score(self.score, "score")
        self.confidence = _validate_confidence(self.confidence, "confidence")


@dataclass
class EventNote:
    """结构化事件说明。"""

    title: str
    message: str
    scope: CoverageScope = CoverageScope.GLOBAL
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scope = _coerce_enum(self.scope, CoverageScope, "scope")
        self.risk_level = _coerce_enum(self.risk_level, RiskLevel, "risk_level")


@dataclass
class BranchVerdict:
    """Research Agent 的统一分支结论。"""

    agent_name: str
    thesis: str
    symbol: str | None = None
    architecture_version: str = ARCHITECTURE_VERSION_CURRENT
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V9
    status: AgentStatus = AgentStatus.SUCCESS
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)
    events: list[EventNote] = field(default_factory=list)
    investment_risks: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    diagnostic_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thesis = _require_non_empty_text(self.thesis, "thesis")
        self.status = _coerce_enum(self.status, AgentStatus, "status")
        self.direction = _coerce_enum(self.direction, Direction, "direction")
        self.action = _coerce_enum(self.action, ActionLabel, "action")
        self.confidence_label = _coerce_enum(
            self.confidence_label, ConfidenceLabel, "confidence_label"
        )
        self.final_score = _validate_score(self.final_score, "final_score")
        self.final_confidence = _validate_confidence(
            self.final_confidence, "final_confidence"
        )


@dataclass
class RiskDecision:
    """RiskGuard 的统一输出。"""

    agent_name: str = "RiskGuard"
    architecture_version: str = ARCHITECTURE_VERSION_CURRENT
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V9
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    status: AgentStatus = AgentStatus.SUCCESS
    risk_level: RiskLevel = RiskLevel.MEDIUM
    hard_veto: bool = False
    veto: bool = False
    action_cap: ActionLabel = ActionLabel.BUY
    max_weight: float = 1.0
    gross_exposure_cap: float = 1.0
    target_exposure_cap: float = 1.0
    blocked_symbols: list[str] = field(default_factory=list)
    position_limits: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    events: list[EventNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_enum(self.status, AgentStatus, "status")
        self.risk_level = _coerce_enum(self.risk_level, RiskLevel, "risk_level")
        self.action_cap = _coerce_enum(self.action_cap, ActionLabel, "action_cap")
        self.max_weight = _validate_confidence(self.max_weight, "max_weight")
        exposure_cap = min(
            _validate_confidence(self.gross_exposure_cap, "gross_exposure_cap"),
            _validate_confidence(self.target_exposure_cap, "target_exposure_cap"),
        )
        self.gross_exposure_cap = exposure_cap
        self.target_exposure_cap = exposure_cap
        combined_veto = bool(self.hard_veto or self.veto)
        self.hard_veto = combined_veto
        self.veto = combined_veto
        self.position_limits = {
            str(symbol): _validate_confidence(limit, f"position_limits[{symbol}]")
            for symbol, limit in self.position_limits.items()
        }


@dataclass
class ICDecision:
    """ICCoordinator 的统一输出。"""

    agent_name: str = "ICCoordinator"
    architecture_version: str = ARCHITECTURE_VERSION_CURRENT
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V9
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    status: AgentStatus = AgentStatus.SUCCESS
    thesis: str = ""
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    selected_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    rationale_points: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_enum(self.status, AgentStatus, "status")
        self.direction = _coerce_enum(self.direction, Direction, "direction")
        self.action = _coerce_enum(self.action, ActionLabel, "action")
        self.confidence_label = _coerce_enum(
            self.confidence_label, ConfidenceLabel, "confidence_label"
        )
        self.final_score = _validate_score(self.final_score, "final_score")
        self.final_confidence = _validate_confidence(
            self.final_confidence, "final_confidence"
        )

    @property
    def action_suggestion(self) -> ActionLabel:
        return self.action

    @property
    def ic_thesis(self) -> str:
        return self.thesis


@dataclass
class PortfolioPlan:
    """PortfolioConstructor 的统一输出。"""

    agent_name: str = "PortfolioConstructor"
    architecture_version: str = ARCHITECTURE_VERSION_CURRENT
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V9
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    status: AgentStatus = AgentStatus.SUCCESS
    target_exposure: float = 0.0
    target_gross_exposure: float = 0.0
    target_net_exposure: float = 0.0
    cash_ratio: float = 1.0
    target_weights: dict[str, float] = field(default_factory=dict)
    target_positions: dict[str, float] = field(default_factory=dict)
    position_limits: dict[str, float] = field(default_factory=dict)
    blocked_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    concentration_metrics: dict[str, float] = field(default_factory=dict)
    turnover_estimate: float = 0.0
    execution_notes: list[str] = field(default_factory=list)
    construction_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_enum(self.status, AgentStatus, "status")
        normalized_positions = self.target_positions or self.target_weights
        normalized_positions = {
            str(symbol): _validate_confidence(weight, f"target_positions[{symbol}]")
            for symbol, weight in normalized_positions.items()
        }
        self.target_positions = normalized_positions
        self.target_weights = dict(normalized_positions)

        self.target_exposure = _validate_confidence(self.target_exposure, "target_exposure")
        self.target_gross_exposure = _validate_confidence(
            self.target_gross_exposure, "target_gross_exposure"
        )
        self.target_net_exposure = _validate_score(
            self.target_net_exposure, "target_net_exposure"
        )
        if self.target_gross_exposure <= 0.0 and self.target_exposure > 0.0:
            self.target_gross_exposure = self.target_exposure
        elif self.target_exposure <= 0.0 and self.target_gross_exposure > 0.0:
            self.target_exposure = self.target_gross_exposure
        elif self.target_exposure <= 0.0 and self.target_gross_exposure <= 0.0 and normalized_positions:
            inferred_gross = _validate_confidence(sum(normalized_positions.values()), "target_gross_exposure")
            self.target_gross_exposure = inferred_gross
            self.target_exposure = inferred_gross

        if self.target_net_exposure == 0.0 and self.target_gross_exposure > 0.0:
            self.target_net_exposure = self.target_gross_exposure

        self.cash_ratio = _validate_confidence(self.cash_ratio, "cash_ratio")
        if self.cash_ratio >= 1.0 and self.target_gross_exposure > 0.0:
            self.cash_ratio = _validate_confidence(
                max(0.0, 1.0 - self.target_gross_exposure),
                "cash_ratio",
            )

        total_target_weight = sum(self.target_positions.values())
        if total_target_weight > self.target_gross_exposure + 1e-8:
            raise ValueError("target_positions 权重和不得超过 target_gross_exposure")

        self.target_weights = {
            str(symbol): _validate_confidence(weight, f"target_weights[{symbol}]")
            for symbol, weight in self.target_weights.items()
        }
        self.position_limits = {
            str(symbol): _validate_confidence(limit, f"position_limits[{symbol}]")
            for symbol, limit in self.position_limits.items()
        }
        self.blocked_symbols = _dedupe_texts(self.blocked_symbols)
        self.rejected_symbols = _dedupe_texts(self.rejected_symbols or self.blocked_symbols)
        if not self.blocked_symbols and self.rejected_symbols:
            self.blocked_symbols = list(self.rejected_symbols)

        self.concentration_metrics = {
            str(name): _validate_non_negative(value, f"concentration_metrics[{name}]")
            for name, value in self.concentration_metrics.items()
        }
        self.turnover_estimate = _validate_non_negative(
            self.turnover_estimate, "turnover_estimate"
        )
        merged_notes = _dedupe_texts(self.construction_notes or self.execution_notes)
        if not merged_notes and self.execution_notes:
            merged_notes = _dedupe_texts(self.execution_notes)
        self.construction_notes = merged_notes
        self.execution_notes = list(merged_notes)


@dataclass
class ReportBundle:
    """NarratorAgent 的统一输入/输出包。"""

    agent_name: str = "NarratorAgent"
    architecture_version: str = ARCHITECTURE_VERSION_CURRENT
    branch_schema_version: str = BRANCH_SCHEMA_VERSION_V9
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION
    headline: str = ""
    summary: str = ""
    macro_verdict: BranchVerdict | None = None
    branch_verdicts: dict[str, BranchVerdict] = field(default_factory=dict)
    risk_decision: RiskDecision | None = None
    ic_decision: ICDecision | None = None
    ic_decisions: list[ICDecision] = field(default_factory=list)
    portfolio_plan: PortfolioPlan | None = None
    markdown_report: str = ""
    executive_summary: list[str] = field(default_factory=list)
    market_view: str = ""
    branch_conclusions: dict[str, str] = field(default_factory=dict)
    stock_cards: list[dict[str, Any]] = field(default_factory=list)
    coverage_summary: list[str] = field(default_factory=list)
    appendix_diagnostics: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[EventNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ic_decision is None and self.ic_decisions:
            self.ic_decision = self.ic_decisions[0]
        if not self.executive_summary and self.highlights:
            self.executive_summary = list(self.highlights[:3])
        if not self.highlights and self.executive_summary:
            self.highlights = list(self.executive_summary)
        if not self.summary and self.executive_summary:
            self.summary = " ".join(self.executive_summary)


__all__ = [
    "ActionLabel",
    "AgentStatus",
    "BranchVerdict",
    "ConfidenceLabel",
    "CoverageScope",
    "Direction",
    "EvidenceItem",
    "EventNote",
    "ICDecision",
    "PortfolioPlan",
    "ReportBundle",
    "RiskDecision",
    "RiskLevel",
]
