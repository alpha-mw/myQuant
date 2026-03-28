"""
当前公开 agent protocol。

主线不再依赖历史 sourceless shadow wrapper。
本文件直接声明当前稳定协议类型，供 Agent bridge、报告层和测试复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dc_fields
from enum import Enum
from typing import Any, Optional

from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


class AgentStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    VETOED = "vetoed"


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ActionLabel(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    WATCH = "watch"
    AVOID = "avoid"


class ConfidenceLabel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class CoverageScope(str, Enum):
    BRANCH = "branch"
    SYMBOL = "symbol"
    MARKET = "market"
    PORTFOLIO = "portfolio"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


def _make_compat_init(cls: type) -> None:
    orig_init = cls.__init__
    known_fields = {item.name for item in dc_fields(cls)}

    def _compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        extras: dict[str, Any] = {}
        for key in list(kwargs):
            if key not in known_fields:
                extras[key] = kwargs.pop(key)
        orig_init(self, *args, **kwargs)
        for key, value in extras.items():
            if isinstance(getattr(type(self), key, None), property):
                prop = getattr(type(self), key)
                if prop.fset is not None:
                    prop.fset(self, value)
                elif hasattr(self, "metadata"):
                    self.metadata[key] = value
            else:
                object.__setattr__(self, key, value)

    cls.__init__ = _compat_init  # type: ignore[assignment]


@dataclass
class EventNote:
    title: str = ""
    message: str = ""
    scope: CoverageScope = CoverageScope.BRANCH
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    source: str = ""
    summary: str = ""
    direction: Direction = Direction.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    scope: CoverageScope = CoverageScope.BRANCH
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchVerdict:
    agent_name: str = ""
    thesis: str = ""
    symbol: Optional[str] = None
    status: AgentStatus = AgentStatus.SUCCESS
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)
    investment_risks: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    diagnostic_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskDecision:
    status: AgentStatus = AgentStatus.SUCCESS
    risk_level: RiskLevel = RiskLevel.LOW
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
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ICDecision:
    status: AgentStatus = AgentStatus.SUCCESS
    thesis: str = ""
    symbol: Optional[str] = None
    direction: Direction = Direction.NEUTRAL
    action: ActionLabel = ActionLabel.HOLD
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    final_score: float = 0.0
    final_confidence: float = 0.0
    agreement_points: list[str] = field(default_factory=list)
    conflict_points: list[str] = field(default_factory=list)
    rationale_points: list[str] = field(default_factory=list)
    selected_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioPlan:
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
    concentration_metrics: dict[str, Any] = field(default_factory=dict)
    turnover_estimate: float = 0.0
    execution_notes: list[str] = field(default_factory=list)
    construction_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportBundle:
    agent_name: str = "NarratorAgent"
    headline: str = ""
    summary: str = ""
    macro_verdict: Optional[BranchVerdict] = None
    branch_verdicts: dict[str, BranchVerdict] = field(default_factory=dict)
    risk_decision: Optional[RiskDecision] = None
    ic_decision: Optional[ICDecision] = None
    ic_decisions: list[ICDecision] = field(default_factory=list)
    portfolio_plan: Optional[PortfolioPlan] = None
    markdown_report: str = ""
    executive_summary: list[str] = field(default_factory=list)
    market_view: list[str] = field(default_factory=list)
    branch_conclusions: list[str] = field(default_factory=list)
    stock_cards: list[str] = field(default_factory=list)
    coverage_summary: list[str] = field(default_factory=list)
    appendix_diagnostics: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[EventNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_make_compat_init(EventNote)
_make_compat_init(EvidenceItem)
_make_compat_init(BranchVerdict)
_make_compat_init(RiskDecision)
_make_compat_init(ICDecision)
_make_compat_init(PortfolioPlan)
_make_compat_init(ReportBundle)


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_SCHEMA_VERSION",
    "IC_PROTOCOL_VERSION",
    "REPORT_PROTOCOL_VERSION",
    "ActionLabel",
    "AgentStatus",
    "ConfidenceLabel",
    "CoverageScope",
    "Direction",
    "EventNote",
    "EvidenceItem",
    "RiskLevel",
    "BranchVerdict",
    "RiskDecision",
    "ICDecision",
    "PortfolioPlan",
    "ReportBundle",
]
