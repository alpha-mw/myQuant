#!/usr/bin/env python3
"""
V9 分支契约定义。

当前主线把原先独立的 llm_debate 顶层分支下沉为各分支内部可复用的
branch-local debate 能力，因此分支结果需要同时保留：

1. 原始 deterministic/base 结果
2. branch-local debate 后的最终结果

本文件作为新的契约定义源；`quant_investor.contracts` 仅保留兼容导出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    DEBATE_TEMPLATE_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
    reject_retired_intelligence_keys,
)


_ALLOWED_BRANCH_RESULT_NAMES = frozenset((*CANONICAL_BRANCH_ORDER, "kline"))


def _require_branch_name(branch_name: str, field_name: str) -> str:
    resolved = str(branch_name).strip()
    if resolved not in _ALLOWED_BRANCH_RESULT_NAMES:
        raise ValueError(f"{field_name} is not a v15 branch: {branch_name!r}.")
    return resolved


def _require_version(actual: str, expected: str, field_name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{field_name} mismatch: expected {expected!r}, got {actual!r}."
        )


@dataclass
class UnifiedDataBundle:
    """数据层统一输出的数据包。"""

    market: str = ""
    symbols: list[str] = field(default_factory=list)
    symbol_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: dict[str, dict[str, Any]] = field(default_factory=dict)
    macro_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stock_data: dict[str, pd.DataFrame] = field(init=False, repr=False)
    stock_pool: list[str] = field(init=False, repr=False)

    def __init__(
        self,
        market: str = "",
        symbols: list[str] | None = None,
        symbol_data: dict[str, pd.DataFrame] | None = None,
        fundamentals: dict[str, dict[str, Any]] | None = None,
        macro_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stock_data: dict[str, pd.DataFrame] | None = None,
        stock_pool: list[str] | None = None,
    ) -> None:
        payload = symbol_data if symbol_data is not None else stock_data
        self.market = market
        self.symbol_data = dict(payload or {})
        self.symbols = list(symbols or stock_pool or self.symbol_data.keys())
        self.fundamentals = dict(fundamentals or {})
        self.macro_data = dict(macro_data or {})
        self.metadata = dict(metadata or {})
        self.stock_data = self.symbol_data
        self.stock_pool = self.symbols

    def combined_frame(self) -> pd.DataFrame:
        """将全部标的的时序数据合并为单张表。"""
        if not self.symbol_data:
            return pd.DataFrame()
        frames = []
        for symbol, df in self.symbol_data.items():
            if df is None or df.empty:
                continue
            frame = df.copy()
            if "symbol" not in frame.columns:
                frame["symbol"] = symbol
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

    def latest_prices(self) -> dict[str, float]:
        """返回各标的最新收盘价。"""
        prices: dict[str, float] = {}
        for symbol, df in self.symbol_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            prices[symbol] = float(df["close"].iloc[-1])
        return prices

    def symbol_provenance(self) -> dict[str, dict[str, Any]]:
        """返回按股票组织的数据来源元信息。"""
        provenance = self.metadata.get("symbol_provenance", {})
        return provenance if isinstance(provenance, dict) else {}

    def synthetic_symbols(self) -> list[str]:
        """返回使用模拟或降级数据的股票列表。"""
        result = []
        for symbol, meta in self.symbol_provenance().items():
            if meta.get("is_synthetic"):
                result.append(symbol)
        return result

    def degraded_symbols(self) -> list[str]:
        """返回非纯真实来源的股票列表。"""
        result = []
        for symbol, meta in self.symbol_provenance().items():
            if meta.get("data_source_status") != "real":
                result.append(symbol)
        return result

    def real_symbols(self) -> list[str]:
        """返回使用真实数据的股票列表。"""
        result = []
        for symbol in self.symbols:
            meta = self.symbol_provenance().get(symbol, {})
            if not meta.get("is_synthetic", False):
                result.append(symbol)
        return result


@dataclass
class EvidencePacket:
    """分支 base result 提炼后的证据包。"""

    branch_name: str
    as_of: str = ""
    scope: str = "symbol"
    summary: str = ""
    symbols: list[str] = field(default_factory=list)
    top_symbols: list[str] = field(default_factory=list)
    bull_points: list[str] = field(default_factory=list)
    bear_points: list[str] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    used_features: list[str] = field(default_factory=list)
    feature_values: dict[str, Any] = field(default_factory=dict)
    symbol_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_name = _require_branch_name(
            self.branch_name,
            "EvidencePacket.branch_name",
        )
        reject_retired_intelligence_keys(
            {
                "feature_values": self.feature_values,
                "symbol_context": self.symbol_context,
                "metadata": self.metadata,
            },
            path="EvidencePacket",
        )


@dataclass
class DebateVerdict:
    """branch-local debate 的统一输出。"""

    direction: str = "neutral"
    confidence: float = 0.0
    score_adjustment: float = 0.0
    bull_points: list[str] = field(default_factory=list)
    bear_points: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    used_features: list[str] = field(default_factory=list)
    hard_veto: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "score_adjustment": self.score_adjustment,
            "bull_points": list(self.bull_points),
            "bear_points": list(self.bear_points),
            "risk_flags": list(self.risk_flags),
            "unknowns": list(self.unknowns),
            "used_features": list(self.used_features),
            "hard_veto": self.hard_veto,
            "metadata": dict(self.metadata),
        }


@dataclass
class FundamentalSnapshot:
    """点时公司财务/估值快照。"""

    symbol: str
    as_of: str = ""
    available: bool = False
    source: str = "neutral"
    publish_time: str = ""
    effective_time: str = ""
    ingest_time: str = ""
    revision_id: str = ""
    is_estimated: bool = False
    data_quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    roe: float = 0.0
    roa: float = 0.0
    gross_margin: float = 0.0
    net_margin: float = 0.0
    revenue_growth: float = 0.0
    profit_growth: float = 0.0
    debt_ratio: float = 0.0
    current_ratio: float = 0.0
    cash_flow: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    ps: float = 0.0
    dividend_yield: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class ForecastSnapshot:
    """盈利预测与修正快照。"""

    symbol: str
    as_of: str = ""
    available: bool = False
    source: str = "neutral"
    publish_time: str = ""
    effective_time: str = ""
    ingest_time: str = ""
    revision_id: str = ""
    is_estimated: bool = False
    data_quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    provider: str = "none"
    horizon_days: int = 0
    expected_return: float = 0.0
    eps_growth: float = 0.0
    revenue_growth_forecast: float = 0.0
    forecast_revision: float = 0.0
    coverage_count: int = 0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ManagementSnapshot:
    """管理层与治理快照。"""

    symbol: str
    as_of: str = ""
    available: bool = False
    source: str = "neutral"
    publish_time: str = ""
    effective_time: str = ""
    ingest_time: str = ""
    revision_id: str = ""
    is_estimated: bool = False
    data_quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    management_stability: float = 0.0
    governance_score: float = 0.0
    management_alignment: float = 0.0
    key_executive_changes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class OwnershipSnapshot:
    """股东结构快照。"""

    symbol: str
    as_of: str = ""
    available: bool = False
    source: str = "neutral"
    publish_time: str = ""
    effective_time: str = ""
    ingest_time: str = ""
    revision_id: str = ""
    is_estimated: bool = False
    data_quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    concentration_score: float = 0.0
    top_holder_pct: float = 0.0
    institutional_holding_pct: float = 0.0
    insider_holding_pct: float = 0.0
    ownership_change_signal: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class CorporateDocumentSnapshot:
    """离线文档语义快照。"""

    symbol: str
    as_of: str = ""
    available: bool = False
    source: str = "offline_snapshot"
    publish_time: str = ""
    effective_time: str = ""
    ingest_time: str = ""
    revision_id: str = ""
    is_estimated: bool = False
    data_quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    latest_document_type: str = ""
    semantic_sentiment: float = 0.0
    execution_confidence: float = 0.0
    governance_red_flag: float = 0.0
    key_phrases: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosticEvent:
    """统一诊断事件结构。"""

    code: str
    message: str
    category: str = "diagnostic"
    scope: str = "global"
    unit: str = "occurrences"
    severity: str = "info"


@dataclass(init=False)
class BranchResult:
    """单个研究分支的标准输出。"""

    branch_name: str
    signals: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    explanation: str = ""
    symbol_scores: dict[str, float] = field(default_factory=dict)
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    base_score: Optional[float] = None
    final_score: Optional[float] = None
    base_confidence: Optional[float] = None
    final_confidence: Optional[float] = None
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    horizon_days: int = 5
    evidence: EvidencePacket = field(default_factory=lambda: EvidencePacket(branch_name="quant"))
    debate_verdict: DebateVerdict = field(default_factory=DebateVerdict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    thesis_points: list[str] = field(default_factory=list)
    investment_risks: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    diagnostic_notes: list[str] = field(default_factory=list)
    support_drivers: list[str] = field(default_factory=list)
    drag_drivers: list[str] = field(default_factory=list)
    weight_cap_reasons: list[str] = field(default_factory=list)
    module_coverage: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        branch_name: str,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        signals: Optional[dict[str, Any]] = None,
        risks: Optional[list[str]] = None,
        explanation: str = "",
        symbol_scores: Optional[dict[str, float]] = None,
        success: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        base_score: Optional[float] = None,
        final_score: Optional[float] = None,
        base_confidence: Optional[float] = None,
        final_confidence: Optional[float] = None,
        architecture_version: str = ARCHITECTURE_VERSION,
        branch_schema_version: str = BRANCH_SCHEMA_VERSION,
        calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION,
        debate_template_version: str = DEBATE_TEMPLATE_VERSION,
        horizon_days: int = 5,
        evidence: Optional[EvidencePacket] = None,
        debate_verdict: Optional[DebateVerdict] = None,
        data_quality: Optional[dict[str, Any]] = None,
        conclusion: str = "",
        thesis_points: Optional[list[str]] = None,
        investment_risks: Optional[list[str]] = None,
        coverage_notes: Optional[list[str]] = None,
        diagnostic_notes: Optional[list[str]] = None,
        support_drivers: Optional[list[str]] = None,
        drag_drivers: Optional[list[str]] = None,
        weight_cap_reasons: Optional[list[str]] = None,
        module_coverage: Optional[dict[str, Any]] = None,
    ) -> None:
        self.branch_name = _require_branch_name(branch_name, "BranchResult.branch_name")
        _require_version(
            architecture_version,
            ARCHITECTURE_VERSION,
            "BranchResult.architecture_version",
        )
        _require_version(
            branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "BranchResult.branch_schema_version",
        )
        _require_version(
            calibration_schema_version,
            CALIBRATION_SCHEMA_VERSION,
            "BranchResult.calibration_schema_version",
        )
        _require_version(
            debate_template_version,
            DEBATE_TEMPLATE_VERSION,
            "BranchResult.debate_template_version",
        )
        self.signals = dict(signals or {})
        self.risks = list(risks or [])
        self.explanation = explanation
        self.symbol_scores = dict(symbol_scores or {})
        self.success = bool(success)
        self.metadata = dict(metadata or {})
        self.base_score = float(base_score if base_score is not None else (score or 0.0))
        self.final_score = float(final_score if final_score is not None else self.base_score)
        self.base_confidence = float(
            base_confidence if base_confidence is not None else (confidence or 0.0)
        )
        self.final_confidence = float(
            final_confidence if final_confidence is not None else self.base_confidence
        )
        self.architecture_version = architecture_version
        self.branch_schema_version = branch_schema_version
        self.calibration_schema_version = calibration_schema_version
        self.debate_template_version = debate_template_version
        self.horizon_days = int(horizon_days or self.metadata.get("horizon_days", 5) or 5)
        self.evidence = evidence if evidence is not None else EvidencePacket(branch_name=branch_name)
        self.debate_verdict = debate_verdict if debate_verdict is not None else DebateVerdict()
        self.data_quality = dict(data_quality or {})
        self.conclusion = str(conclusion or "")
        self.thesis_points = [str(item) for item in (thesis_points or []) if str(item)]
        self.investment_risks = [str(item) for item in (investment_risks or []) if str(item)]
        self.coverage_notes = [str(item) for item in (coverage_notes or []) if str(item)]
        self.diagnostic_notes = [str(item) for item in (diagnostic_notes or []) if str(item)]
        self.support_drivers = [str(item) for item in (support_drivers or []) if str(item)]
        self.drag_drivers = [str(item) for item in (drag_drivers or []) if str(item)]
        self.weight_cap_reasons = [str(item) for item in (weight_cap_reasons or []) if str(item)]
        self.module_coverage = dict(module_coverage or {})
        self.horizon_days = int(self.horizon_days or self.metadata.get("horizon_days", 5) or 5)
        self.metadata.setdefault("horizon_days", self.horizon_days)

        if self.evidence.branch_name != self.branch_name:
            raise ValueError(
                "BranchResult evidence branch mismatch: "
                f"{self.evidence.branch_name!r} != {self.branch_name!r}."
            )

    def validate(self) -> None:
        self.branch_name = _require_branch_name(self.branch_name, "BranchResult.branch_name")
        for field_name, actual, expected in (
            ("architecture_version", self.architecture_version, ARCHITECTURE_VERSION),
            ("branch_schema_version", self.branch_schema_version, BRANCH_SCHEMA_VERSION),
            ("calibration_schema_version", self.calibration_schema_version, CALIBRATION_SCHEMA_VERSION),
            ("debate_template_version", self.debate_template_version, DEBATE_TEMPLATE_VERSION),
        ):
            _require_version(actual, expected, f"BranchResult.{field_name}")
        self.evidence.__post_init__()
        if self.evidence.branch_name != self.branch_name:
            raise ValueError("BranchResult evidence branch mismatch.")
        reject_retired_intelligence_keys(
            {
                "signals": self.signals,
                "metadata": self.metadata,
                "data_quality": self.data_quality,
                "module_coverage": self.module_coverage,
                "debate_metadata": self.debate_verdict.metadata,
            },
            path="BranchResult",
        )

    @property
    def score(self) -> float:
        return float(self.final_score if self.final_score is not None else 0.0)

    @score.setter
    def score(self, value: float) -> None:
        self.final_score = float(value)

    @property
    def confidence(self) -> float:
        return float(self.final_confidence if self.final_confidence is not None else 0.0)

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.final_confidence = float(value)


@dataclass
class CalibratedBranchSignal:
    """单个研究分支经校准后的标准信号。"""

    branch_name: str
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    branch_mode: str = "unknown"
    horizon_days: int = 5
    reliability: float = 0.0
    aggregate_expected_return: float = 0.0
    symbol_expected_returns: dict[str, float] = field(default_factory=dict)
    symbol_confidences: dict[str, float] = field(default_factory=dict)
    symbol_convictions: dict[str, float] = field(default_factory=dict)
    data_source_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_name = _require_branch_name(
            self.branch_name,
            "CalibratedBranchSignal.branch_name",
        )
        _require_version(
            self.branch_schema_version,
            BRANCH_SCHEMA_VERSION,
            "CalibratedBranchSignal.branch_schema_version",
        )
        _require_version(self.architecture_version, ARCHITECTURE_VERSION, "CalibratedBranchSignal.architecture_version")
        _require_version(self.calibration_schema_version, CALIBRATION_SCHEMA_VERSION, "CalibratedBranchSignal.calibration_schema_version")
        _require_version(self.debate_template_version, DEBATE_TEMPLATE_VERSION, "CalibratedBranchSignal.debate_template_version")
        reject_retired_intelligence_keys(
            self.metadata,
            path="CalibratedBranchSignal.metadata",
        )


@dataclass
class LLMUsageRecord:
    stage: str = ""
    branch_or_agent_name: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    success: bool = True
    fallback: bool = False
    estimated_cost_usd: float = 0.0
    timestamp_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsageSummary:
    call_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_stage: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return int(self.call_count)

    @total_calls.setter
    def total_calls(self, value: int) -> None:
        self.call_count = int(value)

    @property
    def total_prompt_tokens(self) -> int:
        return int(self.prompt_tokens)

    @total_prompt_tokens.setter
    def total_prompt_tokens(self, value: int) -> None:
        self.prompt_tokens = int(value)

    @property
    def total_completion_tokens(self) -> int:
        return int(self.completion_tokens)

    @total_completion_tokens.setter
    def total_completion_tokens(self, value: int) -> None:
        self.completion_tokens = int(value)


@dataclass
class PortfolioStrategy:
    """Ensemble Layer 输出的组合级策略。"""

    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    target_exposure: float = 0.0
    total_exposure: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    cash_ratio: float = 1.0
    style_bias: str = "均衡"
    sector_preferences: list[str] = field(default_factory=list)
    candidate_symbols: list[str] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    target_positions: dict[str, float] = field(default_factory=dict)
    position_limits: dict[str, float] = field(default_factory=dict)
    stop_loss_policy: dict[str, Any] = field(default_factory=dict)
    execution_notes: list[str] = field(default_factory=list)
    branch_consensus: dict[str, float] = field(default_factory=dict)
    symbol_convictions: dict[str, float] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    research_mode: str = "production"
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)
    recommendations: list["TradeRecommendation"] = field(default_factory=list)
    trade_recommendations: list["TradeRecommendation"] = field(default_factory=list)

    def __post_init__(self) -> None:
        for field_name, actual, expected in (
            ("architecture_version", self.architecture_version, ARCHITECTURE_VERSION),
            ("branch_schema_version", self.branch_schema_version, BRANCH_SCHEMA_VERSION),
            ("calibration_schema_version", self.calibration_schema_version, CALIBRATION_SCHEMA_VERSION),
            ("debate_template_version", self.debate_template_version, DEBATE_TEMPLATE_VERSION),
        ):
            _require_version(actual, expected, f"PortfolioStrategy.{field_name}")
        unexpected = sorted(set(self.branch_consensus) - set(CANONICAL_BRANCH_ORDER))
        if unexpected:
            raise ValueError(
                "PortfolioStrategy has noncanonical branch keys: "
                + ", ".join(unexpected)
            )
        for recommendation in (*self.recommendations, *self.trade_recommendations):
            recommendation.__post_init__()
        reject_retired_intelligence_keys(
            {
                "stop_loss_policy": self.stop_loss_policy,
                "risk_summary": self.risk_summary,
                "provenance_summary": self.provenance_summary,
                "metadata": self.metadata,
            },
            path="PortfolioStrategy",
        )


@dataclass
class TradeRecommendation:
    """单票可执行交易建议。"""

    symbol: str
    action: str = "watch"
    weight: float = 0.0
    category: str = ""
    current_price: float = 0.0
    recommended_entry_price: float = 0.0
    entry_price_range: dict[str, float] = field(default_factory=dict)
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    support_price: float = 0.0
    resistance_price: float = 0.0
    model_expected_return: float = 0.0
    expected_upside: float = 0.0
    expected_drawdown: float = 0.0
    risk_reward_ratio: float = 0.0
    suggested_weight: float = 0.0
    suggested_amount: float = 0.0
    suggested_shares: int = 0
    lot_size: int = 1
    confidence: float = 0.0
    consensus_score: float = 0.0
    rationale: str = ""
    one_line_conclusion: str = ""
    support_drivers: list[str] = field(default_factory=list)
    drag_drivers: list[str] = field(default_factory=list)
    weight_cap_reasons: list[str] = field(default_factory=list)
    branch_positive_count: int = 0
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_expected_returns: dict[str, float] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    position_management: list[str] = field(default_factory=list)
    horizon_days: int = 5
    trend_regime: str = "震荡"
    data_source_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, values in (
            ("branch_scores", self.branch_scores),
            ("branch_expected_returns", self.branch_expected_returns),
        ):
            unexpected = sorted(set(values) - set(CANONICAL_BRANCH_ORDER))
            if unexpected:
                raise ValueError(
                    f"TradeRecommendation.{field_name} has noncanonical branches: "
                    + ", ".join(unexpected)
                )
        reject_retired_intelligence_keys(
            {
                "entry_price_range": self.entry_price_range,
                "metadata": self.metadata,
            },
            path="TradeRecommendation",
        )


@dataclass
class ResearchPipelineResult:
    """并行研究主流程结果。"""

    data_bundle: UnifiedDataBundle
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION
    likelihood_schema_version: str = LIKELIHOOD_SCHEMA_VERSION
    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    ic_protocol_version: str = IC_PROTOCOL_VERSION
    report_protocol_version: str = REPORT_PROTOCOL_VERSION
    debate_template_version: str = DEBATE_TEMPLATE_VERSION
    branch_results: dict[str, BranchResult] = field(default_factory=dict)
    risk_result: Optional[Any] = None
    final_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    final_report: str = ""
    execution_log: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    calibrated_signals: dict[str, CalibratedBranchSignal] = field(default_factory=dict)
    pipeline_mode: str = "draft"

    def __post_init__(self) -> None:
        for field_name, actual, expected in (
            ("architecture_version", self.architecture_version, ARCHITECTURE_VERSION),
            ("branch_schema_version", self.branch_schema_version, BRANCH_SCHEMA_VERSION),
            ("likelihood_schema_version", self.likelihood_schema_version, LIKELIHOOD_SCHEMA_VERSION),
            ("calibration_schema_version", self.calibration_schema_version, CALIBRATION_SCHEMA_VERSION),
            ("ic_protocol_version", self.ic_protocol_version, IC_PROTOCOL_VERSION),
            ("report_protocol_version", self.report_protocol_version, REPORT_PROTOCOL_VERSION),
            ("debate_template_version", self.debate_template_version, DEBATE_TEMPLATE_VERSION),
        ):
            _require_version(actual, expected, f"ResearchPipelineResult.{field_name}")
        unexpected = sorted(set(self.branch_results) - set(CANONICAL_BRANCH_ORDER))
        if unexpected:
            raise ValueError(
                "ResearchPipelineResult has noncanonical branch keys: "
                + ", ".join(unexpected)
            )
        if self.pipeline_mode not in {"draft", "bayesian"}:
            raise ValueError(
                "ResearchPipelineResult pipeline_mode must be 'draft' or 'bayesian'."
            )
        if self.pipeline_mode == "bayesian" and set(self.branch_results) != set(
            CANONICAL_BRANCH_ORDER
        ):
            missing = sorted(
                set(CANONICAL_BRANCH_ORDER) - set(self.branch_results)
            )
            raise ValueError(
                "ResearchPipelineResult bayesian branch_results must contain "
                "exactly the canonical branches "
                f"{list(CANONICAL_BRANCH_ORDER)!r}; missing branches: "
                + ", ".join(missing)
            )
        unexpected_calibrated = sorted(
            set(self.calibrated_signals) - set(CANONICAL_BRANCH_ORDER)
        )
        if unexpected_calibrated:
            raise ValueError(
                "ResearchPipelineResult has noncanonical calibrated signal keys: "
                + ", ".join(unexpected_calibrated)
            )
        for branch_name, result in self.branch_results.items():
            if not isinstance(result, BranchResult):
                raise ValueError("ResearchPipelineResult branch results must be BranchResult objects.")
            result.validate()
            if result.branch_name != branch_name:
                raise ValueError("ResearchPipelineResult branch result key/name mismatch.")
        for branch_name, signal in self.calibrated_signals.items():
            if not isinstance(signal, CalibratedBranchSignal):
                raise ValueError(
                    "ResearchPipelineResult calibrated signals must be CalibratedBranchSignal objects."
                )
            signal.__post_init__()
            if signal.branch_name != branch_name:
                raise ValueError("ResearchPipelineResult calibrated signal key/name mismatch.")
        self.final_strategy.__post_init__()
        reject_retired_intelligence_keys(
            {
                "data_bundle_metadata": self.data_bundle.metadata,
                "timings": self.timings,
            },
            path="ResearchPipelineResult",
        )


__all__ = [
    "BranchResult",
    "CalibratedBranchSignal",
    "CorporateDocumentSnapshot",
    "DiagnosticEvent",
    "DebateVerdict",
    "EvidencePacket",
    "ForecastSnapshot",
    "FundamentalSnapshot",
    "LLMUsageRecord",
    "LLMUsageSummary",
    "ManagementSnapshot",
    "OwnershipSnapshot",
    "PortfolioStrategy",
    "ResearchPipelineResult",
    "TradeRecommendation",
    "UnifiedDataBundle",
]
