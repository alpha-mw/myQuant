#!/usr/bin/env python3
"""
统一分支契约定义

为“五路并行研究 -> 风控 -> 集成裁判”架构提供标准化中间接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class UnifiedDataBundle:
    """数据层统一输出的数据包。"""

    market: str
    symbols: list[str]
    symbol_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: dict[str, dict[str, Any]] = field(default_factory=dict)
    event_data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sentiment_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    macro_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

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
class BranchResult:
    """单个研究分支的标准输出。"""

    branch_name: str
    score: float = 0.0
    confidence: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    explanation: str = ""
    symbol_scores: dict[str, float] = field(default_factory=dict)
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibratedBranchSignal:
    """单个研究分支经校准后的标准信号。"""

    branch_name: str
    branch_mode: str = "unknown"
    horizon_days: int = 5
    reliability: float = 0.0
    aggregate_expected_return: float = 0.0
    symbol_expected_returns: dict[str, float] = field(default_factory=dict)
    symbol_confidences: dict[str, float] = field(default_factory=dict)
    symbol_convictions: dict[str, float] = field(default_factory=dict)
    data_source_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioStrategy:
    """Ensemble Layer 输出的组合级策略。"""

    target_exposure: float = 0.0
    style_bias: str = "均衡"
    sector_preferences: list[str] = field(default_factory=list)
    candidate_symbols: list[str] = field(default_factory=list)
    position_limits: dict[str, float] = field(default_factory=dict)
    stop_loss_policy: dict[str, Any] = field(default_factory=dict)
    execution_notes: list[str] = field(default_factory=list)
    branch_consensus: dict[str, float] = field(default_factory=dict)
    symbol_convictions: dict[str, float] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    research_mode: str = "production"
    trade_recommendations: list["TradeRecommendation"] = field(default_factory=list)


@dataclass
class TradeRecommendation:
    """单票可执行交易建议。"""

    symbol: str
    action: str = "watch"
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
    branch_positive_count: int = 0
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_expected_returns: dict[str, float] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    position_management: list[str] = field(default_factory=list)
    horizon_days: int = 5
    trend_regime: str = "震荡"
    data_source_status: str = "unknown"


@dataclass
class ResearchPipelineResult:
    """并行研究主流程结果。"""

    data_bundle: UnifiedDataBundle
    branch_results: dict[str, BranchResult] = field(default_factory=dict)
    risk_result: Optional[Any] = None
    final_strategy: PortfolioStrategy = field(default_factory=PortfolioStrategy)
    final_report: str = ""
    execution_log: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    calibrated_signals: dict[str, CalibratedBranchSignal] = field(default_factory=dict)
