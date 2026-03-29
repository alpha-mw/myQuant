"""K-line 双模型综合评估接口。

当前先提供占位版 evaluator，后续可在不改动 K-line 主链的前提下
替换为真实的大模型评估器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.branch_contracts import BranchResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class KLineEvaluationInput:
    """双模型综合评估的输入。"""

    stock_pool: list[str]
    symbol_data: dict[str, pd.DataFrame]
    kronos_result: BranchResult
    chronos_result: BranchResult


@dataclass
class KLineEvaluationOutput:
    """双模型综合评估的标准输出。"""

    score: float
    confidence: float
    symbol_scores: dict[str, float]
    predicted_returns: dict[str, float]
    regimes: dict[str, str]
    explanation: str
    risks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KLineEvaluator(ABC):
    """K-line 综合评估器抽象接口。"""

    name = "base"
    llm_ready = False

    @abstractmethod
    def evaluate(self, evaluation_input: KLineEvaluationInput) -> KLineEvaluationOutput:
        """对 Kronos 与 Chronos 的输出做综合评估。"""


class MarketStateDetector:
    """检测市场状态，用于占位 evaluator 的规则融合。"""

    STATES = {
        0: "strong_uptrend",
        1: "weak_uptrend",
        2: "strong_downtrend",
        3: "weak_downtrend",
        4: "high_volatility_range",
        5: "low_volatility_range",
        6: "potential_top",
        7: "potential_bottom",
    }

    def detect(self, prices: pd.Series, window: int = 20) -> int:
        if len(prices) < window:
            return 5

        recent = prices.tail(window)
        returns = recent.pct_change().dropna()
        sma_short = recent.rolling(window=5).mean().iloc[-1]
        sma_long = recent.rolling(window=window).mean().iloc[-1]
        trend_strength = (sma_short - sma_long) / (sma_long + 1e-10)
        volatility = returns.std() * np.sqrt(252)

        recent_high = recent.max()
        recent_low = recent.min()
        price_range = recent_high - recent_low
        if price_range < 1e-10:
            return 5

        current = recent.iloc[-1]
        dist_to_high = (recent_high - current) / price_range
        dist_to_low = (current - recent_low) / price_range

        if trend_strength > 0.05 and volatility > 0.3:
            return 0
        if trend_strength > 0.02:
            return 1
        if trend_strength < -0.05 and volatility > 0.3:
            return 2
        if trend_strength < -0.02:
            return 3
        if volatility > 0.4 and dist_to_high < 0.3:
            return 6
        if volatility > 0.4 and dist_to_low < 0.3:
            return 7
        if volatility > 0.25:
            return 4
        return 5

    def get_state_name(self, state_id: int) -> str:
        return self.STATES.get(state_id, "unknown")


class PlaceholderLLMEvaluator(KLineEvaluator):
    """占位版 LLM evaluator。

    目前仍用可解释的规则融合双模型，但输出结构已经固定，后续真实大模型
    只需替换此类即可。
    """

    name = "placeholder_llm_reviewer"
    llm_ready = False

    _WEIGHT_TRENDING = (0.65, 0.35)
    _WEIGHT_RANGING = (0.40, 0.60)
    _WEIGHT_REVERSAL = (0.50, 0.50)

    def __init__(self) -> None:
        self._detector = MarketStateDetector()

    def _adaptive_weights(self, state_id: int) -> tuple[float, float]:
        if state_id in (0, 1, 2, 3):
            return self._WEIGHT_TRENDING
        if state_id in (4, 5):
            return self._WEIGHT_RANGING
        return self._WEIGHT_REVERSAL

    def evaluate(self, evaluation_input: KLineEvaluationInput) -> KLineEvaluationOutput:
        kronos_result = evaluation_input.kronos_result
        chronos_result = evaluation_input.chronos_result

        k_scores = kronos_result.symbol_scores or {}
        c_scores = chronos_result.symbol_scores or {}
        k_returns = kronos_result.signals.get("predicted_return", {})
        c_returns = chronos_result.signals.get("predicted_return", {})
        k_reliability = float(kronos_result.metadata.get("reliability", 0.5))
        c_reliability = float(chronos_result.metadata.get("reliability", 0.5))

        all_symbols = set(evaluation_input.stock_pool) | set(k_scores) | set(c_scores)
        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}
        market_states: dict[str, str] = {}
        weights_used: dict[str, dict[str, float]] = {}

        for symbol in all_symbols:
            ks = float(k_scores.get(symbol, 0.0))
            cs = float(c_scores.get(symbol, 0.0))
            kr = float(k_returns.get(symbol, 0.0))
            cr = float(c_returns.get(symbol, 0.0))

            if symbol in evaluation_input.symbol_data and not evaluation_input.symbol_data[symbol].empty:
                state_id = self._detector.detect(evaluation_input.symbol_data[symbol]["close"])
            else:
                state_id = 5

            base_k, base_c = self._adaptive_weights(state_id)
            weighted_k = base_k * max(k_reliability, 0.05)
            weighted_c = base_c * max(c_reliability, 0.05)
            total_weight = weighted_k + weighted_c
            w_k = weighted_k / total_weight
            w_c = weighted_c / total_weight

            fused_score = w_k * ks + w_c * cs
            fused_return = w_k * kr + w_c * cr
            regime = "上行" if fused_score > 0.2 else "下行" if fused_score < -0.2 else "震荡"

            symbol_scores[symbol] = _clamp(fused_score, -1.0, 1.0)
            predicted_returns[symbol] = _clamp(fused_return, -0.3, 0.3)
            regimes[symbol] = regime
            market_states[symbol] = self._detector.get_state_name(state_id)
            weights_used[symbol] = {
                "kronos": round(w_k, 4),
                "chronos": round(w_c, 4),
            }

        agreement_ratio = sum(
            1 for symbol in all_symbols if float(k_scores.get(symbol, 0.0)) * float(c_scores.get(symbol, 0.0)) >= 0
        ) / max(len(all_symbols), 1)
        base_confidence = 0.5 * kronos_result.confidence + 0.5 * chronos_result.confidence
        confidence = _clamp(base_confidence * (0.85 + 0.30 * agreement_ratio), 0.42, 0.92)
        combined_reliability = _clamp(
            (0.5 * k_reliability + 0.5 * c_reliability) * (0.85 + 0.20 * agreement_ratio),
            0.35,
            0.92,
        )

        risks = list(dict.fromkeys([*kronos_result.risks, *chronos_result.risks]))
        explanation = (
            "K线分析当前固定同时运行 Kronos 与 Chronos，"
            "再通过预留的大模型评估接口输出综合结果；当前评估器为占位规则版。"
        )

        return KLineEvaluationOutput(
            score=_safe_mean(list(symbol_scores.values())),
            confidence=confidence,
            symbol_scores=symbol_scores,
            predicted_returns=predicted_returns,
            regimes=regimes,
            explanation=explanation,
            risks=risks,
            metadata={
                "branch_mode": "kline_dual_model",
                "reliability": combined_reliability,
                "horizon_days": 5,
                "evaluator_name": self.name,
                "llm_interface_reserved": True,
                "llm_ready": self.llm_ready,
                "agreement_ratio": round(agreement_ratio, 3),
                "market_states": market_states,
                "weights_used": weights_used,
                "model_components": {
                    "kronos": {
                        "score": kronos_result.score,
                        "confidence": kronos_result.confidence,
                        "runtime_mode": kronos_result.metadata.get("model_runtime_mode", "unknown"),
                        "reliability": k_reliability,
                    },
                    "chronos": {
                        "score": chronos_result.score,
                        "confidence": chronos_result.confidence,
                        "runtime_mode": chronos_result.metadata.get("model_runtime_mode", "unknown"),
                        "reliability": c_reliability,
                    },
                },
            },
        )


def get_kline_evaluator(name: str | None = None) -> KLineEvaluator:
    """返回 K-line 综合评估器实例。"""
    normalized = (name or "placeholder").strip().lower()
    if normalized in {"placeholder", "placeholder_llm", "llm_placeholder"}:
        return PlaceholderLLMEvaluator()
    raise ValueError(f"未知 K-line evaluator: {name!r}")
