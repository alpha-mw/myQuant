"""启发式 K线后端：基于动量与波动率的简单适配器。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from branch_contracts import BranchResult

from .base import KLineBackend


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class HeuristicBackend(KLineBackend):
    name = "heuristic"
    reliability = 0.62
    horizon_days = 5

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        for symbol, df in symbol_data.items():
            if df.empty:
                continue
            ret_20 = float(df["close"].pct_change(20).iloc[-1]) if len(df) > 20 else 0.0
            ret_5 = float(df["close"].pct_change(5).iloc[-1]) if len(df) > 5 else 0.0
            vol_20 = float(df["close"].pct_change().rolling(20).std().iloc[-1]) if len(df) > 20 else 0.02
            forecast = 0.6 * ret_20 + 0.4 * ret_5
            signal = _clamp(forecast / (vol_20 * 8 + 1e-8), -1.0, 1.0)
            regime = "上行" if signal > 0.2 else "下行" if signal < -0.2 else "震荡"
            symbol_scores[symbol] = signal
            predicted_returns[symbol] = _clamp(forecast, -0.3, 0.3)
            regimes[symbol] = regime

        # 补齐无数据的股票
        for symbol in stock_pool:
            symbol_scores.setdefault(symbol, 0.0)
            predicted_returns.setdefault(symbol, 0.0)
            regimes.setdefault(symbol, "震荡")

        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.45 + float(np.std(list(symbol_scores.values()) or [0.0])), 0.35, 0.75)
        return BranchResult(
            branch_name="kline",
            score=score,
            confidence=confidence,
            signals={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "model_mode": "heuristic",
            },
            risks=["K线分支当前为启发式适配，尚未接入预训练模型。"],
            explanation="K线分析（启发式）基于 OHLCV 时序趋势与波动特征生成收益预测和趋势判断。",
            symbol_scores=symbol_scores,
            metadata={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "branch_mode": "kline_heuristic",
                "reliability": self.reliability,
                "horizon_days": self.horizon_days,
            },
        )
