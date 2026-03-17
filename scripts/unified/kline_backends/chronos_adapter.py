"""Chronos-2 时序模型后端适配器。

包装 Amazon Chronos-2 的 Chronos2Pipeline，
使用 predict_df 接口对 OHLCV 收盘价序列进行概率预测。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from branch_contracts import BranchResult

from .base import KLineBackend


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class ChronosBackend(KLineBackend):
    name = "chronos"
    reliability = 0.75
    horizon_days = 5

    def __init__(self, model_name: str | None = None, **_kwargs: object) -> None:
        self._model_name = model_name or os.environ.get(
            "CHRONOS_MODEL_NAME", "amazon/chronos-2-small"
        )
        self._pipeline = None
        # 确保 chronos 包可导入
        chronos_src = "${CHRONOS_SRC_DIR}"
        if chronos_src not in sys.path:
            sys.path.insert(0, chronos_src)

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        from chronos import Chronos2Pipeline  # type: ignore[import-untyped]
        self._pipeline = Chronos2Pipeline.from_pretrained(self._model_name)

    def _build_input_df(self, symbol_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """将多只股票的收盘价合并为 Chronos-2 所需的 long-format DataFrame。"""
        rows: list[pd.DataFrame] = []
        for symbol, df in symbol_data.items():
            if df.empty:
                continue
            sub = df[["date", "close"]].copy()
            sub = sub.rename(columns={"date": "timestamp", "close": "target"})
            sub["item_id"] = symbol
            rows.append(sub)
        if not rows:
            return pd.DataFrame()
        combined = pd.concat(rows, ignore_index=True)
        return combined.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        self._ensure_pipeline()

        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        input_df = self._build_input_df(symbol_data)
        if input_df.empty:
            for symbol in stock_pool:
                symbol_scores[symbol] = 0.0
                predicted_returns[symbol] = 0.0
                regimes[symbol] = "震荡"
            return self._build_result(symbol_scores, predicted_returns, regimes, stock_pool)

        try:
            pred_df = self._pipeline.predict_df(
                input_df,
                prediction_length=self.horizon_days,
                quantile_levels=[0.1, 0.5, 0.9],
                cross_learning=len(symbol_data) > 1,
            )
        except Exception:
            for symbol in stock_pool:
                symbol_scores[symbol] = 0.0
                predicted_returns[symbol] = 0.0
                regimes[symbol] = "震荡"
            return self._build_result(symbol_scores, predicted_returns, regimes, stock_pool)

        # 解析预测结果
        for symbol, df in symbol_data.items():
            if df.empty:
                continue
            last_close = float(df["close"].iloc[-1])
            sym_pred = pred_df[pred_df["item_id"] == symbol] if "item_id" in pred_df.columns else pd.DataFrame()
            if sym_pred.empty or last_close <= 0:
                symbol_scores[symbol] = 0.0
                predicted_returns[symbol] = 0.0
                regimes[symbol] = "震荡"
                continue

            # 取中位数预测的最后一期
            median_col = [c for c in sym_pred.columns if "0.5" in str(c)]
            if median_col:
                median_pred = float(sym_pred[median_col[0]].iloc[-1])
            else:
                median_pred = last_close

            ret = (median_pred - last_close) / last_close
            vol = float(df["close"].pct_change().tail(20).std()) if len(df) > 20 else 0.02
            signal = _clamp(ret / (vol * 5 + 1e-8), -1.0, 1.0)
            regime = "上行" if signal > 0.2 else "下行" if signal < -0.2 else "震荡"

            symbol_scores[symbol] = signal
            predicted_returns[symbol] = _clamp(ret, -0.3, 0.3)
            regimes[symbol] = regime

        for symbol in stock_pool:
            symbol_scores.setdefault(symbol, 0.0)
            predicted_returns.setdefault(symbol, 0.0)
            regimes.setdefault(symbol, "震荡")

        return self._build_result(symbol_scores, predicted_returns, regimes, stock_pool)

    def _build_result(
        self,
        symbol_scores: dict[str, float],
        predicted_returns: dict[str, float],
        regimes: dict[str, str],
        stock_pool: list[str],
    ) -> BranchResult:
        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.48 + float(np.std(list(symbol_scores.values()) or [0.0])), 0.38, 0.82)
        return BranchResult(
            branch_name="kline",
            score=score,
            confidence=confidence,
            signals={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "model_mode": "chronos",
            },
            risks=[],
            explanation="K线分析（Chronos-2）使用 Amazon 时序基础模型进行概率预测。",
            symbol_scores=symbol_scores,
            metadata={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "branch_mode": "kline_chronos",
                "reliability": self.reliability,
                "horizon_days": self.horizon_days,
            },
        )
