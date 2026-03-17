"""Kronos 预训练 K线模型后端适配器。

包装 ${KRONOS_MODEL_PATH}/kronos_lab 的 PredictionService，
将 OHLCV DataFrame 转为 Kronos 所需的 CSV 输入格式并解析预测结果。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from branch_contracts import BranchResult

from .base import KLineBackend


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class KronosBackend(KLineBackend):
    name = "kronos"
    reliability = 0.78
    horizon_days = 5

    def __init__(self, kronos_path: str | None = None, **_kwargs: object) -> None:
        self._kronos_path = kronos_path or os.environ.get(
            "KRONOS_MODEL_PATH", "${KRONOS_MODEL_PATH}"
        )
        self._service = None

    def _ensure_service(self) -> None:
        if self._service is not None:
            return
        lab_path = self._kronos_path
        if lab_path not in sys.path:
            sys.path.insert(0, lab_path)
        from kronos_lab import PredictionService  # type: ignore[import-untyped]
        self._service = PredictionService()

    def _df_to_csv(self, symbol: str, df: pd.DataFrame) -> Path:
        """将 OHLCV DataFrame 转为 Kronos 所需的 CSV 格式。"""
        export = df[["date", "open", "high", "low", "close", "volume"]].copy()
        export = export.rename(columns={"date": "timestamp"})
        export["item_id"] = symbol
        export["target"] = export["close"]
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        export.to_csv(tmp, index=False)
        return tmp

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        self._ensure_service()

        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        for symbol, df in symbol_data.items():
            if df.empty or len(df) < 30:
                continue
            try:
                csv_path = self._df_to_csv(symbol, df)
                result = self._service.predict(
                    csv_path=str(csv_path),
                    item_id=symbol,
                    lookback=min(400, len(df)),
                    horizon=self.horizon_days,
                    sample_count=3,
                )
                csv_path.unlink(missing_ok=True)

                # 解析 Kronos 预测结果
                pred_values = result.get("predicted_values", [])
                if pred_values:
                    last_close = float(df["close"].iloc[-1])
                    if last_close > 0:
                        median_pred = float(np.median([v[-1] if isinstance(v, list) else v for v in pred_values]))
                        ret = (median_pred - last_close) / last_close
                    else:
                        ret = 0.0
                else:
                    ret = 0.0

                vol = float(df["close"].pct_change().tail(20).std()) if len(df) > 20 else 0.02
                signal = _clamp(ret / (vol * 5 + 1e-8), -1.0, 1.0)
                regime = "上行" if signal > 0.2 else "下行" if signal < -0.2 else "震荡"

                symbol_scores[symbol] = signal
                predicted_returns[symbol] = _clamp(ret, -0.3, 0.3)
                regimes[symbol] = regime
            except Exception:
                symbol_scores[symbol] = 0.0
                predicted_returns[symbol] = 0.0
                regimes[symbol] = "震荡"

        for symbol in stock_pool:
            symbol_scores.setdefault(symbol, 0.0)
            predicted_returns.setdefault(symbol, 0.0)
            regimes.setdefault(symbol, "震荡")

        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.50 + float(np.std(list(symbol_scores.values()) or [0.0])), 0.40, 0.85)
        return BranchResult(
            branch_name="kline",
            score=score,
            confidence=confidence,
            signals={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "model_mode": "kronos",
            },
            risks=[],
            explanation="K线分析（Kronos）使用预训练 K线基础模型进行多步预测。",
            symbol_scores=symbol_scores,
            metadata={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "branch_mode": "kline_kronos",
                "reliability": self.reliability,
                "horizon_days": self.horizon_days,
            },
        )
