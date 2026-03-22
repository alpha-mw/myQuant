"""Kronos 预训练 K线模型后端。

不再依赖仓库外的旧版模型路径；统一复用 myQuant 内置的
`KronosIntegrator` 与 vendored `quant_investor._vendor.kronos_model`。
若本地权重不可用，则自动回退到统计替代模式，但仍保持标准输出契约。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.contracts import BranchResult
from quant_investor.kronos_predictor import KronosIntegrator

from .base import KLineBackend


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class KronosBackend(KLineBackend):
    """Kronos 单模型后端，供组合 K-line 主链内部调用。"""

    name = "kronos"
    reliability = 0.78
    horizon_days = 5

    _SIZE_TO_MODEL_NAME = {
        "mini": "kronos-mini",
        "small": "kronos-small",
        "base": "kronos-base",
        "kronos-mini": "kronos-mini",
        "kronos-small": "kronos-small",
        "kronos-base": "kronos-base",
    }

    def __init__(
        self,
        kronos_path: str | None = None,
        kronos_model_size: str | None = None,
        allow_remote_download: bool = False,
        **_kwargs: object,
    ) -> None:
        self._model_name = self._normalize_model_name(kronos_model_size)
        self._integrator = KronosIntegrator(
            model_name=self._model_name,
            model_path=kronos_path,
            sample_count=3,
            allow_remote_download=allow_remote_download,
        )

    @classmethod
    def _normalize_model_name(cls, kronos_model_size: str | None) -> str:
        return cls._SIZE_TO_MODEL_NAME.get(str(kronos_model_size or "base").lower(), "kronos-base")

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        for symbol, df in symbol_data.items():
            if df.empty or len(df) < 30:
                continue

            forecast = self._integrator.predict_single(symbol, df, pred_len=self.horizon_days)
            ret = float(forecast.pred_close_pct) / 100.0
            ann_vol = max(float(forecast.volatility_forecast) / 100.0, 0.10)
            signal = _clamp(ret / (ann_vol * 0.8 + 1e-8), -1.0, 1.0)
            regime = "上行" if signal > 0.2 else "下行" if signal < -0.2 else "震荡"

            symbol_scores[symbol] = signal
            predicted_returns[symbol] = _clamp(ret, -0.3, 0.3)
            regimes[symbol] = regime

        for symbol in stock_pool:
            symbol_scores.setdefault(symbol, 0.0)
            predicted_returns.setdefault(symbol, 0.0)
            regimes.setdefault(symbol, "震荡")

        runtime_mode = self._integrator.runtime_mode
        reliability = self.reliability if runtime_mode == "vendor_native" else 0.58
        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.50 + float(np.std(list(symbol_scores.values()) or [0.0])), 0.40, 0.86)
        risks = []
        if runtime_mode != "vendor_native":
            risks.append("Kronos 原生权重未命中，已使用统计替代预测")

        return BranchResult(
            branch_name="kline",
            score=score,
            confidence=confidence,
            signals={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "model_mode": "kronos",
                "model_runtime_mode": runtime_mode,
                "kronos_model_name": self._model_name,
            },
            risks=risks,
            explanation=(
                "K线分析（Kronos）已切换为 myQuant 内置模型链路，"
                f"当前模型为 {self._model_name}，运行模式为 {runtime_mode}。"
            ),
            symbol_scores=symbol_scores,
            metadata={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "branch_mode": "kline_kronos_internal",
                "reliability": reliability,
                "horizon_days": self.horizon_days,
                "model_runtime_mode": runtime_mode,
                "model_source": "internal_vendor",
                "kronos_model_name": self._model_name,
            },
        )
