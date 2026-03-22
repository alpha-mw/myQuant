"""Chronos-2 时序模型后端。

统一从 myQuant 内 vendored Chronos 源码加载；若本地权重不可用，
则回退到内部统计替代模式，避免依赖仓库外的 `chronos-forecasting` 代码。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant_investor.contracts import BranchResult
from quant_investor._vendor.chronos_loader import load_vendored_chronos

from .base import KLineBackend


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class ChronosBackend(KLineBackend):
    """Chronos 单模型后端，供组合 K-line 主链内部调用。"""

    name = "chronos"
    reliability = 0.75
    horizon_days = 5

    def __init__(
        self,
        model_name: str | None = None,
        allow_remote_download: bool = False,
        **_kwargs: object,
    ) -> None:
        self._model_name = model_name or str(
            Path(__file__).resolve().parents[2] / "data" / "models" / "chronos-2"
        )
        self._allow_remote_download = allow_remote_download
        self._pipeline = None
        self._runtime_mode = "statistical_fallback"
        self._load_error = ""

    @property
    def runtime_mode(self) -> str:
        return self._runtime_mode

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        chronos_module = load_vendored_chronos()
        Chronos2Pipeline = chronos_module.Chronos2Pipeline  # type: ignore[attr-defined]
        kwargs = {}
        is_local_path = Path(self._model_name).expanduser().exists()
        if not self._allow_remote_download and not is_local_path:
            self._load_error = "未配置 myQuant 内本地 Chronos 权重目录"
            self._runtime_mode = "statistical_fallback"
            self._pipeline = None
            return
        if not is_local_path:
            kwargs["local_files_only"] = not self._allow_remote_download
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        if not self._allow_remote_download and not is_local_path:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            self._pipeline = Chronos2Pipeline.from_pretrained(self._model_name, **kwargs)
            self._runtime_mode = "vendor_native"
        except TypeError:
            self._pipeline = Chronos2Pipeline.from_pretrained(self._model_name)
            self._runtime_mode = "vendor_native"
        except Exception as exc:
            self._load_error = str(exc)
            self._runtime_mode = "statistical_fallback"
            self._pipeline = None
        finally:
            if not self._allow_remote_download and not is_local_path:
                if previous_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = previous_offline
                if previous_transformers_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

    def _build_input_df(self, symbol_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        return pd.concat(rows, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    def _statistical_return(self, df: pd.DataFrame) -> float:
        close = df["close"].astype(float)
        returns = close.pct_change().dropna()
        if returns.empty:
            return 0.0

        lookback = returns.tail(min(len(returns), 60))
        drift = float(lookback.median())
        vol = float(lookback.std())
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.02

        rng = np.random.default_rng(abs(hash((len(df), self._model_name))) % (2**32))
        samples = []
        for _ in range(32):
            path = close.iloc[-1]
            for _ in range(self.horizon_days):
                shock = rng.normal(drift, vol)
                path *= max(1.0 + np.clip(shock, -0.12, 0.12), 0.01)
            samples.append(path)

        predicted_close = float(np.median(samples))
        last_close = float(close.iloc[-1])
        return (predicted_close - last_close) / last_close if last_close > 0 else 0.0

    def _predict_with_fallback(
        self,
        symbol_data: dict[str, pd.DataFrame],
        stock_pool: list[str],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        for symbol, df in symbol_data.items():
            if df.empty or len(df) < 30:
                continue
            ret = self._statistical_return(df)
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

        return symbol_scores, predicted_returns, regimes

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        self._ensure_pipeline()

        symbol_scores: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        regimes: dict[str, str] = {}

        input_df = self._build_input_df(symbol_data)
        if self._pipeline is not None and not input_df.empty:
            try:
                pred_df = self._pipeline.predict_df(
                    input_df,
                    prediction_length=self.horizon_days,
                    quantile_levels=[0.1, 0.5, 0.9],
                    cross_learning=len(symbol_data) > 1,
                )
                for symbol, df in symbol_data.items():
                    if df.empty:
                        continue
                    last_close = float(df["close"].iloc[-1])
                    sym_pred = (
                        pred_df[pred_df["item_id"] == symbol]
                        if "item_id" in pred_df.columns
                        else pd.DataFrame()
                    )
                    if sym_pred.empty or last_close <= 0:
                        continue

                    median_cols = [column for column in sym_pred.columns if "0.5" in str(column)]
                    predicted_close = float(sym_pred[median_cols[0]].iloc[-1]) if median_cols else last_close
                    ret = (predicted_close - last_close) / last_close
                    vol = float(df["close"].pct_change().tail(20).std()) if len(df) > 20 else 0.02
                    signal = _clamp(ret / (vol * 5 + 1e-8), -1.0, 1.0)
                    regime = "上行" if signal > 0.2 else "下行" if signal < -0.2 else "震荡"

                    symbol_scores[symbol] = signal
                    predicted_returns[symbol] = _clamp(ret, -0.3, 0.3)
                    regimes[symbol] = regime
            except Exception as exc:
                self._load_error = str(exc)
                self._runtime_mode = "statistical_fallback"
                symbol_scores, predicted_returns, regimes = self._predict_with_fallback(symbol_data, stock_pool)
        else:
            symbol_scores, predicted_returns, regimes = self._predict_with_fallback(symbol_data, stock_pool)

        for symbol in stock_pool:
            symbol_scores.setdefault(symbol, 0.0)
            predicted_returns.setdefault(symbol, 0.0)
            regimes.setdefault(symbol, "震荡")

        reliability = self.reliability if self._runtime_mode == "vendor_native" else 0.56
        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.48 + float(np.std(list(symbol_scores.values()) or [0.0])), 0.38, 0.82)
        risks = []
        if self._runtime_mode != "vendor_native":
            message = "Chronos 原生权重未命中，已使用统计替代预测"
            if self._load_error:
                message = f"{message}: {self._load_error}"
            risks.append(message)

        return BranchResult(
            branch_name="kline",
            score=score,
            confidence=confidence,
            signals={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "model_mode": "chronos",
                "model_runtime_mode": self._runtime_mode,
                "chronos_model_name": self._model_name,
            },
            risks=risks,
            explanation=(
                "K线分析（Chronos）已切换为 myQuant 内置模型链路，"
                f"当前模型为 {self._model_name}，运行模式为 {self._runtime_mode}。"
            ),
            symbol_scores=symbol_scores,
            metadata={
                "predicted_return": predicted_returns,
                "trend_regime": regimes,
                "branch_mode": "kline_chronos_internal",
                "reliability": reliability,
                "horizon_days": self.horizon_days,
                "model_runtime_mode": self._runtime_mode,
                "model_source": "internal_vendor",
                "chronos_model_name": self._model_name,
            },
        )
