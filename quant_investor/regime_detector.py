"""
市场状态识别模块（Market Regime Detector）
==========================================

基于三个维度识别四种市场状态：
  - 波动率状态：滚动 20 日已实现波动率分位数（高/低）
  - 趋势强度：ADX 指标（>25 为趋势市，<20 为震荡市）
  - 相关性状态：股票间平均相关系数（高相关 = 系统性风险偏高）

输出四种状态（MarketRegime）：
  TREND_BULL      趋势上涨
  TREND_BEAR      趋势下跌
  RANGE_LOW_VOL   震荡低波
  RANGE_HIGH_VOL  震荡高波

不同状态下的建议参数调整：
  TREND_BULL:      K线分析权重 ↑，止损放宽（0.10），再平衡频率：周
  TREND_BEAR:      全面降仓（仓位上限 40%），止损收紧（0.05），现金优先
  RANGE_LOW_VOL:   Quant/Intelligence 权重 ↑，均值回归策略，再平衡频率：月
  RANGE_HIGH_VOL:  降仓（仓位上限 60%），风险平价权重，再平衡频率：周
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class MarketRegime(Enum):
    TREND_BULL = "趋势上涨"
    TREND_BEAR = "趋势下跌"
    RANGE_LOW_VOL = "震荡低波"
    RANGE_HIGH_VOL = "震荡高波"
    UNKNOWN = "未知"


@dataclass
class RegimeParams:
    """当前状态建议的策略参数"""
    regime: MarketRegime
    max_position: float          # 最大仓位
    stop_loss_pct: float         # 止损比例（负数）
    rebalance_freq: str          # "D" / "W" / "M"
    branch_weight_adjustments: dict[str, float]  # 对默认权重的倍率调整
    description: str


# 每种状态的推荐参数
_REGIME_PARAMS: dict[MarketRegime, RegimeParams] = {
    MarketRegime.TREND_BULL: RegimeParams(
        regime=MarketRegime.TREND_BULL,
        max_position=0.90,
        stop_loss_pct=-0.10,
        rebalance_freq="W",
        branch_weight_adjustments={"kline": 1.5, "quant": 0.9, "macro": 1.1},
        description="趋势上涨市：K线分析权重上调，止损放宽至 10%，每周再平衡",
    ),
    MarketRegime.TREND_BEAR: RegimeParams(
        regime=MarketRegime.TREND_BEAR,
        max_position=0.40,
        stop_loss_pct=-0.05,
        rebalance_freq="W",
        branch_weight_adjustments={"kline": 1.3, "macro": 1.5, "fundamental": 0.7},
        description="趋势下跌市：仓位上限 40%，止损收紧至 5%，宏观权重上调",
    ),
    MarketRegime.RANGE_LOW_VOL: RegimeParams(
        regime=MarketRegime.RANGE_LOW_VOL,
        max_position=0.85,
        stop_loss_pct=-0.08,
        rebalance_freq="M",
        branch_weight_adjustments={"quant": 1.3, "intelligence": 1.2, "kline": 0.7},
        description="震荡低波市：因子选股权重上调，每月再平衡降低摩擦成本",
    ),
    MarketRegime.RANGE_HIGH_VOL: RegimeParams(
        regime=MarketRegime.RANGE_HIGH_VOL,
        max_position=0.60,
        stop_loss_pct=-0.07,
        rebalance_freq="W",
        branch_weight_adjustments={"macro": 1.4, "quant": 1.1, "kline": 0.8},
        description="震荡高波市：仓位上限 60%，宏观权重上调，风险平价仓位",
    ),
}


class RegimeDetector:
    """
    市场状态识别器。

    输入：市场指数日收益率序列（pd.Series，索引为日期）
    输出：MarketRegime 枚举值 + 对应 RegimeParams

    用法：
        detector = RegimeDetector()
        regime, params = detector.detect(market_returns)
    """

    VOL_WINDOW = 20           # 波动率计算窗口
    VOL_HIGH_QUANTILE = 0.65  # 超过历史 65% 分位数视为高波动
    ADX_WINDOW = 14           # ADX 计算窗口
    ADX_TREND_THRESHOLD = 25  # ADX > 25 判断为趋势市
    ADX_RANGE_THRESHOLD = 20  # ADX < 20 判断为震荡市

    def __init__(self, vol_lookback: int = 252) -> None:
        """
        Args:
            vol_lookback: 用于计算波动率历史分位数的回溯期（交易日数）
        """
        self.vol_lookback = vol_lookback

    def detect(
        self,
        market_returns: pd.Series,
        symbol_returns: Optional[pd.DataFrame] = None,
    ) -> tuple[MarketRegime, RegimeParams]:
        """
        识别当前市场状态。

        Args:
            market_returns: 市场指数（如沪深300）的日收益率，索引为日期
            symbol_returns: 个股日收益率矩阵（行=日期，列=symbol），用于计算相关性状态
                           若为 None，跳过相关性维度

        Returns:
            (MarketRegime, RegimeParams)
        """
        if len(market_returns) < max(self.ADX_WINDOW, self.VOL_WINDOW) + 5:
            return MarketRegime.UNKNOWN, self._default_params()

        recent = market_returns.dropna().tail(self.vol_lookback)

        # 维度1：波动率状态
        rolling_vol = recent.rolling(self.VOL_WINDOW).std() * np.sqrt(252)
        current_vol = float(rolling_vol.iloc[-1]) if len(rolling_vol.dropna()) > 0 else 0.20
        vol_quantile = float((rolling_vol.dropna() <= current_vol).mean())
        is_high_vol = vol_quantile >= self.VOL_HIGH_QUANTILE

        # 维度2：趋势强度（ADX 近似）
        adx = self._calc_adx(recent, self.ADX_WINDOW)
        is_trending = adx >= self.ADX_TREND_THRESHOLD

        # 维度3：方向（多头 or 空头趋势）
        trend_direction = float(recent.tail(self.VOL_WINDOW).mean())

        # 综合判断
        if is_trending:
            regime = MarketRegime.TREND_BULL if trend_direction >= 0 else MarketRegime.TREND_BEAR
        else:
            regime = MarketRegime.RANGE_HIGH_VOL if is_high_vol else MarketRegime.RANGE_LOW_VOL

        params = _REGIME_PARAMS.get(regime, self._default_params())
        return regime, params

    @staticmethod
    def _calc_adx(returns: pd.Series, window: int) -> float:
        """
        用收益率序列近似计算 ADX（方向性运动指数）。

        真正的 ADX 需要 OHLC 数据；这里用收益率的绝对值方向性作为代理：
          DM+ = max(return, 0)，DM- = max(-return, 0)
          DI+ = EWM(DM+) / EWM(|return|)
          DI- = EWM(DM-) / EWM(|return|)
          ADX ≈ EWM(|DI+ - DI-| / (DI+ + DI-)) * 100
        """
        r = returns.dropna().tail(window * 3)
        if len(r) < window:
            return 0.0
        dm_plus = r.clip(lower=0)
        dm_minus = (-r).clip(lower=0)
        tr = r.abs()
        alpha = 1.0 / window
        di_plus = dm_plus.ewm(alpha=alpha, adjust=False).mean()
        di_minus = dm_minus.ewm(alpha=alpha, adjust=False).mean()
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        denom = (di_plus + di_minus) / (tr_smooth + 1e-8)
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)) * 100
        adx = float(dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
        return adx

    @staticmethod
    def _default_params() -> RegimeParams:
        return RegimeParams(
            regime=MarketRegime.UNKNOWN,
            max_position=0.80,
            stop_loss_pct=-0.08,
            rebalance_freq="W",
            branch_weight_adjustments={},
            description="未识别状态，使用默认参数",
        )
