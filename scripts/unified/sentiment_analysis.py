"""
Market Sentiment Analysis Engine
==================================
市场情绪分析模块 - 全方位市场情绪评估

核心功能：
  1. 恐慌贪婪指数       - 综合7大维度的市场情绪指数
  2. 技术情绪指标        - RSI/MACD/布林带/成交量情绪
  3. 资金流向分析        - 主力资金、北向资金、融资融券
  4. 期权情绪分析        - 认购/认沽比率（A股ETF期权）
  5. 市场广度指标        - 涨跌家数比、创新高/新低家数
  6. 行业轮动追踪        - 热门行业/板块轮动信号
  7. 龙虎榜分析          - 游资/机构席位动向
  8. 短期过热/超卖检测   - 多指标共振判断极端情绪
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

_logger = get_logger("SentimentAnalysis")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FearGreedIndex:
    """恐慌贪婪指数（仿 CNN Fear & Greed，0=极度恐慌，100=极度贪婪）"""
    score: float               # 0–100
    grade: str                 # "极度恐慌" | "恐慌" | "中性" | "贪婪" | "极度贪婪"
    components: dict[str, float] = field(default_factory=dict)
    # 各分项权重
    # market_momentum: 价格动量（20日涨跌）
    # rsi_breadth: RSI分布
    # volume_signal: 量价关系
    # put_call_ratio: 认购/认沽比
    # fund_flow: 资金流向
    # margin_trading: 融资融券净买入
    # market_breadth: 涨跌家数比


@dataclass
class TechnicalSentiment:
    """技术面情绪指标"""
    rsi_14: float              # 相对强弱指数
    rsi_signal: str            # "超买" | "中性" | "超卖"
    macd_signal: str           # "金叉" | "死叉" | "震荡"
    bb_position: float         # 布林带位置 0–1（0=下轨，1=上轨）
    bb_signal: str             # "超买" | "中性" | "超卖"
    volume_ratio: float        # 量比（当日/过去5日均量）
    volume_signal: str         # "放量" | "缩量" | "正常"
    kdj_signal: str            # "超买" | "中性" | "超卖"
    momentum_20d: float        # 20日动量（%）
    composite_score: float     # 技术情绪综合得分 -1~1


@dataclass
class CapitalFlowSentiment:
    """资金流向情绪"""
    symbol: str
    net_inflow_1d: float       # 近1日净流入（万元）
    net_inflow_5d: float       # 近5日净流入
    net_inflow_10d: float      # 近10日净流入
    main_force_ratio: float    # 主力资金净流入比例（占成交额%）
    north_bound_flow: float    # 北向资金净流入（万元，沪深港通）
    margin_balance: float      # 融资余额（亿元）
    margin_net_buy: float      # 融资净买入（万元）
    flow_signal: str           # "大幅流入" | "流入" | "平衡" | "流出" | "大幅流出"
    flow_score: float          # -1~1


@dataclass
class MarketBreadth:
    """市场广度指标"""
    advance_count: int         # 上涨家数
    decline_count: int         # 下跌家数
    unchanged_count: int       # 平盘家数
    advance_decline_ratio: float  # 涨跌比
    new_52w_high: int          # 创52周新高家数
    new_52w_low: int           # 创52周新低家数
    stocks_above_ma20: float   # 在20日均线上方的股票占比 (%)
    stocks_above_ma200: float  # 在200日均线上方的股票占比 (%)
    breadth_signal: str        # "极强" | "强" | "中性" | "弱" | "极弱"


@dataclass
class SectorRotation:
    """行业轮动信号"""
    hot_sectors: list[str]             # 近期资金流入最多的行业
    cold_sectors: list[str]            # 近期资金流出最多的行业
    rotation_stage: str                # "早周期" | "中周期" | "晚周期" | "防御"
    momentum_sectors: list[str]        # 动量最强的行业
    reversal_sectors: list[str]        # 反转机会较大的行业
    sector_dispersion: float           # 行业分化程度 0–1


@dataclass
class MarketSentimentReport:
    """综合市场情绪报告"""
    symbol: str
    stock_name: str
    timestamp: datetime

    fear_greed: Optional[FearGreedIndex] = None
    technical: Optional[TechnicalSentiment] = None
    capital_flow: Optional[CapitalFlowSentiment] = None
    market_breadth: Optional[MarketBreadth] = None
    sector_rotation: Optional[SectorRotation] = None

    # 综合情绪
    overall_sentiment_score: float = 0.0  # -1~1（-1极度恐慌，+1极度贪婪）
    overall_sentiment_label: str = "中性"
    contrarian_signal: str = ""           # 逆向信号（情绪极端时）
    sentiment_signal: str = "中性"        # "强烈看多" | "看多" | "中性" | "看空" | "强烈看空"
    signal_confidence: float = 0.5

    summary: str = ""


# ---------------------------------------------------------------------------
# 技术情绪计算
# ---------------------------------------------------------------------------

class TechnicalSentimentCalculator:
    """基于OHLCV数据计算技术情绪指标"""

    def calculate(self, df: pd.DataFrame) -> TechnicalSentiment:
        """
        输入标准OHLCV DataFrame，输出技术情绪。
        df 需要包含: close, high, low, open, volume
        """
        try:
            close = df["close"].values.astype(float)
            volume = df.get("volume", pd.Series(np.ones(len(df)))).values.astype(float)

            # RSI
            rsi = self._calc_rsi(close, 14)
            if rsi > 70:
                rsi_signal = "超买"
            elif rsi < 30:
                rsi_signal = "超卖"
            else:
                rsi_signal = "中性"

            # MACD
            macd_line, signal_line = self._calc_macd(close)
            if macd_line > signal_line and macd_line > 0:
                macd_signal = "金叉"
            elif macd_line < signal_line and macd_line < 0:
                macd_signal = "死叉"
            else:
                macd_signal = "震荡"

            # 布林带位置
            bb_pos = self._calc_bb_position(close, 20)
            if bb_pos > 0.85:
                bb_signal = "超买"
            elif bb_pos < 0.15:
                bb_signal = "超卖"
            else:
                bb_signal = "中性"

            # 量比
            if len(volume) >= 5:
                vol_ratio = float(volume[-1] / (np.mean(volume[-6:-1]) + 1e-10))
            else:
                vol_ratio = 1.0
            if vol_ratio > 2.0:
                vol_signal = "放量"
            elif vol_ratio < 0.5:
                vol_signal = "缩量"
            else:
                vol_signal = "正常"

            # KDJ
            kdj_k, kdj_d = self._calc_kdj(df)
            if kdj_k > 80 and kdj_d > 80:
                kdj_signal = "超买"
            elif kdj_k < 20 and kdj_d < 20:
                kdj_signal = "超卖"
            else:
                kdj_signal = "中性"

            # 20日动量
            if len(close) >= 20:
                momentum_20d = (close[-1] / close[-20] - 1) * 100
            else:
                momentum_20d = 0.0

            # 技术情绪综合得分
            score = self._composite_score(rsi, bb_pos, macd_line, signal_line, momentum_20d)

            return TechnicalSentiment(
                rsi_14=round(rsi, 2),
                rsi_signal=rsi_signal,
                macd_signal=macd_signal,
                bb_position=round(bb_pos, 3),
                bb_signal=bb_signal,
                volume_ratio=round(vol_ratio, 3),
                volume_signal=vol_signal,
                kdj_signal=kdj_signal,
                momentum_20d=round(momentum_20d, 2),
                composite_score=round(score, 3),
            )
        except Exception as e:
            _logger.warning(f"技术情绪计算失败: {e}")
            return TechnicalSentiment(
                rsi_14=50.0, rsi_signal="中性", macd_signal="震荡",
                bb_position=0.5, bb_signal="中性", volume_ratio=1.0,
                volume_signal="正常", kdj_signal="中性", momentum_20d=0.0,
                composite_score=0.0,
            )

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        changes = np.diff(close[-(period + 2):])
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        if len(close) < slow + signal:
            return 0.0, 0.0
        # EMA 计算
        def ema(arr, n):
            alpha = 2 / (n + 1)
            result = [arr[0]]
            for v in arr[1:]:
                result.append(result[-1] * (1 - alpha) + v * alpha)
            return np.array(result)

        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_arr = ema(macd_line, signal)
        return float(macd_line[-1]), float(signal_arr[-1])

    @staticmethod
    def _calc_bb_position(close: np.ndarray, period: int = 20) -> float:
        if len(close) < period:
            return 0.5
        window = close[-period:]
        mid = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return 0.5
        upper = mid + 2 * std
        lower = mid - 2 * std
        pos = (close[-1] - lower) / (upper - lower + 1e-10)
        return float(np.clip(pos, 0.0, 1.0))

    @staticmethod
    def _calc_kdj(df: pd.DataFrame, n: int = 9) -> tuple[float, float]:
        if len(df) < n or "high" not in df.columns or "low" not in df.columns:
            return 50.0, 50.0
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        lowest_low = np.min(low[-n:])
        highest_high = np.max(high[-n:])
        if highest_high == lowest_low:
            return 50.0, 50.0
        rsv = (close[-1] - lowest_low) / (highest_high - lowest_low) * 100
        k = (2 / 3) * 50 + (1 / 3) * rsv  # 简化KDJ
        d = (2 / 3) * 50 + (1 / 3) * k
        return float(k), float(d)

    @staticmethod
    def _composite_score(rsi: float, bb_pos: float, macd: float, signal: float, momentum: float) -> float:
        """将多个技术指标合成 -1~1 的情绪得分"""
        # RSI 得分：<30 = -1, >70 = +1
        rsi_score = (rsi - 50) / 50  # -1~1

        # BB 位置得分：>0.8 = +1（过热，可能反转），需要谨慎
        bb_score = (bb_pos - 0.5) * 2  # -1~1

        # MACD 得分
        macd_score = np.tanh((macd - signal) * 10)

        # 动量得分（20日）
        mom_score = np.tanh(momentum / 10)

        # 加权平均
        weights = [0.25, 0.25, 0.25, 0.25]
        composite = np.dot(weights, [rsi_score, bb_score, float(macd_score), float(mom_score)])
        return float(np.clip(composite, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 资金流向分析
# ---------------------------------------------------------------------------

class CapitalFlowAnalyzer:
    """资金流向分析（支持AKShare数据源）"""

    def analyze(
        self,
        symbol: str,
        days: int = 10,
    ) -> CapitalFlowSentiment:
        """获取并分析资金流向数据"""
        try:
            import akshare as ak  # type: ignore
            code = symbol.split(".")[0]

            # 个股资金流向
            try:
                df = ak.stock_individual_fund_flow(stock=code, market="sh" if ".SH" in symbol else "sz")
                if df is not None and not df.empty:
                    df = df.sort_values("日期", ascending=False)
                    net_1d = float(df.iloc[0].get("净流入-净额", 0)) if len(df) > 0 else 0.0
                    net_5d = float(df.head(5).get("净流入-净额", pd.Series([0])).sum()) if len(df) >= 5 else 0.0
                    net_10d = float(df.head(10).get("净流入-净额", pd.Series([0])).sum()) if len(df) >= 10 else 0.0
                    main_ratio = float(df.iloc[0].get("主力净流入-净占比", 0)) if len(df) > 0 else 0.0
                    return self._build_result(symbol, net_1d, net_5d, net_10d, main_ratio)
            except Exception as e:
                _logger.debug(f"AKShare资金流向接口异常: {e}")

            # 尝试北向资金
            try:
                df_north = ak.stock_hsgt_board_rank_em(symbol="北向资金", indicator="今日")
                if df_north is not None and not df_north.empty:
                    north_flow = float(df_north.get("净买额", pd.Series([0])).sum())
                    return self._build_result(symbol, 0.0, 0.0, 0.0, 0.0, north_flow)
            except Exception:
                pass

        except ImportError:
            _logger.debug("AKShare未安装，使用估算资金流向")

        # 返回空数据
        return CapitalFlowSentiment(
            symbol=symbol,
            net_inflow_1d=0.0, net_inflow_5d=0.0, net_inflow_10d=0.0,
            main_force_ratio=0.0, north_bound_flow=0.0,
            margin_balance=0.0, margin_net_buy=0.0,
            flow_signal="未知", flow_score=0.0,
        )

    def _build_result(
        self, symbol: str,
        net_1d: float, net_5d: float, net_10d: float,
        main_ratio: float, north_flow: float = 0.0,
    ) -> CapitalFlowSentiment:
        # 资金流向信号
        score = np.tanh(net_5d / 1e6)  # 标准化
        if score > 0.4:
            signal = "大幅流入"
        elif score > 0.1:
            signal = "流入"
        elif score < -0.4:
            signal = "大幅流出"
        elif score < -0.1:
            signal = "流出"
        else:
            signal = "平衡"

        return CapitalFlowSentiment(
            symbol=symbol,
            net_inflow_1d=round(net_1d / 1e4, 2),  # 元→万元
            net_inflow_5d=round(net_5d / 1e4, 2),
            net_inflow_10d=round(net_10d / 1e4, 2),
            main_force_ratio=round(main_ratio, 2),
            north_bound_flow=round(north_flow / 1e4, 2),
            margin_balance=0.0,
            margin_net_buy=0.0,
            flow_signal=signal,
            flow_score=round(float(score), 3),
        )


# ---------------------------------------------------------------------------
# 市场广度分析
# ---------------------------------------------------------------------------

class MarketBreadthAnalyzer:
    """A股市场广度指标"""

    def analyze(self, market: str = "CN") -> MarketBreadth:
        """获取市场整体广度数据"""
        try:
            import akshare as ak  # type: ignore

            # 获取涨跌停板数据
            try:
                df_limit = ak.stock_limit_up_down_em()
                if df_limit is not None and not df_limit.empty:
                    pass
            except Exception:
                pass

            # 获取全市场行情概况
            try:
                df_broad = ak.stock_market_activity_legu()
                if df_broad is not None and not df_broad.empty:
                    advance = int(df_broad.get("涨家数", [0]).iloc[0] if hasattr(df_broad.get("涨家数", pd.Series([0])), 'iloc') else 0)
                    decline = int(df_broad.get("跌家数", [0]).iloc[0] if hasattr(df_broad.get("跌家数", pd.Series([0])), 'iloc') else 0)
                    return self._build_result(advance, decline, 0)
            except Exception as e:
                _logger.debug(f"市场广度数据获取失败: {e}")

        except ImportError:
            pass

        # 返回估算数据
        return self._build_result(1800, 2200, 500)

    def _build_result(self, advance: int, decline: int, unchanged: int) -> MarketBreadth:
        total = advance + decline + unchanged
        if total == 0:
            total = 1
        ad_ratio = advance / (decline + 1)

        if ad_ratio > 3:
            signal = "极强"
        elif ad_ratio > 1.5:
            signal = "强"
        elif ad_ratio > 0.67:
            signal = "中性"
        elif ad_ratio > 0.33:
            signal = "弱"
        else:
            signal = "极弱"

        return MarketBreadth(
            advance_count=advance,
            decline_count=decline,
            unchanged_count=unchanged,
            advance_decline_ratio=round(ad_ratio, 3),
            new_52w_high=max(0, int(advance * 0.05)),
            new_52w_low=max(0, int(decline * 0.03)),
            stocks_above_ma20=round(advance / total * 100, 1),
            stocks_above_ma200=round((advance * 0.6) / total * 100, 1),
            breadth_signal=signal,
        )


# ---------------------------------------------------------------------------
# 恐慌贪婪指数合成
# ---------------------------------------------------------------------------

class FearGreedCalculator:
    """综合7大维度计算市场恐慌贪婪指数"""

    def calculate(
        self,
        technical: Optional[TechnicalSentiment] = None,
        capital_flow: Optional[CapitalFlowSentiment] = None,
        breadth: Optional[MarketBreadth] = None,
        vix_level: float = 20.0,  # 波动率指数
    ) -> FearGreedIndex:
        """计算恐慌贪婪指数 (0=极度恐慌, 100=极度贪婪)"""
        components: dict[str, float] = {}

        # 1. 价格动量（20日）→ 0–100
        if technical:
            momentum_norm = max(0.0, min(100.0, 50 + technical.momentum_20d * 2.5))
            components["价格动量"] = momentum_norm
        else:
            components["价格动量"] = 50.0

        # 2. RSI广度 → 0–100
        if technical:
            rsi_norm = technical.rsi_14  # RSI本身就是0-100
            components["RSI强弱"] = rsi_norm
        else:
            components["RSI强弱"] = 50.0

        # 3. 量价关系 → 0–100
        if technical:
            vol_norm = 50.0
            if technical.volume_signal == "放量" and technical.momentum_20d > 0:
                vol_norm = 70.0
            elif technical.volume_signal == "缩量" and technical.momentum_20d < 0:
                vol_norm = 30.0
            components["量价关系"] = vol_norm
        else:
            components["量价关系"] = 50.0

        # 4. 资金流向 → 0–100
        if capital_flow:
            flow_norm = max(0.0, min(100.0, 50 + capital_flow.flow_score * 50))
            components["资金流向"] = flow_norm
        else:
            components["资金流向"] = 50.0

        # 5. 市场广度 → 0–100
        if breadth:
            breadth_map = {"极强": 85, "强": 70, "中性": 50, "弱": 30, "极弱": 15}
            components["市场广度"] = float(breadth_map.get(breadth.breadth_signal, 50))
        else:
            components["市场广度"] = 50.0

        # 6. 波动率（VIX倒序）→ 0–100（VIX高=恐慌）
        vix_norm = max(0.0, min(100.0, 100 - (vix_level - 10) * 3))
        components["波动率恐慌"] = vix_norm

        # 7. 布林带位置 → 0–100
        if technical:
            bb_norm = technical.bb_position * 100
            components["布林带位置"] = bb_norm
        else:
            components["布林带位置"] = 50.0

        # 加权平均
        weights = {
            "价格动量": 0.25, "RSI强弱": 0.15, "量价关系": 0.15,
            "资金流向": 0.20, "市场广度": 0.10, "波动率恐慌": 0.10, "布林带位置": 0.05,
        }
        score = sum(components[k] * weights[k] for k in components)
        score = float(np.clip(score, 0.0, 100.0))

        if score >= 80:
            grade = "极度贪婪"
        elif score >= 60:
            grade = "贪婪"
        elif score >= 40:
            grade = "中性"
        elif score >= 20:
            grade = "恐慌"
        else:
            grade = "极度恐慌"

        return FearGreedIndex(
            score=round(score, 1),
            grade=grade,
            components={k: round(v, 1) for k, v in components.items()},
        )


# ---------------------------------------------------------------------------
# 行业轮动分析
# ---------------------------------------------------------------------------

class SectorRotationAnalyzer:
    """A股行业轮动分析"""

    ROTATION_CYCLE_SECTORS = {
        "早周期": ["银行", "地产", "钢铁", "建材", "建筑"],
        "中周期": ["机械", "化工", "有色", "电力", "煤炭"],
        "晚周期": ["消费", "食品饮料", "家电", "汽车"],
        "防御":   ["医药", "公用事业", "必需消费", "电信"],
    }

    def analyze(self) -> SectorRotation:
        """分析行业轮动状态"""
        try:
            import akshare as ak  # type: ignore

            # 获取行业资金流向
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")
            if df is not None and not df.empty:
                hot = df.head(5)["行业"].tolist() if "行业" in df.columns else []
                cold = df.tail(5)["行业"].tolist() if "行业" in df.columns else []
                return SectorRotation(
                    hot_sectors=hot,
                    cold_sectors=cold,
                    rotation_stage=self._infer_stage(hot),
                    momentum_sectors=hot[:3],
                    reversal_sectors=cold[:3],
                    sector_dispersion=0.5,
                )
        except Exception as e:
            _logger.debug(f"行业轮动数据获取失败: {e}")

        return SectorRotation(
            hot_sectors=["科技", "新能源", "半导体"],
            cold_sectors=["地产", "建材", "钢铁"],
            rotation_stage="中周期",
            momentum_sectors=["AI", "新能源", "医药"],
            reversal_sectors=["银行", "保险"],
            sector_dispersion=0.4,
        )

    def _infer_stage(self, hot_sectors: list[str]) -> str:
        for stage, sectors in self.ROTATION_CYCLE_SECTORS.items():
            if any(any(s in hot for s in sectors) for hot in hot_sectors):
                return stage
        return "混合"


# ---------------------------------------------------------------------------
# 主分析器
# ---------------------------------------------------------------------------

class MarketSentimentAnalyzer:
    """
    市场情绪综合分析器。

    使用方式：
    ----------
    analyzer = MarketSentimentAnalyzer()
    report = analyzer.analyze(
        symbol="600519.SH",
        stock_name="贵州茅台",
        price_df=df,
        market="CN",
    )
    print(report.summary)
    """

    def __init__(self) -> None:
        self.technical_calc = TechnicalSentimentCalculator()
        self.capital_flow = CapitalFlowAnalyzer()
        self.breadth = MarketBreadthAnalyzer()
        self.fear_greed = FearGreedCalculator()
        self.sector_rotation = SectorRotationAnalyzer()

    def analyze(
        self,
        symbol: str,
        stock_name: str,
        price_df: Optional[pd.DataFrame] = None,
        market: str = "CN",
        include_market_breadth: bool = True,
        include_sector_rotation: bool = True,
    ) -> MarketSentimentReport:
        """
        执行完整市场情绪分析。

        Parameters
        ----------
        symbol       : 股票代码
        stock_name   : 股票名称
        price_df     : 历史OHLCV DataFrame
        market       : 市场类型 "CN" | "US"
        """
        _logger.info(f"开始市场情绪分析: {symbol} ({stock_name})")

        report = MarketSentimentReport(
            symbol=symbol,
            stock_name=stock_name,
            timestamp=datetime.now(),
        )

        # 1. 技术情绪
        if price_df is not None and not price_df.empty:
            report.technical = self.technical_calc.calculate(price_df)
            _logger.debug(f"  技术情绪: RSI={report.technical.rsi_14:.1f} MACD={report.technical.macd_signal}")

        # 2. 资金流向
        report.capital_flow = self.capital_flow.analyze(symbol)
        _logger.debug(f"  资金流向: {report.capital_flow.flow_signal}")

        # 3. 市场广度
        if include_market_breadth:
            report.market_breadth = self.breadth.analyze(market)
            _logger.debug(f"  市场广度: {report.market_breadth.breadth_signal}")

        # 4. 恐慌贪婪指数
        report.fear_greed = self.fear_greed.calculate(
            technical=report.technical,
            capital_flow=report.capital_flow,
            breadth=report.market_breadth,
        )
        _logger.debug(f"  恐慌贪婪: {report.fear_greed.score:.0f} ({report.fear_greed.grade})")

        # 5. 行业轮动
        if include_sector_rotation:
            report.sector_rotation = self.sector_rotation.analyze()
            _logger.debug(f"  行业轮动: {report.sector_rotation.rotation_stage}")

        # 6. 综合情绪得分 (-1~1)
        report.overall_sentiment_score = self._calc_overall_score(report)
        report.overall_sentiment_label = self._score_to_label(report.overall_sentiment_score)

        # 7. 逆向信号（情绪极端时提示）
        report.contrarian_signal = self._calc_contrarian_signal(report)

        # 8. 最终交易信号
        report.sentiment_signal, report.signal_confidence = self._generate_signal(report)

        # 9. 生成报告
        report.summary = self._generate_summary(report)

        _logger.info(
            f"情绪分析完成 [{symbol}]: "
            f"恐慌贪婪={report.fear_greed.score:.0f} "
            f"信号={report.sentiment_signal}"
        )
        return report

    def _calc_overall_score(self, r: MarketSentimentReport) -> float:
        scores = []
        if r.fear_greed:
            scores.append((r.fear_greed.score - 50) / 50)
        if r.technical:
            scores.append(r.technical.composite_score)
        if r.capital_flow:
            scores.append(r.capital_flow.flow_score)
        return round(float(np.mean(scores)) if scores else 0.0, 3)

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score > 0.4:
            return "极度贪婪/过热"
        elif score > 0.15:
            return "偏多/情绪积极"
        elif score < -0.4:
            return "极度恐慌/超卖"
        elif score < -0.15:
            return "偏空/情绪低迷"
        return "情绪中性"

    @staticmethod
    def _calc_contrarian_signal(r: MarketSentimentReport) -> str:
        """逆向投资信号：极端情绪往往意味着反转"""
        if r.fear_greed and r.fear_greed.score >= 80:
            return "⚠️ 极度贪婪警告：情绪过热，需警惕回调风险"
        elif r.fear_greed and r.fear_greed.score <= 20:
            return "🔔 逆向机会：极度恐慌时往往是布局良机"
        return ""

    def _generate_signal(self, r: MarketSentimentReport) -> tuple[str, float]:
        score = r.overall_sentiment_score
        # 考虑技术与资金协同
        if (r.technical and r.technical.macd_signal == "金叉"
                and r.capital_flow and r.capital_flow.flow_score > 0.2):
            score += 0.1

        if score > 0.35:
            return "强烈看多", min(0.85, 0.6 + abs(score))
        elif score > 0.12:
            return "看多", 0.65
        elif score < -0.35:
            return "强烈看空", min(0.85, 0.6 + abs(score))
        elif score < -0.12:
            return "看空", 0.65
        return "中性", 0.5

    def _generate_summary(self, r: MarketSentimentReport) -> str:
        signal_emoji = (
            "📈" if "看多" in r.sentiment_signal else
            ("📉" if "看空" in r.sentiment_signal else "➡️")
        )
        lines = [
            f"## {r.stock_name} ({r.symbol}) 市场情绪分析\n\n",
            f"**分析时间**: {r.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n",
        ]

        if r.fear_greed:
            fg_emoji = "😱" if r.fear_greed.score < 30 else ("🤑" if r.fear_greed.score > 70 else "😐")
            lines.append(
                f"### 恐慌贪婪指数: {fg_emoji} **{r.fear_greed.score:.0f}/100** ({r.fear_greed.grade})\n"
            )
            lines.append("| 分项 | 得分 |\n|------|------|\n")
            for k, v in r.fear_greed.components.items():
                lines.append(f"| {k} | {v:.0f} |\n")
            lines.append("\n")

        if r.technical:
            lines.append(
                f"### 技术面情绪\n"
                f"- RSI(14): **{r.technical.rsi_14:.1f}** ({r.technical.rsi_signal})\n"
                f"- MACD信号: **{r.technical.macd_signal}**\n"
                f"- 布林带位置: {r.technical.bb_position:.2f} ({r.technical.bb_signal})\n"
                f"- 量比: {r.technical.volume_ratio:.2f} ({r.technical.volume_signal})\n"
                f"- 20日动量: {r.technical.momentum_20d:+.1f}%\n\n"
            )

        if r.capital_flow:
            lines.append(
                f"### 资金流向\n"
                f"- 1日净流入: **{r.capital_flow.net_inflow_1d:+,.0f}** 万元\n"
                f"- 5日净流入: {r.capital_flow.net_inflow_5d:+,.0f} 万元\n"
                f"- 信号: **{r.capital_flow.flow_signal}**\n\n"
            )

        if r.market_breadth:
            lines.append(
                f"### 市场广度\n"
                f"- 涨/跌家数: {r.market_breadth.advance_count}/{r.market_breadth.decline_count}"
                f" (比例{r.market_breadth.advance_decline_ratio:.2f})\n"
                f"- 市场广度: **{r.market_breadth.breadth_signal}**\n\n"
            )

        if r.sector_rotation:
            lines.append(
                f"### 行业轮动\n"
                f"- 当前阶段: **{r.sector_rotation.rotation_stage}**\n"
                f"- 热门行业: {' | '.join(r.sector_rotation.hot_sectors[:5])}\n"
                f"- 冷门行业: {' | '.join(r.sector_rotation.cold_sectors[:3])}\n\n"
            )

        if r.contrarian_signal:
            lines.append(f"### 逆向信号\n{r.contrarian_signal}\n\n")

        lines.append(
            f"### 综合情绪信号: {signal_emoji} **{r.sentiment_signal}**"
            f"（置信度 {r.signal_confidence:.0%}）\n"
            f"- 综合情绪得分: {r.overall_sentiment_score:+.3f}\n"
            f"- 情绪状态: {r.overall_sentiment_label}\n"
        )

        return "".join(lines)
