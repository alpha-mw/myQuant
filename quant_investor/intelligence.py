"""
Intelligence Layer (Layer 8 & 9)
===================================
智能分析层 — 整合 Kronos + 财务分析 + 新闻分析 + 市场情绪

架构位置：
  Layer 8: Kronos 基础模型预测层
  Layer 9: 多维智能融合层（财务 + 新闻 + 情绪 → LLM综合）

核心创新：
  1. Kronos 金融基础模型提供高质量K线预测信号
  2. 财务分析（Piotroski/Beneish/Altman/DCF）评估基本面健康度
  3. 新闻分析 + LLM情感提取市场叙事信号
  4. 市场情绪（恐慌贪婪指数 + 技术情绪 + 资金流向）测量群体心理
  5. 四维信号加权融合 → 最终投资建议
  6. LLM深度总结（将量化结论转为人类可读的投资报告）

输入：Layer 7（Multi-LLM决策）的输出
输出：增强的九维分析报告

信号权重（可配置）：
  Kronos预测     : 25%
  财务分析       : 25%
  新闻情感       : 20%
  市场情绪       : 20%
  量化基础层(V7) : 10%
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from quant_investor.kronos_predictor import KronosIntegrator, KronosPortfolioSignal
from quant_investor.financial_analysis import FinancialAnalyzer, FinancialHealthReport
from quant_investor.news_analysis import NewsAnalyzer, NewsAnalysisResult
from quant_investor.sentiment_analysis import MarketSentimentAnalyzer, MarketSentimentReport
from quant_investor.logger import get_logger

_logger = get_logger("IntelligenceLayer")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceSignal:
    """单标的智能层信号"""
    symbol: str
    stock_name: str

    # 各层信号
    kronos_signal: str = "中性"           # Kronos预测信号
    kronos_score: float = 0.0             # -1~1
    kronos_pred_pct: float = 0.0          # 预测涨跌幅

    financial_grade: str = "N/A"          # "A+" | "A" | "B" | "C" | "D"
    financial_score: float = 0.5          # 0~1
    financial_summary: str = ""

    news_signal: str = "中性"             # 新闻情感信号
    news_score: float = 0.0               # -1~1
    news_sentiment: float = 0.0           # 原始情感得分

    sentiment_signal: str = "中性"        # 市场情绪信号
    sentiment_score: float = 0.0          # -1~1
    fear_greed_score: float = 50.0        # 0~100

    # 加权融合
    combined_score: float = 0.0           # -1~1
    combined_signal: str = "中性"
    signal_confidence: float = 0.5

    # 风险指标
    key_risks: list[str] = field(default_factory=list)
    key_opportunities: list[str] = field(default_factory=list)

    # LLM综合分析
    llm_synthesis: str = ""


@dataclass
class IntelligenceLayerResult:
    """智能层完整输出"""
    signals: dict[str, IntelligenceSignal] = field(default_factory=dict)
    kronos_portfolio_signal: Optional[KronosPortfolioSignal] = None
    market_overview: str = ""            # 整体市场情绪概述
    top_recommendations: list[str] = field(default_factory=list)  # 强推标的
    avoid_list: list[str] = field(default_factory=list)            # 规避标的
    comprehensive_report: str = ""
    processing_time: float = 0.0


# ---------------------------------------------------------------------------
# 智能层主引擎
# ---------------------------------------------------------------------------

class IntelligenceLayerEngine:
    """
    V9 智能层引擎。

    使用方式：
    ----------
    engine = IntelligenceLayerEngine(
        kronos_model="kronos-base",
        weights={"kronos": 0.25, "financial": 0.25, "news": 0.20, "sentiment": 0.20, "quant_base": 0.10},
    )
    result = engine.analyze(
        stock_pool=["600519.SH", "000858.SZ"],
        price_data_dict={"600519.SH": df_maotai},
        financial_data_dict={"600519.SH": df_fin},
        stock_names={"600519.SH": "贵州茅台"},
        quant_scores={"600519.SH": 0.3},  # 来自V7的量化得分
    )
    """

    DEFAULT_WEIGHTS = {
        "kronos":     0.25,
        "financial":  0.25,
        "news":       0.20,
        "sentiment":  0.20,
        "quant_base": 0.10,
    }

    def __init__(
        self,
        kronos_model: str = "kronos-base",
        weights: Optional[dict[str, float]] = None,
        pred_len: int = 20,
        news_days: int = 7,
        enable_llm_synthesis: bool = True,
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.pred_len = pred_len
        self.news_days = news_days
        self.enable_llm_synthesis = enable_llm_synthesis

        self.kronos = KronosIntegrator(model_name=kronos_model)
        self.financial_analyzer = FinancialAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.sentiment_analyzer = MarketSentimentAnalyzer()

        _logger.info(
            f"IntelligenceLayer初始化: Kronos={kronos_model}, "
            f"权重={self.weights}"
        )

    # ------------------------------------------------------------------
    # 主分析入口
    # ------------------------------------------------------------------

    def analyze(
        self,
        stock_pool: list[str],
        price_data_dict: dict[str, pd.DataFrame],
        financial_data_dict: Optional[dict[str, pd.DataFrame]] = None,
        stock_names: Optional[dict[str, str]] = None,
        quant_scores: Optional[dict[str, float]] = None,
        market: str = "CN",
    ) -> IntelligenceLayerResult:
        """
        执行完整智能层分析。

        Parameters
        ----------
        stock_pool       : 待分析股票列表
        price_data_dict  : {symbol: ohlcv_df} 历史价格数据
        financial_data_dict : {symbol: financial_df} 财务数据（可选）
        stock_names      : {symbol: name} 股票名称（可选）
        quant_scores     : {symbol: score} 来自V7层的量化得分 -1~1（可选）
        market           : 市场类型
        """
        t0 = time.time()
        _logger.info(f"=== 智能层分析开始 ({len(stock_pool)} 只股票) ===")

        result = IntelligenceLayerResult()
        financial_data_dict = financial_data_dict or {}
        stock_names = stock_names or {s: s for s in stock_pool}
        quant_scores = quant_scores or {}

        # Step 1: Kronos 批量预测（Layer 8）
        _logger.info("Step 1/4: Kronos基础模型预测...")
        kronos_result = self.kronos.analyze_portfolio(
            stock_data_dict={s: price_data_dict[s] for s in stock_pool if s in price_data_dict},
            pred_len=self.pred_len,
        )
        result.kronos_portfolio_signal = kronos_result

        # Step 2: 逐只分析（财务 + 新闻 + 情绪）
        _logger.info("Step 2-4: 财务/新闻/情绪逐只分析...")
        for i, symbol in enumerate(stock_pool):
            name = stock_names.get(symbol, symbol)
            _logger.info(f"  [{i+1}/{len(stock_pool)}] 分析 {symbol} ({name})")
            signal = self._analyze_single(
                symbol=symbol,
                stock_name=name,
                price_df=price_data_dict.get(symbol),
                financial_df=financial_data_dict.get(symbol),
                kronos_result=kronos_result,
                quant_score=quant_scores.get(symbol, 0.0),
                market=market,
            )
            result.signals[symbol] = signal

        # Step 3: 组合级汇总
        result.top_recommendations = self._get_top_picks(result.signals, top_n=5)
        result.avoid_list = self._get_avoid_list(result.signals, top_n=5)
        result.market_overview = self._generate_market_overview(result, market)

        # Step 4: 生成综合报告
        result.comprehensive_report = self._generate_report(result, stock_names)

        result.processing_time = round(time.time() - t0, 2)
        _logger.info(f"=== 智能层分析完成 ({result.processing_time}s) ===")

        return result

    # ------------------------------------------------------------------
    # 单标的分析
    # ------------------------------------------------------------------

    def _analyze_single(
        self,
        symbol: str,
        stock_name: str,
        price_df: Optional[pd.DataFrame],
        financial_df: Optional[pd.DataFrame],
        kronos_result: KronosPortfolioSignal,
        quant_score: float,
        market: str,
    ) -> IntelligenceSignal:
        sig = IntelligenceSignal(symbol=symbol, stock_name=stock_name)

        # --- Kronos 信号 ---
        kronos_forecast = kronos_result.forecasts.get(symbol)
        if kronos_forecast and not kronos_forecast.error:
            sig.kronos_signal = kronos_forecast.direction_signal
            sig.kronos_score = (kronos_forecast.pred_close_pct / 20.0)  # 归一化到-1~1
            sig.kronos_score = float(np.clip(sig.kronos_score, -1.0, 1.0))
            sig.kronos_pred_pct = kronos_forecast.pred_close_pct

        # --- 财务分析 ---
        if financial_df is not None and not financial_df.empty:
            fin_report: FinancialHealthReport = self.financial_analyzer.full_analysis(
                symbol=symbol,
                stock_name=stock_name,
                financial_df=financial_df,
            )
            sig.financial_grade = fin_report.overall_grade
            sig.financial_score = fin_report.overall_score / 100.0
            sig.financial_summary = fin_report.summary[:500]
            sig.key_risks.extend(fin_report.key_risks[:2])
            sig.key_opportunities.extend(fin_report.key_strengths[:2])
        else:
            sig.financial_score = 0.5  # 中性

        # --- 新闻分析 ---
        news_result: NewsAnalysisResult = self.news_analyzer.analyze(
            symbol=symbol,
            stock_name=stock_name,
            days=self.news_days,
        )
        sig.news_signal = news_result.news_signal
        sig.news_score = news_result.avg_sentiment_score
        sig.news_sentiment = news_result.avg_sentiment_score

        # --- 市场情绪 ---
        sentiment_result: MarketSentimentReport = self.sentiment_analyzer.analyze(
            symbol=symbol,
            stock_name=stock_name,
            price_df=price_df,
            market=market,
        )
        sig.sentiment_signal = sentiment_result.sentiment_signal
        sig.sentiment_score = sentiment_result.overall_sentiment_score
        sig.fear_greed_score = sentiment_result.fear_greed.score if sentiment_result.fear_greed else 50.0

        # --- 加权融合 ---
        sig.combined_score, sig.combined_signal, sig.signal_confidence = self._combine_signals(
            sig, quant_score
        )

        # --- LLM综合分析 ---
        if self.enable_llm_synthesis:
            sig.llm_synthesis = self._llm_synthesize(sig, news_result, sentiment_result)

        return sig

    def _combine_signals(
        self,
        sig: IntelligenceSignal,
        quant_score: float,
    ) -> tuple[float, str, float]:
        """加权融合四维信号"""
        w = self.weights
        financial_signal_score = (sig.financial_score - 0.5) * 2  # 0~1 → -1~1

        weighted_score = (
            w["kronos"] * sig.kronos_score +
            w["financial"] * financial_signal_score +
            w["news"] * sig.news_score +
            w["sentiment"] * sig.sentiment_score +
            w["quant_base"] * quant_score
        )
        weighted_score = float(np.clip(weighted_score, -1.0, 1.0))

        # 信号确定
        if weighted_score > 0.35:
            label = "强烈买入"
        elif weighted_score > 0.15:
            label = "买入"
        elif weighted_score < -0.35:
            label = "强烈卖出"
        elif weighted_score < -0.15:
            label = "卖出"
        else:
            label = "持有"

        # 置信度：各信号方向一致性越高，置信度越高
        signals_arr = [sig.kronos_score, financial_signal_score, sig.news_score, sig.sentiment_score]
        signs = [np.sign(s) for s in signals_arr if abs(s) > 0.05]
        if len(signs) > 1:
            consistency = sum(s == np.sign(weighted_score) for s in signs) / len(signs)
            confidence = 0.4 + 0.5 * consistency
        else:
            confidence = 0.5

        return round(weighted_score, 3), label, round(float(confidence), 3)

    def _llm_synthesize(
        self,
        sig: IntelligenceSignal,
        news: NewsAnalysisResult,
        sentiment: MarketSentimentReport,
    ) -> str:
        """使用LLM生成人类可读的综合分析"""
        from quant_investor.llm_gateway import LLMClient as GatewayLLMClient, has_any_provider, _run_sync

        if not has_any_provider():
            return self._fallback_synthesis(sig)

        context = f"""你是一位专业的量化投资分析师。请基于以下多维度分析数据，为{sig.stock_name}({sig.symbol})生成简洁的投资分析摘要（200字以内，中文）：

**Kronos基础模型预测**: {sig.kronos_signal}，预测{self.pred_len}日涨跌幅 {sig.kronos_pred_pct:+.1f}%
**财务健康评级**: {sig.financial_grade}（得分 {sig.financial_score*100:.0f}/100）
**新闻情感信号**: {sig.news_signal}（情感得分 {sig.news_sentiment:+.3f}，{news.total_news_count}条新闻）
**市场情绪信号**: {sig.sentiment_signal}（恐慌贪婪指数 {sig.fear_greed_score:.0f}/100）
**综合建议**: {sig.combined_signal}（置信度 {sig.signal_confidence:.0%}）

关键风险: {'; '.join(sig.key_risks[:2]) if sig.key_risks else '无'}
关键机遇: {'; '.join(sig.key_opportunities[:2]) if sig.key_opportunities else '无'}

请用专业简洁的语言综合以上信息，给出核心投资逻辑和主要关注点。"""

        try:
            client = GatewayLLMClient(timeout=15.0, max_retries=1)
            return _run_sync(client.complete_text(
                messages=[{"role": "user", "content": context}],
                model="moonshot-v1-8k",
                max_tokens=400,
                stage="intelligence_summary",
                actor_name=sig.symbol,
            ))
        except Exception as e:
            _logger.debug(f"LLM综合分析失败: {e}")
            return self._fallback_synthesis(sig)

    @staticmethod
    def _fallback_synthesis(sig: IntelligenceSignal) -> str:
        """当LLM不可用时的规则生成摘要"""
        parts = [f"**{sig.stock_name}({sig.symbol}) 智能综合分析**\n"]

        parts.append(
            f"综合 Kronos基础模型、财务分析、新闻情感和市场情绪四个维度，"
            f"当前综合得分 **{sig.combined_score:+.3f}**，建议 **{sig.combined_signal}**"
            f"（置信度 {sig.signal_confidence:.0%}）。\n"
        )

        if sig.kronos_pred_pct != 0:
            parts.append(
                f"Kronos模型预测未来20个交易日涨跌幅约 **{sig.kronos_pred_pct:+.1f}%**。"
            )

        if sig.financial_grade in ("A+", "A"):
            parts.append(f"财务状况良好（{sig.financial_grade}级），基本面支撑有力。")
        elif sig.financial_grade in ("C", "D"):
            parts.append(f"财务状况偏弱（{sig.financial_grade}级），需关注基本面风险。")

        if sig.key_opportunities:
            parts.append(f"\n**机遇**: {sig.key_opportunities[0]}")
        if sig.key_risks:
            parts.append(f"\n**风险**: {sig.key_risks[0]}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 组合汇总
    # ------------------------------------------------------------------

    def _get_top_picks(self, signals: dict[str, IntelligenceSignal], top_n: int) -> list[str]:
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: (x[1].combined_score, x[1].signal_confidence),
            reverse=True,
        )
        return [s for s, sig in sorted_signals[:top_n] if sig.combined_signal in ("强烈买入", "买入")]

    def _get_avoid_list(self, signals: dict[str, IntelligenceSignal], top_n: int) -> list[str]:
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1].combined_score,
        )
        return [s for s, sig in sorted_signals[:top_n] if sig.combined_signal in ("强烈卖出", "卖出")]

    def _generate_market_overview(self, result: IntelligenceLayerResult, market: str) -> str:
        signals = list(result.signals.values())
        if not signals:
            return "暂无市场概览数据"

        bullish = sum(1 for s in signals if s.combined_signal in ("强烈买入", "买入"))
        bearish = sum(1 for s in signals if s.combined_signal in ("强烈卖出", "卖出"))
        neutral = len(signals) - bullish - bearish

        avg_fear_greed = np.mean([s.fear_greed_score for s in signals])
        avg_sentiment = np.mean([s.news_sentiment for s in signals])

        overview = (
            f"**市场整体态势** ({market}市场)\n"
            f"- 看多/中性/看空: {bullish}/{neutral}/{bearish}\n"
            f"- 平均恐慌贪婪指数: {avg_fear_greed:.0f}/100\n"
            f"- 平均新闻情感: {avg_sentiment:+.3f}\n"
        )

        if result.kronos_portfolio_signal:
            overview += (
                f"- Kronos整体看多比例: {result.kronos_portfolio_signal.ensemble_bullish_pct:.0%}\n"
            )

        return overview

    def _generate_report(
        self,
        result: IntelligenceLayerResult,
        stock_names: dict[str, str],
    ) -> str:
        lines = [
            "# 智能层综合分析报告（Layer 8+9）\n\n",
            f"**分析模块**: Kronos基础模型 + 财务分析 + 新闻分析 + 市场情绪\n",
            f"**信号权重**: Kronos {self.weights['kronos']:.0%} | "
            f"财务 {self.weights['financial']:.0%} | "
            f"新闻 {self.weights['news']:.0%} | "
            f"情绪 {self.weights['sentiment']:.0%}\n\n",
            "---\n\n",
            f"## 市场概览\n{result.market_overview}\n\n",
        ]

        if result.top_recommendations:
            lines.append(
                f"## 重点推荐\n"
                + "、".join(f"{s}({stock_names.get(s, s)})" for s in result.top_recommendations)
                + "\n\n"
            )

        if result.avoid_list:
            lines.append(
                f"## 规避标的\n"
                + "、".join(f"{s}({stock_names.get(s, s)})" for s in result.avoid_list)
                + "\n\n"
            )

        # Kronos组合信号
        if result.kronos_portfolio_signal and result.kronos_portfolio_signal.summary:
            lines.append(f"---\n\n{result.kronos_portfolio_signal.summary}\n\n")

        # 逐股详细分析
        lines.append("---\n\n## 逐股详细分析\n\n")
        sorted_signals = sorted(
            result.signals.items(),
            key=lambda x: x[1].combined_score,
            reverse=True,
        )
        for symbol, sig in sorted_signals:
            signal_emoji = (
                "🟢" if "买入" in sig.combined_signal else
                ("🔴" if "卖出" in sig.combined_signal else "🟡")
            )
            lines.append(
                f"### {signal_emoji} {sig.stock_name} ({symbol})\n\n"
                f"| 维度 | 信号 | 得分 |\n"
                f"|------|------|------|\n"
                f"| Kronos预测 | {sig.kronos_signal} | {sig.kronos_pred_pct:+.1f}% |\n"
                f"| 财务健康 | {sig.financial_grade} | {sig.financial_score*100:.0f}/100 |\n"
                f"| 新闻情感 | {sig.news_signal} | {sig.news_sentiment:+.3f} |\n"
                f"| 市场情绪 | {sig.sentiment_signal} | {sig.fear_greed_score:.0f}/100 |\n"
                f"| **综合信号** | **{sig.combined_signal}** | **{sig.combined_score:+.3f}** |\n\n"
            )
            if sig.llm_synthesis:
                lines.append(f"{sig.llm_synthesis}\n\n")
            if sig.key_opportunities:
                lines.append(
                    "**机遇**: " + " | ".join(sig.key_opportunities[:2]) + "\n\n"
                )
            if sig.key_risks:
                lines.append(
                    "**风险**: " + " | ".join(sig.key_risks[:2]) + "\n\n"
                )

        return "".join(lines)
