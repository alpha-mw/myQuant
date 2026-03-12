"""
Quant Investor V9.0 — 九层架构
=================================
在 V8 七层基础上新增：
  Layer 8: Kronos 基础模型预测层（金融K线基础模型）
  Layer 9: 多维智能融合层（财务分析 + 新闻分析 + 市场情绪 + LLM综合）

新增模块：
  - kronos_predictor.py     → Layer 8: Kronos金融基础模型预测
  - financial_analysis.py   → 财务健康评估（Piotroski/Beneish/Altman/DCF/DuPont）
  - news_analysis.py        → 新闻&替代数据分析（情感提取/事件检测/主题建模）
  - sentiment_analysis.py   → 市场情绪分析（恐慌贪婪指数/资金流/市场广度）
  - intelligence_layer.py   → 智能层编排器（Layer 8+9统一入口）

架构演进：
  V8: 七层（量化6层 + Multi-LLM裁判） → 结构化报告
  V9: 九层（V8七层 + Kronos预测 + 四维智能融合） → 全面智能报告

运行方式：
  python quant_investor_v9.py --stocks 000001.SZ 600519.SH --capital 1000000

Kronos集成说明：
  - 支持 Kronos-mini/small/base（HuggingFace Hub）
  - 在 Kronos 原生库未安装时自动降级至统计替代模式
  - 安装Kronos：git clone https://github.com/shiyu-coder/Kronos && pip install -e .
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from logger import get_logger

_logger = get_logger("QuantInvestorV9")


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class V9PipelineResult:
    """V9 完整流水线结果"""
    # V8 输出（继承）
    v8_result: Any = None

    # V9 新增
    intelligence_result: Any = None      # IntelligenceLayerResult
    final_report: str = ""               # 完整 Markdown 报告（V9版本）

    # 时间统计
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# V9 主引擎
# ---------------------------------------------------------------------------

class QuantInvestorV9:
    """
    V9 九层架构主引擎。

    Parameters
    ----------
    stock_pool      : 待分析股票代码列表
    market          : "CN" | "US"
    lookback_years  : 数据回溯年数
    total_capital   : 模拟资金量
    risk_level      : "保守" | "中等" | "积极"
    kronos_model    : Kronos模型规格 "kronos-mini" | "kronos-small" | "kronos-base"
    pred_len        : Kronos预测期数（交易日，默认20≈1个月）
    enable_macro    : 是否启用宏观分析层
    enable_backtest : 是否运行 Walk-Forward 回测
    enable_alpha_mining : 是否运行因子挖掘
    enable_financial_analysis : 是否启用财务分析
    enable_news_analysis : 是否启用新闻分析
    enable_sentiment_analysis : 是否启用情绪分析
    intelligence_weights : 自定义智能层信号权重
    verbose         : 是否打印详细日志
    """

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        kronos_model: str = "kronos-small",
        pred_len: int = 20,
        enable_macro: bool = True,
        enable_backtest: bool = False,
        enable_alpha_mining: bool = False,
        enable_financial_analysis: bool = True,
        enable_news_analysis: bool = True,
        enable_sentiment_analysis: bool = True,
        intelligence_weights: Optional[dict[str, float]] = None,
        verbose: bool = True,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.kronos_model = kronos_model
        self.pred_len = pred_len
        self.enable_macro = enable_macro
        self.enable_backtest = enable_backtest
        self.enable_alpha_mining = enable_alpha_mining
        self.enable_financial_analysis = enable_financial_analysis
        self.enable_news_analysis = enable_news_analysis
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.intelligence_weights = intelligence_weights
        self.verbose = verbose
        self.result = V9PipelineResult()

    # ------------------------------------------------------------------
    # 主流水线
    # ------------------------------------------------------------------

    def run(self) -> V9PipelineResult:
        """执行完整九层分析流水线"""
        t_start = time.time()
        self._log("=" * 70)
        self._log("🚀 Quant Investor V9.0 启动 — 九层智能投资框架")
        self._log(f"   Kronos: {self.kronos_model} | 预测期: {self.pred_len}天")
        self._log(f"   智能模块: 财务={'✅' if self.enable_financial_analysis else '❌'} | "
                  f"新闻={'✅' if self.enable_news_analysis else '❌'} | "
                  f"情绪={'✅' if self.enable_sentiment_analysis else '❌'}")
        self._log(f"   标的: {self.stock_pool}")
        self._log(f"   市场: {self.market}  资金: ¥{self.total_capital:,.0f}")
        self._log("=" * 70)

        # ------ Layers 1-7: 复用 V8 ------
        v8_result = self._run_v8_pipeline()
        self.result.v8_result = v8_result

        # ------ Layer 8+9: 智能层 ------
        self._run_intelligence_layers(v8_result)

        # ------ 生成V9综合报告 ------
        self._generate_v9_report()

        self.result.total_time = round(time.time() - t_start, 2)
        self._log(f"\n✅ V9 分析完成，总耗时 {self.result.total_time}s")
        self._log(f"   报告字数: {len(self.result.final_report)} 字符")
        return self.result

    # ------------------------------------------------------------------
    # Layer 1-7: V8 流水线
    # ------------------------------------------------------------------

    def _run_v8_pipeline(self) -> Any:
        t0 = time.time()
        self._log("\n▶ Layers 1-7: 运行 V8 七层流水线...")

        try:
            from quant_investor_v8 import QuantInvestorV8
            v8 = QuantInvestorV8(
                stock_pool=self.stock_pool,
                market=self.market,
                lookback_years=self.lookback_years,
                total_capital=self.total_capital,
                risk_level=self.risk_level,
                enable_macro=self.enable_macro,
                enable_backtest=self.enable_backtest,
                enable_alpha_mining=self.enable_alpha_mining,
                verbose=self.verbose,
            )
            v8_result = v8.run()
            self.result.layer_timings["v8_pipeline"] = round(time.time() - t0, 2)
            self._log(f"  V8 完成 ({self.result.layer_timings['v8_pipeline']}s)")
            return v8_result
        except ImportError:
            self._log("  ⚠️ quant_investor_v8 不可用，跳过V8层")
            self.result.layer_timings["v8_pipeline"] = round(time.time() - t0, 2)
            return None

    # ------------------------------------------------------------------
    # Layer 8+9: 智能层
    # ------------------------------------------------------------------

    def _run_intelligence_layers(self, v8_result: Any) -> None:
        t0 = time.time()
        self._log("\n▶ Layers 8+9: 智能层分析 (Kronos + 财务 + 新闻 + 情绪)...")

        try:
            from intelligence_layer import IntelligenceLayerEngine

            # 构建智能层权重
            weights = self.intelligence_weights or {
                "kronos": 0.25, "financial": 0.25,
                "news": 0.20, "sentiment": 0.20, "quant_base": 0.10,
            }
            if not self.enable_news_analysis:
                weights["news"] = 0.0
                weights["financial"] = min(weights["financial"] + 0.10, 0.35)
                weights["sentiment"] = min(weights["sentiment"] + 0.10, 0.30)
            if not self.enable_financial_analysis:
                weights["financial"] = 0.0
                weights["kronos"] = min(weights["kronos"] + 0.15, 0.40)
            if not self.enable_sentiment_analysis:
                weights["sentiment"] = 0.0
                weights["kronos"] = min(weights["kronos"] + 0.10, 0.40)

            engine = IntelligenceLayerEngine(
                kronos_model=self.kronos_model,
                weights=weights,
                pred_len=self.pred_len,
                enable_llm_synthesis=True,
            )

            # 准备数据
            price_data = self._collect_price_data(v8_result)
            financial_data = self._collect_financial_data(v8_result) if self.enable_financial_analysis else {}
            stock_names = self._collect_stock_names(v8_result)
            quant_scores = self._extract_v8_quant_scores(v8_result)

            self.result.intelligence_result = engine.analyze(
                stock_pool=self.stock_pool,
                price_data_dict=price_data,
                financial_data_dict=financial_data,
                stock_names=stock_names,
                quant_scores=quant_scores,
                market=self.market,
            )

        except ImportError as e:
            self._log(f"  ⚠️ 智能层模块导入失败: {e}")
        except Exception as e:
            _logger.exception(f"智能层执行异常: {e}")

        self.result.layer_timings["intelligence_layers"] = round(time.time() - t0, 2)
        self._log(f"  智能层完成 ({self.result.layer_timings['intelligence_layers']}s)")

    # ------------------------------------------------------------------
    # 数据收集工具
    # ------------------------------------------------------------------

    def _collect_price_data(self, v8_result: Any) -> dict[str, pd.DataFrame]:
        """从V8结果或独立数据源收集价格数据"""
        price_data: dict[str, pd.DataFrame] = {}

        # 尝试从V8结果提取
        if v8_result is not None:
            raw_data = getattr(v8_result, "raw_data", {})
            for symbol in self.stock_pool:
                df = raw_data.get(symbol)
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    price_data[symbol] = df
                    continue

        # 对缺失的标的，尝试从数据层获取
        missing = [s for s in self.stock_pool if s not in price_data]
        if missing:
            price_data.update(self._fetch_price_data_fallback(missing))

        return price_data

    def _fetch_price_data_fallback(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """从AKShare/yfinance获取价格数据（降级方案）"""
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                import akshare as ak  # type: ignore
                code = symbol.split(".")[0]
                df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                         adjust="qfq",
                                         start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y%m%d"))
                if df is not None and not df.empty:
                    # 标准化列名
                    col_map = {"开盘": "open", "最高": "high", "最低": "low",
                               "收盘": "close", "成交量": "volume", "成交额": "amount"}
                    df = df.rename(columns=col_map)
                    result[symbol] = df
                    self._log(f"  获取 {symbol} 价格数据: {len(df)} 行")
            except Exception as e:
                self._log(f"  ⚠️ 无法获取 {symbol} 价格数据: {e}")
                # 生成合成价格数据（用于测试）
                result[symbol] = self._generate_synthetic_price(symbol)
        return result

    @staticmethod
    def _generate_synthetic_price(symbol: str) -> pd.DataFrame:
        """生成合成价格序列（当数据源不可用时）"""
        import numpy as np
        n = 250
        rng = np.random.default_rng(seed=hash(symbol) % 2**32)
        returns = rng.normal(0.0003, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))
        noise = rng.uniform(0.98, 1.02, n)
        return pd.DataFrame({
            "open": close * noise,
            "high": close * rng.uniform(1.001, 1.025, n),
            "low":  close * rng.uniform(0.975, 0.999, n),
            "close": close,
            "volume": rng.integers(1_000_000, 50_000_000, n),
        })

    def _collect_financial_data(self, v8_result: Any) -> dict[str, pd.DataFrame]:
        """从V8结果或数据层收集财务数据"""
        financial_data: dict[str, pd.DataFrame] = {}

        for symbol in self.stock_pool:
            try:
                import akshare as ak  # type: ignore
                code = symbol.split(".")[0]
                # 获取财务指标
                df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
                if df is not None and not df.empty:
                    financial_data[symbol] = df
                    self._log(f"  获取 {symbol} 财务数据: {len(df)} 期")
            except Exception as e:
                self._log(f"  ⚠️ 无法获取 {symbol} 财务数据: {e}")

        return financial_data

    def _collect_stock_names(self, v8_result: Any) -> dict[str, str]:
        """收集股票名称"""
        names: dict[str, str] = {}
        # 尝试从V8结果提取
        if v8_result is not None:
            raw_data = getattr(v8_result, "raw_data", {})
            for symbol in self.stock_pool:
                name = raw_data.get(f"{symbol}_name") or symbol
                names[symbol] = str(name)

        # 补充缺失名称
        missing = [s for s in self.stock_pool if s not in names or names[s] == s]
        if missing:
            names.update(self._fetch_stock_names(missing))

        return names

    def _fetch_stock_names(self, symbols: list[str]) -> dict[str, str]:
        """查询股票名称"""
        names: dict[str, str] = {}
        try:
            import akshare as ak  # type: ignore
            stock_list = ak.stock_info_a_code_name()
            if stock_list is not None and not stock_list.empty:
                code_name_map = dict(zip(
                    stock_list.get("code", pd.Series()).values,
                    stock_list.get("name", pd.Series()).values,
                ))
                for symbol in symbols:
                    code = symbol.split(".")[0]
                    names[symbol] = code_name_map.get(code, symbol)
        except Exception:
            for s in symbols:
                names[s] = s
        return names

    def _extract_v8_quant_scores(self, v8_result: Any) -> dict[str, float]:
        """从V8的LLM集成结果提取量化得分"""
        scores: dict[str, float] = {}
        if v8_result is None:
            return scores

        llm_results = getattr(v8_result, "llm_ensemble_results", {})
        for symbol, consensus in llm_results.items():
            if consensus is not None:
                combined_score = getattr(consensus, "final_combined_score", 0.0)
                scores[symbol] = float(combined_score)
        return scores

    # ------------------------------------------------------------------
    # V9 综合报告生成
    # ------------------------------------------------------------------

    def _generate_v9_report(self) -> None:
        """合并V8报告和V9智能层报告"""
        lines = [
            "# Quant Investor V9.0 全面智能分析报告\n\n",
            f"**分析日期**: {time.strftime('%Y-%m-%d %H:%M')}\n",
            f"**标的池**: {', '.join(self.stock_pool)}\n",
            f"**市场**: {self.market}  **模拟资金**: ¥{self.total_capital:,.0f}\n",
            f"**Kronos模型**: {self.kronos_model}  **预测期**: {self.pred_len}个交易日\n\n",
            "---\n\n",
        ]

        # V9 智能层报告（最重要，放前面）
        if self.result.intelligence_result:
            lines.append(self.result.intelligence_result.comprehensive_report)
            lines.append("\n\n---\n\n")

        # V8 报告（量化基础层）
        if self.result.v8_result:
            v8_report = getattr(self.result.v8_result, "final_report", "")
            if v8_report:
                lines.append("# V8 量化基础层分析（Layers 1-7）\n\n")
                lines.append(v8_report)

        # 执行时间统计
        lines.append(f"\n\n---\n\n## 性能统计\n\n")
        lines.append("| 层级 | 耗时 |\n|------|------|\n")
        for layer, t in self.result.layer_timings.items():
            lines.append(f"| {layer} | {t:.2f}s |\n")
        lines.append(f"| **总计** | **{self.result.total_time}s** |\n")

        self.result.final_report = "".join(lines)
        self._log(f"\n📄 V9 报告生成完成 ({len(self.result.final_report)} 字符)")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            _logger.info(msg)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quant Investor V9.0 — 九层智能投资框架 (含Kronos基础模型)"
    )
    parser.add_argument(
        "--stocks", nargs="+",
        default=["000001.SZ", "600519.SH"],
        help="股票代码列表（如 000001.SZ 600519.SH）",
    )
    parser.add_argument(
        "--market", default="CN", choices=["CN", "US"],
        help="市场类型（默认CN）",
    )
    parser.add_argument(
        "--capital", type=float, default=1_000_000.0,
        help="模拟资金（默认100万）",
    )
    parser.add_argument(
        "--risk-level", default="中等",
        choices=["保守", "中等", "积极"],
        help="风险偏好（默认中等）",
    )
    parser.add_argument(
        "--kronos-model", default="kronos-small",
        choices=["kronos-mini", "kronos-small", "kronos-base"],
        help="Kronos模型规格（默认kronos-small）",
    )
    parser.add_argument(
        "--pred-len", type=int, default=20,
        help="Kronos预测期数（交易日，默认20）",
    )
    parser.add_argument(
        "--no-macro", action="store_true",
        help="禁用宏观分析层（加快速度）",
    )
    parser.add_argument(
        "--with-backtest", action="store_true",
        help="启用Walk-Forward回测（最慢）",
    )
    parser.add_argument(
        "--no-news", action="store_true",
        help="禁用新闻分析",
    )
    parser.add_argument(
        "--no-sentiment", action="store_true",
        help="禁用情绪分析",
    )
    parser.add_argument(
        "--output", default="",
        help="报告输出路径（默认打印到控制台）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    analyzer = QuantInvestorV9(
        stock_pool=args.stocks,
        market=args.market,
        total_capital=args.capital,
        risk_level=args.risk_level,
        kronos_model=args.kronos_model,
        pred_len=args.pred_len,
        enable_macro=not args.no_macro,
        enable_backtest=args.with_backtest,
        enable_news_analysis=not args.no_news,
        enable_sentiment_analysis=not args.no_sentiment,
        verbose=True,
    )

    result = analyzer.run()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.final_report)
        print(f"\n✅ 报告已保存至: {args.output}")
    else:
        print("\n" + "=" * 70)
        print(result.final_report[:5000] + (
            "\n\n...[报告已截断，使用 --output 保存完整报告]..." if len(result.final_report) > 5000 else ""
        ))


if __name__ == "__main__":
    main()
