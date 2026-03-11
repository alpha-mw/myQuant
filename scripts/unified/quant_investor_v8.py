"""
Quant Investor V8.0 — 七层架构
================================
在 V7 六层基础上新增：
  Layer 7: Multi-LLM 集成裁判（Claude + GPT-4o + DeepSeek + Gemini）

新增模块：
  - multi_llm_ensemble.py  → Layer 7: 四大LLM并行裁判
  - alpha_mining.py        → 系统性因子挖掘（独立运行）
  - portfolio_backtest.py  → Walk-Forward 组合回测
  - investment_report.py   → 结构化用户报告（含执行步骤）

架构变化：
  V7: 量化6层 → LLM多模型辩论（单模型调用多次）→ 输出
  V8: 量化6层 → 四大LLM独立裁判 → 加权融合 → 结构化报告含执行步骤 → 输出

运行方式：
  python quant_investor_v8.py --stocks 000001.SZ 600519.SH --capital 1000000
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from logger import get_logger

_logger = get_logger("QuantInvestorV8")

# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class V8PipelineResult:
    """V8 完整流水线结果"""
    # 各层输出
    raw_data: dict = field(default_factory=dict)
    factor_data: dict = field(default_factory=dict)
    model_predictions: dict = field(default_factory=dict)
    macro_signal: str = "🟡"
    macro_summary: str = ""
    risk_results: dict = field(default_factory=dict)
    v7_decision: Any = None             # DecisionLayerResult from V7

    # V8 新增
    llm_ensemble_results: dict = field(default_factory=dict)  # symbol → EnsembleConsensus
    final_report: str = ""              # 完整 Markdown 报告
    execution_log: list[str] = field(default_factory=list)

    # 时间统计
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------

class QuantInvestorV8:
    """
    V8 七层架构主引擎。

    Parameters
    ----------
    stock_pool      : 待分析股票代码列表（A股: "000001.SZ"，美股: "AAPL"）
    market          : "CN" | "US"
    lookback_years  : 数据回溯年数
    total_capital   : 模拟资金量（用于报告中仓位计算）
    risk_level      : "保守" | "中等" | "积极"
    enable_macro    : 是否启用宏观分析层（慢，可关闭调试）
    enable_backtest : 是否运行 Walk-Forward 回测（最慢，可关闭调试）
    enable_alpha_mining : 是否运行因子挖掘（最慢，可关闭调试）
    verbose         : 是否打印详细日志
    """

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_macro: bool = True,
        enable_backtest: bool = False,
        enable_alpha_mining: bool = False,
        verbose: bool = True,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_macro = enable_macro
        self.enable_backtest = enable_backtest
        self.enable_alpha_mining = enable_alpha_mining
        self.verbose = verbose
        self.result = V8PipelineResult()

    def run(self) -> V8PipelineResult:
        """执行完整七层分析流水线"""
        t_start = time.time()
        self._log("=" * 60)
        self._log("🚀 Quant Investor V8.0 启动")
        self._log(f"   标的: {self.stock_pool}")
        self._log(f"   市场: {self.market}  资金: ¥{self.total_capital:,.0f}")
        self._log("=" * 60)

        # ---------- Layers 1-6: 复用 V7 ----------
        v7_result = self._run_v7_pipeline()

        # ---------- Layer 7: Multi-LLM ----------
        self._layer7_multi_llm(v7_result)

        # ---------- 生成结构化报告 ----------
        self._generate_final_report()

        self.result.total_time = time.time() - t_start
        self._log(f"\n✅ V8 分析完成，总耗时 {self.result.total_time:.1f}s")
        return self.result

    # ----------------------------------------------------------------
    # Layer 1-6: V7 流水线（复用）
    # ----------------------------------------------------------------

    def _run_v7_pipeline(self) -> Any:
        """调用 V7 流水线，提取各层结果"""
        t0 = time.time()
        self._log("\n▶ 运行 V7 六层流水线 (Layers 1-6)...")

        try:
            from quant_investor_v7 import QuantInvestorV7
            v7 = QuantInvestorV7(
                market=self.market,
                stock_pool=self.stock_pool,
                lookback_years=self.lookback_years,
                enable_macro=self.enable_macro,
                verbose=self.verbose,
            )
            v7_result = v7.run()
            self.result.v7_decision = v7_result.decision_result
            self.result.macro_signal = getattr(v7_result, "macro_signal", "🟡") or "🟡"
            self.result.macro_summary = str(getattr(v7_result, "macro_report", "") or "")[:200]

            # 提取各层数据供 Layer 7 使用
            self.result.raw_data = {}
            self.result.factor_data = {}
            self.result.model_predictions = {}

            self._log(f"  V7 完成 ({time.time()-t0:.1f}s)")
            self.result.layer_timings["v7_pipeline"] = time.time() - t0
            return v7_result

        except ImportError:
            self._log("  ⚠️ quant_investor_v7 不可用，使用 Mock 数据")
            self.result.macro_signal = "🟡"
            self.result.macro_summary = "（V7不可用，使用Mock数据）"
            self.result.layer_timings["v7_pipeline"] = time.time() - t0
            return None

    # ----------------------------------------------------------------
    # Layer 7: Multi-LLM 集成裁判
    # ----------------------------------------------------------------

    def _layer7_multi_llm(self, v7_result: Any) -> None:
        t0 = time.time()
        self._log("\n▶ Layer 7: Multi-LLM 集成裁判...")

        try:
            from multi_llm_ensemble import MultiLLMEnsemble, build_quant_context
        except ImportError as e:
            self._log(f"  ⚠️ multi_llm_ensemble 导入失败: {e}")
            return

        ensemble = MultiLLMEnsemble()

        for symbol in self.stock_pool:
            self._log(f"  分析 {symbol}...")
            quant_ctx = self._build_quant_context_for_symbol(symbol, v7_result)
            quant_score = self._extract_quant_score(symbol, v7_result)

            consensus = ensemble.analyze_symbol(symbol, quant_ctx, quant_score)
            self.result.llm_ensemble_results[symbol] = consensus

            self._log(
                f"  {symbol}: {consensus.final_vote.value} "
                f"(得分={consensus.final_combined_score:+.3f}, "
                f"分歧={consensus.disagreement_index:.2f})"
            )

        self.result.layer_timings["layer7_llm"] = time.time() - t0
        self._log(f"  Layer 7 完成 ({time.time()-t0:.1f}s)")

    def _build_quant_context_for_symbol(self, symbol: str, v7_result: Any) -> dict:
        """从 V7 结果提取单只股票的量化上下文"""
        try:
            from multi_llm_ensemble import build_quant_context

            # 尝试从 V7 结果中提取真实数据
            factor_data = self._extract_factor_data(symbol, v7_result)
            model_preds = self._extract_model_predictions(symbol, v7_result)
            risk_metrics = self._extract_risk_metrics(symbol, v7_result)
            current_price = self._extract_current_price(symbol, v7_result)

            return build_quant_context(
                symbol=symbol,
                factor_data=factor_data,
                model_predictions=model_preds,
                macro_signal=self.result.macro_signal,
                risk_metrics=risk_metrics,
                current_price=current_price,
            )
        except Exception as e:
            _logger.debug(f"构建 {symbol} 量化上下文失败: {e}，使用最小上下文")
            return {
                "symbol": symbol,
                "macro_environment": {
                    "signal": self.result.macro_signal,
                    "interpretation": "宏观信号来自V7层",
                },
                "analysis_date": time.strftime("%Y-%m-%d"),
            }

    def _extract_quant_score(self, symbol: str, v7_result: Any) -> float:
        """从 V7 各层结果提取综合量化得分 [-1, 1]"""
        try:
            if v7_result is None:
                return 0.0
            # 尝试从 decision_result 中提取
            decision = getattr(v7_result, "decision_result", None)
            if decision:
                recs = getattr(decision, "stock_recommendations", [])
                for rec in recs:
                    if hasattr(rec, "symbol") and rec.symbol == symbol:
                        rating = getattr(rec, "rating", "持有")
                        score_map = {
                            "强烈买入": 0.8, "买入": 0.4, "持有": 0.0,
                            "卖出": -0.4, "强烈卖出": -0.8
                        }
                        return score_map.get(rating, 0.0)
            return 0.0
        except Exception:
            return 0.0

    def _extract_factor_data(self, symbol: str, v7_result: Any) -> dict:
        try:
            factor_data = getattr(v7_result, "factor_data", None)
            if factor_data and hasattr(factor_data, "selected_factors"):
                return {"selected_factors": factor_data.selected_factors[:10]}
            return {}
        except Exception:
            return {}

    def _extract_model_predictions(self, symbol: str, v7_result: Any) -> dict:
        try:
            model_results = getattr(v7_result, "model_results", None)
            if model_results:
                return {"ensemble_score": float(getattr(model_results, "ensemble_accuracy", 0))}
            return {}
        except Exception:
            return {}

    def _extract_risk_metrics(self, symbol: str, v7_result: Any) -> dict:
        try:
            risk_result = getattr(v7_result, "risk_layer_result", None)
            if risk_result:
                metrics = getattr(risk_result, "risk_metrics", None)
                if metrics:
                    return {
                        "var_95": getattr(metrics, "var_95", None),
                        "max_drawdown": getattr(metrics, "max_drawdown", None),
                        "sharpe_ratio": getattr(metrics, "sharpe_ratio", None),
                    }
            return {}
        except Exception:
            return {}

    def _extract_current_price(self, symbol: str, v7_result: Any) -> Optional[float]:
        try:
            raw_data = getattr(v7_result, "raw_data", {})
            if symbol in raw_data:
                df = raw_data[symbol]
                if hasattr(df, "empty") and not df.empty and "close" in df.columns:
                    return float(df["close"].iloc[-1])
            return None
        except Exception:
            return None

    # ----------------------------------------------------------------
    # 生成最终报告
    # ----------------------------------------------------------------

    def _generate_final_report(self) -> None:
        t0 = time.time()
        self._log("\n▶ 生成结构化投资报告...")

        try:
            from investment_report import (
                InvestmentReportGenerator, ReportInput,
                StockRecommendation, from_ensemble_result,
            )

            stock_recs: list[StockRecommendation] = []

            for symbol, consensus in self.result.llm_ensemble_results.items():
                quant_ctx = self._build_quant_context_for_symbol(
                    symbol, self.result.v7_decision
                )
                rec = from_ensemble_result(consensus, quant_ctx)
                stock_recs.append(rec)

            # 若没有 LLM 结果（所有 LLM 都不可用），使用 V7 结果生成基础报告
            if not stock_recs:
                stock_recs = self._build_recs_from_v7()

            report_input = ReportInput(
                stocks=stock_recs,
                macro_signal=self.result.macro_signal,
                macro_summary=self.result.macro_summary,
                market_outlook=self._build_market_outlook(),
                total_capital=self.total_capital,
                risk_level=self.risk_level,
                analysis_date=time.strftime("%Y-%m-%d"),
            )

            generator = InvestmentReportGenerator(report_input)
            self.result.final_report = generator.generate()
            self._log(f"  报告生成完成 ({time.time()-t0:.1f}s)")

        except Exception as e:
            _logger.error(f"报告生成失败: {e}")
            self.result.final_report = f"报告生成失败: {e}"

    def _build_recs_from_v7(self):
        """从 V7 决策层结果构建基础推荐列表（无 LLM 时的降级方案）"""
        from investment_report import StockRecommendation
        recs = []
        decision = self.result.v7_decision
        if decision is None:
            return recs
        recommendations = getattr(decision, "stock_recommendations", [])
        for rec in recommendations:
            recs.append(StockRecommendation(
                symbol=getattr(rec, "symbol", "—"),
                final_vote=getattr(rec, "rating", "持有"),
                ensemble_confidence=0.5,
                disagreement_index=0.0,
            ))
        return recs

    def _build_market_outlook(self) -> str:
        """从宏观信号生成市场展望文字"""
        signal = self.result.macro_signal
        base = self.result.macro_summary or ""
        outlook_map = {
            "🟢": "宏观指标向好，市场整体风险偏好上升，适合适度进攻。",
            "🟡": "市场方向不明朗，建议精选个股，控制仓位在50%-70%之间。",
            "🔴": "宏观压力较大，建议防御为主，重仓保守资产。",
        }
        return outlook_map.get(signal, "") + " " + base

    # ----------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.result.execution_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self.verbose:
            _logger.info(msg)

    def print_report(self) -> None:
        """打印最终报告到控制台"""
        print("\n" + "=" * 70)
        print(self.result.final_report)
        print("=" * 70)

    def save_report(self, path: str) -> None:
        """保存报告到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.final_report)
        _logger.info(f"报告已保存到: {path}")


# ---------------------------------------------------------------------------
# Alpha 挖掘独立入口
# ---------------------------------------------------------------------------

def run_alpha_mining(df, output_path: str = "/tmp/alpha_mining_report.md") -> str:
    """独立运行 Alpha 因子挖掘（与主流程解耦）"""
    from alpha_mining import AlphaMiner
    miner = AlphaMiner(
        df,
        enable_genetic=True,
        enable_llm=bool(os.getenv("ANTHROPIC_API_KEY")),
        genetic_generations=30,
    )
    result = miner.mine()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.mining_report)
    _logger.info(f"Alpha 挖掘报告已保存到: {output_path}")
    return result.mining_report


# ---------------------------------------------------------------------------
# Walk-Forward 回测独立入口
# ---------------------------------------------------------------------------

def run_portfolio_backtest(
    df,
    n_holdings: int = 10,
    method: str = "equal_weight",
    output_path: str = "/tmp/backtest_report.md",
) -> str:
    """独立运行组合回测"""
    from portfolio_backtest import PortfolioBacktester
    backtester = PortfolioBacktester(
        df, n_holdings=n_holdings,
        construction_method=method,
        rebalance_freq="W",
    )
    result = backtester.run_walkforward()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.backtest_report)
    _logger.info(f"回测报告已保存到: {output_path}")
    return result.backtest_report


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Investor V8.0 — 七层量化 + 四大AI裁判"
    )
    parser.add_argument(
        "--stocks", nargs="+",
        default=["000001.SZ", "600519.SH"],
        help="股票代码列表（A股格式：000001.SZ）",
    )
    parser.add_argument("--market",  default="CN", choices=["CN", "US"])
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="模拟资金（元）")
    parser.add_argument("--risk",    default="中等", choices=["保守", "中等", "积极"])
    parser.add_argument("--lookback", type=float, default=1.0, help="数据回溯年数")
    parser.add_argument("--no-macro",    action="store_true", help="关闭宏观层（调试用）")
    parser.add_argument("--with-backtest", action="store_true", help="开启 Walk-Forward 回测")
    parser.add_argument("--output",  default="", help="报告输出路径（默认打印到控制台）")
    args = parser.parse_args()

    investor = QuantInvestorV8(
        stock_pool=args.stocks,
        market=args.market,
        lookback_years=args.lookback,
        total_capital=args.capital,
        risk_level=args.risk,
        enable_macro=not args.no_macro,
        enable_backtest=args.with_backtest,
        verbose=True,
    )

    result = investor.run()

    if args.output:
        investor.save_report(args.output)
        print(f"\n报告已保存到: {args.output}")
    else:
        investor.print_report()

    # 打印时间统计
    print("\n⏱ 各层耗时：")
    for layer, t in result.layer_timings.items():
        print(f"  {layer}: {t:.1f}s")
    print(f"  总计: {result.total_time:.1f}s")


if __name__ == "__main__":
    main()
