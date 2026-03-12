"""
Quant Investor V10.0 — 五支柱并行智能框架
===========================================
彻底重构 V9 的顺序链式架构，改为真正的并行五分支 + 风控 + 集成裁判。

新架构：

    数据层（EnhancedDataLayer）
         │
    ┌────┴────────────────────────────────────────────────┐
    │                   五大并行分支                       │
    │                                                     │
    │  Branch 1        Branch 2        Branch 3           │
    │  Kronos          传统量化        LLM多空辩论         │
    │  图形预测        (因子+模型)      (5模型辩论)        │
    │                                                     │
    │  Branch 4        Branch 5                           │
    │  多维情报融合    宏观数据                            │
    │  (财务/新闻/情绪)(macro terminal)                   │
    └────────────────────────┬────────────────────────────┘
                             │
                      风控层（RiskManagementLayer）
                             │
                      集成裁判层（EnsembleJudgeEngine）
                             │
                        投资建议 + 完整报告

并行执行：concurrent.futures.ThreadPoolExecutor
降级策略：任意分支失败只影响该分支权重，不阻断流水线

运行方式：
  python quant_investor_v10.py --stocks 000001.SZ 600519.SH --capital 1000000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# 统一路径
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from logger import get_logger

_logger = get_logger("QuantInvestorV10")


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class BranchOutput:
    """单个分支的执行输出"""
    name: str
    result: Any = None
    elapsed: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None


@dataclass
class V10PipelineResult:
    """V10 完整流水线结果"""
    # 五分支输出
    kronos: BranchOutput = field(default_factory=lambda: BranchOutput("kronos"))
    quant: BranchOutput = field(default_factory=lambda: BranchOutput("quant"))
    debate: BranchOutput = field(default_factory=lambda: BranchOutput("debate"))
    intelligence: BranchOutput = field(default_factory=lambda: BranchOutput("intelligence"))
    macro: BranchOutput = field(default_factory=lambda: BranchOutput("macro"))

    # 风控层输出
    risk: BranchOutput = field(default_factory=lambda: BranchOutput("risk"))

    # 集成裁判层输出
    ensemble: Any = None

    # 报告
    final_report: str = ""
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# V10 主引擎
# ---------------------------------------------------------------------------

class QuantInvestorV10:
    """
    V10 五支柱并行智能框架主引擎。

    Parameters
    ----------
    stock_pool          : 待分析股票代码列表
    market              : "CN" | "US"
    lookback_years      : 数据回溯年数（默认1.0）
    total_capital       : 模拟资金量
    risk_level          : "保守" | "中等" | "积极"
    kronos_model        : "kronos-mini" | "kronos-small" | "kronos-base"
    pred_len            : Kronos预测期（交易日数）
    enable_kronos       : 是否启用Branch 1 Kronos
    enable_quant        : 是否启用Branch 2 传统量化
    enable_debate       : 是否启用Branch 3 LLM辩论
    enable_intelligence : 是否启用Branch 4 多维情报融合
    enable_macro        : 是否启用Branch 5 宏观分析
    max_workers         : 并行线程数（默认5）
    ensemble_weights    : 覆盖集成权重（dict）
    verbose             : 是否打印详细日志
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
        enable_kronos: bool = True,
        enable_quant: bool = True,
        enable_debate: bool = True,
        enable_intelligence: bool = True,
        enable_macro: bool = True,
        max_workers: int = 5,
        ensemble_weights: Optional[dict[str, float]] = None,
        verbose: bool = True,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.kronos_model = kronos_model
        self.pred_len = pred_len
        self.enable_kronos = enable_kronos
        self.enable_quant = enable_quant
        self.enable_debate = enable_debate
        self.enable_intelligence = enable_intelligence
        self.enable_macro = enable_macro
        self.max_workers = max_workers
        self.ensemble_weights = ensemble_weights
        self.verbose = verbose

        self._result = V10PipelineResult()
        # 共享价格数据（数据层输出，五分支共用）
        self._price_data: dict[str, pd.DataFrame] = {}
        self._stock_names: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 主流水线
    # ------------------------------------------------------------------

    def run(self) -> V10PipelineResult:
        t_start = time.time()
        self._log("=" * 72)
        self._log("🚀 Quant Investor V10.0 — 五支柱并行智能框架")
        self._log(f"   标的: {self.stock_pool}  市场: {self.market}")
        self._log(f"   资金: ¥{self.total_capital:,.0f}  风险偏好: {self.risk_level}")
        enabled = [n for n, e in [
            ("Kronos", self.enable_kronos), ("量化", self.enable_quant),
            ("辩论", self.enable_debate), ("情报", self.enable_intelligence),
            ("宏观", self.enable_macro)] if e]
        self._log(f"   启用分支: {', '.join(enabled)}  并行线程: {self.max_workers}")
        self._log("=" * 72)

        # ① 数据层：获取价格数据（所有分支共用）
        self._fetch_data_layer()

        # ② 五大分支并行执行
        self._run_branches_parallel()

        # ③ 风控层
        self._run_risk_layer()

        # ④ 集成裁判层
        self._run_ensemble_judge()

        # ⑤ 生成报告
        self._generate_report()

        self._result.total_time = round(time.time() - t_start, 2)
        self._log(f"\n✅ V10 分析完成，总耗时 {self._result.total_time}s")
        return self._result

    # ------------------------------------------------------------------
    # ① 数据层
    # ------------------------------------------------------------------

    def _fetch_data_layer(self) -> None:
        self._log("\n▶ [数据层] 获取价格数据...")
        t0 = time.time()

        for symbol in self.stock_pool:
            df = self._fetch_price(symbol)
            if df is not None and not df.empty:
                self._price_data[symbol] = df
                self._log(f"  {symbol}: {len(df)} 行")
            else:
                self._price_data[symbol] = self._synthetic_price(symbol)
                self._log(f"  {symbol}: 合成价格数据（数据源不可用）")

        self._stock_names = self._fetch_names()
        self._log(f"  数据层完成 ({round(time.time()-t0,2)}s)")

    def _fetch_price(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak  # type: ignore
            code = symbol.split(".")[0]
            days = int(self.lookback_years * 365)
            start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq",
                                     start_date=start)
            if df is not None and not df.empty:
                col_map = {"开盘": "open", "最高": "high", "最低": "low",
                           "收盘": "close", "成交量": "volume", "成交额": "amount",
                           "日期": "date"}
                return df.rename(columns=col_map)
        except Exception as e:
            _logger.debug(f"AKShare fetch failed for {symbol}: {e}")
        return None

    def _fetch_names(self) -> dict[str, str]:
        names: dict[str, str] = {}
        try:
            import akshare as ak  # type: ignore
            tbl = ak.stock_info_a_code_name()
            if tbl is not None and not tbl.empty:
                m = dict(zip(tbl["code"].values, tbl["name"].values))
                for s in self.stock_pool:
                    names[s] = m.get(s.split(".")[0], s)
                return names
        except Exception:
            pass
        return {s: s for s in self.stock_pool}

    @staticmethod
    def _synthetic_price(symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(seed=abs(hash(symbol)) % 2**32)
        n = 250
        ret = rng.normal(3e-4, 0.015, n)
        close = 100.0 * np.exp(np.cumsum(ret))
        return pd.DataFrame({
            "open":   close * rng.uniform(0.99, 1.01, n),
            "high":   close * rng.uniform(1.001, 1.025, n),
            "low":    close * rng.uniform(0.975, 0.999, n),
            "close":  close,
            "volume": rng.integers(1_000_000, 50_000_000, n),
        })

    # ------------------------------------------------------------------
    # ② 五大分支并行
    # ------------------------------------------------------------------

    def _run_branches_parallel(self) -> None:
        self._log("\n▶ [并行分支] 五大分支同步启动...")
        t0 = time.time()

        tasks = {}
        if self.enable_kronos:
            tasks["kronos"] = self._branch_kronos
        if self.enable_quant:
            tasks["quant"] = self._branch_quant
        if self.enable_debate:
            tasks["debate"] = self._branch_debate
        if self.enable_intelligence:
            tasks["intelligence"] = self._branch_intelligence
        if self.enable_macro:
            tasks["macro"] = self._branch_macro

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(fn): name for name, fn in tasks.items()}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    output: BranchOutput = future.result()
                    setattr(self._result, name, output)
                    status = "✅" if output.ok else "⚠️"
                    self._log(f"  {status} Branch-{name}: {output.elapsed}s")
                except Exception as e:
                    setattr(self._result, name, BranchOutput(name, error=str(e)))
                    self._log(f"  ❌ Branch-{name}: {e}")

        self._log(f"  并行分支完成 ({round(time.time()-t0,2)}s)")

    # ------------------------------------------------------------------
    # Branch 1: Kronos 图形预测
    # ------------------------------------------------------------------

    def _branch_kronos(self) -> BranchOutput:
        t0 = time.time()
        try:
            from kronos_predictor import KronosIntegrator
            integrator = KronosIntegrator(
                model_name=self.kronos_model,
                sample_count=5,
            )
            result = integrator.analyze_portfolio(
                stock_data_dict=self._price_data,
                pred_len=self.pred_len,
            )
            return BranchOutput("kronos", result, round(time.time()-t0, 2))
        except Exception as e:
            _logger.warning(f"Branch-Kronos 失败: {e}")
            return BranchOutput("kronos", error=str(e), elapsed=round(time.time()-t0, 2))

    # ------------------------------------------------------------------
    # Branch 2: 传统量化（因子层 + 模型层合并）
    # ------------------------------------------------------------------

    def _branch_quant(self) -> BranchOutput:
        """
        改进版传统量化分支：
          1. Alpha158 生成 130+ 因子
          2. 遗传/LLM Alpha 挖掘（可选）
          3. IC/IR 加权因子筛选
          4. 跨截面标准化 + 行业中性化
          5. 多模型集成预测（RF/XGB + 时序验证）
        """
        t0 = time.time()
        try:
            from quant_investor_v7 import QuantInvestorV7
            v7 = QuantInvestorV7(
                stock_pool=self.stock_pool,
                market=self.market,
                lookback_years=self.lookback_years,
                total_capital=self.total_capital,
                risk_level=self.risk_level,
                enable_macro=False,        # 宏观由Branch5负责
                enable_backtest=False,
                enable_alpha_mining=False,
                verbose=False,
            )
            result = v7.run()
            return BranchOutput("quant", result, round(time.time()-t0, 2))
        except Exception as e:
            _logger.warning(f"Branch-Quant V7 失败，尝试轻量模式: {e}")
            # 降级：直接用 Alpha158 + 简单 RF
            return self._branch_quant_lightweight(t0)

    def _branch_quant_lightweight(self, t0: float) -> BranchOutput:
        """轻量量化分支：当V7不可用时的降级方案（使用 FactorEngineer）"""
        try:
            from alpha158 import FactorEngineer

            engineer = FactorEngineer(ic_threshold=0.02, lookback_ic=60)
            predictions = engineer.cross_sectional_score(self._price_data)

            class LightQuantResult:
                def __init__(self, preds):
                    self.model_predictions = preds
                    self.raw_data = {}
                    self.factor_data = None

            return BranchOutput("quant", LightQuantResult(predictions), round(time.time()-t0, 2))
        except Exception as e2:
            return BranchOutput("quant", error=str(e2), elapsed=round(time.time()-t0, 2))

    # ------------------------------------------------------------------
    # Branch 3: LLM 多空辩论
    # ------------------------------------------------------------------

    def _branch_debate(self) -> BranchOutput:
        t0 = time.time()
        try:
            from decision_layer import DecisionLayer
            layer = DecisionLayer(verbose=False)

            # 构建传入数据
            quant_data: dict[str, dict] = {}
            for symbol in self.stock_pool:
                df = self._price_data.get(symbol)
                quant_data[symbol] = {
                    "price_data": df,
                    "stock_name": self._stock_names.get(symbol, symbol),
                    "market": self.market,
                }

            result = layer.run_decision_process(
                symbols=self.stock_pool,
                quant_data=quant_data,
                macro_data={},
                risk_data={},
            )
            return BranchOutput("debate", result, round(time.time()-t0, 2))
        except Exception as e:
            _logger.warning(f"Branch-Debate 失败: {e}")
            return BranchOutput("debate", error=str(e), elapsed=round(time.time()-t0, 2))

    # ------------------------------------------------------------------
    # Branch 4: 多维情报融合（财务 + 新闻 + 情绪）
    # ------------------------------------------------------------------

    def _branch_intelligence(self) -> BranchOutput:
        t0 = time.time()
        try:
            from intelligence_layer import IntelligenceLayerEngine
            engine = IntelligenceLayerEngine(
                kronos_model=self.kronos_model,
                weights={
                    "kronos": 0.0,          # Kronos 由 Branch 1 独立负责
                    "financial": 0.35,
                    "news": 0.30,
                    "sentiment": 0.30,
                    "quant_base": 0.05,
                },
                pred_len=self.pred_len,
                enable_llm_synthesis=True,
            )
            financial_data = self._fetch_financial_data()
            result = engine.analyze(
                stock_pool=self.stock_pool,
                price_data_dict=self._price_data,
                financial_data_dict=financial_data,
                stock_names=self._stock_names,
                market=self.market,
            )
            return BranchOutput("intelligence", result, round(time.time()-t0, 2))
        except Exception as e:
            _logger.warning(f"Branch-Intelligence 失败: {e}")
            return BranchOutput("intelligence", error=str(e), elapsed=round(time.time()-t0, 2))

    def _fetch_financial_data(self) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for symbol in self.stock_pool:
            try:
                import akshare as ak  # type: ignore
                code = symbol.split(".")[0]
                df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
                if df is not None and not df.empty:
                    data[symbol] = df
            except Exception:
                pass
        return data

    # ------------------------------------------------------------------
    # Branch 5: 宏观数据
    # ------------------------------------------------------------------

    def _branch_macro(self) -> BranchOutput:
        t0 = time.time()
        try:
            from macro_terminal_tushare import create_terminal
            terminal = create_terminal(market=self.market, verbose=False)
            result = terminal.run()
            return BranchOutput("macro", result, round(time.time()-t0, 2))
        except Exception as e:
            _logger.warning(f"Branch-Macro 失败: {e}")
            return BranchOutput("macro", error=str(e), elapsed=round(time.time()-t0, 2))

    # ------------------------------------------------------------------
    # ③ 风控层
    # ------------------------------------------------------------------

    def _run_risk_layer(self) -> None:
        self._log("\n▶ [风控层] 运行风险管理...")
        t0 = time.time()
        try:
            from risk_management_layer import RiskManagementLayer
            risk_layer = RiskManagementLayer(verbose=False)

            # 提取量化预测作为风控输入
            quant_result = self._result.quant.result
            factor_data = getattr(quant_result, "factor_data", None)
            positions_input = {s: 1.0 / len(self.stock_pool) for s in self.stock_pool}

            risk_result = risk_layer.run_risk_management(
                factor_data=factor_data,
                model_predictions=getattr(quant_result, "model_predictions", None),
                current_positions=positions_input,
                price_data={s: df["close"] if "close" in df.columns else df.iloc[:, 0]
                            for s, df in self._price_data.items() if df is not None},
            )
            self._result.risk = BranchOutput("risk", risk_result, round(time.time()-t0, 2))
            self._log(f"  风控层完成 ({self._result.risk.elapsed}s)")
        except Exception as e:
            _logger.warning(f"风控层失败: {e}")
            self._result.risk = BranchOutput("risk", error=str(e), elapsed=round(time.time()-t0, 2))

    # ------------------------------------------------------------------
    # ④ 集成裁判层
    # ------------------------------------------------------------------

    def _run_ensemble_judge(self) -> None:
        self._log("\n▶ [集成裁判层] 汇聚五大分支信号...")
        t0 = time.time()
        try:
            from ensemble_judge import EnsembleJudgeEngine

            # 估算组合波动率（用于市场状态检测）
            portfolio_vol, market_trend = self._estimate_market_metrics()

            judge = EnsembleJudgeEngine(
                max_single_position={"保守": 0.15, "中等": 0.25, "积极": 0.35}[self.risk_level],
                min_confidence=0.30,
                custom_weights=self.ensemble_weights,
            )
            self._result.ensemble = judge.judge(
                stock_pool=self.stock_pool,
                stock_names=self._stock_names,
                kronos_result=self._result.kronos.result,
                quant_result=self._result.quant.result,
                debate_result=self._result.debate.result,
                intelligence_result=self._result.intelligence.result,
                macro_result=self._result.macro.result,
                risk_result=self._result.risk.result,
                portfolio_volatility=portfolio_vol,
                market_trend_5d=market_trend,
            )
            self._log(f"  集成裁判完成 ({round(time.time()-t0,2)}s)  "
                      f"市场={self._result.ensemble.market_regime.value}")
        except Exception as e:
            _logger.error(f"集成裁判层失败: {e}")

    def _estimate_market_metrics(self) -> tuple[float, float]:
        """估算组合平均波动率和近5日趋势"""
        vols, trends = [], []
        for df in self._price_data.values():
            if df is None or "close" not in df.columns or len(df) < 20:
                continue
            close = df["close"].values
            ret = np.diff(np.log(close + 1e-9))
            if len(ret) >= 20:
                vols.append(float(np.std(ret[-20:])))
            if len(ret) >= 5:
                trends.append(float(np.mean(ret[-5:])))
        return (float(np.mean(vols)) if vols else 0.0,
                float(np.mean(trends)) if trends else 0.0)

    # ------------------------------------------------------------------
    # ⑤ 报告生成
    # ------------------------------------------------------------------

    def _generate_report(self) -> None:
        self._log("\n▶ [报告] 生成综合报告...")
        lines = [
            "# Quant Investor V10.0 — 五支柱并行智能分析报告\n\n",
            f"**分析日期**: {time.strftime('%Y-%m-%d %H:%M')}\n",
            f"**标的池**: {', '.join(self.stock_pool)}\n",
            f"**市场**: {self.market}  **模拟资金**: ¥{self.total_capital:,.0f}\n",
            f"**架构**: 五支柱并行 + 风控层 + 集成裁判层\n\n",
            "---\n\n",
        ]

        # 集成裁判报告（最核心）
        if self._result.ensemble:
            lines.append(self._result.ensemble.report)
            lines.append("\n\n---\n\n")

        # 各分支状态摘要
        lines.append("## 分支执行状态\n\n")
        lines.append("| 分支 | 状态 | 耗时 | 说明 |\n|------|------|------|------|\n")
        branch_info = [
            ("kronos",       "Branch 1: Kronos图形预测"),
            ("quant",        "Branch 2: 传统量化（因子+模型）"),
            ("debate",       "Branch 3: LLM多空辩论"),
            ("intelligence", "Branch 4: 多维情报融合"),
            ("macro",        "Branch 5: 宏观数据"),
            ("risk",         "风控层"),
        ]
        for attr, label in branch_info:
            b: BranchOutput = getattr(self._result, attr)
            status = "✅ 成功" if b.ok else f"⚠️ {b.error[:40] if b.error else '跳过'}"
            lines.append(f"| {label} | {status} | {b.elapsed:.1f}s | — |\n")

        lines.append(f"\n**总耗时**: {self._result.total_time}s\n")

        # 宏观分析报告片段
        if self._result.macro.ok:
            macro_report = getattr(self._result.macro.result, "macro_report", "") or \
                           getattr(self._result.macro.result, "report", "")
            if macro_report:
                lines.append("\n\n---\n\n## Branch 5: 宏观环境分析\n\n")
                lines.append(str(macro_report)[:3000])

        self._result.final_report = "".join(lines)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            _logger.info(msg)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quant Investor V10.0 — 五支柱并行智能框架"
    )
    parser.add_argument("--stocks", nargs="+", default=["000001.SZ", "600519.SH"])
    parser.add_argument("--market", default="CN", choices=["CN", "US"])
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--risk-level", default="中等", choices=["保守", "中等", "积极"])
    parser.add_argument("--kronos-model", default="kronos-small",
                        choices=["kronos-mini", "kronos-small", "kronos-base"])
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--no-kronos",      action="store_true")
    parser.add_argument("--no-quant",       action="store_true")
    parser.add_argument("--no-debate",      action="store_true")
    parser.add_argument("--no-intelligence",action="store_true")
    parser.add_argument("--no-macro",       action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyzer = QuantInvestorV10(
        stock_pool=args.stocks,
        market=args.market,
        total_capital=args.capital,
        risk_level=args.risk_level,
        kronos_model=args.kronos_model,
        pred_len=args.pred_len,
        enable_kronos=not args.no_kronos,
        enable_quant=not args.no_quant,
        enable_debate=not args.no_debate,
        enable_intelligence=not args.no_intelligence,
        enable_macro=not args.no_macro,
        max_workers=args.workers,
        verbose=True,
    )
    result = analyzer.run()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.final_report)
        print(f"\n✅ 报告已保存: {args.output}")
    else:
        print("\n" + "=" * 72)
        print(result.final_report[:5000] + (
            "\n\n...[报告截断，使用 --output 保存完整版]..." if len(result.final_report) > 5000 else ""
        ))


if __name__ == "__main__":
    main()
