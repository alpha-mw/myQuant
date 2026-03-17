#!/usr/bin/env python3
"""
五路并行研究编排器

主流程：
数据层 -> 五大研究分支并行 -> 风控层 -> 集成裁判层
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from alpha_mining import AlphaMiner, FactorLibrary
from branch_contracts import (
    BranchResult,
    PortfolioStrategy,
    ResearchPipelineResult,
    TradeRecommendation,
    UnifiedDataBundle,
)
from enhanced_data_layer import EnhancedDataLayer
from logger import get_logger
from macro_terminal_tushare import create_terminal
from risk_management_layer import PortfolioOptimizer, RiskLayerResult, RiskManagementLayer
from signal_calibration import SignalCalibrator


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# ---------------------------------------------------------------------------
# 自适应集成权重追踪器
# ---------------------------------------------------------------------------

class BranchPerformanceTracker:
    """
    基于滚动 IC（信息系数）动态调整五路分支权重。

    每次运行后将分支得分与实际收益存入 JSON 历史文件，
    下次运行时读取历史 IC 计算自适应权重：
      w_i = max(IC_i_rolling, 0) / sum(max(IC_j, 0))
    若所有分支 IC 均非正，回退到预设静态权重。

    历史文件：data/branch_ic_history.json（相对于工作目录）
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "kline": 0.22,
        "quant": 0.28,
        "llm_debate": 0.15,
        "intelligence": 0.20,
        "macro": 0.15,
    }
    MIN_WEIGHT = 0.05      # 分支 IC ≤ 0 时的最低保底权重
    ROLLING_WINDOW = 60    # 滚动 IC 窗口（天/次）
    HISTORY_PATH = "data/branch_ic_history.json"

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {k: [] for k in self.DEFAULT_WEIGHTS}
        self._load()

    def _load(self) -> None:
        try:
            import json
            if os.path.exists(self.HISTORY_PATH):
                with open(self.HISTORY_PATH) as f:
                    data = json.load(f)
                # 向后兼容：旧版 "kronos" key 自动迁移到 "kline"
                if "kronos" in data and "kline" not in data:
                    data["kline"] = data.pop("kronos")
                for branch in self._history:
                    self._history[branch] = data.get(branch, [])[-self.ROLLING_WINDOW:]
        except Exception:
            pass

    def save(self, branch_scores: dict[str, float], realized_return: float) -> None:
        """记录本次运行的分支得分 × 实际收益（IC 代理）"""
        import json
        for branch, score in branch_scores.items():
            if branch in self._history:
                ic_proxy = score * np.sign(realized_return)
                self._history[branch].append(float(ic_proxy))
                self._history[branch] = self._history[branch][-self.ROLLING_WINDOW:]
        try:
            os.makedirs(os.path.dirname(self.HISTORY_PATH) or ".", exist_ok=True)
            with open(self.HISTORY_PATH, "w") as f:
                json.dump(self._history, f)
        except Exception:
            pass

    def get_adaptive_weights(self) -> dict[str, float]:
        """
        计算自适应权重。历史记录不足时回退到静态权重。
        """
        ic_means: dict[str, float] = {}
        for branch, history in self._history.items():
            if len(history) >= 5:
                ic_means[branch] = float(np.mean(history[-self.ROLLING_WINDOW:]))
            else:
                ic_means[branch] = self.DEFAULT_WEIGHTS.get(branch, 0.1)

        positive_ics = {k: max(v, 0.0) for k, v in ic_means.items()}
        total = sum(positive_ics.values())
        if total < 1e-8:
            return dict(self.DEFAULT_WEIGHTS)

        weights = {k: max(v / total, self.MIN_WEIGHT) for k, v in positive_ics.items()}
        # 重新归一化（保底权重可能使总和 > 1）
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}


def _safe_mean(values: list[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def _series_or_empty(values: dict[str, float]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float)


class ParallelResearchPipeline:
    """并行研究主流程编排器。"""

    BRANCH_ORDER = [
        "kline",
        "quant",
        "llm_debate",
        "intelligence",
        "macro",
    ]

    def __init__(
        self,
        stock_pool: list[str],
        market: str = "CN",
        lookback_years: float = 1.0,
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        enable_alpha_mining: bool = True,
        enable_kline: bool = True,
        enable_intelligence: bool = True,
        enable_llm_debate: bool = True,
        enable_macro: bool = True,
        kline_backend: str = "heuristic",
        allow_synthetic_for_research: bool = False,
        verbose: bool = True,
        # 向后兼容
        enable_kronos: bool | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.market = market.upper()
        self.lookback_years = lookback_years
        self.total_capital = total_capital
        self.risk_level = risk_level
        self.enable_alpha_mining = enable_alpha_mining
        # 向后兼容：enable_kronos 映射到 enable_kline
        self.enable_kline = enable_kline if enable_kronos is None else enable_kronos
        self.enable_intelligence = enable_intelligence
        self.enable_llm_debate = enable_llm_debate
        self.enable_macro = enable_macro
        self.kline_backend = kline_backend
        self.allow_synthetic_for_research = allow_synthetic_for_research
        self.verbose = verbose
        self.data_layer = EnhancedDataLayer(market=self.market, verbose=verbose)
        self.risk_layer = RiskManagementLayer(verbose=verbose)
        self._logger = get_logger("ParallelResearchPipeline", verbose)
        self._branch_tracker = BranchPerformanceTracker()
        self._market_regime: str | None = None  # 由 run() 中 RegimeDetector 设定

    def _log(self, message: str) -> None:
        self._logger.info(message)

    def run(self) -> ResearchPipelineResult:
        """执行完整并行研究流程。"""
        t0 = time.time()
        data_bundle = self._build_data_bundle()
        result = ResearchPipelineResult(data_bundle=data_bundle)
        result.execution_log.append("并行研究流程启动")
        result.timings["data_layer"] = time.time() - t0

        # 在分支执行前检测市场状态，供各分支自适应使用
        try:
            from regime_detector import RegimeDetector
            combined_for_regime = data_bundle.combined_frame()
            market_ret = combined_for_regime.groupby("date")["close"].mean().pct_change().dropna()
            if len(market_ret) >= 20:
                regime_result = RegimeDetector().detect(market_ret)
                self._market_regime = regime_result.regime.value
                self._log(f"市场状态识别：{self._market_regime}")
        except Exception as regime_exc:
            self._log(f"市场状态识别失败，使用默认权重: {regime_exc}")

        branch_start = time.time()
        result.branch_results = self._run_branches(data_bundle)
        result.timings["research_branches"] = time.time() - branch_start
        result.calibrated_signals = self._calibrate_signals(data_bundle, result.branch_results)

        # 分级 Quorum 检查
        successful_branches = [b for b in result.branch_results.values() if b.success]
        n_success = len(successful_branches)
        # 分级可靠度乘数：0-1=禁止交易, 2=0.6, 3=0.85, 4=0.95, 5=1.0
        self._quorum_reliability = {0: 0.0, 1: 0.0, 2: 0.6, 3: 0.85, 4: 0.95, 5: 1.0}.get(n_success, 1.0)
        self._quorum_max_exposure = {0: 0.0, 1: 0.0, 2: 0.30, 3: 0.95, 4: 0.95, 5: 0.95}.get(n_success, 0.95)
        if n_success <= 1:
            self._log(
                f"警告：仅 {n_success}/5 个分支成功，"
                "进入 research_only 模式，禁止生成交易建议。"
            )
            result.execution_log.append(
                f"[CRITICAL] 分支成功数严重不足（{n_success}/5），研究模式，不生成交易。"
            )
        elif n_success == 2:
            self._log(
                f"警告：仅 {n_success}/5 个分支成功，"
                "仓位上限降至 30%，可靠度 ×0.6。"
            )
            result.execution_log.append(
                f"[WARN] 分支成功数偏低（{n_success}/5），仓位受限，结果仅供参考。"
            )
        elif n_success == 3:
            result.execution_log.append(
                f"[INFO] {n_success}/5 个分支成功，正常运行（轻度警告）。"
            )

        risk_start = time.time()
        result.risk_result = self._run_risk_layer(data_bundle, result.branch_results, result.calibrated_signals)
        result.timings["risk_layer"] = time.time() - risk_start

        ensemble_start = time.time()
        result.final_strategy = self._run_ensemble_layer(
            data_bundle,
            result.branch_results,
            result.risk_result,
            calibrated_signals=result.calibrated_signals,
        )
        result.final_report = self._build_markdown_report(result)
        result.timings["ensemble_layer"] = time.time() - ensemble_start
        result.timings["total"] = time.time() - t0
        result.execution_log.append("并行研究流程完成")
        return result

    # ---------------------------------------------------------------------
    # 数据层
    # ---------------------------------------------------------------------

    def _build_data_bundle(self) -> UnifiedDataBundle:
        """构建统一数据包。"""
        self._log("构建统一数据包...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(365 * self.lookback_years))

        symbol_data: dict[str, pd.DataFrame] = {}
        fundamentals: dict[str, dict[str, Any]] = {}
        event_data: dict[str, list[dict[str, Any]]] = {}
        sentiment_data: dict[str, dict[str, Any]] = {}
        symbol_provenance: dict[str, dict[str, Any]] = {}

        for symbol in self.stock_pool:
            df, provenance = self._fetch_symbol_frame(symbol, start_date, end_date)
            symbol_data[symbol] = df
            symbol_provenance[symbol] = provenance
            fundamentals[symbol] = self._extract_fundamental_snapshot(df)
            event_data[symbol] = self._build_event_snapshot(symbol, df)
            sentiment_data[symbol] = self._build_sentiment_snapshot(symbol, df)

        macro_data = {
            "market": self.market,
            "as_of": end_date.strftime("%Y-%m-%d"),
            "risk_level": "中风险",
            "signal": "🟡",
        }

        return UnifiedDataBundle(
            market=self.market,
            symbols=self.stock_pool,
            symbol_data=symbol_data,
            fundamentals=fundamentals,
            event_data=event_data,
            sentiment_data=sentiment_data,
            macro_data=macro_data,
            metadata={
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
                "total_capital": self.total_capital,
                "risk_level": self.risk_level,
                "symbol_provenance": symbol_provenance,
                "data_source_status": self._bundle_data_source_status(symbol_provenance),
                "research_mode": self._bundle_research_mode(symbol_provenance),
            },
        )

    def _fetch_symbol_frame(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """优先拉取真实数据，失败时降级到模拟数据。"""
        try:
            df = self.data_layer.fetch_and_process(
                symbol=symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                label_periods=5,
            )
            if df is not None and not df.empty:
                if "forward_ret_5d" not in df.columns:
                    df["forward_ret_5d"] = df["close"].shift(-5) / df["close"] - 1
                return df, {
                    "symbol": symbol,
                    "data_source_status": "real",
                    "is_synthetic": False,
                    "degraded_reason": "",
                    "branch_mode": "real_market_data",
                    "reliability": 1.0,
                }
        except Exception as exc:
            self._log(f"{symbol} 真实数据获取失败，使用模拟数据: {exc}")
            degraded_reason = str(exc)
        else:
            degraded_reason = "empty_frame"

        return self._generate_mock_symbol_frame(symbol, start_date, end_date), {
            "symbol": symbol,
            "data_source_status": "synthetic_fallback",
            "is_synthetic": True,
            "degraded_reason": degraded_reason,
            "branch_mode": "synthetic_fallback",
            "reliability": 0.35,
        }

    def _generate_mock_symbol_frame(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """生成可用于测试和离线运行的模拟数据。"""
        seed = abs(hash(symbol)) % 10000
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start_date, end_date)
        drift = 0.0005 + (seed % 7) * 0.00008
        shocks = rng.normal(drift, 0.02, len(dates))
        close = 100 * np.exp(np.cumsum(shocks))
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, len(dates)))
        volume = rng.integers(2_000_000, 12_000_000, len(dates))
        amount = close * volume

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        )
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["momentum_12_1"] = df["close"].pct_change(120) - df["close"].pct_change(20)
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        df["ma_bias_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).mean()
        df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        df["label_return"] = df["close"].shift(-5) / df["close"] - 1
        df["forward_ret_5d"] = df["label_return"]
        df["symbol"] = symbol
        df["market"] = self.market
        df["roe"] = 0.08 + (seed % 9) * 0.01
        df["gross_margin"] = 0.2 + (seed % 5) * 0.03
        df["profit_growth"] = -0.05 + (seed % 8) * 0.03
        df["revenue_growth"] = -0.02 + (seed % 6) * 0.025
        df["debt_ratio"] = 0.25 + (seed % 4) * 0.08
        df["pe"] = 10 + (seed % 15)
        df["pb"] = 1.0 + (seed % 6) * 0.3
        df["ps"] = 1.0 + (seed % 5) * 0.25
        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def _extract_fundamental_snapshot(self, df: pd.DataFrame) -> dict[str, Any]:
        """从数据表抽取基本面快照。"""
        if df is None or df.empty:
            return {}
        latest = df.iloc[-1]
        fields = [
            "roe",
            "gross_margin",
            "profit_growth",
            "revenue_growth",
            "debt_ratio",
            "pe",
            "pb",
            "ps",
        ]
        snapshot = {field: float(latest[field]) for field in fields if field in latest and pd.notna(latest[field])}
        snapshot["latest_close"] = float(latest.get("close", 0.0))
        return snapshot

    def _build_event_snapshot(self, symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        """构造占位事件摘要。"""
        if df is None or df.empty:
            return []
        latest = df.iloc[-1]
        recent_return = float(latest.get("return_20d", 0.0) or 0.0)
        volume_ratio = float(latest.get("volume_ratio_20d", 1.0) or 1.0)
        events = []
        if recent_return > 0.12:
            events.append({"type": "price_breakout", "headline": f"{symbol} 近期趋势显著增强", "impact": 0.2})
        if recent_return < -0.12:
            events.append({"type": "drawdown", "headline": f"{symbol} 近期出现较大回撤", "impact": -0.25})
        if volume_ratio > 1.5:
            events.append({"type": "abnormal_volume", "headline": f"{symbol} 成交量显著放大", "impact": 0.1})
        return events

    def _build_sentiment_snapshot(self, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
        """构造情绪和广度占位数据。"""
        if df is None or df.empty:
            return {}
        ret = df["close"].pct_change().dropna()

        # --- 增强版恐贪指数（三维度组合） ---
        # 1) 动量信号：20 日 Sharpe
        momentum = float(ret.tail(20).mean() / (ret.tail(20).std() + 1e-8))
        # 2) 波动率排位（低波 = 偏贪婪）
        vol_20 = float(ret.tail(20).std()) if len(ret) >= 20 else float(ret.std())
        vol_252 = float(ret.tail(252).std()) if len(ret) >= 252 else vol_20
        vol_rank = 1.0 - min(vol_20 / (vol_252 + 1e-8), 2.0) / 2.0
        # 3) 价格在 52 周高低间的位置
        close = df["close"]
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
        price_position = (float(close.iloc[-1]) - low_52w) / (high_52w - low_52w + 1e-8)
        # 组合
        raw_fg = 0.35 * momentum + 0.25 * vol_rank + 0.40 * (price_position * 2 - 1)
        fear_greed = _clamp(raw_fg / 2.5, -1.0, 1.0)

        money_flow = float(df["volume"].tail(5).mean() / (df["volume"].tail(20).mean() + 1e-8) - 1)
        breadth = float((ret.tail(20) > 0).mean() * 2 - 1) if not ret.empty else 0.0
        return {
            "symbol": symbol,
            "fear_greed": fear_greed,
            "money_flow": _clamp(money_flow, -1.0, 1.0),
            "breadth": _clamp(breadth, -1.0, 1.0),
        }

    # ---------------------------------------------------------------------
    # 分支层
    # ---------------------------------------------------------------------

    def _run_branches(self, data_bundle: UnifiedDataBundle) -> dict[str, BranchResult]:
        branch_factories: dict[str, Callable[[UnifiedDataBundle], BranchResult]] = {
            "kline": self._run_kline_branch,
            "quant": self._run_quant_branch,
            "llm_debate": self._run_llm_debate_branch,
            "intelligence": self._run_intelligence_branch,
            "macro": self._run_macro_branch,
        }
        enabled = {
            "kline": self.enable_kline,
            "quant": True,
            "llm_debate": self.enable_llm_debate,
            "intelligence": self.enable_intelligence,
            "macro": self.enable_macro,
        }

        results: dict[str, BranchResult] = {}
        branch_timeout = 120  # 单分支最长等待秒数
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(branch_factories[name], data_bundle): name
                for name in self.BRANCH_ORDER
                if enabled.get(name, False)
            }
            for future in as_completed(future_map, timeout=branch_timeout + 10):
                name = future_map[future]
                try:
                    branch_result = future.result(timeout=branch_timeout)
                    branch_result.metadata.setdefault(
                        "data_source_status",
                        self._branch_data_source_status(data_bundle),
                    )
                    branch_result.metadata.setdefault("is_synthetic", False)
                    branch_result.metadata.setdefault("degraded_reason", "")
                    branch_result.metadata.setdefault("branch_mode", self._default_branch_mode(name))
                    branch_result.metadata.setdefault("reliability", 1.0 if branch_result.success else 0.25)
                    results[name] = branch_result
                except Exception as exc:
                    self._log(f"{name} 分支失败，使用降级结果: {exc}")
                    results[name] = BranchResult(
                        branch_name=name,
                        score=0.0,
                        confidence=0.0,
                        risks=[f"{name} 分支失败: {exc}"],
                        explanation=f"{name} 分支异常，已降级为中性结果。",
                        symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
                        success=False,
                        metadata={
                            "data_source_status": "degraded_branch",
                            "is_synthetic": False,
                            "degraded_reason": str(exc),
                            "branch_mode": self._default_branch_mode(name),
                            "reliability": 0.2,
                        },
                    )

        for name in self.BRANCH_ORDER:
            if name not in results and enabled.get(name, False):
                results[name] = BranchResult(
                    branch_name=name,
                    explanation=f"{name} 分支未执行，返回中性结果。",
                    symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
                    success=False,
                    metadata={
                        "data_source_status": "degraded_branch",
                        "is_synthetic": False,
                        "degraded_reason": "branch_not_executed",
                        "branch_mode": self._default_branch_mode(name),
                        "reliability": 0.2,
                    },
                )

        return results

    def _calibrate_signals(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
    ) -> dict[str, Any]:
        """对全部分支结果做统一量纲校准。"""
        calibrator = SignalCalibrator(data_bundle.symbol_provenance())
        return calibrator.calibrate_all(branch_results, self.stock_pool)

    def _run_kline_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """K线分析分支：支持 heuristic / kronos / chronos 三种后端。"""
        from kline_backends import get_backend

        backend = get_backend(self.kline_backend)
        result = backend.predict(data_bundle.symbol_data, self.stock_pool)
        result.branch_name = "kline"
        result.metadata["branch_mode"] = f"kline_{backend.name}"
        result.metadata["reliability"] = backend.reliability
        result.metadata["horizon_days"] = backend.horizon_days
        return result

    def _run_quant_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """传统量化分支：Alpha 挖掘优先，失败时回退到经典因子。"""
        combined = data_bundle.combined_frame()
        if combined.empty:
            return BranchResult(
                branch_name="quant",
                explanation="量化分支没有可用数据，返回中性结果。",
                symbol_scores={symbol: 0.0 for symbol in self.stock_pool},
                success=False,
            )

        if "forward_ret_5d" not in combined.columns:
            combined["forward_ret_5d"] = combined["close"].shift(-5) / combined["close"] - 1

        selected_factors: list[str] = []
        explanation = ""
        if self.enable_alpha_mining:
            try:
                unique_dates = combined["date"].nunique() if "date" in combined.columns else 60
                gen_count = max(10, min(20, unique_dates // 15))
                miner = AlphaMiner(
                    combined.copy(),
                    forward_col="forward_ret_5d",
                    enable_genetic=True,
                    enable_llm=False,
                    genetic_generations=gen_count,
                )
                mining_result = miner.mine()
                selected_factors = [profile.name for profile in mining_result.selected_factors[:6]]
                explanation = (
                    f"Alpha 挖掘完成，优先采用 {len(selected_factors)} 个经过 IC/IR 和正交筛选的因子。"
                )
            except Exception as exc:
                explanation = f"Alpha 挖掘失败，使用经典因子回退: {exc}"
        else:
            explanation = "已关闭 Alpha 挖掘，当前直接使用经典量价因子组合作为量化研究分支。"

        if not selected_factors:
            selected_factors = [
                factor
                for factor in ["momentum_3m", "realized_vol_20d", "rsi_14", "macd_signal", "volume_ratio"]
                if factor in FactorLibrary.all_factor_funcs()
            ]
            explanation = "Alpha 挖掘未筛出稳定因子，当前回退到经典动量/低波/技术/量能因子组合。"

        factor_exposures: dict[str, dict[str, float]] = {symbol: {} for symbol in self.stock_pool}
        factor_scores = pd.DataFrame(index=combined.index)
        factor_funcs = FactorLibrary.all_factor_funcs()

        for factor_name in selected_factors:
            func = factor_funcs.get(factor_name)
            if func is None:
                continue
            try:
                factor_scores[factor_name] = func(combined)
            except Exception:
                continue

        if factor_scores.empty:
            factor_scores["fallback_momentum"] = combined.groupby("symbol")["close"].pct_change(20)
            selected_factors = ["fallback_momentum"]

        ranked = factor_scores.groupby(combined["date"]).transform(
            lambda series: (series - series.mean()) / (series.std(ddof=0) + 1e-8)
        )
        combined_scores = ranked.mean(axis=1).fillna(0.0)
        combined = combined.assign(quant_alpha_score=combined_scores)

        latest_rows = combined.sort_values("date").groupby("symbol").tail(1)
        symbol_scores = {
            str(row["symbol"]): float(_clamp(row["quant_alpha_score"] / 3, -1.0, 1.0))
            for _, row in latest_rows.iterrows()
        }

        for _, row in latest_rows.iterrows():
            symbol = str(row["symbol"])
            for factor_name in selected_factors[:5]:
                if factor_name in factor_scores.columns:
                    factor_exposures[symbol][factor_name] = float(factor_scores.loc[row.name, factor_name])

        expected_returns = {
            symbol: float(_clamp(score * 0.12, -0.25, 0.25))
            for symbol, score in symbol_scores.items()
        }
        score = _safe_mean(list(symbol_scores.values()))
        confidence = _clamp(0.4 + len(selected_factors) * 0.05, 0.4, 0.85)
        risks = []
        if len(selected_factors) <= 2:
            risks.append("可用 Alpha 因子较少，量化信号稳定性有限。")

        return BranchResult(
            branch_name="quant",
            score=score,
            confidence=confidence,
            signals={
                "alpha_factors": selected_factors,
                "factor_exposures": factor_exposures,
                "expected_return": expected_returns,
                "feature_reasoning": explanation,
            },
            risks=risks,
            explanation=explanation,
            symbol_scores=symbol_scores,
            metadata={
                "factor_exposures": factor_exposures,
                "expected_return": expected_returns,
                "branch_mode": "alpha_research",
                "reliability": 0.86 if len(selected_factors) >= 3 else 0.68,
                "horizon_days": 5,
            },
        )

    def _run_llm_debate_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """LLM 多空辩论分支：第一版使用结构化研究摘要生成辩论结论。"""
        symbol_scores: dict[str, float] = {}
        bull_case: dict[str, list[str]] = {}
        bear_case: dict[str, list[str]] = {}
        key_risks: dict[str, list[str]] = {}

        for symbol in self.stock_pool:
            fundamentals = data_bundle.fundamentals.get(symbol, {})
            events = data_bundle.event_data.get(symbol, [])
            sentiment = data_bundle.sentiment_data.get(symbol, {})

            bullish_points: list[str] = []
            bearish_points: list[str] = []
            risks: list[str] = []
            score = 0.0

            roe = float(fundamentals.get("roe", 0.0))
            growth = float(fundamentals.get("profit_growth", fundamentals.get("revenue_growth", 0.0)))
            debt = float(fundamentals.get("debt_ratio", 0.4))
            event_impact = _safe_mean([float(item.get("impact", 0.0)) for item in events])
            fear_greed = float(sentiment.get("fear_greed", 0.0))

            if roe > 0.12:
                bullish_points.append("盈利能力较强，ROE 表现优于中性水平。")
                score += 0.25
            else:
                bearish_points.append("盈利能力尚未形成明显优势。")
                score -= 0.1

            if growth > 0.1:
                bullish_points.append("成长信号积极，利润或收入保持扩张。")
                score += 0.2
            elif growth < -0.05:
                bearish_points.append("成长动能偏弱，基本面改善需要更多验证。")
                score -= 0.2

            if debt > 0.55:
                risks.append("资产负债率偏高，需要关注偿债压力。")
                bearish_points.append("负债水平较高，财务弹性受限。")
                score -= 0.15

            if event_impact > 0.05:
                bullish_points.append("近期事件流偏正面，有利于估值修复。")
                score += 0.15
            elif event_impact < -0.05:
                bearish_points.append("近期事件流偏负面，可能压制风险偏好。")
                score -= 0.15

            if fear_greed > 0.2:
                bullish_points.append("市场情绪边际回暖。")
                score += 0.1
            elif fear_greed < -0.2:
                bearish_points.append("市场情绪仍偏谨慎。")
                score -= 0.1

            bull_case[symbol] = bullish_points or ["暂无强烈看多证据，维持中性观察。"]
            bear_case[symbol] = bearish_points or ["暂无明显看空论据。"]
            key_risks[symbol] = risks
            symbol_scores[symbol] = _clamp(score, -1.0, 1.0)

        return BranchResult(
            branch_name="llm_debate",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=0.58,
            signals={
                "bull_case": bull_case,
                "bear_case": bear_case,
                "key_risks": key_risks,
                "llm_confidence": 0.58,
            },
            risks=["第一版为结构化研究辩论器，新闻与公告摘要仍以占位适配为主。"],
            explanation="LLM 辩论分支围绕基本面、行业、事件和情绪构建多空论据，但不直接负责最终仓位裁决。",
            symbol_scores=symbol_scores,
            metadata={
                "bull_case": bull_case,
                "bear_case": bear_case,
                "branch_mode": "structured_research_debate",
                "debate_mode": "structured_rules",
                "reliability": 0.58,
                "horizon_days": 10,
            },
        )

    def _run_intelligence_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """多维智能融合分支。"""
        financial_health: dict[str, float] = {}
        event_risk: dict[str, float] = {}
        sentiment_score: dict[str, float] = {}
        breadth_score: dict[str, float] = {}
        alerts: list[str] = []
        symbol_scores: dict[str, float] = {}

        advancing = 0
        declining = 0
        for symbol, df in data_bundle.symbol_data.items():
            fundamentals = data_bundle.fundamentals.get(symbol, {})
            sentiments = data_bundle.sentiment_data.get(symbol, {})
            events = data_bundle.event_data.get(symbol, [])

            roe = float(fundamentals.get("roe", 0.0))
            gross_margin = float(fundamentals.get("gross_margin", 0.0))
            debt_ratio = float(fundamentals.get("debt_ratio", 0.5))
            pe = float(fundamentals.get("pe", 15.0))
            pb = float(fundamentals.get("pb", 2.0))

            piotroski_like = (
                (1 if roe > 0.1 else 0)
                + (1 if gross_margin > 0.25 else 0)
                + (1 if debt_ratio < 0.45 else 0)
            ) / 3
            beneish_risk = 1.0 - _clamp((gross_margin * 2 + roe - debt_ratio), 0.0, 1.0)
            altman_like = _clamp(1.5 - debt_ratio + roe, 0.0, 1.0)
            valuation_score = _clamp((20 - pe) / 20 + (2.5 - pb) / 3, -1.0, 1.0)

            financial_score = _clamp(
                (piotroski_like - beneish_risk + altman_like + valuation_score) / 3,
                -1.0,
                1.0,
            )
            event_score = _clamp(-_safe_mean([abs(float(item.get("impact", 0.0))) for item in events]), -1.0, 0.0)
            senti = _clamp(
                0.5 * float(sentiments.get("fear_greed", 0.0))
                + 0.3 * float(sentiments.get("money_flow", 0.0))
                + 0.2 * float(sentiments.get("breadth", 0.0)),
                -1.0,
                1.0,
            )

            if df["close"].pct_change().iloc[-1] >= 0:
                advancing += 1
            else:
                declining += 1

            breadth = 0.0
            if advancing + declining > 0:
                breadth = (advancing - declining) / (advancing + declining)

            if debt_ratio > 0.6:
                alerts.append(f"{symbol} 负债率偏高，需结合现金流进一步验证。")
            if beneish_risk > 0.6:
                alerts.append(f"{symbol} 财务舞弊风险代理信号偏高。")

            financial_health[symbol] = financial_score
            event_risk[symbol] = event_score
            sentiment_score[symbol] = senti
            breadth_score[symbol] = breadth
            # 根据市场状态自适应调整融合权重
            regime = self._market_regime
            if regime == "趋势上涨" or regime == "趋势下跌":
                w_fin, w_sen, w_evt = 0.35, 0.40, 0.25
            elif regime == "震荡低波":
                w_fin, w_sen, w_evt = 0.55, 0.20, 0.25
            elif regime == "震荡高波":
                w_fin, w_sen, w_evt = 0.40, 0.25, 0.35
            else:
                w_fin, w_sen, w_evt = 0.50, 0.25, 0.25
            symbol_scores[symbol] = _clamp(w_fin * financial_score + w_sen * senti + w_evt * event_score, -1.0, 1.0)

        return BranchResult(
            branch_name="intelligence",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=0.62,
            signals={
                "intelligence_score": symbol_scores,
                "financial_health_score": financial_health,
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "breadth_score": breadth_score,
                "alerts": alerts,
            },
            risks=alerts[:5],
            explanation="多维智能融合分支独立整合财务质量、事件风险、情绪、资金流和市场广度信号。",
            symbol_scores=symbol_scores,
            metadata={
                "financial_health_score": financial_health,
                "event_risk_score": event_risk,
                "sentiment_score": sentiment_score,
                "breadth_score": breadth_score,
                "branch_mode": "structured_intelligence_fusion",
                "reliability": 0.72,
                "horizon_days": 10,
            },
        )

    def _run_macro_branch(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        """宏观分支：优先使用现有宏观终端，失败时回退到市场内部统计。"""
        try:
            terminal = create_terminal(self.market)
            report = terminal.generate_risk_report()
            macro_score = {"🔴": -0.8, "🟡": 0.0, "🟢": 0.7, "🔵": 0.9}.get(report.overall_signal, 0.0)
            signal_map = {
                "macro_score": macro_score,
                "macro_regime": report.overall_risk_level,
                "liquidity_signal": report.overall_signal,
                "policy_signal": report.recommendation,
                "risk_level": report.overall_risk_level,
            }
            return BranchResult(
                branch_name="macro",
                score=macro_score,
                confidence=0.75,
                signals=signal_map,
                risks=[] if macro_score >= 0 else ["宏观环境偏谨慎，需控制总仓位。"],
                explanation="宏观分支基于多模块风控终端输出流动性、政策与风险状态。",
                symbol_scores={symbol: macro_score for symbol in self.stock_pool},
                metadata={
                    "report": report,
                    "branch_mode": "macro_terminal",
                    "reliability": 0.82,
                    "horizon_days": 20,
                },
            )
        except Exception as exc:
            self._log(f"宏观终端失败，使用增强降级统计: {exc}")

        combined = data_bundle.combined_frame()
        market_return = combined.groupby("date")["close"].mean().pct_change().dropna()

        # --- 增强降级回退：三维度宏观替代 ---
        # 1) 波动率分位数（低波 = 偏乐观）
        vol_20 = float(market_return.tail(20).std()) if len(market_return) >= 20 else 0.01
        vol_252 = float(market_return.tail(252).std()) if len(market_return) >= 252 else vol_20
        vol_percentile = min(vol_20 / (vol_252 + 1e-8), 2.0) / 2.0
        vol_score = 1.0 - vol_percentile  # 低波 = 积极

        # 2) 市场广度（涨跌比）
        latest_date = combined["date"].max() if "date" in combined.columns else None
        if latest_date is not None:
            latest_slice = combined[combined["date"] == latest_date]
            adv = int((latest_slice["close"].pct_change().fillna(0) >= 0).sum())
            dec = max(int(len(latest_slice) - adv), 1)
        else:
            adv, dec = 1, 1
        breadth_score = (adv - dec) / (adv + dec)

        # 3) 多周期动量结构
        ret_5d = float(market_return.tail(5).mean()) if len(market_return) >= 5 else 0.0
        ret_20d = float(market_return.tail(20).mean()) if len(market_return) >= 20 else 0.0
        ret_60d = float(market_return.tail(60).mean()) if len(market_return) >= 60 else ret_20d
        momentum_structure = 0.5 * ret_5d + 0.3 * ret_20d + 0.2 * ret_60d
        momentum_norm = _clamp(momentum_structure / (vol_20 + 1e-8), -1.0, 1.0)

        # 综合降级宏观得分
        score = _clamp(0.40 * vol_score + 0.35 * breadth_score + 0.25 * momentum_norm, -1.0, 1.0)

        # 集成 RegimeDetector 输出状态标签
        from regime_detector import RegimeDetector
        try:
            detector = RegimeDetector()
            regime_result = detector.detect(market_return)
            regime = regime_result.regime.value
        except Exception:
            regime = "低风险" if score > 0.2 else "高风险" if score < -0.2 else "中风险"

        signal = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
        return BranchResult(
            branch_name="macro",
            score=score,
            confidence=0.50,
            signals={
                "macro_score": score,
                "macro_regime": regime,
                "liquidity_signal": signal,
                "policy_signal": "增强降级统计",
                "risk_level": regime,
                "vol_score": round(vol_score, 3),
                "breadth_score": round(breadth_score, 3),
                "momentum_norm": round(momentum_norm, 3),
            },
            risks=["宏观分支使用增强降级统计（波动率+广度+动量），建议后续补齐真实宏观数据。"],
            explanation="宏观分支降级为市场波动率分位、广度和多周期动量的增强估计。",
            symbol_scores={symbol: score for symbol in self.stock_pool},
            metadata={
                "branch_mode": "macro_enhanced_degraded",
                "reliability": 0.55,
                "horizon_days": 20,
            },
            success=False,
        )

    # ---------------------------------------------------------------------
    # 风控层
    # ---------------------------------------------------------------------

    def _run_risk_layer(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any] | None = None,
    ) -> RiskLayerResult:
        """将统一分支结果适配到现有风控层。"""
        if calibrated_signals is None:
            calibrated_signals = self._calibrate_signals(data_bundle, branch_results)
        consensus_scores = self._aggregate_symbol_scores(calibrated_signals)
        current_prices = data_bundle.latest_prices()
        predicted_returns = self._aggregate_expected_returns(calibrated_signals)
        macro_overlay = self._macro_overlay_factor(calibrated_signals.get("macro"))
        predicted_returns = {
            symbol: value * macro_overlay
            for symbol, value in predicted_returns.items()
        }
        predicted_volatilities = self._estimate_symbol_volatilities(data_bundle)
        portfolio_returns = self._build_portfolio_returns(data_bundle)
        covariance_matrix = self._build_covariance_matrix(data_bundle, list(predicted_returns.keys()))
        macro_signal = self._macro_signal_from_branch(branch_results.get("macro"))
        return self.risk_layer.run_risk_management(
            portfolio_returns=portfolio_returns,
            predicted_returns=predicted_returns,
            predicted_volatilities=predicted_volatilities,
            current_prices=current_prices,
            macro_signal=macro_signal,
            conviction_scores=consensus_scores,
            covariance_matrix=covariance_matrix,
        )

    def _aggregate_symbol_scores(
        self,
        calibrated_signals: dict[str, Any],
    ) -> dict[str, float]:
        """将多个分支的 symbol_scores 聚合为统一共识分数。"""
        branch_weights = self._branch_tracker.get_adaptive_weights()
        self._log("分支权重: " + ", ".join(f"{k}={v:.2f}" for k, v in branch_weights.items()))
        return self._aggregate_symbol_metric(
            calibrated_signals=calibrated_signals,
            metric_name="symbol_convictions",
            clamp_range=(-1.0, 1.0),
            branch_weights=branch_weights,
        )

    def _aggregate_expected_returns(
        self,
        calibrated_signals: dict[str, Any],
    ) -> dict[str, float]:
        """汇总各分支的校准后预期收益。"""
        branch_weights = self._branch_tracker.get_adaptive_weights()
        return self._aggregate_symbol_metric(
            calibrated_signals=calibrated_signals,
            metric_name="symbol_expected_returns",
            clamp_range=(-0.3, 0.3),
            branch_weights=branch_weights,
        )

    def _aggregate_symbol_metric(
        self,
        calibrated_signals: dict[str, Any],
        metric_name: str,
        clamp_range: tuple[float, float],
        branch_weights: dict[str, float],
    ) -> dict[str, float]:
        """按统一链路聚合个股级信号。"""
        weighted_sum = {symbol: 0.0 for symbol in self.stock_pool}
        weight_sum = {symbol: 0.0 for symbol in self.stock_pool}
        lower, upper = clamp_range

        for branch_name, branch in calibrated_signals.items():
            if branch_name == "macro":
                continue
            branch_weight = branch_weights.get(branch_name, 0.1)
            metric = getattr(branch, metric_name, {})
            for symbol in self.stock_pool:
                value = float(metric.get(symbol, 0.0))
                confidence = float(branch.symbol_confidences.get(symbol, 0.0))
                reliability_penalty = max(float(branch.reliability), 0.05)
                weight = branch_weight * reliability_penalty * max(confidence, 0.05)
                weighted_sum[symbol] += value * weight
                weight_sum[symbol] += weight

        return {
            symbol: _clamp(weighted_sum[symbol] / (weight_sum[symbol] + 1e-8), lower, upper)
            for symbol in self.stock_pool
        }

    @staticmethod
    def _macro_overlay_factor(macro_signal: Any | None) -> float:
        """宏观 overlay 只影响总预期收益和总仓位，不参与个股排序。"""
        if macro_signal is None:
            return 1.0
        macro_score = float(getattr(macro_signal, "aggregate_expected_return", 0.0))
        return _clamp(1.0 + macro_score * 1.5, 0.65, 1.15)

    def _estimate_symbol_volatilities(self, data_bundle: UnifiedDataBundle) -> dict[str, float]:
        vols: dict[str, float] = {}
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                vols[symbol] = 0.25
                continue
            ret = df["close"].pct_change().dropna()
            vols[symbol] = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 5 else 0.25
            if not np.isfinite(vols[symbol]) or vols[symbol] <= 0:
                vols[symbol] = 0.25
        return vols

    def _build_portfolio_returns(self, data_bundle: UnifiedDataBundle) -> pd.Series:
        aligned: list[pd.Series] = []
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                continue
            ret = df.set_index("date")["close"].pct_change().rename(symbol)
            aligned.append(ret)
        if not aligned:
            return pd.Series([0.0] * 30)
        combined = pd.concat(aligned, axis=1).dropna(how="all").fillna(0.0)
        return combined.mean(axis=1).tail(252)

    def _build_returns_matrix(self, data_bundle: UnifiedDataBundle) -> pd.DataFrame:
        aligned: list[pd.Series] = []
        for symbol, df in data_bundle.symbol_data.items():
            if df.empty:
                continue
            aligned.append(df.set_index("date")["close"].pct_change().rename(symbol))
        if not aligned:
            return pd.DataFrame()
        return pd.concat(aligned, axis=1).sort_index().fillna(0.0)

    def _build_covariance_matrix(
        self,
        data_bundle: UnifiedDataBundle,
        symbols: list[str],
    ) -> pd.DataFrame | None:
        returns_matrix = self._build_returns_matrix(data_bundle)
        if returns_matrix.empty or not symbols:
            return None
        usable = [symbol for symbol in symbols if symbol in returns_matrix.columns]
        if not usable:
            return None
        cov = PortfolioOptimizer.build_cov_from_returns(returns_matrix, usable)
        return pd.DataFrame(cov, index=usable, columns=usable)

    def _macro_signal_from_branch(self, branch: BranchResult | None) -> str:
        if branch is None:
            return "🟡"
        signal = branch.signals.get("liquidity_signal")
        if signal in {"🔴", "🟡", "🟢", "🔵"}:
            return str(signal)
        if branch.score >= 0.5:
            return "🟢"
        if branch.score <= -0.3:
            return "🔴"
        return "🟡"

    # ---------------------------------------------------------------------
    # 集成裁判层
    # ---------------------------------------------------------------------

    def _run_ensemble_layer(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        risk_result: RiskLayerResult,
        calibrated_signals: dict[str, Any] | None = None,
    ) -> PortfolioStrategy:
        """生成最终组合级策略。"""
        if calibrated_signals is None:
            calibrated_signals = self._calibrate_signals(data_bundle, branch_results)
        branch_consensus = {
            name: round(signal.aggregate_expected_return, 4)
            for name, signal in calibrated_signals.items()
        }
        raw_symbol_convictions = self._aggregate_symbol_scores(calibrated_signals)
        expected_returns = self._aggregate_expected_returns(calibrated_signals)
        synthetic_symbols = set(data_bundle.synthetic_symbols())
        degraded_symbols = set(data_bundle.degraded_symbols())
        macro_overlay = self._macro_overlay_factor(calibrated_signals.get("macro"))
        candidate_symbols = [
            symbol
            for symbol, score in sorted(raw_symbol_convictions.items(), key=lambda item: item[1], reverse=True)
            if score > 0 and (self.allow_synthetic_for_research or symbol not in synthetic_symbols)
        ][: min(8, len(data_bundle.real_symbols()) or len(self.stock_pool))]

        research_score = _safe_mean(list(branch_consensus.values()))
        disagreement = np.std([branch.score for branch in branch_results.values()]) if branch_results else 0.0
        base_exposure = 1 - risk_result.position_sizing.cash_ratio
        exposure_penalty = _clamp(disagreement, 0.0, 0.35)
        target_exposure = _clamp(base_exposure * (1 - exposure_penalty), 0.1, 0.95)
        if synthetic_symbols and not self.allow_synthetic_for_research:
            target_exposure = min(target_exposure, 0.45)
        if risk_result.risk_level == "danger":
            target_exposure = min(target_exposure, 0.25)
        elif risk_result.risk_level == "warning":
            target_exposure = min(target_exposure, 0.55)
        if not data_bundle.real_symbols():
            target_exposure = 0.0
        target_exposure = _clamp(target_exposure * macro_overlay, 0.0, 0.95)

        # 分级 quorum 约束
        quorum_max = getattr(self, "_quorum_max_exposure", 0.95)
        quorum_rel = getattr(self, "_quorum_reliability", 1.0)
        target_exposure = min(target_exposure, quorum_max)
        if quorum_rel <= 0.0:
            target_exposure = 0.0

        style_bias = self._infer_style_bias(branch_results, research_score)
        sector_preferences = self._infer_sector_preferences(style_bias, branch_results.get("macro"))
        stop_loss_policy = {
            "portfolio_drawdown_limit": abs(self.risk_layer.max_drawdown_limit),
            "default_stop_loss_pct": abs(self.risk_layer.stop_loss_pct),
            "default_take_profit_pct": self.risk_layer.take_profit_pct,
        }
        position_limits = self._optimize_positions(
            data_bundle=data_bundle,
            candidate_symbols=candidate_symbols,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
            target_exposure=target_exposure,
            fallback_weights=risk_result.position_sizing.risk_adjusted_weights,
        )

        execution_notes = [
            f"研究共识得分 {research_score:+.2f}，分支分歧度 {disagreement:.2f}。",
            f"当前风险等级为 {risk_result.risk_level}，目标总仓位调整为 {target_exposure:.0%}。",
        ]
        if synthetic_symbols:
            execution_notes.append(
                "存在降级/模拟数据标的，已默认排除其进入候选池。"
                if not self.allow_synthetic_for_research else
                "存在降级/模拟数据标的，当前以研究模式允许其保留。"
            )
        for branch in branch_results.values():
            execution_notes.extend(branch.risks[:1])
        research_mode = "production"
        research_only_reason = ""
        if quorum_rel <= 0.0:
            research_mode = "research_only"
            research_only_reason = "quorum_insufficient"
            target_exposure = 0.0
            execution_notes.append("分支成功数不足（≤1），进入 research_only 模式。")
        elif synthetic_symbols or any(not branch.success for branch in branch_results.values()):
            research_mode = "degraded"
        if not candidate_symbols:
            execution_notes.append("暂无明确优势标的，建议以现金和防御仓位为主。")
            if not data_bundle.real_symbols():
                research_mode = "research_only"
                research_only_reason = "no_real_candidates_after_provenance_filter"
                target_exposure = 0.0
                position_limits = {}

        trade_recommendations = self._build_trade_recommendations(
            data_bundle=data_bundle,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            risk_result=risk_result,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            expected_returns=expected_returns,
            convictions=raw_symbol_convictions,
        )

        return PortfolioStrategy(
            target_exposure=target_exposure,
            style_bias=style_bias,
            sector_preferences=sector_preferences,
            candidate_symbols=candidate_symbols,
            position_limits=position_limits,
            stop_loss_policy=stop_loss_policy,
            execution_notes=execution_notes[:8],
            branch_consensus=branch_consensus,
            symbol_convictions=raw_symbol_convictions,
            risk_summary={
                "risk_level": risk_result.risk_level,
                "volatility": risk_result.risk_metrics.volatility,
                "max_drawdown": risk_result.risk_metrics.max_drawdown,
                "warnings": risk_result.risk_warnings,
                "planned_investment": self.total_capital * target_exposure,
                "cash_reserve": self.total_capital * (1 - target_exposure),
                "max_single_position": self.risk_layer.max_position_size,
            },
            provenance_summary={
                "synthetic_symbols": sorted(synthetic_symbols),
                "degraded_symbols": sorted(degraded_symbols),
                "branch_modes": {
                    name: str(signal.branch_mode) for name, signal in calibrated_signals.items()
                },
                "branch_reliability": {
                    name: float(signal.reliability) for name, signal in calibrated_signals.items()
                },
                "macro_overlay": macro_overlay,
                "research_only_reason": research_only_reason,
                "symbol_provenance": data_bundle.symbol_provenance(),
            },
            research_mode=research_mode,
            trade_recommendations=trade_recommendations,
        )

    def _optimize_positions(
        self,
        data_bundle: UnifiedDataBundle,
        candidate_symbols: list[str],
        expected_returns: dict[str, float],
        convictions: dict[str, float],
        target_exposure: float,
        fallback_weights: dict[str, float],
    ) -> dict[str, float]:
        """使用组合优化器生成最终仓位上限。"""
        if not candidate_symbols or target_exposure <= 0:
            return {}
        returns_matrix = self._build_returns_matrix(data_bundle)
        if returns_matrix.empty:
            return {
                symbol: round(min(fallback_weights.get(symbol, 0.0), self.risk_layer.max_position_size), 4)
                for symbol in candidate_symbols
            }
        optimizer = PortfolioOptimizer(
            max_position=self.risk_layer.max_position_size,
            method="risk_parity",
        )
        covariance_matrix = PortfolioOptimizer.build_cov_from_returns(returns_matrix, candidate_symbols)
        optimized = optimizer.optimize(
            candidate_symbols,
            expected_returns=expected_returns,
            cov_matrix=covariance_matrix,
            conviction_scores=convictions,
            base_position=target_exposure,
        )
        return {
            symbol: round(min(weight, self.risk_layer.max_position_size), 4)
            for symbol, weight in optimized.items()
            if weight > 0
        }

    def _build_trade_recommendations(
        self,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: RiskLayerResult,
        candidate_symbols: list[str],
        position_limits: dict[str, float],
        expected_returns: dict[str, float],
        convictions: dict[str, float],
    ) -> list[TradeRecommendation]:
        """把组合结论展开为单票可执行交易建议。"""
        if not candidate_symbols or not position_limits:
            return []

        current_prices = data_bundle.latest_prices()
        stop_levels = risk_result.stop_loss_take_profit.stop_loss_levels
        take_profit_levels = risk_result.stop_loss_take_profit.take_profit_levels
        symbol_provenance = data_bundle.symbol_provenance()
        recommendations: list[TradeRecommendation] = []

        for symbol in candidate_symbols:
            target_weight = float(position_limits.get(symbol, 0.0))
            current_price = float(current_prices.get(symbol, 0.0))
            if target_weight <= 0 or current_price <= 0:
                continue

            df = data_bundle.symbol_data.get(symbol, pd.DataFrame())
            support, resistance, atr, trend = self._derive_price_levels(df, current_price)
            model_expected_return = max(float(expected_returns.get(symbol, 0.0)), 0.0)
            conviction = float(convictions.get(symbol, 0.0))
            entry_price = self._suggest_entry_price(
                current_price=current_price,
                support_price=support,
                atr_value=atr,
                conviction=conviction,
                trend_regime=trend,
            )
            buy_zone_low = max(0.01, min(entry_price, support) * 0.995)
            buy_zone_high = max(entry_price, min(current_price * 1.01, entry_price + atr * 0.35))

            stop_loss_price = float(
                stop_levels.get(symbol, current_price * (1 + self.risk_layer.stop_loss_pct))
            )
            if support > 0:
                stop_loss_price = min(stop_loss_price, support * 0.99)
            stop_loss_price = _clamp(stop_loss_price, current_price * 0.75, current_price * 0.985)

            target_price = max(
                float(take_profit_levels.get(symbol, current_price * (1 + self.risk_layer.take_profit_pct))),
                current_price * (1 + max(model_expected_return, 0.08)),
                max(resistance, current_price + atr * 1.5),
            )
            target_price = max(target_price, entry_price * 1.05)

            lot_size = self._market_lot_size(symbol)
            budget_amount = self.total_capital * target_weight
            suggested_shares = int(budget_amount // max(entry_price, 0.01) // lot_size) * lot_size
            suggested_amount = suggested_shares * entry_price
            suggested_weight = suggested_amount / self.total_capital if self.total_capital > 0 else 0.0

            confidence = _safe_mean(
                [
                    float(signal.symbol_confidences.get(symbol, 0.0))
                    for name, signal in calibrated_signals.items()
                    if name != "macro"
                ]
            )
            horizon_days = int(
                max(
                    5,
                    round(
                        _safe_mean(
                            [
                                float(signal.horizon_days)
                                for name, signal in calibrated_signals.items()
                                if name != "macro"
                            ]
                        )
                    ),
                )
            )
            branch_scores = {
                name: float(branch.symbol_scores.get(symbol, branch.score))
                for name, branch in branch_results.items()
            }
            branch_expected_returns = {
                name: float(
                    signal.symbol_expected_returns.get(symbol, signal.aggregate_expected_return)
                )
                for name, signal in calibrated_signals.items()
            }
            branch_positive_count = sum(1 for score in branch_scores.values() if score > 0.05)

            expected_upside = target_price / max(entry_price, 0.01) - 1
            expected_drawdown = max(0.0, 1 - stop_loss_price / max(entry_price, 0.01))
            risk_reward_ratio = expected_upside / (expected_drawdown + 1e-8)
            risk_flags = self._collect_symbol_risks(
                symbol=symbol,
                data_bundle=data_bundle,
                branch_results=branch_results,
                branch_positive_count=branch_positive_count,
                confidence=confidence,
                expected_drawdown=expected_drawdown,
                trend_regime=trend,
            )
            position_management = self._build_position_management(
                symbol=symbol,
                suggested_weight=suggested_weight,
                suggested_amount=suggested_amount,
                buy_zone_low=buy_zone_low,
                buy_zone_high=buy_zone_high,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                branch_positive_count=branch_positive_count,
            )

            action = "buy" if suggested_shares > 0 else "watch"
            if suggested_shares == 0:
                risk_flags.append("按当前资金与整手规则暂无法成交，需提高单票预算或等待回落。")

            recommendations.append(
                TradeRecommendation(
                    symbol=symbol,
                    action=action,
                    category=self.market,
                    current_price=round(current_price, 2),
                    recommended_entry_price=round(entry_price, 2),
                    entry_price_range={
                        "low": round(buy_zone_low, 2),
                        "high": round(buy_zone_high, 2),
                    },
                    target_price=round(target_price, 2),
                    stop_loss_price=round(stop_loss_price, 2),
                    support_price=round(support, 2),
                    resistance_price=round(resistance, 2),
                    model_expected_return=round(model_expected_return, 4),
                    expected_upside=round(expected_upside, 4),
                    expected_drawdown=round(expected_drawdown, 4),
                    risk_reward_ratio=round(risk_reward_ratio, 2),
                    suggested_weight=round(suggested_weight, 4),
                    suggested_amount=round(suggested_amount, 2),
                    suggested_shares=suggested_shares,
                    lot_size=lot_size,
                    confidence=round(confidence, 4),
                    consensus_score=round(conviction, 4),
                    branch_positive_count=branch_positive_count,
                    branch_scores={k: round(v, 4) for k, v in branch_scores.items()},
                    branch_expected_returns={
                        k: round(v, 4) for k, v in branch_expected_returns.items()
                    },
                    risk_flags=risk_flags[:4],
                    position_management=position_management,
                    horizon_days=horizon_days,
                    trend_regime=trend,
                    data_source_status=str(
                        symbol_provenance.get(symbol, {}).get("data_source_status", "unknown")
                    ),
                )
            )

        return sorted(
            recommendations,
            key=lambda item: (
                item.suggested_weight,
                item.consensus_score,
                item.expected_upside,
            ),
            reverse=True,
        )

    def _derive_price_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> tuple[float, float, float, str]:
        """基于近期价格行为估计支撑、阻力、ATR 与趋势状态。"""
        if df is None or df.empty or "close" not in df.columns:
            return current_price * 0.96, current_price * 1.08, max(current_price * 0.02, 0.01), "震荡"

        recent = df.sort_values("date").tail(60).copy()
        close = recent["close"].astype(float)
        high = recent["high"].astype(float) if "high" in recent.columns else close
        low = recent["low"].astype(float) if "low" in recent.columns else close
        prev_close = close.shift(1).fillna(close)
        true_range = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )

        atr = float(true_range.tail(14).mean()) if len(true_range) >= 2 else current_price * 0.02
        atr = max(atr, current_price * 0.005)
        ma10 = float(close.tail(10).mean()) if len(close) >= 10 else current_price
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else ma10
        low20 = float(low.tail(20).min()) if len(low) >= 5 else current_price * 0.96
        high20 = float(high.tail(20).max()) if len(high) >= 5 else current_price * 1.06

        support = min(current_price, max(low20, ma20 - 0.75 * atr))
        resistance = max(high20, current_price + 1.5 * atr)

        if current_price > ma10 > ma20:
            trend = "上行"
        elif current_price < ma10 < ma20:
            trend = "承压"
        else:
            trend = "震荡"

        return support, resistance, atr, trend

    @staticmethod
    def _suggest_entry_price(
        current_price: float,
        support_price: float,
        atr_value: float,
        conviction: float,
        trend_regime: str,
    ) -> float:
        """结合趋势和回撤空间给出执行参考买点。"""
        conviction = max(conviction, 0.0)
        if trend_regime == "上行":
            raw_entry = current_price - max(atr_value * 0.45, current_price * 0.01)
        elif trend_regime == "承压":
            raw_entry = min(current_price * 0.98, support_price * 1.01)
        else:
            raw_entry = (current_price + support_price) / 2

        lower = max(0.01, support_price * (1.002 if conviction >= 0.25 else 0.995))
        upper = current_price * 1.005
        return round(_clamp(raw_entry, lower, upper), 2)

    def _collect_symbol_risks(
        self,
        symbol: str,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        branch_positive_count: int,
        confidence: float,
        expected_drawdown: float,
        trend_regime: str,
    ) -> list[str]:
        """收集个股级风险观察。"""
        risk_flags: list[str] = []
        symbol_meta = data_bundle.symbol_provenance().get(symbol, {})
        if symbol_meta.get("data_source_status") != "real":
            risk_flags.append("数据链路并非纯真实行情，当前建议仅作研究参考。")
        if branch_positive_count < 3:
            risk_flags.append("五路分支共识不足，执行上宜降级为观察或轻仓试错。")
        if confidence < 0.35:
            risk_flags.append("综合置信度一般，建议严格分批建仓。")
        if expected_drawdown > 0.10:
            risk_flags.append("止损空间偏大，单票仓位不宜激进。")
        if trend_regime == "承压":
            risk_flags.append("短线趋势仍承压，优先等待企稳信号。")

        llm_branch = branch_results.get("llm_debate")
        if llm_branch is not None:
            llm_risks = llm_branch.signals.get("key_risks", {}).get(symbol, [])
            risk_flags.extend(str(item) for item in llm_risks[:2])

        intelligence_branch = branch_results.get("intelligence")
        if intelligence_branch is not None:
            alerts = intelligence_branch.signals.get("alerts", [])
            risk_flags.extend(str(item) for item in alerts if symbol in str(item))

        deduped: list[str] = []
        for item in risk_flags:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _build_position_management(
        symbol: str,
        suggested_weight: float,
        suggested_amount: float,
        buy_zone_low: float,
        buy_zone_high: float,
        target_price: float,
        stop_loss_price: float,
        branch_positive_count: int,
    ) -> list[str]:
        """生成简洁的分批建仓与止盈止损动作。"""
        return [
            (
                f"{symbol} 建议总仓位 {suggested_weight:.1%}，预算约 ¥{suggested_amount:,.0f}，"
                "首次建仓 60%，余下 40% 仅在回踩买点区间时补齐。"
            ),
            (
                f"买入区间参考 ¥{buy_zone_low:.2f}-¥{buy_zone_high:.2f}；"
                f"若跌破 ¥{stop_loss_price:.2f} 则执行止损。"
            ),
            (
                f"若上行至 ¥{target_price:.2f} 附近先兑现 50%，"
                f"剩余仓位按分支共识 {branch_positive_count}/5 持续跟踪。"
            ),
        ]

    def _market_lot_size(self, symbol: str) -> int:
        """返回市场整手单位。"""
        if self.market == "CN":
            return 100
        return 1

    def _infer_style_bias(
        self,
        branch_results: dict[str, BranchResult],
        research_score: float,
    ) -> str:
        quant_score = branch_results.get("quant", BranchResult("quant")).score
        intelligence_score = branch_results.get("intelligence", BranchResult("intelligence")).score
        macro_score = branch_results.get("macro", BranchResult("macro")).score
        if research_score < -0.2 or macro_score < -0.2:
            return "防御"
        if intelligence_score > 0.2 and quant_score < 0:
            return "高质量"
        if quant_score > 0.2:
            return "成长"
        return "均衡"

    def _infer_sector_preferences(
        self,
        style_bias: str,
        macro_branch: BranchResult | None,
    ) -> list[str]:
        if style_bias == "成长":
            prefs = ["科技成长", "先进制造", "景气消费"]
        elif style_bias == "高质量":
            prefs = ["高股息", "央国企龙头", "现金流稳健"]
        elif style_bias == "防御":
            prefs = ["公用事业", "必选消费", "红利低波"]
        else:
            prefs = ["均衡配置", "盈利改善", "估值合理"]
        if macro_branch and macro_branch.score < -0.2:
            return prefs[:2] + ["现金管理"]
        return prefs

    # ---------------------------------------------------------------------
    # 报告
    # ---------------------------------------------------------------------

    def _build_markdown_report(self, result: ResearchPipelineResult) -> str:
        strategy = result.final_strategy
        risk_result = result.risk_result
        lines = [
            "# 五路并行研究投资策略报告",
            "",
            f"**市场**: {result.data_bundle.market}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 组合级策略结论",
            f"- 目标总仓位: {strategy.target_exposure:.0%}",
            f"- 风格偏好: {strategy.style_bias}",
            f"- 候选标的: {', '.join(strategy.candidate_symbols) if strategy.candidate_symbols else '暂无'}",
            f"- 行业偏好: {', '.join(strategy.sector_preferences) if strategy.sector_preferences else '均衡配置'}",
            "",
            "## 五大研究分支",
        ]

        for name in self.BRANCH_ORDER:
            branch = result.branch_results.get(name)
            if branch is None:
                continue
            lines.extend(
                [
                    f"### {branch.branch_name}",
                    f"- 分支得分: {branch.score:+.2f}",
                    f"- 置信度: {branch.confidence:.0%}",
                    f"- 分支模式: {branch.metadata.get('branch_mode', 'unknown')}",
                    f"- 可靠度: {float(branch.metadata.get('reliability', 0.0)):.0%}",
                    f"- 说明: {branch.explanation}",
                ]
            )
            if branch.risks:
                lines.append(f"- 关键风险: {'；'.join(branch.risks[:3])}")
            top_symbols = sorted(branch.symbol_scores.items(), key=lambda item: item[1], reverse=True)[:3]
            if top_symbols:
                lines.append(
                    "- 高优先级标的: "
                    + " / ".join([f"{symbol}({score:+.2f})" for symbol, score in top_symbols])
                )
            lines.append("")

        if risk_result is not None:
            lines.extend(
                [
                    "## 风控层",
                    f"- 风险等级: {risk_result.risk_level}",
                    f"- 年化波动率: {risk_result.risk_metrics.volatility:.2%}",
                    f"- 最大回撤: {risk_result.risk_metrics.max_drawdown:.2%}",
                    f"- 夏普比率: {risk_result.risk_metrics.sharpe_ratio:.2f}",
                    f"- 计划投入资金: ¥{strategy.risk_summary.get('planned_investment', 0.0):,.0f}",
                    f"- 计划保留现金: ¥{strategy.risk_summary.get('cash_reserve', 0.0):,.0f}",
                    "",
                ]
            )

        lines.extend(
            [
                "## 可信度与降级状态",
                f"- 研究模式: {strategy.research_mode}",
                f"- 使用模拟/降级数据的标的: {', '.join(strategy.provenance_summary.get('synthetic_symbols', [])) or '无'}",
                f"- 宏观 Overlay: {float(strategy.provenance_summary.get('macro_overlay', 1.0)):.2f}",
            ]
        )
        reason = strategy.provenance_summary.get("research_only_reason", "")
        if reason:
            lines.append(f"- 仅研究参考原因: {reason}")
        lines.append("- 分支模式与可靠度:")
        for branch_name in self.BRANCH_ORDER:
            if branch_name not in strategy.provenance_summary.get("branch_modes", {}):
                continue
            lines.append(
                "  "
                + f"{branch_name}: "
                + f"{strategy.provenance_summary['branch_modes'][branch_name]} / "
                + f"{float(strategy.provenance_summary.get('branch_reliability', {}).get(branch_name, 0.0)):.0%}"
            )
        for symbol, meta in sorted(result.data_bundle.symbol_provenance().items()):
            status = meta.get("data_source_status", "unknown")
            reason = meta.get("degraded_reason", "")
            reliability = float(meta.get("reliability", 0.0))
            suffix = f"；原因: {reason}" if reason else ""
            lines.append(f"- {symbol}: {status}，可靠度 {reliability:.0%}{suffix}")
        lines.append("")

        if strategy.trade_recommendations:
            lines.append("## 可执行交易计划")
            lines.append("| 标的 | 建议仓位 | 建议金额 | 建议股数 | 现价 | 建议买入 | 目标价 | 止损价 | 预期空间 | 五路支持 |")
            lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for recommendation in strategy.trade_recommendations:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            recommendation.symbol,
                            f"{recommendation.suggested_weight:.1%}",
                            f"¥{recommendation.suggested_amount:,.0f}",
                            f"{recommendation.suggested_shares:,}",
                            f"¥{recommendation.current_price:.2f}",
                            f"¥{recommendation.recommended_entry_price:.2f}",
                            f"¥{recommendation.target_price:.2f}",
                            f"¥{recommendation.stop_loss_price:.2f}",
                            f"{recommendation.expected_upside:.1%}",
                            f"{recommendation.branch_positive_count}/5",
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.append("## 个股执行与风险观察")
            for recommendation in strategy.trade_recommendations[:5]:
                lines.append(f"### {recommendation.symbol}")
                lines.append(
                    f"- 买入区间: ¥{recommendation.entry_price_range.get('low', 0.0):.2f}"
                    f" - ¥{recommendation.entry_price_range.get('high', 0.0):.2f}"
                )
                lines.append(
                    f"- 建议配置: {recommendation.suggested_weight:.1%}"
                    f" / ¥{recommendation.suggested_amount:,.0f}"
                    f" / {recommendation.suggested_shares:,} 股"
                )
                lines.append(
                    f"- 目标/止损: ¥{recommendation.target_price:.2f}"
                    f" / ¥{recommendation.stop_loss_price:.2f}"
                )
                lines.append(
                    f"- 五路共识: {recommendation.consensus_score:+.2f}，"
                    f"支持分支 {recommendation.branch_positive_count}/5，"
                    f"综合置信度 {recommendation.confidence:.0%}"
                )
                if recommendation.risk_flags:
                    lines.append(f"- 风险观察: {'；'.join(recommendation.risk_flags[:3])}")
                if recommendation.position_management:
                    lines.append(f"- 仓位管理: {'；'.join(recommendation.position_management[:2])}")
                lines.append("")

        lines.append("## 执行建议")
        for note in strategy.execution_notes:
            lines.append(f"- {note}")

        lines.append("")
        lines.append("## 风控约束")
        lines.append(f"- 单票上限: {self.risk_layer.max_position_size:.0%}")
        lines.append(f"- 组合回撤红线: {abs(self.risk_layer.max_drawdown_limit):.0%}")
        lines.append(f"- 默认止损: {abs(self.risk_layer.stop_loss_pct):.0%}")

        return "\n".join(lines)

    @staticmethod
    def _default_branch_mode(branch_name: str) -> str:
        return {
            "kline": "kline_heuristic",
            "quant": "alpha_research",
            "llm_debate": "structured_research_debate",
            "intelligence": "structured_intelligence_fusion",
            "macro": "macro_terminal",
        }.get(branch_name, "unknown")

    @staticmethod
    def _bundle_data_source_status(symbol_provenance: dict[str, dict[str, Any]]) -> str:
        statuses = {meta.get("data_source_status", "unknown") for meta in symbol_provenance.values()}
        if not statuses:
            return "unknown"
        if statuses == {"real"}:
            return "real"
        if statuses <= {"synthetic_fallback"}:
            return "synthetic_only"
        return "mixed"

    def _bundle_research_mode(self, symbol_provenance: dict[str, dict[str, Any]]) -> str:
        if not symbol_provenance:
            return "research_only"
        if all(meta.get("is_synthetic") for meta in symbol_provenance.values()):
            return "research_only"
        if any(meta.get("is_synthetic") for meta in symbol_provenance.values()):
            return "degraded"
        return "production"

    @staticmethod
    def _branch_data_source_status(data_bundle: UnifiedDataBundle) -> str:
        if not data_bundle.symbol_provenance():
            return "unknown"
        statuses = {meta.get("data_source_status", "unknown") for meta in data_bundle.symbol_provenance().values()}
        if statuses == {"real"}:
            return "real"
        if statuses <= {"synthetic_fallback"}:
            return "synthetic_only"
        return "mixed"
