#!/usr/bin/env python3
"""
Risk Management Layer - 风控层 (第6层)

功能:
1. 组合风控 - 波动率控制、最大回撤控制
2. 仓位管理 - 基于风险信号的动态仓位调整
3. 止损止盈 - 个股和组合级别的止损止盈
4. 风险分解 - Barra风格因子风险分解
5. 压力测试 - 极端行情模拟
6. 风险预算 - 基于风险预算的资产配置
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from quant_investor.logger import get_logger
from quant_investor.exceptions import RiskCalculationError


@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float = 0.0                    # 波动率
    max_drawdown: float = 0.0                  # 最大回撤
    var_95: float = 0.0                        # 95% VaR
    cvar_95: float = 0.0                       # 95% CVaR
    sharpe_ratio: float = 0.0                  # 夏普比率
    sortino_ratio: float = 0.0                 # 索提诺比率
    beta: float = 0.0                          # Beta
    alpha: float = 0.0                         # Alpha
    tracking_error: float = 0.0                # 跟踪误差
    information_ratio: float = 0.0             # 信息比率


@dataclass
class PositionSizing:
    """仓位管理结果"""
    target_positions: Dict[str, float] = field(default_factory=dict)         # 目标仓位
    risk_adjusted_weights: Dict[str, float] = field(default_factory=dict)    # 风险调整权重
    cash_ratio: float = 0.0                    # 现金比例
    leverage: float = 1.0                      # 杠杆倍数


@dataclass
class StopLossTakeProfit:
    """止损止盈设置"""
    stop_loss_levels: Dict[str, float] = field(default_factory=dict)         # 止损价位
    take_profit_levels: Dict[str, float] = field(default_factory=dict)       # 止盈价位
    trailing_stops: Dict[str, float] = field(default_factory=dict)           # 跟踪止损


@dataclass
class RiskLayerResult:
    """风控层结果"""
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    stop_loss_take_profit: StopLossTakeProfit = field(default_factory=StopLossTakeProfit)
    risk_decomposition: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    risk_budget_allocation: Dict[str, float] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    risk_level: str = "normal"                 # normal, warning, danger


class RiskManagementLayer:
    """
    风控层 - 组合风控与仓位管理
    """
    
    def __init__(
        self,
        max_position_size: float = 0.2,         # 单票最大仓位
        max_sector_exposure: float = 0.3,       # 行业最大暴露
        max_drawdown_limit: float = -0.15,      # 最大回撤限制
        target_volatility: float = 0.2,         # 目标波动率
        stop_loss_pct: float = -0.08,           # 止损比例
        take_profit_pct: float = 0.15,          # 止盈比例
        risk_free_rate: float = 0.03,           # 无风险利率
        verbose: bool = True
    ):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown_limit = max_drawdown_limit
        self.target_volatility = target_volatility
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
        self._logger = get_logger("RiskManagementLayer", verbose)
        self.result = RiskLayerResult()

    def _log(self, msg: str) -> None:
        self._logger.info(msg)
    
    # ==================== 风险指标计算 ====================
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        计算风险指标
        """
        self._log("计算风险指标...")

        metrics = RiskMetrics()

        if len(returns) < 2:
            return metrics

        try:
            # 年化波动率
            metrics.volatility = returns.std() * np.sqrt(252)

            # 最大回撤
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            metrics.max_drawdown = drawdown.min()

            # VaR / CVaR
            metrics.var_95 = float(np.percentile(returns, 5))
            metrics.cvar_95 = returns[returns <= metrics.var_95].mean()

            # 夏普比率
            excess_returns = returns.mean() * 252 - self.risk_free_rate
            if metrics.volatility > 0:
                metrics.sharpe_ratio = excess_returns / metrics.volatility

            # 索提诺比率
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            if downside_std > 0:
                metrics.sortino_ratio = excess_returns / downside_std

            # Beta / Alpha (如果有基准)
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                if benchmark_variance > 0:
                    metrics.beta = covariance / benchmark_variance
                    metrics.alpha = returns.mean() * 252 - self.risk_free_rate - metrics.beta * (benchmark_returns.mean() * 252 - self.risk_free_rate)

                # 跟踪误差
                tracking_diff = returns - benchmark_returns
                metrics.tracking_error = tracking_diff.std() * np.sqrt(252)

                # 信息比率
                if metrics.tracking_error > 0:
                    metrics.information_ratio = (returns.mean() - benchmark_returns.mean()) * 252 / metrics.tracking_error

        except Exception as e:
            raise RiskCalculationError("risk_metrics", str(e)) from e

        self._log(f"风险指标: 波动率={metrics.volatility:.2%}, 最大回撤={metrics.max_drawdown:.2%}, 夏普={metrics.sharpe_ratio:.2f}")

        return metrics
    
    # ==================== 仓位管理 ====================
    
    def calculate_position_sizing(
        self,
        predicted_returns: Dict[str, float],
        predicted_volatilities: Dict[str, float],
        risk_signals: Dict[str, str],
        macro_signal: str = "🟢",
        conviction_scores: Optional[Dict[str, float]] = None,
        conviction_alpha: float = 1.5,
    ) -> PositionSizing:
        """
        基于风险的仓位管理（含信号置信度直接映射 - 改进八）

        公式：raw_weight_i = inv_vol_i * max(score_i, 0)^alpha
          alpha=1.0 线性映射，alpha=1.5 高置信股放大（默认），alpha=2.0 更激进

        Args:
            predicted_returns:    预期收益率
            predicted_volatilities: 预期波动率
            risk_signals:         个股风险信号
            macro_signal:         宏观信号颜色
            conviction_scores:    分支综合置信分 [-1,1]，若提供则与逆波动率联合加权
            conviction_alpha:     置信分幂函数指数（默认 1.5）
        """
        self._log("计算风险调整仓位...")

        sizing = PositionSizing()

        # 基于宏观信号的基础仓位
        base_position_map = {
            "🔴": 0.3,      # 高风险 - 30%仓位
            "🟡": 0.5,      # 中风险 - 50%仓位
            "🟢": 0.8,      # 低风险 - 80%仓位
            "🔵": 1.0       # 极低风险 - 100%仓位
        }
        base_position = base_position_map.get(macro_signal, 0.5)

        # 计算基础权重：逆波动率 × 置信分凸性映射
        raw_weights: Dict[str, float] = {}
        for symbol, vol in predicted_volatilities.items():
            inv_vol = 1.0 / max(vol, 0.01)
            if conviction_scores and symbol in conviction_scores:
                score = max(conviction_scores[symbol], 0.0)
                # 幂函数映射：高置信分股票权重放大
                conviction_factor = score ** conviction_alpha if score > 0 else 0.0
                raw_weights[symbol] = inv_vol * conviction_factor
            else:
                raw_weights[symbol] = inv_vol

        # 若所有权重为零（全部置信分 ≤ 0），回退到纯反波动率
        if sum(raw_weights.values()) < 1e-8:
            raw_weights = {s: 1.0 / max(v, 0.01) for s, v in predicted_volatilities.items()}

        total_raw = sum(raw_weights.values())
        if total_raw > 0:
            for symbol in raw_weights:
                sizing.risk_adjusted_weights[symbol] = (
                    raw_weights[symbol] / total_raw * base_position
                )

        # 应用最大仓位限制
        for symbol in sizing.risk_adjusted_weights:
            sizing.risk_adjusted_weights[symbol] = min(
                sizing.risk_adjusted_weights[symbol],
                self.max_position_size
            )

        # 重新归一化
        total_weight = sum(sizing.risk_adjusted_weights.values())
        if total_weight > 0:
            for symbol in sizing.risk_adjusted_weights:
                sizing.risk_adjusted_weights[symbol] = (
                    sizing.risk_adjusted_weights[symbol] / total_weight * base_position
                )

        sizing.cash_ratio = 1 - base_position
        sizing.leverage = 1.0 if base_position <= 1.0 else base_position

        self._log(f"仓位管理: 基础仓位={base_position:.0%}, 现金比例={sizing.cash_ratio:.0%}"
                  + (f", 置信加权(α={conviction_alpha})" if conviction_scores else ""))

        return sizing
    
    # ==================== 止损止盈 ====================
    
    def calculate_stop_loss_take_profit(
        self,
        current_prices: Dict[str, float],
        entry_prices: Optional[Dict[str, float]] = None,
        atr_values: Optional[Dict[str, float]] = None
    ) -> StopLossTakeProfit:
        """
        计算止损止盈价位
        """
        self._log("计算止损止盈...")
        
        stp = StopLossTakeProfit()
        
        for symbol, current_price in current_prices.items():
            # 固定百分比止损止盈
            stp.stop_loss_levels[symbol] = current_price * (1 + self.stop_loss_pct)
            stp.take_profit_levels[symbol] = current_price * (1 + self.take_profit_pct)
            
            # ATR-based 跟踪止损 (如果有ATR)
            if atr_values and symbol in atr_values:
                atr = atr_values[symbol]
                stp.trailing_stops[symbol] = current_price - 2 * atr  # 2倍ATR跟踪止损
        
        return stp
    
    # ==================== 风险分解 ====================
    
    def risk_decomposition(
        self,
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Barra风格因子风险分解
        
        将风险分解为不同因子来源
        """
        self._log("风险分解...")
        
        decomposition = {}
        
        # 假设因子暴露已知，计算各因子贡献的风险
        for factor in factor_exposures.columns:
            factor_return = returns.corrwith(factor_exposures[factor])
            factor_risk = factor_return.std() * np.sqrt(252)
            decomposition[factor] = factor_risk
        
        # 计算系统性风险和特异性风险
        total_risk = returns.std().mean() * np.sqrt(252)
        systematic_risk = sum(decomposition.values())
        idiosyncratic_risk = max(0, total_risk - systematic_risk)
        
        decomposition['systematic'] = systematic_risk
        decomposition['idiosyncratic'] = idiosyncratic_risk
        
        return decomposition
    
    # ==================== 压力测试 ====================
    
    def stress_test(
        self,
        portfolio_returns: pd.Series,
        scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        压力测试
        
        模拟极端行情下的组合表现
        """
        self._log("压力测试...")
        
        if scenarios is None:
            # 默认压力场景
            scenarios = {
                'market_crash_2008': -0.40,      # 2008年金融危机
                'market_crash_2015': -0.30,      # 2015年股灾
                'covid_crash_2020': -0.35,       # 2020年疫情崩盘
                'interest_rate_shock': -0.15,    # 利率冲击
                'liquidity_crisis': -0.25,       # 流动性危机
            }
        
        results = {}
        
        for scenario_name, market_decline in scenarios.items():
            # 简化模型：假设组合跌幅与市场相关
            portfolio_decline = market_decline * (1 + np.random.randn() * 0.1)
            results[scenario_name] = portfolio_decline
        
        return results
    
    # ==================== 风险预算 ====================
    
    def risk_budget_allocation(
        self,
        assets: List[str],
        cov_matrix: pd.DataFrame,
        target_risk: float = 0.15
    ) -> Dict[str, float]:
        """
        基于风险预算的资产配置
        
        等风险贡献 (ERC) 策略
        """
        self._log("风险预算配置...")
        
        n = len(assets)
        
        # 简化：等权重作为起点
        weights = {asset: 1/n for asset in assets}
        cov_values = cov_matrix.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype=float)
        
        # 迭代优化以达到等风险贡献
        for _ in range(100):  # 最大迭代次数
            weight_array = np.array([weights[asset] for asset in assets], dtype=float)

            # 计算边际风险贡献
            port_vol = np.sqrt(weight_array.T @ cov_values @ weight_array)
            
            if port_vol == 0:
                break
            
            mrc = cov_values @ weight_array / port_vol
            rc = weight_array * mrc
            
            # 调整权重
            target_rc = port_vol / n
            for i, asset in enumerate(assets):
                if rc[i] > 0:
                    adjustment = (target_rc - rc[i]) / rc[i] * 0.1
                    weights[asset] *= (1 + adjustment)
            
            # 归一化
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    # ==================== 综合风控 ====================
    
    def run_risk_management(
        self,
        portfolio_returns: pd.Series,
        predicted_returns: Dict[str, float],
        predicted_volatilities: Dict[str, float],
        current_prices: Dict[str, float],
        macro_signal: str = "🟢",
        benchmark_returns: Optional[pd.Series] = None,
        conviction_scores: Optional[Dict[str, float]] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
    ) -> RiskLayerResult:
        """
        执行完整风控流程
        """
        self._log("=" * 60)
        self._log("【第6层】风控层 - 组合风控与仓位管理")
        self._log("=" * 60)
        
        result = RiskLayerResult()
        
        # 1. 计算风险指标
        result.risk_metrics = self.calculate_risk_metrics(
            portfolio_returns,
            benchmark_returns
        )
        
        # 2. 仓位管理
        result.position_sizing = self.calculate_position_sizing(
            predicted_returns,
            predicted_volatilities,
            {},  # 个股风险信号
            macro_signal,
            conviction_scores=conviction_scores,
        )
        
        # 3. 止损止盈
        result.stop_loss_take_profit = self.calculate_stop_loss_take_profit(
            current_prices
        )
        
        # 4. 压力测试
        result.stress_test_results = self.stress_test(portfolio_returns)
        
        # 5. 风险预算 (如果有多个资产)
        if len(predicted_returns) > 1:
            # 优先使用主线传入的协方差矩阵，缺失时回退到对角波动率矩阵
            cov_matrix = covariance_matrix
            if cov_matrix is None:
                cov_matrix = pd.DataFrame(
                    np.diag([v**2 for v in predicted_volatilities.values()]),
                    index=list(predicted_returns.keys()),
                    columns=list(predicted_returns.keys())
                )
            result.risk_budget_allocation = self.risk_budget_allocation(
                list(predicted_returns.keys()),
                cov_matrix
            )
        
        # 6. 风险预警
        warnings = []
        
        if result.risk_metrics.max_drawdown < self.max_drawdown_limit:
            warnings.append(f"⚠️ 最大回撤 {result.risk_metrics.max_drawdown:.2%} 超过限制 {self.max_drawdown_limit:.2%}")
        
        if result.risk_metrics.volatility > self.target_volatility * 1.5:
            warnings.append(f"⚠️ 波动率 {result.risk_metrics.volatility:.2%} 远高于目标 {self.target_volatility:.2%}")
        
        if result.risk_metrics.sharpe_ratio < 0:
            warnings.append(f"⚠️ 夏普比率 {result.risk_metrics.sharpe_ratio:.2f} 为负")
        
        # 压力测试预警
        max_stress_loss = min(result.stress_test_results.values())
        if max_stress_loss < -0.30:
            warnings.append(f"⚠️ 压力测试最大损失 {max_stress_loss:.2%}，极端风险较高")
        
        result.risk_warnings = warnings
        
        # 确定风险等级
        if len(warnings) >= 3:
            result.risk_level = "danger"
        elif len(warnings) >= 1:
            result.risk_level = "warning"
        else:
            result.risk_level = "normal"
        
        self._log(f"风控层完成: 风险等级={result.risk_level}, 预警数={len(warnings)}")
        
        if warnings:
            for w in warnings:
                self._log(f"  {w}")
        
        self.result = result
        return result
    
    def generate_risk_report(self) -> str:
        """生成风控报告"""
        lines = []
        
        lines.append("## 🛡️ 风控层报告")
        lines.append("")
        
        # 风险指标
        lines.append("### 风险指标")
        lines.append(f"- 年化波动率: {self.result.risk_metrics.volatility:.2%}")
        lines.append(f"- 最大回撤: {self.result.risk_metrics.max_drawdown:.2%}")
        lines.append(f"- VaR (95%): {self.result.risk_metrics.var_95:.2%}")
        lines.append(f"- 夏普比率: {self.result.risk_metrics.sharpe_ratio:.2f}")
        lines.append(f"- Beta: {self.result.risk_metrics.beta:.2f}")
        lines.append("")
        
        # 仓位管理
        lines.append("### 仓位管理")
        lines.append(f"- 现金比例: {self.result.position_sizing.cash_ratio:.0%}")
        lines.append(f"- 杠杆倍数: {self.result.position_sizing.leverage:.2f}x")
        if self.result.position_sizing.risk_adjusted_weights:
            lines.append("- 风险调整权重:")
            for symbol, weight in sorted(
                self.result.position_sizing.risk_adjusted_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                lines.append(f"  - {symbol}: {weight:.2%}")
        lines.append("")
        
        # 止损止盈
        lines.append("### 止损止盈设置")
        for symbol in list(self.result.stop_loss_take_profit.stop_loss_levels.keys())[:3]:
            sl = self.result.stop_loss_take_profit.stop_loss_levels[symbol]
            tp = self.result.stop_loss_take_profit.take_profit_levels[symbol]
            lines.append(f"- {symbol}: 止损 {sl:.2f}, 止盈 {tp:.2f}")
        lines.append("")
        
        # 压力测试
        lines.append("### 压力测试结果")
        for scenario, loss in sorted(self.result.stress_test_results.items(), key=lambda x: x[1]):
            lines.append(f"- {scenario}: {loss:.2%}")
        lines.append("")
        
        # 风险预警
        if self.result.risk_warnings:
            lines.append("### ⚠️ 风险预警")
            for warning in self.result.risk_warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        lines.append(f"**风险等级**: {self.result.risk_level.upper()}")
        
        return "\n".join(lines)


# ==================== 组合优化器 ====================


class PortfolioOptimizer:
    """
    均值-方差组合优化器（改进四）。

    支持三种优化目标：
      max_sharpe    最大化夏普比率
      min_variance  最小化波动率（风险厌恶型）
      risk_parity   风险平价（每股等风险贡献）

    协方差矩阵使用 Ledoit-Wolf 收缩估计量，避免样本估计噪声。
    内置约束：单股最大仓位、行业集中度（可选）、多头仓位。
    """

    def __init__(
        self,
        max_position: float = 0.20,
        min_position: float = 0.0,
        risk_free_rate: float = 0.025,
        method: str = "risk_parity",
    ) -> None:
        """
        Args:
            max_position:   单股最大仓位
            min_position:   单股最小仓位（>0 可防止某股权重降为0）
            risk_free_rate: 无风险利率（年化），用于夏普计算
            method:         优化目标 max_sharpe / min_variance / risk_parity
        """
        self.max_position = max_position
        self.min_position = min_position
        self.rf = risk_free_rate
        self.method = method

    def optimize(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
        conviction_scores: Optional[Dict[str, float]] = None,
        base_position: float = 1.0,
    ) -> Dict[str, float]:
        """
        执行组合优化。

        Args:
            symbols:          股票列表
            expected_returns: 预期收益率 {symbol: return}
            cov_matrix:       协方差矩阵 (n×n numpy array，与 symbols 顺序对应)
            conviction_scores: 置信分数 [-1,1]，若提供则对 expected_returns 加权
            base_position:    总仓位上限（来自宏观信号）

        Returns:
            {symbol: weight}，权重之和 ≤ base_position
        """
        n = len(symbols)
        if n == 0:
            return {}
        if n == 1:
            return {symbols[0]: min(base_position, self.max_position)}

        # 若提供置信分数，融合到预期收益
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
        if conviction_scores:
            scores = np.array([max(conviction_scores.get(s, 0.0), 0.0) for s in symbols])
            mu = mu * (1 + scores)

        # 保证协方差矩阵正半定
        cov = np.array(cov_matrix, dtype=float)
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-8:
            cov += np.eye(n) * (abs(eigvals.min()) + 1e-6)

        if self.method == "risk_parity":
            weights = self._risk_parity(cov, n)
        elif self.method == "min_variance":
            weights = self._min_variance(cov, n)
        else:
            weights = self._max_sharpe(mu, cov, n)

        # 应用仓位约束
        weights = np.clip(weights, self.min_position, self.max_position)
        total = weights.sum()
        if total < 1e-8:
            weights = np.ones(n) / n
            total = 1.0
        weights = weights / total * base_position

        return {s: float(w) for s, w in zip(symbols, weights)}

    @staticmethod
    def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf 收缩估计量：
          Sigma_lw = delta * mu_target + (1 - delta) * S
        其中目标矩阵为等相关矩阵，delta 为解析最优收缩系数。
        """
        t, n = returns.shape
        S = np.cov(returns, rowvar=False)
        mu = np.trace(S) / n
        delta_hat = np.sum((S - mu * np.eye(n))**2)
        pi_hat = sum(
            np.sum((np.outer(returns[i] - returns.mean(axis=0),
                             returns[i] - returns.mean(axis=0)) - S)**2)
            for i in range(t)
        ) / t**2
        rho = min(1.0, max(0.0, pi_hat / delta_hat)) if delta_hat > 1e-12 else 0.0
        return rho * mu * np.eye(n) + (1 - rho) * S

    @staticmethod
    def _risk_parity(cov: np.ndarray, n: int) -> np.ndarray:
        """风险平价：每个资产贡献相同组合风险"""
        w = np.ones(n) / n
        for _ in range(300):
            port_var = w @ cov @ w
            if port_var < 1e-12:
                break
            mrc = cov @ w / np.sqrt(port_var)
            rc = w * mrc
            target = port_var / n
            w = w * (target / (rc + 1e-10))
            w = np.clip(w, 0, None)
            w /= w.sum() + 1e-10
        return w

    @staticmethod
    def _min_variance(cov: np.ndarray, n: int) -> np.ndarray:
        """最小方差：解析解 w ∝ Sigma^{-1} * 1"""
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(n)
            w = inv_cov @ ones
            w = np.clip(w, 0, None)
            total = w.sum()
            return w / total if total > 1e-8 else np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """最大夏普：解析解 w ∝ Sigma^{-1} * (mu - rf)"""
        excess = mu - self.rf / 252
        try:
            inv_cov = np.linalg.inv(cov)
            w = inv_cov @ excess
            w = np.clip(w, 0, None)
            total = w.sum()
            return w / total if total > 1e-8 else np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    @classmethod
    def build_cov_from_returns(
        cls,
        returns_df: pd.DataFrame,
        symbols: List[str],
        lookback: int = 60,
    ) -> np.ndarray:
        """
        从历史收益率矩阵估算协方差矩阵（使用 Ledoit-Wolf 收缩）。

        Args:
            returns_df: 列为 symbol 的日收益率 DataFrame
            symbols:    目标股票列表
            lookback:   使用最近 N 天数据

        Returns:
            n×n numpy 协方差矩阵（年化）
        """
        r = returns_df[symbols].dropna().tail(lookback)
        if len(r) < 10:
            vols = returns_df[symbols].std().fillna(0.20).values
            return np.diag(vols**2) * 252
        try:
            cov = cls.ledoit_wolf_shrinkage(r.values) * 252
        except Exception:
            cov = r.cov().fillna(0).values * 252
        return cov


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Risk Management Layer - 测试")
    print("=" * 80)
    
    # 创建测试数据
    np.random.seed(42)
    n_days = 252
    
    # 模拟组合收益
    portfolio_returns = pd.Series(
        np.random.normal(0.0005, 0.02, n_days),
        index=pd.date_range('2024-01-01', periods=n_days, freq='B')
    )
    
    # 模拟预测数据
    predicted_returns = {
        'AAPL': 0.02,
        'MSFT': 0.015,
        'GOOGL': 0.01,
        'AMZN': 0.008,
        'NVDA': 0.025
    }
    
    predicted_volatilities = {
        'AAPL': 0.25,
        'MSFT': 0.22,
        'GOOGL': 0.28,
        'AMZN': 0.30,
        'NVDA': 0.35
    }
    
    current_prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 2800.0,
        'AMZN': 3200.0,
        'NVDA': 220.0
    }
    
    # 运行风控
    risk_layer = RiskManagementLayer(verbose=True)
    
    result = risk_layer.run_risk_management(
        portfolio_returns=portfolio_returns,
        predicted_returns=predicted_returns,
        predicted_volatilities=predicted_volatilities,
        current_prices=current_prices,
        macro_signal="🟡"
    )
    
    print("\n" + "=" * 80)
    print("风控报告")
    print("=" * 80)
    print(risk_layer.generate_risk_report())
