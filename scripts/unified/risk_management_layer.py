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
    target_positions: Dict[str, float]         # 目标仓位
    risk_adjusted_weights: Dict[str, float]    # 风险调整权重
    cash_ratio: float = 0.0                    # 现金比例
    leverage: float = 1.0                      # 杠杆倍数


@dataclass
class StopLossTakeProfit:
    """止损止盈设置"""
    stop_loss_levels: Dict[str, float]         # 止损价位
    take_profit_levels: Dict[str, float]       # 止盈价位
    trailing_stops: Dict[str, float]           # 跟踪止损


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
        
        self.result = RiskLayerResult()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[RiskLayer] {msg}")
    
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
        
        # 年化波动率
        metrics.volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        metrics.max_drawdown = drawdown.min()
        
        # VaR / CVaR
        metrics.var_95 = np.percentile(returns, 5)
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
        
        self._log(f"风险指标: 波动率={metrics.volatility:.2%}, 最大回撤={metrics.max_drawdown:.2%}, 夏普={metrics.sharpe_ratio:.2f}")
        
        return metrics
    
    # ==================== 仓位管理 ====================
    
    def calculate_position_sizing(
        self,
        predicted_returns: Dict[str, float],
        predicted_volatilities: Dict[str, float],
        risk_signals: Dict[str, str],
        macro_signal: str = "🟢"
    ) -> PositionSizing:
        """
        基于风险的仓位管理
        
        根据风险信号动态调整仓位
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
        
        # 计算风险调整权重 (逆波动率加权)
        inverse_vols = {}
        for symbol, vol in predicted_volatilities.items():
            if vol > 0:
                inverse_vols[symbol] = 1 / vol
            else:
                inverse_vols[symbol] = 1.0
        
        total_inverse_vol = sum(inverse_vols.values())
        if total_inverse_vol > 0:
            for symbol in inverse_vols:
                sizing.risk_adjusted_weights[symbol] = inverse_vols[symbol] / total_inverse_vol * base_position
        
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
                sizing.risk_adjusted_weights[symbol] /= total_weight
                sizing.risk_adjusted_weights[symbol] *= base_position
        
        sizing.cash_ratio = 1 - base_position
        sizing.leverage = 1.0 if base_position <= 1.0 else base_position
        
        self._log(f"仓位管理: 基础仓位={base_position:.0%}, 现金比例={sizing.cash_ratio:.0%}")
        
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
        
        # 迭代优化以达到等风险贡献
        for _ in range(100):  # 最大迭代次数
            # 计算边际风险贡献
            port_vol = np.sqrt(np.array(list(weights.values())).T @ cov_matrix @ np.array(list(weights.values())))
            
            if port_vol == 0:
                break
            
            mrc = cov_matrix @ np.array(list(weights.values())) / port_vol
            rc = np.array(list(weights.values())) * mrc
            
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
        benchmark_returns: Optional[pd.Series] = None
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
            macro_signal
        )
        
        # 3. 止损止盈
        result.stop_loss_take_profit = self.calculate_stop_loss_take_profit(
            current_prices
        )
        
        # 4. 压力测试
        result.stress_test_results = self.stress_test(portfolio_returns)
        
        # 5. 风险预算 (如果有多个资产)
        if len(predicted_returns) > 1:
            # 简化：使用预测波动率构建协方差矩阵
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
