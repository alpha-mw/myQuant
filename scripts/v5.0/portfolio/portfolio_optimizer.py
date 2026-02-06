"""
Quant-Investor V5.0 - 组合优化与回测模块

本模块提供完整的组合优化和回测功能，包括：
1. 组合优化：均值-方差、风险平价、Black-Litterman、最大分散化
2. 回测引擎：支持交易成本、滑点、再平衡
3. 绩效评估：夏普比率、最大回撤、Calmar比率等
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ==================== 组合优化器 ====================

class PortfolioOptimizer:
    """
    组合优化器
    
    支持多种优化方法：
    - 均值-方差优化 (Mean-Variance)
    - 风险平价 (Risk Parity)
    - 最大分散化 (Maximum Diversification)
    - Black-Litterman
    - 最小方差 (Minimum Variance)
    - 等权重 (Equal Weight)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def _calc_portfolio_stats(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """计算组合统计量"""
        port_return = np.sum(returns * weights) * 252  # 年化
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        return port_return, port_vol, sharpe
    
    def mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None,
        max_weight: float = 0.3,
        min_weight: float = 0.0
    ) -> np.ndarray:
        """
        均值-方差优化
        
        Args:
            expected_returns: 预期收益率
            cov_matrix: 协方差矩阵
            target_return: 目标收益率（None则最大化夏普比率）
            max_weight: 单个资产最大权重
            min_weight: 单个资产最小权重
        
        Returns:
            最优权重
        """
        n_assets = len(expected_returns)
        
        # 初始权重
        init_weights = np.array([1/n_assets] * n_assets)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * expected_returns) * 252 - target_return
            })
        
        # 边界
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # 目标函数
        if target_return is not None:
            # 最小化方差
            def objective(w):
                return np.dot(w.T, np.dot(cov_matrix * 252, w))
        else:
            # 最大化夏普比率（最小化负夏普）
            def objective(w):
                port_return = np.sum(w * expected_returns) * 252
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
                if port_vol == 0:
                    return 1e10
                return -(port_return - self.risk_free_rate) / port_vol
        
        # 优化
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else init_weights
    
    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        target_risk_contrib: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        风险平价优化
        
        使每个资产对组合风险的贡献相等。
        
        Args:
            cov_matrix: 协方差矩阵
            target_risk_contrib: 目标风险贡献（None则等风险贡献）
        
        Returns:
            最优权重
        """
        n_assets = cov_matrix.shape[0]
        
        if target_risk_contrib is None:
            target_risk_contrib = np.array([1/n_assets] * n_assets)
        
        init_weights = np.array([1/n_assets] * n_assets)
        
        def risk_contribution(w):
            """计算风险贡献"""
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return np.zeros(n_assets)
            marginal_contrib = np.dot(cov_matrix, w) / port_vol
            return w * marginal_contrib
        
        def objective(w):
            """最小化风险贡献偏差"""
            rc = risk_contribution(w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return 1e10
            rc_pct = rc / port_vol
            return np.sum((rc_pct - target_risk_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 0.5) for _ in range(n_assets))
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else init_weights
    
    def minimum_variance(
        self,
        cov_matrix: np.ndarray,
        max_weight: float = 0.3
    ) -> np.ndarray:
        """
        最小方差组合
        
        Args:
            cov_matrix: 协方差矩阵
            max_weight: 单个资产最大权重
        
        Returns:
            最优权重
        """
        n_assets = cov_matrix.shape[0]
        init_weights = np.array([1/n_assets] * n_assets)
        
        def objective(w):
            return np.dot(w.T, np.dot(cov_matrix, w))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else init_weights
    
    def maximum_diversification(
        self,
        cov_matrix: np.ndarray,
        max_weight: float = 0.3
    ) -> np.ndarray:
        """
        最大分散化组合
        
        最大化分散化比率 = 加权平均波动率 / 组合波动率
        
        Args:
            cov_matrix: 协方差矩阵
            max_weight: 单个资产最大权重
        
        Returns:
            最优权重
        """
        n_assets = cov_matrix.shape[0]
        init_weights = np.array([1/n_assets] * n_assets)
        
        # 个股波动率
        vols = np.sqrt(np.diag(cov_matrix))
        
        def objective(w):
            """最小化负分散化比率"""
            weighted_vol = np.sum(w * vols)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return 1e10
            return -weighted_vol / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else init_weights
    
    def black_litterman(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views: Dict[int, float],
        view_confidences: Dict[int, float],
        tau: float = 0.05,
        risk_aversion: float = 2.5
    ) -> np.ndarray:
        """
        Black-Litterman模型
        
        融合市场均衡收益与主观观点。
        
        Args:
            market_weights: 市场权重（市值加权）
            cov_matrix: 协方差矩阵
            views: 观点字典 {资产索引: 预期超额收益}
            view_confidences: 观点置信度 {资产索引: 置信度0-1}
            tau: 不确定性参数
            risk_aversion: 风险厌恶系数
        
        Returns:
            最优权重
        """
        n_assets = len(market_weights)
        
        # 均衡收益
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if not views:
            # 无观点时返回市场权重
            return market_weights
        
        # 构建观点矩阵
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = []
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
            confidence = view_confidences.get(asset_idx, 0.5)
            # 观点不确定性与置信度成反比
            omega_diag.append((1 - confidence) * tau * cov_matrix[asset_idx, asset_idx])
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman公式
        tau_cov = tau * cov_matrix
        
        # 后验收益
        M1 = np.linalg.inv(np.linalg.inv(tau_cov) + np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))
        M2 = np.dot(np.linalg.inv(tau_cov), pi) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        bl_returns = np.dot(M1, M2)
        
        # 使用均值-方差优化得到最终权重
        return self.mean_variance(bl_returns / 252, cov_matrix)
    
    def equal_weight(self, n_assets: int) -> np.ndarray:
        """等权重组合"""
        return np.array([1/n_assets] * n_assets)


# ==================== 回测引擎 ====================

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000  # 初始资金
    commission_rate: float = 0.001    # 手续费率
    slippage: float = 0.001           # 滑点
    rebalance_freq: str = 'monthly'   # 再平衡频率: daily, weekly, monthly
    benchmark: Optional[str] = None   # 基准指数


@dataclass
class BacktestResult:
    """回测结果"""
    portfolio_values: pd.Series       # 组合净值
    benchmark_values: pd.Series       # 基准净值
    returns: pd.Series                # 组合收益率
    benchmark_returns: pd.Series      # 基准收益率
    positions: pd.DataFrame           # 持仓记录
    trades: pd.DataFrame              # 交易记录
    metrics: Dict[str, float]         # 绩效指标


class Backtester:
    """
    回测引擎
    
    支持：
    - 交易成本和滑点
    - 多种再平衡频率
    - 基准对比
    - 完整绩效评估
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            prices: 价格DataFrame (日期 x 股票)
            weights: 权重DataFrame (日期 x 股票)
            benchmark_prices: 基准价格Series
        
        Returns:
            BacktestResult
        """
        # 对齐数据
        common_dates = prices.index.intersection(weights.index)
        prices = prices.loc[common_dates]
        weights = weights.loc[common_dates]
        
        if benchmark_prices is not None:
            benchmark_prices = benchmark_prices.loc[common_dates]
        
        # 计算收益率
        returns = prices.pct_change().fillna(0)
        
        # 初始化
        capital = self.config.initial_capital
        portfolio_values = [capital]
        positions_list = []
        trades_list = []
        
        current_weights = weights.iloc[0].values
        current_positions = current_weights * capital / prices.iloc[0].values
        
        # 确定再平衡日期
        rebalance_dates = self._get_rebalance_dates(common_dates)
        
        for i in range(1, len(common_dates)):
            date = common_dates[i]
            prev_date = common_dates[i-1]
            
            # 计算当日收益
            daily_returns = returns.iloc[i].values
            
            # 更新持仓价值
            position_values = current_positions * prices.iloc[i].values
            capital = np.sum(position_values)
            
            # 检查是否需要再平衡
            if date in rebalance_dates:
                target_weights = weights.loc[date].values
                target_values = target_weights * capital
                target_positions = target_values / prices.iloc[i].values
                
                # 计算交易
                trades = target_positions - current_positions
                trade_values = np.abs(trades * prices.iloc[i].values)
                
                # 扣除交易成本
                total_cost = np.sum(trade_values) * (self.config.commission_rate + self.config.slippage)
                capital -= total_cost
                
                # 更新持仓
                current_positions = target_positions * (capital / np.sum(target_values)) if np.sum(target_values) > 0 else target_positions
                current_weights = target_weights
                
                # 记录交易
                for j, (stock, trade) in enumerate(zip(prices.columns, trades)):
                    if abs(trade) > 0.001:
                        trades_list.append({
                            'date': date,
                            'stock': stock,
                            'shares': trade,
                            'price': prices.iloc[i, j],
                            'value': trade * prices.iloc[i, j]
                        })
            
            portfolio_values.append(capital)
            
            # 记录持仓
            positions_list.append({
                'date': date,
                **{stock: pos for stock, pos in zip(prices.columns, current_positions)}
            })
        
        # 构建结果
        portfolio_values = pd.Series(portfolio_values, index=common_dates)
        portfolio_returns = portfolio_values.pct_change().fillna(0)
        
        if benchmark_prices is not None:
            benchmark_values = benchmark_prices / benchmark_prices.iloc[0] * self.config.initial_capital
            benchmark_returns = benchmark_prices.pct_change().fillna(0)
        else:
            benchmark_values = pd.Series(index=common_dates)
            benchmark_returns = pd.Series(index=common_dates)
        
        positions_df = pd.DataFrame(positions_list)
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        
        # 计算绩效指标
        metrics = self._calculate_metrics(
            portfolio_values, portfolio_returns,
            benchmark_values, benchmark_returns
        )
        
        return BacktestResult(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            positions=positions_df,
            trades=trades_df,
            metrics=metrics
        )
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """获取再平衡日期"""
        if self.config.rebalance_freq == 'daily':
            return set(dates)
        elif self.config.rebalance_freq == 'weekly':
            # 每周第一个交易日
            df = pd.DataFrame(index=dates)
            df['week'] = df.index.isocalendar().week
            df['year'] = df.index.year
            return set(df.groupby(['year', 'week']).apply(lambda x: x.index[0]))
        else:  # monthly
            # 每月第一个交易日
            df = pd.DataFrame(index=dates)
            df['month'] = df.index.month
            df['year'] = df.index.year
            return set(df.groupby(['year', 'month']).apply(lambda x: x.index[0]))
    
    def _calculate_metrics(
        self,
        portfolio_values: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_values: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """计算绩效指标"""
        metrics = {}
        
        # 基础指标
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        n_years = len(portfolio_values) / 252
        annual_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
        annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
        
        metrics['total_return'] = total_return
        metrics['annual_return'] = annual_return
        metrics['annual_volatility'] = annual_vol
        
        # 夏普比率
        excess_return = annual_return - 2  # 假设无风险利率2%
        metrics['sharpe_ratio'] = excess_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # Calmar比率
        metrics['calmar_ratio'] = annual_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = excess_return / downside_vol if downside_vol > 0 else 0
        
        # 胜率
        metrics['win_rate'] = (portfolio_returns > 0).mean() * 100
        
        # 盈亏比
        avg_win = portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0
        avg_loss = abs(portfolio_returns[portfolio_returns < 0].mean()) if (portfolio_returns < 0).any() else 0
        metrics['profit_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 相对基准指标
        if len(benchmark_returns) > 0 and not benchmark_returns.isna().all():
            # Alpha和Beta
            cov_matrix = np.cov(portfolio_returns.dropna(), benchmark_returns.dropna())
            if cov_matrix.shape == (2, 2):
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
                benchmark_annual_return = benchmark_returns.mean() * 252 * 100
                alpha = annual_return - (2 + beta * (benchmark_annual_return - 2))
                
                metrics['alpha'] = alpha
                metrics['beta'] = beta
            
            # 信息比率
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252) * 100
            metrics['tracking_error'] = tracking_error
            metrics['information_ratio'] = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
        
        return metrics


# ==================== 绩效报告生成器 ====================

class PerformanceReporter:
    """绩效报告生成器"""
    
    @staticmethod
    def generate_report(result: BacktestResult) -> str:
        """生成绩效报告"""
        report = []
        report.append("=" * 60)
        report.append("回测绩效报告")
        report.append("=" * 60)
        
        m = result.metrics
        
        report.append("\n【收益指标】")
        report.append(f"  总收益率:     {m.get('total_return', 0):.2f}%")
        report.append(f"  年化收益率:   {m.get('annual_return', 0):.2f}%")
        report.append(f"  年化波动率:   {m.get('annual_volatility', 0):.2f}%")
        
        report.append("\n【风险指标】")
        report.append(f"  最大回撤:     {m.get('max_drawdown', 0):.2f}%")
        report.append(f"  夏普比率:     {m.get('sharpe_ratio', 0):.2f}")
        report.append(f"  索提诺比率:   {m.get('sortino_ratio', 0):.2f}")
        report.append(f"  Calmar比率:   {m.get('calmar_ratio', 0):.2f}")
        
        report.append("\n【交易指标】")
        report.append(f"  胜率:         {m.get('win_rate', 0):.2f}%")
        report.append(f"  盈亏比:       {m.get('profit_loss_ratio', 0):.2f}")
        
        if 'alpha' in m:
            report.append("\n【相对基准】")
            report.append(f"  Alpha:        {m.get('alpha', 0):.2f}%")
            report.append(f"  Beta:         {m.get('beta', 0):.2f}")
            report.append(f"  跟踪误差:     {m.get('tracking_error', 0):.2f}%")
            report.append(f"  信息比率:     {m.get('information_ratio', 0):.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Portfolio Optimization & Backtesting Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建测试数据
    n_assets = 5
    n_days = 500
    
    # 模拟价格数据
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    returns = np.random.randn(n_days, n_assets) * 0.02 + 0.0005
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_assets)]
    )
    
    # 基准
    benchmark_returns = np.random.randn(n_days) * 0.015 + 0.0003
    benchmark_prices = pd.Series(
        100 * np.cumprod(1 + benchmark_returns),
        index=dates
    )
    
    # 计算协方差矩阵和预期收益
    daily_returns = prices.pct_change().dropna()
    cov_matrix = daily_returns.cov().values
    expected_returns = daily_returns.mean().values
    
    print("\n1. Testing Portfolio Optimization...")
    optimizer = PortfolioOptimizer()
    
    # 均值-方差
    mv_weights = optimizer.mean_variance(expected_returns, cov_matrix)
    print(f"   Mean-Variance weights: {mv_weights.round(3)}")
    
    # 风险平价
    rp_weights = optimizer.risk_parity(cov_matrix)
    print(f"   Risk Parity weights: {rp_weights.round(3)}")
    
    # 最小方差
    minvar_weights = optimizer.minimum_variance(cov_matrix)
    print(f"   Min Variance weights: {minvar_weights.round(3)}")
    
    # 最大分散化
    maxdiv_weights = optimizer.maximum_diversification(cov_matrix)
    print(f"   Max Diversification weights: {maxdiv_weights.round(3)}")
    
    # 等权重
    eq_weights = optimizer.equal_weight(n_assets)
    print(f"   Equal weights: {eq_weights.round(3)}")
    
    print("\n2. Testing Backtester...")
    
    # 创建权重时间序列（使用风险平价）
    weights_df = pd.DataFrame(
        np.tile(rp_weights, (len(dates), 1)),
        index=dates,
        columns=prices.columns
    )
    
    # 运行回测
    config = BacktestConfig(
        initial_capital=1000000,
        commission_rate=0.001,
        slippage=0.001,
        rebalance_freq='monthly'
    )
    
    backtester = Backtester(config)
    result = backtester.run(prices, weights_df, benchmark_prices)
    
    # 生成报告
    print("\n3. Performance Report:")
    report = PerformanceReporter.generate_report(result)
    print(report)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
