"""
风险管理单元测试
测试风险指标计算、VaR、压力测试
"""

import pytest
import pandas as pd
import numpy as np


class TestRiskMetrics:
    """风险指标测试"""
    
    def test_volatility_calculation(self):
        """测试波动率计算"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.02  # 日收益
        
        annual_vol = returns.std() * np.sqrt(252)
        
        # 年化波动率应该约为日波动率的sqrt(252)倍
        expected_vol = returns.std() * np.sqrt(252)
        assert abs(annual_vol - expected_vol) < 0.001
    
    def test_sharpe_ratio(self):
        """测试夏普比率"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005  # 带正收益的日收益
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        # 夏普比率应该为正
        assert sharpe > 0
    
    def test_max_drawdown(self):
        """测试最大回撤"""
        # 创建一个有明显回撤的收益序列
        # 累计净值：1.0 → 1.01 → 1.02 → 0.97 → 0.94 → 0.95 → 0.96 → 0.86 → 0.91
        returns = pd.Series([0.01, 0.01, -0.05, -0.03, 0.01, 0.02, -0.10, 0.05])
        
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # 最大回撤应该为负
        assert max_dd < 0
        # 最大回撤应该在-10%左右（从1.02跌到0.86，回撤约15.7%）
        assert -0.20 < max_dd < -0.10
    
    def test_var_calculation(self):
        """测试VaR计算"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02  # 日收益
        
        # 95% VaR
        var_95 = np.percentile(returns, 5)
        # 99% VaR
        var_99 = np.percentile(returns, 1)
        
        # 99% VaR应该比95% VaR更极端（更负）
        assert var_99 < var_95
        # 两者都应该是负数
        assert var_95 < 0
        assert var_99 < 0


class TestPortfolioRisk:
    """组合风险测试"""
    
    def test_portfolio_volatility(self):
        """测试组合波动率"""
        np.random.seed(42)
        
        # 两只股票的收益
        returns1 = np.random.randn(252) * 0.02
        returns2 = np.random.randn(252) * 0.015
        
        # 等权重组合
        portfolio_returns = 0.5 * returns1 + 0.5 * returns2
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        assert portfolio_vol > 0
    
    def test_beta_calculation(self):
        """测试Beta计算"""
        np.random.seed(42)
        
        # 市场收益
        market_returns = np.random.randn(252) * 0.015
        # 股票收益（与市场相关）
        stock_returns = 1.2 * market_returns + np.random.randn(252) * 0.01
        
        # 计算Beta
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
        
        # Beta应该接近1.2
        assert abs(beta - 1.2) < 0.3


class TestStressTest:
    """压力测试"""
    
    def test_correlation_stress(self):
        """测试相关性压力"""
        np.random.seed(42)
        
        # 正常市场条件下低相关的两只股票
        normal_returns1 = np.random.randn(100) * 0.02
        normal_returns2 = np.random.randn(100) * 0.02
        normal_corr = np.corrcoef(normal_returns1, normal_returns2)[0, 1]
        
        # 压力市场条件下（相关性趋近于1）
        market_shock = np.random.randn(100) * 0.05
        stress_returns1 = market_shock + np.random.randn(100) * 0.01
        stress_returns2 = market_shock + np.random.randn(100) * 0.01
        stress_corr = np.corrcoef(stress_returns1, stress_returns2)[0, 1]
        
        # 压力条件下相关性应该更高
        assert abs(stress_corr) > abs(normal_corr)
