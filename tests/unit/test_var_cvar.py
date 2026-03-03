"""
VaR和CVaR风险指标测试
"""

import pytest
import numpy as np
from scipy import stats


class TestVaRCalculation:
    """VaR计算测试"""
    
    def test_historical_var(self):
        """测试历史模拟法VaR"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02  # 日收益
        
        # 95% VaR
        var_95 = np.percentile(returns, 5)
        # 99% VaR
        var_99 = np.percentile(returns, 1)
        
        # VaR应该是负数（损失）
        assert var_95 < 0
        assert var_99 < 0
        # 99% VaR应该比95%更极端
        assert var_99 < var_95
    
    def test_parametric_var(self):
        """测试参数法VaR（方差-协方差法）"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # 95% VaR
        var_95 = mean - 1.645 * std
        # 99% VaR
        var_99 = mean - 2.326 * std
        
        assert var_95 < 0
        assert var_99 < var_95
    
    def test_cornish_fisher_var(self):
        """测试Cornish-Fisher调整VaR"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        
        mean = np.mean(returns)
        std = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Cornish-Fisher调整
        z = -1.645  # 95%置信度
        z_cf = (z + 
                (z**2 - 1) * skewness / 6 + 
                (z**3 - 3*z) * kurtosis / 24 - 
                (2*z**3 - 5*z) * skewness**2 / 36)
        
        var_cf = mean + z_cf * std
        
        assert var_cf < 0


class TestCVaRCalculation:
    """CVaR（条件VaR）测试"""
    
    def test_historical_cvar(self):
        """测试历史模拟法CVaR"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        
        # 95% VaR
        var_95 = np.percentile(returns, 5)
        # 95% CVaR = 超过VaR的平均损失
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # CVaR应该比VaR更极端
        assert cvar_95 <= var_95
        assert cvar_95 < 0
    
    def test_parametric_cvar(self):
        """测试参数法CVaR"""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # 正态分布假设下的CVaR
        # CVaR = mean - std * phi(z) / Phi(z)
        z = -1.645
        phi_z = stats.norm.pdf(z)
        Phi_z = 0.05
        
        cvar = mean - std * phi_z / Phi_z
        
        assert cvar < 0


class TestAdvancedRiskMetrics:
    """高级风险指标测试"""
    
    def test_omega_ratio(self):
        """测试Omega比率"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005
        threshold = 0  # 阈值
        
        # Omega = 正收益总和 / 负收益绝对值总和
        positive_returns = returns[returns > threshold]
        negative_returns = returns[returns <= threshold]
        
        omega = (np.sum(positive_returns - threshold) / 
                 np.sum(np.abs(negative_returns - threshold)))
        
        assert omega > 0
    
    def test_sortino_ratio(self):
        """测试Sortino比率"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005
        target_return = 0
        
        # 下行标准差
        downside_returns = returns[returns < target_return]
        downside_std = np.std(downside_returns) * np.sqrt(252)
        
        # Sortino比率
        excess_return = np.mean(returns) * 252 - target_return
        sortino = excess_return / downside_std if downside_std > 0 else 0
        
        assert sortino > -10  # 合理的范围
    
    def test_calmar_ratio(self):
        """测试Calmar比率"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005
        
        # 年化收益
        annual_return = np.mean(returns) * 252
        
        # 最大回撤
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        assert calmar > -100  # 合理的范围
