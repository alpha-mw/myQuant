"""
因子中性化测试
"""

import pytest
import numpy as np
import pandas as pd
from factor_neutralizer import FactorNeutralizer, PortfolioNeutralizer


class TestFactorNeutralizer:
    """因子中性化器测试"""
    
    def test_market_cap_neutralize(self):
        """测试市值中性化"""
        neutralizer = FactorNeutralizer()
        
        # 创建测试数据
        np.random.seed(42)
        n = 100
        
        # 因子值与市值相关
        market_cap = np.random.lognormal(10, 1, n)
        factor_values = np.log(market_cap) * 0.5 + np.random.randn(n)
        
        factor_series = pd.Series(factor_values)
        cap_series = pd.Series(market_cap)
        
        # 中性化
        neutral_factor = neutralizer.market_cap_neutralize(factor_series, cap_series)
        
        # 中性化后的因子应该与市值相关性降低
        original_corr = factor_series.corr(np.log(cap_series))
        neutral_corr = neutral_factor.corr(np.log(cap_series))
        
        assert abs(neutral_corr) < abs(original_corr)
    
    def test_industry_neutralize(self):
        """测试行业中性化"""
        neutralizer = FactorNeutralizer()
        
        np.random.seed(42)
        n = 100
        
        # 创建3个行业
        industries = np.random.choice(['A', 'B', 'C'], n)
        factor_values = np.random.randn(n)
        
        # 给不同行业不同的均值
        for i, ind in enumerate(['A', 'B', 'C']):
            mask = industries == ind
            factor_values[mask] += i * 2
        
        factor_series = pd.Series(factor_values)
        industry_series = pd.Series(industries)
        
        # 中性化
        neutral_factor = neutralizer.industry_neutralize(factor_series, industry_series)
        
        # 每个行业内的均值应该接近0
        for ind in ['A', 'B', 'C']:
            mask = industry_series == ind
            industry_mean = neutral_factor[mask].mean()
            assert abs(industry_mean) < 0.5
    
    def test_double_neutralize(self):
        """测试双重中性化"""
        neutralizer = FactorNeutralizer()
        
        np.random.seed(42)
        n = 100
        
        market_cap = np.random.lognormal(10, 1, n)
        industries = np.random.choice(['A', 'B'], n)
        factor_values = np.random.randn(n)
        
        factor_series = pd.Series(factor_values)
        cap_series = pd.Series(market_cap)
        industry_series = pd.Series(industries)
        
        # 双重中性化
        neutral_factor = neutralizer.double_neutralize(
            factor_series, cap_series, industry_series
        )
        
        # 结果不应该全为NaN
        assert neutral_factor.notna().sum() > 0


class TestPortfolioNeutralizer:
    """组合中性化器测试"""
    
    def test_neutralize_portfolio_weights(self):
        """测试组合权重中性化"""
        neutralizer = PortfolioNeutralizer()
        
        np.random.seed(42)
        n = 10
        
        # 原始权重
        weights = pd.Series(np.random.random(n))
        weights = weights / weights.sum()
        
        market_cap = np.random.lognormal(10, 1, n)
        industries = pd.Series(['A'] * 5 + ['B'] * 5)
        
        # 中性化
        neutral_weights = neutralizer.neutralize_portfolio_weights(
            weights, market_cap, industries
        )
        
        # 权重总和应该为1
        assert abs(neutral_weights.sum() - 1.0) < 0.001
        
        # 每个行业的权重应该接近0.5
        industry_a_weight = neutral_weights[industries == 'A'].sum()
        assert abs(industry_a_weight - 0.5) < 0.1
    
    def test_check_neutrality(self):
        """测试中性化检查"""
        neutralizer = PortfolioNeutralizer()
        
        n = 10
        weights = pd.Series([0.1] * n)
        market_cap = pd.Series(np.random.lognormal(10, 1, n))
        industries = pd.Series(['A'] * 5 + ['B'] * 5)
        
        result = neutralizer.check_neutrality(weights, market_cap, industries)
        
        # 检查结果包含必要字段
        assert 'market_cap_exposure' in result
        assert 'max_industry_deviation' in result
        assert 'is_market_cap_neutral' in result
        assert 'is_industry_neutral' in result
