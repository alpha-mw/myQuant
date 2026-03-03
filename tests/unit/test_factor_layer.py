"""
因子层单元测试
测试因子计算、IC分析、因子筛选
"""

import pytest
import pandas as pd
import numpy as np


class TestFactorCalculation:
    """因子计算测试"""
    
    def test_momentum_factors(self):
        """测试动量因子"""
        prices = pd.Series([10, 11, 12, 11, 13, 14, 15, 14, 16, 17])
        
        return_5d = prices.pct_change(5)
        
        # 第6天相对于第1天的收益
        expected_return = (14 - 10) / 10  # = 0.4
        assert abs(return_5d.iloc[5] - expected_return) < 0.001
    
    def test_mean_reversion_factor(self):
        """测试均值回归因子"""
        # 使用足够长的序列计算20日均线
        prices = pd.Series([10 + i * 0.5 for i in range(30)])  # 30个数据点
        ma_20 = prices.rolling(20).mean()
        ma_bias = (prices - ma_20) / ma_20
        
        # 价格高于均线时，偏离度为正（跳过前20个NaN）
        valid_bias = ma_bias.dropna()
        assert len(valid_bias) > 0
        assert valid_bias.iloc[-1] > 0
    
    def test_volume_factor(self):
        """测试成交量因子"""
        volume = pd.Series([1000, 2000, 1500, 3000, 2500, 1800, 2200])
        volume_ma = volume.rolling(5).mean()
        volume_ratio = volume / volume_ma
        
        assert volume_ratio.iloc[-1] > 0


class TestFactorIC:
    """因子IC测试"""
    
    def test_ic_calculation(self):
        """测试IC计算"""
        np.random.seed(42)
        n = 100
        
        # 模拟因子值和未来收益
        factor = np.random.randn(n)
        future_return = 0.3 * factor + np.random.randn(n) * 0.5
        
        # 计算IC（相关系数）
        ic = np.corrcoef(factor, future_return)[0, 1]
        
        # IC应在-1到1之间
        assert -1 <= ic <= 1
        # 由于我们设置了相关性，IC应该为正
        assert ic > 0
    
    def test_ic_significance(self):
        """测试IC显著性"""
        from scipy import stats
        
        np.random.seed(42)
        n = 100
        
        factor = np.random.randn(n)
        future_return = 0.3 * factor + np.random.randn(n) * 0.5
        
        # 计算IC和p值
        ic, p_value = stats.pearsonr(factor, future_return)
        
        # p值应小于0.05才显著
        assert p_value < 0.05


class TestFactorSelection:
    """因子筛选测试"""
    
    def test_factor_correlation_filter(self):
        """测试因子相关性过滤"""
        np.random.seed(42)
        n = 100
        
        # 创建高度相关的两个因子
        base = np.random.randn(n)
        factor1 = base
        factor2 = base + np.random.randn(n) * 0.1  # 高度相关
        factor3 = np.random.randn(n)  # 不相关
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef([factor1, factor2, factor3])
        
        # factor1和factor2应该高度相关
        assert abs(corr_matrix[0, 1]) > 0.8
        # factor1和factor3应该低相关
        assert abs(corr_matrix[0, 2]) < 0.5
    
    def test_factor_ranking(self):
        """测试因子排序"""
        factors = pd.DataFrame({
            "factor_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "factor_b": [0.5, 0.4, 0.3, 0.2, 0.1],
            "ic": [0.05, 0.03, 0.08, 0.02, 0.06],
        })
        
        # 按IC排序
        ranked = factors.sort_values("ic", ascending=False)
        
        assert ranked.iloc[0]["factor_a"] == 0.3  # IC最高的
        assert ranked.iloc[0]["ic"] == 0.08
