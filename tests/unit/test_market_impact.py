"""
市场冲击成本测试
"""

import pytest
import numpy as np
import pandas as pd
from market_impact import MarketImpactModel, AdvancedCostModel, estimate_optimal_trade_size


class TestMarketImpactModel:
    """市场冲击模型测试"""
    
    def test_market_impact_calculation(self):
        """测试市场冲击计算"""
        model = MarketImpactModel()
        
        trade_volume = 10000
        avg_daily_volume = 100000
        price = 100.0
        
        impact = model.calculate_impact(trade_volume, avg_daily_volume, price)
        
        # 检查冲击成本为正
        assert impact.temporary_impact > 0
        assert impact.permanent_impact > 0
        assert impact.total_cost > 0
        
        # 总成本 = 临时 + 永久
        assert abs(impact.total_cost - (impact.temporary_impact + impact.permanent_impact)) < 0.0001
    
    def test_impact_increases_with_volume(self):
        """测试冲击随成交量增加"""
        model = MarketImpactModel()
        
        small_trade = model.calculate_impact(1000, 100000, 100.0)
        large_trade = model.calculate_impact(50000, 100000, 100.0)
        
        # 大交易的冲击成本应该更高
        assert large_trade.total_cost > small_trade.total_cost
    
    def test_slippage_estimation(self):
        """测试滑点估计"""
        model = MarketImpactModel()
        
        slippage = model.estimate_slippage(10000, 100000, bid_ask_spread=0.001)
        
        # 滑点应该在合理范围内
        assert slippage > 0
        assert slippage < 0.01  # 最大1%


class TestAdvancedCostModel:
    """高级成本模型测试"""
    
    def test_total_cost_calculation(self):
        """测试总成本计算"""
        model = AdvancedCostModel()
        
        costs = model.calculate_total_cost(
            trade_amount=100000,
            trade_volume=1000,
            avg_daily_volume=10000,
            is_buy=True
        )
        
        # 检查所有成本项
        assert 'commission' in costs
        assert 'stamp_duty' in costs
        assert 'slippage' in costs
        assert 'market_impact' in costs
        assert 'total_cost' in costs
        
        # 买入没有印花税
        assert costs['stamp_duty'] == 0
        
        # 总成本为正
        assert costs['total_cost'] > 0
    
    def test_sell_cost_includes_stamp_duty(self):
        """测试卖出成本包含印花税"""
        model = AdvancedCostModel()
        
        costs = model.calculate_total_cost(
            trade_amount=100000,
            trade_volume=1000,
            avg_daily_volume=10000,
            is_buy=False  # 卖出
        )
        
        # 卖出有印花税
        assert costs['stamp_duty'] > 0


class TestOptimalTradeSize:
    """最优交易规模测试"""
    
    def test_optimal_trade_size_calculation(self):
        """测试最优交易规模计算"""
        optimal_size = estimate_optimal_trade_size(
            avg_daily_volume=100000,
            max_impact_pct=0.01,
            volatility=0.25
        )
        
        # 最优规模应该小于日均成交量
        assert optimal_size < 100000
        assert optimal_size > 0
    
    def test_optimal_size_increases_with_volume(self):
        """测试最优规模随成交量增加"""
        small_adv = estimate_optimal_trade_size(100000)
        large_adv = estimate_optimal_trade_size(1000000)
        
        # 日均成交量越大，最优交易规模越大
        assert large_adv > small_adv
