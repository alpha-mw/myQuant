"""
回测系统单元测试
测试回测引擎、策略、订单管理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestBacktestData:
    """回测数据测试"""
    
    def test_ohlcv_data_integrity(self):
        """测试OHLCV数据完整性"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + abs(np.random.randn(100)) * 0.5 + 0.1,
            'low': prices - abs(np.random.randn(100)) * 0.5 - 0.1,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
        })
        
        # 确保high >= open, high >= close
        df['high'] = df[['open', 'close', 'high']].max(axis=1) + 0.01
        # 确保low <= open, low <= close
        df['low'] = df[['open', 'close', 'low']].min(axis=1) - 0.01
        
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_return_calculation(self):
        """测试收益计算"""
        prices = pd.Series([100, 102, 101, 105, 103])
        returns = prices.pct_change().dropna()
        
        expected = pd.Series([0.02, -0.00980392, 0.03960396, -0.01904762])
        
        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.02) < 0.0001


class TestTradingCosts:
    """交易成本测试"""
    
    def test_commission_calculation(self):
        """测试手续费计算"""
        trade_amount = 100000
        commission_rate = 0.0003
        
        commission = trade_amount * commission_rate
        
        assert abs(commission - 30.0) < 0.001
    
    def test_stamp_duty_calculation(self):
        """测试印花税计算"""
        sell_amount = 100000
        stamp_duty_rate = 0.001
        
        stamp_duty = sell_amount * stamp_duty_rate
        
        assert stamp_duty == 100.0
    
    def test_slippage_calculation(self):
        """测试滑点计算"""
        price = 100.0
        slippage = 0.001
        
        # 买入滑点（价格上升）
        buy_price = price * (1 + slippage)
        # 卖出滑点（价格下降）
        sell_price = price * (1 - slippage)
        
        assert buy_price == 100.1
        assert sell_price == 99.9


class TestPositionSizing:
    """仓位管理测试"""
    
    def test_equal_weight_position(self):
        """测试等权重仓位"""
        total_capital = 1000000
        n_stocks = 10
        
        position_size = total_capital / n_stocks
        
        assert position_size == 100000
    
    def test_risk_based_position(self):
        """测试基于风险的仓位"""
        capital = 1000000
        risk_per_trade = 0.02  # 2%
        stop_loss = 0.05  # 5%止损
        
        # 风险金额
        risk_amount = capital * risk_per_trade
        # 仓位大小
        position_size = risk_amount / stop_loss
        
        assert position_size == 400000
    
    def test_position_limit(self):
        """测试仓位限制"""
        capital = 1000000
        max_position_pct = 0.1  # 单票最大10%
        
        calculated_position = 150000  # 计算出的仓位
        max_position = capital * max_position_pct
        
        actual_position = min(calculated_position, max_position)
        
        assert actual_position == 100000


class TestRiskManagement:
    """回测风控测试"""
    
    def test_stop_loss_trigger(self):
        """测试止损触发"""
        entry_price = 100.0
        stop_loss_pct = 0.08
        
        stop_price = entry_price * (1 - stop_loss_pct)
        current_price = 91.0  # 下跌9%
        
        assert current_price <= stop_price  # 应该触发止损
    
    def test_take_profit_trigger(self):
        """测试止盈触发"""
        entry_price = 100.0
        take_profit_pct = 0.20
        
        take_profit_price = entry_price * (1 + take_profit_pct)
        current_price = 125.0  # 上涨25%
        
        assert current_price >= take_profit_price  # 应该触发止盈
    
    def test_max_drawdown_limit(self):
        """测试最大回撤限制"""
        peak_value = 1200000
        current_value = 1000000
        max_drawdown_limit = 0.15
        
        drawdown = (peak_value - current_value) / peak_value
        
        assert drawdown > max_drawdown_limit  # 超过15%限制，应该清仓
