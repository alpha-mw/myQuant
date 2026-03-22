"""
Quant-Investor V7.0 市场冲击成本模型
实现更精细的交易成本建模
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class MarketImpactCost:
    """市场冲击成本"""
    temporary_impact: float  # 临时冲击成本
    permanent_impact: float  # 永久冲击成本
    total_cost: float        # 总成本


class MarketImpactModel:
    """
    市场冲击模型
    
    参考Almgren-Chriss模型:
    Cost = 临时冲击 + 永久冲击
         = (成交量比率)^0.5 * 波动率 + 成交量比率 * 波动率
    """
    
    def __init__(self,
                 eta: float = 0.142,      # 临时冲击系数
                 gamma: float = 0.314,    # 永久冲击系数
                 sigma: Optional[float] = None):  # 波动率
        """
        初始化市场冲击模型
        
        Args:
            eta: 临时冲击系数 (Almgren-Chriss默认值0.142)
            gamma: 永久冲击系数 (Almgren-Chriss默认值0.314)
            sigma: 年化波动率 (默认None，从数据计算)
        """
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
    
    def calculate_impact(self,
                        trade_volume: float,
                        avg_daily_volume: float,
                        price: float,
                        volatility: Optional[float] = None) -> MarketImpactCost:
        """
        计算市场冲击成本
        
        Args:
            trade_volume: 交易量
            avg_daily_volume: 日均成交量
            price: 当前价格
            volatility: 年化波动率 (可选)
            
        Returns:
            市场冲击成本
        """
        # 使用传入的波动率或默认值
        sigma = volatility if volatility is not None else (self.sigma or 0.25)
        
        # 成交量比率 (X/V)
        volume_ratio = trade_volume / avg_daily_volume if avg_daily_volume > 0 else 0
        
        # 临时冲击成本 (与成交量比率的平方根成正比)
        temporary_impact = self.eta * sigma * np.sqrt(volume_ratio)
        
        # 永久冲击成本 (与成交量比率成正比)
        permanent_impact = self.gamma * sigma * volume_ratio
        
        # 总成本
        total_cost = temporary_impact + permanent_impact
        
        return MarketImpactCost(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_cost=total_cost
        )
    
    def estimate_slippage(self,
                         trade_volume: float,
                         avg_daily_volume: float,
                         bid_ask_spread: float = 0.001) -> float:
        """
        估计滑点
        
        Args:
            trade_volume: 交易量
            avg_daily_volume: 日均成交量
            bid_ask_spread: 买卖价差 (默认0.1%)
            
        Returns:
            估计滑点
        """
        # 基础滑点 = 买卖价差的一半
        base_slippage = bid_ask_spread / 2
        
        # 成交量比率
        volume_ratio = trade_volume / avg_daily_volume if avg_daily_volume > 0 else 0
        
        # 滑点随成交量增加而增加
        slippage = base_slippage * (1 + volume_ratio)
        
        return min(slippage, 0.01)  # 最大1%


class AdvancedCostModel:
    """
    高级交易成本模型
    整合所有成本类型
    """
    
    def __init__(self,
                 commission_rate: float = 0.0003,      # 手续费0.03%
                 stamp_duty_rate: float = 0.001,       # 印花税0.1%
                 slippage_rate: float = 0.001,         # 滑点0.1%
                 market_impact_model: Optional[MarketImpactModel] = None):
        """
        初始化成本模型
        
        Args:
            commission_rate: 手续费率
            stamp_duty_rate: 印花税率
            slippage_rate: 滑点率
            market_impact_model: 市场冲击模型
        """
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate
        self.market_impact = market_impact_model or MarketImpactModel()
    
    def calculate_total_cost(self,
                            trade_amount: float,
                            trade_volume: float,
                            avg_daily_volume: float,
                            is_buy: bool = True,
                            price: float = 1.0,
                            volatility: Optional[float] = None) -> dict:
        """
        计算总交易成本
        
        Args:
            trade_amount: 交易金额
            trade_volume: 交易量
            avg_daily_volume: 日均成交量
            is_buy: 是否买入
            price: 价格
            volatility: 波动率
            
        Returns:
            成本明细字典
        """
        # 1. 手续费 (买卖都收)
        commission = trade_amount * self.commission_rate
        
        # 2. 印花税 (仅卖出)
        stamp_duty = trade_amount * self.stamp_duty_rate if not is_buy else 0
        
        # 3. 滑点
        slippage = trade_amount * self.slippage_rate
        
        # 4. 市场冲击成本
        impact = self.market_impact.calculate_impact(
            trade_volume, avg_daily_volume, price, volatility
        )
        market_impact_cost = trade_amount * impact.total_cost
        
        # 总成本
        total_cost = commission + stamp_duty + slippage + market_impact_cost
        
        return {
            'commission': commission,
            'stamp_duty': stamp_duty,
            'slippage': slippage,
            'market_impact': market_impact_cost,
            'temporary_impact': trade_amount * impact.temporary_impact,
            'permanent_impact': trade_amount * impact.permanent_impact,
            'total_cost': total_cost,
            'cost_percentage': total_cost / trade_amount if trade_amount > 0 else 0,
        }


def estimate_optimal_trade_size(avg_daily_volume: float,
                                 max_impact_pct: float = 0.01,
                                 volatility: float = 0.25) -> float:
    """
    估计最优交易规模
    
    Args:
        avg_daily_volume: 日均成交量
        max_impact_pct: 最大冲击成本比例 (默认1%)
        volatility: 年化波动率
        
    Returns:
        最优交易规模
    """
    # Almgren-Chriss逆推
    # 假设总成本 = 临时冲击 + 永久冲击 ≈ 1%
    # 1% = 0.142 * 0.25 * sqrt(X/V) + 0.314 * 0.25 * (X/V)
    
    model = MarketImpactModel()
    
    # 数值求解
    best_ratio = 0
    best_diff = float('inf')
    
    for ratio in np.linspace(0.001, 0.5, 1000):
        impact = model.calculate_impact(ratio * avg_daily_volume, avg_daily_volume, 1.0, volatility)
        diff = abs(impact.total_cost - max_impact_pct)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
    
    return best_ratio * avg_daily_volume
