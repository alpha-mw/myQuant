#!/usr/bin/env python3
"""
增强回测引擎 (Enhanced Backtesting Engine) - quant-investor V2.3

引入滑点、市场冲击和交易限制等更真实的交易成本模型，使回测结果更接近实盘。

作者: Manus AI
日期: 2026-01-31
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """订单类型"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradingCostModel:
    """交易成本模型配置"""
    
    # 佣金率（双边）
    commission_rate: float = 0.0003  # 0.03%
    
    # 印花税（仅卖出，A股特有）
    stamp_tax_rate: float = 0.001  # 0.1%
    
    # 滑点模型
    slippage_type: str = "fixed"  # "fixed" 或 "volume_based"
    slippage_rate: float = 0.001  # 固定滑点率 0.1%
    
    # 市场冲击模型参数
    market_impact_coef: float = 0.1  # 市场冲击系数
    
    # 最小交易单位（A股为100股）
    min_trade_unit: int = 100


class EnhancedBacktestEngine:
    """
    增强回测引擎
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        cost_model: Optional[TradingCostModel] = None
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            cost_model: 交易成本模型
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TradingCostModel()
        
        # 回测状态
        self.cash = initial_capital
        self.positions = {}  # {stock_code: shares}
        self.trades = []  # 交易记录
        self.portfolio_values = []  # 组合价值历史
        
        print(f"[EnhancedBacktest] 初始资金: {initial_capital:,.0f}")
        print(f"[EnhancedBacktest] 交易成本模型:")
        print(f"  - 佣金率: {self.cost_model.commission_rate * 100:.3f}%")
        print(f"  - 印花税: {self.cost_model.stamp_tax_rate * 100:.3f}%")
        print(f"  - 滑点率: {self.cost_model.slippage_rate * 100:.3f}%")
    
    def execute_order(
        self,
        stock_code: str,
        order_type: OrderType,
        target_shares: int,
        current_price: float,
        date: str,
        volume: float = None
    ) -> Dict:
        """
        执行订单
        
        Args:
            stock_code: 股票代码
            order_type: 订单类型
            target_shares: 目标股数
            current_price: 当前价格
            date: 交易日期
            volume: 当日成交量（用于计算市场冲击）
            
        Returns:
            交易记录字典
        """
        # 计算实际成交价
        execution_price = self._calculate_execution_price(
            current_price, order_type, target_shares, volume
        )
        
        # 计算交易成本
        trade_value = target_shares * execution_price
        commission = trade_value * self.cost_model.commission_rate
        stamp_tax = trade_value * self.cost_model.stamp_tax_rate if order_type == OrderType.SELL else 0
        total_cost = commission + stamp_tax
        
        # 更新持仓和现金
        current_shares = self.positions.get(stock_code, 0)
        
        if order_type == OrderType.BUY:
            total_payment = trade_value + total_cost
            if total_payment > self.cash:
                print(f"[Warning] 资金不足，无法买入 {stock_code}")
                return None
            
            self.cash -= total_payment
            self.positions[stock_code] = current_shares + target_shares
            
        else:  # SELL
            if target_shares > current_shares:
                print(f"[Warning] 持仓不足，无法卖出 {stock_code}")
                return None
            
            total_receipt = trade_value - total_cost
            self.cash += total_receipt
            self.positions[stock_code] = current_shares - target_shares
            
            if self.positions[stock_code] == 0:
                del self.positions[stock_code]
        
        # 记录交易
        trade_record = {
            'date': date,
            'stock_code': stock_code,
            'order_type': order_type.value,
            'shares': target_shares,
            'price': current_price,
            'execution_price': execution_price,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'total_cost': total_cost,
            'cash_after': self.cash
        }
        
        self.trades.append(trade_record)
        
        return trade_record
    
    def _calculate_execution_price(
        self,
        current_price: float,
        order_type: OrderType,
        shares: int,
        volume: Optional[float]
    ) -> float:
        """
        计算实际成交价（考虑滑点和市场冲击）
        
        Args:
            current_price: 当前价格
            order_type: 订单类型
            shares: 股数
            volume: 成交量
            
        Returns:
            实际成交价
        """
        # 基础滑点
        if self.cost_model.slippage_type == "fixed":
            slippage = current_price * self.cost_model.slippage_rate
        else:
            # 基于成交量的滑点（简化模型）
            if volume is not None and volume > 0:
                volume_ratio = shares / volume
                slippage = current_price * self.cost_model.slippage_rate * (1 + volume_ratio * 10)
            else:
                slippage = current_price * self.cost_model.slippage_rate
        
        # 市场冲击（简化模型）
        if volume is not None and volume > 0:
            volume_ratio = shares / volume
            market_impact = current_price * self.cost_model.market_impact_coef * volume_ratio
        else:
            market_impact = 0
        
        # 计算实际成交价
        if order_type == OrderType.BUY:
            execution_price = current_price + slippage + market_impact
        else:
            execution_price = current_price - slippage - market_impact
        
        return max(execution_price, 0.01)  # 确保价格为正
    
    def calculate_portfolio_value(self, prices: Dict[str, float], date: str) -> float:
        """
        计算当前组合价值
        
        Args:
            prices: 股票代码到当前价格的字典
            date: 日期
            
        Returns:
            组合总价值
        """
        position_value = sum(
            self.positions.get(stock_code, 0) * prices.get(stock_code, 0)
            for stock_code in self.positions
        )
        
        total_value = self.cash + position_value
        
        self.portfolio_values.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value
        })
        
        return total_value
    
    def get_performance_metrics(self) -> Dict:
        """
        计算回测性能指标
        
        Returns:
            性能指标字典
        """
        if not self.portfolio_values:
            return {}
        
        df = pd.DataFrame(self.portfolio_values)
        df['returns'] = df['total_value'].pct_change()
        
        # 总收益率
        total_return = (df['total_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率（假设252个交易日）
        n_days = len(df)
        annualized_return = ((df['total_value'].iloc[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03 / 252
        excess_returns = df['returns'] - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # 最大回撤
        cummax = df['total_value'].cummax()
        drawdown = (df['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': df['total_value'].iloc[-1],
            'n_trades': len(self.trades)
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_portfolio_values_df(self) -> pd.DataFrame:
        """获取组合价值历史DataFrame"""
        return pd.DataFrame(self.portfolio_values)


def demo():
    """演示增强回测引擎的使用"""
    print("=" * 60)
    print("增强回测引擎演示")
    print("=" * 60)
    
    # 创建回测引擎
    engine = EnhancedBacktestEngine(initial_capital=100000)
    
    # 模拟交易
    print("\n【模拟交易】")
    
    # 第1天：买入股票A
    trade1 = engine.execute_order(
        stock_code='600000.SH',
        order_type=OrderType.BUY,
        target_shares=1000,
        current_price=10.0,
        date='2020-01-02',
        volume=1000000
    )
    print(f"交易1: {trade1['date']} {trade1['order_type']} {trade1['stock_code']} "
          f"{trade1['shares']}股 @ {trade1['execution_price']:.2f}")
    
    # 第2天：计算组合价值
    portfolio_value = engine.calculate_portfolio_value(
        prices={'600000.SH': 10.5},
        date='2020-01-03'
    )
    print(f"第2天组合价值: {portfolio_value:,.2f}")
    
    # 第3天：卖出部分股票
    trade2 = engine.execute_order(
        stock_code='600000.SH',
        order_type=OrderType.SELL,
        target_shares=500,
        current_price=11.0,
        date='2020-01-04',
        volume=1000000
    )
    print(f"交易2: {trade2['date']} {trade2['order_type']} {trade2['stock_code']} "
          f"{trade2['shares']}股 @ {trade2['execution_price']:.2f}")
    
    # 第4天：计算最终组合价值
    portfolio_value = engine.calculate_portfolio_value(
        prices={'600000.SH': 11.5},
        date='2020-01-05'
    )
    print(f"第4天组合价值: {portfolio_value:,.2f}")
    
    # 性能指标
    print("\n【性能指标】")
    metrics = engine.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # 交易记录
    print("\n【交易记录】")
    trades_df = engine.get_trades_df()
    print(trades_df[['date', 'stock_code', 'order_type', 'shares', 'execution_price', 'total_cost']])


if __name__ == "__main__":
    demo()
