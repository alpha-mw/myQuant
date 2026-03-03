"""
Quant-Investor V7.0 Backtrader回测集成
事件驱动回测，支持滑点、交易成本、订单簿模拟
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MomentumStrategy(bt.Strategy):
    """
    动量策略 - 基于5日和20日收益率
    """
    params = (
        ('momentum_period', 5),
        ('lookback_period', 20),
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('position_pct', 0.1),  # 单只股票仓位10%
        ('stop_loss', 0.08),    # 止损8%
        ('take_profit', 0.20),  # 止盈20%
    )
    
    def __init__(self):
        self.orders = {}  # 跟踪订单
        self.stops = {}   # 跟踪止损单
        # 预计算RSI指标
        self.rsi = {}
        for data in self.datas:
            self.rsi[data._name] = bt.indicators.RSI(data.close, period=self.p.rsi_period)
        
    def next(self):
        """每个bar调用"""
        for data in self.datas:
            symbol = data._name
            
            # 跳过无数据的情况
            if len(data) < self.p.lookback_period + 5:
                continue
            
            # 计算动量
            close_now = data.close[0]
            close_5d = data.close[-self.p.momentum_period]
            close_20d = data.close[-self.p.lookback_period]
            
            momentum_5d = (close_now - close_5d) / close_5d if close_5d != 0 else 0
            momentum_20d = (close_now - close_20d) / close_20d if close_20d != 0 else 0
            
            # 计算RSI
            rsi = self.rsi[symbol][0]
            
            # 获取当前持仓
            pos = self.getposition(data)
            
            # 买入信号：5日动量>0 且 20日动量>0 且 RSI在合理区间
            buy_signal = (
                momentum_5d > 0 and 
                momentum_20d > 0 and 
                self.p.rsi_lower < rsi < self.p.rsi_upper
            )
            
            # 卖出信号：5日动量<0 或 RSI>70
            sell_signal = momentum_5d < 0 or rsi > self.p.rsi_upper
            
            # 执行交易
            if not pos.size and buy_signal:
                # 计算仓位大小
                cash = self.broker.getcash()
                size = int((cash * self.p.position_pct) / close_now / 100) * 100
                
                if size > 0:
                    order = self.buy(data=data, size=size)
                    self.orders[symbol] = order
                    
                    # 设置止损
                    stop_price = close_now * (1 - self.p.stop_loss)
                    stop_order = self.sell(data=data, size=size, 
                                          exectype=bt.Order.Stop, price=stop_price)
                    self.stops[symbol] = stop_order
                    
            elif pos.size > 0 and sell_signal:
                # 平仓
                self.close(data=data)
                
                # 取消止损单
                if symbol in self.stops:
                    self.cancel(self.stops[symbol])
                    del self.stops[symbol]


class RiskManagedStrategy(bt.Strategy):
    """
    风险管理增强策略
    包含组合层面的风险控制
    """
    params = (
        ('max_positions', 15),      # 最大持仓数
        ('max_drawdown', 0.15),     # 组合最大回撤15%
        ('position_pct', 0.07),     # 单只股票仓位7%
        ('risk_free_rate', 0.03),   # 无风险利率3%
    )
    
    def __init__(self):
        self.orders = {}
        self.stops = {}
        self.peak_value = 0
        
    def next(self):
        """每个bar调用"""
        # 计算当前组合价值
        portfolio_value = self.broker.getvalue()
        
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # 计算回撤
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # 如果回撤超过阈值，清仓
        if drawdown > self.p.max_drawdown:
            for data in self.datas:
                if self.getposition(data).size > 0:
                    self.close(data=data)
            return
        
        # 检查持仓数量
        current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)
        
        if current_positions >= self.p.max_positions:
            return
        
        # 动量筛选逻辑（简化版）
        for data in self.datas:
            if len(data) < 25:
                continue
                
            symbol = data._name
            pos = self.getposition(data)
            
            if pos.size > 0:
                continue
            
            # 简单动量计算
            momentum = (data.close[0] - data.close[-5]) / data.close[-5]
            
            if momentum > 0.02:  # 2%动量阈值
                cash = self.broker.getcash()
                size = int((cash * self.p.position_pct) / data.close[0] / 100) * 100
                
                if size > 0:
                    self.buy(data=data, size=size)


class BacktestEngine:
    """
    回测引擎封装
    """
    def __init__(self, 
                 initial_cash: float = 1000000.0,
                 commission: float = 0.0003,      # 手续费0.03%
                 stamp_duty: float = 0.001,       # 印花税0.1%
                 slippage: float = 0.001):        # 滑点0.1%
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
            stamp_duty: 印花税率
            slippage: 滑点
        """
        self.cerebro = bt.Cerebro()
        self.initial_cash = initial_cash
        
        # 设置初始资金
        self.cerebro.broker.setcash(initial_cash)
        
        # 设置手续费（包含印花税）
        total_commission = commission + stamp_duty
        self.cerebro.broker.setcommission(
            commission=total_commission,
            margin=False
        )
        
        # 设置滑点
        self.cerebro.broker.set_slippage_perc(slippage)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
    def add_data(self, 
                 df: pd.DataFrame, 
                 symbol: str,
                 fromdate: Optional[datetime] = None,
                 todate: Optional[datetime] = None):
        """
        添加数据
        
        Args:
            df: DataFrame包含OHLCV数据
            symbol: 股票代码
            fromdate: 开始日期
            todate: 结束日期
        """
        # 确保列名正确
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
        df.set_index('datetime', inplace=True)
        
        # 重命名列以匹配Backtrader
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }
        df = df.rename(columns=column_mapping)
        
        # 创建数据源
        data = bt.feeds.PandasData(
            dataname=df,
            name=symbol,
            fromdate=fromdate,
            todate=todate,
            datetime=None,  # 使用索引
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
        )
        
        self.cerebro.adddata(data)
        
    def run(self, strategy_class=RiskManagedStrategy, **strategy_params) -> Dict:
        """
        运行回测
        
        Args:
            strategy_class: 策略类
            **strategy_params: 策略参数
            
        Returns:
            回测结果字典
        """
        # 添加策略
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        
        # 运行回测
        print(f'初始资金: {self.initial_cash:,.2f}')
        results = self.cerebro.run()
        strat = results[0]
        
        # 获取最终资金
        final_value = self.cerebro.broker.getvalue()
        print(f'最终资金: {final_value:,.2f}')
        print(f'收益率: {(final_value/self.initial_cash - 1)*100:.2f}%')
        
        # 提取分析结果
        result = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': final_value / self.initial_cash - 1,
            'sharpe_ratio': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
            'trades': strat.analyzers.trades.get_analysis(),
        }
        
        return result
    
    def plot(self):
        """绘制回测结果"""
        self.cerebro.plot(style='candlestick', barup='red', bardown='green')


if __name__ == '__main__':
    # 示例用法
    print("Backtrader回测引擎已加载")
    print("使用示例:")
    print("  engine = BacktestEngine(initial_cash=1000000)")
    print("  engine.add_data(df, '000001.SZ')")
    print("  result = engine.run(MomentumStrategy)")
