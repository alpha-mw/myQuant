"""
Quant-Investor V7.0 压力测试模块
实现多种压力测试场景
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from dataclasses import dataclass


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    portfolio_value_before: float
    portfolio_value_after: float
    loss_amount: float
    loss_percentage: float
    var_95: float
    var_99: float
    max_drawdown: float


class StressTester:
    """压力测试器"""
    
    def __init__(self, returns: Union[pd.Series, np.ndarray]):
        """
        初始化压力测试器
        
        Args:
            returns: 历史收益率序列
        """
        self.returns = np.array(returns).flatten()
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns)
        
    def correlation_stress(self, correlation_increase: float = 0.3) -> StressTestResult:
        """
        相关性冲击场景
        模拟市场恐慌时资产相关性趋近于1的情况
        
        Args:
            correlation_increase: 相关性增加幅度（默认0.3）
            
        Returns:
            压力测试结果
        """
        # 模拟相关性增加：收益率向均值靠拢
        stressed_returns = self.returns * (1 - correlation_increase) + self.mean * correlation_increase
        
        # 计算损失
        portfolio_value = 1000000  # 假设初始资金100万
        final_value = portfolio_value * np.prod(1 + stressed_returns)
        loss = portfolio_value - final_value
        
        return StressTestResult(
            scenario_name="相关性冲击",
            portfolio_value_before=portfolio_value,
            portfolio_value_after=float(final_value),
            loss_amount=float(loss),
            loss_percentage=float(loss / portfolio_value),
            var_95=float(np.percentile(stressed_returns, 5)),
            var_99=float(np.percentile(stressed_returns, 1)),
            max_drawdown=self._calculate_max_drawdown(stressed_returns)
        )
    
    def liquidity_stress(self, liquidity_shock: float = -0.05) -> StressTestResult:
        """
        流动性冲击场景
        模拟市场流动性枯竭，价格暴跌
        
        Args:
            liquidity_shock: 流动性冲击幅度（默认-5%）
            
        Returns:
            压力测试结果
        """
        # 模拟流动性冲击：所有收益率叠加负向冲击
        stressed_returns = self.returns + liquidity_shock
        
        portfolio_value = 1000000
        final_value = portfolio_value * np.prod(1 + stressed_returns)
        loss = portfolio_value - final_value
        
        return StressTestResult(
            scenario_name="流动性冲击",
            portfolio_value_before=portfolio_value,
            portfolio_value_after=float(final_value),
            loss_amount=float(loss),
            loss_percentage=float(loss / portfolio_value),
            var_95=float(np.percentile(stressed_returns, 5)),
            var_99=float(np.percentile(stressed_returns, 1)),
            max_drawdown=self._calculate_max_drawdown(stressed_returns)
        )
    
    def volatility_spike(self, volatility_multiplier: float = 2.0) -> StressTestResult:
        """
        波动率飙升场景
        模拟市场波动率突然增加
        
        Args:
            volatility_multiplier: 波动率乘数（默认2倍）
            
        Returns:
            压力测试结果
        """
        # 模拟波动率飙升：收益率标准差增加
        stressed_returns = np.random.normal(self.mean, self.std * volatility_multiplier, 
                                           len(self.returns))
        
        portfolio_value = 1000000
        final_value = portfolio_value * np.prod(1 + stressed_returns)
        loss = portfolio_value - final_value
        
        return StressTestResult(
            scenario_name="波动率飙升",
            portfolio_value_before=portfolio_value,
            portfolio_value_after=float(final_value),
            loss_amount=float(loss),
            loss_percentage=float(loss / portfolio_value),
            var_95=float(np.percentile(stressed_returns, 5)),
            var_99=float(np.percentile(stressed_returns, 1)),
            max_drawdown=self._calculate_max_drawdown(stressed_returns)
        )
    
    def crisis_2008(self) -> StressTestResult:
        """
        2008年金融危机场景
        模拟2008年式金融危机
        
        Returns:
            压力测试结果
        """
        # 模拟2008危机：大幅下跌，高波动
        crisis_returns = np.random.normal(-0.02, 0.05, len(self.returns))
        
        portfolio_value = 1000000
        final_value = portfolio_value * np.prod(1 + crisis_returns)
        loss = portfolio_value - final_value
        
        return StressTestResult(
            scenario_name="2008金融危机",
            portfolio_value_before=portfolio_value,
            portfolio_value_after=float(final_value),
            loss_amount=float(loss),
            loss_percentage=float(loss / portfolio_value),
            var_95=float(np.percentile(crisis_returns, 5)),
            var_99=float(np.percentile(crisis_returns, 1)),
            max_drawdown=self._calculate_max_drawdown(crisis_returns)
        )
    
    def crisis_2015_chn(self) -> StressTestResult:
        """
        2015年A股熔断场景
        模拟2015年A股熔断危机
        
        Returns:
            压力测试结果
        """
        # 模拟2015A股熔断：连续跌停，流动性枯竭
        circuit_breaker_returns = np.random.normal(-0.03, 0.04, len(self.returns))
        # 添加连续跌停效应
        for i in range(1, len(circuit_breaker_returns)):
            if circuit_breaker_returns[i-1] < -0.05:
                circuit_breaker_returns[i] -= 0.02
        
        portfolio_value = 1000000
        final_value = portfolio_value * np.prod(1 + circuit_breaker_returns)
        loss = portfolio_value - final_value
        
        return StressTestResult(
            scenario_name="2015A股熔断",
            portfolio_value_before=portfolio_value,
            portfolio_value_after=float(final_value),
            loss_amount=float(loss),
            loss_percentage=float(loss / portfolio_value),
            var_95=float(np.percentile(circuit_breaker_returns, 5)),
            var_99=float(np.percentile(circuit_breaker_returns, 1)),
            max_drawdown=self._calculate_max_drawdown(circuit_breaker_returns)
        )
    
    def run_all_stress_tests(self) -> List[StressTestResult]:
        """
        运行所有压力测试场景
        
        Returns:
            所有场景的结果列表
        """
        scenarios = [
            self.correlation_stress(),
            self.liquidity_stress(),
            self.volatility_spike(),
            self.crisis_2008(),
            self.crisis_2015_chn(),
        ]
        return scenarios
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return np.min(drawdown)


def print_stress_test_report(results: List[StressTestResult]):
    """打印压力测试报告"""
    print("\n" + "=" * 80)
    print("压力测试报告")
    print("=" * 80)
    
    for result in results:
        print(f"\n📊 {result.scenario_name}")
        print(f"  初始资金: {result.portfolio_value_before:,.2f}")
        print(f"  最终资金: {result.portfolio_value_after:,.2f}")
        print(f"  损失金额: {result.loss_amount:,.2f}")
        print(f"  损失比例: {result.loss_percentage*100:.2f}%")
        print(f"  VaR(95%): {result.var_95*100:.2f}%")
        print(f"  VaR(99%): {result.var_99*100:.2f}%")
        print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
    
    print("\n" + "=" * 80)
