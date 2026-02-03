"""
风险度量指标计算模块 (Risk Metrics Calculator)

提供全面的风险度量指标计算，包括：
1. 风险调整后收益指标：Sharpe, Sortino, Calmar, Information Ratio, Treynor
2. 绝对风险度量：波动率、回撤分析
3. 尾部风险度量：VaR, CVaR (ES)
4. 相对风险度量：Beta, 跟踪误差, Capture Ratios

V2.8 - 风险管理模块
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats
from dataclasses import dataclass
from enum import Enum


class VaRMethod(Enum):
    """VaR计算方法"""
    HISTORICAL = "historical"       # 历史模拟法
    PARAMETRIC = "parametric"       # 参数法（方差-协方差法）
    MONTE_CARLO = "monte_carlo"     # 蒙特卡洛模拟法


@dataclass
class DrawdownInfo:
    """回撤信息"""
    start_date: pd.Timestamp       # 回撤开始日期
    trough_date: pd.Timestamp      # 最低点日期
    end_date: Optional[pd.Timestamp]  # 恢复日期（如未恢复则为None）
    drawdown: float                # 回撤幅度（负数）
    duration: int                  # 回撤持续天数
    recovery_duration: Optional[int]  # 恢复天数


class RiskMetrics:
    """
    风险度量指标计算器
    
    提供全面的风险度量指标计算功能，支持策略评估和风险分析。
    """
    
    # 默认参数
    DEFAULT_RISK_FREE_RATE = 0.02  # 默认无风险利率 2%
    DEFAULT_TRADING_DAYS = 252     # 年化交易日数
    DEFAULT_CONFIDENCE_LEVEL = 0.95  # 默认置信水平
    
    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        trading_days: int = DEFAULT_TRADING_DAYS
    ):
        """
        初始化风险度量计算器
        
        Args:
            risk_free_rate: 年化无风险利率
            trading_days: 年化交易日数
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_rf = risk_free_rate / trading_days
    
    # ==================== 基础统计量 ====================
    
    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = 'simple'
    ) -> pd.Series:
        """
        计算收益率序列
        
        Args:
            prices: 价格序列
            method: 'simple' (简单收益率) 或 'log' (对数收益率)
            
        Returns:
            收益率序列
        """
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()
    
    def annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        return (1 + total_return) ** (self.trading_days / n_periods) - 1
    
    def annualized_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        return returns.std() * np.sqrt(self.trading_days)
    
    def downside_volatility(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        计算下行波动率
        
        Args:
            returns: 收益率序列
            threshold: 下行阈值，默认为0
            
        Returns:
            年化下行波动率
        """
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(self.trading_days)
    
    # ==================== 风险调整后收益指标 ====================
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        计算Sharpe比率
        
        衡量每单位总风险（标准差）获得的超额收益。
        
        Args:
            returns: 收益率序列
            
        Returns:
            Sharpe比率
        """
        if len(returns) == 0:
            return np.nan
        excess_returns = returns - self.daily_rf
        std = returns.std()
        if std == 0 or std < 1e-10:  # 处理常数收益率情况
            return 0.0 if excess_returns.mean() <= 0 else np.inf
        return np.sqrt(self.trading_days) * excess_returns.mean() / std
    
    def sortino_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        计算Sortino比率
        
        只考虑下行波动率，更关注负面风险。
        
        Args:
            returns: 收益率序列
            threshold: 下行阈值
            
        Returns:
            Sortino比率
        """
        excess_returns = returns - self.daily_rf
        downside_std = self.downside_volatility(returns, threshold) / np.sqrt(self.trading_days)
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        return np.sqrt(self.trading_days) * excess_returns.mean() / downside_std
    
    def calmar_ratio(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> float:
        """
        计算Calmar比率
        
        年化收益除以最大回撤，衡量回撤风险。
        
        Args:
            returns: 收益率序列
            prices: 价格序列（可选，用于计算回撤）
            
        Returns:
            Calmar比率
        """
        ann_return = self.annualized_return(returns)
        
        if prices is None:
            # 从收益率重建价格序列
            prices = (1 + returns).cumprod()
        
        max_dd = self.maximum_drawdown(prices)
        
        if max_dd == 0:
            return np.inf if ann_return > 0 else 0.0
        
        return ann_return / abs(max_dd)
    
    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算信息比率
        
        超额收益除以跟踪误差，衡量主动管理能力。
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            信息比率
        """
        # 对齐数据
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) == 0:
            return 0.0
        
        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0.0
        
        return (excess_returns.mean() * self.trading_days) / tracking_error
    
    def treynor_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算Treynor比率
        
        衡量每单位系统性风险（Beta）的超额收益。
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            Treynor比率
        """
        beta = self.calculate_beta(returns, benchmark_returns)
        
        if beta == 0:
            return 0.0
        
        ann_return = self.annualized_return(returns)
        return (ann_return - self.risk_free_rate) / beta
    
    def omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        计算Omega比率
        
        收益超过阈值的概率加权和与低于阈值的概率加权和之比。
        
        Args:
            returns: 收益率序列
            threshold: 阈值
            
        Returns:
            Omega比率
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf
        
        return gains.sum() / losses.sum()
    
    # ==================== 回撤分析 ====================
    
    def maximum_drawdown(self, prices: pd.Series) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            最大回撤（负数）
        """
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    def drawdown_series(self, prices: pd.Series) -> pd.Series:
        """
        计算回撤序列
        
        Args:
            prices: 价格序列
            
        Returns:
            回撤序列
        """
        cummax = prices.cummax()
        return (prices - cummax) / cummax
    
    def analyze_drawdowns(
        self,
        prices: pd.Series,
        top_n: int = 5
    ) -> List[DrawdownInfo]:
        """
        分析回撤详情
        
        Args:
            prices: 价格序列
            top_n: 返回最大的N个回撤
            
        Returns:
            回撤信息列表
        """
        drawdown = self.drawdown_series(prices)
        
        # 找到所有回撤区间
        drawdowns = []
        in_drawdown = False
        start_idx = None
        peak_value = None
        
        for i, (date, dd) in enumerate(drawdown.items()):
            if not in_drawdown and dd < 0:
                # 开始新的回撤
                in_drawdown = True
                start_idx = i - 1 if i > 0 else i
                peak_value = prices.iloc[start_idx]
            elif in_drawdown and dd == 0:
                # 回撤结束
                in_drawdown = False
                trough_idx = drawdown.iloc[start_idx:i].idxmin()
                trough_loc = drawdown.index.get_loc(trough_idx)
                
                drawdowns.append(DrawdownInfo(
                    start_date=prices.index[start_idx],
                    trough_date=trough_idx,
                    end_date=date,
                    drawdown=drawdown.loc[trough_idx],
                    duration=(trough_loc - start_idx),
                    recovery_duration=(i - trough_loc)
                ))
        
        # 处理未结束的回撤
        if in_drawdown:
            trough_idx = drawdown.iloc[start_idx:].idxmin()
            trough_loc = drawdown.index.get_loc(trough_idx)
            
            drawdowns.append(DrawdownInfo(
                start_date=prices.index[start_idx],
                trough_date=trough_idx,
                end_date=None,
                drawdown=drawdown.loc[trough_idx],
                duration=(len(prices) - 1 - start_idx),
                recovery_duration=None
            ))
        
        # 按回撤幅度排序，返回最大的N个
        drawdowns.sort(key=lambda x: x.drawdown)
        return drawdowns[:top_n]
    
    def average_drawdown(self, prices: pd.Series) -> float:
        """计算平均回撤"""
        drawdown = self.drawdown_series(prices)
        return drawdown[drawdown < 0].mean()
    
    def drawdown_duration(self, prices: pd.Series) -> Dict[str, float]:
        """
        计算回撤持续时间统计
        
        Returns:
            包含最大、平均、当前回撤持续时间的字典
        """
        drawdowns = self.analyze_drawdowns(prices, top_n=100)
        
        if not drawdowns:
            return {'max_duration': 0, 'avg_duration': 0, 'current_duration': 0}
        
        durations = [d.duration for d in drawdowns]
        
        return {
            'max_duration': max(durations),
            'avg_duration': np.mean(durations),
            'current_duration': drawdowns[-1].duration if drawdowns[-1].end_date is None else 0
        }
    
    # ==================== VaR和CVaR ====================
    
    def value_at_risk(
        self,
        returns: pd.Series,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        method: VaRMethod = VaRMethod.HISTORICAL,
        holding_period: int = 1
    ) -> float:
        """
        计算风险价值 (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平（如0.95表示95%）
            method: 计算方法
            holding_period: 持有期（天数）
            
        Returns:
            VaR值（负数，表示损失）
        """
        if method == VaRMethod.HISTORICAL:
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == VaRMethod.PARAMETRIC:
            # 假设正态分布
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(1 - confidence_level, mu, sigma)
        
        elif method == VaRMethod.MONTE_CARLO:
            # 蒙特卡洛模拟
            mu = returns.mean()
            sigma = returns.std()
            simulated = np.random.normal(mu, sigma, 10000)
            var = np.percentile(simulated, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # 调整持有期
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return var
    
    def conditional_var(
        self,
        returns: pd.Series,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> float:
        """
        计算条件风险价值 (Conditional VaR / Expected Shortfall)
        
        CVaR是超过VaR阈值的平均损失，是一个更保守的风险度量。
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            method: 计算方法
            
        Returns:
            CVaR值（负数，表示损失）
        """
        var = self.value_at_risk(returns, confidence_level, method)
        
        # CVaR是低于VaR的所有收益的平均值
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    def var_cvar_summary(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.90, 0.95, 0.99]
    ) -> pd.DataFrame:
        """
        生成VaR和CVaR汇总表
        
        Args:
            returns: 收益率序列
            confidence_levels: 置信水平列表
            
        Returns:
            包含不同置信水平下VaR和CVaR的DataFrame
        """
        results = []
        
        for cl in confidence_levels:
            for method in [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC]:
                var = self.value_at_risk(returns, cl, method)
                cvar = self.conditional_var(returns, cl, method)
                
                results.append({
                    'Confidence Level': f"{cl*100:.0f}%",
                    'Method': method.value,
                    'VaR': f"{var*100:.2f}%",
                    'CVaR': f"{cvar*100:.2f}%"
                })
        
        return pd.DataFrame(results)
    
    # ==================== 相对风险度量 ====================
    
    def calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算Beta系数
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            Beta系数
        """
        # 对齐数据
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 1.0
        
        cov = aligned.cov().iloc[0, 1]
        var = aligned.iloc[:, 1].var()
        
        if var == 0:
            return 1.0
        
        return cov / var
    
    def calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算Jensen's Alpha（年化）
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            年化Alpha
        """
        beta = self.calculate_beta(returns, benchmark_returns)
        
        strategy_return = self.annualized_return(returns)
        benchmark_return = self.annualized_return(benchmark_returns)
        
        # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
        expected_return = self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
        
        return strategy_return - expected_return
    
    def tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算跟踪误差（年化）
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            年化跟踪误差
        """
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) == 0:
            return 0.0
        
        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        return excess_returns.std() * np.sqrt(self.trading_days)
    
    def capture_ratios(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        计算上行/下行捕获比率
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            包含up_capture和down_capture的字典
        """
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) == 0:
            return {'up_capture': 0.0, 'down_capture': 0.0}
        
        strategy = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        
        # 上行捕获：基准上涨时的表现
        up_mask = benchmark > 0
        if up_mask.sum() > 0:
            up_capture = (1 + strategy[up_mask]).prod() / (1 + benchmark[up_mask]).prod()
            up_capture = up_capture ** (1 / up_mask.sum() * self.trading_days) - 1
        else:
            up_capture = 0.0
        
        # 下行捕获：基准下跌时的表现
        down_mask = benchmark < 0
        if down_mask.sum() > 0:
            down_capture = (1 + strategy[down_mask]).prod() / (1 + benchmark[down_mask]).prod()
            down_capture = down_capture ** (1 / down_mask.sum() * self.trading_days) - 1
        else:
            down_capture = 0.0
        
        return {
            'up_capture': up_capture * 100,  # 转换为百分比
            'down_capture': down_capture * 100
        }
    
    # ==================== 综合风险报告 ====================
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        计算所有风险度量指标
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（可选）
            prices: 价格序列（可选）
            
        Returns:
            包含所有风险指标的字典
        """
        if prices is None:
            prices = (1 + returns).cumprod()
        
        metrics = {
            # 收益指标
            'annualized_return': self.annualized_return(returns),
            'total_return': (1 + returns).prod() - 1,
            
            # 波动率指标
            'annualized_volatility': self.annualized_volatility(returns),
            'downside_volatility': self.downside_volatility(returns),
            
            # 风险调整后收益
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns, prices),
            'omega_ratio': self.omega_ratio(returns),
            
            # 回撤指标
            'max_drawdown': self.maximum_drawdown(prices),
            'avg_drawdown': self.average_drawdown(prices),
            
            # VaR/CVaR
            'var_95': self.value_at_risk(returns, 0.95),
            'cvar_95': self.conditional_var(returns, 0.95),
            'var_99': self.value_at_risk(returns, 0.99),
            'cvar_99': self.conditional_var(returns, 0.99),
            
            # 其他统计量
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_days_ratio': (returns > 0).mean(),
        }
        
        # 如果有基准数据，计算相对指标
        if benchmark_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(returns, benchmark_returns),
                'alpha': self.calculate_alpha(returns, benchmark_returns),
                'information_ratio': self.information_ratio(returns, benchmark_returns),
                'treynor_ratio': self.treynor_ratio(returns, benchmark_returns),
                'tracking_error': self.tracking_error(returns, benchmark_returns),
            })
            
            capture = self.capture_ratios(returns, benchmark_returns)
            metrics['up_capture'] = capture['up_capture']
            metrics['down_capture'] = capture['down_capture']
        
        return metrics
    
    def generate_risk_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        strategy_name: str = "Strategy"
    ) -> str:
        """
        生成风险分析报告（Markdown格式）
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（可选）
            prices: 价格序列（可选）
            strategy_name: 策略名称
            
        Returns:
            Markdown格式的风险报告
        """
        metrics = self.calculate_all_metrics(returns, benchmark_returns, prices)
        
        report = f"""# {strategy_name} 风险分析报告

## 1. 收益概览

| 指标 | 数值 |
|:---|---:|
| 年化收益率 | {metrics['annualized_return']*100:.2f}% |
| 累计收益率 | {metrics['total_return']*100:.2f}% |
| 年化波动率 | {metrics['annualized_volatility']*100:.2f}% |
| 下行波动率 | {metrics['downside_volatility']*100:.2f}% |

## 2. 风险调整后收益

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| Sharpe Ratio | {metrics['sharpe_ratio']:.3f} | 每单位总风险的超额收益 |
| Sortino Ratio | {metrics['sortino_ratio']:.3f} | 每单位下行风险的超额收益 |
| Calmar Ratio | {metrics['calmar_ratio']:.3f} | 年化收益/最大回撤 |
| Omega Ratio | {metrics['omega_ratio']:.3f} | 收益/损失比 |

## 3. 回撤分析

| 指标 | 数值 |
|:---|---:|
| 最大回撤 | {metrics['max_drawdown']*100:.2f}% |
| 平均回撤 | {metrics['avg_drawdown']*100:.2f}% |

## 4. 尾部风险 (VaR/CVaR)

| 置信水平 | VaR | CVaR |
|:---|---:|---:|
| 95% | {metrics['var_95']*100:.2f}% | {metrics['cvar_95']*100:.2f}% |
| 99% | {metrics['var_99']*100:.2f}% | {metrics['cvar_99']*100:.2f}% |

## 5. 收益分布特征

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| 偏度 | {metrics['skewness']:.3f} | {'右偏（正向）' if metrics['skewness'] > 0 else '左偏（负向）'} |
| 峰度 | {metrics['kurtosis']:.3f} | {'厚尾' if metrics['kurtosis'] > 0 else '薄尾'} |
| 正收益天数占比 | {metrics['positive_days_ratio']*100:.1f}% | |
"""
        
        if benchmark_returns is not None:
            report += f"""
## 6. 相对基准表现

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| Beta | {metrics['beta']:.3f} | 系统性风险敞口 |
| Alpha (年化) | {metrics['alpha']*100:.2f}% | 超额收益能力 |
| Information Ratio | {metrics['information_ratio']:.3f} | 主动管理能力 |
| Treynor Ratio | {metrics['treynor_ratio']:.3f} | 每单位Beta的超额收益 |
| 跟踪误差 | {metrics['tracking_error']*100:.2f}% | 与基准的偏离程度 |
| 上行捕获 | {metrics['up_capture']:.1f}% | 牛市表现 |
| 下行捕获 | {metrics['down_capture']:.1f}% | 熊市表现 |
"""
        
        return report


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("风险度量指标计算模块测试")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_days = 252 * 2  # 2年数据
    
    # 模拟策略收益率
    strategy_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),  # 日均0.05%，波动率1.5%
        index=pd.date_range('2024-01-01', periods=n_days, freq='B')
    )
    
    # 模拟基准收益率
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.012, n_days),  # 日均0.03%，波动率1.2%
        index=pd.date_range('2024-01-01', periods=n_days, freq='B')
    )
    
    # 初始化计算器
    rm = RiskMetrics(risk_free_rate=0.02)
    
    # 计算所有指标
    metrics = rm.calculate_all_metrics(strategy_returns, benchmark_returns)
    
    print("\n基础指标:")
    print(f"  年化收益率: {metrics['annualized_return']*100:.2f}%")
    print(f"  年化波动率: {metrics['annualized_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    print("\n尾部风险:")
    print(f"  VaR (95%): {metrics['var_95']*100:.2f}%")
    print(f"  CVaR (95%): {metrics['cvar_95']*100:.2f}%")
    
    print("\n相对指标:")
    print(f"  Beta: {metrics['beta']:.3f}")
    print(f"  Alpha: {metrics['alpha']*100:.2f}%")
    print(f"  Information Ratio: {metrics['information_ratio']:.3f}")
    
    # 生成完整报告
    print("\n" + "=" * 60)
    print("生成风险分析报告...")
    report = rm.generate_risk_report(strategy_returns, benchmark_returns, strategy_name="测试策略")
    print(report)
