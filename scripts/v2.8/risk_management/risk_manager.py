"""
综合风险管理器 (Risk Manager)

整合所有风险分析功能，提供：
1. 一站式风险评估
2. 可视化风险报告
3. 风险预警系统
4. 策略风险对比

V2.8 - 风险管理模块
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# 导入子模块
try:
    from .risk_metrics import RiskMetrics, VaRMethod
    from .factor_risk import FactorRiskAnalyzer, RiskDecomposition
except ImportError:
    from risk_metrics import RiskMetrics, VaRMethod
    from factor_risk import FactorRiskAnalyzer, RiskDecomposition


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    EXTREME = "极高风险"


@dataclass
class RiskAlert:
    """风险预警"""
    alert_type: str
    level: RiskLevel
    message: str
    value: float
    threshold: float


class RiskManager:
    """
    综合风险管理器
    
    整合RiskMetrics和FactorRiskAnalyzer，提供完整的风险管理解决方案。
    """
    
    # 风险阈值配置
    RISK_THRESHOLDS = {
        'max_drawdown': {
            'low': -0.10,
            'medium': -0.20,
            'high': -0.30,
        },
        'volatility': {
            'low': 0.15,
            'medium': 0.25,
            'high': 0.40,
        },
        'var_95': {
            'low': -0.02,
            'medium': -0.03,
            'high': -0.05,
        },
        'sharpe_ratio': {
            'good': 1.0,
            'acceptable': 0.5,
            'poor': 0.0,
        },
    }
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        trading_days: int = 252
    ):
        """
        初始化风险管理器
        
        Args:
            risk_free_rate: 年化无风险利率
            trading_days: 年化交易日数
        """
        self.risk_metrics = RiskMetrics(risk_free_rate, trading_days)
        self.factor_analyzer = FactorRiskAnalyzer(trading_days)
        self.trading_days = trading_days
    
    # ==================== 综合风险评估 ====================
    
    def evaluate_strategy(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        prices: Optional[pd.Series] = None,
        strategy_name: str = "Strategy"
    ) -> Dict:
        """
        综合评估策略风险
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（可选）
            factor_returns: 因子收益率DataFrame（可选）
            prices: 价格序列（可选）
            strategy_name: 策略名称
            
        Returns:
            综合风险评估结果
        """
        if prices is None:
            prices = (1 + returns).cumprod()
        
        # 基础风险指标
        basic_metrics = self.risk_metrics.calculate_all_metrics(
            returns, benchmark_returns, prices
        )
        
        # 因子风险分解（如果提供了因子数据）
        factor_decomposition = None
        factor_exposures = None
        if factor_returns is not None:
            factor_decomposition = self.factor_analyzer.decompose_risk(
                returns, factor_returns
            )
            factor_exposures = self.factor_analyzer.calculate_factor_exposures(
                returns, factor_returns
            )
        
        # 回撤分析
        drawdown_analysis = self.risk_metrics.analyze_drawdowns(prices, top_n=5)
        
        # 风险预警
        alerts = self._generate_risk_alerts(basic_metrics)
        
        # 风险等级评估
        risk_level = self._assess_risk_level(basic_metrics)
        
        return {
            'strategy_name': strategy_name,
            'basic_metrics': basic_metrics,
            'factor_decomposition': factor_decomposition,
            'factor_exposures': factor_exposures,
            'drawdown_analysis': drawdown_analysis,
            'alerts': alerts,
            'risk_level': risk_level,
        }
    
    def _generate_risk_alerts(self, metrics: Dict) -> List[RiskAlert]:
        """生成风险预警"""
        alerts = []
        
        # 最大回撤预警
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < self.RISK_THRESHOLDS['max_drawdown']['high']:
            alerts.append(RiskAlert(
                alert_type='最大回撤',
                level=RiskLevel.EXTREME,
                message=f'最大回撤达到{max_dd*100:.1f}%，超过高风险阈值',
                value=max_dd,
                threshold=self.RISK_THRESHOLDS['max_drawdown']['high']
            ))
        elif max_dd < self.RISK_THRESHOLDS['max_drawdown']['medium']:
            alerts.append(RiskAlert(
                alert_type='最大回撤',
                level=RiskLevel.HIGH,
                message=f'最大回撤达到{max_dd*100:.1f}%，超过中等风险阈值',
                value=max_dd,
                threshold=self.RISK_THRESHOLDS['max_drawdown']['medium']
            ))
        
        # 波动率预警
        vol = metrics.get('annualized_volatility', 0)
        if vol > self.RISK_THRESHOLDS['volatility']['high']:
            alerts.append(RiskAlert(
                alert_type='波动率',
                level=RiskLevel.HIGH,
                message=f'年化波动率达到{vol*100:.1f}%，风险较高',
                value=vol,
                threshold=self.RISK_THRESHOLDS['volatility']['high']
            ))
        
        # VaR预警
        var_95 = metrics.get('var_95', 0)
        if var_95 < self.RISK_THRESHOLDS['var_95']['high']:
            alerts.append(RiskAlert(
                alert_type='VaR',
                level=RiskLevel.HIGH,
                message=f'95% VaR达到{var_95*100:.2f}%，尾部风险较大',
                value=var_95,
                threshold=self.RISK_THRESHOLDS['var_95']['high']
            ))
        
        # Sharpe比率预警
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < self.RISK_THRESHOLDS['sharpe_ratio']['poor']:
            alerts.append(RiskAlert(
                alert_type='Sharpe比率',
                level=RiskLevel.HIGH,
                message=f'Sharpe比率为{sharpe:.2f}，风险调整后收益为负',
                value=sharpe,
                threshold=self.RISK_THRESHOLDS['sharpe_ratio']['poor']
            ))
        elif sharpe < self.RISK_THRESHOLDS['sharpe_ratio']['acceptable']:
            alerts.append(RiskAlert(
                alert_type='Sharpe比率',
                level=RiskLevel.MEDIUM,
                message=f'Sharpe比率为{sharpe:.2f}，风险调整后收益偏低',
                value=sharpe,
                threshold=self.RISK_THRESHOLDS['sharpe_ratio']['acceptable']
            ))
        
        return alerts
    
    def _assess_risk_level(self, metrics: Dict) -> RiskLevel:
        """评估整体风险等级"""
        risk_scores = []
        
        # 根据各指标评分
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < -0.30:
            risk_scores.append(4)
        elif max_dd < -0.20:
            risk_scores.append(3)
        elif max_dd < -0.10:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        vol = metrics.get('annualized_volatility', 0)
        if vol > 0.40:
            risk_scores.append(4)
        elif vol > 0.25:
            risk_scores.append(3)
        elif vol > 0.15:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 0:
            risk_scores.append(4)
        elif sharpe < 0.5:
            risk_scores.append(3)
        elif sharpe < 1.0:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # 综合评分
        avg_score = np.mean(risk_scores)
        
        if avg_score >= 3.5:
            return RiskLevel.EXTREME
        elif avg_score >= 2.5:
            return RiskLevel.HIGH
        elif avg_score >= 1.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # ==================== 策略对比 ====================
    
    def compare_strategies(
        self,
        strategies: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        对比多个策略的风险指标
        
        Args:
            strategies: 策略名称到收益率序列的映射
            benchmark_returns: 基准收益率序列（可选）
            
        Returns:
            策略对比表
        """
        results = []
        
        for name, returns in strategies.items():
            metrics = self.risk_metrics.calculate_all_metrics(
                returns, benchmark_returns
            )
            
            results.append({
                '策略': name,
                '年化收益': f"{metrics['annualized_return']*100:.2f}%",
                '年化波动率': f"{metrics['annualized_volatility']*100:.2f}%",
                '最大回撤': f"{metrics['max_drawdown']*100:.2f}%",
                'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                'Sortino': f"{metrics['sortino_ratio']:.3f}",
                'Calmar': f"{metrics['calmar_ratio']:.3f}",
                'VaR(95%)': f"{metrics['var_95']*100:.2f}%",
            })
        
        return pd.DataFrame(results)
    
    # ==================== 风险报告生成 ====================
    
    def generate_comprehensive_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        prices: Optional[pd.Series] = None,
        strategy_name: str = "Strategy"
    ) -> str:
        """
        生成综合风险分析报告
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（可选）
            factor_returns: 因子收益率DataFrame（可选）
            prices: 价格序列（可选）
            strategy_name: 策略名称
            
        Returns:
            Markdown格式的综合报告
        """
        # 获取评估结果
        evaluation = self.evaluate_strategy(
            returns, benchmark_returns, factor_returns, prices, strategy_name
        )
        
        metrics = evaluation['basic_metrics']
        alerts = evaluation['alerts']
        risk_level = evaluation['risk_level']
        
        # 生成报告
        report = f"""# {strategy_name} 综合风险分析报告

## 风险等级: {risk_level.value}

---

## 1. 执行摘要

| 核心指标 | 数值 | 评价 |
|:---|---:|:---|
| 年化收益率 | {metrics['annualized_return']*100:.2f}% | {'优秀' if metrics['annualized_return'] > 0.15 else '良好' if metrics['annualized_return'] > 0.08 else '一般'} |
| 年化波动率 | {metrics['annualized_volatility']*100:.2f}% | {'低' if metrics['annualized_volatility'] < 0.15 else '中等' if metrics['annualized_volatility'] < 0.25 else '高'} |
| 最大回撤 | {metrics['max_drawdown']*100:.2f}% | {'可控' if metrics['max_drawdown'] > -0.15 else '较大' if metrics['max_drawdown'] > -0.25 else '严重'} |
| Sharpe比率 | {metrics['sharpe_ratio']:.3f} | {'优秀' if metrics['sharpe_ratio'] > 1.5 else '良好' if metrics['sharpe_ratio'] > 1.0 else '一般' if metrics['sharpe_ratio'] > 0.5 else '较差'} |

"""
        
        # 风险预警
        if alerts:
            report += "## 2. 风险预警\n\n"
            for alert in alerts:
                icon = "🔴" if alert.level in [RiskLevel.HIGH, RiskLevel.EXTREME] else "🟡"
                report += f"{icon} **{alert.alert_type}**: {alert.message}\n\n"
        else:
            report += "## 2. 风险预警\n\n✅ 当前无风险预警\n\n"
        
        # 详细指标
        report += f"""## 3. 详细风险指标

### 3.1 收益指标

| 指标 | 数值 |
|:---|---:|
| 累计收益率 | {metrics['total_return']*100:.2f}% |
| 年化收益率 | {metrics['annualized_return']*100:.2f}% |
| 正收益天数占比 | {metrics['positive_days_ratio']*100:.1f}% |

### 3.2 波动率指标

| 指标 | 数值 |
|:---|---:|
| 年化波动率 | {metrics['annualized_volatility']*100:.2f}% |
| 下行波动率 | {metrics['downside_volatility']*100:.2f}% |

### 3.3 风险调整后收益

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| Sharpe Ratio | {metrics['sharpe_ratio']:.3f} | 每单位总风险的超额收益 |
| Sortino Ratio | {metrics['sortino_ratio']:.3f} | 每单位下行风险的超额收益 |
| Calmar Ratio | {metrics['calmar_ratio']:.3f} | 年化收益/最大回撤 |
| Omega Ratio | {metrics['omega_ratio']:.3f} | 收益/损失比 |

### 3.4 回撤分析

| 指标 | 数值 |
|:---|---:|
| 最大回撤 | {metrics['max_drawdown']*100:.2f}% |
| 平均回撤 | {metrics['avg_drawdown']*100:.2f}% |

### 3.5 尾部风险 (VaR/CVaR)

| 置信水平 | VaR | CVaR |
|:---|---:|---:|
| 95% | {metrics['var_95']*100:.2f}% | {metrics['cvar_95']*100:.2f}% |
| 99% | {metrics['var_99']*100:.2f}% | {metrics['cvar_99']*100:.2f}% |

### 3.6 收益分布特征

| 指标 | 数值 | 解读 |
|:---|---:|:---|
| 偏度 | {metrics['skewness']:.3f} | {'收益分布右偏，正向收益更多' if metrics['skewness'] > 0.5 else '收益分布左偏，负向收益更多' if metrics['skewness'] < -0.5 else '收益分布较为对称'} |
| 峰度 | {metrics['kurtosis']:.3f} | {'厚尾分布，极端收益更频繁' if metrics['kurtosis'] > 1 else '薄尾分布，极端收益较少' if metrics['kurtosis'] < -1 else '接近正态分布'} |

"""
        
        # 相对基准表现
        if benchmark_returns is not None and 'beta' in metrics:
            report += f"""### 3.7 相对基准表现

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| Beta | {metrics['beta']:.3f} | {'进攻型' if metrics['beta'] > 1.1 else '防守型' if metrics['beta'] < 0.9 else '中性'} |
| Alpha (年化) | {metrics['alpha']*100:.2f}% | {'正Alpha，有超额收益能力' if metrics['alpha'] > 0 else '负Alpha，跑输基准'} |
| Information Ratio | {metrics['information_ratio']:.3f} | 主动管理能力 |
| Treynor Ratio | {metrics['treynor_ratio']:.3f} | 每单位Beta的超额收益 |
| 跟踪误差 | {metrics['tracking_error']*100:.2f}% | 与基准的偏离程度 |

"""
        
        # 因子风险分解
        if evaluation['factor_decomposition'] is not None:
            decomp = evaluation['factor_decomposition']
            report += f"""## 4. 因子风险分解

| 风险类型 | 数值 | 占比 |
|:---|---:|---:|
| 总风险 | {decomp.total_risk*100:.2f}% | 100% |
| 系统性风险 | {decomp.systematic_risk*100:.2f}% | {decomp.r_squared*100:.1f}% |
| 特异性风险 | {decomp.idiosyncratic_risk*100:.2f}% | {(1-decomp.r_squared)*100:.1f}% |

**R² = {decomp.r_squared:.3f}**：策略收益的{decomp.r_squared*100:.1f}%可以被因子模型解释。

### 因子暴露

| 因子 | 暴露 (Beta) | 显著性 |
|:---|---:|:---|
"""
            if evaluation['factor_exposures']:
                for exp in evaluation['factor_exposures']:
                    sig = "✓" if exp.is_significant else ""
                    report += f"| {exp.factor_name} | {exp.exposure:.4f} | {sig} |\n"
        
        # 风险建议
        report += """
## 5. 风险管理建议

"""
        
        if metrics['max_drawdown'] < -0.20:
            report += "- **回撤控制**：建议设置止损线或动态调整仓位，控制最大回撤在20%以内。\n"
        
        if metrics['annualized_volatility'] > 0.25:
            report += "- **波动率管理**：建议通过分散投资或对冲策略降低组合波动率。\n"
        
        if metrics['sharpe_ratio'] < 0.5:
            report += "- **收益优化**：当前风险调整后收益偏低，建议优化策略或调整风险敞口。\n"
        
        if metrics['skewness'] < -0.5:
            report += "- **尾部风险**：收益分布左偏，建议关注下行风险保护。\n"
        
        if not alerts and metrics['sharpe_ratio'] > 1.0:
            report += "✅ 当前策略风险状况良好，建议继续监控并保持。\n"
        
        return report


# ==================== 模块初始化 ====================

__all__ = [
    'RiskManager',
    'RiskLevel',
    'RiskAlert',
]


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("综合风险管理器测试")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_days = 252 * 2
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # 模拟策略收益率
    strategy_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),
        index=dates
    )
    
    # 模拟基准收益率
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.012, n_days),
        index=dates
    )
    
    # 模拟因子收益率
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, n_days),
        'size': np.random.normal(0.0001, 0.005, n_days),
        'value': np.random.normal(0.0001, 0.006, n_days),
        'momentum': np.random.normal(0.0002, 0.008, n_days),
    }, index=dates)
    
    # 初始化风险管理器
    rm = RiskManager()
    
    # 综合评估
    print("\n1. 综合风险评估:")
    evaluation = rm.evaluate_strategy(
        strategy_returns,
        benchmark_returns,
        factor_returns,
        strategy_name="测试策略"
    )
    print(f"  风险等级: {evaluation['risk_level'].value}")
    print(f"  预警数量: {len(evaluation['alerts'])}")
    
    # 策略对比
    print("\n2. 策略对比:")
    strategies = {
        '策略A': strategy_returns,
        '策略B': pd.Series(np.random.normal(0.0003, 0.012, n_days), index=dates),
        '策略C': pd.Series(np.random.normal(0.0008, 0.020, n_days), index=dates),
    }
    comparison = rm.compare_strategies(strategies, benchmark_returns)
    print(comparison.to_string(index=False))
    
    # 生成综合报告
    print("\n" + "=" * 60)
    print("生成综合风险分析报告...")
    report = rm.generate_comprehensive_report(
        strategy_returns,
        benchmark_returns,
        factor_returns,
        strategy_name="测试策略"
    )
    print(report)
