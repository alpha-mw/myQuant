"""
因子风险分解与归因分析模块 (Factor Risk Decomposition)

基于Barra风险模型思想，提供：
1. 风险因子暴露计算
2. 系统性风险与特异性风险分解
3. 因子风险贡献分析
4. 边际风险贡献 (Marginal Contribution to Risk)
5. 成分风险贡献 (Component Risk Contribution)

V2.8 - 风险管理模块
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class FactorExposure:
    """因子暴露信息"""
    factor_name: str
    exposure: float          # 因子暴露（Beta）
    t_stat: float           # t统计量
    p_value: float          # p值
    is_significant: bool    # 是否显著（p<0.05）


@dataclass
class RiskDecomposition:
    """风险分解结果"""
    total_risk: float           # 总风险（年化波动率）
    systematic_risk: float      # 系统性风险
    idiosyncratic_risk: float   # 特异性风险
    r_squared: float            # R²（系统性风险占比）
    factor_contributions: Dict[str, float]  # 各因子风险贡献


class FactorRiskAnalyzer:
    """
    因子风险分析器
    
    基于多因子模型进行风险分解和归因分析。
    """
    
    # 常用风格因子
    STYLE_FACTORS = [
        'market',      # 市场因子
        'size',        # 市值因子
        'value',       # 价值因子
        'momentum',    # 动量因子
        'volatility',  # 波动率因子
        'quality',     # 质量因子
        'liquidity',   # 流动性因子
    ]
    
    def __init__(self, trading_days: int = 252):
        """
        初始化因子风险分析器
        
        Args:
            trading_days: 年化交易日数
        """
        self.trading_days = trading_days
    
    # ==================== 因子暴露计算 ====================
    
    def calculate_factor_exposures(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> List[FactorExposure]:
        """
        计算策略对各因子的暴露
        
        使用多元回归计算因子暴露（Beta）。
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率DataFrame，每列为一个因子
            
        Returns:
            因子暴露列表
        """
        # 对齐数据
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        if len(aligned) < 30:
            warnings.warn("数据量不足30天，因子暴露估计可能不可靠")
        
        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values
        
        # 添加截距项
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS回归
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            return []
        
        # 计算统计量
        n = len(y)
        k = X.shape[1]
        
        # 残差标准误
        y_pred = X_with_const @ beta
        sse = np.sum((y - y_pred) ** 2)
        mse = sse / (n - k - 1) if n > k + 1 else 0
        
        # 参数标准误
        try:
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se_beta = np.sqrt(var_beta)
        except np.linalg.LinAlgError:
            se_beta = np.zeros(k + 1)
        
        # 构建结果
        exposures = []
        factor_names = factor_returns.columns.tolist()
        
        for i, factor_name in enumerate(factor_names):
            b = beta[i + 1]  # 跳过截距项
            se = se_beta[i + 1] if i + 1 < len(se_beta) else 0
            
            if se > 0:
                t_stat = b / se
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
            else:
                t_stat = 0
                p_value = 1
            
            exposures.append(FactorExposure(
                factor_name=factor_name,
                exposure=b,
                t_stat=t_stat,
                p_value=p_value,
                is_significant=(p_value < 0.05)
            ))
        
        return exposures
    
    def calculate_single_factor_beta(
        self,
        returns: pd.Series,
        factor_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        计算单因子Beta
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率序列
            
        Returns:
            (beta, t_stat, r_squared)
        """
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0, 0.0, 0.0
        
        y = aligned.iloc[:, 0]
        x = aligned.iloc[:, 1]
        
        # 简单线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        t_stat = slope / std_err if std_err > 0 else 0
        
        return slope, t_stat, r_value ** 2
    
    # ==================== 风险分解 ====================
    
    def decompose_risk(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> RiskDecomposition:
        """
        将总风险分解为系统性风险和特异性风险
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率DataFrame
            
        Returns:
            风险分解结果
        """
        # 对齐数据
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        
        # 计算总风险（年化波动率）
        total_risk = y.std() * np.sqrt(self.trading_days)
        
        # 多元回归
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y.values, rcond=None)
        
        # 计算拟合值和残差
        y_pred = X_with_const @ beta
        residuals = y.values - y_pred
        
        # 系统性风险（拟合值的波动率）
        systematic_risk = np.std(y_pred) * np.sqrt(self.trading_days)
        
        # 特异性风险（残差的波动率）
        idiosyncratic_risk = np.std(residuals) * np.sqrt(self.trading_days)
        
        # R² = 1 - SSE/SST
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 计算各因子的风险贡献
        factor_contributions = self._calculate_factor_risk_contributions(
            beta[1:],  # 排除截距
            X.values,
            total_risk
        )
        factor_contributions = dict(zip(factor_returns.columns, factor_contributions))
        
        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            r_squared=r_squared,
            factor_contributions=factor_contributions
        )
    
    def _calculate_factor_risk_contributions(
        self,
        betas: np.ndarray,
        factor_data: np.ndarray,
        total_risk: float
    ) -> np.ndarray:
        """
        计算各因子对总风险的贡献
        
        使用边际风险贡献方法。
        """
        # 因子协方差矩阵
        factor_cov = np.cov(factor_data.T) * self.trading_days
        
        # 组合方差 = β' Σ β
        portfolio_var = betas @ factor_cov @ betas
        
        if portfolio_var <= 0:
            return np.zeros(len(betas))
        
        # 边际风险贡献 = Σ β / σ
        marginal_risk = factor_cov @ betas / np.sqrt(portfolio_var)
        
        # 成分风险贡献 = β * 边际风险贡献
        component_risk = betas * marginal_risk
        
        # 归一化为占总风险的比例
        total_component = np.sum(np.abs(component_risk))
        if total_component > 0:
            component_risk = component_risk / total_component * total_risk
        
        return component_risk
    
    # ==================== 风险归因 ====================
    
    def risk_attribution(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Dict[str, any]:
        """
        投资组合风险归因分析
        
        Args:
            portfolio_weights: 资产权重
            asset_returns: 各资产收益率DataFrame
            factor_returns: 因子收益率DataFrame
            
        Returns:
            风险归因结果
        """
        # 计算组合收益率
        aligned_assets = asset_returns[portfolio_weights.index]
        portfolio_returns = (aligned_assets * portfolio_weights).sum(axis=1)
        
        # 整体风险分解
        decomposition = self.decompose_risk(portfolio_returns, factor_returns)
        
        # 计算各资产的风险贡献
        asset_risk_contributions = self._calculate_asset_risk_contributions(
            portfolio_weights,
            asset_returns
        )
        
        # 计算各资产的因子暴露
        asset_factor_exposures = {}
        for asset in portfolio_weights.index:
            if asset in asset_returns.columns:
                exposures = self.calculate_factor_exposures(
                    asset_returns[asset],
                    factor_returns
                )
                asset_factor_exposures[asset] = {
                    e.factor_name: e.exposure for e in exposures
                }
        
        return {
            'portfolio_decomposition': decomposition,
            'asset_risk_contributions': asset_risk_contributions,
            'asset_factor_exposures': asset_factor_exposures
        }
    
    def _calculate_asset_risk_contributions(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        计算各资产对组合风险的贡献
        
        使用成分VaR方法。
        """
        # 对齐数据
        aligned = returns[weights.index].dropna()
        
        # 协方差矩阵
        cov_matrix = aligned.cov() * self.trading_days
        
        # 组合方差
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std == 0:
            return pd.Series(0, index=weights.index)
        
        # 边际风险贡献
        marginal_risk = cov_matrix @ weights / portfolio_std
        
        # 成分风险贡献
        component_risk = weights * marginal_risk
        
        return component_risk
    
    # ==================== 情景分析 ====================
    
    def scenario_analysis(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        scenarios: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        情景分析：模拟不同市场情景下的策略表现
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率DataFrame
            scenarios: 情景定义，格式为 {情景名: {因子名: 因子收益率}}
            
        Returns:
            各情景下的预期收益
        """
        # 计算因子暴露
        exposures = self.calculate_factor_exposures(returns, factor_returns)
        exposure_dict = {e.factor_name: e.exposure for e in exposures}
        
        # 计算截距（Alpha）
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        beta, _, _, _ = np.linalg.lstsq(X_with_const, y.values, rcond=None)
        alpha = beta[0]
        
        # 计算各情景下的预期收益
        results = []
        for scenario_name, factor_shocks in scenarios.items():
            expected_return = alpha
            for factor, shock in factor_shocks.items():
                if factor in exposure_dict:
                    expected_return += exposure_dict[factor] * shock
            
            results.append({
                'Scenario': scenario_name,
                'Expected Return': expected_return,
                'Expected Return (%)': f"{expected_return * 100:.2f}%"
            })
        
        return pd.DataFrame(results)
    
    def stress_test(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        stress_levels: List[float] = [-3, -2, -1, 1, 2, 3]
    ) -> pd.DataFrame:
        """
        压力测试：测试极端市场条件下的风险
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率DataFrame
            stress_levels: 压力水平（标准差倍数）
            
        Returns:
            压力测试结果
        """
        # 计算因子暴露
        exposures = self.calculate_factor_exposures(returns, factor_returns)
        
        # 因子统计量
        factor_stats = factor_returns.describe()
        
        results = []
        for level in stress_levels:
            scenario = {}
            for exp in exposures:
                factor = exp.factor_name
                if factor in factor_stats.columns:
                    mean = factor_stats.loc['mean', factor]
                    std = factor_stats.loc['std', factor]
                    scenario[factor] = mean + level * std
            
            # 计算预期收益
            expected_return = sum(
                exp.exposure * scenario.get(exp.factor_name, 0)
                for exp in exposures
            )
            
            results.append({
                'Stress Level': f"{level}σ",
                'Expected Return': expected_return,
                'Expected Return (%)': f"{expected_return * 100:.2f}%"
            })
        
        return pd.DataFrame(results)
    
    # ==================== 报告生成 ====================
    
    def generate_factor_risk_report(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        strategy_name: str = "Strategy"
    ) -> str:
        """
        生成因子风险分析报告
        
        Args:
            returns: 策略收益率序列
            factor_returns: 因子收益率DataFrame
            strategy_name: 策略名称
            
        Returns:
            Markdown格式的报告
        """
        # 计算因子暴露
        exposures = self.calculate_factor_exposures(returns, factor_returns)
        
        # 风险分解
        decomposition = self.decompose_risk(returns, factor_returns)
        
        # 生成报告
        report = f"""# {strategy_name} 因子风险分析报告

## 1. 风险分解

| 风险类型 | 数值 | 占比 |
|:---|---:|---:|
| 总风险 | {decomposition.total_risk*100:.2f}% | 100% |
| 系统性风险 | {decomposition.systematic_risk*100:.2f}% | {decomposition.r_squared*100:.1f}% |
| 特异性风险 | {decomposition.idiosyncratic_risk*100:.2f}% | {(1-decomposition.r_squared)*100:.1f}% |

**R² = {decomposition.r_squared:.3f}**：策略收益的{decomposition.r_squared*100:.1f}%可以被因子模型解释。

## 2. 因子暴露

| 因子 | 暴露 (Beta) | t统计量 | p值 | 显著性 |
|:---|---:|---:|---:|:---|
"""
        
        for exp in exposures:
            sig = "✓" if exp.is_significant else ""
            report += f"| {exp.factor_name} | {exp.exposure:.4f} | {exp.t_stat:.2f} | {exp.p_value:.4f} | {sig} |\n"
        
        report += """
## 3. 因子风险贡献

| 因子 | 风险贡献 |
|:---|---:|
"""
        
        for factor, contribution in decomposition.factor_contributions.items():
            report += f"| {factor} | {contribution*100:.2f}% |\n"
        
        # 添加压力测试
        stress_results = self.stress_test(returns, factor_returns)
        
        report += """
## 4. 压力测试

| 压力水平 | 预期收益 |
|:---|---:|
"""
        
        for _, row in stress_results.iterrows():
            report += f"| {row['Stress Level']} | {row['Expected Return (%)']}\n"
        
        return report


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("因子风险分解模块测试")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_days = 252 * 2
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # 模拟因子收益率
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, n_days),
        'size': np.random.normal(0.0001, 0.005, n_days),
        'value': np.random.normal(0.0001, 0.006, n_days),
        'momentum': np.random.normal(0.0002, 0.008, n_days),
    }, index=dates)
    
    # 模拟策略收益率（与因子相关）
    strategy_returns = (
        0.0001 +  # Alpha
        1.2 * factor_returns['market'] +
        0.3 * factor_returns['size'] +
        -0.2 * factor_returns['value'] +
        0.5 * factor_returns['momentum'] +
        np.random.normal(0, 0.005, n_days)  # 特异性收益
    )
    strategy_returns = pd.Series(strategy_returns, index=dates)
    
    # 初始化分析器
    analyzer = FactorRiskAnalyzer()
    
    # 计算因子暴露
    print("\n1. 因子暴露:")
    exposures = analyzer.calculate_factor_exposures(strategy_returns, factor_returns)
    for exp in exposures:
        sig = "***" if exp.is_significant else ""
        print(f"  {exp.factor_name}: Beta={exp.exposure:.4f}, t={exp.t_stat:.2f}, p={exp.p_value:.4f} {sig}")
    
    # 风险分解
    print("\n2. 风险分解:")
    decomposition = analyzer.decompose_risk(strategy_returns, factor_returns)
    print(f"  总风险: {decomposition.total_risk*100:.2f}%")
    print(f"  系统性风险: {decomposition.systematic_risk*100:.2f}%")
    print(f"  特异性风险: {decomposition.idiosyncratic_risk*100:.2f}%")
    print(f"  R²: {decomposition.r_squared:.4f}")
    
    print("\n3. 因子风险贡献:")
    for factor, contribution in decomposition.factor_contributions.items():
        print(f"  {factor}: {contribution*100:.2f}%")
    
    # 情景分析
    print("\n4. 情景分析:")
    scenarios = {
        '牛市': {'market': 0.02, 'momentum': 0.01},
        '熊市': {'market': -0.03, 'momentum': -0.02},
        '价值回归': {'value': 0.02, 'momentum': -0.01},
    }
    scenario_results = analyzer.scenario_analysis(strategy_returns, factor_returns, scenarios)
    print(scenario_results.to_string(index=False))
    
    # 压力测试
    print("\n5. 压力测试:")
    stress_results = analyzer.stress_test(strategy_returns, factor_returns)
    print(stress_results.to_string(index=False))
    
    # 生成完整报告
    print("\n" + "=" * 60)
    print("生成因子风险分析报告...")
    report = analyzer.generate_factor_risk_report(strategy_returns, factor_returns, "测试策略")
    print(report)
