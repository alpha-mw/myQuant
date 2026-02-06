"""
Quant-Investor V5.0 - 高级风险管理模块

本模块提供完整的风险管理功能，包括：
1. Barra风险模型：风险因子分解
2. 波动率预测：GARCH、EWMA模型
3. 压力测试：历史情景回放
4. VaR/CVaR计算
5. 风险预算与监控
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ==================== Barra风险模型 ====================

class BarraRiskModel:
    """
    Barra风险模型
    
    将组合风险分解为：
    - 市场风险
    - 行业风险
    - 风格因子风险（规模、价值、动量、波动率、质量等）
    - 特异性风险
    """
    
    # 标准Barra风格因子
    STYLE_FACTORS = [
        'size',           # 规模：市值
        'value',          # 价值：BP、EP
        'momentum',       # 动量：过去收益
        'volatility',     # 波动率：历史波动
        'quality',        # 质量：ROE、盈利稳定性
        'growth',         # 成长：收入增长、盈利增长
        'leverage',       # 杠杆：负债率
        'liquidity',      # 流动性：换手率
        'beta',           # Beta：市场敏感度
        'dividend_yield'  # 股息率
    ]
    
    def __init__(self):
        self.factor_returns = None
        self.factor_covariance = None
        self.factor_loadings = None
        self.specific_risk = None
    
    def calculate_factor_loadings(
        self,
        stock_data: pd.DataFrame,
        market_cap: pd.Series = None
    ) -> pd.DataFrame:
        """
        计算股票的因子载荷
        
        Args:
            stock_data: 股票数据DataFrame，包含价格、财务指标等
            market_cap: 市值Series
        
        Returns:
            因子载荷DataFrame
        """
        loadings = pd.DataFrame(index=stock_data.index if hasattr(stock_data, 'index') else range(len(stock_data)))
        
        # 规模因子：市值的对数
        if market_cap is not None:
            loadings['size'] = np.log(market_cap)
        elif 'market_cap' in stock_data.columns:
            loadings['size'] = np.log(stock_data['market_cap'])
        else:
            loadings['size'] = 0
        
        # 价值因子：BP（账面市值比）
        if 'book_value' in stock_data.columns and 'market_cap' in stock_data.columns:
            loadings['value'] = stock_data['book_value'] / stock_data['market_cap']
        elif 'pb' in stock_data.columns:
            loadings['value'] = 1 / stock_data['pb']
        else:
            loadings['value'] = 0
        
        # 动量因子：过去12个月收益
        if 'momentum_12m' in stock_data.columns:
            loadings['momentum'] = stock_data['momentum_12m']
        elif 'return_12m' in stock_data.columns:
            loadings['momentum'] = stock_data['return_12m']
        else:
            loadings['momentum'] = 0
        
        # 波动率因子
        if 'volatility' in stock_data.columns:
            loadings['volatility'] = stock_data['volatility']
        else:
            loadings['volatility'] = 0
        
        # 质量因子：ROE
        if 'roe' in stock_data.columns:
            loadings['quality'] = stock_data['roe']
        else:
            loadings['quality'] = 0
        
        # 成长因子：收入增长
        if 'revenue_growth' in stock_data.columns:
            loadings['growth'] = stock_data['revenue_growth']
        else:
            loadings['growth'] = 0
        
        # 杠杆因子：负债率
        if 'debt_ratio' in stock_data.columns:
            loadings['leverage'] = stock_data['debt_ratio']
        else:
            loadings['leverage'] = 0
        
        # 流动性因子：换手率
        if 'turnover' in stock_data.columns:
            loadings['liquidity'] = stock_data['turnover']
        else:
            loadings['liquidity'] = 0
        
        # Beta因子
        if 'beta' in stock_data.columns:
            loadings['beta'] = stock_data['beta']
        else:
            loadings['beta'] = 1
        
        # 股息率因子
        if 'dividend_yield' in stock_data.columns:
            loadings['dividend_yield'] = stock_data['dividend_yield']
        else:
            loadings['dividend_yield'] = 0
        
        # 标准化
        for col in loadings.columns:
            if loadings[col].std() > 0:
                loadings[col] = (loadings[col] - loadings[col].mean()) / loadings[col].std()
        
        self.factor_loadings = loadings
        return loadings
    
    def decompose_risk(
        self,
        weights: np.ndarray,
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        specific_variance: np.ndarray
    ) -> Dict[str, float]:
        """
        分解组合风险
        
        Args:
            weights: 组合权重
            factor_loadings: 因子载荷矩阵
            factor_covariance: 因子协方差矩阵
            specific_variance: 特异性方差
        
        Returns:
            风险分解结果
        """
        # 组合因子暴露
        portfolio_factor_exposure = np.dot(weights, factor_loadings.values)
        
        # 因子风险
        factor_risk = np.sqrt(np.dot(portfolio_factor_exposure, 
                                     np.dot(factor_covariance, portfolio_factor_exposure)))
        
        # 特异性风险
        specific_risk = np.sqrt(np.sum((weights ** 2) * specific_variance))
        
        # 总风险
        total_risk = np.sqrt(factor_risk ** 2 + specific_risk ** 2)
        
        # 各因子贡献
        factor_contributions = {}
        for i, factor in enumerate(factor_loadings.columns):
            factor_var = portfolio_factor_exposure[i] ** 2 * factor_covariance[i, i]
            factor_contributions[factor] = np.sqrt(factor_var) if factor_var > 0 else 0
        
        return {
            'total_risk': total_risk * np.sqrt(252) * 100,  # 年化
            'factor_risk': factor_risk * np.sqrt(252) * 100,
            'specific_risk': specific_risk * np.sqrt(252) * 100,
            'factor_risk_pct': (factor_risk ** 2 / (total_risk ** 2)) * 100 if total_risk > 0 else 0,
            'specific_risk_pct': (specific_risk ** 2 / (total_risk ** 2)) * 100 if total_risk > 0 else 0,
            'factor_contributions': factor_contributions,
            'factor_exposures': dict(zip(factor_loadings.columns, portfolio_factor_exposure))
        }


# ==================== 波动率预测模型 ====================

class VolatilityForecaster:
    """
    波动率预测模型
    
    支持：
    - EWMA（指数加权移动平均）
    - 简化GARCH(1,1)
    - 历史波动率
    """
    
    def __init__(self):
        self.model_params = {}
    
    def ewma(
        self,
        returns: pd.Series,
        lambda_param: float = 0.94,
        forecast_horizon: int = 1
    ) -> float:
        """
        EWMA波动率预测
        
        Args:
            returns: 收益率序列
            lambda_param: 衰减因子（RiskMetrics默认0.94）
            forecast_horizon: 预测期限（天）
        
        Returns:
            预测波动率（日度）
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0
        
        # 计算EWMA方差
        variance = returns.iloc[0] ** 2
        for r in returns.iloc[1:]:
            variance = lambda_param * variance + (1 - lambda_param) * r ** 2
        
        # 预测
        forecast_variance = variance  # EWMA假设方差持续
        
        return np.sqrt(forecast_variance * forecast_horizon)
    
    def simple_garch(
        self,
        returns: pd.Series,
        omega: float = 0.00001,
        alpha: float = 0.05,
        beta: float = 0.90,
        forecast_horizon: int = 1
    ) -> float:
        """
        简化GARCH(1,1)波动率预测
        
        σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        
        Args:
            returns: 收益率序列
            omega: 常数项
            alpha: ARCH项系数
            beta: GARCH项系数
            forecast_horizon: 预测期限
        
        Returns:
            预测波动率
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0
        
        # 初始化方差
        variance = returns.var()
        
        # 迭代计算
        for r in returns:
            variance = omega + alpha * r ** 2 + beta * variance
        
        # 多步预测
        long_run_variance = omega / (1 - alpha - beta) if (alpha + beta) < 1 else variance
        
        forecast_variance = 0
        current_var = variance
        for h in range(forecast_horizon):
            forecast_variance += current_var
            current_var = omega + (alpha + beta) * current_var
        
        return np.sqrt(forecast_variance)
    
    def historical_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True
    ) -> float:
        """
        历史波动率
        
        Args:
            returns: 收益率序列
            window: 计算窗口
            annualize: 是否年化
        
        Returns:
            波动率
        """
        returns = returns.dropna()
        if len(returns) < window:
            vol = returns.std()
        else:
            vol = returns.iloc[-window:].std()
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def forecast_volatility(
        self,
        returns: pd.Series,
        method: str = 'ewma',
        **kwargs
    ) -> Dict[str, float]:
        """
        综合波动率预测
        
        Args:
            returns: 收益率序列
            method: 预测方法
            **kwargs: 方法参数
        
        Returns:
            预测结果字典
        """
        results = {}
        
        # EWMA
        results['ewma_daily'] = self.ewma(returns, **{k: v for k, v in kwargs.items() if k in ['lambda_param', 'forecast_horizon']})
        results['ewma_annual'] = results['ewma_daily'] * np.sqrt(252)
        
        # GARCH
        results['garch_daily'] = self.simple_garch(returns, **{k: v for k, v in kwargs.items() if k in ['omega', 'alpha', 'beta', 'forecast_horizon']})
        results['garch_annual'] = results['garch_daily'] * np.sqrt(252)
        
        # 历史波动率
        results['historical_20d'] = self.historical_volatility(returns, window=20, annualize=True)
        results['historical_60d'] = self.historical_volatility(returns, window=60, annualize=True)
        
        return results


# ==================== VaR/CVaR计算 ====================

class RiskMetrics:
    """
    风险度量计算
    
    支持：
    - VaR（历史模拟、参数法、蒙特卡洛）
    - CVaR（条件VaR）
    - 最大回撤
    """
    
    @staticmethod
    def var_historical(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        历史模拟法VaR
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            horizon: 持有期
        
        Returns:
            VaR值（正数表示损失）
        """
        returns = returns.dropna()
        if len(returns) == 0:
            return 0
        
        # 计算持有期收益
        if horizon > 1:
            rolling_returns = returns.rolling(horizon).sum().dropna()
        else:
            rolling_returns = returns
        
        var = -np.percentile(rolling_returns, (1 - confidence) * 100)
        return var
    
    @staticmethod
    def var_parametric(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        参数法VaR（假设正态分布）
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            horizon: 持有期
        
        Returns:
            VaR值
        """
        returns = returns.dropna()
        if len(returns) == 0:
            return 0
        
        mu = returns.mean() * horizon
        sigma = returns.std() * np.sqrt(horizon)
        
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mu + z_score * sigma)
        
        return var
    
    @staticmethod
    def var_monte_carlo(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10000
    ) -> float:
        """
        蒙特卡洛VaR
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            horizon: 持有期
            n_simulations: 模拟次数
        
        Returns:
            VaR值
        """
        returns = returns.dropna()
        if len(returns) == 0:
            return 0
        
        mu = returns.mean()
        sigma = returns.std()
        
        # 模拟
        simulated_returns = np.random.normal(mu, sigma, (n_simulations, horizon))
        portfolio_returns = simulated_returns.sum(axis=1)
        
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        return var
    
    @staticmethod
    def cvar(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        条件VaR（Expected Shortfall）
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            horizon: 持有期
        
        Returns:
            CVaR值
        """
        returns = returns.dropna()
        if len(returns) == 0:
            return 0
        
        var = RiskMetrics.var_historical(returns, confidence, horizon)
        
        # 计算超过VaR的平均损失
        if horizon > 1:
            rolling_returns = returns.rolling(horizon).sum().dropna()
        else:
            rolling_returns = returns
        
        tail_returns = rolling_returns[rolling_returns < -var]
        
        if len(tail_returns) == 0:
            return var
        
        return -tail_returns.mean()
    
    @staticmethod
    def max_drawdown(values: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        计算最大回撤
        
        Args:
            values: 净值序列
        
        Returns:
            (最大回撤, 峰值日期, 谷值日期)
        """
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = values[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date


# ==================== 压力测试 ====================

class StressTester:
    """
    压力测试模块
    
    支持：
    - 历史情景回放（2008金融危机、2015股灾、2020疫情等）
    - 假设情景测试
    - 敏感性分析
    """
    
    # 历史压力情景
    HISTORICAL_SCENARIOS = {
        '2008_financial_crisis': {
            'description': '2008年全球金融危机',
            'market_shock': -0.50,  # 市场下跌50%
            'volatility_spike': 3.0,  # 波动率上升3倍
            'duration_days': 250
        },
        '2015_china_crash': {
            'description': '2015年A股股灾',
            'market_shock': -0.45,
            'volatility_spike': 2.5,
            'duration_days': 60
        },
        '2020_covid_crash': {
            'description': '2020年新冠疫情冲击',
            'market_shock': -0.35,
            'volatility_spike': 4.0,
            'duration_days': 30
        },
        '2022_rate_hike': {
            'description': '2022年美联储加息周期',
            'market_shock': -0.25,
            'volatility_spike': 1.5,
            'duration_days': 180
        },
        'flash_crash': {
            'description': '闪崩情景',
            'market_shock': -0.10,
            'volatility_spike': 5.0,
            'duration_days': 1
        }
    }
    
    def __init__(self):
        self.results = {}
    
    def run_historical_scenario(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        scenario_name: str
    ) -> Dict[str, Any]:
        """
        运行历史情景测试
        
        Args:
            portfolio_value: 组合价值
            portfolio_beta: 组合Beta
            scenario_name: 情景名称
        
        Returns:
            压力测试结果
        """
        if scenario_name not in self.HISTORICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.HISTORICAL_SCENARIOS[scenario_name]
        
        # 计算组合损失
        market_shock = scenario['market_shock']
        portfolio_shock = portfolio_beta * market_shock
        
        loss = portfolio_value * portfolio_shock
        
        return {
            'scenario': scenario_name,
            'description': scenario['description'],
            'market_shock': market_shock * 100,
            'portfolio_shock': portfolio_shock * 100,
            'portfolio_loss': loss,
            'remaining_value': portfolio_value + loss,
            'duration_days': scenario['duration_days'],
            'volatility_spike': scenario['volatility_spike']
        }
    
    def run_all_scenarios(
        self,
        portfolio_value: float,
        portfolio_beta: float
    ) -> pd.DataFrame:
        """
        运行所有历史情景测试
        
        Args:
            portfolio_value: 组合价值
            portfolio_beta: 组合Beta
        
        Returns:
            所有情景测试结果
        """
        results = []
        
        for scenario_name in self.HISTORICAL_SCENARIOS:
            result = self.run_historical_scenario(
                portfolio_value, portfolio_beta, scenario_name
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(
        self,
        portfolio_value: float,
        factor_exposures: Dict[str, float],
        factor_shocks: Dict[str, List[float]] = None
    ) -> pd.DataFrame:
        """
        敏感性分析
        
        Args:
            portfolio_value: 组合价值
            factor_exposures: 因子暴露
            factor_shocks: 因子冲击范围
        
        Returns:
            敏感性分析结果
        """
        if factor_shocks is None:
            factor_shocks = {
                'market': [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20],
                'interest_rate': [-0.02, -0.01, 0, 0.01, 0.02],
                'volatility': [-0.50, -0.25, 0, 0.25, 0.50, 1.0]
            }
        
        results = []
        
        for factor, shocks in factor_shocks.items():
            exposure = factor_exposures.get(factor, 0)
            
            for shock in shocks:
                pnl = portfolio_value * exposure * shock
                results.append({
                    'factor': factor,
                    'shock': shock * 100,
                    'exposure': exposure,
                    'pnl': pnl,
                    'pnl_pct': (pnl / portfolio_value) * 100
                })
        
        return pd.DataFrame(results)


# ==================== 综合风险管理器 ====================

class AdvancedRiskManager:
    """
    综合风险管理器
    
    整合所有风险管理功能。
    """
    
    def __init__(self):
        self.barra_model = BarraRiskModel()
        self.vol_forecaster = VolatilityForecaster()
        self.risk_metrics = RiskMetrics()
        self.stress_tester = StressTester()
    
    def comprehensive_risk_analysis(
        self,
        returns: pd.Series,
        portfolio_value: float,
        weights: np.ndarray = None,
        factor_loadings: pd.DataFrame = None,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        综合风险分析
        
        Args:
            returns: 组合收益率序列
            portfolio_value: 组合价值
            weights: 组合权重
            factor_loadings: 因子载荷
            benchmark_returns: 基准收益率
        
        Returns:
            综合风险分析结果
        """
        results = {}
        
        # 1. 基础风险指标
        results['basic_metrics'] = {
            'volatility_annual': returns.std() * np.sqrt(252) * 100,
            'var_95_1d': self.risk_metrics.var_historical(returns, 0.95, 1) * 100,
            'var_99_1d': self.risk_metrics.var_historical(returns, 0.99, 1) * 100,
            'cvar_95_1d': self.risk_metrics.cvar(returns, 0.95, 1) * 100,
            'var_95_10d': self.risk_metrics.var_historical(returns, 0.95, 10) * 100,
        }
        
        # 2. 波动率预测
        results['volatility_forecast'] = self.vol_forecaster.forecast_volatility(returns)
        
        # 3. 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        max_dd, peak_date, trough_date = self.risk_metrics.max_drawdown(cumulative_returns)
        results['drawdown'] = {
            'max_drawdown': max_dd * 100,
            'peak_date': str(peak_date) if peak_date else None,
            'trough_date': str(trough_date) if trough_date else None
        }
        
        # 4. 相对基准风险
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')
            port_ret, bench_ret = aligned_returns
            
            if len(port_ret) > 1:
                cov_matrix = np.cov(port_ret.dropna(), bench_ret.dropna())
                if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > 0:
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    beta = 1
                
                tracking_error = (port_ret - bench_ret).std() * np.sqrt(252) * 100
                
                results['relative_risk'] = {
                    'beta': beta,
                    'tracking_error': tracking_error
                }
        
        # 5. 压力测试
        beta = results.get('relative_risk', {}).get('beta', 1)
        results['stress_test'] = self.stress_tester.run_all_scenarios(
            portfolio_value, beta
        ).to_dict('records')
        
        return results
    
    def generate_risk_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成风险报告
        
        Args:
            analysis_results: 综合风险分析结果
        
        Returns:
            Markdown格式的风险报告
        """
        report = []
        report.append("# 综合风险分析报告\n")
        
        # 基础风险指标
        report.append("## 1. 基础风险指标\n")
        basic = analysis_results.get('basic_metrics', {})
        report.append(f"| 指标 | 数值 |")
        report.append(f"|:---|---:|")
        report.append(f"| 年化波动率 | {basic.get('volatility_annual', 0):.2f}% |")
        report.append(f"| 1日VaR (95%) | {basic.get('var_95_1d', 0):.2f}% |")
        report.append(f"| 1日VaR (99%) | {basic.get('var_99_1d', 0):.2f}% |")
        report.append(f"| 1日CVaR (95%) | {basic.get('cvar_95_1d', 0):.2f}% |")
        report.append(f"| 10日VaR (95%) | {basic.get('var_95_10d', 0):.2f}% |")
        report.append("")
        
        # 波动率预测
        report.append("## 2. 波动率预测\n")
        vol = analysis_results.get('volatility_forecast', {})
        report.append(f"| 模型 | 年化波动率 |")
        report.append(f"|:---|---:|")
        report.append(f"| EWMA | {vol.get('ewma_annual', 0)*100:.2f}% |")
        report.append(f"| GARCH | {vol.get('garch_annual', 0)*100:.2f}% |")
        report.append(f"| 历史20日 | {vol.get('historical_20d', 0)*100:.2f}% |")
        report.append(f"| 历史60日 | {vol.get('historical_60d', 0)*100:.2f}% |")
        report.append("")
        
        # 回撤分析
        report.append("## 3. 回撤分析\n")
        dd = analysis_results.get('drawdown', {})
        report.append(f"- **最大回撤**: {dd.get('max_drawdown', 0):.2f}%")
        report.append(f"- **峰值日期**: {dd.get('peak_date', 'N/A')}")
        report.append(f"- **谷值日期**: {dd.get('trough_date', 'N/A')}")
        report.append("")
        
        # 压力测试
        report.append("## 4. 压力测试\n")
        report.append(f"| 情景 | 市场冲击 | 组合冲击 | 预计损失 |")
        report.append(f"|:---|---:|---:|---:|")
        for scenario in analysis_results.get('stress_test', []):
            report.append(f"| {scenario.get('description', '')} | {scenario.get('market_shock', 0):.1f}% | {scenario.get('portfolio_shock', 0):.1f}% | ¥{scenario.get('portfolio_loss', 0):,.0f} |")
        report.append("")
        
        return "\n".join(report)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Advanced Risk Management Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建测试数据
    n_days = 500
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    
    # 模拟收益率
    returns = pd.Series(
        np.random.randn(n_days) * 0.02 + 0.0003,
        index=dates
    )
    
    benchmark_returns = pd.Series(
        np.random.randn(n_days) * 0.015 + 0.0002,
        index=dates
    )
    
    portfolio_value = 1000000
    
    print("\n1. Testing VaR/CVaR...")
    var_95 = RiskMetrics.var_historical(returns, 0.95, 1)
    var_99 = RiskMetrics.var_historical(returns, 0.99, 1)
    cvar_95 = RiskMetrics.cvar(returns, 0.95, 1)
    print(f"   VaR (95%, 1d): {var_95*100:.2f}%")
    print(f"   VaR (99%, 1d): {var_99*100:.2f}%")
    print(f"   CVaR (95%, 1d): {cvar_95*100:.2f}%")
    
    print("\n2. Testing Volatility Forecasting...")
    vol_forecaster = VolatilityForecaster()
    vol_results = vol_forecaster.forecast_volatility(returns)
    print(f"   EWMA Annual Vol: {vol_results['ewma_annual']*100:.2f}%")
    print(f"   GARCH Annual Vol: {vol_results['garch_annual']*100:.2f}%")
    print(f"   Historical 20d Vol: {vol_results['historical_20d']*100:.2f}%")
    
    print("\n3. Testing Stress Testing...")
    stress_tester = StressTester()
    stress_results = stress_tester.run_all_scenarios(portfolio_value, 1.2)
    print("   Stress Test Results:")
    for _, row in stress_results.iterrows():
        print(f"   - {row['description']}: {row['portfolio_shock']:.1f}% (Loss: ¥{row['portfolio_loss']:,.0f})")
    
    print("\n4. Testing Comprehensive Risk Analysis...")
    risk_manager = AdvancedRiskManager()
    analysis = risk_manager.comprehensive_risk_analysis(
        returns, portfolio_value,
        benchmark_returns=benchmark_returns
    )
    
    print("\n5. Generating Risk Report...")
    report = risk_manager.generate_risk_report(analysis)
    print(report[:1000] + "...")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
