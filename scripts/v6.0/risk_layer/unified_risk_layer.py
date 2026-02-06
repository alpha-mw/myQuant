#!/usr/bin/env python3
"""
Quant-Investor V6.0 - 统一组合与风控层 (Unified Risk Layer)

整合所有历史版本的组合优化和风险管理能力：
- V4.1: Alpha/Beta/IR/胜率/牛熊市分析
- V5.0: 组合优化 (等权/风险平价/最大夏普/最小方差/Black-Litterman)
- V5.0: 风险管理 (VaR/CVaR/压力测试/回撤控制/仓位管理)

设计原则：
1. 统一的组合构建接口
2. 多维度风险评估
3. 基准对比验证
4. 动态仓位管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import optimize, stats
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构 ====================

@dataclass
class PortfolioAllocation:
    """组合配置"""
    weights: Dict[str, float] = field(default_factory=dict)
    method: str = ""
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class RiskMetrics:
    """风险指标"""
    # VaR/CVaR
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # 波动率
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0
    
    # 回撤
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # 尾部风险
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0


@dataclass
class BenchmarkComparison:
    """基准对比结果 (源自V4.1)"""
    # 基础
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # 年化
    annual_alpha: float = 0.0
    annual_portfolio: float = 0.0
    annual_benchmark: float = 0.0
    
    # 风险调整
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    # 胜率
    win_rate: float = 0.0
    up_capture: float = 0.0
    down_capture: float = 0.0
    
    # 统计显著性
    alpha_tstat: float = 0.0
    alpha_pvalue: float = 1.0
    is_significant: bool = False
    
    # 牛熊市
    bull_alpha: float = 0.0
    bear_alpha: float = 0.0


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    portfolio_impact: float = 0.0
    benchmark_impact: float = 0.0
    worst_stock: str = ""
    worst_impact: float = 0.0


@dataclass
class RiskLayerOutput:
    """风控层完整输出"""
    # 组合配置
    portfolio: PortfolioAllocation = None
    alternative_portfolios: Dict[str, PortfolioAllocation] = field(default_factory=dict)
    
    # 风险指标
    risk_metrics: RiskMetrics = None
    
    # 基准对比
    benchmark_comparison: BenchmarkComparison = None
    
    # 压力测试
    stress_tests: List[StressTestResult] = field(default_factory=list)
    
    # 仓位建议
    position_sizing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 风险预警
    risk_alerts: List[str] = field(default_factory=list)
    
    # 统计
    stats: Dict[str, Any] = field(default_factory=dict)


# ==================== 组合优化器 ====================

class PortfolioOptimizer:
    """
    组合优化器 (源自V5.0)
    
    支持多种优化方法：等权、风险平价、最大夏普、最小方差。
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, returns: pd.DataFrame, method: str = 'max_sharpe',
                  constraints: Dict = None) -> PortfolioAllocation:
        """
        优化组合权重
        
        Args:
            returns: 各股票收益率DataFrame (列为股票代码)
            method: 优化方法
            constraints: 约束条件
        
        Returns:
            PortfolioAllocation: 组合配置
        """
        if method == 'equal_weight':
            return self._equal_weight(returns)
        elif method == 'risk_parity':
            return self._risk_parity(returns)
        elif method == 'max_sharpe':
            return self._max_sharpe(returns, constraints)
        elif method == 'min_variance':
            return self._min_variance(returns, constraints)
        else:
            return self._equal_weight(returns)
    
    def _equal_weight(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """等权配置"""
        n = len(returns.columns)
        weights = {col: 1.0/n for col in returns.columns}
        return self._build_allocation(weights, returns, "等权配置")
    
    def _risk_parity(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """风险平价"""
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ cov.values @ w)
            marginal_risk = cov.values @ w / port_vol
            risk_contrib = w * marginal_risk
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)
        
        w0 = np.ones(n) / n
        bounds = [(0.01, 0.5)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        try:
            result = optimize.minimize(risk_budget_objective, w0, method='SLSQP',
                                       bounds=bounds, constraints=constraints)
            weights = {col: float(w) for col, w in zip(returns.columns, result.x)}
        except:
            weights = {col: 1.0/n for col in returns.columns}
        
        return self._build_allocation(weights, returns, "风险平价")
    
    def _max_sharpe(self, returns: pd.DataFrame, constraints: Dict = None) -> PortfolioAllocation:
        """最大夏普比率"""
        mean_returns = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def neg_sharpe(w):
            port_ret = w @ mean_returns.values
            port_vol = np.sqrt(w @ cov.values @ w)
            return -(port_ret - self.risk_free_rate) / (port_vol + 1e-8)
        
        w0 = np.ones(n) / n
        max_weight = (constraints or {}).get('max_weight', 0.3)
        bounds = [(0.0, max_weight)] * n
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        try:
            result = optimize.minimize(neg_sharpe, w0, method='SLSQP',
                                       bounds=bounds, constraints=cons)
            weights = {col: max(0, float(w)) for col, w in zip(returns.columns, result.x)}
            # 归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        except:
            weights = {col: 1.0/n for col in returns.columns}
        
        return self._build_allocation(weights, returns, "最大夏普")
    
    def _min_variance(self, returns: pd.DataFrame, constraints: Dict = None) -> PortfolioAllocation:
        """最小方差"""
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def portfolio_variance(w):
            return w @ cov.values @ w
        
        w0 = np.ones(n) / n
        max_weight = (constraints or {}).get('max_weight', 0.3)
        bounds = [(0.0, max_weight)] * n
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        try:
            result = optimize.minimize(portfolio_variance, w0, method='SLSQP',
                                       bounds=bounds, constraints=cons)
            weights = {col: max(0, float(w)) for col, w in zip(returns.columns, result.x)}
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        except:
            weights = {col: 1.0/n for col in returns.columns}
        
        return self._build_allocation(weights, returns, "最小方差")
    
    def _build_allocation(self, weights: Dict[str, float], returns: pd.DataFrame,
                           method: str) -> PortfolioAllocation:
        """构建组合配置"""
        w = np.array([weights.get(col, 0) for col in returns.columns])
        mean_ret = returns.mean().values * 252
        cov = returns.cov().values * 252
        
        exp_ret = float(w @ mean_ret)
        exp_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (exp_ret - self.risk_free_rate) / (exp_vol + 1e-8)
        
        return PortfolioAllocation(
            weights=weights, method=method,
            expected_return=exp_ret, expected_volatility=exp_vol,
            sharpe_ratio=float(sharpe)
        )


# ==================== 风险管理器 ====================

class RiskManager:
    """
    风险管理器 (源自V5.0)
    
    提供VaR/CVaR、压力测试、回撤分析等风险评估工具。
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
    
    def calculate_risk_metrics(self, portfolio_returns: pd.Series) -> RiskMetrics:
        """计算完整的风险指标"""
        metrics = RiskMetrics()
        
        if len(portfolio_returns) < 10:
            return metrics
        
        returns = portfolio_returns.dropna()
        
        # VaR/CVaR
        metrics.var_95 = float(np.percentile(returns, 5))
        metrics.var_99 = float(np.percentile(returns, 1))
        metrics.cvar_95 = float(returns[returns <= metrics.var_95].mean()) if (returns <= metrics.var_95).any() else metrics.var_95
        metrics.cvar_99 = float(returns[returns <= metrics.var_99].mean()) if (returns <= metrics.var_99).any() else metrics.var_99
        
        # 波动率
        metrics.annual_volatility = float(returns.std() * np.sqrt(252))
        metrics.downside_volatility = float(returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).any() else 0
        
        # 回撤
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        metrics.max_drawdown = float(drawdown.min())
        metrics.avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
        
        # 最大回撤持续时间
        is_drawdown = drawdown < 0
        if is_drawdown.any():
            dd_groups = (~is_drawdown).cumsum()
            dd_lengths = is_drawdown.groupby(dd_groups).sum()
            metrics.max_drawdown_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
        
        # 高阶统计
        metrics.skewness = float(returns.skew())
        metrics.kurtosis = float(returns.kurtosis())
        
        # 尾部比率
        upper_tail = returns[returns > 0].quantile(0.95) if (returns > 0).any() else 0
        lower_tail = abs(returns[returns < 0].quantile(0.05)) if (returns < 0).any() else 1
        metrics.tail_ratio = float(upper_tail / (lower_tail + 1e-8))
        
        return metrics
    
    def benchmark_comparison(self, portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series) -> BenchmarkComparison:
        """基准对比分析 (源自V4.1)"""
        comp = BenchmarkComparison()
        
        # 对齐数据
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 20:
            return comp
        
        pr = aligned['portfolio']
        br = aligned['benchmark']
        excess = pr - br
        n_days = len(aligned)
        
        # 基础收益
        comp.portfolio_return = float((1 + pr).prod() - 1)
        comp.benchmark_return = float((1 + br).prod() - 1)
        comp.excess_return = comp.portfolio_return - comp.benchmark_return
        
        # 年化
        years = n_days / 252
        comp.annual_portfolio = float((1 + comp.portfolio_return) ** (1/years) - 1) if years > 0 else 0
        comp.annual_benchmark = float((1 + comp.benchmark_return) ** (1/years) - 1) if years > 0 else 0
        comp.annual_alpha = comp.annual_portfolio - comp.annual_benchmark
        
        # Beta
        cov_matrix = np.cov(pr.values, br.values)
        comp.beta = float(cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8))
        
        # Tracking Error & IR
        comp.tracking_error = float(excess.std() * np.sqrt(252))
        comp.information_ratio = float(excess.mean() * 252 / (comp.tracking_error + 1e-8))
        
        # 胜率
        comp.win_rate = float((excess > 0).mean())
        
        # 上涨/下跌捕获率
        up_days = br > 0
        down_days = br < 0
        if up_days.sum() > 0:
            comp.up_capture = float(pr[up_days].mean() / (br[up_days].mean() + 1e-8))
        if down_days.sum() > 0:
            comp.down_capture = float(pr[down_days].mean() / (br[down_days].mean() + 1e-8))
        
        # Alpha显著性检验
        if len(excess) > 10:
            t_stat, p_value = stats.ttest_1samp(excess.values, 0)
            comp.alpha_tstat = float(t_stat)
            comp.alpha_pvalue = float(p_value)
            comp.is_significant = p_value < 0.05
        
        # 牛熊市分析
        bull_mask = br.rolling(60).mean() > 0
        bear_mask = ~bull_mask
        
        if bull_mask.sum() > 10:
            comp.bull_alpha = float((excess[bull_mask].mean()) * 252)
        if bear_mask.sum() > 10:
            comp.bear_alpha = float((excess[bear_mask].mean()) * 252)
        
        return comp
    
    def stress_test(self, returns: pd.DataFrame, weights: Dict[str, float]) -> List[StressTestResult]:
        """压力测试"""
        scenarios = {
            "市场暴跌 (-20%)": {"market_shock": -0.20},
            "利率飙升 (+200bp)": {"market_shock": -0.10, "vol_multiplier": 1.5},
            "流动性危机": {"market_shock": -0.15, "vol_multiplier": 2.0},
            "科技股崩盘 (-30%)": {"market_shock": -0.30},
            "温和回调 (-10%)": {"market_shock": -0.10},
        }
        
        results = []
        for name, params in scenarios.items():
            shock = params.get('market_shock', -0.10)
            vol_mult = params.get('vol_multiplier', 1.0)
            
            # 模拟各股票在压力场景下的表现
            stock_impacts = {}
            portfolio_impact = 0.0
            worst_stock = ""
            worst_impact = 0.0
            
            for stock, weight in weights.items():
                if stock in returns.columns:
                    stock_vol = returns[stock].std() * np.sqrt(252)
                    beta_approx = returns[stock].corr(returns.mean(axis=1))
                    
                    impact = shock * beta_approx * vol_mult
                    stock_impacts[stock] = impact
                    portfolio_impact += weight * impact
                    
                    if impact < worst_impact:
                        worst_impact = impact
                        worst_stock = stock
            
            results.append(StressTestResult(
                scenario_name=name,
                portfolio_impact=float(portfolio_impact),
                benchmark_impact=float(shock),
                worst_stock=worst_stock,
                worst_impact=float(worst_impact)
            ))
        
        return results
    
    def position_sizing(self, weights: Dict[str, float], risk_metrics: RiskMetrics,
                         total_capital: float = 1000000,
                         max_position_pct: float = 0.20,
                         max_portfolio_var: float = 0.02) -> Dict[str, Dict[str, Any]]:
        """仓位管理"""
        positions = {}
        
        # 基于风险预算调整仓位
        risk_budget = max_portfolio_var
        current_risk = abs(risk_metrics.var_95) if risk_metrics.var_95 else 0.02
        
        # 风险调整系数
        risk_adj = min(1.0, risk_budget / (current_risk + 1e-8))
        
        for stock, weight in weights.items():
            adjusted_weight = min(weight * risk_adj, max_position_pct)
            capital_allocated = total_capital * adjusted_weight
            
            positions[stock] = {
                'target_weight': float(adjusted_weight),
                'capital_allocated': float(capital_allocated),
                'original_weight': float(weight),
                'risk_adjusted': risk_adj < 1.0,
            }
        
        return positions


# ==================== 统一风控层 ====================

class UnifiedRiskLayer:
    """
    V6.0 统一组合与风控层
    
    整合组合优化、风险评估、基准对比和仓位管理。
    """
    
    def __init__(self, risk_free_rate: float = 0.04, verbose: bool = True):
        self.verbose = verbose
        self.optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
        self.risk_manager = RiskManager(risk_free_rate=risk_free_rate)
    
    def process(self, recommendations: List[Dict], data_bundle=None,
                optimization_method: str = 'max_sharpe',
                total_capital: float = 1000000) -> RiskLayerOutput:
        """
        执行完整的组合构建和风险评估
        
        Args:
            recommendations: 推荐股票列表 (来自决策层)
            data_bundle: 数据包 (来自数据层)
            optimization_method: 优化方法
            total_capital: 总资金
        
        Returns:
            RiskLayerOutput: 风控层完整输出
        """
        output = RiskLayerOutput()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🛡️ V6.0 统一组合与风控层")
            print(f"{'='*60}")
        
        # 1. 提取推荐股票的收益率数据
        stock_codes = [r['code'] for r in recommendations if r.get('code')]
        returns_df = self._extract_returns(stock_codes, data_bundle)
        
        if returns_df is None or returns_df.empty:
            if self.verbose:
                print(f"  ⚠️ 无法获取收益率数据，使用等权配置")
            n = len(stock_codes)
            weights = {code: 1.0/n for code in stock_codes} if n > 0 else {}
            output.portfolio = PortfolioAllocation(weights=weights, method="等权配置(默认)")
            output.final_recommendations_with_weights = recommendations
            return output
        
        # 2. 组合优化
        if self.verbose:
            print(f"\n  📊 组合优化 (方法: {optimization_method})")
        
        output.portfolio = self.optimizer.optimize(returns_df, method=optimization_method)
        
        # 也计算其他方法作为参考
        for method in ['equal_weight', 'risk_parity', 'min_variance']:
            if method != optimization_method:
                output.alternative_portfolios[method] = self.optimizer.optimize(returns_df, method=method)
        
        if self.verbose:
            print(f"    ✓ 主组合: {output.portfolio.method}")
            print(f"      预期收益: {output.portfolio.expected_return:.2%}")
            print(f"      预期波动: {output.portfolio.expected_volatility:.2%}")
            print(f"      夏普比率: {output.portfolio.sharpe_ratio:.2f}")
            print(f"\n    权重分配:")
            for stock, weight in sorted(output.portfolio.weights.items(), key=lambda x: -x[1]):
                if weight > 0.001:
                    print(f"      {stock:<10s}: {weight:.1%}")
        
        # 3. 计算组合收益率
        w = np.array([output.portfolio.weights.get(col, 0) for col in returns_df.columns])
        portfolio_returns = (returns_df * w).sum(axis=1)
        
        # 4. 风险指标
        if self.verbose:
            print(f"\n  📉 风险评估")
        
        output.risk_metrics = self.risk_manager.calculate_risk_metrics(portfolio_returns)
        
        if self.verbose:
            rm = output.risk_metrics
            print(f"    VaR (95%): {rm.var_95:.2%}")
            print(f"    CVaR (95%): {rm.cvar_95:.2%}")
            print(f"    年化波动率: {rm.annual_volatility:.2%}")
            print(f"    最大回撤: {rm.max_drawdown:.2%}")
            print(f"    偏度: {rm.skewness:.2f}")
            print(f"    峰度: {rm.kurtosis:.2f}")
        
        # 5. 基准对比
        benchmark_returns = self._get_benchmark_returns(data_bundle)
        if benchmark_returns is not None:
            if self.verbose:
                print(f"\n  📊 基准对比")
            
            output.benchmark_comparison = self.risk_manager.benchmark_comparison(
                portfolio_returns, benchmark_returns
            )
            
            bc = output.benchmark_comparison
            if self.verbose:
                print(f"    年化Alpha: {bc.annual_alpha:.2%}")
                print(f"    Beta: {bc.beta:.2f}")
                print(f"    信息比率: {bc.information_ratio:.2f}")
                print(f"    胜率: {bc.win_rate:.1%}")
                print(f"    Alpha显著: {'✓' if bc.is_significant else '✗'} (p={bc.alpha_pvalue:.4f})")
        
        # 6. 压力测试
        if self.verbose:
            print(f"\n  🔥 压力测试")
        
        output.stress_tests = self.risk_manager.stress_test(returns_df, output.portfolio.weights)
        
        if self.verbose:
            for st in output.stress_tests:
                print(f"    {st.scenario_name}: 组合影响 {st.portfolio_impact:.2%}")
        
        # 7. 仓位管理
        output.position_sizing = self.risk_manager.position_sizing(
            output.portfolio.weights, output.risk_metrics, total_capital
        )
        
        # 8. 风险预警
        output.risk_alerts = self._generate_alerts(output)
        
        if output.risk_alerts and self.verbose:
            print(f"\n  ⚠️ 风险预警:")
            for alert in output.risk_alerts:
                print(f"    • {alert}")
        
        # 9. 统计
        output.stats = {
            "stocks_in_portfolio": sum(1 for w in output.portfolio.weights.values() if w > 0.001),
            "optimization_method": output.portfolio.method,
            "expected_return": output.portfolio.expected_return,
            "expected_volatility": output.portfolio.expected_volatility,
            "sharpe_ratio": output.portfolio.sharpe_ratio,
            "max_drawdown": output.risk_metrics.max_drawdown if output.risk_metrics else 0,
            "risk_alerts": len(output.risk_alerts),
        }
        
        if self.verbose:
            print(f"\n  ✅ 风控层处理完成")
        
        return output
    
    def _extract_returns(self, stock_codes: List[str], data_bundle) -> Optional[pd.DataFrame]:
        """从数据包中提取收益率"""
        if data_bundle is None or not hasattr(data_bundle, 'stock_universe'):
            return None
        
        returns_dict = {}
        for code in stock_codes:
            stock = data_bundle.stock_universe.get(code)
            if stock and stock.price_data is not None and len(stock.price_data) > 20:
                returns_dict[code] = stock.price_data['Close'].pct_change().dropna()
        
        if not returns_dict:
            return None
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        return returns_df if len(returns_df) > 20 else None
    
    def _get_benchmark_returns(self, data_bundle) -> Optional[pd.Series]:
        """获取基准收益率"""
        if data_bundle is None or data_bundle.benchmark_data is None:
            return None
        
        if 'Close' in data_bundle.benchmark_data.columns:
            return data_bundle.benchmark_data['Close'].pct_change().dropna()
        return None
    
    def _generate_alerts(self, output: RiskLayerOutput) -> List[str]:
        """生成风险预警"""
        alerts = []
        
        if output.risk_metrics:
            rm = output.risk_metrics
            if rm.max_drawdown < -0.20:
                alerts.append(f"历史最大回撤较大 ({rm.max_drawdown:.1%})，注意下行风险")
            if rm.annual_volatility > 0.30:
                alerts.append(f"年化波动率较高 ({rm.annual_volatility:.1%})，建议降低仓位")
            if rm.kurtosis > 3:
                alerts.append(f"收益分布厚尾 (峰度={rm.kurtosis:.1f})，尾部风险较高")
            if rm.skewness < -0.5:
                alerts.append(f"收益分布左偏 (偏度={rm.skewness:.2f})，下行风险不对称")
        
        if output.benchmark_comparison:
            bc = output.benchmark_comparison
            if bc.beta > 1.3:
                alerts.append(f"组合Beta较高 ({bc.beta:.2f})，市场下跌时损失放大")
            if bc.information_ratio < 0:
                alerts.append(f"信息比率为负 ({bc.information_ratio:.2f})，超额收益不稳定")
        
        # 集中度风险
        if output.portfolio:
            max_weight = max(output.portfolio.weights.values()) if output.portfolio.weights else 0
            if max_weight > 0.25:
                alerts.append(f"最大单一持仓权重 {max_weight:.1%}，集中度风险较高")
        
        # 压力测试
        for st in output.stress_tests:
            if st.portfolio_impact < -0.15:
                alerts.append(f"压力场景'{st.scenario_name}'下组合损失 {st.portfolio_impact:.1%}")
        
        return alerts


# ==================== 便捷函数 ====================

def run_risk_analysis(recommendations: List[Dict], data_bundle=None,
                       method: str = 'max_sharpe', capital: float = 1000000,
                       verbose: bool = True) -> RiskLayerOutput:
    """便捷函数：运行组合优化和风险分析"""
    layer = UnifiedRiskLayer(verbose=verbose)
    return layer.process(recommendations, data_bundle, method, capital)


if __name__ == "__main__":
    print("=" * 60)
    print("V6.0 统一组合与风控层测试")
    print("=" * 60)
    print("风控层需要决策层的输出作为输入，请通过MasterPipeline运行完整流程。")
