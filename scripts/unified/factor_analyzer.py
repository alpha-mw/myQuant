#!/usr/bin/env python3
"""
Factor Analyzer - 因子检验模块

功能:
1. IC值分析 (信息系数)
2. 分层回测 (单调性检验)
3. 换手率分析 (因子稳定性)
4. 因子相关性分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# 可选可视化依赖
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


@dataclass
class ICAnalysisResult:
    """IC分析结果"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    ic_tstat: float
    ic_series: pd.Series = field(default_factory=pd.Series)


@dataclass
class LayerBacktestResult:
    """分层回测结果"""
    factor_name: str
    layer_returns: Dict[int, pd.Series]
    layer_cum_returns: Dict[int, pd.Series]
    long_short_return: pd.Series
    long_short_cum_return: pd.Series
    monotonicity_score: float


@dataclass
class TurnoverAnalysisResult:
    """换手率分析结果"""
    factor_name: str
    turnover_rate: pd.Series
    mean_turnover: float
    stability_score: float


class FactorAnalyzer:
    """
    因子分析器
    
    单因子检验全流程
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[FactorAnalyzer] {msg}")
    
    # ========== IC值分析 ==========
    
    def calculate_ic(self, df: pd.DataFrame, factor_col: str, 
                     return_col: str = 'label_return',
                     method: str = 'spearman') -> ICAnalysisResult:
        """
        计算IC值 (信息系数)
        
        IC = Correlation(Factor_t, Return_t+1)
        
        Args:
            method: 'spearman' (秩相关, 推荐) 或 'pearson' (线性相关)
        """
        self._log(f"计算IC值: {factor_col}")
        
        # 按日期分组计算IC
        ic_list = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            if len(day_data) < 10:  # 样本太少跳过
                continue
            
            factor_values = day_data[factor_col]
            returns = day_data[return_col]
            
            # 剔除缺失值
            mask = factor_values.notna() & returns.notna()
            if mask.sum() < 10:
                continue
            
            if method == 'spearman':
                ic = factor_values[mask].corr(returns[mask], method='spearman')
            else:
                ic = factor_values[mask].corr(returns[mask], method='pearson')
            
            if not np.isnan(ic):
                ic_list.append({'date': date, 'ic': ic})
        
        ic_df = pd.DataFrame(ic_list)
        if ic_df.empty:
            return ICAnalysisResult(factor_name=factor_col, ic_mean=0, ic_std=0, 
                                   ic_ir=0, ic_positive_ratio=0, ic_tstat=0)
        
        ic_series = ic_df.set_index('date')['ic']
        
        # 计算统计指标
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0  # 信息比率
        ic_positive_ratio = (ic_series > 0).mean()
        ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std != 0 else 0
        
        result = ICAnalysisResult(
            factor_name=factor_col,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_positive_ratio=ic_positive_ratio,
            ic_tstat=ic_tstat,
            ic_series=ic_series
        )
        
        self._log(f"IC分析完成: Mean={ic_mean:.4f}, IR={ic_ir:.4f}, Positive={ic_positive_ratio:.2%}")
        
        return result
    
    def ic_analysis_summary(self, ic_results: List[ICAnalysisResult]) -> pd.DataFrame:
        """IC分析汇总表"""
        summary = []
        for r in ic_results:
            summary.append({
                '因子': r.factor_name,
                'IC均值': r.ic_mean,
                'IC标准差': r.ic_std,
                'IC_IR': r.ic_ir,
                '正IC占比': r.ic_positive_ratio,
                't统计量': r.ic_tstat,
                '有效性': '有效' if abs(r.ic_mean) > 0.02 and r.ic_ir > 0.3 else '无效'
            })
        return pd.DataFrame(summary)
    
    # ========== 分层回测 ==========
    
    def layer_backtest(self, df: pd.DataFrame, factor_col: str,
                       return_col: str = 'label_return',
                       n_layers: int = 5,
                       long_short: bool = True) -> LayerBacktestResult:
        """
        分层回测
        
        按因子值分组，检验收益单调性
        
        Args:
            n_layers: 分层数 (默认5层)
            long_short: 是否计算多空收益
        """
        self._log(f"分层回测: {factor_col}, {n_layers}层")
        
        layer_returns = {}
        layer_cum_returns = {}
        
        # 按日期分组
        for date in sorted(df['date'].unique()):
            day_data = df[df['date'] == date].copy()
            
            if len(day_data) < n_layers * 10:
                continue
            
            # 按因子值分层
            day_data['layer'] = pd.qcut(day_data[factor_col], n_layers, 
                                        labels=range(1, n_layers + 1),
                                        duplicates='drop')
            
            # 计算每层收益
            for layer in range(1, n_layers + 1):
                layer_data = day_data[day_data['layer'] == layer]
                if len(layer_data) > 0:
                    avg_return = layer_data[return_col].mean()
                    if layer not in layer_returns:
                        layer_returns[layer] = []
                    layer_returns[layer].append({'date': date, 'return': avg_return})
        
        # 转换为Series
        for layer in layer_returns:
            df_layer = pd.DataFrame(layer_returns[layer])
            if not df_layer.empty:
                series = df_layer.set_index('date')['return']
                layer_returns[layer] = series
                layer_cum_returns[layer] = (1 + series).cumprod() - 1
        
        # 计算多空收益 (最高层 - 最低层)
        long_short_return = pd.Series()
        if long_short and 1 in layer_returns and n_layers in layer_returns:
            # 假设因子方向：高层=高因子值
            long_short_return = layer_returns[n_layers] - layer_returns[1]
        
        long_short_cum_return = (1 + long_short_return).cumprod() - 1
        
        # 计算单调性得分 (斯皮尔曼秩相关)
        monotonicity_score = 0
        if len(layer_returns) == n_layers:
            layer_mean_returns = [layer_returns[i].mean() for i in range(1, n_layers + 1)]
            ranks = list(range(1, n_layers + 1))
            monotonicity_score = np.corrcoef(ranks, layer_mean_returns)[0, 1]
        
        result = LayerBacktestResult(
            factor_name=factor_col,
            layer_returns=layer_returns,
            layer_cum_returns=layer_cum_returns,
            long_short_return=long_short_return,
            long_short_cum_return=long_short_cum_return,
            monotonicity_score=monotonicity_score
        )
        
        self._log(f"分层回测完成: 单调性={monotonicity_score:.4f}")
        
        return result
    
    def layer_backtest_summary(self, layer_results: List[LayerBacktestResult]) -> pd.DataFrame:
        """分层回测汇总表"""
        summary = []
        for r in layer_results:
            summary.append({
                '因子': r.factor_name,
                '单调性得分': r.monotonicity_score,
                '分层数': len(r.layer_returns),
                '多空收益': r.long_short_return.mean() if not r.long_short_return.empty else 0,
                '单调性': '单调' if abs(r.monotonicity_score) > 0.5 else '非单调'
            })
        return pd.DataFrame(summary)
    
    # ========== 换手率分析 ==========
    
    def calculate_turnover(self, df: pd.DataFrame, factor_col: str,
                          n_quantiles: int = 10) -> TurnoverAnalysisResult:
        """
        计算因子换手率
        
        衡量因子稳定性，避免高频换仓导致成本高
        """
        self._log(f"换手率分析: {factor_col}")
        
        turnover_rates = []
        
        dates = sorted(df['date'].unique())
        
        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]
            
            prev_data = df[df['date'] == prev_date]
            curr_data = df[df['date'] == curr_date]
            
            # 获取前一期最高因子值的股票
            prev_top = set(prev_data.nlargest(int(len(prev_data) / n_quantiles), factor_col)['symbol'])
            
            # 获取当期最高因子值的股票
            curr_top = set(curr_data.nlargest(int(len(curr_data) / n_quantiles), factor_col)['symbol'])
            
            # 计算换手率
            if len(prev_top) > 0:
                turnover = len(prev_top - curr_top) / len(prev_top)
                turnover_rates.append({'date': curr_date, 'turnover': turnover})
        
        turnover_df = pd.DataFrame(turnover_rates)
        if turnover_df.empty:
            return TurnoverAnalysisResult(factor_name=factor_col, 
                                         turnover_rate=pd.Series(),
                                         mean_turnover=0, stability_score=0)
        
        turnover_series = turnover_df.set_index('date')['turnover']
        mean_turnover = turnover_series.mean()
        
        # 稳定性得分 (1 - 平均换手率，越高越稳定)
        stability_score = 1 - mean_turnover
        
        result = TurnoverAnalysisResult(
            factor_name=factor_col,
            turnover_rate=turnover_series,
            mean_turnover=mean_turnover,
            stability_score=stability_score
        )
        
        self._log(f"换手率分析完成: 平均换手率={mean_turnover:.2%}, 稳定性={stability_score:.4f}")
        
        return result
    
    # ========== 因子相关性分析 ==========
    
    def factor_correlation(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        self._log(f"计算因子相关性: {len(factor_cols)}个因子")
        
        corr_matrix = df[factor_cols].corr()
        
        # 找出高相关性因子对
        high_corr_pairs = []
        for i in range(len(factor_cols)):
            for j in range(i + 1, len(factor_cols)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.7:  # 高相关性阈值
                    high_corr_pairs.append({
                        '因子1': factor_cols[i],
                        '因子2': factor_cols[j],
                        '相关系数': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            self._log(f"发现 {len(high_corr_pairs)} 对高相关性因子")
        
        return corr_matrix
    
    # ========== 综合因子检验 ==========
    
    def comprehensive_factor_test(self, df: pd.DataFrame, 
                                   factor_cols: List[str],
                                   return_col: str = 'label_return') -> Dict[str, pd.DataFrame]:
        """
        综合因子检验
        
        包含IC分析、分层回测、换手率分析
        """
        self._log(f"开始综合因子检验: {len(factor_cols)}个因子")
        
        # 1. IC分析
        ic_results = []
        for factor in factor_cols:
            ic_result = self.calculate_ic(df, factor, return_col)
            ic_results.append(ic_result)
        
        ic_summary = self.ic_analysis_summary(ic_results)
        
        # 2. 分层回测
        layer_results = []
        for factor in factor_cols:
            layer_result = self.layer_backtest(df, factor, return_col)
            layer_results.append(layer_result)
        
        layer_summary = self.layer_backtest_summary(layer_results)
        
        # 3. 换手率分析
        turnover_results = []
        for factor in factor_cols:
            turnover_result = self.calculate_turnover(df, factor)
            turnover_results.append({
                '因子': turnover_result.factor_name,
                '平均换手率': turnover_result.mean_turnover,
                '稳定性得分': turnover_result.stability_score
            })
        
        turnover_summary = pd.DataFrame(turnover_results)
        
        # 4. 因子相关性
        corr_matrix = self.factor_correlation(df, factor_cols)
        
        # 5. 综合评分
        comprehensive = ic_summary.merge(layer_summary, on='因子', how='outer')
        comprehensive = comprehensive.merge(turnover_summary, on='因子', how='outer')
        
        # 计算综合得分
        comprehensive['综合得分'] = (
            comprehensive['IC_IR'].fillna(0).abs() * 0.4 +
            comprehensive['单调性得分'].fillna(0).abs() * 0.3 +
            comprehensive['稳定性得分'].fillna(0) * 0.3
        )
        
        comprehensive = comprehensive.sort_values('综合得分', ascending=False)
        
        self._log("综合因子检验完成")
        
        return {
            'ic_analysis': ic_summary,
            'layer_backtest': layer_summary,
            'turnover_analysis': turnover_summary,
            'correlation_matrix': corr_matrix,
            'comprehensive_score': comprehensive
        }


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Factor Analyzer - 测试")
    print("=" * 80)
    
    # 创建测试数据
    np.random.seed(42)
    n_days = 100
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    symbols = [f'STOCK{i:03d}' for i in range(n_stocks)]
    
    data = []
    for date in dates:
        for symbol in symbols:
            # 生成因子值
            momentum = np.random.randn()
            value = np.random.randn()
            quality = np.random.randn()
            
            # 生成收益 (与因子相关)
            returns = 0.001 + 0.01 * momentum + 0.005 * value + np.random.randn() * 0.02
            
            data.append({
                'date': date,
                'symbol': symbol,
                'momentum': momentum,
                'value': value,
                'quality': quality,
                'label_return': returns
            })
    
    df = pd.DataFrame(data)
    
    print(f"\n测试数据: {len(df)} 行")
    
    # 因子检验
    analyzer = FactorAnalyzer(verbose=True)
    
    results = analyzer.comprehensive_factor_test(
        df,
        factor_cols=['momentum', 'value', 'quality'],
        return_col='label_return'
    )
    
    print("\n" + "=" * 80)
    print("IC分析结果")
    print("=" * 80)
    print(results['ic_analysis'])
    
    print("\n" + "=" * 80)
    print("分层回测结果")
    print("=" * 80)
    print(results['layer_backtest'])
    
    print("\n" + "=" * 80)
    print("换手率分析")
    print("=" * 80)
    print(results['turnover_analysis'])
    
    print("\n" + "=" * 80)
    print("综合评分")
    print("=" * 80)
    print(results['comprehensive_score'])
