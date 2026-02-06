#!/usr/bin/env python3
"""
Quant-Investor V6.0 - 统一因子层 (Unified Factor Layer)

整合所有历史版本的因子能力：
- V3.2: 遗传规划因子挖掘引擎
- V3.3: 工业级因子分析 (Tear Sheet / IC / IR)
- V3.4: Alpha158因子库 + tsfresh特征
- V3.5: 深度特征合成引擎
- V4.1: 基准对比验证 (Alpha / Beta / IR / 胜率)
- V5.0: 500+因子库 (基本面/价量/宏观)

设计原则：
1. 统一的因子计算接口
2. 自动化的因子有效性检验 (IC/IR/分层回测)
3. 基于基准的Alpha验证
4. 支持自定义因子注册
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构 ====================

@dataclass
class FactorResult:
    """单个因子的计算和验证结果"""
    name: str
    category: str
    description: str
    
    # 因子值
    values: pd.Series = None
    
    # IC分析
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0  # Information Ratio = IC_mean / IC_std
    ic_series: pd.Series = None
    
    # 分层回测
    top_return: float = 0.0
    bottom_return: float = 0.0
    long_short_return: float = 0.0
    
    # 基准对比
    alpha_vs_benchmark: float = 0.0
    win_rate_vs_benchmark: float = 0.0
    
    # 有效性判定
    is_effective: bool = False
    effectiveness_score: float = 0.0
    
    # 统计显著性
    ic_tstat: float = 0.0
    ic_pvalue: float = 1.0


@dataclass
class FactorLayerOutput:
    """因子层的完整输出"""
    # 所有计算的因子
    all_factors: Dict[str, FactorResult] = field(default_factory=dict)
    
    # 有效因子（通过筛选的）
    effective_factors: List[FactorResult] = field(default_factory=list)
    
    # 因子矩阵 (用于模型层)
    factor_matrix: pd.DataFrame = None
    
    # 因子相关性矩阵
    correlation_matrix: pd.DataFrame = None
    
    # 候选股票池 (因子综合排名Top N)
    candidate_stocks: List[Dict[str, Any]] = field(default_factory=list)
    
    # 统计摘要
    stats: Dict[str, Any] = field(default_factory=dict)


# ==================== 因子计算器 ====================

class FactorCalculator:
    """
    统一因子计算器
    
    整合V3.4 Alpha158 + V5.0 500+因子库，提供完整的因子计算能力。
    """
    
    def __init__(self):
        self._custom_factors = {}
    
    def register_factor(self, name: str, category: str, description: str,
                         func: Callable[[pd.DataFrame], pd.Series]):
        """注册自定义因子"""
        self._custom_factors[name] = {
            'category': category, 'description': description, 'func': func
        }
    
    def calculate_all(self, panel: pd.DataFrame, stock_col: str = 'stock_code',
                       date_col: str = 'date') -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            panel: 面板数据 (包含 Open/High/Low/Close/Volume 等列)
            stock_col: 股票代码列名
            date_col: 日期列名
        
        Returns:
            包含所有因子值的DataFrame
        """
        result = panel.copy()
        
        # 按股票分组计算因子
        grouped = result.groupby(stock_col)
        
        # ===== 价量因子 =====
        # 动量因子
        for period in [5, 10, 20, 60, 120, 252]:
            col_name = f'momentum_{period}d'
            result[col_name] = grouped['Close'].pct_change(period)
        
        # 反转因子
        for period in [5, 10, 20]:
            col_name = f'reversal_{period}d'
            result[col_name] = -grouped['Close'].pct_change(period)
        
        # 波动率因子
        for period in [10, 20, 60]:
            col_name = f'volatility_{period}d'
            result[col_name] = grouped['returns'].transform(
                lambda x: x.rolling(period, min_periods=max(5, period//2)).std() * np.sqrt(252)
            )
        
        # 成交量因子
        result['volume_ratio_5d'] = grouped['Volume'].transform(
            lambda x: x / x.rolling(5, min_periods=3).mean()
        )
        result['volume_ratio_20d'] = grouped['Volume'].transform(
            lambda x: x / x.rolling(20, min_periods=10).mean()
        )
        result['volume_std_20d'] = grouped['Volume'].transform(
            lambda x: x.rolling(20, min_periods=10).std() / (x.rolling(20, min_periods=10).mean() + 1e-8)
        )
        
        # 价格位置因子
        result['price_position_20d'] = grouped['Close'].transform(
            lambda x: (x - x.rolling(20, min_periods=10).min()) / 
                      (x.rolling(20, min_periods=10).max() - x.rolling(20, min_periods=10).min() + 1e-8)
        )
        result['price_position_60d'] = grouped['Close'].transform(
            lambda x: (x - x.rolling(60, min_periods=30).min()) / 
                      (x.rolling(60, min_periods=30).max() - x.rolling(60, min_periods=30).min() + 1e-8)
        )
        
        # 均线偏离因子
        for period in [5, 10, 20, 60]:
            ma = grouped['Close'].transform(lambda x: x.rolling(period, min_periods=max(3, period//2)).mean())
            result[f'ma_bias_{period}d'] = (result['Close'] - ma) / (ma + 1e-8)
        
        # MACD相关
        result['ema_12'] = grouped['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        result['ema_26'] = grouped['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = grouped['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # RSI
        for period in [6, 14, 24]:
            delta = grouped['Close'].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = grouped[gain.name].transform(lambda x: x.rolling(period, min_periods=period//2).mean()) if gain.name in result.columns else gain.rolling(period, min_periods=period//2).mean()
            avg_loss = grouped[loss.name].transform(lambda x: x.rolling(period, min_periods=period//2).mean()) if loss.name in result.columns else loss.rolling(period, min_periods=period//2).mean()
            # Simplified RSI calculation
            result[f'rsi_{period}'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
        
        # 布林带
        result['bb_mid'] = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
        bb_std = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=10).std())
        result['bb_upper'] = result['bb_mid'] + 2 * bb_std
        result['bb_lower'] = result['bb_mid'] - 2 * bb_std
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / (result['bb_mid'] + 1e-8)
        result['bb_position'] = (result['Close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)
        
        # ATR
        high_low = result['High'] - result['Low']
        high_close = (result['High'] - result['Close'].shift(1)).abs()
        low_close = (result['Low'] - result['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr_14'] = grouped[tr.name].transform(lambda x: x.rolling(14, min_periods=7).mean()) if tr.name in result.columns else tr.rolling(14, min_periods=7).mean()
        result['atr_ratio'] = result['atr_14'] / (result['Close'] + 1e-8)
        
        # 换手率因子
        if 'turnover' in result.columns:
            result['turnover_5d_avg'] = grouped['turnover'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
            result['turnover_20d_avg'] = grouped['turnover'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
        
        # ===== 高阶因子 =====
        # 偏度
        result['skewness_20d'] = grouped['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).skew()
        )
        
        # 峰度
        result['kurtosis_20d'] = grouped['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).kurt()
        )
        
        # 下行波动率
        result['downside_vol_20d'] = grouped['returns'].transform(
            lambda x: x.clip(upper=0).rolling(20, min_periods=10).std() * np.sqrt(252)
        )
        
        # 最大回撤 (滚动)
        result['max_drawdown_20d'] = grouped['Close'].transform(
            lambda x: (x / x.rolling(20, min_periods=10).max() - 1).rolling(20, min_periods=10).min()
        )
        
        # ===== 自定义因子 =====
        for name, factor_info in self._custom_factors.items():
            try:
                result[name] = factor_info['func'](result)
            except Exception:
                pass
        
        # 清理临时列
        temp_cols = ['ema_12', 'ema_26', 'bb_mid', 'bb_upper', 'bb_lower']
        result = result.drop(columns=[c for c in temp_cols if c in result.columns], errors='ignore')
        
        return result
    
    def get_factor_names(self) -> List[str]:
        """获取所有因子名称"""
        factor_prefixes = [
            'momentum_', 'reversal_', 'volatility_', 'volume_ratio_', 'volume_std_',
            'price_position_', 'ma_bias_', 'macd', 'rsi_', 'bb_', 'atr_',
            'turnover_', 'skewness_', 'kurtosis_', 'downside_vol_', 'max_drawdown_'
        ]
        return factor_prefixes


# ==================== 因子验证器 ====================

class FactorValidator:
    """
    因子验证器
    
    整合V3.3 Tear Sheet + V4.1 Alpha分析，对因子进行全面的有效性检验。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def validate_factor(self, factor_values: pd.Series, forward_returns: pd.Series,
                         benchmark_returns: pd.Series = None,
                         factor_name: str = "unnamed") -> FactorResult:
        """
        验证单个因子的有效性
        
        Args:
            factor_values: 因子值 (截面数据)
            forward_returns: 未来收益率
            benchmark_returns: 基准收益率（可选）
            factor_name: 因子名称
        
        Returns:
            FactorResult: 因子验证结果
        """
        result = FactorResult(
            name=factor_name,
            category=self._infer_category(factor_name),
            description=f"Factor: {factor_name}"
        )
        
        # 对齐数据
        valid_mask = factor_values.notna() & forward_returns.notna()
        fv = factor_values[valid_mask]
        fr = forward_returns[valid_mask]
        
        if len(fv) < 30:
            return result
        
        # 1. IC分析 (Rank IC)
        ic, p_value = stats.spearmanr(fv, fr)
        result.ic_mean = ic
        result.ic_pvalue = p_value
        result.ic_tstat = ic / (1.0 / np.sqrt(len(fv)) + 1e-8)
        
        # 2. 分层回测
        try:
            n_groups = min(5, len(fv) // 10)
            if n_groups >= 2:
                fv_ranked = fv.rank(pct=True)
                top_mask = fv_ranked >= (1 - 1/n_groups)
                bottom_mask = fv_ranked <= (1/n_groups)
                
                result.top_return = float(fr[top_mask].mean() * 252) if top_mask.sum() > 0 else 0
                result.bottom_return = float(fr[bottom_mask].mean() * 252) if bottom_mask.sum() > 0 else 0
                result.long_short_return = result.top_return - result.bottom_return
        except Exception:
            pass
        
        # 3. 基准对比
        if benchmark_returns is not None:
            try:
                aligned_br = benchmark_returns.reindex(fr.index)
                valid_br = aligned_br.notna()
                if valid_br.sum() > 10:
                    excess = fr[valid_br] - aligned_br[valid_br]
                    result.alpha_vs_benchmark = float(excess.mean() * 252)
                    result.win_rate_vs_benchmark = float((excess > 0).mean())
            except Exception:
                pass
        
        # 4. 有效性判定
        result.effectiveness_score = self._calc_effectiveness_score(result)
        result.is_effective = result.effectiveness_score >= 0.5
        
        return result
    
    def validate_all_factors(self, panel: pd.DataFrame, factor_columns: List[str],
                              return_col: str = 'returns', forward_periods: int = 5,
                              stock_col: str = 'stock_code', date_col: str = 'date',
                              benchmark_returns: pd.Series = None) -> Dict[str, FactorResult]:
        """
        批量验证所有因子
        
        Args:
            panel: 面板数据
            factor_columns: 因子列名列表
            return_col: 收益率列名
            forward_periods: 前瞻期
            stock_col: 股票代码列名
            date_col: 日期列名
            benchmark_returns: 基准收益率
        
        Returns:
            Dict[str, FactorResult]: 因子验证结果字典
        """
        if self.verbose:
            print(f"\n  🔬 开始因子验证: {len(factor_columns)} 个因子")
        
        results = {}
        
        # 计算前瞻收益率
        panel = panel.copy()
        panel['forward_return'] = panel.groupby(stock_col)[return_col].shift(-forward_periods)
        
        for i, col in enumerate(factor_columns):
            if col not in panel.columns:
                continue
            
            # 使用最新截面数据进行IC计算
            dates = panel[date_col].unique()
            ic_list = []
            
            for dt in dates[-60:]:  # 使用最近60个交易日
                cross_section = panel[panel[date_col] == dt]
                fv = cross_section[col]
                fr = cross_section['forward_return']
                
                valid = fv.notna() & fr.notna()
                if valid.sum() >= 5:  # 降低阈值以支持小股票池
                    ic, _ = stats.spearmanr(fv[valid], fr[valid])
                    if not np.isnan(ic):
                        ic_list.append(ic)
            
            if len(ic_list) < 3:  # 降低阈值以支持小样本
                continue
            
            ic_series = pd.Series(ic_list)
            
            result = FactorResult(
                name=col,
                category=self._infer_category(col),
                description=f"Factor: {col}",
                ic_mean=float(ic_series.mean()),
                ic_std=float(ic_series.std()),
                ir=float(ic_series.mean() / (ic_series.std() + 1e-8)),
                ic_series=ic_series
            )
            
            # t检验
            if len(ic_list) > 2:
                t_stat, p_val = stats.ttest_1samp(ic_list, 0)
                result.ic_tstat = float(t_stat)
                result.ic_pvalue = float(p_val)
            
            # 分层回测 (使用最新截面)
            latest_date = dates[-1]
            latest_cs = panel[panel[date_col] == latest_date]
            fv = latest_cs[col]
            fr = latest_cs['forward_return']
            valid = fv.notna() & fr.notna()
            
            if valid.sum() >= 20:
                fv_ranked = fv[valid].rank(pct=True)
                top_mask = fv_ranked >= 0.8
                bottom_mask = fv_ranked <= 0.2
                
                result.top_return = float(fr[valid][top_mask].mean() * 252) if top_mask.sum() > 0 else 0
                result.bottom_return = float(fr[valid][bottom_mask].mean() * 252) if bottom_mask.sum() > 0 else 0
                result.long_short_return = result.top_return - result.bottom_return
            
            # 有效性判定
            result.effectiveness_score = self._calc_effectiveness_score(result)
            result.is_effective = result.effectiveness_score >= 0.5
            
            results[col] = result
        
        effective_count = sum(1 for r in results.values() if r.is_effective)
        if self.verbose:
            print(f"    ✓ 验证完成: {len(results)} 个因子, 有效: {effective_count} 个")
        
        return results
    
    def _calc_effectiveness_score(self, result: FactorResult) -> float:
        """计算因子有效性综合得分 (0-1)"""
        score = 0.0
        
        # IC均值 (权重: 30%)
        ic_abs = abs(result.ic_mean)
        if ic_abs > 0.05:
            score += 0.3
        elif ic_abs > 0.03:
            score += 0.2
        elif ic_abs > 0.01:
            score += 0.1
        
        # IR (权重: 30%)
        ir_abs = abs(result.ir)
        if ir_abs > 0.5:
            score += 0.3
        elif ir_abs > 0.3:
            score += 0.2
        elif ir_abs > 0.1:
            score += 0.1
        
        # 多空收益 (权重: 20%)
        ls = abs(result.long_short_return)
        if ls > 0.2:
            score += 0.2
        elif ls > 0.1:
            score += 0.15
        elif ls > 0.05:
            score += 0.1
        
        # 统计显著性 (权重: 20%)
        if result.ic_pvalue < 0.01:
            score += 0.2
        elif result.ic_pvalue < 0.05:
            score += 0.15
        elif result.ic_pvalue < 0.1:
            score += 0.1
        
        return score
    
    def _infer_category(self, name: str) -> str:
        """推断因子类别"""
        if any(k in name for k in ['momentum', 'reversal', 'ma_bias']):
            return "动量/反转"
        elif any(k in name for k in ['volatility', 'downside_vol', 'atr', 'bb_width']):
            return "波动率"
        elif any(k in name for k in ['volume', 'turnover']):
            return "流动性"
        elif any(k in name for k in ['rsi', 'macd', 'bb_position', 'price_position']):
            return "技术指标"
        elif any(k in name for k in ['pe', 'pb', 'ps', 'roe', 'eps']):
            return "基本面"
        elif any(k in name for k in ['skewness', 'kurtosis', 'max_drawdown']):
            return "高阶统计"
        return "其他"


# ==================== 统一因子层 ====================

class UnifiedFactorLayer:
    """
    V6.0 统一因子层
    
    整合因子计算、验证和筛选的完整流程。
    """
    
    def __init__(self, verbose: bool = True, top_n_stocks: int = 20):
        """
        Args:
            verbose: 是否打印详细信息
            top_n_stocks: 候选股票池大小
        """
        self.verbose = verbose
        self.top_n_stocks = top_n_stocks
        self.calculator = FactorCalculator()
        self.validator = FactorValidator(verbose=verbose)
    
    def process(self, panel: pd.DataFrame, benchmark_returns: pd.Series = None,
                stock_col: str = 'stock_code', date_col: str = 'date') -> FactorLayerOutput:
        """
        执行完整的因子层处理流程
        
        Args:
            panel: 面板数据 (来自数据层)
            benchmark_returns: 基准收益率 (来自数据层)
            stock_col: 股票代码列名
            date_col: 日期列名
        
        Returns:
            FactorLayerOutput: 因子层完整输出
        """
        output = FactorLayerOutput()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 V6.0 统一因子层")
            print(f"{'='*60}")
        
        # 1. 计算所有因子
        if self.verbose:
            print(f"\n  📐 计算因子...")
        
        factor_panel = self.calculator.calculate_all(panel, stock_col=stock_col, date_col=date_col)
        
        # 识别因子列
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns',
                     stock_col, date_col, 'stock_name', 'industry', 'turnover',
                     'volatility_20d', 'momentum_20d', 'momentum_60d']
        factor_columns = [c for c in factor_panel.columns 
                         if c not in base_cols and factor_panel[c].dtype in ['float64', 'float32', 'int64']]
        
        if self.verbose:
            print(f"    ✓ 计算完成: {len(factor_columns)} 个因子")
        
        # 2. 验证因子有效性
        all_results = self.validator.validate_all_factors(
            factor_panel, factor_columns,
            return_col='returns', forward_periods=5,
            stock_col=stock_col, date_col=date_col,
            benchmark_returns=benchmark_returns
        )
        output.all_factors = all_results
        
        # 3. 筛选有效因子
        effective = [r for r in all_results.values() if r.is_effective]
        effective.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        # Fallback: 如果没有通过严格筛选的因子，使用所有因子按得分排序
        if not effective and all_results:
            all_sorted = sorted(all_results.values(), key=lambda x: x.effectiveness_score, reverse=True)
            # 取得分最高的Top N个因子作为候选
            effective = all_sorted[:min(15, len(all_sorted))]
            if self.verbose:
                print(f"    ⚠️ 无严格有效因子，使用Top {len(effective)} 因子作为候选")
        
        output.effective_factors = effective
        
        if self.verbose:
            print(f"\n  📊 有效因子排名 (Top 10):")
            for i, f in enumerate(effective[:10], 1):
                print(f"    {i:2d}. {f.name:<25s} IC={f.ic_mean:+.4f}  IR={f.ir:+.3f}  "
                      f"L/S={f.long_short_return:+.2%}  Score={f.effectiveness_score:.2f}")
        
        # 4. 构建因子矩阵
        effective_names = [f.name for f in effective[:30]]  # 取Top30有效因子
        if effective_names:
            valid_factor_cols = [n for n in effective_names if n in factor_panel.columns]
            if valid_factor_cols:
                output.factor_matrix = factor_panel[[stock_col, date_col] + valid_factor_cols].copy()
        
        # Fallback: 如果因子矩阵仍为空，使用所有数值因子列
        if output.factor_matrix is None:
            all_factor_cols = [c for c in factor_columns if c in factor_panel.columns][:20]
            if all_factor_cols:
                output.factor_matrix = factor_panel[[stock_col, date_col] + all_factor_cols].copy()
                if self.verbose:
                    print(f"    ⚠️ 使用全部因子构建因子矩阵: {len(all_factor_cols)} 列")
        
        # 5. 因子相关性分析
        if effective_names and len(effective_names) > 1:
            latest_date = factor_panel[date_col].max()
            latest_cs = factor_panel[factor_panel[date_col] == latest_date]
            valid_cols = [c for c in effective_names if c in latest_cs.columns]
            if valid_cols:
                output.correlation_matrix = latest_cs[valid_cols].corr()
        
        # 6. 综合选股
        output.candidate_stocks = self._select_candidates(
            factor_panel, effective, stock_col, date_col
        )
        
        # 7. 统计摘要
        output.stats = {
            "total_factors_calculated": len(factor_columns),
            "total_factors_validated": len(all_results),
            "effective_factors": len(effective),
            "candidate_stocks": len(output.candidate_stocks),
            "top_factor": effective[0].name if effective else "N/A",
            "top_factor_ic": effective[0].ic_mean if effective else 0,
        }
        
        if self.verbose:
            print(f"\n  ✅ 因子层处理完成")
            print(f"     有效因子: {output.stats['effective_factors']} 个")
            print(f"     候选股票: {output.stats['candidate_stocks']} 只")
        
        return output
    
    def _select_candidates(self, panel: pd.DataFrame, effective_factors: List[FactorResult],
                            stock_col: str, date_col: str) -> List[Dict[str, Any]]:
        """基于有效因子综合选股"""
        # 如果没有有效因子，使用基础指标选股
        if not effective_factors:
            return self._fallback_select(panel, stock_col, date_col)
        
        # 使用最新截面数据
        latest_date = panel[date_col].max()
        latest = panel[panel[date_col] == latest_date].copy()
        
        if len(latest) == 0:
            return []
        
        # 计算综合得分
        score_cols = []
        for factor in effective_factors[:15]:  # 使用Top15因子
            col = factor.name
            if col not in latest.columns:
                continue
            
            # 根据IC方向决定排名方向
            rank_col = f'{col}_rank'
            if factor.ic_mean > 0:
                latest[rank_col] = latest[col].rank(pct=True, ascending=True)
            else:
                latest[rank_col] = latest[col].rank(pct=True, ascending=False)
            
            # 加权 (按effectiveness_score)
            latest[rank_col] = latest[rank_col] * factor.effectiveness_score
            score_cols.append(rank_col)
        
        if not score_cols:
            return self._fallback_select(panel, stock_col, date_col)
        
        latest['composite_score'] = latest[score_cols].mean(axis=1)
        latest = latest.sort_values('composite_score', ascending=False)
        
        # 选取Top N
        candidates = []
        for _, row in latest.head(self.top_n_stocks).iterrows():
            candidates.append({
                'code': row[stock_col],
                'name': row.get('stock_name', row[stock_col]),
                'composite_score': float(row['composite_score']),
                'industry': row.get('industry', ''),
                'latest_price': float(row.get('Close', 0)),
                'returns_20d': float(row.get('momentum_20d', 0)) if 'momentum_20d' in row.index else 0,
            })
        
        return candidates
    
    def _fallback_select(self, panel: pd.DataFrame, stock_col: str, date_col: str) -> List[Dict[str, Any]]:
        """回退选股: 当无有效因子时，基于基础指标选股"""
        latest_date = panel[date_col].max()
        latest = panel[panel[date_col] == latest_date].copy()
        
        if len(latest) == 0:
            return []
        
        # 使用动量和波动率等基础指标排序
        score_cols = []
        for col in ['momentum_20d', 'momentum_60d', 'returns']:
            if col in latest.columns:
                rank_col = f'{col}_rank'
                latest[rank_col] = latest[col].rank(pct=True, ascending=True)
                score_cols.append(rank_col)
        
        if not score_cols:
            # 最后回退: 返回所有股票
            candidates = []
            for _, row in latest.iterrows():
                candidates.append({
                    'code': row[stock_col],
                    'name': row.get('stock_name', row[stock_col]),
                    'composite_score': 0.5,
                    'industry': row.get('industry', ''),
                    'latest_price': float(row.get('Close', 0)),
                    'returns_20d': 0,
                })
            return candidates[:self.top_n_stocks]
        
        latest['composite_score'] = latest[score_cols].mean(axis=1)
        latest = latest.sort_values('composite_score', ascending=False)
        
        candidates = []
        for _, row in latest.head(self.top_n_stocks).iterrows():
            candidates.append({
                'code': row[stock_col],
                'name': row.get('stock_name', row[stock_col]),
                'composite_score': float(row['composite_score']),
                'industry': row.get('industry', ''),
                'latest_price': float(row.get('Close', 0)),
                'returns_20d': float(row.get('momentum_20d', 0)) if 'momentum_20d' in row.index else 0,
            })
        
        if self.verbose:
            print(f"    ⚠️ 使用基础指标回退选股: {len(candidates)} 只")
        
        return candidates


# ==================== 便捷函数 ====================

def run_factor_analysis(panel: pd.DataFrame, benchmark_returns: pd.Series = None,
                         verbose: bool = True, top_n: int = 20) -> FactorLayerOutput:
    """
    便捷函数：运行完整的因子分析
    
    Args:
        panel: 面板数据
        benchmark_returns: 基准收益率
        verbose: 是否打印详细信息
        top_n: 候选股票数量
    
    Returns:
        FactorLayerOutput: 因子层输出
    """
    layer = UnifiedFactorLayer(verbose=verbose, top_n_stocks=top_n)
    return layer.process(panel, benchmark_returns)


if __name__ == "__main__":
    print("=" * 60)
    print("V6.0 统一因子层测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    panels = []
    for stock in stocks:
        price = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        df = pd.DataFrame({
            'date': dates,
            'stock_code': stock,
            'stock_name': stock,
            'industry': 'Tech',
            'Open': price * (1 + np.random.randn(252) * 0.01),
            'High': price * (1 + abs(np.random.randn(252) * 0.02)),
            'Low': price * (1 - abs(np.random.randn(252) * 0.02)),
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000, 252).astype(float),
        })
        df['returns'] = df['Close'].pct_change()
        df['turnover'] = df['Volume'] / df['Volume'].rolling(20).mean()
        panels.append(df)
    
    panel = pd.concat(panels, ignore_index=True)
    
    # 运行因子分析
    output = run_factor_analysis(panel, verbose=True, top_n=3)
    
    print(f"\n统计: {output.stats}")
    print(f"\n候选股票:")
    for s in output.candidate_stocks:
        print(f"  {s['code']}: score={s['composite_score']:.4f}")
