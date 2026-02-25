#!/usr/bin/env python3
"""
统一因子层 - 简化版
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FactorOutput:
    """因子层输出"""
    effective_factors: List[str] = field(default_factory=list)
    candidate_stocks: List[str] = field(default_factory=list)
    factor_matrix: Optional[pd.DataFrame] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class UnifiedFactorLayer:
    """统一因子层"""
    
    def __init__(self, verbose: bool = True, top_n_stocks: int = 20):
        self.verbose = verbose
        self.top_n_stocks = top_n_stocks
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [FactorLayer] {msg}")
    
    def process(self, panel_data: Optional[pd.DataFrame], 
                benchmark_returns: Optional[pd.Series]) -> FactorOutput:
        """处理因子"""
        output = FactorOutput()
        
        if panel_data is None or panel_data.empty:
            self._log("无数据，返回空结果")
            return output
        
        self._log("计算基础因子...")
        
        # 计算简单因子
        factor_data = self._calculate_factors(panel_data)
        
        if factor_data is not None:
            output.factor_matrix = factor_data
            
            # 确定列名
            stock_col = None
            for col in ['stock_code', 'Stock']:
                if col in factor_data.columns:
                    stock_col = col
                    break
            
            date_col = None
            for col in ['date', 'Date']:
                if col in factor_data.columns:
                    date_col = col
                    break
            
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            if stock_col:
                exclude_cols.append(stock_col)
            if date_col:
                exclude_cols.append(date_col)
                
            output.effective_factors = [c for c in factor_data.columns if c not in exclude_cols]
            
            # 选择候选股票
            if stock_col:
                unique_stocks = factor_data[stock_col].unique()
                output.candidate_stocks = list(unique_stocks)[:self.top_n_stocks]
            
            self._log(f"计算了 {len(output.effective_factors)} 个因子")
            self._log(f"筛选出 {len(output.candidate_stocks)} 只候选股票")
        
        return output
    
    def _calculate_factors(self, panel_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算基础因子"""
        if 'Close' not in panel_data.columns:
            return None
        
        result = panel_data.copy()
        
        # 确定股票代码列名
        stock_col = 'stock_code' if 'stock_code' in result.columns else 'Stock'
        date_col = 'date' if 'date' in result.columns else 'Date'
        
        # 按股票分组计算
        def calc_stock_factors(group):
            group = group.sort_values(date_col)
            close = group['Close']
            
            # 价格动量因子
            group['momentum_5d'] = close.pct_change(5)
            group['momentum_20d'] = close.pct_change(20)
            group['momentum_60d'] = close.pct_change(60)
            
            # 波动率因子
            group['volatility_20d'] = close.pct_change().rolling(20).std() * np.sqrt(252)
            
            # 均值回归因子
            group['ma_20d'] = close.rolling(20).mean()
            group['ma_bias_20d'] = (close - group['ma_20d']) / group['ma_20d']
            
            # 成交量因子
            vol_col = 'Volume' if 'Volume' in group.columns else 'volume'
            if vol_col in group.columns:
                group['volume_ratio'] = group[vol_col] / group[vol_col].rolling(20).mean()
            
            return group
        
        # 保存原始股票代码列
        df = result.copy()
        result = result.groupby(stock_col, group_keys=False).apply(calc_stock_factors)
        # 确保股票代码列保留
        if stock_col not in result.columns:
            result[stock_col] = df[stock_col].values
        
        # 填充缺失值
        exclude_cols = [stock_col, date_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'volume', 'Stock', 'Date']
        factor_cols = [c for c in result.columns if c not in exclude_cols]
        result[factor_cols] = result[factor_cols].fillna(0)
        
        return result
