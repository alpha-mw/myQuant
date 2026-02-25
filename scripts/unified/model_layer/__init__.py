#!/usr/bin/env python3
"""
统一模型层 - 简化版
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelOutput:
    """模型层输出"""
    ranked_stocks: List[Dict] = field(default_factory=list)
    predictions: Optional[pd.DataFrame] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class UnifiedModelLayer:
    """统一模型层"""
    
    def __init__(self, verbose: bool = True, top_n_stocks: int = 20):
        self.verbose = verbose
        self.top_n_stocks = top_n_stocks
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [ModelLayer] {msg}")
    
    def predict(self, factor_matrix: Optional[pd.DataFrame],
                panel: Optional[pd.DataFrame],
                candidate_stocks: List[str]) -> ModelOutput:
        """预测股票排名"""
        output = ModelOutput()

        if factor_matrix is None or factor_matrix.empty:
            self._log("无因子数据，返回空结果")
            return output

        self._log("训练模型...")

        # 获取因子列
        stock_col = None
        for col in ['stock_code', 'Stock']:
            if col in factor_matrix.columns:
                stock_col = col
                break

        if stock_col is None:
            self._log("找不到股票代码列")
            return output

        exclude_cols = [stock_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'date', 'Date']
        factor_cols = [c for c in factor_matrix.columns if c not in exclude_cols]

        if not factor_cols:
            self._log("无有效因子")
            return output

        # 计算每只股票的因子均值作为简单预测
        latest_data = factor_matrix.groupby(stock_col)[factor_cols].last()

        # 简单评分：动量因子加权
        score = pd.Series(0.0, index=latest_data.index)

        if 'momentum_20d' in latest_data.columns:
            score += latest_data['momentum_20d'] * 0.4
        if 'momentum_60d' in latest_data.columns:
            score += latest_data['momentum_60d'] * 0.3
        if 'ma_bias_20d' in latest_data.columns:
            score -= latest_data['ma_bias_20d'] * 0.2  # 均值回归

        # 排序
        ranked = score.sort_values(ascending=False)

        # 生成输出
        for i, (stock, s) in enumerate(ranked.head(self.top_n_stocks).items()):
            output.ranked_stocks.append({
                'code': stock,
                'name': stock,
                'composite_score': float(s),
                'rank': i + 1
            })

        output.stats['models_trained'] = 1
        self._log(f"生成 {len(output.ranked_stocks)} 只股票排名")

        return output
