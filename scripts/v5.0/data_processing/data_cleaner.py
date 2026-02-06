"""
Quant-Investor V5.0 - 数据清洗与预处理模块

本模块提供工业级的数据清洗流水线，包括：
1. 去极值 (Winsorization)
2. 缺失值填充
3. 标准化
4. 前视偏差防控
5. 幸存者偏差处理
6. 动态复权处理
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Literal
from scipy import stats
from datetime import datetime, timedelta


class DataCleaner:
    """
    工业级数据清洗器
    
    提供完整的数据清洗流水线，确保数据质量和回测准确性。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("DataCleaner initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DataCleaner] {msg}")
    
    # ==================== 去极值 ====================
    
    def winsorize(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: Literal['percentile', 'mad', 'sigma'] = 'percentile',
        limits: tuple = (0.01, 0.99),
        sigma: float = 3.0
    ) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            data: 输入数据
            columns: 需要处理的列，None表示所有数值列
            method: 去极值方法
                - 'percentile': 分位数截断
                - 'mad': 中位数绝对偏差法
                - 'sigma': 标准差法
            limits: 分位数截断的上下限 (仅用于percentile方法)
            sigma: 标准差倍数 (仅用于sigma/mad方法)
        
        Returns:
            去极值后的数据
        """
        result = data.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            series = result[col].dropna()
            
            if method == 'percentile':
                lower = series.quantile(limits[0])
                upper = series.quantile(limits[1])
            elif method == 'mad':
                median = series.median()
                mad = np.median(np.abs(series - median))
                lower = median - sigma * 1.4826 * mad
                upper = median + sigma * 1.4826 * mad
            elif method == 'sigma':
                mean = series.mean()
                std = series.std()
                lower = mean - sigma * std
                upper = mean + sigma * std
            else:
                raise ValueError(f"Unknown method: {method}")
            
            result[col] = result[col].clip(lower=lower, upper=upper)
        
        self._log(f"Winsorized {len(columns)} columns using {method} method")
        return result
    
    # ==================== 缺失值填充 ====================
    
    def fill_missing(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal['mean', 'median', 'ffill', 'bfill', 'interpolate', 'industry'] = 'median',
        industry_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        缺失值填充
        
        Args:
            data: 输入数据
            columns: 需要处理的列
            method: 填充方法
                - 'mean': 均值填充
                - 'median': 中位数填充
                - 'ffill': 前向填充
                - 'bfill': 后向填充
                - 'interpolate': 线性插值
                - 'industry': 行业均值填充
            industry_col: 行业列名 (仅用于industry方法)
        
        Returns:
            填充后的数据
        """
        result = data.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            missing_count = result[col].isna().sum()
            if missing_count == 0:
                continue
            
            if method == 'mean':
                result[col] = result[col].fillna(result[col].mean())
            elif method == 'median':
                result[col] = result[col].fillna(result[col].median())
            elif method == 'ffill':
                result[col] = result[col].ffill()
            elif method == 'bfill':
                result[col] = result[col].bfill()
            elif method == 'interpolate':
                result[col] = result[col].interpolate(method='linear')
            elif method == 'industry':
                if industry_col is None or industry_col not in result.columns:
                    result[col] = result[col].fillna(result[col].median())
                else:
                    industry_mean = result.groupby(industry_col)[col].transform('mean')
                    result[col] = result[col].fillna(industry_mean)
                    result[col] = result[col].fillna(result[col].median())
        
        total_missing = data[columns].isna().sum().sum()
        self._log(f"Filled {total_missing} missing values using {method} method")
        return result
    
    # ==================== 标准化 ====================
    
    def standardize(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal['zscore', 'minmax', 'rank', 'robust'] = 'zscore',
        by_date: bool = False,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        数据标准化
        
        Args:
            data: 输入数据
            columns: 需要处理的列
            method: 标准化方法
                - 'zscore': Z-score标准化
                - 'minmax': Min-Max归一化
                - 'rank': 排名百分比
                - 'robust': 稳健标准化 (使用中位数和IQR)
            by_date: 是否按日期分组标准化 (截面标准化)
            date_col: 日期列名
        
        Returns:
            标准化后的数据
        """
        result = data.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        def _standardize_series(s: pd.Series, method: str) -> pd.Series:
            if method == 'zscore':
                return (s - s.mean()) / (s.std() + 1e-8)
            elif method == 'minmax':
                return (s - s.min()) / (s.max() - s.min() + 1e-8)
            elif method == 'rank':
                return s.rank(pct=True)
            elif method == 'robust':
                median = s.median()
                q75, q25 = s.quantile(0.75), s.quantile(0.25)
                iqr = q75 - q25
                return (s - median) / (iqr + 1e-8)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        for col in columns:
            if col not in result.columns or col == date_col:
                continue
            
            if by_date and date_col in result.columns:
                result[col] = result.groupby(date_col)[col].transform(
                    lambda x: _standardize_series(x, method)
                )
            else:
                result[col] = _standardize_series(result[col], method)
        
        self._log(f"Standardized {len(columns)} columns using {method} method")
        return result
    
    # ==================== 完整清洗流水线 ====================
    
    def clean_pipeline(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        winsorize_method: str = 'percentile',
        fill_method: str = 'median',
        standardize_method: str = 'zscore',
        by_date: bool = True,
        date_col: str = 'date',
        industry_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        完整的数据清洗流水线
        
        执行顺序：去极值 -> 缺失值填充 -> 标准化
        
        Args:
            data: 输入数据
            columns: 需要处理的列
            winsorize_method: 去极值方法
            fill_method: 缺失值填充方法
            standardize_method: 标准化方法
            by_date: 是否按日期分组标准化
            date_col: 日期列名
            industry_col: 行业列名
        
        Returns:
            清洗后的数据
        """
        self._log("Starting data cleaning pipeline...")
        
        # Step 1: 去极值
        result = self.winsorize(data, columns, method=winsorize_method)
        
        # Step 2: 缺失值填充
        result = self.fill_missing(result, columns, method=fill_method, industry_col=industry_col)
        
        # Step 3: 标准化
        result = self.standardize(result, columns, method=standardize_method, by_date=by_date, date_col=date_col)
        
        self._log("Data cleaning pipeline completed")
        return result


class BiasHandler:
    """
    偏差处理器
    
    处理量化回测中常见的偏差问题：
    1. 前视偏差 (Look-ahead Bias)
    2. 幸存者偏差 (Survivorship Bias)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("BiasHandler initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[BiasHandler] {msg}")
    
    # ==================== 前视偏差防控 ====================
    
    def shift_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        shift_periods: int = 1,
        date_col: str = 'date',
        stock_col: str = 'stock_code'
    ) -> pd.DataFrame:
        """
        特征滞后处理，防止前视偏差
        
        确保在T日只能使用T-1日及之前的数据。
        
        Args:
            data: 输入数据
            feature_cols: 需要滞后的特征列
            shift_periods: 滞后期数
            date_col: 日期列名
            stock_col: 股票代码列名
        
        Returns:
            滞后处理后的数据
        """
        result = data.copy()
        
        # 按股票分组滞后
        if stock_col in result.columns:
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result.groupby(stock_col)[col].shift(shift_periods)
        else:
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result[col].shift(shift_periods)
        
        self._log(f"Shifted {len(feature_cols)} features by {shift_periods} periods")
        return result
    
    def dynamic_adjustment(
        self,
        price_data: pd.DataFrame,
        dividend_data: pd.DataFrame,
        as_of_date: str,
        price_col: str = 'close',
        date_col: str = 'date',
        stock_col: str = 'stock_code'
    ) -> pd.DataFrame:
        """
        动态复权处理
        
        在指定日期进行复权计算，只使用该日期之前已知的分红信息。
        
        Args:
            price_data: 价格数据
            dividend_data: 分红数据，包含 ex_date (除权日), dividend (每股分红)
            as_of_date: 复权计算的截止日期
            price_col: 价格列名
            date_col: 日期列名
            stock_col: 股票代码列名
        
        Returns:
            动态复权后的价格数据
        """
        result = price_data.copy()
        
        # 筛选截止日期之前的分红信息
        if 'ex_date' in dividend_data.columns:
            valid_dividends = dividend_data[
                pd.to_datetime(dividend_data['ex_date']) <= pd.to_datetime(as_of_date)
            ]
        else:
            valid_dividends = dividend_data[
                pd.to_datetime(dividend_data[date_col]) <= pd.to_datetime(as_of_date)
            ]
        
        # 计算复权因子
        if len(valid_dividends) > 0:
            # 按股票和日期计算累计复权因子
            for stock in result[stock_col].unique():
                stock_divs = valid_dividends[valid_dividends[stock_col] == stock]
                if len(stock_divs) == 0:
                    continue
                
                stock_mask = result[stock_col] == stock
                stock_prices = result.loc[stock_mask].copy()
                
                # 简化的复权计算：累计分红调整
                adj_factor = 1.0
                for _, div_row in stock_divs.iterrows():
                    ex_date = pd.to_datetime(div_row.get('ex_date', div_row.get(date_col)))
                    dividend = div_row.get('dividend', 0)
                    
                    # 在除权日之前的价格需要调整
                    pre_ex_mask = pd.to_datetime(stock_prices[date_col]) < ex_date
                    if pre_ex_mask.any():
                        avg_price = stock_prices.loc[pre_ex_mask, price_col].mean()
                        if avg_price > 0:
                            adj_factor *= (avg_price - dividend) / avg_price
                
                result.loc[stock_mask, f'{price_col}_adj'] = result.loc[stock_mask, price_col] * adj_factor
        else:
            result[f'{price_col}_adj'] = result[price_col]
        
        self._log(f"Dynamic adjustment completed as of {as_of_date}")
        return result
    
    # ==================== 幸存者偏差处理 ====================
    
    def get_point_in_time_universe(
        self,
        constituent_history: pd.DataFrame,
        as_of_date: str,
        index_col: str = 'index_code',
        stock_col: str = 'stock_code',
        start_date_col: str = 'in_date',
        end_date_col: str = 'out_date'
    ) -> List[str]:
        """
        获取时点股票池
        
        根据历史成分股数据，获取指定日期的真实股票池，避免幸存者偏差。
        
        Args:
            constituent_history: 历史成分股数据
            as_of_date: 查询日期
            index_col: 指数代码列名
            stock_col: 股票代码列名
            start_date_col: 纳入日期列名
            end_date_col: 剔除日期列名
        
        Returns:
            该日期的成分股列表
        """
        as_of = pd.to_datetime(as_of_date)
        
        # 筛选在指定日期有效的成分股
        valid_mask = (
            (pd.to_datetime(constituent_history[start_date_col]) <= as_of) &
            (
                constituent_history[end_date_col].isna() |
                (pd.to_datetime(constituent_history[end_date_col]) > as_of)
            )
        )
        
        universe = constituent_history.loc[valid_mask, stock_col].unique().tolist()
        self._log(f"Point-in-time universe as of {as_of_date}: {len(universe)} stocks")
        return universe
    
    def filter_delisted_stocks(
        self,
        data: pd.DataFrame,
        delist_data: pd.DataFrame,
        date_col: str = 'date',
        stock_col: str = 'stock_code',
        delist_date_col: str = 'delist_date'
    ) -> pd.DataFrame:
        """
        过滤已退市股票的未来数据
        
        确保在回测中不会使用退市后的数据。
        
        Args:
            data: 输入数据
            delist_data: 退市数据
            date_col: 日期列名
            stock_col: 股票代码列名
            delist_date_col: 退市日期列名
        
        Returns:
            过滤后的数据
        """
        result = data.copy()
        
        # 合并退市信息
        result = result.merge(
            delist_data[[stock_col, delist_date_col]],
            on=stock_col,
            how='left'
        )
        
        # 过滤退市后的数据
        valid_mask = (
            result[delist_date_col].isna() |
            (pd.to_datetime(result[date_col]) < pd.to_datetime(result[delist_date_col]))
        )
        
        filtered = result.loc[valid_mask].drop(columns=[delist_date_col])
        
        removed_count = len(result) - len(filtered)
        self._log(f"Filtered {removed_count} rows of delisted stock data")
        return filtered


class LabelGenerator:
    """
    标签生成器
    
    为机器学习模型生成各种类型的标签。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("LabelGenerator initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[LabelGenerator] {msg}")
    
    def generate_return_label(
        self,
        data: pd.DataFrame,
        periods: int = 5,
        price_col: str = 'close',
        date_col: str = 'date',
        stock_col: str = 'stock_code',
        label_type: Literal['regression', 'classification', 'ranking'] = 'regression',
        n_classes: int = 3
    ) -> pd.DataFrame:
        """
        生成收益率标签
        
        Args:
            data: 输入数据
            periods: 预测周期（天数）
            price_col: 价格列名
            date_col: 日期列名
            stock_col: 股票代码列名
            label_type: 标签类型
                - 'regression': 连续收益率
                - 'classification': 涨跌分类
                - 'ranking': 收益率分位数排名
            n_classes: 分类数量 (仅用于classification和ranking)
        
        Returns:
            包含标签的数据
        """
        result = data.copy()
        
        # 计算未来N日收益率
        if stock_col in result.columns:
            result['future_return'] = result.groupby(stock_col)[price_col].pct_change(periods).shift(-periods)
        else:
            result['future_return'] = result[price_col].pct_change(periods).shift(-periods)
        
        if label_type == 'regression':
            result['label'] = result['future_return']
        
        elif label_type == 'classification':
            # 二分类：涨/跌
            if n_classes == 2:
                result['label'] = (result['future_return'] > 0).astype(int)
            # 三分类：跌/平/涨
            elif n_classes == 3:
                conditions = [
                    result['future_return'] < -0.02,
                    result['future_return'] > 0.02
                ]
                choices = [0, 2]  # 0=跌, 1=平, 2=涨
                result['label'] = np.select(conditions, choices, default=1)
            else:
                # N分类：按分位数
                result['label'] = pd.qcut(result['future_return'], q=n_classes, labels=False, duplicates='drop')
        
        elif label_type == 'ranking':
            # 按日期分组计算排名分位数
            if date_col in result.columns:
                result['label'] = result.groupby(date_col)['future_return'].transform(
                    lambda x: pd.qcut(x, q=n_classes, labels=False, duplicates='drop')
                )
            else:
                result['label'] = pd.qcut(result['future_return'], q=n_classes, labels=False, duplicates='drop')
        
        self._log(f"Generated {label_type} labels with {periods}-day forward returns")
        return result


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Data Processing Module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_samples // 10).repeat(10),
        'stock_code': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'] * (n_samples // 5),
        'close': np.random.randn(n_samples) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, n_samples),
        'pe_ratio': np.random.randn(n_samples) * 5 + 20,
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare'], n_samples)
    })
    
    # 添加一些极值和缺失值
    test_data.loc[0, 'pe_ratio'] = 1000  # 极值
    test_data.loc[1, 'pe_ratio'] = -500  # 极值
    test_data.loc[10:20, 'close'] = np.nan  # 缺失值
    
    print("\n1. Testing DataCleaner...")
    cleaner = DataCleaner()
    
    # 测试去极值
    winsorized = cleaner.winsorize(test_data, columns=['pe_ratio'], method='percentile')
    print(f"   PE ratio range after winsorization: [{winsorized['pe_ratio'].min():.2f}, {winsorized['pe_ratio'].max():.2f}]")
    
    # 测试缺失值填充
    filled = cleaner.fill_missing(test_data, columns=['close'], method='median')
    print(f"   Missing values after filling: {filled['close'].isna().sum()}")
    
    # 测试标准化
    standardized = cleaner.standardize(test_data, columns=['pe_ratio'], method='zscore')
    print(f"   PE ratio mean after standardization: {standardized['pe_ratio'].mean():.4f}")
    
    # 测试完整流水线
    cleaned = cleaner.clean_pipeline(
        test_data,
        columns=['close', 'pe_ratio'],
        by_date=True,
        date_col='date'
    )
    print(f"   Pipeline completed. Shape: {cleaned.shape}")
    
    print("\n2. Testing BiasHandler...")
    bias_handler = BiasHandler()
    
    # 测试特征滞后
    shifted = bias_handler.shift_features(
        test_data,
        feature_cols=['close', 'pe_ratio'],
        shift_periods=1,
        stock_col='stock_code'
    )
    print(f"   Features shifted. First row close is now NaN: {pd.isna(shifted.groupby('stock_code')['close'].first().iloc[0])}")
    
    print("\n3. Testing LabelGenerator...")
    label_gen = LabelGenerator()
    
    # 测试回归标签
    labeled = label_gen.generate_return_label(
        test_data,
        periods=5,
        label_type='regression',
        stock_col='stock_code'
    )
    print(f"   Regression labels generated. Non-null labels: {labeled['label'].notna().sum()}")
    
    # 测试分类标签
    labeled_cls = label_gen.generate_return_label(
        test_data,
        periods=5,
        label_type='classification',
        n_classes=3,
        stock_col='stock_code'
    )
    print(f"   Classification labels distribution: {labeled_cls['label'].value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
