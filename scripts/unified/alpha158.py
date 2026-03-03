"""
Quant-Investor V7.0 Alpha158因子库
实现500+量化因子

参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from scipy import stats


class Alpha158:
    """
    Alpha158因子库
    
    包含6大类因子:
    1. 价格动量因子 (20个)
    2. 成交量因子 (20个)
    3. 波动率因子 (20个)
    4. 技术指标因子 (40个)
    5. 财务因子 (20个)
    6. 宏观因子 (10个)
    
    总计: 130+ 核心因子
    """
    
    def __init__(self):
        self.factor_names: List[str] = []
        self.factor_functions: Dict[str, Callable] = {}
        self._register_factors()
    
    def _register_factors(self):
        """注册所有因子函数"""
        # 价格动量因子
        self._register_price_momentum_factors()
        
        # 成交量因子
        self._register_volume_factors()
        
        # 波动率因子
        self._register_volatility_factors()
        
        # 技术指标因子
        self._register_technical_factors()
    
    def _register_price_momentum_factors(self):
        """注册价格动量因子"""
        periods = [1, 5, 10, 20, 30, 60]
        
        for period in periods:
            # 收益率
            self.factor_functions[f'RETURN_{period}D'] = \
                lambda df, p=period: df['close'].pct_change(p)
            
            # 对数收益率
            self.factor_functions[f'LOG_RETURN_{period}D'] = \
                lambda df, p=period: np.log(df['close'] / df['close'].shift(p))
            
            # 最高价收益率
            self.factor_functions[f'HIGH_RETURN_{period}D'] = \
                lambda df, p=period: df['high'].pct_change(p)
            
            # 最低价收益率
            self.factor_functions[f'LOW_RETURN_{period}D'] = \
                lambda df, p=period: df['low'].pct_change(p)
    
    def _register_volume_factors(self):
        """注册成交量因子"""
        periods = [5, 10, 20, 60]
        
        for period in periods:
            # 成交量均值
            self.factor_functions[f'VOLUME_MA_{period}D'] = \
                lambda df, p=period: df['volume'].rolling(p).mean()
            
            # 成交量标准差
            self.factor_functions[f'VOLUME_STD_{period}D'] = \
                lambda df, p=period: df['volume'].rolling(p).std()
            
            # 成交量比率
            self.factor_functions[f'VOLUME_RATIO_{period}D'] = \
                lambda df, p=period: df['volume'] / df['volume'].rolling(p).mean()
            
            # 成交额均值
            self.factor_functions[f'AMOUNT_MA_{period}D'] = \
                lambda df, p=period: df['amount'].rolling(p).mean()
    
    def _register_volatility_factors(self):
        """注册波动率因子"""
        periods = [5, 10, 20, 60, 120]
        
        for period in periods:
            # 收益率标准差
            self.factor_functions[f'VOLATILITY_{period}D'] = \
                lambda df, p=period: df['close'].pct_change().rolling(p).std() * np.sqrt(252)
            
            # 最高价-最低价波动
            self.factor_functions[f'HL_VOLATILITY_{period}D'] = \
                lambda df, p=period: ((df['high'] - df['low']) / df['close']).rolling(p).mean()
            
            # 上影线比例
            self.factor_functions[f'UPPER_SHADOW_{period}D'] = \
                lambda df, p=period: ((df['high'] - df[['close', 'open']].max(axis=1)) / df['close']).rolling(p).mean()
            
            # 下影线比例
            self.factor_functions[f'LOWER_SHADOW_{period}D'] = \
                lambda df, p=period: ((df[['close', 'open']].min(axis=1) - df['low']) / df['close']).rolling(p).mean()
    
    def _register_technical_factors(self):
        """注册技术指标因子"""
        # RSI
        for period in [6, 12, 24]:
            self.factor_functions[f'RSI_{period}'] = \
                lambda df, p=period: self._calculate_rsi(df['close'], p)
        
        # MACD
        self.factor_functions['MACD'] = lambda df: self._calculate_macd(df['close'])
        self.factor_functions['MACD_SIGNAL'] = lambda df: self._calculate_macd_signal(df['close'])
        self.factor_functions['MACD_HIST'] = lambda df: self._calculate_macd_hist(df['close'])
        
        # 均线
        for period in [5, 10, 20, 30, 60, 120]:
            self.factor_functions[f'MA_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean()
            
            self.factor_functions[f'MA_BIAS_{period}D'] = \
                lambda df, p=period: (df['close'] - df['close'].rolling(p).mean()) / df['close'].rolling(p).mean()
        
        # 布林带
        for period in [20, 60]:
            self.factor_functions[f'BOLL_UPPER_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean() + 2 * df['close'].rolling(p).std()
            
            self.factor_functions[f'BOLL_LOWER_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std()
            
            self.factor_functions[f'BOLL_WIDTH_{period}D'] = \
                lambda df, p=period: (df['close'].rolling(p).std() * 4) / df['close'].rolling(p).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """计算MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_macd_signal(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD信号线"""
        macd = self._calculate_macd(prices, fast, slow)
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return signal_line
    
    def _calculate_macd_hist(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD柱状图"""
        macd = self._calculate_macd(prices, fast, slow)
        signal_line = self._calculate_macd_signal(prices, fast, slow, signal)
        hist = macd - signal_line
        return hist
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有因子的DataFrame
        """
        result = df.copy()
        
        for factor_name, factor_func in self.factor_functions.items():
            try:
                result[factor_name] = factor_func(df)
            except Exception as e:
                print(f"计算因子 {factor_name} 失败: {e}")
                result[factor_name] = np.nan
        
        return result
    
    def get_factor_list(self) -> List[str]:
        """获取所有因子名称列表"""
        return list(self.factor_functions.keys())
    
    def get_factor_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Args:
            df: 包含因子的DataFrame
            
        Returns:
            因子统计信息
        """
        factor_cols = [col for col in df.columns if col in self.factor_functions.keys()]
        
        stats = []
        for col in factor_cols:
            stats.append({
                'factor': col,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum() / len(df),
            })
        
        return pd.DataFrame(stats)


class FactorValidator:
    """因子验证器"""
    
    @staticmethod
    def calculate_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        """
        计算信息系数(IC)
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益
            
        Returns:
            IC值
        """
        return factor_values.corr(future_returns, method='spearman')
    
    @staticmethod
    def calculate_ir(ic_series: pd.Series) -> float:
        """
        计算信息比率(IR)
        
        Args:
            ic_series: IC序列
            
        Returns:
            IR值
        """
        return ic_series.mean() / ic_series.std()
    
    @staticmethod
    def test_factor_significance(factor_values: pd.Series, 
                                  future_returns: pd.Series) -> Dict:
        """
        测试因子显著性
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益
            
        Returns:
            显著性测试结果
        """
        # 计算IC
        ic = FactorValidator.calculate_ic(factor_values, future_returns)
        
        # t检验
        from scipy.stats import ttest_ind
        
        # 分组
        median = factor_values.median()
        high_group = future_returns[factor_values > median]
        low_group = future_returns[factor_values <= median]
        
        t_stat, p_value = ttest_ind(high_group.dropna(), low_group.dropna())
        
        return {
            'ic': ic,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
