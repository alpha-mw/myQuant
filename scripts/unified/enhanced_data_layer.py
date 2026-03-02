#!/usr/bin/env python3
"""
Enhanced Data Layer - 增强版数据层
支持丰富的数据源、完整的数据清理和特征工程

数据源:
- 价格数据: OHLCV、Tick数据、订单簿
- 基本面数据: 财务报表、分析师预期、宏观指标
- 另类数据: 社交媒体情绪、资金流向等

数据处理:
- 清洗: 去极值、补缺失、复权处理
- 标准化: Z-Score、Rank、行业中性化
- 特征工程: 因子计算、标签定义
- 偏差防控: 前视偏差、幸存者偏差
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 可选依赖
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ==================== 配置 ====================

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '33d6ebd3bad7812192d768a191e29ebe653a1839b3f63ec8a0dd7da94172')
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


# ==================== 数据结构 ====================

@dataclass
class OHLCVData:
    """OHLCV价格数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None  # 复权收盘价
    
@dataclass
class TickData:
    """Tick级别数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_volume: int
    ask_volume: int

@dataclass
class FundamentalData:
    """基本面数据"""
    symbol: str
    report_date: str
    # 盈利能力
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    # 成长能力
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    # 估值指标
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    dividend_yield: Optional[float] = None
    # 财务健康
    debt_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    cash_flow: Optional[float] = None

@dataclass
class MacroData:
    """宏观指标数据"""
    date: str
    gdp_yoy: Optional[float] = None
    cpi_yoy: Optional[float] = None
    ppi_yoy: Optional[float] = None
    m2_yoy: Optional[float] = None
    interest_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None


# ==================== 数据获取基类 ====================

class DataSourceBase(ABC):
    """数据源基类"""
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取OHLCV数据"""
        pass
    
    @abstractmethod
    def get_fundamental(self, symbol: str) -> FundamentalData:
        """获取基本面数据"""
        pass


# ==================== Tushare数据源 ====================

class TushareDataSource(DataSourceBase):
    """Tushare数据源 - A股"""
    
    def __init__(self):
        self.pro = None
        if TUSHARE_AVAILABLE:
            try:
                ts.set_token(TUSHARE_TOKEN)
                self.pro = ts.pro_api()
                self.pro._DataApi__token = TUSHARE_TOKEN
                self.pro._DataApi__http_url = TUSHARE_URL
                print("[TushareDataSource] 初始化成功")
            except Exception as e:
                print(f"[TushareDataSource] 初始化失败: {e}")
    
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, 
                  freq: str = 'D') -> pd.DataFrame:
        """
        获取OHLCV数据
        
        Args:
            symbol: 股票代码 (如 '000001.SZ')
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            freq: 频率 D=日线, W=周线, M=月线
        """
        if not self.pro:
            return pd.DataFrame()
        
        try:
            # 获取复权因子
            adj_df = self.pro.adj_factor(ts_code=symbol)
            
            # 获取日线数据
            df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 合并复权因子
            if adj_df is not None and not adj_df.empty:
                df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
                # 计算复权价格
                df['adj_close'] = df['close'] * df['adj_factor']
                df['adj_open'] = df['open'] * df['adj_factor']
                df['adj_high'] = df['high'] * df['adj_factor']
                df['adj_low'] = df['low'] * df['adj_factor']
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            # 重命名列
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            })
            
            return df
            
        except Exception as e:
            print(f"[TushareDataSource] 获取OHLCV失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental(self, symbol: str) -> FundamentalData:
        """获取基本面数据"""
        if not self.pro:
            return FundamentalData(symbol=symbol)
        
        try:
            # 获取最新财务指标
            df = self.pro.fina_indicator(ts_code=symbol, limit=1)
            
            if df is None or df.empty:
                return FundamentalData(symbol=symbol)
            
            latest = df.iloc[0]
            
            return FundamentalData(
                symbol=symbol,
                report_date=str(latest.get('end_date', '')),
                roe=float(latest.get('roe', 0)) if pd.notna(latest.get('roe')) else None,
                roa=float(latest.get('roa', 0)) if pd.notna(latest.get('roa')) else None,
                gross_margin=float(latest.get('grossprofit_margin', 0)) if pd.notna(latest.get('grossprofit_margin')) else None,
                net_margin=float(latest.get('netprofit_margin', 0)) if pd.notna(latest.get('netprofit_margin')) else None,
                revenue_growth=float(latest.get('or_yoy', 0)) if pd.notna(latest.get('or_yoy')) else None,
                profit_growth=float(latest.get('netprofit_yoy', 0)) if pd.notna(latest.get('netprofit_yoy')) else None,
                debt_ratio=float(latest.get('debt_to_assets', 0)) if pd.notna(latest.get('debt_to_assets')) else None,
            )
            
        except Exception as e:
            print(f"[TushareDataSource] 获取基本面数据失败 {symbol}: {e}")
            return FundamentalData(symbol=symbol)
    
    def get_daily_basic(self, symbol: str, trade_date: str) -> Dict[str, float]:
        """获取每日指标 (估值数据)"""
        if not self.pro:
            return {}
        
        try:
            df = self.pro.daily_basic(ts_code=symbol, trade_date=trade_date)
            if df is not None and not df.empty:
                latest = df.iloc[0]
                return {
                    'pe': float(latest.get('pe', 0)) if pd.notna(latest.get('pe')) else None,
                    'pb': float(latest.get('pb', 0)) if pd.notna(latest.get('pb')) else None,
                    'ps': float(latest.get('ps', 0)) if pd.notna(latest.get('ps')) else None,
                    'dividend_yield': float(latest.get('dv_ratio', 0)) if pd.notna(latest.get('dv_ratio')) else None,
                    'total_mv': float(latest.get('total_mv', 0)) if pd.notna(latest.get('total_mv')) else None,
                    'circ_mv': float(latest.get('circ_mv', 0)) if pd.notna(latest.get('circ_mv')) else None,
                }
        except Exception as e:
            print(f"[TushareDataSource] 获取每日指标失败 {symbol}: {e}")
        
        return {}
    
    def get_money_flow(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取资金流向数据"""
        if not self.pro:
            return pd.DataFrame()
        
        try:
            df = self.pro.moneyflow(ts_code=symbol, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df.sort_values('trade_date')
        except Exception as e:
            print(f"[TushareDataSource] 获取资金流向失败 {symbol}: {e}")
        
        return pd.DataFrame()


# ==================== Yahoo Finance数据源 ====================

class YahooDataSource(DataSourceBase):
    """Yahoo Finance数据源 - 美股"""
    
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str,
                  freq: str = '1d') -> pd.DataFrame:
        """获取OHLCV数据"""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=freq)
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            df.index.name = 'date'
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            print(f"[YahooDataSource] 获取OHLCV失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental(self, symbol: str) -> FundamentalData:
        """获取基本面数据"""
        if not YFINANCE_AVAILABLE:
            return FundamentalData(symbol=symbol)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return FundamentalData(
                symbol=symbol,
                report_date=datetime.now().strftime('%Y-%m-%d'),
                pe=info.get('trailingPE'),
                pb=info.get('priceToBook'),
                ps=info.get('priceToSalesTrailing12Months'),
                dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None,
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                gross_margin=info.get('grossMargins', 0) * 100 if info.get('grossMargins') else None,
                net_margin=info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
                revenue_growth=info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
                profit_growth=info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None,
                debt_ratio=info.get('debtToEquity'),
            )
            
        except Exception as e:
            print(f"[YahooDataSource] 获取基本面数据失败 {symbol}: {e}")
            return FundamentalData(symbol=symbol)


# ==================== 数据清理器 ====================

class DataCleaner:
    """
    数据清理器
    
    功能:
    1. 去极值 (Winsorization)
    2. 补缺失值
    3. 标准化
    4. 前视偏差防控
    5. 幸存者偏差处理
    6. 复权处理
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DataCleaner] {msg}")
    
    def winsorize(self, df: pd.DataFrame, columns: List[str],
                  method: str = 'mad', sigma: float = 3.0) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            method: 'mad' (中位数绝对偏差), 'sigma' (标准差), 'percentile' (百分位数)
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # 只处理数值列
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            
            series = result[col].dropna()
            if len(series) == 0:
                continue
            
            if method == 'mad':
                median = series.median()
                mad = np.median(np.abs(series - median))
                lower = median - sigma * 1.4826 * mad
                upper = median + sigma * 1.4826 * mad
            elif method == 'sigma':
                mean = series.mean()
                std = series.std()
                lower = mean - sigma * std
                upper = mean + sigma * std
            elif method == 'percentile':
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
            else:
                continue
            
            result[col] = result[col].clip(lower=lower, upper=upper)
        
        self._log(f"去极值完成: {len(columns)} 列, 方法={method}")
        return result
    
    def fill_missing(self, df: pd.DataFrame, columns: List[str],
                     method: str = 'ffill') -> pd.DataFrame:
        """补缺失值"""
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # 只处理数值列
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            
            if method == 'ffill':
                result[col] = result[col].ffill()
            elif method == 'bfill':
                result[col] = result[col].bfill()
            elif method == 'mean':
                result[col] = result[col].fillna(result[col].mean())
            elif method == 'median':
                result[col] = result[col].fillna(result[col].median())
            elif method == 'zero':
                result[col] = result[col].fillna(0)
        
        self._log(f"补缺失完成: {len(columns)} 列, 方法={method}")
        return result
    
    def standardize(self, df: pd.DataFrame, columns: List[str],
                    method: str = 'zscore') -> pd.DataFrame:
        """
        标准化
        
        Args:
            method: 'zscore' (Z-Score), 'rank' (排名), 'minmax' (Min-Max)
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # 只处理数值列
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            
            series = result[col].dropna()
            if len(series) == 0:
                continue
            
            if method == 'zscore':
                mean = series.mean()
                std = series.std()
                if std != 0:
                    result[col] = (result[col] - mean) / std
            elif method == 'rank':
                result[col] = result[col].rank(pct=True)
            elif method == 'minmax':
                min_val = series.min()
                max_val = series.max()
                if max_val != min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
        
        self._log(f"标准化完成: {len(columns)} 列, 方法={method}")
        return result
    
    def prevent_look_ahead_bias(self, df: pd.DataFrame, factor_cols: List[str],
                                 delay_periods: int = 1) -> pd.DataFrame:
        """
        前视偏差防控
        
        将因子值滞后，确保使用过去的数据预测未来
        """
        result = df.copy()
        
        for col in factor_cols:
            if col in result.columns:
                result[col] = result[col].shift(delay_periods)
        
        self._log(f"前视偏差防控: 滞后 {delay_periods} 期")
        return result
    
    def process(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """完整的数据处理流程"""
        self._log("开始数据清理流程...")
        
        # 1. 去极值
        df = self.winsorize(df, numeric_cols, method='mad')
        
        # 2. 补缺失
        df = self.fill_missing(df, numeric_cols, method='ffill')
        
        # 3. 标准化
        df = self.standardize(df, numeric_cols, method='zscore')
        
        # 4. 前视偏差防控
        df = self.prevent_look_ahead_bias(df, numeric_cols, delay_periods=1)
        
        self._log("数据清理完成")
        return df


# ==================== 特征工程 ====================

class FeatureEngineer:
    """
    特征工程
    
    将原始数据转化为预测因子
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[FeatureEngineer] {msg}")
    
    # ========== 价量因子 ==========
    
    def calc_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量因子 - 使用历史收益率，避免前视偏差"""
        result = df.copy()
        
        # 历史收益率 (过去N天的收益，不是未来)
        for period in [5, 10, 20, 60, 120]:
            # 过去period天的累计收益: (today - period_days_ago) / period_days_ago
            result[f'momentum_{period}d'] = result['close'] / result['close'].shift(period) - 1
        
        # 经典动量: 过去12个月剔除最近1个月的收益
        result['momentum_12_1'] = result['close'].shift(20) / result['close'].shift(240) - 1
        
        self._log("动量因子计算完成")
        return result
    
    def calc_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率因子"""
        result = df.copy()
        
        # 历史波动率
        for period in [20, 60, 120]:
            result[f'volatility_{period}d'] = result['close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # ATR (Average True Range)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr_14'] = tr.rolling(14).mean()
        
        self._log("波动率因子计算完成")
        return result
    
    def calc_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """技术指标因子"""
        result = df.copy()
        
        # RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = result['close'].ewm(span=12, adjust=False).mean()
        exp2 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = exp1 - exp2
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        
        # 布林带
        result['ma_20'] = result['close'].rolling(20).mean()
        result['std_20'] = result['close'].rolling(20).std()
        result['bollinger_upper'] = result['ma_20'] + 2 * result['std_20']
        result['bollinger_lower'] = result['ma_20'] - 2 * result['std_20']
        result['bollinger_position'] = (result['close'] - result['bollinger_lower']) / (result['bollinger_upper'] - result['bollinger_lower'])
        
        # 移动平均线偏离
        for period in [5, 10, 20, 60]:
            ma = result['close'].rolling(period).mean()
            result[f'ma_bias_{period}'] = (result['close'] - ma) / ma
        
        self._log("技术指标因子计算完成")
        return result
    
    def calc_liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """流动性因子"""
        result = df.copy()
        
        # 换手率
        if 'volume' in result.columns:
            # 需要流通股本数据，这里简化处理
            result['volume_ratio_5d'] = result['volume'] / result['volume'].rolling(5).mean()
            result['volume_ratio_20d'] = result['volume'] / result['volume'].rolling(20).mean()
        
        # Amihud非流动性指标 (|收益率| / 成交金额)
        if 'amount' in result.columns:
            result['amihud'] = np.abs(result['close'].pct_change()) / result['amount']
        
        self._log("流动性因子计算完成")
        return result
    
    # ========== 基本面因子 ==========
    
    def calc_fundamental_factors(self, df: pd.DataFrame, fundamental: FundamentalData) -> pd.DataFrame:
        """基本面因子"""
        result = df.copy()
        
        # 估值指标
        if fundamental.pe:
            result['pe'] = fundamental.pe
        if fundamental.pb:
            result['pb'] = fundamental.pb
        if fundamental.ps:
            result['ps'] = fundamental.ps
        if fundamental.dividend_yield:
            result['dividend_yield'] = fundamental.dividend_yield
        
        # 质量指标
        if fundamental.roe:
            result['roe'] = fundamental.roe
        if fundamental.gross_margin:
            result['gross_margin'] = fundamental.gross_margin
        if fundamental.net_margin:
            result['net_margin'] = fundamental.net_margin
        
        # 成长指标
        if fundamental.revenue_growth:
            result['revenue_growth'] = fundamental.revenue_growth
        if fundamental.profit_growth:
            result['profit_growth'] = fundamental.profit_growth
        
        self._log("基本面因子计算完成")
        return result
    
    # ========== 特征工程主入口 ==========
    
    def create_features(self, df: pd.DataFrame, 
                       fundamental: Optional[FundamentalData] = None) -> pd.DataFrame:
        """创建所有特征"""
        self._log("开始特征工程...")
        
        # 价量因子
        df = self.calc_momentum_factors(df)
        df = self.calc_volatility_factors(df)
        df = self.calc_technical_factors(df)
        df = self.calc_liquidity_factors(df)
        
        # 基本面因子
        if fundamental:
            df = self.calc_fundamental_factors(df, fundamental)
        
        self._log(f"特征工程完成，共 {len(df.columns)} 列")
        return df


# ==================== 标签定义 ====================

class LabelGenerator:
    """
    标签生成器
    
    定义预测目标
    """
    
    @staticmethod
    def future_return(df: pd.DataFrame, periods: int = 5) -> pd.Series:
        """未来N日收益率 (回归问题)"""
        return df['close'].shift(-periods) / df['close'] - 1
    
    @staticmethod
    def future_direction(df: pd.DataFrame, periods: int = 5) -> pd.Series:
        """未来涨跌方向 (分类问题)"""
        future_return = df['close'].shift(-periods) / df['close'] - 1
        return (future_return > 0).astype(int)
    
    @staticmethod
    def future_return_quantile(df: pd.DataFrame, periods: int = 5, quantiles: int = 5) -> pd.Series:
        """未来收益率分位数 (排序问题)"""
        future_return = df['close'].shift(-periods) / df['close'] - 1
        return pd.qcut(future_return, quantiles, labels=False, duplicates='drop')


# ==================== 增强版数据层主类 ====================

class EnhancedDataLayer:
    """
    增强版数据层
    
    集成数据获取、清理、特征工程
    """
    
    def __init__(self, market: str = "CN", verbose: bool = True):
        self.market = market.upper()
        self.verbose = verbose
        
        # 初始化数据源
        if self.market == "CN":
            self.data_source = TushareDataSource()
        elif self.market == "US":
            self.data_source = YahooDataSource()
        else:
            raise ValueError(f"不支持的市场: {market}")
        
        # 初始化处理器
        self.cleaner = DataCleaner(verbose=verbose)
        self.feature_engineer = FeatureEngineer(verbose=verbose)
        self.label_generator = LabelGenerator()
    
    def fetch_and_process(self, symbol: str, start_date: str, end_date: str,
                          label_periods: int = 5) -> pd.DataFrame:
        """
        获取并处理完整数据
        
        Returns:
            包含特征和标签的DataFrame
        """
        print(f"\n[EnhancedDataLayer] 处理 {symbol}...")
        
        # 1. 获取OHLCV数据
        df = self.data_source.get_ohlcv(symbol, start_date, end_date)
        if df.empty:
            print(f"[EnhancedDataLayer] 无数据: {symbol}")
            return pd.DataFrame()
        
        print(f"[EnhancedDataLayer] 获取 {len(df)} 条OHLCV数据")
        
        # 2. 获取基本面数据
        fundamental = self.data_source.get_fundamental(symbol)
        
        # 3. 特征工程
        df = self.feature_engineer.create_features(df, fundamental)
        
        # 4. 数据清理
        factor_cols = [c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        df = self.cleaner.process(df, factor_cols)
        
        # 5. 生成标签
        df['label_return'] = self.label_generator.future_return(df, label_periods)
        df['label_direction'] = self.label_generator.future_direction(df, label_periods)
        
        # 6. 添加元信息
        df['symbol'] = symbol
        df['market'] = self.market
        
        print(f"[EnhancedDataLayer] 处理完成: {len(df)} 行, {len(df.columns)} 列")
        
        return df


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Enhanced Data Layer - 测试")
    print("=" * 80)
    
    # 测试A股
    print("\n【测试A股】")
    data_layer = EnhancedDataLayer(market="CN", verbose=True)
    
    df = data_layer.fetch_and_process(
        symbol="000001.SZ",
        start_date="20240101",
        end_date="20240225",
        label_periods=5
    )
    
    if not df.empty:
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n因子列表:")
        factor_cols = [c for c in df.columns if c.startswith(('return_', 'volatility_', 'rsi_', 'macd_', 'ma_bias_', 'label_'))]
        print(factor_cols)
