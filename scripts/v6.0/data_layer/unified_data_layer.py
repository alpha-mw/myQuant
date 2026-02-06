#!/usr/bin/env python3
"""
Quant-Investor V6.0 - 统一数据层 (Unified Data Layer)

整合所有历史版本的数据能力：
- V2.7: 持久化数据存储 (SQLite + Parquet)
- V3.0: 期货/期权/行业数据
- V4.0: 统一数据获取 (Tushare/yfinance/FRED)
- V4.1: 基准数据 (指数成分股/基准收益)
- V5.0: 数据清洗 (去极值/缺失值/标准化/偏差处理)

设计原则：
1. 所有数据自动持久化，支持增量更新
2. 统一的数据接口，屏蔽底层数据源差异
3. 内置数据清洗流水线，确保数据质量
4. 支持时点数据(Point-in-Time)，防止前视偏差
"""

import os
import sys
import json
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ==================== 数据结构定义 ====================

@dataclass
class MarketConfig:
    """市场配置"""
    name: str
    indices: List[str]
    index_codes: Dict[str, str]
    data_source: str
    currency: str
    benchmark_symbol: str  # 基准指数代码


MARKET_CONFIGS = {
    "CN": MarketConfig(
        name="A股市场",
        indices=["沪深300", "中证1000"],
        index_codes={"沪深300": "000300.SH", "中证1000": "000852.SH"},
        data_source="tushare",
        currency="CNY",
        benchmark_symbol="000300.SH"
    ),
    "US": MarketConfig(
        name="美股市场",
        indices=["纳斯达克100", "标普500"],
        index_codes={"纳斯达克100": "^NDX", "标普500": "^GSPC"},
        data_source="yfinance",
        currency="USD",
        benchmark_symbol="^GSPC"
    )
}


@dataclass
class StockRecord:
    """单只股票的完整数据记录"""
    code: str
    name: str
    market: str
    industry: str = ""
    sector: str = ""
    price_data: pd.DataFrame = None
    financial_data: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class UnifiedDataBundle:
    """统一数据包 - 包含分析所需的全部数据"""
    market: str
    config: MarketConfig
    fetch_date: str
    
    # 核心数据
    stock_universe: Dict[str, StockRecord] = field(default_factory=dict)
    benchmark_data: pd.DataFrame = None
    
    # 宏观数据
    macro_data: Dict[str, pd.Series] = field(default_factory=dict)
    
    # 行业数据
    industry_data: Dict[str, Any] = field(default_factory=dict)
    
    # 清洗后的面板数据 (用于因子计算)
    panel_data: pd.DataFrame = None
    
    # 用户关注的股票 (自定义股票池时非空，表示决策层只分析这些股票)
    focus_stocks: Optional[List[str]] = None
    
    # 元信息
    stats: Dict[str, Any] = field(default_factory=dict)


# ==================== 持久化数据管理器 ====================

class PersistentDataManager:
    """
    持久化数据管理器 (源自V2.7)
    
    使用SQLite存储元数据，Parquet存储时序数据。
    支持增量更新，避免重复下载。
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or os.path.expanduser("~/.quant_investor/data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "metadata.db"
        self._init_db()
    
    def _init_db(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key TEXT PRIMARY KEY,
                data_type TEXT,
                market TEXT,
                last_updated TEXT,
                file_path TEXT,
                row_count INTEGER,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS download_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market TEXT,
                data_type TEXT,
                status TEXT,
                message TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, market: str, data_type: str, symbol: str = "", 
                        start_date: str = "", end_date: str = "") -> str:
        """生成缓存键"""
        raw = f"{market}_{data_type}_{symbol}_{start_date}_{end_date}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def is_cached(self, market: str, data_type: str, symbol: str = "",
                   max_age_hours: int = 24) -> bool:
        """检查数据是否已缓存且未过期"""
        cache_key = self._get_cache_key(market, data_type, symbol)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT last_updated, file_path FROM data_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return False
        
        last_updated = datetime.fromisoformat(row[0])
        file_path = row[1]
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False
        
        # 检查是否过期
        age = datetime.now() - last_updated
        return age.total_seconds() < max_age_hours * 3600
    
    def save_dataframe(self, df: pd.DataFrame, market: str, data_type: str, 
                        symbol: str = "", metadata: Dict = None) -> str:
        """保存DataFrame到Parquet文件"""
        cache_key = self._get_cache_key(market, data_type, symbol)
        
        # 生成文件路径
        subdir = self.data_dir / market.lower() / data_type
        subdir.mkdir(parents=True, exist_ok=True)
        
        safe_symbol = symbol.replace(".", "_").replace("^", "_").replace("/", "_")
        file_path = str(subdir / f"{safe_symbol or 'data'}_{cache_key[:8]}.parquet")
        
        # 保存Parquet
        df.to_parquet(file_path, index=True)
        
        # 更新元数据
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO data_cache 
            (cache_key, data_type, market, last_updated, file_path, row_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cache_key, data_type, market, datetime.now().isoformat(),
            file_path, len(df), json.dumps(metadata or {})
        ))
        conn.commit()
        conn.close()
        
        return file_path
    
    def load_dataframe(self, market: str, data_type: str, symbol: str = "") -> Optional[pd.DataFrame]:
        """从缓存加载DataFrame"""
        cache_key = self._get_cache_key(market, data_type, symbol)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM data_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()
        
        if row and os.path.exists(row[0]):
            return pd.read_parquet(row[0])
        return None
    
    def log_download(self, market: str, data_type: str, status: str, message: str = ""):
        """记录下载日志"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO download_log (timestamp, market, data_type, status, message)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), market, data_type, status, message))
        conn.commit()
        conn.close()


# ==================== 数据清洗器 ====================

class DataCleaner:
    """
    数据清洗器 (源自V5.0)
    
    提供完整的数据清洗流水线：
    1. 去极值 (Winsorization)
    2. 缺失值填充
    3. 标准化
    4. 前视偏差防控
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DataCleaner] {msg}")
    
    def winsorize(self, data: pd.DataFrame, columns: List[str] = None,
                   method: str = 'mad', sigma: float = 3.0) -> pd.DataFrame:
        """去极值处理"""
        result = data.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                continue
            series = result[col].dropna()
            if len(series) == 0:
                continue
            
            if method == 'mad':
                median = series.median()
                mad = np.median(np.abs(series - median))
                lower = median - sigma * 1.4826 * mad
                upper = median + sigma * 1.4826 * mad
            elif method == 'percentile':
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
            elif method == 'sigma':
                mean = series.mean()
                std = series.std()
                lower = mean - sigma * std
                upper = mean + sigma * std
            else:
                continue
            
            result[col] = result[col].clip(lower=lower, upper=upper)
        
        self._log(f"去极值完成: {len(columns)} 列, 方法={method}")
        return result
    
    def fill_missing(self, data: pd.DataFrame, columns: List[str] = None,
                      method: str = 'ffill') -> pd.DataFrame:
        """缺失值填充"""
        result = data.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        total_filled = 0
        for col in columns:
            if col not in result.columns:
                continue
            missing = result[col].isna().sum()
            if missing == 0:
                continue
            
            if method == 'ffill':
                result[col] = result[col].ffill().bfill()
            elif method == 'median':
                result[col] = result[col].fillna(result[col].median())
            elif method == 'mean':
                result[col] = result[col].fillna(result[col].mean())
            elif method == 'interpolate':
                result[col] = result[col].interpolate(method='linear')
            
            total_filled += missing
        
        self._log(f"缺失值填充完成: 共填充 {total_filled} 个缺失值")
        return result
    
    def standardize(self, data: pd.DataFrame, columns: List[str] = None,
                     method: str = 'zscore', by_date: bool = False,
                     date_col: str = 'date') -> pd.DataFrame:
        """标准化"""
        result = data.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        def _std(s, m):
            if m == 'zscore':
                return (s - s.mean()) / (s.std() + 1e-8)
            elif m == 'minmax':
                return (s - s.min()) / (s.max() - s.min() + 1e-8)
            elif m == 'rank':
                return s.rank(pct=True)
            return s
        
        for col in columns:
            if col not in result.columns or col == date_col:
                continue
            if by_date and date_col in result.columns:
                result[col] = result.groupby(date_col)[col].transform(lambda x: _std(x, method))
            else:
                result[col] = _std(result[col], method)
        
        self._log(f"标准化完成: {len(columns)} 列, 方法={method}")
        return result
    
    def shift_features(self, data: pd.DataFrame, feature_cols: List[str],
                        shift_periods: int = 1, stock_col: str = 'stock_code') -> pd.DataFrame:
        """特征滞后处理，防止前视偏差"""
        result = data.copy()
        if stock_col in result.columns:
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result.groupby(stock_col)[col].shift(shift_periods)
        else:
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result[col].shift(shift_periods)
        
        self._log(f"特征滞后处理完成: {len(feature_cols)} 列, 滞后 {shift_periods} 期")
        return result
    
    def clean_pipeline(self, data: pd.DataFrame, columns: List[str] = None,
                        winsorize_method: str = 'mad',
                        fill_method: str = 'ffill',
                        standardize_method: str = 'zscore',
                        by_date: bool = True, date_col: str = 'date') -> pd.DataFrame:
        """完整清洗流水线: 去极值 -> 缺失值填充 -> 标准化"""
        self._log("开始数据清洗流水线...")
        result = self.winsorize(data, columns, method=winsorize_method)
        result = self.fill_missing(result, columns, method=fill_method)
        result = self.standardize(result, columns, method=standardize_method, 
                                   by_date=by_date, date_col=date_col)
        self._log("数据清洗流水线完成")
        return result


# ==================== 统一数据层 ====================

class UnifiedDataLayer:
    """
    V6.0 统一数据层
    
    整合所有数据获取、持久化和清洗能力，提供统一的数据接口。
    """
    
    def __init__(self, market: str = "US", lookback_years: int = 3, 
                  verbose: bool = True, cache_hours: int = 24):
        """
        初始化统一数据层
        
        Args:
            market: 市场类型 ("CN" 或 "US")
            lookback_years: 历史数据回溯年数
            verbose: 是否打印详细信息
            cache_hours: 缓存有效期（小时）
        """
        self.market = market.upper()
        if self.market not in MARKET_CONFIGS:
            raise ValueError(f"不支持的市场: {market}. 支持: {list(MARKET_CONFIGS.keys())}")
        
        self.config = MARKET_CONFIGS[self.market]
        self.lookback_years = lookback_years
        self.verbose = verbose
        self.cache_hours = cache_hours
        
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_years * 365)
        
        # 初始化子模块
        self.storage = PersistentDataManager()
        self.cleaner = DataCleaner(verbose=verbose)
        
        # 初始化数据源客户端
        self._clients = {}
        self._init_clients()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 V6.0 统一数据层初始化")
            print(f"   市场: {self.config.name}")
            print(f"   数据范围: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
            print(f"   缓存目录: {self.storage.data_dir}")
            print(f"{'='*60}")
    
    def _init_clients(self):
        """初始化数据源客户端"""
        # Tushare (A股)
        if self.market == "CN":
            try:
                import tushare as ts
                token = os.getenv("TUSHARE_TOKEN", "")
                if token:
                    ts.set_token(token)
                self._clients['tushare'] = ts.pro_api()
                if self.verbose:
                    print(f"  ✅ Tushare 初始化成功")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ Tushare 初始化失败: {e}")
        
        # yfinance (美股/全球)
        try:
            import yfinance as yf
            self._clients['yfinance'] = yf
            if self.verbose:
                print(f"  ✅ yfinance 初始化成功")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ yfinance 初始化失败: {e}")
        
        # FRED (宏观数据)
        try:
            from fredapi import Fred
            fred_key = os.getenv("FRED_API_KEY", "")
            if fred_key:
                self._clients['fred'] = Fred(api_key=fred_key)
                if self.verbose:
                    print(f"  ✅ FRED 初始化成功")
        except Exception:
            pass
    
    # ==================== 核心数据获取 ====================
    
    def fetch_all(self, stock_pool: List[str] = None) -> UnifiedDataBundle:
        """
        获取全部数据，返回统一数据包
        
        Args:
            stock_pool: 指定股票池（可选，默认使用指数成分股）
        
        Returns:
            UnifiedDataBundle: 包含所有数据的统一数据包
        """
        bundle = UnifiedDataBundle(
            market=self.market,
            config=self.config,
            fetch_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"📥 开始获取 {self.config.name} 数据...")
            print(f"{'─'*50}")
        
        # 1. 获取股票池 (自定义时自动扩充指数成分股)
        focus_stocks = None
        if stock_pool:
            bundle.stock_universe, focus_stocks = self._build_expanded_universe(stock_pool)
        else:
            bundle.stock_universe = self._fetch_index_constituents()
        
        bundle.focus_stocks = focus_stocks  # None表示全部关注
        
        # 2. 获取价格数据
        bundle.stock_universe = self._fetch_price_data(bundle.stock_universe)
        
        # 3. 获取基准数据
        bundle.benchmark_data = self._fetch_benchmark_data()
        
        # 4. 获取财务数据
        bundle.stock_universe = self._fetch_financial_data(bundle.stock_universe)
        
        # 5. 获取宏观数据
        bundle.macro_data = self._fetch_macro_data()
        
        # 6. 构建面板数据
        bundle.panel_data = self._build_panel_data(bundle.stock_universe)
        
        # 7. 统计信息
        valid_stocks = sum(1 for s in bundle.stock_universe.values() 
                          if s.price_data is not None and len(s.price_data) > 0)
        bundle.stats = {
            "total_stocks": len(bundle.stock_universe),
            "valid_stocks": valid_stocks,
            "focus_stocks": len(focus_stocks) if focus_stocks else valid_stocks,
            "benchmark_available": bundle.benchmark_data is not None,
            "macro_indicators": len(bundle.macro_data),
            "panel_rows": len(bundle.panel_data) if bundle.panel_data is not None else 0,
            "date_range": f"{self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}"
        }
        
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"✅ 数据获取完成!")
            print(f"   股票总数: {bundle.stats['total_stocks']}")
            print(f"   有效数据: {bundle.stats['valid_stocks']} 只")
            print(f"   基准数据: {'✓' if bundle.stats['benchmark_available'] else '✗'}")
            print(f"   宏观指标: {bundle.stats['macro_indicators']} 个")
            print(f"   面板行数: {bundle.stats['panel_rows']}")
            print(f"{'─'*50}")
        
        return bundle
    
    # ==================== 股票池获取 ====================
    
    def _build_custom_universe(self, stock_pool: List[str]) -> Dict[str, StockRecord]:
        """构建自定义股票池"""
        universe = {}
        for code in stock_pool:
            universe[code] = StockRecord(code=code, name=code, market=self.market)
        return universe
    
    def _build_expanded_universe(self, stock_pool: List[str]) -> Tuple[Dict[str, StockRecord], List[str]]:
        """
        构建扩充后的股票池：
        - 自定义股票池的股票标记为focus_stocks
        - 自动补充指数成分股作为因子验证的完整截面样本
        
        Returns:
            (expanded_universe, focus_stocks): 扩充后的universe和用户关注的股票列表
        """
        # 1. 先获取指数成分股作为完整样本
        full_universe = self._fetch_index_constituents()
        
        # 2. 确保用户指定的股票都在其中
        focus_stocks = list(stock_pool)
        for code in focus_stocks:
            if code not in full_universe:
                full_universe[code] = StockRecord(code=code, name=code, market=self.market)
        
        if self.verbose:
            print(f"  📊 样本扩充: 用户关注 {len(focus_stocks)} 只 → 完整样本 {len(full_universe)} 只")
        
        return full_universe, focus_stocks
    
    def _fetch_index_constituents(self) -> Dict[str, StockRecord]:
        """获取指数成分股"""
        if self.market == "CN":
            return self._fetch_cn_constituents()
        else:
            return self._fetch_us_constituents()
    
    def _fetch_cn_constituents(self) -> Dict[str, StockRecord]:
        """获取A股指数成分股"""
        universe = {}
        ts_pro = self._clients.get('tushare')
        
        if ts_pro is None:
            if self.verbose:
                print("  ⚠️ Tushare不可用，使用默认A股核心股票池")
            return self._get_default_cn_stocks()
        
        for index_name, index_code in self.config.index_codes.items():
            try:
                if self.verbose:
                    print(f"  获取 {index_name} 成分股...")
                
                df = ts_pro.index_weight(index_code=index_code)
                if df is not None and len(df) > 0:
                    latest_date = df['trade_date'].max()
                    df = df[df['trade_date'] == latest_date]
                    
                    for _, row in df.iterrows():
                        code = row['con_code']
                        if code not in universe:
                            universe[code] = StockRecord(code=code, name="", market="CN")
                    
                    if self.verbose:
                        print(f"    ✓ {index_name}: {len(df)} 只")
            except Exception as e:
                if self.verbose:
                    print(f"    ✗ {index_name} 获取失败: {e}")
        
        # 获取股票基本信息
        try:
            stock_basic = ts_pro.stock_basic(exchange='', list_status='L')
            if stock_basic is not None:
                for code in universe:
                    info = stock_basic[stock_basic['ts_code'] == code]
                    if len(info) > 0:
                        universe[code].name = info.iloc[0]['name']
                        universe[code].industry = info.iloc[0].get('industry', '')
        except Exception:
            pass
        
        return universe if universe else self._get_default_cn_stocks()
    
    def _fetch_us_constituents(self) -> Dict[str, StockRecord]:
        """获取美股指数成分股"""
        # 纳斯达克100 + 标普500核心股票
        nasdaq100 = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "COST", "NFLX",
            "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTC", "CMCSA", "TXN", "QCOM", "AMGN",
            "INTU", "AMAT", "ISRG", "HON", "BKNG", "VRTX", "SBUX", "GILD", "MDLZ", "ADI",
            "ADP", "REGN", "LRCX", "PANW", "KLAC", "SNPS", "CDNS", "MELI", "ASML", "PYPL",
            "CRWD", "ABNB", "MRVL", "ORLY", "FTNT", "DASH", "MNST", "CTAS", "DXCM", "ODFL"
        ]
        
        sp500_supplement = [
            "JPM", "V", "JNJ", "UNH", "PG", "MA", "HD", "XOM", "CVX", "BAC",
            "MRK", "ABBV", "KO", "PFE", "LLY", "WMT", "DIS", "MCD", "VZ", "NKE",
            "CRM", "TMO", "ABT", "DHR", "ORCL", "ACN", "WFC", "PM", "RTX", "NEE",
            "BMY", "SCHW", "LOW", "UPS", "GS", "MS", "BLK", "SPGI", "AXP", "CAT"
        ]
        
        all_stocks = list(set(nasdaq100 + sp500_supplement))
        
        universe = {}
        for symbol in all_stocks:
            universe[symbol] = StockRecord(code=symbol, name=symbol, market="US")
        
        if self.verbose:
            print(f"  ✓ 美股股票池: {len(universe)} 只 (NASDAQ100 + S&P500核心)")
        
        return universe
    
    def _get_default_cn_stocks(self) -> Dict[str, StockRecord]:
        """默认A股核心股票池"""
        stocks = {
            "600519.SH": ("贵州茅台", "白酒"), "000858.SZ": ("五粮液", "白酒"),
            "601318.SH": ("中国平安", "保险"), "600036.SH": ("招商银行", "银行"),
            "000333.SZ": ("美的集团", "家电"), "600276.SH": ("恒瑞医药", "医药"),
            "601012.SH": ("隆基绿能", "光伏"), "002475.SZ": ("立讯精密", "电子"),
            "300750.SZ": ("宁德时代", "电池"), "600900.SH": ("长江电力", "电力"),
        }
        universe = {}
        for code, (name, industry) in stocks.items():
            universe[code] = StockRecord(code=code, name=name, market="CN", industry=industry)
        return universe
    
    # ==================== 价格数据获取 ====================
    
    def _fetch_price_data(self, universe: Dict[str, StockRecord]) -> Dict[str, StockRecord]:
        """获取价格数据（带缓存）"""
        if self.verbose:
            print(f"\n  📈 获取价格数据...")
        
        if self.market == "CN":
            return self._fetch_cn_prices(universe)
        else:
            return self._fetch_us_prices(universe)
    
    def _fetch_cn_prices(self, universe: Dict[str, StockRecord]) -> Dict[str, StockRecord]:
        """获取A股价格数据"""
        ts_pro = self._clients.get('tushare')
        if ts_pro is None:
            return universe
        
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        success = 0
        
        for code, stock in universe.items():
            # 检查缓存
            if self.storage.is_cached(self.market, "price", code, self.cache_hours):
                cached = self.storage.load_dataframe(self.market, "price", code)
                if cached is not None:
                    stock.price_data = cached
                    success += 1
                    continue
            
            try:
                df = ts_pro.daily(ts_code=code, start_date=start_str, end_date=end_str)
                if df is not None and len(df) > 0:
                    df = df.sort_values('trade_date')
                    df['date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('date')
                    df = df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'vol': 'Volume'
                    })
                    stock.price_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    # 持久化
                    self.storage.save_dataframe(stock.price_data, self.market, "price", code)
                    success += 1
            except Exception:
                pass
        
        if self.verbose:
            print(f"    ✓ A股价格数据: {success}/{len(universe)}")
        return universe
    
    def _fetch_us_prices(self, universe: Dict[str, StockRecord]) -> Dict[str, StockRecord]:
        """获取美股价格数据"""
        yf = self._clients.get('yfinance')
        if yf is None:
            return universe
        
        symbols = list(universe.keys())
        
        # 检查哪些需要下载
        to_download = []
        for symbol in symbols:
            if self.storage.is_cached(self.market, "price", symbol, self.cache_hours):
                cached = self.storage.load_dataframe(self.market, "price", symbol)
                if cached is not None:
                    universe[symbol].price_data = cached
                    continue
            to_download.append(symbol)
        
        cached_count = len(symbols) - len(to_download)
        if cached_count > 0 and self.verbose:
            print(f"    ✓ 从缓存加载: {cached_count} 只")
        
        if to_download:
            try:
                if self.verbose:
                    print(f"    ⏳ 下载中: {len(to_download)} 只...")
                
                data = yf.download(
                    to_download,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                success = 0
                for symbol in to_download:
                    try:
                        if len(to_download) > 1 and isinstance(data.columns, pd.MultiIndex):
                            stock_df = data.xs(symbol, level=1, axis=1)
                        elif len(to_download) == 1:
                            stock_df = data
                        else:
                            stock_df = data[symbol] if symbol in data.columns else None
                        
                        if stock_df is not None and len(stock_df.dropna()) > 20:
                            stock_df = stock_df.dropna()
                            universe[symbol].price_data = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                            
                            # 持久化
                            self.storage.save_dataframe(
                                universe[symbol].price_data, self.market, "price", symbol
                            )
                            success += 1
                    except Exception:
                        pass
                
                if self.verbose:
                    print(f"    ✓ 新下载: {success}/{len(to_download)}")
            except Exception as e:
                if self.verbose:
                    print(f"    ✗ 批量下载失败: {e}")
        
        total_valid = sum(1 for s in universe.values() if s.price_data is not None)
        if self.verbose:
            print(f"    ✓ 总计有效价格数据: {total_valid}/{len(universe)}")
        
        return universe
    
    # ==================== 基准数据获取 ====================
    
    def _fetch_benchmark_data(self) -> Optional[pd.DataFrame]:
        """获取基准指数数据"""
        if self.verbose:
            print(f"\n  📊 获取基准数据: {self.config.benchmark_symbol}")
        
        # 检查缓存
        if self.storage.is_cached(self.market, "benchmark", self.config.benchmark_symbol, self.cache_hours):
            cached = self.storage.load_dataframe(self.market, "benchmark", self.config.benchmark_symbol)
            if cached is not None:
                if self.verbose:
                    print(f"    ✓ 从缓存加载基准数据: {len(cached)} 条")
                return cached
        
        yf = self._clients.get('yfinance')
        if yf is None:
            return None
        
        try:
            benchmark = yf.download(
                self.config.benchmark_symbol,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if benchmark is not None and len(benchmark) > 0:
                # 处理MultiIndex列
                if isinstance(benchmark.columns, pd.MultiIndex):
                    benchmark.columns = benchmark.columns.get_level_values(0)
                
                self.storage.save_dataframe(
                    benchmark, self.market, "benchmark", self.config.benchmark_symbol
                )
                if self.verbose:
                    print(f"    ✓ 基准数据: {len(benchmark)} 条")
                return benchmark
        except Exception as e:
            if self.verbose:
                print(f"    ✗ 基准数据获取失败: {e}")
        
        return None
    
    # ==================== 财务数据获取 ====================
    
    def _fetch_financial_data(self, universe: Dict[str, StockRecord]) -> Dict[str, StockRecord]:
        """获取财务数据"""
        if self.verbose:
            print(f"\n  💰 计算财务指标...")
        
        for code, stock in universe.items():
            if stock.price_data is None or len(stock.price_data) < 20:
                continue
            
            prices = stock.price_data['Close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < 10:
                continue
            
            stock.financial_data = {
                'annual_return': float(returns.mean() * 252),
                'annual_volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-8)),
                'max_drawdown': float(self._calc_max_drawdown(prices)),
                'avg_volume': float(stock.price_data['Volume'].mean()),
                'latest_price': float(prices.iloc[-1]),
                'price_52w_high': float(prices.tail(252).max()) if len(prices) >= 252 else float(prices.max()),
                'price_52w_low': float(prices.tail(252).min()) if len(prices) >= 252 else float(prices.min()),
                'return_1m': float(prices.pct_change(21).iloc[-1]) if len(prices) > 21 else 0,
                'return_3m': float(prices.pct_change(63).iloc[-1]) if len(prices) > 63 else 0,
                'return_6m': float(prices.pct_change(126).iloc[-1]) if len(prices) > 126 else 0,
                'return_1y': float(prices.pct_change(252).iloc[-1]) if len(prices) > 252 else 0,
            }
        
        valid = sum(1 for s in universe.values() if s.financial_data)
        if self.verbose:
            print(f"    ✓ 财务指标计算完成: {valid} 只")
        
        return universe
    
    def _calc_max_drawdown(self, prices: pd.Series) -> float:
        """计算最大回撤"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    # ==================== 宏观数据获取 ====================
    
    def _fetch_macro_data(self) -> Dict[str, pd.Series]:
        """获取宏观经济数据"""
        if self.verbose:
            print(f"\n  🌍 获取宏观数据...")
        
        macro = {}
        yf = self._clients.get('yfinance')
        
        if yf:
            # 通过yfinance获取关键市场指标
            macro_symbols = {
                'VIX': '^VIX',        # 恐慌指数
                'DXY': 'DX-Y.NYB',    # 美元指数
                'TNX': '^TNX',         # 10年期美债收益率
                'GOLD': 'GC=F',       # 黄金
                'OIL': 'CL=F',        # 原油
            }
            
            for name, symbol in macro_symbols.items():
                try:
                    data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                    if data is not None and len(data) > 0:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        macro[name] = data['Close']
                except Exception:
                    pass
        
        if self.verbose:
            print(f"    ✓ 宏观指标: {len(macro)} 个 ({', '.join(macro.keys())})")
        
        return macro
    
    # ==================== 面板数据构建 ====================
    
    def _build_panel_data(self, universe: Dict[str, StockRecord]) -> Optional[pd.DataFrame]:
        """构建面板数据（用于因子计算）"""
        if self.verbose:
            print(f"\n  🔧 构建面板数据...")
        
        panels = []
        for code, stock in universe.items():
            if stock.price_data is None or len(stock.price_data) < 20:
                continue
            
            df = stock.price_data.copy()
            df['stock_code'] = code
            df['stock_name'] = stock.name
            df['industry'] = stock.industry
            
            # 计算基础衍生指标
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['turnover'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
            df['momentum_20d'] = df['Close'].pct_change(20)
            df['momentum_60d'] = df['Close'].pct_change(60)
            
            panels.append(df)
        
        if panels:
            panel = pd.concat(panels, axis=0)
            panel = panel.reset_index()
            if 'Date' in panel.columns:
                panel = panel.rename(columns={'Date': 'date'})
            elif 'index' in panel.columns:
                panel = panel.rename(columns={'index': 'date'})
            
            if self.verbose:
                print(f"    ✓ 面板数据: {len(panel)} 行, {len(panel['stock_code'].unique())} 只股票")
            
            return panel
        
        return None
    
    # ==================== 数据清洗接口 ====================
    
    def clean_panel(self, panel: pd.DataFrame, factor_columns: List[str] = None) -> pd.DataFrame:
        """对面板数据进行清洗"""
        return self.cleaner.clean_pipeline(
            panel, columns=factor_columns,
            winsorize_method='mad', fill_method='ffill',
            standardize_method='zscore', by_date=True, date_col='date'
        )
    
    def get_benchmark_returns(self, bundle: UnifiedDataBundle) -> Optional[pd.Series]:
        """获取基准收益率序列"""
        if bundle.benchmark_data is not None and 'Close' in bundle.benchmark_data.columns:
            return bundle.benchmark_data['Close'].pct_change().dropna()
        return None


# ==================== 便捷函数 ====================

def fetch_data(market: str = "US", lookback_years: int = 3, 
               stock_pool: List[str] = None, verbose: bool = True) -> UnifiedDataBundle:
    """
    便捷函数：获取指定市场的完整数据
    
    Args:
        market: 市场类型 ("CN" 或 "US")
        lookback_years: 历史数据回溯年数
        stock_pool: 指定股票池（可选）
        verbose: 是否打印详细信息
    
    Returns:
        UnifiedDataBundle: 统一数据包
    
    示例:
        # 获取美股数据
        bundle = fetch_data("US")
        
        # 获取A股数据
        bundle = fetch_data("CN", lookback_years=3)
        
        # 获取指定股票
        bundle = fetch_data("US", stock_pool=["AAPL", "MSFT", "NVDA"])
    """
    layer = UnifiedDataLayer(market=market, lookback_years=lookback_years, verbose=verbose)
    return layer.fetch_all(stock_pool=stock_pool)


if __name__ == "__main__":
    print("=" * 60)
    print("V6.0 统一数据层测试")
    print("=" * 60)
    
    # 测试美股数据获取
    bundle = fetch_data("US", lookback_years=1, stock_pool=["AAPL", "MSFT", "NVDA"])
    
    print(f"\n数据统计: {bundle.stats}")
    
    if bundle.panel_data is not None:
        print(f"\n面板数据预览:")
        print(bundle.panel_data.head())
