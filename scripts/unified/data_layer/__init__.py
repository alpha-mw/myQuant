#!/usr/bin/env python3
"""
统一数据层 - 简化版
整合 V2.7 持久化 + V6.0 数据层核心功能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# 尝试导入 yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
except:
    YF_AVAILABLE = False


@dataclass
class UnifiedDataBundle:
    """统一数据包"""
    market: str
    stock_universe: Dict[str, Any] = field(default_factory=dict)
    panel_data: Optional[pd.DataFrame] = None
    focus_stocks: Optional[List[str]] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class UnifiedDataLayer:
    """统一数据层"""
    
    def __init__(self, market: str = "US", lookback_years: int = 1, verbose: bool = True):
        self.market = market.upper()
        self.lookback_years = lookback_years
        self.verbose = verbose
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * lookback_years)
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DataLayer] {msg}")
    
    def fetch_all(self, stock_pool: Optional[List[str]] = None) -> UnifiedDataBundle:
        """获取所有数据"""
        bundle = UnifiedDataBundle(market=self.market, focus_stocks=stock_pool)
        
        self._log(f"获取 {self.market} 市场数据...")
        
        # 获取股票数据
        if self.market == "US" and YF_AVAILABLE:
            symbols = stock_pool or ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            bundle = self._fetch_us_data(bundle, symbols)
        else:
            # 模拟数据
            bundle = self._generate_mock_data(bundle, stock_pool or ["STOCK1", "STOCK2"])
        
        return bundle
    
    def _fetch_us_data(self, bundle: UnifiedDataBundle, symbols: List[str]) -> UnifiedDataBundle:
        """获取美股数据"""
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date, end=self.end_date)
                
                if not hist.empty:
                    hist['stock_code'] = symbol
                    hist['date'] = hist.index
                    all_data.append(hist)
                    bundle.stock_universe[symbol] = {'code': symbol, 'name': symbol}
            except Exception as e:
                self._log(f"获取 {symbol} 失败: {e}")
        
        if all_data:
            bundle.panel_data = pd.concat(all_data, ignore_index=True)
            bundle.stats['total_rows'] = len(bundle.panel_data)
            bundle.stats['unique_stocks'] = len(bundle.stock_universe)
        
        return bundle
    
    def _generate_mock_data(self, bundle: UnifiedDataBundle, symbols: List[str]) -> UnifiedDataBundle:
        """生成模拟数据"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        all_data = []
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'date': dates,
                'stock_code': symbol,
                'Open': prices * 0.99,
                'High': prices * 1.02,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates)),
            })
            all_data.append(df)
            bundle.stock_universe[symbol] = {'code': symbol, 'name': symbol}
        
        bundle.panel_data = pd.concat(all_data, ignore_index=True)
        bundle.stats['total_rows'] = len(bundle.panel_data)
        bundle.stats['unique_stocks'] = len(bundle.stock_universe)
        
        return bundle
    
    def get_benchmark_returns(self, bundle: UnifiedDataBundle) -> Optional[pd.Series]:
        """获取基准收益"""
        if bundle.panel_data is None or bundle.panel_data.empty:
            return None
        
        # 使用第一只股票作为简单基准
        first_stock = list(bundle.stock_universe.keys())[0]
        stock_data = bundle.panel_data[bundle.panel_data['stock_code'] == first_stock]
        
        if 'Close' in stock_data.columns:
            prices = stock_data.set_index('date')['Close'].sort_index()
            return prices.pct_change().dropna()
        
        return None
