#!/usr/bin/env python3
"""
Enhanced Data Layer - 增强版数据层 (向后兼容 shim)

所有逻辑已迁移至 data/ 包。本文件保留向后兼容的导入接口。

新代码请使用:
    from quant_investor.data import DataHub, get_data_hub
    from quant_investor.data import DataSourceBase, TushareDataSource, YahooDataSource
    from quant_investor.data import OHLCVData, FundamentalData, MacroData
"""

from pathlib import Path

# ==================== 数据结构 ====================
from quant_investor.data.models import OHLCVData, TickData, FundamentalData, MacroData

# ==================== 数据获取基类 ====================
from quant_investor.data.sources.base import (
    DataSourceBase,
    _parse_any_date,
    _normalize_ohlcv_frame,
    _filter_ohlcv_by_date,
)

# ==================== 数据源 ====================
from quant_investor.data.sources.tushare_cn import TushareDataSource
from quant_investor.data.sources.tushare_us import USTushareDataSource
from quant_investor.data.sources.yahoo import YahooDataSource

# USLocalCSVDataSource & USCompositeDataSource — inline for backward compat
import pandas as pd

class USLocalCSVDataSource(DataSourceBase):
    """本地美股 CSV 数据源，读取 data/us_market_full。"""

    def __init__(self, data_dir=None):
        from quant_investor.data.storage.csv_store import CSVStore
        default_dir = Path(__file__).resolve().parents[1] / "data" / "us_market_full"
        self._store = CSVStore(data_dir or default_dir)

    def get_ohlcv(self, symbol, start_date, end_date, freq="1d"):
        return self._store.read(symbol, start_date, end_date)

    def get_fundamental(self, symbol):
        return FundamentalData(symbol=symbol)


class USCompositeDataSource(DataSourceBase):
    """美股复合数据源：本地缓存优先，刷新回补走 Tushare，最后回退 Yahoo。"""

    def __init__(self, local_data_dir=None, max_staleness_days=7):
        self._local = USLocalCSVDataSource(local_data_dir)
        self._tushare = USTushareDataSource()
        self._yahoo = YahooDataSource()
        self._max_staleness_days = max_staleness_days
        self.last_ohlcv_source = "unknown"
        self.last_fundamental_source = "unknown"

    def get_ohlcv(self, symbol, start_date, end_date):
        requested_end = _parse_any_date(end_date)
        local_df = self._local.get_ohlcv(symbol, start_date, end_date)
        if not local_df.empty:
            latest_date = pd.to_datetime(local_df["date"], errors="coerce").max()
            if pd.isna(requested_end) or (
                pd.notna(latest_date)
                and latest_date >= requested_end - pd.Timedelta(days=self._max_staleness_days)
            ):
                self.last_ohlcv_source = "local_csv"
                return local_df

        tushare_df = self._tushare.get_ohlcv(symbol, start_date, end_date)
        if not tushare_df.empty:
            self.last_ohlcv_source = "tushare_us_daily"
            return tushare_df

        yahoo_df = self._yahoo.get_ohlcv(symbol, start_date, end_date)
        if not yahoo_df.empty:
            self.last_ohlcv_source = "yahoo"
            return yahoo_df

        if not local_df.empty:
            self.last_ohlcv_source = "local_csv_stale"
            return local_df

        self.last_ohlcv_source = "unavailable"
        return pd.DataFrame()

    def get_fundamental(self, symbol):
        if self.last_ohlcv_source == "yahoo":
            self.last_fundamental_source = "yahoo"
            return self._yahoo.get_fundamental(symbol)
        self.last_fundamental_source = "skipped"
        return FundamentalData(symbol=symbol)


# ==================== 数据处理 ====================
from quant_investor.data.processing.cleaner import DataCleaner
from quant_investor.data.processing.features import FeatureEngineer
from quant_investor.data.processing.labels import LabelGenerator


# ==================== 增强版数据层主类 ====================
from quant_investor.data.hub import DataHub


class EnhancedDataLayer(DataHub):
    """向后兼容别名 — 等价于 DataHub"""
    pass


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Enhanced Data Layer - 测试 (via data/ package)")
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
