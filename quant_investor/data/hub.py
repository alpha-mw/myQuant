"""Public DataHub facade for source-backed market data access."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_investor.data.models import FundamentalData, MacroData
from quant_investor.data.sources.macro import MacroDataSource
from quant_investor.data.sources.tushare_cn import TushareDataSource
from quant_investor.data.sources.yahoo import YahooDataSource


class DataHub:
    """Thin compatibility facade around priority-ordered data sources."""

    def __init__(self, market: str = "CN", verbose: bool = True) -> None:
        self.market = str(market or "CN").upper()
        self.verbose = bool(verbose)
        self._source = TushareDataSource(allow_live=False) if self.market == "CN" else YahooDataSource()
        self._macro_source = MacroDataSource()
        self.last_ohlcv_source = "unknown"
        self.last_fundamental_source = "unknown"
        self.last_daily_basic_status = "unknown"
        self.last_daily_basic_source = "unknown"
        self.last_daily_basic_reason = ""

    def get_ohlcv(self, symbol: str, start_date: str = "", end_date: str = "", freq: str = "1d") -> pd.DataFrame:
        frame = self._source.get_ohlcv(symbol, start_date, end_date, freq=freq)
        self.last_ohlcv_source = str(getattr(self._source, "last_ohlcv_source", self._source.source_name))
        return frame

    def get_fundamental(self, symbol: str) -> FundamentalData:
        result = self._source.get_fundamental(symbol)
        self.last_fundamental_source = str(getattr(self._source, "last_fundamental_source", getattr(result, "source", "")) or "unknown")
        return result

    def get_daily_basic(self, symbol: str, trade_date: str | None = None) -> dict[str, Any]:
        payload = self._source.get_daily_basic(symbol, trade_date)
        self.last_daily_basic_status = str(getattr(self._source, "last_daily_basic_status", "available" if payload else "unknown"))
        self.last_daily_basic_source = str(getattr(self._source, "last_daily_basic_source", "unknown"))
        self.last_daily_basic_reason = str(getattr(self._source, "last_daily_basic_reason", ""))
        return payload

    def get_macro(self, market: str | None = None, as_of: str = "") -> MacroData:
        return self._macro_source.get_macro(market or self.market, as_of=as_of)

    def query_tushare(self, api_name: str, **kwargs: Any) -> Any:
        client = getattr(self._source, "_client", None)
        if client is None:
            raise RuntimeError("active source has no Tushare client")
        return client.query(api_name, **kwargs)

    def fetch_and_process(self, symbol: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        return self.get_ohlcv(symbol, start_date, end_date)


def get_data_hub(market: str = "CN", verbose: bool = True) -> DataHub:
    return DataHub(market=market, verbose=verbose)
