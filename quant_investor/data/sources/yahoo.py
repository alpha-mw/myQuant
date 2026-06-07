"""Yahoo fallback source.

This module is importable offline; network access is attempted only when a
caller explicitly invokes the source methods in an environment with yfinance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_investor.data.models import FundamentalData
from quant_investor.data.sources.base import DataSourceBase, _filter_ohlcv_by_date, _normalize_ohlcv_frame


class YahooDataSource(DataSourceBase):
    source_name = "public_structured_fallback"

    def __init__(self) -> None:
        self.last_ohlcv_source = "unknown"
        self.last_fundamental_source = "unknown"

    def get_ohlcv(self, symbol: str, start_date: str = "", end_date: str = "", freq: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except Exception:
            self.last_ohlcv_source = "yahoo_unavailable"
            return pd.DataFrame()
        try:
            frame = yf.download(symbol, start=start_date or None, end=end_date or None, progress=False, auto_adjust=False)
        except Exception:
            self.last_ohlcv_source = "yahoo_error"
            return pd.DataFrame()
        normalized = _normalize_ohlcv_frame(frame.reset_index() if isinstance(frame, pd.DataFrame) else frame)
        self.last_ohlcv_source = "yahoo" if not normalized.empty else "yahoo_empty"
        return _filter_ohlcv_by_date(normalized, start_date, end_date)

    def get_fundamental(self, symbol: str) -> FundamentalData:
        try:
            import yfinance as yf  # type: ignore
        except Exception:
            self.last_fundamental_source = "yahoo_unavailable"
            return FundamentalData(symbol=symbol, source=self.last_fundamental_source)
        try:
            info: dict[str, Any] = dict(yf.Ticker(symbol).info or {})
        except Exception:
            self.last_fundamental_source = "yahoo_error"
            return FundamentalData(symbol=symbol, source=self.last_fundamental_source)
        self.last_fundamental_source = "yahoo"
        return FundamentalData(
            symbol=symbol,
            pe=info.get("trailingPE"),
            pb=info.get("priceToBook"),
            ps=info.get("priceToSalesTrailing12Months"),
            dividend_yield=info.get("dividendYield"),
            gross_margin=info.get("grossMargins"),
            net_margin=info.get("profitMargins"),
            revenue_growth=info.get("revenueGrowth"),
            profit_growth=info.get("earningsGrowth"),
            debt_ratio=info.get("debtToEquity"),
            source=self.last_fundamental_source,
        )
