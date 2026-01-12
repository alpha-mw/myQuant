from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant_platform.data.tushare_client import IFIndClient, TushareClient


@dataclass
class DataBundle:
    prices: Dict[str, pd.DataFrame]
    market_caps: Dict[str, pd.DataFrame]
    industries: pd.DataFrame
    benchmark: pd.DataFrame
    members: pd.DataFrame


class DataManager:
    def __init__(
        self,
        data_dir: Path,
        tushare_client: Optional[TushareClient],
        ifind_client: Optional[IFIndClient] = None,
    ) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tushare_client = tushare_client
        self.ifind_client = ifind_client or IFIndClient(enabled=False)

    def load_or_fetch(self, start_date: str, end_date: str, benchmark_code: str) -> DataBundle:
        members = self._load_members(benchmark_code, start_date, end_date)
        if members.empty:
            raise RuntimeError("HS300 member list not available. Provide data/hs300_members.csv.")
        industries = self._load_industries()
        prices, market_caps = self._load_price_data(members, start_date, end_date)
        benchmark = self._load_benchmark(benchmark_code, start_date, end_date)
        return DataBundle(prices=prices, market_caps=market_caps, industries=industries, benchmark=benchmark, members=members)

    def _load_members(self, benchmark_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_path = self.data_dir / "hs300_members.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        if not self.tushare_client:
            return pd.DataFrame()
        data = self.tushare_client.fetch_index_members(benchmark_code, start_date, end_date)
        data.to_csv(cache_path, index=False)
        return data

    def _load_industries(self) -> pd.DataFrame:
        cache_path = self.data_dir / "stock_basic.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        if not self.tushare_client:
            return pd.DataFrame()
        data = self.tushare_client.fetch_stock_basic()
        data.to_csv(cache_path, index=False)
        return data

    def _load_benchmark(self, benchmark_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_path = self.data_dir / f"benchmark_{benchmark_code}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        if not self.tushare_client:
            return pd.DataFrame()
        data = self.tushare_client.fetch_benchmark(benchmark_code, start_date, end_date)
        data.to_csv(cache_path, index=False)
        return data

    def _load_price_data(
        self, members: pd.DataFrame, start_date: str, end_date: str
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        prices: Dict[str, pd.DataFrame] = {}
        market_caps: Dict[str, pd.DataFrame] = {}
        codes = sorted(members["con_code"].unique().tolist())
        for ts_code in codes:
            price_path = self.data_dir / f"{ts_code}_daily.csv"
            cap_path = self.data_dir / f"{ts_code}_basic.csv"
            if price_path.exists():
                prices[ts_code] = pd.read_csv(price_path)
            else:
                prices[ts_code] = self._fetch_daily(ts_code, start_date, end_date)
                prices[ts_code].to_csv(price_path, index=False)
            if cap_path.exists():
                market_caps[ts_code] = pd.read_csv(cap_path)
            else:
                market_caps[ts_code] = self._fetch_market_cap(ts_code, start_date, end_date)
                market_caps[ts_code].to_csv(cap_path, index=False)
        return prices, market_caps

    def _fetch_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.tushare_client:
            daily = self.tushare_client.fetch_daily(ts_code, start_date, end_date)
            adj = self.tushare_client.fetch_adj_factor(ts_code, start_date, end_date)
            return self._merge_adjusted(daily, adj)
        return self.ifind_client.fetch_daily(ts_code, start_date, end_date)

    def _fetch_market_cap(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.tushare_client:
            return pd.DataFrame()
        return self.tushare_client.fetch_daily_basic(ts_code, start_date, end_date)

    @staticmethod
    def _merge_adjusted(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
        if daily.empty:
            return daily
        merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
        merged["adj_factor"] = merged["adj_factor"].fillna(method="ffill").fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            merged[col] = merged[col] * merged["adj_factor"]
        return merged
