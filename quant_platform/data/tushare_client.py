from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import tushare as ts


@dataclass
class TushareConfig:
    token: str


class TushareClient:
    def __init__(self, config: TushareConfig) -> None:
        ts.set_token(config.token)
        self.pro = ts.pro_api()

    def fetch_index_members(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        data_frames = []
        current = start_date
        while current <= end_date:
            frame = self.pro.index_weight(index_code=index_code, start_date=current, end_date=end_date)
            if frame.empty:
                break
            data_frames.append(frame)
            current = (pd.to_datetime(frame["trade_date"].max()) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if not data_frames:
            return pd.DataFrame()
        data = pd.concat(data_frames, ignore_index=True)
        return data.drop_duplicates(subset=["trade_date", "con_code"])

    def fetch_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_adj_factor(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields="ts_code,trade_date,total_mv")

    def fetch_stock_basic(self) -> pd.DataFrame:
        return self.pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry,name")

    def fetch_benchmark(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)


class IFIndClient:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def fetch_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.enabled:
            raise RuntimeError("iFinD client not configured. Provide iFinD credentials to enable.")
        raise NotImplementedError("iFinD integration placeholder.")


def build_tushare_client(token: Optional[str]) -> Optional[TushareClient]:
    if not token:
        return None
    return TushareClient(TushareConfig(token=token))
