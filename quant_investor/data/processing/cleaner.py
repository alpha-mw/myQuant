"""Data cleaning helpers for public imports."""

from __future__ import annotations

import pandas as pd

from quant_investor.data.sources.base import _normalize_ohlcv_frame


class DataCleaner:
    def clean_ohlcv(self, frame: pd.DataFrame | None) -> pd.DataFrame:
        return _normalize_ohlcv_frame(frame)
