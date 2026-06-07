"""Feature engineering helpers for public imports."""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    def add_returns(self, frame: pd.DataFrame, periods: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
        if frame is None or frame.empty or "close" not in frame.columns:
            return pd.DataFrame() if frame is None else frame.copy()
        out = frame.copy()
        close = pd.to_numeric(out["close"], errors="coerce")
        for period in periods:
            out[f"return_{int(period)}d"] = close.pct_change(int(period))
        return out
