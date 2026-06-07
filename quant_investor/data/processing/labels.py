"""Label generation helpers for public imports."""

from __future__ import annotations

import pandas as pd


class LabelGenerator:
    def forward_return(self, frame: pd.DataFrame, horizon: int = 5) -> pd.Series:
        if frame is None or frame.empty or "close" not in frame.columns:
            return pd.Series(dtype=float)
        close = pd.to_numeric(frame["close"], errors="coerce")
        return close.shift(-int(horizon)).div(close).sub(1.0)
