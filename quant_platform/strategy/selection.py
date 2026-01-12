from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class SelectionResult:
    holdings: Dict[pd.Timestamp, List[str]]


def select_top_n(scores: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, top_n: int) -> SelectionResult:
    holdings: Dict[pd.Timestamp, List[str]] = {}
    for date in rebalance_dates:
        if date not in scores.index:
            continue
        row = scores.loc[date].dropna()
        top = row.sort_values(ascending=False).head(top_n).index.tolist()
        holdings[date] = top
    return SelectionResult(holdings=holdings)
