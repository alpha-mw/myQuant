"""Thin wrapper around the explicit Kline hybrid engine."""

from __future__ import annotations

import pandas as pd

from quant_investor.branch_contracts import BranchResult

from .base import KLineBackend
from .hybrid_engine import KlineHybridEngine


class HybridBackend(KLineBackend):
    """K-line production backend delegating to the hybrid engine."""

    name = "hybrid"
    reliability = 0.84
    horizon_days = 5

    def __init__(self, evaluator_name: str | None = None, **kwargs: object) -> None:
        self._engine = KlineHybridEngine(evaluator_name=evaluator_name, **kwargs)

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        signal = self._engine.predict(symbol_data, stock_pool)
        return signal.branch_result
