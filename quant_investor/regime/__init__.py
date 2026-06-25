"""Deterministic Markov-style market regime engine."""

from quant_investor.regime.engine import MarkovRegimeEngine
from quant_investor.regime.types import (
    REGIME_RANGE_HIGH_VOL,
    REGIME_RANGE_LOW_VOL,
    REGIME_STATES,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_UNKNOWN,
    RegimeFeatureSnapshot,
    RegimeSignal,
)

__all__ = [
    "MarkovRegimeEngine",
    "REGIME_TREND_UP",
    "REGIME_RANGE_LOW_VOL",
    "REGIME_RANGE_HIGH_VOL",
    "REGIME_TREND_DOWN",
    "REGIME_UNKNOWN",
    "REGIME_STATES",
    "RegimeFeatureSnapshot",
    "RegimeSignal",
]
