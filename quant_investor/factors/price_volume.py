"""Runtime price/volume factor implementations for governed mined factors."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _numeric_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(dtype=float)


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    for column in ("trade_date", "date"):
        if column in working.columns:
            return working.sort_values(column).reset_index(drop=True)
    return working.reset_index(drop=True)


def _close_series(frame: pd.DataFrame) -> pd.Series:
    working = _ordered_frame(frame)
    return _numeric_series(working, ("adj_close", "close", "Close")).dropna()


def _volume_series(frame: pd.DataFrame) -> pd.Series:
    working = _ordered_frame(frame)
    return _numeric_series(working, ("vol", "volume", "Volume")).replace(0.0, np.nan).dropna()


def _amount_series(frame: pd.DataFrame) -> pd.Series:
    working = _ordered_frame(frame)
    amount = _numeric_series(working, ("amount", "turnover", "dollar_volume")).replace(0.0, np.nan)
    if not amount.dropna().empty:
        return amount.dropna()
    close = _numeric_series(working, ("adj_close", "close", "Close"))
    volume = _numeric_series(working, ("vol", "volume", "Volume"))
    return close.mul(volume).replace(0.0, np.nan).dropna()


def _window_from_name(name: str) -> int:
    try:
        suffix = str(name).rsplit("_", 1)[1]
        return max(int(suffix.removesuffix("d")), 1)
    except Exception:
        return 20


def _short_reversal(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    if len(close) <= window:
        return np.nan
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-12:
        return np.nan
    return -((latest / base) - 1.0)


def _volume_stability(frame: pd.DataFrame, window: int) -> float:
    volume = _volume_series(frame).tail(window)
    if len(volume) < max(3, min(window, 5)):
        return np.nan
    mean = float(volume.mean())
    if mean <= 1e-12:
        return np.nan
    return -(float(volume.std(ddof=0)) / mean)


def _low_dollar_volume(frame: pd.DataFrame, window: int) -> float:
    amount = _amount_series(frame).tail(window)
    if len(amount) < max(3, min(window, 5)):
        return np.nan
    average_amount = float(amount.mean())
    if average_amount <= 1e-12:
        return np.nan
    return -float(np.log(average_amount))


def _amihud_illiquidity(frame: pd.DataFrame, window: int) -> float:
    working = _ordered_frame(frame)
    close = _numeric_series(working, ("adj_close", "close", "Close"))
    amount = _numeric_series(working, ("amount", "turnover", "dollar_volume")).replace(0.0, np.nan)
    if amount.dropna().empty:
        volume = _numeric_series(working, ("vol", "volume", "Volume"))
        amount = close.mul(volume).replace(0.0, np.nan)
    returns = close.pct_change().abs()
    values = returns.div(amount).replace([np.inf, -np.inf], np.nan).dropna().tail(window)
    if len(values) < max(3, min(window, 5)):
        return np.nan
    return float(values.mean())


def compute_price_volume_factor(name: str, frames: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Compute the latest cross-sectional raw value for a price/volume factor."""

    factor_name = str(name).strip()
    window = _window_from_name(factor_name)
    values: dict[str, float] = {}
    for symbol, frame in frames.items():
        if factor_name.startswith("pv_short_reversal_"):
            value = _short_reversal(frame, window)
        elif factor_name.startswith("pv_volume_stability_"):
            value = _volume_stability(frame, window)
        elif factor_name.startswith("pv_low_dollar_volume_"):
            value = _low_dollar_volume(frame, window)
        elif factor_name.startswith("pv_amihud_illiquidity_"):
            value = _amihud_illiquidity(frame, window)
        else:
            raise ValueError(f"unknown price/volume factor: {factor_name}")
        values[str(symbol)] = value
    return pd.Series(values, dtype=float)


__all__ = ["compute_price_volume_factor"]
