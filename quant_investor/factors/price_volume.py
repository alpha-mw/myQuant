"""Runtime price/volume factor implementations for governed mined factors."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _numeric_series(
    frame: pd.DataFrame, candidates: tuple[str, ...]
) -> pd.Series:
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
    return (
        _numeric_series(working, ("vol", "volume", "Volume"))
        .replace(0.0, np.nan)
        .dropna()
    )


def _amount_series(frame: pd.DataFrame) -> pd.Series:
    working = _ordered_frame(frame)
    amount = _numeric_series(
        working, ("amount", "turnover", "dollar_volume")
    ).replace(0.0, np.nan)
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


def _smooth_windows_from_name(name: str) -> tuple[int, int]:
    parts = str(name).strip().split("_")
    try:
        base = int(parts[-2].removesuffix("d"))
        smooth = int(parts[-1].removesuffix("d"))
        return max(base, 1), max(smooth, 1)
    except Exception:
        return 20, 5


def _weight_from_name(name: str, default: float = 0.75) -> float:
    try:
        suffix = str(name).rsplit("_w", 1)[1]
        return max(0.0, min(float(suffix) / 100.0, 1.0))
    except Exception:
        return default


def _short_reversal(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    if len(close) <= window:
        return np.nan
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-12:
        return np.nan
    return -((latest / base) - 1.0)


def _volatility_penalty(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    returns = close.pct_change().dropna().tail(window)
    if len(returns) < max(3, min(window, 5)):
        return np.nan
    return -float(returns.std())


def _downside_volatility(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    returns = close.pct_change().dropna().tail(window)
    if len(returns) < max(3, min(window, 5)):
        return np.nan
    downside = returns.where(returns < 0.0, 0.0)
    return -float(downside.std())


def _price_efficiency(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    if len(close) <= window:
        return np.nan
    window_close = close.tail(window + 1)
    base = float(window_close.iloc[0])
    latest = float(window_close.iloc[-1])
    if abs(base) <= 1e-12:
        return np.nan
    net_move = abs((latest / base) - 1.0)
    path = float(window_close.pct_change().abs().sum())
    if path <= 1e-12:
        return np.nan
    return net_move / path


def _volume_stability(frame: pd.DataFrame, window: int) -> float:
    volume = _volume_series(frame).tail(window)
    if len(volume) < max(3, min(window, 5)):
        return np.nan
    mean = float(volume.mean())
    if mean <= 1e-12:
        return np.nan
    return -(float(volume.std(ddof=0)) / mean)


def _volume_stability_smooth(
    frame: pd.DataFrame,
    base_window: int,
    smooth_window: int,
) -> float:
    volume = _volume_series(frame)
    min_base = max(3, min(base_window, 5))
    if len(volume) < min_base:
        return np.nan
    mean = volume.rolling(base_window, min_periods=min_base).mean()
    std = volume.rolling(base_window, min_periods=min_base).std(ddof=0)
    raw = -(std.div(mean.replace(0.0, np.nan)))
    values = (
        raw.rolling(
            smooth_window,
            min_periods=max(1, min(smooth_window, 3)),
        )
        .mean()
        .dropna()
    )
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def _low_dollar_volume(frame: pd.DataFrame, window: int) -> float:
    amount = _amount_series(frame).tail(window)
    if len(amount) < max(3, min(window, 5)):
        return np.nan
    average_amount = float(amount.mean())
    if average_amount <= 1e-12:
        return np.nan
    return -float(np.log(average_amount))


def _high_dollar_volume(frame: pd.DataFrame, window: int) -> float:
    amount = _amount_series(frame).tail(window)
    if len(amount) < max(3, min(window, 5)):
        return np.nan
    average_amount = float(amount.mean())
    if average_amount <= 1e-12:
        return np.nan
    return float(np.log(average_amount))


def _dollar_volume_growth(
    frame: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> float:
    amount = _amount_series(frame)
    short = amount.tail(short_window)
    long = amount.tail(long_window)
    if len(short) < max(3, min(short_window, 5)):
        return np.nan
    if len(long) < max(3, min(long_window, 5)):
        return np.nan
    long_avg = float(long.mean())
    if long_avg <= 1e-12:
        return np.nan
    return float(short.mean()) / long_avg - 1.0


def _amihud_illiquidity(frame: pd.DataFrame, window: int) -> float:
    working = _ordered_frame(frame)
    close = _numeric_series(working, ("adj_close", "close", "Close"))
    amount = _numeric_series(
        working, ("amount", "turnover", "dollar_volume")
    ).replace(0.0, np.nan)
    if amount.dropna().empty:
        volume = _numeric_series(working, ("vol", "volume", "Volume"))
        amount = close.mul(volume).replace(0.0, np.nan)
    returns = close.pct_change().abs()
    values = (
        returns.div(amount)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(window)
    )
    if len(values) < max(3, min(window, 5)):
        return np.nan
    return float(values.mean())


def _momentum(frame: pd.DataFrame, window: int) -> float:
    close = _close_series(frame)
    if len(close) <= window:
        return np.nan
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-12:
        return np.nan
    return (latest / base) - 1.0


def _growth_windows_from_name(name: str) -> tuple[int, int]:
    parts = str(name).strip().split("_")
    try:
        short_window = int(parts[-2].removesuffix("d"))
        long_window = int(parts[-1].removesuffix("d"))
        return max(short_window, 1), max(long_window, 1)
    except Exception:
        return 20, 60


def _rank_blend_volstab_momentum_amihud(
    frames: Mapping[str, pd.DataFrame],
    *,
    outer_weight: float,
) -> pd.Series:
    vol_stability: dict[str, float] = {}
    momentum: dict[str, float] = {}
    amihud: dict[str, float] = {}
    for symbol, frame in frames.items():
        key = str(symbol)
        vol_stability[key] = _volume_stability_smooth(
            frame,
            base_window=19,
            smooth_window=2,
        )
        momentum[key] = _momentum(frame, 90)
        amihud[key] = _amihud_illiquidity(frame, 5)
    vol_rank = pd.Series(vol_stability, dtype=float).rank(pct=True)
    momentum_rank = pd.Series(momentum, dtype=float).rank(pct=True)
    amihud_rank = pd.Series(amihud, dtype=float).rank(pct=True)
    inner = momentum_rank.mul(0.60).add(amihud_rank.mul(0.40))
    inner_rank = inner.rank(pct=True)
    return vol_rank.mul(outer_weight).add(
        inner_rank.mul(1.0 - outer_weight)
    )


def compute_price_volume_factor(
    name: str, frames: Mapping[str, pd.DataFrame]
) -> pd.Series:
    """Compute the latest cross-sectional raw value.

    The input factor name selects one governed price/volume implementation.
    """

    factor_name = str(name).strip()
    if factor_name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
        return _rank_blend_volstab_momentum_amihud(
            frames,
            outer_weight=_weight_from_name(factor_name, 0.75),
        )
    window = _window_from_name(factor_name)
    values: dict[str, float] = {}
    for symbol, frame in frames.items():
        if factor_name.startswith("pv_momentum_"):
            value = _momentum(frame, window)
        elif factor_name.startswith("pv_short_reversal_"):
            value = _short_reversal(frame, window)
        elif factor_name.startswith("pv_volume_stability_smooth_"):
            base_window, smooth_window = _smooth_windows_from_name(factor_name)
            value = _volume_stability_smooth(frame, base_window, smooth_window)
        elif factor_name.startswith("pv_volume_stability_"):
            value = _volume_stability(frame, window)
        elif factor_name.startswith("pv_low_dollar_volume_"):
            value = _low_dollar_volume(frame, window)
        elif factor_name.startswith("pv_high_dollar_volume_"):
            value = _high_dollar_volume(frame, window)
        elif factor_name.startswith("pv_amihud_illiquidity_"):
            value = _amihud_illiquidity(frame, window)
        elif factor_name.startswith("pv_volatility_penalty_"):
            value = _volatility_penalty(frame, window)
        elif factor_name.startswith("pv_downside_volatility_"):
            value = _downside_volatility(frame, window)
        elif factor_name.startswith("pv_price_efficiency_"):
            value = _price_efficiency(frame, window)
        elif factor_name.startswith("pv_dollar_volume_growth_"):
            short_window, long_window = _growth_windows_from_name(factor_name)
            value = _dollar_volume_growth(frame, short_window, long_window)
        else:
            raise ValueError(f"unknown price/volume factor: {factor_name}")
        values[str(symbol)] = value
    return pd.Series(values, dtype=float)


__all__ = ["compute_price_volume_factor"]
