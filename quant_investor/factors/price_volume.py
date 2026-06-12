"""Runtime price/volume factor implementations for governed mined factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _numeric_series(
    frame: pd.DataFrame, candidates: tuple[str, ...]
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            values = frame[column]
            if pd.api.types.is_numeric_dtype(values.dtype):
                return values
            return pd.to_numeric(values, errors="coerce")
    return pd.Series(dtype=float)


def _ordered_frame(frame: pd.DataFrame, *, lookback_rows: int = 0) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    rows = int(lookback_rows or 0)
    for column in ("trade_date", "date"):
        if column not in frame.columns:
            continue
        values = frame[column]
        if values.is_monotonic_increasing:
            return frame.tail(rows) if rows > 0 and len(frame) > rows else frame
        ordered = frame.sort_values(column).reset_index(drop=True)
        return ordered.tail(rows) if rows > 0 and len(ordered) > rows else ordered
    return frame.tail(rows) if rows > 0 and len(frame) > rows else frame


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


@dataclass(frozen=True)
class _PreparedPriceVolumeFrame:
    close: pd.Series
    volume: pd.Series
    amount: pd.Series
    close_raw: pd.Series
    amount_raw: pd.Series
    amihud_base: pd.Series | None = None


def _amihud_base_from_raw(close_raw: pd.Series, amount_raw: pd.Series) -> pd.Series:
    close_values = close_raw.to_numpy(dtype=float, copy=False)
    amount_values = amount_raw.to_numpy(dtype=float, copy=False)
    base = np.full(close_values.shape, np.nan, dtype=float)
    if close_values.size >= 2:
        previous = close_values[:-1]
        current = close_values[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.abs(current / previous - 1.0)
            base[1:] = returns / amount_values[1:]
    return (
        pd.Series(base, index=close_raw.index, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _prepare_price_volume_frame(
    frame: pd.DataFrame,
    *,
    include_amihud_base: bool = False,
    lookback_rows: int = 0,
) -> _PreparedPriceVolumeFrame:
    working = _ordered_frame(frame, lookback_rows=lookback_rows)
    close_raw = _numeric_series(working, ("adj_close", "close", "Close"))
    volume_raw = _numeric_series(working, ("vol", "volume", "Volume"))
    volume = volume_raw.replace(0.0, np.nan).dropna()
    amount_raw = _numeric_series(
        working,
        ("amount", "turnover", "dollar_volume"),
    ).replace(0.0, np.nan)
    if amount_raw.dropna().empty:
        amount_raw = close_raw.mul(volume_raw).replace(0.0, np.nan)
    amihud_base = (
        _amihud_base_from_raw(close_raw, amount_raw)
        if include_amihud_base
        else None
    )
    return _PreparedPriceVolumeFrame(
        close=close_raw.dropna(),
        volume=volume,
        amount=amount_raw.dropna(),
        close_raw=close_raw,
        amount_raw=amount_raw,
        amihud_base=amihud_base,
    )


def prepare_price_volume_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    include_amihud_base: bool = False,
    lookback_rows: int = 0,
) -> dict[str, _PreparedPriceVolumeFrame]:
    """Prepare price/volume inputs once for a batch of factor calculations."""

    return {
        str(symbol): _prepare_price_volume_frame(
            frame,
            include_amihud_base=include_amihud_base,
            lookback_rows=lookback_rows,
        )
        for symbol, frame in frames.items()
    }


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


def _short_reversal_from_close(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-12:
        return np.nan
    return -((latest / base) - 1.0)


def _volatility_penalty_from_close(close: pd.Series, window: int) -> float:
    returns = close.pct_change().dropna().tail(window)
    if len(returns) < max(3, min(window, 5)):
        return np.nan
    return -float(returns.std())


def _downside_volatility_from_close(close: pd.Series, window: int) -> float:
    returns = close.pct_change().dropna().tail(window)
    if len(returns) < max(3, min(window, 5)):
        return np.nan
    downside = returns.where(returns < 0.0, 0.0)
    return -float(downside.std())


def _price_efficiency_from_close(close: pd.Series, window: int) -> float:
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


def _volume_stability_from_volume(volume: pd.Series, window: int) -> float:
    volume = volume.tail(window)
    if len(volume) < max(3, min(window, 5)):
        return np.nan
    mean = float(volume.mean())
    if mean <= 1e-12:
        return np.nan
    return -(float(volume.std(ddof=0)) / mean)


def _volume_stability_smooth_from_volume(
    volume: pd.Series,
    base_window: int,
    smooth_window: int,
) -> float:
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


def _low_dollar_volume_from_amount(amount: pd.Series, window: int) -> float:
    amount = amount.tail(window)
    if len(amount) < max(3, min(window, 5)):
        return np.nan
    average_amount = float(amount.mean())
    if average_amount <= 1e-12:
        return np.nan
    return -float(np.log(average_amount))


def _active_price_volume_windows(
    *,
    factor_cache: dict[str, Any] | None,
    prefix: str,
    current_name: str,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[int, ...]:
    names = list((factor_cache or {}).get("active_price_volume_names", ()) or ())
    names.append(str(current_name))
    windows = [
        _window_from_name(name)
        for name in names
        if str(name).startswith(prefix)
        and not str(name).startswith(exclude_prefixes)
    ]
    return tuple(sorted(set(windows))) or (_window_from_name(current_name),)


def _low_dollar_volume_values_prepared(
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame],
    *,
    window: int,
    factor_name: str,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    cache_key = "pv_low_dollar_volume_window_values"
    cached = (factor_cache or {}).get(cache_key)
    values_by_window: dict[int, pd.Series] = (
        cached if isinstance(cached, dict) else {}
    )
    requested_windows = _active_price_volume_windows(
        factor_cache=factor_cache,
        prefix="pv_low_dollar_volume_",
        current_name=factor_name,
    )
    missing_windows = [
        candidate for candidate in requested_windows if candidate not in values_by_window
    ]
    if missing_windows:
        computed: dict[int, dict[str, float]] = {
            candidate: {} for candidate in missing_windows
        }
        for symbol, prepared in prepared_frames.items():
            amount_values = prepared.amount.to_numpy(dtype=float, copy=False)
            for candidate in missing_windows:
                min_count = max(3, min(candidate, 5))
                if amount_values.size < min_count:
                    value = np.nan
                else:
                    window_values = amount_values[-candidate:]
                    average_amount = float(np.nanmean(window_values))
                    value = (
                        -float(np.log(average_amount))
                        if average_amount > 1e-12
                        else np.nan
                    )
                computed[candidate][str(symbol)] = value
        values_by_window = dict(values_by_window)
        values_by_window.update(
            {
                candidate: pd.Series(values, dtype=float)
                for candidate, values in computed.items()
            }
        )
        if factor_cache is not None:
            factor_cache[cache_key] = values_by_window
    return pd.Series(values_by_window.get(window, pd.Series(dtype=float)), dtype=float)


def _volume_stability_values_prepared(
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame],
    *,
    window: int,
    factor_name: str,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    cache_key = "pv_volume_stability_window_values"
    cached = (factor_cache or {}).get(cache_key)
    values_by_window: dict[int, pd.Series] = (
        cached if isinstance(cached, dict) else {}
    )
    requested_windows = _active_price_volume_windows(
        factor_cache=factor_cache,
        prefix="pv_volume_stability_",
        current_name=factor_name,
        exclude_prefixes=("pv_volume_stability_smooth_",),
    )
    missing_windows = [
        candidate for candidate in requested_windows if candidate not in values_by_window
    ]
    if missing_windows:
        computed: dict[int, dict[str, float]] = {
            candidate: {} for candidate in missing_windows
        }
        for symbol, prepared in prepared_frames.items():
            volume_values = prepared.volume.to_numpy(dtype=float, copy=False)
            for candidate in missing_windows:
                min_count = max(3, min(candidate, 5))
                if volume_values.size < min_count:
                    value = np.nan
                else:
                    window_values = volume_values[-candidate:]
                    mean = float(np.nanmean(window_values))
                    value = (
                        -(float(np.nanstd(window_values, ddof=0)) / mean)
                        if mean > 1e-12
                        else np.nan
                    )
                computed[candidate][str(symbol)] = value
        values_by_window = dict(values_by_window)
        values_by_window.update(
            {
                candidate: pd.Series(values, dtype=float)
                for candidate, values in computed.items()
            }
        )
        if factor_cache is not None:
            factor_cache[cache_key] = values_by_window
    return pd.Series(values_by_window.get(window, pd.Series(dtype=float)), dtype=float)


def _amihud_illiquidity_values_prepared(
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame],
    *,
    window: int,
    factor_name: str,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    cache_key = "pv_amihud_illiquidity_window_values"
    cached = (factor_cache or {}).get(cache_key)
    values_by_window: dict[int, pd.Series] = (
        cached if isinstance(cached, dict) else {}
    )
    requested_windows = _active_price_volume_windows(
        factor_cache=factor_cache,
        prefix="pv_amihud_illiquidity_",
        current_name=factor_name,
    )
    missing_windows = [
        candidate for candidate in requested_windows if candidate not in values_by_window
    ]
    if missing_windows:
        computed: dict[int, dict[str, float]] = {
            candidate: {} for candidate in missing_windows
        }
        for symbol, prepared in prepared_frames.items():
            base = prepared.amihud_base
            if base is None:
                base = _amihud_base_from_raw(prepared.close_raw, prepared.amount_raw)
            base_values = base.to_numpy(dtype=float, copy=False)
            for candidate in missing_windows:
                min_count = max(3, min(candidate, 5))
                if base_values.size < min_count:
                    value = np.nan
                else:
                    value = float(np.nanmean(base_values[-candidate:]))
                computed[candidate][str(symbol)] = value
        values_by_window = dict(values_by_window)
        values_by_window.update(
            {
                candidate: pd.Series(values, dtype=float)
                for candidate, values in computed.items()
            }
        )
        if factor_cache is not None:
            factor_cache[cache_key] = values_by_window
    return pd.Series(values_by_window.get(window, pd.Series(dtype=float)), dtype=float)


def _high_dollar_volume_from_amount(amount: pd.Series, window: int) -> float:
    amount = amount.tail(window)
    if len(amount) < max(3, min(window, 5)):
        return np.nan
    average_amount = float(amount.mean())
    if average_amount <= 1e-12:
        return np.nan
    return float(np.log(average_amount))


def _dollar_volume_growth_from_amount(
    amount: pd.Series,
    short_window: int,
    long_window: int,
) -> float:
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


def _amihud_illiquidity_from_prepared(
    prepared: _PreparedPriceVolumeFrame,
    window: int,
) -> float:
    base = prepared.amihud_base
    if base is None:
        base = _amihud_base_from_raw(prepared.close_raw, prepared.amount_raw)
    values = base.tail(window)
    if len(values) < max(3, min(window, 5)):
        return np.nan
    return float(values.mean())


def _momentum_from_close(close: pd.Series, window: int) -> float:
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


def _short_reversal(frame: pd.DataFrame, window: int) -> float:
    return _short_reversal_from_close(_close_series(frame), window)


def _volatility_penalty(frame: pd.DataFrame, window: int) -> float:
    return _volatility_penalty_from_close(_close_series(frame), window)


def _downside_volatility(frame: pd.DataFrame, window: int) -> float:
    return _downside_volatility_from_close(_close_series(frame), window)


def _price_efficiency(frame: pd.DataFrame, window: int) -> float:
    return _price_efficiency_from_close(_close_series(frame), window)


def _volume_stability(frame: pd.DataFrame, window: int) -> float:
    return _volume_stability_from_volume(_volume_series(frame), window)


def _volume_stability_smooth(
    frame: pd.DataFrame,
    base_window: int,
    smooth_window: int,
) -> float:
    return _volume_stability_smooth_from_volume(
        _volume_series(frame),
        base_window,
        smooth_window,
    )


def _low_dollar_volume(frame: pd.DataFrame, window: int) -> float:
    return _low_dollar_volume_from_amount(_amount_series(frame), window)


def _high_dollar_volume(frame: pd.DataFrame, window: int) -> float:
    return _high_dollar_volume_from_amount(_amount_series(frame), window)


def _dollar_volume_growth(
    frame: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> float:
    return _dollar_volume_growth_from_amount(
        _amount_series(frame),
        short_window,
        long_window,
    )


def _amihud_illiquidity(frame: pd.DataFrame, window: int) -> float:
    return _amihud_illiquidity_from_prepared(
        _prepare_price_volume_frame(frame),
        window,
    )


def _momentum(frame: pd.DataFrame, window: int) -> float:
    return _momentum_from_close(_close_series(frame), window)


def _rank_blend_volstab_momentum_amihud(
    frames: Mapping[str, pd.DataFrame],
    *,
    outer_weight: float,
) -> pd.Series:
    return _rank_blend_volstab_momentum_amihud_prepared(
        prepare_price_volume_frames(frames),
        outer_weight=outer_weight,
    )


def _rank_blend_volstab_momentum_amihud_prepared(
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame],
    *,
    outer_weight: float,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    cache_key = "pv_blend_volstab19x2_mom90_amihud5_components"
    components = (factor_cache or {}).get(cache_key)
    if not isinstance(components, dict):
        vol_stability: dict[str, float] = {}
        momentum: dict[str, float] = {}
        amihud: dict[str, float] = {}
        for symbol, prepared in prepared_frames.items():
            key = str(symbol)
            vol_stability[key] = _volume_stability_smooth_from_volume(
                prepared.volume,
                base_window=19,
                smooth_window=2,
            )
            momentum[key] = _momentum_from_close(prepared.close, 90)
            amihud[key] = _amihud_illiquidity_from_prepared(prepared, 5)
        vol_rank = pd.Series(vol_stability, dtype=float).rank(pct=True)
        momentum_rank = pd.Series(momentum, dtype=float).rank(pct=True)
        amihud_rank = pd.Series(amihud, dtype=float).rank(pct=True)
        inner = momentum_rank.mul(0.60).add(amihud_rank.mul(0.40))
        inner_rank = inner.rank(pct=True)
        components = {
            "vol_rank": vol_rank,
            "inner_rank": inner_rank,
        }
        if factor_cache is not None:
            factor_cache[cache_key] = components
    vol_rank = pd.Series(
        components.get("vol_rank", pd.Series(dtype=float)),
        dtype=float,
    )
    inner_rank = pd.Series(
        components.get("inner_rank", pd.Series(dtype=float)),
        dtype=float,
    )
    return vol_rank.mul(outer_weight).add(
        inner_rank.mul(1.0 - outer_weight)
    )


def _compute_prepared_price_volume_factor(
    factor_name: str,
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame],
    *,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    if factor_name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
        return _rank_blend_volstab_momentum_amihud_prepared(
            prepared_frames,
            outer_weight=_weight_from_name(factor_name, 0.75),
            factor_cache=factor_cache,
        )
    window = _window_from_name(factor_name)
    if factor_name.startswith("pv_low_dollar_volume_"):
        return _low_dollar_volume_values_prepared(
            prepared_frames,
            window=window,
            factor_name=factor_name,
            factor_cache=factor_cache,
        )
    if (
        factor_name.startswith("pv_volume_stability_")
        and not factor_name.startswith("pv_volume_stability_smooth_")
    ):
        return _volume_stability_values_prepared(
            prepared_frames,
            window=window,
            factor_name=factor_name,
            factor_cache=factor_cache,
        )
    if factor_name.startswith("pv_amihud_illiquidity_"):
        return _amihud_illiquidity_values_prepared(
            prepared_frames,
            window=window,
            factor_name=factor_name,
            factor_cache=factor_cache,
        )
    values: dict[str, float] = {}
    for symbol, prepared in prepared_frames.items():
        if factor_name.startswith("pv_momentum_"):
            value = _momentum_from_close(prepared.close, window)
        elif factor_name.startswith("pv_short_reversal_"):
            value = _short_reversal_from_close(prepared.close, window)
        elif factor_name.startswith("pv_volume_stability_smooth_"):
            base_window, smooth_window = _smooth_windows_from_name(factor_name)
            value = _volume_stability_smooth_from_volume(
                prepared.volume,
                base_window,
                smooth_window,
            )
        elif factor_name.startswith("pv_volume_stability_"):
            value = _volume_stability_from_volume(prepared.volume, window)
        elif factor_name.startswith("pv_low_dollar_volume_"):
            value = _low_dollar_volume_from_amount(prepared.amount, window)
        elif factor_name.startswith("pv_high_dollar_volume_"):
            value = _high_dollar_volume_from_amount(prepared.amount, window)
        elif factor_name.startswith("pv_amihud_illiquidity_"):
            value = _amihud_illiquidity_from_prepared(prepared, window)
        elif factor_name.startswith("pv_volatility_penalty_"):
            value = _volatility_penalty_from_close(prepared.close, window)
        elif factor_name.startswith("pv_downside_volatility_"):
            value = _downside_volatility_from_close(prepared.close, window)
        elif factor_name.startswith("pv_price_efficiency_"):
            value = _price_efficiency_from_close(prepared.close, window)
        elif factor_name.startswith("pv_dollar_volume_growth_"):
            short_window, long_window = _growth_windows_from_name(factor_name)
            value = _dollar_volume_growth_from_amount(
                prepared.amount,
                short_window,
                long_window,
            )
        else:
            raise ValueError(f"unknown price/volume factor: {factor_name}")
        values[str(symbol)] = value
    return pd.Series(values, dtype=float)


def compute_price_volume_factor(
    name: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    prepared_frames: Mapping[str, _PreparedPriceVolumeFrame] | None = None,
    factor_cache: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute the latest cross-sectional raw value.

    The input factor name selects one governed price/volume implementation.
    """

    factor_name = str(name).strip()
    if prepared_frames is not None:
        return _compute_prepared_price_volume_factor(
            factor_name,
            prepared_frames,
            factor_cache=factor_cache,
        )
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


__all__ = [
    "compute_price_volume_factor",
    "prepare_price_volume_frames",
]
