from __future__ import annotations

"""Factor computation and neutralization helpers."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorInputs:
    prices: Dict[str, pd.DataFrame]
    market_caps: Dict[str, pd.DataFrame]
    industries: pd.DataFrame


@dataclass
class FactorOutputs:
    scores: pd.DataFrame
    momentum: pd.DataFrame
    volatility: pd.DataFrame


def compute_factors(inputs: FactorInputs, weight_momentum: float) -> FactorOutputs:
    """Compute momentum/volatility factors and combined score."""
    price_panel = _build_price_panel(inputs.prices)
    returns = price_panel.pct_change()
    momentum = price_panel / price_panel.shift(20) - 1
    volatility = returns.rolling(20).std() * np.sqrt(252)

    industry_map = inputs.industries.set_index("ts_code")["industry"].to_dict()
    market_caps = _build_market_cap_panel(inputs.market_caps)

    neutral_mom = _neutralize(momentum, industry_map, market_caps)
    neutral_vol = _neutralize(volatility, industry_map, market_caps)

    weight_vol = 1.0 - weight_momentum
    scores = neutral_mom * weight_momentum + neutral_vol * weight_vol

    return FactorOutputs(scores=scores, momentum=momentum, volatility=volatility)


def _build_price_panel(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ts_code, data in prices.items():
        frame = data[["trade_date", "close"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.set_index("trade_date").sort_index()
        frame = frame.rename(columns={"close": ts_code})
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index()


def _build_market_cap_panel(market_caps: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ts_code, data in market_caps.items():
        if data.empty:
            continue
        frame = data[["trade_date", "total_mv"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.set_index("trade_date").sort_index()
        frame = frame.rename(columns={"total_mv": ts_code})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def _neutralize(factor: pd.DataFrame, industry_map: Dict[str, str], market_caps: pd.DataFrame) -> pd.DataFrame:
    neutralized = []
    for date, row in factor.iterrows():
        valid = row.dropna()
        if valid.empty:
            neutralized.append(row)
            continue
        codes = valid.index.tolist()
        industries = pd.Series({code: industry_map.get(code, "Unknown") for code in codes})
        industry_dummies = pd.get_dummies(industries)
        log_mkt = None
        if not market_caps.empty and date in market_caps.index:
            log_mkt = np.log(market_caps.loc[date, codes].replace(0, np.nan))
        if log_mkt is None:
            design = industry_dummies
        else:
            design = industry_dummies.join(log_mkt.rename("log_mkt"))
        design = design.fillna(0.0)
        y = valid.values
        x = design.values
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        residual_series = pd.Series(residual, index=codes)
        neutralized.append(row.combine_first(residual_series))
    return pd.DataFrame(neutralized, index=factor.index, columns=factor.columns)


def compute_rebalance_dates(price_panel: pd.DataFrame) -> pd.DatetimeIndex:
    month_ends = price_panel.resample("M").last().index
    return month_ends


def build_factor_scores(inputs: FactorInputs, weight_momentum: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = compute_factors(inputs, weight_momentum)
    return outputs.scores, outputs.momentum, outputs.volatility
