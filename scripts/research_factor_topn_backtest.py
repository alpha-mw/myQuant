#!/usr/bin/env python3
"""What is left of the LOW + W80 pool after A-share execution, and how much capital fits?

A research-only, long-only backtest of an equal-weight top-N portfolio drawn
from the production ranking (LOW and W80 percentiles averaged 50/50). The signal
is taken at the close of session t and traded at the open of t+1 with:

- no buy at limit-up and no sell at limit-down, judged at the open;
- no trade on a day the name has no bar (suspended);
- 100-share lots on entry (200 on STAR), whole position on exit;
- the owner's paper fee schedule, with the 0.1% stamp duty before 2023-08-28;
- half a tick of spread and a square-root market-impact charge;
- fills capped at a share of the name's trailing 20-session turnover.

It reads the canonical bars table of the current snapshot through
``MarketDataReader``, makes no provider, broker or LLM call, and writes only
under ``reports/research/``. It is not ``market backtest``, which stays
unavailable, and it authorises nothing.

Read every number with three limits in mind. LOW and W80 were chosen with
knowledge of this history, so the backtest is in-sample for that choice. The
snapshot has no limit prices, suspension flags or ST history, so limits are
derived from the previous close and the board's band, and an ST name's 5% band
is only recognised when it opens exactly on that price and never trades through
it. Dividends are reinvested through the adjustment factor, and a delisted
holding is written down to ``--delist-recovery`` of its last close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.execution_cost import (  # noqa: E402
    COST_MODEL_SQRT_IMPACT,
    FactorExecutionCostConfig,
    estimate_market_impact_bps,
)
from quant_investor.paper.execution import (  # noqa: E402
    COMMISSION_MINIMUM,
    COMMISSION_RATE,
    STAMP_RATE,
    TRANSFER_RATE,
)

BARS_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
    "adj_factor",
    "adj_close",
    "total_mv",
]
AMOUNT_UNIT_CNY = 1000.0  # bars ``amount`` is in thousand CNY
TICK = 0.01
STAMP_DUTY_HALVED_ON = "20230828"
STAMP_RATE_BEFORE_HALVING = 0.001
CHINEXT_20_PERCENT_FROM = "20200824"
SESSIONS_PER_YEAR = 252
LOW_WEIGHT = 0.5


@dataclass(frozen=True)
class Frictions:
    """Which execution costs a run charges. Tradability and lots always apply."""

    label: str
    fees: bool = True
    half_spread_ticks: float = 0.5
    impact_coefficient_bps: float = 250.0
    participation_cap: float = 0.10


@dataclass
class Market:
    """Wide (session x symbol) arrays for the simulated window."""

    dates: list[str]
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    pre_close: np.ndarray
    adj_factor: np.ndarray
    adj_close_filled: np.ndarray
    adv_cny: np.ndarray
    band: np.ndarray
    lot: np.ndarray
    delist_index: np.ndarray


@dataclass
class RunResult:
    nav: np.ndarray
    stats: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- signals


def compute_components(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-row LOW, volume-stability, momentum and Amihud, in row space per symbol.

    A vectorised copy of ``governance.bootstrap._strict_components``: windows
    count a symbol's own rows, so a suspension is simply skipped, and a value
    that is undefined on the last row falls back to the last defined one where
    the production code does. ``bars`` must be sorted by symbol then date.
    """

    by_symbol = bars["ts_code"]
    amount = bars["amount"]
    rolled = amount.groupby(by_symbol, sort=False).rolling(5, min_periods=5)
    mean5 = rolled.mean().reset_index(level=0, drop=True)
    min5 = rolled.min().reset_index(level=0, drop=True)
    low = -np.log(mean5.where(min5 > 0.0))

    volume = bars["vol"].groupby(by_symbol, sort=False).rolling(19, min_periods=5)
    rolling_mean = volume.mean().reset_index(level=0, drop=True)
    rolling_std = volume.std(ddof=0).reset_index(level=0, drop=True)
    stability = -(rolling_std / rolling_mean.where(rolling_mean > 0.0))
    smoothed = (
        stability.groupby(by_symbol, sort=False)
        .rolling(2, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    smoothed = smoothed.groupby(by_symbol, sort=False).ffill()

    close = bars["adj_close"]
    base = close.groupby(by_symbol, sort=False).shift(90)
    momentum = (close / base.where(base > 0.0) - 1.0).where(np.isfinite(close))

    returns = close.groupby(by_symbol, sort=False).pct_change(fill_method=None).abs()
    illiquidity = (returns / amount.where(amount > 0.0)).replace([np.inf, -np.inf], np.nan)
    valid = illiquidity.dropna()
    amihud = (
        valid.groupby(by_symbol.loc[valid.index], sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(bars.index)
        .groupby(by_symbol, sort=False)
        .ffill()
    )
    return pd.DataFrame(
        {"low_dollar": low, "stability": smoothed, "momentum": momentum, "amihud": amihud},
        index=bars.index,
    )


def blend_w80(day: pd.DataFrame) -> pd.Series:
    """W80 over one session's cohort, as ``compute_bootstrap_signals`` ranks it."""

    inner = (
        day["momentum"].rank(pct=True).mul(0.60).add(day["amihud"].rank(pct=True).mul(0.40))
    ).rank(pct=True)
    return day["stability"].rank(pct=True).mul(0.80).add(inner.mul(0.20))


def rank_pool(low: pd.Series, w80: pd.Series, *, top_n: int) -> list[str]:
    """The production pool: average-rank percentiles 50/50, ties by symbol."""

    finite = np.isfinite(low) & np.isfinite(w80)
    low, w80 = low[finite], w80[finite]
    count = len(low)
    if count < 2:
        return []
    combined = LOW_WEIGHT * (low.rank() - 1.0) / (count - 1) + (1.0 - LOW_WEIGHT) * (
        w80.rank() - 1.0
    ) / (count - 1)
    symbols = combined.index.to_numpy(dtype=str)
    order = np.lexsort((symbols, -combined.to_numpy()))
    return list(symbols[order[:top_n]])


# --------------------------------------------------------------------------- execution


def round_price(value: float) -> float:
    return math.floor(value / TICK + 0.5 + 1e-9) * TICK


def limit_band(symbol: str, trade_date: str) -> float:
    if symbol.endswith(".BJ"):
        return 0.30
    if symbol.startswith(("688", "689")):
        return 0.20
    if symbol.startswith(("300", "301")) and trade_date >= CHINEXT_20_PERCENT_FROM:
        return 0.20
    return 0.10


def buy_blocked(open_price: float, high: float, pre_close: float, band: float) -> bool:
    """Limit-up at the open; also an open pinned on the 5% ST limit all day."""

    if open_price >= round_price(pre_close * (1.0 + band)) - 1e-9:
        return True
    st_limit = round_price(pre_close * 1.05)
    return abs(open_price - st_limit) < 1e-9 and high <= open_price + 1e-9


def sell_blocked(open_price: float, low: float, pre_close: float, band: float) -> bool:
    if open_price <= round_price(pre_close * (1.0 - band)) + 1e-9:
        return True
    st_limit = round_price(pre_close * 0.95)
    return abs(open_price - st_limit) < 1e-9 and low >= open_price - 1e-9


def trade_fees(value: float, *, side: str, trade_date: str) -> float:
    """Commission with its minimum, transfer fee, and stamp duty on a sale."""

    commission = max(value * float(COMMISSION_RATE), float(COMMISSION_MINIMUM))
    transfer = value * float(TRANSFER_RATE)
    if side != "SELL":
        return commission + transfer
    stamp = float(STAMP_RATE) if trade_date >= STAMP_DUTY_HALVED_ON else STAMP_RATE_BEFORE_HALVING
    return commission + transfer + value * stamp


def _price_drag(
    value: float,
    open_price: float,
    adv_cny: float,
    frictions: Frictions,
    impact: FactorExecutionCostConfig,
) -> float:
    """Spread plus impact as a fraction of the traded value."""

    spread = frictions.half_spread_ticks * TICK / open_price
    if impact.impact_coefficient <= 0.0 or not adv_cny > 0.0:
        return spread
    impact_bps = estimate_market_impact_bps(participation_rate=value / adv_cny, config=impact)
    return spread + impact_bps / 10_000.0


def simulate(
    market: Market,
    targets: dict[int, list[int]],
    *,
    capital: float,
    top_n: int,
    frictions: Frictions,
    delist_recovery: float = 0.0,
) -> RunResult:
    """Run one portfolio. ``targets[i]`` is the pool to hold from the open of session i."""

    cash = float(capital)
    units: dict[int, float] = {}
    pending_exit: set[int] = set()
    nav = np.empty(len(market.dates))
    nav_previous = float(capital)
    stats = dict.fromkeys(
        (
            "buy_value",
            "sell_value",
            "fees",
            "price_drag",
            "buy_orders",
            "buy_blocked_limit",
            "buy_blocked_suspended",
            "buy_below_lot",
            "buy_capped",
            "sell_orders",
            "sell_blocked_limit",
            "sell_blocked_suspended",
            "sell_capped",
            "delisted_positions",
            "delisting_loss",
            "holding_sessions",
        ),
        0.0,
    )
    cap = frictions.participation_cap
    impact = FactorExecutionCostConfig(
        config_id="research-topn-backtest",
        impact_model=COST_MODEL_SQRT_IMPACT,
        impact_coefficient=frictions.impact_coefficient_bps,
    )

    for index, trade_date in enumerate(market.dates):
        for symbol in [s for s in units if market.delist_index[s] <= index]:
            value = units.pop(symbol) * market.adj_close_filled[index, symbol]
            cash += value * delist_recovery
            stats["delisted_positions"] += 1
            stats["delisting_loss"] += value * (1.0 - delist_recovery)
            pending_exit.discard(symbol)

        target = targets.get(index)
        if target is not None:
            wanted = set(target)
            pending_exit = {symbol for symbol in units if symbol not in wanted}

        for symbol in sorted(pending_exit):
            stats["sell_orders"] += 1
            open_price = market.open[index, symbol]
            if not open_price > 0.0:
                stats["sell_blocked_suspended"] += 1
                continue
            if sell_blocked(
                open_price,
                market.low[index, symbol],
                market.pre_close[index, symbol],
                market.band[index, symbol],
            ):
                stats["sell_blocked_limit"] += 1
                continue
            adv = market.adv_cny[index, symbol]
            position = units[symbol] * open_price * market.adj_factor[index, symbol]
            value = position if not cap > 0.0 or not adv > 0.0 else min(position, cap * adv)
            if value < position:
                stats["sell_capped"] += 1
                units[symbol] *= 1.0 - value / position
            else:
                del units[symbol]
                pending_exit.discard(symbol)
            drag = value * _price_drag(value, open_price, adv, frictions, impact)
            fees = (
                trade_fees(value - drag, side="SELL", trade_date=trade_date)
                if frictions.fees
                else 0.0
            )
            cash += value - drag - fees
            stats["sell_value"] += value
            stats["price_drag"] += drag
            stats["fees"] += fees

        if target is not None:
            per_name = nav_previous / top_n
            for symbol in target:
                if symbol in units:
                    continue
                stats["buy_orders"] += 1
                open_price = market.open[index, symbol]
                if not open_price > 0.0:
                    stats["buy_blocked_suspended"] += 1
                    continue
                if buy_blocked(
                    open_price,
                    market.high[index, symbol],
                    market.pre_close[index, symbol],
                    market.band[index, symbol],
                ):
                    stats["buy_blocked_limit"] += 1
                    continue
                adv = market.adv_cny[index, symbol]
                budget = min(per_name, cash)
                if cap > 0.0 and adv > 0.0 and budget > cap * adv:
                    budget = cap * adv
                    stats["buy_capped"] += 1
                lot = int(market.lot[symbol])
                shares = math.floor(budget / open_price / 100.0) * 100
                while shares >= lot:
                    value = shares * open_price
                    drag = value * _price_drag(value, open_price, adv, frictions, impact)
                    fees = (
                        trade_fees(value + drag, side="BUY", trade_date=trade_date)
                        if frictions.fees
                        else 0.0
                    )
                    if value + drag + fees <= cash:
                        break
                    shares -= 100
                if shares < lot:
                    stats["buy_below_lot"] += 1
                    continue
                cash -= value + drag + fees
                units[symbol] = shares / market.adj_factor[index, symbol]
                stats["buy_value"] += value
                stats["price_drag"] += drag
                stats["fees"] += fees

        holdings = sum(
            amount * market.adj_close_filled[index, symbol] for symbol, amount in units.items()
        )
        nav[index] = nav_previous = cash + holdings
        stats["holding_sessions"] += len(units)

    return RunResult(nav=nav, stats=stats)


# --------------------------------------------------------------------------- reporting


def performance(nav: np.ndarray, *, initial: float) -> dict[str, float]:
    series = np.concatenate([[initial], nav])
    returns = series[1:] / series[:-1] - 1.0
    years = len(returns) / SESSIONS_PER_YEAR
    volatility = float(np.std(returns, ddof=1) * math.sqrt(SESSIONS_PER_YEAR))
    peak = np.maximum.accumulate(series)
    return {
        "cagr": float((series[-1] / series[0]) ** (1.0 / years) - 1.0),
        "volatility": volatility,
        "sharpe": float(np.mean(returns) * SESSIONS_PER_YEAR / volatility) if volatility else 0.0,
        "max_drawdown": float(np.min(series / peak - 1.0)),
    }


def run_summary(result: RunResult, *, capital: float, sessions: int) -> dict[str, float]:
    years = sessions / SESSIONS_PER_YEAR
    average_nav = float(np.mean(result.nav))
    stats = result.stats
    return {
        **performance(result.nav, initial=capital),
        "final_nav_multiple": float(result.nav[-1] / capital),
        "annual_one_way_turnover": (stats["buy_value"] + stats["sell_value"])
        / 2.0
        / average_nav
        / years,
        "annual_fee_drag": stats["fees"] / average_nav / years,
        "annual_spread_impact_drag": stats["price_drag"] / average_nav / years,
        "average_holdings": stats["holding_sessions"] / sessions,
        "buy_blocked_limit_rate": stats["buy_blocked_limit"] / max(stats["buy_orders"], 1.0),
        "buy_below_lot_rate": stats["buy_below_lot"] / max(stats["buy_orders"], 1.0),
        "buy_capped_rate": stats["buy_capped"] / max(stats["buy_orders"], 1.0),
        "sell_blocked_limit_rate": stats["sell_blocked_limit"] / max(stats["sell_orders"], 1.0),
        "sell_capped_rate": stats["sell_capped"] / max(stats["sell_orders"], 1.0),
        "delisted_positions": stats["delisted_positions"],
        "delisting_loss_share_of_capital": stats["delisting_loss"] / capital,
    }


# --------------------------------------------------------------------------- data


def load_bars(workspace: Path, *, start: str, end: str) -> tuple[pd.DataFrame, dict[str, str]]:
    import pyarrow.parquet as pq

    from quant_investor.market.market_data_reader import MarketDataReader

    reader = MarketDataReader(market="CN", data_root=str(workspace / "data"))
    paths = reader.table_partition_paths(start, end)
    bars = pq.read_table([str(path) for path in paths], columns=BARS_COLUMNS).to_pandas()
    bars["ts_code"] = bars["ts_code"].astype(str)
    bars["trade_date"] = bars["trade_date"].astype(str)
    bars = bars[(bars["trade_date"] >= start) & (bars["trade_date"] <= end)]
    bars = bars.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    pointer = workspace / "data/parquet/cn/_latest.json"
    raw = pointer.read_bytes()
    return bars, {
        "snapshot_id": json.loads(raw)["snapshot_id"],
        "pointer_sha256": hashlib.sha256(raw).hexdigest(),
    }


def listing_dates(workspace: Path) -> dict[str, tuple[str, str]]:
    from quant_investor.market.pit_universe import PITUniverseStore

    records = PITUniverseStore(
        root_dir=str(workspace / "data/parquet/cn/reference")
    ).records_by_symbol()
    return {symbol: (record.list_date, record.delist_date) for symbol, record in records.items()}


def listed_mask(bars: pd.DataFrame, listing: dict[str, tuple[str, str]]) -> pd.Series:
    """Listed on the row's date. A symbol with no PIT record is excluded."""

    listed_from = bars["ts_code"].map({symbol: value[0] for symbol, value in listing.items()})
    delisted_on = bars["ts_code"].map({symbol: value[1] for symbol, value in listing.items()})
    return (
        listed_from.notna()
        & (listed_from != "")
        & (bars["trade_date"] >= listed_from)
        & ((delisted_on == "") | (bars["trade_date"] < delisted_on))
    )


def build_market(
    bars: pd.DataFrame, dates: list[str], listing: dict[str, tuple[str, str]]
) -> Market:
    window = bars[bars["trade_date"].isin(set(dates))]
    symbols = sorted(window["ts_code"].unique())
    row = window["trade_date"].map({value: i for i, value in enumerate(dates)}).to_numpy()
    column = window["ts_code"].map({value: i for i, value in enumerate(symbols)}).to_numpy()

    def wide(name: str) -> np.ndarray:
        values = np.full((len(dates), len(symbols)), np.nan)
        values[row, column] = window[name].to_numpy(dtype=float)
        return values

    adj_close = pd.DataFrame(wide("adj_close")).ffill().to_numpy()
    band = np.empty((len(dates), len(symbols)))
    for position, day in enumerate((dates[0], CHINEXT_20_PERCENT_FROM)):
        rows = slice(None) if position == 0 else slice(int(np.searchsorted(dates, day)), None)
        band[rows] = [limit_band(symbol, day) for symbol in symbols]
    delist_index = np.full(len(symbols), len(dates) + 1)
    for position, symbol in enumerate(symbols):
        delisted_on = listing.get(symbol, ("", ""))[1]
        if delisted_on:
            delist_index[position] = int(np.searchsorted(dates, delisted_on))
    return Market(
        dates=dates,
        symbols=symbols,
        open=wide("open"),
        high=wide("high"),
        low=wide("low"),
        pre_close=wide("pre_close"),
        adj_factor=wide("adj_factor"),
        adj_close_filled=adj_close,
        adv_cny=wide("adv20") * AMOUNT_UNIT_CNY,
        band=band,
        lot=np.array([200 if symbol.startswith(("688", "689")) else 100 for symbol in symbols]),
        delist_index=delist_index,
    )


def universe_benchmarks(bars: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Close-to-close returns of every listed, traded name: equal and cap weighted.

    Gross of all costs and rebalanced daily, so they are yardsticks, not
    investable portfolios. A name that stops trading drops out the next day.
    The smallest market-cap decile is included because LOW is largely a
    small-cap tilt: it shows how much of the pool's return plain size explains.
    """

    frame = bars[bars["listed"]].copy()
    grouped = frame.groupby("ts_code", sort=False)
    frame["next_return"] = grouped["adj_close"].shift(-1) / frame["adj_close"] - 1.0
    frame["next_date"] = grouped["trade_date"].shift(-1)
    all_dates = sorted(bars["trade_date"].unique())
    following = dict(zip(all_dates[:-1], all_dates[1:]))
    # A name suspended tomorrow contributes a zero return, not its eventual gap.
    consecutive = frame["next_date"] == frame["trade_date"].map(following)
    frame["next_return"] = frame["next_return"].where(consecutive, 0.0)
    frame["weighted"] = frame["next_return"] * frame["total_mv"]
    by_day = frame.groupby("trade_date")
    smallest = frame[by_day["total_mv"].rank(pct=True) <= 0.10]
    table = pd.DataFrame(
        {
            "equal_weight": by_day["next_return"].mean(),
            "cap_weight": by_day["weighted"].sum() / by_day["total_mv"].sum(),
            "smallest_decile": smallest.groupby("trade_date")["next_return"].mean(),
        }
    )
    table.index = table.index.map(following)
    return table.reindex(dates).fillna(0.0)


def build_targets(
    bars: pd.DataFrame,
    market: Market,
    *,
    top_n: int,
    rebalance_sessions: int,
    min_cohort: int,
    signal_start: str,
) -> tuple[dict[int, list[int]], dict[str, float]]:
    """Pools ranked at the close of every k-th session, keyed by the next session's index."""

    symbol_index = {symbol: i for i, symbol in enumerate(market.symbols)}
    all_dates = sorted(bars["trade_date"].unique())
    signal_dates = [day for day in all_dates if day >= signal_start][::rebalance_sessions]
    eligible = bars[bars["listed"] & (bars["total_mv"] > 0.0)]
    by_day = {day: frame for day, frame in eligible.groupby("trade_date") if day in signal_dates}
    targets: dict[int, list[int]] = {}
    cohorts: list[int] = []
    skipped = 0
    for day in signal_dates:
        execution = all_dates.index(day) + 1
        if execution >= len(all_dates) or all_dates[execution] not in market.dates:
            continue
        frame = by_day[day].set_index("ts_code")
        w80 = blend_w80(frame)
        cohort = int((np.isfinite(frame["low_dollar"]) & np.isfinite(w80)).sum())
        cohorts.append(cohort)
        if cohort < min_cohort:
            skipped += 1
            continue
        pool = rank_pool(frame["low_dollar"], w80, top_n=top_n)
        targets[market.dates.index(all_dates[execution])] = [
            symbol_index[symbol] for symbol in pool if symbol in symbol_index
        ]
    return targets, {
        "signal_dates": float(len(cohorts)),
        "rebalances": float(len(targets)),
        "skipped_below_min_cohort": float(skipped),
        "median_cohort": float(np.median(cohorts)) if cohorts else 0.0,
        "min_cohort_seen": float(min(cohorts)) if cohorts else 0.0,
    }


def verify_against_production(
    bars: pd.DataFrame, components: pd.DataFrame, dates: list[str]
) -> list[dict]:
    """Recompute sample sessions with the governed signal code and compare."""

    from quant_investor.factors.governance.bootstrap import (
        BLEND_W80,
        CANONICAL_PARQUET,
        LOW_DOLLAR_VOLUME,
        compute_bootstrap_signals,
    )

    checks = []
    eligible = bars["listed"] & (bars["total_mv"] > 0.0)
    for day in dates:
        cohort = bars.loc[eligible & (bars["trade_date"] == day), "ts_code"]
        history = bars[bars["ts_code"].isin(set(cohort)) & (bars["trade_date"] <= day)]
        frames = {
            symbol: frame[["trade_date", "amount", "adj_close", "vol"]].tail(140)
            for symbol, frame in history.groupby("ts_code", sort=False)
        }
        governed = compute_bootstrap_signals(frames, source_format=CANONICAL_PARQUET)
        mine = components.loc[cohort.index].set_index(cohort.to_numpy())
        w80 = blend_w80(mine)
        low_gap = (governed[LOW_DOLLAR_VOLUME] - mine["low_dollar"]).abs().max()
        w80_gap = (governed[BLEND_W80] - w80).abs().max()
        undefined_differs = int(
            (governed[LOW_DOLLAR_VOLUME].isna() != mine["low_dollar"].isna()).sum()
            + (governed[BLEND_W80].isna() != w80.isna()).sum()
        )
        pool = rank_pool(mine["low_dollar"], w80, top_n=100)
        governed_pool = rank_pool(governed[LOW_DOLLAR_VOLUME], governed[BLEND_W80], top_n=100)
        checks.append(
            {
                "date": day,
                "cohort": len(cohort),
                "max_abs_low_gap": float(low_gap),
                "max_abs_w80_gap": float(w80_gap),
                "undefined_value_mismatches": undefined_differs,
                "top100_identical": pool == governed_pool,
            }
        )
    return checks


# --------------------------------------------------------------------------- entry point

FRICTION_LAYERS = (
    Frictions("tradability_only", fees=False, half_spread_ticks=0.0, impact_coefficient_bps=0.0),
    Frictions("plus_fees", half_spread_ticks=0.0, impact_coefficient_bps=0.0),
    Frictions("plus_spread", impact_coefficient_bps=0.0),
    Frictions("all_costs"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--start", default="20180102", help="first signal session, YYYYMMDD")
    parser.add_argument("--end", default="99991231")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--rebalance-sessions", type=int, default=5)
    parser.add_argument("--min-cohort", type=int, default=3000)
    parser.add_argument("--base-capital", type=float, default=1_000_000.0)
    parser.add_argument("--capacity-capitals", default="1e6,1e7,1e8,1e9")
    parser.add_argument("--delist-recovery", type=float, default=0.0)
    parser.add_argument("--verify-dates", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    warmup_start = f"{int(args.start[:4]) - 1}0101"
    bars, snapshot = load_bars(args.workspace_root, start=warmup_start, end=args.end)
    listing = listing_dates(args.workspace_root)
    bars["listed"] = listed_mask(bars, listing)
    components = compute_components(bars)
    bars = bars.join(components)
    bars["adv20"] = (
        bars.groupby("ts_code", sort=False)["amount"]
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .groupby(bars["ts_code"], sort=False)
        .shift(1)
    )

    all_dates = sorted(bars["trade_date"].unique())
    signal_dates = [day for day in all_dates if day >= args.start]
    dates = all_dates[all_dates.index(signal_dates[0]) + 1 :]
    market = build_market(bars, dates, listing)
    targets, cohort_stats = build_targets(
        bars,
        market,
        top_n=args.top_n,
        rebalance_sessions=args.rebalance_sessions,
        min_cohort=args.min_cohort,
        signal_start=args.start,
    )

    verification = []
    if args.verify_dates > 0:
        step = max(len(signal_dates) // (args.verify_dates + 1), 1)
        sample = signal_dates[step::step][: args.verify_dates]
        verification = verify_against_production(bars, components, sample)

    runs: dict[str, RunResult] = {}
    for frictions in FRICTION_LAYERS:
        runs[frictions.label] = simulate(
            market,
            targets,
            capital=args.base_capital,
            top_n=args.top_n,
            frictions=frictions,
            delist_recovery=args.delist_recovery,
        )
    capacity = {}
    for capital in (float(value) for value in args.capacity_capitals.split(",")):
        result = (
            runs["all_costs"]
            if capital == args.base_capital
            else simulate(
                market,
                targets,
                capital=capital,
                top_n=args.top_n,
                frictions=FRICTION_LAYERS[-1],
                delist_recovery=args.delist_recovery,
            )
        )
        capacity[f"{capital:.0f}"] = run_summary(result, capital=capital, sessions=len(dates))

    benchmark_returns = universe_benchmarks(bars, dates)
    benchmark_nav = (1.0 + benchmark_returns).cumprod()
    nav_table = pd.DataFrame(
        {label: result.nav / args.base_capital for label, result in runs.items()}, index=dates
    ).join(benchmark_nav.add_prefix("universe_"))
    year_end = nav_table.groupby(nav_table.index.str[:4]).last()
    by_year = year_end / year_end.shift(1).fillna(1.0) - 1.0

    summary = {
        "schema": "research-factor-topn-backtest.v1",
        "authority": "RESEARCH_ONLY_IN_SAMPLE_NOT_INVESTMENT_ADVICE",
        "snapshot": snapshot,
        "config": {
            "first_signal_session": signal_dates[0],
            "first_trade_session": dates[0],
            "last_session": dates[-1],
            "sessions": len(dates),
            "top_n": args.top_n,
            "rebalance_sessions": args.rebalance_sessions,
            "min_cohort": args.min_cohort,
            "base_capital": args.base_capital,
            "delist_recovery": args.delist_recovery,
            "friction_layers": [asdict(value) for value in FRICTION_LAYERS],
        },
        "cohort": cohort_stats,
        "signal_verification": verification,
        "layers": {
            label: run_summary(result, capital=args.base_capital, sessions=len(dates))
            for label, result in runs.items()
        },
        "capacity_all_costs": capacity,
        "benchmarks": {
            name: performance(benchmark_nav[name].to_numpy(), initial=1.0)
            for name in benchmark_nav.columns
        },
        "calendar_year_returns": {
            year: {name: float(value) for name, value in row.items()}
            for year, row in by_year.iterrows()
        },
    }

    output = args.output_dir or (
        args.workspace_root
        / "reports/research/factor_topn_backtest"
        / f"{snapshot['snapshot_id']}-top{args.top_n}-every{args.rebalance_sessions}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    nav_table.to_csv(output / "nav.csv", index_label="trade_date")
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
