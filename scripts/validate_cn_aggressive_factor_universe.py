#!/usr/bin/env python3
"""Broad offline validation for CN aggressive factor candidates.

This script is intentionally research-only. It reads strict Parquet canonical
CN bars plus local stock metadata, builds point-in-time price/volume factors,
defines full-market forward-return labels, and writes validation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.market.market_data_reader import MarketDataReader


DEFAULT_OUTPUT = Path(
    "results/strategy_records/CN/aggressive_tech_manufacturing/"
    "20260618_broad_factor_validation"
)
LATEST_PATH = Path("data/parquet/cn/_latest.json")
STOCK_BASIC_PATH = Path("data/parquet/cn/dag_core_raw/table=stock_basic/part.parquet")
DAILY_BASIC_PATH = Path("data/parquet/cn/fundamental_raw/table=daily_basic/part.parquet")


class ActiveMarketSnapshotError(RuntimeError):
    """Raised when the active strict-Parquet market pointer is unsafe."""


@dataclass(frozen=True)
class ValidationConfig:
    start_date: str = "20210104"
    warmup_date: str = "20200101"
    test_start_date: str = "20250101"
    min_amount_ma20: float = 50_000.0
    top_bucket_min_rows: int = 100
    winner_quantile: float = 0.90
    loser_quantile: float = 0.10
    top_per_day_values: tuple[int, ...] = (1, 3, 5, 10)
    cooldown_days: int = 20


FACTOR_COLUMNS = [
    "ret5",
    "ret20",
    "ret60",
    "risk_adj20",
    "risk_adj_fast",
    "fresh_accel",
    "amount_ratio1_20",
    "amount_ratio5_20",
    "high20_pos",
    "high60_pos",
    "close_vs_ma20",
    "low_vol_score",
    "industry_ret20_median",
    "industry_pos20",
    "score_quality_momentum",
    "score_starter_momentum",
    "score_breakout_quality",
]


def _pointer_identity(value: Any) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_active_pointer(path: Path) -> tuple[dict[str, Any], str]:
    """Read the active pointer as one stable, regular, non-symlink file."""

    if path.is_symlink():
        raise ActiveMarketSnapshotError(
            f"strict Parquet snapshot pointer must not be a symlink: {path}"
        )
    try:
        before = path.stat()
        if not path.is_file():
            raise ActiveMarketSnapshotError(
                f"strict Parquet snapshot pointer must be a regular file: {path}"
            )
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ActiveMarketSnapshotError(
            f"strict Parquet snapshot pointer missing or unreadable: {path}"
        ) from exc
    if path.is_symlink() or _pointer_identity(before) != _pointer_identity(after):
        raise ActiveMarketSnapshotError(
            "strict Parquet snapshot pointer changed during read"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActiveMarketSnapshotError(
            f"strict Parquet snapshot pointer is invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ActiveMarketSnapshotError(
            f"strict Parquet snapshot pointer must contain an object: {path}"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _resolve_active_bars_root(
    latest_path: Path = LATEST_PATH,
) -> tuple[Path, dict[str, Any], str]:
    """Resolve bars only from the healthy active market pointer."""

    latest_path = Path(latest_path)
    payload, pointer_sha256 = _read_active_pointer(latest_path)
    raw_table_root = payload.get("table_root")
    if not isinstance(raw_table_root, str) or not raw_table_root.strip():
        raise ActiveMarketSnapshotError(
            "active strict Parquet snapshot pointer table_root is missing"
        )
    if ".." in Path(raw_table_root.strip()).parts:
        raise ActiveMarketSnapshotError(
            "active strict Parquet snapshot pointer table_root contains parent traversal"
        )

    market_root = latest_path.parent.absolute()
    data_root = market_root.parent.parent
    for canonical_root in (data_root, data_root / "parquet", market_root):
        if canonical_root.is_symlink():
            raise ActiveMarketSnapshotError(
                f"CN canonical path component must not be a symlink: {canonical_root}"
            )
    gate = MarketDataReader(
        market="CN",
        data_root=data_root,
        mode_policy="strict",
    ).clean_snapshot_gate(refresh=True)
    payload_after, pointer_sha256_after = _read_active_pointer(latest_path)
    if pointer_sha256_after != pointer_sha256 or payload_after != payload:
        raise ActiveMarketSnapshotError(
            "active strict Parquet snapshot pointer changed during validation"
        )
    if gate.get("healthy") is not True:
        blockers = "; ".join(
            str(item)
            for item in list(gate.get("blockers", []) or [])
            if str(item).strip()
        )
        raise ActiveMarketSnapshotError(
            "active strict Parquet snapshot is blocked"
            + (f": {blockers}" if blockers else "")
        )
    if str(gate.get("snapshot_id") or "") != str(payload.get("snapshot_id") or ""):
        raise ActiveMarketSnapshotError(
            "active strict Parquet snapshot id changed during validation"
        )

    bars_root = Path(str(gate.get("table_root") or ""))
    if not bars_root.is_absolute():
        bars_root = Path.cwd() / bars_root
    bars_root = bars_root.absolute()
    try:
        relative_parts = bars_root.relative_to(market_root).parts
    except ValueError as exc:
        raise ActiveMarketSnapshotError(
            f"active strict Parquet table_root escapes CN canonical root: {bars_root}"
        ) from exc

    current = market_root
    if current.is_symlink():
        raise ActiveMarketSnapshotError(
            f"CN canonical market root must not be a symlink: {current}"
        )
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ActiveMarketSnapshotError(
                f"active strict Parquet table_root contains a symlink: {current}"
            )
    try:
        resolved_market_root = market_root.resolve(strict=True)
        resolved_bars_root = bars_root.resolve(strict=True)
        resolved_bars_root.relative_to(resolved_market_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ActiveMarketSnapshotError(
            f"active strict Parquet table_root is missing, unreadable, or unsafe: {bars_root}"
        ) from exc
    if not resolved_bars_root.is_dir():
        raise ActiveMarketSnapshotError(
            f"active strict Parquet table_root is not a directory: {resolved_bars_root}"
        )
    return resolved_bars_root, payload, pointer_sha256


def _metric_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "n": 0,
            "symbol_count": 0,
            "avg_fwd20": math.nan,
            "median_fwd20": math.nan,
            "win_rate20": math.nan,
            "ge10_rate20": math.nan,
            "le_minus8_rate20": math.nan,
            "avg_max_drawdown20": math.nan,
            "avg_outcome_score20": math.nan,
        }
    return {
        "n": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()),
        "avg_fwd20": float(frame["fwd20"].mean()),
        "median_fwd20": float(frame["fwd20"].median()),
        "win_rate20": float((frame["fwd20"] > 0).mean()),
        "ge10_rate20": float((frame["fwd20"] >= 0.10).mean()),
        "le_minus8_rate20": float((frame["fwd20"] <= -0.08).mean()),
        "avg_max_drawdown20": float(frame["max_drawdown20"].mean()),
        "avg_outcome_score20": float(frame["outcome_score20"].mean()),
    }


def _factor_gate(row: pd.Series) -> tuple[str, str]:
    """Classify a factor as a broad-market buy trigger, filter, or reject."""

    test_avg = float(row.get("test_top_avg_fwd20", math.nan))
    test_median = float(row.get("test_top_median_fwd20", math.nan))
    test_win = float(row.get("test_top_win_rate20", math.nan))
    spread = float(row.get("spread_avg_fwd20", math.nan))
    ic_mean = float(row.get("ic_mean", math.nan))
    loss_rate = float(row.get("top_le_minus8_rate20", math.nan))

    values = [test_avg, test_median, test_win, spread, ic_mean, loss_rate]
    if any(pd.isna(value) for value in values):
        return "reject_as_buy_trigger", "missing required validation metric"

    if (
        test_avg > 0
        and test_median > 0
        and test_win >= 0.50
        and spread > 0
        and ic_mean > 0
        and loss_rate <= 0.15
    ):
        return "pass_filter_only", "broadly positive, but still needs full portfolio cost and fill testing"

    if test_avg > 0 and test_win >= 0.47 and loss_rate <= 0.25:
        return "research_only", "positive sample-outside mean but weak median, spread, or IC"

    return "reject_as_buy_trigger", "fails broad buy-trigger gates"


def _safe_qcut_bucket(values: pd.Series, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    bins = [-0.001, 1 / 3, 2 / 3, 1.001]
    return pd.cut(ranked, bins=bins, labels=labels).astype("string")


def _load_base_frame(config: ValidationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars_root, latest, latest_pointer_sha256 = _resolve_active_bars_root(
        LATEST_PATH
    )
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "amount",
        "vol",
    ]
    bars = pd.read_parquet(
        bars_root,
        columns=columns,
        filters=[("trade_date", ">=", config.warmup_date)],
    ).rename(columns={"ts_code": "symbol"})
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["trade_date"] = bars["trade_date"].astype(str)
    for column in ["open", "high", "low", "close", "adj_close", "amount", "vol"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars["price"] = bars["adj_close"].where(
        bars["adj_close"].notna() & (bars["adj_close"] > 0),
        bars["close"],
    )

    stock_basic = pd.read_parquet(
        STOCK_BASIC_PATH,
        columns=["ts_code", "name", "industry", "market", "list_date"],
    ).rename(columns={"ts_code": "symbol"})
    stock_basic["symbol"] = stock_basic["symbol"].astype(str).str.upper()
    meta = stock_basic.drop_duplicates("symbol").set_index("symbol")
    bars["name"] = bars["symbol"].map(meta["name"]).fillna("")
    bars["industry"] = bars["symbol"].map(meta["industry"]).fillna("UNKNOWN")
    bars["market_board"] = bars["symbol"].map(meta["market"]).fillna("")
    bars["list_date"] = bars["symbol"].map(meta["list_date"]).fillna("")

    st_mask = bars["name"].str.contains("ST|退", regex=True, na=False)
    bars = bars[
        bars["symbol"].str.endswith((".SZ", ".SH"))
        & ~st_mask
        & bars["price"].notna()
    ].copy()
    bars = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    bars["bar_index"] = bars.groupby("symbol").cumcount()

    if DAILY_BASIC_PATH.exists():
        daily_basic = pd.read_parquet(
            DAILY_BASIC_PATH,
            columns=["ts_code", "trade_date", "circ_mv"],
            filters=[("trade_date", ">=", int(config.warmup_date))],
        ).rename(columns={"ts_code": "symbol"})
        daily_basic["symbol"] = daily_basic["symbol"].astype(str).str.upper()
        daily_basic["trade_date"] = daily_basic["trade_date"].astype(str)
        daily_basic["circ_mv"] = pd.to_numeric(
            daily_basic["circ_mv"],
            errors="coerce",
        )
        daily_basic = daily_basic.drop_duplicates(
            ["symbol", "trade_date"],
            keep="last",
        )
        bars = bars.merge(daily_basic, on=["symbol", "trade_date"], how="left")
    else:
        bars["circ_mv"] = np.nan

    lineage = {
        "latest_snapshot": latest,
        "latest_pointer_path": str(LATEST_PATH),
        "latest_pointer_sha256": latest_pointer_sha256,
        "bars_root": str(bars_root),
        "stock_basic_path": str(STOCK_BASIC_PATH),
        "daily_basic_path": str(DAILY_BASIC_PATH),
        "rows_after_filters": int(len(bars)),
        "symbol_count_after_filters": int(bars["symbol"].nunique()),
        "date_min": str(bars["trade_date"].min()),
        "date_max": str(bars["trade_date"].max()),
        "circ_mv_coverage": float(bars["circ_mv"].notna().mean()),
    }
    return bars, lineage


def _future_min_from_entry(series: pd.Series, horizon: int) -> pd.Series:
    entry_series = series.shift(-1)
    return entry_series.iloc[::-1].rolling(
        horizon + 1,
        min_periods=horizon + 1,
    ).min().iloc[::-1]


def _add_features(frame: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    df = frame.copy()
    grouped = df.groupby("symbol", group_keys=False)

    for period in [1, 3, 5, 10, 20, 40, 60, 120]:
        df[f"ret{period}"] = grouped["price"].pct_change(period)

    df["daily_ret"] = grouped["price"].pct_change(1)
    for window in [10, 20, 60]:
        df[f"vol{window}"] = grouped["daily_ret"].transform(
            lambda s, window=window: s.rolling(
                window,
                min_periods=max(6, int(window * 0.60)),
            ).std()
        )
    for window in [5, 10, 20, 60, 120]:
        min_periods = max(3, int(window * 0.75))
        df[f"ma{window}"] = grouped["price"].transform(
            lambda s, window=window, min_periods=min_periods: s.rolling(
                window,
                min_periods=min_periods,
            ).mean()
        )
        df[f"high{window}"] = grouped["price"].transform(
            lambda s, window=window, min_periods=min_periods: s.rolling(
                window,
                min_periods=min_periods,
            ).max()
        )
        df[f"low{window}"] = grouped["price"].transform(
            lambda s, window=window, min_periods=min_periods: s.rolling(
                window,
                min_periods=min_periods,
            ).min()
        )

    df["ret5_prev5"] = grouped["ret5"].shift(5)
    df["ret20_lag20"] = grouped["ret20"].shift(20)
    df["amount_ma5"] = grouped["amount"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["amount_ma20_prev"] = grouped["amount"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean()
    )
    df["amount_ma60_prev"] = grouped["amount"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=30).mean()
    )
    df["amount_ratio1_20"] = df["amount"] / df["amount_ma20_prev"]
    df["amount_ratio5_20"] = df["amount_ma5"] / df["amount_ma20_prev"]
    df["amount_ratio20_60"] = df["amount_ma20_prev"] / df["amount_ma60_prev"]

    df["close_vs_ma20"] = df["price"] / df["ma20"] - 1.0
    df["close_vs_ma60"] = df["price"] / df["ma60"] - 1.0
    df["high20_pos"] = df["price"] / df["high20"]
    df["high60_pos"] = df["price"] / df["high60"]
    df["range20"] = df["high20"] / df["low20"] - 1.0
    df["range60"] = df["high60"] / df["low60"] - 1.0
    df["limit_like"] = df["ret1"] >= 0.095
    df["recent_limit_count10"] = grouped["limit_like"].transform(
        lambda s: s.rolling(10, min_periods=1).sum()
    )

    for horizon in [5, 10, 20, 40]:
        df[f"fwd{horizon}"] = (
            grouped["price"].shift(-(horizon + 1)) / grouped["price"].shift(-1) - 1.0
        )
    df["entry_date"] = grouped["trade_date"].shift(-1)
    df["entry_price"] = grouped["price"].shift(-1)
    future_min20 = grouped["price"].transform(lambda s: _future_min_from_entry(s, 20))
    df["max_drawdown20"] = future_min20 / df["entry_price"] - 1.0
    df["outcome_score20"] = df["fwd20"] + 0.50 * df["max_drawdown20"].clip(
        lower=-0.30,
        upper=0.0,
    )

    validation_mask = df["trade_date"] >= config.start_date
    industry = (
        df[validation_mask]
        .groupby(["trade_date", "industry"])
        .agg(
            industry_count=("symbol", "nunique"),
            industry_ret20_median=("ret20", "median"),
            industry_ret60_median=("ret60", "median"),
            industry_pos20=("ret20", lambda s: float((s > 0).mean())),
            industry_breakout20=("high20_pos", lambda s: float((s >= 0.98).mean())),
        )
        .reset_index()
    )
    for column in [
        "industry_ret20_median",
        "industry_ret60_median",
        "industry_pos20",
        "industry_breakout20",
    ]:
        industry[f"{column}_pct"] = industry.groupby("trade_date")[column].rank(
            pct=True
        )
    df = df.merge(industry, on=["trade_date", "industry"], how="left")

    df["risk_adj20"] = (df["ret20"] + 0.35 * df["ret60"]) / np.maximum(
        df["vol20"],
        0.01,
    )
    df["risk_adj_fast"] = (df["ret10"] + 0.45 * df["ret20"]) / np.maximum(
        df["vol10"],
        0.01,
    )
    df["fresh_accel"] = (
        df["ret5"] - df["ret5_prev5"].fillna(0.0)
    ) + 0.50 * (df["ret20"] - df["ret20_lag20"].fillna(0.0))

    rank_columns = [
        "ret5",
        "ret20",
        "ret60",
        "risk_adj20",
        "risk_adj_fast",
        "fresh_accel",
        "amount_ratio1_20",
        "amount_ratio5_20",
        "high20_pos",
        "high60_pos",
        "close_vs_ma20",
        "vol20",
        "circ_mv",
    ]
    for column in rank_columns:
        mask = validation_mask & df[column].notna() & np.isfinite(df[column])
        df.loc[mask, f"{column}_pct"] = df.loc[mask].groupby("trade_date")[
            column
        ].rank(pct=True)
    df["low_vol_score"] = 1.0 - df["vol20_pct"]
    df["small_cap_score"] = 1.0 - df["circ_mv_pct"]

    df["score_quality_momentum"] = (
        0.35 * df["risk_adj20_pct"]
        + 0.15 * df["fresh_accel_pct"]
        + 0.15 * df["amount_ratio5_20_pct"]
        + 0.15 * df["industry_ret20_median_pct"]
        + 0.10 * df["industry_pos20_pct"]
        + 0.10 * df["low_vol_score"]
    )
    df["score_starter_momentum"] = (
        0.30 * df["risk_adj_fast_pct"]
        + 0.20 * df["fresh_accel_pct"]
        + 0.20 * df["amount_ratio1_20_pct"]
        + 0.15 * df["ret20_pct"]
        + 0.10 * df["industry_ret20_median_pct"]
        + 0.05 * df["low_vol_score"]
    )
    df["score_breakout_quality"] = (
        0.30 * df["risk_adj20_pct"]
        + 0.20 * df["high60_pos_pct"]
        + 0.15 * df["ret20_pct"]
        + 0.15 * df["amount_ratio5_20_pct"]
        + 0.10 * df["industry_ret20_median_pct"]
        + 0.10 * df["low_vol_score"]
    )

    return df


def _add_labels_and_segments(df: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    work = df.copy()
    eligible = (
        (work["trade_date"] >= config.start_date)
        & work["fwd20"].notna()
        & work["outcome_score20"].notna()
        & (work["amount_ma20_prev"] >= config.min_amount_ma20)
    )
    work["validation_eligible"] = eligible

    label_frame = work.loc[eligible, ["trade_date", "outcome_score20"]].copy()
    q = label_frame.groupby("trade_date")["outcome_score20"].quantile(
        [config.loser_quantile, config.winner_quantile]
    ).unstack()
    q = q.rename(
        columns={
            config.loser_quantile: "outcome_q10",
            config.winner_quantile: "outcome_q90",
        }
    ).reset_index()
    work = work.merge(q, on="trade_date", how="left")
    work["winner20"] = (
        work["validation_eligible"]
        & (work["outcome_score20"] >= work["outcome_q90"])
        & (work["fwd20"] > 0)
    )
    work["loser20"] = (
        work["validation_eligible"]
        & (
            (work["outcome_score20"] <= work["outcome_q10"])
            | (work["fwd20"] <= -0.08)
        )
    )

    market = (
        work.loc[work["trade_date"] >= config.start_date]
        .groupby("trade_date")
        .agg(
            market_ret20_median=("ret20", "median"),
            market_pos20=("ret20", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    conditions = [
        (market["market_ret20_median"] > 0.05) & (market["market_pos20"] > 0.55),
        (market["market_ret20_median"] < -0.03) & (market["market_pos20"] < 0.45),
    ]
    market["market_regime"] = np.select(conditions, ["bull", "bear"], "neutral")
    work = work.merge(market, on="trade_date", how="left")

    work["year"] = work["trade_date"].str[:4]
    work["sample_split"] = np.where(
        work["trade_date"] >= config.test_start_date,
        "test_2025_plus",
        "train_2021_2024",
    )
    work["liquidity_bucket"] = work.groupby("trade_date", group_keys=False)[
        "amount_ma20_prev"
    ].transform(lambda s: _safe_qcut_bucket(s, ["low_liquidity", "mid_liquidity", "high_liquidity"]))
    work["size_bucket"] = work.groupby("trade_date", group_keys=False)["circ_mv"].transform(
        lambda s: _safe_qcut_bucket(s, ["small_cap", "mid_cap", "large_cap"])
    )
    return work


def _date_spearman(frame: pd.DataFrame, factor: str, target: str) -> dict[str, float]:
    values: list[float] = []
    for _, group in frame[[factor, target, "trade_date"]].dropna().groupby(
        "trade_date"
    ):
        if len(group) < 200:
            continue
        corr = group[factor].rank().corr(group[target].rank())
        if pd.notna(corr) and np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return {"ic_mean": math.nan, "ic_std": math.nan, "ic_positive_rate": math.nan, "ic_n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "ic_mean": float(arr.mean()),
        "ic_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ic_positive_rate": float((arr > 0).mean()),
        "ic_n": int(len(arr)),
    }


def _factor_deciles(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    cols = [
        "trade_date",
        "symbol",
        factor,
        "fwd20",
        "max_drawdown20",
        "outcome_score20",
        "winner20",
        "loser20",
    ]
    d = frame.loc[frame["validation_eligible"], cols].dropna(subset=[factor, "fwd20"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["factor_pct"] = d.groupby("trade_date")[factor].rank(method="average", pct=True)
    d["decile"] = np.ceil(d["factor_pct"] * 10).clip(1, 10).astype(int)
    grouped = d.groupby("decile", as_index=False).agg(
        n=("symbol", "count"),
        symbol_count=("symbol", "nunique"),
        avg_fwd20=("fwd20", "mean"),
        median_fwd20=("fwd20", "median"),
        win_rate20=("fwd20", lambda s: float((s > 0).mean())),
        ge10_rate20=("fwd20", lambda s: float((s >= 0.10).mean())),
        le_minus8_rate20=("fwd20", lambda s: float((s <= -0.08).mean())),
        avg_max_drawdown20=("max_drawdown20", "mean"),
        avg_outcome_score20=("outcome_score20", "mean"),
        winner_rate=("winner20", "mean"),
        loser_rate=("loser20", "mean"),
    )
    grouped.insert(0, "factor", factor)
    return grouped


def _evaluate_factors(df: pd.DataFrame, config: ValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_rows: list[dict[str, Any]] = []
    decile_frames: list[pd.DataFrame] = []
    eligible = df.loc[df["validation_eligible"]].copy()
    train = eligible[eligible["trade_date"] < config.test_start_date]
    test = eligible[eligible["trade_date"] >= config.test_start_date]

    for factor in FACTOR_COLUMNS:
        deciles = _factor_deciles(eligible, factor)
        if deciles.empty:
            continue
        decile_frames.append(deciles)
        top = deciles[deciles["decile"] == 10].iloc[0].to_dict()
        bottom = deciles[deciles["decile"] == 1].iloc[0].to_dict()

        train_deciles = _factor_deciles(train, factor)
        test_deciles = _factor_deciles(test, factor)
        train_top = (
            train_deciles[train_deciles["decile"] == 10].iloc[0].to_dict()
            if not train_deciles.empty
            else {}
        )
        test_top = (
            test_deciles[test_deciles["decile"] == 10].iloc[0].to_dict()
            if not test_deciles.empty
            else {}
        )

        ic_payload = _date_spearman(eligible, factor, "outcome_score20")
        factor_rows.append(
            {
                "factor": factor,
                "top_decile_n": top.get("n"),
                "top_avg_fwd20": top.get("avg_fwd20"),
                "top_median_fwd20": top.get("median_fwd20"),
                "top_win_rate20": top.get("win_rate20"),
                "top_ge10_rate20": top.get("ge10_rate20"),
                "top_le_minus8_rate20": top.get("le_minus8_rate20"),
                "top_avg_drawdown20": top.get("avg_max_drawdown20"),
                "top_winner_rate": top.get("winner_rate"),
                "bottom_avg_fwd20": bottom.get("avg_fwd20"),
                "spread_avg_fwd20": float(top.get("avg_fwd20", np.nan) - bottom.get("avg_fwd20", np.nan)),
                "spread_outcome_score20": float(top.get("avg_outcome_score20", np.nan) - bottom.get("avg_outcome_score20", np.nan)),
                "train_top_avg_fwd20": train_top.get("avg_fwd20"),
                "train_top_median_fwd20": train_top.get("median_fwd20"),
                "train_top_win_rate20": train_top.get("win_rate20"),
                "test_top_avg_fwd20": test_top.get("avg_fwd20"),
                "test_top_median_fwd20": test_top.get("median_fwd20"),
                "test_top_win_rate20": test_top.get("win_rate20"),
                **ic_payload,
            }
        )

    factor_summary = pd.DataFrame(factor_rows)
    if not factor_summary.empty:
        factor_summary["validation_score"] = (
            2.0 * factor_summary["test_top_avg_fwd20"].fillna(-9)
            + 1.5 * factor_summary["test_top_median_fwd20"].fillna(-9)
            + 0.5 * factor_summary["spread_avg_fwd20"].fillna(-9)
            + 0.5 * factor_summary["ic_mean"].fillna(-9)
            - 0.4 * factor_summary["top_le_minus8_rate20"].fillna(1)
        )
        gates = factor_summary.apply(_factor_gate, axis=1, result_type="expand")
        factor_summary["validation_status"] = gates[0]
        factor_summary["gate_reason"] = gates[1]
        factor_summary = factor_summary.sort_values("validation_score", ascending=False)
    decile_summary = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    return factor_summary, decile_summary


def _top_bucket(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    d = frame.loc[frame["validation_eligible"]].dropna(subset=[factor, "fwd20"]).copy()
    if d.empty:
        return d
    d["factor_pct"] = d.groupby("trade_date")[factor].rank(method="average", pct=True)
    return d[d["factor_pct"] >= 0.90].copy()


def _segment_validation(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    segment_specs = [
        ("year", "year"),
        ("market_regime", "market_regime"),
        ("liquidity_bucket", "liquidity_bucket"),
        ("size_bucket", "size_bucket"),
        ("industry", "industry"),
    ]
    for factor in factors:
        top = _top_bucket(df, factor)
        if top.empty:
            continue
        for segment_type, column in segment_specs:
            for segment, group in top.groupby(column, dropna=True):
                if len(group) < 30:
                    continue
                payload = _metric_payload(group)
                rows.append(
                    {
                        "factor": factor,
                        "segment_type": segment_type,
                        "segment": str(segment),
                        **payload,
                    }
                )
    return pd.DataFrame(rows)


def _portfolio_proxy(df: pd.DataFrame, factors: list[str], config: ValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = df.loc[df["validation_eligible"]].copy()
    base_columns = [
        "trade_date",
        "symbol",
        "name",
        "industry",
        "bar_index",
        "fwd20",
        "max_drawdown20",
        "outcome_score20",
        "limit_like",
        "recent_limit_count10",
        "amount_ma20_prev",
    ]

    for factor in factors:
        data = eligible.dropna(subset=[factor, "fwd20"]).copy()
        if data.empty:
            continue
        data = data.sort_values(["trade_date", factor], ascending=[True, False])
        for per_day in config.top_per_day_values:
            selected: list[dict[str, Any]] = []
            last_bar_by_symbol: dict[str, int] = {}
            for _, group in data.groupby("trade_date", sort=True):
                picked = 0
                for item in group.head(max(50, per_day * 10))[base_columns + [factor]].itertuples(index=False):
                    if picked >= per_day:
                        break
                    last_bar = last_bar_by_symbol.get(str(item.symbol), -100_000)
                    if int(item.bar_index) - last_bar <= config.cooldown_days:
                        continue
                    selected.append(item._asdict())
                    last_bar_by_symbol[str(item.symbol)] = int(item.bar_index)
                    picked += 1
            selected_frame = pd.DataFrame(selected)
            payload = _metric_payload(selected_frame) if not selected_frame.empty else _metric_payload(pd.DataFrame())
            rows.append(
                {
                    "factor": factor,
                    "per_day": per_day,
                    "cooldown_days": config.cooldown_days,
                    **payload,
                    "limit_like_signal_rate": (
                        float(selected_frame["limit_like"].mean()) if not selected_frame.empty else math.nan
                    ),
                    "recent_limit_count10_mean": (
                        float(selected_frame["recent_limit_count10"].mean()) if not selected_frame.empty else math.nan
                    ),
                    "median_amount_ma20": (
                        float(selected_frame["amount_ma20_prev"].median()) if not selected_frame.empty else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _current_candidates(df: pd.DataFrame, factors: list[str], latest_trade_date: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    current = df[df["trade_date"] == latest_trade_date].copy()
    if current.empty:
        return pd.DataFrame()
    for factor in factors:
        if factor not in current.columns:
            continue
        subset = current.dropna(subset=[factor]).copy()
        if subset.empty:
            continue
        subset = subset.sort_values(factor, ascending=False).head(20)
        subset.insert(0, "factor", factor)
        subset["factor_value"] = subset[factor]
        rows.append(
            subset[
                [
                    "factor",
                    "factor_value",
                    "trade_date",
                    "symbol",
                    "name",
                    "industry",
                    "ret5",
                    "ret20",
                    "ret60",
                    "risk_adj20",
                    "amount_ratio1_20",
                    "amount_ma20_prev",
                    "close_vs_ma20",
                    "high60_pos",
                    "industry_ret20_median",
                    "market_regime",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_report(
    output_dir: Path,
    config: ValidationConfig,
    lineage: dict[str, Any],
    label_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    accepted_factors: list[str],
    diagnostic_factors: list[str],
) -> None:
    def pct(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.2%}"

    lines: list[str] = []
    lines.append("# A股全市场因子广泛验证")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().astimezone().isoformat()}")
    lines.append("")
    lines.append("## 数据边界")
    lines.append("")
    latest = lineage.get("latest_snapshot", {})
    lines.append(
        "- 数据源：由当前 strict Parquet active pointer "
        f"`{lineage.get('latest_pointer_path')}` 绑定的 table root "
        f"`{lineage.get('bars_root')}`。"
    )
    lines.append("- 行业映射：`data/parquet/cn/dag_core_raw/table=stock_basic/part.parquet`。")
    lines.append("- 市值分层：`fundamental_raw/table=daily_basic` 去重后的 `circ_mv`。")
    lines.append(f"- Snapshot：`{latest.get('snapshot_id', 'UNKNOWN')}`；latest_trade_date=`{latest.get('latest_trade_date', 'UNKNOWN')}`。")
    lines.append(f"- 样本：{lineage.get('rows_after_filters'):,} 行，{lineage.get('symbol_count_after_filters'):,} 只，日期 {lineage.get('date_min')}..{lineage.get('date_max')}。")
    lines.append("- 未调用 Tushare、yfinance、LLM、broker 或真实下单接口。")
    lines.append("")
    lines.append("## 标签")
    lines.append("")
    lines.append("- `fwd20`：信号次一交易日入场，持有 20 个交易日的复权收益。")
    lines.append("- `max_drawdown20`：入场后 20 个交易日窗口内最大收盘回撤。")
    lines.append("- `outcome_score20 = fwd20 + 0.5 * clipped(max_drawdown20)`，回撤越大标签越差。")
    lines.append("- `winner20`：每日 `outcome_score20` 前 10% 且 `fwd20 > 0`。")
    lines.append("- `loser20`：每日 `outcome_score20` 后 10% 或 `fwd20 <= -8%`。")
    lines.append("")
    lines.append("## 因子排序")
    lines.append("")
    lines.append("| rank | factor | test top avg20 | test top median20 | test top win | spread avg20 | IC mean | top loss<=-8% |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(factor_summary.head(12).itertuples(index=False), start=1):
        lines.append(
            f"| {i} | `{row.factor}` | {pct(row.test_top_avg_fwd20)} | "
            f"{pct(row.test_top_median_fwd20)} | {pct(row.test_top_win_rate20)} | "
            f"{pct(row.spread_avg_fwd20)} | {float(row.ic_mean):.4f} | "
            f"{pct(row.top_le_minus8_rate20)} |"
        )
    lines.append("")
    lines.append("## 组合代理")
    lines.append("")
    lines.append("这里不是正式组合回测，只是每日 top N + 同票冷却的候选质量代理。")
    lines.append("")
    lines.append("| factor | per_day | n | avg20 | median20 | win | >=10% | <=-8% | signal limit-like |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in portfolio_summary.sort_values(
        ["avg_fwd20", "median_fwd20"],
        ascending=False,
    ).head(15).itertuples(index=False):
        lines.append(
            f"| `{row.factor}` | {row.per_day} | {int(row.n)} | {pct(row.avg_fwd20)} | "
            f"{pct(row.median_fwd20)} | {pct(row.win_rate20)} | {pct(row.ge10_rate20)} | "
            f"{pct(row.le_minus8_rate20)} | {pct(row.limit_like_signal_rate)} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    if accepted_factors:
        lines.append(
            "- 严格通过广泛验证闸门的是："
            + "、".join(f"`{factor}`" for factor in accepted_factors)
            + "。"
        )
        if "low_vol_score" in accepted_factors:
            lines.append("- `low_vol_score` 更适合作为回撤/质量过滤，不是 aggressive buy trigger。")
    else:
        lines.append("- 没有因子通过严格广泛验证闸门；今日不应新增 aggressive buy trigger。")
    rejected = factor_summary[factor_summary["validation_status"] == "reject_as_buy_trigger"]["factor"].head(6).tolist()
    if rejected:
        lines.append(
            "- 不应作为买入触发的高排名弱因子："
            + "、".join(f"`{factor}`" for factor in rejected)
            + "。"
        )
    if diagnostic_factors:
        lines.append(
            "- 诊断覆盖因子："
            + "、".join(f"`{factor}`" for factor in diagnostic_factors)
            + "。"
        )
    lines.append("- 接受标准不是命中某几只股票，而是全市场 decile spread、IC、样本外 top bucket 和组合代理同时站得住。")
    lines.append("- 如果一个因子只在 2025+ 有改善、全周期中位数仍弱，应归为 research/paper，不进入 production_factor。")
    lines.append("- 下一步若要进入正式因子治理，需要补交易成本、涨跌停成交可得性、停牌和逐日 MTM 组合回测。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    for name in [
        "manifest.json",
        "label_summary.csv",
        "factor_summary.csv",
        "factor_deciles.csv",
        "segment_validation.csv",
        "portfolio_proxy_summary.csv",
        "current_top_candidates.csv",
    ]:
        lines.append(f"- `{name}`")
    output_dir.joinpath("broad_factor_validation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(output_dir: Path, config: ValidationConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bars, lineage = _load_base_frame(config)
    featured = _add_features(bars, config)
    labelled = _add_labels_and_segments(featured, config)
    eligible = labelled[labelled["validation_eligible"]].copy()

    label_summary = pd.DataFrame(
        [
            {"split": "all", **_metric_payload(eligible)},
            {
                "split": "train_2021_2024",
                **_metric_payload(eligible[eligible["trade_date"] < config.test_start_date]),
            },
            {
                "split": "test_2025_plus",
                **_metric_payload(eligible[eligible["trade_date"] >= config.test_start_date]),
            },
        ]
    )
    factor_summary, decile_summary = _evaluate_factors(labelled, config)
    if factor_summary.empty:
        accepted: list[str] = []
        diagnostic: list[str] = []
    else:
        accepted = factor_summary.loc[
            factor_summary["validation_status"] == "pass_filter_only",
            "factor",
        ].tolist()
        diagnostic = []
        for factor in [*accepted, *factor_summary.head(6)["factor"].tolist()]:
            if factor not in diagnostic:
                diagnostic.append(factor)
    segment_summary = _segment_validation(labelled, diagnostic)
    portfolio_summary = _portfolio_proxy(labelled, diagnostic, config)
    latest_trade_date = str(lineage.get("latest_snapshot", {}).get("latest_trade_date") or labelled["trade_date"].max())
    current_candidates = _current_candidates(labelled, accepted, latest_trade_date)

    label_summary.to_csv(output_dir / "label_summary.csv", index=False, encoding="utf-8-sig")
    factor_summary.to_csv(output_dir / "factor_summary.csv", index=False, encoding="utf-8-sig")
    decile_summary.to_csv(output_dir / "factor_deciles.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(output_dir / "segment_validation.csv", index=False, encoding="utf-8-sig")
    portfolio_summary.to_csv(output_dir / "portfolio_proxy_summary.csv", index=False, encoding="utf-8-sig")
    current_candidates.to_csv(output_dir / "current_top_candidates.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "script": "scripts/validate_cn_aggressive_factor_universe.py",
        "config": asdict(config),
        "lineage": lineage,
        "accepted_factors": accepted,
        "diagnostic_factors": diagnostic,
        "artifact_files": [
            "broad_factor_validation_report.md",
            "manifest.json",
            "label_summary.csv",
            "factor_summary.csv",
            "factor_deciles.csv",
            "segment_validation.csv",
            "portfolio_proxy_summary.csv",
            "current_top_candidates.csv",
        ],
        "live_calls": {
            "tushare": False,
            "yfinance": False,
            "llm": False,
            "broker": False,
            "notion": False,
        },
    }
    output_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        output_dir,
        config,
        lineage,
        label_summary,
        factor_summary,
        segment_summary,
        portfolio_summary,
        accepted,
        diagnostic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="20210104")
    parser.add_argument("--warmup-date", default="20200101")
    parser.add_argument("--test-start-date", default="20250101")
    parser.add_argument("--min-amount-ma20", type=float, default=50_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValidationConfig(
        start_date=str(args.start_date),
        warmup_date=str(args.warmup_date),
        test_start_date=str(args.test_start_date),
        min_amount_ma20=float(args.min_amount_ma20),
    )
    run(args.output_dir, config)


if __name__ == "__main__":
    main()
