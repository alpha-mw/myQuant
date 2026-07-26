"""Five-pillar PIT scoring and deterministic Top-24 sealing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

STATUS_AVAILABLE = "AVAILABLE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
MAIN_METRICS = (
    "fin_roe",
    "fin_ocf_to_profit",
    "fin_net_profit_yoy",
    "fin_debt_to_assets",
    "fcf_to_price",
)
OPTIONAL_METRICS = ("fin_roa", "fin_fcf_to_profit", "forecast_revision")
ALL_METRICS = MAIN_METRICS + OPTIONAL_METRICS
REVERSE_METRICS = frozenset({"fin_debt_to_assets"})
PILLARS: tuple[tuple[str, str, str | None, float], ...] = (
    ("profitability", "fin_roe", "fin_roa", 0.25),
    ("cash_conversion", "fin_ocf_to_profit", "fin_fcf_to_profit", 0.25),
    ("growth_expectations", "fin_net_profit_yoy", "forecast_revision", 0.20),
    ("balance_sheet_resilience", "fin_debt_to_assets", None, 0.15),
    ("valuation", "fcf_to_price", None, 0.15),
)
REQUIRED_SNAPSHOT_COLUMNS = frozenset(
    {
        "symbol",
        "industry",
        "in_universe",
        "research_eligible",
        "membership_conflict",
        "membership_is_pit",
        "universe_id",
        "availability",
        "flow_basis",
        "balance_sheet_basis",
        "capex_sign_convention",
        "net_profit_ttm",
        "market_cap",
        "cfo_ttm",
        "capex_ttm",
        *MAIN_METRICS[:-1],
    }
)
REQUIRED_HISTORY_COLUMNS = frozenset(
    {"symbol", "trade_date", "availability", "is_open_day", "metric", "value"}
)
REQUIRED_WIDE_HISTORY_COLUMNS = frozenset(
    {"symbol", "trade_date", "availability", "is_open_day", *MAIN_METRICS}
)


@dataclass(frozen=True)
class FundamentalCandidateSet:
    scored: pd.DataFrame
    ranked_symbols: tuple[str, ...]
    sealed_symbols: tuple[str, ...]
    appended_holdings: tuple[str, ...]


def _columns(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _timestamps(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{label} contains an invalid timestamp")
    return result


def _cutoff(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return result.tz_convert("UTC")


def _strict_bool(value: object) -> bool | None:
    return bool(value) if isinstance(value, (bool, np.bool_)) else None


def _finite(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _historical_percentile(current: float, values: pd.Series) -> float:
    array = values.to_numpy(dtype=float, copy=False)
    less = int(np.sum(array < current))
    equal = int(np.sum(array == current))
    return float((less + 0.5 * equal) / len(array))


def _score_fundamental_universe(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
    holdings: Iterable[str] = (),
    top_n: int = 24,
    _history_lookup: dict[tuple[str, str], pd.Series] | None = None,
) -> FundamentalCandidateSet:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _columns(snapshot, REQUIRED_SNAPSHOT_COLUMNS, "snapshot")
    cutoff_ts = _cutoff(cutoff)
    working = snapshot.copy(deep=True)
    working["symbol"] = working["symbol"].astype(str).str.strip()
    working["industry"] = working["industry"].where(working["industry"].notna(), "")
    working["industry"] = working["industry"].astype(str).str.strip()
    if (working["symbol"] == "").any() or working["symbol"].duplicated().any():
        raise ValueError("snapshot symbols must be unique and non-empty")
    working["availability"] = _timestamps(working["availability"], "snapshot.availability")
    if "fin_fcf_to_profit" not in working:
        working["fin_fcf_to_profit"] = np.nan
    if _history_lookup is None:
        _columns(history, REQUIRED_HISTORY_COLUMNS, "history")
        hist = history.copy(deep=True)
        hist["symbol"] = hist["symbol"].astype(str).str.strip()
        hist["metric"] = hist["metric"].astype(str).str.strip()
        hist["trade_date"] = _timestamps(hist["trade_date"], "history.trade_date")
        hist["availability"] = _timestamps(hist["availability"], "history.availability")
        if any(_strict_bool(value) is not True for value in hist["is_open_day"]):
            raise ValueError("history rows must be explicit canonical open-market sessions")
        if hist.duplicated(["symbol", "metric", "trade_date"], keep=False).any():
            raise ValueError("history contains duplicate symbol/metric/open-session rows")
        history_lookup: dict[tuple[str, str], pd.Series] = {}
        eligible_history = hist.loc[
            (hist["trade_date"] <= cutoff_ts) & (hist["availability"] <= cutoff_ts)
        ].sort_values(["symbol", "metric", "trade_date"], kind="mergesort")
        for (symbol, metric), rows in eligible_history.groupby(
            ["symbol", "metric"],
            sort=False,
            observed=True,
        ):
            numeric = pd.to_numeric(rows.tail(756)["value"], errors="coerce")
            numeric = numeric[np.isfinite(numeric)]
            if len(numeric) >= 252:
                history_lookup[(str(symbol), str(metric))] = numeric
    else:
        history_lookup = _history_lookup

    reasons: dict[str, list[str]] = {symbol: [] for symbol in working["symbol"]}
    eligible = pd.Series(True, index=working.index, dtype=bool)
    for index, row in working.iterrows():
        symbol = str(row["symbol"])
        for field in (
            "in_universe",
            "research_eligible",
            "membership_conflict",
            "membership_is_pit",
        ):
            if _strict_bool(row[field]) is None:
                reasons[symbol].append(f"invalid_{field}")
        checks = (
            (str(row["universe_id"]).strip() != "CN/full_a", "wrong_universe_id"),
            (_strict_bool(row["membership_is_pit"]) is not True, "membership_not_pit"),
            (_strict_bool(row["in_universe"]) is not True, "not_in_pit_universe"),
            (_strict_bool(row["research_eligible"]) is not True, "not_research_eligible"),
            (_strict_bool(row["membership_conflict"]) is not False, "pit_membership_conflict"),
            (row["availability"] > cutoff_ts, "evidence_after_cutoff"),
            (str(row["flow_basis"]).strip() != "LATEST_TTM", "flow_basis_not_latest_ttm"),
            (
                str(row["balance_sheet_basis"]).strip() != "LATEST_REPORT_PERIOD",
                "balance_sheet_basis_not_latest_report_period",
            ),
            (
                str(row["capex_sign_convention"]).strip() != "POSITIVE_OUTFLOW",
                "capex_sign_convention_invalid",
            ),
            (not str(row["industry"]), "industry_unknown"),
            ((_finite(row["net_profit_ttm"]) or 0.0) <= 0.0, "nonpositive_net_profit_ttm"),
            ((_finite(row["market_cap"]) or 0.0) <= 0.0, "nonpositive_market_cap"),
        )
        reasons[symbol].extend(reason for failed, reason in checks if failed)
        cfo, capex, market_cap = (
            _finite(row["cfo_ttm"]),
            _finite(row["capex_ttm"]),
            _finite(row["market_cap"]),
        )
        if cfo is None or capex is None or market_cap is None or market_cap <= 0:
            reasons[symbol].append("fcf_inputs_missing")
        else:
            fcf = cfo - capex
            working.at[index, "fcf_to_price"] = fcf / market_cap
            net_profit = _finite(row["net_profit_ttm"])
            if net_profit is not None and net_profit > 0:
                working.at[index, "fin_fcf_to_profit"] = fcf / net_profit
        for metric in MAIN_METRICS:
            if _finite(working.at[index, metric]) is None:
                reasons[symbol].append(f"missing_main_metric:{metric}")
        eligible.at[index] = not reasons[symbol]

    sizes = working.loc[eligible].groupby("industry", dropna=False).size()
    for index, row in working.loc[eligible].iterrows():
        if int(sizes.get(row["industry"], 0)) < 20:
            reasons[str(row["symbol"])].append("industry_sample_below_20")
            eligible.at[index] = False
    metric_values = {
        metric: (
            pd.to_numeric(working[metric], errors="coerce")
            if metric in working
            else pd.Series(np.nan, index=working.index)
        )
        for metric in ALL_METRICS
    }

    main_histories: dict[tuple[int, str], pd.Series] = {}
    preflight_indexes = list(working.index[eligible])
    for metric in MAIN_METRICS:
        values = metric_values[metric]
        for _, indexes in working.loc[preflight_indexes].groupby("industry").groups.items():
            valid = [index for index in indexes if np.isfinite(values.at[index])]
            if len(valid) < 20:
                for index in indexes:
                    reasons[str(working.at[index, "symbol"])].append(
                        f"industry_metric_sample_below_20:{metric}"
                    )
                    eligible.at[index] = False
                continue
            for index in valid:
                symbol = str(working.at[index, "symbol"])
                own = history_lookup.get((symbol, metric))
                if own is None:
                    reasons[symbol].append(f"self_history_below_252:{metric}")
                    eligible.at[index] = False
                else:
                    main_histories[(index, metric)] = own
    final_sizes = working.loc[eligible].groupby("industry", dropna=False).size()
    for index, row in working.loc[eligible].iterrows():
        if int(final_sizes.get(row["industry"], 0)) < 20:
            reasons[str(row["symbol"])].append("final_industry_available_sample_below_20")
            eligible.at[index] = False

    optional_histories: dict[tuple[int, str], pd.Series] = {}
    while True:
        failures: dict[int, list[str]] = {}
        indexes_now = list(working.index[eligible])
        for metric in OPTIONAL_METRICS:
            values = metric_values[metric]
            for _, indexes in working.loc[indexes_now].groupby("industry").groups.items():
                provided = [index for index in indexes if np.isfinite(values.at[index])]
                if not provided:
                    continue
                if len(provided) < 20:
                    for index in provided:
                        failures.setdefault(index, []).append(
                            f"industry_optional_metric_sample_below_20:{metric}"
                        )
                    continue
                for index in provided:
                    symbol = str(working.at[index, "symbol"])
                    own = optional_histories.get((index, metric))
                    if own is None:
                        own = history_lookup.get((symbol, metric))
                    if own is None:
                        failures.setdefault(index, []).append(f"self_history_below_252:{metric}")
                    else:
                        optional_histories[(index, metric)] = own
        if not failures:
            break
        for index, failure_reasons in failures.items():
            reasons[str(working.at[index, "symbol"])].extend(failure_reasons)
            eligible.at[index] = False
        sizes = working.loc[eligible].groupby("industry", dropna=False).size()
        for index, row in working.loc[eligible].iterrows():
            if int(sizes.get(row["industry"], 0)) < 20:
                reasons[str(row["symbol"])].append("final_industry_available_sample_below_20")
                eligible.at[index] = False

    metric_scores: dict[str, pd.Series] = {}
    for metric in ALL_METRICS:
        values = metric_values[metric]
        scores = pd.Series(np.nan, index=working.index, dtype=float)
        for _, indexes in working.loc[eligible].groupby("industry").groups.items():
            valid = [index for index in indexes if np.isfinite(values.at[index])]
            if len(valid) < 20:
                if valid:
                    raise ValueError(f"metric readiness preflight drift: {metric}")
                continue
            industry_values = values.loc[valid]
            clipped = industry_values.clip(
                lower=float(industry_values.quantile(0.01, interpolation="linear")),
                upper=float(industry_values.quantile(0.99, interpolation="linear")),
            )
            industry_pct = clipped.rank(method="average", pct=True)
            for index in valid:
                symbol = str(working.at[index, "symbol"])
                own = main_histories.get((index, metric))
                if own is None:
                    own = optional_histories.get((index, metric))
                if own is None:
                    raise ValueError(f"metric history readiness preflight drift: {symbol}:{metric}")
                combined = 0.70 * float(industry_pct.at[index]) + 0.30 * _historical_percentile(
                    float(clipped.at[index]), own
                )
                scores.at[index] = 1.0 - combined if metric in REVERSE_METRICS else combined
        metric_scores[metric] = scores
        working[f"{metric}_score"] = scores

    pillars: list[str] = []
    total = pd.Series(0.0, index=working.index, dtype=float)
    for pillar, main_metric, optional_metric, weight in PILLARS:
        pillar_score = metric_scores[main_metric].copy()
        if optional_metric is not None:
            optional_score = metric_scores[optional_metric]
            ready = optional_score.notna()
            pillar_score.loc[ready] = (
                0.70 * pillar_score.loc[ready] + 0.30 * optional_score.loc[ready]
            )
        column = f"pillar_{pillar}"
        pillars.append(column)
        working[column] = pillar_score
        total = total.add(pillar_score.fillna(0).mul(weight))
    for index, row in working.iterrows():
        if any(pd.isna(working.at[index, column]) for column in pillars):
            reasons[str(row["symbol"])].append("pillar_score_unavailable")
            eligible.at[index] = False
    working["status"] = np.where(eligible, STATUS_AVAILABLE, STATUS_UNAVAILABLE)
    working["total_score"] = total.where(eligible, np.nan)
    working["unavailable_reasons"] = [
        tuple(dict.fromkeys(reasons[str(symbol)])) for symbol in working["symbol"]
    ]

    available = working.loc[working["status"] == STATUS_AVAILABLE].sort_values(
        ["total_score", "symbol"], ascending=[False, True], kind="mergesort"
    )
    ranked = tuple(available["symbol"].head(top_n).astype(str))
    normalized_holdings = tuple(
        dict.fromkeys(str(value).strip() for value in holdings if str(value).strip())
    )
    unknown = sorted(set(normalized_holdings).difference(set(working["symbol"])))
    if unknown:
        raise ValueError(f"holdings absent from sealed PIT snapshot: {unknown}")
    appended = tuple(symbol for symbol in normalized_holdings if symbol not in ranked)
    working["selected_top_n"] = working["symbol"].isin(ranked)
    working["sealed_holding"] = working["symbol"].isin(appended)
    return FundamentalCandidateSet(
        working.sort_values("symbol", kind="mergesort").reset_index(drop=True),
        ranked,
        ranked + appended,
        appended,
    )


def score_fundamental_universe(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
    holdings: Iterable[str] = (),
    top_n: int = 24,
) -> FundamentalCandidateSet:
    return _score_fundamental_universe(
        snapshot,
        history,
        cutoff=cutoff,
        holdings=holdings,
        top_n=top_n,
    )


def score_fundamental_universe_wide_history(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
    holdings: Iterable[str] = (),
    top_n: int = 24,
) -> FundamentalCandidateSet:
    """Score an equivalent one-row-per-symbol/session history without melting it."""

    _columns(history, REQUIRED_WIDE_HISTORY_COLUMNS, "wide_history")
    cutoff_ts = _cutoff(cutoff)
    hist = history.copy(deep=True)
    hist["symbol"] = hist["symbol"].astype(str).str.strip()
    hist["trade_date"] = _timestamps(hist["trade_date"], "wide_history.trade_date")
    hist["availability"] = _timestamps(hist["availability"], "wide_history.availability")
    if (hist["symbol"] == "").any():
        raise ValueError("wide_history symbols must be non-empty")
    if any(_strict_bool(value) is not True for value in hist["is_open_day"]):
        raise ValueError("wide_history rows must be explicit canonical open-market sessions")
    if hist.duplicated(["symbol", "trade_date"], keep=False).any():
        raise ValueError("wide_history contains duplicate symbol/open-session rows")
    eligible = hist.loc[
        (hist["trade_date"] <= cutoff_ts) & (hist["availability"] <= cutoff_ts)
    ].sort_values(["symbol", "trade_date"], kind="mergesort")
    history_lookup: dict[tuple[str, str], pd.Series] = {}
    for symbol, rows in eligible.groupby("symbol", sort=False, observed=True):
        latest = rows.tail(756)
        for metric in ALL_METRICS:
            if metric not in latest:
                continue
            numeric = pd.to_numeric(latest[metric], errors="coerce")
            numeric = numeric[np.isfinite(numeric)]
            if len(numeric) >= 252:
                history_lookup[(str(symbol), metric)] = numeric
    return _score_fundamental_universe(
        snapshot,
        pd.DataFrame(),
        cutoff=cutoff,
        holdings=holdings,
        top_n=top_n,
        _history_lookup=history_lookup,
    )


__all__ = [
    "ALL_METRICS",
    "FundamentalCandidateSet",
    "MAIN_METRICS",
    "OPTIONAL_METRICS",
    "PILLARS",
    "score_fundamental_universe",
    "score_fundamental_universe_wide_history",
]
