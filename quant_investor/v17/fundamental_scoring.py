"""Fail-closed v17 point-in-time fundamental candidate scoring.

This module is deliberately a pure transformation.  It neither discovers a
universe nor fetches missing data.  Callers must supply the canonical PIT
membership snapshot and metric history sealed for the decision cutoff.
"""

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
OPTIONAL_METRICS = (
    "fin_roa",
    "fin_fcf_to_profit",
    "forecast_revision",
)
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


@dataclass(frozen=True)
class FundamentalCandidateSet:
    """Scored rows plus the immutable Top-N and appended-holdings seal."""

    scored: pd.DataFrame
    ranked_symbols: tuple[str, ...]
    sealed_symbols: tuple[str, ...]
    appended_holdings: tuple[str, ...]


def _require_columns(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _as_utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains an invalid timestamp")
    return parsed


def _cutoff_utc(cutoff: datetime | str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(cutoff)
    if parsed.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return parsed.tz_convert("UTC")


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _finite(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _industry_percentiles(values: pd.Series) -> pd.Series:
    """Average-tie percent ranks after the caller applies industry clipping."""

    return values.rank(method="average", pct=True)


def _historical_percentile(current: float, values: pd.Series) -> float:
    array = values.to_numpy(dtype=float, copy=False)
    less = int(np.sum(array < current))
    equal = int(np.sum(array == current))
    # A mid empirical CDF is deterministic and does not inject the current
    # observation into the already-sealed historical sample.
    return float((less + 0.5 * equal) / len(array))


def _history_for_metric(
    history: pd.DataFrame,
    *,
    symbol: str,
    metric: str,
    cutoff: pd.Timestamp,
) -> pd.Series | None:
    rows = history.loc[
        (history["symbol"] == symbol)
        & (history["metric"] == metric)
        & (history["trade_date"] <= cutoff)
        & (history["availability"] <= cutoff)
    ].sort_values("trade_date", kind="mergesort")
    rows = rows.drop_duplicates("trade_date", keep="last").tail(756)
    numeric = pd.to_numeric(rows["value"], errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) < 252:
        return None
    return numeric


def score_fundamental_universe(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
    holdings: Iterable[str] = (),
    top_n: int = 24,
) -> FundamentalCandidateSet:
    """Score the canonical PIT universe and seal Top-N plus current holdings.

    The snapshot must contain exactly one row per security selected by the
    caller's canonical ``CN/full_a`` PIT membership source.  Rows with missing
    or conflicting membership, late evidence, unknown/small industries,
    non-positive TTM attributable profit or market capitalisation, or any
    missing main metric are marked ``UNAVAILABLE``.  They are never zero-filled
    or ranked against the whole market as a fallback.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _require_columns(snapshot, REQUIRED_SNAPSHOT_COLUMNS, "snapshot")
    _require_columns(history, REQUIRED_HISTORY_COLUMNS, "history")
    cutoff_ts = _cutoff_utc(cutoff)

    working = snapshot.copy(deep=True)
    working["symbol"] = working["symbol"].astype(str).str.strip()
    working["industry"] = working["industry"].where(working["industry"].notna(), "")
    working["industry"] = working["industry"].astype(str).str.strip()
    if (working["symbol"] == "").any() or working["symbol"].duplicated().any():
        raise ValueError("snapshot symbols must be unique and non-empty")
    working["availability"] = _as_utc(working["availability"], "snapshot.availability")

    # ``fin_fcf_to_profit`` is optional, but whenever its inputs are present it
    # is derived from the same frozen FCF definition as ``fcf_to_price``.  A
    # caller-supplied value is never trusted over the canonical arithmetic.
    if "fin_fcf_to_profit" not in working:
        working["fin_fcf_to_profit"] = np.nan

    hist = history.copy(deep=True)
    hist["symbol"] = hist["symbol"].astype(str).str.strip()
    hist["metric"] = hist["metric"].astype(str).str.strip()
    hist["trade_date"] = _as_utc(hist["trade_date"], "history.trade_date")
    hist["availability"] = _as_utc(hist["availability"], "history.availability")
    if any(_strict_bool(value) is not True for value in hist["is_open_day"]):
        raise ValueError("history rows must be explicit canonical open-market sessions")
    if hist.duplicated(["symbol", "metric", "trade_date"], keep=False).any():
        raise ValueError("history contains duplicate symbol/metric/open-session rows")

    reasons: dict[str, list[str]] = {symbol: [] for symbol in working["symbol"]}
    eligible_mask = pd.Series(True, index=working.index, dtype=bool)
    for idx, row in working.iterrows():
        symbol = str(row["symbol"])
        for field in (
            "in_universe",
            "research_eligible",
            "membership_conflict",
            "membership_is_pit",
        ):
            if _strict_bool(row[field]) is None:
                reasons[symbol].append(f"invalid_{field}")
        if str(row["universe_id"]).strip() != "CN/full_a":
            reasons[symbol].append("wrong_universe_id")
        if _strict_bool(row["membership_is_pit"]) is not True:
            reasons[symbol].append("membership_not_pit")
        if _strict_bool(row["in_universe"]) is not True:
            reasons[symbol].append("not_in_pit_universe")
        if _strict_bool(row["research_eligible"]) is not True:
            reasons[symbol].append("not_research_eligible")
        if _strict_bool(row["membership_conflict"]) is not False:
            reasons[symbol].append("pit_membership_conflict")
        if row["availability"] > cutoff_ts:
            reasons[symbol].append("evidence_after_cutoff")
        if str(row["flow_basis"]).strip() != "LATEST_TTM":
            reasons[symbol].append("flow_basis_not_latest_ttm")
        if str(row["balance_sheet_basis"]).strip() != "LATEST_REPORT_PERIOD":
            reasons[symbol].append("balance_sheet_basis_not_latest_report_period")
        if str(row["capex_sign_convention"]).strip() != "POSITIVE_OUTFLOW":
            reasons[symbol].append("capex_sign_convention_invalid")
        industry = str(row["industry"])
        if not industry:
            reasons[symbol].append("industry_unknown")
        if (_finite(row["net_profit_ttm"]) or 0.0) <= 0.0:
            reasons[symbol].append("nonpositive_net_profit_ttm")
        market_cap = _finite(row["market_cap"])
        if (market_cap or 0.0) <= 0.0:
            reasons[symbol].append("nonpositive_market_cap")
        cfo = _finite(row["cfo_ttm"])
        capex = _finite(row["capex_ttm"])
        if cfo is None or capex is None or market_cap is None or market_cap <= 0.0:
            reasons[symbol].append("fcf_inputs_missing")
        else:
            # CAPEX is an outflow magnitude in the canonical contract.
            fcf = cfo - capex
            working.at[idx, "fcf_to_price"] = fcf / market_cap
            net_profit = _finite(row["net_profit_ttm"])
            if net_profit is not None and net_profit > 0.0:
                working.at[idx, "fin_fcf_to_profit"] = fcf / net_profit
        for metric in MAIN_METRICS:
            if _finite(working.at[idx, metric]) is None:
                reasons[symbol].append(f"missing_main_metric:{metric}")
        eligible_mask.at[idx] = not reasons[symbol]

    # Small-industry determination uses canonical, otherwise-eligible members,
    # not only securities that happen to have a value for an optional metric.
    industry_sizes = working.loc[eligible_mask].groupby("industry", dropna=False).size()
    for idx, row in working.loc[eligible_mask].iterrows():
        symbol = str(row["symbol"])
        if int(industry_sizes.get(row["industry"], 0)) < 20:
            reasons[symbol].append("industry_sample_below_20")
            eligible_mask.at[idx] = False

    metric_values = {
        metric: (
            pd.to_numeric(working[metric], errors="coerce")
            if metric in working
            else pd.Series(np.nan, index=working.index)
        )
        for metric in ALL_METRICS
    }

    # Preflight every main metric before computing any percentile.  Mutating
    # the peer set while metrics are being ranked makes earlier metrics depend
    # on iteration order, so readiness and scoring are deliberately two
    # separate passes.
    main_histories: dict[tuple[int, str], pd.Series] = {}
    preflight_indexes = list(working.index[eligible_mask])
    for metric in MAIN_METRICS:
        values = metric_values[metric]
        for _, indexes in working.loc[preflight_indexes].groupby("industry").groups.items():
            valid_indexes = [idx for idx in indexes if np.isfinite(values.at[idx])]
            if len(valid_indexes) < 20:
                for idx in indexes:
                    symbol = str(working.at[idx, "symbol"])
                    reasons[symbol].append(f"industry_metric_sample_below_20:{metric}")
                    eligible_mask.at[idx] = False
                continue
            for idx in valid_indexes:
                symbol = str(working.at[idx, "symbol"])
                own_history = _history_for_metric(
                    hist,
                    symbol=symbol,
                    metric=metric,
                    cutoff=cutoff_ts,
                )
                if own_history is None:
                    reasons[symbol].append(f"self_history_below_252:{metric}")
                    eligible_mask.at[idx] = False
                else:
                    main_histories[(idx, metric)] = own_history

    final_industry_sizes = working.loc[eligible_mask].groupby("industry", dropna=False).size()
    for idx, row in working.loc[eligible_mask].iterrows():
        symbol = str(row["symbol"])
        if int(final_industry_sizes.get(row["industry"], 0)) < 20:
            reasons[symbol].append("final_industry_available_sample_below_20")
            eligible_mask.at[idx] = False

    # An optional metric may be omitted for a security, in which case its main
    # metric receives 100% of the pillar weight.  Once a finite optional value
    # is supplied, however, it is evidence rather than an omission: both its
    # industry peer set and its own PIT history must be scoreable.  Iterate to
    # a fixed point because removing an unscoreable optional observation can
    # reduce the remaining peer set below 20.
    optional_histories: dict[tuple[int, str], pd.Series] = {}
    while True:
        optional_failures: dict[int, list[str]] = {}
        current_indexes = list(working.index[eligible_mask])
        for metric in OPTIONAL_METRICS:
            values = metric_values[metric]
            for _, indexes in working.loc[current_indexes].groupby("industry").groups.items():
                provided_indexes = [idx for idx in indexes if np.isfinite(values.at[idx])]
                if not provided_indexes:
                    continue
                if len(provided_indexes) < 20:
                    for idx in provided_indexes:
                        optional_failures.setdefault(idx, []).append(
                            f"industry_optional_metric_sample_below_20:{metric}"
                        )
                    continue
                for idx in provided_indexes:
                    symbol = str(working.at[idx, "symbol"])
                    own_history = optional_histories.get((idx, metric))
                    if own_history is None:
                        own_history = _history_for_metric(
                            hist,
                            symbol=symbol,
                            metric=metric,
                            cutoff=cutoff_ts,
                        )
                    if own_history is None:
                        optional_failures.setdefault(idx, []).append(
                            f"self_history_below_252:{metric}"
                        )
                    else:
                        optional_histories[(idx, metric)] = own_history

        if not optional_failures:
            break
        for idx, failure_reasons in optional_failures.items():
            symbol = str(working.at[idx, "symbol"])
            reasons[symbol].extend(failure_reasons)
            eligible_mask.at[idx] = False

        remaining_industry_sizes = (
            working.loc[eligible_mask].groupby("industry", dropna=False).size()
        )
        for idx, row in working.loc[eligible_mask].iterrows():
            if int(remaining_industry_sizes.get(row["industry"], 0)) < 20:
                symbol = str(row["symbol"])
                reasons[symbol].append("final_industry_available_sample_below_20")
                eligible_mask.at[idx] = False

    metric_scores: dict[str, pd.Series] = {}
    for metric in ALL_METRICS:
        values = metric_values[metric]
        metric_score = pd.Series(np.nan, index=working.index, dtype=float)
        for _, indexes in working.loc[eligible_mask].groupby("industry").groups.items():
            valid_indexes = [idx for idx in indexes if np.isfinite(values.at[idx])]
            # All finite main and optional observations have already passed a
            # fixed-point readiness preflight.  A finite optional observation
            # must never silently degrade into the main-only pillar path.
            if len(valid_indexes) < 20:
                if valid_indexes:
                    raise ValueError(f"metric readiness preflight drift: {metric}")
                continue
            industry_values = values.loc[valid_indexes]
            low = float(industry_values.quantile(0.01, interpolation="linear"))
            high = float(industry_values.quantile(0.99, interpolation="linear"))
            clipped = industry_values.clip(lower=low, upper=high)
            industry_pct = _industry_percentiles(clipped)
            for idx in valid_indexes:
                symbol = str(working.at[idx, "symbol"])
                own_history = main_histories.get((idx, metric))
                if own_history is None:
                    own_history = optional_histories.get((idx, metric))
                if own_history is None:
                    raise ValueError(f"metric history readiness preflight drift: {symbol}:{metric}")
                self_pct = _historical_percentile(float(clipped.at[idx]), own_history)
                combined = 0.70 * float(industry_pct.at[idx]) + 0.30 * self_pct
                metric_score.at[idx] = 1.0 - combined if metric in REVERSE_METRICS else combined
        metric_scores[metric] = metric_score
        working[f"{metric}_score"] = metric_score

    pillar_columns: list[str] = []
    total = pd.Series(0.0, index=working.index, dtype=float)
    for pillar, main_metric, optional_metric, weight in PILLARS:
        main_score = metric_scores[main_metric]
        pillar_score = main_score.copy()
        if optional_metric is not None:
            optional_score = metric_scores[optional_metric]
            optional_ready = optional_score.notna()
            pillar_score.loc[optional_ready] = (
                0.70 * main_score.loc[optional_ready] + 0.30 * optional_score.loc[optional_ready]
            )
        column = f"pillar_{pillar}"
        pillar_columns.append(column)
        working[column] = pillar_score
        total = total.add(pillar_score.fillna(0.0).mul(weight))

    for idx, row in working.iterrows():
        symbol = str(row["symbol"])
        if any(pd.isna(working.at[idx, column]) for column in pillar_columns):
            reasons[symbol].append("pillar_score_unavailable")
            eligible_mask.at[idx] = False
    working["status"] = np.where(eligible_mask, STATUS_AVAILABLE, STATUS_UNAVAILABLE)
    working["total_score"] = total.where(eligible_mask, np.nan)
    working["unavailable_reasons"] = [
        tuple(dict.fromkeys(reasons[str(symbol)])) for symbol in working["symbol"]
    ]

    available = working.loc[working["status"] == STATUS_AVAILABLE].sort_values(
        ["total_score", "symbol"], ascending=[False, True], kind="mergesort"
    )
    ranked = tuple(available["symbol"].head(top_n).astype(str))
    holdings_normalized = tuple(
        dict.fromkeys(str(item).strip() for item in holdings if str(item).strip())
    )
    symbol_set = set(working["symbol"])
    unknown_holdings = sorted(set(holdings_normalized).difference(symbol_set))
    if unknown_holdings:
        raise ValueError(f"holdings absent from sealed PIT snapshot: {unknown_holdings}")
    appended = tuple(symbol for symbol in holdings_normalized if symbol not in ranked)
    sealed = ranked + appended
    working["selected_top_n"] = working["symbol"].isin(ranked)
    working["sealed_holding"] = working["symbol"].isin(appended)
    return FundamentalCandidateSet(
        scored=working.sort_values("symbol", kind="mergesort").reset_index(drop=True),
        ranked_symbols=ranked,
        sealed_symbols=sealed,
        appended_holdings=appended,
    )


__all__ = [
    "ALL_METRICS",
    "FundamentalCandidateSet",
    "MAIN_METRICS",
    "OPTIONAL_METRICS",
    "PILLARS",
    "STATUS_AVAILABLE",
    "STATUS_UNAVAILABLE",
    "score_fundamental_universe",
]
