"""PIT forward-return calibration with three-horizon eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

HORIZONS = (120, 252, 378)
MIN_OBSERVATIONS = 100
MIN_CROSS_SECTIONS = 12
MIN_SYMBOLS = 20
MAX_LOOKBACK_OPEN_DAYS = 2520
REQUIRED_COLUMNS = frozenset(
    {
        "symbol",
        "industry",
        "score_decile",
        "horizon",
        "cross_section_date",
        "availability",
        "age_open_days",
        "realized_open_days",
        "is_pit_month_end",
        "is_mature",
        "stock_start_trade_date",
        "stock_end_trade_date",
        "benchmark_start_trade_date",
        "benchmark_end_trade_date",
        "stock_total_return",
        "benchmark_total_return",
        "benchmark_symbol",
        "stock_return_includes_dividends",
        "benchmark_return_is_pre_tax_total_return",
        "delisted",
        "official_terminal_cash_settlement",
    }
)


@dataclass(frozen=True)
class FundamentalEligibility:
    status: str
    eligible: bool
    base_q25_by_horizon: Mapping[int, float]
    blockers: tuple[str, ...]

    @property
    def optimizer_q25_252(self) -> float | None:
        return self.base_q25_by_horizon.get(252) if self.eligible else None


def _cutoff(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return result.tz_convert("UTC")


def _timestamps(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{label} contains an invalid timestamp")
    return result


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def calibrate_forward_returns(
    observations: pd.DataFrame,
    *,
    cutoff: datetime | str | pd.Timestamp,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(observations.columns))
    if missing:
        raise ValueError(f"observations missing required columns: {missing}")
    cutoff_ts = _cutoff(cutoff)
    frame = observations.copy(deep=True)
    timestamp_columns = (
        "cross_section_date",
        "availability",
        "stock_start_trade_date",
        "stock_end_trade_date",
        "benchmark_start_trade_date",
        "benchmark_end_trade_date",
    )
    for column in timestamp_columns:
        frame[column] = _timestamps(frame[column], column)
    frame = frame.loc[frame["cross_section_date"] <= cutoff_ts].copy()
    result_columns = (
        "industry",
        "score_decile",
        "horizon",
        "status",
        "q25",
        "q50",
        "q75",
        "observation_count",
        "cross_section_count",
        "symbol_count",
        "blockers",
    )
    if frame.empty:
        return pd.DataFrame(columns=result_columns)
    numeric_columns = (
        "score_decile",
        "horizon",
        "age_open_days",
        "realized_open_days",
        "stock_total_return",
        "benchmark_total_return",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["excess_return"] = frame["stock_total_return"] - frame["benchmark_total_return"]
    frame["_invalid_symbol"] = [
        not isinstance(value, str) or not value or value.strip() != value
        for value in frame["symbol"]
    ]
    frame["_duplicate"] = frame.duplicated(["symbol", "cross_section_date", "horizon"], keep=False)
    rows: list[dict[str, object]] = []
    group_columns = ["industry", "score_decile", "horizon"]
    for (industry, raw_decile, raw_horizon), cell in frame.groupby(
        group_columns, dropna=False, sort=True
    ):
        blockers: list[str] = []
        decile_number = float(raw_decile) if pd.notna(raw_decile) else np.nan
        horizon_number = float(raw_horizon) if pd.notna(raw_horizon) else np.nan
        decile = (
            int(decile_number) if np.isfinite(decile_number) and decile_number.is_integer() else -1
        )
        horizon = (
            int(horizon_number)
            if np.isfinite(horizon_number) and horizon_number.is_integer()
            else -1
        )
        if not isinstance(industry, str) or not industry or industry.strip() != industry:
            blockers.append("industry_unknown")
        if decile not in range(1, 11):
            blockers.append("invalid_score_decile")
        if horizon not in HORIZONS:
            blockers.append("invalid_horizon")
        if bool(cell["_invalid_symbol"].any()):
            blockers.append("invalid_symbol")
        if bool(cell["_duplicate"].any()):
            blockers.append("duplicate_symbol_cross_section_horizon")
        if not cell["benchmark_symbol"].astype(str).eq("H00300.CSI").all():
            blockers.append("wrong_benchmark")
        if any(
            _strict_bool(value) is not True for value in cell["stock_return_includes_dividends"]
        ):
            blockers.append("stock_return_not_total_return")
        if any(
            _strict_bool(value) is not True
            for value in cell["benchmark_return_is_pre_tax_total_return"]
        ):
            blockers.append("benchmark_not_pre_tax_total_return")
        if (cell["availability"] > cutoff_ts).any():
            blockers.append("evidence_after_cutoff")
        ages = cell["age_open_days"]
        if (
            ages.isna().any()
            or (ages % 1 != 0).any()
            or ((ages < 0) | (ages > MAX_LOOKBACK_OPEN_DAYS)).any()
        ):
            blockers.append("outside_2520_open_day_window")
        if any(_strict_bool(value) is not True for value in cell["is_pit_month_end"]):
            blockers.append("not_pit_month_end")
        if any(_strict_bool(value) is not True for value in cell["is_mature"]):
            blockers.append("immature_forward_return")
        exact_dates = cell["stock_start_trade_date"].eq(cell["benchmark_start_trade_date"]) & cell[
            "stock_end_trade_date"
        ].eq(cell["benchmark_end_trade_date"])
        if not bool(exact_dates.all()):
            blockers.append("stock_benchmark_trade_dates_mismatch")
        starts = cell["stock_start_trade_date"].eq(cell["cross_section_date"]) & cell[
            "benchmark_start_trade_date"
        ].eq(cell["cross_section_date"])
        if not bool(starts.all()):
            blockers.append("forward_start_not_cross_section")
        if (cell["stock_end_trade_date"] > cutoff_ts).any() or (
            cell["benchmark_end_trade_date"] > cutoff_ts
        ).any():
            blockers.append("forward_end_after_cutoff")
        if (cell["availability"] < cell["stock_end_trade_date"]).any() or (
            cell["availability"] < cell["benchmark_end_trade_date"]
        ).any():
            blockers.append("forward_return_available_before_end")
        if (cell["stock_end_trade_date"] <= cell["stock_start_trade_date"]).any() or (
            cell["benchmark_end_trade_date"] <= cell["benchmark_start_trade_date"]
        ).any():
            blockers.append("forward_date_order_invalid")
        realized = cell["realized_open_days"]
        if (
            realized.isna().any()
            or not np.isfinite(realized).all()
            or not realized.eq(realized.round()).all()
            or not bool(realized.eq(horizon).all())
        ):
            blockers.append("forward_horizon_mismatch")
        if not bool(
            np.isfinite(cell["stock_total_return"]).all()
            and np.isfinite(cell["benchmark_total_return"]).all()
        ):
            blockers.append("nonfinite_total_return")
        for _, item in cell.iterrows():
            delisted = _strict_bool(item["delisted"])
            terminal = _strict_bool(item["official_terminal_cash_settlement"])
            if delisted is None or terminal is None:
                blockers.append("invalid_delisting_evidence")
                break
            if delisted and not terminal:
                blockers.append("delisting_without_official_terminal_cash")
                break
        count = len(cell)
        cross_sections = int(cell["cross_section_date"].nunique())
        symbols = int(cell["symbol"].astype(str).nunique())
        if count < MIN_OBSERVATIONS:
            blockers.append("observations_below_100")
        if cross_sections < MIN_CROSS_SECTIONS:
            blockers.append("cross_sections_below_12")
        if symbols < MIN_SYMBOLS:
            blockers.append("symbols_below_20")
        values = cell["excess_return"].to_numpy(dtype=float)
        available = not blockers
        rows.append(
            {
                "industry": str(industry),
                "score_decile": decile,
                "horizon": horizon,
                "status": "AVAILABLE" if available else "UNAVAILABLE",
                "q25": float(np.quantile(values, 0.25, method="linear")) if available else np.nan,
                "q50": float(np.quantile(values, 0.50, method="linear")) if available else np.nan,
                "q75": float(np.quantile(values, 0.75, method="linear")) if available else np.nan,
                "observation_count": int(count),
                "cross_section_count": cross_sections,
                "symbol_count": symbols,
                "blockers": tuple(dict.fromkeys(blockers)),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns, kind="mergesort").reset_index(drop=True)


def assess_fundamental_eligibility(
    calibration: pd.DataFrame,
    *,
    industry: str,
    score_decile: int,
    deep_research_complete: bool,
    severe_red_flags: bool,
) -> FundamentalEligibility:
    blockers: list[str] = []
    values: dict[int, float] = {}
    if not isinstance(industry, str) or not industry or industry.strip() != industry:
        blockers.append("industry_invalid")
    if (
        isinstance(score_decile, (bool, np.bool_))
        or not isinstance(score_decile, (int, np.integer))
        or int(score_decile) not in range(1, 11)
    ):
        blockers.append("score_decile_invalid")
    if _strict_bool(deep_research_complete) is None:
        blockers.append("deep_research_complete_not_strict_bool")
    if _strict_bool(severe_red_flags) is None:
        blockers.append("severe_red_flags_not_strict_bool")
    required = {"industry", "score_decile", "horizon", "status", "q25"}
    missing = sorted(required.difference(calibration.columns))
    if missing:
        return FundamentalEligibility(
            "F_INELIGIBLE",
            False,
            {},
            tuple(blockers + [f"calibration_columns_missing:{','.join(missing)}"]),
        )
    for horizon in HORIZONS:
        cell = calibration.loc[
            (calibration["industry"].astype(str) == str(industry))
            & (calibration["score_decile"] == score_decile)
            & (calibration["horizon"] == horizon)
        ]
        if len(cell) != 1 or str(cell.iloc[0]["status"]) != "AVAILABLE":
            blockers.append(f"calibration_unavailable:{horizon}")
            continue
        raw = cell.iloc[0]["q25"]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            blockers.append(f"q25_nonfinite:{horizon}")
            continue
        value = float(raw)
        if not np.isfinite(value):
            blockers.append(f"q25_nonfinite:{horizon}")
            continue
        values[horizon] = value
        if value <= 0:
            blockers.append(f"q25_not_positive:{horizon}")
    if _strict_bool(deep_research_complete) is not True:
        blockers.append("deep_research_incomplete")
    if _strict_bool(severe_red_flags) is not False:
        blockers.append("severe_red_flag")
    eligible = not blockers and set(values) == set(HORIZONS)
    return FundamentalEligibility(
        "F_ELIGIBLE" if eligible else "F_INELIGIBLE",
        eligible,
        values,
        tuple(blockers),
    )
