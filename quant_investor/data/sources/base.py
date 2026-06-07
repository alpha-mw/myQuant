"""Common source protocol and OHLCV normalization helpers."""

from __future__ import annotations

from abc import ABC
from typing import Any

import pandas as pd


def _parse_any_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return pd.NaT
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_convert(None)


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {
        str(column).strip().lower().replace(" ", "_"): str(column)
        for column in frame.columns
    }


def _first_column(lookup: dict[str, str], *names: str) -> str:
    for name in names:
        key = name.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return ""


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    if not columns:
        return pd.Series(default, index=frame.index, dtype=float)
    series = pd.to_numeric(frame[columns[0]], errors="coerce")
    for column in columns[1:]:
        series = series.combine_first(pd.to_numeric(frame[column], errors="coerce"))
    return series.fillna(default)


def _normalize_ohlcv_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a stable OHLCV frame with one date and one volume column.

    The helper intentionally accepts common Tushare/Yahoo/local CSV spellings
    and strips timezone information from date-like columns.
    """

    columns = ["date", "open", "high", "low", "close", "volume", "amount", "adj_close"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    lookup = _column_lookup(working)
    date_col = _first_column(lookup, "date", "trade_date", "datetime", "timestamp")
    if date_col:
        dates = pd.to_datetime(working[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        dates = pd.to_datetime(working.index, errors="coerce", utc=True)
        dates = pd.Series(dates.tz_convert(None), index=working.index)

    open_col = _first_column(lookup, "open")
    high_col = _first_column(lookup, "high")
    low_col = _first_column(lookup, "low")
    close_col = _first_column(lookup, "close", "adj_close", "adj close")
    volume_cols = [
        column
        for column in (
            _first_column(lookup, "volume"),
            _first_column(lookup, "vol"),
        )
        if column
    ]
    amount_cols = [
        column
        for column in (
            _first_column(lookup, "amount"),
            _first_column(lookup, "amt"),
        )
        if column
    ]
    adj_col = _first_column(lookup, "adj_close", "adj close")

    out = pd.DataFrame(index=working.index)
    out["date"] = dates
    out["open"] = pd.to_numeric(working[open_col], errors="coerce") if open_col else pd.NA
    out["high"] = pd.to_numeric(working[high_col], errors="coerce") if high_col else pd.NA
    out["low"] = pd.to_numeric(working[low_col], errors="coerce") if low_col else pd.NA
    out["close"] = pd.to_numeric(working[close_col], errors="coerce") if close_col else pd.NA
    out["volume"] = _coalesce_numeric(working, volume_cols, default=0.0)
    out["amount"] = _coalesce_numeric(working, amount_cols, default=0.0)
    out["adj_close"] = pd.to_numeric(working[adj_col], errors="coerce") if adj_col else pd.NA
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True).loc[:, columns]


def _normalize_date_text(value: Any) -> str:
    parsed = _parse_any_date(value)
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y%m%d")


def _filter_ohlcv_by_date(
    frame: pd.DataFrame,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame.columns:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    filtered = frame.copy()
    dates = pd.to_datetime(filtered["date"], errors="coerce")
    mask = dates.notna()
    start = _normalize_date_text(start_date)
    end = _normalize_date_text(end_date)
    if start:
        mask &= dates.dt.strftime("%Y%m%d") >= start
    if end:
        mask &= dates.dt.strftime("%Y%m%d") <= end
    return filtered.loc[mask].sort_values("date").reset_index(drop=True)


class DataSourceBase(ABC):
    """Minimal source interface shared by CN/US/fallback providers."""

    source_name = "base"

    def get_ohlcv(self, symbol: str, start_date: str = "", end_date: str = "", freq: str = "1d") -> pd.DataFrame:
        return pd.DataFrame()

    def get_fundamental(self, symbol: str) -> Any:
        raise NotImplementedError

    def get_daily_basic(self, symbol: str, trade_date: str | None = None) -> dict[str, Any]:
        return {}
