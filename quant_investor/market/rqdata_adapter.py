"""Pure, offline normalization for RQData daily-bar candidate frames.

This module deliberately does not import ``rqdatac`` and performs no provider
or filesystem I/O.  It converts one complete RQData response into the existing
myQuant bar schema so shadow comparison can be built without creating a second
storage path or granting promotion authority.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

RQDATA_PROVIDER = "rqdata"
RQDATA_DAILY_DATASET = "get_price"
RQDATA_DAILY_ADJUSTMENT = "none"

_RQDATA_SYMBOL = re.compile(r"^(?P<code>[0-9]{6})\.(?P<exchange>XSHG|XSHE|XBSE)$")
_EXCHANGE_SUFFIX = {"XSHG": "SH", "XSHE": "SZ", "XBSE": "BJ"}
_PRICE_COLUMNS = ("open", "high", "low", "close")
_REQUIRED_COLUMNS = (
    "order_book_id",
    "date",
    *_PRICE_COLUMNS,
    "volume",
    "total_turnover",
)
_OPTIONAL_COLUMNS = ("prev_close", "limit_up", "limit_down", "num_trades")


class RQDataNormalizationError(ValueError):
    """Raised when a provider frame cannot enter the canonical candidate lane."""


def normalize_rqdata_symbol(value: Any) -> str:
    """Map an exact RQData mainland equity identity to myQuant's symbol form."""

    text = str(value or "").strip().upper()
    match = _RQDATA_SYMBOL.fullmatch(text)
    if match is None:
        raise RQDataNormalizationError(f"unsupported RQData symbol: {text or '<empty>'}")
    return f"{match.group('code')}.{_EXCHANGE_SUFFIX[match.group('exchange')]}"


def _normalize_trade_date(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise RQDataNormalizationError("trade date is missing")
        parsed = value.date()
    elif isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        text = str(value or "").strip()
        try:
            parsed = datetime.strptime(text, "%Y%m%d" if len(text) == 8 else "%Y-%m-%d").date()
        except ValueError as exc:
            raise RQDataNormalizationError(f"invalid RQData trade date: {text}") from exc
    return parsed.strftime("%Y%m%d")


def _materialize_index(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    named_levels = {str(name) for name in work.index.names if name is not None}
    if named_levels.intersection({"order_book_id", "date", "datetime"}):
        work = work.reset_index()
    if "date" not in work.columns and "datetime" in work.columns:
        work = work.rename(columns={"datetime": "date"})
    return work


def _numeric_column(work: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(work[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise RQDataNormalizationError(f"{column} contains non-numeric values") from exc
    for value in values.dropna().tolist():
        if not math.isfinite(float(value)):
            raise RQDataNormalizationError(f"{column} contains non-finite values")
        if float(value) < 0:
            raise RQDataNormalizationError(f"{column} contains negative values")
    return values


def normalize_rqdata_daily_bars(
    frame: pd.DataFrame,
    *,
    adjustment_type: str,
) -> pd.DataFrame:
    """Normalize one complete unadjusted RQData daily response.

    No row-level fallback, unit scaling, forward filling, or deduplication is
    performed.  A malformed or mixed response fails as a whole.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if adjustment_type != RQDATA_DAILY_ADJUSTMENT:
        raise RQDataNormalizationError("RQData daily bars must use adjust_type='none'")
    if frame.empty:
        raise RQDataNormalizationError("RQData daily frame is empty")
    work = _materialize_index(frame)
    missing = [column for column in _REQUIRED_COLUMNS if column not in work.columns]
    if missing:
        raise RQDataNormalizationError(f"RQData daily frame missing columns: {missing}")

    work["provider_symbol"] = work["order_book_id"].astype(str).str.strip().str.upper()
    work["ts_code"] = work["provider_symbol"].map(normalize_rqdata_symbol)
    work["trade_date"] = work["date"].map(_normalize_trade_date)
    for column in (*_PRICE_COLUMNS, "volume", "total_turnover", *_OPTIONAL_COLUMNS):
        if column in work.columns:
            work[column] = _numeric_column(work, column)

    missing_price_count = work[list(_PRICE_COLUMNS)].isna().sum(axis=1)
    if ((missing_price_count != 0) & (missing_price_count != len(_PRICE_COLUMNS))).any():
        raise RQDataNormalizationError("RQData daily frame contains partial OHLC rows")
    trading = missing_price_count.eq(0)
    if (work.loc[trading, "low"] > work.loc[trading, list(_PRICE_COLUMNS)].min(axis=1)).any():
        raise RQDataNormalizationError("RQData daily frame has low above OHLC minimum")
    if (work.loc[trading, "high"] < work.loc[trading, list(_PRICE_COLUMNS)].max(axis=1)).any():
        raise RQDataNormalizationError("RQData daily frame has high below OHLC maximum")

    if work.duplicated(subset=["ts_code", "trade_date"]).any():
        raise RQDataNormalizationError("RQData daily frame contains duplicate symbol/date rows")

    # The stable myQuant/Factor contract names this field ``vol``.  RQData
    # already reports share volume, so the value is renamed without scaling.
    work["vol"] = work["volume"]
    work["amount"] = work["total_turnover"]
    work["provider"] = RQDATA_PROVIDER
    work["provider_dataset"] = RQDATA_DAILY_DATASET
    work["adjustment_type"] = RQDATA_DAILY_ADJUSTMENT
    work["bar_status"] = trading.map({True: "TRADING", False: "SUSPENDED_OR_NO_TRADE"})

    ordered = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        *[column for column in _OPTIONAL_COLUMNS if column in work.columns],
        "bar_status",
        "provider",
        "provider_dataset",
        "provider_symbol",
        "adjustment_type",
    ]
    return work[ordered].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
