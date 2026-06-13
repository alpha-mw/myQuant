"""Pure normalization helpers for the CN fundamental mart."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def run_id(as_of: str | None = None) -> str:
    suffix = str(as_of or "").strip() or now_utc().strftime("%Y%m%d")
    return f"cn_fundamental_{suffix}_{now_utc().strftime('%H%M%S')}"


def date_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def period_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else text


def string_series(values: Any, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.astype("string").fillna("").str.strip()


def period_series(values: Any, index: pd.Index) -> pd.Series:
    text = string_series(values, index)
    text = text.mask(text.str.lower().isin({"nan", "nat", "none"}), "")
    text = text.str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D+", "", regex=True)
    return digits.str[:8].where(digits.str.len() >= 8, text).fillna("")


def date_series(values: Any, index: pd.Index) -> pd.Series:
    text = string_series(values, index)
    text = text.mask(text.str.lower().isin({"nan", "nat", "none"}), "")
    text = text.str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D+", "", regex=True)
    fast_text = digits.str[:8].where(digits.str.len() >= 8)
    parsed_fast = pd.to_datetime(fast_text, format="%Y%m%d", errors="coerce")
    parsed_slow = pd.to_datetime(text.where(fast_text.isna()), errors="coerce")
    parsed = parsed_fast.fillna(parsed_slow)
    output = pd.Series("", index=index, dtype=object)
    valid = parsed.notna()
    output.loc[valid] = parsed.loc[valid].dt.strftime("%Y-%m-%d")
    return output


def num(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def first_number(row: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = num(row.get(name))
        if np.isfinite(value):
            return value
    return float("nan")


def positive_denominator(value: float) -> float:
    return value if np.isfinite(value) and value > 0 else float("nan")


def percent_to_ratio(value: object) -> float:
    number = num(value)
    if not np.isfinite(number):
        return float("nan")
    return number / 100.0 if abs(number) > 2.0 else number


def availability(row: Mapping[str, Any]) -> str:
    for column in ("f_ann_date", "ann_date", "availability_date"):
        text = date_text(row.get(column))
        if text:
            return text
    return ""


__all__ = [
    "availability",
    "date_series",
    "date_text",
    "first_number",
    "now_utc",
    "num",
    "percent_to_ratio",
    "period_series",
    "period_text",
    "positive_denominator",
    "run_id",
    "string_series",
]
