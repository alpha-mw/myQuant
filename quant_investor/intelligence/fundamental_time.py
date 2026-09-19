"""Decision-cutoff selection for native Fundamental frames, without age vetoes."""

from datetime import date, datetime, timedelta, timezone
from numbers import Integral
from zoneinfo import ZoneInfo
import re

import pandas as pd

from ._common import IntelligenceError, company_code, timestamp

SHANGHAI = ZoneInfo("Asia/Shanghai")
TEMPORAL_CODES = frozenset(
    {
        "FUNDAMENTAL_PIT_DATE_INVALID",
        "FUNDAMENTAL_PIT_REVISION_AMBIGUOUS",
        "FUNDAMENTAL_ROW_PROVENANCE_INVALID",
        "FUNDAMENTAL_SOURCE_AVAILABILITY_INVALID",
        "FUNDAMENTAL_SOURCE_NOT_AVAILABLE_AT_DECISION",
        "FUNDAMENTAL_NATIVE_AVAILABILITY_MISSING",
        "FUNDAMENTAL_NATIVE_AVAILABILITY_BACKDATED",
    }
)


class FundamentalTemporalError(IntelligenceError):
    def __init__(self, code):
        if code not in TEMPORAL_CODES:
            raise IntelligenceError("unknown Fundamental temporal diagnosis")
        super().__init__(code)
        self.code = code


def session_date(value) -> date:
    try:
        if value is None or value is pd.NaT or isinstance(value, bool):
            raise ValueError("missing or boolean session")
        if isinstance(value, datetime):
            local = (
                value.replace(tzinfo=SHANGHAI)
                if value.tzinfo is None
                else value.astimezone(SHANGHAI)
            )
            return local.date()
        if isinstance(value, date):
            return value
        if isinstance(value, Integral):
            value = str(int(value))
        if type(value) is str and re.fullmatch(r"[0-9]{8}", value):
            return datetime.strptime(value, "%Y%m%d").date()
        if type(value) is str and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError, OverflowError) as exc:
        raise FundamentalTemporalError("FUNDAMENTAL_PIT_DATE_INVALID") from exc
    raise FundamentalTemporalError("FUNDAMENTAL_PIT_DATE_INVALID")


def availability_instant(value) -> datetime:
    try:
        if type(value) is not str or value != value.strip():
            raise ValueError("availability must be explicit UTC text")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("availability must be UTC")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise FundamentalTemporalError("FUNDAMENTAL_SOURCE_AVAILABILITY_INVALID") from exc


def source_available_at(value, *, as_of: str) -> str:
    """Conservative whole-second envelope time; never floor native availability."""
    cutoff = datetime.fromisoformat(
        timestamp(as_of, label="Fundamental decision cutoff").replace("Z", "+00:00")
    )
    known = availability_instant(value)
    if known > cutoff:
        raise FundamentalTemporalError("FUNDAMENTAL_SOURCE_NOT_AVAILABLE_AT_DECISION")
    return availability_ceiling(value)


def availability_ceiling(value) -> str:
    """Conservative whole-second bound; preserve original native time separately."""
    known = availability_instant(value)
    if known.microsecond:
        known = (known + timedelta(seconds=1)).replace(microsecond=0)
    return known.strftime("%Y-%m-%dT%H:%M:%SZ")


def bind_native_availability(declared, verified_pointer, *, as_of=None):
    metadata = verified_pointer["manifest"]["metadata"]
    derivations = [
        metadata.get("provider_manifest", {}).get("derivation", {}),
        verified_pointer.get("metadata", {}).get("derivation", {}),
    ]
    stamps = [
        value["derivation_timestamp"] for value in derivations if "derivation_timestamp" in value
    ]
    if not stamps:
        raise FundamentalTemporalError("FUNDAMENTAL_NATIVE_AVAILABILITY_MISSING")
    native = max(stamps, key=availability_instant)
    if availability_instant(declared) < availability_instant(native):
        raise FundamentalTemporalError("FUNDAMENTAL_NATIVE_AVAILABILITY_BACKDATED")
    if as_of is not None:
        source_available_at(declared, as_of=as_of)
    return native


def select_fundamental_snapshots(frame: pd.DataFrame, *, as_of: str) -> pd.DataFrame:
    cutoff = datetime.fromisoformat(
        timestamp(as_of, label="Fundamental decision cutoff").replace("Z", "+00:00")
    )
    day = cutoff.astimezone(SHANGHAI).date()
    selected = frame.copy()
    selected["ts_code"] = selected["ts_code"].map(company_code)
    dates = selected["trade_date"].map(session_date)
    selected["trade_date"] = pd.to_datetime(dates)
    selected = selected.loc[dates <= day].copy()
    try:
        selected = selected.drop_duplicates()
    except (TypeError, ValueError) as exc:
        raise FundamentalTemporalError("FUNDAMENTAL_ROW_PROVENANCE_INVALID") from exc
    if selected.duplicated(["ts_code", "trade_date"], keep=False).any():
        raise FundamentalTemporalError("FUNDAMENTAL_PIT_REVISION_AMBIGUOUS")
    return (
        selected.sort_values(["ts_code", "trade_date"], kind="mergesort")
        .groupby("ts_code", as_index=False)
        .tail(1)
    )
