"""Per-held-symbol Market history extracts bound to a frozen canonical snapshot.

Daily materialization pins each held symbol's full bar history as a small
Parquet extract in the day's immutable journal, derived from the frozen
snapshot's canonical ``table/bars``. Consumers read the extract by exact SHA and
re-derive the same rows from the frozen table, so provenance stays bound to the
snapshot without retaining a per-symbol ``serving/`` projection.

Refs are verified by content, not path: a legacy ref to a retained serving file
passes the same row comparison while that file still exists.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.operations.daily_contract import ContractError

_PROJECTION_ONLY_COLUMNS = ("year", "month", "symbol")


def held_market_frame(reader: Any, symbol: str) -> pd.DataFrame:
    """Return ``symbol``'s canonical history from the reader's frozen table."""

    result = reader.read_symbol_frames([symbol])[symbol]
    frame = result.frame
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        raise ContractError("HELD_MARKET_SOURCE_MISSING")
    frame = frame.drop(columns=[c for c in _PROJECTION_ONLY_COLUMNS if c in frame.columns])
    return frame.sort_values("trade_date", kind="stable").reset_index(drop=True)


def legacy_serving_projection_root(reader: Any) -> Path | None:
    """Return the snapshot's transitional serving root when it is still published."""

    from quant_investor.config import config

    if not getattr(config, "CN_MARKET_SERVING_PROJECTION", False):
        return None
    serving_root = reader._require_snapshot().serving_root
    return serving_root if serving_root is not None and serving_root.is_dir() else None


def held_market_extract_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def load_held_market_rows(reader: Any, symbol: str, raw: bytes) -> pd.DataFrame:
    """Parse a pinned extract and require it to equal the frozen table rows."""

    try:
        frame = pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:  # pyarrow raises several unrelated types
        raise ContractError("HELD_MARKET_EXTRACT_UNREADABLE") from exc
    if "ts_code" in frame.columns and set(frame["ts_code"].astype(str)) - {symbol}:
        raise ContractError("HELD_MARKET_EXTRACT_SYMBOL_MISMATCH")
    if reader is not None and not _same_rows(frame, held_market_frame(reader, symbol)):
        raise ContractError("HELD_MARKET_EXTRACT_SNAPSHOT_MISMATCH")
    return frame


def _same_rows(extract: pd.DataFrame, canonical: pd.DataFrame) -> bool:
    extract = extract.drop(columns=[c for c in _PROJECTION_ONLY_COLUMNS if c in extract.columns])
    if set(extract.columns) != set(canonical.columns):
        return False
    ordered = list(canonical.columns)
    left = extract.loc[:, ordered].copy()
    right = canonical.loc[:, ordered].copy()
    for frame in (left, right):
        frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    left = left.sort_values("trade_date", kind="stable").reset_index(drop=True)
    right = right.sort_values("trade_date", kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError:
        return False
    return True
