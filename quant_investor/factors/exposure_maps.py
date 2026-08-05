"""Factor exposure maps bound to the governed fundamental generation.

Gate 2 (coverage concentration) and Gate 6 (industry/size neutralization) need a
sector and a size bucket for every symbol on every rebalance date.  The legacy
loader assembled those from three loose raw tables (``daily_basic``,
``dag_core_raw/stock_basic``, ``dag_core_raw/daily_basic_ext``) plus a
hand-written ``_catalog.json``.  That source stopped being written on
2026-06-11, its ``_catalog.json`` has since disappeared, and even while it was
current it reconstructed about a quarter of its market caps by multiplying a
single as-of ``total_share`` snapshot by close - which is not point-in-time.

The governed fundamental generation already carries a point-in-time ``sector``
and ``total_mv_rmb`` per symbol per trade date, hash-bound to the generation
manifest, so read the exposure there instead.

Size buckets are recomputed here as cross-sectional terciles of the
point-in-time market cap rather than taken from the generation's own
``size_bucket`` column: that column uses fixed absolute thresholds, so a rising
market sweeps the whole cross-section into ``large`` and neutralization stops
removing any size exposure at all.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_investor.factors.pit_fundamentals import normalize_ts_code
from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    load_fundamental_pointer,
)

GOVERNED_EXPOSURE_SOURCE = "governed_fundamental_generation_exposure"
GOVERNED_SIZE_POLICY = "generation_pit_total_mv_cross_section_tercile"
GOVERNED_INDUSTRY_POLICY = "generation_pit_sector_reference"
EXPOSURE_TABLE = "fundamental_daily"
REQUIRED_COLUMNS = (
    "ts_code",
    "trade_date",
    "availability_date",
    "sector",
    "total_mv_rmb",
)
UNKNOWN = "unknown"
_SMALL_QUANTILE = 1.0 / 3.0
_MID_QUANTILE = 2.0 / 3.0

ExposureMaps = tuple[
    dict[str, str],
    dict[str, str],
    pd.DataFrame,
    dict[str, Any],
]


def _blocked(blocker: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "blocker": blocker,
        "source": GOVERNED_EXPOSURE_SOURCE,
        "size_policy": GOVERNED_SIZE_POLICY,
        "industry_policy": GOVERNED_INDUSTRY_POLICY,
        "catalog_validated": False,
        "point_in_time_size": True,
        "reconstructed_size_pair_count": 0,
        "reconstructed_size_pair_ratio": 0.0,
        **extra,
    }


def _bucket_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return UNKNOWN
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else UNKNOWN


def _resolve_exposure_table(mart_root: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Return the generation-bound exposure table path and its pointer."""

    pointer = load_fundamental_pointer(mart_root)
    if pointer is None:
        raise FundamentalGenerationError("fundamental pointer missing")
    base = Path(mart_root).expanduser()
    relative = str(dict(pointer.get("tables", {}) or {}).get(EXPOSURE_TABLE, ""))
    if not relative:
        raise FundamentalGenerationError(
            f"fundamental pointer has no {EXPOSURE_TABLE} table"
        )
    table_path = base / relative
    if not table_path.exists():
        raise FundamentalGenerationError(
            f"fundamental table missing: {EXPOSURE_TABLE}"
        )
    return table_path, pointer


def _read_exposure_frame(
    table_path: Path,
    pointer: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Read only the exposure columns from the exact hash-bound bytes.

    ``load_fundamental_table`` re-fingerprints every column of the whole table,
    which costs minutes on a five-year daily panel.  The manifest ``sha256`` is
    taken over the Parquet file bytes, so verifying those bytes here gives the
    same binding at a fraction of the cost, and the projection keeps memory
    proportional to what the exposure maps actually need.
    """

    declared = dict(
        dict(pointer.get("manifest", {}) or {})
        .get("tables", {})
        .get(EXPOSURE_TABLE, {})
        or {}
    )
    declared_sha256 = str(declared.get("sha256") or "").strip().lower()
    payload = table_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if declared_sha256 and declared_sha256 != actual_sha256:
        raise FundamentalGenerationError(
            f"fundamental table hash mismatch: {EXPOSURE_TABLE}"
        )
    declared_columns = list(declared.get("columns") or [])
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if declared_columns and column not in declared_columns
    ]
    if missing:
        raise FundamentalGenerationError(
            f"{EXPOSURE_TABLE} is missing exposure columns: {sorted(missing)}"
        )
    frame = pd.read_parquet(io.BytesIO(payload), columns=list(REQUIRED_COLUMNS))
    return frame, actual_sha256


def governed_exposure_date_bounds(
    mart_root: str | Path,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the first and last trade date the governed exposure covers."""

    try:
        table_path, _pointer = _resolve_exposure_table(mart_root)
        frame = pd.read_parquet(table_path, columns=["trade_date"])
    except (FundamentalGenerationError, OSError, ValueError):
        return None, None
    trade_dates = pd.to_datetime(
        frame["trade_date"].astype(str), errors="coerce"
    ).dropna()
    if trade_dates.empty:
        return None, None
    return (
        pd.Timestamp(trade_dates.min()).normalize(),
        pd.Timestamp(trade_dates.max()).normalize(),
    )


def clamp_analysis_start_to_exposure(
    resolved_analysis_start: str,
    exposure_start: pd.Timestamp | None,
) -> str:
    """Never evaluate a rebalance date the exposure maps cannot neutralize.

    Gate 2 and Gate 6 both fail closed on dates with no sector/size exposure, so
    an analysis window that reaches back before the governed generation starts
    costs every candidate those gates for a stretch of history nothing can fix.
    """

    if exposure_start is None or pd.isna(exposure_start):
        return resolved_analysis_start
    text = str(resolved_analysis_start or "").strip()
    if not text:
        return pd.Timestamp(exposure_start).strftime("%Y-%m-%d")
    current = pd.to_datetime(text, errors="coerce")
    if pd.isna(current) or current < pd.Timestamp(exposure_start):
        return pd.Timestamp(exposure_start).strftime("%Y-%m-%d")
    return text


def load_governed_exposure_maps(
    *,
    mart_root: str | Path,
    symbols: Sequence[str],
    as_of: pd.Timestamp | None,
    evaluation_dates: Sequence[pd.Timestamp] = (),
    close_by_date: pd.DataFrame | None = None,
) -> ExposureMaps:
    """Build sector and size-bucket exposure from the governed generation."""

    wanted = {normalize_ts_code(symbol) for symbol in symbols}
    dates = sorted(
        {
            pd.Timestamp(date).normalize()
            for date in evaluation_dates
            if not pd.isna(date)
        }
    )
    if not dates and as_of is not None and not pd.isna(as_of):
        dates = [pd.Timestamp(as_of).normalize()]
    date_index = pd.DatetimeIndex(dates)
    empty_by_date = pd.DataFrame(index=date_index, dtype=object)

    try:
        table_path, pointer = _resolve_exposure_table(mart_root)
        frame, table_sha256 = _read_exposure_frame(table_path, pointer)
    except (FundamentalGenerationError, OSError, ValueError) as exc:
        return {}, {}, empty_by_date, _blocked(
            f"governed_exposure_generation_unavailable:{exc}",
            table_path="",
            generation_id="",
            table_sha256="",
        )

    generation_id = str(pointer.get("generation_id", "") or "")
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].map(normalize_ts_code)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"].astype(str), errors="coerce"
    ).dt.normalize()
    frame["availability_date"] = pd.to_datetime(
        frame["availability_date"].astype(str), errors="coerce"
    ).dt.normalize()
    frame = frame[frame["ts_code"].isin(wanted)].dropna(subset=["trade_date"])

    # A row is usable only if the generation says it was knowable on the trade
    # date it is stamped with.  Unknown availability is not point-in-time.
    pit_ok = frame["availability_date"].notna() & (
        frame["availability_date"] <= frame["trade_date"]
    )
    pit_violation_row_count = int((~pit_ok).sum())
    frame = frame[pit_ok]
    if as_of is not None and not pd.isna(as_of):
        frame = frame[frame["trade_date"] <= pd.Timestamp(as_of).normalize()]

    exposure_latest = frame["trade_date"].max() if not frame.empty else pd.NaT
    evaluation_end = date_index.max() if len(date_index) else pd.NaT
    exposure_covers_evaluation_end = bool(
        not pd.isna(exposure_latest)
        and not pd.isna(evaluation_end)
        and exposure_latest >= evaluation_end
    )

    market_caps = (
        frame.pivot_table(
            index="trade_date",
            columns="ts_code",
            values="total_mv_rmb",
            aggfunc="last",
        )
        .sort_index()
        .reindex(index=date_index)
    )
    market_caps = market_caps.where(market_caps > 0.0)

    observable = (
        close_by_date.copy()
        if close_by_date is not None
        else pd.DataFrame(index=date_index)
    )
    if not observable.empty:
        observable.index = pd.to_datetime(observable.index).normalize()
        observable.columns = [
            normalize_ts_code(column) for column in observable.columns
        ]
        observable = observable.reindex(index=date_index)
        market_caps = market_caps.reindex(columns=observable.columns)
    observable_mask = (
        observable.notna()
        if not observable.empty
        else market_caps.notna()
    )

    market_cap_rank = market_caps.rank(axis=1, pct=True)
    size_values = np.where(
        market_cap_rank <= _SMALL_QUANTILE,
        "small",
        np.where(market_cap_rank <= _MID_QUANTILE, "mid", "large"),
    )
    size_bucket_by_date = pd.DataFrame(
        size_values,
        index=market_cap_rank.index,
        columns=market_cap_rank.columns,
    ).where(market_cap_rank.notna())

    latest_sectors = (
        frame.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(subset=["ts_code"], keep="last")
        .set_index("ts_code")["sector"]
        if not frame.empty
        else pd.Series(dtype=object)
    )
    sectors = {
        str(symbol): _bucket_text(value)
        for symbol, value in latest_sectors.items()
    }
    latest_sizes = (
        size_bucket_by_date.iloc[-1]
        if not size_bucket_by_date.empty
        else pd.Series(dtype=object)
    )
    sizes = {
        str(symbol): _bucket_text(bucket)
        for symbol, bucket in latest_sizes.items()
    }

    dynamic_size_symbols = {
        str(symbol)
        for symbol in size_bucket_by_date.columns[
            size_bucket_by_date.notna().any(axis=0)
        ]
    }
    covered = {
        symbol
        for symbol in set(sectors).intersection(dynamic_size_symbols)
        if sectors[symbol] != UNKNOWN
    }
    coverage_ratio = float(len(covered) / max(len(wanted), 1))

    loaded_dates = set(
        pd.DatetimeIndex(
            size_bucket_by_date.index[size_bucket_by_date.notna().any(axis=1)]
        )
    )
    requested_dates = set(date_index)
    evaluation_date_coverage_ratio = float(
        len(requested_dates.intersection(loaded_dates))
        / max(len(requested_dates), 1)
    )

    cross_section_coverage: list[float] = []
    for _date, row in size_bucket_by_date.iterrows():
        valid_sizes = row.dropna()
        if valid_sizes.empty:
            continue
        known_sector_count = sum(
            sectors.get(str(symbol), UNKNOWN) != UNKNOWN
            for symbol in valid_sizes.index
        )
        cross_section_coverage.append(
            float(known_sector_count / len(valid_sizes))
        )
    min_cross_section_coverage_ratio = (
        min(cross_section_coverage) if cross_section_coverage else 0.0
    )

    sized_pairs = size_bucket_by_date.notna() & observable_mask.reindex(
        index=size_bucket_by_date.index,
        columns=size_bucket_by_date.columns,
        fill_value=False,
    )
    observable_pair_count = int(
        observable_mask.reindex(
            index=size_bucket_by_date.index,
            columns=size_bucket_by_date.columns,
            fill_value=False,
        )
        .sum()
        .sum()
    )
    exact_pair_count = int(sized_pairs.sum().sum())
    size_pair_coverage_ratio = float(
        exact_pair_count / max(observable_pair_count, 1)
    )

    sector_count = len({value for value in sectors.values() if value != UNKNOWN})
    size_bucket_count = len(
        {value for value in sizes.values() if value != UNKNOWN}
    )
    ready = (
        coverage_ratio >= 0.95
        and evaluation_date_coverage_ratio == 1.0
        and min_cross_section_coverage_ratio >= 0.95
        and size_pair_coverage_ratio >= 0.95
        and exposure_covers_evaluation_end
        and pit_violation_row_count == 0
        and sector_count >= 2
        and size_bucket_count >= 3
    )
    return sectors, sizes, size_bucket_by_date, {
        "status": "ready" if ready else "blocked",
        "blocker": "" if ready else "governed_exposure_incomplete",
        "source": GOVERNED_EXPOSURE_SOURCE,
        "generation_id": generation_id,
        "table_path": str(table_path),
        "table_sha256": table_sha256,
        "catalog_validated": True,
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d")
        if as_of is not None and not pd.isna(as_of)
        else "",
        "evaluation_start": date_index.min().strftime("%Y-%m-%d")
        if len(date_index)
        else "",
        "evaluation_end": evaluation_end.strftime("%Y-%m-%d")
        if not pd.isna(evaluation_end)
        else "",
        "exposure_latest_date": exposure_latest.strftime("%Y-%m-%d")
        if not pd.isna(exposure_latest)
        else "",
        "exposure_covers_evaluation_end": exposure_covers_evaluation_end,
        # The governed generation is itself the share reference, so the legacy
        # staleness key stays meaningful for readers of the mining evidence.
        "share_reference_latest_date": exposure_latest.strftime("%Y-%m-%d")
        if not pd.isna(exposure_latest)
        else "",
        "share_reference_covers_evaluation_end": exposure_covers_evaluation_end,
        "daily_basic_latest_date": exposure_latest.strftime("%Y-%m-%d")
        if not pd.isna(exposure_latest)
        else "",
        "pit_violation_row_count": pit_violation_row_count,
        "requested_symbol_count": len(wanted),
        "covered_symbol_count": len(covered),
        "coverage_ratio": coverage_ratio,
        "requested_evaluation_date_count": len(requested_dates),
        "covered_evaluation_date_count": len(
            requested_dates.intersection(loaded_dates)
        ),
        "evaluation_date_coverage_ratio": evaluation_date_coverage_ratio,
        "min_cross_section_coverage_ratio": min_cross_section_coverage_ratio,
        "observable_size_pair_count": observable_pair_count,
        "exact_pit_size_pair_count": exact_pair_count,
        "reconstructed_size_pair_count": 0,
        "combined_size_pair_coverage_ratio": size_pair_coverage_ratio,
        "pit_size_pair_coverage_ratio": size_pair_coverage_ratio,
        "reconstructed_size_pair_ratio": 0.0,
        "sector_count": sector_count,
        "size_bucket_count": size_bucket_count,
        "unknown_sector_count": len(wanted)
        - sum(value != UNKNOWN for value in sectors.values()),
        "unknown_size_bucket_count": len(wanted)
        - sum(value != UNKNOWN for value in sizes.values()),
        "point_in_time_size": True,
        "size_policy": GOVERNED_SIZE_POLICY,
        "industry_policy": GOVERNED_INDUSTRY_POLICY,
    }
