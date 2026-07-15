"""Standalone CN point-in-time fundamental mart builder.

The mart is intentionally separate from daily market maintenance. It can be
driven from offline fixtures in tests or from an explicit live provider in
operations, and it treats missing announcement dates as a hard quarantine.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_investor.factors.pit_fundamentals import normalize_ts_code
from quant_investor.market.fundamental_generation import (
    FUNDAMENTAL_TABLES,
    _issue_primary_generation_attestation,
    load_fundamental_pointer,
    publish_fundamental_generation,
    resolve_fundamental_table_path,
)
from quant_investor.market.market_data_reader import MarketDataReader

DEFAULT_FUNDAMENTAL_ROOT = Path("data/parquet/cn")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/fundamental")
DEFAULT_READINESS_ROOT = Path("reports/fundamental_readiness")
DEFAULT_MARKET_DATA_ROOT = Path("data")
DEFAULT_METADATA_ROOT = Path("data/metadata")
DEFAULT_UNIVERSES = ("hs300", "zz500", "zz1000")
FULL_A_UNIVERSE_KEYS = {"full_a", "full_market", "all_a", "all", "full"}
FULL_A_PHYSICAL_DIRECTORIES = ("hs300", "zz500", "zz1000", "other")

SOURCE_TABLES = ("fina_indicator", "income", "balancesheet", "cashflow", "daily_basic", "forecast")
FINANCIAL_SOURCE_TABLES = ("fina_indicator", "income", "balancesheet", "cashflow")
DERIVED_PERIOD_FIELDS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "free_cashflow",
)
DERIVED_DAILY_FIELDS = DERIVED_PERIOD_FIELDS + ("fcf_to_price", "forecast_revision")
FORECAST_DAILY_COLUMNS = (
    "ts_code",
    "forecast_end_date",
    "availability_date",
    "forecast_ann_date",
    "forecast_revision",
    "forecast_type",
    "forecast_summary",
    "forecast_change_reason",
    "forecast_source",
    "forecast_fetched_at",
    "forecast_ingest_run_id",
)


@dataclass(frozen=True)
class FundamentalMartArtifacts:
    run_id: str
    data_root: Path
    raw_snapshot_root: Path
    reports_root: Path
    fundamental_period_path: Path
    fundamental_daily_path: Path
    quarantine_path: Path
    readiness_json_path: Path
    readiness_md_path: Path
    readiness_csv_path: Path


class FundamentalReadinessError(RuntimeError):
    """Raised when an operational mart refresh fails its publication gate."""

    def __init__(self, readiness: Mapping[str, Any]):
        self.readiness = dict(readiness)
        blockers = ",".join(str(item) for item in readiness.get("blockers", []))
        super().__init__(f"fundamental readiness gate failed: {blockers or 'unknown'}")


_LIVE_TUSHARE_CAPABILITY = object()


@dataclass(frozen=True)
class _LiveTushareAttestation:
    capability: object
    source: str
    provider_manifest_sha256: str
    raw_table_fingerprints: tuple[tuple[str, str], ...]


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("fundamental raw tables must contain pandas DataFrames")
    digest = hashlib.sha256()
    schema = [
        {"position": index, "name": repr(column), "dtype": str(dtype)}
        for index, (column, dtype) in enumerate(zip(frame.columns, frame.dtypes))
    ]
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=True)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _raw_table_fingerprints(
    raw_tables: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            table,
            _frame_sha256(raw_tables.get(table, pd.DataFrame())),
        )
        for table in SOURCE_TABLES
    )


def _issue_live_tushare_attestation(
    source: str,
    provider_manifest: Mapping[str, Any],
    raw_tables: Mapping[str, pd.DataFrame],
) -> _LiveTushareAttestation:
    if not _has_verified_live_tushare_provenance(
        source,
        provider_manifest,
        raw_tables,
    ):
        raise ValueError("live Tushare provenance attestation failed")
    return _LiveTushareAttestation(
        capability=_LIVE_TUSHARE_CAPABILITY,
        source=str(source),
        provider_manifest_sha256=_canonical_mapping_sha256(provider_manifest),
        raw_table_fingerprints=_raw_table_fingerprints(raw_tables),
    )


def _live_tushare_attestation_matches(
    attestation: _LiveTushareAttestation | None,
    *,
    source: str,
    provider_manifest: Mapping[str, Any],
    raw_tables: Mapping[str, pd.DataFrame],
) -> bool:
    if not isinstance(attestation, _LiveTushareAttestation):
        return False
    try:
        return bool(
            attestation.capability is _LIVE_TUSHARE_CAPABILITY
            and attestation.source == str(source)
            and attestation.provider_manifest_sha256
            == _canonical_mapping_sha256(provider_manifest)
            and attestation.raw_table_fingerprints
            == _raw_table_fingerprints(raw_tables)
        )
    except (TypeError, ValueError):
        return False


def _provider_source_priority(
    source: str,
    provider_manifest: Mapping[str, Any] | None,
    raw_tables: Mapping[str, pd.DataFrame],
    live_tushare_attestation: _LiveTushareAttestation | None = None,
) -> str:
    manifest = dict(provider_manifest or {})
    explicit = str(manifest.get("source_priority") or "").strip()
    source_is_tushare = "tushare" in str(source or "").lower()
    if explicit == "tushare_primary":
        if _live_tushare_attestation_matches(
            live_tushare_attestation,
            source=source,
            provider_manifest=manifest,
            raw_tables=raw_tables,
        ):
            return explicit
        raise ValueError(
            "tushare_primary requires an internal live Tushare attestation"
        )
    if explicit:
        return explicit
    if source_is_tushare:
        raise ValueError(
            "Tushare source requires verified provider provenance"
        )
    return "manual_offline_snapshot"


def _has_verified_live_tushare_provenance(
    source: str,
    provider_manifest: Mapping[str, Any],
    raw_tables: Mapping[str, pd.DataFrame],
) -> bool:
    if str(source or "").strip() not in {
        "live_tushare",
        "live_tushare_partial",
    }:
        return False
    if (
        str(provider_manifest.get("provider") or "").strip().lower()
        != "tushare"
        or str(
            provider_manifest.get("source_provenance") or ""
        ).strip()
        != "live_tushare_explicit"
    ):
        return False
    try:
        expected_counts = {
            table: int(len(raw_tables.get(table, pd.DataFrame())))
            for table in SOURCE_TABLES
        }
        declared_counts = {
            str(key): int(value)
            for key, value in dict(
                provider_manifest.get("raw_row_counts", {}) or {}
            ).items()
        }
    except (TypeError, ValueError):
        return False
    if declared_counts != expected_counts:
        return False
    try:
        attempted = int(provider_manifest.get("requests_attempted", -1))
        accounted = sum(
            int(provider_manifest.get(field, -1))
            for field in (
                "requests_succeeded_with_rows",
                "requests_empty",
                "requests_failed",
            )
        )
    except (TypeError, ValueError):
        return False
    outcomes = provider_manifest.get("symbol_table_outcomes")
    try:
        declared_tables = set(provider_manifest.get("tables", []))
    except TypeError:
        return False
    return bool(
        attempted > 0
        and attempted == accounted
        and isinstance(outcomes, list)
        and len(outcomes) == attempted
        and declared_tables == set(SOURCE_TABLES)
    )


def _resolve_data_base(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    if root.name in {"fundamental_daily", "fundamental_period", "fundamental_quarantine"}:
        return root.parent
    return root


def _fundamental_table_path(data_root: str | Path, table_name: str) -> Path:
    return resolve_fundamental_table_path(
        _resolve_data_base(data_root),
        table_name,
    )


def _read_existing_fundamental_table(
    data_root: str | Path,
    table_name: str,
) -> pd.DataFrame:
    path = resolve_fundamental_table_path(
        _resolve_data_base(data_root),
        table_name,
    )
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(
            f"existing canonical fundamental table is unreadable: {path}: {exc}"
        ) from exc


def _merge_key_values(series: pd.Series, field_name: str) -> pd.Series:
    if field_name == "ts_code":
        return series.map(normalize_ts_code)
    if field_name == "end_date":
        return series.map(_period_text)
    if field_name.endswith("date"):
        return series.map(_date_text)
    return series.astype(str).str.strip()


def _align_incoming_to_existing_schema(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    aligned = incoming.copy()
    for column in existing.columns.intersection(aligned.columns):
        existing_series = existing[column]
        if isinstance(existing_series.dtype, pd.StringDtype):
            aligned[column] = aligned[column].astype("string")
        elif pd.api.types.is_integer_dtype(existing_series.dtype):
            aligned[column] = pd.to_numeric(
                aligned[column],
                errors="raise",
            ).astype(existing_series.dtype)
        elif pd.api.types.is_float_dtype(existing_series.dtype):
            aligned[column] = pd.to_numeric(
                aligned[column],
                errors="coerce",
            ).astype(existing_series.dtype)
        elif pd.api.types.is_datetime64_any_dtype(existing_series.dtype):
            aligned[column] = pd.to_datetime(
                aligned[column],
                errors="coerce",
                utc=getattr(existing_series.dt, "tz", None) is not None,
            )
        elif existing_series.dtype == object:
            sample = existing_series.dropna()
            if not sample.empty and isinstance(sample.iloc[0], date):
                aligned[column] = pd.to_datetime(
                    aligned[column],
                    errors="coerce",
                ).dt.date
    return aligned


def _merge_fundamental_table(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    key_fields: Sequence[str],
    quality_fields: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if existing.empty and incoming.empty:
        return incoming.copy(), {
            "existing_rows": 0,
            "incoming_rows": 0,
            "merged_rows": 0,
            "retained_existing_rows": 0,
            "accepted_incoming_rows": 0,
        }
    if incoming.empty:
        return existing.copy(), {
            "existing_rows": int(len(existing)),
            "incoming_rows": 0,
            "merged_rows": int(len(existing)),
            "retained_existing_rows": int(len(existing)),
            "accepted_incoming_rows": 0,
            "merge_path": "retain_existing_no_incoming",
        }
    if existing.empty:
        return incoming.copy(), {
            "existing_rows": 0,
            "incoming_rows": int(len(incoming)),
            "merged_rows": int(len(incoming)),
            "retained_existing_rows": 0,
            "accepted_incoming_rows": int(len(incoming)),
            "merge_path": "accept_incoming_no_existing",
        }
    missing_keys = [
        field
        for field in key_fields
        if field not in existing.columns or field not in incoming.columns
    ]
    if missing_keys and not existing.empty and not incoming.empty:
        raise ValueError(
            "fundamental merge missing key columns: "
            + ",".join(missing_keys)
        )
    if (
        not existing.empty
        and not incoming.empty
        and "ts_code" in key_fields
        and not existing.duplicated(subset=list(key_fields)).any()
        and not incoming.duplicated(subset=list(key_fields)).any()
    ):
        existing_symbols = {
            normalize_ts_code(value)
            for value in existing["ts_code"].dropna().drop_duplicates()
        }
        incoming_symbols = {
            normalize_ts_code(value)
            for value in incoming["ts_code"].dropna().drop_duplicates()
        }
        if existing_symbols.isdisjoint(incoming_symbols):
            incoming_aligned = _align_incoming_to_existing_schema(
                existing,
                incoming,
            )
            columns = list(
                dict.fromkeys([*existing.columns, *incoming_aligned.columns])
            )
            output = pd.concat(
                [
                    existing.reindex(columns=columns),
                    incoming_aligned.reindex(columns=columns),
                ],
                ignore_index=True,
                sort=False,
            )
            return output, {
                "existing_rows": int(len(existing)),
                "incoming_rows": int(len(incoming_aligned)),
                "merged_rows": int(len(output)),
                "retained_existing_rows": int(len(existing)),
                "accepted_incoming_rows": int(len(incoming_aligned)),
                "merge_path": "disjoint_symbol_append",
            }
    frames: list[pd.DataFrame] = []
    columns = list(dict.fromkeys([*existing.columns, *incoming.columns]))
    for origin, frame in ((0, existing), (1, incoming)):
        if frame.empty:
            continue
        missing_keys = [field for field in key_fields if field not in frame.columns]
        if missing_keys:
            raise ValueError(
                "fundamental merge missing key columns: "
                + ",".join(missing_keys)
            )
        working = frame.reindex(columns=columns).copy()
        for index, field in enumerate(key_fields):
            key_column = f"_merge_key_{index}"
            working[key_column] = _merge_key_values(working[field], field)
            if working[key_column].astype(str).str.strip().eq("").any():
                raise ValueError(
                    f"fundamental merge has empty key value: {field}"
                )
        numeric = pd.DataFrame(index=working.index)
        for field in quality_fields:
            numeric[field] = pd.to_numeric(
                working[field] if field in working.columns else np.nan,
                errors="coerce",
            )
        working["_merge_quality"] = (
            numeric.replace([np.inf, -np.inf], np.nan).notna().sum(axis=1)
        )
        working["_merge_origin"] = origin
        frames.append(working)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    merge_keys = [f"_merge_key_{index}" for index in range(len(key_fields))]
    combined = combined.sort_values(
        [*merge_keys, "_merge_quality", "_merge_origin"],
        kind="mergesort",
    )
    winners = combined.drop_duplicates(subset=merge_keys, keep="last")
    stats = {
        "existing_rows": int(len(existing)),
        "incoming_rows": int(len(incoming)),
        "merged_rows": int(len(winners)),
        "retained_existing_rows": int((winners["_merge_origin"] == 0).sum()),
        "accepted_incoming_rows": int((winners["_merge_origin"] == 1).sum()),
    }
    output = winners.sort_values(merge_keys, kind="mergesort").drop(
        columns=[*merge_keys, "_merge_quality", "_merge_origin"]
    )
    return output.reset_index(drop=True), stats


def _merge_quarantine_table(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    incoming_aligned = (
        _align_incoming_to_existing_schema(existing, incoming)
        if not existing.empty and not incoming.empty
        else incoming
    )
    columns = list(
        dict.fromkeys([*existing.columns, *incoming_aligned.columns])
    )
    origin_column = "_fundamental_quarantine_merge_origin"
    while origin_column in columns:
        origin_column = f"_{origin_column}"
    frames: list[pd.DataFrame] = []
    for origin, frame in ((0, existing), (1, incoming_aligned)):
        if frame.empty:
            continue
        working = frame.reindex(columns=columns).copy()
        working[origin_column] = origin
        frames.append(working)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        winners = combined.drop_duplicates(subset=columns, keep="last")
        retained_existing_rows = int((winners[origin_column] == 0).sum())
        accepted_incoming_rows = int((winners[origin_column] == 1).sum())
        merged = winners.drop(columns=[origin_column])
    else:
        merged = incoming.copy()
        retained_existing_rows = 0
        accepted_incoming_rows = 0
    return merged.reset_index(drop=True), {
        "existing_rows": int(len(existing)),
        "incoming_rows": int(len(incoming_aligned)),
        "merged_rows": int(len(merged)),
        "retained_existing_rows": retained_existing_rows,
        "accepted_incoming_rows": accepted_incoming_rows,
    }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _run_id(as_of: str | None = None) -> str:
    suffix = str(as_of or "").strip() or _now_utc().strftime("%Y%m%d")
    return f"cn_fundamental_{suffix}_{_now_utc().strftime('%H%M%S')}"


def _date_text(value: object) -> str:
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


def _period_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else text


def _string_series(values: Any, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.astype("string").fillna("").str.strip()


def _period_series(values: Any, index: pd.Index) -> pd.Series:
    text = _string_series(values, index)
    text = text.mask(text.str.lower().isin({"nan", "nat", "none"}), "")
    text = text.str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D+", "", regex=True)
    return digits.str[:8].where(digits.str.len() >= 8, text).fillna("")


def _date_series(values: Any, index: pd.Index) -> pd.Series:
    text = _string_series(values, index)
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


def _num(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _first_number(row: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = _num(row.get(name))
        if np.isfinite(value):
            return value
    return float("nan")


def _positive_denominator(value: float) -> float:
    return value if np.isfinite(value) and value > 0 else float("nan")


def _percent_to_ratio(value: object) -> float:
    number = _num(value)
    if not np.isfinite(number):
        return float("nan")
    return number / 100.0 if abs(number) > 2.0 else number


def _availability(row: Mapping[str, Any]) -> str:
    for column in ("f_ann_date", "ann_date", "availability_date"):
        text = _date_text(row.get(column))
        if text:
            return text
    return ""


def _load_sector_map(metadata_root: Path | None = None) -> dict[str, str]:
    root = metadata_root or DEFAULT_METADATA_ROOT
    candidates = [
        (root / "stock_list.parquet", ("sector", "industry")),
        (root / "stock_profiles.parquet", ("sector", "industry")),
    ]
    sector_map: dict[str, str] = {}
    for path, columns in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame = frame[[column for column in ["ts_code", "sector", "industry"] if column in frame.columns]]
        if "ts_code" not in frame.columns:
            continue
        working = frame.copy()
        working["ts_code"] = working["ts_code"].map(normalize_ts_code)
        sector = pd.Series("", index=working.index, dtype=object)
        for column in columns:
            if column not in working.columns:
                continue
            values = _string_series(working[column], working.index)
            values = values.mask(values.str.lower().isin({"nan", "nat", "none", "unknown"}), "")
            sector = sector.mask(sector.astype(str).str.strip().eq(""), values)
        for symbol, value in zip(working["ts_code"], sector, strict=False):
            text = str(value or "").strip()
            if symbol and text and symbol not in sector_map:
                sector_map[symbol] = text
    return sector_map


def _normalize_table(
    frame: pd.DataFrame | None,
    *,
    table: str,
    run_id: str,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    working = frame.copy()
    if "ts_code" not in working.columns:
        working["ts_code"] = ""
    working["ts_code"] = working["ts_code"].map(normalize_ts_code)
    if "end_date" not in working.columns:
        working["end_date"] = working.get("report_period", "")
    working["end_date"] = _period_series(working["end_date"], working.index)
    f_ann = _date_series(working["f_ann_date"] if "f_ann_date" in working.columns else "", working.index)
    ann = _date_series(working["ann_date"] if "ann_date" in working.columns else "", working.index)
    explicit = _date_series(
        working["availability_date"] if "availability_date" in working.columns else "",
        working.index,
    )
    working["availability_date"] = f_ann.mask(f_ann.eq(""), ann).mask(lambda value: value.eq(""), explicit)
    if "fetched_at" not in working.columns:
        working["fetched_at"] = _now_utc().isoformat()
    if "source" not in working.columns:
        working["source"] = source
    working["ingest_run_id"] = run_id
    working["raw_table"] = table
    missing = (
        working["ts_code"].astype(str).eq("")
        | working["end_date"].astype(str).eq("")
        | working["availability_date"].astype(str).eq("")
    )
    quarantine = working.loc[missing].copy()
    if not quarantine.empty:
        quarantine["quarantine_reason"] = "missing_ts_code_end_date_or_announcement_date"
    clean = working.loc[~missing].copy()
    clean = clean.sort_values(["ts_code", "end_date", "availability_date", "fetched_at"]).drop_duplicates(
        subset=["ts_code", "end_date", "availability_date", "raw_table"],
        keep="last",
    )
    return clean.reset_index(drop=True), quarantine.reset_index(drop=True)


def _prefix_columns(frame: pd.DataFrame, prefix: str, fields: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "availability_date"])
    keep = ["ts_code", "end_date", "availability_date", "fetched_at", "source"]
    keep.extend(field for field in fields if field in frame.columns)
    out = frame.loc[:, list(dict.fromkeys(keep))].copy()
    rename = {
        column: f"{prefix}_{column}"
        for column in out.columns
        if column not in {"ts_code", "end_date", "availability_date"}
    }
    return out.rename(columns=rename)


def _outer_period_frame(
    fina_indicator: pd.DataFrame,
    income: pd.DataFrame,
    balancesheet: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["ts_code", "end_date", "availability_date"]
    frames = [
        _prefix_columns(
            fina_indicator,
            "fi",
            (
                "roe_dt",
                "roe",
                "roa",
                "debt_to_assets",
                "netprofit_yoy",
                "ocf_to_profit",
            ),
        ),
        _prefix_columns(income, "inc", ("n_income_attr_p", "n_income")),
        _prefix_columns(balancesheet, "bs", ("total_liab", "total_assets")),
        _prefix_columns(cashflow, "cf", ("n_cashflow_act", "c_pay_acq_const_fiolta", "free_cashflow")),
    ]
    base = pd.DataFrame(columns=keys)
    for frame in frames:
        if frame.empty:
            continue
        base = frame if base.empty else base.merge(frame, on=keys, how="outer")
    return base.sort_values(keys).reset_index(drop=True) if not base.empty else base


def derive_fundamental_period(
    raw_tables: Mapping[str, pd.DataFrame],
    *,
    run_id: str,
    source: str = "tushare",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one PIT period row per symbol, period, and availability date."""

    clean_tables: dict[str, pd.DataFrame] = {}
    quarantines: list[pd.DataFrame] = []
    for table in ("fina_indicator", "income", "balancesheet", "cashflow"):
        clean, quarantine = _normalize_table(
            raw_tables.get(table),
            table=table,
            run_id=run_id,
            source=source,
        )
        clean_tables[table] = clean
        if not quarantine.empty:
            quarantines.append(quarantine)

    period = _outer_period_frame(
        clean_tables.get("fina_indicator", pd.DataFrame()),
        clean_tables.get("income", pd.DataFrame()),
        clean_tables.get("balancesheet", pd.DataFrame()),
        clean_tables.get("cashflow", pd.DataFrame()),
    )
    if period.empty:
        quarantine_frame = pd.concat(quarantines, ignore_index=True) if quarantines else pd.DataFrame()
        return pd.DataFrame(columns=["ts_code", "end_date", "availability_date", *DERIVED_PERIOD_FIELDS]), quarantine_frame

    records: list[dict[str, Any]] = []
    profit_by_symbol_period: dict[tuple[str, str], float] = {}
    for row in period.to_dict("records"):
        profit = _first_number(row, ("inc_n_income_attr_p", "inc_n_income"))
        profit_by_symbol_period[(str(row["ts_code"]), str(row["end_date"]))] = profit

    for row in period.to_dict("records"):
        total_assets = _num(row.get("bs_total_assets"))
        total_liab = _num(row.get("bs_total_liab"))
        profit = _first_number(row, ("inc_n_income_attr_p", "inc_n_income"))
        ocf = _num(row.get("cf_n_cashflow_act"))
        capex = _num(row.get("cf_c_pay_acq_const_fiolta"))
        free_cashflow_direct = _num(row.get("cf_free_cashflow"))
        free_cashflow = (
            free_cashflow_direct
            if np.isfinite(free_cashflow_direct)
            else (ocf - capex if np.isfinite(ocf) and np.isfinite(capex) else float("nan"))
        )
        end_date = pd.to_datetime(str(row["end_date"]), errors="coerce")
        prev_profit = float("nan")
        if not pd.isna(end_date):
            prev_period = (pd.Timestamp(end_date) - pd.DateOffset(years=1)).strftime("%Y%m%d")
            prev_profit = profit_by_symbol_period.get((str(row["ts_code"]), prev_period), float("nan"))
        direct_yoy = _percent_to_ratio(row.get("fi_netprofit_yoy"))
        fallback_yoy = (
            (profit - prev_profit) / prev_profit
            if np.isfinite(profit) and np.isfinite(prev_profit) and prev_profit > 0
            else float("nan")
        )
        direct_debt = _percent_to_ratio(row.get("fi_debt_to_assets"))
        fallback_debt = total_liab / total_assets if np.isfinite(total_liab) and total_assets > 0 else float("nan")
        direct_ocf_profit = _percent_to_ratio(row.get("fi_ocf_to_profit"))
        fallback_ocf_profit = ocf / _positive_denominator(profit) if np.isfinite(ocf) else float("nan")
        fcf_profit = free_cashflow / _positive_denominator(profit) if np.isfinite(free_cashflow) else float("nan")
        source_parts = sorted(
            {
                str(row.get(column))
                for column in row
                if column.endswith("_source") and str(row.get(column, "")).strip()
            }
        )
        fetched_parts = sorted(
            {
                str(row.get(column))
                for column in row
                if column.endswith("_fetched_at") and str(row.get(column, "")).strip()
            }
        )
        records.append(
            {
                "ts_code": row["ts_code"],
                "end_date": row["end_date"],
                "availability_date": row["availability_date"],
                "source_version": str(row["availability_date"]),
                "source": ";".join(source_parts) or source,
                "fetched_at": max(fetched_parts) if fetched_parts else "",
                "fin_roe": _percent_to_ratio(row.get("fi_roe_dt") if pd.notna(row.get("fi_roe_dt")) else row.get("fi_roe")),
                "fin_roa": _percent_to_ratio(row.get("fi_roa")),
                "fin_debt_to_assets": direct_debt if np.isfinite(direct_debt) else fallback_debt,
                "fin_net_profit_yoy": direct_yoy if np.isfinite(direct_yoy) else fallback_yoy,
                "fin_ocf_to_profit": direct_ocf_profit if np.isfinite(direct_ocf_profit) else fallback_ocf_profit,
                "fin_fcf_to_profit": fcf_profit,
                "free_cashflow": free_cashflow,
            }
        )
    output = pd.DataFrame(records)
    output["availability_date"] = pd.to_datetime(output["availability_date"], errors="coerce")
    output = output.sort_values(["ts_code", "end_date", "availability_date", "fetched_at"]).drop_duplicates(
        subset=["ts_code", "end_date", "availability_date"],
        keep="last",
    )
    quarantine_frame = pd.concat(quarantines, ignore_index=True) if quarantines else pd.DataFrame()
    return output.reset_index(drop=True), quarantine_frame


def _normalize_daily_basic(frame: pd.DataFrame | None, *, run_id: str, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "ts_code" not in working.columns:
        working["ts_code"] = ""
    working["ts_code"] = working["ts_code"].map(normalize_ts_code)
    if "trade_date" not in working.columns:
        working["trade_date"] = working.get("date", "")
    working["trade_date"] = pd.to_datetime(_date_series(working["trade_date"], working.index), errors="coerce")
    if "total_mv_rmb" in working.columns:
        working["total_mv_rmb"] = pd.to_numeric(working["total_mv_rmb"], errors="coerce")
    elif "total_mv" in working.columns:
        working["total_mv_rmb"] = pd.to_numeric(working["total_mv"], errors="coerce") * 10_000.0
    else:
        working["total_mv_rmb"] = np.nan
    sector = (
        _string_series(working["sector"], working.index)
        if "sector" in working.columns
        else pd.Series("", index=working.index, dtype=object)
    )
    if "industry" in working.columns:
        industry = _string_series(working["industry"], working.index)
        sector = sector.mask(sector.astype(str).str.strip().eq(""), industry)
    mapped_sector = working["ts_code"].map(_load_sector_map()).fillna("")
    sector = sector.mask(sector.astype(str).str.strip().eq(""), mapped_sector)
    working["sector"] = sector.replace("", "unknown")
    working["source"] = working.get("source", source)
    working["fetched_at"] = working.get("fetched_at", _now_utc().isoformat())
    working["ingest_run_id"] = run_id
    working = working.dropna(subset=["ts_code", "trade_date"])
    working = working[working["ts_code"].astype(str) != ""]
    return working.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"],
        keep="last",
    )


def _normalize_forecast(frame: pd.DataFrame | None, *, run_id: str, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=FORECAST_DAILY_COLUMNS)
    working = frame.copy()

    def numeric_column(column: str) -> pd.Series:
        if column not in working.columns:
            return pd.Series(np.nan, index=working.index, dtype=float)
        return pd.to_numeric(working[column], errors="coerce")

    if "ts_code" not in working.columns:
        working["ts_code"] = ""
    working["ts_code"] = working["ts_code"].map(normalize_ts_code)
    if "ann_date" not in working.columns:
        working["ann_date"] = working.get("f_ann_date", working.get("first_ann_date", ""))
    availability = _date_series(working["ann_date"], working.index)
    if "end_date" in working.columns:
        end_date = _period_series(working["end_date"], working.index)
    else:
        end_date = pd.Series("", index=working.index, dtype=object)
    p_min = numeric_column("p_change_min")
    p_max = numeric_column("p_change_max")
    forecast_revision = pd.concat([p_min, p_max], axis=1).mean(axis=1, skipna=True) / 100.0
    if "last_parent_net" in working.columns:
        net_min = numeric_column("net_profit_min")
        net_max = numeric_column("net_profit_max")
        last_parent_net = numeric_column("last_parent_net")
        net_mid = pd.concat([net_min, net_max], axis=1).mean(axis=1, skipna=True)
        fallback_revision = (net_mid - last_parent_net).div(last_parent_net.abs().where(last_parent_net.abs() > 0))
        forecast_revision = forecast_revision.fillna(fallback_revision)
    normalized = pd.DataFrame(
        {
            "ts_code": working["ts_code"],
            "forecast_end_date": end_date,
            "availability_date": pd.to_datetime(availability, errors="coerce"),
            "forecast_ann_date": availability,
            "forecast_revision": forecast_revision,
            "forecast_type": _string_series(working.get("type", ""), working.index),
            "forecast_summary": _string_series(working.get("summary", ""), working.index),
            "forecast_change_reason": _string_series(working.get("change_reason", ""), working.index),
            "forecast_source": working.get("source", source),
            "forecast_fetched_at": working.get("fetched_at", _now_utc().isoformat()),
            "forecast_ingest_run_id": run_id,
        }
    )
    normalized = normalized.dropna(subset=["ts_code", "availability_date"])
    normalized = normalized[normalized["ts_code"].astype(str) != ""]
    normalized = normalized[normalized["forecast_revision"].notna()]
    return normalized.sort_values(["ts_code", "availability_date", "forecast_end_date"]).drop_duplicates(
        subset=["ts_code", "availability_date", "forecast_end_date"],
        keep="last",
    )


def build_fundamental_daily(
    period: pd.DataFrame,
    daily_basic: pd.DataFrame | None,
    forecast: pd.DataFrame | None = None,
    *,
    run_id: str,
    source: str = "tushare",
) -> pd.DataFrame:
    """Join period fundamentals to exact daily rows with PIT merge-asof."""

    daily = _normalize_daily_basic(daily_basic, run_id=run_id, source=source)
    forecast_daily = _normalize_forecast(forecast, run_id=run_id, source=source)
    if period.empty or daily.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", *DERIVED_DAILY_FIELDS])
    period_work = period.copy()
    period_work["availability_date"] = pd.to_datetime(period_work["availability_date"], errors="coerce")
    outputs: list[pd.DataFrame] = []
    for symbol, daily_group in daily.groupby("ts_code", sort=True):
        period_group = period_work[period_work["ts_code"] == symbol].sort_values("availability_date")
        if period_group.empty:
            continue
        joined = pd.merge_asof(
            daily_group.sort_values("trade_date"),
            period_group.sort_values("availability_date"),
            left_on="trade_date",
            right_on="availability_date",
            direction="backward",
            suffixes=("", "_period"),
        )
        forecast_group = forecast_daily[forecast_daily["ts_code"] == symbol].sort_values("availability_date")
        if not forecast_group.empty:
            joined = pd.merge_asof(
                joined.sort_values("trade_date"),
                forecast_group.drop(columns=["ts_code"]).sort_values("availability_date"),
                left_on="trade_date",
                right_on="availability_date",
                direction="backward",
                suffixes=("", "_forecast"),
            )
        joined["ts_code"] = symbol
        outputs.append(joined)
    if not outputs:
        return pd.DataFrame(columns=["ts_code", "trade_date", *DERIVED_DAILY_FIELDS])
    out = pd.concat(outputs, ignore_index=True)
    out["fcf_to_price"] = pd.to_numeric(out.get("free_cashflow"), errors="coerce").div(
        pd.to_numeric(out.get("total_mv_rmb"), errors="coerce").where(
            pd.to_numeric(out.get("total_mv_rmb"), errors="coerce") > 0
        )
    )
    out["size_bucket"] = _size_buckets(out)
    keep = [
        "ts_code",
        "trade_date",
        "end_date",
        "availability_date",
        "source_version",
        "source",
        "fetched_at",
        "sector",
        "size_bucket",
        "total_mv_rmb",
        *DERIVED_DAILY_FIELDS,
        "forecast_end_date",
        "forecast_ann_date",
        "forecast_type",
        "forecast_summary",
        "forecast_change_reason",
        "forecast_source",
        "forecast_fetched_at",
        "forecast_ingest_run_id",
    ]
    return out[[column for column in keep if column in out.columns]].sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)


def _size_buckets(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("total_mv_rmb"), errors="coerce")
    if values.notna().sum() < 3:
        return pd.Series("unknown", index=frame.index)
    ranks = values.rank(pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["small", "mid", "large"],
        include_lowest=True,
    ).astype(str).replace("nan", "unknown")


def _coverage(frame: pd.DataFrame, fields: Sequence[str]) -> float:
    if frame.empty or not fields:
        return 0.0
    existing = [field for field in fields if field in frame.columns]
    if not existing:
        return 0.0
    total = len(frame) * len(existing)
    return float(frame[existing].notna().sum().sum() / max(total, 1))


def _bucket_share(frame: pd.DataFrame, group_column: str, fields: Sequence[str]) -> float:
    if frame.empty or group_column not in frame.columns:
        return 1.0
    existing = [field for field in fields if field in frame.columns]
    if not existing:
        return 1.0
    counts = frame.assign(_nonnull=frame[existing].notna().sum(axis=1)).groupby(group_column)["_nonnull"].sum()
    total = float(counts.sum())
    if total <= 0:
        return 1.0
    return float(counts.max() / total)


def build_readiness_payload(
    daily: pd.DataFrame,
    period: pd.DataFrame,
    quarantine: pd.DataFrame,
    *,
    run_id: str,
    fields: Sequence[str] = DERIVED_DAILY_FIELDS,
    expected_symbol_count: int = 0,
    require_expected_symbol_scope: bool = False,
) -> dict[str, Any]:
    coverage_rate = _coverage(daily, fields)
    monthly = (
        daily.assign(month=pd.to_datetime(daily["trade_date"], errors="coerce").dt.to_period("M").astype(str))
        if not daily.empty and "trade_date" in daily.columns
        else pd.DataFrame()
    )
    monthly_rates = (
        monthly.groupby("month").apply(lambda group: _coverage(group, fields), include_groups=False).to_dict()
        if not monthly.empty
        else {}
    )
    symbol_rates = (
        daily.groupby("ts_code").apply(lambda group: _coverage(group, fields), include_groups=False).to_dict()
        if not daily.empty and "ts_code" in daily.columns
        else {}
    )
    sector_rates = (
        daily.groupby("sector").apply(lambda group: _coverage(group, fields), include_groups=False).to_dict()
        if not daily.empty and "sector" in daily.columns
        else {}
    )
    size_rates = (
        daily.groupby("size_bucket").apply(lambda group: _coverage(group, fields), include_groups=False).to_dict()
        if not daily.empty and "size_bucket" in daily.columns
        else {}
    )
    monthly_min = min(monthly_rates.values()) if monthly_rates else 0.0
    nan_rate = 1.0 - coverage_rate
    symbols_with_period = int(
        period["ts_code"].nunique()
        if not period.empty and "ts_code" in period.columns
        else 0
    )
    expected_scope_available = int(expected_symbol_count) > 0
    symbol_coverage_rate = (
        min(symbols_with_period / int(expected_symbol_count), 1.0)
        if expected_scope_available
        else (0.0 if require_expected_symbol_scope else 1.0)
    )
    symbol_scope_surplus_count = max(
        symbols_with_period - int(expected_symbol_count),
        0,
    )
    gate2_passed = (
        coverage_rate >= 0.60
        and nan_rate <= 0.40
        and monthly_min >= 0.39
        and _bucket_share(daily, "sector", fields) <= 0.80
        and _bucket_share(daily, "size_bucket", fields) <= 0.80
        and symbol_coverage_rate >= 0.95
        and (expected_scope_available or not require_expected_symbol_scope)
    )
    blockers: list[str] = []
    if require_expected_symbol_scope and not expected_scope_available:
        blockers.append("expected_symbol_scope_missing")
    if coverage_rate < 0.60:
        blockers.append("coverage_rate_below_60pct")
    if nan_rate > 0.40:
        blockers.append("nan_rate_above_40pct")
    if monthly_min < 0.39:
        blockers.append("monthly_coverage_min_below_39pct")
    if _bucket_share(daily, "sector", fields) > 0.80:
        blockers.append("sector_coverage_concentration_above_80pct")
    if _bucket_share(daily, "size_bucket", fields) > 0.80:
        blockers.append("size_bucket_coverage_concentration_above_80pct")
    if symbol_coverage_rate < 0.95:
        blockers.append("symbol_coverage_below_95pct")
    if not period.empty and "availability_date" in period.columns:
        if pd.to_datetime(period["availability_date"], errors="coerce").isna().any():
            blockers.append("invalid_availability_date_in_period")
    return {
        "run_id": run_id,
        "schema_version": "cn-fundamental-readiness.v1",
        "field_set": list(fields),
        "period_rows": int(len(period)),
        "daily_rows": int(len(daily)),
        "quarantine_rows": int(len(quarantine)),
        "expected_symbol_count": int(expected_symbol_count),
        "expected_symbol_scope_required": bool(require_expected_symbol_scope),
        "expected_symbol_scope_available": bool(expected_scope_available),
        "symbols_with_period": symbols_with_period,
        "symbol_coverage_rate": float(symbol_coverage_rate),
        "symbol_scope_surplus_count": symbol_scope_surplus_count,
        "coverage_rate": coverage_rate,
        "nan_rate": nan_rate,
        "monthly_coverage_min": float(monthly_min),
        "max_sector_coverage_share": _bucket_share(daily, "sector", fields),
        "max_size_bucket_coverage_share": _bucket_share(daily, "size_bucket", fields),
        "gate2_passed": bool(gate2_passed),
        "blockers": blockers,
        "by_symbol": {str(k): float(v) for k, v in symbol_rates.items()},
        "by_month": {str(k): float(v) for k, v in monthly_rates.items()},
        "by_sector": {str(k): float(v) for k, v in sector_rates.items()},
        "by_size_bucket": {str(k): float(v) for k, v in size_rates.items()},
    }


def _readiness_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in ("by_symbol", "by_month", "by_sector", "by_size_bucket"):
        values = payload.get(group_name, {})
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            rows.append({"group": group_name.removeprefix("by_"), "key": key, "coverage_rate": value})
    return pd.DataFrame(rows, columns=["group", "key", "coverage_rate"])


def _render_readiness_md(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# CN Fundamental Readiness",
            "",
            f"- Run: `{payload.get('run_id')}`",
            f"- Period rows: {payload.get('period_rows')}",
            f"- Daily rows: {payload.get('daily_rows')}",
            f"- Quarantine rows: {payload.get('quarantine_rows')}",
            f"- Coverage: {float(payload.get('coverage_rate', 0.0)):.2%}",
            f"- NaN rate: {float(payload.get('nan_rate', 1.0)):.2%}",
            f"- Monthly coverage min: {float(payload.get('monthly_coverage_min', 0.0)):.2%}",
            f"- Symbol coverage: {float(payload.get('symbol_coverage_rate', 0.0)):.2%}",
            f"- Gate 2 passed: {payload.get('gate2_passed')}",
            f"- Blockers: {', '.join(payload.get('blockers', [])) or '-'}",
            "",
        ]
    )


def write_fundamental_mart(
    raw_tables: Mapping[str, pd.DataFrame],
    *,
    data_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    reports_root: str | Path = DEFAULT_READINESS_ROOT,
    run_id: str | None = None,
    source: str = "manual_offline_snapshot",
    provider_manifest: Mapping[str, Any] | None = None,
    write_raw_snapshots: bool = True,
    require_expected_symbol_scope: bool = False,
    publish_on_gate_failure: bool = True,
    _live_tushare_attestation: _LiveTushareAttestation | None = None,
) -> tuple[FundamentalMartArtifacts, dict[str, Any]]:
    run_id = run_id or _run_id()
    source_priority = _provider_source_priority(
        source,
        provider_manifest,
        raw_tables,
        _live_tushare_attestation,
    )
    data_dir = _resolve_data_base(data_root)
    snapshot_dir = Path(raw_snapshot_root).expanduser()
    reports_dir = Path(reports_root).expanduser()
    for path in (data_dir, snapshot_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    if write_raw_snapshots:
        for table in SOURCE_TABLES:
            frame = raw_tables.get(table)
            if frame is None:
                continue
            raw_dir = snapshot_dir / table
            raw_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(raw_dir / f"{run_id}.csv", index=False)

    incoming_period, incoming_quarantine = derive_fundamental_period(
        raw_tables,
        run_id=run_id,
        source=source,
    )
    incoming_daily = build_fundamental_daily(
        incoming_period,
        raw_tables.get("daily_basic"),
        raw_tables.get("forecast"),
        run_id=run_id,
        source=source,
    )
    prior_pointer = load_fundamental_pointer(data_dir)
    existing = {
        table_name: _read_existing_fundamental_table(data_dir, table_name)
        for table_name in FUNDAMENTAL_TABLES
    }
    period, period_merge = _merge_fundamental_table(
        existing["fundamental_period"],
        incoming_period,
        key_fields=("ts_code", "end_date", "availability_date"),
        quality_fields=DERIVED_PERIOD_FIELDS,
    )
    daily, daily_merge = _merge_fundamental_table(
        existing["fundamental_daily"],
        incoming_daily,
        key_fields=("ts_code", "trade_date"),
        quality_fields=DERIVED_DAILY_FIELDS,
    )
    quarantine, quarantine_merge = _merge_quarantine_table(
        existing["fundamental_quarantine"],
        incoming_quarantine,
    )
    retained_existing_rows = sum(
        int(stats.get("retained_existing_rows", 0) or 0)
        for stats in (period_merge, daily_merge, quarantine_merge)
    )
    parent_generation_id = ""
    parent_primary_provenance_sha256 = ""
    if source_priority == "tushare_primary" and retained_existing_rows:
        if (
            prior_pointer is None
            or prior_pointer.get("primary_provenance_verified") is not True
        ):
            raise ValueError(
                "primary generation cannot retain rows from an unverified parent"
            )
        parent_generation_id = str(
            prior_pointer.get("generation_id") or ""
        ).strip()
        parent_primary_provenance_sha256 = str(
            dict(prior_pointer.get("primary_provenance", {}) or {}).get(
                "envelope_sha256"
            )
            or ""
        ).strip()
    expected_symbol_count = int(
        dict(provider_manifest or {}).get("symbols_requested", 0) or 0
    )
    readiness = build_readiness_payload(
        daily,
        period,
        quarantine,
        run_id=run_id,
        expected_symbol_count=expected_symbol_count,
        require_expected_symbol_scope=require_expected_symbol_scope,
    )
    raw_row_counts = {table: int(len(raw_tables.get(table, pd.DataFrame()))) for table in SOURCE_TABLES}
    readiness["provider_status"] = source
    readiness["source_priority"] = source_priority
    readiness["raw_row_counts"] = raw_row_counts
    readiness["raw_snapshot_written"] = bool(write_raw_snapshots)
    readiness["merge"] = {
        "fundamental_period": period_merge,
        "fundamental_daily": daily_merge,
        "fundamental_quarantine": quarantine_merge,
        "absence_is_deletion": False,
        "deletion_policy": "explicit_tombstone_required",
    }
    if provider_manifest:
        readiness["provider_manifest"] = dict(provider_manifest)

    readiness_json = reports_dir / f"{run_id}.json"
    readiness_md = reports_dir / f"{run_id}.md"
    readiness_csv = reports_dir / f"{run_id}.csv"
    readiness_json.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness_md.write_text(_render_readiness_md(readiness), encoding="utf-8")
    _readiness_rows(readiness).to_csv(readiness_csv, index=False)
    if not readiness["gate2_passed"] and not publish_on_gate_failure:
        raise FundamentalReadinessError(readiness)
    generation_metadata = {
        "run_id": run_id,
        "provider_status": source,
        "source_priority": source_priority,
        "source_provenance": dict(provider_manifest or {}).get(
            "source_provenance",
            "",
        ),
        "raw_row_counts": raw_row_counts,
        "raw_snapshot_written": bool(write_raw_snapshots),
        "provider_manifest": dict(provider_manifest or {}),
        "storage_backend": "parquet_canonical",
        "readiness": str(readiness_json),
        "gate2_passed": readiness["gate2_passed"],
        "merge": readiness["merge"],
    }
    if parent_generation_id:
        generation_metadata.update(
            {
                "parent_generation_id": parent_generation_id,
                "parent_primary_provenance_sha256": (
                    parent_primary_provenance_sha256
                ),
            }
        )
    generation_tables = {
        "fundamental_period": period,
        "fundamental_daily": daily,
        "fundamental_quarantine": quarantine,
    }
    primary_generation_attestation = None
    if source_priority == "tushare_primary":
        if not _live_tushare_attestation_matches(
            _live_tushare_attestation,
            source=source,
            provider_manifest=dict(provider_manifest or {}),
            raw_tables=raw_tables,
        ):
            raise ValueError(
                "live Tushare attestation changed before generation publication"
            )
        primary_generation_attestation = (
            _issue_primary_generation_attestation(
                tables=generation_tables,
                metadata=generation_metadata,
                source=_live_tushare_attestation.source,
                provider_manifest_sha256=(
                    _live_tushare_attestation.provider_manifest_sha256
                ),
                raw_table_fingerprints=(
                    _live_tushare_attestation.raw_table_fingerprints
                ),
            )
        )
    generation_paths, pointer = publish_fundamental_generation(
        root=data_dir,
        run_id=run_id,
        tables=generation_tables,
        metadata=generation_metadata,
        _primary_attestation=primary_generation_attestation,
    )
    period_path = generation_paths["fundamental_period"]
    daily_path = generation_paths["fundamental_daily"]
    quarantine_path = generation_paths["fundamental_quarantine"]
    manifest = {
        **generation_metadata,
        "generation_id": pointer["generation_id"],
        "pointer_path": str(data_dir / "_fundamental_latest.json"),
        "fundamental_period": str(period_path),
        "fundamental_daily": str(daily_path),
        "fundamental_quarantine": str(quarantine_path),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (data_dir / "latest_manifest.json").write_text(manifest_text, encoding="utf-8")
    artifacts = FundamentalMartArtifacts(
        run_id=run_id,
        data_root=data_dir,
        raw_snapshot_root=snapshot_dir,
        reports_root=reports_dir,
        fundamental_period_path=period_path,
        fundamental_daily_path=daily_path,
        quarantine_path=quarantine_path,
        readiness_json_path=readiness_json,
        readiness_md_path=readiness_md,
        readiness_csv_path=readiness_csv,
    )
    return artifacts, readiness


def _read_raw_input_dir(path: str | Path | None) -> dict[str, pd.DataFrame]:
    if not path:
        return {}
    base = Path(path).expanduser()
    tables: dict[str, pd.DataFrame] = {}
    for table in SOURCE_TABLES:
        parquet_path = base / f"{table}.parquet"
        if parquet_path.exists():
            tables[table] = pd.read_parquet(parquet_path)
    return tables


def _resolve_symbols_from_parquet_universe(
    data_root: str | Path,
    universes: Sequence[str],
) -> list[str]:
    symbols: list[str] = []
    reader = MarketDataReader(market="CN", data_root=data_root, mode_policy="strict")
    components_path = (
        Path(data_root).expanduser()
        / "cn_universe"
        / "cn_index_components.json"
    )
    components: Mapping[str, Any] = {}
    if components_path.exists():
        try:
            components_payload = json.loads(
                components_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ValueError(
                f"invalid canonical universe components: {components_path}: {exc}"
            ) from exc
        if not isinstance(components_payload, Mapping):
            raise ValueError(
                f"canonical universe components must be an object: {components_path}"
            )
        components = components_payload
    serving_symbols = set(reader.list_symbols(universe_key="full_a"))
    for universe in universes:
        normalized_universe = str(universe or "").strip().lower() or "full_a"
        if normalized_universe in FULL_A_UNIVERSE_KEYS:
            scoped = [
                normalize_ts_code(symbol)
                for symbol in list(components.get("full_a", []) or [])
                if normalize_ts_code(symbol) in serving_symbols
            ]
            symbols.extend(scoped or sorted(serving_symbols))
        else:
            symbols.extend(reader.list_symbols(universe_key=normalized_universe))
    return [symbol for symbol in dict.fromkeys(symbols) if symbol]


def _resolve_symbols_from_daily_root(daily_root: Path, universes: Sequence[str]) -> list[str]:
    """Compatibility wrapper; CN production symbols now come from Parquet."""
    return _resolve_symbols_from_parquet_universe(daily_root, universes)


def _fetch_tushare_tables(
    symbols: Sequence[str],
    *,
    years: int,
    as_of: str,
    workers: int,
    pro: Any,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    end = pd.to_datetime(as_of or _now_utc().date(), errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp(_now_utc().date())
    start = end - pd.DateOffset(years=int(years))
    financial_start = start - pd.DateOffset(years=2)
    start_text = start.strftime("%Y%m%d")
    financial_start_text = financial_start.strftime("%Y%m%d")
    end_text = pd.Timestamp(end).strftime("%Y%m%d")
    fields = {
        "fina_indicator": "ts_code,ann_date,f_ann_date,end_date,roe_dt,roe,roa,debt_to_assets,netprofit_yoy,ocf_to_profit,update_flag",
        "income": "ts_code,ann_date,f_ann_date,end_date,n_income,n_income_attr_p,update_flag",
        "balancesheet": "ts_code,ann_date,f_ann_date,end_date,total_liab,total_assets,update_flag",
        "cashflow": "ts_code,ann_date,f_ann_date,end_date,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow,update_flag",
        "daily_basic": "ts_code,trade_date,total_mv,circ_mv,pe,pb",
        "forecast": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net,summary,change_reason,update_flag",
    }
    collected = {table: [] for table in SOURCE_TABLES}
    errors: list[dict[str, str]] = []
    attempted = 0
    succeeded = 0
    empty = 0
    failed = 0
    outcomes: list[dict[str, Any]] = []

    def fetch_symbol(
        symbol: str,
    ) -> tuple[
        dict[str, list[pd.DataFrame]],
        int,
        int,
        int,
        list[dict[str, str]],
        list[dict[str, Any]],
    ]:
        symbol_frames = {table: [] for table in SOURCE_TABLES}
        symbol_succeeded = 0
        symbol_empty = 0
        symbol_failed = 0
        symbol_errors: list[dict[str, str]] = []
        symbol_outcomes: list[dict[str, Any]] = []
        for table in SOURCE_TABLES:
            method = getattr(pro, table, None)
            if method is None:
                symbol_failed += 1
                error = "provider_endpoint_missing"
                symbol_errors.append({"symbol": symbol, "table": table, "error": error})
                symbol_outcomes.append(
                    {"symbol": symbol, "table": table, "status": "error", "rows": 0, "error": error}
                )
                continue
            table_start_text = financial_start_text if table in FINANCIAL_SOURCE_TABLES or table == "forecast" else start_text
            try:
                frame = method(
                    ts_code=symbol,
                    start_date=table_start_text,
                    end_date=end_text,
                    fields=fields[table],
                )
            except TypeError:
                try:
                    frame = method(ts_code=symbol, fields=fields[table])
                except Exception as exc:
                    symbol_failed += 1
                    symbol_errors.append({"symbol": symbol, "table": table, "error": str(exc)})
                    symbol_outcomes.append(
                        {"symbol": symbol, "table": table, "status": "error", "rows": 0, "error": str(exc)}
                    )
                    continue
            except Exception as exc:
                symbol_failed += 1
                symbol_errors.append({"symbol": symbol, "table": table, "error": str(exc)})
                symbol_outcomes.append(
                    {"symbol": symbol, "table": table, "status": "error", "rows": 0, "error": str(exc)}
                )
                continue
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                symbol_frames[table].append(frame)
                symbol_succeeded += 1
                symbol_outcomes.append(
                    {"symbol": symbol, "table": table, "status": "success", "rows": int(len(frame)), "error": ""}
                )
            else:
                symbol_empty += 1
                symbol_outcomes.append(
                    {"symbol": symbol, "table": table, "status": "empty", "rows": 0, "error": ""}
                )
        time.sleep(0.05)
        return (
            symbol_frames,
            symbol_succeeded,
            symbol_empty,
            symbol_failed,
            symbol_errors,
            symbol_outcomes,
        )

    worker_count = max(1, int(workers or 1))
    if worker_count == 1 or len(symbols) <= 1:
        for symbol in symbols:
            (
                symbol_frames,
                symbol_succeeded,
                symbol_empty,
                symbol_failed,
                symbol_errors,
                symbol_outcomes,
            ) = fetch_symbol(symbol)
            attempted += len(SOURCE_TABLES)
            succeeded += symbol_succeeded
            empty += symbol_empty
            failed += symbol_failed
            for table, frames in symbol_frames.items():
                collected[table].extend(frames)
            if len(errors) < 200:
                errors.extend(symbol_errors[: max(0, 200 - len(errors))])
            outcomes.extend(symbol_outcomes)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fetch_symbol, symbol) for symbol in symbols]
            for future in as_completed(futures):
                (
                    symbol_frames,
                    symbol_succeeded,
                    symbol_empty,
                    symbol_failed,
                    symbol_errors,
                    symbol_outcomes,
                ) = future.result()
                completed += 1
                attempted += len(SOURCE_TABLES)
                succeeded += symbol_succeeded
                empty += symbol_empty
                failed += symbol_failed
                for table, frames in symbol_frames.items():
                    collected[table].extend(frames)
                if len(errors) < 200:
                    errors.extend(symbol_errors[: max(0, 200 - len(errors))])
                outcomes.extend(symbol_outcomes)
                if len(symbols) >= 50 and completed % 100 == 0:
                    print(
                        f"[fundamental-maintain] fetched {completed}/{len(symbols)} symbols; "
                        f"failed_requests={failed}",
                        flush=True,
                    )
    tables = {
        table: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for table, frames in collected.items()
    }
    manifest = {
        "provider": "tushare",
        "symbols_requested": int(len(symbols)),
        "tables": list(SOURCE_TABLES),
        "years": int(years),
        "start_date": start_text,
        "daily_start_date": start_text,
        "financial_start_date": financial_start_text,
        "end_date": end_text,
        "requests_attempted": int(attempted),
        "requests_succeeded_with_rows": int(succeeded),
        "requests_empty": int(empty),
        "requests_failed": int(failed),
        "errors_truncated": failed > len(errors),
        "errors": errors,
        "symbol_table_outcomes": sorted(
            outcomes,
            key=lambda item: (str(item["symbol"]), str(item["table"])),
        ),
        "absence_is_deletion": False,
        "deletion_policy": "explicit_tombstone_required",
        "raw_row_counts": {table: int(len(frame)) for table, frame in tables.items()},
    }
    return tables, manifest


def run_cn_fundamental_maintenance(
    *,
    market: str = "CN",
    universes: Sequence[str] | str | None = None,
    years: int = 5,
    as_of: str = "",
    workers: int = 4,
    data_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    reports_root: str | Path = DEFAULT_READINESS_ROOT,
    raw_input_dir: str | Path | None = None,
    allow_live: bool = False,
    pro: Any | None = None,
    raw_tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if str(market).upper() != "CN":
        raise ValueError("fundamental-maintain currently supports CN only")
    universe_list = (
        [item.strip() for item in universes.split(",") if item.strip()]
        if isinstance(universes, str)
        else list(universes or DEFAULT_UNIVERSES)
    )
    run_id = _run_id(as_of)
    try:
        scope_symbols = _resolve_symbols_from_parquet_universe(
            DEFAULT_MARKET_DATA_ROOT,
            universe_list,
        )
        scope_error = ""
    except Exception as exc:
        scope_symbols = []
        scope_error = f"{type(exc).__name__}:{exc}"
    provider_manifest: dict[str, Any] = {
        "symbols_requested": int(len(scope_symbols)),
        "symbol_scope_status": "resolved" if scope_symbols else "missing",
        "symbol_scope_source": "strict_parquet_serving_intersect_canonical_components",
        "symbol_scope_universes": list(universe_list),
        "source_priority": "manual_offline_snapshot",
        "source_provenance": "offline_input_unverified",
    }
    if scope_error:
        provider_manifest["symbol_scope_error"] = scope_error
    live_tushare_attestation: _LiveTushareAttestation | None = None
    tables = dict(raw_tables or {})
    if not tables:
        tables = _read_raw_input_dir(raw_input_dir)
    provider_status = "offline_input" if tables else "not_requested"
    if not tables and allow_live:
        if pro is None:
            try:
                import tushare as ts  # type: ignore

                from quant_investor.config import config
                from quant_investor.credential_utils import create_tushare_pro

                if not config.TUSHARE_TOKEN:
                    raise RuntimeError("missing TUSHARE_TOKEN")
                pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
            except Exception as exc:
                provider_status = f"provider_unavailable:{exc}"
                provider_manifest.update(
                    {
                        "provider": "tushare",
                        "provider_status": provider_status,
                    }
                )
                pro = None
        if pro is not None:
            tables, fetch_manifest = _fetch_tushare_tables(
                scope_symbols,
                years=int(years),
                as_of=as_of,
                workers=int(workers),
                pro=pro,
            )
            provider_manifest.update(fetch_manifest)
            provider_manifest.update(
                {
                    "symbols_requested": int(len(scope_symbols)),
                    "symbol_scope_status": "resolved" if scope_symbols else "missing",
                    "symbol_scope_source": "strict_parquet_serving_intersect_canonical_components",
                    "symbol_scope_universes": list(universe_list),
                    "source_priority": "tushare_primary",
                    "source_provenance": "live_tushare_explicit",
                }
            )
            provider_status = (
                "live_tushare_partial"
                if int(provider_manifest.get("requests_failed", 0)) > 0
                else "live_tushare"
            )
            provider_manifest["provider_status"] = provider_status
            live_tushare_attestation = _issue_live_tushare_attestation(
                provider_status,
                provider_manifest,
                tables,
            )
    if not tables:
        tables = {table: pd.DataFrame() for table in SOURCE_TABLES}
    artifacts, readiness = write_fundamental_mart(
        tables,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        reports_root=reports_root,
        run_id=run_id,
        source=provider_status,
        provider_manifest=provider_manifest,
        require_expected_symbol_scope=True,
        publish_on_gate_failure=False,
        _live_tushare_attestation=live_tushare_attestation,
    )
    return {
        "run_id": run_id,
        "provider_status": provider_status,
        "universes": universe_list,
        "years": int(years),
        "as_of": as_of,
        "artifacts": {key: str(value) for key, value in artifacts.__dict__.items() if key.endswith("_path")},
        "readiness": readiness,
    }


__all__ = [
    "DEFAULT_FUNDAMENTAL_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "DEFAULT_READINESS_ROOT",
    "DEFAULT_UNIVERSES",
    "DERIVED_DAILY_FIELDS",
    "DERIVED_PERIOD_FIELDS",
    "FundamentalReadinessError",
    "FundamentalMartArtifacts",
    "build_fundamental_daily",
    "build_readiness_payload",
    "derive_fundamental_period",
    "run_cn_fundamental_maintenance",
    "write_fundamental_mart",
]
