"""Standalone CN point-in-time fundamental mart builder.

The mart is intentionally separate from daily market maintenance. It can be
driven from offline fixtures in tests or from an explicit live provider in
operations, and it treats missing announcement dates as a hard quarantine.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

import fcntl

import numpy as np
import pandas as pd

from quant_investor.factors.pit_fundamentals import normalize_ts_code
from quant_investor.market.fundamental_generation import (
    FUNDAMENTAL_TABLES,
    _daily_history_coverage_metrics,
    _issue_primary_generation_attestation,
    _listing_identity_sha256,
    _validate_daily_history_coverage_intervals,
    load_fundamental_pointer,
    pointer_sha256 as fundamental_pointer_sha256,
    publish_fundamental_generation,
    resolve_fundamental_table_path,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.fundamental_provider_contract import (
    FUNDAMENTAL_DERIVATION_CONTRACT,
    FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA,
    FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA,
    FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
    FUNDAMENTAL_FETCH_PIT_CONTRACT,
    FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
    FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
    FundamentalEndpointAuditPolicy,
    assert_frame_semantics_equal,
    build_financial_coverage,
    canonical_json_sha256,
    frame_fingerprint,
    frame_logical_schema,
    matured_quarter_baseline,
    strict_nonnegative_int,
    validate_outcome_accounting_v3,
)

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
SOURCE_REQUIRED_COLUMNS = {
    "fina_indicator": {
        "end_date",
        "roe_dt",
        "roe",
        "roa",
        "debt_to_assets",
        "netprofit_yoy",
    },
    "income": {"end_date", "n_income", "n_income_attr_p", "update_flag"},
    "balancesheet": {"end_date", "total_liab", "total_assets", "update_flag"},
    "cashflow": {
        "end_date",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
        "free_cashflow",
        "update_flag",
    },
    "daily_basic": {"trade_date", "total_mv", "circ_mv", "pe", "pb"},
    "forecast": {
        "end_date",
        "type",
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "last_parent_net",
        "summary",
        "change_reason",
        "update_flag",
    },
}
SOURCE_VALUE_COLUMNS = {
    "fina_indicator": ("roe_dt", "roe", "roa", "debt_to_assets"),
    "income": ("n_income", "n_income_attr_p"),
    "balancesheet": ("total_liab", "total_assets"),
    "cashflow": ("n_cashflow_act", "free_cashflow"),
    "daily_basic": ("total_mv", "circ_mv"),
    "forecast": ("p_change_min", "p_change_max", "summary"),
}
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


SOURCE_REQUEST_FIELDS = {
    "fina_indicator": "ts_code,ann_date,end_date,roe_dt,roe,roa,debt_to_assets,netprofit_yoy",
    "income": "ts_code,ann_date,f_ann_date,end_date,n_income,n_income_attr_p,update_flag",
    "balancesheet": "ts_code,ann_date,f_ann_date,end_date,total_liab,total_assets,update_flag",
    "cashflow": "ts_code,ann_date,f_ann_date,end_date,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow,update_flag",
    "daily_basic": "ts_code,trade_date,total_mv,circ_mv,pe,pb",
    "forecast": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net,summary,change_reason,update_flag",
}
CRITICAL_BASE_TABLES = tuple(table for table in SOURCE_TABLES if table != "forecast")


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


class FundamentalFetchCheckpointError(RuntimeError):
    """Raised when resumable live-fetch evidence is missing or inconsistent."""


class FundamentalFetchAuditError(RuntimeError):
    """Raised when a full-rebuild provider fetch fails its pre-publication audit."""

    def __init__(self, manifest: Mapping[str, Any]):
        self.manifest = dict(manifest)
        audit = dict(manifest.get("endpoint_audit", {}) or {})
        blockers = ",".join(str(item) for item in audit.get("blockers", []))
        super().__init__(f"fundamental endpoint audit failed: {blockers or 'unknown'}")


_LIVE_TUSHARE_CAPABILITY = object()


@dataclass(frozen=True)
class _LiveTushareAttestation:
    capability: object
    source: str
    provider_manifest_sha256: str
    raw_table_fingerprints: tuple[tuple[str, str], ...]


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(value))


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
            frame_fingerprint(raw_tables.get(table, pd.DataFrame())),
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


def _issue_live_tushare_v4_attestation(
    source: str,
    provider_manifest: Mapping[str, Any],
    raw_tables: Mapping[str, pd.DataFrame],
) -> _LiveTushareAttestation:
    """Issue the internal capability only after the sealed v4 replay passed."""

    manifest = dict(provider_manifest)
    if (
        source != "live_tushare_vip"
        or manifest.get("schema_version")
        != "cn-fundamental-provider-manifest.v4"
        or manifest.get("authoritative_full_rebuild") is not True
        or manifest.get("performance_gate_passed") is not True
        or set(dict(manifest.get("raw_table_fingerprints", {}) or {}))
        != set(SOURCE_TABLES)
        or set(raw_tables) != set(SOURCE_TABLES)
    ):
        raise ValueError("live Tushare v4 provenance attestation failed")
    return _LiveTushareAttestation(
        capability=_LIVE_TUSHARE_CAPABILITY,
        source=source,
        provider_manifest_sha256=_canonical_mapping_sha256(manifest),
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
    if manifest.get("schema_version") == "cn-fundamental-provider-manifest.v4":
        if _live_tushare_attestation_matches(
            live_tushare_attestation,
            source=source,
            provider_manifest=manifest,
            raw_tables=raw_tables,
        ):
            return "tushare_primary"
        raise ValueError(
            "Fundamental v4 requires an internal live Tushare attestation"
        )
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
        provider_manifest.get("schema_version")
        != FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA
        or str(provider_manifest.get("provider") or "").strip().lower()
        != "tushare"
        or str(
            provider_manifest.get("source_provenance") or ""
        ).strip()
        != "live_tushare_explicit"
    ):
        return False
    try:
        expected_counts = {
            table: len(raw_tables.get(table, pd.DataFrame()))
            for table in SOURCE_TABLES
        }
        declared_counts = {
            str(key): strict_nonnegative_int(
                value,
                label=f"provider raw row count {key}",
            )
            for key, value in dict(
                provider_manifest.get("raw_row_counts", {}) or {}
            ).items()
        }
    except (TypeError, ValueError):
        return False
    if declared_counts != expected_counts:
        return False
    try:
        attempted = strict_nonnegative_int(
            provider_manifest.get("requests_attempted"),
            label="provider requests_attempted",
        )
        accounted = sum(
            strict_nonnegative_int(
                provider_manifest.get(field), label=f"provider {field}"
            )
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
    if not (
        attempted > 0
        and attempted == accounted
        and isinstance(outcomes, list)
        and len(outcomes) == attempted
        and declared_tables == set(SOURCE_TABLES)
    ):
        return False
    try:
        for index, outcome in enumerate(outcomes):
            validate_outcome_accounting_v3(
                outcome,
                label=f"provider outcome {index}",
            )
        if str(
            provider_manifest.get("request_outcome_accounting_sha256") or ""
        ).lower() != canonical_json_sha256(outcomes):
            return False
        declared_fingerprints = dict(
            provider_manifest.get("raw_table_fingerprints", {}) or {}
        )
        actual_fingerprints = {
            table: frame_fingerprint(raw_tables.get(table, pd.DataFrame()))
            for table in SOURCE_TABLES
        }
        if declared_fingerprints != actual_fingerprints:
            return False
    except (TypeError, ValueError):
        return False
    audit = provider_manifest.get("endpoint_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA:
        return False
    if provider_manifest.get("authoritative_full_rebuild") is True:
        checkpoint = provider_manifest.get("checkpoint")
        derivation = provider_manifest.get("derivation")
        if not isinstance(checkpoint, Mapping) or not isinstance(derivation, Mapping):
            return False
        if checkpoint.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA:
            return False
        if derivation.get("contract_version") != FUNDAMENTAL_DERIVATION_CONTRACT:
            return False
        if derivation.get("selection_rule") != (
            "latest_active_membership_interval_as_of_else_latest_expired"
        ):
            return False
        if dict(derivation.get("raw_table_fingerprints", {}) or {}) != dict(
            provider_manifest.get("raw_table_fingerprints", {}) or {}
        ):
            return False
        if str(
            provider_manifest.get("raw_to_derived_binding_sha256") or ""
        ).lower() != canonical_json_sha256(derivation):
            return False
        if set(dict(derivation.get("output_frame_fingerprints", {}) or {})) != set(
            FUNDAMENTAL_TABLES
        ):
            return False
        for field in (
            "pointer_sha256",
            "manifest_sha256",
            "binding_sha256",
            "outcome_accounting_sha256",
            "table_evidence_sha256",
        ):
            value = str(checkpoint.get(field) or "").lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                return False
    return True


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
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


def _validate_authoritative_derived_bundle(
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Validate an isolated full-rebuild bundle without copying its frames."""

    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise ValueError("v3 derived table bundle is incomplete")
    key_fields = {
        "fundamental_period": ("ts_code", "end_date", "availability_date"),
        "fundamental_daily": ("ts_code", "trade_date"),
        "fundamental_quarantine": ("ts_code", "quarantine_reason"),
    }
    for table_name in FUNDAMENTAL_TABLES:
        frame = tables[table_name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"v3 derived table is not a DataFrame: {table_name}"
            )
        required = key_fields[table_name]
        missing = [field for field in required if field not in frame.columns]
        if missing and not frame.empty:
            raise ValueError(
                f"v3 derived table missing keys: {table_name}:" + ",".join(missing)
            )
        if frame.empty:
            continue
        for field in required:
            values = frame[field].astype("string").fillna("").str.strip()
            if values.eq("").any():
                raise ValueError(
                    f"v3 derived table has empty key: {table_name}:{field}"
                )
        if frame.duplicated(subset=list(required)).any():
            raise ValueError(f"v3 derived table has duplicate keys: {table_name}")


def _authoritative_replace_stats(frame: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(frame))
    return {
        "existing_rows": 0,
        "incoming_rows": rows,
        "merged_rows": rows,
        "retained_existing_rows": 0,
        "accepted_incoming_rows": rows,
        "merge_path": "authoritative_isolated_replace",
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


def _num(value: Any) -> float:
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
    derivation_timestamp: str | None = None,
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
        working["fetched_at"] = derivation_timestamp or _now_utc().isoformat()
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
    derivation_timestamp: str | None = None,
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
            derivation_timestamp=derivation_timestamp,
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


def _normalize_daily_basic(
    frame: pd.DataFrame | None,
    *,
    run_id: str,
    source: str,
    sector_map: Mapping[str, str] | None = None,
    derivation_timestamp: str | None = None,
) -> pd.DataFrame:
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
    if sector_map is not None:
        sector = working["ts_code"].map(dict(sector_map)).fillna("")
    else:
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
    working["fetched_at"] = working.get(
        "fetched_at",
        derivation_timestamp or _now_utc().isoformat(),
    )
    working["ingest_run_id"] = run_id
    working = working.dropna(subset=["ts_code", "trade_date"])
    working = working[working["ts_code"].astype(str) != ""]
    return working.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"],
        keep="last",
    )


def _normalize_forecast(
    frame: pd.DataFrame | None,
    *,
    run_id: str,
    source: str,
    derivation_timestamp: str | None = None,
) -> pd.DataFrame:
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
            "forecast_fetched_at": working.get(
                "fetched_at",
                derivation_timestamp or _now_utc().isoformat(),
            ),
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
    sector_map: Mapping[str, str] | None = None,
    derivation_timestamp: str | None = None,
) -> pd.DataFrame:
    """Join period fundamentals to exact daily rows with PIT merge-asof."""

    daily = _normalize_daily_basic(
        daily_basic,
        run_id=run_id,
        source=source,
        sector_map=sector_map,
        derivation_timestamp=derivation_timestamp,
    )
    forecast_daily = _normalize_forecast(
        forecast,
        run_id=run_id,
        source=source,
        derivation_timestamp=derivation_timestamp,
    )
    if period.empty or daily.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", *DERIVED_DAILY_FIELDS])
    period_work = period.copy()
    period_work["ts_code"] = period_work["ts_code"].map(normalize_ts_code)
    period_work["availability_date"] = pd.to_datetime(
        period_work["availability_date"],
        errors="coerce",
    )
    period_symbols = {
        str(symbol)
        for symbol in period_work["ts_code"].dropna().drop_duplicates()
        if str(symbol)
    }
    daily = daily[daily["ts_code"].isin(period_symbols)]
    if daily.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", *DERIVED_DAILY_FIELDS])
    # ``merge_asof`` selects the last right-hand row when several fiscal
    # periods share an availability date.  The pre-v3 implementation sorted
    # each symbol independently with pandas' default sort before joining.  Its
    # tie winner is therefore part of the historical PIT contract even though
    # that default sort is not stable.  Collapse only the small right-hand
    # tables with the legacy ordering before the global vectorized join; this
    # preserves prior values without rebuilding millions of per-symbol daily
    # frames.
    period_work = _legacy_asof_tie_winners(period_work)
    daily = daily.sort_values(
        ["trade_date", "ts_code"],
        kind="mergesort",
    )
    period_work = period_work.sort_values(
        ["availability_date", "ts_code"],
        kind="mergesort",
    )
    out = pd.merge_asof(
        daily,
        period_work,
        by="ts_code",
        left_on="trade_date",
        right_on="availability_date",
        direction="backward",
        suffixes=("", "_period"),
    )
    del daily, period_work
    if not forecast_daily.empty:
        forecast_daily = _legacy_asof_tie_winners(forecast_daily)
        forecast_work = forecast_daily.sort_values(
            ["availability_date", "ts_code"],
            kind="mergesort",
        )
        out = pd.merge_asof(
            out.sort_values(["trade_date", "ts_code"], kind="mergesort"),
            forecast_work,
            by="ts_code",
            left_on="trade_date",
            right_on="availability_date",
            direction="backward",
            suffixes=("", "_forecast"),
        )
        del forecast_work
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


def _legacy_asof_tie_winners(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the exact right-row winners used by the legacy PIT join."""

    if frame.empty:
        return frame
    row_position = "__legacy_asof_row_position"
    if row_position in frame.columns:
        raise ValueError(f"reserved column present in asof input: {row_position}")
    working = frame.copy()
    working[row_position] = np.arange(len(working), dtype=np.int64)
    winner_positions: list[int] = []
    for _symbol, group in working.groupby("ts_code", sort=False):
        winners = group.sort_values("availability_date").drop_duplicates(
            subset=["availability_date"],
            keep="last",
        )
        winner_positions.extend(int(value) for value in winners[row_position])
    return working.iloc[winner_positions].drop(columns=[row_position])


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


def _sector_map_from_membership_bytes_v3(
    membership_bytes: bytes,
    *,
    membership_sha256: str,
    as_of: str,
    symbols: Sequence[str],
    non_blocking_absent_symbols: Sequence[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(membership_bytes, bytes) or not membership_bytes:
        raise ValueError("v3 derivation requires non-empty membership bytes")
    expected_sha256 = str(membership_sha256 or "").strip().lower()
    actual_sha256 = hashlib.sha256(membership_bytes).hexdigest()
    if expected_sha256 != actual_sha256:
        raise ValueError("v3 derivation membership SHA mismatch")
    as_of_text = _normalize_fetch_as_of(as_of)
    scope = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    if not scope:
        raise ValueError("v3 derivation requires a non-empty exact symbol scope")
    scope_set = set(scope)
    exceptions = sorted(
        {
            normalize_ts_code(symbol)
            for symbol in non_blocking_absent_symbols
            if normalize_ts_code(symbol)
        }
    )
    if not set(exceptions).issubset(scope_set):
        raise ValueError("v3 derivation exception is outside the symbol scope")
    exception_set = set(exceptions)
    try:
        membership = pd.read_parquet(io.BytesIO(membership_bytes))
    except Exception as exc:
        raise ValueError("v3 derivation membership is unreadable") from exc
    required = {"symbol", "effective_from", "effective_to", "industry"}
    if not required.issubset(membership.columns):
        raise ValueError("v3 derivation membership schema is incomplete")
    working = membership.copy()
    if working[["symbol", "effective_from", "effective_to"]].isna().any(axis=None):
        raise ValueError("v3 derivation membership required field is null")
    working["_symbol"] = working["symbol"].map(normalize_ts_code)
    if working["_symbol"].eq("").any():
        raise ValueError("v3 derivation membership symbol is invalid")

    def exact_dates(series: pd.Series, *, allow_empty: bool) -> tuple[pd.Series, pd.Series]:
        text = series.astype("string").fillna("").str.strip()
        exact = text.str.fullmatch(r"\d{8}", na=False)
        valid = exact | (text.eq("") if allow_empty else False)
        parsed = pd.to_datetime(
            text.where(exact),
            format="%Y%m%d",
            errors="coerce",
        )
        valid &= parsed.notna() | (text.eq("") if allow_empty else False)
        return parsed, valid

    effective_from, valid_from = exact_dates(
        working["effective_from"], allow_empty=False
    )
    effective_to, valid_to = exact_dates(working["effective_to"], allow_empty=True)
    if not valid_from.all() or not valid_to.all():
        raise ValueError("v3 derivation membership interval date is invalid")
    working["_effective_from"] = effective_from
    working["_effective_to"] = effective_to
    as_of_ts = pd.Timestamp(pd.to_datetime(as_of_text, format="%Y%m%d"))
    eligible = working[
        working["_symbol"].isin(scope_set)
        & working["_effective_from"].notna()
        & working["_effective_from"].le(as_of_ts)
    ].copy()
    if eligible.empty:
        raise ValueError("v3 derivation membership has no eligible rows")
    eligible["_active"] = eligible["_effective_to"].isna() | eligible[
        "_effective_to"
    ].ge(as_of_ts)
    selected_rows: list[pd.Series] = []
    expired_fallback_symbols: list[str] = []
    for symbol in scope:
        rows = eligible[eligible["_symbol"] == symbol]
        if rows.empty:
            raise ValueError(f"v3 derivation membership missing scope symbol: {symbol}")
        active = rows[rows["_active"]]
        if active.empty and symbol not in exception_set:
            raise ValueError(
                f"v3 derivation membership has no active interval: {symbol}"
            )
        candidates = active if not active.empty else rows
        if active.empty:
            expired_fallback_symbols.append(str(symbol))
        latest_from = candidates["_effective_from"].max()
        latest = candidates[candidates["_effective_from"].eq(latest_from)]
        industries = {
            str(value).strip()
            for value in latest["industry"].tolist()
            if str(value).strip()
            and str(value).strip().lower() not in {"nan", "none", "<na>"}
        }
        if len(industries) > 1:
            raise ValueError(
                f"v3 derivation membership has conflicting industries: {symbol}"
            )
        selected_rows.append(latest.sort_index().iloc[-1])
    selected = pd.DataFrame(selected_rows)
    sector_map: dict[str, str] = {}
    for row in selected.to_dict("records"):
        value = row.get("industry")
        text = "" if pd.isna(value) else str(value).strip()
        sector_map[str(row["_symbol"])] = (
            "" if text.lower() in {"nan", "none", "<na>"} else text
        )
    evidence = {
        "contract_version": FUNDAMENTAL_DERIVATION_CONTRACT,
        "membership_sha256": actual_sha256,
        "as_of": as_of_text,
        "selection_rule": (
            "latest_active_membership_interval_as_of_else_latest_expired"
        ),
        "selected_symbol_count": int(len(sector_map)),
        "symbol_set_sha256": _symbol_scope_sha256(scope),
        "non_blocking_absent_symbols": exceptions,
        "expired_fallback_symbol_count": int(len(expired_fallback_symbols)),
        "expired_fallback_symbols_sha256": hashlib.sha256(
            "\n".join(expired_fallback_symbols).encode("utf-8")
        ).hexdigest(),
        "sector_map_sha256": canonical_json_sha256(sector_map),
    }
    return sector_map, evidence


def rederive_fundamental_tables_v3(
    raw_tables: Mapping[str, pd.DataFrame],
    *,
    membership_bytes: bytes,
    membership_sha256: str,
    as_of: str,
    symbols: Sequence[str],
    non_blocking_absent_symbols: Sequence[str],
    run_id: str,
    source: str,
    derivation_timestamp: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Deterministically derive mart tables from accepted raw and bound membership."""

    if not str(run_id or "").strip():
        raise ValueError("v3 derivation requires run_id")
    if not str(source or "").strip():
        raise ValueError("v3 derivation requires source")
    timestamp = str(derivation_timestamp or "").strip()
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if not timestamp or pd.isna(parsed_timestamp):
        raise ValueError("v3 derivation timestamp is invalid")
    normalized_scope = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    if not normalized_scope:
        raise ValueError("v3 derivation requires a non-empty exact symbol scope")
    scope_set = set(normalized_scope)
    for table in SOURCE_TABLES:
        frame = raw_tables.get(table, pd.DataFrame())
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"v3 derivation raw table is not a DataFrame: {table}")
        if frame.empty:
            continue
        if "ts_code" not in frame.columns:
            raise ValueError(f"v3 derivation raw table has no ts_code: {table}")
        raw_symbols = {
            normalize_ts_code(value)
            for value in frame["ts_code"].tolist()
            if normalize_ts_code(value)
        }
        if not raw_symbols.issubset(scope_set):
            raise ValueError(f"v3 derivation raw table is outside scope: {table}")
    sector_map, membership_evidence = _sector_map_from_membership_bytes_v3(
        membership_bytes,
        membership_sha256=membership_sha256,
        as_of=as_of,
        symbols=symbols,
        non_blocking_absent_symbols=non_blocking_absent_symbols,
    )
    if set(sector_map) != scope_set:
        raise ValueError("v3 derivation sector map does not match exact scope")
    period, quarantine = derive_fundamental_period(
        raw_tables,
        run_id=run_id,
        source=source,
        derivation_timestamp=timestamp,
    )
    daily = build_fundamental_daily(
        period,
        raw_tables.get("daily_basic"),
        raw_tables.get("forecast"),
        run_id=run_id,
        source=source,
        sector_map=sector_map,
        derivation_timestamp=timestamp,
    )
    tables = {
        "fundamental_period": period,
        "fundamental_daily": daily,
        "fundamental_quarantine": quarantine,
    }
    evidence = {
        **membership_evidence,
        "derivation_timestamp": timestamp,
        "run_id": str(run_id),
        "source": str(source),
        "raw_table_fingerprints": {
            table: frame_fingerprint(raw_tables.get(table, pd.DataFrame()))
            for table in SOURCE_TABLES
        },
        "output_frame_fingerprints": {
            table: frame_fingerprint(frame) for table, frame in tables.items()
        },
    }
    return tables, evidence


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
    _derived_tables_v3: Mapping[str, pd.DataFrame] | None = None,
    _provider_evidence_bytes: Mapping[str, bytes] | None = None,
    _published_pointer_out: dict[str, Any] | None = None,
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

    if _derived_tables_v3 is None:
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
    else:
        if set(_derived_tables_v3) != set(FUNDAMENTAL_TABLES):
            raise ValueError("v3 derived table bundle is incomplete")
        incoming_period = _derived_tables_v3["fundamental_period"]
        incoming_daily = _derived_tables_v3["fundamental_daily"]
        incoming_quarantine = _derived_tables_v3["fundamental_quarantine"]
    predecessor_pointer_sha256 = fundamental_pointer_sha256(data_dir)
    prior_pointer = load_fundamental_pointer(data_dir)
    existing = {
        table_name: _read_existing_fundamental_table(data_dir, table_name)
        for table_name in FUNDAMENTAL_TABLES
    }
    authoritative_isolated_replace = bool(
        _derived_tables_v3 is not None
        and dict(provider_manifest or {}).get("authoritative_full_rebuild") is True
    )
    if authoritative_isolated_replace:
        if source_priority != "tushare_primary":
            raise ValueError(
                "authoritative isolated rebuild requires tushare_primary"
            )
        if prior_pointer is not None or any(
            not frame.empty for frame in existing.values()
        ):
            raise ValueError(
                "authoritative isolated rebuild data root must be empty"
            )
        _validate_authoritative_derived_bundle(_derived_tables_v3)
        period = incoming_period
        daily = incoming_daily
        quarantine = incoming_quarantine
        period_merge = _authoritative_replace_stats(period)
        daily_merge = _authoritative_replace_stats(daily)
        quarantine_merge = _authoritative_replace_stats(quarantine)
    else:
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
        expected_pointer_sha256=predecessor_pointer_sha256,
    )
    if _provider_evidence_bytes is not None:
        if dict(provider_manifest or {}).get("schema_version") != (
            "cn-fundamental-provider-manifest.v4"
        ):
            raise ValueError("provider evidence bytes require a v4 manifest")
        from ..intelligence_v2.sources.tushare.fundamental_v4.storage import (
            capture_provider_evidence_directory,
        )
        from .fundamental_generation import (
            _write_captured_provider_evidence,
        )

        generation_root = data_dir / Path(
            str(pointer["manifest_path"])
        ).parent
        _write_captured_provider_evidence(
            generation_root,
            _provider_evidence_bytes,
        )
        if capture_provider_evidence_directory(
            generation_root / "provider_evidence"
        ) != dict(_provider_evidence_bytes):
            raise ValueError(
                "Fundamental v4 provider evidence readback failed"
            )
    if _published_pointer_out is not None:
        _published_pointer_out.clear()
        _published_pointer_out.update(pointer)
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


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd().resolve(strict=True) / absolute
    absolute = Path(os.path.abspath(absolute))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ValueError(f"{label} is unreadable: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink: {cursor}")
    try:
        before = os.lstat(absolute)
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {absolute}") from exc
    try:
        opened = os.fstat(descriptor)
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != signature:
            raise ValueError(f"{label} changed during open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(absolute)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != signature
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            != signature
        ):
            raise ValueError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_regular_file_sha256(path: Path, *, label: str) -> str:
    """Hash one stable regular file without retaining its bytes in memory."""

    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd().resolve(strict=True) / absolute
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ValueError(f"{label} is unreadable: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink: {cursor}")
    try:
        before = os.lstat(absolute)
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {absolute}") from exc
    try:
        opened = os.fstat(descriptor)
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != signature
        ):
            raise ValueError(f"{label} changed during open")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(absolute)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != signature
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            != signature
        ):
            raise ValueError(f"{label} changed during read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(
        _stable_regular_file_bytes(path, label="fundamental evidence file")
    ).hexdigest()


def _symbol_scope_sha256(symbols: Sequence[str]) -> str:
    normalized = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _canonical_bar_file_evidence(root: Path) -> list[dict[str, Any]]:
    expanded_root = root.expanduser()
    if not expanded_root.is_absolute():
        expanded_root = Path.cwd().resolve(strict=True) / expanded_root
    root = Path(os.path.abspath(expanded_root))
    paths = _canonical_bar_paths(root)
    evidence: list[dict[str, Any]] = []
    for path in paths:
        payload = _stable_regular_file_bytes(
            path,
            label="canonical market bar dataset file",
        )
        evidence.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(len(payload)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if _canonical_bar_paths(root) != paths:
        raise ValueError("canonical market bar dataset changed during read")
    return evidence


def _canonical_bar_paths(root: Path) -> list[Path]:
    """Enumerate a canonical Parquet tree while rejecting every symlink."""

    absolute = root.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd().resolve(strict=True) / absolute
    absolute = Path(os.path.abspath(absolute))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ValueError(
                f"canonical market bar dataset is unreadable: {cursor}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"canonical market bar dataset contains a symlink: {cursor}"
            )
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError("canonical market bar root is not a directory")

    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(
        absolute,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        directory_metadata = os.lstat(directory_path)
        if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
            directory_metadata.st_mode
        ):
            raise ValueError(
                "canonical market bar dataset contains an unsafe directory"
            )
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            child = directory_path / dirname
            child_metadata = os.lstat(child)
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ValueError(
                    f"canonical market bar dataset contains a symlink: {child}"
                )
            if not stat.S_ISDIR(child_metadata.st_mode):
                raise ValueError(
                    "canonical market bar dataset contains an unsafe directory"
                )
        for filename in filenames:
            child = directory_path / filename
            child_metadata = os.lstat(child)
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ValueError(
                    f"canonical market bar dataset contains a symlink: {child}"
                )
            if filename.startswith(".") or child.suffix.lower() != ".parquet":
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise ValueError(
                    "canonical market bar dataset contains an unsafe file"
                )
            paths.append(child)
    paths.sort(key=lambda path: path.relative_to(absolute).as_posix())
    if not paths:
        raise ValueError("canonical market bar dataset has no Parquet files")
    return paths


DAILY_BASIC_COVERAGE_BOUNDARY_PATH = (
    Path("data") / "cn_universe" / "daily_basic_coverage_boundaries.json"
)


def _json_object_without_duplicate_keys(payload: bytes) -> dict[str, Any]:
    def build(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("daily history coverage declaration has duplicate keys")
            result[key] = value
        return result

    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=build)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("daily history coverage declaration is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("daily history coverage declaration must be an object")
    return parsed


def _declared_coverage_intervals(
    scope: set[str],
    *,
    listing_identities: Mapping[str, str],
    listing_dates: Mapping[str, str],
    history_end_dates: Mapping[str, str],
    membership_sha256: str,
    cutoff: str,
) -> dict[str, Any]:
    """Load exact intervals; legacy response-derived floors are not authority."""

    path = DAILY_BASIC_COVERAGE_BOUNDARY_PATH
    if not path.is_absolute():
        path = Path.cwd().resolve(strict=True) / path
    if not path.is_file():
        return {
            "daily_history_coverage_interval_path": "",
            "daily_history_coverage_interval_source_sha256": "",
            "daily_history_coverage_intervals": [],
        }
    raw = _stable_regular_file_bytes(path, label="daily_basic coverage boundaries")
    payload = _json_object_without_duplicate_keys(raw)
    if payload.get("schema_version") != "daily-basic-coverage-intervals.v2":
        legacy_symbols = {
            normalize_ts_code(symbol)
            for symbol in dict(payload.get("coverage_starts", {}) or {})
        }
        if scope.intersection(legacy_symbols):
            raise ValueError(
                "legacy daily_basic coverage starts are not exact provider authority"
            )
        return {
            "daily_history_coverage_interval_path": "",
            "daily_history_coverage_interval_source_sha256": "",
            "daily_history_coverage_intervals": [],
        }
    if set(payload) != {"schema_version", "intervals", "record_sha256"}:
        raise ValueError("daily history coverage declaration shape is invalid")
    record = {
        "schema_version": payload["schema_version"],
        "intervals": payload["intervals"],
    }
    if canonical_json_sha256(record) != str(payload["record_sha256"]).strip().lower():
        raise ValueError("daily history coverage declaration seal is invalid")
    values = payload.get("intervals")
    if not isinstance(values, list):
        raise ValueError("daily history coverage intervals must be a list")
    relevant = [
        row
        for row in values
        if isinstance(row, Mapping) and normalize_ts_code(row.get("symbol")) in scope
    ]
    normalized: list[dict[str, Any]] = []
    for symbol in sorted(scope):
        rows = [
            row
            for row in relevant
            if normalize_ts_code(row.get("symbol")) == symbol
        ]
        validated = _validate_daily_history_coverage_intervals(
            rows,
            symbol=symbol,
            listing_identity=str(listing_identities.get(symbol) or ""),
            listing_start=str(listing_dates.get(symbol) or ""),
            listing_end=str(history_end_dates.get(symbol) or ""),
            listing_source_sha256=membership_sha256,
            cutoff=cutoff,
        )
        normalized.extend(
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in validated
        )
    return {
        "daily_history_coverage_interval_path": str(path),
        "daily_history_coverage_interval_source_sha256": hashlib.sha256(raw).hexdigest(),
        "daily_history_coverage_intervals": sorted(
            normalized, key=lambda row: str(row["interval_id"])
        ),
    }


def _canonical_bar_history_bounds(
    pointer_payload: Mapping[str, Any],
    *,
    symbols: Sequence[str],
    listing_dates: Mapping[str, str],
    history_end_dates: Mapping[str, str],
    daily_start: str,
    as_of: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    table_root_value = str(pointer_payload.get("table_root") or "").strip()
    if not table_root_value:
        raise ValueError("canonical market pointer table_root is missing")
    raw_root = Path(table_root_value).expanduser()
    if not raw_root.is_absolute():
        raw_root = Path.cwd().resolve(strict=True) / raw_root
    root = Path(os.path.abspath(raw_root))
    file_paths = _canonical_bar_paths(root)
    requested_symbols = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    symbol_set = set(requested_symbols)
    start_ts = pd.Timestamp(pd.to_datetime(daily_start, format="%Y%m%d"))
    as_of_ts = pd.Timestamp(pd.to_datetime(as_of, format="%Y%m%d"))
    if start_ts > as_of_ts:
        raise ValueError("canonical daily history window is reversed")
    eligibility_starts: dict[str, pd.Timestamp] = {}
    eligibility_ends: dict[str, pd.Timestamp] = {}
    for symbol in requested_symbols:
        listing_ts = pd.Timestamp(
            pd.to_datetime(str(listing_dates.get(symbol) or ""), format="%Y%m%d")
        )
        history_end_ts = pd.Timestamp(
            pd.to_datetime(str(history_end_dates.get(symbol) or ""), format="%Y%m%d")
        )
        eligibility_starts[symbol] = max(start_ts, listing_ts)
        eligibility_ends[symbol] = min(as_of_ts, history_end_ts)
        if eligibility_starts[symbol] > eligibility_ends[symbol]:
            raise ValueError(
                f"canonical daily history eligibility is reversed: {symbol}"
            )
    first_bounds: dict[str, pd.Timestamp] = {}
    last_bounds: dict[str, pd.Timestamp] = {}
    file_evidence_before: list[dict[str, Any]] = []
    for path in file_paths:
        payload = _stable_regular_file_bytes(
            path,
            label="canonical market bar dataset file",
        )
        file_evidence_before.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(len(payload)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        try:
            bars = pd.read_parquet(
                io.BytesIO(payload),
                columns=["ts_code", "trade_date"],
            )
        except Exception as exc:
            raise ValueError("canonical market bar dataset is unreadable") from exc
        if set(bars.columns) != {"ts_code", "trade_date"}:
            raise ValueError("canonical market bar dataset schema is invalid")
        bars["_symbol"] = bars["ts_code"].map(normalize_ts_code)
        trade_date_text = bars["trade_date"].astype("string").fillna("").str.strip()
        trade_date_exact = trade_date_text.str.fullmatch(r"\d{8}", na=False)
        bars["_trade_date"] = pd.to_datetime(
            trade_date_text.where(trade_date_exact),
            format="%Y%m%d",
            errors="coerce",
        )
        relevant = bars["_symbol"].isin(symbol_set)
        if (relevant & (~trade_date_exact | bars["_trade_date"].isna())).any():
            raise ValueError(
                "canonical market bar dataset contains an invalid trade date"
            )
        scoped = bars.loc[relevant, ["_symbol", "_trade_date"]].copy()
        scoped["_eligibility_start"] = scoped["_symbol"].map(eligibility_starts)
        scoped["_eligibility_end"] = scoped["_symbol"].map(eligibility_ends)
        scoped = scoped[
            scoped["_trade_date"].ge(scoped["_eligibility_start"])
            & scoped["_trade_date"].le(scoped["_eligibility_end"])
        ]
        grouped = scoped.groupby("_symbol", sort=False)["_trade_date"].agg(
            ["min", "max"]
        )
        for symbol, row in grouped.iterrows():
            normalized_symbol = str(symbol)
            first = pd.Timestamp(row["min"])
            last = pd.Timestamp(row["max"])
            first_bounds[normalized_symbol] = min(
                first,
                first_bounds.get(normalized_symbol, first),
            )
            last_bounds[normalized_symbol] = max(
                last,
                last_bounds.get(normalized_symbol, last),
            )
        del payload, bars, scoped, grouped
    file_evidence_after = _canonical_bar_file_evidence(root)
    if file_evidence_after != file_evidence_before:
        raise ValueError("canonical market bar dataset changed during read")
    first_dates: dict[str, str] = {}
    last_dates: dict[str, str] = {}
    for symbol in requested_symbols:
        if symbol not in first_bounds or symbol not in last_bounds:
            raise ValueError(f"canonical market bar bounds missing symbol: {symbol}")
        first = first_bounds[symbol]
        last = last_bounds[symbol]
        if (
            first < eligibility_starts[symbol]
            or last > eligibility_ends[symbol]
            or first > last
        ):
            raise ValueError(f"canonical market bar bounds are invalid: {symbol}")
        first_dates[symbol] = first.strftime("%Y%m%d")
        last_dates[symbol] = last.strftime("%Y%m%d")
    bounds_lines = [
        f"{symbol}|{first_dates[symbol]}|{last_dates[symbol]}"
        for symbol in requested_symbols
    ]
    return first_dates, last_dates, {
        "canonical_bar_table_root": str(root),
        "canonical_bar_file_count": int(len(file_evidence_after)),
        "canonical_bar_files_sha256": canonical_json_sha256(file_evidence_after),
        "canonical_bar_bounds_sha256": hashlib.sha256(
            "\n".join(bounds_lines).encode("utf-8")
        ).hexdigest(),
        "canonical_bar_daily_start": daily_start,
        "canonical_bar_as_of": as_of,
    }


def build_canonical_scope_evidence(
    symbols: Sequence[str],
    *,
    canonical_path: str | Path,
    market_pointer_path: str | Path,
    membership_path: str | Path,
    as_of: str,
    daily_start: str | None = None,
) -> dict[str, Any]:
    """Bind a provider rebuild to the exact canonical scope source and symbols."""

    resolved = Path(canonical_path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd().resolve(strict=True) / resolved
    scope_bytes = _stable_regular_file_bytes(
        resolved,
        label="canonical scope evidence",
    )
    pointer_path = Path(market_pointer_path).expanduser()
    if not pointer_path.is_absolute():
        pointer_path = Path.cwd().resolve(strict=True) / pointer_path
    pointer_bytes = _stable_regular_file_bytes(
        pointer_path,
        label="canonical market pointer",
    )
    resolved_membership_path = Path(membership_path).expanduser()
    if not resolved_membership_path.is_absolute():
        resolved_membership_path = (
            Path.cwd().resolve(strict=True) / resolved_membership_path
        )
    membership_bytes = _stable_regular_file_bytes(
        resolved_membership_path,
        label="canonical PIT membership",
    )
    normalized = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    if not normalized:
        raise ValueError("canonical scope evidence requires at least one symbol")
    try:
        scope_payload = json.loads(scope_bytes.decode("utf-8"))
        pointer_payload = json.loads(pointer_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError("canonical scope evidence JSON is invalid") from exc
    if not isinstance(scope_payload, Mapping) or not isinstance(pointer_payload, Mapping):
        raise ValueError("canonical scope evidence must contain JSON objects")
    declared_scope = sorted(
        {
            normalize_ts_code(symbol)
            for symbol in list(scope_payload.get("full_a", []) or [])
            if normalize_ts_code(symbol)
        }
    )
    if declared_scope != normalized:
        raise ValueError("canonical full_a scope does not match requested symbols")
    scope_sha256 = _symbol_scope_sha256(normalized)
    coverage = dict(pointer_payload.get("coverage", {}) or {})
    if (
        int(coverage.get("expected_scope_count", -1)) != len(normalized)
        or str(coverage.get("expected_scope_sha256") or "").strip().lower()
        != scope_sha256
    ):
        raise ValueError("canonical market pointer scope binding mismatch")
    pointer_membership_path = Path(
        str(coverage.get("pit_membership_path") or "")
    ).expanduser()
    if not pointer_membership_path.is_absolute():
        pointer_membership_path = (
            Path.cwd().resolve(strict=True) / pointer_membership_path
        )
    pointer_membership_sha256 = str(
        coverage.get("pit_membership_sha256") or ""
    ).strip().lower()
    if (
        pointer_membership_path.resolve(strict=True)
        != resolved_membership_path.resolve(strict=True)
        or len(pointer_membership_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in pointer_membership_sha256
        )
        or hashlib.sha256(membership_bytes).hexdigest()
        != pointer_membership_sha256
    ):
        raise ValueError("canonical market pointer PIT membership binding mismatch")
    requested_as_of = _normalize_fetch_as_of(as_of)
    requested_daily_start = _normalize_fetch_as_of(daily_start or "19000101")
    if requested_daily_start > requested_as_of:
        raise ValueError("canonical daily history start is after as_of")
    complete_trade_date = str(
        pointer_payload.get("latest_complete_trade_date") or ""
    ).strip()
    if complete_trade_date != requested_as_of:
        raise ValueError(
            "canonical market pointer latest_complete_trade_date does not match as_of"
        )
    normalized_set = set(normalized)
    non_blocking_absent = sorted(
        {
            normalize_ts_code(symbol)
            for symbol in list(coverage.get("non_blocking_absent_symbols", []) or [])
            if normalize_ts_code(symbol) in normalized_set
        }
    )
    non_blocking_absent_set = set(non_blocking_absent)
    try:
        membership = pd.read_parquet(io.BytesIO(membership_bytes))
    except Exception as exc:
        raise ValueError("canonical PIT membership is unreadable") from exc
    required_membership_columns = {
        "symbol",
        "list_date",
        "effective_from",
        "effective_to",
    }
    if not required_membership_columns.issubset(membership.columns):
        raise ValueError("canonical PIT membership schema is incomplete")
    membership = membership.copy()
    membership["_symbol"] = membership["symbol"].map(normalize_ts_code)
    relevant_membership = membership["_symbol"].isin(normalized_set)
    if membership.loc[
        relevant_membership,
        ["list_date", "effective_from", "effective_to"],
    ].isna().any(axis=None):
        raise ValueError("canonical PIT membership required date is null")
    list_date_text = membership["list_date"].astype("string").fillna("").str.strip()
    list_date_exact = list_date_text.str.fullmatch(r"\d{8}", na=False)
    membership["_list_date"] = pd.to_datetime(
        list_date_text.where(list_date_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_list_date = relevant_membership & (
        ~list_date_exact | membership["_list_date"].isna()
    )
    if invalid_list_date.any():
        raise ValueError("canonical PIT membership list_date is invalid")
    effective_from_text = (
        membership["effective_from"].astype("string").fillna("").str.strip()
    )
    effective_from_exact = effective_from_text.str.fullmatch(r"\d{8}", na=False)
    membership["_effective_from"] = pd.to_datetime(
        effective_from_text.where(effective_from_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_effective_from = relevant_membership & (
        ~effective_from_exact | membership["_effective_from"].isna()
    )
    if invalid_effective_from.any():
        raise ValueError("canonical PIT membership effective_from is invalid")
    effective_to_text = (
        membership["effective_to"].astype("string").fillna("").str.strip()
    )
    effective_to_open = effective_to_text.eq("")
    effective_to_exact = effective_to_text.str.fullmatch(r"\d{8}", na=False)
    membership["_effective_to"] = pd.to_datetime(
        effective_to_text.where(~effective_to_open & effective_to_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_effective_to = relevant_membership & (
        ~effective_to_open
        & (~effective_to_exact | membership["_effective_to"].isna())
    )
    if invalid_effective_to.any():
        raise ValueError("canonical PIT membership effective_to is invalid")
    membership["_effective_to_open"] = effective_to_open
    if "delist_date" in membership.columns:
        delist_date_text = (
            membership["delist_date"].astype("string").fillna("").str.strip()
        )
    else:
        delist_date_text = pd.Series("", index=membership.index, dtype="string")
    delist_date_empty = delist_date_text.eq("")
    delist_date_exact = delist_date_text.str.fullmatch(r"\d{8}", na=False)
    membership["_delist_date"] = pd.to_datetime(
        delist_date_text.where(~delist_date_empty & delist_date_exact),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_delist_date = relevant_membership & (
        ~delist_date_empty
        & (~delist_date_exact | membership["_delist_date"].isna())
    )
    if invalid_delist_date.any():
        raise ValueError("canonical PIT membership delist_date is invalid")
    as_of_ts = pd.Timestamp(pd.to_datetime(requested_as_of, format="%Y%m%d"))
    listing_dates: dict[str, str] = {}
    history_end_dates: dict[str, str] = {}
    listing_identities: dict[str, str] = {}
    membership_sha256 = hashlib.sha256(membership_bytes).hexdigest()
    for symbol in normalized:
        rows = membership[
            (membership["_symbol"] == symbol)
            & membership["_effective_from"].notna()
            & membership["_effective_from"].le(as_of_ts)
        ].sort_values("_effective_from")
        if rows.empty:
            raise ValueError(f"canonical PIT membership missing symbol: {symbol}")
        active_rows = rows[
            rows["_effective_to_open"]
            | rows["_effective_to"].ge(as_of_ts)
        ]
        selected_active = not active_rows.empty
        if selected_active:
            row = active_rows.iloc[-1]
        elif symbol in non_blocking_absent_set:
            row = rows.iloc[-1]
        else:
            raise ValueError(
                f"canonical PIT membership interval is not active as_of: {symbol}"
            )
        list_date = row["_list_date"].strftime("%Y%m%d")
        if list_date > requested_as_of:
            raise ValueError(f"canonical PIT membership has future list date: {symbol}")
        end_candidates: list[str] = []
        for field in ("_effective_to", "_delist_date"):
            end_value = row.get(field)
            if not pd.isna(end_value):
                end_date = pd.Timestamp(end_value).strftime("%Y%m%d")
                if end_date <= requested_as_of:
                    end_candidates.append(end_date)
        listing_dates[symbol] = list_date
        if (
            selected_active
            and any(candidate < requested_as_of for candidate in end_candidates)
            and symbol not in non_blocking_absent_set
        ):
            raise ValueError(
                f"canonical PIT membership active interval conflicts with delist date: {symbol}"
            )
        if not selected_active and not end_candidates:
            raise ValueError(
                f"canonical PIT membership expired interval has no end date: {symbol}"
            )
        history_end = max(end_candidates) if end_candidates else requested_as_of
        effective_from = row["_effective_from"].strftime("%Y%m%d")
        if history_end < list_date or history_end < effective_from:
            raise ValueError(
                f"canonical PIT membership date order is invalid: {symbol}"
            )
        history_end_dates[symbol] = history_end
        listing_identities[symbol] = _listing_identity_sha256(
            symbol=symbol,
            listing_date=list_date,
            effective_from=effective_from,
            history_end=history_end,
            membership_sha256=membership_sha256,
        )
    canonical_bar_first_dates, canonical_bar_last_dates, bar_evidence = (
        _canonical_bar_history_bounds(
            pointer_payload,
            symbols=normalized,
            listing_dates=listing_dates,
            history_end_dates=history_end_dates,
            daily_start=requested_daily_start,
            as_of=requested_as_of,
        )
    )
    eligibility_lines = [
        "|".join(
            (
                symbol,
                listing_dates[symbol],
                history_end_dates[symbol],
                canonical_bar_first_dates[symbol],
                canonical_bar_last_dates[symbol],
            )
        )
        for symbol in normalized
    ]
    return {
        "canonical_path": str(resolved),
        "canonical_file_sha256": hashlib.sha256(scope_bytes).hexdigest(),
        "canonical_market_pointer_path": str(pointer_path),
        "canonical_market_pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "canonical_market_snapshot_id": str(pointer_payload.get("snapshot_id") or ""),
        "canonical_market_trade_date": complete_trade_date,
        "canonical_membership_path": str(resolved_membership_path),
        "canonical_membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "symbol_count": int(len(normalized)),
        "symbol_set_sha256": scope_sha256,
        "listing_dates": listing_dates,
        "history_end_dates": history_end_dates,
        "listing_identities": listing_identities,
        "canonical_bar_first_dates": canonical_bar_first_dates,
        "canonical_bar_last_dates": canonical_bar_last_dates,
        **bar_evidence,
        "history_eligibility_sha256": hashlib.sha256(
            "\n".join(eligibility_lines).encode("utf-8")
        ).hexdigest(),
        "non_blocking_absent_symbols": non_blocking_absent,
        **_declared_coverage_intervals(
            normalized_set,
            listing_identities=listing_identities,
            listing_dates=listing_dates,
            history_end_dates=history_end_dates,
            membership_sha256=membership_sha256,
            cutoff=requested_as_of,
        ),
    }


def _validate_canonical_scope_evidence(
    evidence: Mapping[str, Any],
    symbols: Sequence[str],
) -> dict[str, Any]:
    required = {
        "canonical_path",
        "canonical_file_sha256",
        "canonical_market_pointer_path",
        "canonical_market_pointer_sha256",
        "canonical_market_snapshot_id",
        "canonical_market_trade_date",
        "canonical_membership_path",
        "canonical_membership_sha256",
        "symbol_count",
        "symbol_set_sha256",
        "listing_dates",
        "history_end_dates",
        "listing_identities",
        "canonical_bar_first_dates",
        "canonical_bar_last_dates",
        "canonical_bar_table_root",
        "canonical_bar_file_count",
        "canonical_bar_files_sha256",
        "canonical_bar_bounds_sha256",
        "canonical_bar_daily_start",
        "canonical_bar_as_of",
        "history_eligibility_sha256",
        "non_blocking_absent_symbols",
        "daily_history_coverage_interval_path",
        "daily_history_coverage_interval_source_sha256",
        "daily_history_coverage_intervals",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError("canonical scope evidence missing fields: " + ",".join(missing))
    normalized = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    rebuilt = build_canonical_scope_evidence(
        normalized,
        canonical_path=str(evidence.get("canonical_path") or ""),
        market_pointer_path=str(
            evidence.get("canonical_market_pointer_path") or ""
        ),
        membership_path=str(evidence.get("canonical_membership_path") or ""),
        as_of=str(evidence.get("canonical_market_trade_date") or ""),
        daily_start=str(evidence.get("canonical_bar_daily_start") or ""),
    )
    if _canonical_mapping_sha256(rebuilt) != _canonical_mapping_sha256(evidence):
        raise ValueError("canonical scope evidence changed after binding")
    return rebuilt


def _normalize_fetch_as_of(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return _now_utc().strftime("%Y%m%d")
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) != 8:
        raise ValueError("fundamental fetch as_of must resolve to YYYYMMDD")
    parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed) or pd.Timestamp(parsed).strftime("%Y%m%d") != digits:
        raise ValueError("fundamental fetch as_of must be a valid YYYYMMDD date")
    return digits


def _strict_pit_cutoff(
    frame: pd.DataFrame,
    *,
    table: str,
    symbol: str,
    as_of: str,
) -> tuple[pd.DataFrame, dict[str, int], str]:
    received = int(len(frame))
    stats: dict[str, int] = {
        "rows_received": received,
        "rows": 0,
        "rows_hard_invalid": 0,
        "rows_filtered_future": 0,
        "rows_filtered_missing_availability": 0,
        "rows_filtered_core_values": 0,
        "rows_deduplicated": 0,
        "rows_discarded_request_malformed": 0,
        "rows_hard_invalid_schema": 0,
        "rows_hard_invalid_symbol": 0,
        "rows_hard_invalid_availability_date": 0,
        "rows_hard_invalid_end_date": 0,
        "rows_hard_invalid_end_after_availability": 0,
        "rows_hard_invalid_core_numeric": 0,
    }
    if frame.empty:
        return frame.copy(), stats, ""

    def malformed(
        reason: str,
        hard_counts: Mapping[str, int],
    ) -> tuple[pd.DataFrame, dict[str, int], str]:
        result = dict(stats)
        hard_total = sum(int(value) for value in hard_counts.values())
        result.update({key: int(value) for key, value in hard_counts.items()})
        result["rows_hard_invalid"] = hard_total
        result["rows_discarded_request_malformed"] = received - hard_total
        return pd.DataFrame(), result, reason

    if "ts_code" not in frame.columns:
        return malformed(
            "missing_ts_code",
            {"rows_hard_invalid_schema": received},
        )
    missing_required = sorted(SOURCE_REQUIRED_COLUMNS[table].difference(frame.columns))
    if missing_required:
        return malformed(
            "missing_required_columns:" + ",".join(missing_required),
            {"rows_hard_invalid_schema": received},
        )
    requested_symbol = normalize_ts_code(symbol)
    returned_symbols = frame["ts_code"].map(normalize_ts_code)
    if returned_symbols.eq("").any() or not returned_symbols.eq(requested_symbol).all():
        return malformed(
            "response_symbol_scope_mismatch",
            {"rows_hard_invalid_symbol": received},
        )
    cutoff_columns = (
        ("trade_date",)
        if table == "daily_basic"
        else tuple(column for column in ("ann_date", "f_ann_date") if column in frame.columns)
    )
    if table == "daily_basic" and "trade_date" not in frame.columns:
        return malformed(
            "missing_trade_date",
            {"rows_hard_invalid_schema": received},
        )
    if table != "daily_basic" and not cutoff_columns:
        return malformed(
            "missing_availability_date_columns",
            {"rows_hard_invalid_schema": received},
        )
    cutoff = pd.Timestamp(pd.to_datetime(as_of, format="%Y%m%d"))
    hard_unclassified = pd.Series(True, index=frame.index, dtype=bool)
    hard_masks: dict[str, pd.Series] = {
        "rows_hard_invalid_availability_date": pd.Series(
            False, index=frame.index, dtype=bool
        ),
        "rows_hard_invalid_end_date": pd.Series(
            False, index=frame.index, dtype=bool
        ),
        "rows_hard_invalid_end_after_availability": pd.Series(
            False, index=frame.index, dtype=bool
        ),
        "rows_hard_invalid_core_numeric": pd.Series(
            False, index=frame.index, dtype=bool
        ),
    }
    parsed_cutoffs: dict[str, pd.Series] = {}
    populated_cutoffs: dict[str, pd.Series] = {}

    def exact_date_series(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        raw = values.astype("string").fillna("").str.strip()
        exact_yyyymmdd = raw.str.fullmatch(r"\d{8}", na=False)
        parsed = pd.to_datetime(
            raw.where(exact_yyyymmdd),
            format="%Y%m%d",
            errors="coerce",
        )
        populated = raw.ne("")
        invalid = populated & (~exact_yyyymmdd | parsed.isna())
        return parsed, populated, invalid

    for column in cutoff_columns:
        parsed, populated, invalid = exact_date_series(frame[column])
        parsed_cutoffs[column] = parsed
        populated_cutoffs[column] = populated
        assigned = invalid & hard_unclassified
        hard_masks["rows_hard_invalid_availability_date"] |= assigned
        hard_unclassified &= ~assigned
    if table == "daily_basic":
        missing_trade_date = ~populated_cutoffs["trade_date"] & hard_unclassified
        hard_masks["rows_hard_invalid_availability_date"] |= missing_trade_date
        hard_unclassified &= ~missing_trade_date
        selected_availability = parsed_cutoffs["trade_date"]
        missing_availability = pd.Series(False, index=frame.index, dtype=bool)
    else:
        selected_availability = pd.Series(
            pd.NaT,
            index=frame.index,
            dtype="datetime64[ns]",
        )
        for column in ("f_ann_date", "ann_date"):
            if column in parsed_cutoffs:
                selected_availability = selected_availability.fillna(
                    parsed_cutoffs[column]
                )
        missing_availability = selected_availability.isna() & hard_unclassified

    end_date = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if table != "daily_basic":
        end_date, end_present, invalid_end = exact_date_series(frame["end_date"])
        bad_end = (invalid_end | ~end_present | end_date.isna()) & hard_unclassified
        hard_masks["rows_hard_invalid_end_date"] |= bad_end
        hard_unclassified &= ~bad_end
        if table in FINANCIAL_SOURCE_TABLES:
            after_availability = (
                end_date.notna()
                & selected_availability.notna()
                & end_date.gt(selected_availability)
                & hard_unclassified
            )
            hard_masks["rows_hard_invalid_end_after_availability"] |= (
                after_availability
            )
            hard_unclassified &= ~after_availability

    numeric_core_columns = {
        "fina_indicator": (
            "roe_dt",
            "roe",
            "roa",
            "debt_to_assets",
            "netprofit_yoy",
        ),
        "income": ("n_income_attr_p", "n_income"),
        "balancesheet": ("total_assets", "total_liab"),
        "cashflow": ("free_cashflow", "n_cashflow_act"),
        "daily_basic": ("total_mv",),
        "forecast": (
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            "last_parent_net",
        ),
    }[table]
    numeric: dict[str, pd.Series] = {}
    invalid_core_numeric = pd.Series(False, index=frame.index, dtype=bool)
    for column in numeric_core_columns:
        raw = frame[column].astype("string").fillna("").str.strip()
        populated = raw.ne("")
        parsed = pd.to_numeric(raw.where(populated), errors="coerce")
        finite = pd.Series(
            np.isfinite(parsed.to_numpy(dtype=float, na_value=np.nan)),
            index=frame.index,
        )
        invalid_core_numeric |= populated & (~parsed.notna() | ~finite)
        numeric[column] = parsed.where(finite)
    assigned_core = invalid_core_numeric & hard_unclassified
    hard_masks["rows_hard_invalid_core_numeric"] |= assigned_core
    hard_unclassified &= ~assigned_core

    hard_counts = {name: int(mask.sum()) for name, mask in hard_masks.items()}
    if any(hard_counts.values()):
        reason_by_counter = {
            "rows_hard_invalid_availability_date": "invalid_availability_date",
            "rows_hard_invalid_end_date": "invalid_end_date",
            "rows_hard_invalid_end_after_availability": "end_after_availability",
            "rows_hard_invalid_core_numeric": "invalid_core_values",
        }
        reason = next(
            reason_by_counter[name]
            for name in reason_by_counter
            if hard_counts[name]
        )
        return malformed(reason, hard_counts)

    future_mask = selected_availability.gt(cutoff).fillna(False)
    missing_mask = missing_availability & ~future_mask
    remaining = ~(future_mask | missing_mask)
    if table in {"fina_indicator", "income", "cashflow"}:
        usable_core = pd.concat(
            [numeric[column].notna() for column in numeric_core_columns],
            axis=1,
        ).any(axis=1)
    elif table == "balancesheet":
        usable_core = (
            numeric["total_assets"].notna()
            & numeric["total_liab"].notna()
            & numeric["total_assets"].gt(0)
        )
    elif table == "daily_basic":
        usable_core = numeric["total_mv"].notna() & numeric["total_mv"].gt(0)
    else:
        p_mid = pd.concat(
            [numeric["p_change_min"], numeric["p_change_max"]], axis=1
        ).mean(axis=1, skipna=True)
        net_mid = pd.concat(
            [numeric["net_profit_min"], numeric["net_profit_max"]], axis=1
        ).mean(axis=1, skipna=True)
        last_parent = numeric["last_parent_net"]
        usable_core = p_mid.notna() | (
            net_mid.notna() & last_parent.notna() & last_parent.abs().gt(0)
        )
    filtered_core = remaining & ~usable_core
    accepted = frame.loc[remaining & usable_core].copy()
    stats["rows_filtered_future"] = int(future_mask.sum())
    stats["rows_filtered_missing_availability"] = int(missing_mask.sum())
    stats["rows_filtered_core_values"] = int(filtered_core.sum())
    sort_columns = [
        column
        for column in (
            "ts_code",
            "trade_date",
            "ann_date",
            "f_ann_date",
            "end_date",
            "update_flag",
        )
        if column in accepted.columns
    ]
    if sort_columns:
        accepted = accepted.sort_values(
            sort_columns,
            kind="mergesort",
            key=lambda values: values.astype("string"),
        )
    before_dedup = len(accepted)
    accepted = accepted.drop_duplicates()
    stats["rows_deduplicated"] = int(before_dedup - len(accepted))
    stats["rows"] = int(len(accepted))
    return accepted.reset_index(drop=True), stats, ""


def _zero_request_outcome_accounting() -> dict[str, int]:
    return {
        "rows_received": 0,
        "rows": 0,
        "rows_hard_invalid": 0,
        "rows_filtered_future": 0,
        "rows_filtered_missing_availability": 0,
        "rows_filtered_core_values": 0,
        "rows_deduplicated": 0,
        "rows_discarded_request_malformed": 0,
        "rows_hard_invalid_schema": 0,
        "rows_hard_invalid_symbol": 0,
        "rows_hard_invalid_availability_date": 0,
        "rows_hard_invalid_end_date": 0,
        "rows_hard_invalid_end_after_availability": 0,
        "rows_hard_invalid_core_numeric": 0,
    }


class _RequestRateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        sleep_fn: Callable[[float], None],
        monotonic_fn: Callable[[], float],
    ) -> None:
        rate = float(requests_per_second or 0.0)
        if rate < 0:
            raise ValueError("requests_per_second must be non-negative")
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._lock = threading.Lock()
        self._last_started = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = self._monotonic()
            remaining = self._interval - (now - self._last_started)
            if self._last_started and remaining > 0:
                self._sleep(remaining)
            self._last_started = self._monotonic()


_DAILY_HISTORY_OUTCOME_FIELDS = (
    "expected_history_start",
    "expected_history_end",
    "evaluated_history_end",
    "observed_history_start",
    "observed_history_end",
    "observed_history_rows",
    "minimum_history_rows",
    "expected_history_months",
    "observed_history_months",
    "monthly_history_coverage_ratio",
    "max_consecutive_missing_months",
    "coverage_interval_refs",
    "coverage_reason_counts",
    "coverage_blocker_codes",
    "history_start_complete",
    "history_end_complete",
    "history_density_complete",
    "history_monthly_complete",
    "history_boundary_tolerance_days",
    "history_complete",
    "history_exception_evidence_bound",
)


def _attach_daily_history_coverage(
    symbols: Sequence[str],
    outcomes: Sequence[Mapping[str, Any]],
    tables: Mapping[str, pd.DataFrame],
    *,
    daily_start: str,
    as_of: str,
    scope_evidence: Mapping[str, Any] | None,
    policy: FundamentalEndpointAuditPolicy,
) -> list[dict[str, Any]]:
    """Replay daily-history coverage against the bound canonical bar window."""

    evidence = dict(scope_evidence or {})
    listing_dates = dict(evidence.get("listing_dates", {}) or {})
    history_end_dates = dict(evidence.get("history_end_dates", {}) or {})
    listing_identities = dict(evidence.get("listing_identities", {}) or {})
    coverage_intervals = list(
        evidence.get("daily_history_coverage_intervals", []) or []
    )
    bar_first_dates = dict(evidence.get("canonical_bar_first_dates", {}) or {})
    bar_last_dates = dict(evidence.get("canonical_bar_last_dates", {}) or {})
    by_key = {
        (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        ): dict(outcome)
        for outcome in outcomes
    }
    frame = tables.get("daily_basic", pd.DataFrame())
    dates_by_symbol: dict[str, pd.Series] = {}
    if not frame.empty:
        if "ts_code" not in frame.columns or "trade_date" not in frame.columns:
            raise ValueError("daily_basic table is missing history columns")
        working = frame.loc[:, ["ts_code", "trade_date"]].copy()
        working["_symbol"] = working["ts_code"].map(normalize_ts_code)
        working["_trade_date"] = pd.to_datetime(
            working["trade_date"].astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        if working["_trade_date"].isna().any():
            raise ValueError("daily_basic table contains an invalid trade date")
        dates_by_symbol = {
            str(symbol): group["_trade_date"]
            for symbol, group in working.groupby("_symbol", sort=False)
            if str(symbol)
        }
    authoritative = bool(
        listing_dates and history_end_dates and bar_first_dates and bar_last_dates
    )
    normalized_symbols = [
        normalize_ts_code(symbol)
        for symbol in symbols
        if normalize_ts_code(symbol)
    ]
    for symbol in normalized_symbols:
        outcome = by_key.get((symbol, "daily_basic"))
        if outcome is None:
            continue
        for field in _DAILY_HISTORY_OUTCOME_FIELDS:
            outcome.pop(field, None)
        if str(outcome.get("status") or "") != "success":
            continue
        dates = dates_by_symbol.get(symbol, pd.Series(dtype="datetime64[ns]"))
        if authoritative:
            required = {
                "listing_date": str(listing_dates.get(symbol) or ""),
                "history_end": str(history_end_dates.get(symbol) or ""),
                "bar_first": str(bar_first_dates.get(symbol) or ""),
                "bar_last": str(bar_last_dates.get(symbol) or ""),
            }
            if any(not value for value in required.values()):
                raise ValueError(f"canonical daily history bounds missing symbol: {symbol}")
            expected_start = max(
                daily_start,
                required["listing_date"],
                required["bar_first"],
            )
            expected_end = min(
                as_of,
                required["history_end"],
                required["bar_last"],
            )
            if expected_start > expected_end:
                raise ValueError(f"canonical daily history bounds are reversed: {symbol}")
        else:
            expected_start = daily_start
            expected_end = as_of
        symbol_intervals = [
            interval
            for interval in coverage_intervals
            if normalize_ts_code(interval.get("symbol")) == symbol
        ]
        outcome.update(
            _daily_history_coverage_metrics(
                dates,
                expected_start=expected_start,
                expected_end=expected_end,
                allow_tail_gap=False,
                coverage_intervals=symbol_intervals,
                symbol=symbol,
                listing_identity=str(listing_identities.get(symbol) or ""),
                listing_start=str(listing_dates.get(symbol) or expected_start),
                listing_end=str(history_end_dates.get(symbol) or expected_end),
                listing_source_sha256=str(
                    evidence.get("canonical_membership_sha256") or ""
                ),
                cutoff=as_of,
                boundary_tolerance_days=int(
                    policy.daily_history_boundary_tolerance_days
                ),
            )
        )
    return sorted(
        by_key.values(),
        key=lambda item: (str(item.get("symbol")), str(item.get("table"))),
    )


def _attach_financial_coverage(
    symbols: Sequence[str],
    outcomes: Sequence[Mapping[str, Any]],
    tables: Mapping[str, pd.DataFrame],
    *,
    financial_start: str,
    as_of: str,
    scope_evidence: Mapping[str, Any] | None,
    policy: FundamentalEndpointAuditPolicy,
) -> list[dict[str, Any]]:
    """Attach financial-period evidence without widening expectations from rejects."""

    evidence = dict(scope_evidence or {})
    listing_dates = dict(evidence.get("listing_dates", {}) or {})
    history_end_dates = dict(evidence.get("history_end_dates", {}) or {})
    by_key = {
        (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        ): dict(outcome)
        for outcome in outcomes
    }
    periods_by_table_and_symbol: dict[str, dict[str, list[str]]] = {}
    for table in FINANCIAL_SOURCE_TABLES:
        frame = tables.get(table, pd.DataFrame())
        if frame.empty or "ts_code" not in frame.columns or "end_date" not in frame.columns:
            periods_by_table_and_symbol[table] = {}
            continue
        working = frame.loc[:, ["ts_code", "end_date"]].copy()
        working["_symbol"] = working["ts_code"].map(normalize_ts_code)
        periods_by_table_and_symbol[table] = {
            str(symbol): sorted(
                {
                    str(value).strip()
                    for value in group["end_date"].tolist()
                    if str(value).strip()
                }
            )
            for symbol, group in working.groupby("_symbol", sort=False)
            if str(symbol)
        }
    for symbol in symbols:
        normalized_symbol = normalize_ts_code(symbol)
        if listing_dates and history_end_dates:
            baseline = matured_quarter_baseline(
                financial_start,
                str(listing_dates.get(normalized_symbol) or ""),
                str(history_end_dates.get(normalized_symbol) or ""),
                as_of,
            )
            expected = list(baseline)
            expected_set = set(expected)
        else:
            # Non-authoritative fetches have no exact PIT membership bounds and
            # therefore cannot claim a financial-history denominator.
            baseline = []
            expected = []
            expected_set = set()
        for table in FINANCIAL_SOURCE_TABLES:
            outcome = by_key.get((normalized_symbol, table))
            if outcome is None:
                continue
            covered = [
                period
                for period in periods_by_table_and_symbol[table].get(
                    normalized_symbol,
                    [],
                )
                if period in expected_set
            ]
            coverage = build_financial_coverage(
                expected,
                baseline,
                covered,
                minimum_ratio=float(policy.financial_period_min_coverage_ratio),
                max_consecutive_missing_baseline=int(
                    policy.financial_max_consecutive_missing_baseline_periods
                ),
                require_latest_baseline=bool(policy.financial_require_latest_baseline),
            )
            outcome["financial_coverage"] = coverage
            outcome["financial_coverage_passed"] = bool(coverage["passed"])
    return sorted(
        by_key.values(),
        key=lambda item: (str(item.get("symbol")), str(item.get("table"))),
    )


def _build_endpoint_audit(
    symbols: Sequence[str],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    policy: FundamentalEndpointAuditPolicy,
    daily_basic_empty_exception_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    normalized_symbols = sorted(
        {normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)}
    )
    expected_keys = {(symbol, table) for symbol in normalized_symbols for table in SOURCE_TABLES}
    observed_keys: set[tuple[str, str]] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    invalid_status_count = 0
    by_table: dict[str, dict[str, int]] = {
        table: {
            status: 0
            for status in (
                "success",
                "empty",
                "error",
                "malformed",
                "financial_coverage_failed",
                "financial_coverage_not_applicable",
                "financial_coverage_applicable_passed",
            )
        }
        for table in SOURCE_TABLES
    }
    effective_success_keys: set[tuple[str, str]] = set()
    for outcome in outcomes:
        key = (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        )
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        status = str(outcome.get("status") or "")
        if key[1] not in by_table or status not in {
            "success",
            "empty",
            "error",
            "malformed",
        }:
            invalid_status_count += 1
            continue
        try:
            validate_outcome_accounting_v3(
                outcome,
                label=f"endpoint audit outcome {key[0]}/{key[1]}",
            )
        except (TypeError, ValueError):
            invalid_status_count += 1
            continue
        if key[1] in FINANCIAL_SOURCE_TABLES and status == "success":
            coverage = outcome.get("financial_coverage")
            if not isinstance(coverage, Mapping):
                by_table[key[1]]["financial_coverage_failed"] += 1
            elif str(coverage.get("status") or "") == "not_applicable":
                by_table[key[1]]["financial_coverage_not_applicable"] += 1
                by_table[key[1]]["success"] += 1
                effective_success_keys.add(key)
            elif outcome.get("financial_coverage_passed") is True:
                by_table[key[1]]["success"] += 1
                by_table[key[1]]["financial_coverage_applicable_passed"] += 1
                effective_success_keys.add(key)
            else:
                by_table[key[1]]["financial_coverage_failed"] += 1
        else:
            by_table[key[1]][status] += 1
            if key[1] in FINANCIAL_SOURCE_TABLES:
                coverage = outcome.get("financial_coverage")
                if (
                    isinstance(coverage, Mapping)
                    and str(coverage.get("status") or "") == "not_applicable"
                ):
                    by_table[key[1]]["financial_coverage_not_applicable"] += 1
            if status == "success":
                effective_success_keys.add(key)
    missing_keys = expected_keys.difference(observed_keys)
    surplus_keys = observed_keys.difference(expected_keys)
    total_error = sum(counts["error"] for counts in by_table.values())
    total_malformed = sum(counts["malformed"] for counts in by_table.values())
    non_blocking_absent = {
        normalize_ts_code(symbol)
        for symbol in daily_basic_empty_exception_symbols
        if normalize_ts_code(symbol)
    }
    daily_history_incomplete_set: set[str] = set()
    for outcome in outcomes:
        if str(outcome.get("table") or "") != "daily_basic":
            continue
        symbol = normalize_ts_code(outcome.get("symbol"))
        status = str(outcome.get("status") or "")
        if status == "success":
            if outcome.get("history_complete") is not True:
                daily_history_incomplete_set.add(symbol)
            continue
        if status == "empty" and symbol in non_blocking_absent:
            continue
        daily_history_incomplete_set.add(symbol)
    daily_history_incomplete_symbols = sorted(daily_history_incomplete_set)
    daily_history_incomplete = len(daily_history_incomplete_symbols)
    blockers: list[str] = []
    if missing_keys:
        blockers.append("request_outcomes_incomplete")
    if surplus_keys:
        blockers.append("request_outcomes_out_of_scope")
    if duplicate_keys:
        blockers.append("request_outcomes_duplicate")
    if invalid_status_count:
        blockers.append("request_outcomes_invalid_status")
    if total_error > int(policy.max_error_requests):
        blockers.append("provider_error_requests_above_threshold")
    if total_malformed > int(policy.max_malformed_requests):
        blockers.append("provider_malformed_requests_above_threshold")
    if daily_history_incomplete:
        blockers.append("daily_basic_per_symbol_history_incomplete")
    endpoint_payload: dict[str, Any] = {}
    denominator = int(len(normalized_symbols))
    for table, counts in by_table.items():
        success_ratio = counts["success"] / denominator if denominator else 0.0
        minimum = (
            float(policy.daily_basic_min_success_ratio)
            if table == "daily_basic"
            else float(policy.critical_min_success_ratio)
        )
        critical = table in CRITICAL_BASE_TABLES
        endpoint_passed = bool(
            not critical
            or (
                success_ratio >= minimum
                and (table != "daily_basic" or daily_history_incomplete == 0)
            )
        )
        if critical and success_ratio < minimum:
            blockers.append(f"{table}_success_ratio_below_threshold")
        financial_coverage_denominator = (
            counts["financial_coverage_applicable_passed"]
            + counts["financial_coverage_failed"]
            if table in FINANCIAL_SOURCE_TABLES
            else 0
        )
        financial_coverage_passed = (
            counts["financial_coverage_applicable_passed"]
            if table in FINANCIAL_SOURCE_TABLES
            else 0
        )
        endpoint_payload[table] = {
            "request_denominator": denominator,
            "success": int(counts["success"]),
            "empty": int(counts["empty"]),
            "error": int(counts["error"]),
            "malformed": int(counts["malformed"]),
            "financial_coverage_failed": int(counts["financial_coverage_failed"]),
            "financial_coverage_not_applicable": int(
                counts["financial_coverage_not_applicable"]
            ),
            "accounted": int(
                counts["success"]
                + counts["empty"]
                + counts["error"]
                + counts["malformed"]
                + counts["financial_coverage_failed"]
            ),
            "success_ratio": float(success_ratio),
            "critical_base_endpoint": critical,
            "minimum_success_ratio": minimum if critical else 0.0,
            "legitimate_empty_allowed": table == "forecast",
            "passed": endpoint_passed,
            "financial_coverage_denominator": int(financial_coverage_denominator),
            "financial_coverage_passed": int(financial_coverage_passed),
            "financial_coverage_pass_ratio": (
                financial_coverage_passed / financial_coverage_denominator
                if financial_coverage_denominator
                else None
            ),
            "per_symbol_history_incomplete": (
                int(daily_history_incomplete) if table == "daily_basic" else 0
            ),
        }
    all_critical_success = 0
    for symbol in normalized_symbols:
        if all((symbol, table) in effective_success_keys for table in CRITICAL_BASE_TABLES):
            all_critical_success += 1
    expected_requests = len(expected_keys)
    return {
        "schema_version": FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA,
        "passed": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "symbol_denominator": denominator,
        "request_denominator": int(expected_requests),
        "requests_accounted": int(len(observed_keys.intersection(expected_keys))),
        "requests_error": int(total_error),
        "requests_malformed": int(total_malformed),
        "symbols_with_all_critical_success": int(all_critical_success),
        "symbols_with_all_critical_success_ratio": (
            all_critical_success / denominator if denominator else 0.0
        ),
        "forecast_empty_is_legitimate": True,
        "daily_basic_history_incomplete_symbols": daily_history_incomplete_symbols,
        "daily_basic_history_exception_symbols": sorted(non_blocking_absent),
        "policy": {
            "critical_min_success_ratio": float(policy.critical_min_success_ratio),
            "daily_basic_min_success_ratio": float(policy.daily_basic_min_success_ratio),
            "financial_period_min_coverage_ratio": float(
                policy.financial_period_min_coverage_ratio
            ),
            "financial_max_consecutive_missing_baseline_periods": int(
                policy.financial_max_consecutive_missing_baseline_periods
            ),
            "financial_require_latest_baseline": bool(
                policy.financial_require_latest_baseline
            ),
            "daily_history_boundary_tolerance_days": int(
                policy.daily_history_boundary_tolerance_days
            ),
            "max_error_requests": int(policy.max_error_requests),
            "max_malformed_requests": int(policy.max_malformed_requests),
        },
        "endpoints": endpoint_payload,
    }


def _active_daily_tail_gap_exceptions(
    scope_evidence: Mapping[str, Any] | None,
    *,
    as_of: str,
) -> list[str]:
    scope = dict(scope_evidence or {})
    history_end_dates = dict(scope.get("history_end_dates", {}) or {})
    return sorted(
        {
            normalize_ts_code(symbol)
            for symbol in list(scope.get("non_blocking_absent_symbols", []) or [])
            if normalize_ts_code(symbol)
            and str(history_end_dates.get(normalize_ts_code(symbol)) or "") == as_of
        }
    )


def _canonical_json_file_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() and path.is_symlink():
        raise FundamentalFetchCheckpointError("checkpoint JSON target is a symlink")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    encoded = _canonical_json_file_bytes(payload)
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_checkpoint_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise FundamentalFetchCheckpointError("checkpoint Parquet target is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b") as handle:
            descriptor = -1
            frame.to_parquet(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_checkpoint_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_checkpoint_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_checkpoint_root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd().resolve(strict=True) / raw
    if ".." in absolute.parts:
        raise FundamentalFetchCheckpointError("checkpoint root contains parent traversal")
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, mode=0o700)
                metadata = os.lstat(cursor)
            except FileExistsError:
                metadata = os.lstat(cursor)
        except OSError as exc:
            raise FundamentalFetchCheckpointError("checkpoint root is unsafe") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FundamentalFetchCheckpointError("checkpoint root is unsafe")
    return cursor.resolve(strict=True)


@contextmanager
def _checkpoint_root_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".checkpoint.lock"
    if lock_path.exists() and lock_path.is_symlink():
        raise FundamentalFetchCheckpointError("checkpoint lock is a symlink")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FundamentalFetchCheckpointError("checkpoint lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _checkpoint_path(root: Path, relative: str) -> Path:
    relative_path = Path(str(relative or ""))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FundamentalFetchCheckpointError(
            "checkpoint manifest references a path outside its root"
        )
    cursor = root
    for part in relative_path.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FundamentalFetchCheckpointError("checkpoint path is unsafe") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalFetchCheckpointError("checkpoint path contains a symlink")
    return root / relative_path


def _checkpoint_binding(
    *,
    symbols: Sequence[str],
    years: int,
    start_date: str,
    financial_start_date: str,
    as_of: str,
    canonical_scope_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "symbols_requested": int(len(symbols)),
        "symbol_set_sha256": _symbol_scope_sha256(symbols),
        "years": int(years),
        "daily_start_date": start_date,
        "financial_start_date": financial_start_date,
        "as_of": as_of,
        "tables": list(SOURCE_TABLES),
        "request_fields": dict(SOURCE_REQUEST_FIELDS),
        "pit_contract_version": FUNDAMENTAL_FETCH_PIT_CONTRACT,
        "canonical_scope_evidence": dict(canonical_scope_evidence or {}),
    }


@dataclass(frozen=True)
class _FetchCheckpointSnapshot:
    tables: dict[str, pd.DataFrame]
    outcomes: list[dict[str, Any]]
    revision: int
    generation_id: str
    pointer_sha256: str
    manifest_sha256: str
    binding_sha256: str
    outcome_accounting_sha256: str
    table_evidence_sha256: str


def _empty_fetch_checkpoint_snapshot(
    expected_binding: Mapping[str, Any],
) -> _FetchCheckpointSnapshot:
    return _FetchCheckpointSnapshot(
        tables={table: pd.DataFrame() for table in SOURCE_TABLES},
        outcomes=[],
        revision=0,
        generation_id="",
        pointer_sha256="",
        manifest_sha256="",
        binding_sha256=_canonical_mapping_sha256(expected_binding),
        outcome_accounting_sha256=canonical_json_sha256([]),
        table_evidence_sha256=canonical_json_sha256({}),
    )


def _decode_checkpoint_object(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise FundamentalFetchCheckpointError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FundamentalFetchCheckpointError(f"{label} must be an object")
    return value


def _verify_fetch_checkpoint_pointer_bytes(
    checkpoint_root: Path,
    pointer_bytes: bytes,
    *,
    expected_binding: Mapping[str, Any],
) -> _FetchCheckpointSnapshot:
    pointer = _decode_checkpoint_object(pointer_bytes, label="checkpoint pointer")
    if pointer.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA:
        raise FundamentalFetchCheckpointError("checkpoint pointer schema mismatch")
    try:
        pointer_revision = strict_nonnegative_int(
            pointer.get("revision"), label="checkpoint pointer revision"
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalFetchCheckpointError(str(exc)) from exc
    if pointer_revision < 1:
        raise FundamentalFetchCheckpointError("checkpoint pointer revision must be positive")
    generation_id = str(pointer.get("generation_id") or "")
    manifest_path = _checkpoint_path(
        checkpoint_root,
        str(pointer.get("manifest_path") or ""),
    )
    if not manifest_path.is_file():
        raise FundamentalFetchCheckpointError("checkpoint manifest is missing")
    manifest_bytes = _stable_regular_file_bytes(
        manifest_path, label="checkpoint manifest"
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if str(pointer.get("manifest_sha256") or "").lower() != manifest_sha256:
        raise FundamentalFetchCheckpointError("checkpoint manifest SHA mismatch")
    manifest = _decode_checkpoint_object(manifest_bytes, label="checkpoint manifest")
    if manifest.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA:
        raise FundamentalFetchCheckpointError("checkpoint manifest schema mismatch")
    try:
        manifest_revision = strict_nonnegative_int(
            manifest.get("revision"), label="checkpoint manifest revision"
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalFetchCheckpointError(str(exc)) from exc
    if manifest_revision != pointer_revision:
        raise FundamentalFetchCheckpointError("checkpoint revision mismatch")
    if str(manifest.get("generation_id") or "") != generation_id:
        raise FundamentalFetchCheckpointError("checkpoint generation mismatch")
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise FundamentalFetchCheckpointError("checkpoint binding is missing")
    binding_sha256 = _canonical_mapping_sha256(binding)
    if str(manifest.get("binding_sha256") or "").lower() != binding_sha256:
        raise FundamentalFetchCheckpointError("checkpoint binding SHA mismatch")
    if binding_sha256 != _canonical_mapping_sha256(expected_binding):
        raise FundamentalFetchCheckpointError(
            "checkpoint binding mismatch (scope, as_of, or fetch window drift)"
        )
    generation_root = manifest_path.parent
    table_files = manifest.get("table_files")
    if not isinstance(table_files, Mapping) or set(table_files) != set(SOURCE_TABLES):
        raise FundamentalFetchCheckpointError("checkpoint table manifest is incomplete")
    table_evidence_sha256 = canonical_json_sha256(dict(table_files))
    if str(manifest.get("table_evidence_sha256") or "").lower() != table_evidence_sha256:
        raise FundamentalFetchCheckpointError("checkpoint table evidence SHA mismatch")
    tables: dict[str, pd.DataFrame] = {}
    for table in SOURCE_TABLES:
        entry = table_files.get(table)
        if not isinstance(entry, Mapping):
            raise FundamentalFetchCheckpointError(f"checkpoint table entry is invalid: {table}")
        path = _checkpoint_path(generation_root, str(entry.get("path") or ""))
        if not path.is_file():
            raise FundamentalFetchCheckpointError(f"checkpoint table file is missing: {table}")
        parquet_bytes = _stable_regular_file_bytes(
            path, label=f"checkpoint table {table}"
        )
        if str(entry.get("sha256") or "").lower() != hashlib.sha256(parquet_bytes).hexdigest():
            raise FundamentalFetchCheckpointError(f"checkpoint table SHA mismatch: {table}")
        try:
            frame = pd.read_parquet(io.BytesIO(parquet_bytes))
        except Exception as exc:
            raise FundamentalFetchCheckpointError(
                f"checkpoint table is unreadable: {table}: {exc}"
            ) from exc
        try:
            row_count = strict_nonnegative_int(
                entry.get("row_count"), label=f"checkpoint {table} row_count"
            )
        except (TypeError, ValueError) as exc:
            raise FundamentalFetchCheckpointError(str(exc)) from exc
        if row_count != len(frame):
            raise FundamentalFetchCheckpointError(f"checkpoint table row count mismatch: {table}")
        if entry.get("columns") != list(frame.columns):
            raise FundamentalFetchCheckpointError(
                f"checkpoint table columns mismatch: {table}"
            )
        if str(entry.get("frame_fingerprint") or "").lower() != frame_fingerprint(frame):
            raise FundamentalFetchCheckpointError(
                f"checkpoint table frame fingerprint mismatch: {table}"
            )
        if entry.get("logical_schema") != frame_logical_schema(frame):
            raise FundamentalFetchCheckpointError(
                f"checkpoint table logical schema mismatch: {table}"
            )
        tables[table] = frame
    outcomes_entry = manifest.get("request_outcomes")
    if not isinstance(outcomes_entry, Mapping):
        raise FundamentalFetchCheckpointError("checkpoint request outcomes are missing")
    outcomes_path = _checkpoint_path(
        generation_root,
        str(outcomes_entry.get("path") or ""),
    )
    if not outcomes_path.is_file():
        raise FundamentalFetchCheckpointError("checkpoint request outcomes file is missing")
    outcomes_bytes = _stable_regular_file_bytes(
        outcomes_path, label="checkpoint request outcomes"
    )
    if str(outcomes_entry.get("sha256") or "").lower() != hashlib.sha256(outcomes_bytes).hexdigest():
        raise FundamentalFetchCheckpointError("checkpoint request outcomes SHA mismatch")
    outcomes_payload = _decode_checkpoint_object(
        outcomes_bytes, label="checkpoint request outcomes"
    )
    outcomes_value = outcomes_payload.get("outcomes")
    if not isinstance(outcomes_value, list):
        raise FundamentalFetchCheckpointError("checkpoint outcomes must be a list")
    try:
        outcome_count = strict_nonnegative_int(
            outcomes_entry.get("count"), label="checkpoint outcome count"
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalFetchCheckpointError(str(exc)) from exc
    if outcome_count != len(outcomes_value):
        raise FundamentalFetchCheckpointError("checkpoint outcome count mismatch")
    outcomes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_outcome in outcomes_value:
        if not isinstance(raw_outcome, Mapping):
            raise FundamentalFetchCheckpointError("checkpoint outcome must be an object")
        outcome = dict(raw_outcome)
        key = (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        )
        if key in seen:
            raise FundamentalFetchCheckpointError("checkpoint outcomes contain duplicates")
        seen.add(key)
        if not key[0] or key[1] not in SOURCE_TABLES:
            raise FundamentalFetchCheckpointError("checkpoint outcome is malformed")
        try:
            validate_outcome_accounting_v3(
                outcome, label=f"checkpoint outcome {key[0]}/{key[1]}"
            )
        except (TypeError, ValueError) as exc:
            raise FundamentalFetchCheckpointError(str(exc)) from exc
        outcomes.append(outcome)
    if outcomes != sorted(
        outcomes,
        key=lambda item: (str(item.get("symbol")), str(item.get("table"))),
    ):
        raise FundamentalFetchCheckpointError("checkpoint outcomes are not sorted")
    outcome_accounting_sha256 = canonical_json_sha256(outcomes)
    if str(outcomes_entry.get("accounting_sha256") or "").lower() != outcome_accounting_sha256:
        raise FundamentalFetchCheckpointError("checkpoint outcome accounting SHA mismatch")
    if str(manifest.get("outcome_accounting_sha256") or "").lower() != outcome_accounting_sha256:
        raise FundamentalFetchCheckpointError("checkpoint manifest outcome SHA mismatch")
    for table in SOURCE_TABLES:
        declared_rows = sum(
            strict_nonnegative_int(
                outcome.get("rows"),
                label=f"checkpoint {table} outcome rows",
            )
            for outcome in outcomes
            if str(outcome.get("table") or "") == table
        )
        if declared_rows != len(tables[table]):
            raise FundamentalFetchCheckpointError(
                f"checkpoint outcome/table row count mismatch: {table}"
            )
    return _FetchCheckpointSnapshot(
        tables=tables,
        outcomes=outcomes,
        revision=pointer_revision,
        generation_id=generation_id,
        pointer_sha256=hashlib.sha256(pointer_bytes).hexdigest(),
        manifest_sha256=manifest_sha256,
        binding_sha256=binding_sha256,
        outcome_accounting_sha256=outcome_accounting_sha256,
        table_evidence_sha256=table_evidence_sha256,
    )


def _current_checkpoint_pointer_bytes(checkpoint_root: Path) -> bytes | None:
    pointer_path = checkpoint_root / "latest.json"
    if not pointer_path.exists():
        if pointer_path.is_symlink():
            raise FundamentalFetchCheckpointError("checkpoint pointer is a symlink")
        unexpected = []
        for path in checkpoint_root.iterdir():
            if path.name == ".checkpoint.lock":
                continue
            if path.name == "_generations":
                _validate_orphan_checkpoint_generations(path)
                continue
            unexpected.append(path.name)
        if unexpected:
            raise FundamentalFetchCheckpointError(
                "checkpoint root is non-empty but latest.json is missing"
            )
        return None
    return _stable_regular_file_bytes(pointer_path, label="checkpoint pointer")


def _validate_orphan_checkpoint_generations(generations_root: Path) -> None:
    try:
        root_metadata = os.lstat(generations_root)
    except OSError as exc:
        raise FundamentalFetchCheckpointError(
            "checkpoint orphan generations root is unsafe"
        ) from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FundamentalFetchCheckpointError(
            "checkpoint orphan generations root is unsafe"
        )

    generation_prefix = "checkpoint_00000001_"
    for generation_root in generations_root.iterdir():
        try:
            generation_metadata = os.lstat(generation_root)
        except OSError as exc:
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan generation is unsafe"
            ) from exc
        generation_suffix = generation_root.name.removeprefix(generation_prefix)
        if (
            stat.S_ISLNK(generation_metadata.st_mode)
            or not stat.S_ISDIR(generation_metadata.st_mode)
            or not generation_root.name.startswith(generation_prefix)
            or not generation_suffix.isdigit()
        ):
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan generation is unsafe"
            )
        _validate_orphan_checkpoint_generation_tree(generation_root)


def _validate_orphan_checkpoint_generation_tree(generation_root: Path) -> None:
    json_names = {"manifest.json", "request_outcomes.json"}
    for path in generation_root.iterdir():
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan generation contains an unsafe entry"
            ) from exc
        if path.name == "tables":
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise FundamentalFetchCheckpointError(
                    "checkpoint orphan tables entry is unsafe"
                )
            _validate_orphan_checkpoint_table_tree(path)
            continue
        is_known_json = path.name in json_names
        is_known_json_temporary = any(
            path.name.startswith(f".{name}.tmp-") for name in json_names
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not (is_known_json or is_known_json_temporary)
        ):
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan generation contains an unsafe entry"
            )


def _validate_orphan_checkpoint_table_tree(tables_root: Path) -> None:
    table_names = {f"{table}.parquet" for table in SOURCE_TABLES}
    for path in tables_root.iterdir():
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan tables contain an unsafe entry"
            ) from exc
        is_known_table = path.name in table_names
        is_known_temporary = any(
            path.name.startswith(f".{name}.tmp-") for name in table_names
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not (is_known_table or is_known_temporary)
        ):
            raise FundamentalFetchCheckpointError(
                "checkpoint orphan tables contain an unsafe entry"
            )


def _load_fetch_checkpoint(
    checkpoint_root: Path,
    *,
    expected_binding: Mapping[str, Any],
) -> _FetchCheckpointSnapshot:
    with _checkpoint_root_lock(checkpoint_root):
        pointer_bytes = _current_checkpoint_pointer_bytes(checkpoint_root)
        if pointer_bytes is None:
            return _empty_fetch_checkpoint_snapshot(expected_binding)
        return _verify_fetch_checkpoint_pointer_bytes(
            checkpoint_root,
            pointer_bytes,
            expected_binding=expected_binding,
        )


def _checkpoint_pointer_revision_for_cas(
    checkpoint_root: Path,
    pointer_bytes: bytes,
    *,
    expected_binding: Mapping[str, Any],
) -> int:
    pointer = _decode_checkpoint_object(
        pointer_bytes,
        label="checkpoint pointer CAS readback",
    )
    if pointer.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA:
        raise FundamentalFetchCheckpointError("checkpoint pointer schema mismatch")
    try:
        revision = strict_nonnegative_int(
            pointer.get("revision"),
            label="checkpoint pointer revision",
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalFetchCheckpointError(str(exc)) from exc
    if revision < 1:
        raise FundamentalFetchCheckpointError(
            "checkpoint pointer revision must be positive"
        )
    generation_id = str(pointer.get("generation_id") or "")
    manifest_path = _checkpoint_path(
        checkpoint_root,
        str(pointer.get("manifest_path") or ""),
    )
    if not manifest_path.is_file():
        raise FundamentalFetchCheckpointError("checkpoint manifest is missing")
    manifest_bytes = _stable_regular_file_bytes(
        manifest_path,
        label="checkpoint manifest CAS readback",
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != str(
        pointer.get("manifest_sha256") or ""
    ).lower():
        raise FundamentalFetchCheckpointError("checkpoint manifest SHA mismatch")
    manifest = _decode_checkpoint_object(
        manifest_bytes,
        label="checkpoint manifest CAS readback",
    )
    if (
        manifest.get("schema_version") != FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA
        or str(manifest.get("generation_id") or "") != generation_id
    ):
        raise FundamentalFetchCheckpointError(
            "checkpoint pointer/manifest CAS mismatch"
        )
    try:
        manifest_revision = strict_nonnegative_int(
            manifest.get("revision"),
            label="checkpoint manifest revision",
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalFetchCheckpointError(str(exc)) from exc
    binding = manifest.get("binding")
    manifest_binding_sha256 = (
        _canonical_mapping_sha256(binding)
        if isinstance(binding, Mapping)
        else ""
    )
    if (
        manifest_revision != revision
        or not isinstance(binding, Mapping)
        or manifest_binding_sha256
        != str(manifest.get("binding_sha256") or "").lower()
        or manifest_binding_sha256 != _canonical_mapping_sha256(expected_binding)
    ):
        raise FundamentalFetchCheckpointError(
            "checkpoint revision or binding CAS mismatch"
        )
    return revision


def _checkpoint_financial_coverage_contract_valid(
    outcome: Mapping[str, Any],
) -> bool:
    table = str(outcome.get("table") or "")
    status = str(outcome.get("status") or "")
    if table not in FINANCIAL_SOURCE_TABLES or status not in {
        "success",
        "empty",
    }:
        return True
    coverage = outcome.get("financial_coverage")
    if not isinstance(coverage, Mapping):
        return False
    coverage_status = str(coverage.get("status") or "")
    coverage_passed = coverage.get("passed")
    declared_passed = outcome.get("financial_coverage_passed")
    return bool(
        coverage_status in {"applicable", "not_applicable"}
        and type(coverage_passed) is bool
        and type(declared_passed) is bool
        and declared_passed is coverage_passed
    )


def _write_fetch_checkpoint(
    checkpoint_root: Path,
    *,
    binding: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    outcomes: Sequence[Mapping[str, Any]],
    expected_pointer_sha256: str,
    expected_revision: int,
) -> _FetchCheckpointSnapshot:
    with _checkpoint_root_lock(checkpoint_root):
        current_pointer_bytes = _current_checkpoint_pointer_bytes(checkpoint_root)
        current_pointer_sha256 = (
            hashlib.sha256(current_pointer_bytes).hexdigest()
            if current_pointer_bytes is not None
            else ""
        )
        if current_pointer_sha256 != expected_pointer_sha256:
            raise FundamentalFetchCheckpointError("checkpoint pointer CAS mismatch")
        if current_pointer_bytes is None:
            current_revision = 0
        else:
            # The exact pointer bytes were fully verified when the resume state
            # was loaded. Under the root lock the byte-level CAS is sufficient;
            # reloading every Parquet table here would validate the same revision
            # again before it is immediately superseded.
            current_revision = _checkpoint_pointer_revision_for_cas(
                checkpoint_root,
                current_pointer_bytes,
                expected_binding=binding,
            )
        if current_revision != expected_revision:
            raise FundamentalFetchCheckpointError("checkpoint revision CAS mismatch")
        next_revision = expected_revision + 1
        generation_id = f"checkpoint_{next_revision:08d}_{time.time_ns()}"
        generation_root = checkpoint_root / "_generations" / generation_id
        if generation_root.exists():
            raise FundamentalFetchCheckpointError(
                f"checkpoint generation already exists: {generation_root}"
            )
        generation_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(generation_root.parent, 0o700)
        os.chmod(generation_root, 0o700)
        table_files: dict[str, Any] = {}
        readback_tables: dict[str, pd.DataFrame] = {}
        for table in SOURCE_TABLES:
            frame = tables.get(table, pd.DataFrame())
            if not isinstance(frame, pd.DataFrame):
                raise FundamentalFetchCheckpointError(
                    f"checkpoint table is not a DataFrame: {table}"
                )
            relative = f"tables/{table}.parquet"
            path = generation_root / relative
            _atomic_parquet_write(path, frame)
            parquet_bytes = _stable_regular_file_bytes(
                path, label=f"checkpoint candidate table {table}"
            )
            try:
                readback = pd.read_parquet(io.BytesIO(parquet_bytes))
                assert_frame_semantics_equal(
                    frame,
                    readback,
                    label=f"checkpoint candidate table {table}",
                )
            except Exception as exc:
                raise FundamentalFetchCheckpointError(
                    f"checkpoint table semantic readback mismatch: {table}: {exc}"
                ) from exc
            table_files[table] = {
                "path": relative,
                "sha256": hashlib.sha256(parquet_bytes).hexdigest(),
                "frame_fingerprint": frame_fingerprint(readback),
                "logical_schema": frame_logical_schema(readback),
                "columns": list(readback.columns),
                "row_count": int(len(readback)),
            }
            readback_tables[table] = readback
        sorted_outcomes = sorted(
            (dict(outcome) for outcome in outcomes),
            key=lambda item: (str(item.get("symbol")), str(item.get("table"))),
        )
        seen_outcome_keys: set[tuple[str, str]] = set()
        for index, outcome in enumerate(sorted_outcomes):
            try:
                validate_outcome_accounting_v3(
                    outcome, label=f"checkpoint candidate outcome {index}"
                )
            except (TypeError, ValueError) as exc:
                raise FundamentalFetchCheckpointError(str(exc)) from exc
            key = (
                normalize_ts_code(outcome.get("symbol")),
                str(outcome.get("table") or ""),
            )
            if not key[0] or key[1] not in SOURCE_TABLES:
                raise FundamentalFetchCheckpointError(
                    "checkpoint outcome is malformed"
                )
            if key in seen_outcome_keys:
                raise FundamentalFetchCheckpointError(
                    "checkpoint outcomes contain duplicates"
                )
            seen_outcome_keys.add(key)
            if not _checkpoint_financial_coverage_contract_valid(outcome):
                raise FundamentalFetchCheckpointError(
                    "checkpoint financial coverage is missing or inconsistent: "
                    f"{key[0]}/{key[1]}"
                )
        for table in SOURCE_TABLES:
            declared_rows = sum(
                strict_nonnegative_int(
                    outcome.get("rows"),
                    label=f"checkpoint {table} outcome rows",
                )
                for outcome in sorted_outcomes
                if str(outcome.get("table") or "") == table
            )
            if declared_rows != len(readback_tables[table]):
                raise FundamentalFetchCheckpointError(
                    f"checkpoint outcome/table row count mismatch: {table}"
                )
        outcome_accounting_sha256 = canonical_json_sha256(sorted_outcomes)
        outcomes_path = generation_root / "request_outcomes.json"
        _atomic_json_write(outcomes_path, {"outcomes": sorted_outcomes})
        outcomes_bytes = _stable_regular_file_bytes(
            outcomes_path, label="checkpoint candidate request outcomes"
        )
        if outcomes_bytes != _canonical_json_file_bytes(
            {"outcomes": sorted_outcomes}
        ):
            raise FundamentalFetchCheckpointError(
                "checkpoint request outcome readback changed"
            )
        outcomes_readback = _decode_checkpoint_object(
            outcomes_bytes, label="checkpoint candidate request outcomes"
        )
        if outcomes_readback.get("outcomes") != sorted_outcomes:
            raise FundamentalFetchCheckpointError(
                "checkpoint request outcome readback changed"
            )
        manifest = {
            "schema_version": FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
            "revision": next_revision,
            "generation_id": generation_id,
            "created_at": _now_utc().isoformat().replace("+00:00", "Z"),
            "binding": dict(binding),
            "binding_sha256": _canonical_mapping_sha256(binding),
            "outcome_accounting_sha256": outcome_accounting_sha256,
            "table_files": table_files,
            "table_evidence_sha256": canonical_json_sha256(table_files),
            "request_outcomes": {
                "path": outcomes_path.name,
                "sha256": hashlib.sha256(outcomes_bytes).hexdigest(),
                "accounting_sha256": outcome_accounting_sha256,
                "count": int(len(sorted_outcomes)),
            },
        }
        manifest_path = generation_root / "manifest.json"
        _atomic_json_write(manifest_path, manifest)
        manifest_bytes = _stable_regular_file_bytes(
            manifest_path, label="checkpoint candidate manifest"
        )
        if manifest_bytes != _canonical_json_file_bytes(manifest):
            raise FundamentalFetchCheckpointError(
                "checkpoint candidate manifest readback changed"
            )
        _fsync_checkpoint_directory(generation_root)
        pointer = {
            "schema_version": FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA,
            "generation_id": generation_id,
            "manifest_path": str(
                manifest_path.relative_to(checkpoint_root).as_posix()
            ),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "revision": next_revision,
        }
        candidate_pointer_bytes = _canonical_json_file_bytes(pointer)

        def revalidate_candidate() -> None:
            for table in SOURCE_TABLES:
                table_path = generation_root / str(table_files[table]["path"])
                if _stable_regular_file_sha256(
                    table_path,
                    label=f"checkpoint candidate table {table} publication recheck",
                ) != str(table_files[table]["sha256"]):
                    raise FundamentalFetchCheckpointError(
                        f"checkpoint candidate table changed before publication: {table}"
                    )
            if _stable_regular_file_bytes(
                outcomes_path,
                label="checkpoint candidate outcomes publication recheck",
            ) != outcomes_bytes:
                raise FundamentalFetchCheckpointError(
                    "checkpoint candidate outcomes changed before publication"
                )
            if _stable_regular_file_bytes(
                manifest_path,
                label="checkpoint candidate manifest publication recheck",
            ) != manifest_bytes:
                raise FundamentalFetchCheckpointError(
                    "checkpoint candidate manifest changed before publication"
                )

        latest_path = checkpoint_root / "latest.json"
        latest_before_switch = (
            _stable_regular_file_bytes(
                latest_path, label="checkpoint pointer CAS readback"
            )
            if latest_path.exists()
            else None
        )
        latest_before_switch_sha256 = (
            hashlib.sha256(latest_before_switch).hexdigest()
            if latest_before_switch is not None
            else ""
        )
        if latest_before_switch_sha256 != expected_pointer_sha256:
            raise FundamentalFetchCheckpointError("checkpoint pointer CAS mismatch")
        try:
            # Recheck exact candidate identity after the pointer CAS read and as
            # close as possible to the canonical latest.json switch.  This
            # closes the validation-to-switch window for same-user mutation.
            revalidate_candidate()
        except ValueError as exc:
            raise FundamentalFetchCheckpointError(str(exc)) from exc
        _atomic_json_write(checkpoint_root / "latest.json", pointer)
        published_pointer_bytes = _stable_regular_file_bytes(
            checkpoint_root / "latest.json", label="published checkpoint pointer"
        )
        if published_pointer_bytes != candidate_pointer_bytes:
            raise FundamentalFetchCheckpointError(
                "published checkpoint pointer changed from candidate"
            )
        return _FetchCheckpointSnapshot(
            tables=readback_tables,
            outcomes=[dict(outcome) for outcome in sorted_outcomes],
            revision=next_revision,
            generation_id=generation_id,
            pointer_sha256=hashlib.sha256(published_pointer_bytes).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            binding_sha256=_canonical_mapping_sha256(binding),
            outcome_accounting_sha256=outcome_accounting_sha256,
            table_evidence_sha256=canonical_json_sha256(table_files),
        )


def _checkpoint_outcome_requires_refetch(
    outcome: Mapping[str, Any],
    *,
    daily_basic_empty_exception_symbols: Collection[str],
) -> bool:
    status = str(outcome.get("status") or "")
    if status in {"error", "malformed"}:
        return True
    table = str(outcome.get("table") or "")
    symbol = normalize_ts_code(outcome.get("symbol"))
    if table == "daily_basic":
        if status == "empty" and symbol in daily_basic_empty_exception_symbols:
            return False
        return status != "success" or outcome.get("history_complete") is not True
    if table in FINANCIAL_SOURCE_TABLES:
        if status != "success":
            return True
        coverage = outcome.get("financial_coverage")
        if not _checkpoint_financial_coverage_contract_valid(outcome):
            return True
        assert isinstance(coverage, Mapping)
        if (
            coverage.get("passed") is not True
            or outcome.get("financial_coverage_passed") is not True
        ):
            return True
    return status not in {"success", "empty"}


def _fetch_tushare_tables(
    symbols: Sequence[str],
    *,
    years: int,
    as_of: str,
    workers: int,
    pro: Any,
    canonical_scope_evidence: Mapping[str, Any] | None = None,
    checkpoint_root: str | Path | None = None,
    checkpoint_batch_size: int = 500,
    max_attempts: int = 1,
    retry_backoff_seconds: float = 0.0,
    max_retry_backoff_seconds: float = 0.0,
    requests_per_second: float = 0.0,
    endpoint_audit_policy: FundamentalEndpointAuditPolicy | None = None,
    enforce_endpoint_audit: bool = False,
    symbol_pause_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    attempt_limit = int(max_attempts)
    if attempt_limit < 1:
        raise ValueError("max_attempts must be at least 1")
    initial_backoff = float(retry_backoff_seconds)
    maximum_backoff = float(max_retry_backoff_seconds)
    if initial_backoff < 0 or maximum_backoff < 0:
        raise ValueError("retry backoff values must be non-negative")
    batch_size = int(checkpoint_batch_size)
    if batch_size < 1:
        raise ValueError("checkpoint_batch_size must be at least 1")
    pause_seconds = float(symbol_pause_seconds)
    if pause_seconds < 0:
        raise ValueError("symbol_pause_seconds must be non-negative")
    sleep = sleep_fn or time.sleep
    monotonic = monotonic_fn or time.monotonic
    end_text = _normalize_fetch_as_of(as_of)
    end = pd.Timestamp(pd.to_datetime(end_text, format="%Y%m%d"))
    start = end - pd.DateOffset(years=int(years))
    financial_start = start - pd.DateOffset(years=2)
    start_text = start.strftime("%Y%m%d")
    financial_start_text = financial_start.strftime("%Y%m%d")
    normalized_symbols = [
        symbol for symbol in dict.fromkeys(normalize_ts_code(value) for value in symbols) if symbol
    ]
    validated_scope_evidence = (
        _validate_canonical_scope_evidence(
            canonical_scope_evidence,
            normalized_symbols,
        )
        if canonical_scope_evidence is not None
        else None
    )
    binding = _checkpoint_binding(
        symbols=normalized_symbols,
        years=int(years),
        start_date=start_text,
        financial_start_date=financial_start_text,
        as_of=end_text,
        canonical_scope_evidence=validated_scope_evidence,
    )
    resolved_checkpoint_root = (
        _safe_checkpoint_root(checkpoint_root)
        if checkpoint_root is not None
        else None
    )
    checkpoint_state = _empty_fetch_checkpoint_snapshot(binding)
    if resolved_checkpoint_root is not None:
        checkpoint_state = _load_fetch_checkpoint(
            resolved_checkpoint_root,
            expected_binding=binding,
        )
    audit_tail_gap_exceptions = set(
        _active_daily_tail_gap_exceptions(
            validated_scope_evidence,
            as_of=end_text,
        )
    )
    audit_policy = endpoint_audit_policy or FundamentalEndpointAuditPolicy()
    expected_keys = {(symbol, table) for symbol in normalized_symbols for table in SOURCE_TABLES}
    base_tables: dict[str, pd.DataFrame] = {
        table: checkpoint_state.tables[table]
        for table in SOURCE_TABLES
    }

    def attach_checkpoint_coverage(
        checkpoint_tables: Mapping[str, pd.DataFrame],
        checkpoint_outcomes: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        attached = _attach_daily_history_coverage(
            normalized_symbols,
            checkpoint_outcomes,
            checkpoint_tables,
            daily_start=start_text,
            as_of=end_text,
            scope_evidence=validated_scope_evidence,
            policy=audit_policy,
        )
        return _attach_financial_coverage(
            normalized_symbols,
            attached,
            checkpoint_tables,
            financial_start=financial_start_text,
            as_of=end_text,
            scope_evidence=validated_scope_evidence,
            policy=audit_policy,
        )

    invalid_checkpoint_coverage_keys = {
        (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        )
        for outcome in checkpoint_state.outcomes
        if not _checkpoint_financial_coverage_contract_valid(outcome)
    }
    checkpoint_outcomes = attach_checkpoint_coverage(
        base_tables,
        checkpoint_state.outcomes,
    )
    outcome_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for outcome in checkpoint_outcomes:
        key = (
            normalize_ts_code(outcome.get("symbol")),
            str(outcome.get("table") or ""),
        )
        if key not in expected_keys:
            raise FundamentalFetchCheckpointError(
                "checkpoint request outcome is outside the bound scope"
            )
        outcome_by_key[key] = dict(outcome)
    resumed_valid_request_count = sum(
        (
            (
                normalize_ts_code(outcome.get("symbol")),
                str(outcome.get("table") or ""),
            )
            not in invalid_checkpoint_coverage_keys
            and not _checkpoint_outcome_requires_refetch(
                outcome,
                daily_basic_empty_exception_symbols=audit_tail_gap_exceptions,
            )
        )
        for outcome in checkpoint_outcomes
    )
    replacement_frames: dict[tuple[str, str], pd.DataFrame] = {}
    limiter = _RequestRateLimiter(
        float(requests_per_second),
        sleep_fn=sleep,
        monotonic_fn=monotonic,
    )

    def _bounded_error(exc: object) -> str:
        return str(exc).strip().replace("\n", " ")[:500] or type(exc).__name__

    def _call_endpoint(
        method: Any,
        *,
        symbol: str,
        table: str,
        table_start_text: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        provider_calls = 0
        last_error = ""
        last_malformed = ""
        last_stats = _zero_request_outcome_accounting()
        for attempt in range(1, attempt_limit + 1):
            try:
                limiter.wait()
                provider_calls += 1
                try:
                    response = method(
                        ts_code=symbol,
                        start_date=table_start_text,
                        end_date=end_text,
                        fields=SOURCE_REQUEST_FIELDS[table],
                    )
                except TypeError:
                    limiter.wait()
                    provider_calls += 1
                    response = method(
                        ts_code=symbol,
                        fields=SOURCE_REQUEST_FIELDS[table],
                    )
                if not isinstance(response, pd.DataFrame):
                    last_malformed = "provider_response_not_dataframe"
                    last_stats = _zero_request_outcome_accounting()
                else:
                    accepted, cutoff_stats, malformed_reason = _strict_pit_cutoff(
                        response,
                        table=table,
                        symbol=symbol,
                        as_of=end_text,
                    )
                    last_stats = cutoff_stats
                    if not malformed_reason:
                        status = "success" if not accepted.empty else "empty"
                        history_evidence: dict[str, Any] = {}
                        if table == "daily_basic" and status == "success":
                            scope_evidence = dict(validated_scope_evidence or {})
                            listing_dates = dict(
                                scope_evidence.get("listing_dates", {}) or {}
                            )
                            history_end_dates = dict(
                                scope_evidence.get("history_end_dates", {}) or {}
                            )
                            bar_first_dates = dict(
                                scope_evidence.get("canonical_bar_first_dates", {})
                                or {}
                            )
                            bar_last_dates = dict(
                                scope_evidence.get("canonical_bar_last_dates", {})
                                or {}
                            )
                            expected_start = max(
                                start_text,
                                str(listing_dates.get(symbol) or start_text),
                                str(bar_first_dates.get(symbol) or start_text),
                            )
                            expected_end = min(
                                end_text,
                                str(history_end_dates.get(symbol) or end_text),
                                str(bar_last_dates.get(symbol) or end_text),
                            )
                            if expected_start > expected_end:
                                raise ValueError(
                                    "canonical daily history bounds are reversed: "
                                    f"{symbol}"
                                )
                            dates = pd.to_datetime(
                                accepted["trade_date"].astype("string"),
                                format="%Y%m%d",
                                errors="coerce",
                            )
                            interval_rows = [
                                interval
                                for interval in list(
                                    scope_evidence.get(
                                        "daily_history_coverage_intervals", []
                                    )
                                    or []
                                )
                                if normalize_ts_code(interval.get("symbol"))
                                == symbol
                            ]
                            history_evidence = {
                                **_daily_history_coverage_metrics(
                                    dates,
                                    expected_start=expected_start,
                                    expected_end=expected_end,
                                    allow_tail_gap=False,
                                    boundary_tolerance_days=int(
                                        audit_policy.daily_history_boundary_tolerance_days
                                    ),
                                    coverage_intervals=interval_rows,
                                    symbol=symbol,
                                    listing_identity=str(
                                        dict(
                                            scope_evidence.get(
                                                "listing_identities", {}
                                            )
                                            or {}
                                        ).get(symbol)
                                        or ""
                                    ),
                                    listing_start=str(
                                        listing_dates.get(symbol) or expected_start
                                    ),
                                    listing_end=str(
                                        history_end_dates.get(symbol) or expected_end
                                    ),
                                    listing_source_sha256=str(
                                        scope_evidence.get(
                                            "canonical_membership_sha256"
                                        )
                                        or ""
                                    ),
                                    cutoff=end_text,
                                ),
                            }
                        outcome = {
                            "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
                            "symbol": symbol,
                            "table": table,
                            "status": status,
                            "error": "",
                            "attempts": attempt,
                            "provider_calls": provider_calls,
                            **cutoff_stats,
                            **history_evidence,
                        }
                        validate_outcome_accounting_v3(
                            outcome,
                            label=f"provider outcome {symbol}/{table}",
                        )
                        return accepted, outcome
                    last_malformed = malformed_reason
            except Exception as exc:
                last_error = _bounded_error(exc)
                last_malformed = ""
                last_stats = _zero_request_outcome_accounting()
            if attempt < attempt_limit and initial_backoff > 0:
                delay = initial_backoff * (2 ** (attempt - 1))
                if maximum_backoff > 0:
                    delay = min(delay, maximum_backoff)
                sleep(delay)
        status = "malformed" if last_malformed else "error"
        error = last_malformed or last_error or "provider_request_failed"
        outcome = {
            "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
            "symbol": symbol,
            "table": table,
            "status": status,
            "error": error,
            "attempts": attempt_limit,
            "provider_calls": provider_calls,
            **last_stats,
        }
        validate_outcome_accounting_v3(
            outcome,
            label=f"provider outcome {symbol}/{table}",
        )
        return pd.DataFrame(), outcome

    def fetch_symbol(
        symbol: str,
        pending_tables: Sequence[str],
    ) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
        symbol_frames: dict[str, pd.DataFrame] = {}
        symbol_outcomes: list[dict[str, Any]] = []
        for table in pending_tables:
            method = getattr(pro, table, None)
            if method is None:
                error = "provider_endpoint_missing"
                symbol_outcomes.append(
                    {
                        "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
                        "symbol": symbol,
                        "table": table,
                        "status": "error",
                        "error": error,
                        "attempts": 0,
                        "provider_calls": 0,
                        **_zero_request_outcome_accounting(),
                    }
                )
                continue
            table_start_text = (
                financial_start_text
                if table in FINANCIAL_SOURCE_TABLES or table == "forecast"
                else start_text
            )
            frame, outcome = _call_endpoint(
                method,
                symbol=symbol,
                table=table,
                table_start_text=table_start_text,
            )
            if not frame.empty:
                symbol_frames[table] = frame
            symbol_outcomes.append(outcome)
        if pause_seconds:
            sleep(pause_seconds)
        return symbol_frames, symbol_outcomes

    def materialize_tables() -> dict[str, pd.DataFrame]:
        materialized: dict[str, pd.DataFrame] = {}
        for table in SOURCE_TABLES:
            replacements = [
                (symbol, frame)
                for (symbol, replacement_table), frame in replacement_frames.items()
                if replacement_table == table
            ]
            base = base_tables[table]
            if not replacements:
                materialized[table] = base
                continue
            replaced_symbols = {symbol for symbol, _frame in replacements}
            if not base.empty:
                if "ts_code" not in base.columns:
                    raise FundamentalFetchCheckpointError(
                        f"checkpoint table cannot replace symbol rows: {table}"
                    )
                retained = base.loc[
                    ~base["ts_code"].map(normalize_ts_code).isin(replaced_symbols)
                ]
            else:
                retained = base
            pieces = [retained] if not retained.empty else []
            pieces.extend(frame for _symbol, frame in replacements if not frame.empty)
            materialized[table] = (
                pd.concat(pieces, ignore_index=True)
                if pieces
                else pd.DataFrame()
            )
        return materialized

    symbols_with_pending_tables = {
        symbol: [
            table
            for table in SOURCE_TABLES
            if (
                (symbol, table) in invalid_checkpoint_coverage_keys
                or _checkpoint_outcome_requires_refetch(
                    outcome_by_key.get(
                        (symbol, table),
                        {"symbol": symbol, "table": table, "status": ""},
                    ),
                    daily_basic_empty_exception_symbols=audit_tail_gap_exceptions,
                )
            )
        ]
        for symbol in normalized_symbols
    }
    symbols_with_pending_tables = {
        symbol: tables_to_fetch
        for symbol, tables_to_fetch in symbols_with_pending_tables.items()
        if tables_to_fetch
    }
    fetched_request_count = sum(
        len(tables_to_fetch) for tables_to_fetch in symbols_with_pending_tables.values()
    )
    completed_since_checkpoint = 0

    def consume_result(
        symbol_frames: Mapping[str, pd.DataFrame],
        symbol_outcomes: Sequence[Mapping[str, Any]],
    ) -> None:
        nonlocal completed_since_checkpoint, checkpoint_state
        for outcome in symbol_outcomes:
            key = (
                normalize_ts_code(outcome.get("symbol")),
                str(outcome.get("table") or ""),
            )
            if key not in expected_keys:
                raise FundamentalFetchCheckpointError(
                    "provider outcome is outside the bound scope"
                )
            replacement_frames[key] = symbol_frames.get(
                key[1],
                pd.DataFrame(),
            )
            previous = dict(outcome_by_key.get(key, {}) or {})
            combined = dict(outcome)
            combined["attempts_cumulative"] = int(
                previous.get("attempts_cumulative", previous.get("attempts", 0)) or 0
            ) + int(outcome.get("attempts", 0) or 0)
            combined["provider_calls_cumulative"] = int(
                previous.get(
                    "provider_calls_cumulative",
                    previous.get("provider_calls", 0),
                )
                or 0
            ) + int(outcome.get("provider_calls", 0) or 0)
            outcome_by_key[key] = combined
        completed_since_checkpoint += 1
        if resolved_checkpoint_root is not None and completed_since_checkpoint >= batch_size:
            checkpoint_tables = materialize_tables()
            covered_checkpoint_outcomes = attach_checkpoint_coverage(
                checkpoint_tables,
                list(outcome_by_key.values()),
            )
            checkpoint_state = _write_fetch_checkpoint(
                resolved_checkpoint_root,
                binding=binding,
                tables=checkpoint_tables,
                outcomes=covered_checkpoint_outcomes,
                expected_pointer_sha256=checkpoint_state.pointer_sha256,
                expected_revision=checkpoint_state.revision,
            )
            base_tables.clear()
            base_tables.update(checkpoint_state.tables)
            replacement_frames.clear()
            outcome_by_key.clear()
            outcome_by_key.update(
                {
                    (
                        normalize_ts_code(outcome.get("symbol")),
                        str(outcome.get("table") or ""),
                    ): dict(outcome)
                    for outcome in checkpoint_state.outcomes
                }
            )
            completed_since_checkpoint = 0

    worker_count = max(1, int(workers or 1))
    pending_items = list(symbols_with_pending_tables.items())
    if worker_count == 1 or len(pending_items) <= 1:
        for symbol, pending_tables in pending_items:
            consume_result(*fetch_symbol(symbol, pending_tables))
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_symbol, symbol, pending_tables): symbol
                for symbol, pending_tables in pending_items
            }
            for future in as_completed(futures):
                consume_result(*future.result())
                completed += 1
                if len(pending_items) >= 50 and completed % 100 == 0:
                    failed_so_far = sum(
                        str(outcome.get("status")) in {"error", "malformed"}
                        for outcome in outcome_by_key.values()
                    )
                    print(
                        f"[fundamental-maintain] fetched {completed}/{len(pending_items)} symbols; "
                        f"failed_requests={failed_so_far}",
                        flush=True,
                    )
    tables = materialize_tables()
    if validated_scope_evidence is not None:
        _validate_canonical_scope_evidence(
            validated_scope_evidence,
            normalized_symbols,
        )
    outcomes = attach_checkpoint_coverage(
        tables,
        list(outcome_by_key.values()),
    )
    if resolved_checkpoint_root is not None:
        checkpoint_unchanged = bool(
            checkpoint_state.revision > 0
            and fetched_request_count == 0
            and outcomes == checkpoint_state.outcomes
        )
        if not checkpoint_unchanged:
            checkpoint_state = _write_fetch_checkpoint(
                resolved_checkpoint_root,
                binding=binding,
                tables=tables,
                outcomes=outcomes,
                expected_pointer_sha256=checkpoint_state.pointer_sha256,
                expected_revision=checkpoint_state.revision,
            )
        # _write_fetch_checkpoint returns the already verified Parquet readback
        # snapshot. Independent resume/load calls still perform a full reload.
        tables = checkpoint_state.tables
        outcomes = checkpoint_state.outcomes
    succeeded = sum(str(item.get("status")) == "success" for item in outcomes)
    empty = sum(str(item.get("status")) == "empty" for item in outcomes)
    malformed = sum(str(item.get("status")) == "malformed" for item in outcomes)
    errors_only = sum(str(item.get("status")) == "error" for item in outcomes)
    failed = errors_only + malformed
    errors = [
        {
            "symbol": str(item.get("symbol") or ""),
            "table": str(item.get("table") or ""),
            "status": str(item.get("status") or ""),
            "error": str(item.get("error") or ""),
        }
        for item in outcomes
        if str(item.get("status")) in {"error", "malformed"}
    ][:200]
    endpoint_audit = _build_endpoint_audit(
        normalized_symbols,
        outcomes,
        policy=audit_policy,
        daily_basic_empty_exception_symbols=sorted(audit_tail_gap_exceptions),
    )
    manifest = {
        "schema_version": FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
        "provider": "tushare",
        "symbols_requested": int(len(normalized_symbols)),
        "tables": list(SOURCE_TABLES),
        "request_fields": dict(SOURCE_REQUEST_FIELDS),
        "pit_contract_version": FUNDAMENTAL_FETCH_PIT_CONTRACT,
        "years": int(years),
        "start_date": start_text,
        "daily_start_date": start_text,
        "financial_start_date": financial_start_text,
        "end_date": end_text,
        "requests_attempted": int(len(outcomes)),
        "requests_succeeded_with_rows": int(succeeded),
        "requests_empty": int(empty),
        "requests_failed": int(failed),
        "requests_malformed": int(malformed),
        "provider_calls_attempted": int(
            sum(
                int(
                    item.get(
                        "provider_calls_cumulative",
                        item.get("provider_calls", 0),
                    )
                )
                for item in outcomes
            )
        ),
        "requests_retried": int(
            sum(
                max(
                    0,
                    int(
                        item.get(
                            "attempts_cumulative",
                            item.get("attempts", 0),
                        )
                    )
                    - 1,
                )
                for item in outcomes
            )
        ),
        "errors_truncated": failed > len(errors),
        "errors": errors,
        "symbol_table_outcomes": outcomes,
        "absence_is_deletion": False,
        "deletion_policy": "explicit_tombstone_required",
        "raw_row_counts": {table: int(len(frame)) for table, frame in tables.items()},
        "raw_table_fingerprints": {
            table: frame_fingerprint(frame) for table, frame in tables.items()
        },
        "request_outcome_accounting_sha256": canonical_json_sha256(outcomes),
        "strict_pit_as_of": end_text,
        "pit_rows_filtered_future": int(
            sum(int(item.get("rows_filtered_future", 0)) for item in outcomes)
        ),
        "pit_rows_filtered_missing_availability": int(
            sum(
                int(item.get("rows_filtered_missing_availability", 0))
                for item in outcomes
            )
        ),
        "pit_rows_filtered_core_values": int(
            sum(int(item.get("rows_filtered_core_values", 0)) for item in outcomes)
        ),
        "pit_rows_filtered_invalid_date": int(
            sum(
                int(item.get("rows_hard_invalid_availability_date", 0))
                + int(item.get("rows_hard_invalid_end_date", 0))
                + int(item.get("rows_hard_invalid_end_after_availability", 0))
                for item in outcomes
            )
        ),
        "pit_rows_deduplicated": int(
            sum(int(item.get("rows_deduplicated", 0)) for item in outcomes)
        ),
        "retry_policy": {
            "max_attempts": attempt_limit,
            "initial_backoff_seconds": initial_backoff,
            "max_backoff_seconds": maximum_backoff,
        },
        "requests_per_second": float(requests_per_second),
        "endpoint_audit": endpoint_audit,
    }
    if validated_scope_evidence is not None:
        manifest["canonical_scope_evidence"] = validated_scope_evidence
        manifest["derivation"] = {
            "contract_version": FUNDAMENTAL_DERIVATION_CONTRACT,
            "pit_membership_path": str(
                validated_scope_evidence.get("canonical_membership_path") or ""
            ),
            "pit_membership_sha256": str(
                validated_scope_evidence.get("canonical_membership_sha256") or ""
            ),
            "sector_selection_rule": (
                "latest_active_membership_interval_as_of_else_latest_expired"
            ),
        }
    if resolved_checkpoint_root is not None:
        manifest["checkpoint"] = {
            "schema_version": FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
            "root": str(resolved_checkpoint_root),
            "generation_id": checkpoint_state.generation_id,
            "revision": int(checkpoint_state.revision),
            "pointer_sha256": checkpoint_state.pointer_sha256,
            "manifest_sha256": checkpoint_state.manifest_sha256,
            "binding_sha256": checkpoint_state.binding_sha256,
            "outcome_accounting_sha256": checkpoint_state.outcome_accounting_sha256,
            "table_evidence_sha256": checkpoint_state.table_evidence_sha256,
            "batch_size_symbols": batch_size,
            "resumed_valid_request_count": int(resumed_valid_request_count),
            "requests_fetched_this_run": int(fetched_request_count),
        }
    if enforce_endpoint_audit and not endpoint_audit["passed"]:
        raise FundamentalFetchAuditError(manifest)
    return tables, manifest


def fetch_tushare_fundamental_full_rebuild(
    symbols: Sequence[str],
    *,
    canonical_scope_path: str | Path,
    canonical_market_pointer_path: str | Path,
    canonical_membership_path: str | Path,
    years: int,
    as_of: str,
    workers: int,
    pro: Any,
    checkpoint_root: str | Path,
    checkpoint_batch_size: int = 500,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 0.5,
    max_retry_backoff_seconds: float = 8.0,
    requests_per_second: float = 8.0,
    endpoint_audit_policy: FundamentalEndpointAuditPolicy | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fetch an authoritative, resumable full-scope Tushare input bundle.

    This primitive does not write or publish a fundamental generation.  It
    raises before returning data when request-level audit thresholds fail.
    """

    requested_as_of = str(as_of or "").strip()
    if len(requested_as_of) != 8 or not requested_as_of.isdigit():
        raise ValueError("full rebuild as_of must be YYYYMMDD")
    requested_daily_start = (
        pd.Timestamp(pd.to_datetime(requested_as_of, format="%Y%m%d"))
        - pd.DateOffset(years=int(years))
    ).strftime("%Y%m%d")
    normalized_symbols = [
        symbol for symbol in dict.fromkeys(normalize_ts_code(value) for value in symbols) if symbol
    ]
    scope_evidence = build_canonical_scope_evidence(
        normalized_symbols,
        canonical_path=canonical_scope_path,
        market_pointer_path=canonical_market_pointer_path,
        membership_path=canonical_membership_path,
        as_of=requested_as_of,
        daily_start=requested_daily_start,
    )
    return _fetch_tushare_tables(
        normalized_symbols,
        years=years,
        as_of=requested_as_of,
        workers=workers,
        pro=pro,
        canonical_scope_evidence=scope_evidence,
        checkpoint_root=checkpoint_root,
        checkpoint_batch_size=checkpoint_batch_size,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        max_retry_backoff_seconds=max_retry_backoff_seconds,
        requests_per_second=requests_per_second,
        endpoint_audit_policy=endpoint_audit_policy,
        enforce_endpoint_audit=True,
        symbol_pause_seconds=0.0,
    )


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
    run_id: str = "",
    authoritative_full_rebuild: bool = False,
    safe_incremental_successor: bool = False,
    append_first_successor: bool = False,
    historical_taint_evidence: Sequence[Mapping[str, Any]] = (),
    income_support_dependencies: Sequence[Mapping[str, Any]] = (),
    financial_support_dependencies: Sequence[Mapping[str, Any]] = (),
    taint_analysis_dry_run: bool = False,
    audit_run_root: str | Path | None = None,
    canonical_predecessor_root: str | Path | None = None,
    expected_pointer_sha256: str = "",
    canonical_scope_path: str | Path | None = None,
    canonical_market_pointer_path: str | Path | None = None,
    canonical_pit_pointer_path: str | Path | None = None,
    canonical_membership_path: str | Path | None = None,
    history_audit_path: str | Path | None = None,
    expected_history_audit_sha256: str = "",
    checkpoint_root: str | Path | None = None,
    checkpoint_batch_size: int = 500,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 0.5,
    max_retry_backoff_seconds: float = 8.0,
    requests_per_second: float = 8.0,
) -> dict[str, Any]:
    if str(market).upper() != "CN":
        raise ValueError("fundamental-maintain currently supports CN only")
    universe_list = (
        [item.strip() for item in universes.split(",") if item.strip()]
        if isinstance(universes, str)
        else list(universes or DEFAULT_UNIVERSES)
    )
    resolved_run_id = str(run_id or "").strip() or _run_id(as_of)
    if append_first_successor and not safe_incremental_successor:
        raise ValueError(
            "append-first successor requires safe incremental successor mode"
        )
    if historical_taint_evidence and not append_first_successor:
        raise ValueError(
            "historical taint evidence requires append-first successor mode"
        )
    if (
        income_support_dependencies or financial_support_dependencies
    ) and not append_first_successor:
        raise ValueError(
            "financial support dependencies require append-first successor mode"
        )
    if taint_analysis_dry_run:
        if authoritative_full_rebuild or safe_incremental_successor:
            raise ValueError(
                "taint analysis dry-run is exclusive of rebuild and successor modes"
            )
        if raw_input_dir or raw_tables or checkpoint_root:
            raise ValueError(
                "taint analysis dry-run cannot use raw input or checkpoint-root"
            )
        if not allow_live:
            raise ValueError("taint analysis dry-run requires --allow-live")
        if [item.lower() for item in universe_list] != ["full_a"]:
            raise ValueError("taint analysis dry-run requires universes=full_a")
        required_taint_values = {
            "audit_run_root": audit_run_root,
            "canonical_predecessor_root": canonical_predecessor_root,
            "expected_pointer_sha256": expected_pointer_sha256,
            "canonical_scope_path": canonical_scope_path,
            "canonical_market_pointer_path": canonical_market_pointer_path,
            "canonical_pit_pointer_path": canonical_pit_pointer_path,
            "canonical_membership_path": canonical_membership_path,
            "history_audit_path": history_audit_path,
            "expected_history_audit_sha256": expected_history_audit_sha256,
            "run_id": resolved_run_id,
            "as_of": as_of,
        }
        missing = sorted(
            name for name, value in required_taint_values.items() if not value
        )
        if missing:
            raise ValueError(
                "taint analysis dry-run missing required values: "
                + ", ".join(missing)
            )
        retry_backoffs = tuple(
            min(
                float(max_retry_backoff_seconds),
                float(retry_backoff_seconds) * (2**attempt),
            )
            for attempt in range(max(0, int(max_attempts) - 1))
        )
        from .fundamental_successor import run_cn_fundamental_taint_dry_run

        return run_cn_fundamental_taint_dry_run(
            as_of=as_of,
            run_id=resolved_run_id,
            audit_run_root=audit_run_root,
            canonical_root=canonical_predecessor_root,
            expected_pointer_sha256=expected_pointer_sha256,
            canonical_market_pointer_path=canonical_market_pointer_path,
            canonical_pit_pointer_path=canonical_pit_pointer_path,
            canonical_membership_path=canonical_membership_path,
            canonical_scope_path=canonical_scope_path,
            history_audit_path=history_audit_path,
            expected_history_audit_sha256=expected_history_audit_sha256,
            allow_live=allow_live,
            universes=universe_list,
            max_attempts=int(max_attempts),
            retry_backoff_seconds=retry_backoffs,
            requests_per_second=float(requests_per_second),
            client=pro,
        )
    if safe_incremental_successor:
        if authoritative_full_rebuild:
            raise ValueError(
                "safe incremental successor and authoritative full rebuild are mutually exclusive"
            )
        if raw_input_dir or raw_tables:
            raise ValueError(
                "safe incremental successor cannot use offline raw input"
            )
        required_safe_values = {
            "canonical_predecessor_root": canonical_predecessor_root,
            "expected_pointer_sha256": expected_pointer_sha256,
            "canonical_scope_path": canonical_scope_path,
            "canonical_market_pointer_path": canonical_market_pointer_path,
            "canonical_pit_pointer_path": canonical_pit_pointer_path,
            "canonical_membership_path": canonical_membership_path,
            "history_audit_path": history_audit_path,
            "expected_history_audit_sha256": expected_history_audit_sha256,
            "checkpoint_root": checkpoint_root,
            "run_id": resolved_run_id,
            "as_of": as_of,
        }
        missing = sorted(
            name for name, value in required_safe_values.items() if not value
        )
        if missing:
            raise ValueError(
                "safe incremental successor missing required values: "
                + ", ".join(missing)
            )
        retry_backoffs = tuple(
            min(
                float(max_retry_backoff_seconds),
                float(retry_backoff_seconds) * (2**attempt),
            )
            for attempt in range(max(0, int(max_attempts) - 1))
        )
        from .fundamental_successor import run_cn_fundamental_safe_successor

        return run_cn_fundamental_safe_successor(
            as_of=as_of,
            run_id=resolved_run_id,
            staging_root=data_root,
            canonical_root=canonical_predecessor_root,
            expected_pointer_sha256=expected_pointer_sha256,
            canonical_market_pointer_path=canonical_market_pointer_path,
            canonical_pit_pointer_path=canonical_pit_pointer_path,
            canonical_membership_path=canonical_membership_path,
            canonical_scope_path=canonical_scope_path,
            history_audit_path=history_audit_path,
            expected_history_audit_sha256=expected_history_audit_sha256,
            support_fileset_root=checkpoint_root,
            allow_live=allow_live,
            universes=universe_list,
            max_attempts=int(max_attempts),
            retry_backoff_seconds=retry_backoffs,
            requests_per_second=float(requests_per_second),
            client=pro,
            append_first=append_first_successor,
            historical_taint_evidence=historical_taint_evidence,
            income_support_dependencies=income_support_dependencies,
            financial_support_dependencies=financial_support_dependencies,
        )
    if authoritative_full_rebuild:
        if not allow_live:
            raise ValueError("authoritative full rebuild requires --allow-live")
        if raw_input_dir or raw_tables:
            raise ValueError("authoritative full rebuild cannot use offline raw input")
        if [item.lower() for item in universe_list] != ["full_a"]:
            raise ValueError("authoritative full rebuild requires universes=full_a")
        _normalize_fetch_as_of(as_of)
        if (
            canonical_scope_path is None
            or canonical_market_pointer_path is None
            or canonical_membership_path is None
        ):
            raise ValueError(
                "authoritative full rebuild requires canonical scope, market pointer, "
                "and PIT membership paths"
            )
        if checkpoint_root is None:
            raise ValueError("authoritative full rebuild requires a checkpoint root")
        staging_base = _resolve_data_base(data_root).expanduser()
        canonical_base = DEFAULT_FUNDAMENTAL_ROOT.expanduser()
        if staging_base.resolve() == canonical_base.resolve(strict=True):
            raise ValueError("authoritative full rebuild must use an isolated staging data root")
        staging_pointer = staging_base / "_fundamental_latest.json"
        if staging_pointer.exists() or staging_pointer.is_symlink():
            raise ValueError("authoritative full rebuild staging pointer already exists")
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
        if authoritative_full_rebuild:
            raise ValueError(f"authoritative full rebuild scope unavailable: {scope_error}")
    live_tushare_attestation: _LiveTushareAttestation | None = None
    derived_tables_v3: dict[str, pd.DataFrame] | None = None
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
            if authoritative_full_rebuild:
                tables, fetch_manifest = fetch_tushare_fundamental_full_rebuild(
                    scope_symbols,
                    canonical_scope_path=canonical_scope_path,
                    canonical_market_pointer_path=canonical_market_pointer_path,
                    canonical_membership_path=canonical_membership_path,
                    years=int(years),
                    as_of=as_of,
                    workers=int(workers),
                    pro=pro,
                    checkpoint_root=checkpoint_root,
                    checkpoint_batch_size=int(checkpoint_batch_size),
                    max_attempts=int(max_attempts),
                    retry_backoff_seconds=float(retry_backoff_seconds),
                    max_retry_backoff_seconds=float(max_retry_backoff_seconds),
                    requests_per_second=float(requests_per_second),
                )
            else:
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
                    "authoritative_full_rebuild": bool(authoritative_full_rebuild),
                }
            )
            provider_status = (
                "live_tushare_partial"
                if int(provider_manifest.get("requests_failed", 0)) > 0
                else "live_tushare"
            )
            provider_manifest["provider_status"] = provider_status
            if authoritative_full_rebuild:
                derivation = dict(provider_manifest.get("derivation", {}) or {})
                membership_path = Path(
                    str(derivation.get("pit_membership_path") or "")
                )
                membership_bytes = _stable_regular_file_bytes(
                    membership_path,
                    label="v3 derivation PIT membership",
                )
                derivation_timestamp = _now_utc().isoformat().replace("+00:00", "Z")
                scope_evidence = dict(
                    provider_manifest.get("canonical_scope_evidence", {}) or {}
                )
                derived_tables_v3, derivation_evidence = rederive_fundamental_tables_v3(
                    tables,
                    membership_bytes=membership_bytes,
                    membership_sha256=str(
                        derivation.get("pit_membership_sha256") or ""
                    ),
                    as_of=_normalize_fetch_as_of(as_of),
                    symbols=scope_symbols,
                    non_blocking_absent_symbols=list(
                        scope_evidence.get("non_blocking_absent_symbols", []) or []
                    ),
                    run_id=resolved_run_id,
                    source=provider_status,
                    derivation_timestamp=derivation_timestamp,
                )
                provider_manifest["derivation"] = {
                    **derivation,
                    **derivation_evidence,
                }
                provider_manifest["raw_to_derived_binding_sha256"] = (
                    canonical_json_sha256(provider_manifest["derivation"])
                )
            live_tushare_attestation = _issue_live_tushare_attestation(
                provider_status,
                provider_manifest,
                tables,
            )
    if authoritative_full_rebuild and pro is None:
        raise RuntimeError(f"authoritative full rebuild provider unavailable: {provider_status}")
    if not tables:
        tables = {table: pd.DataFrame() for table in SOURCE_TABLES}
    published_pointer: dict[str, Any] = {}
    artifacts, readiness = write_fundamental_mart(
        tables,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        reports_root=reports_root,
        run_id=resolved_run_id,
        source=provider_status,
        provider_manifest=provider_manifest,
        write_raw_snapshots=not authoritative_full_rebuild,
        require_expected_symbol_scope=True,
        publish_on_gate_failure=False,
        _live_tushare_attestation=live_tushare_attestation,
        _derived_tables_v3=derived_tables_v3,
        _published_pointer_out=published_pointer,
    )
    pointer = published_pointer
    if authoritative_full_rebuild and (
        pointer is None
        or pointer.get("generation_id") != resolved_run_id
        or pointer.get("primary_provenance_verified") is not True
        or dict(pointer.get("metadata", {}) or {}).get("gate2_passed") is not True
    ):
        raise ValueError("authoritative full rebuild generation readback failed")
    return {
        "run_id": resolved_run_id,
        "provider_status": provider_status,
        "universes": universe_list,
        "years": int(years),
        "as_of": as_of,
        "artifacts": {key: str(value) for key, value in artifacts.__dict__.items() if key.endswith("_path")},
        "readiness": readiness,
        "generation_id": str(dict(pointer or {}).get("generation_id") or ""),
        "primary_provenance_verified": bool(
            dict(pointer or {}).get("primary_provenance_verified") is True
        ),
    }


__all__ = [
    "DEFAULT_FUNDAMENTAL_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "DEFAULT_READINESS_ROOT",
    "DEFAULT_UNIVERSES",
    "DERIVED_DAILY_FIELDS",
    "DERIVED_PERIOD_FIELDS",
    "FundamentalEndpointAuditPolicy",
    "FundamentalFetchAuditError",
    "FundamentalFetchCheckpointError",
    "FundamentalReadinessError",
    "FundamentalMartArtifacts",
    "build_canonical_scope_evidence",
    "build_fundamental_daily",
    "build_readiness_payload",
    "derive_fundamental_period",
    "fetch_tushare_fundamental_full_rebuild",
    "rederive_fundamental_tables_v3",
    "run_cn_fundamental_maintenance",
    "write_fundamental_mart",
]
