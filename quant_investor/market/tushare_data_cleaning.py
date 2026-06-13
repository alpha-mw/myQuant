"""Offline Tushare CSV cleaning, factor-readiness, and storage audit helpers.

This module deliberately contains no Tushare client imports.  It operates on
local pandas frames and files so the download layer can preserve raw data,
promote cleaned CSVs atomically, and emit sidecars for factor research.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.tushare_cleaning_types import (
    CLEANING_ISSUE_BLOCKER,
    CLEANING_ISSUE_CONFLICTING_DUPLICATE_PRIMARY_KEY,
    CLEANING_ISSUE_DUPLICATE_PRIMARY_KEY,
    CLEANING_ISSUE_EMPTY_ROW,
    CLEANING_ISSUE_INVALID_DATE,
    CLEANING_ISSUE_INVALID_NUMERIC,
    CLEANING_ISSUE_INVALID_OHLC_RELATION,
    CLEANING_ISSUE_INVALID_PRICE,
    CLEANING_ISSUE_INVALID_TS_CODE,
    CLEANING_ISSUE_LOW_COVERAGE,
    CLEANING_ISSUE_MISSING_REQUIRED_COLUMN,
    CLEANING_ISSUE_NEGATIVE_AMOUNT,
    CLEANING_ISSUE_NEGATIVE_VOLUME,
    CLEANING_ISSUE_QUARANTINED_ROW,
    CLEANING_ISSUE_WARNING,
    CLEANING_STATUS_FAIL,
    CLEANING_STATUS_PASS,
    CLEANING_STATUS_WARN,
    DEFAULT_CLEANING_REPORT_DIR,
    DEFAULT_FACTOR_READINESS_DIR,
    DEFAULT_PARQUET_SHADOW_DIR,
    DEFAULT_QUARANTINE_DIR,
    DEFAULT_RAW_BACKUP_DIR,
    FACTOR_READINESS_INSUFFICIENT_DATA,
    FACTOR_READINESS_NOT_READY,
    FACTOR_READINESS_READY,
    FACTOR_READINESS_WARN,
    PARQUET_STATUS_FAILED,
    PARQUET_STATUS_SHADOW_WRITTEN,
    PARQUET_STATUS_SKIPPED,
    PARQUET_STATUS_SUPPORTED,
    PARQUET_STATUS_UNSUPPORTED,
    STORAGE_ISSUE_PARQUET_BACKEND_MISSING,
    STORAGE_STATUS_EFFICIENT,
    STORAGE_STATUS_FAIL,
    STORAGE_STATUS_REDUNDANT,
    STORAGE_STATUS_WARN,
    TUSHARE_FACTOR_READINESS_SCHEMA_VERSION,
    FactorReadinessConfig,
    FactorReadinessIssue,
    FactorReadinessReport,
    FactorReadyMaskManifest,
    MatrixCoverageSummary,
    ParquetMigrationReport,
    TushareCleanProfile,
    TushareCleaningIssue,
    TushareCleaningReport,
    TushareStorageAuditReport,
    TushareStorageOptimizationConfig,
    _now_iso,
    _primary_key_payload,
    _json_ready,
    _safe_stem,
    _stable_json,
    _timestamp_slug,
)
from quant_investor.market.tushare_cleaning_core import (
    _build_cleaning_report,
    _cell_flag,
    _issue,
    _normalize_date_value,
    _resolve_profile,
    _valid_ts_code,
)
from quant_investor.market.tushare_cleaning_profiles import get_default_tushare_clean_profiles
from quant_investor.market.tushare_cleaning_storage import (
    atomic_write_dataframe_csv,
    build_storage_audit_report,
    detect_parquet_backend,
    safe_json_dump,
    sha256_file,
    write_parquet_shadow_if_supported,
)
from quant_investor.market.tushare_factor_readiness import (
    build_factor_readiness_report,
    build_factor_ready_mask_manifest,
    build_matrix_coverage_summary,
)

ROW_FLAGS_COMPACTION_SCHEMA = "myquant.uniform_row_flags_compaction.v1"
ROW_FLAGS_COMPACTION_MIN_ROWS = 100


def _uniform_row_flags_compaction_metadata(
    row_flags: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if (
        row_flags is None
        or len(row_flags) < ROW_FLAGS_COMPACTION_MIN_ROWS
        or "row_index" not in row_flags.columns
    ):
        return None
    row_indices = pd.to_numeric(row_flags["row_index"], errors="coerce")
    if row_indices.isna().any():
        return None
    if row_indices.astype(int).tolist() != list(range(len(row_flags))):
        return None

    uniform_values: dict[str, Any] = {}
    for column in row_flags.columns:
        if column == "row_index":
            continue
        values = row_flags[column].drop_duplicates()
        if len(values) != 1:
            return None
        value = _json_ready(values.iloc[0])
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except (AttributeError, ValueError, TypeError):
                pass
        uniform_values[str(column)] = _json_ready(value)
    if not uniform_values:
        return None
    return {
        "row_flags_compaction_schema": ROW_FLAGS_COMPACTION_SCHEMA,
        "row_flags_row_count": int(len(row_flags)),
        "row_flags_columns": [str(column) for column in row_flags.columns],
        "row_flags_uniform_values": uniform_values,
    }


def clean_tushare_dataframe(
    df: Any,
    *,
    table_name: str = "daily",
    profile: TushareCleanProfile | None = None,
    source_path: str | Path | None = None,
    raw_backup_path: str | Path | None = None,
    cleaned_path: str | Path | None = None,
    parquet_shadow_path: str | Path | None = None,
    quarantine_path: str | Path | None = None,
    row_flags_path: str | Path | None = None,
    cell_flags_path: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, TushareCleaningReport]:
    resolved_profile = _resolve_profile(table_name, profile)
    generated = generated_at or _now_iso()
    meta = dict(metadata or {})
    raw = (df.copy(deep=True) if isinstance(df, pd.DataFrame) else pd.DataFrame(df).copy(deep=True))
    work = raw.copy(deep=True)
    work.columns = [str(column).strip() for column in work.columns]
    if resolved_profile.table_name == "daily" and "ts_code" not in work.columns and meta.get("symbol"):
        work["ts_code"] = str(meta["symbol"]).strip().upper()
    work["_source_row_index"] = list(range(len(work)))

    issues: list[TushareCleaningIssue] = []
    cell_flags: list[dict[str, Any]] = []
    missing_required = [
        column for column in resolved_profile.required_columns if column not in work.columns
    ]
    for column in missing_required:
        issues.append(
            _issue(
                table_name=resolved_profile.table_name,
                issue_code=CLEANING_ISSUE_MISSING_REQUIRED_COLUMN,
                severity=CLEANING_ISSUE_BLOCKER,
                message=f"required column missing: {column}",
                column=column,
            )
        )

    row_flag_records: list[dict[str, Any]] = []
    if missing_required:
        row_flags = pd.DataFrame(
            {
                "row_index": list(range(len(raw))),
                "missing_required_column": True,
                "quarantined": False,
                "dropped": False,
            }
        )
        report = _build_cleaning_report(
            profile=resolved_profile,
            generated_at=generated,
            source_path=source_path,
            raw_backup_path=raw_backup_path,
            cleaned_path=None,
            parquet_shadow_path=parquet_shadow_path,
            quarantine_path=quarantine_path,
            row_flags_path=row_flags_path,
            cell_flags_path=cell_flags_path,
            raw_row_count=len(raw),
            cleaned_row_count=0,
            dropped_row_count=0,
            quarantined_row_count=0,
            duplicate_row_count=0,
            conflicting_duplicate_count=0,
            latest_trade_date=None,
            issues=issues,
            metadata=meta,
        )
        return pd.DataFrame(columns=work.columns.drop("_source_row_index", errors="ignore")), None, row_flags, pd.DataFrame(cell_flags), report

    for column in resolved_profile.ts_code_columns:
        if column not in work.columns:
            continue
        work[column] = work[column].astype("string").str.strip().str.upper()
        invalid_mask = ~work[column].map(_valid_ts_code)
        invalid_mask = invalid_mask.fillna(True)
        for idx, row in work[invalid_mask].iterrows():
            row_index = int(row["_source_row_index"])
            pk = _primary_key_payload(row, resolved_profile.primary_key)
            message = f"invalid ts_code in {column}"
            issues.append(
                _issue(
                    table_name=resolved_profile.table_name,
                    issue_code=CLEANING_ISSUE_INVALID_TS_CODE,
                    severity=CLEANING_ISSUE_WARNING,
                    message=message,
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                )
            )
            cell_flags.append(
                _cell_flag(
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                    issue_code=CLEANING_ISSUE_INVALID_TS_CODE,
                    severity=CLEANING_ISSUE_WARNING,
                    value=row.get(column),
                    message=message,
                )
            )

    for column in resolved_profile.date_columns:
        if column not in work.columns:
            continue
        original = work[column].copy()
        normalized = original.map(_normalize_date_value)
        invalid_mask = normalized.isna() & original.notna() & (original.astype(str).str.strip() != "")
        work[column] = normalized
        for idx, row in work[invalid_mask].iterrows():
            row_index = int(row["_source_row_index"])
            pk = _primary_key_payload(row, resolved_profile.primary_key)
            message = f"invalid date in {column}"
            issues.append(
                _issue(
                    table_name=resolved_profile.table_name,
                    issue_code=CLEANING_ISSUE_INVALID_DATE,
                    severity=CLEANING_ISSUE_WARNING,
                    message=message,
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                )
            )
            cell_flags.append(
                _cell_flag(
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                    issue_code=CLEANING_ISSUE_INVALID_DATE,
                    severity=CLEANING_ISSUE_WARNING,
                    value=original.loc[idx],
                    message=message,
                )
            )

    for column in resolved_profile.numeric_columns:
        if column not in work.columns:
            continue
        original = work[column].copy()
        work[column] = pd.to_numeric(work[column], errors="coerce")
        invalid_mask = work[column].isna() & original.notna() & (original.astype(str).str.strip() != "")
        for idx, row in work[invalid_mask].iterrows():
            row_index = int(row["_source_row_index"])
            pk = _primary_key_payload(row, resolved_profile.primary_key)
            message = f"invalid numeric value in {column}"
            issues.append(
                _issue(
                    table_name=resolved_profile.table_name,
                    issue_code=CLEANING_ISSUE_INVALID_NUMERIC,
                    severity=CLEANING_ISSUE_WARNING,
                    message=message,
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                )
            )
            cell_flags.append(
                _cell_flag(
                    row_index=row_index,
                    primary_key=pk,
                    column=column,
                    issue_code=CLEANING_ISSUE_INVALID_NUMERIC,
                    severity=CLEANING_ISSUE_WARNING,
                    value=original.loc[idx],
                    message=message,
                )
            )

    empty_row_mask = work.drop(columns=["_source_row_index"], errors="ignore").isna().all(axis=1)
    invalid_date_mask = pd.Series(False, index=work.index)
    for column in resolved_profile.date_columns:
        if column in resolved_profile.primary_key and column in work.columns:
            invalid_date_mask = invalid_date_mask | work[column].isna()
    invalid_ts_mask = pd.Series(False, index=work.index)
    for column in resolved_profile.ts_code_columns:
        if column in resolved_profile.primary_key and column in work.columns:
            invalid_ts_mask = invalid_ts_mask | ~work[column].map(_valid_ts_code).fillna(False)

    invalid_price_mask = pd.Series(False, index=work.index)
    for column in resolved_profile.price_columns:
        if column in work.columns:
            invalid_price_mask = invalid_price_mask | (work[column].notna() & (work[column] <= 0))
    negative_volume_mask = pd.Series(False, index=work.index)
    for column in resolved_profile.volume_columns:
        if column in work.columns:
            negative_volume_mask = negative_volume_mask | (work[column].notna() & (work[column] < 0))
    negative_amount_mask = pd.Series(False, index=work.index)
    for column in resolved_profile.amount_columns:
        if column in work.columns:
            negative_amount_mask = negative_amount_mask | (work[column].notna() & (work[column] < 0))

    invalid_ohlc_mask = pd.Series(False, index=work.index)
    ohlc = resolved_profile.ohlc_columns
    if ohlc and all(column in work.columns for column in ohlc.values()):
        open_col = ohlc.get("open", "open")
        high_col = ohlc.get("high", "high")
        low_col = ohlc.get("low", "low")
        close_col = ohlc.get("close", "close")
        required_ohlc = work[[open_col, high_col, low_col, close_col]]
        full_ohlc = required_ohlc.notna().all(axis=1)
        invalid_ohlc_mask = full_ohlc & (
            (work[high_col] < work[[open_col, close_col, low_col]].max(axis=1))
            | (work[low_col] > work[[open_col, close_col, high_col]].min(axis=1))
            | (work[high_col] < work[low_col])
        )

    def _record_row_issue(mask: pd.Series, code: str, message: str) -> None:
        for _idx, row in work[mask].iterrows():
            row_index = int(row["_source_row_index"])
            pk = _primary_key_payload(row, resolved_profile.primary_key)
            issues.append(
                _issue(
                    table_name=resolved_profile.table_name,
                    issue_code=code,
                    severity=CLEANING_ISSUE_WARNING,
                    message=message,
                    row_index=row_index,
                    primary_key=pk,
                )
            )

    _record_row_issue(empty_row_mask, CLEANING_ISSUE_EMPTY_ROW, "empty row quarantined")
    _record_row_issue(invalid_price_mask, CLEANING_ISSUE_INVALID_PRICE, "invalid non-positive price")
    _record_row_issue(negative_volume_mask, CLEANING_ISSUE_NEGATIVE_VOLUME, "negative volume")
    _record_row_issue(negative_amount_mask, CLEANING_ISSUE_NEGATIVE_AMOUNT, "negative amount")
    _record_row_issue(invalid_ohlc_mask, CLEANING_ISSUE_INVALID_OHLC_RELATION, "invalid OHLC relation")

    duplicate_mask = pd.Series(False, index=work.index)
    conflicting_mask = pd.Series(False, index=work.index)
    if resolved_profile.allow_deduplicate and all(column in work.columns for column in resolved_profile.primary_key):
        key_valid = work[resolved_profile.primary_key].notna().all(axis=1)
        duplicate_mask = key_valid & work.duplicated(subset=resolved_profile.primary_key, keep=False)
        non_key_columns = [
            column
            for column in work.columns
            if column not in set(resolved_profile.primary_key) | {"_source_row_index"}
        ]
        for _key, group in work[key_valid & duplicate_mask].groupby(resolved_profile.primary_key, dropna=False):
            if not non_key_columns:
                continue
            comparable = group[non_key_columns].fillna("<NA>").astype(str)
            if len(comparable.drop_duplicates()) > 1:
                conflicting_mask.loc[group.index] = True
        for _idx, row in work[duplicate_mask].iterrows():
            row_index = int(row["_source_row_index"])
            pk = _primary_key_payload(row, resolved_profile.primary_key)
            code = (
                CLEANING_ISSUE_CONFLICTING_DUPLICATE_PRIMARY_KEY
                if bool(conflicting_mask.loc[_idx])
                else CLEANING_ISSUE_DUPLICATE_PRIMARY_KEY
            )
            issues.append(
                _issue(
                    table_name=resolved_profile.table_name,
                    issue_code=code,
                    severity=CLEANING_ISSUE_WARNING,
                    message="duplicate primary key encountered",
                    row_index=row_index,
                    primary_key=pk,
                )
            )

    quarantine_mask = (
        empty_row_mask
        | invalid_date_mask
        | invalid_ts_mask
        | invalid_price_mask
        | (negative_volume_mask if resolved_profile.quarantine_invalid_volume_amount else False)
        | (negative_amount_mask if resolved_profile.quarantine_invalid_volume_amount else False)
        | (invalid_ohlc_mask if resolved_profile.quarantine_invalid_ohlc else False)
    )
    quarantine_reasons: list[str] = []
    for idx in work.index:
        reasons = []
        if bool(empty_row_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_EMPTY_ROW)
        if bool(invalid_date_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_INVALID_DATE)
        if bool(invalid_ts_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_INVALID_TS_CODE)
        if bool(invalid_price_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_INVALID_PRICE)
        if bool(negative_volume_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_NEGATIVE_VOLUME)
        if bool(negative_amount_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_NEGATIVE_AMOUNT)
        if bool(invalid_ohlc_mask.loc[idx]):
            reasons.append(CLEANING_ISSUE_INVALID_OHLC_RELATION)
        quarantine_reasons.append(",".join(reasons))
        pk = _primary_key_payload(work.loc[idx], resolved_profile.primary_key)
        row_flag_records.append(
            {
                "row_index": int(work.loc[idx, "_source_row_index"]),
                "primary_key_json": _stable_json(pk),
                "is_empty_row": bool(empty_row_mask.loc[idx]),
                "invalid_date": bool(invalid_date_mask.loc[idx]),
                "invalid_ts_code": bool(invalid_ts_mask.loc[idx]),
                "invalid_price": bool(invalid_price_mask.loc[idx]),
                "invalid_ohlc": bool(invalid_ohlc_mask.loc[idx]),
                "negative_volume": bool(negative_volume_mask.loc[idx]),
                "negative_amount": bool(negative_amount_mask.loc[idx]),
                "duplicate_primary_key": bool(duplicate_mask.loc[idx]),
                "conflicting_duplicate_primary_key": bool(conflicting_mask.loc[idx]),
                "quarantined": bool(quarantine_mask.loc[idx]),
                "dropped": False,
                "factor_candidate_row": False,
            }
        )

    quarantined = work[quarantine_mask].copy()
    if not quarantined.empty:
        quarantined["quarantine_reasons"] = [
            quarantine_reasons[pos]
            for pos, idx in enumerate(work.index)
            if bool(quarantine_mask.loc[idx])
        ]
        issues.append(
            _issue(
                table_name=resolved_profile.table_name,
                issue_code=CLEANING_ISSUE_QUARANTINED_ROW,
                severity=CLEANING_ISSUE_WARNING,
                message=f"{len(quarantined)} rows quarantined",
                metadata={"quarantined_row_count": len(quarantined)},
            )
        )
    cleaned = work[~quarantine_mask].copy()
    dropped_row_count = 0
    if resolved_profile.allow_deduplicate and all(column in cleaned.columns for column in resolved_profile.primary_key):
        before_dedupe = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=resolved_profile.primary_key, keep="last")
        dropped_row_count += before_dedupe - len(cleaned)
    if all(column in cleaned.columns for column in resolved_profile.primary_key):
        cleaned = cleaned.sort_values(resolved_profile.primary_key).reset_index(drop=True)
    else:
        sort_cols = [column for column in resolved_profile.date_columns if column in cleaned.columns]
        if sort_cols:
            cleaned = cleaned.sort_values(sort_cols).reset_index(drop=True)

    helper_columns = ["_source_row_index"]
    cleaned = cleaned.drop(columns=helper_columns, errors="ignore")
    if not quarantined.empty:
        quarantined = quarantined.drop(columns=helper_columns, errors="ignore")

    factor_required_present = [
        column for column in resolved_profile.factor_required_columns if column in cleaned.columns
    ]
    if factor_required_present:
        eligible_keys = set()
        for _idx, row in cleaned.iterrows():
            if row[factor_required_present].notna().all():
                eligible_keys.add(_stable_json(_primary_key_payload(row, resolved_profile.primary_key)))
        for record in row_flag_records:
            record["factor_candidate_row"] = (
                bool(record.get("primary_key_json") in eligible_keys)
                and not bool(record.get("quarantined"))
                and not bool(record.get("dropped"))
            )

    for record in row_flag_records:
        if bool(record["duplicate_primary_key"]) and not bool(record["quarantined"]):
            pk = json.loads(str(record["primary_key_json"]))
            matching = cleaned
            for key, value in pk.items():
                if key in matching.columns:
                    matching = matching[matching[key].astype(str) == str(value)]
            if len(matching) == 0:
                record["dropped"] = True

    row_flags = pd.DataFrame(row_flag_records)
    cell_flags_df = pd.DataFrame(cell_flags)
    latest_trade_date = None
    if "trade_date" in cleaned.columns and not cleaned.empty:
        latest_trade_date = str(cleaned["trade_date"].dropna().max())
    elif resolved_profile.date_columns:
        for column in resolved_profile.date_columns:
            if column in cleaned.columns and not cleaned.empty:
                latest_trade_date = str(cleaned[column].dropna().max())
                break

    if len(raw) > 0 and cleaned.empty:
        issues.append(
            _issue(
                table_name=resolved_profile.table_name,
                issue_code=CLEANING_ISSUE_LOW_COVERAGE,
                severity=CLEANING_ISSUE_BLOCKER,
                message="all rows were invalid or quarantined; canonical file not safe to promote",
                metadata={"raw_row_count": len(raw)},
            )
        )

    report = _build_cleaning_report(
        profile=resolved_profile,
        generated_at=generated,
        source_path=source_path,
        raw_backup_path=raw_backup_path,
        cleaned_path=cleaned_path,
        parquet_shadow_path=parquet_shadow_path,
        quarantine_path=quarantine_path,
        row_flags_path=row_flags_path,
        cell_flags_path=cell_flags_path,
        raw_row_count=len(raw),
        cleaned_row_count=len(cleaned),
        dropped_row_count=dropped_row_count,
        quarantined_row_count=len(quarantined),
        duplicate_row_count=int(duplicate_mask.sum()),
        conflicting_duplicate_count=int(conflicting_mask.sum()),
        latest_trade_date=latest_trade_date,
        issues=issues,
        metadata=meta,
    )
    return cleaned, (quarantined if not quarantined.empty else None), row_flags, cell_flags_df, report


def _artifact_paths(
    *,
    canonical_path: Path,
    table_name: str,
    generated_at: str,
    metadata: Mapping[str, Any],
    raw_backup_dir: str | Path | None,
    quarantine_dir: str | Path | None,
    report_dir: str | Path | None,
    factor_readiness_dir: str | Path | None,
    parquet_dir: str | Path | None,
) -> dict[str, Path]:
    category = _safe_stem(str(metadata.get("category") or canonical_path.parent.name or "uncategorized"))
    stem = _safe_stem(str(metadata.get("symbol") or canonical_path.stem))
    stamp = _timestamp_slug(generated_at)
    file_stem = f"{category}_{stem}_{stamp}"
    report_root = Path(report_dir or DEFAULT_CLEANING_REPORT_DIR) / table_name
    readiness_root = Path(factor_readiness_dir or DEFAULT_FACTOR_READINESS_DIR) / table_name
    paths = {
        "raw_backup": Path(raw_backup_dir or DEFAULT_RAW_BACKUP_DIR) / table_name / f"{file_stem}_raw.csv",
        "quarantine": Path(quarantine_dir or DEFAULT_QUARANTINE_DIR) / table_name / f"{file_stem}_quarantine.csv",
        "row_flags": report_root / f"{file_stem}_row_flags.csv",
        "cell_flags": report_root / f"{file_stem}_cell_flags.csv",
        "cleaning_report": report_root / f"{file_stem}_cleaning_report.json",
        "factor_readiness_report": readiness_root / f"{file_stem}_factor_readiness_report.json",
        "matrix_coverage": readiness_root / f"{file_stem}_matrix_coverage.json",
        "factor_ready_masks": readiness_root / f"{file_stem}_factor_ready_masks.json",
        "storage_audit_report": report_root / "storage" / f"{file_stem}_storage_audit_report.json",
        "parquet_migration_report": report_root / "storage" / f"{file_stem}_parquet_migration_report.json",
        "parquet_shadow": Path(parquet_dir or DEFAULT_PARQUET_SHADOW_DIR) / category / f"{stem}.parquet",
    }
    return paths


def clean_tushare_dataframe_to_file(
    df: Any,
    *,
    canonical_path: str | Path,
    table_name: str = "daily",
    profile: TushareCleanProfile | None = None,
    promote: bool = True,
    raw_backup_dir: str | Path | None = None,
    quarantine_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    factor_readiness_dir: str | Path | None = None,
    enable_factor_readiness: bool = True,
    enable_storage_audit: bool = True,
    storage_config: TushareStorageOptimizationConfig | None = None,
    factor_readiness_config: FactorReadinessConfig | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    generated = generated_at or _now_iso()
    meta = dict(metadata or {})
    target = Path(canonical_path)
    resolved_storage_config = storage_config or TushareStorageOptimizationConfig()
    paths = _artifact_paths(
        canonical_path=target,
        table_name=table_name,
        generated_at=generated,
        metadata=meta,
        raw_backup_dir=raw_backup_dir,
        quarantine_dir=quarantine_dir,
        report_dir=report_dir,
        factor_readiness_dir=factor_readiness_dir,
        parquet_dir=resolved_storage_config.parquet_dir,
    )
    raw_frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    raw_backup_path = atomic_write_dataframe_csv(raw_frame, paths["raw_backup"])
    cleaned, quarantined, row_flags, cell_flags, report = clean_tushare_dataframe(
        raw_frame,
        table_name=table_name,
        profile=profile,
        source_path=target,
        raw_backup_path=raw_backup_path,
        cleaned_path=target if promote else None,
        parquet_shadow_path=paths["parquet_shadow"] if resolved_storage_config.parquet_shadow_write else None,
        quarantine_path=paths["quarantine"],
        row_flags_path=paths["row_flags"],
        cell_flags_path=paths["cell_flags"],
        generated_at=generated,
        metadata=meta,
    )
    report.raw_hash = sha256_file(raw_backup_path)

    if quarantined is not None and not quarantined.empty:
        atomic_write_dataframe_csv(quarantined, paths["quarantine"])
    else:
        report.quarantine_path = None
    if row_flags is not None:
        row_flags_compaction = _uniform_row_flags_compaction_metadata(row_flags)
        if row_flags_compaction:
            report.row_flags_path = None
            report.metadata.update(
                {
                    "row_flags_compacted": True,
                    "row_flags_path_suppressed": True,
                    "row_flags_planned_path": str(paths["row_flags"]),
                    **row_flags_compaction,
                }
            )
        else:
            atomic_write_dataframe_csv(row_flags, paths["row_flags"])
            report.metadata.update(
                {
                    "row_flags_compacted": False,
                    "row_flags_path_suppressed": False,
                }
            )
    if cell_flags is not None:
        if cell_flags.empty:
            report.cell_flags_path = None
            report.metadata.update(
                {
                    "cell_flags_empty": True,
                    "cell_flags_path_suppressed": True,
                    "cell_flags_planned_path": str(paths["cell_flags"]),
                }
            )
        else:
            atomic_write_dataframe_csv(cell_flags, paths["cell_flags"])
            report.metadata.update(
                {
                    "cell_flags_empty": False,
                    "cell_flags_path_suppressed": False,
                }
            )

    parquet_report: ParquetMigrationReport | None = None
    if report.status != CLEANING_STATUS_FAIL and promote:
        atomic_write_dataframe_csv(cleaned, target)
        report.cleaned_path = str(target)
        report.cleaned_hash = sha256_file(target)
        if resolved_storage_config.parquet_shadow_write or resolved_storage_config.parquet_canonical:
            parquet_report = write_parquet_shadow_if_supported(
                cleaned,
                table_name=table_name,
                csv_path=target,
                parquet_path=paths["parquet_shadow"],
                config=resolved_storage_config,
                generated_at=generated,
                metadata=meta,
            )
            safe_json_dump(parquet_report.to_dict(), paths["parquet_migration_report"])
            if parquet_report.status == PARQUET_STATUS_SHADOW_WRITTEN:
                report.parquet_shadow_path = parquet_report.parquet_path
                if parquet_report.parquet_path:
                    report.parquet_hash = sha256_file(parquet_report.parquet_path)
        elif resolved_storage_config.parquet_canonical:
            report.issues.append(
                _issue(
                    table_name=table_name,
                    issue_code=STORAGE_ISSUE_PARQUET_BACKEND_MISSING,
                    severity=CLEANING_ISSUE_WARNING,
                    message="canonical parquet requested but unavailable; CSV retained",
                )
            )

    readiness_report: FactorReadinessReport | None = None
    if enable_factor_readiness and report.status != CLEANING_STATUS_FAIL:
        readiness_report = build_factor_readiness_report(
            table_reports={table_name: report},
            cleaned_frames={table_name: cleaned},
            quarantined_frames={table_name: quarantined} if quarantined is not None else {},
            config=factor_readiness_config,
            source_root=str(target.parent),
            generated_at=generated,
            metadata=meta,
        )
        safe_json_dump(readiness_report.to_dict(), paths["factor_readiness_report"])
        if readiness_report.coverage_summaries:
            safe_json_dump(
                {
                    "schema_version": TUSHARE_FACTOR_READINESS_SCHEMA_VERSION,
                    "generated_at": generated,
                    "summaries": [item.to_dict() for item in readiness_report.coverage_summaries],
                },
                paths["matrix_coverage"],
            )
        if readiness_report.mask_manifests:
            safe_json_dump(
                {
                    "schema_version": TUSHARE_FACTOR_READINESS_SCHEMA_VERSION,
                    "generated_at": generated,
                    "manifests": [item.to_dict() for item in readiness_report.mask_manifests],
                },
                paths["factor_ready_masks"],
            )
        report.metadata.update(
            {
                "factor_readiness_report_path": str(paths["factor_readiness_report"]),
                "matrix_coverage_path": str(paths["matrix_coverage"]),
                "factor_ready_masks_path": str(paths["factor_ready_masks"]),
                "factor_readiness_status": readiness_report.overall_status,
            }
        )

    storage_report: TushareStorageAuditReport | None = None
    if enable_storage_audit:
        storage_report = build_storage_audit_report(
            table_name=table_name,
            csv_path=target if target.exists() else None,
            parquet_path=report.parquet_shadow_path,
            config=resolved_storage_config,
            generated_at=generated,
            metadata=meta,
        )
        safe_json_dump(storage_report.to_dict(), paths["storage_audit_report"])
        report.metadata.update(
            {
                "storage_audit_report_path": str(paths["storage_audit_report"]),
                "storage_status": storage_report.status,
            }
        )

    safe_json_dump(report.to_dict(), paths["cleaning_report"])
    return {
        "cleaned_df": cleaned,
        "quarantined_df": quarantined,
        "row_flags_df": row_flags,
        "cell_flags_df": cell_flags,
        "cleaning_report": report,
        "cleaning_report_path": str(paths["cleaning_report"]),
        "factor_readiness_report": readiness_report,
        "factor_readiness_report_path": str(paths["factor_readiness_report"]) if readiness_report else None,
        "matrix_coverage_path": str(paths["matrix_coverage"]) if readiness_report else None,
        "factor_ready_masks_path": str(paths["factor_ready_masks"]) if readiness_report else None,
        "storage_audit_report": storage_report,
        "storage_audit_report_path": str(paths["storage_audit_report"]) if storage_report else None,
        "parquet_migration_report": parquet_report,
        "parquet_migration_report_path": str(paths["parquet_migration_report"]) if parquet_report else None,
        "raw_backup_path": str(raw_backup_path),
        "quarantine_path": report.quarantine_path,
        "row_flags_path": report.row_flags_path,
        "cell_flags_path": report.cell_flags_path,
        "parquet_shadow_path": report.parquet_shadow_path,
        "cleaning_status": report.status,
        "factor_readiness_status": readiness_report.overall_status if readiness_report else None,
        "storage_status": storage_report.status if storage_report else None,
        "parquet_status": parquet_report.status if parquet_report else PARQUET_STATUS_SKIPPED,
    }


def clean_tushare_file(
    path: str | Path,
    *,
    table_name: str = "daily",
    profile: TushareCleanProfile | None = None,
    promote: bool = True,
    raw_backup_dir: str | Path | None = None,
    quarantine_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    factor_readiness_dir: str | Path | None = None,
    enable_factor_readiness: bool = True,
    enable_storage_audit: bool = True,
    storage_config: TushareStorageOptimizationConfig | None = None,
    factor_readiness_config: FactorReadinessConfig | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        supported, backend, warnings = detect_parquet_backend()
        if not supported or backend is None:
            generated = generated_at or _now_iso()
            report = _build_cleaning_report(
                profile=_resolve_profile(table_name, profile),
                generated_at=generated,
                source_path=source,
                raw_backup_path=None,
                cleaned_path=None,
                parquet_shadow_path=None,
                quarantine_path=None,
                row_flags_path=None,
                cell_flags_path=None,
                raw_row_count=0,
                cleaned_row_count=0,
                dropped_row_count=0,
                quarantined_row_count=0,
                duplicate_row_count=0,
                conflicting_duplicate_count=0,
                latest_trade_date=None,
                issues=[
                    _issue(
                        table_name=table_name,
                        issue_code=STORAGE_ISSUE_PARQUET_BACKEND_MISSING,
                        severity=CLEANING_ISSUE_BLOCKER,
                        message="parquet input requested but no backend is available",
                        metadata={"warnings": warnings},
                    )
                ],
                metadata=dict(metadata or {}),
            )
            return {"cleaning_report": report, "cleaning_status": report.status}
        frame = pd.read_parquet(source, engine=backend)
    else:
        frame = pd.read_csv(source)
    return clean_tushare_dataframe_to_file(
        frame,
        canonical_path=source,
        table_name=table_name,
        profile=profile,
        promote=promote,
        raw_backup_dir=raw_backup_dir,
        quarantine_dir=quarantine_dir,
        report_dir=report_dir,
        factor_readiness_dir=factor_readiness_dir,
        enable_factor_readiness=enable_factor_readiness,
        enable_storage_audit=enable_storage_audit,
        storage_config=storage_config,
        factor_readiness_config=factor_readiness_config,
        generated_at=generated_at,
        metadata={**dict(metadata or {}), "source_path": str(source)},
    )


def clean_tushare_download_directory(
    root_dir: str | Path,
    *,
    table_name: str = "daily",
    include: str | None = None,
    exclude: str | None = None,
    promote: bool = True,
    raw_backup_dir: str | Path | None = None,
    quarantine_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    factor_readiness_dir: str | Path | None = None,
    enable_factor_readiness: bool = True,
    enable_storage_audit: bool = True,
    storage_config: TushareStorageOptimizationConfig | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    selected: list[Path] = []
    for path in files:
        rel = str(path.relative_to(root))
        if include and not fnmatch.fnmatch(rel, include):
            continue
        if exclude and fnmatch.fnmatch(rel, exclude):
            continue
        selected.append(path)

    summary: dict[str, Any] = {
        "total_files": len(selected),
        "pass_count": 0,
        "warn_count": 0,
        "fail_count": 0,
        "raw_rows": 0,
        "clean_rows": 0,
        "quarantine_rows": 0,
        "results": [],
    }
    for path in selected:
        result = clean_tushare_file(
            path,
            table_name=table_name,
            promote=promote,
            raw_backup_dir=raw_backup_dir,
            quarantine_dir=quarantine_dir,
            report_dir=report_dir,
            factor_readiness_dir=factor_readiness_dir,
            enable_factor_readiness=enable_factor_readiness,
            enable_storage_audit=enable_storage_audit,
            storage_config=storage_config,
            metadata={"category": path.parent.name, "symbol": path.stem, "directory_clean": True},
        )
        report = result.get("cleaning_report")
        if isinstance(report, TushareCleaningReport):
            summary["raw_rows"] += report.raw_row_count
            summary["clean_rows"] += report.cleaned_row_count
            summary["quarantine_rows"] += report.quarantined_row_count
            if report.status == CLEANING_STATUS_PASS:
                summary["pass_count"] += 1
            elif report.status == CLEANING_STATUS_WARN:
                summary["warn_count"] += 1
            else:
                summary["fail_count"] += 1
        summary["results"].append(
            {
                "path": str(path),
                "cleaning_status": result.get("cleaning_status"),
                "factor_readiness_status": result.get("factor_readiness_status"),
                "storage_status": result.get("storage_status"),
                "parquet_status": result.get("parquet_status"),
                "cleaning_report_path": result.get("cleaning_report_path"),
                "raw_backup_path": result.get("raw_backup_path"),
                "quarantine_path": result.get("quarantine_path"),
                "row_flags_path": result.get("row_flags_path"),
                "cell_flags_path": result.get("cell_flags_path"),
                "factor_ready_masks_path": result.get("factor_ready_masks_path"),
                "matrix_coverage_path": result.get("matrix_coverage_path"),
                "storage_audit_report_path": result.get("storage_audit_report_path"),
                "parquet_migration_report_path": result.get("parquet_migration_report_path"),
            }
        )
    return summary


__all__ = [
    "CLEANING_STATUS_PASS",
    "CLEANING_STATUS_WARN",
    "CLEANING_STATUS_FAIL",
    "FACTOR_READINESS_READY",
    "FACTOR_READINESS_WARN",
    "FACTOR_READINESS_NOT_READY",
    "FACTOR_READINESS_INSUFFICIENT_DATA",
    "STORAGE_STATUS_EFFICIENT",
    "STORAGE_STATUS_REDUNDANT",
    "STORAGE_STATUS_WARN",
    "STORAGE_STATUS_FAIL",
    "PARQUET_STATUS_SUPPORTED",
    "PARQUET_STATUS_UNSUPPORTED",
    "PARQUET_STATUS_SHADOW_WRITTEN",
    "PARQUET_STATUS_SKIPPED",
    "PARQUET_STATUS_FAILED",
    "TushareCleanProfile",
    "TushareCleaningIssue",
    "TushareCleaningReport",
    "FactorReadinessConfig",
    "FactorReadinessIssue",
    "MatrixCoverageSummary",
    "FactorReadyMaskManifest",
    "FactorReadinessReport",
    "TushareStorageOptimizationConfig",
    "TushareStorageAuditReport",
    "ParquetMigrationReport",
    "get_default_tushare_clean_profiles",
    "clean_tushare_dataframe",
    "build_factor_ready_mask_manifest",
    "build_matrix_coverage_summary",
    "build_factor_readiness_report",
    "clean_tushare_file",
    "clean_tushare_download_directory",
    "clean_tushare_dataframe_to_file",
    "detect_parquet_backend",
    "write_parquet_shadow_if_supported",
    "build_storage_audit_report",
    "safe_json_dump",
    "sha256_file",
    "atomic_write_dataframe_csv",
]
