"""Offline Tushare CSV cleaning, factor-readiness, and storage audit helpers.

This module deliberately contains no Tushare client imports.  It operates on
local pandas frames and files so the download layer can preserve raw data,
promote cleaned CSVs atomically, and emit sidecars for factor research.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.versioning import (
    TUSHARE_DATA_CLEANING_SCHEMA_VERSION,
    TUSHARE_FACTOR_READINESS_SCHEMA_VERSION,
    TUSHARE_PARQUET_MIGRATION_SCHEMA_VERSION,
    TUSHARE_STORAGE_OPTIMIZATION_SCHEMA_VERSION,
)


CLEANING_STATUS_PASS = "pass"
CLEANING_STATUS_WARN = "warn"
CLEANING_STATUS_FAIL = "fail"

FACTOR_READINESS_READY = "ready"
FACTOR_READINESS_WARN = "warn"
FACTOR_READINESS_NOT_READY = "not_ready"
FACTOR_READINESS_INSUFFICIENT_DATA = "insufficient_data"

STORAGE_STATUS_EFFICIENT = "efficient"
STORAGE_STATUS_REDUNDANT = "redundant"
STORAGE_STATUS_WARN = "warn"
STORAGE_STATUS_FAIL = "fail"

PARQUET_STATUS_SUPPORTED = "supported"
PARQUET_STATUS_UNSUPPORTED = "unsupported"
PARQUET_STATUS_SHADOW_WRITTEN = "shadow_written"
PARQUET_STATUS_SKIPPED = "skipped"
PARQUET_STATUS_FAILED = "failed"

CLEANING_ISSUE_INFO = "info"
CLEANING_ISSUE_WARNING = "warning"
CLEANING_ISSUE_BLOCKER = "blocker"

CLEANING_ISSUE_MISSING_REQUIRED_COLUMN = "missing_required_column"
CLEANING_ISSUE_DUPLICATE_PRIMARY_KEY = "duplicate_primary_key"
CLEANING_ISSUE_CONFLICTING_DUPLICATE_PRIMARY_KEY = "conflicting_duplicate_primary_key"
CLEANING_ISSUE_INVALID_DATE = "invalid_date"
CLEANING_ISSUE_INVALID_TS_CODE = "invalid_ts_code"
CLEANING_ISSUE_INVALID_NUMERIC = "invalid_numeric"
CLEANING_ISSUE_NEGATIVE_VOLUME = "negative_volume"
CLEANING_ISSUE_NEGATIVE_AMOUNT = "negative_amount"
CLEANING_ISSUE_INVALID_PRICE = "invalid_price"
CLEANING_ISSUE_INVALID_OHLC_RELATION = "invalid_ohlc_relation"
CLEANING_ISSUE_EMPTY_ROW = "empty_row"
CLEANING_ISSUE_STALE_DATA = "stale_data"
CLEANING_ISSUE_LOW_COVERAGE = "low_coverage"
CLEANING_ISSUE_WRITE_FAILURE = "write_failure"
CLEANING_ISSUE_QUARANTINED_ROW = "quarantined_row"

READINESS_ISSUE_MISSING_TRADE_CAL = "missing_trade_cal"
READINESS_ISSUE_MISSING_REQUIRED_TABLE = "missing_required_table"
READINESS_ISSUE_LOW_SYMBOL_COVERAGE = "low_symbol_coverage"
READINESS_ISSUE_LOW_DATE_COVERAGE = "low_date_coverage"
READINESS_ISSUE_LOW_FIELD_COVERAGE = "low_field_coverage"
READINESS_ISSUE_UNBALANCED_SYMBOL_DATE_PANEL = "unbalanced_symbol_date_panel"
READINESS_ISSUE_MISSING_ADJ_FACTOR = "missing_adj_factor"
READINESS_ISSUE_INVALID_ADJ_FACTOR = "invalid_adj_factor"
READINESS_ISSUE_MISSING_LIMIT_DATA = "missing_limit_data"
READINESS_ISSUE_MISSING_SUSPEND_DATA = "missing_suspend_data"
READINESS_ISSUE_MISSING_INDEX_WEIGHT = "missing_index_weight"
READINESS_ISSUE_NON_POINT_IN_TIME_INDEX_WEIGHT = "non_point_in_time_index_weight"
READINESS_ISSUE_FACTOR_FIELD_NOT_MATRIX_READY = "factor_field_not_matrix_ready"
READINESS_ISSUE_EXCESSIVE_QUARANTINE_RATIO = "excessive_quarantine_ratio"
READINESS_ISSUE_NO_TRADABILITY_MASK = "no_tradability_mask"
READINESS_ISSUE_NO_BENCHMARK_MEMBERSHIP_MASK = "no_benchmark_membership_mask"

STORAGE_ISSUE_REDUNDANT_DUPLICATE_FILE = "redundant_duplicate_file"
STORAGE_ISSUE_OVERLAPPING_SYMBOL_HISTORY = "overlapping_symbol_history"
STORAGE_ISSUE_CSV_LARGE_FOR_ANALYTICS = "csv_large_for_analytics"
STORAGE_ISSUE_PARQUET_BACKEND_MISSING = "parquet_backend_missing"
STORAGE_ISSUE_PARQUET_WRITE_FAILED = "parquet_write_failed"
STORAGE_ISSUE_PARQUET_READBACK_MISMATCH = "parquet_readback_mismatch"
STORAGE_ISSUE_CANONICAL_PARQUET_DISABLED = "canonical_parquet_disabled"
STORAGE_ISSUE_DELETE_CSV_DISABLED = "delete_csv_disabled"
STORAGE_ISSUE_CSV_RETAINED_FOR_COMPATIBILITY = "csv_retained_for_compatibility"

DEFAULT_CLEANING_REPORT_DIR = "data/cleaning_reports/tushare"
DEFAULT_RAW_BACKUP_DIR = "data/raw_backups/tushare"
DEFAULT_QUARANTINE_DIR = "data/quarantine/tushare"
DEFAULT_FACTOR_READINESS_DIR = "data/factor_readiness/tushare"
DEFAULT_STORAGE_AUDIT_DIR = "data/cleaning_reports/tushare/storage_audit"
DEFAULT_PARQUET_SHADOW_DIR = "data/cn_market_parquet"
DEFAULT_CLEANING_REPORTS_FILENAME = "cleaning_reports.json"
DEFAULT_FACTOR_READINESS_REPORTS_FILENAME = "factor_readiness_reports.json"
DEFAULT_ROW_FLAGS_FILENAME = "row_flags.csv"
DEFAULT_CELL_FLAGS_FILENAME = "cell_flags.csv"
DEFAULT_MATRIX_COVERAGE_FILENAME = "matrix_coverage.json"
DEFAULT_FACTOR_READY_MASKS_FILENAME = "factor_ready_masks.json"
DEFAULT_STORAGE_AUDIT_REPORTS_FILENAME = "storage_audit_reports.json"
DEFAULT_PARQUET_MIGRATION_REPORTS_FILENAME = "parquet_migration_reports.json"

_VALID_CLEANING_STATUSES = {
    CLEANING_STATUS_PASS,
    CLEANING_STATUS_WARN,
    CLEANING_STATUS_FAIL,
}
_VALID_READINESS_STATUSES = {
    FACTOR_READINESS_READY,
    FACTOR_READINESS_WARN,
    FACTOR_READINESS_NOT_READY,
    FACTOR_READINESS_INSUFFICIENT_DATA,
}
_VALID_STORAGE_STATUSES = {
    STORAGE_STATUS_EFFICIENT,
    STORAGE_STATUS_REDUNDANT,
    STORAGE_STATUS_WARN,
    STORAGE_STATUS_FAIL,
}
_VALID_PARQUET_STATUSES = {
    PARQUET_STATUS_SUPPORTED,
    PARQUET_STATUS_UNSUPPORTED,
    PARQUET_STATUS_SHADOW_WRITTEN,
    PARQUET_STATUS_SKIPPED,
    PARQUET_STATUS_FAILED,
}
_VALID_STORAGE_FORMATS = {"csv", "parquet", "dual"}
_MATRIX_RELEVANT_TABLES = {"daily", "adj_factor", "daily_basic", "index_weight"}


def _sorted_unique(values: Any) -> list[str]:
    if values is None:
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes, list, tuple, dict)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_dict(instance: Any) -> dict[str, Any]:
    return _json_ready(asdict(instance))


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _safe_stem(value: str) -> str:
    text = str(value or "").strip() or "tushare"
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "tushare"


def _timestamp_slug(generated_at: str) -> str:
    return _safe_stem(generated_at.replace(":", "").replace("-", "").replace("T", "_").replace("Z", ""))


def _path_or_none(path: str | Path | None) -> str | None:
    return str(path) if path is not None else None


def _primary_key_payload(row: Mapping[str, Any], primary_key: list[str]) -> dict[str, Any]:
    return {key: _json_ready(row.get(key)) for key in primary_key if key in row}


@dataclass
class TushareCleanProfile:
    schema_version: str = TUSHARE_DATA_CLEANING_SCHEMA_VERSION
    table_name: str = ""
    primary_key: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    optional_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    ts_code_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    positive_columns: list[str] = field(default_factory=list)
    non_negative_columns: list[str] = field(default_factory=list)
    price_columns: list[str] = field(default_factory=list)
    volume_columns: list[str] = field(default_factory=list)
    amount_columns: list[str] = field(default_factory=list)
    ohlc_columns: dict[str, str] = field(default_factory=dict)
    factor_required_columns: list[str] = field(default_factory=list)
    model_optional_columns: list[str] = field(default_factory=list)
    point_in_time_columns: list[str] = field(default_factory=list)
    preferred_storage_format: str = "csv"
    parquet_partition_columns: list[str] = field(default_factory=list)
    allow_drop_invalid_rows: bool = True
    allow_deduplicate: bool = True
    quarantine_invalid_ohlc: bool = True
    quarantine_invalid_volume_amount: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.table_name = str(self.table_name or "").strip()
        if not self.table_name:
            raise ValueError("table_name must be non-empty")
        self.primary_key = _sorted_unique(self.primary_key)
        if not self.primary_key:
            raise ValueError("primary_key must be non-empty")
        for field_name in [
            "required_columns",
            "optional_columns",
            "date_columns",
            "ts_code_columns",
            "numeric_columns",
            "positive_columns",
            "non_negative_columns",
            "price_columns",
            "volume_columns",
            "amount_columns",
            "factor_required_columns",
            "model_optional_columns",
            "point_in_time_columns",
            "parquet_partition_columns",
        ]:
            setattr(self, field_name, _sorted_unique(getattr(self, field_name)))
        self.ohlc_columns = {
            str(key).strip(): str(value).strip()
            for key, value in sorted((self.ohlc_columns or {}).items())
            if str(key).strip() and str(value).strip()
        }
        if self.preferred_storage_format not in _VALID_STORAGE_FORMATS:
            raise ValueError("preferred_storage_format must be csv, parquet, or dual")
        self.allow_drop_invalid_rows = bool(self.allow_drop_invalid_rows)
        self.allow_deduplicate = bool(self.allow_deduplicate)
        self.quarantine_invalid_ohlc = bool(self.quarantine_invalid_ohlc)
        self.quarantine_invalid_volume_amount = bool(self.quarantine_invalid_volume_amount)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TushareCleanProfile":
        return cls(**dict(payload))


@dataclass
class TushareCleaningIssue:
    schema_version: str = TUSHARE_DATA_CLEANING_SCHEMA_VERSION
    issue_id: str = ""
    table_name: str = ""
    issue_code: str = ""
    severity: str = CLEANING_ISSUE_WARNING
    message: str = ""
    row_index: int | None = None
    primary_key: dict[str, Any] = field(default_factory=dict)
    column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TushareCleaningIssue":
        return cls(**dict(payload))


@dataclass
class TushareCleaningReport:
    schema_version: str = TUSHARE_DATA_CLEANING_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    table_name: str = ""
    source_path: str | None = None
    raw_backup_path: str | None = None
    cleaned_path: str | None = None
    parquet_shadow_path: str | None = None
    quarantine_path: str | None = None
    row_flags_path: str | None = None
    cell_flags_path: str | None = None
    raw_hash: str | None = None
    cleaned_hash: str | None = None
    parquet_hash: str | None = None
    raw_row_count: int = 0
    cleaned_row_count: int = 0
    dropped_row_count: int = 0
    quarantined_row_count: int = 0
    duplicate_row_count: int = 0
    conflicting_duplicate_count: int = 0
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    latest_trade_date: str | None = None
    status: str = CLEANING_STATUS_PASS
    issues: list[TushareCleaningIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_CLEANING_STATUSES:
            raise ValueError("invalid cleaning status")
        self.issues = [
            issue if isinstance(issue, TushareCleaningIssue) else TushareCleaningIssue.from_dict(issue)
            for issue in self.issues
        ]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        payload = _to_dict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TushareCleaningReport":
        data = dict(payload)
        data["issues"] = [TushareCleaningIssue.from_dict(item) for item in data.get("issues", [])]
        return cls(**data)


@dataclass
class FactorReadinessConfig:
    schema_version: str = TUSHARE_FACTOR_READINESS_SCHEMA_VERSION
    config_id: str = ""
    universe: str | None = None
    benchmark: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    min_symbol_coverage_ratio: float = 0.80
    min_date_coverage_ratio: float = 0.95
    min_field_coverage_ratio: float = 0.90
    max_quarantine_ratio: float = 0.05
    require_adj_factor: bool = True
    require_trade_cal: bool = True
    require_limit_data: bool = True
    require_suspend_data: bool = True
    require_index_weight: bool = False
    require_point_in_time_index_weight: bool = False
    required_factor_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.required_factor_fields = _sorted_unique(self.required_factor_fields)
        for name in [
            "min_symbol_coverage_ratio",
            "min_date_coverage_ratio",
            "min_field_coverage_ratio",
            "max_quarantine_ratio",
        ]:
            value = float(getattr(self, name))
            setattr(self, name, max(0.0, min(value, 1.0)))
        if not self.config_id:
            self.config_id = make_factor_readiness_config_id(self)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorReadinessConfig":
        return cls(**dict(payload))


@dataclass
class FactorReadinessIssue:
    schema_version: str = TUSHARE_FACTOR_READINESS_SCHEMA_VERSION
    issue_id: str = ""
    issue_code: str = ""
    severity: str = CLEANING_ISSUE_WARNING
    message: str = ""
    table_name: str | None = None
    symbol: str | None = None
    trade_date: str | None = None
    field_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorReadinessIssue":
        return cls(**dict(payload))


@dataclass
class MatrixCoverageSummary:
    schema_version: str = TUSHARE_FACTOR_READINESS_SCHEMA_VERSION
    table_name: str = ""
    symbol_count: int = 0
    date_count: int = 0
    expected_cell_count: int = 0
    observed_cell_count: int = 0
    missing_cell_count: int = 0
    quarantined_cell_count: int = 0
    symbol_coverage_ratio: float | None = None
    date_coverage_ratio: float | None = None
    field_coverage_ratio: float | None = None
    field_coverage: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatrixCoverageSummary":
        return cls(**dict(payload))


@dataclass
class FactorReadyMaskManifest:
    schema_version: str = TUSHARE_FACTOR_READINESS_SCHEMA_VERSION
    manifest_id: str = ""
    table_name: str = ""
    symbols: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    masks: dict[str, list[list[bool]]] = field(default_factory=dict)
    mask_meanings: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbols = _sorted_unique(self.symbols)
        self.dates = _sorted_unique(self.dates)
        self.metadata = dict(self.metadata or {})
        if not self.manifest_id:
            self.manifest_id = make_factor_ready_mask_manifest_id(
                table_name=self.table_name,
                symbols=self.symbols,
                dates=self.dates,
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorReadyMaskManifest":
        return cls(**dict(payload))


@dataclass
class FactorReadinessReport:
    schema_version: str = TUSHARE_FACTOR_READINESS_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    config: FactorReadinessConfig = field(default_factory=FactorReadinessConfig)
    source_root: str | None = None
    table_reports: dict[str, TushareCleaningReport] = field(default_factory=dict)
    coverage_summaries: list[MatrixCoverageSummary] = field(default_factory=list)
    mask_manifests: list[FactorReadyMaskManifest] = field(default_factory=list)
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    overall_status: str = FACTOR_READINESS_INSUFFICIENT_DATA
    issues: list[FactorReadinessIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.overall_status not in _VALID_READINESS_STATUSES:
            raise ValueError("invalid factor readiness status")
        if not isinstance(self.config, FactorReadinessConfig):
            self.config = FactorReadinessConfig.from_dict(self.config)
        self.table_reports = {
            str(name): report if isinstance(report, TushareCleaningReport) else TushareCleaningReport.from_dict(report)
            for name, report in self.table_reports.items()
        }
        self.coverage_summaries = [
            item if isinstance(item, MatrixCoverageSummary) else MatrixCoverageSummary.from_dict(item)
            for item in self.coverage_summaries
        ]
        self.mask_manifests = [
            item if isinstance(item, FactorReadyMaskManifest) else FactorReadyMaskManifest.from_dict(item)
            for item in self.mask_manifests
        ]
        self.issues = [
            item if isinstance(item, FactorReadinessIssue) else FactorReadinessIssue.from_dict(item)
            for item in self.issues
        ]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        payload = _to_dict(self)
        payload["config"] = self.config.to_dict()
        payload["table_reports"] = {
            name: report.to_dict() for name, report in self.table_reports.items()
        }
        payload["coverage_summaries"] = [item.to_dict() for item in self.coverage_summaries]
        payload["mask_manifests"] = [item.to_dict() for item in self.mask_manifests]
        payload["issues"] = [item.to_dict() for item in self.issues]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorReadinessReport":
        return cls(**dict(payload))


@dataclass
class TushareStorageOptimizationConfig:
    schema_version: str = TUSHARE_STORAGE_OPTIMIZATION_SCHEMA_VERSION
    config_id: str = ""
    prefer_parquet_for_factor_research: bool = True
    parquet_shadow_write: bool = False
    parquet_canonical: bool = False
    delete_redundant_csv: bool = False
    parquet_dir: str | None = None
    parquet_compression: str = "snappy"
    min_csv_size_for_parquet_bytes: int = 1_000_000
    min_rows_for_parquet: int = 10_000
    require_readback_validation: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in [
            "prefer_parquet_for_factor_research",
            "parquet_shadow_write",
            "parquet_canonical",
            "delete_redundant_csv",
            "require_readback_validation",
        ]:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be bool")
        self.min_csv_size_for_parquet_bytes = max(0, int(self.min_csv_size_for_parquet_bytes))
        self.min_rows_for_parquet = max(0, int(self.min_rows_for_parquet))
        self.parquet_compression = str(self.parquet_compression or "").strip()
        if not self.parquet_compression:
            raise ValueError("parquet_compression must be non-empty")
        if self.delete_redundant_csv and not self.parquet_canonical:
            raise ValueError("delete_redundant_csv requires parquet_canonical")
        self.metadata = dict(self.metadata or {})
        if not self.config_id:
            self.config_id = make_storage_optimization_config_id(self)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TushareStorageOptimizationConfig":
        return cls(**dict(payload))


@dataclass
class TushareStorageAuditReport:
    schema_version: str = TUSHARE_STORAGE_OPTIMIZATION_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    table_name: str = ""
    source_path: str | None = None
    csv_path: str | None = None
    parquet_path: str | None = None
    csv_size_bytes: int | None = None
    parquet_size_bytes: int | None = None
    estimated_size_reduction_ratio: float | None = None
    duplicate_file_count: int = 0
    overlapping_history_count: int = 0
    recommended_storage_format: str = "csv"
    parquet_supported: bool = False
    parquet_backend: str | None = None
    issue_codes: list[str] = field(default_factory=list)
    status: str = STORAGE_STATUS_EFFICIENT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STORAGE_STATUSES:
            raise ValueError("invalid storage status")
        if (
            self.estimated_size_reduction_ratio is not None
            and not 0.0 <= float(self.estimated_size_reduction_ratio) <= 1.0
        ):
            raise ValueError("estimated_size_reduction_ratio must be in [0, 1]")
        self.issue_codes = _sorted_unique(self.issue_codes)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TushareStorageAuditReport":
        return cls(**dict(payload))


@dataclass
class ParquetMigrationReport:
    schema_version: str = TUSHARE_PARQUET_MIGRATION_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    table_name: str = ""
    csv_path: str | None = None
    parquet_path: str | None = None
    backend: str | None = None
    compression: str = "snappy"
    row_count: int = 0
    column_count: int = 0
    csv_size_bytes: int | None = None
    parquet_size_bytes: int | None = None
    size_reduction_ratio: float | None = None
    readback_validated: bool = False
    status: str = PARQUET_STATUS_SKIPPED
    issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_PARQUET_STATUSES:
            raise ValueError("invalid parquet migration status")
        self.issue_codes = _sorted_unique(self.issue_codes)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParquetMigrationReport":
        return cls(**dict(payload))


def make_cleaning_issue_id(
    *,
    table_name: str,
    issue_code: str,
    row_index: int | None = None,
    primary_key: Mapping[str, Any] | None = None,
    column: str | None = None,
    message: str = "",
) -> str:
    return _stable_id(
        "tushare-cleaning-issue",
        {
            "table_name": table_name,
            "issue_code": issue_code,
            "row_index": row_index,
            "primary_key": dict(primary_key or {}),
            "column": column,
            "message": message,
        },
    )


def make_cleaning_report_id(*, table_name: str, source_path: str | None, generated_at: str) -> str:
    return _stable_id(
        "tushare-cleaning-report",
        {"table_name": table_name, "source_path": source_path, "generated_at": generated_at},
    )


def make_factor_readiness_issue_id(
    *,
    issue_code: str,
    table_name: str | None = None,
    symbol: str | None = None,
    trade_date: str | None = None,
    field_name: str | None = None,
) -> str:
    return _stable_id(
        "tushare-factor-readiness-issue",
        {
            "issue_code": issue_code,
            "table_name": table_name,
            "symbol": symbol,
            "trade_date": trade_date,
            "field_name": field_name,
        },
    )


def make_factor_readiness_config_id(config: FactorReadinessConfig) -> str:
    payload = dict(config.__dict__)
    payload.pop("config_id", None)
    return _stable_id("tushare-factor-readiness-config", payload)


def make_factor_readiness_report_id(
    *,
    source_root: str | None,
    generated_at: str,
    config_id: str,
) -> str:
    return _stable_id(
        "tushare-factor-readiness-report",
        {"source_root": source_root, "generated_at": generated_at, "config_id": config_id},
    )


def make_factor_ready_mask_manifest_id(
    *,
    table_name: str,
    symbols: list[str],
    dates: list[str],
) -> str:
    return _stable_id(
        "tushare-factor-ready-mask",
        {"table_name": table_name, "symbols": _sorted_unique(symbols), "dates": _sorted_unique(dates)},
    )


def make_storage_optimization_config_id(config: TushareStorageOptimizationConfig) -> str:
    payload = dict(config.__dict__)
    payload.pop("config_id", None)
    return _stable_id("tushare-storage-config", payload)


def make_storage_audit_report_id(
    *,
    table_name: str,
    source_path: str | None,
    generated_at: str,
) -> str:
    return _stable_id(
        "tushare-storage-audit",
        {"table_name": table_name, "source_path": source_path, "generated_at": generated_at},
    )


def make_parquet_migration_report_id(
    *,
    table_name: str,
    csv_path: str | None,
    generated_at: str,
) -> str:
    return _stable_id(
        "tushare-parquet-migration",
        {"table_name": table_name, "csv_path": csv_path, "generated_at": generated_at},
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_dump(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(target)
    return target


def atomic_write_dataframe_csv(df: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def detect_parquet_backend() -> tuple[bool, str | None, list[str]]:
    warnings: list[str] = []
    probe = pd.DataFrame({"a": [1], "b": ["x"]})
    for backend in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(backend) is None:
            warnings.append(f"{backend} not installed")
            continue
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "probe.parquet"
            try:
                probe.to_parquet(path, engine=backend)
                readback = pd.read_parquet(path, engine=backend)
                if list(readback.columns) == list(probe.columns) and len(readback) == len(probe):
                    return True, backend, warnings
                warnings.append(f"{backend} readback mismatch")
            except Exception as exc:
                warnings.append(f"{backend} unusable: {exc}")
    return False, None, warnings


def _file_size(path: str | Path | None) -> int | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return resolved.stat().st_size


def _csv_row_count(path: str | Path | None) -> int:
    if not path:
        return 0
    resolved = Path(path)
    if not resolved.exists():
        return 0
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _line in handle) - 1, 0)
    except UnicodeDecodeError:
        try:
            return len(pd.read_csv(resolved))
        except Exception:
            return 0


def write_parquet_shadow_if_supported(
    df: Any,
    *,
    table_name: str,
    csv_path: str | Path | None,
    parquet_path: str | Path,
    config: TushareStorageOptimizationConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> ParquetMigrationReport:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    target = Path(parquet_path)
    report_id = make_parquet_migration_report_id(
        table_name=table_name,
        csv_path=_path_or_none(csv_path),
        generated_at=generated_at,
    )
    csv_size = _file_size(csv_path)
    if not config.parquet_shadow_write and not config.parquet_canonical:
        return ParquetMigrationReport(
            report_id=report_id,
            generated_at=generated_at,
            table_name=table_name,
            csv_path=_path_or_none(csv_path),
            parquet_path=str(target),
            compression=config.parquet_compression,
            row_count=len(frame),
            column_count=len(frame.columns),
            csv_size_bytes=csv_size,
            status=PARQUET_STATUS_SKIPPED,
            issue_codes=[STORAGE_ISSUE_CANONICAL_PARQUET_DISABLED],
            metadata=dict(metadata or {}),
        )

    supported, backend, warnings = detect_parquet_backend()
    if not supported or backend is None:
        return ParquetMigrationReport(
            report_id=report_id,
            generated_at=generated_at,
            table_name=table_name,
            csv_path=_path_or_none(csv_path),
            parquet_path=str(target),
            backend=backend,
            compression=config.parquet_compression,
            row_count=len(frame),
            column_count=len(frame.columns),
            csv_size_bytes=csv_size,
            status=PARQUET_STATUS_UNSUPPORTED,
            issue_codes=[STORAGE_ISSUE_PARQUET_BACKEND_MISSING],
            metadata={**dict(metadata or {}), "warnings": warnings},
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    issue_codes: list[str] = []
    readback_validated = False
    try:
        frame.to_parquet(
            tmp_path,
            engine=backend,
            compression=config.parquet_compression,
            index=False,
        )
        readback_metadata: dict[str, Any] = {}
        if config.require_readback_validation:
            readback = pd.read_parquet(tmp_path, engine=backend)
            readback_validated = True
            if len(readback) != len(frame) or set(readback.columns) != set(frame.columns):
                issue_codes.append(STORAGE_ISSUE_PARQUET_READBACK_MISMATCH)
            dtype_changes = {
                column: {
                    "before": str(frame[column].dtype),
                    "after": str(readback[column].dtype),
                }
                for column in frame.columns
                if column in readback.columns and str(frame[column].dtype) != str(readback[column].dtype)
            }
            if dtype_changes:
                readback_metadata["dtype_changes"] = dtype_changes
        if issue_codes:
            status = PARQUET_STATUS_FAILED
        else:
            tmp_path.replace(target)
            status = PARQUET_STATUS_SHADOW_WRITTEN
        parquet_size = _file_size(target if target.exists() else tmp_path)
        ratio = None
        if csv_size and parquet_size is not None and csv_size > 0:
            ratio = max(0.0, min(1.0, 1 - parquet_size / csv_size))
        return ParquetMigrationReport(
            report_id=report_id,
            generated_at=generated_at,
            table_name=table_name,
            csv_path=_path_or_none(csv_path),
            parquet_path=str(target),
            backend=backend,
            compression=config.parquet_compression,
            row_count=len(frame),
            column_count=len(frame.columns),
            csv_size_bytes=csv_size,
            parquet_size_bytes=parquet_size,
            size_reduction_ratio=ratio,
            readback_validated=readback_validated,
            status=status,
            issue_codes=issue_codes,
            metadata={**dict(metadata or {}), **readback_metadata, "warnings": warnings},
        )
    except Exception as exc:
        return ParquetMigrationReport(
            report_id=report_id,
            generated_at=generated_at,
            table_name=table_name,
            csv_path=_path_or_none(csv_path),
            parquet_path=str(target),
            backend=backend,
            compression=config.parquet_compression,
            row_count=len(frame),
            column_count=len(frame.columns),
            csv_size_bytes=csv_size,
            status=PARQUET_STATUS_FAILED,
            issue_codes=[STORAGE_ISSUE_PARQUET_WRITE_FAILED],
            metadata={**dict(metadata or {}), "error": str(exc), "warnings": warnings},
        )
    finally:
        if tmp_path.exists() and tmp_path != target:
            tmp_path.unlink(missing_ok=True)


def build_storage_audit_report(
    *,
    table_name: str,
    csv_path: str | Path | None,
    parquet_path: str | Path | None = None,
    config: TushareStorageOptimizationConfig | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> TushareStorageAuditReport:
    resolved_config = config or TushareStorageOptimizationConfig()
    supported, backend, warnings = detect_parquet_backend()
    csv_size = _file_size(csv_path)
    parquet_size = _file_size(parquet_path)
    row_count = _csv_row_count(csv_path)
    ratio = None
    if csv_size and parquet_size is not None and csv_size > 0:
        ratio = max(0.0, min(1.0, 1 - parquet_size / csv_size))
    matrix_relevant = str(table_name).strip().lower() in _MATRIX_RELEVANT_TABLES
    size_trigger = (
        (csv_size or 0) >= resolved_config.min_csv_size_for_parquet_bytes
        or row_count >= resolved_config.min_rows_for_parquet
    )
    recommend_parquet = (
        resolved_config.prefer_parquet_for_factor_research
        and matrix_relevant
        and size_trigger
    )
    issue_codes: list[str] = []
    if recommend_parquet:
        issue_codes.append(STORAGE_ISSUE_CSV_LARGE_FOR_ANALYTICS)
        if not supported:
            issue_codes.append(STORAGE_ISSUE_PARQUET_BACKEND_MISSING)
    if parquet_path and parquet_size is not None and csv_size and parquet_size < csv_size:
        issue_codes.append(STORAGE_ISSUE_REDUNDANT_DUPLICATE_FILE)
    if not resolved_config.parquet_canonical:
        issue_codes.append(STORAGE_ISSUE_CANONICAL_PARQUET_DISABLED)
    if not resolved_config.delete_redundant_csv:
        issue_codes.append(STORAGE_ISSUE_DELETE_CSV_DISABLED)
        if parquet_path and parquet_size is not None:
            issue_codes.append(STORAGE_ISSUE_CSV_RETAINED_FOR_COMPATIBILITY)

    if parquet_path and parquet_size is not None and csv_size and parquet_size < csv_size and recommend_parquet:
        status = STORAGE_STATUS_REDUNDANT
    elif recommend_parquet and not supported:
        status = STORAGE_STATUS_WARN
    elif recommend_parquet and supported and not parquet_path:
        status = STORAGE_STATUS_WARN
    else:
        status = STORAGE_STATUS_EFFICIENT

    return TushareStorageAuditReport(
        report_id=make_storage_audit_report_id(
            table_name=table_name,
            source_path=_path_or_none(csv_path),
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        table_name=table_name,
        source_path=_path_or_none(csv_path),
        csv_path=_path_or_none(csv_path),
        parquet_path=_path_or_none(parquet_path),
        csv_size_bytes=csv_size,
        parquet_size_bytes=parquet_size,
        estimated_size_reduction_ratio=ratio,
        recommended_storage_format="parquet" if recommend_parquet and supported else "csv",
        parquet_supported=supported,
        parquet_backend=backend,
        issue_codes=issue_codes,
        status=status,
        metadata={**dict(metadata or {}), "row_count": row_count, "warnings": warnings},
    )


def get_default_tushare_clean_profiles() -> dict[str, TushareCleanProfile]:
    daily_numeric = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "adj_factor",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    ]
    return {
        "daily": TushareCleanProfile(
            table_name="daily",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            optional_columns=["pre_close", "change", "pct_chg", "adj_factor", "adj_open", "adj_high", "adj_low", "adj_close"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            numeric_columns=daily_numeric,
            positive_columns=["open", "high", "low", "close", "adj_factor", "adj_open", "adj_high", "adj_low", "adj_close"],
            non_negative_columns=["vol", "amount"],
            price_columns=["open", "high", "low", "close", "pre_close", "adj_open", "adj_high", "adj_low", "adj_close"],
            volume_columns=["vol"],
            amount_columns=["amount"],
            ohlc_columns={"open": "open", "high": "high", "low": "low", "close": "close"},
            factor_required_columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"],
            model_optional_columns=["pre_close", "change", "pct_chg", "adj_open", "adj_high", "adj_low", "adj_close"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["ts_code"],
        ),
        "adj_factor": TushareCleanProfile(
            table_name="adj_factor",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date", "adj_factor"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            numeric_columns=["adj_factor"],
            positive_columns=["adj_factor"],
            factor_required_columns=["ts_code", "trade_date", "adj_factor"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["ts_code"],
        ),
        "daily_basic": TushareCleanProfile(
            table_name="daily_basic",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date"],
            optional_columns=["close", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            numeric_columns=["close", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"],
            positive_columns=["close"],
            factor_required_columns=["ts_code", "trade_date"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["ts_code"],
        ),
        "stock_basic": TushareCleanProfile(
            table_name="stock_basic",
            primary_key=["ts_code"],
            required_columns=["ts_code", "name"],
            optional_columns=["symbol", "area", "industry", "market", "list_date", "delist_date", "list_status"],
            date_columns=["list_date", "delist_date"],
            ts_code_columns=["ts_code"],
            factor_required_columns=["ts_code", "name"],
            point_in_time_columns=["list_date", "delist_date"],
            preferred_storage_format="csv",
        ),
        "suspend_d": TushareCleanProfile(
            table_name="suspend_d",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date"],
            optional_columns=["suspend_type"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            factor_required_columns=["ts_code", "trade_date"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="csv",
        ),
        "suspend": TushareCleanProfile(
            table_name="suspend",
            primary_key=["ts_code", "suspend_date"],
            required_columns=["ts_code", "suspend_date"],
            optional_columns=["resume_date", "ann_date", "suspend_reason", "reason_type"],
            date_columns=["suspend_date", "resume_date", "ann_date"],
            ts_code_columns=["ts_code"],
            factor_required_columns=["ts_code", "suspend_date"],
            point_in_time_columns=["ann_date", "suspend_date", "resume_date"],
            preferred_storage_format="csv",
        ),
        "stk_limit": TushareCleanProfile(
            table_name="stk_limit",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            numeric_columns=["up_limit", "down_limit"],
            positive_columns=["up_limit", "down_limit"],
            price_columns=["up_limit", "down_limit"],
            factor_required_columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["ts_code"],
        ),
        "index_daily": TushareCleanProfile(
            table_name="index_daily",
            primary_key=["ts_code", "trade_date"],
            required_columns=["ts_code", "trade_date", "open", "high", "low", "close"],
            optional_columns=["pre_close", "change", "pct_chg", "vol", "amount"],
            date_columns=["trade_date"],
            ts_code_columns=["ts_code"],
            numeric_columns=["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"],
            positive_columns=["open", "high", "low", "close"],
            non_negative_columns=["vol", "amount"],
            price_columns=["open", "high", "low", "close", "pre_close"],
            volume_columns=["vol"],
            amount_columns=["amount"],
            ohlc_columns={"open": "open", "high": "high", "low": "low", "close": "close"},
            factor_required_columns=["ts_code", "trade_date", "open", "high", "low", "close"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["ts_code"],
        ),
        "index_weight": TushareCleanProfile(
            table_name="index_weight",
            primary_key=["index_code", "con_code", "trade_date"],
            required_columns=["index_code", "con_code", "trade_date", "weight"],
            optional_columns=["trade_date"],
            date_columns=["trade_date"],
            ts_code_columns=["index_code", "con_code"],
            numeric_columns=["weight"],
            non_negative_columns=["weight"],
            factor_required_columns=["index_code", "con_code", "trade_date", "weight"],
            point_in_time_columns=["trade_date"],
            preferred_storage_format="dual",
            parquet_partition_columns=["index_code"],
        ),
        "trade_cal": TushareCleanProfile(
            table_name="trade_cal",
            primary_key=["cal_date"],
            required_columns=["cal_date", "is_open"],
            optional_columns=["exchange", "pretrade_date"],
            date_columns=["cal_date", "pretrade_date"],
            numeric_columns=["is_open"],
            factor_required_columns=["cal_date", "is_open"],
            point_in_time_columns=["cal_date"],
            preferred_storage_format="csv",
        ),
    }


def _resolve_profile(table_name: str, profile: TushareCleanProfile | None = None) -> TushareCleanProfile:
    if profile is not None:
        return profile
    profiles = get_default_tushare_clean_profiles()
    key = str(table_name or "daily").strip().lower()
    if key not in profiles:
        raise ValueError(f"unknown Tushare clean profile: {table_name}")
    return profiles[key]


def _normalize_date_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if re.fullmatch(r"\d{8}", text):
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _valid_ts_code(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().upper()
    if not text:
        return False
    return bool(re.fullmatch(r"[0-9A-Z]{5,12}\.(SH|SZ|BJ|CSI|CNI|WI|HK)$", text))


def _issue(
    *,
    table_name: str,
    issue_code: str,
    severity: str,
    message: str,
    row_index: int | None = None,
    primary_key: Mapping[str, Any] | None = None,
    column: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TushareCleaningIssue:
    payload_pk = dict(primary_key or {})
    return TushareCleaningIssue(
        issue_id=make_cleaning_issue_id(
            table_name=table_name,
            issue_code=issue_code,
            row_index=row_index,
            primary_key=payload_pk,
            column=column,
            message=message,
        ),
        table_name=table_name,
        issue_code=issue_code,
        severity=severity,
        message=message,
        row_index=row_index,
        primary_key=payload_pk,
        column=column,
        metadata=dict(metadata or {}),
    )


def _cell_flag(
    *,
    row_index: int,
    primary_key: Mapping[str, Any],
    column: str,
    issue_code: str,
    severity: str,
    value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "primary_key_json": _stable_json(dict(primary_key)),
        "column": column,
        "issue_code": issue_code,
        "severity": severity,
        "value": _json_ready(value),
        "message": message,
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


def _build_cleaning_report(
    *,
    profile: TushareCleanProfile,
    generated_at: str,
    source_path: str | Path | None,
    raw_backup_path: str | Path | None,
    cleaned_path: str | Path | None,
    parquet_shadow_path: str | Path | None,
    quarantine_path: str | Path | None,
    row_flags_path: str | Path | None,
    cell_flags_path: str | Path | None,
    raw_row_count: int,
    cleaned_row_count: int,
    dropped_row_count: int,
    quarantined_row_count: int,
    duplicate_row_count: int,
    conflicting_duplicate_count: int,
    latest_trade_date: str | None,
    issues: list[TushareCleaningIssue],
    metadata: Mapping[str, Any],
) -> TushareCleaningReport:
    blocker_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_BLOCKER)
    warning_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_WARNING)
    info_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_INFO)
    status = CLEANING_STATUS_FAIL if blocker_count else (CLEANING_STATUS_WARN if warning_count else CLEANING_STATUS_PASS)
    return TushareCleaningReport(
        report_id=make_cleaning_report_id(
            table_name=profile.table_name,
            source_path=_path_or_none(source_path),
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        table_name=profile.table_name,
        source_path=_path_or_none(source_path),
        raw_backup_path=_path_or_none(raw_backup_path),
        cleaned_path=_path_or_none(cleaned_path),
        parquet_shadow_path=_path_or_none(parquet_shadow_path),
        quarantine_path=_path_or_none(quarantine_path),
        row_flags_path=_path_or_none(row_flags_path),
        cell_flags_path=_path_or_none(cell_flags_path),
        raw_row_count=int(raw_row_count),
        cleaned_row_count=int(cleaned_row_count),
        dropped_row_count=int(dropped_row_count),
        quarantined_row_count=int(quarantined_row_count),
        duplicate_row_count=int(duplicate_row_count),
        conflicting_duplicate_count=int(conflicting_duplicate_count),
        issue_count=len(issues),
        blocker_count=blocker_count,
        warning_count=warning_count,
        info_count=info_count,
        latest_trade_date=latest_trade_date,
        status=status,
        issues=issues,
        metadata=dict(metadata or {}),
    )


def _symbol_column_for(df: pd.DataFrame) -> str | None:
    for column in ("ts_code", "con_code", "index_code"):
        if column in df.columns:
            return column
    return None


def _date_column_for(df: pd.DataFrame) -> str | None:
    for column in ("trade_date", "cal_date", "suspend_date", "ann_date"):
        if column in df.columns:
            return column
    return None


def _valid_ohlc_row(row: Mapping[str, Any]) -> bool:
    columns = ["open", "high", "low", "close"]
    if not all(column in row for column in columns):
        return True
    values = [pd.to_numeric(row.get(column), errors="coerce") for column in columns]
    if any(pd.isna(value) for value in values):
        return False
    open_v, high_v, low_v, close_v = values
    return bool(
        min(open_v, high_v, low_v, close_v) > 0
        and high_v >= max(open_v, low_v, close_v)
        and low_v <= min(open_v, high_v, close_v)
        and high_v >= low_v
    )


def build_matrix_coverage_summary(
    df: Any,
    *,
    table_name: str = "daily",
    profile: TushareCleanProfile | None = None,
    fields: list[str] | None = None,
    expected_symbols: list[str] | None = None,
    expected_dates: list[str] | None = None,
    quarantined_df: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MatrixCoverageSummary:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    resolved_profile = _resolve_profile(table_name, profile)
    symbol_column = _symbol_column_for(frame)
    date_column = _date_column_for(frame)
    symbols = _sorted_unique(expected_symbols or (frame[symbol_column].dropna().tolist() if symbol_column else []))
    dates = _sorted_unique(expected_dates or (frame[date_column].dropna().tolist() if date_column else []))
    expected_pairs = {(symbol, date) for symbol in symbols for date in dates}
    observed_pairs: set[tuple[str, str]] = set()
    if symbol_column and date_column and not frame.empty:
        observed_pairs = {
            (str(row[symbol_column]), str(row[date_column]))
            for _idx, row in frame[[symbol_column, date_column]].dropna().iterrows()
        }
    quarantined_pairs: set[tuple[str, str]] = set()
    if quarantined_df is not None:
        qdf = quarantined_df if isinstance(quarantined_df, pd.DataFrame) else pd.DataFrame(quarantined_df)
        if symbol_column in qdf.columns and date_column in qdf.columns:
            quarantined_pairs = {
                (str(row[symbol_column]), str(row[date_column]))
                for _idx, row in qdf[[symbol_column, date_column]].dropna().iterrows()
            }
    field_list = fields or [
        column
        for column in resolved_profile.factor_required_columns
        if column not in set(resolved_profile.primary_key)
    ]
    field_coverage: dict[str, float] = {}
    denominator = len(frame) if len(frame) > 0 else 0
    for column in _sorted_unique(field_list):
        if column in frame.columns and denominator:
            field_coverage[column] = float(frame[column].notna().mean())
        else:
            field_coverage[column] = 0.0
    expected_cell_count = len(expected_pairs)
    observed_cell_count = len(observed_pairs & expected_pairs) if expected_pairs else len(observed_pairs)
    missing_cell_count = max(expected_cell_count - observed_cell_count, 0)
    symbol_coverage = None
    if expected_symbols:
        observed_symbols = {symbol for symbol, _date in observed_pairs}
        symbol_coverage = len(observed_symbols & set(symbols)) / len(symbols) if symbols else None
    else:
        symbol_coverage = 1.0 if symbols else None
    date_coverage = None
    if expected_dates:
        observed_dates = {date for _symbol, date in observed_pairs}
        date_coverage = len(observed_dates & set(dates)) / len(dates) if dates else None
    else:
        date_coverage = 1.0 if dates else None
    field_ratio = min(field_coverage.values()) if field_coverage else None
    return MatrixCoverageSummary(
        table_name=resolved_profile.table_name,
        symbol_count=len(symbols),
        date_count=len(dates),
        expected_cell_count=expected_cell_count,
        observed_cell_count=observed_cell_count,
        missing_cell_count=missing_cell_count,
        quarantined_cell_count=len(quarantined_pairs),
        symbol_coverage_ratio=symbol_coverage,
        date_coverage_ratio=date_coverage,
        field_coverage_ratio=field_ratio,
        field_coverage=field_coverage,
        metadata=dict(metadata or {}),
    )


def build_factor_ready_mask_manifest(
    df: Any,
    *,
    table_name: str = "daily",
    profile: TushareCleanProfile | None = None,
    quarantined_df: Any | None = None,
    symbols: list[str] | None = None,
    dates: list[str] | None = None,
    tradability_mask: Mapping[tuple[str, str], bool] | None = None,
    benchmark_mask: Mapping[tuple[str, str], bool] | None = None,
    index_weight_mask: Mapping[tuple[str, str], bool] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorReadyMaskManifest:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    resolved_profile = _resolve_profile(table_name, profile)
    symbol_column = _symbol_column_for(frame)
    date_column = _date_column_for(frame)
    resolved_symbols = _sorted_unique(symbols or (frame[symbol_column].dropna().tolist() if symbol_column else []))
    resolved_dates = _sorted_unique(dates or (frame[date_column].dropna().tolist() if date_column else []))
    row_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    if symbol_column and date_column and not frame.empty:
        for _idx, row in frame.iterrows():
            symbol = row.get(symbol_column)
            trade_date = row.get(date_column)
            if pd.isna(symbol) or pd.isna(trade_date):
                continue
            row_by_pair[(str(symbol), str(trade_date))] = row.to_dict()
    quarantined_pairs: set[tuple[str, str]] = set()
    if quarantined_df is not None:
        qdf = quarantined_df if isinstance(quarantined_df, pd.DataFrame) else pd.DataFrame(quarantined_df)
        if symbol_column in qdf.columns and date_column in qdf.columns:
            quarantined_pairs = {
                (str(row[symbol_column]), str(row[date_column]))
                for _idx, row in qdf[[symbol_column, date_column]].dropna().iterrows()
            }

    mask_names = [
        "has_row",
        "valid_ohlc",
        "valid_volume",
        "valid_amount",
        "tradable",
        "factor_eligible",
        "benchmark_member",
        "index_weight_available",
        "adjusted_price_ready",
    ]
    masks: dict[str, list[list[bool]]] = {name: [] for name in mask_names}
    for symbol in resolved_symbols:
        row_values = {name: [] for name in mask_names}
        for trade_date in resolved_dates:
            pair = (symbol, trade_date)
            row = row_by_pair.get(pair)
            has_row = row is not None and pair not in quarantined_pairs
            valid_ohlc = bool(has_row and _valid_ohlc_row(row or {}))
            valid_volume = bool(
                has_row
                and all(
                    column not in (row or {})
                    or pd.isna(pd.to_numeric((row or {}).get(column), errors="coerce"))
                    or pd.to_numeric((row or {}).get(column), errors="coerce") >= 0
                    for column in resolved_profile.volume_columns
                )
            )
            valid_amount = bool(
                has_row
                and all(
                    column not in (row or {})
                    or pd.isna(pd.to_numeric((row or {}).get(column), errors="coerce"))
                    or pd.to_numeric((row or {}).get(column), errors="coerce") >= 0
                    for column in resolved_profile.amount_columns
                )
            )
            adjusted_ready = bool(
                has_row
                and "adj_factor" in (row or {})
                and pd.notna(pd.to_numeric((row or {}).get("adj_factor"), errors="coerce"))
                and pd.to_numeric((row or {}).get("adj_factor"), errors="coerce") > 0
            )
            tradable = bool(tradability_mask.get(pair, False)) if tradability_mask is not None else False
            benchmark_member = bool(benchmark_mask.get(pair, False)) if benchmark_mask is not None else False
            index_weight_available = (
                bool(index_weight_mask.get(pair, False)) if index_weight_mask is not None else False
            )
            factor_eligible = bool(
                has_row
                and valid_ohlc
                and valid_volume
                and valid_amount
                and adjusted_ready
                and tradable
                and benchmark_member
            )
            values = {
                "has_row": has_row,
                "valid_ohlc": valid_ohlc,
                "valid_volume": valid_volume,
                "valid_amount": valid_amount,
                "tradable": tradable,
                "factor_eligible": factor_eligible,
                "benchmark_member": benchmark_member,
                "index_weight_available": index_weight_available,
                "adjusted_price_ready": adjusted_ready,
            }
            for name in mask_names:
                row_values[name].append(values[name])
        for name in mask_names:
            masks[name].append(row_values[name])

    meanings = {
        "has_row": "cleaned symbol-date row exists and was not quarantined",
        "valid_ohlc": "OHLC fields pass non-forward-filled price relation checks",
        "valid_volume": "volume fields are non-negative when present",
        "valid_amount": "amount fields are non-negative when present",
        "tradable": "tradability source marks the cell usable; false when source is absent",
        "factor_eligible": "all required masks for factor mining are true",
        "benchmark_member": "benchmark membership source marks the symbol-date in universe",
        "index_weight_available": "point-in-time index weight is available",
        "adjusted_price_ready": "positive adj_factor is present for adjusted-price research",
    }
    return FactorReadyMaskManifest(
        table_name=resolved_profile.table_name,
        symbols=resolved_symbols,
        dates=resolved_dates,
        masks=masks,
        mask_meanings=meanings,
        metadata={
            **dict(metadata or {}),
            "tradability_source": "provided" if tradability_mask is not None else "missing",
            "benchmark_source": "provided" if benchmark_mask is not None else "missing",
            "index_weight_source": "provided" if index_weight_mask is not None else "missing",
        },
    )


def _readiness_issue(
    *,
    issue_code: str,
    severity: str,
    message: str,
    table_name: str | None = None,
    symbol: str | None = None,
    trade_date: str | None = None,
    field_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorReadinessIssue:
    return FactorReadinessIssue(
        issue_id=make_factor_readiness_issue_id(
            issue_code=issue_code,
            table_name=table_name,
            symbol=symbol,
            trade_date=trade_date,
            field_name=field_name,
        ),
        issue_code=issue_code,
        severity=severity,
        message=message,
        table_name=table_name,
        symbol=symbol,
        trade_date=trade_date,
        field_name=field_name,
        metadata=dict(metadata or {}),
    )


def build_factor_readiness_report(
    *,
    table_reports: Mapping[str, TushareCleaningReport],
    cleaned_frames: Mapping[str, Any] | None = None,
    quarantined_frames: Mapping[str, Any] | None = None,
    config: FactorReadinessConfig | None = None,
    source_root: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorReadinessReport:
    generated = generated_at or _now_iso()
    resolved_config = config or FactorReadinessConfig(required_factor_fields=["open", "high", "low", "close", "vol", "amount", "adj_factor"])
    frames = dict(cleaned_frames or {})
    quarantined = dict(quarantined_frames or {})
    reports = dict(table_reports or {})
    issues: list[FactorReadinessIssue] = []
    coverage: list[MatrixCoverageSummary] = []
    masks: list[FactorReadyMaskManifest] = []

    if not frames:
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_MISSING_REQUIRED_TABLE,
                severity=CLEANING_ISSUE_BLOCKER,
                message="no cleaned frames available for factor readiness",
            )
        )

    for table_name, cleaning_report in reports.items():
        if cleaning_report.status == CLEANING_STATUS_FAIL:
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_MISSING_REQUIRED_TABLE,
                    severity=CLEANING_ISSUE_BLOCKER,
                    message=f"cleaning failed for table {table_name}",
                    table_name=table_name,
                )
            )

    for table_name, frame in frames.items():
        profile = _resolve_profile(table_name)
        qdf = quarantined.get(table_name)
        summary = build_matrix_coverage_summary(
            frame,
            table_name=table_name,
            profile=profile,
            fields=resolved_config.required_factor_fields or profile.factor_required_columns,
            quarantined_df=qdf,
        )
        coverage.append(summary)
        masks.append(
            build_factor_ready_mask_manifest(
                frame,
                table_name=table_name,
                profile=profile,
                quarantined_df=qdf,
            )
        )
        if (
            summary.symbol_coverage_ratio is not None
            and summary.symbol_coverage_ratio < resolved_config.min_symbol_coverage_ratio
        ):
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_LOW_SYMBOL_COVERAGE,
                    severity=CLEANING_ISSUE_WARNING,
                    message="symbol coverage below threshold",
                    table_name=table_name,
                    metadata={"coverage": summary.symbol_coverage_ratio},
                )
            )
        if (
            summary.date_coverage_ratio is not None
            and summary.date_coverage_ratio < resolved_config.min_date_coverage_ratio
        ):
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_LOW_DATE_COVERAGE,
                    severity=CLEANING_ISSUE_WARNING,
                    message="date coverage below threshold",
                    table_name=table_name,
                    metadata={"coverage": summary.date_coverage_ratio},
                )
            )
        for field_name, ratio in summary.field_coverage.items():
            if ratio < resolved_config.min_field_coverage_ratio:
                issues.append(
                    _readiness_issue(
                        issue_code=READINESS_ISSUE_LOW_FIELD_COVERAGE,
                        severity=CLEANING_ISSUE_WARNING,
                        message="field coverage below threshold",
                        table_name=table_name,
                        field_name=field_name,
                        metadata={"coverage": ratio},
                    )
                )
        raw_count = max(int(reports.get(table_name, TushareCleaningReport(table_name=table_name)).raw_row_count), len(pd.DataFrame(frame)))
        quarantine_count = int(reports.get(table_name, TushareCleaningReport(table_name=table_name)).quarantined_row_count)
        if raw_count and quarantine_count / raw_count > resolved_config.max_quarantine_ratio:
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_EXCESSIVE_QUARANTINE_RATIO,
                    severity=CLEANING_ISSUE_WARNING,
                    message="quarantine ratio above threshold",
                    table_name=table_name,
                    metadata={"quarantine_ratio": quarantine_count / raw_count},
                )
            )

    if resolved_config.require_trade_cal and "trade_cal" not in frames:
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_MISSING_TRADE_CAL,
                severity=CLEANING_ISSUE_BLOCKER,
                message="trade calendar table is required for factor readiness",
                table_name="trade_cal",
            )
        )
    if resolved_config.require_limit_data and "stk_limit" not in frames:
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_MISSING_LIMIT_DATA,
                severity=CLEANING_ISSUE_WARNING,
                message="limit-up/down data missing; limit masks cannot be proven",
                table_name="stk_limit",
            )
        )
    if resolved_config.require_suspend_data and not ({"suspend_d", "suspend"} & set(frames)):
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_MISSING_SUSPEND_DATA,
                severity=CLEANING_ISSUE_WARNING,
                message="suspend data missing; tradability masks cannot be proven",
                table_name="suspend_d",
            )
        )
    if resolved_config.require_index_weight and "index_weight" not in frames:
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_MISSING_INDEX_WEIGHT,
                severity=CLEANING_ISSUE_WARNING,
                message="index weight table missing",
                table_name="index_weight",
            )
        )
    if resolved_config.require_point_in_time_index_weight and "index_weight" not in frames:
        issues.append(
            _readiness_issue(
                issue_code=READINESS_ISSUE_NON_POINT_IN_TIME_INDEX_WEIGHT,
                severity=CLEANING_ISSUE_BLOCKER,
                message="point-in-time index weight is required but unavailable",
                table_name="index_weight",
            )
        )

    if resolved_config.require_adj_factor:
        daily = pd.DataFrame(frames.get("daily", pd.DataFrame()))
        if daily.empty or "adj_factor" not in daily.columns:
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_MISSING_ADJ_FACTOR,
                    severity=CLEANING_ISSUE_BLOCKER,
                    message="adj_factor missing for adjusted-price factor research",
                    table_name="daily",
                    field_name="adj_factor",
                )
            )
        elif not (pd.to_numeric(daily["adj_factor"], errors="coerce") > 0).all():
            issues.append(
                _readiness_issue(
                    issue_code=READINESS_ISSUE_INVALID_ADJ_FACTOR,
                    severity=CLEANING_ISSUE_BLOCKER,
                    message="adj_factor contains missing or non-positive values",
                    table_name="daily",
                    field_name="adj_factor",
                )
            )

    issues.append(
        _readiness_issue(
            issue_code=READINESS_ISSUE_NO_TRADABILITY_MASK,
            severity=CLEANING_ISSUE_WARNING,
            message="no external tradability mask provided; tradable mask remains false",
        )
    )
    issues.append(
        _readiness_issue(
            issue_code=READINESS_ISSUE_NO_BENCHMARK_MEMBERSHIP_MASK,
            severity=CLEANING_ISSUE_WARNING,
            message="no benchmark membership mask provided; benchmark_member mask remains false",
        )
    )

    blocker_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_BLOCKER)
    warning_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_WARNING)
    info_count = sum(1 for issue in issues if issue.severity == CLEANING_ISSUE_INFO)
    if not frames:
        status = FACTOR_READINESS_INSUFFICIENT_DATA
    elif blocker_count:
        status = FACTOR_READINESS_NOT_READY
    elif warning_count:
        status = FACTOR_READINESS_WARN
    else:
        status = FACTOR_READINESS_READY
    readiness_report = FactorReadinessReport(
        report_id=make_factor_readiness_report_id(
            source_root=_path_or_none(source_root),
            generated_at=generated,
            config_id=resolved_config.config_id,
        ),
        generated_at=generated,
        config=resolved_config,
        source_root=_path_or_none(source_root),
        table_reports=reports,
        coverage_summaries=coverage,
        mask_manifests=masks,
        issue_count=len(issues),
        blocker_count=blocker_count,
        warning_count=warning_count,
        info_count=info_count,
        overall_status=status,
        issues=issues,
        metadata=dict(metadata or {}),
    )
    return readiness_report


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
        atomic_write_dataframe_csv(row_flags, paths["row_flags"])
    if cell_flags is not None:
        atomic_write_dataframe_csv(cell_flags, paths["cell_flags"])

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
