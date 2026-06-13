"""Tushare cleaning schemas, statuses, and stable IDs.

This module is intentionally pure and offline.  The heavier cleaning runtime in
``tushare_data_cleaning`` imports and re-exports these names for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
