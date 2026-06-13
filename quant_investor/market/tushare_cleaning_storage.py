"""Storage and artifact helpers for offline Tushare cleaning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.tushare_cleaning_types import (
    PARQUET_STATUS_FAILED,
    PARQUET_STATUS_SHADOW_WRITTEN,
    PARQUET_STATUS_SKIPPED,
    PARQUET_STATUS_UNSUPPORTED,
    STORAGE_ISSUE_CANONICAL_PARQUET_DISABLED,
    STORAGE_ISSUE_CSV_LARGE_FOR_ANALYTICS,
    STORAGE_ISSUE_CSV_RETAINED_FOR_COMPATIBILITY,
    STORAGE_ISSUE_DELETE_CSV_DISABLED,
    STORAGE_ISSUE_PARQUET_BACKEND_MISSING,
    STORAGE_ISSUE_PARQUET_READBACK_MISMATCH,
    STORAGE_ISSUE_PARQUET_WRITE_FAILED,
    STORAGE_ISSUE_REDUNDANT_DUPLICATE_FILE,
    STORAGE_STATUS_EFFICIENT,
    STORAGE_STATUS_REDUNDANT,
    STORAGE_STATUS_WARN,
    ParquetMigrationReport,
    TushareStorageAuditReport,
    TushareStorageOptimizationConfig,
    _MATRIX_RELEVANT_TABLES,
    _json_ready,
    _path_or_none,
    make_parquet_migration_report_id,
    make_storage_audit_report_id,
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


__all__ = [
    "atomic_write_dataframe_csv",
    "build_storage_audit_report",
    "detect_parquet_backend",
    "safe_json_dump",
    "sha256_file",
    "write_parquet_shadow_if_supported",
]
