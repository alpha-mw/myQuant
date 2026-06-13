"""Core helper functions for offline Tushare cleaning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.tushare_cleaning_profiles import get_default_tushare_clean_profiles
from quant_investor.market.tushare_cleaning_types import (
    CLEANING_ISSUE_BLOCKER,
    CLEANING_ISSUE_INFO,
    CLEANING_ISSUE_WARNING,
    CLEANING_STATUS_FAIL,
    CLEANING_STATUS_PASS,
    CLEANING_STATUS_WARN,
    TushareCleanProfile,
    TushareCleaningIssue,
    TushareCleaningReport,
    _json_ready,
    _path_or_none,
    _stable_json,
    make_cleaning_issue_id,
    make_cleaning_report_id,
)


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


__all__ = [
    "_build_cleaning_report",
    "_cell_flag",
    "_issue",
    "_normalize_date_value",
    "_resolve_profile",
    "_valid_ts_code",
]
