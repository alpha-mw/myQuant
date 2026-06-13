"""Factor-readiness summaries and masks for cleaned Tushare tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market.tushare_cleaning_profiles import get_default_tushare_clean_profiles
from quant_investor.market.tushare_cleaning_types import (
    CLEANING_ISSUE_BLOCKER,
    CLEANING_ISSUE_INFO,
    CLEANING_ISSUE_WARNING,
    CLEANING_STATUS_FAIL,
    FACTOR_READINESS_INSUFFICIENT_DATA,
    FACTOR_READINESS_NOT_READY,
    FACTOR_READINESS_READY,
    FACTOR_READINESS_WARN,
    READINESS_ISSUE_EXCESSIVE_QUARANTINE_RATIO,
    READINESS_ISSUE_INVALID_ADJ_FACTOR,
    READINESS_ISSUE_LOW_DATE_COVERAGE,
    READINESS_ISSUE_LOW_FIELD_COVERAGE,
    READINESS_ISSUE_LOW_SYMBOL_COVERAGE,
    READINESS_ISSUE_MISSING_ADJ_FACTOR,
    READINESS_ISSUE_MISSING_INDEX_WEIGHT,
    READINESS_ISSUE_MISSING_LIMIT_DATA,
    READINESS_ISSUE_MISSING_REQUIRED_TABLE,
    READINESS_ISSUE_MISSING_SUSPEND_DATA,
    READINESS_ISSUE_MISSING_TRADE_CAL,
    READINESS_ISSUE_NO_BENCHMARK_MEMBERSHIP_MASK,
    READINESS_ISSUE_NO_TRADABILITY_MASK,
    READINESS_ISSUE_NON_POINT_IN_TIME_INDEX_WEIGHT,
    FactorReadinessConfig,
    FactorReadinessIssue,
    FactorReadinessReport,
    FactorReadyMaskManifest,
    MatrixCoverageSummary,
    TushareCleanProfile,
    TushareCleaningReport,
    _now_iso,
    _path_or_none,
    _sorted_unique,
    make_factor_readiness_issue_id,
    make_factor_readiness_report_id,
)


def _resolve_profile(table_name: str, profile: TushareCleanProfile | None = None) -> TushareCleanProfile:
    if profile is not None:
        return profile
    profiles = get_default_tushare_clean_profiles()
    key = str(table_name or "daily").strip().lower()
    if key not in profiles:
        raise ValueError(f"unknown Tushare clean profile: {table_name}")
    return profiles[key]


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
    resolved_config = config or FactorReadinessConfig(
        required_factor_fields=["open", "high", "low", "close", "vol", "amount", "adj_factor"]
    )
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
        fallback_report = TushareCleaningReport(table_name=table_name)
        table_report = reports.get(table_name, fallback_report)
        raw_count = max(int(table_report.raw_row_count), len(pd.DataFrame(frame)))
        quarantine_count = int(table_report.quarantined_row_count)
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
    return FactorReadinessReport(
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


__all__ = [
    "build_factor_readiness_report",
    "build_factor_ready_mask_manifest",
    "build_matrix_coverage_summary",
]
