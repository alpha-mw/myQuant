"""Offline factor backtest alignment audit helpers.

This module diagnoses whether factor signals, delayed execution windows, price
fields, and execution-return matrices use the same time axis. It is deliberately
read-only and does not wire factors into stock selection, posterior scoring,
RiskGuard, PortfolioConstructor, orders, providers, LLMs, or execution.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from quant_investor.factors.alignment_audit_types import (
    ALIGNMENT_AUDIT_FAIL,
    ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE,
    ALIGNMENT_AUDIT_PASS,
    ALIGNMENT_AUDIT_WARN,
    ALIGNMENT_ISSUE_ALIGNMENT_GAP,
    ALIGNMENT_ISSUE_BLOCKER,
    ALIGNMENT_ISSUE_DATE_ORDER_INVALID,
    ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING,
    ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL,
    ALIGNMENT_ISSUE_INFO,
    ALIGNMENT_ISSUE_INSUFFICIENT_DATES,
    ALIGNMENT_ISSUE_NON_POSITIVE_DELAY,
    ALIGNMENT_ISSUE_PRICE_FIELD_MISSING,
    ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD,
    ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
    ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
    ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY,
    ALIGNMENT_ISSUE_WARNING,
    ALIGNMENT_POLICY_CUSTOM,
    ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1,
    DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME,
    DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME,
    DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR,
    AlignmentAuditRecord,
    FactorBacktestAlignmentAuditConfig,
    FactorBacktestAlignmentAuditReport,
    FactorBacktestAlignmentIssue,
    _FLOAT_TOLERANCE,
    _coerce_metadata,
    _issue_sort_key,
    _positive_int,
    _resolve_execution_price,
    _to_finite_float,
    _to_positive_price,
    make_alignment_audit_config_id,
    make_alignment_audit_report_id,
    make_alignment_issue_id,
    make_alignment_record_id,
)
from quant_investor.factors.backtest import (
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    FactorDailyBacktestRecord,
    SingleFactorBacktestRun,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VOLUME,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
)
from quant_investor.factors.schema import FactorBacktestConfig
from quant_investor.versioning import FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION


def validate_strictly_ascending_dates(dates: Sequence[str]) -> None:
    resolved_dates = [str(value).strip() for value in dates if str(value).strip()]
    if not resolved_dates:
        raise ValueError("dates must be non-empty.")
    parsed_dates: list[date] = []
    for value in resolved_dates:
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dates must be ISO dates; got {value!r}.") from exc
        if parsed_value.isoformat() != value:
            raise ValueError(f"dates must be canonical ISO dates; got {value!r}.")
        parsed_dates.append(parsed_value)
    if len(set(resolved_dates)) != len(resolved_dates):
        raise ValueError("dates must not contain duplicates.")
    if any(current >= next_value for current, next_value in zip(parsed_dates, parsed_dates[1:])):
        raise ValueError("dates must be strictly ascending ISO dates.")


def expected_alignment_tuples(
    dates: Sequence[str],
    *,
    delay_days: int,
    holding_period_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
    execution_price: str = EXECUTION_PRICE_VWAP,
) -> list[dict[str, Any]]:
    validate_strictly_ascending_dates(dates)
    resolved_dates = [str(value).strip() for value in dates if str(value).strip()]
    resolved_delay = _positive_int(delay_days, "delay_days")
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    resolved_execution_price = _resolve_execution_price(execution_price)
    if start_date is not None:
        date.fromisoformat(start_date)
    if end_date is not None:
        date.fromisoformat(end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date.")

    alignments: list[dict[str, Any]] = []
    for signal_index, signal_date in enumerate(resolved_dates):
        if start_date is not None and signal_date < start_date:
            continue
        if end_date is not None and signal_date > end_date:
            continue
        execution_start_index = signal_index + resolved_delay
        execution_end_index = execution_start_index + resolved_holding_period
        if execution_end_index >= len(resolved_dates):
            continue
        execution_start_date = resolved_dates[execution_start_index]
        execution_end_date = resolved_dates[execution_end_index]
        alignments.append(
            {
                "signal_date": signal_date,
                "execution_start_date": execution_start_date,
                "execution_end_date": execution_end_date,
                "signal_index": signal_index,
                "execution_start_index": execution_start_index,
                "execution_end_index": execution_end_index,
                "delay_days": resolved_delay,
                "holding_period_days": resolved_holding_period,
                "execution_price": resolved_execution_price,
            }
        )
    return alignments


def _make_issue(
    *,
    issue_code: str,
    severity: str,
    message: str,
    signal_date: str | None = None,
    execution_start_date: str | None = None,
    execution_end_date: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestAlignmentIssue:
    return FactorBacktestAlignmentIssue(
        issue_id=make_alignment_issue_id(
            issue_code=issue_code,
            signal_date=signal_date,
            execution_start_date=execution_start_date,
            execution_end_date=execution_end_date,
            message=message,
        ),
        issue_code=issue_code,
        severity=severity,
        message=message,
        signal_date=signal_date,
        execution_start_date=execution_start_date,
        execution_end_date=execution_end_date,
        metadata=dict(metadata or {}),
    )


def _field_for_execution_price(execution_price: str) -> str:
    resolved = _resolve_execution_price(execution_price)
    if resolved == EXECUTION_PRICE_OPEN:
        return FIELD_OPEN
    if resolved == EXECUTION_PRICE_CLOSE:
        return FIELD_CLOSE
    return FIELD_VWAP


def _blank_price_matrix(bundle: MatrixDataBundle) -> list[list[float | None]]:
    return [[None for _date in bundle.contract.dates] for _symbol in bundle.contract.symbols]


def _derive_vwap_matrix(bundle: MatrixDataBundle) -> list[list[float | None]]:
    amount = bundle.get_field(FIELD_AMOUNT)
    volume = bundle.get_field(FIELD_VOLUME)
    output = _blank_price_matrix(bundle)
    for row_index, (amount_row, volume_row) in enumerate(zip(amount, volume)):
        for column_index, (amount_value, volume_value) in enumerate(zip(amount_row, volume_row)):
            amount_number = _to_finite_float(amount_value)
            volume_number = _to_finite_float(volume_value)
            if amount_number is None or volume_number is None or volume_number == 0.0:
                continue
            output[row_index][column_index] = amount_number / volume_number
    return output


def _resolve_price_matrix(
    bundle: MatrixDataBundle,
    execution_price: str,
    *,
    require_vwap_derivable: bool = True,
) -> tuple[list[list[Any]] | None, list[FactorBacktestAlignmentIssue]]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    field_name = _field_for_execution_price(resolved_execution_price)
    if field_name != FIELD_VWAP:
        if bundle.has_field(field_name):
            return bundle.get_field(field_name), []
        return None, [
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_PRICE_FIELD_MISSING,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message=f"execution price field {field_name!r} is missing from bundle.",
                metadata={"execution_price": resolved_execution_price, "field_name": field_name},
            )
        ]

    if bundle.has_field(FIELD_VWAP):
        return bundle.get_field(FIELD_VWAP), []
    if bundle.has_field(FIELD_AMOUNT) and bundle.has_field(FIELD_VOLUME):
        return _derive_vwap_matrix(bundle), []
    issue_code = (
        ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING
        if require_vwap_derivable
        else ALIGNMENT_ISSUE_PRICE_FIELD_MISSING
    )
    return None, [
        _make_issue(
            issue_code=issue_code,
            severity=ALIGNMENT_ISSUE_BLOCKER,
            message="vwap is missing and cannot be derived from amount/volume.",
            metadata={
                "execution_price": resolved_execution_price,
                "required_fields": [FIELD_AMOUNT, FIELD_VOLUME],
                "has_amount": bundle.has_field(FIELD_AMOUNT),
                "has_volume": bundle.has_field(FIELD_VOLUME),
            },
        )
    ]


def _validate_matrix_shape(
    matrix: Sequence[Sequence[Any]],
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if len(matrix) != rows:
        raise ValueError(f"{field_name} must have {rows} rows; got {len(matrix)}.")
    for row_index, row in enumerate(matrix):
        if len(row) != columns:
            raise ValueError(
                f"{field_name} row {row_index} must have {columns} columns; got {len(row)}."
            )


def _forward_return_at(
    price_row: Sequence[Any],
    column_index: int,
    holding_period_days: int,
) -> float | None:
    future_index = column_index + holding_period_days
    if future_index >= len(price_row):
        return None
    start_price = _to_positive_price(price_row[column_index])
    end_price = _to_positive_price(price_row[future_index])
    if start_price is None or end_price is None:
        return None
    return end_price / start_price - 1.0


def _previous_return_at(price_row: Sequence[Any], column_index: int) -> float | None:
    if column_index <= 0:
        return None
    previous_price = _to_positive_price(price_row[column_index - 1])
    current_price = _to_positive_price(price_row[column_index])
    if previous_price is None or current_price is None:
        return None
    return current_price / previous_price - 1.0


def _approximately_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= _FLOAT_TOLERANCE


def audit_execution_return_matrix_alignment(
    *,
    bundle: MatrixDataBundle,
    execution_return_matrix: Sequence[Sequence[float | None]],
    execution_price: str,
    holding_period_days: int,
) -> list[FactorBacktestAlignmentIssue]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    symbols = list(bundle.contract.symbols)
    dates = list(bundle.contract.dates)
    validate_strictly_ascending_dates(dates)
    _validate_matrix_shape(
        execution_return_matrix,
        rows=len(symbols),
        columns=len(dates),
        field_name="execution_return_matrix",
    )
    prices, issues = _resolve_price_matrix(
        bundle,
        resolved_execution_price,
        require_vwap_derivable=True,
    )
    if prices is None:
        return sorted(issues, key=_issue_sort_key)
    _validate_matrix_shape(
        prices,
        rows=len(symbols),
        columns=len(dates),
        field_name="execution_price_matrix",
    )

    output = list(issues)
    for row_index, symbol in enumerate(symbols):
        price_row = prices[row_index]
        observed_row = execution_return_matrix[row_index]
        for column_index, signal_date in enumerate(dates):
            execution_end_index = column_index + resolved_holding_period
            execution_end_date = (
                dates[execution_end_index]
                if execution_end_index < len(dates)
                else None
            )
            expected_return = _forward_return_at(
                price_row,
                column_index,
                resolved_holding_period,
            )
            observed_return = _to_finite_float(observed_row[column_index])
            if expected_return is None and observed_return is None:
                continue
            if _approximately_equal(observed_return, expected_return):
                continue
            previous_return = _previous_return_at(price_row, column_index)
            if (
                previous_return is not None
                and observed_return is not None
                and _approximately_equal(observed_return, previous_return)
            ):
                output.append(
                    _make_issue(
                        issue_code=ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD,
                        severity=ALIGNMENT_ISSUE_BLOCKER,
                        message=(
                            f"execution return for {symbol} on {signal_date} matches "
                            "a prior close-to-close return instead of the forward execution window."
                        ),
                        signal_date=signal_date,
                        execution_start_date=signal_date,
                        execution_end_date=execution_end_date,
                        metadata={
                            "symbol": symbol,
                            "row_index": row_index,
                            "date_index": column_index,
                            "expected_return": expected_return,
                            "observed_return": observed_return,
                            "prior_return": previous_return,
                            "execution_price": resolved_execution_price,
                            "holding_period_days": resolved_holding_period,
                        },
                    )
                )
                continue
            output.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_WARNING,
                    message=(
                        f"execution return for {symbol} on {signal_date} does not match "
                        "the expected forward execution window."
                    ),
                    signal_date=signal_date,
                    execution_start_date=signal_date,
                    execution_end_date=execution_end_date,
                    metadata={
                        "symbol": symbol,
                        "row_index": row_index,
                        "date_index": column_index,
                        "expected_return": expected_return,
                        "observed_return": observed_return,
                        "execution_price": resolved_execution_price,
                        "holding_period_days": resolved_holding_period,
                    },
                )
            )
    return sorted(output, key=_issue_sort_key)


def _holding_period_from_run(run: SingleFactorBacktestRun | None) -> int | None:
    if run is None:
        return None
    candidates: list[Any] = []
    candidates.append(run.metadata.get("holding_period_days"))
    candidates.append(run.aggregate_result.metadata.get("holding_period_days"))
    for record in run.daily_records:
        candidates.append(record.metadata.get("holding_period_days"))
        alignment_payload = record.metadata.get("alignment")
        if isinstance(alignment_payload, Mapping):
            candidates.append(alignment_payload.get("holding_period_days"))
    for candidate in candidates:
        if candidate is None:
            continue
        return _positive_int(candidate, "holding_period_days")
    return None


def _build_default_audit_config(
    config: FactorBacktestConfig,
    run: SingleFactorBacktestRun | None,
) -> FactorBacktestAlignmentAuditConfig:
    holding_period_days = _holding_period_from_run(run) or 1
    expected_policy = (
        ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1
        if config.delay_days == 1 and holding_period_days == 1
        else ALIGNMENT_POLICY_CUSTOM
    )
    audit_config = FactorBacktestAlignmentAuditConfig(
        config_id="placeholder",
        expected_policy=expected_policy,
        expected_delay_days=config.delay_days,
        expected_holding_period_days=holding_period_days,
        execution_price=config.execution_price,
        allow_custom_policy=expected_policy == ALIGNMENT_POLICY_CUSTOM,
        metadata={"source": "FactorBacktestConfig"},
    )
    audit_config.config_id = make_alignment_audit_config_id(audit_config)
    return audit_config


def _factor_bundle_symbols_dates_match(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
) -> bool:
    return (
        list(factor_matrix.symbols) == list(bundle.contract.symbols)
        and list(factor_matrix.dates) == list(bundle.contract.dates)
    )


def _calendar_index(dates: Sequence[str]) -> dict[str, int]:
    return {date_value: index for index, date_value in enumerate(dates)}


def _record_chronology_issues(
    record: FactorDailyBacktestRecord,
    date_to_index: Mapping[str, int],
) -> list[FactorBacktestAlignmentIssue]:
    output: list[FactorBacktestAlignmentIssue] = []
    signal_index = date_to_index.get(record.signal_date)
    execution_start_index = date_to_index.get(record.execution_start_date)
    execution_end_index = date_to_index.get(record.execution_end_date)
    if signal_index is None or execution_start_index is None or execution_end_index is None:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                severity=ALIGNMENT_ISSUE_WARNING,
                message="run daily record contains dates outside the factor matrix calendar.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={
                    "record_date": record.date,
                    "signal_date_in_calendar": signal_index is not None,
                    "execution_start_date_in_calendar": execution_start_index is not None,
                    "execution_end_date_in_calendar": execution_end_index is not None,
                },
            )
        )
        return output
    if execution_start_index == signal_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record executes on the same date as the signal.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if execution_start_index < signal_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record execution starts before the signal date.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if execution_end_index <= execution_start_index:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record return window does not end after execution start.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    if record.date != record.execution_end_date:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_DATE_ORDER_INVALID,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="run daily record date must equal execution_end_date.",
                signal_date=record.signal_date,
                execution_start_date=record.execution_start_date,
                execution_end_date=record.execution_end_date,
                metadata={"record_date": record.date},
            )
        )
    return output


def _run_record_alignment_issues(
    alignments: Sequence[Mapping[str, Any]],
    run: SingleFactorBacktestRun,
    dates: Sequence[str],
) -> list[FactorBacktestAlignmentIssue]:
    output: list[FactorBacktestAlignmentIssue] = []
    date_to_index = _calendar_index(dates)
    run_records = sorted(run.daily_records, key=lambda record: record.date)
    expected_count = len(alignments)
    observed_count = len(run_records)
    if expected_count != observed_count:
        output.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                severity=ALIGNMENT_ISSUE_WARNING,
                message=(
                    "run daily record count does not match expected alignment count "
                    f"({observed_count} observed vs {expected_count} expected)."
                ),
                metadata={
                    "observed_count": observed_count,
                    "expected_count": expected_count,
                    "run_id": run.run_id,
                },
            )
        )
    for record in run_records:
        output.extend(_record_chronology_issues(record, date_to_index))
    for index, alignment in enumerate(alignments):
        if index >= len(run_records):
            break
        record = run_records[index]
        expected_fields = {
            "signal_date": str(alignment["signal_date"]),
            "execution_start_date": str(alignment["execution_start_date"]),
            "execution_end_date": str(alignment["execution_end_date"]),
            "date": str(alignment["execution_end_date"]),
        }
        observed_fields = {
            "signal_date": record.signal_date,
            "execution_start_date": record.execution_start_date,
            "execution_end_date": record.execution_end_date,
            "date": record.date,
        }
        mismatches = {
            field_name: {
                "expected": expected_value,
                "observed": observed_fields[field_name],
            }
            for field_name, expected_value in expected_fields.items()
            if observed_fields[field_name] != expected_value
        }
        if mismatches:
            output.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_WARNING,
                    message="run daily record does not match expected alignment tuple.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata={
                        "run_id": run.run_id,
                        "record_index": index,
                        "record_date": record.date,
                        "mismatches": mismatches,
                    },
                )
            )
    return output


def _issues_for_alignment(
    issues: Sequence[FactorBacktestAlignmentIssue],
    alignment: Mapping[str, Any],
) -> list[FactorBacktestAlignmentIssue]:
    signal_date = str(alignment["signal_date"])
    execution_start_date = str(alignment["execution_start_date"])
    execution_end_date = str(alignment["execution_end_date"])
    return [
        issue for issue in issues
        if (
            issue.signal_date in (None, signal_date)
            and issue.execution_start_date in (None, execution_start_date)
            and issue.execution_end_date in (None, execution_end_date)
        )
    ]


def _verdict_from_issues(issues: Sequence[FactorBacktestAlignmentIssue]) -> str:
    if any(issue.severity == ALIGNMENT_ISSUE_BLOCKER for issue in issues):
        return ALIGNMENT_AUDIT_FAIL
    if any(issue.severity == ALIGNMENT_ISSUE_WARNING for issue in issues):
        return ALIGNMENT_AUDIT_WARN
    return ALIGNMENT_AUDIT_PASS


def _deduplicate_issues(
    issues: Sequence[FactorBacktestAlignmentIssue],
) -> list[FactorBacktestAlignmentIssue]:
    by_id: dict[str, FactorBacktestAlignmentIssue] = {}
    for issue in issues:
        by_id.setdefault(issue.issue_id, issue)
    return sorted(by_id.values(), key=_issue_sort_key)


def audit_factor_backtest_alignment(
    *,
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    run: SingleFactorBacktestRun | None = None,
    audit_config: FactorBacktestAlignmentAuditConfig | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestAlignmentAuditReport:
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    validate_strictly_ascending_dates(factor_matrix.dates)
    validate_strictly_ascending_dates(bundle.contract.dates)
    resolved_audit_config = audit_config or _build_default_audit_config(config, run)
    resolved_execution_price = _resolve_execution_price(resolved_audit_config.execution_price)
    issues: list[FactorBacktestAlignmentIssue] = []

    _price_matrix, price_issues = _resolve_price_matrix(
        bundle,
        resolved_execution_price,
        require_vwap_derivable=resolved_audit_config.require_vwap_derivable,
    )
    issues.extend(price_issues)

    if resolved_audit_config.expected_policy == ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1:
        if resolved_audit_config.expected_delay_days != 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="T+1 alignment policy requires expected_delay_days=1.",
                    metadata={"expected_delay_days": resolved_audit_config.expected_delay_days},
                )
            )
        if resolved_audit_config.expected_holding_period_days != 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_ALIGNMENT_GAP,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="T+1 alignment policy requires expected_holding_period_days=1.",
                    metadata={
                        "expected_holding_period_days": (
                            resolved_audit_config.expected_holding_period_days
                        )
                    },
                )
            )
    if (
        resolved_audit_config.expected_policy == ALIGNMENT_POLICY_CUSTOM
        and not resolved_audit_config.allow_custom_policy
    ):
        issues.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY,
                severity=ALIGNMENT_ISSUE_WARNING,
                message="custom alignment policy is not explicitly allowed.",
                metadata={"expected_policy": resolved_audit_config.expected_policy},
            )
        )

    alignments = expected_alignment_tuples(
        factor_matrix.dates,
        delay_days=resolved_audit_config.expected_delay_days,
        holding_period_days=resolved_audit_config.expected_holding_period_days,
        start_date=config.start_date or None,
        end_date=config.end_date or None,
        execution_price=resolved_execution_price,
    )
    if not alignments:
        issues.append(
            _make_issue(
                issue_code=ALIGNMENT_ISSUE_INSUFFICIENT_DATES,
                severity=ALIGNMENT_ISSUE_BLOCKER,
                message="no signal/execution/return windows fit the requested audit config.",
                metadata={
                    "date_count": len(factor_matrix.dates),
                    "delay_days": resolved_audit_config.expected_delay_days,
                    "holding_period_days": resolved_audit_config.expected_holding_period_days,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                },
            )
        )

    for alignment in alignments:
        if int(alignment["delay_days"]) < 1:
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_NON_POSITIVE_DELAY,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="delay_days must be positive.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata={"delay_days": alignment["delay_days"]},
                )
            )
        if int(alignment["execution_start_index"]) <= int(alignment["signal_index"]):
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_SAME_DAY_EXECUTION,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="execution_start_index must be after signal_index.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata=dict(alignment),
                )
            )
        if int(alignment["execution_end_index"]) <= int(alignment["execution_start_index"]):
            issues.append(
                _make_issue(
                    issue_code=ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL,
                    severity=ALIGNMENT_ISSUE_BLOCKER,
                    message="execution_end_index must be after execution_start_index.",
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    metadata=dict(alignment),
                )
            )

    if run is not None:
        issues.extend(_run_record_alignment_issues(alignments, run, factor_matrix.dates))

    issues = _deduplicate_issues(issues)
    records: list[AlignmentAuditRecord] = []
    for alignment in alignments:
        alignment_issues = _issues_for_alignment(issues, alignment)
        record_issue_codes = [issue.issue_code for issue in alignment_issues]
        record_passed = not any(
            issue.severity == ALIGNMENT_ISSUE_BLOCKER for issue in alignment_issues
        )
        records.append(
            AlignmentAuditRecord(
                record_id=make_alignment_record_id(
                    signal_date=str(alignment["signal_date"]),
                    execution_start_date=str(alignment["execution_start_date"]),
                    execution_end_date=str(alignment["execution_end_date"]),
                    execution_price=resolved_execution_price,
                ),
                signal_date=str(alignment["signal_date"]),
                execution_start_date=str(alignment["execution_start_date"]),
                execution_end_date=str(alignment["execution_end_date"]),
                signal_index=int(alignment["signal_index"]),
                execution_start_index=int(alignment["execution_start_index"]),
                execution_end_index=int(alignment["execution_end_index"]),
                delay_days=int(alignment["delay_days"]),
                holding_period_days=int(alignment["holding_period_days"]),
                execution_price=resolved_execution_price,
                expected_return_source_index=int(alignment["execution_start_index"]),
                observed_weight_source_index=int(alignment["signal_index"]),
                passed=record_passed,
                issue_codes=record_issue_codes,
                metadata={
                    "return_window_after_signal": True,
                    "return_window_after_execution_start": (
                        int(alignment["execution_end_index"])
                        > int(alignment["execution_start_index"])
                    ),
                    "weight_source": "signal_date",
                    "return_source": "execution_start_date",
                },
            )
        )

    report_metadata = _coerce_metadata(metadata)
    report_metadata.update(
        {
            "factor_backtest_alignment_audit_schema_version": (
                FACTOR_BACKTEST_ALIGNMENT_AUDIT_SCHEMA_VERSION
            ),
            "delay_days": resolved_audit_config.expected_delay_days,
            "holding_period_days": resolved_audit_config.expected_holding_period_days,
            "execution_price": resolved_execution_price,
            "non_runtime_impact": True,
        }
    )
    verdict = _verdict_from_issues(issues)
    return FactorBacktestAlignmentAuditReport(
        report_id=make_alignment_audit_report_id(
            factor_matrix_id=factor_matrix.matrix_id,
            backtest_run_id=run.run_id if run is not None else None,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        factor_matrix_id=factor_matrix.matrix_id,
        backtest_run_id=run.run_id if run is not None else None,
        config=resolved_audit_config,
        total_records=len(records),
        passed_records=sum(1 for record in records if record.passed),
        failed_records=sum(1 for record in records if not record.passed),
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == ALIGNMENT_ISSUE_INFO),
        records=records,
        issues=issues,
        verdict=verdict,
        metadata=report_metadata,
    )


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_alignment_audit_markdown(
    report: FactorBacktestAlignmentAuditReport,
) -> str:
    lines = [
        "# Factor Backtest Alignment Audit",
        "",
        f"Generated at: `{_markdown_cell(report.generated_at)}`",
        "",
        f"Verdict: `{_markdown_cell(report.verdict)}`",
        "",
        "## Config Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Policy | `{_markdown_cell(report.config.expected_policy)}` |",
        f"| Delay days | `{report.config.expected_delay_days}` |",
        f"| Holding period days | `{report.config.expected_holding_period_days}` |",
        f"| Execution price | `{_markdown_cell(report.config.execution_price)}` |",
        f"| Require VWAP derivable | `{report.config.require_vwap_derivable}` |",
        f"| Require return window after execution | `{report.config.require_return_window_after_execution}` |",
        f"| Allow custom policy | `{report.config.allow_custom_policy}` |",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Total records | {report.total_records} |",
        f"| Passed records | {report.passed_records} |",
        f"| Failed records | {report.failed_records} |",
        f"| Issues | {report.issue_count} |",
        f"| Blockers | {report.blocker_count} |",
        f"| Warnings | {report.warning_count} |",
        f"| Info | {report.info_count} |",
        "",
        "## Alignment Records",
        "",
        "| Signal | Execute start | Execute end | Delay | Hold | Price | Weight idx | Return idx | Passed | Issues |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    if report.records:
        for record in report.records:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(record.signal_date),
                        _markdown_cell(record.execution_start_date),
                        _markdown_cell(record.execution_end_date),
                        str(record.delay_days),
                        str(record.holding_period_days),
                        _markdown_cell(record.execution_price),
                        str(record.observed_weight_source_index),
                        str(record.expected_return_source_index),
                        _markdown_cell(record.passed),
                        _markdown_cell(", ".join(record.issue_codes)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _No records_ |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Issues",
            "",
            "| Severity | Code | Signal | Execute start | Execute end | Message |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(issue.severity),
                        _markdown_cell(issue.issue_code),
                        _markdown_cell(issue.signal_date),
                        _markdown_cell(issue.execution_start_date),
                        _markdown_cell(issue.execution_end_date),
                        _markdown_cell(issue.message),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _No issues_ |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Non-Runtime Impact",
            "",
            ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ALIGNMENT_AUDIT_PASS",
    "ALIGNMENT_AUDIT_WARN",
    "ALIGNMENT_AUDIT_FAIL",
    "ALIGNMENT_ISSUE_INFO",
    "ALIGNMENT_ISSUE_WARNING",
    "ALIGNMENT_ISSUE_BLOCKER",
    "ALIGNMENT_ISSUE_NON_POSITIVE_DELAY",
    "ALIGNMENT_ISSUE_SAME_DAY_EXECUTION",
    "ALIGNMENT_ISSUE_EXECUTION_BEFORE_SIGNAL",
    "ALIGNMENT_ISSUE_RETURN_WINDOW_OVERLAP_SIGNAL",
    "ALIGNMENT_ISSUE_RETURN_MATRIX_LOOKAHEAD",
    "ALIGNMENT_ISSUE_PRICE_FIELD_MISSING",
    "ALIGNMENT_ISSUE_DERIVED_VWAP_MISSING",
    "ALIGNMENT_ISSUE_DATE_ORDER_INVALID",
    "ALIGNMENT_ISSUE_ALIGNMENT_GAP",
    "ALIGNMENT_ISSUE_UNEXPLAINED_DELAY_POLICY",
    "ALIGNMENT_ISSUE_INSUFFICIENT_DATES",
    "ALIGNMENT_POLICY_SIGNAL_T_EXECUTE_T_PLUS_1",
    "ALIGNMENT_POLICY_CUSTOM",
    "DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR",
    "DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME",
    "DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME",
    "ALIGNMENT_AUDIT_NON_RUNTIME_IMPACT_NOTE",
    "FactorBacktestAlignmentIssue",
    "FactorBacktestAlignmentAuditConfig",
    "AlignmentAuditRecord",
    "FactorBacktestAlignmentAuditReport",
    "make_alignment_audit_config_id",
    "make_alignment_issue_id",
    "make_alignment_record_id",
    "make_alignment_audit_report_id",
    "validate_strictly_ascending_dates",
    "expected_alignment_tuples",
    "audit_execution_return_matrix_alignment",
    "audit_factor_backtest_alignment",
    "render_alignment_audit_markdown",
]
