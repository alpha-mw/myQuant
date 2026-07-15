"""Pure v3 contracts for authoritative Fundamental provider evidence.

This module intentionally performs no filesystem or provider I/O.  It is shared
by fetch/checkpoint code and the independent promotion validator so both sides
apply the same accounting, frame-semantics, and financial-coverage rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA = "myquant-fundamental-provider-manifest.v3"
FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA = "myquant-fundamental-request-outcome.v3"
FUNDAMENTAL_FETCH_PIT_CONTRACT = "myquant-fundamental-fetch-pit.v3"
FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA = "myquant-fundamental-endpoint-audit.v3"
FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA = "myquant-fundamental-fetch-checkpoint.v3"
FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA = (
    "myquant-fundamental-fetch-checkpoint-pointer.v3"
)
FUNDAMENTAL_DERIVATION_CONTRACT = "myquant-fundamental-derivation.v3"

OUTCOME_ACCOUNTING_FIELDS = (
    "rows_received",
    "rows",
    "rows_hard_invalid",
    "rows_filtered_future",
    "rows_filtered_missing_availability",
    "rows_filtered_core_values",
    "rows_deduplicated",
    "rows_discarded_request_malformed",
)
HARD_INVALID_SUBCOUNTER_FIELDS = (
    "rows_hard_invalid_schema",
    "rows_hard_invalid_symbol",
    "rows_hard_invalid_availability_date",
    "rows_hard_invalid_end_date",
    "rows_hard_invalid_end_after_availability",
    "rows_hard_invalid_core_numeric",
)


@dataclass(frozen=True)
class FundamentalEndpointAuditPolicy:
    """Fail-closed thresholds shared by fetch and promotion validation."""

    critical_min_success_ratio: float = 0.95
    daily_basic_min_success_ratio: float = 0.95
    financial_period_min_coverage_ratio: float = 0.90
    financial_max_consecutive_missing_baseline_periods: int = 1
    financial_require_latest_baseline: bool = True
    max_error_requests: int = 0
    max_malformed_requests: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "critical_min_success_ratio",
            "daily_basic_min_success_ratio",
            "financial_period_min_coverage_ratio",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        for field_name in (
            "financial_max_consecutive_missing_baseline_periods",
            "max_error_requests",
            "max_malformed_requests",
        ):
            strict_nonnegative_int(getattr(self, field_name), label=field_name)
        if not isinstance(self.financial_require_latest_baseline, bool):
            raise TypeError("financial_require_latest_baseline must be bool")


def canonical_json_sha256(value: Any) -> str:
    """Hash one canonical, finite JSON value."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def strict_nonnegative_int(value: Any, *, label: str) -> int:
    """Return a non-negative integer without accepting bool or coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


def _is_null_scalar(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _scalar_token(value: Any) -> tuple[str, str]:
    if _is_null_scalar(value):
        return ("null", "")
    if isinstance(value, (bool, np.bool_)):
        return ("boolean", "true" if bool(value) else "false")
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(value).hex())
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime)):
        return ("datetime", pd.Timestamp(value).isoformat())
    if isinstance(value, date):
        return ("date", value.isoformat())
    if isinstance(value, (pd.Timedelta, np.timedelta64, timedelta)):
        return ("timedelta_ns", str(pd.Timedelta(value).value))
    if isinstance(value, Integral):
        return ("integer", str(int(value)))
    if isinstance(value, Decimal):
        if not value.is_finite():
            return ("decimal", str(value))
        return ("decimal", value.normalize().to_eng_string())
    if isinstance(value, Real):
        return ("real", float(value).hex())
    raise TypeError(f"unsupported frame scalar type: {type(value).__name__}")


def frame_logical_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe positional columns by logical scalar types, not container dtype."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    schema: list[dict[str, Any]] = []
    for position, column in enumerate(frame.columns):
        logical_types: set[str] = set()
        nullable = False
        for value in frame.iloc[:, position].array:
            scalar_type, _payload = _scalar_token(value)
            if scalar_type == "null":
                nullable = True
            else:
                logical_types.add(scalar_type)
        schema.append(
            {
                "position": position,
                "name": list(_scalar_token(column)),
                "logical_scalar_types": sorted(logical_types),
                "nullable": nullable,
            }
        )
    return schema


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint row order, null positions, logical scalar types, and values."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"rows": len(frame), "schema": frame_logical_schema(frame)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for row in frame.itertuples(index=False, name=None):
        tokens = [list(_scalar_token(value)) for value in row]
        digest.update(b"\x00")
        digest.update(
            json.dumps(
                tokens,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def assert_frame_semantics_equal(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Assert exact logical frame equality while ignoring container dtype only."""

    if not isinstance(expected, pd.DataFrame) or not isinstance(actual, pd.DataFrame):
        raise TypeError(f"{label} requires pandas DataFrames")
    if list(expected.columns) != list(actual.columns):
        raise ValueError(f"{label} column order or names changed")
    if len(expected) != len(actual):
        raise ValueError(f"{label} row count changed")
    if frame_logical_schema(expected) != frame_logical_schema(actual):
        raise ValueError(f"{label} logical schema or nullability changed")
    if frame_fingerprint(expected) != frame_fingerprint(actual):
        raise ValueError(f"{label} row order, scalar type, null mask, or value changed")


def validate_outcome_accounting_v3(
    outcome: Mapping[str, Any],
    *,
    label: str = "outcome",
) -> dict[str, int]:
    """Validate one v3 request outcome and return normalized counters."""

    if not isinstance(outcome, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if outcome.get("schema_version") != FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA:
        raise ValueError(f"{label} schema version mismatch")
    status = str(outcome.get("status") or "")
    if status not in {"success", "empty", "error", "malformed"}:
        raise ValueError(f"{label} status is invalid")
    counters = {
        field: strict_nonnegative_int(outcome.get(field), label=f"{label}.{field}")
        for field in (*OUTCOME_ACCOUNTING_FIELDS, *HARD_INVALID_SUBCOUNTER_FIELDS)
    }
    if sum(counters[field] for field in HARD_INVALID_SUBCOUNTER_FIELDS) != counters[
        "rows_hard_invalid"
    ]:
        raise ValueError(f"{label} hard-invalid subcounters do not reconcile")

    if status in {"success", "empty"}:
        if counters["rows_hard_invalid"] or counters[
            "rows_discarded_request_malformed"
        ]:
            raise ValueError(f"{label} clean outcome contains malformed rows")
        expected_received = sum(
            counters[field]
            for field in (
                "rows",
                "rows_filtered_future",
                "rows_filtered_missing_availability",
                "rows_filtered_core_values",
                "rows_deduplicated",
            )
        )
        if counters["rows_received"] != expected_received:
            raise ValueError(f"{label} clean row accounting does not reconcile")
        if status == "success" and counters["rows"] == 0:
            raise ValueError(f"{label} success outcome has no accepted rows")
        if status == "empty" and counters["rows"] != 0:
            raise ValueError(f"{label} empty outcome has accepted rows")
    elif status == "malformed":
        for field in (
            "rows",
            "rows_filtered_future",
            "rows_filtered_missing_availability",
            "rows_filtered_core_values",
            "rows_deduplicated",
        ):
            if counters[field] != 0:
                raise ValueError(f"{label} malformed outcome has clean row counters")
        if counters["rows_received"] != (
            counters["rows_hard_invalid"]
            + counters["rows_discarded_request_malformed"]
        ):
            raise ValueError(f"{label} malformed row accounting does not reconcile")
    else:
        if any(counters.values()):
            raise ValueError(f"{label} provider error must not claim received rows")
    return counters


def _strict_yyyymmdd(value: Any, *, label: str) -> date:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError(f"{label} is not a valid date")
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{label} must be exact YYYYMMDD")
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid YYYYMMDD date") from exc


def matured_quarter_baseline(
    financial_start: Any,
    listing_date: Any,
    history_end: Any,
    as_of: Any,
    *,
    max_periods: int = 20,
    lag_days: int = 120,
) -> list[str]:
    """Return the last matured standard quarter ends within eligibility bounds."""

    start = max(
        _strict_yyyymmdd(financial_start, label="financial_start"),
        _strict_yyyymmdd(listing_date, label="listing_date"),
    )
    end = _strict_yyyymmdd(history_end, label="history_end")
    cutoff = _strict_yyyymmdd(as_of, label="as_of")
    period_limit = strict_nonnegative_int(max_periods, label="max_periods")
    lag = strict_nonnegative_int(lag_days, label="lag_days")
    if period_limit < 1:
        raise ValueError("max_periods must be at least 1")
    if end < start:
        raise ValueError("financial history bounds are reversed")
    effective_end = min(end, cutoff)
    quarter_ends: list[date] = []
    for year in range(start.year, effective_end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if (
                start <= period <= effective_end
                and period + timedelta(days=lag) <= cutoff
            ):
                quarter_ends.append(period)
    return [period.strftime("%Y%m%d") for period in quarter_ends[-period_limit:]]


def _strict_periods(values: Sequence[Any], *, label: str) -> list[str]:
    normalized = sorted(
        {
            _strict_yyyymmdd(value, label=f"{label} period").strftime("%Y%m%d")
            for value in values
        }
    )
    return normalized


def build_financial_coverage(
    expected: Sequence[Any],
    baseline: Sequence[Any],
    covered: Sequence[Any],
    *,
    minimum_ratio: float = 0.90,
    max_consecutive_missing_baseline: int = 1,
    require_latest_baseline: bool = True,
) -> dict[str, Any]:
    """Build deterministic financial-period coverage evidence."""

    if isinstance(minimum_ratio, bool) or not isinstance(minimum_ratio, Real):
        raise TypeError("minimum_ratio must be a real number")
    ratio_floor = float(minimum_ratio)
    if not math.isfinite(ratio_floor) or not 0.0 <= ratio_floor <= 1.0:
        raise ValueError("minimum_ratio must be between 0 and 1")
    missing_limit = strict_nonnegative_int(
        max_consecutive_missing_baseline,
        label="max_consecutive_missing_baseline",
    )
    if not isinstance(require_latest_baseline, bool):
        raise TypeError("require_latest_baseline must be bool")

    expected_periods = _strict_periods(expected, label="expected")
    baseline_periods = _strict_periods(baseline, label="baseline")
    covered_periods = _strict_periods(covered, label="covered")
    expected_set = set(expected_periods)
    baseline_set = set(baseline_periods)
    covered_set = set(covered_periods)
    if not baseline_set.issubset(expected_set):
        raise ValueError("baseline periods must be a subset of expected periods")
    if not covered_set.issubset(expected_set):
        raise ValueError("covered periods must be a subset of expected periods")

    if not expected_periods:
        return {
            "status": "not_applicable",
            "expected_periods": [],
            "baseline_periods": [],
            "covered_periods": [],
            "missing_expected_periods": [],
            "missing_baseline_periods": [],
            "expected_period_count": 0,
            "baseline_period_count": 0,
            "covered_period_count": 0,
            "coverage_ratio": None,
            "minimum_coverage_ratio": ratio_floor,
            "latest_baseline_period": "",
            "latest_baseline_present": True,
            "max_consecutive_missing_baseline_periods": 0,
            "max_allowed_consecutive_missing_baseline_periods": missing_limit,
            "passed": True,
            "blockers": [],
        }

    missing_expected = sorted(expected_set.difference(covered_set))
    missing_baseline = sorted(baseline_set.difference(covered_set))
    maximum_run = 0
    current_run = 0
    for period in baseline_periods:
        if period in covered_set:
            current_run = 0
        else:
            current_run += 1
            maximum_run = max(maximum_run, current_run)
    latest_baseline = baseline_periods[-1] if baseline_periods else ""
    latest_present = not latest_baseline or latest_baseline in covered_set
    ratio = len(covered_set) / len(expected_set) if expected_set else 0.0

    blockers: list[str] = []
    if ratio < ratio_floor:
        blockers.append("financial_period_coverage_below_threshold")
    if require_latest_baseline and not latest_present:
        blockers.append("financial_latest_baseline_missing")
    if maximum_run > missing_limit:
        blockers.append("financial_consecutive_baseline_missing_above_threshold")
    return {
        "status": "applicable",
        "expected_periods": expected_periods,
        "baseline_periods": baseline_periods,
        "covered_periods": covered_periods,
        "missing_expected_periods": missing_expected,
        "missing_baseline_periods": missing_baseline,
        "expected_period_count": len(expected_periods),
        "baseline_period_count": len(baseline_periods),
        "covered_period_count": len(covered_periods),
        "coverage_ratio": float(ratio),
        "minimum_coverage_ratio": ratio_floor,
        "latest_baseline_period": latest_baseline,
        "latest_baseline_present": latest_present,
        "max_consecutive_missing_baseline_periods": maximum_run,
        "max_allowed_consecutive_missing_baseline_periods": missing_limit,
        "passed": not blockers,
        "blockers": blockers,
    }


__all__ = [
    "FUNDAMENTAL_DERIVATION_CONTRACT",
    "FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA",
    "FUNDAMENTAL_FETCH_CHECKPOINT_POINTER_SCHEMA",
    "FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA",
    "FUNDAMENTAL_FETCH_PIT_CONTRACT",
    "FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA",
    "FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA",
    "HARD_INVALID_SUBCOUNTER_FIELDS",
    "OUTCOME_ACCOUNTING_FIELDS",
    "FundamentalEndpointAuditPolicy",
    "assert_frame_semantics_equal",
    "build_financial_coverage",
    "canonical_json_sha256",
    "frame_fingerprint",
    "frame_logical_schema",
    "matured_quarter_baseline",
    "strict_nonnegative_int",
    "validate_outcome_accounting_v3",
]
