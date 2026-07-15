from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_investor.market.fundamental_provider_contract import (
    FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
    HARD_INVALID_SUBCOUNTER_FIELDS,
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


def _outcome(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
        "status": "success",
        "rows_received": 7,
        "rows": 3,
        "rows_hard_invalid": 0,
        "rows_filtered_future": 1,
        "rows_filtered_missing_availability": 1,
        "rows_filtered_core_values": 1,
        "rows_deduplicated": 1,
        "rows_discarded_request_malformed": 0,
        **{field: 0 for field in HARD_INVALID_SUBCOUNTER_FIELDS},
    }
    payload.update(overrides)
    return payload


def test_canonical_json_hash_is_order_independent_and_finite() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError, match="finite JSON"):
        canonical_json_sha256({"bad": float("nan")})


@pytest.mark.parametrize("value", [True, False, 1.0, "1", -1])
def test_strict_nonnegative_int_rejects_coercion_and_negative(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        strict_nonnegative_int(value, label="counter")


def test_strict_nonnegative_int_accepts_numpy_integer() -> None:
    assert strict_nonnegative_int(np.int64(3), label="counter") == 3


def test_frame_semantics_allow_object_to_string_container_dtype() -> None:
    expected = pd.DataFrame(
        {
            "symbol": pd.Series(["002204.SZ", None], dtype=object),
            "value": pd.Series([1.5, np.nan], dtype=float),
        }
    )
    actual = pd.DataFrame(
        {
            "symbol": pd.Series(["002204.SZ", pd.NA], dtype="string"),
            "value": pd.Series([1.5, pd.NA], dtype="Float64"),
        }
    )

    assert_frame_semantics_equal(expected, actual, label="roundtrip")
    assert frame_logical_schema(expected) == frame_logical_schema(actual)
    assert frame_fingerprint(expected) == frame_fingerprint(actual)


@pytest.mark.parametrize(
    "actual",
    [
        pd.DataFrame({"value": ["1"]}),
        pd.DataFrame({"value": [pd.Timestamp("2024-05-10")]}),
    ],
)
def test_frame_semantics_reject_scalar_type_rewrite(actual: pd.DataFrame) -> None:
    expected = pd.DataFrame({"value": [1 if actual.iloc[0, 0] == "1" else "20240510"]})

    with pytest.raises(ValueError, match="logical schema"):
        assert_frame_semantics_equal(expected, actual, label="roundtrip")


def test_frame_semantics_reject_row_order_null_mask_and_column_order() -> None:
    expected = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    with pytest.raises(ValueError):
        assert_frame_semantics_equal(
            expected,
            expected.iloc[::-1].reset_index(drop=True),
            label="rows",
        )
    with pytest.raises(ValueError):
        assert_frame_semantics_equal(expected, expected[["b", "a"]], label="columns")


def test_validate_clean_outcome_accounting() -> None:
    counters = validate_outcome_accounting_v3(_outcome())

    assert counters["rows_received"] == 7


def test_validate_malformed_outcome_accounting() -> None:
    payload = _outcome(
        status="malformed",
        rows_received=7,
        rows=0,
        rows_hard_invalid=2,
        rows_filtered_future=0,
        rows_filtered_missing_availability=0,
        rows_filtered_core_values=0,
        rows_deduplicated=0,
        rows_discarded_request_malformed=5,
        rows_hard_invalid_schema=2,
    )

    counters = validate_outcome_accounting_v3(payload)

    assert counters["rows_hard_invalid"] == 2


def test_validate_malformed_outcome_accounts_core_numeric_hard_invalid() -> None:
    payload = _outcome(
        status="malformed",
        rows_received=3,
        rows=0,
        rows_hard_invalid=1,
        rows_filtered_future=0,
        rows_filtered_missing_availability=0,
        rows_filtered_core_values=0,
        rows_deduplicated=0,
        rows_discarded_request_malformed=2,
        rows_hard_invalid_core_numeric=1,
    )

    assert validate_outcome_accounting_v3(payload)["rows_hard_invalid"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        _outcome(rows_received=8),
        _outcome(rows=True),
        _outcome(rows_hard_invalid=1),
        _outcome(
            status="malformed",
            rows_received=7,
            rows=0,
            rows_hard_invalid=2,
            rows_filtered_future=0,
            rows_filtered_missing_availability=0,
            rows_filtered_core_values=0,
            rows_deduplicated=0,
            rows_discarded_request_malformed=4,
            rows_hard_invalid_schema=2,
        ),
    ],
)
def test_validate_outcome_rejects_non_reconciling_or_coerced_counts(
    payload: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_outcome_accounting_v3(payload)


def test_matured_quarter_baseline_uses_120_day_lag_bounds_and_last_twenty() -> None:
    periods = matured_quarter_baseline(
        "20180101",
        date(2018, 4, 10),
        "20260714",
        "20260714",
    )

    assert len(periods) == 20
    assert periods[-1] == "20251231"
    assert "20260331" not in periods
    assert periods[0] == "20210331"


def test_matured_quarter_baseline_includes_period_on_exact_maturity_day() -> None:
    assert matured_quarter_baseline(
        "20240101",
        "20240101",
        "20241231",
        "20240729",
    ) == ["20240331"]


def test_financial_coverage_requires_ratio_latest_and_consecutive_baseline() -> None:
    baseline = ["20230331", "20230630", "20230930", "20231231"]
    expected = [*baseline, "20240331"]
    passing = build_financial_coverage(
        expected,
        baseline,
        expected,
    )
    missing_latest = build_financial_coverage(
        expected,
        baseline,
        expected[:-2] + ["20240331"],
    )
    consecutive_missing = build_financial_coverage(
        expected,
        baseline,
        ["20230331", "20231231", "20240331"],
        minimum_ratio=0.50,
    )

    assert passing["passed"] is True
    assert missing_latest["latest_baseline_present"] is False
    assert "financial_latest_baseline_missing" in missing_latest["blockers"]
    assert consecutive_missing["max_consecutive_missing_baseline_periods"] == 2
    assert consecutive_missing["passed"] is False


def test_financial_coverage_is_inclusive_at_ninety_percent() -> None:
    expected = [f"{year}1231" for year in range(2015, 2025)]
    coverage = build_financial_coverage(
        expected,
        expected,
        expected[1:],
        max_consecutive_missing_baseline=1,
        require_latest_baseline=True,
    )

    assert coverage["coverage_ratio"] == pytest.approx(0.9)
    assert coverage["passed"] is True


def test_financial_coverage_empty_expected_is_not_applicable() -> None:
    coverage = build_financial_coverage([], [], [])

    assert coverage["status"] == "not_applicable"
    assert coverage["passed"] is True
    assert coverage["coverage_ratio"] is None
    assert coverage["blockers"] == []


def test_policy_rejects_bool_threshold_and_negative_request_limit() -> None:
    with pytest.raises(TypeError):
        FundamentalEndpointAuditPolicy(critical_min_success_ratio=True)
    with pytest.raises(ValueError):
        FundamentalEndpointAuditPolicy(max_error_requests=-1)
    with pytest.raises(ValueError):
        FundamentalEndpointAuditPolicy(daily_history_boundary_tolerance_days=-1)


def test_matured_baseline_excludes_report_due_after_eligibility_end() -> None:
    baseline = matured_quarter_baseline(
        "20210101",
        "20210101",
        "20260413",
        "20260714",
    )

    assert baseline[-1] == "20250930"
    assert "20251231" not in baseline
