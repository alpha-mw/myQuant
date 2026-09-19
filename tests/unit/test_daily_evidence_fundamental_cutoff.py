"""Native Fundamental scoring cannot see future rows or choose ambiguous revisions."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from quant_investor.intelligence.daily_evidence import (
    build_fundamental_assessments_from_frame,
    FUNDAMENTAL_METRICS,
)
from quant_investor.intelligence.fundamental_time import (
    session_date,
    source_available_at,
    FundamentalTemporalError,
    bind_native_availability,
)

AS_OF = "2026-08-28T13:30:00Z"


def row(company, day, value):
    return {
        "ts_code": company,
        "trade_date": day,
        **{metric: value for metric in FUNDAMENTAL_METRICS},
    }


def assess(rows):
    return build_fundamental_assessments_from_frame(
        frame=pd.DataFrame(rows),
        companies=["002384.SZ", "002463.SZ"],
        source_path="fixture.parquet",
        source_sha256="a" * 64,
        source_available_at="2026-08-20T08:00:00Z",
        as_of=AS_OF,
        industry_assessments={},
        theme_assessments={},
    )


def test_future_rows_cannot_affect_selection_or_normalization():
    baseline = [row("002384.SZ", "20260820", 1.5), row("002463.SZ", "20260820", 1.0)]
    expected = assess(baseline)
    mixed = [*baseline, row("002463.SZ", "20260830", 3.0), row("000001.SZ", "20260831", 999.0)]
    assert assess(mixed) == expected
    assessments, sources = assess([row("002463.SZ", "20260830", 3.0)])
    assert assessments == {} and sources == []


def test_latest_eligible_snapshot_and_exact_duplicate_are_deterministic():
    rows = [row("002463.SZ", "20260820", 1.0), row("002463.SZ", "20260827", 2.0)]
    assert assess(rows) == assess([rows[-1], *rows])
    assert (
        assess(rows)[1][0]["payload"]["metrics"]["snapshot_trade_date"] == "20260827.000000000000"
    )


@pytest.mark.parametrize("provenance", [False, True])
def test_conflicting_same_day_rows_reject_in_either_order(provenance):
    rows = [
        row("002463.SZ", "20260820", 1.0),
        row("002463.SZ", "20260820", 1.0 if provenance else 2.0),
    ]
    if provenance:
        rows[0]["source_revision"] = "original"
        rows[1]["source_revision"] = "revised"
    for selected in (rows, list(reversed(rows))):
        with pytest.raises(FundamentalTemporalError, match="REVISION_AMBIGUOUS"):
            assess(selected)


@pytest.mark.parametrize(
    "value",
    [
        "20260828",
        "2026-08-28",
        20260828,
        date(2026, 8, 28),
        datetime(2026, 8, 28, 0, 0),
        pd.Timestamp("2026-08-28"),
        datetime(2026, 8, 27, 16, tzinfo=timezone.utc),
    ],
)
def test_explicit_session_representations(value):
    assert session_date(value) == date(2026, 8, 28)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        20260828.0,
        Decimal("20260828"),
        "08/28/2026",
        "20260230",
        pd.NaT,
        float("nan"),
    ],
)
def test_ambiguous_or_invalid_dates_reject(value):
    with pytest.raises(FundamentalTemporalError, match="PIT_DATE_INVALID"):
        session_date(value)


def test_availability_rounds_up_and_never_crosses_decision_cutoff():
    assert source_available_at("2026-08-28T13:29:59.900000Z", as_of=AS_OF) == AS_OF
    with pytest.raises(FundamentalTemporalError, match="NOT_AVAILABLE_AT_DECISION"):
        source_available_at("2026-08-28T13:30:00.000001Z", as_of=AS_OF)


def test_native_availability_uses_latest_verified_stamp_and_retains_precision():
    manifest_time = "2026-08-28T13:29:59.000001Z"
    pointer_time = "2026-08-28T13:29:59.900001Z"
    verified = {
        "manifest": {
            "metadata": {
                "provider_manifest": {
                    "derivation": {
                        "derivation_timestamp": manifest_time,
                    }
                }
            }
        },
        "metadata": {"derivation": {"derivation_timestamp": pointer_time}},
    }
    assert bind_native_availability(AS_OF, verified, as_of=AS_OF) == pointer_time
    with pytest.raises(FundamentalTemporalError, match="NATIVE_AVAILABILITY_BACKDATED"):
        bind_native_availability(manifest_time, verified, as_of=AS_OF)
    with pytest.raises(FundamentalTemporalError, match="NOT_AVAILABLE_AT_DECISION"):
        bind_native_availability("2026-08-28T13:30:00.000001Z", verified, as_of=AS_OF)
    with pytest.raises(FundamentalTemporalError, match="NATIVE_AVAILABILITY_MISSING"):
        bind_native_availability(AS_OF, {"manifest": {"metadata": {}}}, as_of=AS_OF)
