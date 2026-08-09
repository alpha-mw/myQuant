"""Reason-coded PIT daily-history coverage must fail closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market import fundamental_mart
from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    _daily_history_coverage_metrics,
    _listing_identity_sha256,
)
from quant_investor.market.fundamental_provider_contract import canonical_json_sha256


SYMBOL = "000001.SZ"
MEMBERSHIP_SHA = "1" * 64
LISTING_IDENTITY = _listing_identity_sha256(
    symbol=SYMBOL,
    listing_date="20150101",
    effective_from="20150101",
    history_end="20151231",
    membership_sha256=MEMBERSHIP_SHA,
)
AUTHORITY_BY_REASON = {
    "PRE_LISTING": "PIT_LISTING_RECORD",
    "POST_DELISTING": "PIT_LISTING_RECORD",
    "EXCHANGE_SUSPENDED": "DATED_SUSPENSION_EVIDENCE",
    "PROVIDER_COVERAGE_BOUNDARY": "PROVIDER_METADATA_RECEIPT",
    "TRUE_MISSING": "CANONICAL_COVERAGE_AUDIT",
    "UNCONFIRMED": "NONE",
}


def _dense_dates(months: tuple[int, ...]) -> pd.Series:
    return pd.Series(
        pd.to_datetime(
            [f"2015-{month:02d}-{day:02d}" for month in months for day in range(1, 21)]
        )
    )


def _interval(
    reason: str,
    effective_from: str,
    effective_to: str,
    *,
    source_sha256: str = "2" * 64,
    available_at: str = "20151231",
) -> dict[str, str]:
    body = {
        "symbol": SYMBOL,
        "listing_identity": LISTING_IDENTITY,
        "reason": reason,
        "authority": AUTHORITY_BY_REASON[reason],
        "effective_from": effective_from,
        "effective_to": effective_to,
        "available_at": available_at,
        "cutoff": "20151231",
        "source_sha256": (
            MEMBERSHIP_SHA
            if reason in {"PRE_LISTING", "POST_DELISTING"}
            else source_sha256
        ),
    }
    return {"interval_id": canonical_json_sha256(body), **body}


def _metrics(
    dates: pd.Series,
    intervals: tuple[dict[str, str], ...] = (),
) -> dict[str, object]:
    return _daily_history_coverage_metrics(
        dates,
        expected_start="20150101",
        expected_end="20151231",
        allow_tail_gap=False,
        coverage_intervals=intervals,
        symbol=SYMBOL,
        listing_identity=LISTING_IDENTITY,
        listing_start="20150101",
        listing_end="20151231",
        listing_source_sha256=MEMBERSHIP_SHA,
        cutoff="20151231",
    )


def test_bar_gap_alone_never_implies_exchange_suspension() -> None:
    result = _metrics(_dense_dates((1, 2, 3, 10, 11, 12)))

    assert result["coverage_reason_counts"] == {}
    assert result["max_consecutive_missing_months"] == 6
    assert result["history_complete"] is False


def test_exact_suspension_interval_exempts_only_fully_covered_months() -> None:
    result = _metrics(
        _dense_dates((1, 2, 3, 10, 11, 12)),
        (_interval("EXCHANGE_SUSPENDED", "20150401", "20150930"),),
    )

    assert result["coverage_reason_counts"] == {"EXCHANGE_SUSPENDED": 6}
    assert result["coverage_blocker_codes"] == []
    assert result["history_exception_evidence_bound"] is True
    assert result["history_complete"] is True


def test_partial_month_suspension_does_not_exempt_the_month() -> None:
    result = _metrics(
        _dense_dates((1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)),
        (_interval("EXCHANGE_SUSPENDED", "20150410", "20150420"),),
    )

    assert result["coverage_reason_counts"] == {}
    assert result["expected_history_months"] == 12


def test_exact_provider_metadata_interval_can_exempt_a_prefix() -> None:
    result = _metrics(
        _dense_dates((4, 5, 6, 7, 8, 9, 10, 11, 12)),
        (_interval("PROVIDER_COVERAGE_BOUNDARY", "20150101", "20150331"),),
    )

    assert result["coverage_reason_counts"] == {"PROVIDER_COVERAGE_BOUNDARY": 3}
    assert result["history_complete"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("available_at", "20160101", "time ordering"),
        ("effective_to", "20160101", "time ordering"),
        ("listing_identity", "3" * 64, "identity mismatch"),
        ("source_sha256", "not-a-sha", "source SHA"),
    ],
)
def test_future_identity_and_source_mismatches_are_rejected(
    field: str,
    value: str,
    message: str,
) -> None:
    interval = _interval("EXCHANGE_SUSPENDED", "20150401", "20150930")
    interval[field] = value
    interval["interval_id"] = canonical_json_sha256(
        {key: value for key, value in interval.items() if key != "interval_id"}
    )

    with pytest.raises(FundamentalGenerationError, match=message):
        _metrics(_dense_dates((1, 2, 3, 10, 11, 12)), (interval,))


def test_duplicate_interval_identity_is_rejected() -> None:
    interval = _interval("EXCHANGE_SUSPENDED", "20150401", "20150930")

    with pytest.raises(FundamentalGenerationError, match="duplicated"):
        _metrics(_dense_dates((1, 2, 3, 10, 11, 12)), (interval, interval))


@pytest.mark.parametrize("reason", ["TRUE_MISSING", "UNCONFIRMED"])
def test_blocking_reason_never_becomes_an_exemption(reason: str) -> None:
    result = _metrics(
        _dense_dates((1, 2, 3, 10, 11, 12)),
        (_interval(reason, "20150401", "20150930"),),
    )

    assert result["coverage_reason_counts"] == {}
    assert result["coverage_blocker_codes"] == [f"COVERAGE_{reason}"]
    assert result["history_complete"] is False


def test_higher_precedence_suspension_wins_over_provider_boundary() -> None:
    result = _metrics(
        _dense_dates((1, 2, 3, 10, 11, 12)),
        (
            _interval("PROVIDER_COVERAGE_BOUNDARY", "20150401", "20150930"),
            _interval(
                "EXCHANGE_SUSPENDED",
                "20150401",
                "20150930",
                source_sha256="4" * 64,
            ),
        ),
    )

    assert result["coverage_reason_counts"] == {"EXCHANGE_SUSPENDED": 6}
    assert result["history_complete"] is True


def test_equal_precedence_overlap_becomes_unconfirmed() -> None:
    result = _metrics(
        _dense_dates((1, 2, 3, 10, 11, 12)),
        (
            _interval("EXCHANGE_SUSPENDED", "20150401", "20150930"),
            _interval(
                "EXCHANGE_SUSPENDED",
                "20150401",
                "20150930",
                source_sha256="4" * 64,
            ),
        ),
    )

    assert result["coverage_reason_counts"] == {}
    assert result["coverage_blocker_codes"] == ["COVERAGE_UNCONFIRMED"]
    assert result["history_complete"] is False


def test_legacy_provider_floor_is_rejected_for_an_affected_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = tmp_path / "legacy.json"
    declaration.write_text(
        json.dumps(
            {
                "schema_version": "daily-basic-provider-coverage-boundaries.v1",
                "coverage_starts": {SYMBOL: "20150401"},
            }
        )
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DAILY_BASIC_COVERAGE_BOUNDARY_PATH",
        declaration,
    )

    with pytest.raises(ValueError, match="not exact provider authority"):
        fundamental_mart._declared_coverage_intervals(
            {SYMBOL},
            listing_identities={SYMBOL: LISTING_IDENTITY},
            listing_dates={SYMBOL: "20150101"},
            history_end_dates={SYMBOL: "20151231"},
            membership_sha256=MEMBERSHIP_SHA,
            cutoff="20151231",
        )


def test_v2_interval_file_is_content_sealed_and_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _interval("EXCHANGE_SUSPENDED", "20150401", "20150531"),
        _interval(
            "PROVIDER_COVERAGE_BOUNDARY",
            "20150101",
            "20150331",
            source_sha256="4" * 64,
        ),
    ]
    record = {"schema_version": "daily-basic-coverage-intervals.v2", "intervals": rows}
    declaration = tmp_path / "intervals.json"
    declaration.write_text(
        json.dumps({**record, "record_sha256": canonical_json_sha256(record)})
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DAILY_BASIC_COVERAGE_BOUNDARY_PATH",
        declaration,
    )

    result = fundamental_mart._declared_coverage_intervals(
        {SYMBOL},
        listing_identities={SYMBOL: LISTING_IDENTITY},
        listing_dates={SYMBOL: "20150101"},
        history_end_dates={SYMBOL: "20151231"},
        membership_sha256=MEMBERSHIP_SHA,
        cutoff="20151231",
    )

    assert result["daily_history_coverage_interval_source_sha256"] == hashlib.sha256(
        declaration.read_bytes()
    ).hexdigest()
    assert [row["interval_id"] for row in result["daily_history_coverage_intervals"]] == sorted(
        row["interval_id"] for row in rows
    )
