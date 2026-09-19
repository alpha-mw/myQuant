"""A defective row must cost the run that row, not the whole response.

The case these tests were written from is real. On 2026-09-01 Tushare returned
nineteen ``fina_indicator`` rows for 603400.SH, one of which carried the 2026
半年报 dated ``ann_date=20260422`` — two months before the 20260630 period it
reports on had closed. A correctly dated duplicate of that same filing
(``ann_date=20260803``) was in the same payload with identical values, so the
bad row held nothing the panel needed. The old classification voided all
nineteen rows, marked the request malformed, and through the promotion gate's
``requests_malformed != 0`` check blocked a 5,556-symbol rebuild.

The bad row is excluded either way. What these tests pin down is the blast
radius, and the accounting that keeps the exclusion visible.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.market.fundamental_mart import _strict_pit_cutoff
from quant_investor.market.fundamental_provider_contract import (
    FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
    HARD_INVALID_SUBCOUNTER_FIELDS,
    FundamentalEndpointAuditPolicy,
    validate_outcome_accounting_v3,
)

AS_OF = "20260904"
CLEAN_COUNTERS = (
    "rows",
    "rows_filtered_future",
    "rows_filtered_missing_availability",
    "rows_filtered_core_values",
    "rows_deduplicated",
)


def _fina_indicator(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a fina_indicator payload from (ann_date, end_date, roe) triples."""
    return pd.DataFrame(
        [
            {
                "ts_code": "603400.SH",
                "ann_date": ann_date,
                "end_date": end_date,
                "roe_dt": roe,
                "roe": roe,
                "roa": roe / 2,
                "debt_to_assets": 40.0,
                "netprofit_yoy": 1.0,
            }
            for ann_date, end_date, roe in rows
        ]
    )


def _assert_accounting_reconciles(stats: dict[str, int]) -> None:
    accounted = sum(int(stats[field]) for field in CLEAN_COUNTERS)
    accounted += int(stats["rows_hard_invalid"])
    accounted += int(stats["rows_discarded_request_malformed"])
    assert accounted == int(stats["rows_received"])
    assert sum(int(stats[f]) for f in HARD_INVALID_SUBCOUNTER_FIELDS) == int(
        stats["rows_hard_invalid"]
    )


def test_one_misdated_row_does_not_void_the_other_rows() -> None:
    frame = _fina_indicator(
        [
            ("20250530", "20241231", 23.0),
            ("20250818", "20250630", 8.3),
            ("20251028", "20250930", 11.0),
            ("20260803", "20260630", 3.6869),
            # The defect: announced before the period it reports on closed.
            ("20260422", "20260630", 3.6869),
        ]
    )

    accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert reason == ""
    assert stats["rows_hard_invalid"] == 1
    assert stats["rows_hard_invalid_end_after_availability"] == 1
    assert stats["rows_discarded_request_malformed"] == 0
    assert stats["rows"] == 4
    assert "20260422" not in set(accepted["ann_date"])
    _assert_accounting_reconciles(stats)


def test_rejected_row_is_counted_under_its_own_defect() -> None:
    frame = _fina_indicator(
        [
            ("20250530", "20241231", 23.0),
            ("not-a-date", "20250630", 8.3),
        ]
    )

    _accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert reason == ""
    assert stats["rows_hard_invalid_availability_date"] == 1
    assert stats["rows_hard_invalid_end_after_availability"] == 0
    _assert_accounting_reconciles(stats)


def test_response_level_defects_still_void_the_whole_request() -> None:
    frame = _fina_indicator([("20250530", "20241231", 23.0)])
    frame["ts_code"] = "600000.SH"  # not the symbol that was asked for

    accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert reason == "response_symbol_scope_mismatch"
    assert accepted.empty
    assert stats["rows_hard_invalid_symbol"] == 1
    _assert_accounting_reconciles(stats)


def test_missing_required_column_still_voids_the_whole_request() -> None:
    frame = _fina_indicator([("20250530", "20241231", 23.0)]).drop(columns=["roe"])

    _accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert reason.startswith("missing_required_columns:")
    assert stats["rows_hard_invalid_schema"] == 1
    _assert_accounting_reconciles(stats)


def test_a_payload_of_only_bad_rows_is_empty_not_malformed() -> None:
    frame = _fina_indicator([("20260422", "20260630", 3.6869)])

    accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert reason == ""
    assert accepted.empty
    assert stats["rows"] == 0
    assert stats["rows_hard_invalid"] == 1
    _assert_accounting_reconciles(stats)


def test_accepted_rows_are_a_fixed_point_of_a_second_pass() -> None:
    """The generation replay re-runs the cutoff over stored rows and requires
    that nothing more is rejected. Row-level rejection must not break that."""
    frame = _fina_indicator(
        [
            ("20250530", "20241231", 23.0),
            ("20260803", "20260630", 3.6869),
            ("20260422", "20260630", 3.6869),
        ]
    )

    accepted, _stats, _reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )
    replayed, replay_stats, replay_reason = _strict_pit_cutoff(
        accepted, table="fina_indicator", symbol="603400.SH", as_of=AS_OF
    )

    assert replay_reason == ""
    assert replay_stats["rows_hard_invalid"] == 0
    assert replay_stats["rows"] == len(accepted)
    pd.testing.assert_frame_equal(replayed, accepted.reset_index(drop=True))


def _outcome(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
        "status": "success",
        "rows_received": 5,
        "rows": 4,
        "rows_hard_invalid": 1,
        "rows_filtered_future": 0,
        "rows_filtered_missing_availability": 0,
        "rows_filtered_core_values": 0,
        "rows_deduplicated": 0,
        "rows_discarded_request_malformed": 0,
        **{field: 0 for field in HARD_INVALID_SUBCOUNTER_FIELDS},
        "rows_hard_invalid_end_after_availability": 1,
    }
    payload.update(overrides)
    return payload


def test_clean_outcome_may_carry_counted_row_rejections() -> None:
    counters = validate_outcome_accounting_v3(_outcome())

    assert counters["rows_hard_invalid"] == 1
    assert counters["rows"] == 4


def test_clean_outcome_may_not_discard_rows_as_request_malformed() -> None:
    payload = _outcome(rows_received=6, rows_discarded_request_malformed=1)

    with pytest.raises(ValueError, match="discarded rows as malformed"):
        validate_outcome_accounting_v3(payload)


def test_clean_outcome_rejects_unaccounted_row_rejections() -> None:
    # rows_received still counts only the four accepted rows, so the rejected
    # row would vanish from the ledger.
    payload = _outcome(rows_received=4)

    with pytest.raises(ValueError, match="does not reconcile"):
        validate_outcome_accounting_v3(payload)


def test_malformed_requests_remain_fail_closed() -> None:
    policy = FundamentalEndpointAuditPolicy()

    assert policy.max_malformed_requests == 0
    assert policy.max_error_requests == 0
