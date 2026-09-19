"""Both request-level thresholds are fail-closed, and why that is affordable.

An errored request may have lost data. A malformed request delivered a payload
that was not what was asked for — the wrong symbol, missing required columns,
not a frame at all. Neither is survivable, so both stay at zero.

This file previously argued the opposite for malformed requests, because a
single mis-stamped row in 603400.SH's fina_indicator had ended a 103-minute
rebuild and a tolerance of five looked like the way out. It was not: the defect
was one row out of nineteen, and the fix belonged at the row, not at the
threshold. ``_strict_pit_cutoff`` now rejects such a row on its own and leaves
the request clean, so nothing incidental reaches this counter any more — and a
tolerance here would have forgiven genuine response-level corruption too.

Scenarios:
  T01  both thresholds are fail-closed
  T02  one malformed or errored request blocks
  T03  a row-level defect never reaches these counters
  T04  the thresholds cannot be loosened by passing a bad value
"""

import pandas as pd
import pytest

from quant_investor.market.fundamental_mart import _strict_pit_cutoff
from quant_investor.market.fundamental_provider_contract import (
    FundamentalEndpointAuditPolicy,
)


def test_T01_both_request_thresholds_are_fail_closed():
    policy = FundamentalEndpointAuditPolicy()
    assert policy.max_error_requests == 0, "a lost request must fail closed"
    assert policy.max_malformed_requests == 0, "an unusable response must fail closed"


@pytest.mark.parametrize("count", [1, 5, 50])
def test_T02_any_malformed_or_errored_request_blocks(count: int):
    policy = FundamentalEndpointAuditPolicy()
    assert count > policy.max_malformed_requests
    assert count > policy.max_error_requests


def test_T03_a_row_level_defect_does_not_reach_the_malformed_counter():
    """The 603400.SH case: the request stays clean, so the counter stays zero."""
    frame = pd.DataFrame(
        [
            {
                "ts_code": "603400.SH",
                "ann_date": ann_date,
                "end_date": "20260630",
                "roe_dt": 3.6869,
                "roe": 3.6869,
                "roa": 2.3606,
                "debt_to_assets": 41.3721,
                "netprofit_yoy": -46.7808,
            }
            # The first row is announced before its own period closed.
            for ann_date in ("20260422", "20260803")
        ]
    )

    accepted, stats, reason = _strict_pit_cutoff(
        frame, table="fina_indicator", symbol="603400.SH", as_of="20260904"
    )

    assert reason == "", "a row defect must not mark the request malformed"
    assert stats["rows_hard_invalid"] == 1
    assert len(accepted) == 1


def test_T04_thresholds_reject_non_integer_and_negative_values():
    for value in (True, -1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            FundamentalEndpointAuditPolicy(max_malformed_requests=value)
