"""An identity with no trades is admissible only against complete evidence.

Every test here is a refusal except the first. The value of this contract is
entirely in what it declines to accept: a missing day, a renamed identity, a
shifted date, or an edited payload must each stop the document from verifying,
because each of them would otherwise let "we have no data" pass as "there was
legitimately nothing to have".
"""
from __future__ import annotations

import json

import pytest

from quant_investor.market.cn_research_suspension_evidence import (
    SuspensionEvidenceError,
    build_suspension_evidence,
    canonical_payload_sha256,
    expected_trading_days,
    verify_suspension_evidence,
)

SYMBOL = "600068.SH"
CALENDAR = ["20210906", "20210907", "20210908", "20210909", "20210910", "20210913"]


def _window(effective_to: str = "20210913"):
    return expected_trading_days(
        symbol=SYMBOL,
        effective_from="19970526",
        effective_to=effective_to,
        window_start="20210904",
        window_end="20260904",
        calendar_open_dates=CALENDAR,
    )


def _payloads(dates, *, symbol: str = SYMBOL, suspend_type: str = "S") -> dict:
    return {
        date: {
            "query_params": {"trade_date": date},
            "rows": [
                {"ts_code": symbol, "trade_date": date, "suspend_type": suspend_type}
            ],
        }
        for date in dates
    }


def _document(**kwargs):
    window = kwargs.pop("window", None) or _window()
    payloads = kwargs.pop("payloads", None) or _payloads(window.trade_dates)
    return (
        build_suspension_evidence(
            window=window,
            per_date_payloads=payloads,
            calendar_reference={"api": "tushare.trade_cal", "exchange": "SSE"},
            membership_reference={"path": "membership.parquet", "sha256": "0" * 64},
        ),
        window,
        payloads,
    )


def test_expected_days_come_from_calendar_and_membership_not_from_bars() -> None:
    window = _window()

    assert window.window_start == "20210904"
    # effective_to is inclusive for membership, so the window ends there...
    assert window.window_end == "20210913"
    # ...but the removal date is not an expected *trading* day: the identity is
    # being removed that day, not suspended, so no suspension record exists and
    # expecting a trade would create a gap no evidence could ever close.
    assert window.removal_date_excluded == "20210913"
    assert window.trade_dates == tuple(CALENDAR[:-1])


def test_removal_date_is_excluded_only_when_inside_the_window() -> None:
    window = expected_trading_days(
        symbol=SYMBOL,
        effective_from="19970526",
        effective_to="",  # still listed: no removal date at all
        window_start="20210904",
        window_end="20260904",
        calendar_open_dates=CALENDAR,
    )

    assert window.removal_date_excluded == ""
    assert window.trade_dates == tuple(CALENDAR)


def test_complete_evidence_verifies_and_grants_no_coverage() -> None:
    document, window, payloads = _document()

    result = verify_suspension_evidence(
        document,
        symbol=SYMBOL,
        expected_trade_dates=window.trade_dates,
        per_date_payloads=payloads,
    )

    assert result["verified_day_count"] == len(CALENDAR) - 1
    assert result["grants_price_coverage"] is False
    assert result["grants_financial_coverage"] is False


def test_a_single_missing_day_is_refused() -> None:
    window = _window()
    partial = _payloads(window.trade_dates[:-1])
    document, _window_obj, _payloads_obj = _document(window=window, payloads=partial)

    assert document["unexplained_dates"] == [window.trade_dates[-1]]
    with pytest.raises(SuspensionEvidenceError, match="unexplained dates"):
        verify_suspension_evidence(
            document, symbol=SYMBOL, expected_trade_dates=window.trade_dates
        )


def test_evidence_for_another_identity_is_refused() -> None:
    window = _window()
    document, _w, _p = _document(
        window=window, payloads=_payloads(window.trade_dates, symbol="600000.SH")
    )

    # Rows naming a different identity never match, so every day is unexplained.
    assert document["unexplained_dates"] == list(window.trade_dates)
    with pytest.raises(SuspensionEvidenceError):
        verify_suspension_evidence(
            document, symbol=SYMBOL, expected_trade_dates=window.trade_dates
        )


def test_verifying_against_a_different_symbol_is_refused() -> None:
    document, window, _p = _document()

    with pytest.raises(SuspensionEvidenceError, match="different identity"):
        verify_suspension_evidence(
            document, symbol="600000.SH", expected_trade_dates=window.trade_dates
        )


def test_wrong_expected_day_set_is_refused() -> None:
    document, window, _p = _document()
    shifted = list(window.trade_dates[:-1]) + ["20210914"]

    with pytest.raises(SuspensionEvidenceError, match="day set mismatch"):
        verify_suspension_evidence(
            document, symbol=SYMBOL, expected_trade_dates=shifted
        )


def test_a_row_that_is_not_a_suspension_is_refused() -> None:
    window = _window()
    document, _w, _p = _document(
        window=window, payloads=_payloads(window.trade_dates, suspend_type="R")
    )

    assert document["unexplained_dates"] == list(window.trade_dates)


def test_tampered_document_is_refused() -> None:
    document, window, payloads = _document()
    document["per_date"][0]["trade_date"] = "20210914"

    with pytest.raises(SuspensionEvidenceError, match="record_sha256 does not verify"):
        verify_suspension_evidence(
            document,
            symbol=SYMBOL,
            expected_trade_dates=window.trade_dates,
            per_date_payloads=payloads,
        )


def test_tampered_raw_payload_is_refused() -> None:
    document, window, payloads = _document()
    altered = json.loads(json.dumps(payloads))
    altered[window.trade_dates[0]]["rows"][0]["suspend_type"] = "R"

    with pytest.raises(SuspensionEvidenceError, match="payload SHA mismatch"):
        verify_suspension_evidence(
            document,
            symbol=SYMBOL,
            expected_trade_dates=window.trade_dates,
            per_date_payloads=altered,
        )


def test_reversed_window_is_refused() -> None:
    with pytest.raises(SuspensionEvidenceError, match="reversed"):
        expected_trading_days(
            symbol=SYMBOL,
            effective_from="20260101",
            effective_to="20210913",
            window_start="20210904",
            window_end="20260904",
            calendar_open_dates=CALENDAR,
        )


def test_payload_hash_is_stable_across_key_order() -> None:
    left = {"query_params": {"trade_date": "20210906"}, "rows": [{"a": 1, "b": 2}]}
    right = {"rows": [{"b": 2, "a": 1}], "query_params": {"trade_date": "20210906"}}

    assert canonical_payload_sha256(left) == canonical_payload_sha256(right)
