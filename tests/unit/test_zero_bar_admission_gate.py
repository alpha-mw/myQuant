"""Admitting an identity with no bars is an evidence decision, not a skip.

`_canonical_bar_history_bounds` refuses an identity with no bars in its
eligibility window. That refusal is correct: without evidence, "no bars" and "we
failed to fetch the bars" look identical. These tests hold the exemption to the
only thing that makes it safe — a document that verifies, for this identity,
over exactly the window the caller computed independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.cn_research_suspension_evidence import (
    build_suspension_evidence,
    expected_trading_days,
)
from quant_investor.market.fundamental_mart import _verify_zero_bar_admission

SYMBOL = "600068.SH"
CALENDAR = ["20210906", "20210907", "20210908", "20210909", "20210910", "20210913"]


def _evidence(symbol: str = SYMBOL, effective_to: str = "20210913") -> dict:
    window = expected_trading_days(
        symbol=symbol,
        effective_from="19970526",
        effective_to=effective_to,
        window_start="20210904",
        window_end="20260904",
        calendar_open_dates=CALENDAR,
    )
    payloads = {
        date: {
            "query_params": {"trade_date": date},
            "rows": [{"ts_code": symbol, "trade_date": date, "suspend_type": "S"}],
        }
        for date in window.trade_dates
    }
    return build_suspension_evidence(
        window=window,
        per_date_payloads=payloads,
        calendar_reference={"api": "tushare.trade_cal"},
        membership_reference={"path": "m.parquet", "sha256": "0" * 64},
    )


def _bounds(start: str = "20210904", end: str = "20210913"):
    return pd.Timestamp(start), pd.Timestamp(end)


def test_verified_evidence_admits_the_identity_without_granting_coverage() -> None:
    start, end = _bounds()

    result = _verify_zero_bar_admission(
        symbol=SYMBOL,
        admission=_evidence(),
        eligibility_start=start,
        eligibility_end=end,
    )

    assert result["verified_day_count"] == 5
    assert result["grants_price_coverage"] is False
    assert result["grants_financial_coverage"] is False


def test_absent_evidence_is_not_an_admission(tmp_path: Path) -> None:
    start, end = _bounds()

    with pytest.raises(ValueError, match="unreadable"):
        _verify_zero_bar_admission(
            symbol=SYMBOL,
            admission=None,
            eligibility_start=start,
            eligibility_end=end,
        )


def test_window_must_match_the_independently_computed_eligibility() -> None:
    start, _end = _bounds()
    wrong_end = pd.Timestamp("20210914")

    with pytest.raises(ValueError, match="does not match canonical eligibility"):
        _verify_zero_bar_admission(
            symbol=SYMBOL,
            admission=_evidence(),
            eligibility_start=start,
            eligibility_end=wrong_end,
        )


def test_evidence_for_another_identity_is_refused() -> None:
    start, end = _bounds()

    with pytest.raises(ValueError, match="rejected"):
        _verify_zero_bar_admission(
            symbol="600000.SH",
            admission=_evidence(),
            eligibility_start=start,
            eligibility_end=end,
        )


def test_tampered_evidence_is_refused() -> None:
    start, end = _bounds()
    document = _evidence()
    document["per_date"][0]["suspend_type"] = "R"

    with pytest.raises(ValueError, match="rejected"):
        _verify_zero_bar_admission(
            symbol=SYMBOL,
            admission=document,
            eligibility_start=start,
            eligibility_end=end,
        )


def test_evidence_with_an_unexplained_day_is_refused() -> None:
    start, end = _bounds()
    document = _evidence()
    document["unexplained_dates"] = ["20210907"]

    with pytest.raises(ValueError, match="rejected"):
        _verify_zero_bar_admission(
            symbol=SYMBOL,
            admission=document,
            eligibility_start=start,
            eligibility_end=end,
        )


def test_admission_may_be_supplied_as_a_path(tmp_path: Path) -> None:
    start, end = _bounds()
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")

    result = _verify_zero_bar_admission(
        symbol=SYMBOL,
        admission=str(path),
        eligibility_start=start,
        eligibility_end=end,
    )

    # The path is recorded so a replay re-verifies the document itself rather
    # than trusting this summary.
    assert result["evidence_path"] == str(path)


def test_evidence_claiming_coverage_is_refused() -> None:
    start, end = _bounds()
    document = _evidence()
    document["grants_price_coverage"] = True

    with pytest.raises(ValueError, match="rejected|must not claim"):
        _verify_zero_bar_admission(
            symbol=SYMBOL,
            admission=document,
            eligibility_start=start,
            eligibility_end=end,
        )
