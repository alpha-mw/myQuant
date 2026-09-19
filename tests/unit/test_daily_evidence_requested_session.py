"""Requested OPEN/CLOSED decisions replay real close-authority wire fixtures."""

from datetime import datetime, timedelta
import json
import pytest
from quant_investor.market.close_session_authority import (
    acquire_close_session_authority,
    CloseSessionAuthorityError,
)
from quant_investor.market.requested_session import classify_requested_session
from quant_investor.market.tushare_transport import replay_tushare_response_bytes


def capture(now, *, holidays=(), omit=None):
    class Client:
        def request(self, *, params, expected_fields, **kwargs):
            day = datetime.strptime(params["start_date"], "%Y%m%d")
            end = datetime.strptime(params["end_date"], "%Y%m%d")
            previous = day - timedelta(days=1)
            while previous.weekday() >= 5:
                previous -= timedelta(days=1)
            items = []
            while day <= end:
                key = day.strftime("%Y%m%d")
                opened = day.weekday() < 5 and key not in holidays
                if key != omit:
                    items.append(["SSE", key, int(opened), previous.strftime("%Y%m%d")])
                if opened:
                    previous = day
                day += timedelta(days=1)
            raw = json.dumps(
                {
                    "code": 0,
                    "msg": "",
                    "detail": "",
                    "request_id": "synthetic-session",
                    "data": {
                        "fields": list(expected_fields),
                        "items": items,
                        "has_more": False,
                        "count": 0,
                    },
                }
            ).encode()
            return replay_tushare_response_bytes(
                raw, api_name="trade_cal", expected_fields=expected_fields
            )

    return acquire_close_session_authority(now=datetime.fromisoformat(now), client=Client())


@pytest.mark.parametrize(
    "now,holidays,expected",
    [
        ("2026-09-04T20:20:00+08:00", (), "MATCHED_OPEN"),
        ("2026-09-05T20:20:00+08:00", (), "CONFIRMED_CLOSED"),
        ("2026-09-08T20:20:00+08:00", ("20260908",), "CONFIRMED_CLOSED"),
    ],
)
def test_exact_native_calendar_classification(now, holidays, expected):
    value = capture(now, holidays=holidays)
    day = now[:10].replace("-", "")
    result = classify_requested_session(
        requested_trade_date=day, receipt=value.receipt, raw=value.raw_response_bytes
    )
    assert result["classification"] == expected


@pytest.mark.parametrize("fault", ["before_close", "wrong_date", "raw", "receipt", "missing"])
def test_invalid_or_unclosed_open_session_cannot_be_holiday(fault):
    if fault == "missing":
        with pytest.raises(CloseSessionAuthorityError):
            capture("2026-09-04T20:20:00+08:00", omit="20260904")
        return
    value = capture(
        "2026-09-04T10:00:00+08:00" if fault == "before_close" else "2026-09-04T20:20:00+08:00"
    )
    raw = value.raw_response_bytes
    day = "20260905" if fault == "wrong_date" else "20260904"
    if fault == "raw":
        raw += b" "
    elif fault == "receipt":
        value.receipt["target_trade_date"] = "20260903"
    with pytest.raises(CloseSessionAuthorityError):
        classify_requested_session(requested_trade_date=day, receipt=value.receipt, raw=raw)
