from __future__ import annotations

import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.exchange_calendar_official import (
    decode_daily_status,
    decode_session_intervals,
    decoder_code_sha256,
    decoder_id,
)
from quant_investor.system import SystemContractError

ROWS = [
    {"date": "2024-01-01", "status": "CLOSED"},
    {"date": "2024-01-02", "status": "OPEN"},
]
INTERVALS = [
    {"opens_local": "09:30:00", "closes_local": "11:30:00"},
    {"opens_local": "13:00:00", "closes_local": "15:00:00"},
]


def _daily(exchange: str) -> tuple[bytes, str]:
    if exchange == "SSE":
        payload = {
            "result": [
                {"tradeDate": row["date"], "isOpen": row["status"] == "OPEN"} for row in ROWS
            ]
        }
        return b"sseCalendarCallback(" + canonical_json_bytes(payload) + b");", (
            "application/javascript"
        )
    if exchange == "SZSE":
        return (
            canonical_json_bytes(
                {
                    "code": "0",
                    "data": [
                        {
                            "tradeDate": row["date"].replace("-", ""),
                            "tradeFlag": "1" if row["status"] == "OPEN" else "0",
                        }
                        for row in ROWS
                    ],
                }
            ),
            "application/json",
        )
    return (
        canonical_json_bytes(
            {
                "result": [
                    {
                        "market_status": ("TRADING" if row["status"] == "OPEN" else "CLOSED"),
                        "trade_date": row["date"],
                    }
                    for row in ROWS
                ],
                "success": True,
            }
        ),
        "application/json",
    )


def _rules(exchange: str) -> bytes:
    if exchange == "SSE":
        value = {
            "continuousAuction": [
                {"beginTime": row["opens_local"], "endTime": row["closes_local"]}
                for row in INTERVALS
            ],
            "market": "SSE",
        }
    elif exchange == "SZSE":
        value = {
            "exchange": "SZSE",
            "sessions": [
                {
                    "end": row["closes_local"],
                    "start": row["opens_local"],
                    "type": "CONTINUOUS",
                }
                for row in INTERVALS
            ],
        }
    else:
        value = {
            "code": 0,
            "data": {
                "continuousTradingPeriods": [
                    f"{row['opens_local']}/{row['closes_local']}" for row in INTERVALS
                ],
                "marketCode": "BSE",
            },
        }
    return canonical_json_bytes(value)


@pytest.mark.parametrize("exchange", ["SSE", "SZSE", "BSE"])
def test_exchange_specific_native_decoders_replay_exact_daily_and_rule_bytes(
    exchange: str,
) -> None:
    raw, media_type = _daily(exchange)
    assert decode_daily_status(exchange, raw, media_type=media_type) == ROWS
    assert (
        decode_session_intervals(exchange, _rules(exchange), media_type="application/json")
        == INTERVALS
    )
    assert decoder_id(exchange, "DAILY_STATUS") != decoder_id(exchange, "SESSION_RULE")
    assert len(decoder_code_sha256()) == 64


@pytest.mark.parametrize("exchange", ["SSE", "SZSE", "BSE"])
def test_native_calendar_tamper_and_noncanonical_bytes_fail_closed(exchange: str) -> None:
    raw, media_type = _daily(exchange)
    with pytest.raises(SystemContractError):
        decode_daily_status(exchange, raw + b" ", media_type=media_type)

    rules = json.loads(_rules(exchange))
    if exchange == "SSE":
        rules["continuousAuction"][0]["beginTime"] = "weekday-default"
    elif exchange == "SZSE":
        rules["sessions"][0]["type"] = "INFERRED"
    else:
        rules["data"]["continuousTradingPeriods"][0] = "weekday-default"
    with pytest.raises(SystemContractError):
        decode_session_intervals(
            exchange, canonical_json_bytes(rules), media_type="application/json"
        )
