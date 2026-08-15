"""Strict native-body decoders for official CN exchange calendar evidence.

The decoder never infers weekdays, holidays, or session hours.  Each accepted
body must state every daily OPEN/CLOSED value or every continuous-session
interval explicitly in the exchange-specific wire shape.  Unsupported issuer
responses remain a production blocker until this stable module is updated and
released with captured native fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Literal

from quant_investor.system.errors import SystemContractError

EvidenceRole = Literal["DAILY_STATUS", "SESSION_RULE"]

DECODER_IDS: Final[dict[tuple[str, str], str]] = {
    ("SSE", "DAILY_STATUS"): "myquant.exchange-calendar.sse-native-jsonp.v1",
    ("SSE", "SESSION_RULE"): "myquant.exchange-calendar.sse-native-rules-json.v1",
    ("SZSE", "DAILY_STATUS"): "myquant.exchange-calendar.szse-native-json.v1",
    ("SZSE", "SESSION_RULE"): "myquant.exchange-calendar.szse-native-rules-json.v1",
    ("BSE", "DAILY_STATUS"): "myquant.exchange-calendar.bse-native-json.v1",
    ("BSE", "SESSION_RULE"): "myquant.exchange-calendar.bse-native-rules-json.v1",
}

_DATE_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_TIME_RE: Final = re.compile(r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
_SSE_JSONP_RE: Final = re.compile(rb"^sseCalendarCallback\((\{.*\})\);\n?$", re.DOTALL)


def decoder_code_sha256() -> str:
    """Return the exact installed decoder module byte identity."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemContractError(f"{label} is not strict native JSON") from exc
    if json.dumps(value, ensure_ascii=False, separators=(",", ":")) != text.rstrip("\n"):
        raise SystemContractError(f"{label} JSON bytes are not canonical issuer bytes")
    return value


def _daily_row(date_value: Any, open_value: Any, *, label: str) -> dict[str, str]:
    if type(date_value) is not str or _DATE_RE.fullmatch(date_value) is None:
        raise SystemContractError(f"{label} date is invalid")
    if type(open_value) is not bool:
        raise SystemContractError(f"{label} status is not explicit")
    return {"date": date_value, "status": "OPEN" if open_value else "CLOSED"}


def decode_daily_status(  # noqa: C901
    exchange: str, raw: bytes, *, media_type: str
) -> list[dict[str, str]]:
    """Decode one exact native daily-status response without date inference."""

    rows: list[dict[str, str]] = []
    if exchange == "SSE":
        if media_type != "application/javascript":
            raise SystemContractError("SSE daily calendar media type differs")
        matched = _SSE_JSONP_RE.fullmatch(raw)
        if matched is None:
            raise SystemContractError("SSE daily calendar JSONP framing differs")
        value = _json(matched.group(1), label="SSE daily calendar payload")
        if type(value) is not dict or set(value) != {"result"} or type(value["result"]) is not list:
            raise SystemContractError("SSE daily calendar native fields differ")
        for index, item in enumerate(value["result"]):
            if type(item) is not dict or set(item) != {"tradeDate", "isOpen"}:
                raise SystemContractError("SSE daily calendar row fields differ")
            rows.append(_daily_row(item["tradeDate"], item["isOpen"], label=f"SSE[{index}]"))
    elif exchange == "SZSE":
        if media_type != "application/json":
            raise SystemContractError("SZSE daily calendar media type differs")
        value = _json(raw, label="SZSE daily calendar response")
        if (
            type(value) is not dict
            or set(value) != {"code", "data"}
            or value["code"] != "0"
            or type(value["data"]) is not list
        ):
            raise SystemContractError("SZSE daily calendar native fields differ")
        for index, item in enumerate(value["data"]):
            if type(item) is not dict or set(item) != {"tradeDate", "tradeFlag"}:
                raise SystemContractError("SZSE daily calendar row fields differ")
            flag = item["tradeFlag"]
            if flag not in {"0", "1"}:
                raise SystemContractError("SZSE daily calendar status is not explicit")
            date_value = item["tradeDate"]
            if type(date_value) is not str or re.fullmatch(r"[0-9]{8}", date_value) is None:
                raise SystemContractError("SZSE daily calendar date is invalid")
            dashed = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:]}"
            rows.append(_daily_row(dashed, flag == "1", label=f"SZSE[{index}]"))
    elif exchange == "BSE":
        if media_type != "application/json":
            raise SystemContractError("BSE daily calendar media type differs")
        value = _json(raw, label="BSE daily calendar response")
        if (
            type(value) is not dict
            or set(value) != {"result", "success"}
            or value["success"] is not True
            or type(value["result"]) is not list
        ):
            raise SystemContractError("BSE daily calendar native fields differ")
        for index, item in enumerate(value["result"]):
            if type(item) is not dict or set(item) != {"market_status", "trade_date"}:
                raise SystemContractError("BSE daily calendar row fields differ")
            status = item["market_status"]
            if status not in {"TRADING", "CLOSED"}:
                raise SystemContractError("BSE daily calendar status is not explicit")
            rows.append(_daily_row(item["trade_date"], status == "TRADING", label=f"BSE[{index}]"))
    else:
        raise SystemContractError("official exchange daily decoder is unsupported")
    if not rows:
        raise SystemContractError("official daily calendar response is empty")
    return rows


def _interval(start: Any, end: Any, *, label: str) -> dict[str, str]:
    if (
        type(start) is not str
        or type(end) is not str
        or _TIME_RE.fullmatch(start) is None
        or _TIME_RE.fullmatch(end) is None
        or start >= end
    ):
        raise SystemContractError(f"{label} interval is invalid")
    return {"opens_local": start, "closes_local": end}


def decode_session_intervals(  # noqa: C901
    exchange: str, raw: bytes, *, media_type: str
) -> list[dict[str, str]]:
    """Decode explicitly published continuous-session rule bytes."""

    if media_type != "application/json":
        raise SystemContractError("official session rule media type differs")
    value = _json(raw, label=f"{exchange} session rule response")
    intervals: list[dict[str, str]] = []
    if exchange == "SSE":
        if (
            type(value) is not dict
            or set(value) != {"continuousAuction", "market"}
            or value["market"] != "SSE"
            or type(value["continuousAuction"]) is not list
        ):
            raise SystemContractError("SSE session rule native fields differ")
        for index, item in enumerate(value["continuousAuction"]):
            if type(item) is not dict or set(item) != {"beginTime", "endTime"}:
                raise SystemContractError("SSE session interval fields differ")
            intervals.append(
                _interval(item["beginTime"], item["endTime"], label=f"SSE session[{index}]")
            )
    elif exchange == "SZSE":
        if (
            type(value) is not dict
            or set(value) != {"exchange", "sessions"}
            or value["exchange"] != "SZSE"
            or type(value["sessions"]) is not list
        ):
            raise SystemContractError("SZSE session rule native fields differ")
        for index, item in enumerate(value["sessions"]):
            if (
                type(item) is not dict
                or set(item) != {"end", "start", "type"}
                or item["type"] != "CONTINUOUS"
            ):
                raise SystemContractError("SZSE session interval fields differ")
            intervals.append(_interval(item["start"], item["end"], label=f"SZSE[{index}]"))
    elif exchange == "BSE":
        if (
            type(value) is not dict
            or set(value) != {"code", "data"}
            or value["code"] != 0
            or type(value["data"]) is not dict
            or set(value["data"]) != {"continuousTradingPeriods", "marketCode"}
            or value["data"]["marketCode"] != "BSE"
            or type(value["data"]["continuousTradingPeriods"]) is not list
        ):
            raise SystemContractError("BSE session rule native fields differ")
        for index, item in enumerate(value["data"]["continuousTradingPeriods"]):
            if type(item) is not str or item.count("/") != 1:
                raise SystemContractError("BSE session interval fields differ")
            start, end = item.split("/")
            intervals.append(_interval(start, end, label=f"BSE session[{index}]"))
    else:
        raise SystemContractError("official exchange session decoder is unsupported")
    if not intervals:
        raise SystemContractError("official exchange session rules are empty")
    return intervals


def decoder_id(exchange: str, role: EvidenceRole) -> str:
    try:
        return DECODER_IDS[(exchange, role)]
    except KeyError as exc:
        raise SystemContractError("official exchange decoder identity is unsupported") from exc


def decode_capture_projection(
    exchange: str,
    role: EvidenceRole,
    raw: bytes,
    *,
    media_type: str,
) -> Mapping[str, object]:
    if role == "DAILY_STATUS":
        return {"daily_status_rows": decode_daily_status(exchange, raw, media_type=media_type)}
    if role == "SESSION_RULE":
        return {"session_intervals": decode_session_intervals(exchange, raw, media_type=media_type)}
    raise SystemContractError("official calendar evidence role is unsupported")


__all__ = [
    "DECODER_IDS",
    "decode_capture_projection",
    "decode_daily_status",
    "decode_session_intervals",
    "decoder_code_sha256",
    "decoder_id",
]
