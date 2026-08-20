"""Fresh Tushare SSE authority for one post-close CN maintenance attempt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from typing import Any, Final, Mapping, Protocol
from zoneinfo import ZoneInfo

from .tushare_transport import OFFICIAL_TUSHARE_URL, OfficialTushareHttpsClient

TIMEZONE: Final = "Asia/Shanghai"
SSE: Final = "SSE"
API_NAME: Final = "trade_cal"
EXPECTED_FIELDS: Final = ("exchange", "cal_date", "is_open", "pretrade_date")
LOOKBACK_DAYS: Final = 31
SESSION_CLOSE_LOCAL: Final = time(15, 0)


class CloseSessionAuthorityError(RuntimeError):
    """One controlled close-authority failure code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _TushareClient(Protocol):
    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: tuple[str, ...],
    ) -> Any: ...


@dataclass(frozen=True)
class CloseSessionAuthorityResult:
    """Receipt projection plus exact provider bytes for caller-owned sealing."""

    receipt: dict[str, Any]
    raw_response_bytes: bytes


def _utc_seconds(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local_seconds(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S%z")


def _parse_cal_date(value: Any) -> date:
    if type(value) is not str:
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID")
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID")
    return parsed


def _validated_rows(
    response: Any, *, start_date: date, end_date: date
) -> list[tuple[date, bool, str]]:
    if (
        response.api_name != API_NAME
        or tuple(response.fields) != EXPECTED_FIELDS
        or response.has_more is not False
        or response.item_count != len(response.rows)
        or response.reported_count != response.item_count
        or response.provider_reported_count not in {0, response.item_count}
    ):
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_ENVELOPE_INVALID")
    result: list[tuple[date, bool, str]] = []
    observed: set[date] = set()
    for raw_row in response.rows:
        if type(raw_row) is not tuple or len(raw_row) != len(EXPECTED_FIELDS):
            raise CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID")
        exchange, raw_date, raw_is_open, pretrade_date = raw_row
        cal_date = _parse_cal_date(raw_date)
        if (
            exchange != SSE
            or type(raw_is_open) is not int
            or raw_is_open not in {0, 1}
            or type(pretrade_date) is not str
            or (pretrade_date and len(pretrade_date) != 8)
            or cal_date in observed
        ):
            raise CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID")
        if not pretrade_date:
            raise CloseSessionAuthorityError("CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID")
        parsed_pretrade = _parse_cal_date(pretrade_date)
        if parsed_pretrade >= cal_date:
            raise CloseSessionAuthorityError("CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID")
        observed.add(cal_date)
        result.append((cal_date, raw_is_open == 1, pretrade_date))
    if not result:
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_EMPTY")
    # The official endpoint currently returns newest-first.  Provider order is
    # retained in raw evidence; the semantic calendar is normalized by date.
    result.sort(key=lambda row: row[0])
    requested_dates = [
        start_date + timedelta(days=offset) for offset in range((end_date - start_date).days + 1)
    ]
    if [row[0] for row in result] != requested_dates:
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_DATE_COVERAGE_INCOMPLETE")
    latest_open = _parse_cal_date(result[0][2])
    for cal_date, is_open, pretrade_date in result:
        if _parse_cal_date(pretrade_date) != latest_open:
            raise CloseSessionAuthorityError("CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID")
        if is_open:
            latest_open = cal_date
    return result


def acquire_close_session_authority(
    *,
    now: datetime | None = None,
    client: _TushareClient | None = None,
) -> CloseSessionAuthorityResult:
    """Fetch a fresh exact SSE partition and select the latest closed open day."""

    local_zone = ZoneInfo(TIMEZONE)
    observed_now = now or datetime.now(tz=local_zone)
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        raise CloseSessionAuthorityError("CLOSE_AUTHORITY_TIME_INVALID")
    local_now = observed_now.astimezone(local_zone)
    end_date = local_now.date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    params = {
        "exchange": SSE,
        "start_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
    }
    # Official Tushare batch responses use ``count=0`` as a placeholder even
    # when items are present.  Strict decoding preserves the provider count
    # separately while binding the accepted count to exact item cardinality.
    response = (client or OfficialTushareHttpsClient(strict_decimal_decode=True)).request(
        api_name=API_NAME,
        params=params,
        expected_fields=EXPECTED_FIELDS,
    )
    rows = _validated_rows(response, start_date=start_date, end_date=end_date)
    eligible = [
        cal_date
        for cal_date, is_open, _pretrade in rows
        if is_open
        and (
            cal_date < local_now.date()
            or (cal_date == local_now.date() and local_now.time() >= SESSION_CLOSE_LOCAL)
        )
    ]
    if not eligible:
        raise CloseSessionAuthorityError("CLOSE_SESSION_NOT_AVAILABLE")
    target = eligible[-1]
    today_open = next(
        (is_open for cal_date, is_open, _pretrade in rows if cal_date == local_now.date()),
        None,
    )
    if (
        local_now.time() >= SESSION_CLOSE_LOCAL
        and today_open is True
        and target != local_now.date()
    ):
        raise CloseSessionAuthorityError("CLOSE_SESSION_TARGET_NOT_TODAY")
    ordered_open_dates = [
        cal_date.strftime("%Y%m%d") for cal_date, is_open, _pretrade in rows if is_open
    ]
    raw = bytes(response.raw_body)
    receipt = {
        "schema_version": "cn-close-session-receipt.v1",
        "status": "TARGET_AUTHORIZED",
        "authority": "TUSHARE_SSE_TRUSTED_PROVIDER",
        "authority_limitations": ["NOT_EXCHANGE_OFFICIAL", "DAILY_CLOSE_ONLY"],
        "captured_at": _utc_seconds(observed_now),
        "observed_local_time": _local_seconds(local_now),
        "timezone": TIMEZONE,
        "endpoint_url": OFFICIAL_TUSHARE_URL,
        "api_name": API_NAME,
        "request_params": params,
        "expected_fields": list(EXPECTED_FIELDS),
        "request_id_sha256": hashlib.sha256(response.request_id.encode("utf-8")).hexdigest(),
        "raw_response_byte_length": len(raw),
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "provider_reported_count": response.provider_reported_count,
        "item_count": response.item_count,
        "has_more": response.has_more,
        "calendar_start_date": start_date.strftime("%Y%m%d"),
        "calendar_end_date": end_date.strftime("%Y%m%d"),
        "calendar_date_count": len(rows),
        "ordered_open_dates": ordered_open_dates,
        "target_trade_date": target.strftime("%Y%m%d"),
        "session_close_local": SESSION_CLOSE_LOCAL.isoformat(),
    }
    return CloseSessionAuthorityResult(receipt=receipt, raw_response_bytes=raw)


__all__ = [
    "CloseSessionAuthorityError",
    "CloseSessionAuthorityResult",
    "acquire_close_session_authority",
]
