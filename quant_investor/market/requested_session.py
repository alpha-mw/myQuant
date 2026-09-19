"""Pure requested-session classification from native close receipt and exact wire bytes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from .close_session_authority import CloseSessionAuthorityError, replay_close_session_authority
from .tushare_transport import TushareHttpsError


def classify_catchup_session(
    *, requested_trade_date: str, previous_trade_date: str, receipt: dict, raw: bytes
) -> dict:
    """Prove one historical Calendar edge, without granting execution authority."""
    for day in (requested_trade_date, previous_trade_date):
        try:
            parsed = datetime.strptime(day, "%Y%m%d")
        except (TypeError, ValueError) as exc:
            raise CloseSessionAuthorityError("CATCHUP_SESSION_DATE_INVALID") from exc
        if parsed.strftime("%Y%m%d") != day:
            raise CloseSessionAuthorityError("CATCHUP_SESSION_DATE_INVALID")
    if previous_trade_date >= requested_trade_date:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_DATE_ORDER_INVALID")
    try:
        verified = replay_close_session_authority(receipt, raw).receipt
    except (CloseSessionAuthorityError, TushareHttpsError) as exc:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_REPLAY_INVALID") from exc
    if not (
        verified["calendar_start_date"]
        <= previous_trade_date
        < requested_trade_date
        <= verified["calendar_end_date"]
    ):
        raise CloseSessionAuthorityError("CATCHUP_SESSION_COVERAGE_INCOMPLETE")
    days = verified["ordered_open_dates"]
    if previous_trade_date not in days:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_PREVIOUS_NOT_OPEN")
    if requested_trade_date not in days:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_REQUESTED_NOT_OPEN")
    if days.index(requested_trade_date) != days.index(previous_trade_date) + 1:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_NOT_ADJACENT")
    if requested_trade_date > verified["target_trade_date"]:
        raise CloseSessionAuthorityError("CATCHUP_SESSION_NOT_AUTHORIZED")
    observed = datetime.strptime(verified["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    if requested_trade_date >= observed.strftime("%Y%m%d"):
        raise CloseSessionAuthorityError("CATCHUP_SESSION_NOT_HISTORICAL")
    return {
        "requested_trade_date": requested_trade_date,
        "previous_trade_date": previous_trade_date,
        "authorized_close_trade_date": verified["target_trade_date"],
        "observed_at": observed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "HISTORICAL_CLOSED",
        "evidence_classification": "RETROSPECTIVE_RECOMPUTE",
        "prospective": False,
        "execution_authorized": False,
    }


def classify_requested_session(*, requested_trade_date: str, receipt: dict, raw: bytes) -> dict:
    try:
        requested = datetime.strptime(requested_trade_date, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_DATE_INVALID") from exc
    if requested.strftime("%Y%m%d") != requested_trade_date:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_DATE_INVALID")
    verified = replay_close_session_authority(receipt, raw).receipt
    observed = datetime.strptime(verified["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    if observed.strftime("%Y%m%d") != requested_trade_date:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_OBSERVATION_DATE_MISMATCH")
    if not verified["calendar_start_date"] <= requested_trade_date <= verified["calendar_end_date"]:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_NOT_COVERED")
    authorized = verified["target_trade_date"]
    if requested_trade_date in verified["ordered_open_dates"]:
        if authorized != requested_trade_date:
            raise CloseSessionAuthorityError("REQUESTED_OPEN_SESSION_NOT_CLOSED")
        classification = "MATCHED_OPEN"
    else:
        if authorized >= requested_trade_date:
            raise CloseSessionAuthorityError("REQUESTED_CLOSED_SESSION_TARGET_INVALID")
        classification = "CONFIRMED_CLOSED"
    return {
        "requested_trade_date": requested_trade_date,
        "authorized_close_trade_date": authorized,
        "classification": classification,
    }


def requested_session_attempt_result(
    *,
    requested_trade_date: str,
    receipt: dict,
    raw: bytes,
    close_ref: dict,
    mode: str,
    slot: str,
    protected_surfaces: list,
    expected_previous_trade_date: str | None = None,
) -> dict | None:
    """Return an evidence-only exit, or None to allow the matched OPEN components."""
    try:
        classified = classify_requested_session(
            requested_trade_date=requested_trade_date, receipt=receipt, raw=raw
        )
        if (
            classified["classification"] == "MATCHED_OPEN"
            and expected_previous_trade_date is not None
        ):
            require_calendar_predecessor(
                requested_trade_date=requested_trade_date,
                previous_trade_date=expected_previous_trade_date,
                receipt=receipt,
                raw=raw,
            )
    except (CloseSessionAuthorityError, TushareHttpsError) as exc:
        classified = None
        blockers = [exc.code]
    else:
        if classified["classification"] == "MATCHED_OPEN":
            return None
        blockers = []
    status = "NO_ACTION" if classified else "BLOCKED"
    outcome = {
        "schema_version": "cn-daily-maintenance-attempt.v1",
        "status": status,
        "maintenance_status": status,
        "same_day_status": status,
        "execution_disposition": "NON_TRADING_DAY" if classified else "REQUESTED_SESSION_BLOCKED",
        "request_target_trade_date": requested_trade_date,
        "target_date": requested_trade_date,
        "authorized_close_trade_date": receipt["target_trade_date"],
        "factor_input_readiness": "NOT_APPLICABLE" if classified else "BLOCKED",
        "factor_rollover_eligible": False,
        "fundamental_integrity_status": "UNCONFIRMED",
        "fundamental_refresh_status": "HEALTH_ONLY",
        "mode": mode,
        "attempt_slot": slot,
        "canonical_unchanged": True,
        "canonical_write_count": 0,
        "usable_for_investment_research": "UNCONFIRMED",
        "close_session_receipt_ref": dict(close_ref),
        "stage_results": [],
        "blockers": blockers,
        "protected_surfaces": list(protected_surfaces),
    }
    if expected_previous_trade_date is not None and classified is None:
        outcome["request_previous_trade_date"] = expected_previous_trade_date
    return outcome


def recorded_requested_session(
    *,
    attempt: Path,
    receipt: dict,
    requested_trade_date: str,
    read,
    expected_previous_trade_date=None,
) -> dict:
    """Replay exact attempt-local Calendar refs; never read a supplied alternate path."""
    ref = receipt.get("close_session_receipt_ref")
    if (
        type(ref) is not dict
        or set(ref) != {"path", "sha256"}
        or ref["path"] != str(attempt / "close-session-receipt.json")
    ):
        raise CloseSessionAuthorityError("REQUESTED_SESSION_RECEIPT_REF_INVALID")
    raw_receipt = read(attempt / "close-session-receipt.json")
    if hashlib.sha256(raw_receipt).hexdigest() != ref["sha256"]:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_RECEIPT_SHA_MISMATCH")
    try:
        calendar = json.loads(raw_receipt)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_RECEIPT_INVALID") from exc
    if type(calendar) is not dict:
        raise CloseSessionAuthorityError("REQUESTED_SESSION_RECEIPT_INVALID")
    raw_path = attempt / "close-session.raw.json"
    if calendar.get("raw_response_path") != str(raw_path):
        raise CloseSessionAuthorityError("REQUESTED_SESSION_RAW_PATH_INVALID")
    wire = read(raw_path)
    classified = classify_requested_session(
        requested_trade_date=requested_trade_date, receipt=calendar, raw=wire
    )
    if classified["classification"] == "MATCHED_OPEN" and expected_previous_trade_date is not None:
        require_calendar_predecessor(
            requested_trade_date=requested_trade_date,
            previous_trade_date=expected_previous_trade_date,
            receipt=calendar,
            raw=wire,
        )
    return {
        **classified,
        "close_session_receipt_ref": dict(ref),
        "raw_response_ref": {"path": str(raw_path), "sha256": hashlib.sha256(wire).hexdigest()},
        "business_callbacks_this_call": False,
    }


def validate_closed_attempt(
    *, attempt: Path, receipt: dict, requested_trade_date: str, read
) -> dict:
    expected = {
        "status": "NO_ACTION",
        "maintenance_status": "NO_ACTION",
        "same_day_status": "NO_ACTION",
        "execution_disposition": "NON_TRADING_DAY",
        "target_date": requested_trade_date,
        "request_target_trade_date": requested_trade_date,
        "factor_input_readiness": "NOT_APPLICABLE",
        "mode": "execute",
        "stage_results": [],
        "blockers": [],
    }
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("canonical_unchanged") is not True
        or receipt.get("factor_rollover_eligible") is not False
        or type(receipt.get("canonical_write_count")) is not int
        or receipt["canonical_write_count"] != 0
        or "core_completion_ref" in receipt
    ):
        raise CloseSessionAuthorityError("REQUESTED_CLOSED_ATTEMPT_INVALID")
    result = recorded_requested_session(
        attempt=attempt, receipt=receipt, requested_trade_date=requested_trade_date, read=read
    )
    if (
        result["classification"] != "CONFIRMED_CLOSED"
        or receipt.get("authorized_close_trade_date") != result["authorized_close_trade_date"]
    ):
        raise CloseSessionAuthorityError("REQUESTED_CLOSED_ATTEMPT_CALENDAR_MISMATCH")
    return result


def require_calendar_predecessor(*, requested_trade_date, previous_trade_date, receipt, raw):
    """Prove an exact closed Calendar edge, regardless of current/history consumption."""
    for day in (requested_trade_date, previous_trade_date):
        try:
            parsed = datetime.strptime(day, "%Y%m%d")
        except (TypeError, ValueError) as exc:
            raise CloseSessionAuthorityError("REQUESTED_PREDECESSOR_DATE_INVALID") from exc
        if parsed.strftime("%Y%m%d") != day:
            raise CloseSessionAuthorityError("REQUESTED_PREDECESSOR_DATE_INVALID")
    verified = replay_close_session_authority(receipt, raw).receipt
    days = verified["ordered_open_dates"]
    if (
        not verified["calendar_start_date"]
        <= previous_trade_date
        < requested_trade_date
        <= verified["calendar_end_date"]
        or previous_trade_date not in days
        or requested_trade_date not in days
        or requested_trade_date > verified["target_trade_date"]
        or days.index(requested_trade_date) != days.index(previous_trade_date) + 1
    ):
        raise CloseSessionAuthorityError("REQUESTED_PREDECESSOR_NOT_ADJACENT")
    return {
        "requested_trade_date": requested_trade_date,
        "previous_trade_date": previous_trade_date,
        "authorized_close_trade_date": verified["target_trade_date"],
    }


def classify_current_session_edge(*, requested_trade_date, previous_trade_date, receipt, raw):
    value = classify_requested_session(
        requested_trade_date=requested_trade_date, receipt=receipt, raw=raw
    )
    if value["classification"] != "MATCHED_OPEN":
        raise CloseSessionAuthorityError("REQUESTED_CURRENT_OPEN_SESSION_REQUIRED")
    edge = require_calendar_predecessor(
        requested_trade_date=requested_trade_date,
        previous_trade_date=previous_trade_date,
        receipt=receipt,
        raw=raw,
    )
    return {**value, "previous_trade_date": edge["previous_trade_date"]}
