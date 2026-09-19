"""Read-only Calendar range planning, before native input/previous-EOD preflight.

This planner never authorizes execution. The launcher must independently validate
its prior completion/bootstrap anchor and every native day input before writing.
"""

from typing import Mapping
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.close_session_authority import replay_close_session_authority
from .daily_contract import ContractError, validate_ref
from .daily_journal import _validate_day, FALSE_AUTHORITY
from .native_input_contract import validate_native_input_shape
from quant_investor.market.requested_session import classify_catchup_session


def catchup_result_dates(
    plan: dict, *, previous_trade_date: str, target_trade_date: str
) -> list[str]:
    """A completed target has one replay row even when there are no missing dates."""
    dates = list(plan["ordered_trade_dates"])
    if not dates and target_trade_date == previous_trade_date:
        return [previous_trade_date]
    return dates


def plan_catchup(
    *,
    workspace: str,
    calendar_ref: Mapping[str, str],
    raw_calendar_ref: Mapping[str, str],
    previous_trade_date: str,
    target_trade_date: str,
    day_input_refs: Mapping[str, Mapping[str, str]],
) -> dict:
    _validate_day(previous_trade_date)
    _validate_day(target_trade_date)
    if previous_trade_date > target_trade_date:
        raise ContractError("CATCHUP_DATE_ORDER_INVALID")
    reader = SecureSystemStorage(workspace)

    def read(ref):
        validate_ref(ref)
        stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=4 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise ContractError("CATCHUP_SOURCE_SHA_MISMATCH")
        return stored.data

    document = parse_json_bytes(
        read(calendar_ref), label="catch-up Calendar", require_canonical=False
    )
    raw = read(raw_calendar_ref)
    receipt = replay_close_session_authority(document, raw).receipt
    if not (
        receipt["calendar_start_date"]
        <= previous_trade_date
        <= target_trade_date
        <= receipt["calendar_end_date"]
    ):
        raise ContractError("CATCHUP_CALENDAR_RANGE_INCOMPLETE")
    days = receipt["ordered_open_dates"]
    if previous_trade_date not in days:
        raise ContractError("CATCHUP_ANCHOR_NOT_OPEN_SESSION")
    closed = target_trade_date not in days
    if not closed and target_trade_date > receipt["target_trade_date"]:
        raise ContractError("CATCHUP_TARGET_NOT_AUTHORIZED")
    required = (
        [] if closed else [day for day in days if previous_trade_date < day <= target_trade_date]
    )
    if type(day_input_refs) is not dict or set(day_input_refs) - set(required):
        raise ContractError("CATCHUP_INPUT_DATE_SET_INVALID")
    previous = previous_trade_date
    observed_day = receipt["observed_local_time"][:10].replace("-", "")
    for day in required:
        if day < observed_day:
            classify_catchup_session(
                requested_trade_date=day,
                previous_trade_date=previous,
                receipt=document,
                raw=raw,
            )
        previous = day
    inputs = {}
    for day in required:
        if day not in day_input_refs:
            continue
        ref = validate_ref(day_input_refs[day])
        value = parse_json_bytes(read(ref), label="catch-up day input", require_canonical=True)
        validate_native_input_shape(value)
        if value["trade_date"] != day:
            raise ContractError("CATCHUP_DAY_INPUT_BINDING_INVALID")
        inputs[day] = ref
    missing = [day for day in required if day not in inputs]
    return {
        "schema_version": "cn-daily-catchup-plan.v1",
        "status": "NON_TRADING_DAY_NO_ACTION" if closed else "BLOCKED" if missing else "PLANNED",
        "previous_trade_date": previous_trade_date,
        "target_trade_date": target_trade_date,
        "calendar_ref": dict(calendar_ref),
        "raw_calendar_ref": dict(raw_calendar_ref),
        "ordered_trade_dates": required,
        "day_input_refs": inputs,
        "missing_input_dates": missing,
        "native_preflight_required": True,
        "prior_completion_validation_required": True,
        "execution_authorized": False,
        "authority": FALSE_AUTHORITY,
    }
