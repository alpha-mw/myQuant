"""Pure v2 Morning request, owner scope and explicit next-session guards."""

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
import re

from .daily_contract import ContractError, validate_ref
from .daily_journal import _validate_day, _false_authority

REQUEST_SCHEMA = "morning-strategy-request.v2"
REQUEST_SCHEMAS = {REQUEST_SCHEMA: "v2", "morning-strategy-request.v3": "v3"}


def request_version(value):
    version = REQUEST_SCHEMAS.get(value.get("schema_version")) if type(value) is dict else None
    if version is None:
        raise ContractError("MORNING_REQUEST_SCHEMA_INVALID")
    return version


POLICY_SCHEMA = "morning-quote-policy.v1"
_SYMBOL = re.compile(r"[0-9]{6}\.(SH|SZ|BJ)")


def _day(value: str) -> str:
    _validate_day(value)
    return value


def validate_morning_request(value: dict) -> dict:
    version = request_version(value)
    fields = {
        "schema_version",
        "action",
        "run_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "output_ref",
    }
    if version == "v3":
        fields.add("threshold_policy_refs")
        refs = value.get("threshold_policy_refs")
        if type(refs) is not dict or set(refs) != {"trailing", "initial_stop"}:
            raise ContractError("MORNING_THRESHOLD_POLICY_REFS_INVALID")
        for ref in refs.values():
            validate_ref(ref)
    if type(value) is not dict or set(value) != fields:
        raise ContractError("MORNING_V2_REQUEST_INVALID")
    if value["action"] not in {"PREFLIGHT", "REPLAY", "SEAL"}:
        raise ContractError("MORNING_V2_ACTION_INVALID")
    day = _day(value["run_date"])
    for name in (
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
    ):
        validate_ref(value[name])
    if version == "v3":
        previous = PurePosixPath(value["previous_completion_ref"]["path"]).parent.name
        _validate_day(previous)
        if (
            previous >= day
            or value["previous_completion_ref"]["path"]
            != f"results/operations/daily_production/CN/{previous}/completion.v1.json"
        ):
            raise ContractError("MORNING_PREVIOUS_COMPLETION_REF_INVALID")
    if value["action"] == "SEAL":
        output = validate_ref(value["output_ref"])
        if (
            output["path"]
            != f"results/operations/morning_strategy/CN/{day}/0945-strategy.{version}.md"
        ):
            raise ContractError("MORNING_V2_OUTPUT_PATH_INVALID")
    elif value["output_ref"] is not None:
        raise ContractError("MORNING_V2_READ_ACTION_HAS_OUTPUT")
    return dict(value)


def expected_quote_symbols(*, policy: dict, holdings: list[dict], run_date: str) -> list[str]:
    _day(run_date)
    fields = {
        "schema_version",
        "strategy_id",
        "market",
        "effective_from",
        "effective_through",
        "revoked_at",
        "additional_symbols",
        "authority",
    }
    if (
        type(policy) is not dict
        or set(policy) != fields
        or policy["schema_version"] != POLICY_SCHEMA
        or policy["market"] != "CN"
        or policy["strategy_id"] != "aggressive_tech_manufacturing"
        or policy["revoked_at"] is not None
        or not _false_authority(policy["authority"])
    ):
        raise ContractError("MORNING_QUOTE_POLICY_INVALID")
    start, end = _day(policy["effective_from"]), _day(policy["effective_through"])
    if not start <= run_date <= end:
        raise ContractError("MORNING_QUOTE_POLICY_NOT_EFFECTIVE")
    extra = policy["additional_symbols"]
    if (
        type(extra) is not list
        or any(type(s) is not str or _SYMBOL.fullmatch(s) is None for s in extra)
        or extra != sorted(set(extra))
    ):
        raise ContractError("MORNING_QUOTE_POLICY_SYMBOLS_INVALID")
    selected, seen = set(extra), set()
    for row in holdings:
        symbol = row.get("symbol")
        if type(symbol) is not str or _SYMBOL.fullmatch(symbol) is None or symbol in seen:
            raise ContractError("MORNING_HOLDING_SYMBOL_INVALID")
        seen.add(symbol)
        try:
            shares = Decimal(str(row["shares"]))
        except (KeyError, InvalidOperation) as exc:
            raise ContractError("MORNING_HOLDING_SHARES_INVALID") from exc
        if not shares.is_finite() or shares < 0:
            raise ContractError("MORNING_HOLDING_SHARES_INVALID")
        if shares > 0:
            selected.add(symbol)
    if not selected:
        raise ContractError("MORNING_EMPTY_QUOTE_SCOPE_UNSUPPORTED")
    return sorted(selected)


def require_next_open_session(
    *, runtime_projection: list[dict], previous_date: str, run_date: str
) -> None:
    """Rows must already have native source/custody validation; never infer holidays."""
    previous, current = _day(previous_date), _day(run_date)
    if previous >= current:
        raise ContractError("MORNING_PREVIOUS_SESSION_INVALID")
    rows = {}
    for row in runtime_projection:
        day = _day(str(row["date"]).replace("-", ""))
        if day in rows or row.get("status") not in {"OPEN", "CLOSED"}:
            raise ContractError("MORNING_CALENDAR_ROWS_INVALID")
        rows[day] = row["status"]
    date = datetime.strptime(previous, "%Y%m%d").date()
    end = datetime.strptime(current, "%Y%m%d").date()
    while date <= end:
        if date.strftime("%Y%m%d") not in rows:
            raise ContractError("MORNING_CALENDAR_COVERAGE_MISSING")
        date += timedelta(days=1)
    if (
        rows[previous] != "OPEN"
        or rows[current] != "OPEN"
        or any(state == "OPEN" for day, state in rows.items() if previous < day < current)
    ):
        raise ContractError("MORNING_NOT_IMMEDIATELY_NEXT_OPEN_SESSION")
