"""Exact public daily-close response contract; no execution or evidence authority."""

from collections.abc import Sequence
from .daily_contract import ContractError, validate_ref
from .daily_journal import _false_authority, _validate_day

SCHEMA = "cn-daily-production-result.v1"
ACTIONS = frozenset({"PLAN", "EXECUTE", "RESUME", "CATCH_UP"})
FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "target_trade_date",
        "execution_state",
        "business_state",
        "days",
        "authority",
    }
)
DAY_FIELDS = frozenset({"trade_date", "execution_state", "business_state", "completion_ref"})
INCOMPLETE = frozenset({"PARTIAL", "BLOCKED", "FAILED"})


def _require(condition: bool) -> None:
    if not condition:
        raise ContractError("DAILY_PRODUCTION_RESULT_INVALID")


def _validate_row(row: dict, *, action: str) -> None:
    _require(type(row) is dict and set(row) == DAY_FIELDS)
    _validate_day(row["trade_date"])
    state, business, ref = row["execution_state"], row["business_state"], row["completion_ref"]
    if action == "PLAN":
        _require(state == "PLANNED" and business == "NOT_EVALUATED" and ref is None)
    elif state in {"SUCCEEDED", "NO_ACTION"}:
        _require(business == "COMPLETE" and ref is not None)
        checked = validate_ref(ref)
        _require(
            checked["path"]
            == f'results/operations/daily_production/CN/{row["trade_date"]}/completion.v1.json'
        )
    else:
        _require(state in INCOMPLETE and business == "INCOMPLETE" and ref is None)


def validate_production_result(
    value: dict, *, action: str, target_trade_date: str, expected_dates: Sequence[str]
) -> dict:
    """Check the handler result against code-derived Calendar dates and request echoes.

    Native EOD refs must already have been replayed by the fixed handler. This
    shape validator does not read files or turn a well-formed ref into evidence.
    """
    _validate_day(target_trade_date)
    _require(action in ACTIONS)
    dates = list(expected_dates)
    for day in dates:
        _validate_day(day)
    _require(dates == sorted(set(dates)) and all(d <= target_trade_date for d in dates))
    _require(type(value) is dict and set(value) == FIELDS)
    _require(
        value["schema_version"] == SCHEMA
        and value["action"] == action
        and value["target_trade_date"] == target_trade_date
        and _false_authority(value["authority"])
        and type(value["days"]) is list
    )
    for row in value["days"]:
        _validate_row(row, action=action)
    _require([row["trade_date"] for row in value["days"]] == dates)
    state, business = value["execution_state"], value["business_state"]
    if not dates:
        _require(state == "NO_ACTION" and business == "NON_TRADING_DAY")
    elif action == "PLAN":
        _require(state == "PLANNED" and business == "NOT_EVALUATED")
    else:
        states = {row["execution_state"] for row in value["days"]}
        if states <= {"SUCCEEDED", "NO_ACTION"}:
            expected = "NO_ACTION" if states == {"NO_ACTION"} else "SUCCEEDED"
            _require(state == expected and business == "COMPLETE")
        else:
            expected = (
                "FAILED"
                if "FAILED" in states
                else "PARTIAL" if states & {"SUCCEEDED", "NO_ACTION", "PARTIAL"} else "BLOCKED"
            )
            _require(state == expected and business == "INCOMPLETE")
    return value


def production_result_exit_code(value: dict) -> int:
    """Use only after validation; stable expected outcomes are exit 0 or 2."""
    return 2 if value["execution_state"] in INCOMPLETE else 0
