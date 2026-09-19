"""Exact daily deadline policy syntax; native Calendar and custody validate separately."""

from zoneinfo import ZoneInfo
from .daily_contract import ContractError, GRAPH_SHA256, utc_stamp
from .daily_journal import _validate_day

SCHEMA = "cn-daily-prediction-policy.v1"
FIELDS = frozenset(
    {"schema_version", "trade_date", "graph_sha256", "session_rule", "prediction_deadline"}
)


def validate_prediction_policy(value: dict, *, trade_date: str) -> dict:
    """No default deadline, current clock, registration or admission inference."""
    _validate_day(trade_date)
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value["schema_version"] != SCHEMA
        or value["trade_date"] != trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or value["session_rule"] != "CN_SSE_SZSE_OPEN"
    ):
        raise ContractError("PREDICTION_POLICY_CONTRACT_INVALID")
    deadline = utc_stamp(value["prediction_deadline"])
    if deadline.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != trade_date:
        raise ContractError("PREDICTION_POLICY_DEADLINE_DATE_INVALID")
    return dict(value)
