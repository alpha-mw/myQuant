"""Explicit policy dates use Shanghai sessions, never guessed deadline defaults."""

import pytest
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.operations.prediction_policy import validate_prediction_policy


def policy():
    return dict(
        schema_version="cn-daily-prediction-policy.v1",
        trade_date="20260908",
        graph_sha256=GRAPH_SHA256,
        session_rule="CN_SSE_SZSE_OPEN",
        prediction_deadline="2026-09-08T15:59:59Z",
    )


def test_explicit_policy_is_not_admission_and_uses_shanghai_date():
    value = policy()
    assert validate_prediction_policy(value, trade_date="20260908") == value
    value["prediction_deadline"] = "2026-09-07T16:00:00Z"
    assert validate_prediction_policy(value, trade_date="20260908") == value
    value["prediction_deadline"] = "2026-09-08T16:00:00Z"
    with pytest.raises(ContractError, match="DEADLINE_DATE_INVALID"):
        validate_prediction_policy(value, trade_date="20260908")


@pytest.mark.parametrize("fault", ["missing", "extra", "naive", "offset", "graph", "day", "rule"])
def test_invalid_policy_never_falls_back_to_default(fault):
    value = policy()
    if fault == "missing":
        value.pop("prediction_deadline")
    elif fault == "extra":
        value["prospective"] = True
    elif fault == "naive":
        value["prediction_deadline"] = "2026-09-08T15:00:00"
    elif fault == "offset":
        value["prediction_deadline"] = "2026-09-08T23:00:00+08:00"
    elif fault == "graph":
        value["graph_sha256"] = "a" * 64
    elif fault == "day":
        value["trade_date"] = "20260907"
    else:
        value["session_rule"] = "WEEKDAY"
    with pytest.raises(ContractError):
        validate_prediction_policy(value, trade_date="20260908")
