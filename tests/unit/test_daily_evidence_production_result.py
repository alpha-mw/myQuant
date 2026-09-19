"""Result schema rejects false completion, wrong dates, and authority escalation."""

from copy import deepcopy
import pytest
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.production_result import (
    SCHEMA,
    validate_production_result,
    production_result_exit_code,
)


def result(action="RESUME"):
    return {
        "schema_version": SCHEMA,
        "action": action,
        "target_trade_date": "20260904",
        "execution_state": "SUCCEEDED",
        "business_state": "COMPLETE",
        "days": [
            {
                "trade_date": "20260904",
                "execution_state": "SUCCEEDED",
                "business_state": "COMPLETE",
                "completion_ref": {
                    "path": "results/operations/daily_production/CN/20260904/completion.v1.json",
                    "sha256": "a" * 64,
                },
            }
        ],
        "authority": deepcopy(FALSE_AUTHORITY),
    }


def check(value, dates=("20260904",)):
    return validate_production_result(
        value, action=value["action"], target_trade_date="20260904", expected_dates=dates
    )


@pytest.mark.parametrize("action", ["EXECUTE", "RESUME", "CATCH_UP"])
def test_complete_actions_have_exact_bound_ref(action):
    assert production_result_exit_code(check(result(action))) == 0


@pytest.mark.parametrize(
    "fault", ["missing_ref", "wrong_date", "extra_field", "authority", "wrong_aggregate"]
)
def test_malformed_completion_rejected(fault):
    value = result()
    if fault == "missing_ref":
        value["days"][0]["completion_ref"] = None
    elif fault == "wrong_date":
        value["days"][0]["completion_ref"][
            "path"
        ] = "results/operations/daily_production/CN/20260903/completion.v1.json"
    elif fault == "extra_field":
        value["arbitrary"] = True
    elif fault == "authority":
        value["authority"][next(iter(value["authority"]))] = True
    else:
        value["execution_state"] = "NO_ACTION"
    with pytest.raises(ContractError):
        check(value)


def test_plan_and_nontrading_are_not_completed_evidence():
    value = result("PLAN")
    value.update(execution_state="PLANNED", business_state="NOT_EVALUATED")
    value["days"][0].update(
        execution_state="PLANNED", business_state="NOT_EVALUATED", completion_ref=None
    )
    assert production_result_exit_code(check(value)) == 0
    value.update(execution_state="NO_ACTION", business_state="NON_TRADING_DAY", days=[])
    assert production_result_exit_code(check(value, ())) == 0


def test_partial_catchup_preserves_completed_day_and_blocks_false_aggregate():
    value = result("CATCH_UP")
    first = deepcopy(value["days"][0])
    first["trade_date"] = "20260903"
    first["completion_ref"][
        "path"
    ] = "results/operations/daily_production/CN/20260903/completion.v1.json"
    value["days"][0].update(
        execution_state="BLOCKED", business_state="INCOMPLETE", completion_ref=None
    )
    value["days"].insert(0, first)
    value.update(execution_state="PARTIAL", business_state="INCOMPLETE")
    assert production_result_exit_code(check(value, ("20260903", "20260904"))) == 2
    with pytest.raises(ContractError):
        check(value, ("20260904",))
    value.update(execution_state="SUCCEEDED", business_state="COMPLETE")
    with pytest.raises(ContractError):
        check(value, ("20260903", "20260904"))
