"""Native timestamp projection never promotes effective/creation dates to availability."""

from copy import deepcopy
import pytest
from quant_investor.operations.core_timing import recorded_core_timing
from quant_investor.operations.daily_contract import ContractError


def context():
    ref = {"path": "evidence.json", "sha256": "a" * 64}
    nodes = {
        n: {"finished_at": "2026-09-07T20:30:00Z", "recovered": True}
        for n in ("factor", "low_observation", "w80_observation")
    }
    return dict(
        pointer={"activated_at": "2026-09-07T20:10:00Z"},
        pointer_ref=ref,
        generation={"created_at": "2026-08-27T07:00:00Z"},
        observations=[
            {
                "payload": {
                    "factor_alias": a,
                    "registered_at": "2026-09-07T20:14:00Z",
                    "signal_date": "20260827",
                }
            }
            for a in ("LOW", "W80")
        ],
        observation_refs={a: ref for a in ("LOW", "W80")},
        terminal_refs={n: ref for n in nodes},
        terminals=nodes,
    )


def test_historical_creation_and_signal_dates_cannot_backdate_custody():
    args = context()
    original = deepcopy(args)
    result = recorded_core_timing(**args)
    assert result["effective_trade_date"] == "20260827"
    assert result["generation_seal_upper_bound"] == "2026-09-07T20:10:00Z"
    assert result["observation_registration"]["LOW"]["registered_at"] == "2026-09-07T20:14:00Z"
    assert result["generation_created_at_is_availability_proof"] is False
    assert result["prospective_admission"] is False
    assert args == original


@pytest.mark.parametrize("field", ["registration", "seal", "naive", "duplicate"])
def test_unproven_or_impossible_timing_rejects(field):
    args = context()
    if field == "registration":
        args["observations"][0]["payload"]["registered_at"] = "2026-09-08T00:00:00Z"
    elif field == "seal":
        args["pointer"]["activated_at"] = "2026-09-08T00:00:00Z"
    elif field == "naive":
        args["pointer"]["activated_at"] = "2026-09-07T20:10:00"
    else:
        args["observations"][1] = args["observations"][0]
    with pytest.raises(ContractError):
        recorded_core_timing(**args)
