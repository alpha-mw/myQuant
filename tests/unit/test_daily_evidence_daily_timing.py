"""Full node custody projection preserves late/recovered time without OOS admission."""

from copy import deepcopy
import pytest

from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_timing import recorded_daily_timing


def context():
    nodes, refs = {}, {}
    for node in EOD_NODE_IDS:
        prefix = f"results/operations/daily_production/CN/20260827/nodes/{node}/key/attempt-0001/"
        ref = {"path": prefix + "terminal.json", "sha256": "a" * 64}
        refs[node] = ref
        nodes[node] = {
            "state": "SUCCEEDED",
            "start_ref": {"path": prefix + "start.json", "sha256": "b" * 64},
            "terminal_ref": ref,
            "start": {"started_at": "2026-09-08T07:00:00Z"},
            "terminal": {
                "finished_at": "2026-09-08T07:01:00Z",
                "recovered": node == "store",
                "output_refs": {"evidence": {"path": "output.json", "sha256": "c" * 64}},
            },
        }
    return dict(
        trade_date="20260827", nodes=nodes, terminal_refs=refs, verified_at="2026-09-08T07:01:00Z"
    )


def test_effective_date_and_start_do_not_backdate_verified_availability():
    args = context()
    original = deepcopy(args)
    value = recorded_daily_timing(**args)
    assert set(value["nodes"]) == EOD_NODE_IDS
    assert value["effective_trade_date"] == "20260827"
    assert value["nodes"]["store"]["recovered_at"] == "2026-09-08T07:01:00Z"
    assert value["nodes"]["factor"]["recovered_at"] is None
    assert all(
        row["first_verified_at"] == "2026-09-08T07:01:00Z" for row in value["nodes"].values()
    )
    assert value["recovered_unknown"] is True
    assert value["prospective_admission"] is False
    assert args == original
    assert recorded_daily_timing(**args) == value


@pytest.mark.parametrize(
    "fault", ["missing", "changed_ref", "attempt", "naive", "future", "reversed", "bool"]
)
def test_invalid_custody_rejected(fault):
    args = context()
    row = args["nodes"]["store"]
    if fault == "missing":
        args["nodes"].pop("macro")
    elif fault == "changed_ref":
        row["terminal_ref"] = {"path": "different.json", "sha256": "a" * 64}
    elif fault == "attempt":
        row["start_ref"]["path"] = "other/start.json"
    elif fault == "naive":
        row["terminal"]["finished_at"] = "2026-09-08T07:01:00"
    elif fault == "future":
        args["verified_at"] = "2026-09-08T07:00:59Z"
    elif fault == "reversed":
        row["start"]["started_at"] = "2026-09-08T07:02:00Z"
    else:
        row["terminal"]["recovered"] = 1
    with pytest.raises(ContractError):
        recorded_daily_timing(**args)
