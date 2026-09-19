"""Initial PLAN has no produced inputs or business readiness claims."""

import json
import pytest
from test_daily_evidence_execution_controls import fixture
from test_daily_evidence_execute_wiring import module
from quant_investor.operations.daily_contract import ContractError


def test_plan_reads_recipe_without_claim_calendar_or_business_writes(tmp_path, monkeypatch):
    ref, put = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    request["action"] = "PLAN"
    recipe = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    recipe["factor_loop_context_ref"] = None
    request["recipe_ref"] = put("recipe.json", recipe)
    ref = put("request.json", request)

    def forbidden(*args, **kwargs):
        pytest.fail("PLAN reached maintenance, network or writer lock")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden
    )
    monkeypatch.setattr("quant_investor.operations.daily_journal.DailyJournal.locked", forbidden)
    before = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    }
    result = module.plan_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert result["execution_state"] == "PLANNED"
    assert result["business_state"] == "NOT_EVALUATED"
    assert result["days"] == [
        {
            "trade_date": request["target_trade_date"],
            "execution_state": "PLANNED",
            "business_state": "NOT_EVALUATED",
            "completion_ref": None,
        }
    ]
    assert all(value is False for value in result["authority"].values())
    assert before == {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    }


def test_plan_rejects_execute_action_without_running_it(tmp_path):
    ref, _ = fixture(tmp_path)
    with pytest.raises(ContractError, match="PLAN_ACTION_REQUIRED"):
        module.plan_daily_recipe(workspace=str(tmp_path), request_ref=ref)
