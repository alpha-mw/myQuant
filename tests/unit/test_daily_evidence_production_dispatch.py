"""Action/result dispatch with controlled native boundaries, not native execution proof."""

import importlib
import sys
from pathlib import Path
import json
import pytest
from test_daily_evidence_execution_controls import fixture
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
module = importlib.import_module("scripts.daily_production")


@pytest.mark.parametrize(
    "action,status",
    [("EXECUTE", "COMPLETE"), ("RESUME", "COMPLETE"), ("EXECUTE", "PARTIAL"), ("RESUME", "FAILED")],
)
def test_dispatch_reuses_native_paths_and_replays_completion(tmp_path, monkeypatch, action, status):
    ref, put = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    day = request["target_trade_date"]
    request["action"] = action
    handoff = put("handoff.json", {})
    if action == "RESUME":
        request.update(recipe_ref=None, maintenance_handoff_ref=handoff)
    ref = put("request.json", request)
    complete = {
        "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
        "sha256": "c" * 64,
    }
    calls = []
    native_result = {"status": status, "completion_ref": complete if status == "COMPLETE" else None}

    def execute(**kw):
        calls.append("execute")
        assert action == "EXECUTE" and kw["request_ref"] == ref
        return native_result

    def inputs(**kw):
        calls.append("materialize")
        assert action == "RESUME" and kw["handoff_ref"] == handoff
        assert kw.get("_execute_theme", False) is False
        return {"native_inputs_ref": handoff}

    monkeypatch.setattr(module, "execute_daily_recipe", execute)
    monkeypatch.setattr(
        module,
        "read_maintenance_handoff",
        lambda **kw: {
            "handoff": {"trade_date": day, "release_install_ref": request["release_install_ref"]},
            "handoff_ref": handoff,
            "recipe": {"schema_version": "cn-daily-execute-recipe.v1"},
        },
    )
    monkeypatch.setattr(module, "materialize_daily_inputs", inputs)
    monkeypatch.setattr(
        module, "run_materialized_native_input", lambda **kw: calls.append("run") or native_result
    )

    def replay(**kw):
        calls.append("replay")
        assert kw["completion_ref"] == complete
        return {
            "native_replay_validated": True,
            "completion_ref": complete,
            "trade_date": day,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": True,
        }

    monkeypatch.setattr(module, "replay_native_completion", replay)
    result = module.dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )
    assert result["execution_state"] == ("SUCCEEDED" if status == "COMPLETE" else status)
    assert result["days"][0]["completion_ref"] == native_result["completion_ref"]
    assert calls == (["execute"] if action == "EXECUTE" else ["materialize", "run"]) + (
        ["replay"] if status == "COMPLETE" else []
    )


def test_dispatch_plan_and_provenance_guard_precede_execution(tmp_path, monkeypatch):
    ref, put = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    put("SYNTHETIC-DATA-CLOCK.json", {"synthetic": True})

    def forbidden(**kw):
        pytest.fail("marked synthetic workspace reached public execution")

    monkeypatch.setattr(module, "execute_daily_recipe", forbidden)
    with pytest.raises(ContractError, match="SYNTHETIC_WORKSPACE"):
        module.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
        )
    request["action"] = "PLAN"
    ref = put("request.json", request)
    result = module.dispatch_daily_request(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=request["release_install_ref"]
    )
    assert result["execution_state"] == "PLANNED"
    assert result["business_state"] == "NOT_EVALUATED"


@pytest.mark.parametrize(
    "result",
    [
        {"status": "COMPLETE"},
        {"status": "MADE_UP"},
        {"status": "PARTIAL", "completion_ref": {"path": "fake.json", "sha256": "a" * 64}},
    ],
)
def test_malformed_native_outcome_is_internal_failure_not_business_success(
    tmp_path, monkeypatch, result
):
    ref, _ = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    monkeypatch.setattr(module, "execute_daily_recipe", lambda **kw: result)
    with pytest.raises(RuntimeError, match="native daily"):
        module.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
        )


@pytest.mark.parametrize("empty", [False, True])
def test_catchup_uses_calendar_planner_and_fixed_existing_handler(tmp_path, monkeypatch, empty):
    from quant_investor.operations.daily_journal import FALSE_AUTHORITY

    ref, put = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    day = request["target_trade_date"]
    source = request["release_install_ref"]
    request.update(
        action="CATCH_UP",
        recipe_ref=None,
        calendar_ref=source,
        raw_calendar_ref=source,
        previous_completion_ref={
            "path": "results/operations/daily_production/CN/20260903/completion.v1.json",
            "sha256": "a" * 64,
        },
    )
    ref = put("request.json", request)
    calls = []
    monkeypatch.setattr(
        module,
        "plan_catchup",
        lambda **kw: calls.append("plan") or {"ordered_trade_dates": [] if empty else [day]},
    )
    result = {
        "schema_version": "cn-daily-production-result.v1",
        "action": "CATCH_UP",
        "target_trade_date": day,
        "execution_state": "NO_ACTION" if empty else "BLOCKED",
        "business_state": "NON_TRADING_DAY" if empty else "INCOMPLETE",
        "days": (
            []
            if empty
            else [
                {
                    "trade_date": day,
                    "execution_state": "BLOCKED",
                    "business_state": "INCOMPLETE",
                    "completion_ref": None,
                }
            ]
        ),
        "authority": dict(FALSE_AUTHORITY),
    }
    monkeypatch.setattr(
        module, "run_native_catchup", lambda **kw: calls.append("catchup") or result
    )
    assert (
        module.dispatch_daily_request(
            workspace=str(tmp_path), request_ref=ref, release_install_ref=source, synthetic=True
        )
        == result
    )
    assert calls == ["plan", "catchup"]
