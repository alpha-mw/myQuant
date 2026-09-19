"""Fixed coordinator ordering with controlled native boundaries, not a full DAG proof."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace
import pytest
from test_daily_evidence_store_materialization import context as _scripts_context  # noqa: F401
from test_daily_evidence_execution_controls import fixture
from scripts import daily_materialization as module
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal


def setup(tmp_path, monkeypatch, *, fail_gate=False):
    ref, _ = fixture(tmp_path)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    recipe = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    events = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 4, 13, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", Clock)

    def install(controls):
        events.append("install")
        if fail_gate:
            raise ContractError("TEST_INSTALL_REJECTED")

    monkeypatch.setattr(
        "quant_investor.operations.execution_controls."
        "verify_execution_install_and_research_policies",
        install,
    )
    monkeypatch.setattr(
        "scripts.daily_store_materialization.verify_initial_store_controls",
        lambda **k: events.append("store"),
    )
    monkeypatch.setattr(
        "scripts.daily_completion_replay.verify_initial_previous_completion",
        lambda **k: events.append("previous"),
    )
    handoff = {"path": "handoff.json", "sha256": "a" * 64}

    def publish(**kwargs):
        events.append("handoff")
        assert kwargs["request_ref"] == ref
        return handoff

    monkeypatch.setattr(
        "quant_investor.operations.maintenance_handoff.publish_maintenance_handoff", publish
    )

    class Loop:
        def __init__(self, **kwargs):
            events.append("loop")
            assert kwargs["run_root"] == str(tmp_path / "data/private/cn_daily_maintenance")
            self.hook = kwargs["core_handoff_completed"]

        def core_completed(self, checkpoint):
            events.append("top100")
            self.hook({"fixture": "native core state"})
            events.append("settlement")
            return {}

        def report(self, **kwargs):
            events.append("report")

        def _replay_core_completed(self, checkpoint):
            return self.core_completed(checkpoint)

    monkeypatch.setattr("quant_investor.market.daily_factor_loop.DailyFactorLoop", Loop)

    def maintenance(**kwargs):
        events.append("maintenance")
        assert kwargs["attempt_slot"] == "2020" and kwargs["mode"] == "execute"
        assert kwargs["_expected_target_trade_date"] == "20260904"
        kwargs["core_completed"]({})
        events.append("auxiliaries")
        return {"status": "READY"}

    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", maintenance
    )
    monkeypatch.setattr(
        module,
        "read_maintenance_handoff",
        lambda **k: {"handoff": {"request_ref": ref}, "recipe": recipe},
    )

    def materialize(**kwargs):
        assert kwargs["_execute_theme"] is True
        events.append("materialize")
        return {"native_inputs_ref": handoff}

    monkeypatch.setattr(module, "materialize_daily_inputs", materialize)
    monkeypatch.setattr(
        "scripts.daily_completion.run_materialized_native_input",
        lambda **k: events.append("seal") or {"status": "COMPLETE"},
    )
    return ref, events


def test_fresh_execute_calls_maintenance_once_and_anchors_before_auxiliaries(tmp_path, monkeypatch):
    ref, events = setup(tmp_path, monkeypatch)
    assert (
        module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref, synthetic=True)[
            "status"
        ]
        == "COMPLETE"
    )
    assert events == [
        "install",
        "store",
        "previous",
        "loop",
        "maintenance",
        "top100",
        "handoff",
        "settlement",
        "auxiliaries",
        "report",
        "materialize",
        "seal",
    ]


def test_failed_preflight_cannot_construct_loop_or_call_maintenance(tmp_path, monkeypatch):
    ref, events = setup(tmp_path, monkeypatch, fail_gate=True)
    with pytest.raises(ContractError, match="TEST_INSTALL_REJECTED"):
        module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert events == ["install"]
    assert not (tmp_path / "data/private/cn_daily_maintenance").exists()


def test_existing_handoff_bypasses_new_maintenance_and_mutable_preimages(tmp_path, monkeypatch):
    ref, events = setup(tmp_path, monkeypatch)
    journal = DailyJournal(str(tmp_path), "20260904")
    journal.storage.write(
        str(journal.root / "executions" / ref["sha256"] / "maintenance-handoff.v1.json"), b"{}\n"
    )
    (tmp_path / "input.json").unlink()
    assert (
        module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)["status"]
        == "COMPLETE"
    )
    assert events == ["materialize", "seal"]


def test_completed_request_replays_before_current_preimages(tmp_path, monkeypatch):
    ref, events = setup(tmp_path, monkeypatch)
    journal = DailyJournal(str(tmp_path), "20260904")
    journal.storage.write(str(journal.root / "completion.v1.json"), b"{}\n")
    monkeypatch.setattr(
        "quant_investor.operations.completion_readback.inspect_recorded_completion",
        lambda **k: {
            "completed_handoff_snapshot": SimpleNamespace(
                document=lambda role: {"request_ref": ref}
            )
        },
    )
    monkeypatch.setattr(
        module, "replay_native_completion", lambda **k: events.append("native-replay")
    )
    (tmp_path / "input.json").unlink()
    result = module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert result["execution_state"] == "NO_ACTION"
    assert events == ["native-replay"]


def test_finalized_attempt_repairs_handoff_under_existing_lock_without_maintenance(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    ref, events = setup(tmp_path, monkeypatch)
    claim = (
        tmp_path
        / "data/private/cn_daily_maintenance/logical_tasks/20260904-2020-execute/claim.json"
    )
    claim.parent.mkdir(parents=True)
    claim.write_text("{}")

    @contextmanager
    def existing(**kwargs):
        assert kwargs["run_date"] == "20260904"
        events.append("existing-lock")
        yield {
            "logical_replay": "VERIFIED_SAME_INPUT",
            "provider_calls_this_attempt": False,
            "core_completion_ref": {"path": "core.json", "sha256": "a" * 64},
        }
        events.append("released-lock")

    monkeypatch.setattr(
        "quant_investor.operations.maintenance_readback.locked_finalized_maintenance_replay",
        existing,
    )
    result = module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref, synthetic=True)
    assert result["status"] == "COMPLETE"
    assert events == [
        "install",
        "store",
        "existing-lock",
        "loop",
        "top100",
        "handoff",
        "settlement",
        "report",
        "released-lock",
        "materialize",
        "seal",
    ]


def test_fresh_closed_result_stops_before_report_or_handoff(tmp_path, monkeypatch):
    ref, events = setup(tmp_path, monkeypatch)

    def closed(**kwargs):
        events.append("closed-calendar")
        assert kwargs["_expected_target_trade_date"] == "20260904"
        return {"execution_disposition": "NON_TRADING_DAY"}

    monkeypatch.setattr("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", closed)
    result = module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert result["business_state"] == "NON_TRADING_DAY" and result["days"] == []
    assert events == ["install", "store", "previous", "loop", "closed-calendar"]


def test_locked_closed_projection_returns_before_loop_creation(tmp_path, monkeypatch):
    from contextlib import contextmanager

    ref, events = setup(tmp_path, monkeypatch)
    claim = (
        tmp_path
        / "data/private/cn_daily_maintenance/logical_tasks/20260904-2020-execute/claim.json"
    )
    claim.parent.mkdir(parents=True)
    claim.write_text("{}")

    @contextmanager
    def closed(**kwargs):
        events.append("locked")
        yield {"requested_session_result": {"classification": "CONFIRMED_CLOSED"}}
        events.append("unlocked")

    monkeypatch.setattr(
        "quant_investor.operations.maintenance_readback.locked_finalized_maintenance_replay", closed
    )
    result = module.execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert result["business_state"] == "NON_TRADING_DAY" and result["days"] == []
    assert events == ["install", "store", "locked", "unlocked"]
