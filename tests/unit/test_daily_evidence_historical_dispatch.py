"""Real maintenance orchestration and native core validation; producers are controlled."""

from datetime import datetime, timezone
from pathlib import Path
import json

import pytest

from quant_investor.market import daily_maintenance as daily
from quant_investor.market.historical_session import HistoricalSessionError
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
from test_daily_evidence_requested_session import capture
from test_daily_factor_loop_recovery import _components
from test_unified_factor_production_rollover import _write
from quant_investor.market.maintenance_journal import DailyOperationJournal


def context(root):
    workspace, _, _, _, producers = _components(root)
    value = capture("2026-08-21T20:20:00+08:00")
    refs = {}
    for key, name, data in (
        ("calendar_ref", "history/calendar.json", value.receipt),
        ("raw_calendar_ref", "history/calendar.raw", value.raw_response_bytes),
    ):
        refs[key] = {"path": name, "sha256": _write(workspace / name, data)}
    seen, validations = [], []

    def component(stage):
        def run(ctx):
            seen.append((stage, ctx.target_date, ctx.historical_session_ref))
            return {**producers[stage](ctx), "write_performed": False}

        return run

    def checked(ref):
        result = validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=ref["path"],
            expected_receipt_sha256=ref["sha256"],
        )
        validations.append(result)
        return {"status": "NATIVE_CORE_VERIFIED_TEST_ONLY"}

    def forbidden(**kwargs):
        pytest.fail("historical input must bypass Calendar acquisition")

    args = dict(
        workspace_root=workspace,
        run_root=workspace / "data/private/cn_daily_maintenance",
        mode="execute",
        attempt_slot="2020",
        now=datetime(2026, 8, 22, 13, tzinfo=timezone.utc),
        components=daily.MaintenanceComponents(
            pit=component("PIT"),
            market=component("MARKET"),
            history=component("HISTORY"),
            fundamental=component("FUNDAMENTAL"),
            macro_release=component("MACRO_RELEASE"),
        ),
        close_authority=forbidden,
        core_completed=checked,
        _core_replay_completed=checked,
        _historical_calendar_input={
            "target_trade_date": "20260820",
            "previous_trade_date": "20260819",
            **refs,
        },
    )
    return args, seen, validations


def test_historical_dispatch_and_completed_replay(tmp_path):
    args, seen, validations = context(tmp_path)
    workspace = args["workspace_root"]
    original = {
        key: (workspace / ref["path"]).read_bytes()
        for key, ref in args["_historical_calendar_input"].items()
        if key.endswith("_ref")
    }
    result = daily.run_cn_daily_maintenance(**args)
    assert result["status"] == "COMPLETE" and result["target_date"] == "20260820"
    assert [row[0] for row in seen] == ["PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE"]
    assert all(day == "20260820" and ref is not None for _, day, ref in seen)
    assert validations[0]["historical_session"]["observed_at"] == "2026-08-21T12:20:00Z"
    assert validations[0]["historical_session"]["prospective"] is False
    core = json.loads(Path(result["core_completion_ref"]["path"]).read_bytes())
    attempt = Path(result["core_completion_ref"]["path"]).parent
    assert (
        json.loads((attempt / "started.json").read_bytes())["started_at"] == "2026-08-22T13:00:00Z"
    )
    assert core["schema_version"] == "cn-daily-maintenance-core.v2"
    assert "transport_retry" not in result
    assert not list(args["run_root"].rglob("close-request-*.json"))
    assert original == {
        key: (workspace / ref["path"]).read_bytes()
        for key, ref in args["_historical_calendar_input"].items()
        if key.endswith("_ref")
    }
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in args["run_root"].rglob("*")
        if p.is_file()
    }
    seen.clear()
    repeated = daily.run_cn_daily_maintenance(
        **{**args, "now": datetime(2026, 8, 23, 13, tzinfo=timezone.utc)}
    )
    assert repeated["logical_replay"] == "VERIFIED_SAME_INPUT"
    assert repeated["core_completion_ref"] == result["core_completion_ref"]
    assert seen == [] and len(validations) == 2
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in args["run_root"].rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "fault", ["fields", "sha", "future", "date", "mode", "callback", "missing"]
)
def test_bad_historical_input_does_not_create_run_tree(tmp_path, fault):
    args, seen, _ = context(tmp_path)
    args["run_root"] = args["workspace_root"] / "fresh-run"
    value = args["_historical_calendar_input"]
    if fault == "fields":
        value["ready"] = True
    elif fault == "sha":
        value["calendar_ref"]["sha256"] = "0" * 64
    elif fault == "future":
        args["now"] = datetime(2026, 8, 21, 12, tzinfo=timezone.utc)
    elif fault == "date":
        value["target_trade_date"] = "20260822"
    elif fault == "mode":
        args["mode"] = "shadow"
    elif fault == "missing":
        (args["workspace_root"] / value["raw_calendar_ref"]["path"]).unlink()
    else:
        args["_core_replay_completed"] = None
    with pytest.raises(daily.DailyMaintenanceError):
        daily.run_cn_daily_maintenance(**args)
    assert not args["run_root"].exists() and seen == []


def test_source_changed_under_lock_consumes_no_attempt(tmp_path, monkeypatch):
    args, seen, _ = context(tmp_path)
    original = daily._RunLock.__enter__

    def enter(lock):
        result = original(lock)
        path = (
            args["workspace_root"] / args["_historical_calendar_input"]["raw_calendar_ref"]["path"]
        )
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(daily._RunLock, "__enter__", enter)
    with pytest.raises(HistoricalSessionError, match="input_changed"):
        daily.run_cn_daily_maintenance(**args)
    assert not (args["run_root"] / "logical_tasks").exists() and seen == []


def test_historical_in_doubt_recovery_never_invokes_callbacks(tmp_path):
    args, seen, validations = context(tmp_path)
    journal = DailyOperationJournal(
        args["run_root"],
        args["workspace_root"],
        now=args["now"],
        slot="2020",
        mode="execute",
        _historical_trade_date="20260820",
    )
    attempt = args["run_root"] / "attempts/interrupted"
    _write(
        attempt / "started.json",
        {"state": "STARTED", "mode": "execute", "started_at": "2026-08-22T13:00:00Z"},
    )
    _write(attempt / "start-MARKET.json", {"state": "STAGE_STARTED"})
    journal.bind(attempt)
    result = daily.run_cn_daily_maintenance(**args)
    assert result["status"] == "IN_DOUBT"
    assert seen == validations == []
    assert journal.attempts() == [attempt]
    assert not (journal.path / "attempt-2.json").exists()


@pytest.mark.parametrize("fault", ["target", "core_path"])
def test_completed_historical_replay_binds_target_and_checkpoint(tmp_path, fault):
    args, seen, validations = context(tmp_path)
    result = daily.run_cn_daily_maintenance(**args)
    path = Path(result["core_completion_ref"]["path"]).with_name("attempt.json")
    value = json.loads(path.read_bytes())
    if fault == "target":
        value["target_date"] = "20260819"
    else:
        value["core_completion_ref"]["path"] = str(
            path.parent.parent / "elsewhere/core-completion.json"
        )
    _write(path, value)
    seen.clear()
    count = len(validations)
    with pytest.raises(daily.DailyMaintenanceError, match="REPLAY_BINDING_INVALID"):
        daily.run_cn_daily_maintenance(**args)
    assert seen == [] and len(validations) == count
