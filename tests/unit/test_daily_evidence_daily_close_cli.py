"""Public one-command action routing and 0/2/3 outputs; installed dispatch controlled."""

from contextlib import contextmanager
import json
import pytest
from quant_investor.cli.main import main
from quant_investor.cli import daily_production as cli
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from test_daily_evidence_execution_controls import fixture
from test_daily_evidence_morning_cli import output


def command(root, action):
    ref, put = fixture(root)
    request = json.loads((root / ref["path"]).read_bytes())
    request["action"] = action
    source = request["release_install_ref"]
    if action == "RESUME":
        request.update(recipe_ref=None, maintenance_handoff_ref=source)
    elif action == "CATCH_UP":
        request.update(
            recipe_ref=None,
            calendar_ref=source,
            raw_calendar_ref=source,
            previous_completion_ref={
                "path": "results/operations/daily_production/CN/20260903/completion.v1.json",
                "sha256": "c" * 64,
            },
        )
    ref = put("request.json", request)
    args = [
        "production",
        "daily-close",
        "--workspace-root",
        str(root),
        "--request",
        ref["path"],
        "--expected-request-sha256",
        ref["sha256"],
    ]
    group = [
        "--release-repository-root",
        str(root),
        "--release-install-input",
        source["path"],
        "--expected-release-install-input-sha256",
        source["sha256"],
    ]
    return args, group, request, ref


def test_catchup_already_at_target_survives_real_public_dispatch(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from quant_investor.contracts import canonical_json_bytes
    from test_daily_evidence_catchup import fixture as calendar_fixture, put
    from scripts import daily_catchup, daily_production

    args, group, request, ref = command(tmp_path, "CATCH_UP")
    calendar = calendar_fixture(tmp_path)
    day = calendar["previous_trade_date"]
    complete = {
        "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
        "sha256": "a" * 64,
    }
    request.update(
        target_trade_date=day,
        calendar_ref=calendar["calendar_ref"],
        raw_calendar_ref=calendar["raw_calendar_ref"],
        previous_completion_ref=complete,
        day_input_refs={},
    )
    ref = put(tmp_path, "request.json", canonical_json_bytes(request))
    args[-1] = ref["sha256"]
    calls = []

    def replay(**kwargs):
        calls.append(kwargs["trade_date"])
        return {
            "native_replay_validated": True,
            "completion_ref": kwargs["completion_ref"],
            "trade_date": kwargs["trade_date"],
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": False,
        }

    @contextmanager
    def bridge(**kwargs):
        yield {"daily_close": daily_production.dispatch_daily_request, "completion_replay": replay}

    monkeypatch.setattr(cli, "verified_native_context", bridge)
    monkeypatch.setattr(daily_catchup, "replay_native_completion", replay)
    # This legacy CLI fixture controls EOD admission and contains no native EOD
    # files; its serving admission is part of the same explicit boundary.
    from scripts import daily_dashboard_publication

    monkeypatch.setattr(
        daily_dashboard_publication,
        "complete_serving_result",
        lambda **kw: {"status": "COMPLETE", "completion_ref": kw["completion_ref"]},
    )
    monkeypatch.setattr(
        daily_catchup,
        "run_materialized_native_input",
        lambda **kw: pytest.fail("completed target invoked writer"),
    )
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    main(args + group)
    result = output(capsys)
    assert result["execution_state"] == "NO_ACTION" and result["business_state"] == "COMPLETE"
    assert result["days"][0]["completion_ref"] == complete
    assert calls == [day, day]
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "action,state",
    [
        ("PLAN", "PLANNED"),
        ("EXECUTE", "SUCCEEDED"),
        ("RESUME", "NO_ACTION"),
        ("CATCH_UP", "BLOCKED"),
    ],
)
def test_all_actions_share_one_cli_and_validate_exact_native_result(
    tmp_path, monkeypatch, capsys, action, state
):
    args, group, request, ref = command(tmp_path, action)
    day = request["target_trade_date"]
    business = (
        "NOT_EVALUATED" if action == "PLAN" else "INCOMPLETE" if state == "BLOCKED" else "COMPLETE"
    )
    complete = (
        {
            "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
            "sha256": "a" * 64,
        }
        if business == "COMPLETE"
        else None
    )
    result = {
        "schema_version": "cn-daily-production-result.v1",
        "action": action,
        "target_trade_date": day,
        "execution_state": state,
        "business_state": business,
        "days": [
            {
                "trade_date": day,
                "execution_state": state,
                "business_state": business,
                "completion_ref": complete,
            }
        ],
        "authority": dict(FALSE_AUTHORITY),
    }
    events = []

    def dispatch(**kw):
        events.append("dispatch")
        assert (
            kw["request_ref"] == ref and kw["release_install_ref"] == request["release_install_ref"]
        )
        assert "synthetic" not in kw
        return result

    def replay(**kw):
        events.append("replay")
        return {
            "native_replay_validated": True,
            "completion_ref": complete,
            "trade_date": day,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": False,
        }

    @contextmanager
    def bridge(**kw):
        events.append("enter")
        try:
            yield {"daily_close": dispatch, "completion_replay": replay}
        finally:
            events.append("cleanup")

    monkeypatch.setattr(cli, "verified_native_context", bridge)
    monkeypatch.setattr(cli, "plan_catchup", lambda **kw: {"ordered_trade_dates": [day]})
    if state == "BLOCKED":
        with pytest.raises(SystemExit) as caught:
            main(args + group)
        assert caught.value.code == 2
    else:
        main(args + group)
    assert output(capsys) == result
    assert events == ["enter", "dispatch"] + (["replay"] if complete else []) + ["cleanup"]


@pytest.mark.parametrize(
    "kind,code", [("business", 2), ("unexpected", 3), ("result", 3), ("runtime", 2)]
)
def test_daily_close_failures_have_exact_exit_and_cleanup(
    tmp_path, monkeypatch, capsys, kind, code
):
    args, group, _, _ = command(tmp_path, "EXECUTE")
    events = []

    def dispatch(**kw):
        if kind == "business":
            raise ContractError("private failure details")
        if kind == "unexpected":
            raise RuntimeError("private traceback")
        return {"success": True}

    @contextmanager
    def bridge(**kw):
        if kind == "runtime":
            raise ContractError("bad runtime")
        try:
            yield {"daily_close": dispatch}
        finally:
            events.append("cleanup")

    monkeypatch.setattr(cli, "verified_native_context", bridge)
    with pytest.raises(SystemExit) as caught:
        main(args + group)
    assert caught.value.code == code
    value = output(capsys)
    assert set(value) == {"status", "blocker_code"}
    assert value["status"] == ("BLOCKED" if code == 2 else "ERROR")
    assert "private" not in repr(value)
    assert events == ([] if kind == "runtime" else ["cleanup"])


def test_daily_close_requires_runtime_group(tmp_path, capsys):
    args, _, _, _ = command(tmp_path, "PLAN")
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert output(capsys)["blocker_code"] == "DAILY_PRODUCTION_RELEASE_ARGUMENTS_REQUIRED"
