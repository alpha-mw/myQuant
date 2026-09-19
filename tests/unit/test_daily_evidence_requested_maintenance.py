"""Native requested-date boundary stops before business callbacks and preserves Calendar."""

from datetime import datetime
from pathlib import Path
import json
import pytest
from quant_investor.market.daily_maintenance import (
    run_cn_daily_maintenance,
    MaintenanceComponents,
    DailyMaintenanceError,
)
from test_daily_evidence_requested_session import capture


@pytest.mark.parametrize("fault", [None, "wrong_observation", "veto"])
def test_closed_native_attempt_never_calls_business_components(tmp_path, fault):
    now = "2026-09-05T20:20:00+08:00"
    result = capture("2026-09-04T20:20:00+08:00" if fault == "wrong_observation" else now)
    calls = []

    def authority(**kwargs):
        calls.append("calendar")
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("requested closed/mismatched date reached business component")

    run_root = tmp_path / "data/private/cn_daily_maintenance"
    if fault == "veto":
        run_root.mkdir(parents=True, mode=0o700)
        veto = run_root / "WRITE_VETO.json"
        veto.write_text('{"fixture":"active veto"}')
        veto.chmod(0o600)
    kwargs = dict(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="2020",
        now=datetime.fromisoformat(now),
        _expected_target_trade_date="20260905",
        close_authority=authority,
        core_completed=forbidden,
        components=MaintenanceComponents(
            pit=forbidden,
            market=forbidden,
            history=forbidden,
            fundamental=forbidden,
            macro_release=forbidden,
            system_status=forbidden,
        ),
    )
    value = run_cn_daily_maintenance(**kwargs)
    assert value["canonical_unchanged"] is True
    assert value["stage_results"] == []
    assert "core_completion_ref" not in value
    if fault == "veto":
        assert value["status"] == "WRITE_VETO_ACTIVE"
        assert calls == []
    else:
        assert calls == ["calendar"]
        assert value["status"] == ("BLOCKED" if fault else "NO_ACTION")
        assert value["execution_disposition"] == (
            "REQUESTED_SESSION_BLOCKED" if fault else "NON_TRADING_DAY"
        )
        ref = value["close_session_receipt_ref"]
        assert json.loads(Path(ref["path"]).read_bytes())["target_trade_date"] == "20260904"
        assert not (run_root / "WRITE_VETO.json").exists()
    assert not (tmp_path / "results/operations/daily_production").exists()
    assert not (tmp_path / "results/prospective").exists()
    if fault is None:
        from quant_investor.operations.maintenance_readback import (
            locked_finalized_maintenance_replay,
        )

        before = {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in run_root.rglob("*")
            if p.is_file()
        }
        replayed = run_cn_daily_maintenance(**{**kwargs, "close_authority": forbidden})
        assert replayed["requested_session_result"]["classification"] == "CONFIRMED_CLOSED"
        assert replayed["attempt_receipt_ref"] == value["attempt_receipt_ref"]
        assert replayed["logical_claim_ref"] == value["logical_claim_ref"]
        assert replayed["provider_calls_this_attempt"] is False
        with locked_finalized_maintenance_replay(
            workspace=str(tmp_path),
            run_root=str(run_root.relative_to(tmp_path)),
            run_date="20260905",
        ) as retained:
            assert retained["requested_session_result"] == replayed["requested_session_result"]
        assert {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in run_root.rglob("*")
            if p.is_file()
        } == before
        veto = run_root / "WRITE_VETO.json"
        veto.write_text('{"fixture":"veto introduced after closed attempt"}')
        veto.chmod(0o600)
        before = {str(p): p.read_bytes() for p in run_root.rglob("*") if p.is_file()}
        with pytest.raises(DailyMaintenanceError, match="WRITE_VETO_ACTIVE"):
            run_cn_daily_maintenance(**{**kwargs, "close_authority": forbidden})
        assert {str(p): p.read_bytes() for p in run_root.rglob("*") if p.is_file()} == before


def test_wrong_logical_date_is_rejected_before_claim_or_provider(tmp_path):
    root = tmp_path / "maintenance"
    with pytest.raises(DailyMaintenanceError, match="REQUESTED_SESSION_RUN_DATE_MISMATCH"):
        run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=root,
            mode="execute",
            now=datetime.fromisoformat("2026-09-05T20:20:00+08:00"),
            _expected_target_trade_date="20260904",
        )
    assert not root.exists()


def test_same_install_ordinary_receipt_closed_projection_preserves_history(tmp_path):
    from test_daily_factor_loop_recovery import _components
    from quant_investor.operations.maintenance_readback import locked_finalized_maintenance_replay

    workspace, _, _, _, callbacks = _components(tmp_path)
    now = "2026-08-21T20:20:00+08:00"
    calendar = capture(now, holidays=("20260821",))
    run_root = workspace / "data/private/cn_daily_maintenance"
    components = MaintenanceComponents(
        pit=callbacks["PIT"],
        market=callbacks["MARKET"],
        history=callbacks["HISTORY"],
        fundamental=callbacks["FUNDAMENTAL"],
        macro_release=lambda ctx: {"status": "BLOCKED", "blockers": ["SYNTHETIC_AUXILIARY"]},
    )
    kwargs = dict(
        workspace_root=workspace,
        run_root=run_root,
        mode="execute",
        attempt_slot="2020",
        now=datetime.fromisoformat(now),
        components=components,
    )
    original = run_cn_daily_maintenance(**kwargs, close_authority=lambda **kw: calendar)
    assert original["factor_input_readiness"] == "READY"
    assert original["target_date"] == "20260820"
    before = {str(p): p.read_bytes() for p in run_root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary closed projection attempted provider or Factor callback")

    projected = run_cn_daily_maintenance(
        **kwargs,
        _expected_target_trade_date="20260821",
        close_authority=forbidden,
        core_completed=forbidden,
    )
    assert projected["requested_session_result"]["classification"] == "CONFIRMED_CLOSED"
    for key in (
        "target_date",
        "status",
        "canonical_unchanged",
        "canonical_write_count",
        "core_completion_ref",
        "attempt_receipt_ref",
    ):
        assert projected[key] == original[key]
    with locked_finalized_maintenance_replay(
        workspace=str(workspace), run_root=str(run_root.relative_to(workspace)), run_date="20260821"
    ) as replay:
        assert replay["target_date"] == "20260820"
        assert replay["requested_session_result"] == projected["requested_session_result"]
    assert {str(p): p.read_bytes() for p in run_root.rglob("*") if p.is_file()} == before


def test_unfinished_replay_is_not_classified_or_sent_to_core(tmp_path, monkeypatch):
    from quant_investor.market.maintenance_journal import DailyOperationJournal

    pending = {"status": "IN_DOUBT", "provider_calls_this_attempt": False}
    monkeypatch.setattr(DailyOperationJournal, "recover_or_replay", lambda self: pending)

    def forbidden(*args, **kwargs):
        pytest.fail("unfinished replay reached provider/core")

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=tmp_path / "run",
        mode="execute",
        attempt_slot="2020",
        now=datetime.fromisoformat("2026-09-05T20:20:00+08:00"),
        _expected_target_trade_date="20260905",
        close_authority=forbidden,
        core_completed=forbidden,
        _core_replay_completed=forbidden,
    )
    assert {key: result[key] for key in pending} == pending
    assert result["provider_calls"] is False and result["provider_request_attempts"] == {}
    assert "requested_session_result" not in result and "factor_loop" not in result


def test_current_predecessor_mismatch_seals_failure_before_business(tmp_path):
    calendar = capture("2026-09-04T20:20:00+08:00")

    def forbidden(*args, **kwargs):
        pytest.fail("wrong predecessor reached a business callback")

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=tmp_path / "run",
        mode="execute",
        attempt_slot="2020",
        now=datetime.fromisoformat("2026-09-04T20:20:00+08:00"),
        _expected_target_trade_date="20260904",
        _expected_previous_trade_date="20260902",
        close_authority=lambda **kw: calendar,
        core_completed=forbidden,
        components=MaintenanceComponents(
            pit=forbidden,
            market=forbidden,
            history=forbidden,
            fundamental=forbidden,
            macro_release=forbidden,
            system_status=forbidden,
        ),
    )
    assert result["status"] == "BLOCKED"
    assert result["execution_disposition"] == "REQUESTED_SESSION_BLOCKED"
    assert result["blockers"] == ["REQUESTED_PREDECESSOR_NOT_ADJACENT"]
    assert result["canonical_unchanged"] and result["canonical_write_count"] == 0
    assert result["stage_results"] == [] and "core_completion_ref" not in result
    assert result["request_previous_trade_date"] == "20260902"
    assert Path(result["attempt_receipt_ref"]["path"]).is_file()


@pytest.mark.parametrize("previous", ["bad", "20260904", "20260905", 20260903])
def test_current_predecessor_bad_argument_creates_no_claim(tmp_path, previous):
    with pytest.raises(DailyMaintenanceError, match="REQUESTED_PREDECESSOR_ARGUMENT_INVALID"):
        run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=tmp_path / "run",
            mode="execute",
            now=datetime.fromisoformat("2026-09-04T20:20:00+08:00"),
            _expected_target_trade_date="20260904",
            _expected_previous_trade_date=previous,
        )
    assert not (tmp_path / "run").exists()


def test_current_predecessor_native_success_and_finalized_replay_guard(tmp_path):
    from test_daily_factor_loop_recovery import _components
    from quant_investor.market.close_session_authority import CloseSessionAuthorityError
    from quant_investor.operations.maintenance_readback import locked_finalized_maintenance_replay

    workspace, _, _, _, callbacks = _components(tmp_path)
    calendar = capture("2026-08-20T20:20:00+08:00")
    run_root = workspace / "data/private/cn_daily_maintenance"
    calls = []

    def component(stage):
        def run(ctx):
            calls.append(stage)
            return callbacks[stage](ctx)

        return run

    kwargs = dict(
        workspace_root=workspace,
        run_root=run_root,
        mode="execute",
        attempt_slot="2020",
        now=datetime.fromisoformat("2026-08-20T20:20:00+08:00"),
        _expected_target_trade_date="20260820",
        _expected_previous_trade_date="20260819",
        components=MaintenanceComponents(
            pit=component("PIT"),
            market=component("MARKET"),
            history=component("HISTORY"),
            fundamental=component("FUNDAMENTAL"),
            macro_release=lambda ctx: {"status": "BLOCKED", "blockers": ["SYNTHETIC_AUXILIARY"]},
        ),
    )
    original = run_cn_daily_maintenance(**kwargs, close_authority=lambda **kw: calendar)
    assert calls == ["PIT", "MARKET", "HISTORY", "FUNDAMENTAL"]
    assert original["factor_input_readiness"] == "READY"
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in workspace.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("finalized predecessor mismatch reached provider/core")

    with pytest.raises(CloseSessionAuthorityError, match="REQUESTED_PREDECESSOR_NOT_ADJACENT"):
        run_cn_daily_maintenance(
            **{**kwargs, "_expected_previous_trade_date": "20260818"},
            close_authority=forbidden,
            core_completed=forbidden,
            _core_replay_completed=forbidden,
        )
    with pytest.raises(CloseSessionAuthorityError, match="REQUESTED_PREDECESSOR_NOT_ADJACENT"):
        with locked_finalized_maintenance_replay(
            workspace=str(workspace),
            run_root=str(run_root.relative_to(workspace)),
            run_date="20260820",
            expected_previous_trade_date="20260818",
        ):
            pytest.fail("wrong predecessor yielded a finalized maintenance core")
    with locked_finalized_maintenance_replay(
        workspace=str(workspace),
        run_root=str(run_root.relative_to(workspace)),
        run_date="20260820",
        expected_previous_trade_date="20260819",
    ) as replay:
        assert replay["core_completion_ref"] == original["core_completion_ref"]
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in workspace.rglob("*") if p.is_file()
    }
