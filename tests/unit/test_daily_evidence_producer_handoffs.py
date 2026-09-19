"""Native maintenance receipts and Factor reports retain exact recovery refs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest
from quant_investor.market import daily_maintenance as maintenance
from quant_investor.market.daily_factor_loop import DailyFactorLoop
from quant_investor.market.close_session_authority import CloseSessionAuthorityResult
from test_daily_factor_loop_recovery import _components


def inventory(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("fault", [None, "claim", "binding", "gap", "core_ref", "during_use"])
def test_native_same_task_replay_retains_claim_attempt_and_core_without_new_budget(tmp_path, fault):
    workspace, _, close, raw, callbacks = _components(tmp_path)
    components = maintenance.MaintenanceComponents(
        pit=callbacks["PIT"],
        market=callbacks["MARKET"],
        history=callbacks["HISTORY"],
        fundamental=callbacks["FUNDAMENTAL"],
        macro_release=lambda ctx: {"status": "UNCONFIRMED", "blockers": ["SYNTHETIC_AUXILIARY"]},
    )
    run_root = workspace / "data/private/cn_daily_maintenance"
    calls = []

    def authority(**kw):
        calls.append("calendar")
        return CloseSessionAuthorityResult(close, raw)

    kwargs = dict(
        workspace_root=workspace,
        run_root=run_root,
        mode="execute",
        attempt_slot="2020",
        now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        components=components,
    )
    first = maintenance.run_cn_daily_maintenance(**kwargs, close_authority=authority)
    assert first["factor_input_readiness"] == "READY"
    sealed = json.loads(Path(first["attempt_receipt_ref"]["path"]).read_text())
    claim = first["logical_claim_ref"]
    assert sealed["logical_claim_ref"] == claim
    checkpoint = json.loads(Path(first["core_completion_ref"]["path"]).read_text())
    assert checkpoint["logical_claim_ref"] == claim
    assert hashlib.sha256(Path(claim["path"]).read_bytes()).hexdigest() == claim["sha256"]
    before = inventory(run_root)

    def forbidden(**kw):
        raise AssertionError("replay requested Calendar again")

    replay_callbacks = []

    def replay_core(ref):
        assert ref == first["core_completion_ref"]
        replay_callbacks.append(ref)
        return {"status": "NO_ACTION", "fixture": "existing-only callback selected"}

    replay = maintenance.run_cn_daily_maintenance(
        **kwargs,
        close_authority=forbidden,
        core_completed=forbidden,
        _core_replay_completed=replay_core,
    )
    assert replay["logical_replay"] == "VERIFIED_SAME_INPUT"
    assert replay["provider_calls_this_attempt"] is False
    for key in ["logical_claim_ref", "attempt_receipt_ref", "core_completion_ref"]:
        assert replay[key] == first[key]
    assert calls == ["calendar"]
    assert replay_callbacks == [first["core_completion_ref"]]
    assert inventory(run_root) == before

    from quant_investor.operations.maintenance_readback import locked_finalized_maintenance_replay
    from quant_investor.operations.daily_contract import ContractError

    final_path = Path(first["attempt_receipt_ref"]["path"])
    binding_path = Path(claim["path"]).parent / "attempt-1.json"

    def change(path, mutate):
        value = json.loads(path.read_bytes())
        mutate(value)
        path.write_bytes(maintenance._canonical_json_bytes(value))

    if fault == "claim":
        change(Path(claim["path"]), lambda x: x.update(attempt_budget=99))
    elif fault == "binding":
        change(binding_path, lambda x: x["claim_ref"].update(sha256="a" * 64))
    elif fault == "gap":
        binding_path.rename(binding_path.with_name("attempt-2.json"))
    elif fault == "core_ref":
        change(final_path, lambda x: x["core_completion_ref"].update(sha256="a" * 64))
    before_read = inventory(run_root)

    def inspect():
        nonlocal before_read
        with locked_finalized_maintenance_replay(
            workspace=str(workspace),
            run_root=str(run_root.relative_to(workspace)),
            run_date="20260820",
        ) as recovered:
            assert recovered["logical_replay"] == "VERIFIED_SAME_INPUT"
            assert recovered["attempt_receipt_ref"] == first["attempt_receipt_ref"]
            assert recovered["core_completion_ref"] == first["core_completion_ref"]
            assert recovered["provider_calls_this_attempt"] is False
            if fault == "during_use":
                change(final_path, lambda x: x.update(test_mutation=True))
                before_read = inventory(run_root)

    if fault:
        with pytest.raises((ContractError, maintenance.DailyMaintenanceError)):
            inspect()
    else:
        inspect()
    assert inventory(run_root) == before_read


@pytest.mark.parametrize("stage", ["top100_publication", "core_handoff_recovery"])
def test_factor_report_returns_and_seals_exact_core_handoff(tmp_path, stage):
    loop = object.__new__(DailyFactorLoop)
    loop.workspace = tmp_path
    loop.run_root = tmp_path / "run"
    loop.run_root.mkdir(mode=0o700)
    loop.started_at = "2026-09-08T12:20:00Z"
    loop.installation = {"state": "PASS"}
    ref = {
        "path": "results/operations/daily_production/CN/20260908/core-handoff.v1.json",
        "sha256": "a" * 64,
    }
    loop.stages = {stage: {"status": "SUCCEEDED", "core_handoff_ref": ref}}
    result = loop.report(maintenance={"status": "IN_PROGRESS", "target_date": "20260908"})
    document = json.loads(Path(result["report_ref"]["path"]).read_text())
    assert result["core_handoff_ref"] == document["core_handoff_ref"] == ref
    loop.stages = {}
    empty = loop.report(maintenance=None)
    assert "core_handoff_ref" not in empty


@pytest.mark.parametrize("has_core", [True, False])
def test_early_handoff_hook_precedes_independent_settlement(tmp_path, has_core):
    loop = object.__new__(DailyFactorLoop)
    ref = {"path": "core-handoff.json", "sha256": "a" * 64}
    loop.stages = {"top100_publication": {"core_handoff_ref": ref} if has_core else {}}
    events = []
    loop._save_state = lambda state: events.append("saved")

    def hook(state):
        assert state["core_handoff_ref"] == ref
        events.append("handoff")
        return {"status": "SEALED"}

    loop._core_handoff_completed = hook
    loop._settle = lambda ref: (events.append("settle") or {})
    loop._handoff_before_settlement({}, {"path": "calendar.json", "sha256": "b" * 64})
    assert events == (["saved", "handoff", "settle"] if has_core else ["saved", "settle"])
