"""Committed recovery uses native C1 accounting and original synthetic DAG inputs."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from _public_catchup_fixture import put
from test_registered_close_native import prepared_case, inventory
from scripts import daily_registered_recovery as recovery
from scripts import cn_official_close_batch as native
from quant_investor.operations.daily_contract import ContractError


def chronology():
    return {
        "handoff": {"sealed_at": "2026-08-28T13:20:00Z"},
        "plan": {"transaction_planned_at": "2026-08-28T13:21:00Z"},
        "cutoff": {
            "physical_reads_completed_at": "2026-08-28T13:29:00.250000Z",
            "portfolio_created_at": "2026-08-28T13:30:00Z",
            "as_of": "2026-08-28T13:30:00Z",
            "sealed_at": "2026-08-28T13:30:01Z",
        },
        "materialization": {"sealed_at": "2026-08-28T13:30:02Z"},
        "completion": {"cas_observed_at": "2026-08-28T13:31:00Z"},
    }


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "handoff",
        "plan",
        "physical",
        "portfolio",
        "as_of",
        "cutoff_seal",
        "materialization",
        "cas_early",
        "cas_future",
    ],
)
def test_original_custody_time_chain(fault, monkeypatch):
    values = chronology()
    monkeypatch.setattr(recovery, "_now", lambda: datetime(2026, 8, 29, tzinfo=timezone.utc))
    if fault is None:
        assert (
            recovery.validate_recovery_chronology(**values)
            == values["materialization"]["sealed_at"]
        )
        return
    paths = {
        "handoff": ("handoff", "sealed_at"),
        "plan": ("plan", "transaction_planned_at"),
        "physical": ("cutoff", "physical_reads_completed_at"),
        "portfolio": ("cutoff", "portfolio_created_at"),
        "as_of": ("cutoff", "as_of"),
        "cutoff_seal": ("cutoff", "sealed_at"),
        "materialization": ("materialization", "sealed_at"),
        "cas_early": ("completion", "cas_observed_at"),
        "cas_future": ("completion", "cas_observed_at"),
    }
    group, field = paths[fault]
    values[group][field] = (
        "2026-08-30T00:00:00Z" if fault == "cas_future" else "2026-08-28T13:35:00Z"
    )
    if fault == "cas_early":
        values[group][field] = "2026-08-28T13:30:01Z"
    with pytest.raises(ContractError, match="RECOVERY_.*(TIME|OBSERVATION)_INVALID"):
        recovery.validate_recovery_chronology(**values)


def crash_after_cas(adapter, monkeypatch):
    publish = native.publish_catalog

    def crash(*args, **kwargs):
        publish(*args, **kwargs)
        raise RuntimeError("synthetic interruption immediately after CAS")

    with monkeypatch.context() as patch:
        patch.setattr(native, "publish_catalog", crash)
        with pytest.raises(RuntimeError, match="immediately after CAS"):
            adapter.execute(adapter.template())
    monkeypatch.setattr(
        native, "publish_catalog", lambda *a, **kw: pytest.fail("second financial CAS")
    )


def test_native_metadata_guard_runs_before_any_write_and_preserves_existing_completion(
    tmp_path, monkeypatch
):
    _, _, _, adapter = prepared_case(tmp_path, monkeypatch)
    crash_after_cas(adapter, monkeypatch)
    before = inventory(tmp_path)
    calls = []

    def deny(**kwargs):
        calls.append(kwargs)
        assert kwargs["proof"]["completion"] is None
        assert kwargs["completion"]["status"] == "RECOVERED_AFTER_CAS"
        raise ContractError("original package unavailable")

    with (
        recovery._operation_lock(str(adapter.args["record_root"])),
        pytest.raises(ContractError, match="original package"),
    ):
        native.recover_close_completion(**adapter._proof_args(), _metadata_guard=deny)
    assert len(calls) == 1 and inventory(tmp_path) == before
    with recovery._operation_lock(str(adapter.args["record_root"])):
        result = native.recover_close_completion(
            **adapter._proof_args(), _metadata_guard=lambda **kw: calls.append(kw)
        )
    assert result["metadata_write_performed"] and result["catalog_cas_performed"] is False
    before = inventory(tmp_path)
    with recovery._operation_lock(str(adapter.args["record_root"])):
        repeated = native.recover_close_completion(
            **adapter._proof_args(), _metadata_guard=lambda **kw: calls.append(kw)
        )
    assert not repeated["metadata_write_performed"]
    assert repeated["completion"] == result["completion"]
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["changed_source", "wrong_native_commit", "clock_regression"])
def test_dag_guard_rechecks_custody_before_native_metadata_write(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from quant_investor.operations.daily_preparation import Sources
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage

    _, _, _, adapter = prepared_case(tmp_path, monkeypatch)
    crash_after_cas(adapter, monkeypatch)
    protected = put(tmp_path, "original-package.json", {"synthetic": "original custody"})
    source = Sources(str(tmp_path))
    source.raw(protected)
    request = {"target_trade_date": "20260825"}
    stamp = adapter.plan["transaction_planned_at"]
    recovered = {
        "handoff_ref": protected,
        "handoff": {"sealed_at": stamp},
        "request": request,
        "recipe": {},
    }
    proof = native.inspect_close_commit(**adapter._proof_args())
    scope = {
        "source": source,
        "configured": None,
        "recovered": recovered,
        "plan": adapter.plan,
        "proof": proof,
        "adapter": adapter,
        "metadata_complete": False,
        "cutoff": {
            "receipt": {
                key: stamp
                for key in (
                    "physical_reads_completed_at",
                    "portfolio_created_at",
                    "as_of",
                    "sealed_at",
                )
            }
        },
        "record": {"sealed_at": stamp},
        "materialized": SimpleNamespace(native_inputs_ref=protected),
    }
    # Isolate the final guard under real strategy/day/Store locks and native C1
    # recovery. Complete package admission is exercised by the full native case.
    monkeypatch.setattr(recovery, "_original_inputs", lambda *a: (source, request, {}))
    monkeypatch.setattr(recovery, "inspect_registered_recovery", lambda **kw: scope)
    monkeypatch.setattr(recovery, "read_maintenance_handoff", lambda **kw: recovered)
    from scripts import daily_completion

    monkeypatch.setattr(
        daily_completion,
        "run_materialized_native_input",
        lambda **kw: pytest.fail("unverified DAG resumed"),
    )
    original = native.recover_close_completion

    def inject(**kwargs):
        if fault == "changed_source":
            put(tmp_path, protected["path"], {"synthetic": "source changed after inspection"})
        return original(**kwargs)

    monkeypatch.setattr(native, "recover_close_completion", inject)
    if fault == "wrong_native_commit":
        scope["proof"] = {**proof, "pointer_sha256": "a" * 64}
    elif fault == "clock_regression":
        monkeypatch.setattr(recovery, "_now", lambda: datetime(2000, 1, 1, tzinfo=timezone.utc))
    with (
        AutomaticRunStorage(str(tmp_path)).locked(),
        DailyJournal(str(tmp_path), "20260825").locked(),
    ):
        pass
    before = inventory(tmp_path)
    with pytest.raises(ContractError):
        recovery.recover_registered_committed(
            workspace=str(tmp_path),
            request_ref=protected,
            release_install_ref=protected,
            synthetic=True,
        )
    after = inventory(tmp_path)
    assert set(after) == set(before)
    assert all(
        after[p] == value
        for p, value in before.items()
        if fault != "changed_source" or p != protected["path"]
    )


def test_registered_dag_recovery_requires_original_package_and_never_repeats_financial_cas(
    tmp_path, monkeypatch
):
    from _native_cutoff_sources_fixture import build
    from test_registered_cutoff_materialization import clock
    from scripts import daily_materialization, daily_completion
    from scripts.daily_native_inputs import load_native_inputs
    from scripts.daily_production_store_adapter import StoreCloseAdapter

    case = build(tmp_path, monkeypatch, registered_buy=True)
    clock(monkeypatch)
    workspace, journal, recovered = case["workspace"], case["journal"], case["recovered"]
    request = json.loads((workspace / "cutoff-controls/request.json").read_bytes())
    request_ref = {
        "path": "cutoff-controls/request.json",
        "sha256": recovered["handoff"]["request_ref"]["sha256"],
    }
    recovered["request"] = request
    # Core/Factor/install validation remains the existing fixture's bounded seam.
    # Cutoff, materialization, native financial/declaration and Store proof are real.
    monkeypatch.setattr(recovery, "read_maintenance_handoff", lambda **kw: deepcopy(recovered))
    with journal.locked():
        materialized = daily_materialization.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    day, inputs = load_native_inputs(
        workspace=str(workspace), input_ref=materialized.native_inputs_ref
    )
    adapter = StoreCloseAdapter(
        arguments=inputs.store_arguments,
        trade_date=day,
        plan_ref=inputs.store_plan_ref,
        release_ref=inputs.release_ref,
    )
    crash_after_cas(adapter, monkeypatch)
    kwargs = dict(
        workspace=str(workspace),
        request_ref=request_ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )
    called = []
    monkeypatch.setattr(
        daily_completion,
        "run_materialized_native_input",
        lambda **kw: called.append(kw) or {"status": "PARTIAL", "completion_ref": None},
    )
    # Missing exact materialization must prevent even committed-pointer metadata.
    path = workspace / materialized.materialization_ref["path"]
    original = path.read_bytes()
    path.unlink()
    # Establish lock files before measuring forbidden writes; these are coordination
    # metadata, not a repaired Store completion or a newly manufactured package.
    from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage

    with AutomaticRunStorage(str(workspace)).locked():
        pass
    before = inventory(workspace)
    with pytest.raises(ContractError, match="MATERIALIZATION_MISSING"):
        recovery.recover_registered_committed(**kwargs)
    assert inventory(workspace) == before and not called
    put(workspace, materialized.materialization_ref["path"], original)
    # Re-read on the next calendar day; all original package bytes stay unchanged.
    before = inventory(workspace)
    inspected = recovery.inspect_registered_recovery(**kwargs)
    assert not inspected["metadata_complete"]
    assert inventory(workspace) == before
    result = recovery.recover_registered_committed(**kwargs)
    assert result["status"] == "PARTIAL" and len(called) == 1
    assert called[0]["input_ref"] == materialized.native_inputs_ref and called[0]["resume"]
    after = inventory(workspace)
    added = set(after) - set(before)
    assert {PathName.rsplit("/", 1)[-1] for PathName in added} == {
        "committed-pointer.v1.json",
        "completion.v2.json",
    }
    assert all(after[p] == value for p, value in before.items())
    assert adapter.probe(adapter.template()).outcome.state.value == "SUCCEEDED"
    completion = native.inspect_frozen_close_commit(**adapter._proof_args())["completion"]
    # Completion records the actual recovery observation, never the historical T.
    assert completion["cas_observed_at"][:10] != "2026-08-28"
    before = inventory(workspace)
    assert recovery.recover_registered_committed(**kwargs)["status"] == "PARTIAL"
    assert inventory(workspace) == before and len(called) == 2
