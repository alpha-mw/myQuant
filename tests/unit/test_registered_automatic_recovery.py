"""Real automatic/origin storage and registered source custody; Core/CAS/EOD seams explicit."""

from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from _daily_preparation_fixture import put, snapshot
from _registered_event_fixture import NOW
from _public_catchup_fixture import completion
from test_registered_automatic_origin import fixture as original_fixture, active, publish
from test_daily_evidence_catchup_routing import native_inputs
from quant_investor.operations import (
    catchup_binding,
    completion_readback,
    automatic_catchup_closure,
)
from quant_investor.operations.automatic_catchup_contract import AutomaticCatchupError, run_path
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage, PENDING
from quant_investor.operations.daily_contract import ContractError
from scripts import (
    daily_automatic_catchup as auto,
    daily_catchup as batch,
    daily_launch_inspection as launch,
)
from scripts import daily_completion_replay, daily_materialization, daily_production
from scripts import daily_registered_recovery as committed, daily_dashboard_publication as serving


def fixture(root, monkeypatch, *, started=True, complete=True, publication="expired"):
    case = original_fixture(root, monkeypatch)
    with active(case):
        origin_ref = publish(root, case)
    put(root, automatic_catchup_closure.RUN_ROOT + "/.daily-maintenance.lock", b"")
    if started:
        put(
            root,
            automatic_catchup_closure.RUN_ROOT + "/logical_tasks/20260825-2020-execute/claim.json",
            {"synthetic": "Core/CAS proof seam"},
        )
    clock = [NOW + timedelta(days=1)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)

    monkeypatch.setattr(auto, "datetime", Clock)
    monkeypatch.setattr(automatic_catchup_closure, "datetime", Clock)
    monkeypatch.setattr(committed, "registered_commit_required", lambda **kw: started)
    checks, executions, records, natives = [], [], {}, {}

    def inspect(**kwargs):
        checks.append(kwargs)
        assert kwargs["automatic_origin_ref"] == origin_ref
        assert kwargs["request_ref"] == case["origin"]["execution_request_ref"]
        return {"synthetic": "full original package/CAS admission seam"}

    monkeypatch.setattr(committed, "inspect_registered_recovery", inspect)
    monkeypatch.setattr(
        batch, "replay_native_completion", daily_completion_replay.replay_native_completion
    )
    monkeypatch.setattr(catchup_binding, "verify_completed_edge", lambda **kw: natives[kw["day"]])
    monkeypatch.setattr(
        batch, "inspect_recorded_completion", lambda **kw: records[kw["trade_date"]]
    )
    monkeypatch.setattr(
        completion_readback, "inspect_recorded_completion", lambda **kw: records[kw["trade_date"]]
    )
    root_request = case["resolution"]["derived_request"]

    def seal():
        day = "20260825"
        native = native_inputs(root_request, day=day, previous="20260824", current=True)
        native.update(
            schema_version="cn-daily-native-inputs.v7",
            cutoff_ref=root_request["release_install_ref"],
            registered_event_declaration_ref=case["derived"]["recipe"][
                "registered_event_declaration_ref"
            ],
        )
        natives[day] = native
        ref = completion(root, day, root_request)
        eod = json.loads((root / ref["path"]).read_bytes())
        eod.update(
            synthetic=True,
            native_inputs_ref=put(root, "synthetic-native/20260825.json", native),
            native_validation_completed_at=clock[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        ref = put(root, ref["path"], eod)
        documents = {
            "handoff": {
                "request_ref": case["origin"]["execution_request_ref"],
                "automatic_origin_ref": origin_ref,
            },
            "recipe": case["derived"]["recipe"],
        }
        records[day] = {
            "recorded_completion": eod,
            "completed_handoff_snapshot": SimpleNamespace(
                document=documents.__getitem__, recheck=lambda: None
            ),
        }
        return ref

    def execute(**kwargs):
        executions.append(kwargs)
        assert kwargs["committed_recovery_only"] is True
        assert kwargs["_automatic_origin_ref"] == origin_ref
        assert kwargs["request_ref"] == case["origin"]["execution_request_ref"]
        if complete:
            ref = seal()
            return {"status": "PARTIAL", "completion_ref": None, "sealed_evidence_ref": ref}
        return {"status": "PARTIAL", "completion_ref": None}

    monkeypatch.setattr(daily_materialization, "execute_daily_recipe", execute)

    def observed(workspace, day):
        if publication == "unknown":
            return None
        eod = records[day]["recorded_completion"]
        from quant_investor.operations.automatic_catchup_contract import document_ref

        ref = document_ref(f"results/operations/daily_production/CN/{day}/completion.v1.json", eod)
        return {
            "sealed_evidence_ref": ref,
            "validation_scope": "RECORDED_SERVING_BYTES_ONLY",
            "publication_state": (
                "EVIDENCE_SEALED_PUBLICATION_EXPIRED"
                if publication == "expired"
                else "EVIDENCE_SEALED_PUBLICATION_PENDING"
            ),
        }

    monkeypatch.setattr(serving, "observed_serving_status", observed)
    monkeypatch.setattr(
        serving, "complete_serving_result", lambda **kw: pytest.fail("expired evidence published")
    )
    case.update(
        origin_ref=origin_ref,
        clock=clock,
        checks=checks,
        executions=executions,
        records=records,
        seal=seal,
    )
    return case


def run(root, case, **kwargs):
    return daily_production.dispatch_daily_request(
        workspace=str(root),
        request_ref=case["current"]["auto_request_ref"],
        release_install_ref=case["config"]["release_install_ref"],
        synthetic=True,
        **kwargs,
    )


def inspect(root, case):
    return launch.inspect_daily_launch(
        workspace=str(root),
        request_ref=case["current"]["auto_request_ref"],
        release_install_ref=case["config"]["release_install_ref"],
        synthetic=True,
    )


def test_crossday_recovery_uses_original_scope_then_retires_only_completed_expired_eod(
    tmp_path, monkeypatch
):
    import test_registered_automatic_origin as origin_fixture

    original_sources = origin_fixture.source_fixture

    def canonical_dashboard_sources(root, patch):
        data = original_sources(root, patch)
        data["config"]["risk_free_ref"] = {
            **data["config"]["risk_free_ref"],
            "path": "portfolio_dashboard/inputs/cn_govt_bond_yield.csv",
        }
        data["config_ref"] = put(root, data["config_ref"]["path"], data["config"])
        return data

    monkeypatch.setattr(origin_fixture, "source_fixture", canonical_dashboard_sources)
    case = fixture(tmp_path, monkeypatch)
    # Mutated heads cannot be treated as fresh inputs; frozen original custody remains.
    risk_free = case["derived"]["recipe"]["dashboard_sources"]["risk_free_ref"]
    (tmp_path / risk_free["path"]).chmod(0o644)
    for ref in [
        *case["derived"]["recipe"]["store_preimages"].values(),
        case["derived"]["recipe"]["dashboard_sources"]["benchmark_ref"],
    ]:
        (tmp_path / ref["path"]).unlink()
    before = snapshot(tmp_path)
    value = inspect(tmp_path, case)
    assert value["schema_version"] == "cn-daily-launch-inspection.v3"
    assert (value["mode"], value["recovery_scope"]) == ("LOCAL_REPAIR", "COMMITTED_DAG_RECOVERY")
    assert snapshot(tmp_path) == before and not case["executions"]
    with pytest.raises(AutomaticCatchupError, match="AUTO_PUBLICATION_EXPIRED") as expired:
        run(tmp_path, case, committed_recovery_only=True)
    refs = expired.value.fields["completed_eod_refs"]
    assert [row["trade_date"] for row in refs] == ["20260825"]
    assert len(case["executions"]) == 1
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "IDLE"
    assert not (
        tmp_path / run_path(case["current"]["auto_request_ref"], "closure.v1.json")
    ).exists()
    after = snapshot(tmp_path)
    assert all(after[path] == value for path, value in before.items() if path != PENDING)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PUBLICATION_EXPIRED") as repeated:
        run(tmp_path, case)
    assert repeated.value.fields == expired.value.fields
    assert snapshot(tmp_path) == after and len(case["executions"]) == 1


@pytest.mark.parametrize("complete,publication", [(False, "expired"), (True, "unknown")])
def test_incomplete_or_unconfirmed_work_keeps_active_lease(
    tmp_path, monkeypatch, complete, publication
):
    case = fixture(tmp_path, monkeypatch, complete=complete, publication=publication)
    if complete:
        with pytest.raises(ContractError, match="AUTO_SERVING_SCOPE_UNCONFIRMED"):
            run(tmp_path, case)
    else:
        assert run(tmp_path, case)["status"] == "PARTIAL"
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "ACTIVE"
    assert len(case["executions"]) == 1


def test_completed_expired_active_lease_is_local_serving_repair(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    case["seal"]()
    before = snapshot(tmp_path)
    value = inspect(tmp_path, case)
    assert (value["mode"], value["recovery_scope"]) == ("LOCAL_REPAIR", "SERVING_ONLY")
    assert snapshot(tmp_path) == before
    with pytest.raises(AutomaticCatchupError, match="AUTO_PUBLICATION_EXPIRED"):
        run(tmp_path, case, no_producers=True)
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "IDLE"
    assert not case["executions"]


def test_same_day_committed_work_uses_protected_recovery_before_fresh_controls(
    tmp_path, monkeypatch
):
    case = fixture(tmp_path, monkeypatch, complete=False)
    case["clock"][0] = NOW + timedelta(seconds=2)
    ref = case["derived"]["recipe"]["dashboard_sources"]["benchmark_ref"]
    (tmp_path / ref["path"]).unlink()
    assert run(tmp_path, case)["status"] == "PARTIAL"
    assert case["executions"][0]["committed_recovery_only"] is True
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "ACTIVE"


def test_serving_only_restriction_cannot_upgrade_to_dag_recovery(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PRODUCERS_REQUIRED"):
        run(tmp_path, case, no_producers=True)
    assert snapshot(tmp_path) == before and not case["executions"]


def test_origin_only_preparation_can_expire_with_native_absence_proof(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch, started=False)
    before = snapshot(tmp_path)
    value = inspect(tmp_path, case)
    assert (value["mode"], value["recovery_scope"]) == ("LOCAL_REPAIR", "NONE")
    assert snapshot(tmp_path) == before
    with pytest.raises(AutomaticCatchupError, match="AUTO_RESOLUTION_EXPIRED_UNSTARTED"):
        run(tmp_path, case, no_producers=True)
    closure = json.loads(
        (tmp_path / run_path(case["current"]["auto_request_ref"], "closure.v1.json")).read_bytes()
    )
    assert closure["state"] == "EXPIRED_UNSTARTED"
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "IDLE"
    assert not case["executions"]


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_preparation",
        "maintenance_handoff",
        "foreign_execution",
        "changed_marker",
        "native_claim",
        "record_evidence",
    ],
)
def test_preparatory_exception_never_hides_started_or_unknown_work(tmp_path, monkeypatch, fault):
    from pathlib import PurePosixPath
    from quant_investor.operations.source_slot_contract import paths

    case = fixture(tmp_path, monkeypatch, started=False)
    execution = PurePosixPath(case["origin_ref"]["path"]).parent.parent
    prepared = paths(case["config_ref"], "20260825", 2)
    targets = {
        "unknown_preparation": prepared["root"] + "/objects/unexpected.json",
        "maintenance_handoff": str(execution / "maintenance-handoff.v1.json"),
        "foreign_execution": str(execution.parent / ("a" * 64) / "unexpected.json"),
        "changed_marker": prepared["marker1"],
        "native_claim": automatic_catchup_closure.RUN_ROOT
        + "/logical_tasks/20260825-2020-execute/claim.json",
        "record_evidence": "results/operations/daily_production/CN/20260825/nodes/unknown.json",
    }
    put(tmp_path, targets[fault], {"synthetic": "unknown or started evidence"})
    before = snapshot(tmp_path)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PENDING_RECOVERY_UNCONFIRMED"):
        run(tmp_path, case, no_producers=True)
    assert snapshot(tmp_path) == before
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "ACTIVE"


@pytest.mark.parametrize("fault", ["unconfirmed", "foreign", "unstarted_restricted"])
def test_unconfirmed_or_foreign_scope_cannot_start_recovery(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path, monkeypatch, started=fault != "unstarted_restricted")
    if fault == "unconfirmed":
        monkeypatch.setattr(
            committed,
            "inspect_registered_recovery",
            lambda **kw: (_ for _ in ()).throw(ContractError("original materialization missing")),
        )
    before = snapshot(tmp_path)
    if fault == "foreign":
        original = case["current"]["auto_request_ref"]
        copied = put(tmp_path, "foreign-auto.json", (tmp_path / original["path"]).read_bytes())
        before = snapshot(tmp_path)
        with pytest.raises(AutomaticCatchupError, match="AUTO_PENDING_RECOVERY_UNCONFIRMED"):
            daily_production.dispatch_daily_request(
                workspace=str(tmp_path),
                request_ref=copied,
                release_install_ref=case["config"]["release_install_ref"],
                synthetic=True,
            )
    else:
        with pytest.raises(AutomaticCatchupError, match="AUTO_PENDING_RECOVERY_UNCONFIRMED"):
            run(tmp_path, case, committed_recovery_only=True)
    assert snapshot(tmp_path) == before and not case["executions"]
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "ACTIVE"


@pytest.mark.parametrize("publication", ["expired", "unknown"])
def test_foreign_request_cannot_bypass_old_completed_publication_proof(
    tmp_path, monkeypatch, publication
):
    case = fixture(tmp_path, monkeypatch, publication=publication)
    case["seal"]()
    original = case["current"]["auto_request_ref"]
    copied = put(tmp_path, "newer-auto.json", (tmp_path / original["path"]).read_bytes())
    code = (
        "AUTO_PUBLICATION_EXPIRED"
        if publication == "expired"
        else "AUTO_PENDING_RECOVERY_UNCONFIRMED"
    )
    with pytest.raises(AutomaticCatchupError, match=code):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=copied,
            release_install_ref=case["config"]["release_install_ref"],
            synthetic=True,
        )
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == (
        "IDLE" if publication == "expired" else "ACTIVE"
    )
    assert not (tmp_path / run_path(copied, "resolution.v1.json")).exists()
    assert not case["executions"]
