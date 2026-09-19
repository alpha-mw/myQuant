"""Native inspection/control flow; full EOD and installed policy admission are explicit seams."""

import pytest
from _public_catchup_fixture import put
from test_automatic_catchup_execution import execution_fixture, run
from test_daily_evidence_public_catchup import snapshot
from scripts import daily_launch_inspection as inspection
from scripts import daily_dashboard_publication as publication
from scripts import daily_production
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.automatic_catchup_contract import AutomaticCatchupError
from quant_investor.operations.daily_contract import ContractError


def inspect(root, request, ref):
    return inspection.inspect_daily_launch(
        workspace=str(root),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )


def serving_state(root, day, state):
    path = f"results/operations/daily_production/CN/{day}/completion.v1.json"
    from quant_investor.operations.automatic_catchup_contract import digest

    return {
        "sealed_evidence_ref": {"path": path, "sha256": digest((root / path).read_bytes())},
        "publication_state": state,
        "publication_ref": (
            {
                "path": f"results/operations/daily_production/CN/{day}/"
                "dashboard/serving-publication.v2.json",
                "sha256": "c" * 64,
            }
            if state == "RECORDED_EOD_PUBLICATION"
            else None
        ),
        "validation_scope": "RECORDED_SERVING_BYTES_ONLY",
    }


def test_missing_eod_inspection_checks_static_controls_without_writes(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    checked = []
    monkeypatch.setattr(
        inspection,
        "verify_recipe_static_controls",
        lambda **kw: checked.append(kw["recipe"]["target_trade_date"]),
    )
    before = snapshot(tmp_path)
    assert inspect(tmp_path, request, ref)["mode"] == "PRODUCER_REQUIRED"
    assert checked == ["20260827", "20260828"] and snapshot(tmp_path) == before and calls == []


@pytest.mark.parametrize(
    "state,expected",
    [
        ("RECORDED_EOD_PUBLICATION", "COMPLETE_READ_ONLY"),
        ("EVIDENCE_SEALED_PUBLICATION_PENDING", "LOCAL_REPAIR"),
        ("RECORDED_EOD_PUBLICATION_EXPIRED", "AUTO_PUBLICATION_EXPIRED"),
        ("EVIDENCE_SEALED_PUBLICATION_EXPIRED", "AUTO_PUBLICATION_EXPIRED"),
    ],
)
def test_completed_inspection_uses_readonly_serving_not_publisher(
    tmp_path, monkeypatch, state, expected
):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    run(tmp_path, request, ref)
    monkeypatch.setattr(
        publication,
        "observed_serving_status",
        lambda workspace, day: serving_state(tmp_path, day, state),
    )
    monkeypatch.setattr(
        publication, "complete_serving_result", lambda **kw: pytest.fail("inspection published")
    )
    before = snapshot(tmp_path)
    if expected.startswith("AUTO_"):
        with pytest.raises(ContractError, match=expected):
            inspect(tmp_path, request, ref)
    else:
        value = inspect(tmp_path, request, ref)
        assert value["mode"] == expected
        if expected == "COMPLETE_READ_ONLY":
            assert value["result"]["status"] == "NO_ACTION"
        else:
            assert value["result"] is None
    assert len(calls) == 2 and snapshot(tmp_path) == before


def test_active_complete_lease_needs_local_repair_and_foreign_active_lease_blocks(
    tmp_path, monkeypatch
):
    request, ref, _, _, _ = execution_fixture(tmp_path, monkeypatch)
    run(tmp_path, request, ref)
    monkeypatch.setattr(
        publication,
        "observed_serving_status",
        lambda workspace, day: serving_state(tmp_path, day, "RECORDED_EOD_PUBLICATION"),
    )
    storage = AutomaticRunStorage(str(tmp_path))
    with storage.locked():
        storage.set_pending({**storage.pending(), "state": "ACTIVE"})
    before = snapshot(tmp_path)
    assert inspect(tmp_path, request, ref)["mode"] == "LOCAL_REPAIR"
    assert snapshot(tmp_path) == before
    other = put(tmp_path, "other-request.json", request)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PENDING_REQUEST_CONFLICT"):
        inspect(tmp_path, request, other)


def test_no_producers_blocks_before_bindings_and_rechecks_after_inspection(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PRODUCERS_REQUIRED"):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
            no_producers=True,
        )
    assert calls == [] and not list(tmp_path.rglob("binding.v2.json"))
    run(tmp_path, request, ref)
    monkeypatch.setattr(
        publication,
        "observed_serving_status",
        lambda workspace, day: serving_state(tmp_path, day, "EVIDENCE_SEALED_PUBLICATION_PENDING"),
    )
    assert inspect(tmp_path, request, ref)["mode"] == "LOCAL_REPAIR"
    path = tmp_path / "results/operations/daily_production/CN/20260828/completion.v1.json"
    path.unlink()  # Simulated evidence loss after inspection.
    before = snapshot(tmp_path)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PRODUCERS_REQUIRED"):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
            no_producers=True,
        )
    assert snapshot(tmp_path) == before and len(calls) == 2


def test_no_producers_can_repair_existing_publication_without_any_producer(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    run(tmp_path, request, ref)
    publications = []

    def publish(**kw):
        publications.append(kw["completion_ref"])
        return {"status": "COMPLETE", "completion_ref": kw["completion_ref"]}

    monkeypatch.setattr(publication, "complete_serving_result", publish)
    result = daily_production.dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
        no_producers=True,
    )
    assert result["status"] == "NO_ACTION" and len(calls) == 2 and len(publications) == 1


def test_static_policy_failure_and_serving_corruption_are_blockers(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)

    def invalid(**kw):
        raise ContractError("EXECUTION_RESEARCH_POLICY_INVALID")

    monkeypatch.setattr(inspection, "verify_recipe_static_controls", invalid)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError, match="EXECUTION_RESEARCH_POLICY_INVALID"):
        inspect(tmp_path, request, ref)
    assert snapshot(tmp_path) == before and calls == []
    run(tmp_path, request, ref)
    monkeypatch.setattr(
        publication,
        "observed_serving_status",
        lambda *args: (_ for _ in ()).throw(ContractError("CORRUPT_SERVING")),
    )
    with pytest.raises(ContractError, match="CORRUPT_SERVING"):
        inspect(tmp_path, request, ref)


def test_expired_pending_local_closure_never_becomes_completed_result(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from scripts import daily_automatic_catchup as auto
    from quant_investor.operations.automatic_catchup_closure import RUN_ROOT

    request, ref, clock, calls, _ = execution_fixture(tmp_path, monkeypatch)
    put(tmp_path, RUN_ROOT + "/.daily-maintenance.lock", b"")

    @contextmanager
    def crash(*args, **kwargs):
        raise OSError("before producers")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(auto, "automatic_execution", crash)
        with pytest.raises(OSError, match="before producers"):
            run(tmp_path, request, ref)
    clock[0] = datetime(2026, 8, 29, 14, tzinfo=timezone.utc)
    before = snapshot(tmp_path)
    value = inspect(tmp_path, request, ref)
    assert value["mode"] == "LOCAL_REPAIR" and value["result"] is None
    assert snapshot(tmp_path) == before
    with pytest.raises(AutomaticCatchupError, match="AUTO_RESOLUTION_EXPIRED_UNSTARTED"):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
            no_producers=True,
        )
    assert calls == [] and AutomaticRunStorage(str(tmp_path)).pending()["state"] == "IDLE"
