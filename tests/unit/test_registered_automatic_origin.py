"""Real registered financial/source custody; full EOD and installation are controlled."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from _daily_preparation_fixture import put, snapshot
from _registered_event_fixture import NOW
from test_registered_daily_preparation import fixture as source_fixture, run as source_run
from quant_investor.operations import automatic_origin as origin
from quant_investor.operations import automatic_catchup_resolution as resolver
from quant_investor.operations.automatic_catchup_contract import document_ref, run_path
from quant_investor.operations.automatic_catchup_storage import (
    AutomaticRunStorage,
    PENDING_SCHEMA,
    automatic_execution,
    current_automatic_origin,
)
from quant_investor.operations.catchup_binding import (
    derive_catchup_binding,
    persist_catchup_binding,
)
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.system.errors import SystemStorageError
from scripts import daily_completion_replay, daily_completion_store


def fixture(root, monkeypatch):
    case = source_fixture(root, monkeypatch)
    seed = case["config"]["seed_completion_ref"]
    prior = json.loads((root / seed["path"]).read_bytes())
    native = json.loads((root / prior["native_inputs_ref"]["path"]).read_bytes())
    native["previous_trade_date"] = "20260821"
    prior.update(
        schema_version="cn-daily-eod-completion.v2",
        status="SUCCEEDED",
        trade_date="20260824",
        synthetic=True,
        native_validation_completed_at="2026-08-24T13:00:00Z",
        native_inputs_ref=put(root, prior["native_inputs_ref"]["path"], native),
        authority=FALSE_AUTHORITY,
        **{k: case["config"][k] for k in ("market", "strategy_id", "graph_sha256")},
    )
    case["config"]["seed_completion_ref"] = put(root, seed["path"], prior)
    case["config_ref"] = put(root, case["config_ref"]["path"], case["config"])
    # Actual previous Store terminal and C1 financial proof remain native.
    # The outer all-node EOD admission is the only completion seam here.
    monkeypatch.setattr(
        daily_completion_store,
        "inspect_recorded_completion",
        lambda **kw: {"recorded_completion": prior},
    )
    monkeypatch.setattr(
        daily_completion_replay,
        "replay_native_completion",
        lambda **kw: {
            **kw,
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": True,
        },
    )
    monkeypatch.setattr(resolver, "verify_completed_edge", lambda **kw: native)
    prepared = source_run(root, case, mode="provision")
    # Native book helpers also create public-display CSV fixtures. The automatic
    # request reader requires the private owner-only layout for every input.
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    selected = resolver.resolve_automatic_request(
        workspace=str(root),
        request_ref=prepared["request_ref"],
        release_install_ref=case["config"]["release_install_ref"],
        synthetic=True,
        now=NOW + timedelta(seconds=1),
    )
    resolution = selected["resolution"]
    resolution_ref = document_ref(
        run_path(prepared["request_ref"], "resolution.v1.json"), resolution
    )
    store = AutomaticRunStorage(str(root))
    with store.locked():
        from quant_investor.contracts import canonical_json_bytes

        for ref, value in (
            (resolution_ref, resolution),
            (resolution["derived_collection_ref"], resolution["derived_collection"]),
            (resolution["derived_request_ref"], resolution["derived_request"]),
        ):
            store.write(ref["path"], canonical_json_bytes(value))
        store.set_pending(
            {
                "schema_version": PENDING_SCHEMA,
                "state": "ACTIVE",
                "auto_request_ref": prepared["request_ref"],
                "resolution_ref": resolution_ref,
            }
        )
    derived = derive_catchup_binding(
        workspace=str(root),
        request_ref=resolution["derived_request_ref"],
        day="20260825",
        previous_completion_ref=resolution["anchor_ref"],
    )
    journal = DailyJournal(str(root), "20260825")
    with journal.locked():
        binding_ref = persist_catchup_binding(journal=journal, derived=derived)
    current = {
        "auto_request_ref": prepared["request_ref"],
        "resolution_ref": resolution_ref,
        "derived_request_ref": resolution["derived_request_ref"],
    }
    value = origin.origin_document(
        current=current,
        binding_ref=binding_ref,
        execution_request_ref=derived["binding"]["execution_request_ref"],
        day="20260825",
    )
    return {
        **case,
        "origin": value,
        "current": current,
        "store": store,
        "resolution": resolution,
        "resolution_ref": resolution_ref,
        "derived": derived,
    }


@contextmanager
def active(case):
    with case["store"].locked():
        with automatic_execution(
            case["store"],
            resolution_ref=case["resolution_ref"],
            resolution=case["resolution"],
            synthetic=True,
        ) as capability:
            yield capability


def publish(root, case):
    return origin.publish_automatic_origin(
        workspace=str(root), origin=case["origin"], synthetic=True
    )


def test_native_origin_replay_survives_lock_exit_and_mutable_heads_removal(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    with active(case):
        assert (
            current_automatic_origin(
                workspace=str(tmp_path),
                derived_request_ref=case["current"]["derived_request_ref"],
                synthetic=True,
            )
            == case["current"]
        )
        ref = publish(tmp_path, case)
        before = snapshot(tmp_path)
        assert publish(tmp_path, case) == ref
        assert snapshot(tmp_path) == before
    recipe = case["derived"]["recipe"]
    for reference in [
        *recipe["store_preimages"].values(),
        recipe["dashboard_sources"]["benchmark_ref"],
    ]:
        (tmp_path / reference["path"]).unlink()
    before = snapshot(tmp_path)
    replay = origin.read_automatic_origin(workspace=str(tmp_path), reference=ref)
    assert replay["origin"] == case["origin"] and replay["synthetic"] is True
    assert replay["bound"]["recipe"] == recipe
    assert snapshot(tmp_path) == before
    with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
        publish(tmp_path, case)
    assert snapshot(tmp_path) == before
    # Frozen source custody does not excuse fresh work from checking heads.
    with pytest.raises((SystemStorageError, ContractError)):
        resolver.resolve_automatic_request(
            workspace=str(tmp_path),
            request_ref=case["current"]["auto_request_ref"],
            release_install_ref=case["config"]["release_install_ref"],
            synthetic=True,
            now=NOW + timedelta(seconds=2),
        )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    [
        "no_capability",
        "wrong_request",
        "idle",
        "foreign_pending",
        "changed_capability",
        "late_origin",
    ],
)
def test_origin_refuses_unowned_or_retroactive_write(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path, monkeypatch)
    if fault == "no_capability":
        before = snapshot(tmp_path)
        with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
            publish(tmp_path, case)
    else:
        with active(case) as capability:
            if fault == "idle":
                case["store"].set_pending({**case["store"].pending(), "state": "IDLE"})
            elif fault == "foreign_pending":
                foreign = {"path": "another-auto.json", "sha256": "c" * 64}
                case["store"].set_pending(
                    {
                        "schema_version": PENDING_SCHEMA,
                        "state": "ACTIVE",
                        "auto_request_ref": foreign,
                        "resolution_ref": {
                            "path": run_path(foreign, "resolution.v1.json"),
                            "sha256": "d" * 64,
                        },
                    }
                )
            elif fault == "changed_capability":
                capability.raw += b" "
            elif fault == "late_origin":
                path = origin.origin_path("20260825", case["origin"]["execution_request_ref"])
                from pathlib import PurePosixPath

                put(
                    tmp_path,
                    str(PurePosixPath(path).parent.parent / "maintenance-handoff.v1.json"),
                    {"synthetic": "already-started"},
                )
            before = snapshot(tmp_path)
            with pytest.raises(ContractError):
                if fault == "wrong_request":
                    current_automatic_origin(
                        workspace=str(tmp_path),
                        derived_request_ref={"path": "wrong-request.json", "sha256": "a" * 64},
                        synthetic=True,
                    )
                else:
                    publish(tmp_path, case)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    [
        "auto_request_ref",
        "resolution_ref",
        "derived_request_ref",
        "catchup_binding_ref",
        "execution_request_ref",
        "trade_date",
        "authority",
        "extra",
        "path",
    ],
)
def test_resealed_origin_tampering_is_rejected_without_writes(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path, monkeypatch)
    with active(case):
        ref = publish(tmp_path, case)
    value = deepcopy(case["origin"])
    if fault.endswith("_ref"):
        value[fault] = {**value[fault], "sha256": "a" * 64}
    elif fault == "trade_date":
        value[fault] = "20260826"
    elif fault == "authority":
        value[fault]["execution_authorized"] = True
    elif fault == "extra":
        value["allow_recovery"] = True
    ref = put(tmp_path, "copied-origin.json" if fault == "path" else ref["path"], value)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError):
        origin.read_automatic_origin(workspace=str(tmp_path), reference=ref, synthetic=True)
    assert snapshot(tmp_path) == before


def test_upstream_controller_passes_original_origin_separately_from_historical_binding(
    tmp_path, monkeypatch
):
    from scripts import daily_catchup, daily_materialization

    case = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daily_catchup, "replay_native_completion", daily_completion_replay.replay_native_completion
    )
    received = []

    class BeforeMaintenance(Exception):
        pass

    def execute(**kwargs):
        received.append(kwargs)
        verified = origin.read_automatic_origin(
            workspace=str(tmp_path), reference=kwargs["_automatic_origin_ref"], synthetic=True
        )
        assert verified["origin"] == case["origin"]
        assert kwargs["_catchup_binding_ref"] is None
        assert kwargs["request_ref"] == case["origin"]["execution_request_ref"]
        raise BeforeMaintenance

    monkeypatch.setattr(daily_materialization, "execute_daily_recipe", execute)
    with active(case), pytest.raises(BeforeMaintenance):
        daily_catchup.run_upstream_catchup(
            workspace=str(tmp_path),
            request_ref=case["current"]["derived_request_ref"],
            synthetic=True,
        )
    assert len(received) == 1
    before = snapshot(tmp_path)
    ref = received[0]["_automatic_origin_ref"]
    assert (
        origin.read_automatic_origin(workspace=str(tmp_path), reference=ref)["origin"]
        == case["origin"]
    )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("wrong_handoff", [False, True])
def test_execute_resumes_only_the_handoff_with_the_original_origin(
    tmp_path, monkeypatch, wrong_handoff
):
    from pathlib import PurePosixPath
    from scripts import daily_materialization, daily_completion
    from quant_investor.operations import execution_controls

    case = fixture(tmp_path, monkeypatch)
    called = []

    def no_producers(**kwargs):
        pytest.fail("retained handoff must not enter fresh controls/maintenance")

    monkeypatch.setattr(execution_controls, "read_execution_controls", no_producers)
    request_ref = case["origin"]["execution_request_ref"]
    with active(case):
        ref = publish(tmp_path, case)
        handoff_value = {
            "request_ref": request_ref,
            "automatic_origin_ref": None if wrong_handoff else ref,
        }
        handoff_ref = put(
            tmp_path,
            str(PurePosixPath(ref["path"]).parent.parent / "maintenance-handoff.v1.json"),
            handoff_value,
        )
        monkeypatch.setattr(
            daily_materialization,
            "read_maintenance_handoff",
            lambda **kw: {"handoff": handoff_value, "recipe": case["derived"]["recipe"]},
        )
        monkeypatch.setattr(
            daily_materialization,
            "materialize_daily_inputs",
            lambda **kw: called.append(kw) or {"native_inputs_ref": handoff_ref},
        )
        monkeypatch.setattr(
            daily_completion,
            "run_materialized_native_input",
            lambda **kw: {"status": "PARTIAL", "completion_ref": None},
        )
        before = snapshot(tmp_path)
        kwargs = dict(
            workspace=str(tmp_path),
            request_ref=request_ref,
            synthetic=True,
            _automatic_origin_ref=ref,
        )
        if wrong_handoff:
            with pytest.raises(ContractError, match="HANDOFF_AUTOMATIC_ORIGIN_MISMATCH"):
                daily_materialization.execute_daily_recipe(**kwargs)
            assert called == []
        else:
            assert daily_materialization.execute_daily_recipe(**kwargs)["status"] == "PARTIAL"
            assert len(called) == 1 and called[0]["handoff_ref"] == handoff_ref
        assert snapshot(tmp_path) == before
