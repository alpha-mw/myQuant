"""Configured source composition with native stores and synthetic transport/clock."""

import json
from datetime import timedelta

import pytest

from _configured_source_fixture import build
from _daily_preparation_fixture import snapshot, put
from _native_source_producer_fixture import BenchmarkClient, NOW
from test_daily_evidence_requested_session import capture as calendar_fixture
from quant_investor.operations import source_slot_inputs as inputs
from quant_investor.operations.source_slot_contract import SourceSlotError, paths
from quant_investor.operations.source_slot_storage import SourceLocatorStorage
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.automatic_catchup_contract import run_path, document_ref
from quant_investor.operations.daily_preparation_contract import PreparationError
from quant_investor.market import cn_benchmark_capture as benchmark_capture
from test_daily_evidence_execute_wiring import module as _native  # noqa: F401
from scripts import daily_source_inputs as module


def fixture(root, monkeypatch, *, holidays=()):
    data = build(root, monkeypatch)
    calls = []
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW)
    monkeypatch.setattr(inputs, "verify_recipe_static_controls", lambda **kwargs: {})

    def calendar(**kwargs):
        calls.append("Calendar")
        return calendar_fixture(kwargs["now"].isoformat(), holidays=holidays)

    monkeypatch.setattr(inputs, "acquire_close_session_authority", calendar)
    client = BenchmarkClient()
    monkeypatch.setattr(benchmark_capture, "OfficialTushareHttpsClient", lambda **kwargs: client)
    monkeypatch.setattr(benchmark_capture, "TUSHARE_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(benchmark_capture, "_now", lambda: NOW)
    monkeypatch.setattr(module.manager, "_manager_utc_now", lambda: NOW)
    data.update(calendar_calls=calls, client=client)
    return data


def run(root, data, mode="inspect", **kwargs):
    return module.configured_source_inputs(
        workspace=str(root),
        config_ref=data["config_ref"],
        release_install_ref=data["release_install_ref"],
        mode=mode,
        synthetic=True,
        **kwargs,
    )


def test_first_source_run_creates_bootstrap_and_repeat_is_readonly(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    assert run(tmp_path, data)["mode"] == "ACQUISITION_REQUIRED"
    assert snapshot(tmp_path) == before and data["calendar_calls"] == []
    result = run(tmp_path, data, "provision")
    assert result["mode"] == "REQUEST_AVAILABLE"
    assert data["calendar_calls"] == ["Calendar"] and len(data["client"].calls) == 3
    request = json.loads((tmp_path / result["request_ref"]["path"]).read_bytes())
    assert request["action"] == "EXECUTE"
    chain = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()
    assert [value["state"] for value, _ in chain] == ["REQUEST_AVAILABLE", "PREPARING_SOURCES"]
    after = snapshot(tmp_path)
    assert run(tmp_path, data) == result
    assert run(tmp_path, data, "provision", no_providers=True) == result
    assert snapshot(tmp_path) == after and len(data["client"].calls) == 3


def test_no_provider_mode_cannot_create_marker(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    with pytest.raises(SourceSlotError, match="PROVIDER_EVIDENCE_REQUIRED"):
        run(tmp_path, data, "provision", no_providers=True)
    selected = paths(data["config_ref"], "20260825")
    assert not (tmp_path / selected["marker1"]).exists() and not data["calendar_calls"]


def test_closed_session_creates_no_locator_or_source_plan(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch, holidays=("20260825",))
    before_pointer = module.event_store.pointer_sha256(data["book"].root / "_event_store")
    result = run(tmp_path, data, "provision")
    assert result["mode"] == "NON_TRADING_DAY" and result["request_ref"] is None
    assert not SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()
    assert not (tmp_path / paths(data["config_ref"], "20260825")["plan"]).exists()
    assert not data["client"].calls
    assert module.event_store.pointer_sha256(data["book"].root / "_event_store") == before_pointer
    before = snapshot(tmp_path)
    assert run(tmp_path, data) == result and snapshot(tmp_path) == before


def test_calendar_budget_is_consumed_before_each_failure(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    calls = []

    def fail(**kwargs):
        calls.append(1)
        raise OSError("synthetic transport failure")

    monkeypatch.setattr(inputs, "acquire_close_session_authority", fail)
    for _ in range(2):
        with pytest.raises(OSError):
            run(tmp_path, data, "provision")
    with pytest.raises(SourceSlotError, match="BUDGET_EXHAUSTED"):
        run(tmp_path, data, "provision")
    assert len(calls) == 2


def test_ledger_history_missing_blocks_index_fetch_and_event_write(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    next_day = NOW + timedelta(days=1)
    monkeypatch.setattr(inputs, "utc_now", lambda: next_day)
    with pytest.raises(SourceSlotError, match="HISTORICAL_EVENTS_MISSING") as caught:
        run(tmp_path, data, "provision")
    assert caught.value.fields["missing_event_dates"] == ["20260825"]
    assert not data["client"].calls
    assert not SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()


def test_cross_day_after_preparation_commitment_repairs_original_request(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = SourceLocatorStorage.publish

    def interrupt(storage, value, **kwargs):
        if value["state"] == "REQUEST_AVAILABLE":
            raise OSError("synthetic request-boundary interruption")
        return original(storage, value, **kwargs)

    monkeypatch.setattr(SourceLocatorStorage, "publish", interrupt)
    with pytest.raises(OSError):
        run(tmp_path, data, "provision")
    selected = paths(data["config_ref"], "20260825")
    committed = (tmp_path / selected["commitment"]).read_bytes()
    assert not (tmp_path / selected["request"]).exists()
    monkeypatch.setattr(SourceLocatorStorage, "publish", original)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(module.manager, "_manager_utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(benchmark_capture, "_now", lambda: NOW + timedelta(days=1))
    result = run(tmp_path, data, "provision", no_providers=True)
    assert result["trade_date"] == "20260825" and result["mode"] == "REQUEST_AVAILABLE"
    assert (tmp_path / selected["commitment"]).read_bytes() == committed
    assert data["calendar_calls"] == ["Calendar"] and len(data["client"].calls) == 3


def test_locator_exists_before_first_source_cas(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = benchmark_capture.capture_benchmark_close
    calls = []

    def guarded(**kwargs):
        loc = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][0]
        assert loc["state"] == "PREPARING_SOURCES" and loc["request_ref"] is None
        calls.append(loc["trade_date"])
        return original(**kwargs)

    monkeypatch.setattr(benchmark_capture, "capture_benchmark_close", guarded)
    run(tmp_path, data, "provision")
    assert calls == ["20260825"]


def test_cross_day_missing_event_stays_on_original_plan(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = benchmark_capture.capture_benchmark_close

    def interrupt(**kwargs):
        original(**kwargs)
        raise OSError("synthetic interruption after benchmark")

    monkeypatch.setattr(benchmark_capture, "capture_benchmark_close", interrupt)
    with pytest.raises(OSError):
        run(tmp_path, data, "provision")
    old = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][1].data
    monkeypatch.setattr(benchmark_capture, "capture_benchmark_close", original)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(module.manager, "_manager_utc_now", lambda: NOW + timedelta(days=1))
    with pytest.raises(SourceSlotError, match="HISTORICAL_EVENT_GENERATION_MISSING"):
        run(tmp_path, data, "provision", no_providers=True)
    assert SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][1].data == old
    assert len(data["client"].calls) == 3 and data["calendar_calls"] == ["Calendar"]


def test_cross_day_staged_event_is_adopted_without_new_seal(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = module.event_store.pointer_sha256
    fired = []

    def interrupt(root):
        chain = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()
        if chain:
            plan = json.loads((tmp_path / chain[0][0]["source_plan_ref"]["path"]).read_bytes())
            generation = root / "generations" / (plan["event_generation_id"] + ".v1.json")
            if generation.exists() and not fired:
                fired.append(generation)
                raise OSError("synthetic staged Event interruption")
        return original(root)

    monkeypatch.setattr(module.event_store, "pointer_sha256", interrupt)
    with pytest.raises(OSError):
        run(tmp_path, data, "provision")
    raw = fired[0].read_bytes()
    monkeypatch.setattr(module.event_store, "pointer_sha256", original)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(module.manager, "_manager_utc_now", lambda: NOW + timedelta(days=1))
    monkeypatch.setattr(benchmark_capture, "_now", lambda: NOW + timedelta(days=1))
    # Uncommitted CURRENT research cannot be backdated. Sources still recover exactly.
    with pytest.raises(PreparationError, match="FRESH_CALENDAR_REQUIRED"):
        run(tmp_path, data, "provision", no_providers=True)
    loaded = module.event_store.load_generation(data["book"].root / "_event_store")
    assert fired[0].read_bytes() == raw
    assert loaded["generation"] == json.loads(raw)
    assert len(data["client"].calls) == 3 and data["calendar_calls"] == ["Calendar"]


@pytest.mark.parametrize("fault", ["history_missing", "history_bytes", "locator_symlink"])
def test_locator_history_or_alias_corruption_blocks(tmp_path, monkeypatch, fault):
    data = fixture(tmp_path, monkeypatch)
    run(tmp_path, data, "provision")
    storage = SourceLocatorStorage(str(tmp_path), data["config_ref"])
    chain = storage.chain()
    if fault == "locator_symlink":
        raw = (tmp_path / storage.current).read_bytes()
        (tmp_path / storage.current).unlink()
        put(tmp_path, "fixtures/aliased-locator.json", raw)
        (tmp_path / storage.current).symlink_to(tmp_path / "fixtures/aliased-locator.json")
    else:
        history = (
            tmp_path
            / storage.prefix
            / "history"
            / (chain[0][0]["previous_locator_sha256"] + ".json")
        )
        if fault == "history_missing":
            history.unlink()
        else:
            history.write_bytes(b"{}")
    with pytest.raises(Exception):
        run(tmp_path, data)
    assert len(data["client"].calls) == 3


def pending(root, ref):
    with AutomaticRunStorage(str(root)).locked() as lock:
        lock.set_pending(
            {
                "schema_version": "cn-daily-catchup-pending.v1",
                "state": "ACTIVE",
                "auto_request_ref": ref,
                "resolution_ref": document_ref(run_path(ref, "resolution.v1.json"), {}),
            }
        )


def test_foreign_active_pending_blocks_before_calendar(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    ref = {"path": "foreign.json", "sha256": "b" * 64}
    pending(tmp_path, ref)
    before = snapshot(tmp_path)
    with pytest.raises(SourceSlotError, match="FOREIGN_ACTIVE_REQUEST") as caught:
        run(tmp_path, data, "provision")
    assert caught.value.fields["pending_request_ref"] == ref
    assert snapshot(tmp_path) == before and data["calendar_calls"] == []


def test_calendar_cross_midnight_consumes_budget_without_capture(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)

    def late(**kwargs):
        result = calendar_fixture(kwargs["now"].isoformat())
        monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
        return result

    monkeypatch.setattr(inputs, "acquire_close_session_authority", late)
    with pytest.raises(SourceSlotError, match="CAPTURE_DATE_CHANGED"):
        run(tmp_path, data, "provision")
    selected = paths(data["config_ref"], "20260825")
    assert (tmp_path / selected["marker1"]).exists()
    assert not (tmp_path / selected["capture"]).exists()
    assert not SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()


def test_expired_old_page_does_not_block_source_locator_advancement(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    first = run(tmp_path, data, "provision")
    seen = []

    def completed(ctx, selected, **kwargs):
        seen.append(selected["locator"]["request_ref"])
        return True  # Explicit native EOD/head seam; no serving API is called.

    monkeypatch.setattr(module, "_completed_target", completed)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    # Stop before new-day sources so this test only verifies historical selection.
    result = run(tmp_path, data)
    assert result["mode"] == "ACQUISITION_REQUIRED" and result["trade_date"] == "20260826"
    assert seen == [first["request_ref"]]
    assert data["calendar_calls"] == ["Calendar"]


def test_matching_active_automatic_request_is_reused_and_repaired(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    value = dict(data["config"])
    value["seed_completion_ref"] = put(
        tmp_path,
        "results/operations/daily_production/CN/20260824/completion.v1.json",
        {"synthetic": True},
    )
    data["config_ref"] = put(tmp_path, "fixtures/source-config.json", value)
    result = run(tmp_path, data, "provision")
    request = json.loads((tmp_path / result["request_ref"]["path"]).read_bytes())
    assert request["schema_version"] == "cn-daily-automatic-request.v2"
    pending(tmp_path, result["request_ref"])
    assert run(tmp_path, data) == result
    (tmp_path / result["request_ref"]["path"]).unlink()
    assert run(tmp_path, data)["mode"] == "LOCAL_PREPARATION"
    assert run(tmp_path, data, "provision", no_providers=True) == result
    assert len(data["client"].calls) == 3


def test_bad_static_policy_stops_before_calendar(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    value = dict(data["config"])
    policy = json.loads((tmp_path / value["store_policy_ref"]["path"]).read_bytes())
    policy["broker_order_trade_authority"] = True
    value["store_policy_ref"] = put(tmp_path, "fixtures/rejected-policy.json", policy)
    data["config_ref"] = put(tmp_path, "fixtures/source-config.json", value)
    with pytest.raises(Exception):
        run(tmp_path, data, "provision")
    assert not data["calendar_calls"] and not data["client"].calls


def test_request_file_interruption_leaves_recoverable_available_locator(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    original = module.inputs.JournalStorage.write

    def stop(storage, path, raw, **kwargs):
        if path == paths(data["config_ref"], "20260825")["request"]:
            raise OSError("synthetic request file interruption")
        return original(storage, path, raw, **kwargs)

    monkeypatch.setattr(module.inputs.JournalStorage, "write", stop)
    with pytest.raises(OSError):
        run(tmp_path, data, "provision")
    locator = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][0]
    assert locator["state"] == "REQUEST_AVAILABLE"
    monkeypatch.setattr(module.inputs.JournalStorage, "write", original)
    monkeypatch.setattr(inputs, "utc_now", lambda: NOW + timedelta(days=1))
    result = run(tmp_path, data, "provision", no_providers=True)
    assert result["request_ref"] == locator["request_ref"] and result["trade_date"] == "20260825"
    assert data["calendar_calls"] == ["Calendar"] and len(data["client"].calls) == 3


def test_locator_writer_requires_real_active_lock_and_exact_preimage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    data = fixture(tmp_path, monkeypatch)
    run(tmp_path, data, "provision")
    storage = SourceLocatorStorage(str(tmp_path), data["config_ref"])
    current, stored = storage.chain()[0]
    with pytest.raises(SourceSlotError, match="LOCK_WORKSPACE_MISMATCH"):
        storage.publish(
            current,
            expected_sha256=stored.byte_sha256,
            lock=SimpleNamespace(workspace=str(tmp_path), require_lock=lambda: None),
        )
    with pytest.raises(Exception):
        storage.publish(
            current, expected_sha256=stored.byte_sha256, lock=AutomaticRunStorage(str(tmp_path))
        )
    before = snapshot(tmp_path)
    with AutomaticRunStorage(str(tmp_path)).locked() as lock:
        assert (
            storage.publish(current, expected_sha256=stored.byte_sha256, lock=lock).data
            == stored.data
        )
    assert snapshot(tmp_path) == before


def test_calendar_budget_history_cannot_reclaim_deleted_first_marker(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise OSError("synthetic failed request")

    monkeypatch.setattr(inputs, "acquire_close_session_authority", fail)
    for _ in range(2):
        with pytest.raises(OSError):
            run(tmp_path, data, "provision")
    selected = paths(data["config_ref"], "20260825")
    (tmp_path / selected["marker1"]).unlink()
    with pytest.raises(SourceSlotError, match="REQUEST_HISTORY_MISSING"):
        run(tmp_path, data)


def test_concurrent_source_runner_cannot_take_over_preparing_locator(tmp_path, monkeypatch):
    import threading
    from quant_investor.operations.automatic_catchup_contract import AutomaticCatchupError

    data = fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = benchmark_capture.capture_benchmark_close
    outcomes = []

    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    def first():
        try:
            outcomes.append(run(tmp_path, data, "provision"))
        except Exception as exc:
            outcomes.append(exc)

    monkeypatch.setattr(benchmark_capture, "capture_benchmark_close", held)
    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(5)
    locator = SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][1].data
    try:
        with pytest.raises(AutomaticCatchupError, match="AUTO_RUN_BUSY"):
            run(tmp_path, data, "provision")
        assert SourceLocatorStorage(str(tmp_path), data["config_ref"]).chain()[0][1].data == locator
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and len(outcomes) == 1
    assert isinstance(outcomes[0], dict) and outcomes[0]["mode"] == "REQUEST_AVAILABLE"
    assert len(data["client"].calls) == 3
