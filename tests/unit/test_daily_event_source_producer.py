"""Native Event/Calendar tests, with synthetic data and a controlled actual clock."""

from contextlib import contextmanager
from datetime import timedelta
import json

import pytest

from _daily_preparation_fixture import put, snapshot
from _native_source_producer_fixture import NOW, event_arguments
from quant_investor.strategy_records import event_store
from quant_investor.strategy_records.daily_event_source import (
    MAINTENANCE_FIELDS,
    STATE_FIELDS,
    validate_daily_closure_source,
)
from quant_investor.strategy_records.store import StrategyRecordStoreError
from scripts import manage_cn_strategy_records as manager


def setup(root, monkeypatch, **kwargs):
    book, args = event_arguments(root, **kwargs)
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    return book, args


def maintenance(root, args, modern=False):
    from quant_investor.market.daily_maintenance import STAGES

    value = dict.fromkeys(MAINTENANCE_FIELDS)
    value.update(
        schema_version="cn-daily-maintenance-attempt.v1",
        status="PARTIAL",
        maintenance_status="PARTIAL",
        same_day_status="BLOCKED",
        factor_input_readiness="READY",
        factor_input_shadow_readiness="READY",
        factor_input_change="UNKNOWN",
        core_blockers=[],
        macro_status="BLOCKED",
        macro_blockers=["SYNTHETIC_MACRO_BLOCKED"],
        macro_used_by_factor=False,
        fundamental_used_by_factor=False,
        factor_rollover_eligible=True,
        fundamental_integrity_status="READY",
        fundamental_refresh_status="HEALTH_ONLY",
        mode="execute",
        attempt_slot="2020",
        target_date="20260825",
        canonical_unchanged=True,
        canonical_write_count=0,
        usable_for_investment_research="UNCONFIRMED",
        blockers=["SYNTHETIC_MACRO_BLOCKED"],
        protected_surfaces=[],
        close_session_receipt_ref={
            "path": args.calendar_receipt,
            "sha256": args.calendar_receipt_sha256,
        },
        stage_results=[
            {
                "stage": stage,
                "status": "BLOCKED" if stage == "MACRO_RELEASE" else "NO_ACTION",
                "write_performed": False,
                "blockers": ["SYNTHETIC_MACRO_BLOCKED"] if stage == "MACRO_RELEASE" else [],
                "evidence": {},
            }
            for stage in STAGES
        ],
    )
    state = {k: value[k] for k in STATE_FIELDS - {"schema_version", "stage_states"}}
    state.update(
        schema_version="cn-daily-maintenance-state.v1",
        stage_states={r["stage"]: r["status"] for r in value["stage_results"]},
    )
    ref = put(root, "fixtures/maintenance/state.json", state)
    value["state_ref"] = {**ref, "path": str(root / ref["path"])}
    if modern:
        value.update(
            provider_calls=False,
            provider_request_attempts={},
            request_count_scope="OFFICIAL_TRANSPORT_CURRENT_OPERATION_CONTEXT",
        )
    ref = put(root, "fixtures/maintenance/attempt.json", value)
    if modern:
        put(
            root,
            "fixtures/maintenance/ended.json",
            {
                "state": "COMPLETED",
                "workflow_status": "PARTIAL",
                "receipt_ref": {**ref, "path": str(root / ref["path"])},
                "ended_at": "2026-08-25T12:20:00Z",
            },
        )
    args.maintenance_receipt, args.maintenance_receipt_sha256 = (
        str(root / ref["path"]),
        ref["sha256"],
    )
    args.calendar_receipt = args.calendar_receipt_sha256 = args.raw_calendar = (
        args.raw_calendar_sha256
    ) = None
    return ref


def test_current_calendar_seals_once_preserving_prior_dates(tmp_path, monkeypatch):
    book, args = setup(tmp_path, monkeypatch)
    before = event_store.load_generation(book.root / "_event_store")
    result = manager.command_publish_daily_event_closure(args)
    assert result["status"] == "PUBLISHED"
    after = event_store.load_generation(book.root / "_event_store")
    assert after["closures"][:-1] == before["closures"]
    proof = validate_daily_closure_source(workspace=tmp_path, closure=after["closures"][-1])
    assert proof["eod_admission"] is False and proof["execution_authorized"] is False
    assert any(r["path"] == "fixtures/current-calendar.raw.json" for r in proof["source_refs"])
    args.expected_event_pointer_sha256 = result["pointer_sha256"]
    original = snapshot(tmp_path)
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW + timedelta(days=2))
    assert manager.command_publish_daily_event_closure(args)["status"] == "NO_ACTION"
    assert snapshot(tmp_path) == original


@pytest.mark.parametrize("modern", [False, True])
def test_exact_maintenance_profile_preserves_partial_status(tmp_path, monkeypatch, modern):
    book, args = setup(tmp_path, monkeypatch)
    maintenance(tmp_path, args, modern)
    manager.command_publish_daily_event_closure(args)
    closure = event_store.load_generation(book.root / "_event_store")["closures"][-1]
    proof = validate_daily_closure_source(workspace=tmp_path, closure=closure)
    assert proof["maintenance_status"] == "PARTIAL" and proof["eod_admission"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "yesterday",
        "tomorrow",
        "pre_cutoff",
        "closed",
        "future_observation",
        "raw_sha",
        "raw_alias",
        "input_mix",
        "target_only",
    ],
)
def test_invalid_new_day_or_source_cannot_create_closure(tmp_path, monkeypatch, fault):
    book, args = setup(tmp_path, monkeypatch, holidays=("20260825",) if fault == "closed" else ())
    if fault == "yesterday":
        monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW + timedelta(days=1))
    elif fault == "tomorrow":
        args.trade_date = "2026-08-26"
    elif fault == "pre_cutoff":
        monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW - timedelta(hours=5))
    elif fault == "future_observation":
        monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW - timedelta(seconds=1))
    elif fault == "raw_sha":
        args.raw_calendar_sha256 = "a" * 64
    elif fault == "raw_alias":
        ref = put(
            tmp_path,
            "fixtures/other-raw.json",
            (tmp_path / "fixtures/current-calendar.raw.json").read_bytes(),
        )
        args.raw_calendar = str(tmp_path / ref["path"])
    elif fault == "input_mix":
        args.maintenance_receipt, args.maintenance_receipt_sha256 = (
            args.calendar_receipt,
            args.calendar_receipt_sha256,
        )
    elif fault == "target_only":
        ref = maintenance(tmp_path, args)
        args.maintenance_receipt_sha256 = put(tmp_path, ref["path"], {"target_date": "20260825"})[
            "sha256"
        ]
    old = event_store.load_generation(book.root / "_event_store")["pointer_sha256"]
    with pytest.raises(Exception):
        manager.command_publish_daily_event_closure(args)
    assert event_store.load_generation(book.root / "_event_store")["pointer_sha256"] == old


def test_invalid_original_is_not_replaced_by_supplied_valid_calendar(tmp_path, monkeypatch):
    _, args = setup(tmp_path, monkeypatch)
    result = manager.command_publish_daily_event_closure(args)
    args.expected_event_pointer_sha256 = result["pointer_sha256"]
    args.calendar_receipt = str(tmp_path / "fixtures/new-calendar.json")
    put(
        tmp_path,
        "fixtures/new-calendar.json",
        (tmp_path / "fixtures/current-calendar.json").read_bytes(),
    )
    (tmp_path / "fixtures/current-calendar.raw.json").write_bytes(b"{}")
    original = snapshot(tmp_path)
    with pytest.raises(StrategyRecordStoreError):
        manager.command_publish_daily_event_closure(args)
    assert snapshot(tmp_path) == original


def test_programmatic_entry_acquires_record_lock(tmp_path, monkeypatch):
    _, args = setup(tmp_path, monkeypatch)
    original, calls = manager._operation_lock, []

    @contextmanager
    def lock(path):
        calls.append("entered")
        with original(path):
            yield
        calls.append("released")

    monkeypatch.setattr(manager, "_operation_lock", lock)
    manager.command_publish_daily_event_closure(args)
    assert calls == ["entered", "released"]


def test_mutable_veto_is_only_recorded_diagnostic_metadata(tmp_path, monkeypatch):
    book, args = setup(tmp_path, monkeypatch)
    ref = maintenance(tmp_path, args)
    value = json.loads((tmp_path / ref["path"]).read_bytes())
    veto_ref = put(tmp_path, "fixtures/MACRO_WRITE_VETO.json", {"old": True})
    value["macro_write_veto_ref"] = veto_ref
    args.maintenance_receipt_sha256 = put(tmp_path, ref["path"], value)["sha256"]
    put(tmp_path, veto_ref["path"], {"new": True})
    manager.command_publish_daily_event_closure(args)
    closure = event_store.load_generation(book.root / "_event_store")["closures"][-1]
    proof = validate_daily_closure_source(workspace=tmp_path, closure=closure)
    assert proof["maintenance_status"] == "PARTIAL" and proof["eod_admission"] is False
    assert veto_ref not in proof["source_refs"]


@pytest.mark.parametrize(
    "fault", ["write_count", "stage_type", "extra", "state", "ended_status", "ended_future"]
)
def test_invalid_maintenance_terminal_profile_blocks(tmp_path, monkeypatch, fault):
    book, args = setup(tmp_path, monkeypatch)
    ref = maintenance(tmp_path, args, modern=True)
    value = json.loads((tmp_path / ref["path"]).read_bytes())
    if fault == "write_count":
        value["canonical_write_count"] = 1
    elif fault == "stage_type":
        value["stage_results"][0]["write_performed"] = 1
    elif fault == "extra":
        value["fake_completed"] = True
    elif fault == "state":
        state = json.loads((tmp_path / "fixtures/maintenance/state.json").read_bytes())
        state["status"] = "COMPLETE"
        state_ref = put(tmp_path, "fixtures/maintenance/state.json", state)
        value["state_ref"] = state_ref
    args.maintenance_receipt_sha256 = put(tmp_path, ref["path"], value)["sha256"]
    ended = json.loads((tmp_path / "fixtures/maintenance/ended.json").read_bytes())
    ended["receipt_ref"]["sha256"] = args.maintenance_receipt_sha256
    if fault == "ended_status":
        ended["workflow_status"] = "COMPLETE"
    elif fault == "ended_future":
        ended["ended_at"] = "2026-08-25T12:21:00Z"
    put(tmp_path, "fixtures/maintenance/ended.json", ended)
    with pytest.raises(StrategyRecordStoreError):
        manager.command_publish_daily_event_closure(args)
    assert (
        event_store.load_generation(book.root / "_event_store")["pointer_sha256"]
        == args.expected_event_pointer_sha256
    )


def test_cli_calendar_form_uses_one_owning_lock(tmp_path, monkeypatch, capsys):
    _, args = setup(tmp_path, monkeypatch)
    original, calls = manager._operation_lock, []

    @contextmanager
    def lock(path):
        calls.append(path)
        with original(path):
            yield

    monkeypatch.setattr(manager, "_operation_lock", lock)
    argv = ["publish-daily-event-closure"]
    for name in (
        "project_root",
        "record_root",
        "trade_date",
        "policy_path",
        "policy_sha256",
        "calendar_receipt",
        "calendar_receipt_sha256",
        "raw_calendar",
        "raw_calendar_sha256",
        "expected_event_pointer_sha256",
        "generation_id",
    ):
        argv.extend(["--" + name.replace("_", "-"), getattr(args, name)])
    assert manager.main(argv) == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLISHED"
