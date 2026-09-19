"""Native single-day registered BUY -> official valuation, with synthetic owner data."""

from datetime import timedelta
import hashlib
import json

import pytest

from _registered_event_fixture import build, NOW
from _native_daily_store_fixture import write
from test_daily_evidence_requested_session import capture
from scripts import manage_cn_strategy_records as manager
from scripts import cn_official_close_batch as native
from quant_investor.strategy_records import store, performance
from scripts.daily_production_store_adapter import StoreCloseAdapter, prepare_store_plan
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.portfolio_binding import (
    NativePortfolioSource,
    retain_portfolio_source,
)


def case(root, monkeypatch):
    fixture = build(root)
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    declared = manager.command_publish_registered_event_declaration(fixture["args"])
    args = fixture["book"].advance("2026-08-25", publish_events=False)
    calendar = capture(NOW.isoformat())
    path = root / "fixtures/registered-calendar.json"
    args["calendar_receipt_path"] = path
    args["calendar_receipt_sha"] = write(path, calendar.receipt)
    args["registered_event_declaration_ref"] = declared["declaration_ref"]
    return fixture, args


def test_native_registered_close_replaces_intraday_performance(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    pointer, catalog = store.load_registered_catalog(fixture["book"].root)
    before = performance.load_performance_history(
        fixture["book"].root, catalog["performance_history_ref"]
    )["rows"]
    with manager._operation_lock(fixture["book"].root):
        planned = native.close_through_latest(
            **args, execute=False, prepare_only=True, now=NOW + timedelta(seconds=1)
        )
        result = native.close_through_latest(
            **args,
            execute=True,
            expected_plan_sha=planned["plan_sha256"],
            now=NOW + timedelta(seconds=2),
        )
    assert planned["plan_path"].endswith("/plan.v2.json")
    assert result["status"] == "COMMITTED"
    _, catalog = store.load_registered_catalog(fixture["book"].root)
    after = performance.load_performance_history(
        fixture["book"].root, catalog["performance_history_ref"]
    )["rows"]
    assert before[:-1] == after[:-1]
    assert before[-1]["unit_count"] == after[-1]["unit_count"]
    assert after[-1]["evidence_kind"] == "REGISTERED_OFFICIAL_FINANCIAL_STATE"
    proof = native.inspect_frozen_close_commit(
        record_root=fixture["book"].root,
        transaction_id=planned["transaction_id"],
        expected_plan_sha=planned["plan_sha256"],
        expected_source_pointer_sha=fixture["writer_sha"],
        expected_target="2026-08-25",
        plan_version=2,
    )
    assert proof["status"] == "VERIFIED"


def test_adapter_keeps_decision_baseline_separate_from_writer(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    prepared = prepare_store_plan(args)
    plan_ref = {"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]}
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260825",
        plan_ref=plan_ref,
        release_ref={"path": "fixture-release.json", "sha256": "a" * 64},
    )
    journal = DailyJournal(str(tmp_path), "20260825")
    source = NativePortfolioSource(
        workspace=tmp_path, trade_date="20260825", store_plan_ref=plan_ref
    )
    with journal.locked():
        fields = retain_portfolio_source(journal=journal, source=source)
    assert fields["frozen_pointer_ref"]["sha256"] == fixture["baseline_ref"]["sha256"]
    assert fields["source_record_id"] != "20260825_1000"
    assert adapter.probe(adapter.template()).safe_to_execute
    adapter.execute(adapter.template())
    outcome = adapter.probe(adapter.template()).outcome
    assert outcome.state.value == "SUCCEEDED"
    assert outcome.output_refs["completion"]["path"].endswith("/completion.v2.json")


def prepared_case(root, monkeypatch):
    fixture, args = case(root, monkeypatch)
    prepared = prepare_store_plan(args)
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260825",
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref={"path": "fixture-release.json", "sha256": "a" * 64},
    )
    return fixture, args, prepared, adapter


def inventory(root):
    return {
        p.relative_to(root).as_posix(): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def test_readonly_plan_and_completed_adoption_do_not_write(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    before = inventory(tmp_path)
    readonly = native.close_through_latest(**args, execute=False)
    assert readonly["status"] == "PLAN_READY"
    assert inventory(tmp_path) == before
    prepared = prepare_store_plan(args)
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260825",
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref={"path": "fixture-release.json", "sha256": "a" * 64},
    )
    original = native.publish_catalog
    calls = []

    def publish(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    monkeypatch.setattr(native, "publish_catalog", publish)
    adapter.execute(adapter.template())
    assert len(calls) == 1
    before = inventory(tmp_path)
    repeated = prepare_store_plan(args)
    assert repeated["status"] == "PLAN_ADOPTED"
    current = native._pointer_sha(fixture["book"].root / "_record_store/current.v1.json")
    adopted = prepare_store_plan({**args, "expected_store_pointer_sha": current})
    assert adopted["plan_sha256"] == prepared["plan_sha256"]
    assert adopted["source_pointer_ref"]["sha256"] == fixture["baseline_ref"]["sha256"]
    adapter.execute(adapter.template())
    assert inventory(tmp_path) == before and len(calls) == 1


@pytest.mark.parametrize("cut", ["after_cas", "before_completion"])
def test_registered_close_crash_recovers_without_second_cas(tmp_path, monkeypatch, cut):
    fixture, args, prepared, adapter = prepared_case(tmp_path, monkeypatch)
    publish, write = native.publish_catalog, native._write_exact_json
    calls = []

    def crash_publish(*a, **kw):
        calls.append(kw)
        result = publish(*a, **kw)
        if cut == "after_cas":
            raise RuntimeError("synthetic after CAS")
        return result

    def crash_write(path, value):
        if cut == "before_completion" and path.name == "completion.v2.json":
            raise RuntimeError("synthetic before completion")
        return write(path, value)

    monkeypatch.setattr(native, "publish_catalog", crash_publish)
    monkeypatch.setattr(native, "_write_exact_json", crash_write)
    with pytest.raises(RuntimeError, match="synthetic"):
        adapter.execute(adapter.template())
    assert len(calls) == 1
    monkeypatch.setattr(native, "_write_exact_json", write)
    monkeypatch.setattr(native, "publish_catalog", lambda *a, **kw: pytest.fail("second CAS"))
    assert adapter.probe(adapter.template()).recovery_only
    adapter.execute(adapter.template())
    assert adapter.probe(adapter.template()).outcome.state.value == "SUCCEEDED"


def test_frozen_registered_close_uses_no_current_heads(tmp_path, monkeypatch):
    fixture, args, prepared, adapter = prepared_case(tmp_path, monkeypatch)
    adapter.execute(adapter.template())
    expected = adapter.probe(adapter.template()).outcome.output_refs
    for path in (
        fixture["book"].root / "_record_store/current.v1.json",
        fixture["book"].root / "_event_store/current.v1.json",
        tmp_path / "data/parquet/cn/_latest.json",
        tmp_path / "data/parquet/cn/benchmarks/_latest.json",
    ):
        path.unlink()
    monkeypatch.setattr(
        native, "_assert_registered_ancestor", lambda *a: pytest.fail("current ancestry")
    )
    monkeypatch.setattr(native, "publish_catalog", lambda *a, **kw: pytest.fail("financial writer"))
    before = inventory(tmp_path)
    assert adapter.probe(adapter.template()).outcome.output_refs == expected
    source = NativePortfolioSource(
        workspace=tmp_path, trade_date="20260825", store_plan_ref=adapter.plan_ref
    )
    journal = DailyJournal(str(tmp_path), "20260825")
    with journal.locked():
        fields = retain_portfolio_source(journal=journal, source=source)
    assert fields["frozen_pointer_ref"]["sha256"] == fixture["baseline_ref"]["sha256"]
    for path, value in before.items():
        assert inventory(tmp_path)[path] == value


def test_undeclared_intraday_state_is_not_official_no_action(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    args.pop("registered_event_declaration_ref")
    with pytest.raises(store.StrategyRecordStoreError, match="DECLARATION_REQUIRED"):
        native.close_through_latest(**args, execute=False)


@pytest.mark.parametrize(
    "fault",
    [
        "version_path",
        "other_version",
        "decision_pointer",
        "writer_pointer",
        "receipt",
        "performance",
    ],
)
def test_registered_commit_corruption_blocks_readback(tmp_path, monkeypatch, fault):
    fixture, args, prepared, adapter = prepared_case(tmp_path, monkeypatch)
    adapter.execute(adapter.template())
    outputs = adapter.probe(adapter.template()).outcome.output_refs
    directory = tmp_path / adapter.plan_ref["path"]
    if fault == "version_path":
        value = json.loads(directory.read_bytes())
        value["schema_id"] = native.BATCH_PLAN_SCHEMA
        write(directory, value)
    elif fault == "other_version":
        directory.with_name("plan.v1.json").write_bytes(b"{}")
    elif fault in {"decision_pointer", "writer_pointer"}:
        directory.with_name(
            "decision-source-pointer.v1.json"
            if fault == "decision_pointer"
            else "source-pointer.v1.json"
        ).write_bytes(b"corrupt")
    elif fault == "receipt":
        (tmp_path / outputs["completion"]["path"]).write_bytes(b"corrupt")
    else:
        (tmp_path / outputs["performance_series"]["path"]).write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError, store.StrategyRecordStoreError)):
        adapter.probe(adapter.template())


@pytest.mark.parametrize(
    "path,sha", [("retrospective.json", None), (None, "a" * 64), ("retrospective.json", "a" * 64)]
)
def test_completed_adoption_rejects_every_retrospective_input(tmp_path, monkeypatch, path, sha):
    fixture, args, prepared, adapter = prepared_case(tmp_path, monkeypatch)
    adapter.execute(adapter.template())
    monkeypatch.setattr(
        native, "_adopt_registered_close", lambda **kw: pytest.fail("adoption before profile check")
    )
    before = inventory(tmp_path)
    with pytest.raises(store.StrategyRecordStoreError, match="RETROSPECTIVE_PROFILE_CONFLICT"):
        native.close_through_latest(
            **{**args, "retrospective_path": path, "retrospective_sha": sha}, execute=False
        )
    assert inventory(tmp_path) == before


def test_adoption_uses_calendar_closed_target_not_later_open_session(tmp_path, monkeypatch):
    fixture, args = case(tmp_path, monkeypatch)
    calendar = capture("2026-08-26T00:45:00+00:00")
    assert calendar.receipt["target_trade_date"] == "20260825"
    assert calendar.receipt["ordered_open_dates"][-1] == "20260826"
    args["calendar_receipt_sha"] = write(args["calendar_receipt_path"], calendar.receipt)
    prepared = prepare_store_plan(args)
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260825",
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref={"path": "fixture-release.json", "sha256": "a" * 64},
    )
    adapter.execute(adapter.template())
    assert prepare_store_plan(args)["status"] == "PLAN_ADOPTED"
