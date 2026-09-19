"""Native-v3 Store integration: no monkeypatched writers or source validators."""

import hashlib
import json

from _native_daily_store_fixture import NativeStoreFixture, DAYS, write
from scripts.daily_production_store_adapter import StoreCloseAdapter, prepare_store_plan
from quant_investor.strategy_records.store import load_registered_catalog
from quant_investor.strategy_records.performance import load_performance_history
from scripts import cn_official_close_batch as native
import pytest


def run_sessions(project):
    fixture = NativeStoreFixture(project)
    release = {
        "path": "fixtures/release.json",
        "sha256": write(project / "fixtures/release.json", {"synthetic": True}),
    }
    adapters = []
    proof = []
    for day in DAYS:
        args = fixture.advance(day)
        prepared = prepare_store_plan(args)
        assert prepared["status"] == "PLAN_PREPARED"
        adapter = StoreCloseAdapter(
            arguments=args,
            trade_date=day.replace("-", ""),
            plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
            release_ref=release,
        )
        assert adapter.probe(adapter.template()).safe_to_execute
        adapter.execute(adapter.template())
        result = adapter.probe(adapter.template())
        assert result.outcome.state.value == "SUCCEEDED"
        pointer, catalog = load_registered_catalog(fixture.root)
        perf = load_performance_history(fixture.root, catalog["performance_history_ref"])
        assert perf["rows"][-1]["valuation_date"] == day
        proof.append(
            {
                "day": day,
                "pointer_sha256": hashlib.sha256(
                    (fixture.root / "_record_store/current.v1.json").read_bytes()
                ).hexdigest(),
                "output_refs": result.outcome.output_refs,
                "plan": prepared,
            }
        )
        adapters.append(adapter)
    # Replay every historical day after the head has advanced through all five.
    before = {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in project.rglob("*")
        if p.is_file()
    }
    for adapter, row in zip(adapters, proof):
        result = adapter.probe(adapter.template())
        assert result.outcome.output_refs == row["output_refs"]
    after = {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in project.rglob("*")
        if p.is_file()
    }
    assert before == after
    return {
        "synthetic": True,
        "scope": "NATIVE_STORE_SEGMENT_ONLY",
        "full_dag_proof": False,
        "sessions": proof,
        "replay_zero_writes": True,
        "real_unattended_proof": False,
    }


def test_five_native_store_sessions_and_historical_replay(tmp_path):
    proof = run_sessions(tmp_path)
    assert len(proof["sessions"]) == 5
    (tmp_path / "native-store-proof.json").write_text(json.dumps(proof, indent=2))


def make_adapter(fixture, args, release):
    prepared = prepare_store_plan(args)
    return StoreCloseAdapter(
        arguments=args,
        trade_date=prepared["latest_required_close_date"].replace("-", ""),
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref=release,
    )


def test_real_cas_crash_recovers_after_later_day_without_another_cas(tmp_path, monkeypatch):
    fixture = NativeStoreFixture(tmp_path)
    release = {
        "path": "fixtures/release.json",
        "sha256": write(tmp_path / "fixtures/release.json", {"synthetic": True}),
    }
    calls = []
    real_cas = native.publish_catalog

    def counted(*args, **kwargs):
        calls.append(1)
        return real_cas(*args, **kwargs)

    monkeypatch.setattr(native, "publish_catalog", counted)
    first = make_adapter(fixture, fixture.advance(DAYS[0]), release)
    with monkeypatch.context() as faults:

        def crash(*args, **kwargs):
            raise RuntimeError("injected crash after native CAS")

        faults.setattr(native, "_completion_value", crash)
        with pytest.raises(RuntimeError, match="after native CAS"):
            first.execute(first.template())
    assert first.probe(first.template()).recovery_only
    second = make_adapter(fixture, fixture.advance(DAYS[1]), release)
    second.execute(second.template())
    current = (fixture.root / "_record_store/current.v1.json").read_bytes()
    first.execute(first.template())
    assert first.probe(first.template()).outcome.state.value == "SUCCEEDED"
    assert (fixture.root / "_record_store/current.v1.json").read_bytes() == current
    assert len(calls) == 2


def test_native_three_session_backlog_is_one_cas_with_each_date_preserved(tmp_path, monkeypatch):
    fixture = NativeStoreFixture(tmp_path)
    for day in DAYS[:3]:
        args = fixture.advance(day)
    prepared = prepare_store_plan(args)
    assert prepared["missing_dates"] == DAYS[:3]
    calls = []
    real = native.publish_catalog

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(native, "publish_catalog", counted)
    release = {
        "path": "fixtures/release.json",
        "sha256": write(tmp_path / "fixtures/release.json", {"synthetic": True}),
    }
    for day in DAYS[:3]:
        adapter = StoreCloseAdapter(
            arguments=args,
            trade_date=day.replace("-", ""),
            plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
            release_ref=release,
        )
        if day == DAYS[0]:
            adapter.execute(adapter.template())
        outcome = adapter.probe(adapter.template()).outcome
        assert outcome.state.value == "SUCCEEDED"
        manual = json.loads((tmp_path / outcome.output_refs["manual"]["path"]).read_bytes())
        assert manual["valuation_trade_date"] == day.replace("-", "")
    assert len(calls) == 1


def test_stale_event_preimage_blocks_without_store_mutation(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    args = fixture.advance(DAYS[0])
    # Publish Market/benchmarks for day2 but keep the actual registered day1 event input.
    old_event = args["expected_event_pointer_sha"]
    args = fixture.advance(DAYS[1])
    # Read-only preparation bound to the old event generation must reject drift;
    # it cannot silently swap to the new pointer or infer empty dates.
    args["expected_event_pointer_sha"] = old_event
    before = (fixture.root / "_record_store/current.v1.json").read_bytes()
    with pytest.raises(native.StrategyRecordStoreError, match="event pointer preimage mismatch"):
        prepare_store_plan(args)
    assert (fixture.root / "_record_store/current.v1.json").read_bytes() == before


def test_missing_event_day_cannot_be_inferred_empty(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    args = fixture.advance(DAYS[1], publish_events=False)
    before = (fixture.root / "_record_store/current.v1.json").read_bytes()
    with pytest.raises(
        native.StrategyRecordStoreError, match="EVENT_STATE_CLOSURE_MISSING:2026-08-25"
    ):
        prepare_store_plan(args)
    assert (fixture.root / "_record_store/current.v1.json").read_bytes() == before


def test_same_second_record_ids_use_registered_suffix_without_backdating():
    from datetime import datetime, timezone

    planned = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
    ids = native._allocate_record_ids(
        {"records": [{"record_id": "20260907_160000-b01"}, {"record_id": "20260907_160000-b02"}]},
        planned,
        3,
    )
    assert ids == ["20260907_160000-b03", "20260907_160000-b04", "20260907_160000-b05"]


@pytest.mark.parametrize("hour,publication_date", [(15, "2026-09-07"), (16, "2026-09-08")])
def test_native_close_publication_date_uses_shanghai_midnight(tmp_path, hour, publication_date):
    from datetime import datetime, timezone
    from scripts.manage_cn_strategy_records import _operation_lock

    fixture = NativeStoreFixture(tmp_path)
    args = fixture.advance(DAYS[0])
    with _operation_lock(str(fixture.root)):
        prepared = native.close_through_latest(
            **args,
            execute=False,
            prepare_only=True,
            now=datetime(2026, 9, 7, hour, 20, 4, tzinfo=timezone.utc),
        )
    release = {
        "path": "fixtures/release.json",
        "sha256": write(tmp_path / "fixtures/release.json", {"synthetic": True}),
    }
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date=DAYS[0].replace("-", ""),
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref=release,
    )
    adapter.execute(adapter.template())
    outcome = adapter.probe(adapter.template()).outcome
    assert outcome.state.value == "SUCCEEDED"
    manual = json.loads((tmp_path / outcome.output_refs["manual"]["path"]).read_bytes())
    assert (tmp_path / outcome.output_refs["manual"]["path"]).parent.name.startswith(
        publication_date.replace("-", "")
    )
    assert manual["valuation_trade_date"].replace("-", "") == "20260824"
