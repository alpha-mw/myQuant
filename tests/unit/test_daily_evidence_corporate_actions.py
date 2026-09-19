"""Native closed event ancestry and non-executable corporate-action evidence."""

import hashlib
import json
import pytest
from _native_daily_store_fixture import NativeStoreFixture, DAYS, STOCKS, write
from quant_investor.operations.corporate_actions import CorporateActionAdapter, EVENT_ROOT
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.strategy_records.event_store import (
    load_historical_generation,
    StrategyEventStoreError,
)


def ref(root, path):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_native_event_ancestor_and_corporate_replay(tmp_path):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    sha = fixture.event_pointer
    release_path = tmp_path / "fixtures/release.json"
    write(release_path, {"synthetic": True})
    manifest = tmp_path / "data/parquet/cn/_snapshots/synthetic-20260824.json"
    frame = (
        tmp_path
        / f"data/parquet/cn/_snapshots/synthetic-20260824/serving/bars/symbol={STOCKS[0]}/bars.parquet"
    )
    journal = DailyJournal(str(tmp_path), "20260824")
    adapter = CorporateActionAdapter(
        workspace=str(tmp_path),
        journal=journal,
        event_pointer_sha256=sha,
        release_ref=ref(tmp_path, release_path),
        previous_trade_date="20260821",
        calendar_ref=ref(tmp_path, fixture.calendar_path),
        market_snapshot_ref=ref(tmp_path, manifest),
        market_refs={STOCKS[0]: ref(tmp_path, frame)},
    )
    with journal.locked():
        adapter.prepare()
        request = adapter.template()
        assert adapter.probe(request).safe_to_execute
        adapter.execute(request)
        original = adapter.probe(request).outcome
    summary = json.loads((tmp_path / original.output_refs["financial_events"]["path"]).read_bytes())
    assert summary["financial_event_state"] == "VERIFIED_CLOSED_EMPTY"
    assert (
        summary["adjustment_checks"][0]["threshold_state"]
        == "NON_EXECUTABLE_MISSING_ADJUSTMENT_EVIDENCE"
    )
    assert summary["threshold_anchor_mutation"] is False
    fixture.advance(DAYS[1])
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert adapter.probe(request).outcome == original
    historical = load_historical_generation(tmp_path / EVENT_ROOT, expected_pointer_sha256=sha)
    assert historical["pointer_sha256"] == sha
    assert len(historical["closures"]) == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    history = tmp_path / EVENT_ROOT / f"pointer_history/{sha}.json"
    history.write_bytes(history.read_bytes() + b" ")
    with pytest.raises(StrategyEventStoreError, match="SHA mismatch"):
        load_historical_generation(tmp_path / EVENT_ROOT, expected_pointer_sha256=sha)


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (1.0, 1.0, "NO_ADJUSTMENT_FACTOR_CHANGE"),
        (1.0, 2.0, "NON_EXECUTABLE_CORPORATE_ACTION_UNRECONCILED"),
        (None, 1.0, "NON_EXECUTABLE_MISSING_ADJUSTMENT_EVIDENCE"),
        (0.0, 1.0, "NON_EXECUTABLE_MISSING_ADJUSTMENT_EVIDENCE"),
    ],
)
def test_adjustment_state_is_evidence_bound_and_non_executable(tmp_path, before, after, expected):
    fixture = NativeStoreFixture(tmp_path)
    symbol = STOCKS[0]
    fixture.advance(DAYS[0], adjustment_factors={symbol: {"2026-08-21": before, DAYS[0]: after}})
    release = tmp_path / "fixtures/release.json"
    write(release, {"synthetic": True})
    manifest = tmp_path / "data/parquet/cn/_snapshots/synthetic-20260824.json"
    frame = (
        tmp_path
        / f"data/parquet/cn/_snapshots/synthetic-20260824/serving/bars/symbol={symbol}/bars.parquet"
    )
    journal = DailyJournal(str(tmp_path), "20260824")
    adapter = CorporateActionAdapter(
        workspace=str(tmp_path),
        journal=journal,
        event_pointer_sha256=fixture.event_pointer,
        release_ref=ref(tmp_path, release),
        previous_trade_date="20260821",
        calendar_ref=ref(tmp_path, fixture.calendar_path),
        market_snapshot_ref=ref(tmp_path, manifest),
        market_refs={symbol: ref(tmp_path, frame)},
    )
    original = frame.read_bytes()
    with journal.locked():
        adapter.prepare()
        adapter.execute(adapter.template())
        output = adapter.probe(adapter.template()).outcome.output_refs["financial_events"]
    value = json.loads((tmp_path / output["path"]).read_bytes())
    assert value["adjustment_checks"][0]["threshold_state"] == expected
    assert value["threshold_anchor_mutation"] is False
    assert frame.read_bytes() == original
    adapter.market_refs[symbol] = {**adapter.market_refs[symbol], "sha256": "0" * 64}
    from quant_investor.operations.daily_contract import ContractError

    with pytest.raises(ContractError, match="RECIPE_BINDING_MISMATCH"):
        adapter.probe(adapter.template())


@pytest.mark.parametrize("head", ["missing", "corrupt"])
def test_prepared_corporate_probe_uses_retained_pointer_only(tmp_path, monkeypatch, head):
    fixture = NativeStoreFixture(tmp_path)
    fixture.advance(DAYS[0])
    release = tmp_path / "fixtures/release.json"
    write(release, {"synthetic": True})
    journal = DailyJournal(str(tmp_path), "20260824")
    adapter = CorporateActionAdapter(
        workspace=str(tmp_path),
        journal=journal,
        event_pointer_sha256=fixture.event_pointer,
        release_ref=ref(tmp_path, release),
    )
    with journal.locked():
        adapter.prepare()
        request = adapter.template()
        adapter.execute(request)
        expected = adapter.probe(request).outcome
    current = tmp_path / EVENT_ROOT / "current.v1.json"
    if head == "missing":
        current.unlink()
    else:
        current.write_bytes(b"broken-current-pointer")
    import quant_investor.operations.corporate_actions as module

    def forbidden(*args, **kwargs):
        pytest.fail("prepared corporate replay traversed current ancestry")

    monkeypatch.setattr(module, "load_historical_generation", forbidden)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert adapter.probe(request).outcome == expected
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
