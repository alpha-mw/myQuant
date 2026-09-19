"""Native Store plan/preimage custody; actual file/generation/ledger validators."""

import hashlib
import json

import pytest

from _native_daily_store_fixture import NativeStoreFixture, DAYS, write
from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.portfolio_binding import (
    freeze_portfolio_state,
    PortfolioSource,
    NativePortfolioSource,
    retain_portfolio_source,
)
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter


def setup(root):
    fixture = NativeStoreFixture(root)
    args = fixture.advance(DAYS[0])
    prepared = prepare_store_plan(args)
    plan_ref = {"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]}
    journal = DailyJournal(str(root), DAYS[0].replace("-", ""))
    return fixture, args, prepared, plan_ref, journal


def test_native_book_can_be_retained_before_cutoff_without_a_state(tmp_path):
    _, _, _, plan_ref, journal = setup(tmp_path)
    source = NativePortfolioSource(
        workspace=tmp_path, trade_date=journal.trade_date, store_plan_ref=plan_ref
    )
    with journal.locked():
        fields = retain_portfolio_source(journal=journal, source=source)
        assert "as_of" not in fields and not hasattr(source, "as_of")
        assert journal.storage.read(source.state_path) is None
        assert (
            journal.storage.read(source.pointer_path).byte_sha256
            == fields["frozen_pointer_ref"]["sha256"]
        )
        fixed = PortfolioSource(
            workspace=tmp_path,
            trade_date=journal.trade_date,
            store_plan_ref=plan_ref,
            as_of=DAYS[0] + "T13:30:00Z",
        )
        assert fixed.source_fields(fields["frozen_pointer_ref"]) == {"as_of": fixed.as_of, **fields}
        assert retain_portfolio_source(journal=journal, source=source) == fields
        assert journal.storage.read(source.state_path) is None


def test_source_custody_rejects_another_day_lock(tmp_path):
    _, _, _, plan_ref, journal = setup(tmp_path)
    source = NativePortfolioSource(
        workspace=tmp_path, trade_date=journal.trade_date, store_plan_ref=plan_ref
    )
    other = DailyJournal(str(tmp_path), DAYS[1].replace("-", ""))
    with other.locked(), pytest.raises(ValueError, match="JOURNAL_MISMATCH"):
        retain_portfolio_source(journal=other, source=source)
    assert journal.storage.read(source.pointer_path) is None


def test_native_preimage_replays_after_store_advances_without_current_selection(
    tmp_path, monkeypatch
):
    fixture, args, prepared, plan_ref, journal = setup(tmp_path)
    as_of = DAYS[0] + "T13:30:00Z"
    with journal.locked():
        frozen = freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of)
    release = {
        "path": "release.json",
        "sha256": write(tmp_path / "release.json", {"synthetic": True}),
    }
    adapter = StoreCloseAdapter(
        arguments=args, trade_date=journal.trade_date, plan_ref=plan_ref, release_ref=release
    )
    adapter.execute(adapter.template())
    original = SecureSystemStorage.read_workspace_file_bytes

    def no_current(self, path, **kwargs):
        if str(path).endswith("_record_store/current.v1.json"):
            pytest.fail("portfolio replay selected current Store")
        return original(self, path, **kwargs)

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", no_current)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    with journal.locked():
        assert (
            freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of) == frozen
        )
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert frozen["state"]["payload"]["prospective"] is False


def test_partial_pointer_custody_adopts_exact_plan_source(tmp_path, monkeypatch):
    _, _, _, plan_ref, journal = setup(tmp_path)
    as_of = DAYS[0] + "T13:30:00Z"
    original = journal.storage.write

    def crash(path, raw, **kwargs):
        if str(path).endswith("state.v1.json"):
            raise RuntimeError("after pointer custody")
        return original(path, raw, **kwargs)

    with journal.locked():
        with monkeypatch.context() as fault:
            fault.setattr(journal.storage, "write", crash)
            with pytest.raises(RuntimeError, match="after pointer custody"):
                freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of)
        frozen = freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of)
        assert (
            freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of) == frozen
        )


def test_changed_current_preimage_rejects_before_state_publication(tmp_path, monkeypatch):
    fixture, _, _, plan_ref, journal = setup(tmp_path)
    original = PortfolioSource.source_fields
    current = fixture.root / "_record_store/current.v1.json"

    def moved(self, pointer_ref):
        result = original(self, pointer_ref)
        current.write_bytes(current.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(PortfolioSource, "source_fields", moved)
    with journal.locked():
        with pytest.raises(ValueError, match="PREIMAGE_CHANGED_DURING_CAPTURE"):
            freeze_portfolio_state(
                journal=journal, store_plan_ref=plan_ref, as_of=DAYS[0] + "T13:30:00Z"
            )
    assert not list(
        tmp_path.glob("results/operations/daily_production/CN/*/inputs/portfolio/*/state.v1.json")
    )


def test_rewritten_state_cannot_change_native_positions(tmp_path):
    _, _, _, plan_ref, journal = setup(tmp_path)
    as_of = DAYS[0] + "T13:30:00Z"
    with journal.locked():
        frozen = freeze_portfolio_state(journal=journal, store_plan_ref=plan_ref, as_of=as_of)
    state = json.loads((tmp_path / frozen["state_ref"]["path"]).read_bytes())
    state["payload"]["positions"][0]["shares"] = "9999"
    forged = seal_artifact(state["kind"], state["payload"], created_at=state["created_at"])
    raw = canonical_json_bytes(forged)
    (tmp_path / frozen["state_ref"]["path"]).write_bytes(raw)
    ref = {**frozen["state_ref"], "sha256": hashlib.sha256(raw).hexdigest()}
    source = PortfolioSource(
        workspace=tmp_path, trade_date=journal.trade_date, as_of=as_of, store_plan_ref=plan_ref
    )
    with pytest.raises(ValueError, match="STATE_REPLAY_MISMATCH"):
        source.replay(ref)
