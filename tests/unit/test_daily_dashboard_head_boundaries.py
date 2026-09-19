"""Bounded head traversal and native publication lifecycle, without EOD admission claims."""

from datetime import datetime, timedelta
from dataclasses import fields
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import pickle
import multiprocessing

import pytest

from scripts import daily_dashboard_publication as pub
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.dashboard_publication_guard import (
    sealed_publication_scope,
    publication_scope,
)
from test_daily_dashboard_sealed_publication import fixture, publish, inventory


def ref(day):
    return {
        "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
        "sha256": hashlib.sha256(day.encode()).hexdigest(),
    }


def inspected(day, previous, bootstrap=None):
    recipe = {"previous_completion_ref": previous, "bootstrap_ref": bootstrap}
    return {
        "recorded_completion": {"trade_date": day},
        "completed_handoff_snapshot": SimpleNamespace(document=lambda _: recipe),
    }


def test_explicit_head_chain_stops_at_exact_selected_completion(tmp_path, monkeypatch):
    rows = {
        "20260903": inspected("20260903", ref("20260902")),
        "20260902": inspected("20260902", ref("20260901")),
    }
    seen = []
    monkeypatch.setattr(pub, "_replay", lambda root, r: seen.append(r))
    monkeypatch.setattr(pub, "_inspect", lambda root, r: rows[Path(r["path"]).parent.name])
    chain = pub._predecessors(
        tmp_path, inspected("20260904", ref("20260903")), {"completion_ref": ref("20260901")}
    )
    assert list(chain) == seen == [ref("20260903"), ref("20260902"), ref("20260901")]
    seen.clear()
    assert pub._predecessors(tmp_path, inspected("20260904", ref("20260903")), None) == (
        ref("20260903"),
    )
    assert len(seen) == 1


@pytest.mark.parametrize("fault", ["cycle", "missing", "unreached", "overflow"])
def test_bad_predecessor_chains_fail_with_bounded_reads(tmp_path, monkeypatch, fault):
    start = datetime(2026, 9, 10)
    calls = []
    monkeypatch.setattr(pub, "_replay", lambda root, r: calls.append(r))

    def prior(root, r):
        day = Path(r["path"]).parent.name
        date = datetime.strptime(day, "%Y%m%d")
        if fault == "missing":
            return {"completed_handoff_snapshot": None}
        if fault == "unreached":
            return inspected(day, None)
        return inspected(
            day,
            ref(day) if fault == "cycle" else ref((date - timedelta(days=1)).strftime("%Y%m%d")),
        )

    monkeypatch.setattr(pub, "_inspect", prior)
    with pytest.raises(ContractError):
        pub._predecessors(
            tmp_path,
            inspected(start.strftime("%Y%m%d"), ref("20260909")),
            {"completion_ref": ref("20260101")},
        )
    assert len(calls) <= 32


def test_first_head_cannot_use_missing_bootstrap(tmp_path):
    with pytest.raises(ContractError, match="ANCESTRY_CONFLICT"):
        pub._predecessors(tmp_path, inspected("20260904", None), None)


def test_capability_is_scoped_nonserializable_and_path_confined(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    with node.journal.locked(), sealed_publication_scope(tmp_path, proof) as cap:
        with pytest.raises(TypeError):
            pickle.dumps(cap)
        assert not hasattr(cap, "bind")
        with pytest.raises(TypeError):
            cap.expected["bad.json"] = b"bad"
        with pytest.raises(ContractError, match="PATH_NOT_AUTHORIZED"):
            cap.bytes_for(tmp_path / "../outside.json")
        with publication_scope(tmp_path, cap):
            cap.require()
        with pytest.raises(ContractError, match="EOD_PUBLICATION_REQUIRED"):
            with publication_scope(tmp_path):
                pass
    with pytest.raises(ContractError, match="EXPIRED"):
        cap.require()


def test_expired_recovery_preserves_existing_intent_and_serving_bytes(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    import scripts.cn_dashboard_v2_selector as selector

    with monkeypatch.context() as fault:
        fault.setattr(
            selector,
            "publish_selector",
            lambda *a, **k: (_ for _ in ()).throw(OSError("before selector")),
        )
        with pytest.raises(OSError):
            publish(node, proof)
    before = inventory(tmp_path)
    until = pub.instant(json.loads(proof.v2)["freshness"]["valid_through"])

    class ExpiredClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = until + timedelta(seconds=1)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(pub, "datetime", ExpiredClock)
    with pytest.raises(ContractError, match="PUBLICATION_EXPIRED"):
        publish(node, proof)
    assert inventory(tmp_path) == before


def _race_worker(admitted_fixture_fields, start, queue):
    try:
        # Independent processes explicitly recreate the controlled admission seam.
        proof = pub.VerifiedDashboard(**admitted_fixture_fields, _key=pub._PROOF_KEY)
        if not start.wait(10):
            raise RuntimeError("test start timeout")
        journal = DailyJournal(str(proof.workspace), proof.day)
        with journal.locked(), sealed_publication_scope(proof.workspace, proof) as cap:
            result = pub._publish_locked(proof, cap, journal)
        queue.put(("published", result))
    except Exception as exc:
        queue.put(("rejected", str(exc)))


def test_concurrent_preflight_snapshots_have_only_one_commit(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    ctx = multiprocessing.get_context("spawn")
    start, queue = ctx.Event(), ctx.Queue()
    values = {f.name: getattr(proof, f.name) for f in fields(proof) if not f.name.startswith("_")}
    children = [ctx.Process(target=_race_worker, args=(values, start, queue)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        start.set()
        results = [queue.get(timeout=20) for _ in children]
        for child in children:
            child.join(timeout=10)
        assert all(not child.is_alive() and child.exitcode == 0 for child in children)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        queue.close()
    assert sorted(row[0] for row in results) == ["published", "rejected"]
    assert "HEAD_PREIMAGE_CHANGED" in next(row[1] for row in results if row[0] == "rejected")
    before = inventory(tmp_path)
    publish(node, proof)
    assert inventory(tmp_path) == before
