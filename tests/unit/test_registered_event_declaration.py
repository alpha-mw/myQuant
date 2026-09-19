"""Native synthetic registered BUY custody; no real owner/account/provider facts."""

from argparse import Namespace
from copy import deepcopy
from datetime import timedelta
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from _registered_event_fixture import build, NOW, ref
from _native_daily_store_fixture import write
from scripts import manage_cn_strategy_records as manager
from scripts import registered_daily_event_sources as native
from scripts.registered_daily_event_sources import read_declaration
from quant_investor.strategy_records import store, event_store
from quant_investor.strategy_records import registered_event_contracts as contracts


def inventory(root):
    return {
        p.relative_to(root).as_posix(): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def publish(fixture, monkeypatch):
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW)
    return manager.command_publish_registered_event_declaration(fixture["args"])


def test_retired_publication_helper_rejects_before_any_io(tmp_path):
    fixture = build(tmp_path)
    prepared = native.prepare_publication(
        workspace=tmp_path,
        owner_fact_ref={
            "path": fixture["args"].owner_fact,
            "sha256": fixture["args"].owner_fact_sha256,
        },
        expected_pointer_sha=fixture["writer_sha"],
    )
    before = inventory(tmp_path)
    directories = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_dir()}
    for candidate in (prepared, None):
        with pytest.raises(native.Error, match="^REGISTERED_EVENT_MANAGER_PUBLICATION_REQUIRED$"):
            native.publish_prepared(candidate, registered_at="unused")
    assert inventory(tmp_path) == before
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_dir()} == directories


def test_manager_holds_same_real_lock_through_publication_and_readback(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    lock = fixture["book"].root / "_record_store/.operation.v2.lock"
    identity = (lock.stat().st_dev, lock.stat().st_ino)
    observed = []

    def check(stage):
        assert (lock.stat().st_dev, lock.stat().st_ino) == identity
        program = """import fcntl,os,sys
fd=os.open(sys.argv[1],os.O_RDWR)
try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(0)
raise SystemExit(9)
"""
        assert (
            subprocess.run([sys.executable, "-c", program, str(lock)], timeout=10).returncode == 0
        )
        observed.append(stage)

    mkdir, write_once, readback = os.mkdir, store._write_exact_once, native.read_declaration

    def guarded_mkdir(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and path in ("registered", fixture["writer_sha"]):
            check("mkdir:" + path)
        return mkdir(path, *args, **kwargs)

    def guarded_write(path, raw):
        check("before:" + path.name)
        write_once(path, raw)
        check("after:" + path.name)

    def guarded_read(**kwargs):
        check("before:readback")
        proof = readback(**kwargs)
        check("after:readback")
        return proof

    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    monkeypatch.setattr(store, "_write_exact_once", guarded_write)
    monkeypatch.setattr(native, "read_declaration", guarded_read)
    assert publish(fixture, monkeypatch)["status"] == "PUBLISHED"
    assert len(observed) == 8
    assert observed[-2:] == ["before:readback", "after:readback"]
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_pointer_only_interruption_retries_exact_immutable_custody(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    paths = contracts.paths(fixture["writer_sha"])
    original = store._write_exact_once

    def interrupt(path, raw):
        if path == tmp_path / paths["declaration"]:
            raise RuntimeError("interrupted between immutable writes")
        original(path, raw)

    monkeypatch.setattr(store, "_write_exact_once", interrupt)
    before = inventory(tmp_path)
    with pytest.raises(RuntimeError, match="between immutable writes"):
        publish(fixture, monkeypatch)
    partial = inventory(tmp_path)
    assert set(partial) - set(before) == {paths["pointer"]}
    assert all(partial[path] == state for path, state in before.items())
    monkeypatch.setattr(store, "_write_exact_once", original)
    result = publish(fixture, monkeypatch)
    after = inventory(tmp_path)
    assert result["status"] == "PUBLISHED"
    assert all(after[path] == state for path, state in partial.items())
    assert publish(fixture, monkeypatch)["status"] == "NO_ACTION"
    assert inventory(tmp_path) == after


@pytest.mark.parametrize("fault", ["lock_replaced", "lock_released", "source_changed"])
def test_lost_custody_between_writes_blocks_declaration(tmp_path, monkeypatch, fault):
    fixture = build(tmp_path)
    paths = contracts.paths(fixture["writer_sha"])
    lock = fixture["book"].root / "_record_store/.operation.v2.lock"
    operation_lock = manager._operation_lock
    descriptor = None

    @contextmanager
    def capture(root):
        nonlocal descriptor
        with operation_lock(root) as descriptor:
            yield descriptor

    monkeypatch.setattr(manager, "_operation_lock", capture)
    original = store._write_exact_once
    writes = []

    def inject(path, raw):
        original(path, raw)
        writes.append(path)
        if fault == "lock_replaced":
            lock.unlink()
            lock.touch(mode=0o600)
        elif fault == "lock_released":
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        else:
            source = tmp_path / fixture["args"].owner_fact
            source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(store, "_write_exact_once", inject)
    with pytest.raises((manager.StrategyRecordStoreError, ValueError)):
        publish(fixture, monkeypatch)
    assert writes == [tmp_path / paths["pointer"]]
    assert not (tmp_path / paths["declaration"]).exists()


def empty_args(fixture):
    book = fixture["book"]
    return Namespace(
        project_root=str(book.project),
        record_root=str(book.root),
        trade_date="2026-08-25",
        expected_event_pointer_sha256=event_store.pointer_sha256(book.root / "_event_store"),
        generation_id="must-not-publish-empty",
        policy_path=book.policy_path,
        policy_sha256=book.policy_sha,
        maintenance_receipt=None,
        maintenance_receipt_sha256=None,
        calendar_receipt="not-read.json",
        calendar_receipt_sha256="a" * 64,
        raw_calendar="not-read.raw.json",
        raw_calendar_sha256="b" * 64,
    )


def test_native_registered_buy_declaration_and_frozen_read(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    before = inventory(tmp_path)
    monkeypatch.setattr(store, "publish_catalog", lambda **kw: pytest.fail("second financial CAS"))
    monkeypatch.setattr(
        manager, "publish_catalog", lambda **kw: pytest.fail("manager financial CAS")
    )
    monkeypatch.setattr(
        manager, "command_seal_publish", lambda *a, **kw: pytest.fail("financial writer")
    )
    result = publish(fixture, monkeypatch)
    assert result["status"] == "PUBLISHED"
    proof = read_declaration(workspace=tmp_path, declaration_ref=result["declaration_ref"])
    assert proof["writer"]["record"]["record"] == "20260825_1000"
    assert proof["profile"]["trade_count"] == 1
    assert proof["declaration"]["broker_statement_verified"] is False
    for path, state in before.items():
        assert inventory(tmp_path)[path] == state
    assert set(inventory(tmp_path)) - set(before) == set(
        contracts.paths(fixture["writer_sha"]).values()
    )


def test_repeat_preserves_original_registration_and_every_file(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    first = publish(fixture, monkeypatch)
    before = inventory(tmp_path)
    monkeypatch.setattr(manager, "_manager_utc_now", lambda: NOW + timedelta(days=3))
    again = manager.command_publish_registered_event_declaration(fixture["args"])
    assert again["status"] == "NO_ACTION"
    assert again["registered_at"] == first["registered_at"]
    assert again["declaration_ref"] == first["declaration_ref"]
    assert inventory(tmp_path) == before


def test_frozen_replay_has_no_current_pointer_or_writer_dependency(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    result = publish(fixture, monkeypatch)
    root = fixture["book"].root
    for path in (
        root / "_record_store/current.v1.json",
        root / "_event_store/current.v1.json",
        tmp_path / "data/parquet/cn/_latest.json",
        tmp_path / "data/parquet/cn/benchmarks/_latest.json",
    ):
        path.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("frozen replay touched current state or a writer")

    monkeypatch.setattr(store, "load_registered_catalog", forbidden)
    monkeypatch.setattr(event_store, "load_generation", forbidden)
    monkeypatch.setattr(store, "publish_catalog", forbidden)
    monkeypatch.setattr(store, "_write_exact_once", forbidden)
    before = inventory(tmp_path)
    proof = read_declaration(workspace=tmp_path, declaration_ref=result["declaration_ref"])
    assert (
        proof["writer"]["pointer"]["previous_pointer_sha256"] == fixture["baseline_ref"]["sha256"]
    )
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("target", ["pointer", "baseline", "ledger", "performance"])
def test_corrupt_frozen_source_rejects_without_reconstruction(tmp_path, monkeypatch, target):
    fixture = build(tmp_path)
    result = publish(fixture, monkeypatch)
    proof = read_declaration(workspace=tmp_path, declaration_ref=result["declaration_ref"])
    refs = {
        "pointer": proof["declaration"]["writer_store_pointer_ref"],
        "baseline": fixture["baseline_ref"],
        "ledger": proof["declaration"]["writer_record_refs"]["ledger"],
        "performance": proof["declaration"]["writer_record_refs"]["performance_series"],
    }
    (tmp_path / refs[target]["path"]).write_bytes(b"corrupt")
    before = inventory(tmp_path)
    with pytest.raises((ValueError, OSError, store.StrategyRecordStoreError)):
        read_declaration(workspace=tmp_path, declaration_ref=result["declaration_ref"])
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_domain",
        "owner",
        "date",
        "authority",
        "broker",
        "future",
        "wrong_pointer",
        "current_baseline",
        "unbound_ref",
    ],
)
def test_owner_fact_conflicts_block_before_any_publication(tmp_path, monkeypatch, fault):
    fixture = build(tmp_path)
    value = deepcopy(fixture["fact"])
    if fault == "missing_domain":
        del value["domains"]["orders"]
    elif fault == "owner":
        value["owner"] = "AnotherOwner"
    elif fault == "date":
        value["trade_date"] = "2026-08-26"
    elif fault == "authority":
        value["authority"]["trade"] = True
    elif fault == "broker":
        value["broker_statement_verified"] = True
    elif fault == "future":
        value["owner_declared_at"] = "2026-08-26T12:00:00Z"
    elif fault == "wrong_pointer":
        value["writer_pointer_sha256"] = "0" * 64
    elif fault == "current_baseline":
        value["baseline_store_pointer_ref"] = ref(
            tmp_path, fixture["book"].root / "_record_store/current.v1.json"
        )
    else:
        value["domains"]["fills"]["fact_refs"] = [fixture["baseline_ref"]]
    fixture["args"].owner_fact_sha256 = write(
        Path(fixture["args"].owner_fact), contracts.seal(value)
    )
    before = inventory(tmp_path)
    with pytest.raises((ValueError, store.StrategyRecordStoreError)):
        publish(fixture, monkeypatch)
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("declared", [False, True])
def test_registered_trade_blocks_empty_even_before_declaration(tmp_path, monkeypatch, declared):
    fixture = build(tmp_path)
    if declared:
        publish(fixture, monkeypatch)
    before = inventory(tmp_path)
    with pytest.raises(store.StrategyRecordStoreError, match="NONEMPTY_DAY_CANNOT_BE_EMPTY"):
        manager.command_publish_daily_event_closure(empty_args(fixture))
    assert inventory(tmp_path) == before


def test_existing_empty_blocks_declaration_and_cannot_return_no_action(tmp_path, monkeypatch):
    fixture = build(tmp_path, empty_first=True)
    before = inventory(tmp_path)
    with pytest.raises(store.StrategyRecordStoreError, match="EMPTY_CLOSURE_CONFLICT"):
        publish(fixture, monkeypatch)
    with pytest.raises(store.StrategyRecordStoreError, match="NONEMPTY_DAY_CANNOT_BE_EMPTY"):
        manager.command_publish_daily_event_closure(empty_args(fixture))
    assert inventory(tmp_path) == before


def test_changed_fact_cannot_overwrite_same_registered_source(tmp_path, monkeypatch):
    fixture = build(tmp_path)
    first = publish(fixture, monkeypatch)
    changed = {**fixture["fact"], "owner_fact_id": "changed"}
    path = tmp_path / "fixtures/changed-fact.json"
    sha = write(path, contracts.seal(changed))
    fixture["args"].owner_fact, fixture["args"].owner_fact_sha256 = str(path), sha
    before = inventory(tmp_path)
    with pytest.raises(store.StrategyRecordStoreError, match="FACT_IDENTITY_CONFLICT"):
        publish(fixture, monkeypatch)
    assert inventory(tmp_path) == before
    assert read_declaration(workspace=tmp_path, declaration_ref=first["declaration_ref"])


def test_cli_waits_for_real_record_operation_lock_without_double_lock(tmp_path):
    fixture = build(tmp_path)
    ready, entered = tmp_path / "ready", tmp_path / "entered"
    argv = [
        "publish-registered-event-declaration",
        "--project-root",
        str(tmp_path),
        "--record-root",
        fixture["args"].record_root,
        "--owner-fact",
        fixture["args"].owner_fact,
        "--owner-fact-sha256",
        fixture["args"].owner_fact_sha256,
        "--expected-pointer-sha",
        fixture["writer_sha"],
    ]
    program = """import json,sys
from pathlib import Path
from scripts import manage_cn_strategy_records as manager
from scripts import registered_daily_event_sources as native
original=native.prepare_publication
def observed(**kwargs):
    Path(sys.argv[2]).write_text("entered")
    return original(**kwargs)
native.prepare_publication=observed
Path(sys.argv[1]).write_text("ready")
raise SystemExit(manager.main(json.loads(sys.argv[3])))
"""
    child = None
    try:
        with manager._operation_lock(fixture["book"].root):
            child = subprocess.Popen(
                [sys.executable, "-c", program, str(ready), str(entered), json.dumps(argv)],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists()
            time.sleep(0.1)
            assert not entered.exists()
            assert child.poll() is None
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stderr.decode()
        assert json.loads(stdout)["status"] == "PUBLISHED"
        assert entered.exists()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize(
    "fault",
    [
        "sell",
        "fees",
        "value",
        "basis",
        "average",
        "cash",
        "quantity",
        "cost",
        "funding",
        "corporate",
        "missing_fee",
        "duplicate",
        "nonfinite",
    ],
)
def test_financial_attribution_rejects_unbound_or_unsupported_changes(tmp_path, fault):
    fixture = build(tmp_path)
    prepared = native.prepare_publication(
        workspace=tmp_path,
        owner_fact_ref=fixture["fact_ref"],
        expected_pointer_sha=fixture["writer_sha"],
    )
    pair = prepared["pair"]
    manual = deepcopy(pair["writer"]["manual"])
    writer = deepcopy(pair["writer"]["record"])
    trade = manual["applied_owner_declared_trades"][0]
    if fault == "sell":
        trade["side"] = "SELL"
    elif fault == "fees":
        trade["final_total_fee_cny"] += 1
    elif fault == "value":
        trade["trade_value"] += 1
    elif fault == "basis":
        trade["cost_basis_cny"] += 1
    elif fault == "average":
        trade["avg_cost_cny_per_share"] += 1
    elif fault == "cash":
        writer["accounting"]["cash_after"] += 1
    elif fault == "quantity":
        writer["positions"][0]["shares"] += 1
    elif fault == "cost":
        writer["positions"][0]["cost_basis"] += 1
    elif fault == "funding":
        writer["funding"] = {"amount": 1}
    elif fault == "corporate":
        manual["corporate_action_application_ref"] = fixture["fact_ref"]
    elif fault == "missing_fee":
        del trade["commission_cny"]
    elif fault == "duplicate":
        manual["applied_owner_declared_trades"].append(deepcopy(trade))
    else:
        trade["shares"] = "NaN"
    before = inventory(tmp_path)
    with pytest.raises(contracts.RegisteredEventError):
        contracts.validate_buy_transition(
            baseline=pair["baseline"]["record"],
            writer=writer,
            manual=manual,
            fact=fixture["fact"],
            record_refs=pair["writer"]["refs"],
        )
    assert inventory(tmp_path) == before


def test_companion_only_recovery_keeps_original_pointer_and_needs_current_source(
    tmp_path, monkeypatch
):
    fixture = build(tmp_path)
    original = store._write_exact_once

    def crash(path, raw):
        if path.name == "declaration.v1.json":
            raise RuntimeError("synthetic crash before declaration")
        return original(path, raw)

    monkeypatch.setattr(store, "_write_exact_once", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        publish(fixture, monkeypatch)
    paths = contracts.paths(fixture["writer_sha"])
    companion = tmp_path / paths["pointer"]
    original_pointer = companion.read_bytes()
    assert not (tmp_path / paths["declaration"]).exists()
    assert hashlib.sha256(original_pointer).hexdigest() == fixture["writer_sha"]
    monkeypatch.setattr(store, "_write_exact_once", original)
    current = fixture["book"].root / "_record_store/current.v1.json"
    current.write_bytes(b"corrupt advanced current")
    with pytest.raises((ValueError, store.StrategyRecordStoreError)):
        publish(fixture, monkeypatch)
    assert companion.read_bytes() == original_pointer
    assert not (tmp_path / paths["declaration"]).exists()
    current.write_bytes(original_pointer)
    assert publish(fixture, monkeypatch)["status"] == "PUBLISHED"
    assert companion.read_bytes() == original_pointer


@pytest.mark.parametrize("target", ["owner_file", "registered_directory"])
def test_unsafe_source_or_output_path_rejects_before_publication(tmp_path, monkeypatch, target):
    fixture = build(tmp_path)
    if target == "owner_file":
        path = Path(fixture["args"].owner_fact)
        other = path.with_name("owner-actual.json")
        path.rename(other)
        path.symlink_to(other)
    else:
        path = fixture["book"].root / "_event_store/registered"
        other = tmp_path / "outside"
        other.mkdir()
        path.symlink_to(other, target_is_directory=True)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        publish(fixture, monkeypatch)
    assert not (tmp_path / contracts.paths(fixture["writer_sha"])["declaration"]).exists()


def test_later_no_trade_head_does_not_hide_same_day_applied_ancestry(tmp_path, monkeypatch):
    """Guard unit test with an explicit catalog-admission seam; Part C owns publication."""
    fixture = build(tmp_path)
    pointer, catalog = store.load_registered_catalog(fixture["book"].root)
    pointer, catalog = deepcopy(pointer), deepcopy(catalog)
    later = "20260825_1700"
    catalog["lineage_index"].append(
        {
            **catalog["lineage_index"][-1],
            "record_id": later,
            "source_record_id": pointer["active_record_id"],
            "execution_class": "NO_TRADE",
        }
    )
    pointer["active_record_id"] = catalog["active_record_id"] = later
    monkeypatch.setattr(store, "load_catalog_snapshot_bytes", lambda *a, **kw: (pointer, catalog))
    original = native.RegisteredSources.document

    def document(self, reference):
        if reference["path"] == contracts.RECORD_ROOT + "/" + pointer["catalog_path"]:
            return catalog
        return original(self, reference)

    monkeypatch.setattr(native.RegisteredSources, "document", document)
    monkeypatch.setattr(
        native,
        "validate_record",
        lambda *a, **kw: pytest.fail("ancestry guard missed the applied record"),
    )
    with pytest.raises(store.StrategyRecordStoreError, match="NONEMPTY_DAY_CANNOT_BE_EMPTY"):
        native.assert_no_registered_event(workspace=tmp_path, trade_date="2026-08-25")
