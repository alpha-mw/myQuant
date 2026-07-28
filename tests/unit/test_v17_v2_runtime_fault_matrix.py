from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Barrier, Thread

import pytest

from quant_investor.v17_v2_runtime import (
    CASMismatchError,
    ExactOnceConflictError,
    LedgerStore,
    SecureStore,
    StorageCommitError,
    TerminalPublicationError,
    TerminalPublisher,
)

from test_v17_v2_ledger_store import build_ledger_chain
from test_v17_v2_terminal_publication import terminal_documents


def test_concurrent_exact_once_has_one_identity(tmp_path: Path) -> None:
    store = SecureStore(tmp_path)
    store.initialize()
    path = "results/v17_shadow/protocol-v2/runs/run-1/events/concurrent.json"
    barrier = Barrier(2)
    outcomes: list[str] = []

    def write(payload: bytes) -> None:
        barrier.wait()
        try:
            store.write_exact_once(path, payload)
            outcomes.append("committed")
        except ExactOnceConflictError:
            outcomes.append("conflict")

    threads = [Thread(target=write, args=(payload,)) for payload in (b"a", b"b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["committed", "conflict"]
    assert store.read(path) in {b"a", b"b"}


def test_ledger_boundary_failure_recovers_without_rewriting_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = build_ledger_chain()
    store = SecureStore(tmp_path)
    store.initialize()
    ledgers = LedgerStore(store)
    ledgers.initialize("run-1", chain[0])
    expected = hashlib.sha256(chain[0]).hexdigest()
    original = store.replace_cas
    attempts = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StorageCommitError("injected before ledger CAS", possibly_committed=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "replace_cas", fail_once)
    with pytest.raises(StorageCommitError):
        ledgers.append("run-1", expected, chain[1])
    immutable = store.read(
        "results/v17_shadow/protocol-v2/runs/run-1/events/ledger-000001.json"
    )
    assert immutable == chain[1]
    ledgers.append("run-1", expected, chain[1])
    assert ledgers.read_chain("run-1") == tuple(chain)


def test_latest_cas_mismatch_writes_no_payload(tmp_path: Path) -> None:
    store = SecureStore(tmp_path)
    store.initialize()
    store.replace_cas(
        "results/v17_shadow/protocol-v2/_latest/shadow.json",
        "EMPTY",
        b"old\n",
    )
    before = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    with pytest.raises(CASMismatchError):
        store.replace_cas(
            "results/v17_shadow/protocol-v2/_latest/shadow.json",
            "0" * 64,
            b"new\n",
        )
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_terminal_latest_boundary_failure_recovers_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = build_ledger_chain(terminal=True)
    store = SecureStore(tmp_path)
    store.initialize()
    ledgers = LedgerStore(store)
    ledgers.initialize("run-1", chain[0])
    for predecessor, successor in zip(chain, chain[1:]):
        ledgers.append("run-1", hashlib.sha256(predecessor).hexdigest(), successor)
    outcome_path, outcome_bytes, latest_bytes = terminal_documents(chain[-1])
    publisher = TerminalPublisher(store)
    original = store.replace_cas
    attempts = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StorageCommitError("injected latest boundary", possibly_committed=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "replace_cas", fail_once)
    with pytest.raises(TerminalPublicationError) as raised:
        publisher.publish(
            "run-1",
            hashlib.sha256(chain[-1]).hexdigest(),
            outcome_path,
            outcome_bytes,
            "EMPTY",
            latest_bytes,
        )
    assert raised.value.phase == "LATEST"
    assert raised.value.outcome_committed
    assert store.read(outcome_path) == outcome_bytes
    result = publisher.publish(
        "run-1",
        hashlib.sha256(chain[-1]).hexdigest(),
        outcome_path,
        outcome_bytes,
        "EMPTY",
        latest_bytes,
    )
    assert not result.outcome_created
    assert result.latest_replaced
