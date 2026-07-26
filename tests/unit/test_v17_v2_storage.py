from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

import pytest

from quant_investor.v17_v2_runtime import (
    CASMismatchError,
    ExactOnceConflictError,
    LockUnavailableError,
    SecureStore,
    StorageSecurityError,
)


FILE_PATH = "results/v17_shadow/protocol-v2/runs/run-1/events/payload.json"
LATEST_PATH = "results/v17_shadow/protocol-v2/_latest/shadow.json"
LATEST_LOCK = "results/v17_shadow/protocol-v2/_latest/.latest.lock"


def test_secure_exact_once_modes_and_cas(tmp_path: Path) -> None:
    store = SecureStore(tmp_path)
    store.initialize()
    first = store.write_exact_once(FILE_PATH, b"one\n")
    retry = store.write_exact_once(FILE_PATH, b"one\n")
    assert first.created and not retry.created
    with pytest.raises(ExactOnceConflictError):
        store.write_exact_once(FILE_PATH, b"two\n")
    absolute = tmp_path / FILE_PATH
    observed = absolute.stat()
    assert stat.S_IMODE(observed.st_mode) == 0o600
    assert observed.st_nlink == 1
    assert stat.S_IMODE(absolute.parent.stat().st_mode) == 0o700

    created = store.replace_cas(LATEST_PATH, "EMPTY", b"pointer-1\n")
    replaced = store.replace_cas(
        LATEST_PATH,
        created.byte_sha256,
        b"pointer-2\n",
    )
    assert created.replaced and replaced.replaced
    before = (tmp_path / LATEST_PATH).read_bytes()
    with pytest.raises(CASMismatchError):
        store.replace_cas(LATEST_PATH, "0" * 64, b"not-written\n")
    assert (tmp_path / LATEST_PATH).read_bytes() == before


def test_symlink_hardlink_and_lock_contention_fail_closed(tmp_path: Path) -> None:
    store = SecureStore(tmp_path)
    store.initialize()
    target = tmp_path / "outside"
    target.write_bytes(b"x")
    leaf = tmp_path / FILE_PATH
    seed = f"{FILE_PATH}.seed"
    store.write_exact_once(seed, b"seed")
    (tmp_path / seed).unlink()
    leaf.symlink_to(target)
    with pytest.raises(StorageSecurityError):
        store.read(FILE_PATH)
    leaf.unlink()
    store.write_exact_once(FILE_PATH, b"x")
    hardlink = leaf.with_name("hardlink.json")
    os.link(leaf, hardlink)
    with pytest.raises(StorageSecurityError, match="exactly one link"):
        store.read(FILE_PATH)
    hardlink.unlink()

    second = SecureStore(tmp_path)
    with store.locked(LATEST_LOCK):
        with pytest.raises(LockUnavailableError):
            with second.locked(LATEST_LOCK, blocking=False):
                pass
