"""Acceptance guard permits real read locks and rejects publication."""

import pytest

from _verify_five_native_days import readonly_replay_guard, inventory
from quant_investor.factors.production_authority import (
    FactorProductionStore,
    FACTOR_ACTIVE_LOCK_PATH,
)


def make_store(tmp_path):
    store = FactorProductionStore(tmp_path)
    lock = tmp_path / FACTOR_ACTIVE_LOCK_PATH
    lock.parent.mkdir(parents=True, mode=0o700)
    lock.write_bytes(b"")
    lock.chmod(0o600)
    return store, lock


def test_native_lock_retains_workspace(tmp_path):
    store, lock = make_store(tmp_path)
    before = inventory(tmp_path)
    with readonly_replay_guard(), store._active_lock():
        assert lock.exists()
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("mode", [None, 0o644])
def test_guard_cannot_create_or_repair_lock(tmp_path, mode):
    store, lock = make_store(tmp_path)
    if mode is None:
        lock.unlink()
    else:
        lock.chmod(mode)
    before = inventory(tmp_path)
    with pytest.raises((AssertionError, FileNotFoundError)):
        with readonly_replay_guard(), store._active_lock():
            pytest.fail("unsafe lock admitted")
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "method",
    [
        "write_exact_once",
        "_write_blob_exact_once",
        "_write_exact_once",
        "_write_reserved_atomic_no_replace",
        "write_initial_pointer_under_lock",
        "write_permanent_marker_under_lock",
        "replace_active_pointer_under_lock",
    ],
)
def test_all_factor_publication_entries_forbidden(tmp_path, method):
    store, _ = make_store(tmp_path)
    before = inventory(tmp_path)
    with readonly_replay_guard(), store._active_lock():
        with pytest.raises(AssertionError, match="data write"):
            getattr(store._storage, method)()
    assert inventory(tmp_path) == before
