"""Ledger writes are immutable, fixed-path and serialized by the existing day lock."""

import os
import pytest

from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.prospective_storage import (
    ProspectiveLedgerStorage,
    ledger_path,
    _LedgerIO,
)
from quant_investor.system.errors import SystemImmutableConflict, SystemSecurityError


def test_write_once_reuses_exact_bytes_under_day_lock(tmp_path):
    storage = ProspectiveLedgerStorage(str(tmp_path))
    journal = DailyJournal(str(tmp_path), "20260908")
    assert storage.read(journal.trade_date) is None
    assert not (tmp_path / "results").exists()
    with pytest.raises(ContractError, match="JOURNAL_LOCK_REQUIRED"):
        storage.write(journal=journal, raw=b"{}")
    with journal.locked():
        first = storage.write(journal=journal, raw=b"{}")
        path = tmp_path / first.relative_path
        inode = path.stat().st_ino
        assert storage.write(journal=journal, raw=b"{}") == first
        assert path.stat().st_ino == inode
        with pytest.raises(SystemImmutableConflict):
            storage.write(journal=journal, raw=b'{"changed":true}')
    assert storage.read(journal.trade_date) == first
    assert first.relative_path == ledger_path(journal.trade_date)
    assert not hasattr(storage, "lock")
    assert not hasattr(storage, "activate")


@pytest.mark.parametrize(
    "path",
    [
        "results/system/_active.json",
        "results/prospective/CN/20260908/other.json",
        "results/prospective/CN/20260908/sub/evidence-ledger.v1.json",
        "results/prospective/CN/20260230/evidence-ledger.v1.json",
        "results/prospective/CN/../evidence-ledger.v1.json",
        "results/operations/daily_production/CN/20260908/dag-status.v1.json",
    ],
)
def test_no_alternate_write_root_or_filename(tmp_path, path):
    storage = _LedgerIO(str(tmp_path))
    with pytest.raises((ContractError, SystemSecurityError)):
        storage.write(path, b"{}")
    assert not (tmp_path / "results").exists()


def test_different_workspace_journal_rejected(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    journal = DailyJournal(str(other), "20260908")
    with journal.locked(), pytest.raises(ContractError, match="JOURNAL_MISMATCH"):
        ProspectiveLedgerStorage(str(tmp_path)).write(journal=journal, raw=b"{}")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "directory_mode"])
def test_unsafe_ledger_cannot_be_read_or_replaced(tmp_path, unsafe):
    storage = ProspectiveLedgerStorage(str(tmp_path))
    journal = DailyJournal(str(tmp_path), "20260908")
    with journal.locked():
        first = storage.write(journal=journal, raw=b"{}")
    path = tmp_path / first.relative_path
    if unsafe == "symlink":
        target = tmp_path / "outside.json"
        target.write_bytes(b"{}")
        path.unlink()
        path.symlink_to(target)
    elif unsafe == "hardlink":
        os.link(path, tmp_path / "alias.json")
    else:
        path.parent.chmod(0o755)
    with pytest.raises(SystemSecurityError):
        storage.read(journal.trade_date)
    with journal.locked(), pytest.raises(SystemSecurityError):
        storage.write(journal=journal, raw=b"{}")
