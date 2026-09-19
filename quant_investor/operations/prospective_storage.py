"""Fixed daily ledger storage, sharing the existing descriptor-relative primitives."""

from pathlib import Path, PurePosixPath

from quant_investor.system.errors import SystemSecurityError
from quant_investor.system.storage import StoredBytes
from .daily_contract import ContractError, validate_ref
from .daily_journal import DailyJournal, _validate_day
from .journal_storage import JournalStorage

ROOT = PurePosixPath("results/prospective/CN")


def ledger_path(trade_date: str) -> str:
    _validate_day(trade_date)
    return str(ROOT / trade_date / "evidence-ledger.v1.json")


class _LedgerIO(JournalStorage):
    @staticmethod
    def _path(value: str) -> PurePosixPath:
        validate_ref({"path": value, "sha256": "0" * 64})
        path = PurePosixPath(value)
        if len(path.parts) != 5 or path.parts[:3] != ROOT.parts:
            raise SystemSecurityError("PROSPECTIVE_LEDGER_PATH_INVALID")
        if value != ledger_path(path.parts[3]):
            raise SystemSecurityError("PROSPECTIVE_LEDGER_PATH_INVALID")
        return path

    @staticmethod
    def _governed_directory(path: PurePosixPath) -> bool:
        return path == ROOT or ROOT in path.parents


class ProspectiveLedgerStorage:
    """No arbitrary path, projection replacement, independent lock or activation API."""

    def __init__(self, workspace: str):
        self._storage = _LedgerIO(workspace)

    def read(self, trade_date: str) -> StoredBytes | None:
        return self._storage.read(ledger_path(trade_date))

    def write(self, *, journal: DailyJournal, raw: bytes) -> StoredBytes:
        if type(journal) is not DailyJournal or journal.storage._io.workspace_root != Path(
            self._storage._io.workspace_root
        ).resolve(strict=True):
            raise ContractError("PROSPECTIVE_LEDGER_JOURNAL_MISMATCH")
        journal._require_lock()
        return self._storage.write(ledger_path(journal.trade_date), raw)
