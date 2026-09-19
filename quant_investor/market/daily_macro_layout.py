"""Deterministic native Macro transaction selection, without filesystem writes."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from quant_investor.contracts import canonical_json_bytes
from quant_investor.macro.readiness_closure import TRANSACTION_ROOT


@dataclass(frozen=True)
class MacroLayout:
    transaction_id: str
    preparation_parent: Path
    prepared_path: Path
    journal_root: Path
    journal_id: str
    state: str


def _safe_existing(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        if current.exists():
            if any(
                p.name != part and p.name.casefold() == part.casefold() for p in current.iterdir()
            ):
                raise RuntimeError("MACRO_TRANSACTION_CASE_ALIAS")
        current = current / part
        if not os.path.lexists(current):
            return
        metadata = current.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("MACRO_TRANSACTION_PATH_UNSAFE")
        if current != path and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("MACRO_TRANSACTION_PATH_UNSAFE")


def select_macro_layout(context) -> MacroLayout:
    """Resolve occupied states before NO_ACTION; never adopt a different date/run."""
    root = context.workspace_root.resolve(strict=True)
    run = context.run_root.absolute()
    _safe_existing(root, run)
    relative = run.relative_to(root).as_posix()
    from datetime import datetime

    day = context.target_date
    if datetime.strptime(day, "%Y%m%d").strftime("%Y%m%d") != day:
        raise RuntimeError("MACRO_TRANSACTION_DATE_INVALID")
    digest = hashlib.sha256(
        canonical_json_bytes({"schema_version": "daily-macro-identity.v1", "run_root": relative})
    ).hexdigest()
    identity = f"daily-{day}-{digest}"
    parent = root / TRANSACTION_ROOT
    transaction = parent / identity
    journal_root = transaction / "journals"
    journal = journal_root / identity
    prepared = transaction / "prepared/prepared.json"
    legacy_root = run / "journals/macro" / day
    legacy_id = f"macro-{day}"
    legacy = legacy_root / legacy_id
    for path in (transaction, journal, prepared, legacy):
        _safe_existing(root, path)
    old = os.path.lexists(legacy)
    occupied = os.path.lexists(transaction)
    if old and occupied:
        raise RuntimeError("MACRO_TRANSACTION_LAYOUT_CONFLICT")
    if old:
        if not legacy.is_dir() or not any(legacy.iterdir()):
            raise RuntimeError("MACRO_TRANSACTION_EMPTY_LEGACY_JOURNAL")
        return MacroLayout(identity, parent, prepared, legacy_root, legacy_id, "LEGACY")
    if not occupied:
        state = "FRESH"
    elif os.path.lexists(journal):
        if not journal.is_dir() or not any(journal.iterdir()):
            raise RuntimeError("MACRO_TRANSACTION_EMPTY_JOURNAL")
        state = "JOURNALED"
    elif prepared.is_file():
        state = "PREPARED"
    else:
        raise RuntimeError("MACRO_PREPARATION_INCOMPLETE")
    return MacroLayout(identity, parent, prepared, journal_root, identity, state)
