"""Crash-safe state machine for permanently retiring two exact result roots.

The journal is an execution record, never a source of deletion authority.  On
every read this module re-derives both approved roots and their quarantine
paths from the bound repository root, cutover id, and nonce.  A separate lock
file serializes the complete compare-and-swap and filesystem transition.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
from typing import Any

JOURNAL_SCHEMA = "v17-retirement-purge-journal.v1"
RECEIPT_SCHEMA = "v17-retirement-purge-receipt.v1"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
MAX_JOURNAL_BYTES = 4 * 1024 * 1024

SOURCE_RELATIVE_PATHS = (
    "results/v16",
    "results/v16_operator_advisory",
)

REQUIRED_SCHEDULE_NAMES = (
    "automation",
    "a-2",
    "a-dag",
    "cn-dashboard",
    "cn-daily-data-update",
    "myquant-2",
    "myquant-cn",
    "a",
    "myquant",
)

GATE_STATES = (
    "QUIESCED",
    "CODE_VERIFIED",
    "SKILL_SCHEDULES_VERIFIED",
    "PURGE_ELIGIBLE",
)
ACTIVE_PURGE_STATE = "PURGE_IN_PROGRESS"
PURGED_STATE = "PURGED"
RESUMED_STATE = "RESUMED"
ROLLED_BACK_STATE = "ROLLED_BACK_PRE_PURGE"
BLOCKED_STATE = "POST_PURGE_BLOCKED"

ROOT_STEPS = (
    "PENDING",
    "RENAME_INTENT",
    "RENAMED",
    "DELETE_INTENT",
    "DELETE_STARTED",
    "DELETED",
)

PURGE_LOGICALLY_COMMITTED = "PURGE_LOGICALLY_COMMITTED"
PURGE_PHYSICALLY_IRREVERSIBLE = "PURGE_PHYSICALLY_IRREVERSIBLE"

_JOURNAL_BASE_KEYS = frozenset(
    {
        "schema_version",
        "cutover_id",
        "repo_root",
        "repo_sha256",
        "skill_sha256",
        "schedule_sha256",
        "state",
        "purge_phase",
        "authority",
        "roots",
        "history",
    }
)
_ROOT_KEYS = frozenset(
    {
        "relative_path",
        "source_realpath",
        "source_path",
        "source_st_dev",
        "source_st_ino",
        "quarantine_path",
        "nonce",
        "step",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")


class RetirementError(RuntimeError):
    """Base fail-closed retirement error."""


class RetirementConflictError(RetirementError):
    """The journal CAS, lock, or cutover identity no longer matches."""


class PostPurgeBlockedError(RetirementError):
    """The purge crossed the first-rename boundary and needs repair."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RetirementError(f"journal is not finite canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and set(text) <= _HEX_DIGITS


def _normalized_sha256(value: object, field: str) -> str:
    text = str(value).strip().lower()
    if not _is_sha256(text):
        raise RetirementError(f"{field} must be a lowercase SHA256")
    return text


def _normalized_schedule_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_SCHEDULE_NAMES):
        raise RetirementError("schedule SHA map must contain the exact nine schedules")
    return {
        name: _normalized_sha256(value[name], f"schedule_sha256[{name}]")
        for name in sorted(REQUIRED_SCHEDULE_NAMES)
    }


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_private_dir(path: Path) -> Path:
    if path.is_symlink():
        raise RetirementError(f"private directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    os.chmod(path, PRIVATE_DIR_MODE)
    if stat.S_IMODE(path.stat().st_mode) != PRIVATE_DIR_MODE:
        raise RetirementError(f"private directory mode is not 0700: {path}")
    return path


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RetirementError(f"journal is missing or unsafe: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RetirementError(f"journal is not a regular file: {path}")
        if before.st_size > MAX_JOURNAL_BYTES:
            raise RetirementError("journal exceeds maximum size")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise RetirementConflictError("journal changed while it was read")
    finally:
        os.close(fd)
    try:
        path_after = os.lstat(path)
    except FileNotFoundError as exc:
        raise RetirementConflictError("journal disappeared while it was read") from exc
    if _stat_fingerprint(before) != _stat_fingerprint(path_after):
        raise RetirementConflictError("journal path identity changed while it was read")
    return b"".join(chunks), before


def _atomic_write(path: Path, payload: bytes) -> str:
    parent = _ensure_private_dir(path.parent)
    if path.is_symlink():
        raise RetirementError(f"journal cannot be a symlink: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    temp = Path(temp_name)
    digest = _sha256(payload)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, PRIVATE_FILE_MODE)
        _fsync_dir(parent)
        readback, readback_stat = _stable_regular_bytes(path)
        if readback != payload:
            raise RetirementError("journal readback differs from committed bytes")
        if stat.S_IMODE(readback_stat.st_mode) != PRIVATE_FILE_MODE:
            raise RetirementError("journal mode readback is not 0600")
        return digest
    finally:
        if temp.exists() or temp.is_symlink():
            temp.unlink()


@contextmanager
def _exclusive_journal_lock(path: Path) -> Iterator[None]:
    parent = _ensure_private_dir(path.absolute().parent)
    lock_path = parent / f".{path.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, PRIVATE_FILE_MODE)
    except OSError as exc:
        raise RetirementConflictError(f"retirement lock is unsafe: {lock_path}: {exc}") from exc
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RetirementConflictError("retirement lock is not a regular file")
        os.fchmod(fd, PRIVATE_FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RetirementConflictError(
                "another retirement operation holds the cutover lock"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetirementError(f"duplicate journal key: {key}")
        result[key] = value
    return result


def _validated_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise RetirementError("repository root must be a real directory")
    return root.resolve(strict=True)


def _source_path(repo_root: Path, relative_path: str) -> Path:
    if relative_path not in SOURCE_RELATIVE_PATHS:
        raise RetirementError(f"unapproved purge target: {relative_path}")
    results_parent = repo_root / "results"
    if results_parent.is_symlink() or not results_parent.is_dir():
        raise RetirementError("results parent must already be a real directory")
    target = (repo_root / relative_path).absolute()
    if target.parent.resolve(strict=True) != results_parent.resolve(strict=True):
        raise RetirementError("purge target escaped the exact results parent")
    return target


def _expected_quarantine(
    repo_root: Path,
    relative_path: str,
    cutover_id: str,
    nonce: str,
) -> Path:
    source = _source_path(repo_root, relative_path)
    return repo_root / "results" / f".{source.name}.v17-purge-{cutover_id}-{nonce}"


def _validate_history(history: object) -> None:
    if not isinstance(history, list) or not history:
        raise RetirementError("journal history must be a non-empty list")
    simple_events = set(GATE_STATES) | {
        ACTIVE_PURGE_STATE,
        PURGED_STATE,
        RESUMED_STATE,
    }
    reason_events = {ROLLED_BACK_STATE, BLOCKED_STATE}
    root_events = set(ROOT_STEPS) - {"PENDING"}
    for sequence, record in enumerate(history, start=1):
        if not isinstance(record, dict) or record.get("sequence") != sequence:
            raise RetirementError("journal history sequence is invalid")
        event = record.get("event")
        if event in simple_events:
            expected_keys = {"sequence", "event"}
        elif event in reason_events:
            expected_keys = {"sequence", "event", "reason"}
            if not str(record.get("reason", "")).strip():
                raise RetirementError("journal reason event is empty")
        elif event in root_events:
            expected_keys = {"sequence", "event", "relative_path"}
            if record.get("relative_path") not in SOURCE_RELATIVE_PATHS:
                raise RetirementError("journal root event has an invalid path")
        elif event == "POST_PURGE_REPAIR_ACKNOWLEDGED":
            expected_keys = {"sequence", "event", "acknowledgement"}
            if not str(record.get("acknowledgement", "")).strip():
                raise RetirementError("repair acknowledgement is empty")
        else:
            raise RetirementError(f"journal history event is invalid: {event}")
        if set(record) != expected_keys:
            raise RetirementError(f"journal history event has unexpected shape: {event}")


def _validate_journal(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != JOURNAL_SCHEMA:
        raise RetirementError("unexpected retirement journal schema")
    state = value.get("state")
    allowed_states = set(GATE_STATES) | {
        ACTIVE_PURGE_STATE,
        PURGED_STATE,
        RESUMED_STATE,
        ROLLED_BACK_STATE,
        BLOCKED_STATE,
    }
    if state not in allowed_states:
        raise RetirementError("retirement journal state is invalid")
    expected_keys = set(_JOURNAL_BASE_KEYS)
    if state == BLOCKED_STATE:
        expected_keys.add("blocked_reason")
    if state == RESUMED_STATE:
        expected_keys.update(
            {"final_scan_sha256", "active_schedules_restored", "legacy_schedules_deleted"}
        )
    if set(value) != expected_keys:
        raise RetirementError("retirement journal has an unexpected top-level shape")
    if value.get("authority") is not False:
        raise RetirementError("retirement journal must declare authority=false")

    cutover_id = value.get("cutover_id")
    if (
        not isinstance(cutover_id, str)
        or not cutover_id
        or "/" in cutover_id
        or "\\" in cutover_id
        or ".." in cutover_id
    ):
        raise RetirementError("cutover_id is not a safe identifier")
    repo_root = _validated_repo_root(str(value.get("repo_root", "")))
    if value["repo_root"] != str(repo_root):
        raise RetirementError("journal repo_root is not canonical")
    _normalized_sha256(value.get("repo_sha256"), "repo_sha256")
    _normalized_sha256(value.get("skill_sha256"), "skill_sha256")
    if not isinstance(value.get("schedule_sha256"), dict):
        raise RetirementError("schedule_sha256 must be an object")
    normalized_schedules = _normalized_schedule_hashes(value["schedule_sha256"])
    if value["schedule_sha256"] != normalized_schedules:
        raise RetirementError("schedule SHA map is not canonical")

    phase = value.get("purge_phase")
    if phase not in {None, PURGE_LOGICALLY_COMMITTED, PURGE_PHYSICALLY_IRREVERSIBLE}:
        raise RetirementError("journal purge_phase is invalid")
    roots = value.get("roots")
    if not isinstance(roots, list) or len(roots) != len(SOURCE_RELATIVE_PATHS):
        raise RetirementError("journal must contain exactly the two approved roots")

    shared_nonce: str | None = None
    steps: list[str] = []
    for index, (root_entry, relative_path) in enumerate(zip(roots, SOURCE_RELATIVE_PATHS)):
        if not isinstance(root_entry, dict) or set(root_entry) != _ROOT_KEYS:
            raise RetirementError(f"root journal entry {index} has an unexpected shape")
        if root_entry.get("relative_path") != relative_path:
            raise RetirementError("journal roots are missing, reordered, or substituted")
        nonce = root_entry.get("nonce")
        if not isinstance(nonce, str) or not nonce.isalnum():
            raise RetirementError("root nonce must be alphanumeric")
        if shared_nonce is None:
            shared_nonce = nonce
        elif nonce != shared_nonce:
            raise RetirementError("root nonces differ")
        source = _source_path(repo_root, relative_path)
        quarantine = _expected_quarantine(repo_root, relative_path, cutover_id, nonce)
        if root_entry.get("source_path") != str(source):
            raise RetirementError("journal source path is not the derived approved root")
        if root_entry.get("source_realpath") != str(source):
            raise RetirementError("journal source realpath is not the derived approved root")
        if root_entry.get("quarantine_path") != str(quarantine):
            raise RetirementError("journal quarantine path is not derived from the cutover")
        for field in ("source_st_dev", "source_st_ino"):
            number = root_entry.get(field)
            if type(number) is not int or number <= 0:
                raise RetirementError(f"root {field} must be a positive integer")
        step = root_entry.get("step")
        if step not in ROOT_STEPS:
            raise RetirementError("root step is invalid")
        steps.append(step)

    _validate_history(value.get("history"))
    if state in GATE_STATES[:-1] and (steps != ["PENDING", "PENDING"] or phase is not None):
        raise RetirementError("pre-eligibility journal cannot contain purge progress")
    if state == "PURGE_ELIGIBLE":
        if phase is not None or steps not in (
            ["PENDING", "PENDING"],
            ["RENAME_INTENT", "PENDING"],
        ):
            raise RetirementError("PURGE_ELIGIBLE has invalid root progress")
    if state == ACTIVE_PURGE_STATE and phase not in {
        PURGE_LOGICALLY_COMMITTED,
        PURGE_PHYSICALLY_IRREVERSIBLE,
    }:
        raise RetirementError("PURGE_IN_PROGRESS must follow the first rename")
    if state == BLOCKED_STATE:
        if not str(value.get("blocked_reason", "")).strip() or phase not in {
            PURGE_LOGICALLY_COMMITTED,
            PURGE_PHYSICALLY_IRREVERSIBLE,
        }:
            raise RetirementError("POST_PURGE_BLOCKED is missing durable boundary evidence")
    if state in {PURGED_STATE, RESUMED_STATE}:
        if steps != ["DELETED", "DELETED"] or phase != PURGE_PHYSICALLY_IRREVERSIBLE:
            raise RetirementError(f"{state} requires both roots durably DELETED")
    if state == ROLLED_BACK_STATE and phase is not None:
        raise RetirementError("ROLLED_BACK_PRE_PURGE cannot cross the rename boundary")
    if state == RESUMED_STATE:
        _normalized_sha256(value.get("final_scan_sha256"), "final_scan_sha256")
        if value.get("active_schedules_restored") != 7:
            raise RetirementError("RESUMED active schedule count is invalid")
        if value.get("legacy_schedules_deleted") != 2:
            raise RetirementError("RESUMED legacy schedule count is invalid")
    return value


def _parse_journal(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RetirementError(f"non-finite journal value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"invalid journal JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RetirementError("retirement journal must be an object")
    return _validate_journal(value)


def journal_sha256(path: str | Path) -> str:
    payload, file_stat = _stable_regular_bytes(Path(path))
    if stat.S_IMODE(file_stat.st_mode) != PRIVATE_FILE_MODE:
        raise RetirementError("journal mode must be 0600")
    return _sha256(payload)


def load_journal(path: str | Path) -> dict[str, Any]:
    payload, file_stat = _stable_regular_bytes(Path(path))
    if stat.S_IMODE(file_stat.st_mode) != PRIVATE_FILE_MODE:
        raise RetirementError("journal mode must be 0600")
    return _parse_journal(payload)


def _require_cas_locked(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = _normalized_sha256(expected_sha256, "expected_journal_sha256")
    payload, file_stat = _stable_regular_bytes(path)
    if stat.S_IMODE(file_stat.st_mode) != PRIVATE_FILE_MODE:
        raise RetirementError("journal mode must be 0600")
    current = _sha256(payload)
    if expected != current:
        raise RetirementConflictError(
            f"journal CAS mismatch: expected {expected}, current {current}"
        )
    return _parse_journal(payload), current


def _write_with_cas_locked(
    path: Path,
    journal: Mapping[str, Any],
    expected_sha256: str,
) -> str:
    expected = _normalized_sha256(expected_sha256, "expected_journal_sha256")
    payload, _ = _stable_regular_bytes(path)
    current = _sha256(payload)
    if current != expected:
        raise RetirementConflictError(
            f"journal CAS mismatch: expected {expected}, current {current}"
        )
    validated = _validate_journal(dict(journal))
    return _atomic_write(path, _canonical_bytes(validated))


def _stat_source(path: Path) -> os.stat_result:
    try:
        value = os.lstat(path)
    except FileNotFoundError as exc:
        raise RetirementError(f"purge source is missing: {path}") from exc
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise RetirementError(f"purge source is not a real directory: {path}")
    return value


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _identity_matches_stat(value: os.stat_result | None, *, device: int, inode: int) -> bool:
    return bool(
        value is not None
        and stat.S_ISDIR(value.st_mode)
        and not stat.S_ISLNK(value.st_mode)
        and value.st_dev == int(device)
        and value.st_ino == int(inode)
    )


def _identity_matches(path: Path, *, device: int, inode: int) -> bool:
    return _identity_matches_stat(_lstat(path), device=device, inode=inode)


def _append_event(journal: dict[str, Any], event: str, **details: Any) -> None:
    history = journal.setdefault("history", [])
    if not isinstance(history, list):
        raise RetirementError("journal history must be a list")
    record: dict[str, Any] = {"sequence": len(history) + 1, "event": event}
    record.update(details)
    history.append(record)


def initialize_cutover(
    *,
    repo_root: str | Path,
    journal_path: str | Path,
    cutover_id: str,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
    nonce: str | None = None,
) -> str:
    """Create a QUIESCED journal after resolving both exact source inodes."""

    path = Path(journal_path).absolute()
    root = _validated_repo_root(repo_root)
    cutover_nonce = str(nonce or secrets.token_hex(16))
    if not cutover_nonce.isalnum():
        raise RetirementError("nonce must be alphanumeric")
    if not cutover_id or "/" in cutover_id or "\\" in cutover_id or ".." in cutover_id:
        raise RetirementError("cutover_id is not a safe identifier")
    normalized_repo_sha = _normalized_sha256(repo_sha256, "repo_sha256")
    normalized_skill_sha = _normalized_sha256(skill_sha256, "skill_sha256")
    normalized_schedules = _normalized_schedule_hashes(schedule_sha256)

    with _exclusive_journal_lock(path):
        if path.exists() or path.is_symlink():
            raise RetirementConflictError("cutover journal already exists")
        roots: list[dict[str, Any]] = []
        for relative_path in SOURCE_RELATIVE_PATHS:
            source = _source_path(root, relative_path)
            source_stat = _stat_source(source)
            quarantine = _expected_quarantine(root, relative_path, cutover_id, cutover_nonce)
            if _lstat(quarantine) is not None:
                raise RetirementError(f"quarantine path already exists: {quarantine}")
            roots.append(
                {
                    "relative_path": relative_path,
                    "source_realpath": str(source),
                    "source_path": str(source),
                    "source_st_dev": source_stat.st_dev,
                    "source_st_ino": source_stat.st_ino,
                    "quarantine_path": str(quarantine),
                    "nonce": cutover_nonce,
                    "step": "PENDING",
                }
            )

        journal: dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA,
            "cutover_id": cutover_id,
            "repo_root": str(root),
            "repo_sha256": normalized_repo_sha,
            "skill_sha256": normalized_skill_sha,
            "schedule_sha256": normalized_schedules,
            "state": "QUIESCED",
            "purge_phase": None,
            "authority": False,
            "roots": roots,
            "history": [],
        }
        _append_event(journal, "QUIESCED")
        _validate_journal(journal)
        return _atomic_write(path, _canonical_bytes(journal))


def assert_cutover_identity(
    journal: Mapping[str, Any],
    *,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
) -> None:
    expected = {
        "cutover_id": cutover_id,
        "repo_root": str(_validated_repo_root(repo_root)),
        "repo_sha256": _normalized_sha256(repo_sha256, "repo_sha256"),
        "skill_sha256": _normalized_sha256(skill_sha256, "skill_sha256"),
        "schedule_sha256": _normalized_schedule_hashes(schedule_sha256),
    }
    actual = {key: journal.get(key) for key in expected}
    if actual != expected:
        raise RetirementConflictError("cutover identity/hash binding mismatch")


def advance_gate(
    *,
    journal_path: str | Path,
    expected_journal_sha256: str,
    next_state: str,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
) -> str:
    path = Path(journal_path)
    with _exclusive_journal_lock(path):
        journal, current_sha = _require_cas_locked(path, expected_journal_sha256)
        assert_cutover_identity(
            journal,
            cutover_id=cutover_id,
            repo_root=repo_root,
            repo_sha256=repo_sha256,
            skill_sha256=skill_sha256,
            schedule_sha256=schedule_sha256,
        )
        current = journal.get("state")
        if next_state not in GATE_STATES:
            raise RetirementError(f"invalid pre-purge gate state: {next_state}")
        try:
            expected_next = GATE_STATES[GATE_STATES.index(str(current)) + 1]
        except (ValueError, IndexError) as exc:
            raise RetirementConflictError(f"cannot advance gate from {current}") from exc
        if next_state != expected_next:
            raise RetirementConflictError(f"gate transition must be {current} -> {expected_next}")
        journal["state"] = next_state
        _append_event(journal, next_state)
        return _write_with_cas_locked(path, journal, current_sha)


def _check_root_replay(root: Mapping[str, Any]) -> str:
    source = Path(str(root["source_path"]))
    quarantine = Path(str(root["quarantine_path"]))
    source_stat = _lstat(source)
    quarantine_stat = _lstat(quarantine)
    source_present = source_stat is not None
    quarantine_present = quarantine_stat is not None
    identity = {
        "device": int(root["source_st_dev"]),
        "inode": int(root["source_st_ino"]),
    }
    source_exact = _identity_matches_stat(source_stat, **identity)
    quarantine_exact = _identity_matches_stat(quarantine_stat, **identity)
    step = str(root["step"])

    if source_present and quarantine_present:
        return "source and quarantine both exist"
    if source_present and not source_exact:
        return "source inode or type drifted"
    if quarantine_present and not quarantine_exact:
        return "quarantine inode or type drifted"
    if step == "PENDING":
        return "" if source_exact and not quarantine_present else "PENDING requires source only"
    if step == "RENAME_INTENT":
        if (source_exact and not quarantine_present) or (not source_present and quarantine_exact):
            return ""
        return "RENAME_INTENT has an impossible source/quarantine layout"
    if step in {"RENAMED", "DELETE_INTENT"}:
        return "" if not source_present and quarantine_exact else f"{step} requires quarantine only"
    if step == "DELETE_STARTED":
        if not source_present and (quarantine_exact or not quarantine_present):
            return ""
        return "DELETE_STARTED has an impossible source/quarantine layout"
    if step == "DELETED":
        return (
            ""
            if not source_present and not quarantine_present
            else "DELETED requires both paths absent"
        )
    return "unknown root replay step"


def _rename_boundary_crossed(journal: Mapping[str, Any]) -> bool:
    if journal.get("purge_phase") in {
        PURGE_LOGICALLY_COMMITTED,
        PURGE_PHYSICALLY_IRREVERSIBLE,
    }:
        return True
    for root in journal.get("roots", []):
        step = root.get("step")
        if step in {"RENAMED", "DELETE_INTENT", "DELETE_STARTED", "DELETED"}:
            return True
        if step == "RENAME_INTENT":
            quarantine = Path(str(root["quarantine_path"]))
            # The exact original inode at quarantine proves rename happened.
            # A subsequently recreated source name must not move the boundary
            # backwards or make the cutover rollbackable again.
            if _identity_matches(
                quarantine,
                device=int(root["source_st_dev"]),
                inode=int(root["source_st_ino"]),
            ):
                return True
    return False


def rollback_pre_purge(
    *,
    journal_path: str | Path,
    expected_journal_sha256: str,
    reason: str,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
) -> str:
    path = Path(journal_path)
    with _exclusive_journal_lock(path):
        journal, current_sha = _require_cas_locked(path, expected_journal_sha256)
        assert_cutover_identity(
            journal,
            cutover_id=cutover_id,
            repo_root=repo_root,
            repo_sha256=repo_sha256,
            skill_sha256=skill_sha256,
            schedule_sha256=schedule_sha256,
        )
        if journal.get("state") not in GATE_STATES:
            raise RetirementConflictError("rollback is forbidden after the purge starts")
        if _rename_boundary_crossed(journal):
            raise RetirementConflictError("rollback is forbidden after the first rename")
        for root in journal["roots"]:
            replay_reason = _check_root_replay(root)
            if replay_reason:
                raise RetirementConflictError(f"rollback preflight failed: {replay_reason}")
        if not str(reason).strip():
            raise RetirementError("rollback reason is required")
        journal["state"] = ROLLED_BACK_STATE
        _append_event(journal, ROLLED_BACK_STATE, reason=str(reason))
        return _write_with_cas_locked(path, journal, current_sha)


def _record_blocked_locked(
    path: Path,
    journal: dict[str, Any],
    expected_sha256: str,
    reason: str,
) -> str:
    journal["state"] = BLOCKED_STATE
    journal["blocked_reason"] = str(reason)
    if journal.get("purge_phase") is None:
        journal["purge_phase"] = PURGE_LOGICALLY_COMMITTED
    _append_event(journal, BLOCKED_STATE, reason=str(reason))
    return _write_with_cas_locked(path, journal, expected_sha256)


def _raise_durably_blocked(
    path: Path,
    journal: dict[str, Any],
    current_sha: str,
    reason: str,
    *,
    cause: BaseException | None = None,
) -> None:
    try:
        blocked_sha = _record_blocked_locked(path, journal, current_sha, reason)
    except Exception as block_exc:
        message = f"{reason}; failed to persist POST_PURGE_BLOCKED: {block_exc}"
        if cause is not None:
            raise PostPurgeBlockedError(message) from cause
        raise PostPurgeBlockedError(message) from block_exc
    message = f"{reason}; durable blocked journal {blocked_sha}"
    if cause is not None:
        raise PostPurgeBlockedError(message) from cause
    raise PostPurgeBlockedError(message)


def _commit_event_locked(
    *,
    path: Path,
    journal: dict[str, Any],
    current_sha: str,
    event: str,
    root: dict[str, Any],
) -> str:
    root["step"] = event
    _append_event(journal, event, relative_path=root["relative_path"])
    return _write_with_cas_locked(path, journal, current_sha)


def _call_hook(
    hook: Callable[[str, Mapping[str, Any]], None] | None,
    event: str,
    root: Mapping[str, Any],
) -> None:
    if hook is not None:
        hook(event, root)


def purge(
    *,
    journal_path: str | Path,
    expected_journal_sha256: str,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
    repair_acknowledgement: str = "",
    _event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> str:
    """Rename then permanently unlink both roots using durable replay steps."""

    path = Path(journal_path)
    with _exclusive_journal_lock(path):
        journal, current_sha = _require_cas_locked(path, expected_journal_sha256)
        assert_cutover_identity(
            journal,
            cutover_id=cutover_id,
            repo_root=repo_root,
            repo_sha256=repo_sha256,
            skill_sha256=skill_sha256,
            schedule_sha256=schedule_sha256,
        )
        state = journal.get("state")
        if state == BLOCKED_STATE:
            if not str(repair_acknowledgement).strip():
                raise PostPurgeBlockedError(
                    "blocked purge requires an explicit repair acknowledgement"
                )
            for root in journal["roots"]:
                replay_reason = _check_root_replay(root)
                if replay_reason:
                    raise PostPurgeBlockedError(f"repair is incomplete: {replay_reason}")
            journal.pop("blocked_reason", None)
            journal["state"] = ACTIVE_PURGE_STATE
            _append_event(
                journal,
                "POST_PURGE_REPAIR_ACKNOWLEDGED",
                acknowledgement=str(repair_acknowledgement),
            )
            current_sha = _write_with_cas_locked(path, journal, current_sha)
        elif state not in {"PURGE_ELIGIBLE", ACTIVE_PURGE_STATE}:
            raise RetirementConflictError(f"purge cannot start from {state}")

        # Both roots are checked before the first rename.  Before that boundary
        # any failure leaves PURGE_ELIGIBLE byte-identical and rollbackable.
        crossed_boundary = _rename_boundary_crossed(journal)
        replay_failures = [
            (root["relative_path"], _check_root_replay(root)) for root in journal["roots"]
        ]
        replay_failures = [(name, reason) for name, reason in replay_failures if reason]
        if replay_failures:
            reason = "; ".join(f"{name}: {message}" for name, message in replay_failures)
            if crossed_boundary:
                _raise_durably_blocked(path, journal, current_sha, reason)
            raise RetirementError(f"pre-purge root preflight failed: {reason}")

        for root in journal["roots"]:
            source = Path(str(root["source_path"]))
            quarantine = Path(str(root["quarantine_path"]))
            parent = source.parent
            step = str(root["step"])

            if step == "PENDING":
                current_sha = _commit_event_locked(
                    path=path,
                    journal=journal,
                    current_sha=current_sha,
                    event="RENAME_INTENT",
                    root=root,
                )
                step = "RENAME_INTENT"
                _call_hook(_event_hook, "RENAME_INTENT_COMMITTED", root)

            if step == "RENAME_INTENT":
                replay_reason = _check_root_replay(root)
                if replay_reason:
                    if _rename_boundary_crossed(journal):
                        _raise_durably_blocked(path, journal, current_sha, replay_reason)
                    raise RetirementError(f"rename precondition failed: {replay_reason}")
                if _lstat(source) is not None:
                    try:
                        os.rename(source, quarantine)
                        _fsync_dir(parent)
                    except OSError as exc:
                        if _rename_boundary_crossed(journal):
                            journal["state"] = ACTIVE_PURGE_STATE
                            journal["purge_phase"] = PURGE_LOGICALLY_COMMITTED
                            _raise_durably_blocked(
                                path,
                                journal,
                                current_sha,
                                f"rename/fsync failed after first rename: {exc}",
                                cause=exc,
                            )
                        raise RetirementError(
                            f"rename failed before purge boundary: {exc}"
                        ) from exc
                    _call_hook(_event_hook, "RENAMED_FILESYSTEM", root)
                replay_reason = _check_root_replay(root)
                if replay_reason or _lstat(source) is not None:
                    journal["state"] = ACTIVE_PURGE_STATE
                    journal["purge_phase"] = PURGE_LOGICALLY_COMMITTED
                    _raise_durably_blocked(
                        path,
                        journal,
                        current_sha,
                        replay_reason or "rename readback found a remaining source",
                    )
                journal["state"] = ACTIVE_PURGE_STATE
                journal["purge_phase"] = PURGE_LOGICALLY_COMMITTED
                current_sha = _commit_event_locked(
                    path=path,
                    journal=journal,
                    current_sha=current_sha,
                    event="RENAMED",
                    root=root,
                )
                step = "RENAMED"
                _call_hook(_event_hook, "RENAMED_COMMITTED", root)

            if step == "RENAMED":
                current_sha = _commit_event_locked(
                    path=path,
                    journal=journal,
                    current_sha=current_sha,
                    event="DELETE_INTENT",
                    root=root,
                )
                step = "DELETE_INTENT"
                _call_hook(_event_hook, "DELETE_INTENT_COMMITTED", root)

            if step == "DELETE_INTENT":
                replay_reason = _check_root_replay(root)
                if replay_reason:
                    _raise_durably_blocked(path, journal, current_sha, replay_reason)
                journal["purge_phase"] = PURGE_PHYSICALLY_IRREVERSIBLE
                current_sha = _commit_event_locked(
                    path=path,
                    journal=journal,
                    current_sha=current_sha,
                    event="DELETE_STARTED",
                    root=root,
                )
                step = "DELETE_STARTED"
                _call_hook(_event_hook, "DELETE_STARTED_COMMITTED", root)

            if step == "DELETE_STARTED":
                replay_reason = _check_root_replay(root)
                if replay_reason:
                    _raise_durably_blocked(path, journal, current_sha, replay_reason)
                if _lstat(quarantine) is not None:
                    try:
                        shutil.rmtree(quarantine)
                        _fsync_dir(parent)
                    except OSError as exc:
                        _raise_durably_blocked(
                            path,
                            journal,
                            current_sha,
                            f"delete/fsync failed after DELETE_STARTED: {exc}",
                            cause=exc,
                        )
                    _call_hook(_event_hook, "DELETED_FILESYSTEM", root)
                replay_reason = _check_root_replay(root)
                if replay_reason:
                    _raise_durably_blocked(path, journal, current_sha, replay_reason)
                current_sha = _commit_event_locked(
                    path=path,
                    journal=journal,
                    current_sha=current_sha,
                    event="DELETED",
                    root=root,
                )
                _call_hook(_event_hook, "DELETED_COMMITTED", root)

        if any(root["step"] != "DELETED" for root in journal["roots"]):
            _raise_durably_blocked(
                path,
                journal,
                current_sha,
                "not every approved root reached DELETED",
            )
        final_failures = [
            (root["relative_path"], _check_root_replay(root)) for root in journal["roots"]
        ]
        final_failures = [(name, reason) for name, reason in final_failures if reason]
        if final_failures:
            reason = "; ".join(f"{name}: {message}" for name, message in final_failures)
            _raise_durably_blocked(path, journal, current_sha, reason)
        journal["state"] = PURGED_STATE
        journal["purge_phase"] = PURGE_PHYSICALLY_IRREVERSIBLE
        _append_event(journal, PURGED_STATE)
        return _write_with_cas_locked(path, journal, current_sha)


def mark_resumed(
    *,
    journal_path: str | Path,
    expected_journal_sha256: str,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
    active_schedules_restored: int,
    legacy_schedules_deleted: int,
    final_scan_clean: bool,
    final_scan_sha256: str,
) -> tuple[str, dict[str, Any]]:
    path = Path(journal_path)
    with _exclusive_journal_lock(path):
        journal, current_sha = _require_cas_locked(path, expected_journal_sha256)
        assert_cutover_identity(
            journal,
            cutover_id=cutover_id,
            repo_root=repo_root,
            repo_sha256=repo_sha256,
            skill_sha256=skill_sha256,
            schedule_sha256=schedule_sha256,
        )
        if journal.get("state") != PURGED_STATE:
            raise RetirementConflictError("RESUMED requires a durable PURGED state")
        if active_schedules_restored != 7 or legacy_schedules_deleted != 2:
            raise RetirementError("schedule postconditions are not satisfied")
        final_scan_sha = _normalized_sha256(final_scan_sha256, "final_scan_sha256")
        if not final_scan_clean:
            raise RetirementError("a clean hash-bound final scan is required")
        for root in journal["roots"]:
            replay_reason = _check_root_replay(root)
            if replay_reason:
                raise RetirementError(
                    f"source or quarantine reappeared after PURGED: {replay_reason}"
                )

        journal["state"] = RESUMED_STATE
        journal["final_scan_sha256"] = final_scan_sha
        journal["active_schedules_restored"] = 7
        journal["legacy_schedules_deleted"] = 2
        _append_event(journal, RESUMED_STATE)
        final_journal_sha = _write_with_cas_locked(path, journal, current_sha)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "cutover_id": journal["cutover_id"],
            "repo_root": journal["repo_root"],
            "repo_sha256": journal["repo_sha256"],
            "skill_sha256": journal["skill_sha256"],
            "schedule_sha256": journal["schedule_sha256"],
            "final_scan_sha256": final_scan_sha,
            "journal_sha256": final_journal_sha,
            "state": RESUMED_STATE,
            "purged_relative_paths": list(SOURCE_RELATIVE_PATHS),
            "secure_erasure": False,
            "deleted_file_bytes_preserved": False,
            "authority": False,
        }
        return final_journal_sha, receipt


__all__ = [
    "ACTIVE_PURGE_STATE",
    "BLOCKED_STATE",
    "GATE_STATES",
    "JOURNAL_SCHEMA",
    "PostPurgeBlockedError",
    "PURGED_STATE",
    "REQUIRED_SCHEDULE_NAMES",
    "RESUMED_STATE",
    "RetirementConflictError",
    "RetirementError",
    "SOURCE_RELATIVE_PATHS",
    "advance_gate",
    "assert_cutover_identity",
    "initialize_cutover",
    "journal_sha256",
    "load_journal",
    "mark_resumed",
    "purge",
    "rollback_pre_purge",
]
