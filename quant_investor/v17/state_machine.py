"""Durable compare-and-swap lifecycle for v17 shadow runs.

The mutable surface is one 0600 ``ledger.json`` per run.  Every transition
holds the pre-created run lock, compares the caller-supplied byte SHA before
any write, writes immutable transition artifacts, and atomically replaces the
ledger.  A CAS mismatch is strictly zero-write.  Other failed transition
attempts may write only an immutable ``UNPUBLISHED`` receipt; they never alter
the ledger, terminal outcome, or latest pointer.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Iterator
import uuid

from .contracts import (
    V17ContractError,
    parse_utc_timestamp,
    require_authority_false,
    require_exact_keys,
    require_identifier,
)
from .resources import resource_path
from .semantic import (
    canonical_json_bytes,
    require_sha256,
    seal_semantic,
    validate_semantic_seal,
)
from .storage import (
    atomic_write_json,
    ensure_private_directory,
    ensure_v17_shadow_layout,
    file_sha256,
    read_json,
)

LEDGER_VERSION = "myquant.v17.shadow-ledger.v1"
TERMINAL_OUTPUT_VERSION = "myquant.v17.shadow-output.v1"
UNPUBLISHED_RECEIPT_VERSION = "myquant.v17.unpublished-receipt.v1"
STATE_MACHINE_RESOURCE_VERSION = "myquant-v17-shadow-state-machine.v1"
STATE_MACHINE_RESOURCE_SHA256 = "241b9c784cc3623a1d81a7a706b15abe44bb0157ccdbd3da36a15ef4ff6f60f4"
EMPTY_SHA = "EMPTY"

LEDGER_KEYS = frozenset(
    {
        "version",
        "run_id",
        "strategy_id",
        "market",
        "cutoff",
        "state",
        "sequence",
        "created_at",
        "updated_at",
        "previous_ledger_sha256",
        "input_bindings",
        "artifacts",
        "history",
        "authority",
        "semantic_sha256",
    }
)
HISTORY_KEYS = frozenset(
    {
        "sequence",
        "from_state",
        "to_state",
        "at",
        "expected_ledger_sha256",
        "artifact_roles",
    }
)
ARTIFACT_BINDING_KEYS = frozenset(
    {
        "relative_path",
        "byte_sha256",
        "semantic_sha256",
        "sequence",
        "state",
    }
)
TERMINAL_OUTPUT_KEYS = frozenset(
    {
        "version",
        "run_id",
        "strategy_id",
        "market",
        "cutoff",
        "terminal_state",
        "rank_output",
        "portfolio_output",
        "blockers",
        "source_manifest_sha256",
        "ledger_predecessor_sha256",
        "generated_at",
        "authority",
        "semantic_sha256",
    }
)


class V17StateMachineError(V17ContractError):
    """The requested v17 state transition is invalid or cannot be committed."""


class V17LedgerCASMismatch(V17StateMachineError):
    """The expected ledger byte SHA did not match; no bytes were written."""


class V17PostCommitReadbackError(V17StateMachineError):
    """The ledger replace occurred, but durable post-commit readback failed."""


def _repo_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise V17StateMachineError("repository root unavailable or symlinked")
    return root


def _validate_expected_sha(value: str, *, allow_empty: bool) -> str:
    if allow_empty and value == EMPTY_SHA:
        return value
    try:
        return require_sha256(value, label="expected ledger SHA-256")
    except ValueError as exc:
        raise V17StateMachineError(str(exc)) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise V17StateMachineError("fsync target is not a directory")
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise V17StateMachineError("directory identity drift during fsync")
    finally:
        os.close(descriptor)


def _write_exclusive_bytes(path: Path, payload: bytes, *, root: Path) -> str:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise V17StateMachineError("exclusive write escaped fixed root") from exc
    parent = ensure_private_directory(path.parent, root=root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    completed = False
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise V17StateMachineError(f"immutable artifact already exists: {path}") from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise V17StateMachineError("short immutable artifact write")
            written += count
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        completed = True
    finally:
        os.close(descriptor)
        if not completed and os.path.lexists(path):
            path.unlink()
            _fsync_directory(parent)
    _fsync_directory(parent)
    observed = file_sha256(path)
    import hashlib

    expected = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise V17StateMachineError("immutable artifact readback mismatch")
    return observed


def _write_exclusive_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> str:
    raw = canonical_json_bytes(payload) + b"\n"
    return _write_exclusive_bytes(path, raw, root=root)


def _ensure_lock_file(path: Path, *, root: Path) -> None:
    ensure_private_directory(path.parent, root=root)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        descriptor = os.open(
            path,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o600
        ):
            raise V17StateMachineError("lifecycle lock identity invalid")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _load_machine_definition() -> dict[str, Any]:
    path = resource_path("state_machine.v1.json")
    if file_sha256(path) != STATE_MACHINE_RESOURCE_SHA256:
        raise V17StateMachineError("state-machine resource byte SHA mismatch")
    payload = read_json(path)
    require_exact_keys(
        payload,
        frozenset(
            {
                "schema",
                "version",
                "authority",
                "initial_state",
                "terminal_states",
                "transitions",
            }
        ),
        label="state-machine resource",
    )
    if payload.get("schema") != STATE_MACHINE_RESOURCE_VERSION:
        raise V17StateMachineError("state-machine resource version mismatch")
    if payload.get("version") != "17.0.0":
        raise V17StateMachineError("state-machine package version mismatch")
    require_authority_false(payload.get("authority"))
    transitions = payload.get("transitions")
    terminals = payload.get("terminal_states")
    if not isinstance(transitions, Mapping) or not transitions:
        raise V17StateMachineError("state-machine transitions missing")
    if not isinstance(terminals, list) or len(terminals) != 5:
        raise V17StateMachineError("state-machine terminal set mismatch")
    terminal_set = set(terminals)
    if len(terminal_set) != 5 or not terminal_set.issubset(transitions):
        raise V17StateMachineError("state-machine terminal set invalid")
    for state, targets in transitions.items():
        require_identifier(state, label="state-machine state")
        if not isinstance(targets, list) or len(targets) != len(set(targets)):
            raise V17StateMachineError("state-machine transition list invalid")
        if any(target not in transitions for target in targets):
            raise V17StateMachineError("state-machine transition target unknown")
        if state in terminal_set and targets:
            raise V17StateMachineError("terminal state must be immutable")
    if payload.get("initial_state") != "PREPARED":
        raise V17StateMachineError("state-machine initial state mismatch")
    return payload


def terminal_states() -> frozenset[str]:
    return frozenset(_load_machine_definition()["terminal_states"])


def is_terminal_state(state: str) -> bool:
    return state in terminal_states()


def _validate_artifact_binding(
    role: str,
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    maximum_sequence: int,
    verify_payload: bool,
) -> None:
    require_identifier(role, label="artifact role")
    require_exact_keys(binding, ARTIFACT_BINDING_KEYS, label=f"artifact {role}")
    relative = PurePosixPath(str(binding.get("relative_path") or ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("results", "v17_shadow")
    ):
        raise V17StateMachineError(f"artifact path invalid: {role}")
    target = repo_root / Path(*relative.parts)
    declared = require_sha256(binding.get("byte_sha256"), label=f"artifact {role} byte SHA-256")
    require_sha256(
        binding.get("semantic_sha256"),
        label=f"artifact {role} semantic SHA-256",
    )
    sequence = binding.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 1 <= sequence <= maximum_sequence
    ):
        raise V17StateMachineError(f"artifact sequence invalid: {role}")
    if not isinstance(binding.get("state"), str):
        raise V17StateMachineError(f"artifact state invalid: {role}")
    if not verify_payload:
        return
    if file_sha256(target) != declared:
        raise V17StateMachineError(f"artifact byte SHA mismatch: {role}")
    payload = validate_semantic_seal(read_json(target))
    if payload.get("semantic_sha256") != binding.get("semantic_sha256"):
        raise V17StateMachineError(f"artifact semantic SHA mismatch: {role}")


def validate_ledger(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, LEDGER_KEYS, label="v17 shadow ledger")
    if sealed.get("version") != LEDGER_VERSION:
        raise V17StateMachineError("shadow ledger version mismatch")
    run_id = require_identifier(sealed.get("run_id"), label="run_id")
    require_identifier(sealed.get("strategy_id"), label="strategy_id")
    if sealed.get("market") != "CN":
        raise V17StateMachineError("shadow ledger market must be CN")
    parse_utc_timestamp(sealed.get("cutoff"), label="cutoff")
    created_at = parse_utc_timestamp(sealed.get("created_at"), label="created_at")
    updated_at = parse_utc_timestamp(sealed.get("updated_at"), label="updated_at")
    if updated_at < created_at:
        raise V17StateMachineError("ledger updated_at precedes created_at")
    require_authority_false(sealed.get("authority"))

    machine = _load_machine_definition()
    state = sealed.get("state")
    if state not in machine["transitions"]:
        raise V17StateMachineError("shadow ledger state unknown")
    sequence = sealed.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise V17StateMachineError("shadow ledger sequence invalid")
    previous = sealed.get("previous_ledger_sha256")
    if sequence == 0:
        if previous != EMPTY_SHA:
            raise V17StateMachineError("initial ledger predecessor must be EMPTY")
    else:
        require_sha256(previous, label="previous ledger SHA-256")
    if not isinstance(sealed.get("input_bindings"), Mapping):
        raise V17StateMachineError("ledger input_bindings must be an object")
    history = sealed.get("history")
    if not isinstance(history, list) or len(history) != sequence + 1:
        raise V17StateMachineError("ledger history length mismatch")
    previous_state = EMPTY_SHA
    previous_at = None
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise V17StateMachineError("ledger history item must be an object")
        require_exact_keys(item, HISTORY_KEYS, label=f"ledger history[{index}]")
        if item.get("sequence") != index:
            raise V17StateMachineError("ledger history sequence mismatch")
        if item.get("from_state") != previous_state:
            raise V17StateMachineError("ledger history state chain mismatch")
        to_state = item.get("to_state")
        if to_state not in machine["transitions"]:
            raise V17StateMachineError("ledger history target unknown")
        if index == 0:
            if to_state != machine["initial_state"]:
                raise V17StateMachineError("ledger history initial state mismatch")
            if item.get("expected_ledger_sha256") != EMPTY_SHA:
                raise V17StateMachineError("initial expected ledger SHA must be EMPTY")
        else:
            if to_state not in machine["transitions"][previous_state]:
                raise V17StateMachineError("ledger history transition invalid")
            require_sha256(
                item.get("expected_ledger_sha256"),
                label="history expected ledger SHA-256",
            )
        item_at = parse_utc_timestamp(item.get("at"), label="history.at")
        if previous_at is not None and item_at < previous_at:
            raise V17StateMachineError("ledger history timestamp regressed")
        previous_at = item_at
        roles = item.get("artifact_roles")
        if (
            not isinstance(roles, list)
            or roles != sorted(roles)
            or len(roles) != len(set(roles))
            or any(not isinstance(role, str) for role in roles)
        ):
            raise V17StateMachineError("ledger history artifact roles invalid")
        previous_state = str(to_state)
    if previous_state != state:
        raise V17StateMachineError("ledger history does not end at current state")
    if history[-1]["at"] != sealed.get("updated_at"):
        raise V17StateMachineError("ledger updated_at/history mismatch")
    if sequence > 0 and history[-1]["expected_ledger_sha256"] != previous:
        raise V17StateMachineError("ledger predecessor/history mismatch")

    artifacts = sealed.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise V17StateMachineError("ledger artifacts must be an object")
    for role, binding in artifacts.items():
        if not isinstance(binding, Mapping):
            raise V17StateMachineError(f"artifact binding invalid: {role}")
        _validate_artifact_binding(
            role,
            binding,
            repo_root=root,
            maximum_sequence=sequence,
            verify_payload=verify_artifacts,
        )
    if run_id != Path(str(run_id)).name:
        raise V17StateMachineError("run_id is not path-safe")
    return sealed


def _run_paths(repo_root: Path, run_id: str) -> dict[str, Path]:
    safe = require_identifier(run_id, label="run_id")
    results = repo_root / "results" / "v17_shadow"
    run = results / "runs" / safe
    return {
        "results": results,
        "run": run,
        "ledger": run / "ledger.json",
        "lock": run / ".ledger.lock",
        "events": run / "events",
        "receipts": run / "receipts",
        "outcomes": results / "outcomes",
        "latest_lock": results / "_latest" / ".latest.lock",
    }


def load_run_ledger(
    repo_root: str | Path,
    run_id: str,
    *,
    verify_artifacts: bool = True,
) -> tuple[dict[str, Any], str]:
    root = _repo_root(repo_root)
    path = _run_paths(root, run_id)["ledger"]
    before = file_sha256(path)
    payload = read_json(path)
    after = file_sha256(path)
    if before != after:
        raise V17StateMachineError("ledger changed during read")
    validated = validate_ledger(
        payload,
        repo_root=root,
        verify_artifacts=verify_artifacts,
    )
    return validated, before


@contextmanager
def _locked_run(repo_root: Path, run_id: str) -> Iterator[dict[str, Path]]:
    paths = _run_paths(repo_root, run_id)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(paths["lock"], flags)
    except OSError as exc:
        raise V17StateMachineError("run ledger lock unavailable") from exc
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o600
        ):
            raise V17StateMachineError("run ledger lock identity invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield paths
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def initialize_run(
    repo_root: str | Path,
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    prepared_at: str,
    input_bindings: Mapping[str, Any],
    expected_ledger_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Atomically create a PREPARED run from the explicit EMPTY CAS sentinel."""

    if _validate_expected_sha(expected_ledger_sha256, allow_empty=True) != EMPTY_SHA:
        raise V17LedgerCASMismatch("new run expected ledger SHA must be EMPTY")
    root = _repo_root(repo_root)
    run = require_identifier(run_id, label="run_id")
    strategy = require_identifier(strategy_id, label="strategy_id")
    parse_utc_timestamp(cutoff, label="cutoff")
    parse_utc_timestamp(prepared_at, label="prepared_at")
    if not isinstance(input_bindings, Mapping):
        raise V17StateMachineError("input_bindings must be an object")
    canonical_json_bytes(input_bindings)
    _load_machine_definition()

    paths = _run_paths(root, run)
    if os.path.lexists(paths["run"]):
        raise V17LedgerCASMismatch("run already exists; EMPTY CAS rejected")
    layout = ensure_v17_shadow_layout(root)
    if os.path.lexists(paths["run"]):
        raise V17LedgerCASMismatch("run already exists; EMPTY CAS rejected")
    _ensure_lock_file(paths["latest_lock"], root=layout["results"])

    ledger = seal_semantic(
        {
            "version": LEDGER_VERSION,
            "run_id": run,
            "strategy_id": strategy,
            "market": "CN",
            "cutoff": cutoff,
            "state": "PREPARED",
            "sequence": 0,
            "created_at": prepared_at,
            "updated_at": prepared_at,
            "previous_ledger_sha256": EMPTY_SHA,
            "input_bindings": dict(input_bindings),
            "artifacts": {},
            "history": [
                {
                    "sequence": 0,
                    "from_state": EMPTY_SHA,
                    "to_state": "PREPARED",
                    "at": prepared_at,
                    "expected_ledger_sha256": EMPTY_SHA,
                    "artifact_roles": [],
                }
            ],
            "authority": False,
        }
    )
    temporary = layout["runs"] / f".{run}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        _write_exclusive_bytes(
            temporary / ".ledger.lock",
            b"",
            root=temporary,
        )
        _write_exclusive_json(
            temporary / "ledger.json",
            ledger,
            root=temporary,
        )
        (temporary / "events").mkdir(mode=0o700)
        (temporary / "receipts").mkdir(mode=0o700)
        _fsync_directory(temporary)
        os.rename(temporary, paths["run"])
        _fsync_directory(layout["runs"])
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
            _fsync_directory(layout["runs"])
        raise
    observed, ledger_sha = load_run_ledger(root, run)
    return observed, ledger_sha


def _artifact_binding(
    *,
    repo_root: Path,
    path: Path,
    byte_sha256: str,
    semantic_sha256: str,
    sequence: int,
    state: str,
) -> dict[str, Any]:
    relative = path.relative_to(repo_root).as_posix()
    return {
        "relative_path": relative,
        "byte_sha256": byte_sha256,
        "semantic_sha256": semantic_sha256,
        "sequence": sequence,
        "state": state,
    }


def _validate_terminal_output(
    payload: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    next_state: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, TERMINAL_OUTPUT_KEYS, label="terminal shadow output")
    if sealed.get("version") != TERMINAL_OUTPUT_VERSION:
        raise V17StateMachineError("terminal output version mismatch")
    for key in ("run_id", "strategy_id", "market", "cutoff"):
        if sealed.get(key) != ledger.get(key):
            raise V17StateMachineError(f"terminal output binding mismatch: {key}")
    if sealed.get("terminal_state") != next_state:
        raise V17StateMachineError("terminal output state mismatch")
    if sealed.get("ledger_predecessor_sha256") != expected_ledger_sha256:
        raise V17StateMachineError("terminal output predecessor SHA mismatch")
    source_manifest_sha = require_sha256(
        sealed.get("source_manifest_sha256"),
        label="terminal source manifest SHA-256",
    )
    input_bindings = ledger.get("input_bindings")
    if not isinstance(input_bindings, Mapping) or source_manifest_sha != require_sha256(
        input_bindings.get("source_manifest_sha256"),
        label="ledger source manifest SHA-256",
    ):
        raise V17StateMachineError("terminal source manifest SHA must match ledger input binding")
    parse_utc_timestamp(sealed.get("generated_at"), label="generated_at")
    require_authority_false(sealed.get("authority"))
    blockers = sealed.get("blockers")
    if (
        not isinstance(blockers, list)
        or len(blockers) != len(set(blockers))
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        raise V17StateMachineError("terminal blockers invalid")
    if next_state == "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION":
        if not isinstance(sealed.get("rank_output"), Mapping) or not isinstance(
            sealed.get("portfolio_output"), Mapping
        ):
            raise V17StateMachineError("complete terminal requires rank and portfolio")
        if blockers:
            raise V17StateMachineError("complete terminal cannot carry blockers")
    elif next_state in {
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "SHADOW_PORTFOLIO_INFEASIBLE",
    }:
        if not isinstance(sealed.get("rank_output"), Mapping):
            raise V17StateMachineError("rank terminal requires rank output")
        if sealed.get("portfolio_output") is not None or not blockers:
            raise V17StateMachineError("non-portfolio terminal shape invalid")
    else:
        if sealed.get("rank_output") is not None or sealed.get("portfolio_output") is not None:
            raise V17StateMachineError("hard stop cannot carry rank or portfolio output")
        if not blockers:
            raise V17StateMachineError("hard stop requires blockers")
    return sealed


def _write_unpublished_receipt(
    *,
    paths: Mapping[str, Path],
    repo_root: Path,
    run_id: str,
    current_state: str,
    requested_state: str,
    expected_ledger_sha256: str,
    attempted_at: str,
    attempt_id: str,
    error: Exception,
    ledger_commit_status: str,
) -> None:
    receipt = seal_semantic(
        {
            "version": UNPUBLISHED_RECEIPT_VERSION,
            "status": "UNPUBLISHED",
            "run_id": run_id,
            "current_state": current_state,
            "requested_state": requested_state,
            "expected_ledger_sha256": expected_ledger_sha256,
            "attempted_at": attempted_at,
            "attempt_id": attempt_id,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "ledger_commit_status": ledger_commit_status,
            "authority": False,
        }
    )
    receipt_path = paths["receipts"] / f"{attempt_id}.json"
    _write_exclusive_json(receipt_path, receipt, root=paths["run"])


def _cleanup_unpublished_artifacts(paths: list[Path], directories: list[Path]) -> None:
    for path in reversed(paths):
        if os.path.lexists(path):
            path.unlink()
            _fsync_directory(path.parent)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if directory.exists():
            try:
                directory.rmdir()
            except OSError:
                continue
            _fsync_directory(directory.parent)


def _advance_run_state(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    next_state: str,
    transitioned_at: str,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_output: Mapping[str, Any] | None = None,
    verify_existing_artifacts: bool,
) -> tuple[dict[str, Any], str]:
    """Commit one CAS transition and optional immutable JSON artifacts."""

    expected = _validate_expected_sha(expected_ledger_sha256, allow_empty=False)
    root = _repo_root(repo_root)
    safe_run = require_identifier(run_id, label="run_id")
    require_identifier(next_state, label="next_state")
    parse_utc_timestamp(transitioned_at, label="transitioned_at")
    supplied_artifacts = dict(artifacts or {})

    with _locked_run(root, safe_run) as paths:
        ledger, observed_sha = load_run_ledger(
            root,
            safe_run,
            verify_artifacts=verify_existing_artifacts,
        )
        if observed_sha != expected:
            raise V17LedgerCASMismatch("ledger CAS mismatch; transition performed zero writes")
        attempt_id = uuid.uuid4().hex
        created_paths: list[Path] = []
        created_directories: list[Path] = []
        ledger_committed = False
        try:
            machine = _load_machine_definition()
            current_state = str(ledger["state"])
            if next_state not in machine["transitions"][current_state]:
                raise V17StateMachineError(
                    f"transition not permitted: {current_state}->{next_state}"
                )
            if parse_utc_timestamp(transitioned_at, label="transitioned_at") < parse_utc_timestamp(
                ledger["updated_at"], label="ledger.updated_at"
            ):
                raise V17StateMachineError("transition timestamp regressed")
            is_terminal = next_state in set(machine["terminal_states"])
            if is_terminal != (terminal_output is not None):
                raise V17StateMachineError(
                    "terminal transitions require exactly one terminal output"
                )
            sequence = int(ledger["sequence"]) + 1
            prepared_artifacts: dict[str, dict[str, Any]] = {}
            for role, raw_payload in sorted(supplied_artifacts.items()):
                require_identifier(role, label="artifact role")
                if not isinstance(raw_payload, Mapping):
                    raise V17StateMachineError(f"artifact payload invalid: {role}")
                payload = validate_semantic_seal(raw_payload)
                require_authority_false(payload.get("authority"))
                prepared_artifacts[role] = payload

            output_payload: dict[str, Any] | None = None
            if terminal_output is not None:
                output_payload = _validate_terminal_output(
                    terminal_output,
                    ledger=ledger,
                    next_state=next_state,
                    expected_ledger_sha256=expected,
                )

            event_dir = paths["events"] / f"{sequence:04d}-{next_state.lower()}-{attempt_id}"
            new_bindings: dict[str, dict[str, Any]] = {}
            for role, payload in prepared_artifacts.items():
                target = event_dir / f"{role}.json"
                byte_sha = _write_exclusive_json(
                    target,
                    payload,
                    root=paths["run"],
                )
                created_paths.append(target)
                created_directories.append(event_dir)
                new_bindings[role] = _artifact_binding(
                    repo_root=root,
                    path=target,
                    byte_sha256=byte_sha,
                    semantic_sha256=str(payload["semantic_sha256"]),
                    sequence=sequence,
                    state=next_state,
                )

            if output_payload is not None:
                output_path = paths["outcomes"] / f"{safe_run}-{sequence:04d}-{attempt_id}.json"
                output_sha = _write_exclusive_json(
                    output_path,
                    output_payload,
                    root=paths["results"],
                )
                created_paths.append(output_path)
                new_bindings["terminal_output"] = _artifact_binding(
                    repo_root=root,
                    path=output_path,
                    byte_sha256=output_sha,
                    semantic_sha256=str(output_payload["semantic_sha256"]),
                    sequence=sequence,
                    state=next_state,
                )

            artifact_bindings = dict(ledger["artifacts"])
            artifact_bindings.update(new_bindings)
            history = list(ledger["history"])
            history.append(
                {
                    "sequence": sequence,
                    "from_state": current_state,
                    "to_state": next_state,
                    "at": transitioned_at,
                    "expected_ledger_sha256": expected,
                    "artifact_roles": sorted(new_bindings),
                }
            )
            successor = seal_semantic(
                {
                    "version": LEDGER_VERSION,
                    "run_id": ledger["run_id"],
                    "strategy_id": ledger["strategy_id"],
                    "market": ledger["market"],
                    "cutoff": ledger["cutoff"],
                    "state": next_state,
                    "sequence": sequence,
                    "created_at": ledger["created_at"],
                    "updated_at": transitioned_at,
                    "previous_ledger_sha256": expected,
                    "input_bindings": dict(ledger["input_bindings"]),
                    "artifacts": artifact_bindings,
                    "history": history,
                    "authority": False,
                }
            )
            successor_sha = atomic_write_json(
                paths["ledger"],
                successor,
                root=paths["run"],
            )
            ledger_committed = True
            readback, readback_sha = load_run_ledger(
                root,
                safe_run,
                verify_artifacts=verify_existing_artifacts,
            )
            if readback != successor or readback_sha != successor_sha:
                raise V17PostCommitReadbackError(
                    "successor ledger post-commit readback mismatch; latest was not touched"
                )
            return readback, readback_sha
        except V17LedgerCASMismatch:
            raise
        except Exception as exc:
            if not ledger_committed:
                try:
                    _cleanup_unpublished_artifacts(
                        created_paths,
                        created_directories,
                    )
                except Exception:
                    pass
            try:
                _write_unpublished_receipt(
                    paths=paths,
                    repo_root=root,
                    run_id=safe_run,
                    current_state=str(ledger["state"]),
                    requested_state=next_state,
                    expected_ledger_sha256=expected,
                    attempted_at=transitioned_at,
                    attempt_id=attempt_id,
                    error=exc,
                    ledger_commit_status=(
                        "POST_COMMIT_UNCERTAIN" if ledger_committed else "NOT_COMMITTED"
                    ),
                )
            except Exception:
                pass
            if isinstance(exc, V17StateMachineError):
                raise
            raise V17StateMachineError(str(exc)) from exc


def advance_run_state(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    next_state: str,
    transitioned_at: str,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_output: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Commit a normal transition after verifying every existing artifact."""

    return _advance_run_state(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=expected_ledger_sha256,
        next_state=next_state,
        transitioned_at=transitioned_at,
        artifacts=artifacts,
        terminal_output=terminal_output,
        verify_existing_artifacts=True,
    )


def advance_snapshot_drift_hard_stop(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    transitioned_at: str,
    terminal_output: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Commit the one terminal allowed when a prior artifact cannot be read.

    Ledger bytes, semantic seal, history, CAS, artifact binding metadata, and
    the new terminal output are still validated.  Only payload readback of
    already-bound artifacts is skipped.  No other state or artifact write can
    use this path.
    """

    return _advance_run_state(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=expected_ledger_sha256,
        next_state="HARD_STOP_SNAPSHOT_DRIFT",
        transitioned_at=transitioned_at,
        artifacts=None,
        terminal_output=terminal_output,
        verify_existing_artifacts=False,
    )


__all__ = [
    "ARTIFACT_BINDING_KEYS",
    "EMPTY_SHA",
    "LEDGER_KEYS",
    "LEDGER_VERSION",
    "STATE_MACHINE_RESOURCE_SHA256",
    "TERMINAL_OUTPUT_KEYS",
    "TERMINAL_OUTPUT_VERSION",
    "UNPUBLISHED_RECEIPT_VERSION",
    "V17LedgerCASMismatch",
    "V17PostCommitReadbackError",
    "V17StateMachineError",
    "advance_snapshot_drift_hard_stop",
    "advance_run_state",
    "initialize_run",
    "is_terminal_state",
    "load_run_ledger",
    "terminal_states",
    "validate_ledger",
]
