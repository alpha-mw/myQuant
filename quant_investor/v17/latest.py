"""Terminal-only latest pointer publication and explicit repair for v17."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterator

from .contracts import (
    V17ContractError,
    parse_utc_timestamp,
    require_authority_false,
    require_exact_keys,
    require_identifier,
)
from .semantic import require_sha256, seal_semantic, validate_semantic_seal
from .state_machine import (
    ARTIFACT_BINDING_KEYS,
    EMPTY_SHA,
    TERMINAL_OUTPUT_KEYS,
    TERMINAL_OUTPUT_VERSION,
    V17LedgerCASMismatch,
    is_terminal_state,
    load_run_ledger,
)
from .storage import atomic_write_json, file_sha256, read_json

LATEST_POINTER_VERSION = "myquant.v17.shadow-latest-pointer.v1"
LATEST_POINTER_KEYS = frozenset(
    {
        "version",
        "run_id",
        "terminal_state",
        "ledger_path",
        "ledger_sha256",
        "output_path",
        "output_sha256",
        "published_at",
        "publication_mode",
        "authority",
        "semantic_sha256",
    }
)
PUBLICATION_MODES = frozenset({"NORMAL", "REPAIR"})


class V17LatestError(V17ContractError):
    """A terminal latest pointer cannot be validated or CAS-published."""


class V17LatestPostCommitReadbackError(V17LatestError):
    """The latest pointer replace occurred, but its readback failed closed."""


def _load_terminal_ledger_for_latest(
    repo_root: Path,
    run_id: str,
) -> tuple[dict[str, Any], str]:
    ledger, ledger_sha = load_run_ledger(
        repo_root,
        run_id,
        verify_artifacts=False,
    )
    if ledger.get("state") != "HARD_STOP_SNAPSHOT_DRIFT":
        return load_run_ledger(repo_root, run_id, verify_artifacts=True)
    return ledger, ledger_sha


def _root(repo_root: str | Path) -> Path:
    root = Path(repo_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise V17LatestError("repository root unavailable or symlinked")
    return root


def _paths(repo_root: Path) -> dict[str, Path]:
    latest_root = repo_root / "results" / "v17_shadow" / "_latest"
    return {
        "results": repo_root / "results" / "v17_shadow",
        "pointer": latest_root / "shadow.json",
        "lock": latest_root / ".latest.lock",
    }


def _require_expected(value: str, *, label: str, allow_empty: bool) -> str:
    if allow_empty and value == EMPTY_SHA:
        return value
    try:
        return require_sha256(value, label=label)
    except ValueError as exc:
        raise V17LatestError(str(exc)) from exc


@contextmanager
def _locked_latest(repo_root: Path) -> Iterator[dict[str, Path]]:
    paths = _paths(repo_root)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(paths["lock"], flags)
    except OSError as exc:
        raise V17LatestError("latest lock unavailable; prepare a run first") from exc
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o600
        ):
            raise V17LatestError("latest lock identity invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield paths
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_latest_pointer(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_targets: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, LATEST_POINTER_KEYS, label="v17 latest pointer")
    if sealed.get("version") != LATEST_POINTER_VERSION:
        raise V17LatestError("latest pointer version mismatch")
    run_id = require_identifier(sealed.get("run_id"), label="run_id")
    state = str(sealed.get("terminal_state") or "")
    if not is_terminal_state(state):
        raise V17LatestError("latest pointer must reference an immutable terminal")
    parse_utc_timestamp(sealed.get("published_at"), label="published_at")
    if sealed.get("publication_mode") not in PUBLICATION_MODES:
        raise V17LatestError("latest pointer publication mode invalid")
    require_authority_false(sealed.get("authority"))
    ledger_sha = require_sha256(sealed.get("ledger_sha256"), label="latest ledger SHA-256")
    output_sha = require_sha256(sealed.get("output_sha256"), label="latest output SHA-256")
    expected_ledger = PurePosixPath("results/v17_shadow/runs") / run_id / "ledger.json"
    ledger_relative = PurePosixPath(str(sealed.get("ledger_path") or ""))
    if ledger_relative != expected_ledger:
        raise V17LatestError("latest ledger path mismatch")
    output_relative = PurePosixPath(str(sealed.get("output_path") or ""))
    if (
        output_relative.is_absolute()
        or ".." in output_relative.parts
        or output_relative.parts[:3] != ("results", "v17_shadow", "outcomes")
    ):
        raise V17LatestError("latest output path mismatch")
    if verify_targets:
        ledger, observed_ledger_sha = _load_terminal_ledger_for_latest(root, run_id)
        if observed_ledger_sha != ledger_sha or ledger.get("state") != state:
            raise V17LatestError("latest ledger target mismatch")
        output_path = root / Path(*output_relative.parts)
        if file_sha256(output_path) != output_sha:
            raise V17LatestError("latest output target SHA mismatch")
        output = validate_semantic_seal(read_json(output_path))
        require_exact_keys(output, TERMINAL_OUTPUT_KEYS, label="terminal output")
        if (
            output.get("version") != TERMINAL_OUTPUT_VERSION
            or output.get("run_id") != run_id
            or output.get("terminal_state") != state
        ):
            raise V17LatestError("latest terminal output identity mismatch")
        binding = ledger.get("artifacts", {}).get("terminal_output")
        if not isinstance(binding, Mapping):
            raise V17LatestError("terminal output binding missing from ledger")
        require_exact_keys(
            binding,
            ARTIFACT_BINDING_KEYS,
            label="terminal output binding",
        )
        if (
            binding.get("relative_path") != output_relative.as_posix()
            or binding.get("byte_sha256") != output_sha
        ):
            raise V17LatestError("latest/ledger output binding mismatch")
    return sealed


def read_latest_pointer(
    repo_root: str | Path,
    *,
    verify_targets: bool = True,
) -> tuple[dict[str, Any], str] | None:
    root = _root(repo_root)
    target = _paths(root)["pointer"]
    if not os.path.lexists(target):
        return None
    before = file_sha256(target)
    payload = read_json(target)
    after = file_sha256(target)
    if before != after:
        raise V17LatestError("latest pointer changed during read")
    return (
        validate_latest_pointer(
            payload,
            repo_root=root,
            verify_targets=verify_targets,
        ),
        before,
    )


def _publish(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    published_at: str,
    publication_mode: str,
) -> tuple[dict[str, Any], str]:
    root = _root(repo_root)
    safe_run = require_identifier(run_id, label="run_id")
    expected_ledger = _require_expected(
        expected_ledger_sha256,
        label="expected ledger SHA-256",
        allow_empty=False,
    )
    expected_latest = _require_expected(
        expected_latest_sha256,
        label="expected latest SHA-256",
        allow_empty=True,
    )
    parse_utc_timestamp(published_at, label="published_at")
    if publication_mode not in PUBLICATION_MODES:
        raise V17LatestError("publication mode invalid")

    with _locked_latest(root) as paths:
        if os.path.lexists(paths["pointer"]):
            observed_latest = file_sha256(paths["pointer"])
            if expected_latest == EMPTY_SHA or observed_latest != expected_latest:
                raise V17LedgerCASMismatch("latest CAS mismatch; publication performed zero writes")
        elif expected_latest != EMPTY_SHA:
            raise V17LedgerCASMismatch("latest CAS mismatch; publication performed zero writes")

        ledger, observed_ledger = _load_terminal_ledger_for_latest(root, safe_run)
        if observed_ledger != expected_ledger:
            raise V17LedgerCASMismatch(
                "ledger CAS mismatch; latest publication performed zero writes"
            )
        state = str(ledger["state"])
        if not is_terminal_state(state):
            raise V17LatestError("latest publication requires a terminal ledger")
        binding = ledger.get("artifacts", {}).get("terminal_output")
        if not isinstance(binding, Mapping):
            raise V17LatestError("terminal ledger lacks output binding")
        output_relative = PurePosixPath(str(binding.get("relative_path") or ""))
        output_path = root / Path(*output_relative.parts)
        output_sha = require_sha256(binding.get("byte_sha256"), label="terminal output SHA-256")
        if file_sha256(output_path) != output_sha:
            raise V17LatestError("terminal output drift blocks latest publication")
        pointer = seal_semantic(
            {
                "version": LATEST_POINTER_VERSION,
                "run_id": safe_run,
                "terminal_state": state,
                "ledger_path": (
                    Path("results/v17_shadow/runs") / safe_run / "ledger.json"
                ).as_posix(),
                "ledger_sha256": observed_ledger,
                "output_path": output_relative.as_posix(),
                "output_sha256": output_sha,
                "published_at": published_at,
                "publication_mode": publication_mode,
                "authority": False,
            }
        )
        validate_latest_pointer(pointer, repo_root=root, verify_targets=True)
        pointer_sha = atomic_write_json(
            paths["pointer"],
            pointer,
            root=paths["results"],
        )
        readback = read_latest_pointer(root, verify_targets=True)
        if readback is None or readback != (pointer, pointer_sha):
            raise V17LatestPostCommitReadbackError("latest pointer post-commit readback mismatch")
        return pointer, pointer_sha


def publish_terminal_latest(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    published_at: str,
) -> tuple[dict[str, Any], str]:
    return _publish(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        published_at=published_at,
        publication_mode="NORMAL",
    )


def repair_terminal_latest(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    repaired_at: str,
) -> tuple[dict[str, Any], str]:
    """Explicitly republish one already-terminal run after complete readback."""

    return _publish(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        published_at=repaired_at,
        publication_mode="REPAIR",
    )


__all__ = [
    "LATEST_POINTER_KEYS",
    "LATEST_POINTER_VERSION",
    "PUBLICATION_MODES",
    "V17LatestError",
    "V17LatestPostCommitReadbackError",
    "publish_terminal_latest",
    "read_latest_pointer",
    "repair_terminal_latest",
    "validate_latest_pointer",
]
