"""Journaled two-store promotion for CN Macro maintenance candidates.

The release-calendar and observation stores intentionally own their candidate
generation formats.  This module only transports already validated immutable
generations from an isolated private preparation root and switches their two
pointers under a fixed lock order.  A durable append-only journal makes every
interruption classifiable before an operator or automation elects to continue.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping

from quant_investor.macro.release_calendar import load_release_calendar
from quant_investor.macro.store import load_observations

TRANSACTION_SCHEMA = "cn-macro-dual-pointer-transaction.v1"
PREPARED_SCHEMA = "cn-macro-dual-pointer-prepared.v1"
JOURNAL_SCHEMA = "cn-macro-dual-pointer-journal.v1"
EMPTY_POINTER_SHA256 = "EMPTY"
PHASES = (
    "INTENT",
    "BOTH_GENERATIONS_PREPARED",
    "BOTH_GENERATIONS_INSTALLED",
    "RELEASE_POINTER_COMMITTED",
    "OBSERVATIONS_POINTER_COMMITTED",
    "POSTCHECK_PASSED",
    "TERMINAL",
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_POINTER = "_latest.json"
_GENERATIONS = "_generations"
_MAX_JSON_BYTES = 8 * 1024 * 1024


class MacroMaintenanceTransactionError(RuntimeError):
    """Fail-closed transaction error with a machine-readable status."""

    def __init__(self, message: str, *, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.status = status


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_id(value: str, blocker: str) -> str:
    text = str(value or "")
    if not _SAFE_ID_RE.fullmatch(text):
        raise MacroMaintenanceTransactionError(blocker)
    return text


def _required_sha(value: Any, blocker: str, *, empty: bool = False) -> str:
    raw = str(value or "")
    if empty and raw == EMPTY_POINTER_SHA256:
        return EMPTY_POINTER_SHA256
    text = raw.lower()
    if _SHA_RE.fullmatch(text):
        return text
    raise MacroMaintenanceTransactionError(blocker)


def _absolute_directory(path: str | Path, blocker: str) -> Path:
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        raise MacroMaintenanceTransactionError(blocker)
    try:
        current = os.lstat(unresolved)
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise MacroMaintenanceTransactionError(blocker) from exc
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise MacroMaintenanceTransactionError(blocker)
    return resolved


def _private_directory(path: str | Path, blocker: str) -> Path:
    resolved = _absolute_directory(path, blocker)
    current = os.stat(resolved, follow_symlinks=False)
    if stat.S_IMODE(current.st_mode) & 0o077:
        raise MacroMaintenanceTransactionError(blocker)
    return resolved


def _regular_bytes(
    path: Path,
    blocker: str,
    *,
    max_bytes: int = _MAX_JSON_BYTES,
    expected_sha256: str | None = None,
) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise MacroMaintenanceTransactionError(blocker) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > max_bytes
    ):
        raise MacroMaintenanceTransactionError(blocker)
    raw = path.read_bytes()
    after = os.lstat(path)
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature != after_signature or len(raw) != before.st_size:
        raise MacroMaintenanceTransactionError(f"{blocker}_changed_during_read")
    if expected_sha256 is not None and _sha(raw) != expected_sha256:
        raise MacroMaintenanceTransactionError(f"{blocker}_sha256_mismatch")
    return raw


def _pointer_bytes(root: Path) -> tuple[bytes | None, str]:
    path = root / _POINTER
    if not os.path.lexists(path):
        return None, EMPTY_POINTER_SHA256
    raw = _regular_bytes(path, "macro_transaction_pointer_unsafe")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise MacroMaintenanceTransactionError("macro_transaction_pointer_json_invalid") from exc
    if not isinstance(payload, Mapping) or not _safe_id(
        str(payload.get("generation_id") or ""),
        "macro_transaction_pointer_generation_invalid",
    ):
        raise MacroMaintenanceTransactionError("macro_transaction_pointer_shape_invalid")
    return raw, _sha(raw)


def _generation_id(pointer_raw: bytes) -> str:
    try:
        payload = json.loads(pointer_raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - already checked by caller
        raise MacroMaintenanceTransactionError("macro_transaction_pointer_json_invalid") from exc
    if not isinstance(payload, Mapping):
        raise MacroMaintenanceTransactionError("macro_transaction_pointer_shape_invalid")
    return _safe_id(
        str(payload.get("generation_id") or ""),
        "macro_transaction_pointer_generation_invalid",
    )


def _tree_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise MacroMaintenanceTransactionError("macro_transaction_generation_unsafe")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        PurePosixPath(relative)
        current = os.lstat(path)
        if stat.S_ISLNK(current.st_mode):
            raise MacroMaintenanceTransactionError("macro_transaction_generation_symlink_rejected")
        if stat.S_ISDIR(current.st_mode):
            entries.append({"path": relative, "kind": "directory"})
            continue
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise MacroMaintenanceTransactionError("macro_transaction_generation_file_unsafe")
        raw = _regular_bytes(
            path,
            "macro_transaction_generation_file_unsafe",
            max_bytes=max(_MAX_JSON_BYTES, current.st_size),
        )
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "size_bytes": len(raw),
                "sha256": _sha(raw),
            }
        )
    if not entries or not (root / "manifest.json").is_file():
        raise MacroMaintenanceTransactionError("macro_transaction_generation_incomplete")
    return _sha(_json_bytes({"entries": entries}))


def generation_tree_sha256(path: str | Path) -> str:
    """Return the transaction's exact immutable generation-tree digest."""

    return _tree_digest(Path(path).expanduser().resolve(strict=True))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes, blocker: str) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise MacroMaintenanceTransactionError(blocker) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_pointer(root: Path, raw: bytes) -> str:
    path = root / _POINTER
    if os.path.lexists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
        raise MacroMaintenanceTransactionError("macro_transaction_pointer_symlink_rejected")
    descriptor, name = tempfile.mkstemp(prefix="._latest.", suffix=".tmp", dir=root)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()
    persisted, digest = _pointer_bytes(root)
    if persisted != raw:
        raise MacroMaintenanceTransactionError(
            "macro_transaction_pointer_readback_mismatch",
            status="PROMOTION_UNCERTAIN",
        )
    return digest


@contextmanager
def _store_lock(root: Path, filename: str, blocker: str) -> Iterator[None]:
    path = root / filename
    if os.path.lexists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
        raise MacroMaintenanceTransactionError(blocker)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise MacroMaintenanceTransactionError(blocker)
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _transaction_locks(
    authorities: Mapping[str, Mapping[str, Any]],
    release_root: Path,
    observations_root: Path,
) -> Iterator[None]:
    """Acquire the cross-store authority locks in the only permitted order."""

    market_root = Path(authorities["market"]["pointer_path"]).parent
    pit_root = Path(authorities["pit"]["pointer_path"]).parent
    lock_bindings = (
        (
            market_root,
            ".market_writer.lock",
            "macro_transaction_market_lock_unsafe",
        ),
        (
            pit_root,
            ".pit_writer.lock",
            "macro_transaction_pit_lock_unsafe",
        ),
        (
            release_root,
            ".release-calendar.lock",
            "macro_transaction_release_lock_unsafe",
        ),
        (
            observations_root,
            ".promotion.lock",
            "macro_transaction_observations_lock_unsafe",
        ),
    )
    lock_paths = [root / filename for root, filename, _blocker in lock_bindings]
    if len(set(lock_paths)) != 4:
        raise MacroMaintenanceTransactionError("macro_transaction_lock_paths_not_distinct")
    with ExitStack() as stack:
        for root, filename, blocker in lock_bindings:
            stack.enter_context(_store_lock(root, filename, blocker))
        yield


def _copy_generation(source: Path, destination: Path, expected_tree_sha: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            if _tree_digest(destination) == expected_tree_sha:
                return
        raise MacroMaintenanceTransactionError(
            "macro_transaction_generation_no_clobber",
            status="PROMOTION_UNCERTAIN",
        )
    generations = destination.parent
    if generations.is_symlink() or not generations.is_dir():
        raise MacroMaintenanceTransactionError("macro_transaction_generations_root_unsafe")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=generations))
    os.chmod(staging, 0o700)
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=False)
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        if _tree_digest(staging) != expected_tree_sha:
            raise MacroMaintenanceTransactionError("macro_transaction_generation_copy_mismatch")
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(generations)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _prepared_path(prepared_root: Path) -> Path:
    return prepared_root / "prepared.json"


def _authority_pointer(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, bytes, str]:
    pointer = Path(path).expanduser()
    if not pointer.is_absolute():
        raise MacroMaintenanceTransactionError(f"macro_transaction_{label}_pointer_path_invalid")
    expected = _required_sha(
        expected_sha256,
        f"macro_transaction_{label}_pointer_sha_invalid",
    )
    try:
        resolved = pointer.resolve(strict=True)
    except OSError as exc:
        raise MacroMaintenanceTransactionError(f"macro_transaction_{label}_pointer_unsafe") from exc
    if resolved != pointer:
        raise MacroMaintenanceTransactionError(f"macro_transaction_{label}_pointer_path_not_exact")
    raw = _regular_bytes(
        resolved,
        f"macro_transaction_{label}_pointer_unsafe",
        expected_sha256=expected,
    )
    return resolved, raw, expected


def _prepared_authorities(
    prepared_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_authorities = payload.get("authorities")
    if not isinstance(raw_authorities, Mapping) or set(raw_authorities) != {
        "market",
        "pit",
    }:
        raise MacroMaintenanceTransactionError("macro_transaction_authorities_invalid")
    artifact_root = prepared_path.parent.resolve()
    result: dict[str, dict[str, Any]] = {}
    for label in ("market", "pit"):
        raw = raw_authorities.get(label)
        if not isinstance(raw, Mapping):
            raise MacroMaintenanceTransactionError(f"macro_transaction_{label}_authority_invalid")
        pointer_path = Path(str(raw.get("pointer_path") or ""))
        if not pointer_path.is_absolute() or pointer_path.resolve() != pointer_path:
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_pointer_path_invalid"
            )
        pointer_sha = _required_sha(
            raw.get("pointer_sha256"),
            f"macro_transaction_{label}_pointer_sha_invalid",
        )
        relative = PurePosixPath(str(raw.get("pointer_artifact") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_pointer_artifact_path_unsafe"
            )
        artifact = artifact_root.joinpath(*relative.parts).resolve(strict=True)
        if artifact_root not in artifact.parents:
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_pointer_artifact_path_unsafe"
            )
        pointer_raw = _regular_bytes(
            artifact,
            f"macro_transaction_{label}_pointer_artifact_unsafe",
            expected_sha256=pointer_sha,
        )
        result[label] = {
            "pointer_path": pointer_path,
            "pointer_sha256": pointer_sha,
            "pointer_artifact": artifact,
            "pointer_raw": pointer_raw,
        }
    if result["market"]["pointer_path"] == result["pit"]["pointer_path"]:
        raise MacroMaintenanceTransactionError(
            "macro_transaction_authority_pointer_paths_not_distinct"
        )
    return result


def _revalidate_authorities(
    authorities: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint: str,
) -> None:
    for label in ("market", "pit"):
        authority = authorities[label]
        try:
            raw = _regular_bytes(
                Path(authority["pointer_path"]),
                f"macro_transaction_{label}_authority_unsafe",
                expected_sha256=str(authority["pointer_sha256"]),
            )
        except MacroMaintenanceTransactionError as exc:
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_authority_drift:{checkpoint}",
                status="PROMOTION_UNCERTAIN",
            ) from exc
        if raw != authority["pointer_raw"]:
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_authority_bytes_drift:{checkpoint}",
                status="PROMOTION_UNCERTAIN",
            )


def _assert_authority_arguments(
    authorities: Mapping[str, Mapping[str, Any]],
    *,
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
) -> None:
    supplied = {
        "market": (
            Path(market_pointer_path).expanduser(),
            _required_sha(
                expected_market_pointer_sha256,
                "macro_transaction_market_pointer_sha_invalid",
            ),
        ),
        "pit": (
            Path(pit_pointer_path).expanduser(),
            _required_sha(
                expected_pit_pointer_sha256,
                "macro_transaction_pit_pointer_sha_invalid",
            ),
        ),
    }
    for label, (pointer_path, pointer_sha) in supplied.items():
        if not pointer_path.is_absolute():
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_pointer_path_invalid"
            )
        if (
            pointer_path != authorities[label]["pointer_path"]
            or pointer_sha != authorities[label]["pointer_sha256"]
        ):
            raise MacroMaintenanceTransactionError(
                f"macro_transaction_{label}_authority_argument_mismatch"
            )


def seal_prepared_macro_transaction(
    *,
    prepared_root: str | Path,
    release_candidate_root: str | Path,
    observations_candidate_root: str | Path,
    release_canonical_root: str | Path,
    observations_canonical_root: str | Path,
    expected_release_pointer_sha256: str,
    expected_observations_pointer_sha256: str,
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
    authority_mode: str,
    target_date: str,
    input_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Seal candidates plus exact Market, PIT and Macro pointer authority."""

    prepared = _private_directory(prepared_root, "macro_transaction_prepared_root_not_private")
    release_candidate = _private_directory(
        release_candidate_root, "macro_transaction_release_candidate_unsafe"
    )
    observations_candidate = _private_directory(
        observations_candidate_root,
        "macro_transaction_observations_candidate_unsafe",
    )
    release_root = _absolute_directory(
        release_canonical_root, "macro_transaction_release_root_unsafe"
    )
    observations_root = _absolute_directory(
        observations_canonical_root,
        "macro_transaction_observations_root_unsafe",
    )
    target = str(target_date).replace("-", "")
    if len(target) != 8 or not target.isdigit():
        raise MacroMaintenanceTransactionError("macro_transaction_target_date_invalid")
    normalized_authority_mode = str(authority_mode or "").lower()
    if normalized_authority_mode not in {"canonical", "candidate"}:
        raise MacroMaintenanceTransactionError("macro_transaction_authority_mode_invalid")
    market_authority_path, market_authority_raw, market_authority_sha = _authority_pointer(
        market_pointer_path,
        expected_market_pointer_sha256,
        label="market",
    )
    pit_authority_path, pit_authority_raw, pit_authority_sha = _authority_pointer(
        pit_pointer_path,
        expected_pit_pointer_sha256,
        label="pit",
    )
    if market_authority_path == pit_authority_path:
        raise MacroMaintenanceTransactionError(
            "macro_transaction_authority_pointer_paths_not_distinct"
        )

    old_release_raw, old_release_sha = _pointer_bytes(release_root)
    old_observations_raw, old_observations_sha = _pointer_bytes(observations_root)
    expected_release = _required_sha(
        expected_release_pointer_sha256,
        "macro_transaction_expected_release_pointer_sha_invalid",
        empty=True,
    )
    expected_observations = _required_sha(
        expected_observations_pointer_sha256,
        "macro_transaction_expected_observations_pointer_sha_invalid",
        empty=True,
    )
    if old_release_sha != expected_release or old_observations_sha != expected_observations:
        raise MacroMaintenanceTransactionError("macro_transaction_parent_pointer_cas_mismatch")

    new_release_raw, new_release_sha = _pointer_bytes(release_candidate)
    new_observations_raw, new_observations_sha = _pointer_bytes(observations_candidate)
    if new_release_raw is None or new_observations_raw is None:
        raise MacroMaintenanceTransactionError("macro_transaction_candidate_pointer_missing")
    if new_release_sha == old_release_sha or new_observations_sha == old_observations_sha:
        raise MacroMaintenanceTransactionError("macro_transaction_candidate_did_not_advance")
    release_generation = _generation_id(new_release_raw)
    observations_generation = _generation_id(new_observations_raw)
    release_source = release_candidate / _GENERATIONS / release_generation
    observations_source = observations_candidate / _GENERATIONS / observations_generation
    release_tree_sha = _tree_digest(release_source)
    observations_tree_sha = _tree_digest(observations_source)
    dependency_generations: list[dict[str, str]] = []
    observations_generations = observations_candidate / _GENERATIONS
    canonical_observations_generations = observations_root / _GENERATIONS
    for dependency in sorted(observations_generations.iterdir(), key=lambda item: item.name):
        if dependency.name == observations_generation:
            continue
        _safe_id(dependency.name, "macro_transaction_dependency_generation_id_invalid")
        canonical_dependency = canonical_observations_generations / dependency.name
        dependency_sha = _tree_digest(dependency)
        if canonical_dependency.exists():
            if _tree_digest(canonical_dependency) != dependency_sha:
                raise MacroMaintenanceTransactionError(
                    "macro_transaction_dependency_generation_conflict"
                )
            continue
        dependency_generations.append(
            {
                "generation_id": dependency.name,
                "generation_tree_sha256": dependency_sha,
            }
        )

    bindings: dict[str, dict[str, str]] = {}
    for name, raw_binding in sorted((input_bindings or {}).items()):
        path = Path(str(raw_binding.get("path") or "")).expanduser()
        if not path.is_absolute():
            raise MacroMaintenanceTransactionError("macro_transaction_input_binding_path_invalid")
        digest = _required_sha(
            raw_binding.get("sha256"),
            "macro_transaction_input_binding_sha_invalid",
        )
        _regular_bytes(
            path.resolve(strict=True),
            "macro_transaction_input_binding_unsafe",
            max_bytes=max(_MAX_JSON_BYTES, path.stat().st_size),
            expected_sha256=digest,
        )
        bindings[str(name)] = {"path": str(path.resolve()), "sha256": digest}
    bindings["market_pointer_authority"] = {
        "path": str(market_authority_path),
        "sha256": market_authority_sha,
    }
    bindings["pit_pointer_authority"] = {
        "path": str(pit_authority_path),
        "sha256": pit_authority_sha,
    }

    artifacts = prepared / "artifacts"
    if os.path.lexists(artifacts):
        raise MacroMaintenanceTransactionError("macro_transaction_prepared_no_clobber")
    os.mkdir(artifacts, 0o700)
    files: dict[str, tuple[bytes, str]] = {
        "release_new_pointer.json": (new_release_raw, new_release_sha),
        "observations_new_pointer.json": (new_observations_raw, new_observations_sha),
        "market_authority_pointer.json": (
            market_authority_raw,
            market_authority_sha,
        ),
        "pit_authority_pointer.json": (pit_authority_raw, pit_authority_sha),
    }
    if old_release_raw is not None:
        files["release_old_pointer.json"] = (old_release_raw, old_release_sha)
    if old_observations_raw is not None:
        files["observations_old_pointer.json"] = (
            old_observations_raw,
            old_observations_sha,
        )
    for name, (raw, _digest) in files.items():
        _write_exclusive(
            artifacts / name,
            raw,
            "macro_transaction_prepared_no_clobber",
        )

    payload = {
        "schema_version": PREPARED_SCHEMA,
        "target_date": target,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "authority_mode": normalized_authority_mode,
        "release": {
            "canonical_root": str(release_root),
            "candidate_root": str(release_candidate),
            "generation_id": release_generation,
            "generation_tree_sha256": release_tree_sha,
            "old_pointer_sha256": old_release_sha,
            "new_pointer_sha256": new_release_sha,
            "old_pointer_artifact": (
                "artifacts/release_old_pointer.json" if old_release_raw is not None else ""
            ),
            "new_pointer_artifact": "artifacts/release_new_pointer.json",
        },
        "observations": {
            "canonical_root": str(observations_root),
            "candidate_root": str(observations_candidate),
            "generation_id": observations_generation,
            "generation_tree_sha256": observations_tree_sha,
            "old_pointer_sha256": old_observations_sha,
            "new_pointer_sha256": new_observations_sha,
            "old_pointer_artifact": (
                "artifacts/observations_old_pointer.json"
                if old_observations_raw is not None
                else ""
            ),
            "new_pointer_artifact": "artifacts/observations_new_pointer.json",
            "dependency_generations": dependency_generations,
        },
        "input_bindings": bindings,
        "authorities": {
            "market": {
                "pointer_path": str(market_authority_path),
                "pointer_sha256": market_authority_sha,
                "pointer_artifact": "artifacts/market_authority_pointer.json",
            },
            "pit": {
                "pointer_path": str(pit_authority_path),
                "pointer_sha256": pit_authority_sha,
                "pointer_artifact": "artifacts/pit_authority_pointer.json",
            },
        },
    }
    raw = _json_bytes(payload)
    _write_exclusive(_prepared_path(prepared), raw, "macro_transaction_prepared_no_clobber")
    return {
        **payload,
        "prepared_path": str(_prepared_path(prepared)),
        "prepared_sha256": _sha(raw),
    }


def _load_prepared(
    path: str | Path,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], bytes]:
    prepared_path = Path(path).expanduser()
    if not prepared_path.is_absolute():
        raise MacroMaintenanceTransactionError("macro_transaction_prepared_path_invalid")
    raw = _regular_bytes(
        prepared_path.resolve(strict=True),
        "macro_transaction_prepared_unsafe",
        expected_sha256=(
            _required_sha(expected_sha256, "macro_transaction_prepared_sha_invalid")
            if expected_sha256
            else None
        ),
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise MacroMaintenanceTransactionError("macro_transaction_prepared_json_invalid") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != PREPARED_SCHEMA:
        raise MacroMaintenanceTransactionError("macro_transaction_prepared_schema_invalid")
    return prepared_path, dict(payload), raw


def _prepared_component(
    prepared_path: Path, payload: Mapping[str, Any], name: str
) -> dict[str, Any]:
    raw = payload.get(name)
    if not isinstance(raw, Mapping):
        raise MacroMaintenanceTransactionError("macro_transaction_prepared_component_invalid")
    component = dict(raw)
    component["generation_id"] = _safe_id(
        str(component.get("generation_id") or ""),
        "macro_transaction_generation_id_invalid",
    )
    for key in (
        "generation_tree_sha256",
        "new_pointer_sha256",
    ):
        component[key] = _required_sha(component.get(key), f"macro_transaction_{key}_invalid")
    component["old_pointer_sha256"] = _required_sha(
        component.get("old_pointer_sha256"),
        "macro_transaction_old_pointer_sha_invalid",
        empty=True,
    )
    component["canonical_root"] = _absolute_directory(
        component.get("canonical_root"),
        f"macro_transaction_{name}_canonical_root_unsafe",
    )
    component["candidate_root"] = _private_directory(
        component.get("candidate_root"),
        f"macro_transaction_{name}_candidate_root_unsafe",
    )
    artifact_root = prepared_path.parent.resolve()
    for role in ("new_pointer_artifact", "old_pointer_artifact"):
        relative = str(component.get(role) or "")
        if not relative:
            if (
                role == "old_pointer_artifact"
                and component["old_pointer_sha256"] == EMPTY_POINTER_SHA256
            ):
                component[role] = None
                continue
            raise MacroMaintenanceTransactionError("macro_transaction_pointer_artifact_missing")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise MacroMaintenanceTransactionError("macro_transaction_pointer_artifact_path_unsafe")
        artifact = artifact_root.joinpath(*relative_path.parts).resolve(strict=True)
        if artifact_root not in artifact.parents:
            raise MacroMaintenanceTransactionError("macro_transaction_pointer_artifact_path_unsafe")
        expected = (
            component["new_pointer_sha256"]
            if role == "new_pointer_artifact"
            else component["old_pointer_sha256"]
        )
        component[role] = _regular_bytes(
            artifact,
            "macro_transaction_pointer_artifact_unsafe",
            expected_sha256=expected,
        )
    source = component["candidate_root"] / _GENERATIONS / component["generation_id"]
    if _tree_digest(source) != component["generation_tree_sha256"]:
        raise MacroMaintenanceTransactionError("macro_transaction_candidate_generation_drift")
    component["generation_source"] = source
    dependencies = component.get("dependency_generations", [])
    if not isinstance(dependencies, list):
        raise MacroMaintenanceTransactionError("macro_transaction_dependency_generations_invalid")
    normalized_dependencies: list[dict[str, Any]] = []
    seen_dependencies: set[str] = set()
    for raw_dependency in dependencies:
        if not isinstance(raw_dependency, Mapping) or set(raw_dependency) != {
            "generation_id",
            "generation_tree_sha256",
        }:
            raise MacroMaintenanceTransactionError(
                "macro_transaction_dependency_generation_invalid"
            )
        dependency_id = _safe_id(
            str(raw_dependency["generation_id"]),
            "macro_transaction_dependency_generation_id_invalid",
        )
        if dependency_id == component["generation_id"] or dependency_id in seen_dependencies:
            raise MacroMaintenanceTransactionError(
                "macro_transaction_dependency_generation_invalid"
            )
        seen_dependencies.add(dependency_id)
        dependency_sha = _required_sha(
            raw_dependency["generation_tree_sha256"],
            "macro_transaction_dependency_generation_sha_invalid",
        )
        dependency_source = component["candidate_root"] / _GENERATIONS / dependency_id
        if _tree_digest(dependency_source) != dependency_sha:
            raise MacroMaintenanceTransactionError("macro_transaction_dependency_generation_drift")
        normalized_dependencies.append(
            {
                "generation_id": dependency_id,
                "generation_tree_sha256": dependency_sha,
                "generation_source": dependency_source,
            }
        )
    component["dependency_generations"] = normalized_dependencies
    return component


def _revalidate_inputs(payload: Mapping[str, Any]) -> None:
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise MacroMaintenanceTransactionError("macro_transaction_input_bindings_invalid")
    for raw in bindings.values():
        if not isinstance(raw, Mapping):
            raise MacroMaintenanceTransactionError("macro_transaction_input_binding_invalid")
        path = Path(str(raw.get("path") or ""))
        digest = _required_sha(raw.get("sha256"), "macro_transaction_input_binding_sha_invalid")
        _regular_bytes(
            path,
            "macro_transaction_input_binding_unsafe",
            max_bytes=max(_MAX_JSON_BYTES, path.stat().st_size),
            expected_sha256=digest,
        )


def _journal_directory(journal_root: str | Path, journal_run_id: str, *, create: bool) -> Path:
    root = _private_directory(journal_root, "macro_transaction_journal_root_not_private")
    run_id = _safe_id(journal_run_id, "macro_transaction_journal_run_id_invalid")
    run = root / run_id
    if create:
        try:
            os.mkdir(run, 0o700)
            _fsync_directory(root)
        except FileExistsError as exc:
            raise MacroMaintenanceTransactionError("macro_transaction_journal_no_clobber") from exc
    else:
        run = _private_directory(run, "macro_transaction_journal_missing")
    return run


def _journal_records(run: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(run.glob("*.json")), start=1):
        raw = _regular_bytes(path, "macro_transaction_journal_record_unsafe")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise MacroMaintenanceTransactionError(
                "macro_transaction_journal_record_invalid"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != JOURNAL_SCHEMA
            or payload.get("sequence") != index
        ):
            raise MacroMaintenanceTransactionError("macro_transaction_journal_record_invalid")
        records.append(dict(payload))
    if not records:
        raise MacroMaintenanceTransactionError("macro_transaction_journal_empty")
    return records


def _append_journal(
    run: Path,
    phase: str,
    *,
    prepared_path: Path,
    prepared_sha256: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise MacroMaintenanceTransactionError("macro_transaction_journal_phase_invalid")
    sequence = len(list(run.glob("*.json"))) + 1
    payload = {
        "schema_version": JOURNAL_SCHEMA,
        "sequence": sequence,
        "phase": phase,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "prepared_path": str(prepared_path),
        "prepared_sha256": prepared_sha256,
        "details": dict(details or {}),
    }
    path = run / f"{sequence:04d}-{phase.lower()}.json"
    _write_exclusive(path, _json_bytes(payload), "macro_transaction_journal_append_failed")
    return payload


def _inject(injector: Callable[[str], None] | None, phase: str) -> None:
    if injector is not None:
        injector(phase)


def _postcheck(release: Mapping[str, Any], observations: Mapping[str, Any]) -> None:
    load_release_calendar(
        canonical_root=release["canonical_root"],
        expected_pointer_sha256=release["new_pointer_sha256"],
    )
    rows, pointer = load_observations(observations["canonical_root"])
    if (
        not rows
        or pointer.get("pointer_sha256") != observations["new_pointer_sha256"]
        or pointer.get("generation_id") != observations["generation_id"]
    ):
        raise MacroMaintenanceTransactionError(
            "macro_transaction_observations_postcheck_failed",
            status="PROMOTION_UNCERTAIN",
        )


def _state(release: Mapping[str, Any], observations: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _pointer_bytes(release["canonical_root"])[1],
        _pointer_bytes(observations["canonical_root"])[1],
    )


def _classify_state(release: Mapping[str, Any], observations: Mapping[str, Any]) -> str:
    release_sha, observations_sha = _state(release, observations)
    old_release = release["old_pointer_sha256"]
    new_release = release["new_pointer_sha256"]
    old_observations = observations["old_pointer_sha256"]
    new_observations = observations["new_pointer_sha256"]
    if release_sha == new_release and observations_sha == new_observations:
        return "CAN_FINALIZE"
    if release_sha == old_release and observations_sha == old_observations:
        return "CAN_EXECUTE_FORWARD"
    if release_sha == new_release and observations_sha == old_observations:
        return "CAN_EXECUTE_FORWARD"
    return "PROMOTION_UNCERTAIN"


def _execute_forward(
    *,
    prepared_path: Path,
    prepared_payload: Mapping[str, Any],
    prepared_sha256: str,
    run: Path,
    authorities: Mapping[str, Mapping[str, Any]],
    release: Mapping[str, Any],
    observations: Mapping[str, Any],
    injector: Callable[[str], None] | None,
) -> dict[str, Any]:
    with _transaction_locks(
        authorities,
        release["canonical_root"],
        observations["canonical_root"],
    ):
        _revalidate_authorities(authorities, checkpoint="before_install")
        classification = _classify_state(release, observations)
        if classification == "PROMOTION_UNCERTAIN":
            raise MacroMaintenanceTransactionError(
                "macro_transaction_third_party_pointer_drift",
                status="PROMOTION_UNCERTAIN",
            )
        _revalidate_inputs(prepared_payload)
        if classification in {"CAN_EXECUTE_FORWARD", "CAN_FINALIZE"}:
            phases = [record["phase"] for record in _journal_records(run)]
            if "BOTH_GENERATIONS_PREPARED" not in phases:
                _append_journal(
                    run,
                    "BOTH_GENERATIONS_PREPARED",
                    prepared_path=prepared_path,
                    prepared_sha256=prepared_sha256,
                    details={"recovered_from_sealed_preparation": True},
                )
                _inject(injector, "BOTH_GENERATIONS_PREPARED")
            _copy_generation(
                release["generation_source"],
                release["canonical_root"] / _GENERATIONS / release["generation_id"],
                release["generation_tree_sha256"],
            )
            for dependency in observations["dependency_generations"]:
                _copy_generation(
                    dependency["generation_source"],
                    observations["canonical_root"] / _GENERATIONS / dependency["generation_id"],
                    dependency["generation_tree_sha256"],
                )
            _copy_generation(
                observations["generation_source"],
                observations["canonical_root"] / _GENERATIONS / observations["generation_id"],
                observations["generation_tree_sha256"],
            )
            _revalidate_authorities(authorities, checkpoint="after_install")
            phases = [record["phase"] for record in _journal_records(run)]
            if "BOTH_GENERATIONS_INSTALLED" not in phases:
                _append_journal(
                    run,
                    "BOTH_GENERATIONS_INSTALLED",
                    prepared_path=prepared_path,
                    prepared_sha256=prepared_sha256,
                )
                _inject(injector, "BOTH_GENERATIONS_INSTALLED")

            release_sha, observations_sha = _state(release, observations)
            _revalidate_authorities(
                authorities,
                checkpoint="before_release_pointer_switch",
            )
            if release_sha == release["old_pointer_sha256"]:
                release_sha = _atomic_pointer(
                    release["canonical_root"], release["new_pointer_artifact"]
                )
                if release_sha != release["new_pointer_sha256"]:
                    raise MacroMaintenanceTransactionError(
                        "macro_transaction_release_pointer_readback_mismatch",
                        status="PROMOTION_UNCERTAIN",
                    )
                if "RELEASE_POINTER_COMMITTED" not in phases:
                    _append_journal(
                        run,
                        "RELEASE_POINTER_COMMITTED",
                        prepared_path=prepared_path,
                        prepared_sha256=prepared_sha256,
                        details={"pointer_sha256": release_sha},
                    )
                    _inject(injector, "RELEASE_POINTER_COMMITTED")
            elif release_sha == release["new_pointer_sha256"]:
                if "RELEASE_POINTER_COMMITTED" not in phases:
                    _append_journal(
                        run,
                        "RELEASE_POINTER_COMMITTED",
                        prepared_path=prepared_path,
                        prepared_sha256=prepared_sha256,
                        details={
                            "pointer_sha256": release_sha,
                            "recovered_after_pointer_write": True,
                        },
                    )
                    _inject(injector, "RELEASE_POINTER_COMMITTED")
            else:
                raise MacroMaintenanceTransactionError(
                    "macro_transaction_release_pointer_drift",
                    status="PROMOTION_UNCERTAIN",
                )

            _revalidate_authorities(
                authorities,
                checkpoint="after_release_pointer_switch",
            )
            current_observations_sha = _pointer_bytes(observations["canonical_root"])[1]
            if current_observations_sha == observations["old_pointer_sha256"]:
                observations_sha = _atomic_pointer(
                    observations["canonical_root"],
                    observations["new_pointer_artifact"],
                )
                if observations_sha != observations["new_pointer_sha256"]:
                    raise MacroMaintenanceTransactionError(
                        "macro_transaction_observations_pointer_readback_mismatch",
                        status="PROMOTION_UNCERTAIN",
                    )
                if "OBSERVATIONS_POINTER_COMMITTED" not in phases:
                    _append_journal(
                        run,
                        "OBSERVATIONS_POINTER_COMMITTED",
                        prepared_path=prepared_path,
                        prepared_sha256=prepared_sha256,
                        details={"pointer_sha256": observations_sha},
                    )
                    _inject(injector, "OBSERVATIONS_POINTER_COMMITTED")
            elif current_observations_sha == observations["new_pointer_sha256"]:
                if "OBSERVATIONS_POINTER_COMMITTED" not in phases:
                    _append_journal(
                        run,
                        "OBSERVATIONS_POINTER_COMMITTED",
                        prepared_path=prepared_path,
                        prepared_sha256=prepared_sha256,
                        details={
                            "pointer_sha256": current_observations_sha,
                            "recovered_after_pointer_write": True,
                        },
                    )
                    _inject(injector, "OBSERVATIONS_POINTER_COMMITTED")
            else:
                raise MacroMaintenanceTransactionError(
                    "macro_transaction_observations_pointer_drift",
                    status="PROMOTION_UNCERTAIN",
                )

        _revalidate_authorities(
            authorities,
            checkpoint="after_observations_pointer_switch",
        )
        _revalidate_authorities(authorities, checkpoint="before_postcheck")
        _postcheck(release, observations)
        _revalidate_authorities(authorities, checkpoint="after_postcheck")
        phases = [record["phase"] for record in _journal_records(run)]
        if "POSTCHECK_PASSED" not in phases:
            _append_journal(
                run,
                "POSTCHECK_PASSED",
                prepared_path=prepared_path,
                prepared_sha256=prepared_sha256,
            )
            _inject(injector, "POSTCHECK_PASSED")
        phases = [record["phase"] for record in _journal_records(run)]
        if "TERMINAL" not in phases:
            _append_journal(
                run,
                "TERMINAL",
                prepared_path=prepared_path,
                prepared_sha256=prepared_sha256,
                details={"status": "SUCCESS"},
            )
            _inject(injector, "TERMINAL")
    return {
        "schema_version": TRANSACTION_SCHEMA,
        "status": "SUCCESS",
        "terminal": True,
        "target_date": prepared_payload["target_date"],
        "release_pointer_sha256": release["new_pointer_sha256"],
        "observations_pointer_sha256": observations["new_pointer_sha256"],
        "market_pointer_sha256": authorities["market"]["pointer_sha256"],
        "pit_pointer_sha256": authorities["pit"]["pointer_sha256"],
        "journal_path": str(run),
    }


def commit_prepared_macro_transaction(
    *,
    prepared_path: str | Path,
    expected_prepared_sha256: str,
    journal_root: str | Path,
    journal_run_id: str,
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Install both candidates and CAS both pointers with a durable journal."""

    path, payload, raw = _load_prepared(prepared_path, expected_prepared_sha256)
    if payload.get("authority_mode") != "canonical":
        raise MacroMaintenanceTransactionError(
            "macro_transaction_candidate_authority_not_executable"
        )
    authorities = _prepared_authorities(path, payload)
    _assert_authority_arguments(
        authorities,
        market_pointer_path=market_pointer_path,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
        pit_pointer_path=pit_pointer_path,
        expected_pit_pointer_sha256=expected_pit_pointer_sha256,
    )
    release = _prepared_component(path, payload, "release")
    observations = _prepared_component(path, payload, "observations")
    run = _journal_directory(journal_root, journal_run_id, create=True)
    prepared_sha = _sha(raw)
    _append_journal(
        run,
        "INTENT",
        prepared_path=path,
        prepared_sha256=prepared_sha,
        details={
            "release_old": release["old_pointer_sha256"],
            "release_new": release["new_pointer_sha256"],
            "observations_old": observations["old_pointer_sha256"],
            "observations_new": observations["new_pointer_sha256"],
            "market_authority": {
                "pointer_path": str(authorities["market"]["pointer_path"]),
                "pointer_sha256": authorities["market"]["pointer_sha256"],
                "pointer_bytes_b64": base64.b64encode(authorities["market"]["pointer_raw"]).decode(
                    "ascii"
                ),
            },
            "pit_authority": {
                "pointer_path": str(authorities["pit"]["pointer_path"]),
                "pointer_sha256": authorities["pit"]["pointer_sha256"],
                "pointer_bytes_b64": base64.b64encode(authorities["pit"]["pointer_raw"]).decode(
                    "ascii"
                ),
            },
        },
    )
    _inject(failure_injector, "INTENT")
    _revalidate_inputs(payload)
    _append_journal(
        run,
        "BOTH_GENERATIONS_PREPARED",
        prepared_path=path,
        prepared_sha256=prepared_sha,
    )
    _inject(failure_injector, "BOTH_GENERATIONS_PREPARED")
    return _execute_forward(
        prepared_path=path,
        prepared_payload=payload,
        prepared_sha256=prepared_sha,
        run=run,
        authorities=authorities,
        release=release,
        observations=observations,
        injector=failure_injector,
    )


def recover_macro_transaction(
    *,
    journal_root: str | Path,
    journal_run_id: str,
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
    execute_forward: bool = False,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Classify recovery read-only, or deterministically continue forward."""

    run = _journal_directory(journal_root, journal_run_id, create=False)
    records = _journal_records(run)
    first = records[0]
    prepared_path = Path(str(first["prepared_path"]))
    prepared_sha = _required_sha(first["prepared_sha256"], "macro_transaction_prepared_sha_invalid")
    path, payload, raw = _load_prepared(prepared_path, prepared_sha)
    if payload.get("authority_mode") != "canonical":
        raise MacroMaintenanceTransactionError(
            "macro_transaction_candidate_authority_not_executable"
        )
    authorities = _prepared_authorities(path, payload)
    _assert_authority_arguments(
        authorities,
        market_pointer_path=market_pointer_path,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
        pit_pointer_path=pit_pointer_path,
        expected_pit_pointer_sha256=expected_pit_pointer_sha256,
    )
    release = _prepared_component(path, payload, "release")
    observations = _prepared_component(path, payload, "observations")
    terminal = next(
        (record for record in reversed(records) if record["phase"] == "TERMINAL"),
        None,
    )
    if terminal is not None:
        return {
            "schema_version": TRANSACTION_SCHEMA,
            "status": str(terminal.get("details", {}).get("status") or "TERMINAL"),
            "classification": "TERMINAL",
            "terminal": True,
            "journal_path": str(run),
            "market_pointer_sha256": authorities["market"]["pointer_sha256"],
            "pit_pointer_sha256": authorities["pit"]["pointer_sha256"],
        }
    authority_blocker = ""
    try:
        with _transaction_locks(
            authorities,
            release["canonical_root"],
            observations["canonical_root"],
        ):
            _revalidate_authorities(
                authorities,
                checkpoint="recovery_classification",
            )
            classification = _classify_state(release, observations)
    except MacroMaintenanceTransactionError as exc:
        if exc.status != "PROMOTION_UNCERTAIN":
            raise
        classification = "PROMOTION_UNCERTAIN"
        authority_blocker = str(exc)
    result = {
        "schema_version": TRANSACTION_SCHEMA,
        "status": classification,
        "classification": classification,
        "terminal": False,
        "journal_path": str(run),
        "execute_forward_eligible": classification
        in {
            "CAN_EXECUTE_FORWARD",
            "CAN_FINALIZE",
        },
        "blockers": [authority_blocker] if authority_blocker else [],
        "market_pointer_sha256": authorities["market"]["pointer_sha256"],
        "pit_pointer_sha256": authorities["pit"]["pointer_sha256"],
    }
    if not execute_forward:
        return result
    if not result["execute_forward_eligible"]:
        raise MacroMaintenanceTransactionError(
            "macro_transaction_recovery_not_deterministic",
            status="PROMOTION_UNCERTAIN",
        )
    return _execute_forward(
        prepared_path=path,
        prepared_payload=payload,
        prepared_sha256=_sha(raw),
        run=run,
        authorities=authorities,
        release=release,
        observations=observations,
        injector=failure_injector,
    )


def rollback_macro_transaction(
    *,
    journal_root: str | Path,
    journal_run_id: str,
    old_release_pointer_sha256: str,
    new_release_pointer_sha256: str,
    old_observations_pointer_sha256: str,
    new_observations_pointer_sha256: str,
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
) -> dict[str, Any]:
    """Explicit operator rollback guarded by all four old/new SHA values."""

    run = _journal_directory(journal_root, journal_run_id, create=False)
    records = _journal_records(run)
    first = records[0]
    path, payload, raw = _load_prepared(first["prepared_path"], first["prepared_sha256"])
    if payload.get("authority_mode") != "canonical":
        raise MacroMaintenanceTransactionError(
            "macro_transaction_candidate_authority_not_executable"
        )
    authorities = _prepared_authorities(path, payload)
    _assert_authority_arguments(
        authorities,
        market_pointer_path=market_pointer_path,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
        pit_pointer_path=pit_pointer_path,
        expected_pit_pointer_sha256=expected_pit_pointer_sha256,
    )
    release = _prepared_component(path, payload, "release")
    observations = _prepared_component(path, payload, "observations")
    supplied = (
        _required_sha(
            old_release_pointer_sha256,
            "macro_transaction_rollback_sha_invalid",
            empty=True,
        ),
        _required_sha(
            new_release_pointer_sha256,
            "macro_transaction_rollback_sha_invalid",
        ),
        _required_sha(
            old_observations_pointer_sha256,
            "macro_transaction_rollback_sha_invalid",
            empty=True,
        ),
        _required_sha(
            new_observations_pointer_sha256,
            "macro_transaction_rollback_sha_invalid",
        ),
    )
    expected = (
        release["old_pointer_sha256"],
        release["new_pointer_sha256"],
        observations["old_pointer_sha256"],
        observations["new_pointer_sha256"],
    )
    if supplied != expected:
        raise MacroMaintenanceTransactionError("macro_transaction_rollback_identity_mismatch")
    if release["old_pointer_artifact"] is None or observations["old_pointer_artifact"] is None:
        raise MacroMaintenanceTransactionError(
            "macro_transaction_rollback_empty_parent_unsupported"
        )
    with _transaction_locks(
        authorities,
        release["canonical_root"],
        observations["canonical_root"],
    ):
        _revalidate_authorities(authorities, checkpoint="before_rollback")
        release_sha, observations_sha = _state(release, observations)
        if release_sha not in {expected[0], expected[1]} or observations_sha not in {
            expected[2],
            expected[3],
        }:
            raise MacroMaintenanceTransactionError(
                "macro_transaction_rollback_third_party_drift",
                status="PROMOTION_UNCERTAIN",
            )
        # Reverse commit order.
        if observations_sha == expected[3]:
            _atomic_pointer(observations["canonical_root"], observations["old_pointer_artifact"])
        if release_sha == expected[1]:
            _atomic_pointer(release["canonical_root"], release["old_pointer_artifact"])
        if _state(release, observations) != (expected[0], expected[2]):
            raise MacroMaintenanceTransactionError(
                "macro_transaction_rollback_readback_mismatch",
                status="PROMOTION_UNCERTAIN",
            )
        _revalidate_authorities(authorities, checkpoint="after_rollback")
        _append_journal(
            run,
            "TERMINAL",
            prepared_path=path,
            prepared_sha256=_sha(raw),
            details={"status": "ROLLED_BACK"},
        )
    return {
        "schema_version": TRANSACTION_SCHEMA,
        "status": "ROLLED_BACK",
        "terminal": True,
        "journal_path": str(run),
        "market_pointer_sha256": authorities["market"]["pointer_sha256"],
        "pit_pointer_sha256": authorities["pit"]["pointer_sha256"],
    }


__all__ = [
    "MacroMaintenanceTransactionError",
    "PHASES",
    "commit_prepared_macro_transaction",
    "generation_tree_sha256",
    "recover_macro_transaction",
    "rollback_macro_transaction",
    "seal_prepared_macro_transaction",
]
