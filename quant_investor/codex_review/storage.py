"""Bounded strict-JSON and private atomic storage primitives."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

REQUEST_MAX_BYTES = 128 * 1024 * 1024
RESPONSE_MAX_BYTES = 64 * 1024 * 1024
CONTROL_MAX_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 32
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


class ProtocolError(ValueError):
    """Base error for fail-closed local protocol operations."""


class StrictJSONError(ProtocolError):
    pass


class StateConflictError(ProtocolError):
    pass


class DifferentBytesError(ProtocolError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
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
        raise StrictJSONError(f"value is not canonical finite JSON: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProtocolError(f"bound path is not a regular non-symlink file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _validate_value(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise StrictJSONError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJSONError("non-finite JSON number is forbidden")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJSONError("JSON object keys must be strings")
            _validate_value(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_value(item, depth + 1)


def parse_strict_json_bytes(payload: bytes, *, max_bytes: int) -> Any:
    if len(payload) > max_bytes:
        raise StrictJSONError(f"JSON artifact exceeds {max_bytes} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJSONError("JSON artifact must be UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"invalid JSON: {exc}") from exc
    _validate_value(value)
    return value


def read_strict_json(path: str | Path, *, max_bytes: int) -> tuple[bytes, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProtocolError(f"JSON path is not a regular non-symlink file: {target}")
    size = target.stat().st_size
    if size > max_bytes:
        raise StrictJSONError(f"JSON artifact exceeds {max_bytes} bytes")
    payload = target.read_bytes()
    return payload, parse_strict_json_bytes(payload, max_bytes=max_bytes)


def ensure_private_root(root: str | Path) -> Path:
    target = Path(root).absolute()
    if target.is_symlink():
        raise ProtocolError("Codex review root cannot be a symlink")
    target.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    if target.stat().st_mode & 0o777 != PRIVATE_DIR_MODE:
        raise ProtocolError("Codex review root permissions must be 0700")
    return target.resolve(strict=True)


def ensure_private_dir(path: Path, *, root: Path) -> Path:
    candidate = path.absolute()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("private path escapes Codex review root") from exc
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ProtocolError("symlinks are forbidden in Codex review storage")
        current.mkdir(exist_ok=True, mode=PRIVATE_DIR_MODE)
        if current.stat().st_mode & 0o777 != PRIVATE_DIR_MODE:
            raise ProtocolError("Codex review directories must be mode 0700")
    return candidate.resolve(strict=True)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes, *, root: Path) -> str:
    parent = ensure_private_dir(path.parent, root=root)
    if path.is_symlink():
        raise ProtocolError("symlink artifact targets are forbidden")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    temp = Path(temp_name)
    digest = sha256_bytes(payload)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, PRIVATE_FILE_MODE)
        _fsync_dir(parent)
        if path.stat().st_mode & 0o777 != PRIVATE_FILE_MODE:
            raise ProtocolError("private artifact mode readback is not 0600")
        if sha256_bytes(path.read_bytes()) != digest:
            raise ProtocolError("private artifact hash readback mismatch")
        return digest
    finally:
        if temp.exists():
            temp.unlink()


def write_exact_once(path: Path, payload: bytes, *, root: Path) -> tuple[str, bool]:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"artifact target is not a regular file: {path}")
        existing = path.read_bytes()
        if existing != payload:
            raise DifferentBytesError(
                f"different bytes already exist for {path.name}; use a new run_id"
            )
        if path.stat().st_mode & 0o777 != PRIVATE_FILE_MODE:
            raise ProtocolError("existing private artifact mode is not 0600")
        return sha256_bytes(existing), False
    return atomic_write_bytes(path, payload, root=root), True


def read_private_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"private artifact is missing or unsafe: {path}")
    if path.stat().st_mode & 0o777 != PRIVATE_FILE_MODE:
        raise ProtocolError("private artifact permissions are not 0600")
    if path.stat().st_size > max_bytes:
        raise StrictJSONError(f"private artifact exceeds {max_bytes} bytes")
    payload = path.read_bytes()
    parse_strict_json_bytes(payload, max_bytes=max_bytes)
    return payload


@contextmanager
def run_lock(root: str | Path, run_id: str) -> Iterator[tuple[Path, Path]]:
    root_path = ensure_private_root(root)
    if not run_id or any(token in run_id for token in ("/", "\\", "..")):
        raise ProtocolError("run_id is not a safe path component")
    run_dir = ensure_private_dir(root_path / run_id, root=root_path)
    lock_path = run_dir / ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, PRIVATE_FILE_MODE)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield root_path, run_dir
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def assert_cas(state_path: Path, expected_sha256: str) -> str:
    expected = str(expected_sha256).strip().lower()
    if not state_path.exists():
        if expected != "empty":
            raise StateConflictError("state CAS mismatch: expected EMPTY")
        return "EMPTY"
    current = sha256_bytes(read_private_bytes(state_path, max_bytes=CONTROL_MAX_BYTES))
    if expected != current:
        raise StateConflictError(
            f"state CAS mismatch: expected {expected or '<missing>'}, current {current}"
        )
    return current
