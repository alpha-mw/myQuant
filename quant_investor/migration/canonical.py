"""Canonical bytes and safe local-file primitives used by cutover tooling."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final

from ..contracts import (
    canonical_json_bytes as contract_canonical_json_bytes,
    parse_canonical_json_bytes,
)

from .errors import (
    FILE_TOO_LARGE,
    RECEIPT_CONFLICT,
    SYMLINK_REFUSED,
    UNPARSEABLE_JSON,
    UNSAFE_PATH,
    UNSTABLE_FILE,
    UnifiedCutoverError,
)

DEFAULT_MAX_FILE_BYTES: Final = 512 * 1024 * 1024
SHA256_RE: Final = __import__("re").compile(r"^[0-9a-f]{64}$")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return contract_canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(UNPARSEABLE_JSON, "value is not canonical JSON data") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UnifiedCutoverError(UNPARSEABLE_JSON, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(
    raw: bytes,
    *,
    label: str,
    require_canonical: bool = False,
) -> Any:
    if require_canonical:
        try:
            return parse_canonical_json_bytes(raw, label=label)
        except (TypeError, ValueError) as exc:
            raise UnifiedCutoverError(UNPARSEABLE_JSON, f"{label} is not canonical JSON") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except UnifiedCutoverError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedCutoverError(UNPARSEABLE_JSON, f"{label} is not valid UTF-8 JSON") from exc
    return value


def canonical_relative_path(value: Any, *, label: str = "path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is not a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is not a canonical relative path")
    return value


def workspace_path(
    workspace_root: str | os.PathLike[str],
    relative_path: str,
    *,
    label: str = "path",
) -> Path:
    relative = canonical_relative_path(relative_path, label=label)
    root = Path(workspace_root).resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by canonical path
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} escapes workspace root") from exc
    return candidate


def assert_real_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise UnifiedCutoverError(SYMLINK_REFUSED, f"{label} is a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is not a directory")
    return metadata


def assert_no_symlink_components(root: Path, relative_path: str, *, include_leaf: bool) -> None:
    current = root
    parts = PurePosixPath(canonical_relative_path(relative_path)).parts
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise UnifiedCutoverError(UNSAFE_PATH, f"cannot inspect {relative_path}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"symlink component in {relative_path}")


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_stable_regular_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        try:
            metadata = os.lstat(path)
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"{label} is a symlink") from exc
        raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UnifiedCutoverError(UNSAFE_PATH, f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise UnifiedCutoverError(FILE_TOO_LARGE, f"{label} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if remaining or len(raw) != after.st_size or _identity(before) != _identity(after):
            raise UnifiedCutoverError(UNSTABLE_FILE, f"{label} changed during read")
        return raw
    finally:
        os.close(descriptor)


def content_sha256(document: Mapping[str, Any], *, omit: tuple[str, ...] = ()) -> str:
    body = dict(document)
    body.pop("content_sha256", None)
    for key in omit:
        body.pop(key, None)
    return sha256_bytes(canonical_json_bytes(body))


def seal_document(
    document: Mapping[str, Any],
    *,
    identity_field: str | None = None,
    identity_prefix: str = "",
) -> dict[str, Any]:
    result = dict(document)
    result.pop("content_sha256", None)
    if identity_field is not None:
        result.pop(identity_field, None)
        identity = content_sha256(result)
        result[identity_field] = f"{identity_prefix}{identity}" if identity_prefix else identity
    result["content_sha256"] = content_sha256(result)
    return result


def validate_sealed_document(
    document: Mapping[str, Any],
    *,
    identity_field: str | None = None,
    identity_prefix: str = "",
    label: str,
) -> dict[str, Any]:
    result = dict(document)
    observed_content = result.get("content_sha256")
    if not isinstance(observed_content, str) or not SHA256_RE.fullmatch(observed_content):
        raise UnifiedCutoverError(UNPARSEABLE_JSON, f"{label} content_sha256 is invalid")
    if observed_content != content_sha256(result):
        raise UnifiedCutoverError(UNPARSEABLE_JSON, f"{label} content_sha256 mismatch")
    if identity_field is not None:
        observed_id = result.get(identity_field)
        body = dict(result)
        body.pop("content_sha256", None)
        body.pop(identity_field, None)
        expected_hash = content_sha256(body)
        expected_id = f"{identity_prefix}{expected_hash}" if identity_prefix else expected_hash
        if observed_id != expected_id:
            raise UnifiedCutoverError(UNPARSEABLE_JSON, f"{label} {identity_field} mismatch")
    return result


def write_idempotent_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> bool:
    """Create ``path`` once; return False only for an exact existing replay."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"output {path} is a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise UnifiedCutoverError(RECEIPT_CONFLICT, f"output {path} is not a file")
        observed = read_stable_regular_file(path, label=f"existing output {path}")
        if observed == raw:
            return False
        raise UnifiedCutoverError(RECEIPT_CONFLICT, f"output {path} already has different bytes")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError:
        return write_idempotent_bytes(path, raw, mode=mode)
    try:
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = read_stable_regular_file(path, label=f"written output {path}")
    if observed != raw:
        raise UnifiedCutoverError(UNSTABLE_FILE, f"output {path} readback mismatch")
    return True


__all__ = [
    "SHA256_RE",
    "assert_no_symlink_components",
    "assert_real_directory",
    "canonical_json_bytes",
    "canonical_relative_path",
    "content_sha256",
    "parse_json_bytes",
    "read_stable_regular_file",
    "seal_document",
    "sha256_bytes",
    "validate_sealed_document",
    "workspace_path",
    "write_idempotent_bytes",
]
