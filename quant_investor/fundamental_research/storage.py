"""Private, bounded and atomic JSON storage helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_JSON_DEPTH = 12
PRIVATE_MODE = 0o600
_T = TypeVar("_T", bound=BaseModel)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def model_sha256(model: BaseModel) -> str:
    return sha256_bytes(canonical_json_bytes(model.model_dump(mode="json")))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_value(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is forbidden")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_value(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_value(item, depth + 1)


def _assert_contained(
    root: Path,
    path: Path,
    *,
    allow_missing_leaf: bool,
    create_parents: bool = True,
) -> tuple[Path, Path]:
    root = root.absolute()
    path = path.absolute()
    if root.is_symlink():
        raise ValueError("configured private artifact root cannot be a symlink")
    if create_parents:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    elif not root.exists():
        raise FileNotFoundError(root)
    if root.stat().st_mode & 0o077:
        raise ValueError("configured private artifact root permissions must be 0700")
    resolved_root = root.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes configured root") from exc
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise ValueError("symlinks are forbidden in private artifact paths")
        elif current != path or not allow_missing_leaf:
            if current != path and create_parents:
                current.mkdir(mode=0o700)
            else:
                raise FileNotFoundError(path)
    parent_resolved = path.parent.resolve(strict=True)
    try:
        parent_resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("resolved path escapes configured root") from exc
    return resolved_root, path


def load_json_model(root: str | Path, path: str | Path, model_type: type[_T]) -> _T:
    _, target = _assert_contained(
        Path(root), Path(path), allow_missing_leaf=False, create_parents=False
    )
    if target.stat().st_mode & 0o777 != PRIVATE_MODE:
        raise ValueError("private JSON artifact permissions are not 0600")
    if target.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes")
    payload = target.read_bytes()
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes")
    value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    _validate_value(value)
    return model_type.model_validate(value)


def atomic_write_json_model(root: str | Path, path: str | Path, model: BaseModel) -> str:
    _, target = _assert_contained(Path(root), Path(path), allow_missing_leaf=True)
    payload = canonical_json_bytes(model.model_dump(mode="json"))
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes")
    digest = sha256_bytes(payload)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp = Path(temp_name)
    try:
        os.fchmod(fd, PRIVATE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        os.chmod(target, PRIVATE_MODE)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.stat().st_mode & 0o777 != PRIVATE_MODE:
            raise RuntimeError("private artifact permissions are not 0600")
        if sha256_bytes(target.read_bytes()) != digest:
            raise RuntimeError("atomic JSON readback hash mismatch")
        return digest
    finally:
        if temp.exists():
            temp.unlink()
