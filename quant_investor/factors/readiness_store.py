"""Version-neutral, nonauthorizing storage for Factor Governance v4 readiness."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from quant_investor.factors.governance_protocol_v4 import READINESS_SCHEMA_VERSION

FACTOR_READINESS_PATH = Path("results/factor_governance/readiness.json")
MAX_READINESS_BYTES = 4 * 1024 * 1024
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class FactorReadinessStoreError(RuntimeError):
    """The neutral readiness artifact is unsafe, malformed, or authorizing."""


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FactorReadinessStoreError(f"readiness artifact is unavailable: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FactorReadinessStoreError("readiness artifact must be a regular file")
        if before.st_nlink != 1:
            raise FactorReadinessStoreError("readiness artifact must not have hard links")
        if before.st_size > MAX_READINESS_BYTES:
            raise FactorReadinessStoreError("readiness artifact exceeds the size limit")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise FactorReadinessStoreError("readiness artifact changed while being read")
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise FactorReadinessStoreError("readiness artifact disappeared after read") from exc
    if _fingerprint(after) != _fingerprint(path_after):
        raise FactorReadinessStoreError("readiness artifact path identity changed")
    return b"".join(chunks), after


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FactorReadinessStoreError(f"duplicate readiness key: {key}")
        result[key] = value
    return result


def validate_factor_readiness(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if payload.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise FactorReadinessStoreError("unexpected Factor readiness schema")
    if payload.get("status") != "no_new_risk":
        raise FactorReadinessStoreError("retirement migration accepts no_new_risk only")
    required_false = (
        "factor_governance_ready",
        "new_risk_eligible",
        "new_risk_authorized",
        "production_apply_enabled",
    )
    if any(payload.get(field) is not False for field in required_false):
        raise FactorReadinessStoreError("Factor readiness contains activation authority")
    receipt = payload.get("activation_receipt")
    if receipt is not None:
        if not isinstance(receipt, Mapping):
            raise FactorReadinessStoreError("activation_receipt must be null or an object")
        receipt_payload = dict(receipt)
        if receipt_payload.get("valid") is not False:
            raise FactorReadinessStoreError("activation receipt must not be valid")
        if "receipt" in receipt_payload and receipt_payload["receipt"] is not None:
            raise FactorReadinessStoreError("activation receipt payload must be null")
        if "receipt_sha256" in receipt_payload:
            raise FactorReadinessStoreError("activation receipt SHA must be absent")
    return payload


def _decode_factor_readiness(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FactorReadinessStoreError(f"non-finite readiness value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorReadinessStoreError(f"invalid readiness JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FactorReadinessStoreError("readiness artifact must contain an object")
    return validate_factor_readiness(value)


def read_factor_readiness(path: str | Path = FACTOR_READINESS_PATH) -> dict[str, Any]:
    target = Path(path)
    payload, metadata = _stable_read(target)
    if stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE:
        raise FactorReadinessStoreError("readiness artifact mode must be 0600")
    return _decode_factor_readiness(payload)


def install_factor_readiness_exact_once(
    *,
    source_path: str | Path,
    target_path: str | Path = FACTOR_READINESS_PATH,
) -> str:
    """Copy one verified readiness artifact without overwriting an existing target."""

    source = Path(source_path)
    target = Path(target_path)
    source_bytes, source_stat = _stable_read(source)
    if stat.S_IMODE(source_stat.st_mode) != PRIVATE_FILE_MODE:
        raise FactorReadinessStoreError("source readiness mode must be 0600")
    _decode_factor_readiness(source_bytes)

    target_parent = target.parent
    if target_parent.is_symlink():
        raise FactorReadinessStoreError("readiness parent must not be a symlink")
    target_parent.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    os.chmod(target_parent, PRIVATE_DIR_MODE)

    if target.exists() or target.is_symlink():
        target_bytes, target_stat = _stable_read(target)
        if stat.S_IMODE(target_stat.st_mode) != PRIVATE_FILE_MODE:
            raise FactorReadinessStoreError("existing readiness target mode must be 0600")
        if target_bytes != source_bytes:
            raise FactorReadinessStoreError("existing readiness target differs from source")
        read_factor_readiness(target)
        return "already_installed_identical"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".readiness.",
        suffix=".tmp",
        dir=str(target_parent),
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
            linked = True
        except FileExistsError:
            target_bytes, target_stat = _stable_read(target)
            if (
                stat.S_IMODE(target_stat.st_mode) != PRIVATE_FILE_MODE
                or target_bytes != source_bytes
            ):
                raise FactorReadinessStoreError(
                    "readiness target appeared with different bytes"
                )
        directory_descriptor = os.open(target_parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
            if linked:
                directory_descriptor = os.open(target_parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)

    target_bytes, target_stat = _stable_read(target)
    if target_bytes != source_bytes:
        raise FactorReadinessStoreError("readiness target readback differs from source")
    if stat.S_IMODE(target_stat.st_mode) != PRIVATE_FILE_MODE:
        raise FactorReadinessStoreError("readiness target mode readback is not 0600")
    read_factor_readiness(target)
    return "installed"


__all__ = [
    "FACTOR_READINESS_PATH",
    "FactorReadinessStoreError",
    "install_factor_readiness_exact_once",
    "read_factor_readiness",
    "validate_factor_readiness",
]
