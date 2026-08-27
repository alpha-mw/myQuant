"""Non-secret evidence for the shared CN maintenance credential launcher."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Final

SERVICE: Final = "com.maxwell.myquant.tushare"
ACCOUNT: Final = "maxwell"
SOURCE: Final = "MACOS_KEYCHAIN"
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CredentialPreflightError(RuntimeError):
    """One controlled credential-preflight error."""


def _canonical(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_NOT_CANONICAL") from exc


def _directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute():
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_ROOT_NOT_ABSOLUTE")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o077
    ):
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_ROOT_UNSAFE")
    return path


def _write_once(path: Path, raw: bytes) -> tuple[str, bool]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        observed_metadata = os.lstat(path)
        if (
            not stat.S_ISREG(observed_metadata.st_mode)
            or stat.S_ISLNK(observed_metadata.st_mode)
            or observed_metadata.st_uid != os.geteuid()
            or observed_metadata.st_nlink != 1
            or stat.S_IMODE(observed_metadata.st_mode) != 0o600
        ):
            raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_CONFLICT_UNSAFE")
        observed = path.read_bytes()
        if observed != raw:
            raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_CONFLICT")
        return hashlib.sha256(observed).hexdigest(), False
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest(), True


def write_credential_preflight(
    *,
    run_root: str | os.PathLike[str],
    attempt_slot: str,
    receipt_id: str,
    access_state: str,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Write one immutable non-secret credential access receipt."""

    if attempt_slot not in {"1620", "1720", "1820", "2020"}:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_SLOT_INVALID")
    if _ID_RE.fullmatch(receipt_id) is None:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_ID_INVALID")
    if access_state not in {"READY", "BLOCKED"}:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_STATE_INVALID")
    stamp = checked_at or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        parsed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_TIME_INVALID") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != stamp:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_TIME_INVALID")
    root = _directory(Path(run_root), create=True)
    receipts = _directory(root / "credential_preflight", create=True)
    payload = {
        "schema_version": "cn-maintenance-credential-preflight.v1",
        "receipt_id": receipt_id,
        "attempt_slot": attempt_slot,
        "checked_at": stamp,
        "credential_source": SOURCE,
        "service": SERVICE,
        "account": ACCOUNT,
        "access_state": access_state,
        "token_material_recorded": False,
        "token_hash_recorded": False,
    }
    path = receipts / f"{receipt_id}.json"
    digest, created = _write_once(path, _canonical(payload))
    return {
        "status": "RECORDED" if created else "NO_ACTION",
        "credential_source": SOURCE,
        "credential_access": access_state,
        "receipt_path": str(path),
        "receipt_sha256": digest,
        "token_material_recorded": False,
    }


def validate_credential_preflight(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Stable-read one exact READY preflight without reading any token material."""

    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) & 0o077
    ):
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_RECEIPT_UNSAFE")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second or hashlib.sha256(first).hexdigest() != expected_sha256:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_RECEIPT_SHA_MISMATCH")
    try:
        value = json.loads(first)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_RECEIPT_INVALID") from exc
    if _canonical(value) != first:
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_RECEIPT_NOT_CANONICAL")
    if (
        value.get("schema_version") != "cn-maintenance-credential-preflight.v1"
        or value.get("credential_source") != SOURCE
        or value.get("service") != SERVICE
        or value.get("account") != ACCOUNT
        or value.get("access_state") != "READY"
        or value.get("token_material_recorded") is not False
        or value.get("token_hash_recorded") is not False
    ):
        raise CredentialPreflightError("CREDENTIAL_PREFLIGHT_RECEIPT_NOT_READY")
    return value


__all__ = [
    "ACCOUNT",
    "CredentialPreflightError",
    "SERVICE",
    "SOURCE",
    "validate_credential_preflight",
    "write_credential_preflight",
]
