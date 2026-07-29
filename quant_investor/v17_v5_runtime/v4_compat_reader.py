"""Read-only, bounded compatibility reader for explicitly allowed V17 v4 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final, Mapping

from quant_investor.v17_v4_contract import (
    artifact_identity_field as v4_identity_field,
    load_canonical_artifact as load_v4_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.resources import (
    COMPATIBILITY_POLICY_PATH,
    load_compatibility_policy,
    read_packaged_asset,
    verify_predecessor,
)

_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY: Final = getattr(os, "O_DIRECTORY", 0)


class V4CompatibilityError(RuntimeError):
    """Raised when a V17 v4 predecessor input cannot be trusted."""

    exit_code = 2


@dataclass(frozen=True)
class V4ClosureNode:
    artifact_id: str
    byte_sha256: str
    relative_path: str
    semantic_sha256: str
    version: str


@dataclass(frozen=True)
class V4CompatibilityRead:
    closure: tuple[V4ClosureNode, ...]
    compatibility_policy_byte_sha256: str
    document: Mapping[str, Any]
    predecessor_git_commit: str


def _instant(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        raise V4CompatibilityError(f"{label} must be a UTC-second timestamp")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise V4CompatibilityError(f"{label} must be a UTC-second timestamp") from exc
    if parsed.strftime(_UTC_FORMAT) != value:
        raise V4CompatibilityError(f"{label} is not canonical")
    return parsed


def _file_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_workspace_root(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V4CompatibilityError("workspace_root must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise V4CompatibilityError("workspace_root is unavailable") from exc
    if resolved != path or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise V4CompatibilityError("workspace_root must be a real canonical directory")
    return path


def _case_exact_entry(directory_fd: int, name: str) -> None:
    try:
        entries = os.listdir(directory_fd)
    except OSError as exc:
        raise V4CompatibilityError("trusted directory cannot be enumerated") from exc
    matches = [entry for entry in entries if entry.casefold() == name.casefold()]
    if matches != [name]:
        raise V4CompatibilityError("path component is absent or casefold-ambiguous")


def _secure_read_relative(
    workspace_root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    try:
        normalized = require_relative_path(relative_path)
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    parts = PurePosixPath(normalized).parts
    root_fd = -1
    directory_fd = -1
    file_fd = -1
    try:
        root_fd = os.open(workspace_root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        directory_fd = root_fd
        for part in parts[:-1]:
            _case_exact_entry(directory_fd, part)
            before = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise V4CompatibilityError("path parent is not a real directory")
            child_fd = os.open(
                part,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=directory_fd,
            )
            after = os.fstat(child_fd)
            if _file_fingerprint(before) != _file_fingerprint(after):
                os.close(child_fd)
                raise V4CompatibilityError("path parent changed during open")
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = child_fd
        name = parts[-1]
        _case_exact_entry(directory_fd, name)
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes:
            raise V4CompatibilityError("artifact is not a bounded owner file")
        file_fd = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        if _file_fingerprint(before) != _file_fingerprint(opened):
            raise V4CompatibilityError("artifact changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(file_fd, min(1_048_576, remaining))
            if not chunk:
                raise V4CompatibilityError("artifact was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            raise V4CompatibilityError("artifact grew during read")
        closed = os.fstat(file_fd)
        if _file_fingerprint(opened) != _file_fingerprint(closed):
            raise V4CompatibilityError("artifact changed during read")
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            raise V4CompatibilityError("artifact read length mismatch")
        return raw
    except V4CompatibilityError:
        raise
    except OSError as exc:
        raise V4CompatibilityError("artifact secure read failed") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0 and directory_fd != root_fd:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _allowed_row(policy: Mapping[str, Any], version: str) -> Mapping[str, Any]:
    rows = [row for row in policy["allowed_artifacts"] if row["version"] == version]
    if len(rows) != 1:
        raise V4CompatibilityError(f"V17 v4 artifact version is not allowed: {version}")
    return rows[0]


def _path_allowed(relative_path: str, row: Mapping[str, Any]) -> None:
    if not any(
        relative_path == prefix or relative_path.startswith(f"{prefix}/")
        for prefix in row["allowed_path_prefixes"]
    ):
        raise V4CompatibilityError("V17 v4 artifact is outside its allowed namespace")


def _artifact_ref_paths(value: Any, *, pointer: str = "") -> tuple[str, ...]:
    result: list[str] = []
    if type(value) is dict:
        required = {
            "artifact_id",
            "artifact_version",
            "byte_sha256",
            "cutoff",
            "relative_path",
            "semantic_sha256",
            "strategy_id",
        }
        if required.issubset(value):
            result.append(pointer or "/")
        for key in sorted(value):
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.extend(_artifact_ref_paths(value[key], pointer=f"{pointer}/{escaped}"))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.extend(_artifact_ref_paths(item, pointer=f"{pointer}/{index}"))
    return tuple(result)


def read_v4_artifact(
    workspace_root: str | os.PathLike[str],
    *,
    relative_path: str,
    expected_byte_sha256: str,
    expected_strategy_id: str,
    decision_cutoff: str,
) -> V4CompatibilityRead:
    """Read and validate one allowlisted V17 v4 artifact without writing."""

    root = _canonical_workspace_root(workspace_root)
    try:
        expected_sha = require_sha256(expected_byte_sha256)
        strategy = require_identifier(expected_strategy_id, label="expected_strategy_id")
        path = require_relative_path(relative_path)
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    cutoff = _instant(decision_cutoff, label="decision_cutoff")
    predecessor = verify_predecessor()
    policy = load_compatibility_policy()
    limits = policy["closure_limits"]
    raw = _secure_read_relative(
        root,
        path,
        max_bytes=limits["max_artifact_bytes"],
    )
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != expected_sha:
        raise V4CompatibilityError("V17 v4 artifact byte SHA-256 mismatch")
    try:
        unvalidated = load_canonical_resource(raw, label=path)
    except Exception as exc:
        raise V4CompatibilityError("V17 v4 artifact is not canonical JSON") from exc
    if type(unvalidated) is not dict:
        raise V4CompatibilityError("V17 v4 artifact root must be an object")
    version = unvalidated.get("version")
    if type(version) is not str:
        raise V4CompatibilityError("V17 v4 artifact version is absent")
    row = _allowed_row(policy, version)
    _path_allowed(path, row)
    if len(raw) > limits["max_closure_bytes"] or limits["max_nodes"] < 1:
        raise V4CompatibilityError("V17 v4 closure resource limit exceeded")
    if row["transitive_edges"]:
        raise V4CompatibilityError("Phase-0 reader does not admit transitive artifacts")
    if _artifact_ref_paths(unvalidated):
        raise V4CompatibilityError("artifact contains an unallowlisted transitive reference")
    try:
        load_v4_artifact(raw, expected_version=version, label=path)
        identity_field = v4_identity_field(version)
        artifact_id = require_identifier(unvalidated[identity_field], label=identity_field)
        semantic_sha = require_sha256(unvalidated["semantic_sha256"])
    except Exception as exc:
        raise V4CompatibilityError("V17 v4 schema or semantic validation failed") from exc
    if unvalidated.get("protocol_version") != "myquant.v17.v4":
        raise V4CompatibilityError("V17 v4 protocol identity mismatch")
    if unvalidated.get("strategy_id") != strategy:
        raise V4CompatibilityError("V17 v4 strategy binding mismatch")
    if _instant(unvalidated.get("cutoff"), label="artifact.cutoff") > cutoff:
        raise V4CompatibilityError("V17 v4 artifact cutoff is in the future")
    available_at = unvalidated.get("available_at")
    if (
        available_at is not None
        and _instant(
            available_at,
            label="artifact.available_at",
        )
        > cutoff
    ):
        raise V4CompatibilityError("V17 v4 artifact availability is in the future")
    policy_raw = read_packaged_asset(COMPATIBILITY_POLICY_PATH)
    node = V4ClosureNode(
        artifact_id=artifact_id,
        byte_sha256=observed_sha,
        relative_path=path,
        semantic_sha256=semantic_sha,
        version=version,
    )
    return V4CompatibilityRead(
        closure=(node,),
        compatibility_policy_byte_sha256=hashlib.sha256(policy_raw).hexdigest(),
        document=dict(unvalidated),
        predecessor_git_commit=predecessor["source_git_commit"],
    )


__all__ = [
    "V4ClosureNode",
    "V4CompatibilityError",
    "V4CompatibilityRead",
    "read_v4_artifact",
]
