"""Authority-only archive planning and explicit offline copy custody."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final

from ..contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from .canonical import (
    canonical_relative_path,
    read_stable_regular_file,
    sha256_bytes,
    workspace_path,
)
from .errors import (
    ARCHIVE_COPY_CONFLICT,
    ARCHIVE_COPY_FORBIDDEN,
    ARCHIVE_PLAN_INVALID,
    REFERENCE_HASH_MISMATCH,
    SYMLINK_REFUSED,
    UnifiedCutoverError,
)
from .resolver import INVENTORY_CONTRACT_SHA256, INVENTORY_KIND
from .rules import ACTIVE_AUTHORITY, RULES_RELATIVE_PATH, load_rules

ARCHIVE_PLAN_KIND: Final = "system.migration.archive_plan"
ARCHIVE_PLAN_CONTRACT_SHA256: Final = get_contract(ARCHIVE_PLAN_KIND).contract_sha256
ARCHIVE_PLAN_PAYLOAD_FIELDS: Final = frozenset(
    {
        "archive_plan_id",
        "archive_root",
        "blocker_codes",
        "cutover_id",
        "entries",
        "inventory_ref",
        "status",
        "summary",
    }
)
_CUTOVER_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def artifact_exact_ref(document_or_bytes: Mapping[str, Any] | bytes) -> dict[str, str]:
    artifact = validate_artifact(document_or_bytes)
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _identity(kind: str, body: Mapping[str, Any], *, prefix: str) -> str:
    preimage = {"domain": "myquant-migration-identity", "kind": kind, "payload": dict(body)}
    return prefix + sha256_bytes(canonical_json_bytes(preimage))


def _inventory(document_or_bytes: Mapping[str, Any] | bytes) -> dict[str, Any]:
    try:
        inventory = validate_artifact(
            document_or_bytes,
            expected_kind=INVENTORY_KIND,
            expected_contract_sha256=INVENTORY_CONTRACT_SHA256,
        )
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "inventory artifact is invalid") from exc
    payload = inventory["payload"]
    if payload.get("status") != "COMPLETE" or payload.get("summary", {}).get("blocker_codes") != []:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "inventory is not blocker-free COMPLETE")
    files = payload.get("files")
    if type(files) is not list:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "inventory files are invalid")
    return inventory


def _validate_cutover_id(value: Any) -> str:
    if type(value) is not str or _CUTOVER_ID_RE.fullmatch(value) is None:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "cutover_id is not canonical")
    return value


def build_authority_archive_plan(
    workspace_root: str | os.PathLike[str],
    inventory: Mapping[str, Any] | bytes,
    *,
    cutover_id: str,
    created_at: str,
    rules_path: str | os.PathLike[str] = RULES_RELATIVE_PATH,
) -> dict[str, Any]:
    """Plan copies for the exact current authority closure, and nothing else."""

    root = Path(workspace_root).resolve(strict=True)
    normalized_cutover = _validate_cutover_id(cutover_id)
    normalized_inventory = _inventory(inventory)
    rules = load_rules(root, rules_path).rules
    archive_root = rules.unified_layout.archive_root(normalized_cutover)
    entries: list[dict[str, Any]] = []
    targets: set[str] = set()
    for row in normalized_inventory["payload"]["files"]:
        if type(row) is not dict:
            raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "inventory file row is invalid")
        if row.get("classification") != ACTIVE_AUTHORITY:
            continue
        source = canonical_relative_path(row.get("relative_path"), label="archive source")
        target = canonical_relative_path(f"{archive_root}/{source}", label="archive target")
        if target in targets:
            raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, f"archive target collision: {target}")
        targets.add(target)
        raw = read_stable_regular_file(
            workspace_path(root, source), label=f"authority archive source {source}"
        )
        digest = sha256_bytes(raw)
        if digest != row.get("byte_sha256") or len(raw) != row.get("bytes"):
            raise UnifiedCutoverError(
                REFERENCE_HASH_MISMATCH, f"authority source changed after inventory: {source}"
            )
        entries.append(
            {
                "source_relative_path": source,
                "source_byte_sha256": digest,
                "source_bytes": len(raw),
                "archive_relative_path": target,
                "copy_action": "COPY_AUTHORITY_BYTES",
            }
        )
    entries.sort(key=lambda row: row["source_relative_path"])
    body = {
        "archive_root": archive_root,
        "blocker_codes": [],
        "cutover_id": normalized_cutover,
        "entries": entries,
        "inventory_ref": artifact_exact_ref(normalized_inventory),
        "status": "PLANNED",
        "summary": {
            "file_count": len(entries),
            "total_bytes": sum(row["source_bytes"] for row in entries),
            "non_authority_copy_count": 0,
        },
    }
    payload = {
        **body,
        "archive_plan_id": _identity(ARCHIVE_PLAN_KIND, body, prefix="archive-plan-"),
    }
    if set(payload) != ARCHIVE_PLAN_PAYLOAD_FIELDS:
        raise AssertionError("archive plan payload fields drifted")
    return seal_artifact(
        ARCHIVE_PLAN_KIND,
        payload,
        created_at=created_at,
        contract_sha256=ARCHIVE_PLAN_CONTRACT_SHA256,
    )


def validate_authority_archive_plan(
    document_or_bytes: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        document = validate_artifact(
            document_or_bytes,
            expected_kind=ARCHIVE_PLAN_KIND,
            expected_contract_sha256=ARCHIVE_PLAN_CONTRACT_SHA256,
        )
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan artifact is invalid") from exc
    payload = document["payload"]
    if set(payload) != ARCHIVE_PLAN_PAYLOAD_FIELDS:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan payload fields are not exact")
    identity = payload["archive_plan_id"]
    body = dict(payload)
    body.pop("archive_plan_id")
    if identity != _identity(ARCHIVE_PLAN_KIND, body, prefix="archive-plan-"):
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan identity mismatch")
    if (
        payload["status"] != "PLANNED"
        or payload["blocker_codes"] != []
        or payload["summary"].get("non_authority_copy_count") != 0
    ):
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan is not copy-ready")
    entries = payload["entries"]
    if type(entries) is not list or entries != sorted(
        entries, key=lambda row: row.get("source_relative_path", "") if type(row) is dict else ""
    ):
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive entries are not sorted")
    seen: set[str] = set()
    total = 0
    for row in entries:
        if type(row) is not dict or set(row) != {
            "source_relative_path",
            "source_byte_sha256",
            "source_bytes",
            "archive_relative_path",
            "copy_action",
        }:
            raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive entry is invalid")
        source = canonical_relative_path(row["source_relative_path"])
        target = canonical_relative_path(row["archive_relative_path"])
        if (
            row["copy_action"] != "COPY_AUTHORITY_BYTES"
            or not target.startswith(payload["archive_root"] + "/")
            or target != f"{payload['archive_root']}/{source}"
            or target in seen
            or type(row["source_bytes"]) is not int
            or row["source_bytes"] < 0
        ):
            raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive entry escapes exact plan")
        seen.add(target)
        total += row["source_bytes"]
    if payload["summary"] != {
        "file_count": len(entries),
        "total_bytes": total,
        "non_authority_copy_count": 0,
    }:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan summary mismatch")
    return document


def _mkdir_real_tree(root: Path, relative_directory: str) -> None:
    current = root
    for part in PurePosixPath(canonical_relative_path(relative_directory)).parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                metadata = os.lstat(current)
            else:
                metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"archive parent is symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnifiedCutoverError(
                ARCHIVE_COPY_CONFLICT, f"archive parent is not directory: {current}"
            )


def copy_authority_archive(
    workspace_root: str | os.PathLike[str],
    archive_plan: Mapping[str, Any] | bytes,
    *,
    allow_copy: bool = False,
) -> dict[str, Any]:
    """Execute an authority-only plan after an explicit opt-in.

    This helper is deliberately absent from the resolver and receipt scripts;
    tests exercise it only against temporary workspaces.
    """

    if allow_copy is not True:
        raise UnifiedCutoverError(ARCHIVE_COPY_FORBIDDEN, "archive copy requires allow_copy=True")
    root = Path(workspace_root).resolve(strict=True)
    plan = validate_authority_archive_plan(archive_plan)
    copied = 0
    already_present = 0
    for row in plan["payload"]["entries"]:
        source = workspace_path(root, row["source_relative_path"])
        raw = read_stable_regular_file(source, label=f"authority source {source}")
        if sha256_bytes(raw) != row["source_byte_sha256"] or len(raw) != row["source_bytes"]:
            raise UnifiedCutoverError(
                REFERENCE_HASH_MISMATCH,
                f"authority source changed before copy: {row['source_relative_path']}",
            )
        target_relative = row["archive_relative_path"]
        _mkdir_real_tree(root, PurePosixPath(target_relative).parent.as_posix())
        target = workspace_path(root, target_relative)
        created = _write_read_only_archive_file(target, raw, label=target_relative)
        if created:
            copied += 1
        else:
            already_present += 1
    return {
        "status": "COPIED",
        "archive_plan_ref": artifact_exact_ref(plan),
        "copied_file_count": copied,
        "already_present_file_count": already_present,
        "total_file_count": copied + already_present,
    }


def _validate_archive_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise UnifiedCutoverError(
            ARCHIVE_COPY_CONFLICT, f"archive file unavailable: {label}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise UnifiedCutoverError(SYMLINK_REFUSED, f"archive file is symlink: {label}")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        raise UnifiedCutoverError(
            ARCHIVE_COPY_CONFLICT,
            f"archive file must be owner-controlled single-link mode 0444: {label}",
        )
    return metadata


def _write_read_only_archive_file(path: Path, raw: bytes, *, label: str) -> bool:
    """Create exact archive bytes once and seal them owner-read-only."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        _validate_archive_metadata(path, label=label)
        observed = read_stable_regular_file(path, label=f"existing archive {label}")
        if observed != raw:
            raise UnifiedCutoverError(
                ARCHIVE_COPY_CONFLICT, f"archive replay bytes differ: {label}"
            )
        return False

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _write_read_only_archive_file(path, raw, label=label)
    try:
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            written += os.write(descriptor, view[written:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _validate_archive_metadata(path, label=label)
    observed = read_stable_regular_file(path, label=f"written archive {label}")
    if observed != raw:
        raise UnifiedCutoverError(
            ARCHIVE_COPY_CONFLICT, f"archive exact-byte readback mismatch: {label}"
        )
    return True


__all__ = [
    "ARCHIVE_PLAN_CONTRACT_SHA256",
    "ARCHIVE_PLAN_KIND",
    "artifact_exact_ref",
    "build_authority_archive_plan",
    "copy_authority_archive",
    "validate_authority_archive_plan",
]
