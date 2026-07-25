"""Offline, explicit source-object maintenance for the v17 shadow lane."""

from __future__ import annotations

from collections.abc import Mapping
import errno
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from .contracts import Availability
from .semantic import require_sha256, seal_semantic
from .source_bindings import (
    SOURCE_MANIFEST_VERSION,
    V17SourceBindingError,
    validate_source_manifest,
    validate_source_plan,
)
from .state_machine import EMPTY_SHA
from .storage import (
    atomic_write_bytes,
    atomic_write_json,
    ensure_v17_shadow_layout,
    file_sha256,
    read_json,
)

MAX_SOURCE_OBJECT_BYTES = 128 * 1024 * 1024


def _read_stable_source(path: str | Path, *, expected_sha256: str) -> bytes:
    expected = require_sha256(expected_sha256, label="expected source byte SHA-256")
    source = Path(path)
    if file_sha256(source) != expected:
        raise V17SourceBindingError("source byte SHA mismatch")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise V17SourceBindingError(f"source symlink rejected: {source}") from exc
        raise V17SourceBindingError(f"source unavailable: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise V17SourceBindingError("source must be one regular single-link file")
        if before.st_size <= 0 or before.st_size > MAX_SOURCE_OBJECT_BYTES:
            raise V17SourceBindingError("source object size is outside fixed bounds")
        chunks: list[bytes] = []
        remaining = MAX_SOURCE_OBJECT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise V17SourceBindingError("source changed during read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size:
        raise V17SourceBindingError("source read length mismatch")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise V17SourceBindingError("source byte SHA mismatch")
    current = os.lstat(source)
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise V17SourceBindingError("source path replaced during read")
    return raw


def _manifest_target(repo_root: Path, manifest_id: str) -> Path:
    return repo_root / "data" / "private" / "v17_sources" / "manifests" / f"{manifest_id}.json"


def _require_existing_matches_plan(
    existing: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    for key in ("manifest_id", "market", "cutoff", "created_at"):
        if existing.get(key) != plan.get(key):
            raise V17SourceBindingError(f"existing source manifest does not match plan: {key}")
    existing_sources = list(existing.get("sources", []))
    plan_sources = list(plan.get("sources", []))
    if len(existing_sources) != len(plan_sources):
        raise V17SourceBindingError("existing source manifest role count mismatch")
    for existing_item, plan_item in zip(existing_sources, plan_sources, strict=True):
        for key in ("source_id", "role", "availability"):
            if existing_item.get(key) != plan_item.get(key):
                raise V17SourceBindingError(f"existing source manifest binding mismatch: {key}")
        if plan_item["availability"] == Availability.AVAILABLE.value:
            if existing_item.get("byte_sha256") != plan_item.get("expected_byte_sha256"):
                raise V17SourceBindingError(
                    "existing source manifest byte binding differs from plan"
                )
        elif existing_item.get("reason") != plan_item.get("reason"):
            raise V17SourceBindingError(
                "existing source manifest unavailable reason differs from plan"
            )


def maintain_sources(
    repo_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    """Copy only explicitly hash-bound local files into private object storage."""

    root = Path(repo_root).absolute()
    validated_plan = validate_source_plan(plan)
    target = _manifest_target(root, str(validated_plan["manifest_id"]))
    if expected_manifest_sha256 != EMPTY_SHA:
        expected_existing = require_sha256(
            expected_manifest_sha256,
            label="expected source manifest SHA-256",
        )
        if not target.exists() or file_sha256(target) != expected_existing:
            raise V17SourceBindingError("source manifest CAS mismatch")
        existing = validate_source_manifest(
            read_json(target),
            repo_root=root,
            revalidate_objects=True,
        )
        _require_existing_matches_plan(existing, validated_plan)
        return existing, target, expected_existing
    if os.path.lexists(target):
        raise V17SourceBindingError("source manifest EMPTY CAS rejected")

    captured: list[tuple[dict[str, Any], bytes | None]] = []
    for item in validated_plan["sources"]:
        source = dict(item)
        if source["availability"] == Availability.UNAVAILABLE.value:
            captured.append((source, None))
            continue
        raw = _read_stable_source(
            source["path"],
            expected_sha256=source["expected_byte_sha256"],
        )
        captured.append((source, raw))

    layout = ensure_v17_shadow_layout(root)
    manifest_sources: list[dict[str, Any]] = []
    for source, raw in captured:
        if raw is None:
            manifest_sources.append(
                {
                    "source_id": source["source_id"],
                    "role": source["role"],
                    "availability": Availability.UNAVAILABLE.value,
                    "reason": source["reason"],
                }
            )
            continue
        byte_sha = hashlib.sha256(raw).hexdigest()
        object_path = layout["source_objects"] / byte_sha[:2] / f"{byte_sha}.bin"
        if os.path.lexists(object_path):
            if file_sha256(object_path) != byte_sha:
                raise V17SourceBindingError("content-addressed source object mismatch")
        else:
            atomic_write_bytes(
                object_path,
                raw,
                root=layout["private_sources"],
            )
        manifest_sources.append(
            {
                "source_id": source["source_id"],
                "role": source["role"],
                "availability": Availability.AVAILABLE.value,
                "object_path": object_path.relative_to(root).as_posix(),
                "byte_sha256": byte_sha,
                "size_bytes": len(raw),
            }
        )

    manifest = seal_semantic(
        {
            "version": SOURCE_MANIFEST_VERSION,
            "manifest_id": validated_plan["manifest_id"],
            "market": "CN",
            "cutoff": validated_plan["cutoff"],
            "created_at": validated_plan["created_at"],
            "sources": manifest_sources,
            "authority": False,
        }
    )
    manifest = validate_source_manifest(
        manifest,
        repo_root=root,
        revalidate_objects=True,
    )
    manifest_sha = atomic_write_json(
        target,
        manifest,
        root=layout["private_sources"],
    )
    readback = validate_source_manifest(
        read_json(target),
        repo_root=root,
        revalidate_objects=True,
    )
    if readback != manifest or file_sha256(target) != manifest_sha:
        raise V17SourceBindingError("source manifest readback mismatch")
    return readback, target, manifest_sha


def maintain_sources_from_plan_file(
    repo_root: str | Path,
    plan_path: str | Path,
    *,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    expected = require_sha256(
        expected_plan_sha256,
        label="expected source plan SHA-256",
    )
    before = file_sha256(plan_path)
    if before != expected:
        raise V17SourceBindingError("source plan byte SHA mismatch")
    plan = read_json(plan_path)
    after = file_sha256(plan_path)
    if after != before:
        raise V17SourceBindingError("source plan changed during read")
    return maintain_sources(
        repo_root,
        plan,
        expected_manifest_sha256=expected_manifest_sha256,
    )


__all__ = [
    "MAX_SOURCE_OBJECT_BYTES",
    "maintain_sources",
    "maintain_sources_from_plan_file",
]
