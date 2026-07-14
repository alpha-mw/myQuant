"""Append-only immutable generation store for macro v2 observations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from quant_investor.macro.contracts import MacroObservation, canonical_hash

DEFAULT_OBSERVATIONS_ROOT = Path("data/parquet/cn/macro_observations")
POINTER_FILENAME = "_latest.json"
GENERATIONS_DIRNAME = "_generations"
OBSERVATION_COLUMNS = tuple(MacroObservation.__dataclass_fields__)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class MacroObservationStoreError(RuntimeError):
    """Raised when observation storage cannot be validated or advanced."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_id(value: str) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_ID.fullmatch(text) or text in {".", ".."}:
        raise MacroObservationStoreError("macro_observation_run_id_unsafe")
    return text


def _safe_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise MacroObservationStoreError("macro_observation_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root.resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise MacroObservationStoreError("macro_observation_pointer_symlink_rejected")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    path = root / ".promotion.lock"
    if path.exists() and path.is_symlink():
        raise MacroObservationStoreError("macro_observation_lock_symlink_rejected")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _strict_pointer(root: Path) -> dict[str, Any] | None:
    path = root / POINTER_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise MacroObservationStoreError("macro_observation_pointer_unsafe")
    try:
        pointer_bytes = path.read_bytes()
        payload = json.loads(pointer_bytes.decode("utf-8"))
    except Exception as exc:
        raise MacroObservationStoreError("macro_observation_pointer_invalid") from exc
    if not isinstance(payload, Mapping):
        raise MacroObservationStoreError("macro_observation_pointer_not_object")
    pointer = dict(payload)
    if pointer.get("schema_version") != "macro-observation-pointer.v1" or pointer.get("status") != "OK":
        raise MacroObservationStoreError("macro_observation_pointer_shape_invalid")
    pointer["pointer_sha256"] = hashlib.sha256(pointer_bytes).hexdigest()
    return pointer


def pointer_sha256(root: str | Path) -> str:
    path = Path(root).expanduser() / POINTER_FILENAME
    if not path.exists() or path.is_symlink() or not path.is_file():
        return ""
    return _sha256(path)


def _resolve_generation(root: Path, pointer: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    generation_id = str(pointer.get("generation_id") or "").strip()
    relative = Path(str(pointer.get("table_path") or ""))
    manifest_relative = Path(str(pointer.get("manifest_path") or ""))
    declared = str(pointer.get("parquet_sha256") or "")
    manifest_declared = str(pointer.get("manifest_sha256") or "")
    invalid_pointer = any(
        (
            not generation_id,
            not relative,
            relative.is_absolute(),
            ".." in relative.parts,
            not manifest_relative,
            manifest_relative.is_absolute(),
            ".." in manifest_relative.parts,
            not declared,
            not manifest_declared,
        )
    )
    if invalid_pointer:
        raise MacroObservationStoreError("macro_observation_pointer_generation_invalid")
    for child in (relative, manifest_relative):
        cursor = root
        for part in child.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise MacroObservationStoreError("macro_observation_generation_symlink_rejected")
    table = (root / relative).resolve()
    manifest_path = (root / manifest_relative).resolve()
    expected_parent = (root / GENERATIONS_DIRNAME / generation_id).resolve()
    if table.parent != expected_parent or manifest_path.parent != expected_parent:
        raise MacroObservationStoreError("macro_observation_generation_path_mismatch")
    if root not in table.parents or not table.is_file():
        raise MacroObservationStoreError("macro_observation_generation_table_unsafe")
    if root not in manifest_path.parents or not manifest_path.is_file():
        raise MacroObservationStoreError("macro_observation_generation_manifest_unsafe")
    if _sha256(table) != declared:
        raise MacroObservationStoreError("macro_observation_generation_hash_mismatch")
    if _sha256(manifest_path) != manifest_declared:
        raise MacroObservationStoreError("macro_observation_manifest_hash_mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MacroObservationStoreError("macro_observation_manifest_invalid") from exc
    invalid_manifest = not isinstance(manifest, Mapping) or any(
        (
            manifest.get("schema_version") != "macro-observation-generation.v1",
            manifest.get("status") != "OK",
            manifest.get("generation_id") != generation_id,
            manifest.get("parquet_sha256") != declared,
        )
    )
    if invalid_manifest:
        raise MacroObservationStoreError("macro_observation_manifest_shape_invalid")
    return table, dict(manifest)


def load_observations(
    root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    *,
    generation_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = _safe_root(root)
    pointer = _strict_pointer(base)
    if pointer is None and not generation_id:
        return [], {}
    if generation_id:
        safe_generation = _safe_id(generation_id)
        generation_root = Path(GENERATIONS_DIRNAME) / safe_generation
        manifest_path = base / generation_root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise MacroObservationStoreError("macro_observation_generation_missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MacroObservationStoreError("macro_observation_manifest_invalid") from exc
        pointer = {
            "schema_version": "macro-observation-pointer.v1",
            "status": "OK",
            "generation_id": safe_generation,
            "table_path": str(generation_root / "observations.parquet"),
            "manifest_path": str(generation_root / "manifest.json"),
            "parquet_sha256": manifest.get("parquet_sha256"),
            "manifest_sha256": _sha256(manifest_path),
            "content_set_hash": manifest.get("content_set_hash"),
            "row_count": manifest.get("row_count"),
        }
    assert pointer is not None
    table, manifest = _resolve_generation(base, pointer)
    frame = pd.read_parquet(table)
    if tuple(frame.columns) != OBSERVATION_COLUMNS:
        raise MacroObservationStoreError("macro_observation_generation_schema_mismatch")
    rows: list[dict[str, Any]] = []
    for payload in frame.to_dict(orient="records"):
        rows.append(MacroObservation.from_mapping(payload).to_dict())
    if int(pointer.get("row_count", -1)) != len(rows):
        raise MacroObservationStoreError("macro_observation_generation_row_count_mismatch")
    content_set_hash = canonical_hash({"hashes": sorted(row["content_hash"] for row in rows)})
    if content_set_hash != str(pointer.get("content_set_hash") or ""):
        raise MacroObservationStoreError("macro_observation_content_set_hash_mismatch")
    if manifest.get("content_set_hash") != content_set_hash or int(manifest.get("row_count", -1)) != len(rows):
        raise MacroObservationStoreError("macro_observation_manifest_content_mismatch")
    return rows, {**pointer, "generation_manifest": manifest}


def _normalize_rows(observations: Iterable[Mapping[str, Any] | MacroObservation]) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    identity: dict[tuple[str, str, str, str, str, str], str] = {}
    for value in observations:
        payload = value.to_dict() if isinstance(value, MacroObservation) else value
        item = MacroObservation.from_mapping(payload)
        row = item.to_dict()
        key = (
            item.indicator_id,
            item.period_end,
            item.available_at,
            item.vintage_id,
            item.source_system,
            item.source_record_id,
        )
        previous = identity.get(key)
        if previous is not None and previous != item.content_hash:
            raise MacroObservationStoreError("macro_observation_conflicting_vintage")
        identity[key] = item.content_hash
        by_hash[item.content_hash] = row
    return [by_hash[key] for key in sorted(by_hash)]


def publish_observations(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    run_id: str,
    expected_pointer_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = _safe_root(root)
    generation_id = _safe_id(run_id)
    incoming = _normalize_rows(observations)
    if not incoming:
        return {"status": "no_update", "promoted": False, "reason": "empty_input"}

    with _locked(base):
        current_sha = pointer_sha256(base)
        if expected_pointer_sha256 is not None and current_sha != expected_pointer_sha256:
            raise MacroObservationStoreError("macro_observation_pointer_cas_mismatch")
        existing, pointer = load_observations(base)
        combined = _normalize_rows([*existing, *incoming])
        if len(combined) == len(existing):
            return {**pointer, "status": "no_update", "promoted": False, "reason": "all_rows_exist"}

        generations = base / GENERATIONS_DIRNAME
        if generations.exists() and (generations.is_symlink() or not generations.is_dir()):
            raise MacroObservationStoreError("macro_observation_generations_root_unsafe")
        generations.mkdir(parents=True, exist_ok=True, mode=0o700)
        final = generations / generation_id
        if final.exists():
            raise MacroObservationStoreError("macro_observation_generation_exists")
        staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
        try:
            frame = pd.DataFrame(combined, columns=OBSERVATION_COLUMNS)
            table = staging / "observations.parquet"
            frame.to_parquet(table, index=False)
            os.chmod(table, 0o600)
            table_descriptor = os.open(table, os.O_RDONLY)
            try:
                os.fsync(table_descriptor)
            finally:
                os.close(table_descriptor)
            readback = pd.read_parquet(table)
            if len(readback) != len(frame) or tuple(readback.columns) != OBSERVATION_COLUMNS:
                raise MacroObservationStoreError("macro_observation_generation_readback_failed")
            parquet_hash = _sha256(table)
            content_set_hash = canonical_hash({"hashes": sorted(frame["content_hash"].astype(str))})
            added_hashes = sorted(set(frame["content_hash"].astype(str)) - {row["content_hash"] for row in existing})
            manifest = {
                "schema_version": "macro-observation-generation.v1",
                "status": "OK",
                "generation_id": generation_id,
                "row_count": len(frame),
                "parquet_sha256": parquet_hash,
                "content_set_hash": content_set_hash,
                "created_at": _now_utc(),
                "parent_generation_id": str(pointer.get("generation_id") or ""),
                "parent_pointer_sha256": current_sha,
                "added_content_hashes": added_hashes,
                "min_available_at": min(row["available_at"] for row in combined),
                "max_available_at": max(row["available_at"] for row in combined),
                "metadata": dict(metadata or {}),
            }
            manifest_path = staging / "manifest.json"
            _atomic_json(manifest_path, manifest)
            manifest_hash = _sha256(manifest_path)
            _fsync_directory(staging)
            os.replace(staging, final)
            _fsync_directory(final)
            _fsync_directory(generations)
            relative = final.relative_to(base)
            next_pointer = {
                "schema_version": "macro-observation-pointer.v1",
                "status": "OK",
                "generation_id": generation_id,
                "table_path": str(relative / "observations.parquet"),
                "manifest_path": str(relative / "manifest.json"),
                "parquet_sha256": parquet_hash,
                "manifest_sha256": manifest_hash,
                "content_set_hash": content_set_hash,
                "row_count": len(frame),
                "previous_generation_id": str(pointer.get("generation_id") or ""),
                "metadata": dict(metadata or {}),
            }
            _atomic_json(base / POINTER_FILENAME, next_pointer)
            return {**next_pointer, "promoted": True}
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if final.exists() and pointer_sha256(base) == current_sha:
                shutil.rmtree(final)
            raise


__all__ = [
    "DEFAULT_OBSERVATIONS_ROOT",
    "MacroObservationStoreError",
    "load_observations",
    "pointer_sha256",
    "publish_observations",
]
