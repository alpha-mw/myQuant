"""Atomic generation storage for the CN PIT fundamental mart."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


FUNDAMENTAL_POINTER_FILENAME = "_fundamental_latest.json"
FUNDAMENTAL_GENERATIONS_DIRNAME = "_fundamental_generations"
FUNDAMENTAL_TABLES = (
    "fundamental_period",
    "fundamental_daily",
    "fundamental_quarantine",
)


class FundamentalGenerationError(ValueError):
    """Raised when a fundamental generation cannot be read or published."""


def fundamental_data_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path.parent.parent
    if path.name in FUNDAMENTAL_TABLES:
        return path.parent
    return path


def legacy_fundamental_table_path(
    root: str | Path,
    table_name: str,
) -> Path:
    if table_name not in FUNDAMENTAL_TABLES:
        raise FundamentalGenerationError(f"unsupported fundamental table: {table_name}")
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path
    base = fundamental_data_root(path)
    return base / table_name / "part.parquet"


def _safe_generation_id(run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip())
    value = value.strip("._")
    if not value:
        raise FundamentalGenerationError("fundamental generation run_id is empty")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FundamentalGenerationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FundamentalGenerationError(f"JSON object required: {path}")
    return dict(payload)


def _resolve_inside(base: Path, value: str, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise FundamentalGenerationError(f"{label} path is empty")
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise FundamentalGenerationError(f"{label} escapes fundamental root: {resolved}")
    return resolved


@lru_cache(maxsize=8)
def _load_fundamental_pointer_cached(
    base_text: str,
    pointer_mtime_ns: int,
    pointer_size: int,
) -> dict[str, Any]:
    del pointer_mtime_ns, pointer_size
    base = Path(base_text)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    payload = _strict_json(pointer_path)
    if payload.get("status") != "OK":
        raise FundamentalGenerationError("fundamental pointer status is not OK")
    generation_id = str(payload.get("generation_id", "")).strip()
    if not generation_id:
        raise FundamentalGenerationError("fundamental pointer generation_id missing")
    manifest_path = _resolve_inside(
        base,
        str(payload.get("manifest_path", "")),
        label="manifest",
    )
    manifest = _strict_json(manifest_path)
    if str(manifest.get("generation_id", "")) != generation_id:
        raise FundamentalGenerationError("fundamental pointer/manifest generation mismatch")
    tables = dict(payload.get("tables", {}) or {})
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental pointer table set mismatch")
    for table_name, table_value in tables.items():
        table_path = _resolve_inside(base, str(table_value), label=table_name)
        if not table_path.exists() or not table_path.is_file():
            raise FundamentalGenerationError(f"fundamental table missing: {table_path}")
        expected_hash = str(
            dict(manifest.get("tables", {}) or {})
            .get(table_name, {})
            .get("sha256", "")
        )
        actual_hash = hashlib.sha256(table_path.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise FundamentalGenerationError(
                f"fundamental table hash mismatch: {table_name}"
            )
    payload["pointer_path"] = str(pointer_path)
    payload["manifest"] = manifest
    return payload


def load_fundamental_pointer(root: str | Path) -> dict[str, Any] | None:
    base = fundamental_data_root(root).resolve()
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    stat = pointer_path.stat()
    return dict(
        _load_fundamental_pointer_cached(
            str(base),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
    )


def resolve_fundamental_table_path(
    root: str | Path,
    table_name: str,
) -> Path:
    if Path(root).expanduser().suffix.lower() == ".parquet":
        return Path(root).expanduser()
    base = fundamental_data_root(root)
    pointer = load_fundamental_pointer(base)
    if pointer is None:
        return legacy_fundamental_table_path(base, table_name)
    return _resolve_inside(
        base,
        str(dict(pointer.get("tables", {}) or {}).get(table_name, "")),
        label=table_name,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def publish_fundamental_generation(
    *,
    root: str | Path,
    run_id: str,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    base = fundamental_data_root(root)
    generation_id = _safe_generation_id(run_id)
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental publish table set mismatch")
    generations_root = base / FUNDAMENTAL_GENERATIONS_DIRNAME
    generations_root.mkdir(parents=True, exist_ok=True)
    final_root = generations_root / generation_id
    if final_root.exists():
        pointer = load_fundamental_pointer(base)
        if pointer is not None and pointer.get("generation_id") == generation_id:
            return (
                {
                    table_name: resolve_fundamental_table_path(base, table_name)
                    for table_name in FUNDAMENTAL_TABLES
                },
                pointer,
            )
        raise FundamentalGenerationError(
            f"fundamental generation already exists: {generation_id}"
        )
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations_root)
    )
    table_paths: dict[str, Path] = {}
    table_manifest: dict[str, dict[str, Any]] = {}
    try:
        for table_name in FUNDAMENTAL_TABLES:
            frame = tables[table_name]
            path = staging_root / f"{table_name}.parquet"
            frame.to_parquet(path, index=False)
            readback = pd.read_parquet(path)
            if len(readback) != len(frame) or list(readback.columns) != list(frame.columns):
                raise FundamentalGenerationError(
                    f"fundamental table readback mismatch: {table_name}"
                )
            table_manifest[table_name] = {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": "cn-fundamental-generation.v1",
            "generation_id": generation_id,
            "status": "OK",
            "tables": table_manifest,
            "metadata": dict(metadata),
        }
        _atomic_write_json(staging_root / "manifest.json", manifest)
        os.replace(staging_root, final_root)
        relative_root = final_root.relative_to(base)
        for table_name in FUNDAMENTAL_TABLES:
            table_paths[table_name] = final_root / f"{table_name}.parquet"
        pointer_metadata = {
            key: metadata.get(key)
            for key in (
                "run_id",
                "provider_status",
                "source_priority",
                "source_provenance",
                "storage_backend",
                "readiness",
                "gate2_passed",
                "merge",
            )
            if key in metadata
        }
        pointer = {
            "schema_version": "cn-fundamental-pointer.v1",
            "status": "OK",
            "generation_id": generation_id,
            "manifest_path": str(relative_root / "manifest.json"),
            "tables": {
                table_name: str(relative_root / f"{table_name}.parquet")
                for table_name in FUNDAMENTAL_TABLES
            },
            "metadata": pointer_metadata,
        }
        _atomic_write_json(base / FUNDAMENTAL_POINTER_FILENAME, pointer)
        return table_paths, pointer
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if final_root.exists():
            try:
                current_pointer = load_fundamental_pointer(base)
            except FundamentalGenerationError:
                current_pointer = None
            if (
                current_pointer is not None
                and current_pointer.get("generation_id") == generation_id
            ):
                return table_paths, current_pointer
            shutil.rmtree(final_root)
        raise


__all__ = [
    "FUNDAMENTAL_GENERATIONS_DIRNAME",
    "FUNDAMENTAL_POINTER_FILENAME",
    "FUNDAMENTAL_TABLES",
    "FundamentalGenerationError",
    "fundamental_data_root",
    "legacy_fundamental_table_path",
    "load_fundamental_pointer",
    "publish_fundamental_generation",
    "resolve_fundamental_table_path",
]
