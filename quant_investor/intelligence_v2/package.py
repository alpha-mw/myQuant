"""Fail-closed verifier for the isolated Investment Intelligence v2 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from ._core import FROZEN_V1_MANIFEST_SHA256, IntelligenceV2ContractError, canonical_bytes

PACKAGE_MANIFEST_VERSION: Final = "myquant.v17.intelligence-v2.package-manifest.v1"
MANIFEST_RESOURCE: Final = "resources/package_manifest.v1.json"
FROZEN_V1_RESOURCE: Final = "resources/frozen_v1_manifest.v1.json"
_MANIFEST_FIELDS: Final = {
    "array_order_semantics",
    "frozen_v1_manifest_semantic_sha256",
    "resource_paths",
    "semantic_sha256",
    "source_paths",
    "version",
}


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntelligenceV2ContractError("package manifest contains a duplicate key")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except IntelligenceV2ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntelligenceV2ContractError(f"{label} is invalid") from exc
    if type(document) is not dict:
        raise IntelligenceV2ContractError(f"{label} root must be an object")
    return document


def _validate_semantic(document: dict[str, Any], *, label: str) -> dict[str, Any]:
    digest = document.get("semantic_sha256")
    if type(digest) is not str or len(digest) != 64:
        raise IntelligenceV2ContractError(f"{label} semantic SHA is invalid")
    body = dict(document)
    del body["semantic_sha256"]
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != digest:
        raise IntelligenceV2ContractError(f"{label} semantic SHA mismatch")
    return document


def _validate_rows(value: Any, *, label: str) -> list[dict[str, str]]:
    if type(value) is not list:
        raise IntelligenceV2ContractError(f"{label} must be a list")
    rows: list[dict[str, str]] = []
    for row in value:
        if type(row) is not dict or set(row) != {"byte_sha256", "relative_path"}:
            raise IntelligenceV2ContractError(f"{label} row is invalid")
        relative_path = row.get("relative_path")
        digest = row.get("byte_sha256")
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise IntelligenceV2ContractError(f"{label} row is invalid")
        rows.append({"byte_sha256": digest, "relative_path": relative_path})
    paths = [row["relative_path"] for row in rows]
    if paths != sorted(paths, key=lambda item: item.encode("ascii")) or len(paths) != len(
        set(paths)
    ):
        raise IntelligenceV2ContractError(f"{label} paths are not ASCII-sorted unique")
    return rows


def _verify_rows(root: Path, rows: list[dict[str, str]], *, label: str) -> None:
    for row in rows:
        path = root / row["relative_path"]
        if path.is_symlink() or not path.is_file():
            raise IntelligenceV2ContractError(f"{label} path is unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["byte_sha256"]:
            raise IntelligenceV2ContractError(f"{label} byte SHA drift detected")


def verify_package(package_root: Path | None = None) -> dict[str, Any]:
    """Verify the exact v2 Python/resource set and frozen-v1 identity."""

    root = Path(__file__).resolve().parent if package_root is None else Path(package_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise IntelligenceV2ContractError("v2 package root is unsafe")
    manifest = _validate_semantic(
        _load_json(root / MANIFEST_RESOURCE, label="v2 package manifest"),
        label="v2 package manifest",
    )
    if set(manifest) != _MANIFEST_FIELDS or manifest.get("version") != PACKAGE_MANIFEST_VERSION:
        raise IntelligenceV2ContractError("v2 package manifest shape/version mismatch")
    if manifest.get("array_order_semantics") != {
        "/resource_paths": "relative_path ASCII ascending",
        "/source_paths": "relative_path ASCII ascending",
    }:
        raise IntelligenceV2ContractError("v2 package ordering contract mismatch")
    if manifest.get("frozen_v1_manifest_semantic_sha256") != FROZEN_V1_MANIFEST_SHA256:
        raise IntelligenceV2ContractError("frozen-v1 manifest binding mismatch")

    source_rows = _validate_rows(manifest.get("source_paths"), label="source_paths")
    resource_rows = _validate_rows(manifest.get("resource_paths"), label="resource_paths")
    discovered_sources = sorted(
        (path.relative_to(root).as_posix() for path in root.rglob("*.py")),
        key=lambda item: item.encode("ascii"),
    )
    if [row["relative_path"] for row in source_rows] != discovered_sources:
        raise IntelligenceV2ContractError("v2 package source set drift detected")
    resource_root = root / "resources"
    discovered_resources = sorted(
        (
            path.relative_to(root).as_posix()
            for path in resource_root.rglob("*")
            if path.is_file() and path.name != Path(MANIFEST_RESOURCE).name
        ),
        key=lambda item: item.encode("ascii"),
    )
    if [row["relative_path"] for row in resource_rows] != discovered_resources:
        raise IntelligenceV2ContractError("v2 package resource set drift detected")
    _verify_rows(root, source_rows, label="source")
    _verify_rows(root, resource_rows, label="resource")

    frozen = _validate_semantic(
        _load_json(root / FROZEN_V1_RESOURCE, label="frozen-v1 manifest"),
        label="frozen-v1 manifest",
    )
    if frozen.get("semantic_sha256") != FROZEN_V1_MANIFEST_SHA256:
        raise IntelligenceV2ContractError("frozen-v1 resource identity mismatch")
    return manifest


__all__ = ["PACKAGE_MANIFEST_VERSION", "verify_package"]
