"""Fail-closed verifier for the additive I0 package source manifest."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import load_canonical_resource, validate_semantic_sha

from ._core import IntelligenceContractError

PACKAGE_MANIFEST_VERSION: Final = "myquant.v17.research-intelligence.package-manifest.v1"
MANIFEST_RESOURCE: Final = "resources/package_manifest.v1.json"


def _load_manifest() -> dict[str, Any]:
    try:
        raw = (
            resources.files("quant_investor.intelligence").joinpath(MANIFEST_RESOURCE).read_bytes()
        )
        document = load_canonical_resource(raw, label=PACKAGE_MANIFEST_VERSION)
        if type(document) is not dict:
            raise IntelligenceContractError("package manifest root must be an object")
        normalized = validate_semantic_sha(document)
    except IntelligenceContractError:
        raise
    except Exception as exc:
        raise IntelligenceContractError("package manifest is invalid") from exc
    if normalized.get("version") != PACKAGE_MANIFEST_VERSION:
        raise IntelligenceContractError("package manifest version mismatch")
    return normalized


def verify_package(package_root: Path | None = None) -> dict[str, Any]:
    """Verify the exact Python source set and byte hashes without modifying files."""

    root = Path(__file__).resolve().parent if package_root is None else package_root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise IntelligenceContractError("package root is unsafe")
    manifest = _load_manifest()
    rows = manifest.get("source_paths")
    if type(rows) is not list:
        raise IntelligenceContractError("package source manifest is missing")
    declared_paths = [row.get("relative_path") for row in rows if type(row) is dict]
    discovered_paths = sorted(
        (str(path.relative_to(root)) for path in root.rglob("*.py")),
        key=lambda value: value.encode("ascii"),
    )
    if declared_paths != discovered_paths:
        raise IntelligenceContractError("package source set drift detected")
    for row in rows:
        if type(row) is not dict or set(row) != {"byte_sha256", "relative_path"}:
            raise IntelligenceContractError("package source row is invalid")
        path = root / str(row["relative_path"])
        if path.is_symlink() or not path.is_file():
            raise IntelligenceContractError("package source path is unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["byte_sha256"]:
            raise IntelligenceContractError("package source byte SHA drift detected")
    return manifest


__all__ = ["PACKAGE_MANIFEST_VERSION", "verify_package"]
