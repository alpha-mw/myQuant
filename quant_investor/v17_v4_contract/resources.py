"""Hash-bound package and runtime resources for the V17 v4 contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePath
from typing import Any, Final, Mapping

from .canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from .identities import IdentityContractError, require_sha256

PROTOCOL_VERSION: Final = "myquant.v17.v4"
PACKAGE_MANIFEST_PATH: Final = "resources/package_manifest.v1.json"
RUNTIME_BUILD_MANIFEST_PATH: Final = "resources/runtime_build_manifest.v1.json"
PACKAGE_MANIFEST_SHA256: Final = (
    "01fd6bc8c39ac52880be0bb3ad17005537f8c1566d8f7a26a27a0f465b4eb7b4"
)
_PACKAGE_ROOT: Final = Path(__file__).resolve().parent
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class PackageResourceError(RuntimeError):
    """Raised when manifest-bound package or runtime bytes drift."""

    exit_code = 2


def _read_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = load_canonical_resource(raw, label=label)
    except (OSError, CanonicalContractError) as exc:
        raise PackageResourceError(f"{label} is unreadable or invalid") from exc
    if type(payload) is not dict:
        raise PackageResourceError(f"{label} root must be an object")
    return raw, payload


def load_package_manifest(*, package_root: Path | None = None) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    raw, manifest = _read_object(
        root / PACKAGE_MANIFEST_PATH,
        label="v4 package manifest",
    )
    try:
        require_sha256(PACKAGE_MANIFEST_SHA256, label="package manifest SHA-256")
        validate_semantic_sha(manifest)
    except (IdentityContractError, CanonicalContractError) as exc:
        raise PackageResourceError("v4 package manifest seal is invalid") from exc
    if hashlib.sha256(raw).hexdigest() != PACKAGE_MANIFEST_SHA256:
        raise PackageResourceError("v4 package manifest byte SHA-256 mismatch")
    if set(manifest) != {
        "array_order_semantics",
        "assets",
        "authority",
        "protocol_version",
        "self_binding",
        "semantic_sha256",
        "source_paths",
        "version",
    } or (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("version") != "myquant.v17.v4.package-manifest.v1"
        or manifest.get("authority") != _NO_AUTHORITY
        or manifest.get("self_binding")
        != {
            "byte_sha256_source": (
                "quant_investor.v17_v4_contract.resources.PACKAGE_MANIFEST_SHA256"
            ),
            "relative_path": PACKAGE_MANIFEST_PATH,
        }
    ):
        raise PackageResourceError("v4 package manifest identity mismatch")
    return manifest


def _asset_rows(manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    rows = manifest.get("assets")
    if type(rows) is not list:
        raise PackageResourceError("v4 package assets must be an array")
    result: list[dict[str, str]] = []
    previous: str | None = None
    seen: set[str] = set()
    artifact_ids: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "artifact_id",
            "byte_sha256",
            "relative_path",
        }:
            raise PackageResourceError(f"v4 package asset row {index} shape mismatch")
        relative_path = row["relative_path"]
        artifact_id = row["artifact_id"]
        byte_sha256 = row["byte_sha256"]
        if (
            type(relative_path) is not str
            or type(artifact_id) is not str
            or relative_path == PACKAGE_MANIFEST_PATH
            or not relative_path.startswith(("resources/", "schemas/"))
            or not relative_path.endswith(".json")
            or "__pycache__" in PurePath(relative_path).parts
        ):
            raise PackageResourceError(f"v4 package asset row {index} is noncanonical")
        try:
            require_sha256(byte_sha256)
        except IdentityContractError as exc:
            raise PackageResourceError(
                f"v4 package asset row {index} SHA is invalid"
            ) from exc
        if previous is not None and relative_path <= previous:
            raise PackageResourceError("v4 package assets are not ASCII path ordered")
        if relative_path.casefold() in seen:
            raise PackageResourceError("v4 package asset paths casefold-collide")
        if artifact_id.casefold() in artifact_ids:
            raise PackageResourceError("v4 package artifact identities casefold-collide")
        previous = relative_path
        seen.add(relative_path.casefold())
        artifact_ids.add(artifact_id.casefold())
        result.append(dict(row))
    return tuple(result)


def _source_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = manifest.get("source_paths")
    if (
        type(values) is not list
        or any(
            type(value) is not str
            or not value.endswith(".py")
            or "/" in value
            for value in values
        )
        or values != sorted(values)
        or len(values) != len({value.casefold() for value in values})
    ):
        raise PackageResourceError("v4 source path inventory is invalid")
    return tuple(values)


def read_packaged_asset(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> bytes:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    expected = {
        row["relative_path"]: row["byte_sha256"]
        for row in _asset_rows(load_package_manifest(package_root=root))
    }
    if type(relative_path) is not str or relative_path not in expected:
        raise PackageResourceError(f"unknown v4 package asset: {relative_path!r}")
    try:
        raw = (root / relative_path).read_bytes()
    except OSError as exc:
        raise PackageResourceError(
            f"v4 package asset is unreadable: {relative_path}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected[relative_path]:
        raise PackageResourceError(
            f"v4 package asset byte SHA-256 mismatch: {relative_path}"
        )
    try:
        payload = load_canonical_resource(raw, label=relative_path)
        if type(payload) is not dict:
            raise PackageResourceError(
                f"v4 package asset root is not an object: {relative_path}"
            )
        if relative_path.startswith("resources/"):
            validate_semantic_sha(payload)
    except CanonicalContractError as exc:
        raise PackageResourceError(
            f"v4 package asset is invalid: {relative_path}"
        ) from exc
    return raw


def load_packaged_json(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    payload = load_canonical_resource(
        read_packaged_asset(relative_path, package_root=package_root),
        label=relative_path,
    )
    if type(payload) is not dict:
        raise PackageResourceError(f"{relative_path} root must be an object")
    return deepcopy(payload)


def _runtime_rows(manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    if set(manifest) != {
        "array_order_semantics",
        "authority",
        "protocol_version",
        "semantic_sha256",
        "sources",
        "version",
    } or (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("version")
        != "myquant.v17.v4.runtime-build-manifest.v1"
        or manifest.get("authority") != _NO_AUTHORITY
    ):
        raise PackageResourceError("v4 runtime-build manifest identity mismatch")
    rows = manifest.get("sources")
    if type(rows) is not list or not rows:
        raise PackageResourceError("v4 runtime source inventory is empty")
    result: list[dict[str, str]] = []
    previous: str | None = None
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "byte_sha256",
            "relative_path",
        }:
            raise PackageResourceError(f"v4 runtime source row {index} shape mismatch")
        relative_path = row["relative_path"]
        if (
            type(relative_path) is not str
            or not relative_path.startswith("v17_v4_runtime/")
            or not relative_path.endswith(".py")
            or "__pycache__" in PurePath(relative_path).parts
            or (previous is not None and relative_path <= previous)
        ):
            raise PackageResourceError(f"v4 runtime source row {index} is noncanonical")
        try:
            require_sha256(row["byte_sha256"])
        except IdentityContractError as exc:
            raise PackageResourceError(
                f"v4 runtime source row {index} SHA is invalid"
            ) from exc
        previous = relative_path
        result.append(dict(row))
    return tuple(result)


def verify_runtime_build(
    *,
    package_root: Path | None = None,
) -> dict[str, str]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_packaged_json(
        RUNTIME_BUILD_MANIFEST_PATH,
        package_root=root,
    )
    rows = _runtime_rows(manifest)
    quant_investor_root = root.parent
    runtime_root = quant_investor_root / "v17_v4_runtime"
    discovered = sorted(
        path.relative_to(quant_investor_root).as_posix()
        for path in runtime_root.glob("*.py")
    )
    expected = [row["relative_path"] for row in rows]
    if discovered != expected:
        raise PackageResourceError(
            "v4 runtime Python inventory differs from the build manifest"
        )
    result: dict[str, str] = {}
    for row in rows:
        relative_path = row["relative_path"]
        try:
            raw = (quant_investor_root / relative_path).read_bytes()
        except OSError as exc:
            raise PackageResourceError(
                f"v4 runtime source is unreadable: {relative_path}"
            ) from exc
        observed = hashlib.sha256(raw).hexdigest()
        if observed != row["byte_sha256"]:
            raise PackageResourceError(
                f"v4 runtime source byte SHA-256 mismatch: {relative_path}"
            )
        result[relative_path] = observed
    return dict(sorted(result.items()))


def verify_package(*, package_root: Path | None = None) -> dict[str, str]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_package_manifest(package_root=root)
    rows = _asset_rows(manifest)
    discovered_assets = sorted(
        path.relative_to(root).as_posix()
        for directory in ("resources", "schemas")
        for path in (root / directory).glob("*.json")
        if path.relative_to(root).as_posix() != PACKAGE_MANIFEST_PATH
    )
    if discovered_assets != [row["relative_path"] for row in rows]:
        raise PackageResourceError(
            "v4 package JSON inventory differs from the manifest"
        )
    discovered_sources = sorted(path.name for path in root.glob("*.py"))
    if discovered_sources != list(_source_paths(manifest)):
        raise PackageResourceError(
            "v4 package source inventory differs from the manifest"
        )
    result = {PACKAGE_MANIFEST_PATH: PACKAGE_MANIFEST_SHA256}
    for row in rows:
        raw = read_packaged_asset(row["relative_path"], package_root=root)
        result[row["relative_path"]] = hashlib.sha256(raw).hexdigest()
        if row["relative_path"].startswith("schemas/"):
            from .schema_validation import preflight_schema

            preflight_schema(
                load_packaged_json(
                    row["relative_path"],
                    package_root=root,
                )
            )
    verify_runtime_build(package_root=root)
    return dict(sorted(result.items()))


__all__ = [
    "PACKAGE_MANIFEST_PATH",
    "PACKAGE_MANIFEST_SHA256",
    "PROTOCOL_VERSION",
    "PackageResourceError",
    "load_package_manifest",
    "load_packaged_json",
    "read_packaged_asset",
    "verify_package",
    "verify_runtime_build",
]
