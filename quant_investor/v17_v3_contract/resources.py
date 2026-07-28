"""Hash-bound package resources and source/distribution parity for v3.

The manifest never lists its own byte hash.  Its hash is held independently in
``PACKAGE_MANIFEST_SHA256``; the manifest then binds every other JSON asset.
Contract Python source paths are inventoried without hashes so an sdist and
wheel can be compared byte-for-byte without introducing a
manifest/resources.py hash cycle. Runtime and algorithm Python bytes are bound
separately by the packaged runtime-build manifest.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
from pathlib import Path, PurePath
from typing import Any, Final, Iterator, Mapping, cast

from .canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)

PROTOCOL_VERSION: Final = "myquant.v17.v3"
PACKAGE_MANIFEST_PATH: Final = "resources/package_manifest.v1.json"
PACKAGE_MANIFEST_SHA256: Final = "e8dc183437a631f60aa06e17dd59f10dccce21dd3c4d576adfa479dad6b2674c"
RUNTIME_BUILD_MANIFEST_PATH: Final = "resources/runtime_build_manifest.v1.json"
_PACKAGE_ROOT: Final = Path(__file__).resolve().parent
_RESOURCE_SESSION: ContextVar[dict[tuple[str, ...], object] | None] = ContextVar(
    "v17_v3_package_resource_session",
    default=None,
)


class PackageResourceError(RuntimeError):
    """Raised when the package inventory is absent, changed, or ambiguous."""

    exit_code = 2


@contextmanager
def package_resource_session() -> Iterator[None]:
    """Cache exact immutable package reads within one validation transaction."""

    if _RESOURCE_SESSION.get() is not None:
        yield
        return
    token = _RESOURCE_SESSION.set({})
    try:
        yield
    finally:
        _RESOURCE_SESSION.reset(token)


def _read_canonical_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = load_canonical_resource(raw, label=label)
        if type(payload) is not dict:
            raise PackageResourceError(f"{label} root must be an object")
        validate_semantic_sha(payload)
    except PackageResourceError:
        raise
    except (OSError, CanonicalContractError) as exc:
        raise PackageResourceError(f"{label} is unreadable or invalid") from exc
    return raw, payload


def _manifest_at(package_root: Path) -> tuple[bytes, dict[str, Any]]:
    cache = _RESOURCE_SESSION.get()
    key = ("manifest", str(package_root.resolve()))
    if cache is not None and key in cache:
        raw, manifest = cast(
            tuple[bytes, dict[str, Any]],
            cache[key],
        )
        return raw, deepcopy(manifest)
    raw, manifest = _read_canonical_object(
        package_root / PACKAGE_MANIFEST_PATH,
        label="v3 package manifest",
    )
    observed = hashlib.sha256(raw).hexdigest()
    if observed != PACKAGE_MANIFEST_SHA256:
        raise PackageResourceError("v3 package manifest byte SHA-256 mismatch")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise PackageResourceError("v3 package manifest protocol mismatch")
    if manifest.get("version") != "myquant.v17.v3.package-manifest.v1":
        raise PackageResourceError("v3 package manifest version mismatch")
    self_binding = manifest.get("self_binding")
    if self_binding != {
        "byte_sha256_source": ("quant_investor.v17_v3_contract.resources.PACKAGE_MANIFEST_SHA256"),
        "relative_path": PACKAGE_MANIFEST_PATH,
    }:
        raise PackageResourceError("v3 package manifest self-binding policy mismatch")
    if cache is not None:
        cache[key] = (raw, deepcopy(manifest))
    return raw, manifest


def load_package_manifest(*, package_root: Path | None = None) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    return _manifest_at(root)[1]


def _asset_rows(manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    rows = manifest.get("assets")
    if type(rows) is not list:
        raise PackageResourceError("v3 package asset inventory must be an array")
    result: list[dict[str, str]] = []
    previous: str | None = None
    seen_casefold: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "artifact_id",
            "byte_sha256",
            "relative_path",
        }:
            raise PackageResourceError(f"v3 package asset row {index} shape mismatch")
        artifact_id = row["artifact_id"]
        relative_path = row["relative_path"]
        byte_sha256 = row["byte_sha256"]
        if not all(type(value) is str for value in (artifact_id, relative_path, byte_sha256)):
            raise PackageResourceError(f"v3 package asset row {index} values are invalid")
        if previous is not None and relative_path <= previous:
            raise PackageResourceError("v3 package assets are not in ASCII path order")
        if relative_path.casefold() in seen_casefold:
            raise PackageResourceError("v3 package asset paths have a casefold collision")
        if (
            relative_path == PACKAGE_MANIFEST_PATH
            or not relative_path.endswith(".json")
            or not relative_path.startswith(("resources/", "schemas/"))
            or len(byte_sha256) != 64
            or any(character not in "0123456789abcdef" for character in byte_sha256)
        ):
            raise PackageResourceError(f"v3 package asset row {index} is noncanonical")
        previous = relative_path
        seen_casefold.add(relative_path.casefold())
        result.append(
            {
                "artifact_id": artifact_id,
                "byte_sha256": byte_sha256,
                "relative_path": relative_path,
            }
        )
    return tuple(result)


def _source_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = manifest.get("source_paths")
    if (
        type(values) is not list
        or any(type(value) is not str or not value.endswith(".py") for value in values)
        or values != sorted(values)
        or len(values) != len({value.casefold() for value in values})
    ):
        raise PackageResourceError("v3 source path inventory is invalid")
    return tuple(values)


def _runtime_source_rows(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    if (
        manifest.get("version") != "myquant.v17.v3.runtime-build-manifest.v1"
        or manifest.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise PackageResourceError("v3 runtime-build manifest identity mismatch")
    rows = manifest.get("sources")
    if type(rows) is not list or not rows:
        raise PackageResourceError("v3 runtime source inventory must be a nonempty array")
    result: list[dict[str, str]] = []
    previous: str | None = None
    seen_casefold: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "byte_sha256",
            "relative_path",
        }:
            raise PackageResourceError(f"v3 runtime source row {index} shape mismatch")
        relative_path = row["relative_path"]
        byte_sha256 = row["byte_sha256"]
        if (
            type(relative_path) is not str
            or type(byte_sha256) is not str
            or not relative_path.startswith("v17_v3_runtime/")
            or not relative_path.endswith(".py")
            or "__pycache__" in PurePath(relative_path).parts
            or len(byte_sha256) != 64
            or any(character not in "0123456789abcdef" for character in byte_sha256)
        ):
            raise PackageResourceError(f"v3 runtime source row {index} is noncanonical")
        if previous is not None and relative_path <= previous:
            raise PackageResourceError("v3 runtime sources are not in ASCII path order")
        if relative_path.casefold() in seen_casefold:
            raise PackageResourceError("v3 runtime source paths have a casefold collision")
        previous = relative_path
        seen_casefold.add(relative_path.casefold())
        result.append(
            {
                "byte_sha256": byte_sha256,
                "relative_path": relative_path,
            }
        )
    return tuple(result)


def read_packaged_asset(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> bytes:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    cache = _RESOURCE_SESSION.get()
    key = ("asset-raw", str(root.resolve()), relative_path)
    if cache is not None and key in cache:
        return cast(bytes, cache[key])
    manifest = load_package_manifest(package_root=root)
    expected = {row["relative_path"]: row["byte_sha256"] for row in _asset_rows(manifest)}
    if type(relative_path) is not str or relative_path not in expected:
        raise PackageResourceError(f"unknown v3 package asset: {relative_path!r}")
    try:
        raw = (root / relative_path).read_bytes()
    except OSError as exc:
        raise PackageResourceError(f"v3 package asset is unreadable: {relative_path}") from exc
    if hashlib.sha256(raw).hexdigest() != expected[relative_path]:
        raise PackageResourceError(f"v3 package asset byte SHA-256 mismatch: {relative_path}")
    try:
        payload = load_canonical_resource(raw, label=relative_path)
        if type(payload) is not dict:
            raise PackageResourceError(f"v3 package asset root is not an object: {relative_path}")
        if relative_path.startswith("resources/"):
            validate_semantic_sha(payload)
    except CanonicalContractError as exc:
        raise PackageResourceError(f"v3 package asset is invalid: {relative_path}") from exc
    if cache is not None:
        cache[key] = raw
    return raw


def load_packaged_json(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    cache = _RESOURCE_SESSION.get()
    key = ("asset-json", str(root.resolve()), relative_path)
    if cache is not None and key in cache:
        return deepcopy(cast(dict[str, Any], cache[key]))
    payload = load_canonical_resource(
        read_packaged_asset(relative_path, package_root=root),
        label=relative_path,
    )
    if type(payload) is not dict:
        raise PackageResourceError(f"v3 package asset root is not an object: {relative_path}")
    if cache is not None:
        cache[key] = deepcopy(payload)
    return payload


def verify_runtime_build(
    *,
    package_root: Path | None = None,
) -> dict[str, str]:
    """Verify every manifest-bound v3 runtime and algorithm Python byte."""

    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_packaged_json(
        RUNTIME_BUILD_MANIFEST_PATH,
        package_root=root,
    )
    rows = _runtime_source_rows(manifest)
    quant_investor_root = root.parent
    runtime_root = quant_investor_root / "v17_v3_runtime"
    discovered = sorted(
        path.relative_to(quant_investor_root).as_posix()
        for path in runtime_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    expected = [row["relative_path"] for row in rows]
    if discovered != expected:
        raise PackageResourceError("v3 runtime Python inventory differs from the build manifest")
    result: dict[str, str] = {}
    for row in rows:
        relative_path = row["relative_path"]
        try:
            raw = (quant_investor_root / relative_path).read_bytes()
        except OSError as exc:
            raise PackageResourceError(f"v3 runtime source is unreadable: {relative_path}") from exc
        observed = hashlib.sha256(raw).hexdigest()
        if observed != row["byte_sha256"]:
            raise PackageResourceError(f"v3 runtime source byte SHA-256 mismatch: {relative_path}")
        result[relative_path] = observed
    return dict(sorted(result.items()))


def verify_package(*, package_root: Path | None = None) -> dict[str, str]:
    """Verify the exact manifest-bound package asset and source-path inventory."""

    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_package_manifest(package_root=root)
    rows = _asset_rows(manifest)
    source_paths = _source_paths(manifest)
    discovered_assets = sorted(
        path.relative_to(root).as_posix()
        for directory in ("resources", "schemas")
        for path in (root / directory).glob("*.json")
        if path.relative_to(root).as_posix() != PACKAGE_MANIFEST_PATH
    )
    if discovered_assets != [row["relative_path"] for row in rows]:
        raise PackageResourceError("v3 package JSON inventory differs from the manifest")
    discovered_sources = sorted(path.relative_to(root).as_posix() for path in root.glob("*.py"))
    if discovered_sources != list(source_paths):
        raise PackageResourceError("v3 package source path inventory differs from the manifest")
    result = {PACKAGE_MANIFEST_PATH: PACKAGE_MANIFEST_SHA256}
    for row in rows:
        raw = read_packaged_asset(row["relative_path"], package_root=root)
        result[row["relative_path"]] = hashlib.sha256(raw).hexdigest()
    verify_runtime_build(package_root=root)
    return dict(sorted(result.items()))


def manifest_inventory(*, package_root: Path | None = None) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_package_manifest(package_root=root)
    return {
        "assets": _asset_rows(manifest),
        "runtime_sources": _runtime_source_rows(
            load_packaged_json(
                RUNTIME_BUILD_MANIFEST_PATH,
                package_root=root,
            )
        ),
        "source_paths": _source_paths(manifest),
    }


def verify_source_parity(
    source_root: Path,
    distribution_root: Path,
) -> dict[str, str]:
    """Compare manifest-inventoried sdist/wheel package trees byte-for-byte."""

    source = Path(source_root)
    distribution = Path(distribution_root)
    manifest = load_package_manifest(package_root=source)
    relative_paths = [
        PACKAGE_MANIFEST_PATH,
        *_source_paths(manifest),
        *(row["relative_path"] for row in _asset_rows(manifest)),
    ]
    result: dict[str, str] = {}
    for relative_path in relative_paths:
        try:
            source_raw = (source / relative_path).read_bytes()
            distribution_raw = (distribution / relative_path).read_bytes()
        except OSError as exc:
            raise PackageResourceError(
                f"v3 source parity member is missing: {relative_path}"
            ) from exc
        if source_raw != distribution_raw:
            raise PackageResourceError(f"v3 source parity mismatch: {relative_path}")
        result[relative_path] = hashlib.sha256(source_raw).hexdigest()
    expected = set(relative_paths)
    for root, label in ((source, "source"), (distribution, "distribution")):
        discovered = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json"}
        }
        if discovered != expected:
            raise PackageResourceError(f"v3 {label} tree contains unmanifested members")
    runtime_rows = _runtime_source_rows(
        load_packaged_json(
            RUNTIME_BUILD_MANIFEST_PATH,
            package_root=source,
        )
    )
    runtime_relative_paths = [row["relative_path"] for row in runtime_rows]
    for relative_path in runtime_relative_paths:
        try:
            source_raw = (source.parent / relative_path).read_bytes()
            distribution_raw = (distribution.parent / relative_path).read_bytes()
        except OSError as exc:
            raise PackageResourceError(
                f"v3 runtime source parity member is missing: {relative_path}"
            ) from exc
        if source_raw != distribution_raw:
            raise PackageResourceError(f"v3 runtime source parity mismatch: {relative_path}")
        result[relative_path] = hashlib.sha256(source_raw).hexdigest()
    for root, label in (
        (source.parent / "v17_v3_runtime", "source"),
        (distribution.parent / "v17_v3_runtime", "distribution"),
    ):
        discovered_runtime = sorted(
            path.relative_to(root.parent).as_posix()
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
        if discovered_runtime != runtime_relative_paths:
            raise PackageResourceError(f"v3 {label} runtime tree contains unmanifested members")
    return dict(sorted(result.items()))


__all__ = [
    "PACKAGE_MANIFEST_PATH",
    "PACKAGE_MANIFEST_SHA256",
    "PROTOCOL_VERSION",
    "RUNTIME_BUILD_MANIFEST_PATH",
    "PackageResourceError",
    "load_package_manifest",
    "load_packaged_json",
    "manifest_inventory",
    "package_resource_session",
    "read_packaged_asset",
    "verify_package",
    "verify_runtime_build",
    "verify_source_parity",
]
