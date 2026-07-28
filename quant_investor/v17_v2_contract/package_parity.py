"""Offline source/sdist/wheel/install byte-parity verification.

This module verifies already-built local artifacts.  It never resolves
dependencies, installs packages, opens a network connection, or mutates the
artifacts under review.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import configparser
from dataclasses import dataclass
from email.parser import Parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tomllib
from typing import Final, Mapping, NoReturn, Sequence
from urllib.parse import unquote, urlparse
import zipfile

PACKAGE_NAME: Final = "quant_investor"
DEFAULT_DISTRIBUTION_NAME: Final = "quant-investor"
DEFAULT_DISTRIBUTION_VERSION: Final = "17.0.0"
EXPECTED_PROJECT_SCRIPTS: Final = {
    "app": "web.main:app",
    "quant-investor": "quant_investor.cli.main:main",
}
WHEEL_REQUIRED_DIST_INFO_FILES: Final = frozenset(
    {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
)
INSTALLER_GENERATED_DIST_INFO_FILES: Final = frozenset(
    {"INSTALLER", "REQUESTED", "direct_url.json"}
)
MUTABLE_INSTALLED_DIST_INFO_FILES: Final = INSTALLER_GENERATED_DIST_INFO_FILES | {"RECORD"}

# Conservative fail-closed artifact ceilings.  The v17 contract package is small
# source code plus JSON/schema assets; these bounds leave operational headroom
# while preventing archive or RECORD materialization from growing unbounded.
MAX_ARTIFACT_BYTES: Final = 256 * 1024 * 1024
MAX_MEMBER_COUNT: Final = 10_000
MAX_MEMBER_BYTES: Final = 16 * 1024 * 1024
MAX_TOTAL_PAYLOAD_BYTES: Final = 128 * 1024 * 1024
MAX_RECORD_BYTES: Final = 4 * 1024 * 1024
MAX_RECORD_ROWS: Final = 10_000
MAX_RECORD_FIELD_CHARS: Final = 4_096
MAX_RECORD_PATH_CHARS: Final = 4_096
MAX_ZIP_EOCD_SEARCH_BYTES: Final = 22 + 65_535
MAX_ZIP_CENTRAL_DIRECTORY_BYTES: Final = 4 * 1024 * 1024
_ZIP_EOCD_SIGNATURE: Final = b"PK\x05\x06"
_ZIP64_EOCD_LOCATOR_SIGNATURE: Final = b"PK\x06\x07"
_ZIP_CENTRAL_DIRECTORY_SIGNATURE: Final = b"PK\x01\x02"
_ZIP64_EXTRA_FIELD_ID: Final = 0x0001

HashInfo = dict[str, object]
Inventory = dict[str, HashInfo]
RecordRows = dict[str, tuple[str, str]]
NodeKinds = dict[str, str]
StatSignature = tuple[int, int, int, int, int, int]
PhysicalSourceRow = dict[str, object]
PhysicalSourceSuperset = dict[str, object]
HatchNamespaceRow = dict[str, object]
HatchNamespaceBinding = dict[str, object]


@dataclass(frozen=True)
class _SourceProject:
    name: str
    version: str
    scripts: Mapping[str, str]
    pyproject_raw: bytes


@dataclass(frozen=True)
class _WheelInspection:
    package_inventory: Inventory
    provenance: dict[str, object]
    dist_info_files: Mapping[str, bytes]
    console_scripts: Mapping[str, str]


@dataclass
class _OpenStableFile:
    path: Path
    label: str
    descriptor: int
    signature: StatSignature
    opened_stat: os.stat_result
    raw: bytes

    def revalidate(self) -> None:
        try:
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = os.lstat(self.path)
        except OSError as exc:
            raise PackageParityError(f"{self.label} changed during verification") from exc
        if (
            _stat_signature(descriptor_stat) != self.signature
            or _stat_signature(path_stat) != self.signature
        ):
            raise PackageParityError(f"{self.label} changed during verification")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass
class _StableSourceDirectory:
    path: Path
    label: str
    descriptor: int
    signature: StatSignature
    parent_descriptor: int | None
    entry_name: str | None
    expected_names: tuple[str, ...] | None = None

    def revalidate(self) -> None:
        try:
            descriptor_stat = os.fstat(self.descriptor)
            if _stat_signature(descriptor_stat) != self.signature:
                raise PackageParityError(f"{self.label} changed during verification")
            if self.parent_descriptor is None or self.entry_name is None:
                path_stat = os.lstat(self.path)
                if _stat_signature(path_stat) != self.signature:
                    raise PackageParityError(f"{self.label} changed during verification")
            else:
                reopened = _open_directory_at(
                    self.parent_descriptor,
                    self.entry_name,
                    label=self.label,
                )
                try:
                    if reopened.signature != self.signature:
                        raise PackageParityError(f"{self.label} changed during verification")
                finally:
                    reopened.close()
            if self.expected_names is not None:
                observed_names = _directory_names(self.descriptor, label=self.label)
                if observed_names != self.expected_names:
                    raise PackageParityError(f"{self.label} namespace changed during verification")
        except PackageParityError:
            raise
        except OSError as exc:
            raise PackageParityError(f"{self.label} changed during verification") from exc

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass
class _StableSourceFile:
    path: Path
    label: str
    descriptor: int
    signature: StatSignature
    parent_descriptor: int
    entry_name: str
    raw: bytes

    def revalidate(self) -> None:
        try:
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = os.stat(
                self.entry_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_signature(descriptor_stat) != self.signature
                or _stat_signature(path_stat) != self.signature
            ):
                raise PackageParityError(f"{self.label} changed during verification")
            reopened = _open_source_file_at(
                self.parent_descriptor,
                self.entry_name,
                path=self.path,
                label=self.label,
            )
            try:
                if reopened.signature != self.signature or reopened.raw != self.raw:
                    raise PackageParityError(f"{self.label} changed during verification")
            finally:
                reopened.close()
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            if (
                _read_source_fd_bytes(
                    self.descriptor,
                    max_bytes=MAX_MEMBER_BYTES,
                    label=self.label,
                )
                != self.raw
            ):
                raise PackageParityError(f"{self.label} changed during verification")
            if _stat_signature(os.fstat(self.descriptor)) != self.signature:
                raise PackageParityError(f"{self.label} changed during verification")
        except PackageParityError:
            raise
        except OSError as exc:
            raise PackageParityError(f"{self.label} changed during verification") from exc

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True)
class _InstalledRecordInspection:
    summary: dict[str, object]
    package_inventory: Inventory
    dist_info_files: Mapping[str, bytes]


class PackageParityError(RuntimeError):
    """Raised when package payload inventories or bytes differ."""

    exit_code = 2


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")


def _canonical_json(payload: Mapping[str, object]) -> str:
    return _canonical_bytes(payload).decode("utf-8", errors="strict")


def _expected_dist_info_root(*, expected_name: str, expected_version: str) -> str:
    normalized_name = re.sub(r"[-_.]+", "_", expected_name).lower()
    return f"{normalized_name}-{expected_version}.dist-info"


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PackageParityError(f"{label} is unreadable") from exc


def _enforce_count_limit(count: int, *, limit: int, label: str, noun: str) -> None:
    if count > limit:
        raise PackageParityError(f"{label} exceeds {noun} limit: {count} > {limit}")


def _enforce_bytes_limit(size: int, *, limit: int, label: str) -> None:
    if size < 0:
        raise PackageParityError(f"{label} has invalid negative size")
    if size > limit:
        raise PackageParityError(f"{label} exceeds byte limit: {size} > {limit}")


def _enforce_total_bytes_limit(total: int, *, label: str) -> None:
    _enforce_bytes_limit(total, limit=MAX_TOTAL_PAYLOAD_BYTES, label=f"{label} total payload")


def _enforce_path_length(path: str, *, label: str) -> None:
    if len(path) > MAX_RECORD_PATH_CHARS:
        raise PackageParityError(
            f"{label} contains an oversized RECORD path: {len(path)} > {MAX_RECORD_PATH_CHARS}"
        )


def _resolve_path(path: Path, *, label: str, strict: bool = True) -> Path:
    try:
        return path.resolve(strict=strict)
    except OSError as exc:
        raise PackageParityError(f"{label} cannot be resolved") from exc


def _include_relative_path(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    return bool(parts) and "__pycache__" not in parts and not relative_path.endswith(".pyc")


def _hash_record(raw: bytes) -> HashInfo:
    return {"sha256": _sha256(raw), "size": len(raw)}


def _validate_archive_path(
    member_name: str,
    *,
    label: str,
    is_directory: bool = False,
) -> PurePosixPath:
    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise PackageParityError(f"{label} contains unsafe archive path: {member_name!r}")
    normalized = member_name
    if is_directory and normalized.endswith("/"):
        normalized = normalized[:-1]
    raw_parts = normalized.split("/")
    if (
        not normalized
        or any(part in ("", ".", "..") for part in raw_parts)
        or (not is_directory and member_name.endswith("/"))
    ):
        raise PackageParityError(f"{label} contains unsafe archive path: {member_name!r}")
    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or path.as_posix() != normalized:
        raise PackageParityError(f"{label} contains unsafe archive path: {member_name!r}")
    return path


def _validate_record_path(
    record_path: str,
    *,
    label: str,
    allow_leading_parents: bool = False,
) -> PurePosixPath:
    _enforce_path_length(record_path, label=label)
    if not record_path or "\x00" in record_path or "\\" in record_path or record_path.endswith("/"):
        raise PackageParityError(f"{label} contains unsafe RECORD path: {record_path!r}")
    raw_parts = record_path.split("/")
    if any(part in ("", ".") for part in raw_parts):
        raise PackageParityError(f"{label} contains unsafe RECORD path: {record_path!r}")
    seen_non_parent = False
    for part in raw_parts:
        if part == "..":
            if not allow_leading_parents or seen_non_parent:
                raise PackageParityError(f"{label} contains unsafe RECORD path: {record_path!r}")
        else:
            seen_non_parent = True
    if not seen_non_parent:
        raise PackageParityError(f"{label} contains unsafe RECORD path: {record_path!r}")
    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or path.as_posix() != record_path:
        raise PackageParityError(f"{label} contains unsafe RECORD path: {record_path!r}")
    return path


def _casefold_collision(path: str, existing: Mapping[str, object]) -> bool:
    key = path.casefold()
    return any(observed.casefold() == key for observed in existing)


def _record_node(
    nodes: NodeKinds,
    *,
    relative_path: str,
    is_directory: bool,
    label: str,
) -> None:
    canonical_key = relative_path.casefold()
    node_kind = "directory" if is_directory else "file"
    for observed_path, observed_kind in nodes.items():
        observed_key = observed_path.casefold()
        if observed_key == canonical_key:
            raise PackageParityError(
                f"{label} contains duplicate or casefold-colliding path: " f"{relative_path!r}"
            )
        if canonical_key.startswith(observed_key + "/") and observed_kind == "file":
            raise PackageParityError(
                f"{label} contains a file-directory collision: {relative_path!r}"
            )
        if observed_key.startswith(canonical_key + "/") and node_kind == "file":
            raise PackageParityError(
                f"{label} contains a file-directory collision: {relative_path!r}"
            )
    nodes[relative_path] = node_kind


def _record_file(
    inventory: Inventory,
    *,
    relative_path: str,
    raw: bytes,
    label: str,
) -> None:
    path = _validate_archive_path(relative_path, label=label)
    canonical = path.as_posix()
    if not _include_relative_path(canonical):
        return
    if _casefold_collision(canonical, inventory):
        raise PackageParityError(
            f"{label} contains duplicate or casefold-colliding path: {canonical!r}"
        )
    inventory[canonical] = _hash_record(raw)


def _normalize_package_root(package_root: Path, *, label: str) -> Path:
    if package_root.is_symlink():
        raise PackageParityError(f"{label} package root cannot be a symlink")
    root = _resolve_path(package_root, label=label)
    if not root.is_dir():
        raise PackageParityError(f"{label} is not a package directory")
    for quant_index in range(len(root.parts) - 1, -1, -1):
        if root.parts[quant_index] != PACKAGE_NAME:
            continue
        candidate = Path(*root.parts[: quant_index + 1])
        package_init = candidate / "__init__.py"
        if candidate.is_dir() and package_init.is_file() and not package_init.is_symlink():
            return _resolve_path(
                candidate,
                label=f"{label} {PACKAGE_NAME!r} package root",
            )
    raise PackageParityError(f"{label} is not inside a qualifying {PACKAGE_NAME!r} package")


def collect_directory_payload(package_root: Path, *, label: str) -> Inventory:
    """Hash all regular files below one concrete ``quant_investor`` directory."""

    root = _normalize_package_root(package_root, label=label)
    snapshot = _StableSourceNamespace.from_repo_root(
        root.parent,
        extra_paths=(),
        label=label,
    )
    try:
        inventory = snapshot.package_inventory()
        snapshot.revalidate()
        return inventory
    finally:
        snapshot.close()


def _sdist_package_relative(path: PurePosixPath) -> str | None:
    parts = path.parts
    if len(parts) < 2 or parts[1] != PACKAGE_NAME:
        if PACKAGE_NAME in parts:
            raise PackageParityError(
                f"sdist contains an incorrectly nested package path: {path.as_posix()}"
            )
        return None
    return PurePosixPath(*parts[1:]).as_posix()


def _expected_sdist_root(*, expected_name: str, expected_version: str) -> str:
    normalized_name = re.sub(r"[-_.]+", "_", expected_name).lower()
    return f"{normalized_name}-{expected_version}"


def _collect_sdist_payload_from_bytes(
    sdist_raw: bytes,
    *,
    expected_name: str,
    expected_version: str,
    expected_pyproject_raw: bytes | None,
) -> Inventory:
    inventory: Inventory = {}
    nodes: NodeKinds = {}
    pyproject_raw: bytes | None = None
    member_count = 0
    total_payload_bytes = 0
    expected_root = _expected_sdist_root(
        expected_name=expected_name,
        expected_version=expected_version,
    )
    pyproject_member = f"{expected_root}/pyproject.toml"
    try:
        with tarfile.open(fileobj=io.BytesIO(sdist_raw), mode="r:*") as archive:
            for member in archive:
                member_count += 1
                _enforce_count_limit(
                    member_count,
                    limit=MAX_MEMBER_COUNT,
                    label="sdist",
                    noun="member count",
                )
                is_directory = member.isdir()
                if not member.isfile() and not is_directory:
                    raise PackageParityError(f"sdist contains a non-regular member: {member.name}")
                if member.isfile():
                    _enforce_bytes_limit(
                        member.size,
                        limit=MAX_MEMBER_BYTES,
                        label=f"sdist member {member.name}",
                    )
                    total_payload_bytes += member.size
                    _enforce_total_bytes_limit(total_payload_bytes, label="sdist")
                path = _validate_archive_path(
                    member.name,
                    label="sdist",
                    is_directory=is_directory,
                )
                canonical = path.as_posix()
                _record_node(
                    nodes,
                    relative_path=canonical,
                    is_directory=is_directory,
                    label="sdist",
                )
                if path.parts[0] != expected_root:
                    raise PackageParityError(f"sdist archive root mismatch: {path.parts[0]!r}")
                if is_directory:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageParityError(f"sdist member is unreadable: {member.name}")
                raw = stream.read()
                if len(raw) != member.size:
                    raise PackageParityError(
                        f"sdist member changed while it was read: {member.name}"
                    )
                if canonical == pyproject_member:
                    pyproject_raw = raw
                relative = _sdist_package_relative(path)
                if relative is None:
                    continue
                if relative == PACKAGE_NAME:
                    raise PackageParityError("sdist package root is not a directory")
                _record_file(
                    inventory,
                    relative_path=relative,
                    raw=raw,
                    label="sdist",
                )
    except (OSError, tarfile.TarError) as exc:
        raise PackageParityError("sdist is unreadable") from exc
    if pyproject_raw is None:
        raise PackageParityError("sdist pyproject.toml is missing")
    if expected_pyproject_raw is not None and pyproject_raw != expected_pyproject_raw:
        raise PackageParityError("sdist pyproject.toml differs from source")
    if not inventory:
        raise PackageParityError("sdist contract payload is empty")
    return dict(sorted(inventory.items()))


def _collect_sdist_payload(
    sdist_path: Path,
    *,
    expected_name: str,
    expected_version: str,
    expected_pyproject_raw: bytes | None,
) -> Inventory:
    opened = _open_stable_file_no_follow(sdist_path, label="sdist")
    try:
        inventory = _collect_sdist_payload_from_bytes(
            opened.raw,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_pyproject_raw=expected_pyproject_raw,
        )
        opened.revalidate()
        return inventory
    finally:
        opened.close()


def collect_sdist_payload(sdist_path: Path) -> Inventory:
    """Hash the ``quant_investor`` payload inside one source distribution."""

    return _collect_sdist_payload(
        sdist_path,
        expected_name=DEFAULT_DISTRIBUTION_NAME,
        expected_version=DEFAULT_DISTRIBUTION_VERSION,
        expected_pyproject_raw=None,
    )


def _wheel_file_type(member: zipfile.ZipInfo) -> int:
    return (member.external_attr >> 16) & 0o170000


def _zip_u16(raw: bytes, offset: int) -> int:
    return int.from_bytes(raw[offset : offset + 2], "little")


def _zip_u32(raw: bytes, offset: int) -> int:
    return int.from_bytes(raw[offset : offset + 4], "little")


def _zip_extra_contains_zip64(extra: bytes) -> bool:
    offset = 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise PackageParityError("wheel central directory extra field is malformed")
        header_id = _zip_u16(extra, offset)
        data_size = _zip_u16(extra, offset + 2)
        next_offset = offset + 4 + data_size
        if next_offset > len(extra):
            raise PackageParityError("wheel central directory extra field is malformed")
        if header_id == _ZIP64_EXTRA_FIELD_ID:
            return True
        offset = next_offset
    return False


def _zip_eocd_candidates(raw: bytes) -> list[int]:
    search_start = max(0, len(raw) - MAX_ZIP_EOCD_SEARCH_BYTES)
    candidates: list[int] = []
    offset = raw.find(_ZIP_EOCD_SIGNATURE, search_start)
    while offset != -1:
        if offset + 22 <= len(raw):
            comment_length = _zip_u16(raw, offset + 20)
            if offset + 22 + comment_length == len(raw):
                candidates.append(offset)
        offset = raw.find(_ZIP_EOCD_SIGNATURE, offset + 1)
    return candidates


def _preflight_wheel_central_directory(wheel_raw: bytes) -> int:
    _enforce_bytes_limit(len(wheel_raw), limit=MAX_ARTIFACT_BYTES, label="wheel")
    if len(wheel_raw) < 22:
        raise PackageParityError("wheel is unreadable")
    candidates = _zip_eocd_candidates(wheel_raw)
    if not candidates:
        raise PackageParityError("wheel end-of-central-directory is missing")
    if len(candidates) != 1:
        raise PackageParityError("wheel end-of-central-directory is ambiguous")
    eocd_offset = candidates[0]
    if (
        eocd_offset >= 20
        and wheel_raw[eocd_offset - 20 : eocd_offset - 16] == _ZIP64_EOCD_LOCATOR_SIGNATURE
    ):
        raise PackageParityError("wheel ZIP64 records are not supported")

    disk_number = _zip_u16(wheel_raw, eocd_offset + 4)
    central_directory_disk = _zip_u16(wheel_raw, eocd_offset + 6)
    entries_on_disk = _zip_u16(wheel_raw, eocd_offset + 8)
    total_entries = _zip_u16(wheel_raw, eocd_offset + 10)
    central_directory_size = _zip_u32(wheel_raw, eocd_offset + 12)
    central_directory_offset = _zip_u32(wheel_raw, eocd_offset + 16)
    if disk_number != 0 or central_directory_disk != 0 or entries_on_disk != total_entries:
        raise PackageParityError("wheel multi-disk ZIP archives are not supported")
    if (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or central_directory_size == 0xFFFFFFFF
        or central_directory_offset == 0xFFFFFFFF
    ):
        raise PackageParityError("wheel ZIP64 records are not supported")
    _enforce_count_limit(
        total_entries,
        limit=MAX_MEMBER_COUNT,
        label="wheel",
        noun="member count",
    )
    _enforce_bytes_limit(
        central_directory_size,
        limit=MAX_ZIP_CENTRAL_DIRECTORY_BYTES,
        label="wheel central directory",
    )
    if central_directory_offset + central_directory_size != eocd_offset:
        raise PackageParityError("wheel central directory terminus is invalid")

    offset = central_directory_offset
    observed_entries = 0
    while offset < eocd_offset:
        if offset + 46 > eocd_offset:
            raise PackageParityError("wheel central directory header is truncated")
        if wheel_raw[offset : offset + 4] != _ZIP_CENTRAL_DIRECTORY_SIGNATURE:
            raise PackageParityError("wheel central directory header is malformed")
        compressed_size = _zip_u32(wheel_raw, offset + 20)
        uncompressed_size = _zip_u32(wheel_raw, offset + 24)
        filename_length = _zip_u16(wheel_raw, offset + 28)
        extra_length = _zip_u16(wheel_raw, offset + 30)
        comment_length = _zip_u16(wheel_raw, offset + 32)
        disk_start = _zip_u16(wheel_raw, offset + 34)
        local_header_offset = _zip_u32(wheel_raw, offset + 42)
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or disk_start == 0xFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            raise PackageParityError("wheel ZIP64 records are not supported")
        if disk_start != 0:
            raise PackageParityError("wheel multi-disk ZIP archives are not supported")
        if filename_length == 0 or filename_length > MAX_RECORD_PATH_CHARS:
            raise PackageParityError("wheel central directory filename length is invalid")
        variable_start = offset + 46
        variable_end = variable_start + filename_length + extra_length + comment_length
        if variable_end > eocd_offset:
            raise PackageParityError("wheel central directory variable fields are truncated")
        extra = wheel_raw[
            variable_start + filename_length : variable_start + filename_length + extra_length
        ]
        if _zip_extra_contains_zip64(extra):
            raise PackageParityError("wheel ZIP64 records are not supported")
        observed_entries += 1
        _enforce_count_limit(
            observed_entries,
            limit=MAX_MEMBER_COUNT,
            label="wheel",
            noun="member count",
        )
        offset = variable_end
    if offset != eocd_offset or observed_entries != total_entries:
        raise PackageParityError("wheel central directory count mismatch")
    return observed_entries


def _wheel_member_inventory(wheel_raw: bytes) -> tuple[Inventory, dict[str, bytes], str]:
    all_members: Inventory = {}
    raw_members: dict[str, bytes] = {}
    dist_info_roots: set[str] = set()
    nodes: NodeKinds = {}
    total_payload_bytes = 0
    try:
        preflight_member_count = _preflight_wheel_central_directory(wheel_raw)
        with zipfile.ZipFile(io.BytesIO(wheel_raw)) as archive:
            members = archive.infolist()
            if len(members) != preflight_member_count:
                raise PackageParityError("wheel central directory count mismatch")
            _enforce_count_limit(
                len(members),
                limit=MAX_MEMBER_COUNT,
                label="wheel",
                noun="member count",
            )
            for member in members:
                file_type = _wheel_file_type(member)
                is_directory = member.is_dir()
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise PackageParityError(
                        f"wheel contains a non-regular member: {member.filename}"
                    )
                if (is_directory and file_type == stat.S_IFREG) or (
                    not is_directory and file_type == stat.S_IFDIR
                ):
                    raise PackageParityError(
                        f"wheel member type disagrees with its path: {member.filename}"
                    )
                if not is_directory:
                    member_limit = (
                        MAX_RECORD_BYTES
                        if member.filename.endswith(".dist-info/RECORD")
                        else MAX_MEMBER_BYTES
                    )
                    _enforce_bytes_limit(
                        member.file_size,
                        limit=member_limit,
                        label=f"wheel member {member.filename}",
                    )
                    total_payload_bytes += member.file_size
                    _enforce_total_bytes_limit(total_payload_bytes, label="wheel")
                path = _validate_archive_path(
                    member.filename,
                    label="wheel",
                    is_directory=is_directory,
                )
                canonical = path.as_posix()
                _record_node(
                    nodes,
                    relative_path=canonical,
                    is_directory=is_directory,
                    label="wheel",
                )
                top = path.parts[0]
                if top.endswith(".dist-info"):
                    dist_info_roots.add(top)
                elif top != PACKAGE_NAME:
                    raise PackageParityError(f"wheel contains an unexpected root: {top!r}")
                if is_directory:
                    continue
                raw = archive.read(member)
                if len(raw) != member.file_size:
                    raise PackageParityError(f"wheel member changed while it was read: {canonical}")
                raw_members[canonical] = raw
                all_members[canonical] = _hash_record(raw)
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise PackageParityError("wheel is unreadable") from exc
    if len(dist_info_roots) != 1:
        raise PackageParityError("wheel must contain exactly one root dist-info directory")
    return dict(sorted(all_members.items())), raw_members, next(iter(dist_info_roots))


def _metadata_summary_from_bytes(
    raw: bytes,
    *,
    expected_name: str,
    expected_version: str,
    label: str,
) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise PackageParityError(f"{label} METADATA is not valid UTF-8") from exc
    message = Parser().parsestr(text)
    observed_name = message.get("Name")
    observed_version = message.get("Version")
    if observed_name != expected_name or observed_version != expected_version:
        raise PackageParityError(
            f"{label} METADATA name/version mismatch: {observed_name!r} {observed_version!r}"
        )
    return {"name": observed_name, "version": observed_version}


def _record_digest(raw: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(
        "ascii"
    )


def _parse_record_bytes(
    raw: bytes,
    *,
    label: str,
    allow_leading_parents: bool = False,
) -> RecordRows:
    _enforce_bytes_limit(len(raw), limit=MAX_RECORD_BYTES, label=f"{label} RECORD")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise PackageParityError(f"{label} RECORD is malformed") from exc

    parsed_rows = _parse_strict_record_rows(text, label=label)
    rows: RecordRows = {}
    nodes: NodeKinds = {}
    for row in parsed_rows:
        if len(row) != 3:
            raise PackageParityError(f"{label} RECORD contains a malformed row")
        path, digest, size = row
        canonical = _validate_record_path(
            path,
            label=label,
            allow_leading_parents=allow_leading_parents,
        ).as_posix()
        _record_node(
            nodes,
            relative_path=canonical,
            is_directory=False,
            label=f"{label} RECORD",
        )
        rows[canonical] = (digest, size)
    return rows


def _parse_strict_record_rows(text: str, *, label: str) -> list[tuple[str, ...]]:
    parsed_rows: list[tuple[str, ...]] = []
    fields: list[str] = []
    field_chars: list[str] = []
    row_started = False
    field_start = True
    in_quotes = False
    after_quote = False
    index = 0

    def malformed() -> NoReturn:
        raise PackageParityError(f"{label} RECORD is malformed")

    def append_char(char: str) -> None:
        if len(field_chars) + 1 > MAX_RECORD_FIELD_CHARS:
            raise PackageParityError(
                f"{label} RECORD contains an oversized field: "
                f"{len(field_chars) + 1} > {MAX_RECORD_FIELD_CHARS}"
            )
        field_chars.append(char)

    def append_field() -> None:
        nonlocal field_chars, field_start, after_quote
        fields.append("".join(field_chars))
        field_chars = []
        field_start = True
        after_quote = False

    def append_row() -> None:
        nonlocal fields, row_started
        append_field()
        if len(fields) == 1 and fields[0] == "":
            malformed()
        _enforce_count_limit(
            len(parsed_rows) + 1,
            limit=MAX_RECORD_ROWS,
            label=f"{label} RECORD",
            noun="row count",
        )
        parsed_rows.append(tuple(fields))
        fields = []
        row_started = False

    while index < len(text):
        char = text[index]
        if in_quotes:
            if char == '"':
                if index + 1 < len(text) and text[index + 1] == '"':
                    append_char('"')
                    index += 2
                    continue
                in_quotes = False
                after_quote = True
            elif char in "\r\n":
                malformed()
            else:
                append_char(char)
            index += 1
            continue

        if after_quote:
            if char == ",":
                append_field()
                row_started = True
            elif char == "\n":
                append_row()
            elif char == "\r":
                if index + 1 >= len(text) or text[index + 1] != "\n":
                    malformed()
                append_row()
                index += 1
            else:
                malformed()
            index += 1
            continue

        if field_start and char == '"':
            in_quotes = True
            row_started = True
            field_start = False
        elif char == ",":
            append_field()
            row_started = True
        elif char == "\n":
            if not row_started and not fields and not field_chars:
                malformed()
            append_row()
        elif char == "\r":
            if index + 1 >= len(text) or text[index + 1] != "\n":
                malformed()
            if not row_started and not fields and not field_chars:
                malformed()
            append_row()
            index += 1
        elif char == '"':
            malformed()
        else:
            append_char(char)
            row_started = True
            field_start = False
        index += 1

    if in_quotes:
        malformed()
    if after_quote or row_started or fields or field_chars:
        append_row()
    return parsed_rows


def _verify_record_hash_size(
    *,
    rows: Mapping[str, tuple[str, str]],
    files: Mapping[str, bytes],
    record_path: str,
    label: str,
) -> None:
    if set(rows) != set(files):
        missing = sorted(set(files) - set(rows))
        extra = sorted(set(rows) - set(files))
        raise PackageParityError(
            f"{label} RECORD paths mismatch: missing={missing[:20]!r} extra={extra[:20]!r}"
        )
    for relative_path, raw in files.items():
        digest, size = rows[relative_path]
        if relative_path == record_path and digest == "" and size == "":
            continue
        if digest != _record_digest(raw):
            raise PackageParityError(f"{label} RECORD sha256 mismatch: {relative_path}")
        if size != str(len(raw)):
            raise PackageParityError(f"{label} RECORD size mismatch: {relative_path}")


def _inventory_sha256(inventory: Mapping[str, Mapping[str, object]]) -> str:
    raw = _canonical_json(dict(inventory)).encode("utf-8")
    return _sha256(raw)


def _source_project_from_pyproject_bytes(raw: bytes) -> _SourceProject:
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except UnicodeError as exc:
        raise PackageParityError("pyproject.toml is unreadable as UTF-8") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PackageParityError("pyproject.toml is invalid TOML") from exc
    project = payload.get("project")
    if not isinstance(project, dict):
        raise PackageParityError("pyproject.toml is missing [project]")
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise PackageParityError("pyproject.toml is missing project name/version")
    if name != DEFAULT_DISTRIBUTION_NAME or version != DEFAULT_DISTRIBUTION_VERSION:
        raise PackageParityError("source pyproject.toml distribution identity mismatch")
    scripts = project.get("scripts")
    if not isinstance(scripts, dict) or not all(
        isinstance(script_name, str) and isinstance(target, str)
        for script_name, target in scripts.items()
    ):
        raise PackageParityError("source pyproject.toml is missing valid [project.scripts]")
    normalized_scripts = {str(script_name): str(target) for script_name, target in scripts.items()}
    if normalized_scripts != EXPECTED_PROJECT_SCRIPTS:
        raise PackageParityError("source pyproject.toml project scripts mismatch")
    return _SourceProject(
        name=name,
        version=version,
        scripts=dict(sorted(normalized_scripts.items())),
        pyproject_raw=raw,
    )


def _source_project_from_pyproject(source_root: Path) -> _SourceProject:
    snapshot = _StableSourceNamespace.from_repo_root(
        source_root.parent,
        extra_paths=("pyproject.toml",),
        label="source",
    )
    try:
        project = _source_project_from_pyproject_bytes(snapshot.file_bytes("pyproject.toml"))
        snapshot.revalidate()
        return project
    finally:
        snapshot.close()


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _console_scripts_from_bytes(raw: bytes, *, label: str) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise PackageParityError(f"{label} entry_points.txt is not valid UTF-8") from exc
    parser = _CaseSensitiveConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
    )
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise PackageParityError(f"{label} entry_points.txt is malformed") from exc
    if not parser.has_section("console_scripts"):
        raise PackageParityError(f"{label} entry_points.txt is missing [console_scripts]")
    if parser.defaults() or set(parser.sections()) != {"console_scripts"}:
        raise PackageParityError(f"{label} entry_points.txt contains unexpected entry-point groups")
    scripts = {
        script_name: target.strip()
        for script_name, target in parser.items("console_scripts", raw=True)
    }
    if any(not script_name or not target for script_name, target in scripts.items()):
        raise PackageParityError(f"{label} entry_points.txt contains an empty entry point")
    return dict(sorted(scripts.items()))


def _inspect_wheel_provenance_from_bytes(
    *,
    wheel_raw: bytes,
    expected_name: str,
    expected_version: str,
    expected_scripts: Mapping[str, str],
) -> _WheelInspection:
    members, raw_members, dist_info_root = _wheel_member_inventory(wheel_raw)
    expected_dist_info = _expected_dist_info_root(
        expected_name=expected_name,
        expected_version=expected_version,
    )
    if dist_info_root != expected_dist_info:
        raise PackageParityError(f"wheel dist-info directory name mismatch: {dist_info_root!r}")
    metadata_path = f"{dist_info_root}/METADATA"
    wheel_metadata_path = f"{dist_info_root}/WHEEL"
    record_path = f"{dist_info_root}/RECORD"
    entry_points_path = f"{dist_info_root}/entry_points.txt"
    for required_name in WHEEL_REQUIRED_DIST_INFO_FILES:
        required = f"{dist_info_root}/{required_name}"
        if required not in raw_members:
            raise PackageParityError(f"wheel is missing required dist-info file: {required}")
    dist_info_files = {
        path.removeprefix(f"{dist_info_root}/"): raw
        for path, raw in sorted(raw_members.items())
        if path.startswith(f"{dist_info_root}/")
    }
    forbidden = sorted(INSTALLER_GENERATED_DIST_INFO_FILES & set(dist_info_files))
    if forbidden:
        raise PackageParityError(
            f"wheel contains installer-generated dist-info files: {forbidden!r}"
        )
    metadata = _metadata_summary_from_bytes(
        raw_members[metadata_path],
        expected_name=expected_name,
        expected_version=expected_version,
        label="wheel",
    )
    console_scripts = _console_scripts_from_bytes(
        raw_members[entry_points_path],
        label="wheel",
    )
    if console_scripts != dict(expected_scripts):
        raise PackageParityError("wheel entry_points.txt does not match source project scripts")
    rows = _parse_record_bytes(raw_members[record_path], label="wheel")
    _verify_record_hash_size(
        rows=rows,
        files=raw_members,
        record_path=record_path,
        label="wheel",
    )
    package_inventory = {
        path: info for path, info in members.items() if path.startswith(f"{PACKAGE_NAME}/")
    }
    if not package_inventory:
        raise PackageParityError("wheel package payload is empty")
    dist_info_hashes = {
        relative_path: _sha256(raw) for relative_path, raw in dist_info_files.items()
    }
    return _WheelInspection(
        package_inventory=dict(sorted(package_inventory.items())),
        provenance={
            "dist_info_root": dist_info_root,
            "metadata": metadata,
            "record": {
                "file_count": len(rows),
                "record_sha256": _sha256(raw_members[record_path]),
            },
            "dist_info_file_sha256s": dist_info_hashes,
        },
        dist_info_files=dist_info_files,
        console_scripts=console_scripts,
    )


def _inspect_wheel_provenance(
    *,
    wheel_path: Path,
    expected_name: str,
    expected_version: str,
    expected_scripts: Mapping[str, str],
) -> _WheelInspection:
    opened = _open_stable_file_no_follow(wheel_path, label="wheel")
    try:
        inspection = _inspect_wheel_provenance_from_bytes(
            wheel_raw=opened.raw,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_scripts=expected_scripts,
        )
        opened.revalidate()
        return inspection
    finally:
        opened.close()


def inspect_wheel_provenance(
    *,
    wheel_path: Path,
    expected_name: str,
    expected_version: str,
) -> tuple[Inventory, dict[str, object]]:
    inspection = _inspect_wheel_provenance(
        wheel_path=wheel_path,
        expected_name=expected_name,
        expected_version=expected_version,
        expected_scripts=EXPECTED_PROJECT_SCRIPTS,
    )
    return inspection.package_inventory, inspection.provenance


def collect_wheel_payload(wheel_path: Path) -> Inventory:
    """Hash the ``quant_investor`` payload inside one wheel."""

    inventory, _ = inspect_wheel_provenance(
        wheel_path=wheel_path,
        expected_name=DEFAULT_DISTRIBUTION_NAME,
        expected_version=DEFAULT_DISTRIBUTION_VERSION,
    )
    return inventory


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _path_relative_to_base(path: Path, *, base: Path) -> str:
    return PurePosixPath(os.path.relpath(path, base)).as_posix()


def _stat_signature(observed: os.stat_result) -> StatSignature:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _read_fd_bytes(descriptor: int, *, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            descriptor_stat = os.fstat(descriptor)
        except OSError as exc:
            raise PackageParityError(f"{label} is unreadable") from exc
        _enforce_bytes_limit(descriptor_stat.st_size, limit=max_bytes, label=label)
        remaining = max_bytes - total
        if remaining <= 0:
            break
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PackageParityError(f"{label} exceeds byte limit: {total} > {max_bytes}")
        chunks.append(chunk)
    return b"".join(chunks)


def _open_stable_file_no_follow(
    path: Path,
    *,
    label: str,
    before: os.stat_result | None = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> _OpenStableFile:
    try:
        path_stat = os.lstat(path) if before is None else before
    except OSError as exc:
        raise PackageParityError(f"{label} does not exist") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise PackageParityError(f"{label} is not a regular non-symlink file")
    _enforce_bytes_limit(path_stat.st_size, limit=max_bytes, label=label)
    signature = _stat_signature(path_stat)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PackageParityError(f"{label} cannot be opened without following symlinks") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or _stat_signature(opened_stat) != signature:
            raise PackageParityError(f"{label} changed while it was opened")
        _enforce_bytes_limit(opened_stat.st_size, limit=max_bytes, label=label)
        raw = _read_fd_bytes(descriptor, max_bytes=max_bytes, label=label)
        post_descriptor_stat = os.fstat(descriptor)
        post_path_stat = os.lstat(path)
        if (
            len(raw) != signature[3]
            or _stat_signature(post_descriptor_stat) != signature
            or _stat_signature(post_path_stat) != signature
        ):
            raise PackageParityError(f"{label} changed while it was read")
    except PackageParityError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise PackageParityError(f"{label} is unreadable") from exc
    return _OpenStableFile(
        path=path,
        label=label,
        descriptor=descriptor,
        signature=signature,
        opened_stat=opened_stat,
        raw=raw,
    )


def _read_source_fd_bytes(
    descriptor: int,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    return _read_fd_bytes(descriptor, max_bytes=max_bytes, label=label)


def _directory_names(descriptor: int, *, label: str) -> tuple[str, ...]:
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise PackageParityError(f"{label} namespace cannot be listed") from exc
    for name in names:
        try:
            name.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise PackageParityError(f"{label} contains a non-UTF-8 path") from exc
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise PackageParityError(f"{label} contains an unsafe path component")
    return tuple(sorted(names))


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_absolute_directory(path: Path, *, label: str) -> _StableSourceDirectory:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise PackageParityError(f"{label} does not exist") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise PackageParityError(f"{label} is not a regular non-symlink directory")
    signature = _stat_signature(before)
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise PackageParityError(f"{label} cannot be opened without following symlinks") from exc
    try:
        opened = os.fstat(descriptor)
        path_stat = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_signature(opened) != signature
            or _stat_signature(path_stat) != signature
        ):
            raise PackageParityError(f"{label} changed while it was opened")
    except PackageParityError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise PackageParityError(f"{label} cannot be inspected") from exc
    return _StableSourceDirectory(
        path=path,
        label=label,
        descriptor=descriptor,
        signature=signature,
        parent_descriptor=None,
        entry_name=None,
    )


def _open_directory_at(
    parent_descriptor: int,
    entry_name: str,
    *,
    label: str,
    path: Path | None = None,
) -> _StableSourceDirectory:
    try:
        before = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise PackageParityError(f"{label} does not exist") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise PackageParityError(f"{label} is not a regular non-symlink directory")
    signature = _stat_signature(before)
    try:
        descriptor = os.open(
            entry_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise PackageParityError(f"{label} cannot be opened without following symlinks") from exc
    try:
        opened = os.fstat(descriptor)
        after = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_signature(opened) != signature
            or _stat_signature(after) != signature
        ):
            raise PackageParityError(f"{label} changed while it was opened")
    except PackageParityError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise PackageParityError(f"{label} cannot be inspected") from exc
    return _StableSourceDirectory(
        path=path or Path(entry_name),
        label=label,
        descriptor=descriptor,
        signature=signature,
        parent_descriptor=parent_descriptor,
        entry_name=entry_name,
    )


def _open_source_file_at(
    parent_descriptor: int,
    entry_name: str,
    *,
    path: Path,
    label: str,
    max_bytes: int = MAX_MEMBER_BYTES,
) -> _StableSourceFile:
    try:
        before = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise PackageParityError(f"{label} does not exist") from exc
    if not stat.S_ISREG(before.st_mode):
        raise PackageParityError(f"{label} is not a regular non-symlink file")
    if before.st_nlink != 1:
        raise PackageParityError(f"{label} is a hardlinked file")
    _enforce_bytes_limit(before.st_size, limit=max_bytes, label=label)
    signature = _stat_signature(before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(entry_name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise PackageParityError(f"{label} cannot be opened without following symlinks") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_signature(opened) != signature
        ):
            raise PackageParityError(f"{label} changed while it was opened")
        _enforce_bytes_limit(opened.st_size, limit=max_bytes, label=label)
        raw = _read_source_fd_bytes(descriptor, max_bytes=max_bytes, label=label)
        post_descriptor = os.fstat(descriptor)
        post_path = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            len(raw) != signature[3]
            or _stat_signature(post_descriptor) != signature
            or _stat_signature(post_path) != signature
        ):
            raise PackageParityError(f"{label} changed while it was read")
    except PackageParityError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise PackageParityError(f"{label} is unreadable") from exc
    return _StableSourceFile(
        path=path,
        label=label,
        descriptor=descriptor,
        signature=signature,
        parent_descriptor=parent_descriptor,
        entry_name=entry_name,
        raw=raw,
    )


def _validate_repo_relative_path(value: str | Path, *, label: str) -> str:
    raw = value.as_posix() if isinstance(value, Path) else value
    if not isinstance(raw, str):
        raise PackageParityError(f"{label} must be a repository-relative path")
    path = _validate_archive_path(raw, label=label)
    return path.as_posix()


class _StableSourceNamespace:
    def __init__(
        self,
        *,
        repo_root: Path,
        extra_paths: Sequence[str | Path],
        label: str,
    ) -> None:
        self._repo_root = Path(os.path.abspath(repo_root))
        self._label = label
        self._directories: list[_StableSourceDirectory] = []
        self._directories_by_relative_path: dict[str, _StableSourceDirectory] = {}
        self._files: dict[str, _StableSourceFile] = {}
        self._file_identities: dict[tuple[int, int], str] = {}
        self._node_kinds: NodeKinds = {}
        self._physical_rows: dict[str, PhysicalSourceRow] = {}
        self._file_count = 0
        self._payload_bytes = 0
        self._closed = False
        try:
            self._open(extra_paths=extra_paths)
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        extra_paths: Sequence[str | Path],
        label: str,
    ) -> _StableSourceNamespace:
        return cls(repo_root=repo_root, extra_paths=extra_paths, label=label)

    def _open(self, *, extra_paths: Sequence[str | Path]) -> None:
        if self._repo_root == self._repo_root.parent or not self._repo_root.name:
            raise PackageParityError(f"{self._label} repository root is invalid")
        repo_parent = _open_absolute_directory(
            self._repo_root.parent,
            label=f"{self._label} repository parent",
        )
        self._directories.append(repo_parent)
        repo_directory = _open_directory_at(
            repo_parent.descriptor,
            self._repo_root.name,
            label=f"{self._label} repository root",
            path=self._repo_root,
        )
        self._directories.append(repo_directory)
        self._directories_by_relative_path[""] = repo_directory
        package_directory = _open_directory_at(
            repo_directory.descriptor,
            PACKAGE_NAME,
            label=f"{self._label} package root",
            path=self._repo_root / PACKAGE_NAME,
        )
        self._directories.append(package_directory)
        self._directories_by_relative_path[PACKAGE_NAME] = package_directory
        self._record_node_row(
            relative_path=PACKAGE_NAME,
            kind="directory",
            observed=os.fstat(package_directory.descriptor),
            include_row=True,
        )
        self._scan_package_directory(
            package_directory,
            relative_path=PACKAGE_NAME,
        )
        if f"{PACKAGE_NAME}/__init__.py" not in self._files:
            raise PackageParityError(f"{self._label} is not a qualifying {PACKAGE_NAME!r} package")
        canonical_extra_paths: list[str] = []
        for index, value in enumerate(extra_paths):
            canonical = _validate_repo_relative_path(
                value,
                label=f"{self._label} extra_paths[{index}]",
            )
            if canonical == PACKAGE_NAME or canonical.startswith(f"{PACKAGE_NAME}/"):
                raise PackageParityError(
                    f"{self._label} extra path is already inside {PACKAGE_NAME}: {canonical!r}"
                )
            canonical_extra_paths.append(canonical)
        if len(canonical_extra_paths) != len(set(canonical_extra_paths)):
            raise PackageParityError(f"{self._label} extra paths contain duplicates")
        if len({path.casefold() for path in canonical_extra_paths}) != len(canonical_extra_paths):
            raise PackageParityError(f"{self._label} extra paths contain casefold collisions")
        for canonical in sorted(canonical_extra_paths):
            self._open_extra_file(canonical)

    def _record_node_row(
        self,
        *,
        relative_path: str,
        kind: str,
        observed: os.stat_result,
        include_row: bool,
        raw: bytes | None = None,
    ) -> None:
        _record_node(
            self._node_kinds,
            relative_path=relative_path,
            is_directory=kind == "directory",
            label=self._label,
        )
        if not include_row:
            return
        self._physical_rows[relative_path] = {
            "path": relative_path,
            "kind": kind,
            "mode": stat.S_IMODE(observed.st_mode),
            "sha256": None if raw is None else _sha256(raw),
            "size_bytes": 0 if raw is None else len(raw),
        }

    def _record_file(
        self,
        *,
        relative_path: str,
        parent: _StableSourceDirectory,
        entry_name: str,
        include_row: bool,
    ) -> None:
        try:
            observed = os.stat(
                entry_name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise PackageParityError(f"{self._label} file {relative_path} does not exist") from exc
        _enforce_count_limit(
            self._file_count + 1,
            limit=MAX_MEMBER_COUNT,
            label=self._label,
            noun="file count",
        )
        _enforce_bytes_limit(
            observed.st_size,
            limit=MAX_MEMBER_BYTES,
            label=f"{self._label} file {relative_path}",
        )
        _enforce_total_bytes_limit(
            self._payload_bytes + observed.st_size,
            label=self._label,
        )
        opened = _open_source_file_at(
            parent.descriptor,
            entry_name,
            path=self._repo_root / Path(*PurePosixPath(relative_path).parts),
            label=f"{self._label} file {relative_path}",
        )
        identity = (opened.signature[0], opened.signature[1])
        prior = self._file_identities.get(identity)
        if prior is not None:
            opened.close()
            raise PackageParityError(
                f"{self._label} contains hardlink-colliding paths: "
                f"{prior!r} and {relative_path!r}"
            )
        try:
            self._record_node_row(
                relative_path=relative_path,
                kind="file",
                observed=os.fstat(opened.descriptor),
                include_row=include_row,
                raw=opened.raw,
            )
        except BaseException:
            opened.close()
            raise
        self._file_count += 1
        self._payload_bytes += opened.signature[3]
        self._file_identities[identity] = relative_path
        self._files[relative_path] = opened

    def _scan_package_directory(
        self,
        directory: _StableSourceDirectory,
        *,
        relative_path: str,
    ) -> None:
        names = _directory_names(directory.descriptor, label=directory.label)
        directory.expected_names = names
        for name in names:
            child_relative = f"{relative_path}/{name}"
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise PackageParityError(
                    f"{self._label} package namespace changed while it was listed"
                ) from exc
            if stat.S_ISDIR(observed.st_mode):
                child_directory = _open_directory_at(
                    directory.descriptor,
                    name,
                    label=f"{self._label} directory {child_relative}",
                    path=self._repo_root / Path(*PurePosixPath(child_relative).parts),
                )
                self._directories.append(child_directory)
                self._directories_by_relative_path[child_relative] = child_directory
                self._record_node_row(
                    relative_path=child_relative,
                    kind="directory",
                    observed=os.fstat(child_directory.descriptor),
                    include_row=True,
                )
                self._scan_package_directory(
                    child_directory,
                    relative_path=child_relative,
                )
            elif stat.S_ISREG(observed.st_mode):
                self._record_file(
                    relative_path=child_relative,
                    parent=directory,
                    entry_name=name,
                    include_row=True,
                )
            elif stat.S_ISLNK(observed.st_mode):
                raise PackageParityError(f"{self._label} contains a symlink: {child_relative}")
            else:
                raise PackageParityError(
                    f"{self._label} contains a non-regular node: {child_relative}"
                )

    def _open_extra_file(self, relative_path: str) -> None:
        parts = PurePosixPath(relative_path).parts
        directory = self._directories_by_relative_path[""]
        parent_relative = ""
        for component in parts[:-1]:
            child_relative = component if not parent_relative else f"{parent_relative}/{component}"
            existing = self._directories_by_relative_path.get(child_relative)
            if existing is None:
                child_directory = _open_directory_at(
                    directory.descriptor,
                    component,
                    label=f"{self._label} extra-path directory {child_relative}",
                    path=self._repo_root / Path(*PurePosixPath(child_relative).parts),
                )
                self._directories.append(child_directory)
                self._directories_by_relative_path[child_relative] = child_directory
                self._record_node_row(
                    relative_path=child_relative,
                    kind="directory",
                    observed=os.fstat(child_directory.descriptor),
                    include_row=False,
                )
                existing = child_directory
            directory = existing
            parent_relative = child_relative
        try:
            self._record_file(
                relative_path=relative_path,
                parent=directory,
                entry_name=parts[-1],
                include_row=True,
            )
        except PackageParityError as exc:
            if relative_path == "pyproject.toml" and "does not exist" in str(exc):
                raise PackageParityError("source pyproject.toml is missing") from exc
            raise

    def package_inventory(self) -> Inventory:
        inventory: Inventory = {}
        for relative_path, opened in sorted(self._files.items()):
            if not relative_path.startswith(f"{PACKAGE_NAME}/"):
                continue
            _record_file(
                inventory,
                relative_path=relative_path,
                raw=opened.raw,
                label=self._label,
            )
        if not inventory:
            raise PackageParityError(f"{self._label} package payload is empty")
        return dict(sorted(inventory.items()))

    def file_bytes(self, relative_path: str) -> bytes:
        opened = self._files.get(relative_path)
        if opened is None:
            raise PackageParityError(
                f"{self._label} required source file is missing: {relative_path}"
            )
        return opened.raw

    def physical_binding(self) -> PhysicalSourceSuperset:
        rows = [dict(self._physical_rows[path]) for path in sorted(self._physical_rows)]
        return {
            "rows": rows,
            "row_count": len(rows),
            "sha256": _sha256(_canonical_bytes(rows)),
        }

    def revalidate(self) -> None:
        for relative_path in sorted(self._files):
            self._files[relative_path].revalidate()
        for directory in reversed(self._directories):
            directory.revalidate()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for opened in self._files.values():
            try:
                opened.close()
            except OSError:
                continue
        self._files.clear()
        for directory in reversed(self._directories):
            try:
                directory.close()
            except OSError:
                continue
        self._directories.clear()


class _StableDirectoryTree:
    def __init__(self, root: Path, *, label: str) -> None:
        self._root = Path(os.path.abspath(root))
        self._label = label
        self._directories: list[_StableSourceDirectory] = []
        self._files: dict[str, _StableSourceFile] = {}
        self._file_identities: dict[tuple[int, int], str] = {}
        self._nodes: NodeKinds = {}
        self._file_count = 0
        self._payload_bytes = 0
        self._closed = False
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        if self._root == self._root.parent or not self._root.name:
            raise PackageParityError(f"{self._label} root is invalid")
        parent = _open_absolute_directory(
            self._root.parent,
            label=f"{self._label} parent",
        )
        self._directories.append(parent)
        root = _open_directory_at(
            parent.descriptor,
            self._root.name,
            label=f"{self._label} root",
            path=self._root,
        )
        self._directories.append(root)
        self._scan(root, relative_path="")

    def _scan(
        self,
        directory: _StableSourceDirectory,
        *,
        relative_path: str,
    ) -> None:
        names = _directory_names(directory.descriptor, label=directory.label)
        directory.expected_names = names
        for name in names:
            child_relative = name if not relative_path else f"{relative_path}/{name}"
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise PackageParityError(
                    f"{self._label} namespace changed while it was listed"
                ) from exc
            if stat.S_ISDIR(observed.st_mode):
                child_directory = _open_directory_at(
                    directory.descriptor,
                    name,
                    label=f"{self._label} directory {child_relative}",
                    path=self._root / Path(*PurePosixPath(child_relative).parts),
                )
                self._directories.append(child_directory)
                _record_node(
                    self._nodes,
                    relative_path=child_relative,
                    is_directory=True,
                    label=self._label,
                )
                self._scan(child_directory, relative_path=child_relative)
            elif stat.S_ISREG(observed.st_mode):
                _enforce_count_limit(
                    self._file_count + 1,
                    limit=MAX_MEMBER_COUNT,
                    label=self._label,
                    noun="file count",
                )
                _enforce_bytes_limit(
                    observed.st_size,
                    limit=MAX_MEMBER_BYTES,
                    label=f"{self._label} file {child_relative}",
                )
                _enforce_total_bytes_limit(
                    self._payload_bytes + observed.st_size,
                    label=self._label,
                )
                opened = _open_source_file_at(
                    directory.descriptor,
                    name,
                    path=self._root / Path(*PurePosixPath(child_relative).parts),
                    label=f"{self._label} file {child_relative}",
                )
                identity = (opened.signature[0], opened.signature[1])
                prior = self._file_identities.get(identity)
                if prior is not None:
                    opened.close()
                    raise PackageParityError(
                        f"{self._label} contains hardlink-colliding paths: "
                        f"{prior!r} and {child_relative!r}"
                    )
                try:
                    _record_node(
                        self._nodes,
                        relative_path=child_relative,
                        is_directory=False,
                        label=self._label,
                    )
                except BaseException:
                    opened.close()
                    raise
                self._file_count += 1
                self._payload_bytes += opened.signature[3]
                self._file_identities[identity] = child_relative
                self._files[child_relative] = opened
            elif stat.S_ISLNK(observed.st_mode):
                raise PackageParityError(f"{self._label} contains a symlink: {child_relative}")
            else:
                raise PackageParityError(
                    f"{self._label} contains a non-regular node: {child_relative}"
                )

    def file_paths(
        self,
        *,
        prefix: str | None = None,
        exclude_ignored_package_files: bool = False,
    ) -> set[str]:
        paths: set[str] = set()
        for relative_path in self._files:
            output_path = relative_path if prefix is None else f"{prefix}/{relative_path}"
            if exclude_ignored_package_files and not _include_relative_path(output_path):
                continue
            paths.add(output_path)
        return paths

    def revalidate(self) -> None:
        for relative_path in sorted(self._files):
            self._files[relative_path].revalidate()
        for directory in reversed(self._directories):
            directory.revalidate()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for opened in self._files.values():
            try:
                opened.close()
            except OSError:
                continue
        self._files.clear()
        for directory in reversed(self._directories):
            try:
                directory.close()
            except OSError:
                continue
        self._directories.clear()


def collect_physical_source_superset(
    repo_root: Path,
    extra_paths: Sequence[str | Path] = (),
) -> PhysicalSourceSuperset:
    """Bind the stable physical package namespace plus explicit repository files."""

    snapshot = _StableSourceNamespace.from_repo_root(
        repo_root,
        extra_paths=extra_paths,
        label="physical source",
    )
    try:
        binding = snapshot.physical_binding()
        snapshot.revalidate()
        return binding
    finally:
        snapshot.close()


def _validated_physical_rows(
    physical_superset: Mapping[str, object],
) -> list[PhysicalSourceRow]:
    if set(physical_superset) != {"row_count", "rows", "sha256"}:
        raise PackageParityError("physical source superset fields are invalid")
    raw_rows = physical_superset.get("rows")
    row_count = physical_superset.get("row_count")
    binding_sha256 = physical_superset.get("sha256")
    if type(raw_rows) is not list:
        raise PackageParityError("physical source superset rows must be a list")
    if type(row_count) is not int or row_count != len(raw_rows):
        raise PackageParityError("physical source superset row_count mismatch")
    if (
        not isinstance(binding_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", binding_sha256, re.ASCII) is None
    ):
        raise PackageParityError("physical source superset sha256 is invalid")
    rows: list[PhysicalSourceRow] = []
    nodes: NodeKinds = {}
    for index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping) or set(value) != {
            "kind",
            "mode",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise PackageParityError(f"physical source superset rows[{index}] fields are invalid")
        path_value = value.get("path")
        kind = value.get("kind")
        mode = value.get("mode")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if not isinstance(path_value, str):
            raise PackageParityError(f"physical source superset rows[{index}].path is invalid")
        path = _validate_archive_path(
            path_value,
            label=f"physical source superset rows[{index}]",
        ).as_posix()
        if kind not in {"directory", "file"}:
            raise PackageParityError(f"physical source superset rows[{index}].kind is invalid")
        if type(mode) is not int or mode < 0 or mode > 0o7777:
            raise PackageParityError(f"physical source superset rows[{index}].mode is invalid")
        if kind == "directory":
            if sha256 is not None or size_bytes != 0:
                raise PackageParityError(
                    f"physical source superset rows[{index}] directory bytes are invalid"
                )
            if path != PACKAGE_NAME and not path.startswith(f"{PACKAGE_NAME}/"):
                raise PackageParityError(
                    "physical source superset contains an out-of-package directory"
                )
        else:
            if (
                not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256, re.ASCII) is None
                or type(size_bytes) is not int
                or size_bytes < 0
            ):
                raise PackageParityError(
                    f"physical source superset rows[{index}] file bytes are invalid"
                )
        _record_node(
            nodes,
            relative_path=path,
            is_directory=kind == "directory",
            label="physical source superset",
        )
        rows.append(
            {
                "path": path,
                "kind": kind,
                "mode": mode,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if rows != sorted(rows, key=lambda row: str(row["path"])):
        raise PackageParityError("physical source superset rows are not canonically sorted")
    if not any(row["path"] == PACKAGE_NAME and row["kind"] == "directory" for row in rows):
        raise PackageParityError("physical source superset package root is missing")
    if _sha256(_canonical_bytes(rows)) != binding_sha256:
        raise PackageParityError("physical source superset sha256 mismatch")
    return rows


def validate_hatch_namespace_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    physical_superset: Mapping[str, object],
) -> HatchNamespaceBinding:
    """Validate externally selected Hatch rows without importing Hatchling."""

    physical_rows = _validated_physical_rows(physical_superset)
    physical_files = {str(row["path"]): row for row in physical_rows if row["kind"] == "file"}
    canonical_rows: list[HatchNamespaceRow] = []
    distribution_nodes: dict[str, NodeKinds] = {"sdist": {}, "wheel": {}}
    selected_sources: dict[str, set[str]] = {"sdist": set(), "wheel": set()}
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping) or set(value) != {
            "distribution_path",
            "mode",
            "sha256",
            "size_bytes",
            "source_path",
            "target",
        }:
            raise PackageParityError(f"Hatch namespace rows[{index}] fields are invalid")
        target = value.get("target")
        source_value = value.get("source_path")
        distribution_value = value.get("distribution_path")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        mode = value.get("mode")
        if target not in {"sdist", "wheel"}:
            raise PackageParityError(f"Hatch namespace rows[{index}].target is invalid")
        if not isinstance(source_value, str) or not isinstance(distribution_value, str):
            raise PackageParityError(f"Hatch namespace rows[{index}] path is invalid")
        source_path = _validate_archive_path(
            source_value,
            label=f"Hatch namespace rows[{index}].source_path",
        ).as_posix()
        distribution_path = _validate_archive_path(
            distribution_value,
            label=f"Hatch namespace rows[{index}].distribution_path",
        ).as_posix()
        if source_path != distribution_path:
            raise PackageParityError(f"Hatch namespace rows[{index}] remaps its source path")
        physical = physical_files.get(source_path)
        if physical is None:
            raise PackageParityError(
                f"Hatch namespace rows[{index}] source is outside the physical superset"
            )
        if (
            not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256, re.ASCII) is None
            or type(size_bytes) is not int
            or size_bytes < 0
            or type(mode) is not int
            or mode < 0
            or mode > 0o7777
        ):
            raise PackageParityError(f"Hatch namespace rows[{index}] metadata is invalid")
        if (
            sha256 != physical["sha256"]
            or size_bytes != physical["size_bytes"]
            or mode != physical["mode"]
        ):
            raise PackageParityError(f"Hatch namespace rows[{index}] differs from physical source")
        _record_node(
            distribution_nodes[str(target)],
            relative_path=distribution_path,
            is_directory=False,
            label=f"Hatch {target} namespace",
        )
        if source_path in selected_sources[str(target)]:
            raise PackageParityError(f"Hatch {target} namespace selects a source more than once")
        selected_sources[str(target)].add(source_path)
        canonical_rows.append(
            {
                "target": target,
                "source_path": source_path,
                "distribution_path": distribution_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "mode": mode,
            }
        )
    expected_package_sources = {
        path
        for path in physical_files
        if path.startswith(f"{PACKAGE_NAME}/") and _include_relative_path(path)
    }
    expected_extra_sources = {
        path for path in physical_files if not path.startswith(f"{PACKAGE_NAME}/")
    }
    expected_sources = {
        "sdist": expected_package_sources | expected_extra_sources,
        "wheel": expected_package_sources,
    }
    for target in ("sdist", "wheel"):
        if selected_sources[target] != expected_sources[target]:
            missing = sorted(expected_sources[target] - selected_sources[target])
            extra = sorted(selected_sources[target] - expected_sources[target])
            raise PackageParityError(
                f"Hatch {target} namespace source coverage mismatch: "
                f"missing={missing[:20]!r} extra={extra[:20]!r}"
            )
    sorted_rows = sorted(
        canonical_rows,
        key=lambda row: (
            str(row["target"]),
            str(row["distribution_path"]),
            str(row["source_path"]),
        ),
    )
    if canonical_rows != sorted_rows:
        raise PackageParityError("Hatch namespace rows are not canonically sorted")
    wheel_projection: Inventory = {
        str(row["distribution_path"]): {
            "sha256": row["sha256"],
            "size": row["size_bytes"],
        }
        for row in canonical_rows
        if row["target"] == "wheel"
    }
    return {
        "rows": [dict(row) for row in canonical_rows],
        "row_count": len(canonical_rows),
        "sha256": _sha256(_canonical_bytes(canonical_rows)),
        "wheel_projection_sha256": _inventory_sha256(wheel_projection),
    }


def _read_regular_file_no_follow(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    opened = _open_stable_file_no_follow(path, label=label)
    try:
        opened.revalidate()
        return opened.raw, opened.opened_stat
    finally:
        opened.close()


class _InstalledRecordSnapshot:
    def __init__(
        self,
        *,
        dist_info_parent: Path,
        installed_environment_root: Path,
        namespace_snapshots: Sequence[_StableDirectoryTree] = (),
    ) -> None:
        self._dist_info_parent = dist_info_parent
        self._environment_root = installed_environment_root
        self._namespace_snapshots = tuple(namespace_snapshots)
        self._opened: dict[str, _OpenStableFile] = {}
        self._resolved_destinations: dict[Path, str] = {}
        self._opened_identities: dict[tuple[int, int], str] = {}
        self._payload_bytes = 0

    def read(self, relative_path: str) -> bytes:
        existing = self._opened.get(relative_path)
        if existing is not None:
            return existing.raw
        parsed = _validate_record_path(
            relative_path,
            label="installed",
            allow_leading_parents=True,
        )
        full_path = self._dist_info_parent / Path(*parsed.parts)
        try:
            lexical_stat = os.lstat(full_path)
        except OSError as exc:
            raise PackageParityError(
                f"installed RECORD path does not exist: {relative_path}"
            ) from exc
        if not stat.S_ISREG(lexical_stat.st_mode):
            raise PackageParityError(
                f"installed RECORD path is not a regular non-symlink file: {relative_path}"
            )
        _enforce_count_limit(
            len(self._opened) + 1,
            limit=MAX_RECORD_ROWS,
            label="installed RECORD materialization",
            noun="file count",
        )
        max_bytes = (
            MAX_RECORD_BYTES if relative_path.endswith(".dist-info/RECORD") else MAX_MEMBER_BYTES
        )
        _enforce_bytes_limit(
            lexical_stat.st_size,
            limit=max_bytes,
            label=f"installed RECORD file {relative_path}",
        )
        _enforce_total_bytes_limit(
            self._payload_bytes + lexical_stat.st_size,
            label="installed RECORD materialization",
        )
        resolved_path = _resolve_path(
            full_path,
            label=f"installed RECORD path {relative_path}",
        )
        lexically_normal_path = Path(os.path.abspath(full_path))
        if resolved_path != lexically_normal_path:
            raise PackageParityError(f"installed RECORD path traverses a symlink: {relative_path}")
        if not _is_relative_to(resolved_path, self._environment_root):
            raise PackageParityError(
                f"installed RECORD path escapes environment root: {relative_path}"
            )
        prior_path = self._resolved_destinations.get(resolved_path)
        if prior_path is not None:
            raise PackageParityError(
                "installed RECORD paths resolve to the same destination: "
                f"{prior_path!r} and {relative_path!r}"
            )
        opened = _open_stable_file_no_follow(
            full_path,
            label=f"installed RECORD file {relative_path}",
            before=lexical_stat,
            max_bytes=max_bytes,
        )
        identity = (opened.opened_stat.st_dev, opened.opened_stat.st_ino)
        prior_identity_path = self._opened_identities.get(identity)
        if prior_identity_path is not None:
            opened.close()
            raise PackageParityError(
                "installed RECORD paths identify the same file: "
                f"{prior_identity_path!r} and {relative_path!r}"
            )
        self._resolved_destinations[resolved_path] = relative_path
        self._opened_identities[identity] = relative_path
        self._opened[relative_path] = opened
        self._payload_bytes += opened.opened_stat.st_size
        return opened.raw

    def revalidate(self) -> None:
        for opened in self._opened.values():
            opened.revalidate()
        for snapshot in self._namespace_snapshots:
            snapshot.revalidate()

    def close(self) -> None:
        for opened in self._opened.values():
            try:
                opened.close()
            except OSError:
                continue
        self._opened.clear()
        for snapshot in self._namespace_snapshots:
            snapshot.close()


def _expected_script_record_paths(
    *,
    installed_environment_root: Path,
    dist_info_parent: Path,
    console_scripts: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    if os.name != "posix":
        raise PackageParityError("installed console-script verification requires POSIX")
    script_directory = installed_environment_root / "bin"
    try:
        script_directory_stat = os.lstat(script_directory)
    except OSError as exc:
        raise PackageParityError("installed script directory is missing") from exc
    if not stat.S_ISDIR(script_directory_stat.st_mode) or script_directory.is_symlink():
        raise PackageParityError("installed script directory must be a regular directory")
    expected: dict[str, tuple[str, str]] = {}
    for script_name, target in sorted(console_scripts.items()):
        path = script_directory / script_name
        try:
            script_stat = os.lstat(path)
        except OSError as exc:
            raise PackageParityError(f"installed generated script is missing: {path}") from exc
        if not stat.S_ISREG(script_stat.st_mode) or path.is_symlink():
            raise PackageParityError(f"installed generated script is not a regular file: {path}")
        resolved_path = _resolve_path(path, label=f"installed generated script {script_name}")
        if not _is_relative_to(resolved_path, installed_environment_root):
            raise PackageParityError(f"installed generated script escapes environment root: {path}")
        expected[_path_relative_to_base(path, base=dist_info_parent)] = (
            script_name,
            target,
        )
    return expected


def _pip_25_2_console_script_body(target: str) -> bytes:
    module, separator, callable_name = target.partition(":")
    qualified_name = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
    if (
        separator != ":"
        or qualified_name.fullmatch(module) is None
        or qualified_name.fullmatch(callable_name) is None
    ):
        raise PackageParityError(f"source console-script target is unsupported: {target!r}")
    import_name = callable_name.split(".", 1)[0]
    return (
        "import sys\n"
        f"from {module} import {import_name}\n"
        "if __name__ == '__main__':\n"
        "    if sys.argv[0].endswith('.exe'):\n"
        "        sys.argv[0] = sys.argv[0][:-4]\n"
        f"    sys.exit({callable_name}())\n"
    ).encode("utf-8")


def _verify_installed_script_wrappers(
    *,
    files: Mapping[str, bytes],
    expected_scripts: Mapping[str, tuple[str, str]],
    installed_environment_root: Path,
) -> None:
    interpreter = installed_environment_root / "bin" / "python"
    for record_path, (script_name, target) in sorted(expected_scripts.items()):
        raw = files[record_path]
        expected_raw = f"#!{interpreter.as_posix()}\n".encode(
            "utf-8"
        ) + _pip_25_2_console_script_body(target)
        if raw != expected_raw:
            raise PackageParityError(
                f"installed generated script wrapper differs from pip 25.2 contract: {script_name}"
            )


def _collect_installed_package_paths(installed_root: Path) -> set[str]:
    snapshot = _StableDirectoryTree(installed_root, label="installed package")
    try:
        files = snapshot.file_paths(
            prefix=installed_root.name,
            exclude_ignored_package_files=True,
        )
        if not files:
            raise PackageParityError("installed package payload is empty")
        snapshot.revalidate()
        return files
    finally:
        snapshot.close()


def _collect_installed_dist_info_paths(dist_info_path: Path) -> set[str]:
    snapshot = _StableDirectoryTree(
        dist_info_path,
        label="installed dist-info",
    )
    try:
        files = snapshot.file_paths()
        snapshot.revalidate()
        return files
    finally:
        snapshot.close()


def _dist_info_file_hashes(files: Mapping[str, bytes]) -> dict[str, str]:
    return {relative_path: _sha256(raw) for relative_path, raw in sorted(files.items())}


def _verify_installed_dist_info_files(
    *,
    installed_files: Mapping[str, bytes],
    wheel_files: Mapping[str, bytes],
) -> None:
    installed_immutable = {
        path: raw
        for path, raw in installed_files.items()
        if path not in MUTABLE_INSTALLED_DIST_INFO_FILES
    }
    wheel_immutable = {
        path: raw
        for path, raw in wheel_files.items()
        if path not in MUTABLE_INSTALLED_DIST_INFO_FILES
    }
    missing = sorted(set(wheel_immutable) - set(installed_immutable))
    extra = sorted(set(installed_immutable) - set(wheel_immutable))
    changed = sorted(
        path
        for path in set(wheel_immutable) & set(installed_immutable)
        if wheel_immutable[path] != installed_immutable[path]
    )
    if missing or extra or changed:
        raise PackageParityError(
            "installed wheel-owned dist-info differs from wheel: "
            f"missing={missing[:20]!r} extra={extra[:20]!r} changed={changed[:20]!r}"
        )


def _installed_direct_url_summary(
    dist_info_path: Path,
    *,
    wheel_path: Path,
    wheel_raw: bytes,
    direct_url_raw: bytes,
) -> dict[str, object]:
    try:
        payload = json.loads(direct_url_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageParityError("installed direct_url.json is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PackageParityError("installed direct_url.json is not an object")
    dir_info = payload.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        raise PackageParityError("installed direct_url.json marks the install as editable")
    parsed = urlparse(str(payload.get("url", "")))
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise PackageParityError("installed direct_url.json must point to a local wheel file")
    wheel = _resolve_path(wheel_path, label="wheel")
    url_path = _resolve_path(Path(unquote(parsed.path)), label="direct_url wheel path")
    if url_path != wheel:
        raise PackageParityError("installed direct_url.json does not point at the verified wheel")
    archive_info = payload.get("archive_info")
    if not isinstance(archive_info, dict):
        raise PackageParityError("installed direct_url.json is missing archive_info")
    expected_sha256 = _sha256(wheel_raw)
    observed_hash = archive_info.get("hash")
    observed_hashes = archive_info.get("hashes")
    candidates: set[str] = set()
    if isinstance(observed_hash, str):
        candidates.add(observed_hash)
    if isinstance(observed_hashes, dict) and isinstance(observed_hashes.get("sha256"), str):
        candidates.add("sha256=" + str(observed_hashes["sha256"]))
    if candidates != {"sha256=" + expected_sha256}:
        raise PackageParityError("installed direct_url.json wheel sha256 mismatch")
    return {
        "present": True,
        "editable": False,
        "url": wheel.as_uri(),
        "archive_info_sha256": expected_sha256,
        "sha256": _sha256(direct_url_raw),
    }


def _validate_installed_record(
    *,
    dist_info_path: Path,
    installed_environment_root: Path,
    installed_package_paths: set[str],
    installed_dist_info_paths: set[str],
    console_scripts: Mapping[str, str],
    snapshot: _InstalledRecordSnapshot,
) -> _InstalledRecordInspection:
    record_path = dist_info_path / "RECORD"
    record_relative = _path_relative_to_base(record_path, base=dist_info_path.parent)
    record_raw = snapshot.read(record_relative)
    rows = _parse_record_bytes(
        record_raw,
        label="installed",
        allow_leading_parents=True,
    )
    expected_script_paths = _expected_script_record_paths(
        installed_environment_root=installed_environment_root,
        dist_info_parent=dist_info_path.parent,
        console_scripts=console_scripts,
    )
    files = {record_relative: record_raw}
    for relative_path in rows:
        if relative_path != record_relative:
            files[relative_path] = snapshot.read(relative_path)
    _verify_record_hash_size(
        rows=rows,
        files=files,
        record_path=record_relative,
        label="installed",
    )
    dist_info_paths = {
        f"{dist_info_path.name}/{relative_path}" for relative_path in installed_dist_info_paths
    }
    expected_paths = installed_package_paths | dist_info_paths | set(expected_script_paths)
    observed_paths = set(rows)
    if not dist_info_paths.issubset(observed_paths):
        missing = sorted(dist_info_paths - observed_paths)
        raise PackageParityError(f"installed RECORD missing required dist-info rows: {missing!r}")
    if observed_paths != expected_paths:
        missing = sorted(expected_paths - observed_paths)
        extra = sorted(observed_paths - expected_paths)
        raise PackageParityError(
            f"installed RECORD paths mismatch: missing={missing[:20]!r} extra={extra[:20]!r}"
        )
    _verify_installed_script_wrappers(
        files=files,
        expected_scripts=expected_script_paths,
        installed_environment_root=installed_environment_root,
    )
    package_inventory = {
        relative_path: _hash_record(files[relative_path])
        for relative_path in sorted(installed_package_paths)
    }
    dist_info_files = {
        relative_path: files[f"{dist_info_path.name}/{relative_path}"]
        for relative_path in sorted(installed_dist_info_paths)
    }
    return _InstalledRecordInspection(
        summary={
            "file_count": len(rows),
            "package_file_count": len(installed_package_paths),
            "dist_info_file_count": len(dist_info_paths),
            "record_sha256": _sha256(record_raw),
        },
        package_inventory=package_inventory,
        dist_info_files=dist_info_files,
    )


def _verify_installed_dist_info(
    *,
    dist_info_path: Path,
    installed_root: Path,
    installed_environment_root: Path,
    expected_package_inventory: Mapping[str, Mapping[str, object]],
    wheel_path: Path,
    wheel_raw: bytes,
    expected_name: str,
    expected_version: str,
    wheel_inspection: _WheelInspection,
) -> tuple[dict[str, object], Inventory]:
    try:
        dist_info_stat = os.lstat(dist_info_path)
    except OSError as exc:
        raise PackageParityError("installed dist-info path is missing") from exc
    if not stat.S_ISDIR(dist_info_stat.st_mode) or dist_info_path.is_symlink():
        raise PackageParityError("installed dist-info path must be a .dist-info directory")
    dist_info = _resolve_path(dist_info_path, label="installed dist-info")
    environment_root = _resolve_path(installed_environment_root, label="installed environment root")
    if not dist_info.name.endswith(".dist-info"):
        raise PackageParityError("installed dist-info path must be a .dist-info directory")
    expected_dist_info = _expected_dist_info_root(
        expected_name=expected_name,
        expected_version=expected_version,
    )
    if dist_info.name != expected_dist_info:
        raise PackageParityError(f"installed dist-info directory name mismatch: {dist_info.name!r}")
    if dist_info.parent != installed_root.parent:
        raise PackageParityError(
            "installed dist-info must share the installed package site-packages parent"
        )
    if not _is_relative_to(dist_info, environment_root) or not _is_relative_to(
        installed_root, environment_root
    ):
        raise PackageParityError("installed package and dist-info must be inside environment root")
    package_namespace = _StableDirectoryTree(
        installed_root,
        label="installed package payload differs from source",
    )
    try:
        dist_info_namespace = _StableDirectoryTree(
            dist_info,
            label="installed wheel-owned dist-info differs from wheel",
        )
    except BaseException:
        package_namespace.close()
        raise
    snapshot = _InstalledRecordSnapshot(
        dist_info_parent=dist_info.parent,
        installed_environment_root=environment_root,
        namespace_snapshots=(package_namespace, dist_info_namespace),
    )
    try:
        installed_package_paths = _collect_installed_package_paths(installed_root)
        namespace_package_paths = package_namespace.file_paths(
            prefix=installed_root.name,
            exclude_ignored_package_files=True,
        )
        if installed_package_paths != namespace_package_paths:
            changed = sorted(installed_package_paths ^ namespace_package_paths)
            raise PackageParityError(
                "installed package payload differs from source: "
                f"namespace changed={changed[:20]!r}"
            )
        expected_package_paths = set(expected_package_inventory)
        if installed_package_paths != expected_package_paths:
            changed = sorted(installed_package_paths ^ expected_package_paths)
            raise PackageParityError(
                f"installed package payload differs from source: {changed[:20]!r}"
            )
        installed_dist_info_paths = _collect_installed_dist_info_paths(dist_info)
        namespace_dist_info_paths = dist_info_namespace.file_paths()
        if installed_dist_info_paths != namespace_dist_info_paths:
            changed = sorted(installed_dist_info_paths ^ namespace_dist_info_paths)
            raise PackageParityError(
                "installed wheel-owned dist-info differs from wheel: "
                f"namespace changed={changed[:20]!r}"
            )
        record_inspection = _validate_installed_record(
            dist_info_path=dist_info,
            installed_environment_root=environment_root,
            installed_package_paths=installed_package_paths,
            installed_dist_info_paths=installed_dist_info_paths,
            console_scripts=wheel_inspection.console_scripts,
            snapshot=snapshot,
        )
        installed_inventory = record_inspection.package_inventory
        if installed_inventory != expected_package_inventory:
            changed = sorted(
                path
                for path in set(expected_package_inventory) | set(installed_inventory)
                if expected_package_inventory.get(path) != installed_inventory.get(path)
            )
            raise PackageParityError(
                f"installed package payload differs from source: {changed[:20]!r}"
            )
        installed_dist_info_files = record_inspection.dist_info_files
        metadata_raw = installed_dist_info_files.get("METADATA")
        if metadata_raw is None:
            raise PackageParityError("installed dist-info METADATA is missing")
        metadata = _metadata_summary_from_bytes(
            metadata_raw,
            expected_name=expected_name,
            expected_version=expected_version,
            label="installed",
        )
        if "WHEEL" not in installed_dist_info_files:
            raise PackageParityError("installed dist-info WHEEL is missing")
        _verify_installed_dist_info_files(
            installed_files=installed_dist_info_files,
            wheel_files=wheel_inspection.dist_info_files,
        )
        direct_url_raw = installed_dist_info_files.get("direct_url.json")
        if direct_url_raw is None:
            raise PackageParityError("installed direct_url.json is required")
        direct_url = _installed_direct_url_summary(
            dist_info,
            wheel_path=wheel_path,
            wheel_raw=wheel_raw,
            direct_url_raw=direct_url_raw,
        )
        provenance = {
            "metadata": metadata,
            "record": record_inspection.summary,
            "direct_url": direct_url,
            "dist_info_file_sha256s": _dist_info_file_hashes(installed_dist_info_files),
            "non_editable_verified": True,
            "environment_root": environment_root.as_posix(),
            "site_packages_root": dist_info.parent.as_posix(),
            "installed_package_root": installed_root.as_posix(),
            "dist_info_path": dist_info.as_posix(),
        }
        snapshot.revalidate()
        return provenance, installed_inventory
    finally:
        snapshot.close()


def verify_installed_dist_info(
    *,
    dist_info_path: Path,
    source_root: Path,
    installed_root: Path,
    installed_environment_root: Path,
    installed_inventory: Mapping[str, Mapping[str, object]],
    wheel_path: Path,
    expected_name: str,
    expected_version: str,
) -> dict[str, object]:
    normalized_source_root = _normalize_package_root(source_root, label="source")
    normalized_installed_root = _normalize_package_root(installed_root, label="installed")
    source_snapshot = _StableSourceNamespace.from_repo_root(
        normalized_source_root.parent,
        extra_paths=("pyproject.toml",),
        label="source",
    )
    try:
        project = _source_project_from_pyproject_bytes(source_snapshot.file_bytes("pyproject.toml"))
        if expected_name != project.name or expected_version != project.version:
            raise PackageParityError(
                "installed provenance identity does not match source pyproject"
            )
        source_inventory = source_snapshot.package_inventory()
        opened_wheel = _open_stable_file_no_follow(wheel_path, label="wheel")
        try:
            wheel_inspection = _inspect_wheel_provenance_from_bytes(
                wheel_raw=opened_wheel.raw,
                expected_name=expected_name,
                expected_version=expected_version,
                expected_scripts=project.scripts,
            )
            provenance, _ = _verify_installed_dist_info(
                dist_info_path=dist_info_path,
                installed_root=normalized_installed_root,
                installed_environment_root=installed_environment_root,
                expected_package_inventory=source_inventory,
                wheel_path=wheel_path,
                wheel_raw=opened_wheel.raw,
                expected_name=expected_name,
                expected_version=expected_version,
                wheel_inspection=wheel_inspection,
            )
            opened_wheel.revalidate()
            source_snapshot.revalidate()
            return provenance
        finally:
            opened_wheel.close()
    finally:
        source_snapshot.close()


def verify_package_payload_parity(
    *,
    source_package_root: Path,
    sdist_path: Path,
    wheel_path: Path,
    installed_package_root: Path,
    installed_dist_info_path: Path,
    installed_environment_root: Path,
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> dict[str, object]:
    """Require identical normalized paths and bytes across all four surfaces."""

    source_root = _normalize_package_root(source_package_root, label="source")
    installed_root = _normalize_package_root(installed_package_root, label="installed")
    if source_root == installed_root:
        raise PackageParityError("installed package root resolves to the source checkout")
    source_snapshot = _StableSourceNamespace.from_repo_root(
        source_root.parent,
        extra_paths=("pyproject.toml",),
        label="source",
    )
    try:
        project = _source_project_from_pyproject_bytes(source_snapshot.file_bytes("pyproject.toml"))
        if expected_name is not None and expected_name != project.name:
            raise PackageParityError(
                "explicit expected distribution name does not match source pyproject"
            )
        if expected_version is not None and expected_version != project.version:
            raise PackageParityError(
                "explicit expected distribution version does not match source pyproject"
            )
        expected_distribution_name = expected_name or project.name
        expected_distribution_version = expected_version or project.version
        source = source_snapshot.package_inventory()
        opened_sdist = _open_stable_file_no_follow(sdist_path, label="sdist")
        try:
            opened_wheel = _open_stable_file_no_follow(wheel_path, label="wheel")
            try:
                wheel_inspection = _inspect_wheel_provenance_from_bytes(
                    wheel_raw=opened_wheel.raw,
                    expected_name=expected_distribution_name,
                    expected_version=expected_distribution_version,
                    expected_scripts=project.scripts,
                )
                surfaces = {
                    "sdist": _collect_sdist_payload_from_bytes(
                        opened_sdist.raw,
                        expected_name=expected_distribution_name,
                        expected_version=expected_distribution_version,
                        expected_pyproject_raw=project.pyproject_raw,
                    ),
                    "wheel": wheel_inspection.package_inventory,
                }
                for label, observed in surfaces.items():
                    if observed != source:
                        changed = sorted(
                            path
                            for path in set(source) | set(observed)
                            if source.get(path) != observed.get(path)
                        )
                        raise PackageParityError(
                            f"{label} package payload differs from source: {changed[:20]!r}"
                        )
                installed_provenance, _ = _verify_installed_dist_info(
                    dist_info_path=installed_dist_info_path,
                    installed_root=installed_root,
                    installed_environment_root=installed_environment_root,
                    expected_package_inventory=source,
                    wheel_path=wheel_path,
                    wheel_raw=opened_wheel.raw,
                    expected_name=expected_distribution_name,
                    expected_version=expected_distribution_version,
                    wheel_inspection=wheel_inspection,
                )
                inventory_hash = _inventory_sha256(source)
                result = {
                    "package_file_count": len(source),
                    "package_inventory_sha256": inventory_hash,
                    "package_inventory": {
                        "file_count": len(source),
                        "sha256": inventory_hash,
                    },
                    "installed_provenance": installed_provenance,
                    "sdist_sha256": _sha256(opened_sdist.raw),
                    "source_equals_sdist_equals_wheel_equals_installed": True,
                    "wheel_provenance": wheel_inspection.provenance,
                    "wheel_sha256": _sha256(opened_wheel.raw),
                }
                opened_sdist.revalidate()
                opened_wheel.revalidate()
                source_snapshot.revalidate()
                return result
            finally:
                opened_wheel.close()
        finally:
            opened_sdist.close()
    finally:
        source_snapshot.close()


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PackageParityError(f"argument parsing failed: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-package-root", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--installed-package-root", type=Path, required=True)
    parser.add_argument("--installed-dist-info", type=Path, required=True)
    parser.add_argument("--installed-environment-root", type=Path, required=True)
    parser.add_argument("--expected-name")
    parser.add_argument("--expected-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = verify_package_payload_parity(
            source_package_root=args.source_package_root,
            sdist_path=args.sdist,
            wheel_path=args.wheel,
            installed_package_root=args.installed_package_root,
            installed_dist_info_path=args.installed_dist_info,
            installed_environment_root=args.installed_environment_root,
            expected_name=args.expected_name,
            expected_version=args.expected_version,
        )
    except (
        PackageParityError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        binascii.Error,
        ValueError,
    ) as exc:
        print(_canonical_json({"accepted": False, "error": str(exc)}))
        return PackageParityError.exit_code
    print(_canonical_json({"accepted": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PackageParityError",
    "collect_directory_payload",
    "collect_physical_source_superset",
    "collect_sdist_payload",
    "collect_wheel_payload",
    "inspect_wheel_provenance",
    "main",
    "validate_hatch_namespace_rows",
    "verify_installed_dist_info",
    "verify_package_payload_parity",
]
