# flake8: noqa: E501
"""Crash-safe promotion for mixed Fundamental successor generations.

This module deliberately owns only the transaction boundary.  The successor
builder remains responsible for provider, derivation, readiness, provenance,
and chain validation.  A promotion captures that already-validated staging
generation, rechecks exact file identities, and publishes it behind the fixed
market -> PIT -> Fundamental lock order.

Large Parquet files are hashed and copied in chunks.  They are never retained
as ``bytes`` by this module.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


FUNDAMENTAL_POINTER_FILENAME = "_fundamental_latest.json"
FUNDAMENTAL_GENERATIONS_DIRNAME = "_fundamental_generations"
FUNDAMENTAL_PROMOTION_LOCK_FILENAME = ".fundamental-promotion.lock"
MARKET_WRITER_LOCK_FILENAME = ".market_writer.lock"
PIT_WRITER_LOCK_FILENAME = ".pit_writer.lock"
FUNDAMENTAL_TABLES = (
    "fundamental_period",
    "fundamental_daily",
    "fundamental_quarantine",
)

SUCCESSOR_PROVENANCE_SCHEMA = "cn-fundamental-primary-provenance.v3"
SUCCESSOR_PROVENANCE_STATUS = "verified_safe_successor_mixed"
SUCCESSOR_PROMOTION_JOURNAL_SCHEMA = (
    "cn-fundamental-successor-promotion-journal.v2"
)
SUCCESSOR_PROVIDER_FILESET_SCHEMA = (
    "cn-fundamental-successor-provider-fileset.v1"
)
SUCCESSOR_PROMOTION_PLAN_SCHEMA = "cn-fundamental-successor-promotion-plan.v1"

DEFAULT_LOCK_TIMEOUT_SECONDS = 120.0
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_PREDECESSOR_MANIFEST_JSON_BYTES = 128 * 1024 * 1024
_MAX_CHAIN_DEPTH = 4096
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_JOURNAL_PHASES = (
    "INTENT",
    "PRECAS_VALIDATED",
    "INSTALL_INTENT",
    "GENERATION_INSTALLED",
    "CAS_COMMITTED",
    "POSTCHECK_PASSED",
    "ROLLBACK_COMMITTED",
    "TERMINAL",
)


class SuccessorPromotionError(ValueError):
    """Raised when a successor promotion cannot be proven safe."""

    def __init__(
        self,
        message: str,
        *,
        status: str = "BLOCKED",
        journal_run_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.journal_run_id = journal_run_id


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    sha256: str
    size: int
    signature: tuple[int, ...]


@dataclass(frozen=True)
class _SuccessorCapture:
    staging_root: Path
    generation_root: Path
    generation_id: str
    pointer_path: Path
    pointer_bytes: bytes
    pointer_sha256: str
    pointer: dict[str, Any]
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    manifest: dict[str, Any]
    table_files: dict[str, _FileIdentity]
    provider_files: dict[str, _FileIdentity]
    provider_fileset_sha256: str
    predecessor: dict[str, Any]
    market_binding: dict[str, Any]
    pit_binding: dict[str, Any]
    original_seam: str
    immediate_parent_cutoff: str
    target_cutoff: str
    provenance_binding_sha256: str
    successor_chain: dict[str, Any]
    validator_receipt_sha256: str


StagingValidator = Callable[[Path], Any]
ChainValidator = Callable[[Mapping[str, Any]], Any]
LiveBindingValidator = Callable[[Mapping[str, Any], str], Any]
FaultInjector = Callable[[str], None]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256.fullmatch(text):
        raise SuccessorPromotionError("successor SHA256 is invalid")
    return text


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_ID.fullmatch(text) or text in {".", ".."}:
        raise SuccessorPromotionError(f"{label} is unsafe")
    return text


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SuccessorPromotionError(
            "successor promotion payload is not canonical JSON"
        ) from exc


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _json_object(
    payload: bytes,
    *,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any]:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise SuccessorPromotionError(f"{label} JSON limit is invalid")
    if len(payload) > maximum_bytes:
        raise SuccessorPromotionError(f"{label} JSON is unreasonably large")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise SuccessorPromotionError(f"{label} JSON is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise SuccessorPromotionError(f"{label} JSON object is required")
    return dict(parsed)


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _lexical_absolute(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise SuccessorPromotionError(f"{label} contains parent traversal")
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _secure_existing_directory(path: str | Path, *, label: str) -> Path:
    absolute = _lexical_absolute(path, label=label)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorPromotionError(f"{label} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SuccessorPromotionError(f"{label} contains an unsafe ancestor")
    return absolute


def _secure_directory_tree(
    path: str | Path,
    *,
    label: str,
    create: bool,
    private: bool = False,
) -> Path:
    absolute = _lexical_absolute(path, label=label)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            if not create:
                raise SuccessorPromotionError(f"{label} is missing")
            try:
                os.mkdir(cursor, 0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorPromotionError(f"{label} is inaccessible") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SuccessorPromotionError(f"{label} contains an unsafe ancestor")
    if private:
        mode = stat.S_IMODE(os.lstat(absolute).st_mode)
        if mode != 0o700:
            raise SuccessorPromotionError(f"{label} mode must be 0700")
    return absolute


def _safe_regular_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorPromotionError(f"{label} is unreadable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise SuccessorPromotionError(f"{label} is not a private regular file")
    return metadata


def _stable_file_hash(path: Path, *, label: str) -> _FileIdentity:
    before = _safe_regular_metadata(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorPromotionError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        expected_signature = _signature(before)
        if _signature(opened) != expected_signature:
            raise SuccessorPromotionError(f"{label} changed during open")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            _signature(after) != expected_signature
            or _signature(current) != expected_signature
            or size != before.st_size
        ):
            raise SuccessorPromotionError(f"{label} changed during hashing")
        return _FileIdentity(
            path=path,
            sha256=digest.hexdigest(),
            size=size,
            signature=expected_signature,
        )
    except OSError as exc:
        raise SuccessorPromotionError(f"{label} changed during hashing") from exc
    finally:
        os.close(descriptor)


def _stable_small_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> tuple[bytes, _FileIdentity]:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise SuccessorPromotionError(f"{label} size limit is invalid")
    identity = _stable_file_hash(path, label=label)
    if identity.size > maximum_bytes:
        raise SuccessorPromotionError(f"{label} is unreasonably large")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    descriptor = os.open(path, flags)
    try:
        if _signature(os.fstat(descriptor)) != identity.signature:
            raise SuccessorPromotionError(f"{label} changed before readback")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, min(_COPY_CHUNK_SIZE, maximum_bytes))
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        if (
            _signature(os.fstat(descriptor)) != identity.signature
            or _signature(os.lstat(path)) != identity.signature
            or len(payload) != identity.size
            or _sha256_bytes(payload) != identity.sha256
        ):
            raise SuccessorPromotionError(f"{label} changed during readback")
        return payload, identity
    finally:
        os.close(descriptor)


def _resolve_inside(root: Path, value: Any, *, label: str) -> Path:
    raw = str(value or "").strip()
    candidate = Path(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise SuccessorPromotionError(f"{label} is not a safe relative path")
    cursor = root
    for part in candidate.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorPromotionError(f"{label} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SuccessorPromotionError(f"{label} contains a symlink")
    if root not in cursor.parents and cursor != root:
        raise SuccessorPromotionError(f"{label} escapes its root")
    return cursor


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise SuccessorPromotionError(f"{label} mapping is required")
    return dict(value)


def _nested_mapping(
    sources: Sequence[Mapping[str, Any]],
    *names: str,
) -> dict[str, Any]:
    for source in sources:
        for name in names:
            value = source.get(name)
            if is_dataclass(value) and not isinstance(value, type):
                value = asdict(value)
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _first_value(sources: Sequence[Mapping[str, Any]], *names: str) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] not in (None, ""):
                return source[name]
    return None


def _date(value: Any, *, label: str) -> str:
    text = str(value or "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise SuccessorPromotionError(f"{label} must be YYYYMMDD")
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise SuccessorPromotionError(f"{label} is invalid") from exc
    return text


def _decode_pointer_evidence(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> tuple[bytes, str, dict[str, Any]]:
    encoded = str(
        binding.get("exact_pointer_bytes_b64")
        or binding.get("pointer_bytes_b64")
        or ""
    ).strip()
    if not encoded:
        raise SuccessorPromotionError(f"{label} exact pointer bytes are missing")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise SuccessorPromotionError(f"{label} pointer base64 is invalid") from exc
    declared_sha256 = _valid_sha256(binding.get("pointer_sha256"))
    if _sha256_bytes(payload) != declared_sha256:
        raise SuccessorPromotionError(f"{label} pointer SHA256 mismatch")
    parsed = _json_object(payload, label=f"{label} pointer")
    return payload, declared_sha256, parsed


def _normalize_ref_items(value: Any, *, label: str) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    items: list[Any]
    if isinstance(value, Mapping):
        items = []
        for key, raw in value.items():
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("path", str(key))
            else:
                item = {"path": str(key), "sha256": raw}
            items.append(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        raise SuccessorPromotionError(f"{label} immutable refs are invalid")
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in items:
        item = _as_mapping(raw, label=f"{label} immutable ref")
        path = str(item.get("path") or "").strip()
        if not path or path in paths:
            raise SuccessorPromotionError(f"{label} immutable ref path is invalid")
        paths.add(path)
        normalized.append(
            {
                "path": path,
                "sha256": _valid_sha256(item.get("sha256")),
                **(
                    {"size": int(item["size"])}
                    if item.get("size") is not None
                    else {}
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def _validate_embedded_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> tuple[bytes, str, dict[str, Any]]:
    payload, pointer_sha256, pointer = _decode_pointer_evidence(
        binding,
        label=label,
    )
    refs = _normalize_ref_items(binding.get("immutable_refs"), label=label)
    if not refs:
        raise SuccessorPromotionError(f"{label} immutable refs are missing")
    for ref in refs:
        ref_path = _lexical_absolute(ref["path"], label=f"{label} immutable ref")
        identity = _stable_file_hash(ref_path, label=f"{label} immutable ref")
        if identity.sha256 != ref["sha256"]:
            raise SuccessorPromotionError(f"{label} immutable ref SHA256 mismatch")
        if "size" in ref and identity.size != int(ref["size"]):
            raise SuccessorPromotionError(f"{label} immutable ref size mismatch")
    return payload, pointer_sha256, pointer


def _binding_live_pointer_path(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    value = binding.get("live_pointer_path") or binding.get("pointer_path")
    if not value:
        raise SuccessorPromotionError(f"{label} live pointer path is missing")
    path = _lexical_absolute(str(value), label=f"{label} live pointer")
    _secure_existing_directory(path.parent, label=f"{label} pointer root")
    return path


def _validate_live_binding(binding: Mapping[str, Any], *, label: str) -> Path:
    expected_bytes, expected_sha256, _pointer = _validate_embedded_binding(
        binding,
        label=label,
    )
    pointer_path = _binding_live_pointer_path(binding, label=label)
    current_bytes, identity = _stable_small_bytes(
        pointer_path,
        label=f"live {label} pointer",
    )
    if current_bytes != expected_bytes or identity.sha256 != expected_sha256:
        raise SuccessorPromotionError(f"live {label} pointer CAS mismatch")
    return pointer_path


def _provider_fileset_payload(
    files: Mapping[str, _FileIdentity],
) -> dict[str, Any]:
    return {
        "schema_version": SUCCESSOR_PROVIDER_FILESET_SCHEMA,
        "files": {
            relative: {
                "sha256": identity.sha256,
                "size": identity.size,
            }
            for relative, identity in sorted(files.items())
        },
    }


def _scan_provider_files(generation_root: Path) -> dict[str, _FileIdentity]:
    evidence_root = generation_root / "provider_evidence"
    try:
        root_metadata = os.lstat(evidence_root)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SuccessorPromotionError("provider evidence root is unreadable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise SuccessorPromotionError("provider evidence root is unsafe")
    files: dict[str, _FileIdentity] = {}
    for directory, directory_names, file_names in os.walk(
        evidence_root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in list(directory_names):
            metadata = os.lstat(directory_path / name)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise SuccessorPromotionError("provider evidence directory is unsafe")
        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(generation_root).as_posix()
            if relative in files:
                raise SuccessorPromotionError("provider evidence path is duplicated")
            files[relative] = _stable_file_hash(
                path,
                label=f"provider evidence {relative}",
            )
    return dict(sorted(files.items()))


def _normalize_declared_provider_files(
    value: Any,
    *,
    generation_root: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_value, Mapping):
            item = dict(raw_value)
            raw_path = item.get("path") or raw_key
            raw_sha = item.get("sha256")
            raw_size = item.get("size")
        else:
            raw_path = raw_key
            raw_sha = raw_value
            raw_size = None
        path = Path(str(raw_path))
        if path.is_absolute():
            try:
                relative = path.relative_to(generation_root).as_posix()
            except ValueError as exc:
                raise SuccessorPromotionError(
                    "declared provider evidence path escapes generation"
                ) from exc
        else:
            if ".." in path.parts:
                raise SuccessorPromotionError(
                    "declared provider evidence path is unsafe"
                )
            if not path.parts or path.parts[0] != "provider_evidence":
                path = Path("provider_evidence") / path
            relative = path.as_posix()
        if relative in result:
            raise SuccessorPromotionError(
                "declared provider evidence path is duplicated"
            )
        result[relative] = {
            "sha256": _valid_sha256(raw_sha),
            **({"size": int(raw_size)} if raw_size is not None else {}),
        }
    return dict(sorted(result.items()))


def _default_staging_validator(
    root: Path,
    *,
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    generation_root: Path,
    historical_only: bool,
) -> Any:
    import_errors: list[Exception] = []
    for module_name in (
        "quant_investor.market.fundamental_incremental",
        "quant_investor.market.fundamental_successor",
        "quant_investor.market.fundamental_successor_derivation",
    ):
        try:
            module = __import__(
                module_name,
                fromlist=["validate_successor_provenance"],
            )
            validator = getattr(module, "validate_successor_provenance")
        except (ImportError, AttributeError) as exc:
            import_errors.append(exc)
            continue
        return validator(
            dict(pointer),
            dict(manifest),
            generation_root=root,
            historical_only=historical_only,
        )
    raise SuccessorPromotionError(
        "successor staging validator is unavailable"
    ) from (import_errors[-1] if import_errors else None)


def _scan_generation(root: str | Path) -> dict[str, Any]:
    staging_root = _secure_existing_directory(root, label="successor staging root")
    pointer_path = staging_root / FUNDAMENTAL_POINTER_FILENAME
    pointer_bytes, pointer_identity = _stable_small_bytes(
        pointer_path,
        label="successor staging pointer",
    )
    pointer = _json_object(pointer_bytes, label="successor staging pointer")
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
    ):
        raise SuccessorPromotionError("successor pointer shape is invalid")
    generation_id = _safe_id(pointer.get("generation_id"), label="generation_id")
    expected_generation_relative = (
        Path(FUNDAMENTAL_GENERATIONS_DIRNAME) / generation_id
    )
    if str(pointer.get("manifest_path") or "") != (
        expected_generation_relative / "manifest.json"
    ).as_posix():
        raise SuccessorPromotionError("successor manifest path is noncanonical")
    manifest_path = _resolve_inside(
        staging_root,
        pointer.get("manifest_path"),
        label="successor manifest",
    )
    manifest_bytes, manifest_identity = _stable_small_bytes(
        manifest_path,
        label="successor manifest",
    )
    manifest = _json_object(manifest_bytes, label="successor manifest")
    if (
        manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or str(manifest.get("generation_id") or "") != generation_id
    ):
        raise SuccessorPromotionError("successor manifest shape is invalid")
    pointer_tables = _as_mapping(pointer.get("tables"), label="pointer tables")
    manifest_tables = _as_mapping(manifest.get("tables"), label="manifest tables")
    if set(pointer_tables) != set(FUNDAMENTAL_TABLES) or set(manifest_tables) != set(
        FUNDAMENTAL_TABLES
    ):
        raise SuccessorPromotionError("successor table set is invalid")
    table_files: dict[str, _FileIdentity] = {}
    for table_name in FUNDAMENTAL_TABLES:
        if str(pointer_tables[table_name]) != (
            expected_generation_relative / f"{table_name}.parquet"
        ).as_posix():
            raise SuccessorPromotionError(
                f"successor {table_name} path is noncanonical"
            )
        path = _resolve_inside(
            staging_root,
            pointer_tables[table_name],
            label=f"successor {table_name}",
        )
        identity = _stable_file_hash(path, label=f"successor {table_name}")
        contract = _as_mapping(
            manifest_tables[table_name],
            label=f"manifest {table_name}",
        )
        if identity.sha256 != _valid_sha256(contract.get("sha256")):
            raise SuccessorPromotionError(
                f"successor {table_name} manifest SHA256 mismatch"
            )
        if contract.get("bytes") is not None and int(contract["bytes"]) != identity.size:
            raise SuccessorPromotionError(
                f"successor {table_name} manifest size mismatch"
            )
        table_files[table_name] = identity
    generation_root = manifest_path.parent
    provider_files = _scan_provider_files(generation_root)
    return {
        "staging_root": staging_root,
        "generation_root": generation_root,
        "generation_id": generation_id,
        "pointer_path": pointer_path,
        "pointer_bytes": pointer_bytes,
        "pointer_sha256": pointer_identity.sha256,
        "pointer": pointer,
        "manifest_path": manifest_path,
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": manifest_identity.sha256,
        "manifest": manifest,
        "table_files": table_files,
        "provider_files": provider_files,
        "provider_fileset_sha256": _canonical_sha256(
            _provider_fileset_payload(provider_files)
        ),
    }


def _absolute_reference_path(value: Any, *, label: str) -> Path:
    path = _lexical_absolute(str(value or ""), label=label)
    _secure_existing_directory(path.parent, label=f"{label} parent")
    return path


def _pointer_relative_reference(
    pointer_root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    candidate = Path(str(value or "").strip())
    if not candidate.parts or ".." in candidate.parts:
        raise SuccessorPromotionError(f"{label} path is unsafe")
    path = candidate if candidate.is_absolute() else pointer_root / candidate
    path = _lexical_absolute(path, label=label)
    _secure_existing_directory(path.parent, label=f"{label} parent")
    return path


def _source_capture_entry(
    captured: Mapping[str, Any],
    name: str,
) -> tuple[bytes, str]:
    entry = _as_mapping(
        captured.get(name),
        label=f"source captured {name} pointer",
    )
    encoded = str(entry.get("bytes_base64") or "").strip()
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise SuccessorPromotionError(
            f"source captured {name} pointer base64 is invalid"
        ) from exc
    sha256 = _valid_sha256(entry.get("sha256"))
    if (
        _sha256_bytes(payload) != sha256
        or int(entry.get("byte_length", -1)) != len(payload)
    ):
        raise SuccessorPromotionError(
            f"source captured {name} pointer identity mismatch"
        )
    _json_object(payload, label=f"source captured {name} pointer")
    return payload, sha256


def _durable_bindings_from_support_manifest(
    *,
    generation_root: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    permanent_refs = _as_mapping(
        provenance.get("permanent_support_refs"),
        label="permanent support refs",
    )
    support_ref = _as_mapping(
        permanent_refs.get("support_manifest"),
        label="support manifest ref",
    )
    relative = Path(str(support_ref.get("path") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SuccessorPromotionError("support manifest ref path is unsafe")
    support_path = generation_root / "provider_evidence" / relative
    support_bytes, support_identity = _stable_small_bytes(
        support_path,
        label="sealed support manifest",
    )
    if support_identity.sha256 != _valid_sha256(support_ref.get("sha256")):
        raise SuccessorPromotionError("sealed support manifest SHA256 mismatch")
    support_manifest = _json_object(
        support_bytes,
        label="sealed support manifest",
    )
    captured = _as_mapping(
        support_manifest.get("captured_pointers"),
        label="source captured pointers",
    )
    if set(captured) != {"market", "pit", "predecessor"}:
        raise SuccessorPromotionError("source captured pointer set is invalid")
    immutable = _as_mapping(
        support_manifest.get("immutable_refs"),
        label="source immutable refs",
    )
    live_paths = _as_mapping(
        immutable.get("live_pointer_paths"),
        label="source live pointer paths",
    )
    if set(live_paths) != {"market", "pit", "predecessor"}:
        raise SuccessorPromotionError("source live pointer path set is invalid")

    predecessor_bytes, predecessor_sha = _source_capture_entry(
        captured,
        "predecessor",
    )
    predecessor_pointer = _json_object(
        predecessor_bytes,
        label="captured predecessor pointer",
    )
    predecessor_live_path = _absolute_reference_path(
        live_paths["predecessor"],
        label="predecessor live pointer",
    )
    predecessor_root = predecessor_live_path.parent
    predecessor_info = _as_mapping(
        immutable.get("predecessor"),
        label="source predecessor immutable refs",
    )
    predecessor_manifest = _absolute_reference_path(
        predecessor_info.get("manifest_path"),
        label="predecessor manifest",
    )
    predecessor_refs = [
        {
            "path": str(predecessor_manifest),
            "sha256": _valid_sha256(predecessor_info.get("manifest_sha256")),
        }
    ]
    pointer_tables = _as_mapping(
        predecessor_pointer.get("tables"),
        label="captured predecessor tables",
    )
    table_sha = _as_mapping(
        predecessor_info.get("table_sha256"),
        label="predecessor table SHA256",
    )
    if set(pointer_tables) != set(FUNDAMENTAL_TABLES) or set(table_sha) != set(
        FUNDAMENTAL_TABLES
    ):
        raise SuccessorPromotionError("predecessor immutable table set is invalid")
    for table_name in FUNDAMENTAL_TABLES:
        predecessor_refs.append(
            {
                "path": str(
                    _pointer_relative_reference(
                        predecessor_root,
                        pointer_tables[table_name],
                        label=f"predecessor {table_name}",
                    )
                ),
                "sha256": _valid_sha256(table_sha[table_name]),
            }
        )
    predecessor_declared = _as_mapping(
        provenance.get("predecessor"),
        label="successor predecessor",
    )
    predecessor = {
        **predecessor_declared,
        "live_pointer_path": str(predecessor_live_path),
        "pointer_sha256": predecessor_sha,
        "exact_pointer_bytes_b64": base64.b64encode(
            predecessor_bytes
        ).decode("ascii"),
        "immutable_refs": predecessor_refs,
    }

    market_bytes, market_sha = _source_capture_entry(captured, "market")
    market_info = _as_mapping(
        immutable.get("market"),
        label="source market immutable refs",
    )
    market_refs = [
        {
            "path": str(
                _absolute_reference_path(
                    market_info.get("manifest_path"),
                    label="market immutable manifest",
                )
            ),
            "sha256": _valid_sha256(market_info.get("manifest_sha256")),
        }
    ]
    scope_info = _as_mapping(
        immutable.get("scope"),
        label="source scope immutable ref",
    )
    market_refs.append(
        {
            "path": str(
                _absolute_reference_path(
                    scope_info.get("path"),
                    label="expected scope",
                )
            ),
            "sha256": _valid_sha256(scope_info.get("sha256")),
        }
    )
    history_info = _as_mapping(
        immutable.get("history_audit"),
        label="source history audit immutable ref",
    )
    market_refs.append(
        {
            "path": str(
                _absolute_reference_path(
                    history_info.get("path"),
                    label="history audit",
                )
            ),
            "sha256": _valid_sha256(history_info.get("sha256")),
        }
    )
    market = {
        "live_pointer_path": str(
            _absolute_reference_path(
                live_paths["market"],
                label="market live pointer",
            )
        ),
        "pointer_sha256": market_sha,
        "exact_pointer_bytes_b64": base64.b64encode(market_bytes).decode("ascii"),
        "immutable_refs": market_refs,
    }

    pit_bytes, pit_sha = _source_capture_entry(captured, "pit")
    pit_info = _as_mapping(
        immutable.get("pit"),
        label="source PIT immutable refs",
    )
    pit_refs = [
        {
            "path": str(
                _absolute_reference_path(
                    pit_info.get("manifest_path"),
                    label="PIT immutable manifest",
                )
            ),
            "sha256": _valid_sha256(pit_info.get("manifest_sha256")),
        },
        {
            "path": str(
                _absolute_reference_path(
                    pit_info.get("membership_path"),
                    label="PIT immutable membership",
                )
            ),
            "sha256": _valid_sha256(pit_info.get("membership_sha256")),
        },
    ]
    pit = {
        "live_pointer_path": str(
            _absolute_reference_path(
                live_paths["pit"],
                label="PIT live pointer",
            )
        ),
        "pointer_sha256": pit_sha,
        "exact_pointer_bytes_b64": base64.b64encode(pit_bytes).decode("ascii"),
        "immutable_refs": pit_refs,
    }
    return predecessor, market, pit


def _validate_chain_metadata(
    capture: Mapping[str, Any],
    *,
    chain_validator: ChainValidator | None,
) -> tuple[dict[str, Any], str, str, str]:
    predecessor = _as_mapping(capture["predecessor"], label="predecessor")
    chain = _as_mapping(capture["successor_chain"], label="successor_chain")
    original_seam = _date(capture["original_seam"], label="original_seam")
    parent_cutoff = _date(
        capture["immediate_parent_cutoff"],
        label="immediate_parent_cutoff",
    )
    target_cutoff = _date(capture["target_cutoff"], label="target_cutoff")
    if not parent_cutoff < target_cutoff:
        raise SuccessorPromotionError("successor cutoff interval is not increasing")
    predecessor_generation_id = _safe_id(
        predecessor.get("generation_id"),
        label="predecessor generation_id",
    )
    declared_parent_cutoff = predecessor.get("effective_cutoff") or predecessor.get(
        "cutoff"
    )
    if declared_parent_cutoff and _date(
        declared_parent_cutoff,
        label="predecessor cutoff",
    ) != parent_cutoff:
        raise SuccessorPromotionError("predecessor cutoff binding mismatch")
    predecessor_schema = str(
        predecessor.get("provenance_schema_version")
        or predecessor.get("primary_provenance_schema")
        or predecessor.get("provenance_schema")
        or ""
    ).strip()
    if predecessor_schema.endswith(".v2"):
        if original_seam != parent_cutoff:
            raise SuccessorPromotionError("first successor seam mismatch")
    elif predecessor_schema == SUCCESSOR_PROVENANCE_SCHEMA:
        predecessor_seam = _date(
            predecessor.get("original_seam"),
            label="predecessor original_seam",
        )
        if predecessor_seam != original_seam:
            raise SuccessorPromotionError("successor original seam changed")
    elif predecessor_schema:
        raise SuccessorPromotionError("predecessor provenance schema is unsupported")

    depth_raw = chain.get("depth", 1)
    if isinstance(depth_raw, bool):
        raise SuccessorPromotionError("successor chain depth is invalid")
    depth = int(depth_raw)
    if depth < 1 or depth > _MAX_CHAIN_DEPTH:
        raise SuccessorPromotionError("successor chain resource bound exceeded")
    ids_raw = chain.get("generation_ids") or chain.get("ancestor_generation_ids") or []
    if not isinstance(ids_raw, Sequence) or isinstance(ids_raw, (str, bytes)):
        raise SuccessorPromotionError("successor chain generation IDs are invalid")
    ids = [_safe_id(value, label="successor chain generation_id") for value in ids_raw]
    if len(ids) > _MAX_CHAIN_DEPTH or len(ids) != len(set(ids)):
        raise SuccessorPromotionError("successor chain contains a cycle")
    generation_id = _safe_id(
        capture.get("generation_id"),
        label="successor generation_id",
    )
    if "ancestor_generation_ids" in chain:
        if (
            len(ids) < 2
            or ids[-1] != generation_id
            or ids[-2] != predecessor_generation_id
        ):
            raise SuccessorPromotionError("successor chain tip is invalid")
        if len(ids) - 1 > _MAX_CHAIN_DEPTH:
            raise SuccessorPromotionError("successor chain resource bound exceeded")
    elif predecessor_generation_id in ids[:-1]:
        raise SuccessorPromotionError("successor chain repeats its predecessor")
    chain_seam = chain.get("original_seam")
    if isinstance(chain_seam, Mapping):
        chain_seam = chain_seam.get("cutoff")
    if chain_seam and _date(
        chain_seam,
        label="successor chain original_seam",
    ) != original_seam:
        raise SuccessorPromotionError("successor chain seam mismatch")
    if chain_validator is not None:
        try:
            receipt = chain_validator(dict(capture))
        except SuccessorPromotionError:
            raise
        except Exception as exc:
            raise SuccessorPromotionError(
                "successor chain validator failed"
            ) from exc
        if receipt is False:
            raise SuccessorPromotionError("successor chain validator rejected capture")
    return chain, original_seam, parent_cutoff, target_cutoff


def _capture_successor(
    root: str | Path,
    *,
    staging_validator: StagingValidator | None,
    chain_validator: ChainValidator | None,
    historical_only: bool,
) -> _SuccessorCapture:
    scanned = _scan_generation(root)
    pointer = scanned["pointer"]
    manifest = scanned["manifest"]
    if staging_validator is None:
        try:
            validation = _default_staging_validator(
                scanned["staging_root"],
                pointer=pointer,
                manifest=manifest,
                generation_root=scanned["generation_root"],
                historical_only=historical_only,
            )
        except SuccessorPromotionError:
            raise
        except Exception as exc:
            raise SuccessorPromotionError(
                "successor staging provenance validation failed"
            ) from exc
    else:
        try:
            validation = staging_validator(scanned["staging_root"])
        except SuccessorPromotionError:
            raise
        except Exception as exc:
            raise SuccessorPromotionError(
                "successor staging validator failed"
            ) from exc
    validation_map = _as_mapping(validation, label="successor validator result")
    provenance = _nested_mapping(
        (validation_map, manifest, pointer),
        "primary_provenance",
        "provenance",
    )
    pointer_provenance = _as_mapping(
        pointer.get("primary_provenance"),
        label="pointer primary_provenance",
    )
    manifest_provenance = _as_mapping(
        manifest.get("primary_provenance"),
        label="manifest primary_provenance",
    )
    if pointer_provenance != manifest_provenance:
        raise SuccessorPromotionError("successor pointer/manifest provenance mismatch")
    if not provenance:
        provenance = manifest_provenance
    machine_value = provenance.get("machine_states")
    machine_states = (
        _as_mapping(
            machine_value,
            label="successor machine states",
        )
        if machine_value is not None
        else {}
    )
    if (
        provenance.get("schema_version") != SUCCESSOR_PROVENANCE_SCHEMA
        or provenance.get("status") != SUCCESSOR_PROVENANCE_STATUS
        or provenance.get("history_state") not in (None, "mixed")
        or (
            provenance.get("mixed_generation") is not True
            and machine_states.get("mixed") is not True
        )
    ):
        raise SuccessorPromotionError("successor v3 mixed provenance is invalid")

    sources: tuple[Mapping[str, Any], ...] = (
        validation_map,
        provenance,
        manifest,
        pointer,
    )
    predecessor = _nested_mapping(
        sources,
        "predecessor",
        "predecessor_binding",
        "immediate_predecessor",
    )
    target_bindings = _nested_mapping(sources, "target_bindings")
    market_binding = _nested_mapping(
        (validation_map, provenance, target_bindings),
        "market_binding",
        "market",
    )
    pit_binding = _nested_mapping(
        (validation_map, provenance, target_bindings),
        "pit_binding",
        "pit",
    )
    if (
        not predecessor
        or not predecessor.get("exact_pointer_bytes_b64")
        or not market_binding.get("exact_pointer_bytes_b64")
        or not pit_binding.get("exact_pointer_bytes_b64")
    ):
        durable_predecessor, durable_market, durable_pit = (
            _durable_bindings_from_support_manifest(
                generation_root=scanned["generation_root"],
                provenance=provenance,
            )
        )
        predecessor = durable_predecessor
        market_binding = durable_market
        pit_binding = durable_pit
    if not predecessor or not market_binding or not pit_binding:
        raise SuccessorPromotionError("successor durable pointer bindings are incomplete")
    _validate_embedded_binding(predecessor, label="predecessor Fundamental")
    _validate_embedded_binding(market_binding, label="captured market")
    _validate_embedded_binding(pit_binding, label="captured PIT")

    declared_generation = _first_value(sources, "generation_id")
    if declared_generation and str(declared_generation) != scanned["generation_id"]:
        raise SuccessorPromotionError("validator generation_id mismatch")
    declared_pointer_sha = _first_value(sources, "pointer_sha256")
    if declared_pointer_sha and _valid_sha256(declared_pointer_sha) != scanned[
        "pointer_sha256"
    ]:
        raise SuccessorPromotionError("validator pointer SHA256 mismatch")
    declared_manifest_sha = _first_value(sources, "manifest_sha256")
    if declared_manifest_sha and _valid_sha256(declared_manifest_sha) != scanned[
        "manifest_sha256"
    ]:
        raise SuccessorPromotionError("validator manifest SHA256 mismatch")

    declared_table_sha = _nested_mapping(
        sources,
        "table_sha256",
        "table_sha256s",
    )
    if declared_table_sha:
        if set(declared_table_sha) != set(FUNDAMENTAL_TABLES):
            raise SuccessorPromotionError("validator table SHA256 set mismatch")
        for table_name in FUNDAMENTAL_TABLES:
            if _valid_sha256(declared_table_sha[table_name]) != scanned["table_files"][
                table_name
            ].sha256:
                raise SuccessorPromotionError(
                    f"validator {table_name} SHA256 mismatch"
                )

    declared_provider_value = _first_value(
        sources,
        "provider_evidence_files",
        "provider_files",
    )
    if not isinstance(declared_provider_value, Mapping):
        provider_manifest = _nested_mapping(
            (
                _as_mapping(
                    manifest.get("metadata"),
                    label="successor manifest metadata",
                ),
            ),
            "provider_manifest",
        )
        evidence_files = _as_mapping(
            provider_manifest.get("evidence_files"),
            label="provider manifest evidence files",
        )
        declared_provider_value = {
            **evidence_files,
            "provider_manifest.json": _sha256_bytes(
                _canonical_json_bytes(provider_manifest)
            ),
        }
    declared_provider_files = _normalize_declared_provider_files(
        declared_provider_value,
        generation_root=scanned["generation_root"],
    )
    actual_provider_files = scanned["provider_files"]
    if not actual_provider_files:
        raise SuccessorPromotionError("successor provider evidence files are missing")
    if not declared_provider_files:
        raise SuccessorPromotionError(
            "successor validator did not seal provider evidence files"
        )
    if set(declared_provider_files) != set(actual_provider_files):
        raise SuccessorPromotionError("provider evidence fileset mismatch")
    for relative, declared in declared_provider_files.items():
        actual = actual_provider_files[relative]
        if actual.sha256 != declared["sha256"]:
            raise SuccessorPromotionError("provider evidence SHA256 mismatch")
        if "size" in declared and actual.size != int(declared["size"]):
            raise SuccessorPromotionError("provider evidence size mismatch")

    capture_map: dict[str, Any] = {
        **validation_map,
        "generation_id": scanned["generation_id"],
        "predecessor": predecessor,
        "market_binding": market_binding,
        "pit_binding": pit_binding,
        "successor_chain": _nested_mapping(sources, "successor_chain", "chain"),
        "original_seam": _first_value(sources, "original_seam", "seam_trade_date"),
        "immediate_parent_cutoff": _first_value(
            sources,
            "immediate_parent_cutoff",
            "trusted_prefix_cutoff",
            "parent_cutoff",
        ),
        "target_cutoff": _first_value(
            sources,
            "target_cutoff",
            "strict_pit_as_of",
            "as_of",
        ),
    }
    append_boundary = _nested_mapping(
        (capture_map["successor_chain"],),
        "append_boundary",
    )
    original_seam_binding = _nested_mapping(
        (capture_map["successor_chain"],),
        "original_seam",
    )
    if not capture_map["original_seam"]:
        capture_map["original_seam"] = original_seam_binding.get("cutoff")
    if not capture_map["immediate_parent_cutoff"]:
        capture_map["immediate_parent_cutoff"] = append_boundary.get(
            "parent_cutoff"
        )
    if not capture_map["target_cutoff"]:
        capture_map["target_cutoff"] = append_boundary.get("target_cutoff")
    chain, original_seam, parent_cutoff, target_cutoff = _validate_chain_metadata(
        capture_map,
        chain_validator=chain_validator,
    )
    provenance_binding_sha256 = _valid_sha256(
        _first_value(
            sources,
            "provenance_binding_sha256",
            "successor_binding_sha256",
            "binding_sha256",
            "envelope_sha256",
        )
    )
    validator_receipt_sha256 = _canonical_sha256(validation_map)
    return _SuccessorCapture(
        staging_root=scanned["staging_root"],
        generation_root=scanned["generation_root"],
        generation_id=scanned["generation_id"],
        pointer_path=scanned["pointer_path"],
        pointer_bytes=scanned["pointer_bytes"],
        pointer_sha256=scanned["pointer_sha256"],
        pointer=pointer,
        manifest_path=scanned["manifest_path"],
        manifest_bytes=scanned["manifest_bytes"],
        manifest_sha256=scanned["manifest_sha256"],
        manifest=manifest,
        table_files=scanned["table_files"],
        provider_files=actual_provider_files,
        provider_fileset_sha256=scanned["provider_fileset_sha256"],
        predecessor=predecessor,
        market_binding=market_binding,
        pit_binding=pit_binding,
        original_seam=original_seam,
        immediate_parent_cutoff=parent_cutoff,
        target_cutoff=target_cutoff,
        provenance_binding_sha256=provenance_binding_sha256,
        successor_chain=chain,
        validator_receipt_sha256=validator_receipt_sha256,
    )


def _capture_identity(capture: _SuccessorCapture) -> dict[str, Any]:
    return {
        "generation_id": capture.generation_id,
        "pointer_sha256": capture.pointer_sha256,
        "manifest_sha256": capture.manifest_sha256,
        "table_sha256": {
            name: capture.table_files[name].sha256 for name in FUNDAMENTAL_TABLES
        },
        "provider_fileset_sha256": capture.provider_fileset_sha256,
        "provenance_binding_sha256": capture.provenance_binding_sha256,
        "predecessor_pointer_sha256": str(
            capture.predecessor.get("pointer_sha256") or ""
        ).lower(),
        "market_pointer_sha256": str(
            capture.market_binding.get("pointer_sha256") or ""
        ).lower(),
        "pit_pointer_sha256": str(
            capture.pit_binding.get("pointer_sha256") or ""
        ).lower(),
        "original_seam": capture.original_seam,
        "immediate_parent_cutoff": capture.immediate_parent_cutoff,
        "target_cutoff": capture.target_cutoff,
    }


def _recapture_file_identity(capture: _SuccessorCapture) -> None:
    scanned = _scan_generation(capture.staging_root)
    expected = _capture_identity(capture)
    actual = {
        "generation_id": scanned["generation_id"],
        "pointer_sha256": scanned["pointer_sha256"],
        "manifest_sha256": scanned["manifest_sha256"],
        "table_sha256": {
            name: scanned["table_files"][name].sha256 for name in FUNDAMENTAL_TABLES
        },
        "provider_fileset_sha256": scanned["provider_fileset_sha256"],
    }
    for key in (
        "generation_id",
        "pointer_sha256",
        "manifest_sha256",
        "table_sha256",
        "provider_fileset_sha256",
    ):
        if actual[key] != expected[key]:
            raise SuccessorPromotionError(f"successor staging {key} drifted")


def _validate_pointer_references(
    root: Path,
    pointer_bytes: bytes,
    *,
    expected_pointer_sha256: str,
    expected_manifest_sha256: str | None = None,
    manifest_maximum_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any]:
    if _sha256_bytes(pointer_bytes) != expected_pointer_sha256:
        raise SuccessorPromotionError("Fundamental pointer SHA256 mismatch")
    pointer = _json_object(pointer_bytes, label="Fundamental pointer")
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
    ):
        raise SuccessorPromotionError("Fundamental pointer shape is invalid")
    generation_id = _safe_id(pointer.get("generation_id"), label="generation_id")
    manifest_path = _resolve_inside(
        root,
        pointer.get("manifest_path"),
        label="Fundamental manifest",
    )
    manifest_bytes, manifest_identity = _stable_small_bytes(
        manifest_path,
        label="Fundamental manifest",
        maximum_bytes=manifest_maximum_bytes,
    )
    if expected_manifest_sha256 and manifest_identity.sha256 != _valid_sha256(
        expected_manifest_sha256
    ):
        raise SuccessorPromotionError("Fundamental manifest SHA256 mismatch")
    manifest = _json_object(
        manifest_bytes,
        label="Fundamental manifest",
        maximum_bytes=manifest_maximum_bytes,
    )
    if (
        manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or str(manifest.get("generation_id") or "") != generation_id
    ):
        raise SuccessorPromotionError("Fundamental generation binding mismatch")
    pointer_tables = _as_mapping(pointer.get("tables"), label="pointer tables")
    manifest_tables = _as_mapping(manifest.get("tables"), label="manifest tables")
    if set(pointer_tables) != set(FUNDAMENTAL_TABLES) or set(manifest_tables) != set(
        FUNDAMENTAL_TABLES
    ):
        raise SuccessorPromotionError("Fundamental table set mismatch")
    for table_name in FUNDAMENTAL_TABLES:
        table_path = _resolve_inside(
            root,
            pointer_tables[table_name],
            label=f"Fundamental {table_name}",
        )
        identity = _stable_file_hash(table_path, label=f"Fundamental {table_name}")
        contract = _as_mapping(
            manifest_tables[table_name],
            label=f"Fundamental {table_name} contract",
        )
        if identity.sha256 != _valid_sha256(contract.get("sha256")):
            raise SuccessorPromotionError(
                f"Fundamental {table_name} SHA256 mismatch"
            )
    return {
        "pointer": pointer,
        "manifest": manifest,
        "manifest_sha256": manifest_identity.sha256,
    }


def _validate_current_predecessor(
    canonical_root: Path,
    capture: _SuccessorCapture,
    *,
    expected_pointer_sha256: str,
) -> bytes:
    expected_bytes, embedded_sha256, embedded_pointer = _decode_pointer_evidence(
        capture.predecessor,
        label="predecessor Fundamental",
    )
    if embedded_sha256 != expected_pointer_sha256:
        raise SuccessorPromotionError(
            "embedded predecessor SHA256 does not equal CLI expected SHA256"
        )
    pointer_path = canonical_root / FUNDAMENTAL_POINTER_FILENAME
    current_bytes, current_identity = _stable_small_bytes(
        pointer_path,
        label="canonical Fundamental pointer",
    )
    if current_identity.sha256 != expected_pointer_sha256 or current_bytes != expected_bytes:
        raise SuccessorPromotionError("canonical Fundamental pointer CAS mismatch")
    if str(embedded_pointer.get("generation_id") or "") != str(
        capture.predecessor.get("generation_id") or ""
    ):
        raise SuccessorPromotionError("predecessor generation binding mismatch")
    validated = _validate_pointer_references(
        canonical_root,
        current_bytes,
        expected_pointer_sha256=expected_pointer_sha256,
        expected_manifest_sha256=str(
            capture.predecessor.get("manifest_sha256") or ""
        )
        or None,
        manifest_maximum_bytes=_MAX_PREDECESSOR_MANIFEST_JSON_BYTES,
    )
    if str(validated["pointer"].get("generation_id") or "") != str(
        capture.predecessor.get("generation_id") or ""
    ):
        raise SuccessorPromotionError("live predecessor generation mismatch")
    return current_bytes


def _candidate_pointer(
    capture: _SuccessorCapture,
    canonical_root: Path,
) -> tuple[dict[str, Any], bytes, str, Path]:
    final_root = (
        canonical_root / FUNDAMENTAL_GENERATIONS_DIRNAME / capture.generation_id
    )
    relative_root = final_root.relative_to(canonical_root)
    pointer = deepcopy(capture.pointer)
    pointer["manifest_path"] = (relative_root / "manifest.json").as_posix()
    pointer["tables"] = {
        table_name: (relative_root / f"{table_name}.parquet").as_posix()
        for table_name in FUNDAMENTAL_TABLES
    }
    pointer_bytes = _canonical_json_bytes(pointer)
    return pointer, pointer_bytes, _sha256_bytes(pointer_bytes), final_root


def _generation_aggregate(capture: _SuccessorCapture) -> str:
    return _canonical_sha256(
        {
            "generation_id": capture.generation_id,
            "manifest_sha256": capture.manifest_sha256,
            "table_sha256": {
                name: capture.table_files[name].sha256
                for name in FUNDAMENTAL_TABLES
            },
            "provider_fileset_sha256": capture.provider_fileset_sha256,
            "provenance_binding_sha256": capture.provenance_binding_sha256,
        }
    )


def _invoke_live_validator(
    validator: LiveBindingValidator | None,
    capture: _SuccessorCapture,
    phase: str,
) -> None:
    if validator is None:
        return
    try:
        receipt = validator(_capture_identity(capture), phase)
    except SuccessorPromotionError:
        raise
    except Exception as exc:
        raise SuccessorPromotionError(
            f"successor live binding validation failed at {phase}"
        ) from exc
    if receipt is False:
        raise SuccessorPromotionError(
            f"successor live binding validator rejected {phase}"
        )


def _validate_live_state(
    canonical_root: Path,
    capture: _SuccessorCapture,
    *,
    expected_pointer_sha256: str,
    live_binding_validator: LiveBindingValidator | None,
    phase: str,
) -> bytes:
    previous_bytes = _validate_current_predecessor(
        canonical_root,
        capture,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    _validate_live_binding(capture.market_binding, label="market")
    _validate_live_binding(capture.pit_binding, label="PIT")
    _invoke_live_validator(live_binding_validator, capture, phase)
    return previous_bytes


def _preflight(
    *,
    staging_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
    staging_validator: StagingValidator | None,
    chain_validator: ChainValidator | None,
    live_binding_validator: LiveBindingValidator | None,
) -> tuple[dict[str, Any], _SuccessorCapture, Path]:
    expected_sha256 = _valid_sha256(expected_pointer_sha256)
    capture = _capture_successor(
        staging_root,
        staging_validator=staging_validator,
        chain_validator=chain_validator,
        historical_only=False,
    )
    canonical = _secure_existing_directory(
        canonical_root,
        label="canonical Fundamental root",
    )
    if (
        canonical == capture.staging_root
        or canonical in capture.staging_root.parents
        or capture.staging_root in canonical.parents
    ):
        raise SuccessorPromotionError(
            "staging and canonical roots must be disjoint"
        )
    embedded_sha = _valid_sha256(capture.predecessor.get("pointer_sha256"))
    if embedded_sha != expected_sha256:
        raise SuccessorPromotionError(
            "embedded predecessor SHA256 does not equal CLI expected SHA256"
        )
    _validate_live_state(
        canonical,
        capture,
        expected_pointer_sha256=expected_sha256,
        live_binding_validator=live_binding_validator,
        phase="preflight",
    )
    pointer, pointer_bytes, pointer_sha256, final_root = _candidate_pointer(
        capture,
        canonical,
    )
    try:
        os.lstat(final_root)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SuccessorPromotionError("candidate generation path is unsafe") from exc
    else:
        raise SuccessorPromotionError("candidate generation already exists")
    plan = {
        "schema_version": SUCCESSOR_PROMOTION_PLAN_SCHEMA,
        "status": "PREFLIGHT_OK",
        "execute": False,
        "promoted": False,
        "generation_id": capture.generation_id,
        "expected_pointer_sha256": expected_sha256,
        "candidate_pointer_sha256": pointer_sha256,
        "candidate_pointer": pointer,
        "candidate_pointer_bytes_b64": base64.b64encode(pointer_bytes).decode("ascii"),
        "manifest_sha256": capture.manifest_sha256,
        "table_sha256": {
            name: capture.table_files[name].sha256 for name in FUNDAMENTAL_TABLES
        },
        "provider_fileset_sha256": capture.provider_fileset_sha256,
        "provenance_binding_sha256": capture.provenance_binding_sha256,
        "generation_aggregate_sha256": _generation_aggregate(capture),
        "original_seam": capture.original_seam,
        "immediate_parent_cutoff": capture.immediate_parent_cutoff,
        "target_cutoff": capture.target_cutoff,
        "market_pointer_sha256": _valid_sha256(
            capture.market_binding.get("pointer_sha256")
        ),
        "pit_pointer_sha256": _valid_sha256(
            capture.pit_binding.get("pointer_sha256")
        ),
        "final_generation_path": str(final_root),
        "staging_validation_receipt_sha256": capture.validator_receipt_sha256,
    }
    return plan, capture, canonical


def validate_successor_historical_evidence(
    *,
    staging_root: str | Path,
    staging_validator: StagingValidator | None = None,
    chain_validator: ChainValidator | None = None,
) -> dict[str, Any]:
    """Validate sealed history without requiring live pointers to stay frozen.

    This is the verification path for an already-published generation.  Live
    market, PIT, and Fundamental pointers may legitimately advance after it was
    published; their captured exact bytes and immutable references must not.
    """

    capture = _capture_successor(
        staging_root,
        staging_validator=staging_validator,
        chain_validator=chain_validator,
        historical_only=True,
    )
    _validate_embedded_binding(
        capture.predecessor,
        label="predecessor Fundamental",
    )
    _validate_embedded_binding(capture.market_binding, label="captured market")
    _validate_embedded_binding(capture.pit_binding, label="captured PIT")
    return {
        "status": "OK",
        "historical_only": True,
        **_capture_identity(capture),
        "generation_aggregate_sha256": _generation_aggregate(capture),
    }


def preflight_successor_promotion(
    *,
    staging_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
    staging_validator: StagingValidator | None = None,
    chain_validator: ChainValidator | None = None,
    live_binding_validator: LiveBindingValidator | None = None,
) -> dict[str, Any]:
    """Validate a successor promotion without writing canonical or journal state."""

    plan, _capture, _canonical = _preflight(
        staging_root=staging_root,
        canonical_root=canonical_root,
        expected_pointer_sha256=expected_pointer_sha256,
        staging_validator=staging_validator,
        chain_validator=chain_validator,
        live_binding_validator=live_binding_validator,
    )
    return plan


def _lock_file_path(root: Path, filename: str) -> Path:
    _secure_existing_directory(root, label=f"{filename} root")
    return root / filename


@contextmanager
def _secure_timed_lock(
    lock_path: Path,
    *,
    deadline: float,
    poll_seconds: float = 0.05,
) -> Iterator[None]:
    root = _secure_existing_directory(lock_path.parent, label="writer lock root")
    root_identity = _directory_identity(os.lstat(root))
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SuccessorPromotionError("writer lock is unsafe") from exc
    locked = False
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or _signature(opened) != _signature(current)
        ):
            raise SuccessorPromotionError("writer lock identity is unsafe")
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SuccessorPromotionError(
                        "writer lock acquisition timed out"
                    )
                time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
        current = os.lstat(lock_path)
        if (
            _signature(os.fstat(descriptor)) != _signature(current)
            or _directory_identity(os.lstat(root)) != root_identity
        ):
            raise SuccessorPromotionError("writer lock changed after acquisition")
        yield
    except OSError as exc:
        raise SuccessorPromotionError("writer lock operation failed") from exc
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _ordered_writer_locks(
    *,
    capture: _SuccessorCapture,
    canonical_root: Path,
    timeout_seconds: float,
) -> Iterator[None]:
    if isinstance(timeout_seconds, bool) or float(timeout_seconds) < 0:
        raise SuccessorPromotionError("lock timeout is invalid")
    market_pointer = _binding_live_pointer_path(
        capture.market_binding,
        label="market",
    )
    pit_pointer = _binding_live_pointer_path(capture.pit_binding, label="PIT")
    lock_paths = (
        _lock_file_path(market_pointer.parent, MARKET_WRITER_LOCK_FILENAME),
        _lock_file_path(pit_pointer.parent, PIT_WRITER_LOCK_FILENAME),
        _lock_file_path(canonical_root, FUNDAMENTAL_PROMOTION_LOCK_FILENAME),
    )
    if len(set(lock_paths)) != 3:
        raise SuccessorPromotionError("writer lock paths are not distinct")
    deadline = time.monotonic() + float(timeout_seconds)
    with ExitStack() as stack:
        for lock_path in lock_paths:
            stack.enter_context(_secure_timed_lock(lock_path, deadline=deadline))
        yield


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_private_file(path: Path, payload: bytes, *, label: str) -> None:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SuccessorPromotionError(f"{label} cannot be created") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SuccessorPromotionError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    written_bytes, identity = _stable_small_bytes(path, label=label)
    if written_bytes != payload or stat.S_IMODE(os.lstat(path).st_mode) != 0o600:
        raise SuccessorPromotionError(f"{label} exact readback failed")
    if identity.sha256 != _sha256_bytes(payload):
        raise SuccessorPromotionError(f"{label} SHA256 readback failed")


def _copy_private_streamed(
    source: _FileIdentity,
    destination: Path,
    *,
    label: str,
) -> _FileIdentity:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    source_before = _safe_regular_metadata(source.path, label=label)
    if _signature(source_before) != source.signature:
        raise SuccessorPromotionError(f"{label} source changed before copy")
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    destination_flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    source_descriptor = os.open(source.path, source_flags)
    try:
        destination_descriptor = os.open(destination, destination_flags, 0o600)
    except Exception:
        os.close(source_descriptor)
        raise
    try:
        if _signature(os.fstat(source_descriptor)) != source.signature:
            raise SuccessorPromotionError(f"{label} source changed during open")
        source_digest = hashlib.sha256()
        bytes_written = 0
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                break
            source_digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise SuccessorPromotionError(f"{label} copy made no progress")
                view = view[written:]
                bytes_written += written
        os.fsync(destination_descriptor)
        if (
            _signature(os.fstat(source_descriptor)) != source.signature
            or _signature(os.lstat(source.path)) != source.signature
            or source_digest.hexdigest() != source.sha256
            or bytes_written != source.size
        ):
            raise SuccessorPromotionError(f"{label} source changed during copy")
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    installed = _stable_file_hash(destination, label=f"installed {label}")
    if (
        installed.sha256 != source.sha256
        or installed.size != source.size
        or stat.S_IMODE(os.lstat(destination).st_mode) != 0o600
    ):
        raise SuccessorPromotionError(f"{label} installed identity mismatch")
    return installed


def _atomic_write_private_bytes(path: Path, payload: bytes, *, label: str) -> None:
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode)
        or not stat.S_ISREG(existing.st_mode)
        or existing.st_nlink != 1
    ):
        raise SuccessorPromotionError(f"{label} target is unsafe")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SuccessorPromotionError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        readback, identity = _stable_small_bytes(temp_path, label=f"temporary {label}")
        if readback != payload or identity.sha256 != _sha256_bytes(payload):
            raise SuccessorPromotionError(f"{label} temporary readback failed")
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
        final, final_identity = _stable_small_bytes(path, label=label)
        if final != payload or final_identity.sha256 != _sha256_bytes(payload):
            raise SuccessorPromotionError(f"{label} exact readback failed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.lstat(temp_path)
        except FileNotFoundError:
            pass
        else:
            os.unlink(temp_path)


def _copy_generation(
    capture: _SuccessorCapture,
    *,
    generations_root: Path,
    final_root: Path,
    fault_injector: FaultInjector | None,
) -> None:
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{capture.generation_id}.promotion.",
            dir=generations_root,
        )
    )
    os.chmod(staging_directory, 0o700)
    renamed = False
    try:
        provider_directories = {
            (staging_directory / Path(relative)).parent
            for relative in capture.provider_files
        }
        for directory in sorted(
            provider_directories,
            key=lambda value: (len(value.parts), value.as_posix()),
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            cursor = directory
            while cursor != staging_directory:
                os.chmod(cursor, 0o700)
                cursor = cursor.parent
        manifest_identity = _stable_file_hash(
            capture.manifest_path,
            label="successor manifest",
        )
        _copy_private_streamed(
            manifest_identity,
            staging_directory / "manifest.json",
            label="successor manifest",
        )
        for table_name in FUNDAMENTAL_TABLES:
            _copy_private_streamed(
                capture.table_files[table_name],
                staging_directory / f"{table_name}.parquet",
                label=table_name,
            )
        for relative, identity in capture.provider_files.items():
            destination = staging_directory / Path(relative)
            if staging_directory not in destination.parents:
                raise SuccessorPromotionError("provider evidence copy path escapes")
            _copy_private_streamed(
                identity,
                destination,
                label=f"provider evidence {relative}",
            )
        for walk_directory, _names, _files in os.walk(
            staging_directory,
            topdown=False,
        ):
            _fsync_directory(Path(walk_directory))
        if fault_injector is not None:
            fault_injector("before_generation_rename")
        os.replace(staging_directory, final_root)
        renamed = True
        _fsync_directory(final_root)
        _fsync_directory(generations_root)
        if fault_injector is not None:
            fault_injector("after_generation_rename")
    finally:
        if not renamed and staging_directory.exists():
            shutil.rmtree(staging_directory)


def _phase_transition_allowed(phases: Sequence[str], next_phase: str) -> bool:
    if not phases:
        return next_phase == "INTENT"
    previous = phases[-1]
    if previous == "TERMINAL" or next_phase == "INTENT":
        return False
    if next_phase == "TERMINAL":
        return True
    allowed = {
        "INTENT": {"PRECAS_VALIDATED"},
        "PRECAS_VALIDATED": {"INSTALL_INTENT"},
        "INSTALL_INTENT": {"GENERATION_INSTALLED"},
        "GENERATION_INSTALLED": {"CAS_COMMITTED"},
        "CAS_COMMITTED": {"POSTCHECK_PASSED", "ROLLBACK_COMMITTED"},
        "POSTCHECK_PASSED": set(),
        "ROLLBACK_COMMITTED": set(),
    }
    return next_phase in allowed.get(previous, set())


def _validate_installed_generation(
    *,
    capture: _SuccessorCapture,
    canonical_root: Path,
    final_root: Path,
    expected_pointer_bytes: bytes,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    current_pointer, pointer_identity = _stable_small_bytes(
        canonical_root / FUNDAMENTAL_POINTER_FILENAME,
        label="promoted Fundamental pointer",
    )
    if (
        current_pointer != expected_pointer_bytes
        or pointer_identity.sha256 != expected_pointer_sha256
    ):
        raise SuccessorPromotionError("promoted Fundamental pointer identity mismatch")
    installed_manifest, manifest_identity = _stable_small_bytes(
        final_root / "manifest.json",
        label="installed successor manifest",
    )
    if (
        installed_manifest != capture.manifest_bytes
        or manifest_identity.sha256 != capture.manifest_sha256
    ):
        raise SuccessorPromotionError("installed successor manifest mismatch")
    for table_name in FUNDAMENTAL_TABLES:
        identity = _stable_file_hash(
            final_root / f"{table_name}.parquet",
            label=f"installed {table_name}",
        )
        expected = capture.table_files[table_name]
        if identity.sha256 != expected.sha256 or identity.size != expected.size:
            raise SuccessorPromotionError(f"installed {table_name} mismatch")
    installed_provider = _scan_provider_files(final_root)
    if _canonical_sha256(_provider_fileset_payload(installed_provider)) != (
        capture.provider_fileset_sha256
    ):
        raise SuccessorPromotionError("installed provider evidence mismatch")
    return _validate_pointer_references(
        canonical_root,
        current_pointer,
        expected_pointer_sha256=expected_pointer_sha256,
        expected_manifest_sha256=capture.manifest_sha256,
    )


class _Journal:
    def __init__(
        self,
        root: str | Path,
        run_id: str,
        *,
        create: bool,
    ) -> None:
        raw_root = _lexical_absolute(root, label="promotion journal root")
        if not Path(root).expanduser().is_absolute():
            raise SuccessorPromotionError("promotion journal root must be absolute")
        self.root = _secure_directory_tree(
            raw_root,
            label="promotion journal root",
            create=create,
            private=True,
        )
        self.run_id = _safe_id(run_id, label="journal run_id")
        self.run_root = self.root / self.run_id
        if create:
            try:
                os.mkdir(self.run_root, 0o700)
            except FileExistsError as exc:
                raise SuccessorPromotionError(
                    "promotion journal run already exists; use recovery"
                ) from exc
        self.run_root = _secure_directory_tree(
            self.run_root,
            label="promotion journal run root",
            create=False,
            private=True,
        )
        self.records = self._load_records()
        self.terminal = bool(self.records and self.records[-1]["phase"] == "TERMINAL")

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        entries = sorted(self.run_root.iterdir(), key=lambda path: path.name)
        for entry in entries:
            if not re.fullmatch(r"\d{6}_[A-Z_]+\.json", entry.name):
                raise SuccessorPromotionError("promotion journal contains an unsafe entry")
            if stat.S_IMODE(_safe_regular_metadata(entry, label="journal record").st_mode) != 0o600:
                raise SuccessorPromotionError("promotion journal record mode must be 0600")
            payload, _identity = _stable_small_bytes(entry, label="journal record")
            record = _json_object(payload, label="promotion journal record")
            expected_sequence = len(records)
            if (
                record.get("schema_version") != SUCCESSOR_PROMOTION_JOURNAL_SCHEMA
                or record.get("run_id") != self.run_id
                or record.get("sequence") != expected_sequence
            ):
                raise SuccessorPromotionError("promotion journal sequence is invalid")
            phase = str(record.get("phase") or "")
            if phase not in _JOURNAL_PHASES:
                raise SuccessorPromotionError("promotion journal phase is invalid")
            if not _phase_transition_allowed(
                [str(value["phase"]) for value in records],
                phase,
            ):
                raise SuccessorPromotionError(
                    "promotion journal phase transition is invalid"
                )
            expected_name = f"{expected_sequence:06d}_{phase}.json"
            if entry.name != expected_name:
                raise SuccessorPromotionError("promotion journal filename mismatch")
            body = dict(record)
            declared_record_sha = _valid_sha256(body.pop("record_sha256", ""))
            if _canonical_sha256(body) != declared_record_sha:
                raise SuccessorPromotionError("promotion journal record SHA mismatch")
            previous_sha = records[-1]["record_sha256"] if records else ""
            if str(record.get("previous_record_sha256") or "") != previous_sha:
                raise SuccessorPromotionError("promotion journal chain is invalid")
            records.append(record)
        terminal_indexes = [
            index for index, record in enumerate(records) if record["phase"] == "TERMINAL"
        ]
        if terminal_indexes and terminal_indexes != [len(records) - 1]:
            raise SuccessorPromotionError("promotion journal terminal is not final")
        return records

    def append(self, phase: str, details: Mapping[str, Any]) -> dict[str, Any]:
        if phase not in _JOURNAL_PHASES:
            raise SuccessorPromotionError("promotion journal phase is invalid")
        if self.terminal:
            raise SuccessorPromotionError("promotion journal is already terminal")
        if not _phase_transition_allowed(
            [str(record["phase"]) for record in self.records],
            phase,
        ):
            raise SuccessorPromotionError(
                "promotion journal phase transition is invalid"
            )
        sequence = len(self.records)
        body = {
            "schema_version": SUCCESSOR_PROMOTION_JOURNAL_SCHEMA,
            "run_id": self.run_id,
            "sequence": sequence,
            "phase": phase,
            "created_at": _utc_now(),
            "previous_record_sha256": (
                self.records[-1]["record_sha256"] if self.records else ""
            ),
            "details": dict(details),
        }
        record = {**body, "record_sha256": _canonical_sha256(body)}
        path = self.run_root / f"{sequence:06d}_{phase}.json"
        _write_new_private_file(
            path,
            _canonical_json_bytes(record),
            label="promotion journal record",
        )
        _fsync_directory(self.run_root)
        self.records.append(record)
        self.terminal = phase == "TERMINAL"
        return record


def _journal_capture_payload(
    capture: _SuccessorCapture,
    *,
    final_root: Path,
    candidate_pointer_bytes: bytes,
    candidate_pointer_sha256: str,
) -> dict[str, Any]:
    predecessor_bytes, predecessor_sha, _pointer = _decode_pointer_evidence(
        capture.predecessor,
        label="predecessor Fundamental",
    )
    return {
        "generation_id": capture.generation_id,
        "staging_root": str(capture.staging_root),
        "intended_generation_path": str(final_root),
        "manifest_sha256": capture.manifest_sha256,
        "table_sha256": {
            name: capture.table_files[name].sha256 for name in FUNDAMENTAL_TABLES
        },
        "table_size": {
            name: capture.table_files[name].size for name in FUNDAMENTAL_TABLES
        },
        "provider_fileset_sha256": capture.provider_fileset_sha256,
        "provider_files": {
            relative: {"sha256": identity.sha256, "size": identity.size}
            for relative, identity in capture.provider_files.items()
        },
        "provenance_binding_sha256": capture.provenance_binding_sha256,
        "candidate_pointer_sha256": candidate_pointer_sha256,
        "candidate_pointer_bytes_b64": base64.b64encode(
            candidate_pointer_bytes
        ).decode("ascii"),
        "predecessor_pointer_sha256": predecessor_sha,
        "predecessor_manifest_sha256": _valid_sha256(
            capture.predecessor.get("manifest_sha256")
        ),
        "predecessor_pointer_bytes_b64": base64.b64encode(
            predecessor_bytes
        ).decode("ascii"),
        "market_binding": capture.market_binding,
        "pit_binding": capture.pit_binding,
        "original_seam": capture.original_seam,
        "immediate_parent_cutoff": capture.immediate_parent_cutoff,
        "target_cutoff": capture.target_cutoff,
        "capture_identity": _capture_identity(capture),
    }


def _fault(injector: FaultInjector | None, phase: str) -> None:
    if injector is not None:
        injector(phase)


def _pointer_state(
    canonical_root: Path,
    *,
    predecessor_sha256: str,
    candidate_sha256: str,
) -> str:
    try:
        _payload, identity = _stable_small_bytes(
            canonical_root / FUNDAMENTAL_POINTER_FILENAME,
            label="canonical Fundamental pointer",
        )
    except SuccessorPromotionError:
        return "UNKNOWN"
    if identity.sha256 == candidate_sha256:
        return "CANDIDATE"
    if identity.sha256 == predecessor_sha256:
        return "PREDECESSOR"
    return "THIRD_PARTY"


def promote_successor_generation(
    *,
    staging_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
    execute: bool,
    journal_root: str | Path | None = None,
    journal_run_id: str | None = None,
    staging_validator: StagingValidator | None = None,
    chain_validator: ChainValidator | None = None,
    live_binding_validator: LiveBindingValidator | None = None,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    """Preflight or CAS-publish one already-derived successor generation.

    ``execute`` is intentionally required.  ``False`` is a strictly read-only
    preflight.  ``True`` requires an absolute durable journal root and a unique
    journal run ID.
    """

    if not isinstance(execute, bool):
        raise SuccessorPromotionError("execute must be an explicit boolean")
    plan, capture, canonical = _preflight(
        staging_root=staging_root,
        canonical_root=canonical_root,
        expected_pointer_sha256=expected_pointer_sha256,
        staging_validator=staging_validator,
        chain_validator=chain_validator,
        live_binding_validator=live_binding_validator,
    )
    if not execute:
        return plan
    if journal_root is None or journal_run_id is None:
        raise SuccessorPromotionError(
            "execute requires journal_root and journal_run_id"
        )
    run_id = _safe_id(journal_run_id, label="journal run_id")
    journal = _Journal(journal_root, run_id, create=True)
    if journal.records:
        raise SuccessorPromotionError(
            "promotion journal run already exists; use recovery",
            journal_run_id=run_id,
        )
    expected_sha256 = _valid_sha256(expected_pointer_sha256)
    candidate_pointer_bytes = base64.b64decode(
        plan["candidate_pointer_bytes_b64"].encode("ascii"),
        validate=True,
    )
    candidate_sha256 = str(plan["candidate_pointer_sha256"])
    final_root = Path(plan["final_generation_path"])
    generations_root = canonical / FUNDAMENTAL_GENERATIONS_DIRNAME
    journal.append(
        "INTENT",
        {
            "generation_id": capture.generation_id,
            "expected_pointer_sha256": expected_sha256,
            "candidate_pointer_sha256": candidate_sha256,
            "generation_aggregate_sha256": plan["generation_aggregate_sha256"],
        },
    )
    installed = False
    cas_committed = False
    try:
        with _ordered_writer_locks(
            capture=capture,
            canonical_root=canonical,
            timeout_seconds=lock_timeout_seconds,
        ):
            _recapture_file_identity(capture)
            previous_pointer_bytes = _validate_live_state(
                canonical,
                capture,
                expected_pointer_sha256=expected_sha256,
                live_binding_validator=live_binding_validator,
                phase="locked_precas",
            )
            try:
                os.lstat(final_root)
            except FileNotFoundError:
                pass
            else:
                raise SuccessorPromotionError("candidate generation already exists")
            generations_root = _secure_directory_tree(
                generations_root,
                label="Fundamental generations root",
                create=True,
            )
            journal.append(
                "PRECAS_VALIDATED",
                {
                    "generation_id": capture.generation_id,
                    "expected_pointer_sha256": expected_sha256,
                    "market_pointer_sha256": plan["market_pointer_sha256"],
                    "pit_pointer_sha256": plan["pit_pointer_sha256"],
                    "capture_identity": _capture_identity(capture),
                },
            )
            journal.append(
                "INSTALL_INTENT",
                _journal_capture_payload(
                    capture,
                    final_root=final_root,
                    candidate_pointer_bytes=candidate_pointer_bytes,
                    candidate_pointer_sha256=candidate_sha256,
                ),
            )
            _copy_generation(
                capture,
                generations_root=generations_root,
                final_root=final_root,
                fault_injector=fault_injector,
            )
            installed = True
            journal.append(
                "GENERATION_INSTALLED",
                {
                    "generation_id": capture.generation_id,
                    "intended_generation_path": str(final_root),
                    "manifest_sha256": capture.manifest_sha256,
                    "table_sha256": plan["table_sha256"],
                    "provider_fileset_sha256": capture.provider_fileset_sha256,
                },
            )
            _fault(fault_injector, "after_generation_installed_record")
            _recapture_file_identity(capture)
            repeated_previous = _validate_live_state(
                canonical,
                capture,
                expected_pointer_sha256=expected_sha256,
                live_binding_validator=live_binding_validator,
                phase="locked_precommit",
            )
            if repeated_previous != previous_pointer_bytes:
                raise SuccessorPromotionError("predecessor pointer bytes changed")
            _atomic_write_private_bytes(
                canonical / FUNDAMENTAL_POINTER_FILENAME,
                candidate_pointer_bytes,
                label="canonical Fundamental pointer",
            )
            cas_committed = True
            _fault(fault_injector, "after_pointer_write")
            journal.append(
                "CAS_COMMITTED",
                {
                    "generation_id": capture.generation_id,
                    "previous_pointer_sha256": expected_sha256,
                    "pointer_sha256": candidate_sha256,
                },
            )
            try:
                _validate_installed_generation(
                    capture=capture,
                    canonical_root=canonical,
                    final_root=final_root,
                    expected_pointer_bytes=candidate_pointer_bytes,
                    expected_pointer_sha256=candidate_sha256,
                )
                _validate_embedded_binding(
                    capture.predecessor,
                    label="predecessor Fundamental",
                )
                _validate_embedded_binding(
                    capture.market_binding,
                    label="captured market",
                )
                _validate_embedded_binding(capture.pit_binding, label="captured PIT")
                _invoke_live_validator(
                    live_binding_validator,
                    capture,
                    "locked_postcheck",
                )
                _fault(fault_injector, "before_postcheck_record")
            except Exception as postcheck_exc:
                state = _pointer_state(
                    canonical,
                    predecessor_sha256=expected_sha256,
                    candidate_sha256=candidate_sha256,
                )
                if state != "CANDIDATE":
                    journal.append(
                        "TERMINAL",
                        {
                            "status": "PROMOTION_UNCERTAIN",
                            "pointer_state": state,
                            "reason": type(postcheck_exc).__name__,
                        },
                    )
                    raise SuccessorPromotionError(
                        "postcheck failed after third-party pointer drift; rollback refused",
                        status="PROMOTION_UNCERTAIN",
                        journal_run_id=run_id,
                    ) from postcheck_exc
                _atomic_write_private_bytes(
                    canonical / FUNDAMENTAL_POINTER_FILENAME,
                    previous_pointer_bytes,
                    label="Fundamental predecessor rollback pointer",
                )
                rollback_bytes, rollback_identity = _stable_small_bytes(
                    canonical / FUNDAMENTAL_POINTER_FILENAME,
                    label="Fundamental predecessor rollback pointer",
                )
                if (
                    rollback_bytes != previous_pointer_bytes
                    or rollback_identity.sha256 != expected_sha256
                ):
                    raise SuccessorPromotionError(
                        "Fundamental predecessor rollback readback failed",
                        status="PROMOTION_UNCERTAIN",
                        journal_run_id=run_id,
                    ) from postcheck_exc
                try:
                    _validate_pointer_references(
                        canonical,
                        rollback_bytes,
                        expected_pointer_sha256=expected_sha256,
                        expected_manifest_sha256=_valid_sha256(
                            capture.predecessor.get("manifest_sha256")
                        ),
                        manifest_maximum_bytes=(
                            _MAX_PREDECESSOR_MANIFEST_JSON_BYTES
                        ),
                    )
                except Exception as rollback_ref_exc:
                    journal.append(
                        "TERMINAL",
                        {
                            "status": "PROMOTION_UNCERTAIN",
                            "reason": "PREDECESSOR_REFERENCE_READBACK_FAILED",
                            "postcheck_reason": type(postcheck_exc).__name__,
                        },
                    )
                    raise SuccessorPromotionError(
                        "predecessor pointer bytes were restored but its "
                        "immutable references failed readback",
                        status="PROMOTION_UNCERTAIN",
                        journal_run_id=run_id,
                    ) from rollback_ref_exc
                journal.append(
                    "ROLLBACK_COMMITTED",
                    {
                        "generation_id": capture.generation_id,
                        "pointer_sha256": expected_sha256,
                        "rolled_back_from_sha256": candidate_sha256,
                    },
                )
                journal.append(
                    "TERMINAL",
                    {
                        "status": "ROLLED_BACK",
                        "reason": type(postcheck_exc).__name__,
                        "orphan_generation_path": str(final_root),
                    },
                )
                raise SuccessorPromotionError(
                    "successor postcheck failed; exact predecessor bytes restored",
                    status="ROLLED_BACK",
                    journal_run_id=run_id,
                ) from postcheck_exc
            journal.append(
                "POSTCHECK_PASSED",
                {
                    "generation_id": capture.generation_id,
                    "pointer_sha256": candidate_sha256,
                    "generation_aggregate_sha256": plan[
                        "generation_aggregate_sha256"
                    ],
                },
            )
            journal.append(
                "TERMINAL",
                {
                    "status": "SUCCESS",
                    "generation_id": capture.generation_id,
                    "pointer_sha256": candidate_sha256,
                },
            )
        return {
            **plan,
            "status": "OK",
            "execute": True,
            "promoted": True,
            "journal_run_id": run_id,
            "journal_root": str(journal.root),
        }
    except Exception as exc:
        if not journal.terminal:
            pointer_state = _pointer_state(
                canonical,
                predecessor_sha256=expected_sha256,
                candidate_sha256=candidate_sha256,
            )
            if installed and pointer_state == "CANDIDATE":
                # A write/record failure can happen after the atomic pointer
                # replacement but before CAS_COMMITTED (or a later phase) is
                # durable.  Keep the journal open so recovery can revalidate
                # the installed generation under the same three locks.  A
                # terminal record here would strand an active, unchecked
                # candidate and make the recovery API refuse to act.
                if (
                    isinstance(exc, SuccessorPromotionError)
                    and exc.journal_run_id
                ):
                    raise
                raise SuccessorPromotionError(
                    "successor candidate is active but promotion did not "
                    "finish; recovery is required",
                    status="PROMOTION_UNCERTAIN",
                    journal_run_id=run_id,
                ) from exc
            status = (
                "PROMOTION_UNCERTAIN"
                if cas_committed and pointer_state not in {"CANDIDATE", "PREDECESSOR"}
                else "BLOCKED_ORPHAN"
                if installed
                else "BLOCKED"
            )
            journal.append(
                "TERMINAL",
                {
                    "status": status,
                    "reason": type(exc).__name__,
                    "pointer_state": pointer_state,
                    **(
                        {"orphan_generation_path": str(final_root)}
                        if installed
                        else {}
                    ),
                },
            )
        if isinstance(exc, SuccessorPromotionError) and exc.journal_run_id:
            raise
        raise SuccessorPromotionError(
            str(exc),
            status=(
                str(journal.records[-1]["details"].get("status") or "BLOCKED")
                if journal.records
                else "BLOCKED"
            ),
            journal_run_id=run_id,
        ) from exc


def _load_install_intent(journal: _Journal) -> dict[str, Any]:
    intents = [record for record in journal.records if record["phase"] == "INSTALL_INTENT"]
    if len(intents) != 1:
        raise SuccessorPromotionError("journal INSTALL_INTENT is missing or duplicated")
    return _as_mapping(intents[0]["details"], label="INSTALL_INTENT details")


def _decode_journal_bytes(value: Any, *, label: str) -> bytes:
    try:
        return base64.b64decode(str(value).encode("ascii"), validate=True)
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise SuccessorPromotionError(f"{label} base64 is invalid") from exc


def _journal_pointer_paths(intent: Mapping[str, Any]) -> tuple[Path, Path]:
    market = _as_mapping(intent.get("market_binding"), label="journal market binding")
    pit = _as_mapping(intent.get("pit_binding"), label="journal PIT binding")
    return (
        _binding_live_pointer_path(market, label="journal market"),
        _binding_live_pointer_path(pit, label="journal PIT"),
    )


@contextmanager
def _recovery_locks(
    *,
    intent: Mapping[str, Any],
    canonical_root: Path,
    timeout_seconds: float,
) -> Iterator[None]:
    if isinstance(timeout_seconds, bool) or float(timeout_seconds) < 0:
        raise SuccessorPromotionError("lock timeout is invalid")
    market_pointer, pit_pointer = _journal_pointer_paths(intent)
    paths = (
        _lock_file_path(market_pointer.parent, MARKET_WRITER_LOCK_FILENAME),
        _lock_file_path(pit_pointer.parent, PIT_WRITER_LOCK_FILENAME),
        _lock_file_path(canonical_root, FUNDAMENTAL_PROMOTION_LOCK_FILENAME),
    )
    if len(set(paths)) != 3:
        raise SuccessorPromotionError("writer lock paths are not distinct")
    deadline = time.monotonic() + float(timeout_seconds)
    with ExitStack() as stack:
        for path in paths:
            stack.enter_context(_secure_timed_lock(path, deadline=deadline))
        yield


def _journal_generation_path(
    intent: Mapping[str, Any],
    canonical_root: Path,
) -> Path:
    generation_id = _safe_id(
        intent.get("generation_id"),
        label="journal generation_id",
    )
    final_root = _lexical_absolute(
        str(intent.get("intended_generation_path") or ""),
        label="journal generation path",
    )
    expected = (
        canonical_root / FUNDAMENTAL_GENERATIONS_DIRNAME / generation_id
    )
    if final_root != expected:
        raise SuccessorPromotionError(
            "journal generation path is outside the canonical target"
        )
    return final_root


def _installed_from_journal(
    intent: Mapping[str, Any],
    canonical_root: Path,
) -> bool:
    final_root = _journal_generation_path(intent, canonical_root)
    try:
        metadata = os.lstat(final_root)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SuccessorPromotionError("journal generation path is unsafe")
    _secure_existing_directory(
        final_root,
        label="journal generation path",
    )
    return True


def _validate_journal_installed_files(
    intent: Mapping[str, Any],
    canonical_root: Path,
) -> None:
    final_root = _journal_generation_path(intent, canonical_root)
    _secure_existing_directory(
        final_root,
        label="journal generation path",
    )
    manifest_identity = _stable_file_hash(
        final_root / "manifest.json",
        label="recovery manifest",
    )
    if manifest_identity.sha256 != _valid_sha256(intent.get("manifest_sha256")):
        raise SuccessorPromotionError("recovery manifest SHA256 mismatch")
    table_sha = _as_mapping(intent.get("table_sha256"), label="journal table SHA256")
    table_size = _as_mapping(intent.get("table_size"), label="journal table size")
    if set(table_sha) != set(FUNDAMENTAL_TABLES):
        raise SuccessorPromotionError("journal table SHA256 set mismatch")
    for table_name in FUNDAMENTAL_TABLES:
        identity = _stable_file_hash(
            final_root / f"{table_name}.parquet",
            label=f"recovery {table_name}",
        )
        if identity.sha256 != _valid_sha256(table_sha[table_name]):
            raise SuccessorPromotionError(f"recovery {table_name} SHA256 mismatch")
        if table_name in table_size and identity.size != int(table_size[table_name]):
            raise SuccessorPromotionError(f"recovery {table_name} size mismatch")
    provider = _scan_provider_files(final_root)
    if _canonical_sha256(_provider_fileset_payload(provider)) != _valid_sha256(
        intent.get("provider_fileset_sha256")
    ):
        raise SuccessorPromotionError("recovery provider evidence mismatch")


def _recovery_classification(
    journal: _Journal,
    canonical_root: Path,
) -> dict[str, Any]:
    phases = [record["phase"] for record in journal.records]
    terminal = journal.records[-1] if journal.terminal else None
    if terminal is not None:
        return {
            "status": str(terminal["details"].get("status") or "TERMINAL"),
            "action": "NONE",
            "terminal": True,
            "phases": phases,
        }
    if "INSTALL_INTENT" not in phases:
        return {
            "status": "ABANDONED_BEFORE_INSTALL",
            "action": "TERMINATE",
            "terminal": False,
            "phases": phases,
        }
    intent = _load_install_intent(journal)
    predecessor_sha = _valid_sha256(intent.get("predecessor_pointer_sha256"))
    candidate_sha = _valid_sha256(intent.get("candidate_pointer_sha256"))
    pointer_state = _pointer_state(
        canonical_root,
        predecessor_sha256=predecessor_sha,
        candidate_sha256=candidate_sha,
    )
    installed = _installed_from_journal(intent, canonical_root)
    if "POSTCHECK_PASSED" in phases:
        status = "SUCCESS"
        action = "TERMINATE_SUCCESS"
    elif "ROLLBACK_COMMITTED" in phases:
        status = "ROLLED_BACK"
        action = "TERMINATE_ROLLBACK"
    elif pointer_state == "CANDIDATE" and installed:
        status = "CAS_COMMIT_NEEDS_POSTCHECK"
        action = "POSTCHECK_OR_ROLLBACK"
    elif pointer_state == "PREDECESSOR" and installed:
        status = "ORPHAN_RETAINED"
        action = "TERMINATE_ORPHAN"
    elif pointer_state == "PREDECESSOR" and not installed:
        status = "ABANDONED_BEFORE_INSTALL"
        action = "TERMINATE"
    else:
        status = "PROMOTION_UNCERTAIN"
        action = "NO_BLIND_ROLLBACK"
    return {
        "status": status,
        "action": action,
        "terminal": False,
        "pointer_state": pointer_state,
        "generation_installed": installed,
        "phases": phases,
    }


def recover_successor_promotion(
    *,
    canonical_root: str | Path,
    journal_root: str | Path,
    journal_run_id: str,
    execute: bool,
    live_binding_validator: LiveBindingValidator | None = None,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Inspect or finalize an interrupted successor promotion journal.

    Recovery never deletes an installed orphan.  Rollback is attempted only
    when the current pointer is byte-for-byte the journal's intended candidate.
    """

    if not isinstance(execute, bool):
        raise SuccessorPromotionError("execute must be an explicit boolean")
    canonical = _secure_existing_directory(
        canonical_root,
        label="canonical Fundamental root",
    )
    journal = _Journal(journal_root, journal_run_id, create=False)
    if not journal.records:
        raise SuccessorPromotionError("promotion journal is empty")
    classification = _recovery_classification(journal, canonical)
    result = {
        **classification,
        "execute": execute,
        "journal_run_id": journal.run_id,
        "journal_root": str(journal.root),
    }
    if not execute or classification["terminal"]:
        return result
    if "INSTALL_INTENT" not in classification["phases"]:
        journal.append("TERMINAL", {"status": "ABANDONED_BEFORE_INSTALL"})
        return {**result, "status": "ABANDONED_BEFORE_INSTALL", "terminal": True}
    intent = _load_install_intent(journal)
    with _recovery_locks(
        intent=intent,
        canonical_root=canonical,
        timeout_seconds=lock_timeout_seconds,
    ):
        refreshed = _recovery_classification(journal, canonical)
        action = refreshed["action"]
        predecessor_sha = _valid_sha256(intent.get("predecessor_pointer_sha256"))
        candidate_sha = _valid_sha256(intent.get("candidate_pointer_sha256"))
        predecessor_bytes = _decode_journal_bytes(
            intent.get("predecessor_pointer_bytes_b64"),
            label="journal predecessor pointer",
        )
        candidate_bytes = _decode_journal_bytes(
            intent.get("candidate_pointer_bytes_b64"),
            label="journal candidate pointer",
        )
        if _sha256_bytes(predecessor_bytes) != predecessor_sha:
            raise SuccessorPromotionError("journal predecessor pointer SHA mismatch")
        if _sha256_bytes(candidate_bytes) != candidate_sha:
            raise SuccessorPromotionError("journal candidate pointer SHA mismatch")

        if action == "POSTCHECK_OR_ROLLBACK":
            try:
                _validate_journal_installed_files(intent, canonical)
                current_bytes, current_identity = _stable_small_bytes(
                    canonical / FUNDAMENTAL_POINTER_FILENAME,
                    label="recovery candidate pointer",
                )
                if current_bytes != candidate_bytes or current_identity.sha256 != candidate_sha:
                    raise SuccessorPromotionError("recovery candidate pointer drifted")
                _validate_pointer_references(
                    canonical,
                    current_bytes,
                    expected_pointer_sha256=candidate_sha,
                    expected_manifest_sha256=_valid_sha256(
                        intent.get("manifest_sha256")
                    ),
                )
                market = _as_mapping(
                    intent.get("market_binding"),
                    label="journal market binding",
                )
                pit = _as_mapping(intent.get("pit_binding"), label="journal PIT binding")
                _validate_embedded_binding(market, label="journal captured market")
                _validate_embedded_binding(pit, label="journal captured PIT")
                if live_binding_validator is not None:
                    receipt = live_binding_validator(
                        _as_mapping(
                            intent.get("capture_identity"),
                            label="journal capture identity",
                        ),
                        "recovery_postcheck",
                    )
                    if receipt is False:
                        raise SuccessorPromotionError(
                            "recovery live binding validator rejected postcheck"
                        )
            except Exception as postcheck_exc:
                state = _pointer_state(
                    canonical,
                    predecessor_sha256=predecessor_sha,
                    candidate_sha256=candidate_sha,
                )
                if state != "CANDIDATE":
                    journal.append(
                        "TERMINAL",
                        {
                            "status": "PROMOTION_UNCERTAIN",
                            "pointer_state": state,
                            "reason": type(postcheck_exc).__name__,
                        },
                    )
                    return {
                        **result,
                        "status": "PROMOTION_UNCERTAIN",
                        "terminal": True,
                    }
                _atomic_write_private_bytes(
                    canonical / FUNDAMENTAL_POINTER_FILENAME,
                    predecessor_bytes,
                    label="recovery predecessor rollback pointer",
                )
                rollback_bytes, rollback_identity = _stable_small_bytes(
                    canonical / FUNDAMENTAL_POINTER_FILENAME,
                    label="recovery predecessor rollback pointer",
                )
                if rollback_bytes != predecessor_bytes or rollback_identity.sha256 != predecessor_sha:
                    journal.append(
                        "TERMINAL",
                        {
                            "status": "PROMOTION_UNCERTAIN",
                            "reason": "ROLLBACK_READBACK_FAILED",
                        },
                    )
                    return {
                        **result,
                        "status": "PROMOTION_UNCERTAIN",
                        "terminal": True,
                    }
                try:
                    _validate_pointer_references(
                        canonical,
                        rollback_bytes,
                        expected_pointer_sha256=predecessor_sha,
                        expected_manifest_sha256=_valid_sha256(
                            intent.get("predecessor_manifest_sha256")
                        ),
                        manifest_maximum_bytes=(
                            _MAX_PREDECESSOR_MANIFEST_JSON_BYTES
                        ),
                    )
                except Exception:
                    journal.append(
                        "TERMINAL",
                        {
                            "status": "PROMOTION_UNCERTAIN",
                            "reason": "PREDECESSOR_REFERENCE_READBACK_FAILED",
                        },
                    )
                    return {
                        **result,
                        "status": "PROMOTION_UNCERTAIN",
                        "terminal": True,
                    }
                journal.append(
                    "ROLLBACK_COMMITTED",
                    {
                        "pointer_sha256": predecessor_sha,
                        "rolled_back_from_sha256": candidate_sha,
                    },
                )
                journal.append(
                    "TERMINAL",
                    {
                        "status": "ROLLED_BACK",
                        "reason": type(postcheck_exc).__name__,
                        "orphan_generation_path": str(
                            intent.get("intended_generation_path") or ""
                        ),
                    },
                )
                return {**result, "status": "ROLLED_BACK", "terminal": True}
            if "CAS_COMMITTED" not in refreshed["phases"]:
                journal.append(
                    "CAS_COMMITTED",
                    {
                        "recovered": True,
                        "pointer_sha256": candidate_sha,
                        "previous_pointer_sha256": predecessor_sha,
                    },
                )
            journal.append(
                "POSTCHECK_PASSED",
                {"recovered": True, "pointer_sha256": candidate_sha},
            )
            journal.append(
                "TERMINAL",
                {"status": "SUCCESS", "recovered": True},
            )
            return {**result, "status": "SUCCESS", "terminal": True}

        if action == "TERMINATE_ORPHAN":
            journal.append(
                "TERMINAL",
                {
                    "status": "ORPHAN_RETAINED",
                    "orphan_generation_path": str(
                        intent.get("intended_generation_path") or ""
                    ),
                },
            )
            return {**result, "status": "ORPHAN_RETAINED", "terminal": True}
        if action == "TERMINATE_SUCCESS":
            journal.append("TERMINAL", {"status": "SUCCESS", "recovered": True})
            return {**result, "status": "SUCCESS", "terminal": True}
        if action == "TERMINATE_ROLLBACK":
            journal.append(
                "TERMINAL",
                {"status": "ROLLED_BACK", "recovered": True},
            )
            return {**result, "status": "ROLLED_BACK", "terminal": True}
        if action == "TERMINATE":
            journal.append("TERMINAL", {"status": refreshed["status"]})
            return {**result, "status": refreshed["status"], "terminal": True}
        journal.append(
            "TERMINAL",
            {
                "status": "PROMOTION_UNCERTAIN",
                "pointer_state": refreshed.get("pointer_state", "UNKNOWN"),
            },
        )
        return {**result, "status": "PROMOTION_UNCERTAIN", "terminal": True}


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "SUCCESSOR_PROMOTION_JOURNAL_SCHEMA",
    "SuccessorPromotionError",
    "preflight_successor_promotion",
    "promote_successor_generation",
    "recover_successor_promotion",
    "validate_successor_historical_evidence",
]
