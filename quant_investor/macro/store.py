"""Append-only immutable generation store for macro v2 observations."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from quant_investor.macro.contracts import MacroObservation, canonical_hash

DEFAULT_OBSERVATIONS_ROOT = Path("data/parquet/cn/macro_observations")
POINTER_FILENAME = "_latest.json"
GENERATIONS_DIRNAME = "_generations"
OBSERVATION_COLUMNS = tuple(MacroObservation.__dataclass_fields__)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_EXTENSIONS = frozenset({".html", ".bin"})
_GENERATION_V1 = "macro-observation-generation.v1"
_GENERATION_V2 = "macro-observation-generation.v2"
_OBSERVER_FLAGS = {
    "observer_only": True,
    "production_eligible": False,
    "applied": False,
}


class MacroObservationStoreError(RuntimeError):
    """Raised when observation storage cannot be validated or advanced."""


def _add_error_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_metadata(value: Any) -> dict[str, Any]:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise MacroObservationStoreError(
                    "macro_observation_evidence_metadata_key_invalid"
                )
            return {
                key: normalize(item[key])
                for key in sorted(item)
            }
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        raise MacroObservationStoreError(
            "macro_observation_evidence_metadata_value_invalid"
        )

    if not isinstance(value, Mapping):
        raise MacroObservationStoreError(
            "macro_observation_evidence_metadata_not_object"
        )
    normalized = normalize(value)
    assert isinstance(normalized, dict)
    try:
        _canonical_json_bytes(normalized)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise MacroObservationStoreError(
            "macro_observation_evidence_metadata_invalid"
        ) from exc
    return normalized


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(
    path: Path,
    *,
    unsafe_blocker: str,
    changed_blocker: str,
) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise MacroObservationStoreError(unsafe_blocker)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except MacroObservationStoreError:
        raise
    except OSError as exc:
        raise MacroObservationStoreError(unsafe_blocker) from exc
    try:
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise MacroObservationStoreError(changed_blocker)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(path)) != signature
        ):
            raise MacroObservationStoreError(changed_blocker)
        return b"".join(chunks)
    except MacroObservationStoreError:
        raise
    except OSError as exc:
        raise MacroObservationStoreError(changed_blocker) from exc
    finally:
        os.close(descriptor)


def _stable_directory_names(path: Path) -> list[str]:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise MacroObservationStoreError(
                "macro_observation_evidence_parent_unsafe"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except MacroObservationStoreError:
        raise
    except OSError as exc:
        raise MacroObservationStoreError(
            "macro_observation_evidence_parent_unsafe"
        ) from exc
    try:
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise MacroObservationStoreError(
                "macro_observation_evidence_directory_changed_during_read"
            )
        names = sorted(os.listdir(descriptor))
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(path)) != signature
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_directory_changed_during_read"
            )
        return names
    except MacroObservationStoreError:
        raise
    except OSError as exc:
        raise MacroObservationStoreError(
            "macro_observation_evidence_directory_changed_during_read"
        ) from exc
    finally:
        os.close(descriptor)


def _normalize_evidence_inputs(
    incoming: Iterable[Mapping[str, Any]],
    *,
    evidence_bytes: Mapping[str, bytes] | None,
    evidence_metadata: Mapping[str, Mapping[str, Any]] | None,
    observation_evidence: Mapping[str, Iterable[str]] | None,
) -> tuple[
    dict[str, bytes],
    list[dict[str, Any]],
    dict[str, list[str]],
]:
    enabled = any(
        value is not None
        for value in (
            evidence_bytes,
            evidence_metadata,
            observation_evidence,
        )
    )
    if not enabled:
        return {}, [], {}
    if (
        evidence_bytes is None
        or evidence_metadata is None
        or observation_evidence is None
    ):
        raise MacroObservationStoreError(
            "macro_observation_evidence_inputs_incomplete"
        )
    if not all(
        isinstance(value, Mapping)
        for value in (
            evidence_bytes,
            evidence_metadata,
            observation_evidence,
        )
    ):
        raise MacroObservationStoreError(
            "macro_observation_evidence_inputs_not_mappings"
        )

    incoming_hashes = {
        str(item.get("content_hash") or "") for item in incoming
    }
    if set(observation_evidence) != incoming_hashes:
        raise MacroObservationStoreError(
            "macro_observation_evidence_observation_set_mismatch"
        )
    if set(evidence_bytes) != set(evidence_metadata):
        raise MacroObservationStoreError(
            "macro_observation_evidence_metadata_set_mismatch"
        )

    bodies: dict[str, bytes] = {}
    files: list[dict[str, Any]] = []
    for digest in sorted(evidence_bytes):
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise MacroObservationStoreError(
                "macro_observation_evidence_sha256_invalid"
            )
        body = evidence_bytes[digest]
        if not isinstance(body, bytes):
            raise MacroObservationStoreError(
                "macro_observation_evidence_bytes_required"
            )
        if not body:
            raise MacroObservationStoreError(
                "macro_observation_evidence_empty"
            )
        if hashlib.sha256(body).hexdigest() != digest:
            raise MacroObservationStoreError(
                "macro_observation_evidence_hash_mismatch"
            )
        metadata = _canonical_metadata(evidence_metadata[digest])
        extension = str(metadata.get("extension") or "")
        if extension not in _EVIDENCE_EXTENSIONS:
            raise MacroObservationStoreError(
                "macro_observation_evidence_extension_invalid"
            )
        metadata_hash = hashlib.sha256(
            _canonical_json_bytes(metadata)
        ).hexdigest()
        bodies[digest] = body
        files.append(
            {
                "path": f"evidence/raw/{digest}{extension}",
                "sha256": digest,
                "size_bytes": len(body),
                "metadata": metadata,
                "metadata_sha256": metadata_hash,
            }
        )

    mappings: dict[str, list[str]] = {}
    referenced: set[str] = set()
    for content_hash in sorted(observation_evidence):
        if not _SHA256_RE.fullmatch(content_hash):
            raise MacroObservationStoreError(
                "macro_observation_evidence_content_hash_invalid"
            )
        raw_digests = observation_evidence[content_hash]
        if isinstance(raw_digests, (str, bytes)):
            raise MacroObservationStoreError(
                "macro_observation_evidence_list_invalid"
            )
        try:
            digests = sorted(set(raw_digests))
        except TypeError as exc:
            raise MacroObservationStoreError(
                "macro_observation_evidence_list_invalid"
            ) from exc
        if not digests or any(
            not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            for digest in digests
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_list_invalid"
            )
        if any(digest not in bodies for digest in digests):
            raise MacroObservationStoreError(
                "macro_observation_evidence_reference_missing"
            )
        mappings[content_hash] = digests
        referenced.update(digests)
    if referenced != set(bodies):
        raise MacroObservationStoreError(
            "macro_observation_evidence_unreferenced_file"
        )
    return bodies, files, mappings


def _evidence_file_map(
    files: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in files:
        digest = str(item.get("sha256") or "")
        if digest in result:
            raise MacroObservationStoreError(
                "macro_observation_evidence_file_duplicate"
            )
        result[digest] = dict(item)
    return result


def _validate_generation_evidence(
    generation: Path,
    manifest: Mapping[str, Any],
    *,
    include_bytes: bool = False,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, bytes],
]:
    raw_files = manifest.get("evidence_files")
    raw_mapping = manifest.get("observation_evidence")
    if (
        not isinstance(raw_files, list)
        or not raw_files
        or not isinstance(raw_mapping, Mapping)
    ):
        raise MacroObservationStoreError(
            "macro_observation_evidence_manifest_shape_invalid"
        )
    if int(manifest.get("evidence_file_count", -1)) != len(raw_files):
        raise MacroObservationStoreError(
            "macro_observation_evidence_file_count_mismatch"
        )

    evidence_directory = generation / "evidence"
    raw_directory = evidence_directory / "raw"
    for directory in (evidence_directory, raw_directory):
        try:
            directory_metadata = os.lstat(directory)
        except OSError as exc:
            raise MacroObservationStoreError(
                "macro_observation_evidence_parent_unsafe"
            ) from exc
        if (
            stat.S_ISLNK(directory_metadata.st_mode)
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_parent_unsafe"
            )
    if _stable_directory_names(evidence_directory) != ["raw"]:
        raise MacroObservationStoreError(
            "macro_observation_evidence_directory_set_mismatch"
        )

    files: list[dict[str, Any]] = []
    bodies: dict[str, bytes] = {}
    seen_paths: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
            "metadata",
            "metadata_sha256",
        }:
            raise MacroObservationStoreError(
                "macro_observation_evidence_file_shape_invalid"
            )
        digest = str(raw.get("sha256") or "")
        if not _SHA256_RE.fullmatch(digest):
            raise MacroObservationStoreError(
                "macro_observation_evidence_sha256_invalid"
            )
        metadata = _canonical_metadata(raw.get("metadata"))
        extension = str(metadata.get("extension") or "")
        if extension not in _EVIDENCE_EXTENSIONS:
            raise MacroObservationStoreError(
                "macro_observation_evidence_extension_invalid"
            )
        metadata_hash = hashlib.sha256(
            _canonical_json_bytes(metadata)
        ).hexdigest()
        if str(raw.get("metadata_sha256") or "") != metadata_hash:
            raise MacroObservationStoreError(
                "macro_observation_evidence_metadata_hash_mismatch"
            )
        expected_relative = f"evidence/raw/{digest}{extension}"
        relative_text = str(raw.get("path") or "")
        relative = Path(relative_text)
        if (
            relative_text != expected_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in relative_text
            or relative_text in seen_paths
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_path_unsafe"
            )
        seen_paths.add(relative_text)
        size = raw.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise MacroObservationStoreError(
                "macro_observation_evidence_size_invalid"
            )

        cursor = generation
        for part in relative.parts[:-1]:
            cursor = cursor / part
            try:
                directory_metadata = os.lstat(cursor)
            except OSError as exc:
                raise MacroObservationStoreError(
                    "macro_observation_evidence_parent_unsafe"
                ) from exc
            if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
                directory_metadata.st_mode
            ):
                raise MacroObservationStoreError(
                    "macro_observation_evidence_parent_unsafe"
                )
        evidence_path = generation / relative
        try:
            file_metadata = os.lstat(evidence_path)
        except OSError as exc:
            raise MacroObservationStoreError(
                "macro_observation_evidence_file_unsafe"
            ) from exc
        if (
            stat.S_ISLNK(file_metadata.st_mode)
            or not stat.S_ISREG(file_metadata.st_mode)
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_file_unsafe"
            )
        if stat.S_IMODE(file_metadata.st_mode) != 0o600:
            raise MacroObservationStoreError(
                "macro_observation_evidence_permissions_unsafe"
            )
        body = _stable_file_bytes(
            evidence_path,
            unsafe_blocker="macro_observation_evidence_file_unsafe",
            changed_blocker=(
                "macro_observation_evidence_file_changed_during_read"
            ),
        )
        if len(body) != size:
            raise MacroObservationStoreError(
                "macro_observation_evidence_size_mismatch"
            )
        if hashlib.sha256(body).hexdigest() != digest:
            raise MacroObservationStoreError(
                "macro_observation_evidence_hash_mismatch"
            )
        normalized = {
            "path": relative_text,
            "sha256": digest,
            "size_bytes": size,
            "metadata": metadata,
            "metadata_sha256": metadata_hash,
        }
        files.append(normalized)
        if include_bytes:
            bodies[digest] = body
    expected_names = sorted(Path(item["path"]).name for item in files)
    if _stable_directory_names(raw_directory) != expected_names:
        raise MacroObservationStoreError(
            "macro_observation_evidence_directory_set_mismatch"
        )
    if files != sorted(files, key=lambda item: item["sha256"]):
        raise MacroObservationStoreError(
            "macro_observation_evidence_files_not_canonical"
        )
    if len(_evidence_file_map(files)) != len(files):
        raise MacroObservationStoreError(
            "macro_observation_evidence_file_duplicate"
        )
    evidence_set_hash = canonical_hash({"evidence_files": files})
    if manifest.get("evidence_set_sha256") != evidence_set_hash:
        raise MacroObservationStoreError(
            "macro_observation_evidence_set_hash_mismatch"
        )

    mappings: dict[str, list[str]] = {}
    referenced: set[str] = set()
    file_hashes = {item["sha256"] for item in files}
    for content_hash in sorted(raw_mapping):
        if not isinstance(content_hash, str) or not _SHA256_RE.fullmatch(
            content_hash
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_content_hash_invalid"
            )
        raw_digests = raw_mapping[content_hash]
        if not isinstance(raw_digests, list):
            raise MacroObservationStoreError(
                "macro_observation_evidence_list_invalid"
            )
        if (
            not raw_digests
            or raw_digests != sorted(set(raw_digests))
            or any(
                not isinstance(digest, str)
                or digest not in file_hashes
                for digest in raw_digests
            )
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_list_invalid"
            )
        mappings[content_hash] = list(raw_digests)
        referenced.update(raw_digests)
    if referenced != file_hashes:
        raise MacroObservationStoreError(
            "macro_observation_evidence_reference_set_mismatch"
        )
    return files, mappings, bodies


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS invariant
                raise OSError("macro_observation_evidence_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _observer_flags_valid(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(key) is value for key, value in _OBSERVER_FLAGS.items())


def _absolute_root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise MacroObservationStoreError("macro_observation_root_unsafe")
    return raw if raw.is_absolute() else Path.cwd() / raw


def _read_root(value: str | Path) -> Path:
    root = _absolute_root(value)
    cursor = Path(root.anchor)
    for part in root.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise MacroObservationStoreError(
                "macro_observation_root_missing"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise MacroObservationStoreError(
                "macro_observation_root_unsafe"
            )
    return cursor.resolve(strict=True)


def _write_root(value: str | Path) -> Path:
    root = _absolute_root(value)
    cursor = Path(root.anchor)
    for part in root.parts[1:]:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, mode=0o700)
                metadata = os.lstat(cursor)
            except FileExistsError:
                metadata = os.lstat(cursor)
        except OSError as exc:
            raise MacroObservationStoreError(
                "macro_observation_root_unsafe"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise MacroObservationStoreError(
                "macro_observation_root_unsafe"
            )
    return cursor.resolve(strict=True)


def _json_document_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _run_precommit_validator(
    validator: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]],
        None,
    ]
    | None,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if validator is None:
        return
    # The validator must not be able to mutate the store's in-flight state.
    rows_copy = json.loads(
        json.dumps(
            list(rows),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    manifest_copy = json.loads(_json_document_bytes(manifest))
    validator(rows_copy, manifest_copy)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise MacroObservationStoreError("macro_observation_pointer_symlink_rejected")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _json_document_bytes(payload))


def _optional_pointer_bytes(root: Path) -> bytes | None:
    path = root / POINTER_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    return _stable_file_bytes(
        path,
        unsafe_blocker="macro_observation_pointer_unsafe",
        changed_blocker="macro_observation_pointer_changed_during_read",
    )


def _transactional_pointer_switch(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_previous: bytes | None,
) -> None:
    path = root / POINTER_FILENAME
    next_bytes = _json_document_bytes(payload)
    if _optional_pointer_bytes(root) != expected_previous:
        raise MacroObservationStoreError(
            "macro_observation_pointer_changed_before_switch"
        )
    try:
        _atomic_bytes(path, next_bytes)
    except Exception as original_error:
        try:
            current = _optional_pointer_bytes(root)
        except Exception as ownership_error:
            _add_error_note(
                original_error,
                "pointer rollback ownership check failed: "
                f"{ownership_error!r}",
            )
        else:
            if current == next_bytes:
                try:
                    if expected_previous is None:
                        os.unlink(path)
                        _fsync_directory(root)
                    else:
                        _atomic_bytes(path, expected_previous)
                except Exception as rollback_error:
                    _add_error_note(
                        original_error,
                        "pointer rollback failed: "
                        f"{rollback_error!r}",
                    )
            elif current != expected_previous:
                _add_error_note(
                    original_error,
                    "pointer rollback skipped because ownership was lost",
                )
        raise


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
    pointer_bytes = _optional_pointer_bytes(root)
    if pointer_bytes is None:
        return None
    try:
        payload = json.loads(pointer_bytes.decode("utf-8"))
    except MacroObservationStoreError:
        raise
    except Exception as exc:
        raise MacroObservationStoreError("macro_observation_pointer_invalid") from exc
    if not isinstance(payload, Mapping):
        raise MacroObservationStoreError("macro_observation_pointer_not_object")
    pointer = dict(payload)
    if pointer.get("schema_version") != "macro-observation-pointer.v1" or pointer.get("status") != "OK":
        raise MacroObservationStoreError("macro_observation_pointer_shape_invalid")
    if not _observer_flags_valid(pointer):
        raise MacroObservationStoreError(
            "macro_observation_pointer_observer_flags_invalid"
        )
    pointer["pointer_sha256"] = hashlib.sha256(pointer_bytes).hexdigest()
    return pointer


def pointer_sha256(root: str | Path) -> str:
    unresolved = Path(root).expanduser()
    path = unresolved / POINTER_FILENAME
    if not path.exists() and not path.is_symlink():
        return ""
    base = _read_root(unresolved)
    pointer_bytes = _optional_pointer_bytes(base)
    if pointer_bytes is None:  # pragma: no cover - raced after path probe
        return ""
    return hashlib.sha256(pointer_bytes).hexdigest()


def _resolve_generation(
    root: Path,
    pointer: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    generation_id = _safe_id(
        str(pointer.get("generation_id") or "").strip()
    )
    relative = Path(str(pointer.get("table_path") or ""))
    manifest_relative = Path(str(pointer.get("manifest_path") or ""))
    declared = str(pointer.get("parquet_sha256") or "")
    manifest_declared = str(pointer.get("manifest_sha256") or "")
    if not _observer_flags_valid(pointer):
        raise MacroObservationStoreError(
            "macro_observation_pointer_observer_flags_invalid"
        )
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
    table_bytes = _stable_file_bytes(
        table,
        unsafe_blocker="macro_observation_generation_table_unsafe",
        changed_blocker="macro_observation_generation_table_changed_during_read",
    )
    manifest_bytes = _stable_file_bytes(
        manifest_path,
        unsafe_blocker="macro_observation_generation_manifest_unsafe",
        changed_blocker="macro_observation_generation_manifest_changed_during_read",
    )
    if hashlib.sha256(table_bytes).hexdigest() != declared:
        raise MacroObservationStoreError("macro_observation_generation_hash_mismatch")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_declared:
        raise MacroObservationStoreError("macro_observation_manifest_hash_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise MacroObservationStoreError("macro_observation_manifest_invalid") from exc
    invalid_manifest = not isinstance(manifest, Mapping) or any(
        (
            manifest.get("schema_version")
            not in {_GENERATION_V1, _GENERATION_V2},
            manifest.get("status") != "OK",
            manifest.get("generation_id") != generation_id,
            manifest.get("parquet_sha256") != declared,
        )
    )
    if invalid_manifest:
        raise MacroObservationStoreError("macro_observation_manifest_shape_invalid")
    if not _observer_flags_valid(manifest):
        raise MacroObservationStoreError(
            "macro_observation_manifest_observer_flags_invalid"
        )
    if manifest.get("schema_version") == _GENERATION_V2:
        _validate_generation_evidence(expected_parent, manifest)
    return table_bytes, dict(manifest)


def load_observations(
    root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    *,
    generation_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = _read_root(root)
    pointer = _strict_pointer(base)
    if pointer is None and not generation_id:
        raise MacroObservationStoreError("macro_observation_pointer_missing")
    if generation_id:
        safe_generation = _safe_id(generation_id)
        generation_root = Path(GENERATIONS_DIRNAME) / safe_generation
        manifest_path = base / generation_root / "manifest.json"
        try:
            manifest_bytes = _stable_file_bytes(
                manifest_path,
                unsafe_blocker="macro_observation_generation_missing",
                changed_blocker=(
                    "macro_observation_generation_manifest_changed_during_read"
                ),
            )
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except MacroObservationStoreError:
            raise
        except Exception as exc:
            raise MacroObservationStoreError("macro_observation_manifest_invalid") from exc
        pointer = {
            "schema_version": "macro-observation-pointer.v1",
            "status": "OK",
            "generation_id": safe_generation,
            "table_path": str(generation_root / "observations.parquet"),
            "manifest_path": str(generation_root / "manifest.json"),
            "parquet_sha256": manifest.get("parquet_sha256"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "content_set_hash": manifest.get("content_set_hash"),
            "row_count": manifest.get("row_count"),
            **_OBSERVER_FLAGS,
        }
    assert pointer is not None
    table_bytes, manifest = _resolve_generation(base, pointer)
    try:
        frame = pd.read_parquet(io.BytesIO(table_bytes))
    except Exception as exc:
        raise MacroObservationStoreError(
            "macro_observation_generation_table_invalid"
        ) from exc
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
    if manifest.get("schema_version") == _GENERATION_V2:
        raw_mapping = manifest.get("observation_evidence")
        assert isinstance(raw_mapping, Mapping)  # validated by resolver
        if not set(raw_mapping).issubset(
            {row["content_hash"] for row in rows}
        ):
            raise MacroObservationStoreError(
                "macro_observation_evidence_observation_hash_missing"
            )
        _validate_evidence_record_drift(rows, raw_mapping)
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


def _observation_source_record_identity(
    row: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_system") or ""),
        str(row.get("source_record_id") or ""),
        str(row.get("indicator_id") or ""),
        str(row.get("period_end") or ""),
    )


def _validate_evidence_record_drift(
    rows: Iterable[Mapping[str, Any]],
    observation_evidence: Mapping[str, Iterable[str]],
) -> None:
    evidence_by_identity: dict[tuple[str, str, str, str], tuple[str, ...]] = {}
    for row in rows:
        content_hash = str(row.get("content_hash") or "")
        if content_hash not in observation_evidence:
            continue
        evidence = tuple(sorted(observation_evidence[content_hash]))
        identity = _observation_source_record_identity(row)
        previous = evidence_by_identity.get(identity)
        if previous is not None and previous != evidence:
            raise MacroObservationStoreError(
                "macro_observation_official_source_record_evidence_drift"
            )
        evidence_by_identity[identity] = evidence


def _merge_evidence_files(
    existing_files: Iterable[Mapping[str, Any]],
    existing_bodies: Mapping[str, bytes],
    incoming_files: Iterable[Mapping[str, Any]],
    incoming_bodies: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    merged_files = _evidence_file_map(existing_files)
    merged_bodies = dict(existing_bodies)
    for raw in incoming_files:
        item = dict(raw)
        digest = str(item.get("sha256") or "")
        previous = merged_files.get(digest)
        if previous is not None and previous != item:
            raise MacroObservationStoreError(
                "macro_observation_evidence_metadata_conflict"
            )
        body = incoming_bodies[digest]
        previous_body = merged_bodies.get(digest)
        if previous_body is not None and previous_body != body:
            raise MacroObservationStoreError(
                "macro_observation_evidence_content_conflict"
            )
        merged_files[digest] = item
        merged_bodies[digest] = body
    files = [merged_files[digest] for digest in sorted(merged_files)]
    return files, merged_bodies


def _merge_observation_evidence(
    existing: Mapping[str, Iterable[str]],
    incoming: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    merged = {
        content_hash: sorted(set(digests))
        for content_hash, digests in existing.items()
    }
    for content_hash, digests in incoming.items():
        normalized = sorted(set(digests))
        previous = merged.get(content_hash)
        if previous is not None and previous != normalized:
            raise MacroObservationStoreError(
                "macro_observation_official_source_record_evidence_drift"
            )
        merged[content_hash] = normalized
    return {key: merged[key] for key in sorted(merged)}


def publish_observations(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    run_id: str,
    expected_pointer_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    evidence_bytes: Mapping[str, bytes] | None = None,
    evidence_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    observation_evidence: Mapping[str, Iterable[str]] | None = None,
    precommit_validator: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]],
        None,
    ]
    | None = None,
) -> dict[str, Any]:
    base = _write_root(root)
    generation_id = _safe_id(run_id)
    incoming = _normalize_rows(observations)
    (
        incoming_evidence_bodies,
        incoming_evidence_files,
        incoming_observation_evidence,
    ) = _normalize_evidence_inputs(
        incoming,
        evidence_bytes=evidence_bytes,
        evidence_metadata=evidence_metadata,
        observation_evidence=observation_evidence,
    )
    if not incoming:
        return {"status": "no_update", "promoted": False, "reason": "empty_input"}

    with _locked(base):
        previous_pointer_bytes = _optional_pointer_bytes(base)
        current_sha = (
            hashlib.sha256(previous_pointer_bytes).hexdigest()
            if previous_pointer_bytes is not None
            else ""
        )
        if expected_pointer_sha256 is not None and current_sha != expected_pointer_sha256:
            raise MacroObservationStoreError("macro_observation_pointer_cas_mismatch")
        current_pointer = _strict_pointer(base)
        if (
            (current_pointer is None) != (previous_pointer_bytes is None)
            or (
                current_pointer is not None
                and current_pointer.get("pointer_sha256") != current_sha
            )
        ):
            raise MacroObservationStoreError(
                "macro_observation_pointer_changed_during_publish"
            )
        if current_pointer is None:
            existing, pointer = [], {}
        else:
            existing, pointer = load_observations(base)
        existing_evidence_files: list[dict[str, Any]] = []
        existing_observation_evidence: dict[str, list[str]] = {}
        existing_evidence_bodies: dict[str, bytes] = {}
        current_manifest = pointer.get("generation_manifest")
        if (
            isinstance(current_manifest, Mapping)
            and current_manifest.get("schema_version") == _GENERATION_V2
        ):
            current_generation = (
                base
                / GENERATIONS_DIRNAME
                / _safe_id(str(pointer.get("generation_id") or ""))
            )
            (
                existing_evidence_files,
                existing_observation_evidence,
                existing_evidence_bodies,
            ) = _validate_generation_evidence(
                current_generation,
                current_manifest,
                include_bytes=True,
            )
        combined = _normalize_rows([*existing, *incoming])
        existing_hashes = {row["content_hash"] for row in existing}
        new_content_hashes = {
            row["content_hash"] for row in combined
        } - existing_hashes
        if (
            isinstance(current_manifest, Mapping)
            and current_manifest.get("schema_version") == _GENERATION_V2
            and not new_content_hashes.issubset(
                incoming_observation_evidence
            )
        ):
            raise MacroObservationStoreError(
                "macro_observation_v2_new_rows_require_evidence"
            )
        merged_evidence_files, merged_evidence_bodies = _merge_evidence_files(
            existing_evidence_files,
            existing_evidence_bodies,
            incoming_evidence_files,
            incoming_evidence_bodies,
        )
        merged_observation_evidence = _merge_observation_evidence(
            existing_observation_evidence,
            incoming_observation_evidence,
        )
        if set(merged_evidence_bodies) != {
            digest
            for digests in merged_observation_evidence.values()
            for digest in digests
        }:
            raise MacroObservationStoreError(
                "macro_observation_evidence_reference_set_mismatch"
            )
        _validate_evidence_record_drift(
            combined,
            merged_observation_evidence,
        )
        evidence_changed = (
            merged_evidence_files != existing_evidence_files
            or merged_observation_evidence != existing_observation_evidence
        )
        if len(combined) == len(existing) and not evidence_changed:
            _run_precommit_validator(
                precommit_validator,
                combined,
                (
                    current_manifest
                    if isinstance(current_manifest, Mapping)
                    else {}
                ),
            )
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
            added_hashes = sorted(new_content_hashes)
            has_evidence = bool(merged_evidence_files)
            manifest = {
                "schema_version": (
                    _GENERATION_V2 if has_evidence else _GENERATION_V1
                ),
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
                **_OBSERVER_FLAGS,
                "metadata": dict(metadata or {}),
            }
            if has_evidence:
                evidence_directory = staging / "evidence"
                raw_directory = evidence_directory / "raw"
                evidence_directory.mkdir(mode=0o700)
                os.chmod(evidence_directory, 0o700)
                raw_directory.mkdir(mode=0o700)
                os.chmod(raw_directory, 0o700)
                for item in merged_evidence_files:
                    digest = str(item["sha256"])
                    evidence_path = staging / str(item["path"])
                    _write_private_bytes(
                        evidence_path,
                        merged_evidence_bodies[digest],
                    )
                _fsync_directory(raw_directory)
                _fsync_directory(evidence_directory)
                manifest.update(
                    {
                        "evidence_file_count": len(
                            merged_evidence_files
                        ),
                        "evidence_files": merged_evidence_files,
                        "evidence_set_sha256": canonical_hash(
                            {"evidence_files": merged_evidence_files}
                        ),
                        "observation_evidence": (
                            merged_observation_evidence
                        ),
                    }
                )
            manifest_path = staging / "manifest.json"
            _atomic_json(manifest_path, manifest)
            if has_evidence:
                (
                    readback_files,
                    readback_mapping,
                    readback_bodies,
                ) = _validate_generation_evidence(
                    staging,
                    manifest,
                    include_bytes=True,
                )
                if (
                    readback_files != merged_evidence_files
                    or readback_mapping != merged_observation_evidence
                    or readback_bodies != merged_evidence_bodies
                ):
                    raise MacroObservationStoreError(
                        "macro_observation_evidence_readback_mismatch"
                    )
            manifest_hash = _sha256(manifest_path)
            _fsync_directory(staging)
            os.replace(staging, final)
            if has_evidence:
                (
                    final_files,
                    final_mapping,
                    final_bodies,
                ) = _validate_generation_evidence(
                    final,
                    manifest,
                    include_bytes=True,
                )
                if (
                    final_files != merged_evidence_files
                    or final_mapping != merged_observation_evidence
                    or final_bodies != merged_evidence_bodies
                ):
                    raise MacroObservationStoreError(
                        "macro_observation_evidence_readback_mismatch"
                    )
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
                **_OBSERVER_FLAGS,
                "metadata": dict(metadata or {}),
            }
            _run_precommit_validator(
                precommit_validator,
                combined,
                manifest,
            )
            _transactional_pointer_switch(
                base,
                next_pointer,
                expected_previous=previous_pointer_bytes,
            )
            return {
                **next_pointer,
                "pointer_sha256": hashlib.sha256(
                    _json_document_bytes(next_pointer)
                ).hexdigest(),
                "generation_manifest": manifest,
                "promoted": True,
            }
        except Exception as original_error:
            if staging.exists():
                try:
                    shutil.rmtree(staging)
                except Exception as cleanup_error:
                    _add_error_note(
                        original_error,
                        f"staging cleanup failed: {cleanup_error!r}",
                    )
            try:
                owns_previous_pointer = (
                    _optional_pointer_bytes(base) == previous_pointer_bytes
                )
            except Exception as ownership_error:
                _add_error_note(
                    original_error,
                    "generation cleanup ownership check failed: "
                    f"{ownership_error!r}",
                )
                owns_previous_pointer = False
            if final.exists() and owns_previous_pointer:
                try:
                    shutil.rmtree(final)
                except Exception as cleanup_error:
                    _add_error_note(
                        original_error,
                        f"generation cleanup failed: {cleanup_error!r}",
                    )
            raise


__all__ = [
    "DEFAULT_OBSERVATIONS_ROOT",
    "MacroObservationStoreError",
    "load_observations",
    "pointer_sha256",
    "publish_observations",
]
