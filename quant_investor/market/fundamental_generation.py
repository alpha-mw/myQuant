"""Atomic generation storage for the CN PIT fundamental mart."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from copy import deepcopy
from dataclasses import dataclass
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
FUNDAMENTAL_RAW_TABLES = (
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "daily_basic",
    "forecast",
)
PRIMARY_PROVENANCE_SCHEMA_VERSION = "cn-fundamental-primary-provenance.v1"


class FundamentalGenerationError(ValueError):
    """Raised when a fundamental generation cannot be read or published."""


_PRIMARY_GENERATION_CAPABILITY = object()


@dataclass(frozen=True)
class _PrimaryGenerationAttestation:
    capability: object
    source: str
    provider_manifest_sha256: str
    raw_table_fingerprints: tuple[tuple[str, str], ...]
    metadata_sha256: str
    table_fingerprints: tuple[tuple[str, str], ...]


def _metadata_sha256(metadata: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(
            "fundamental generation metadata is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalGenerationError(
            "fundamental generation tables must contain pandas DataFrames"
        )
    digest = hashlib.sha256()
    schema = [
        {"position": index, "name": repr(column), "dtype": str(dtype)}
        for index, (column, dtype) in enumerate(zip(frame.columns, frame.dtypes))
    ]
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    try:
        digest.update(
            pd.util.hash_pandas_object(frame, index=True, categorize=True)
            .to_numpy(dtype="uint64", copy=False)
            .tobytes()
        )
    except (TypeError, ValueError) as exc:
        raise FundamentalGenerationError(
            "fundamental generation table fingerprint failed"
        ) from exc
    return digest.hexdigest()


def _table_fingerprints(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, str], ...]:
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental publish table set mismatch")
    return tuple(
        (table_name, _frame_fingerprint(tables[table_name]))
        for table_name in FUNDAMENTAL_TABLES
    )


def _issue_primary_generation_attestation(
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    source: str,
    provider_manifest_sha256: str,
    raw_table_fingerprints: tuple[tuple[str, str], ...],
) -> _PrimaryGenerationAttestation:
    if str(metadata.get("source_priority") or "").strip() != "tushare_primary":
        raise FundamentalGenerationError(
            "primary generation attestation requires tushare_primary metadata"
        )
    resolved_source = str(source or "").strip()
    if resolved_source not in {"live_tushare", "live_tushare_partial"}:
        raise FundamentalGenerationError(
            "primary generation attestation requires live Tushare source"
        )
    if str(metadata.get("provider_status") or "").strip() != resolved_source:
        raise FundamentalGenerationError(
            "primary generation provider status/source mismatch"
        )
    if (
        str(metadata.get("source_provenance") or "").strip()
        != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "primary generation source provenance is not live Tushare"
        )
    if metadata.get("gate2_passed") is not True:
        raise FundamentalGenerationError(
            "primary generation requires a passed readiness gate"
        )
    manifest_hash = str(provider_manifest_sha256 or "").strip().lower()
    if not _valid_sha256(manifest_hash):
        raise FundamentalGenerationError(
            "primary generation provider manifest hash is invalid"
        )
    provider_manifest = dict(metadata.get("provider_manifest", {}) or {})
    if _metadata_sha256(provider_manifest) != manifest_hash:
        raise FundamentalGenerationError(
            "primary generation provider manifest hash mismatch"
        )
    raw_fingerprints = tuple(raw_table_fingerprints)
    if not _valid_named_fingerprints(
        raw_fingerprints,
        expected_names=FUNDAMENTAL_RAW_TABLES,
    ):
        raise FundamentalGenerationError(
            "primary generation raw table fingerprints are invalid"
        )
    return _PrimaryGenerationAttestation(
        capability=_PRIMARY_GENERATION_CAPABILITY,
        source=resolved_source,
        provider_manifest_sha256=manifest_hash,
        raw_table_fingerprints=raw_fingerprints,
        metadata_sha256=_metadata_sha256(metadata),
        table_fingerprints=_table_fingerprints(tables),
    )


def _primary_generation_attestation_matches(
    attestation: _PrimaryGenerationAttestation | None,
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> bool:
    if not isinstance(attestation, _PrimaryGenerationAttestation):
        return False
    try:
        return bool(
            attestation.capability is _PRIMARY_GENERATION_CAPABILITY
            and attestation.metadata_sha256 == _metadata_sha256(metadata)
            and attestation.table_fingerprints == _table_fingerprints(tables)
        )
    except FundamentalGenerationError:
        return False


def _valid_sha256(value: str) -> bool:
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_named_fingerprints(
    values: tuple[tuple[str, str], ...],
    *,
    expected_names: tuple[str, ...],
) -> bool:
    return bool(
        tuple(name for name, _digest in values) == expected_names
        and all(_valid_sha256(str(digest)) for _name, digest in values)
    )


def _primary_provenance_envelope(
    attestation: _PrimaryGenerationAttestation,
    *,
    metadata: Mapping[str, Any],
    table_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": PRIMARY_PROVENANCE_SCHEMA_VERSION,
        "status": "verified_live_tushare",
        "source": attestation.source,
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "provider_manifest_sha256": attestation.provider_manifest_sha256,
        "metadata_sha256": _metadata_sha256(metadata),
        "raw_table_fingerprints": dict(attestation.raw_table_fingerprints),
        "output_frame_fingerprints": dict(attestation.table_fingerprints),
        "output_parquet_sha256": {
            table_name: str(table_manifest[table_name].get("sha256") or "")
            for table_name in FUNDAMENTAL_TABLES
        },
    }
    body["envelope_sha256"] = _metadata_sha256(body)
    return body


def _verify_primary_provenance(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    pointer_metadata = dict(pointer.get("metadata", {}) or {})
    manifest_metadata = dict(manifest.get("metadata", {}) or {})
    pointer_priority = str(pointer_metadata.get("source_priority") or "").strip()
    manifest_priority = str(manifest_metadata.get("source_priority") or "").strip()
    pointer_envelope = pointer.get("primary_provenance")
    manifest_envelope = manifest.get("primary_provenance")
    if pointer_priority != "tushare_primary" and manifest_priority != "tushare_primary":
        if pointer_envelope is not None or manifest_envelope is not None:
            raise FundamentalGenerationError(
                "non-primary fundamental generation has primary provenance"
            )
        return False
    if pointer_priority != manifest_priority:
        raise FundamentalGenerationError(
            "fundamental primary source priority mismatch"
        )
    if not isinstance(pointer_envelope, Mapping) or not isinstance(
        manifest_envelope,
        Mapping,
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance envelope missing"
        )
    envelope = dict(manifest_envelope)
    if dict(pointer_envelope) != envelope:
        raise FundamentalGenerationError(
            "fundamental primary provenance pointer/manifest mismatch"
        )
    envelope_hash = str(envelope.pop("envelope_sha256", "")).strip().lower()
    if not _valid_sha256(envelope_hash) or _metadata_sha256(envelope) != envelope_hash:
        raise FundamentalGenerationError(
            "fundamental primary provenance envelope hash mismatch"
        )
    if (
        envelope.get("schema_version") != PRIMARY_PROVENANCE_SCHEMA_VERSION
        or envelope.get("status") != "verified_live_tushare"
        or str(envelope.get("source") or "").strip()
        not in {"live_tushare", "live_tushare_partial"}
        or envelope.get("source_priority") != "tushare_primary"
        or envelope.get("source_provenance") != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance contract mismatch"
        )
    if str(manifest_metadata.get("provider_status") or "").strip() != str(
        envelope.get("source") or ""
    ).strip():
        raise FundamentalGenerationError(
            "fundamental primary provenance source mismatch"
        )
    if (
        str(manifest_metadata.get("source_provenance") or "").strip()
        != "live_tushare_explicit"
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance source lineage mismatch"
        )
    if (
        manifest_metadata.get("gate2_passed") is not True
        or pointer_metadata.get("gate2_passed") is not True
    ):
        raise FundamentalGenerationError(
            "fundamental primary provenance readiness gate mismatch"
        )
    provider_manifest_hash = str(
        envelope.get("provider_manifest_sha256") or ""
    ).strip().lower()
    if (
        not _valid_sha256(provider_manifest_hash)
        or _metadata_sha256(
            dict(manifest_metadata.get("provider_manifest", {}) or {})
        )
        != provider_manifest_hash
    ):
        raise FundamentalGenerationError(
            "fundamental primary provider manifest hash mismatch"
        )
    if str(envelope.get("metadata_sha256") or "").strip().lower() != (
        _metadata_sha256(manifest_metadata)
    ):
        raise FundamentalGenerationError(
            "fundamental primary metadata hash mismatch"
        )
    raw_fingerprints = tuple(
        (str(name), str(digest).strip().lower())
        for name, digest in dict(
            envelope.get("raw_table_fingerprints", {}) or {}
        ).items()
    )
    if not _valid_named_fingerprints(
        raw_fingerprints,
        expected_names=FUNDAMENTAL_RAW_TABLES,
    ):
        raise FundamentalGenerationError(
            "fundamental primary raw fingerprints mismatch"
        )
    output_frame_fingerprints = tuple(
        (str(name), str(digest).strip().lower())
        for name, digest in dict(
            envelope.get("output_frame_fingerprints", {}) or {}
        ).items()
    )
    if not _valid_named_fingerprints(
        output_frame_fingerprints,
        expected_names=FUNDAMENTAL_TABLES,
    ):
        raise FundamentalGenerationError(
            "fundamental primary output fingerprints mismatch"
        )
    expected_parquet_hashes = {
        table_name: str(
            dict(manifest.get("tables", {}) or {})
            .get(table_name, {})
            .get("sha256", "")
        ).strip().lower()
        for table_name in FUNDAMENTAL_TABLES
    }
    if dict(envelope.get("output_parquet_sha256", {}) or {}) != (
        expected_parquet_hashes
    ) or not all(_valid_sha256(value) for value in expected_parquet_hashes.values()):
        raise FundamentalGenerationError(
            "fundamental primary parquet provenance mismatch"
        )
    return True


def _tables_claim_primary(
    tables: Mapping[str, pd.DataFrame],
) -> bool:
    for table_name in FUNDAMENTAL_TABLES:
        frame = tables.get(table_name)
        if not isinstance(frame, pd.DataFrame) or "source_priority" not in frame.columns:
            continue
        priorities = {
            str(value).strip()
            for value in frame["source_priority"].dropna().tolist()
        }
        if "tushare_primary" in priorities:
            return True
    return False


def fundamental_data_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path.parent.parent
    if path.name in FUNDAMENTAL_TABLES:
        return path.parent
    return path


def _absolute_data_root(root: str | Path) -> Path:
    raw = fundamental_data_root(root)
    if ".." in raw.parts:
        raise FundamentalGenerationError(
            "fundamental root must not contain parent traversal"
        )
    return raw if raw.is_absolute() else Path.cwd().resolve(strict=True) / raw


def _read_data_root(root: str | Path) -> Path:
    absolute = _absolute_data_root(root)
    cursor = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            return cursor.joinpath(*parts[index + 1:])
        except OSError as exc:
            raise FundamentalGenerationError(
                "fundamental root unreadable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root ancestor symlink rejected"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root is not a directory"
            )
    return cursor.resolve(strict=True)


def _write_data_root(root: str | Path) -> Path:
    absolute = _absolute_data_root(root)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
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
            raise FundamentalGenerationError(
                "fundamental root unwritable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root ancestor symlink rejected"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise FundamentalGenerationError(
                "fundamental root is not a directory"
            )
    return cursor.resolve(strict=True)


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


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(path: Path) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FundamentalGenerationError(
                f"fundamental artifact is not a regular file: {path}"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise FundamentalGenerationError(
            f"fundamental artifact unreadable: {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(before):
            raise FundamentalGenerationError(
                f"fundamental artifact changed during open: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        signature = _file_signature(before)
        if (
            _file_signature(after) != signature
            or _file_signature(current) != signature
        ):
            raise FundamentalGenerationError(
                f"fundamental artifact changed during read: {path}"
            )
        return b"".join(chunks), signature
    except OSError as exc:
        raise FundamentalGenerationError(
            f"fundamental artifact changed during read: {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise FundamentalGenerationError(
            f"invalid fundamental {label} JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FundamentalGenerationError(
            f"fundamental {label} JSON object required"
        )
    return dict(value)


def _resolve_inside(base: Path, value: str, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise FundamentalGenerationError(f"{label} path is empty")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise FundamentalGenerationError(
            f"{label} must be a safe relative fundamental path"
        )
    cursor = base
    for part in candidate.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise FundamentalGenerationError(
                f"{label} path missing: {cursor}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FundamentalGenerationError(
                f"{label} path contains symlink: {cursor}"
            )
    resolved = (base / candidate).resolve(strict=True)
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise FundamentalGenerationError(f"{label} escapes fundamental root: {resolved}")
    return resolved


@lru_cache(maxsize=8)
def _validate_fundamental_pointer_cached(
    base_text: str,
    pointer_bytes: bytes,
    manifest_bytes: bytes,
    table_signatures: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    base = Path(base_text)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    payload = _json_object_from_bytes(pointer_bytes, label="pointer")
    if (
        payload.get("schema_version") != "cn-fundamental-pointer.v1"
        or payload.get("status") != "OK"
    ):
        raise FundamentalGenerationError("fundamental pointer status is not OK")
    generation_id = str(payload.get("generation_id", "")).strip()
    if not generation_id:
        raise FundamentalGenerationError("fundamental pointer generation_id missing")
    _resolve_inside(
        base,
        str(payload.get("manifest_path", "")),
        label="manifest",
    )
    manifest = _json_object_from_bytes(manifest_bytes, label="manifest")
    if (
        manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest.get("status") != "OK"
        or str(manifest.get("generation_id", "")) != generation_id
    ):
        raise FundamentalGenerationError("fundamental pointer/manifest generation mismatch")
    tables = dict(payload.get("tables", {}) or {})
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental pointer table set mismatch")
    signatures_by_name = {
        str(item[0]): tuple(int(value) for value in item[1:])
        for item in table_signatures
    }
    for table_name, table_value in tables.items():
        table_path = _resolve_inside(base, str(table_value), label=table_name)
        expected_hash = str(
            dict(manifest.get("tables", {}) or {})
            .get(table_name, {})
            .get("sha256", "")
        )
        table_bytes, observed_signature = _stable_file_bytes(table_path)
        if observed_signature != signatures_by_name.get(table_name):
            raise FundamentalGenerationError(
                f"fundamental table changed before validation: {table_name}"
            )
        actual_hash = hashlib.sha256(table_bytes).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise FundamentalGenerationError(
                f"fundamental table hash mismatch: {table_name}"
            )
    primary_provenance_verified = _verify_primary_provenance(payload, manifest)
    payload["primary_provenance_verified"] = primary_provenance_verified
    pointer_metadata = dict(payload.get("metadata", {}) or {})
    pointer_metadata["primary_provenance_verified"] = primary_provenance_verified
    payload["metadata"] = pointer_metadata
    payload["pointer_path"] = str(pointer_path)
    payload["manifest"] = manifest
    return payload


def load_fundamental_pointer(root: str | Path) -> dict[str, Any] | None:
    base = _read_data_root(root)
    pointer_path = base / FUNDAMENTAL_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    pointer_bytes, _pointer_signature = _stable_file_bytes(pointer_path)
    pointer_payload = _json_object_from_bytes(
        pointer_bytes,
        label="pointer",
    )
    manifest_path = _resolve_inside(
        base,
        str(pointer_payload.get("manifest_path", "")),
        label="manifest",
    )
    manifest_bytes, _manifest_signature = _stable_file_bytes(manifest_path)
    tables = dict(pointer_payload.get("tables", {}) or {})
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental pointer table set mismatch")
    table_signatures: list[tuple[Any, ...]] = []
    for table_name in FUNDAMENTAL_TABLES:
        table_path = _resolve_inside(
            base,
            str(tables.get(table_name, "")),
            label=table_name,
        )
        table_signatures.append(
            (table_name, *_file_signature(os.lstat(table_path)))
        )
    return deepcopy(
        _validate_fundamental_pointer_cached(
            str(base),
            pointer_bytes,
            manifest_bytes,
            tuple(table_signatures),
        )
    )


def load_fundamental_table(
    root: str | Path,
    table_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the same immutable bytes that satisfy the generation hash."""

    if table_name not in FUNDAMENTAL_TABLES:
        raise FundamentalGenerationError(
            f"unsupported fundamental table: {table_name}"
        )
    base = _read_data_root(root)
    pointer = load_fundamental_pointer(base)
    if pointer is None:
        raise FundamentalGenerationError("fundamental pointer missing")
    table_path = _resolve_inside(
        base,
        str(dict(pointer.get("tables", {}) or {}).get(table_name, "")),
        label=table_name,
    )
    table_bytes, _signature = _stable_file_bytes(table_path)
    table_manifest = dict(
        dict(pointer.get("manifest", {}) or {})
        .get("tables", {})
        .get(table_name, {})
        or {}
    )
    if hashlib.sha256(table_bytes).hexdigest() != str(
        table_manifest.get("sha256") or ""
    ):
        raise FundamentalGenerationError(
            f"fundamental table hash mismatch: {table_name}"
        )
    try:
        frame = pd.read_parquet(io.BytesIO(table_bytes))
    except Exception as exc:
        raise FundamentalGenerationError(
            f"fundamental table unreadable: {table_name}"
        ) from exc
    if "rows" in table_manifest and int(table_manifest["rows"]) != len(frame):
        raise FundamentalGenerationError(
            f"fundamental table row count mismatch: {table_name}"
        )
    if "columns" in table_manifest and list(
        table_manifest["columns"]
    ) != list(frame.columns):
        raise FundamentalGenerationError(
            f"fundamental table columns mismatch: {table_name}"
        )
    return frame, pointer


def resolve_fundamental_table_path(
    root: str | Path,
    table_name: str,
) -> Path:
    if Path(root).expanduser().suffix.lower() == ".parquet":
        return Path(root).expanduser()
    base = _read_data_root(root)
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
    _primary_attestation: _PrimaryGenerationAttestation | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    generation_id = _safe_generation_id(run_id)
    if set(tables) != set(FUNDAMENTAL_TABLES):
        raise FundamentalGenerationError("fundamental publish table set mismatch")
    source_priority = str(metadata.get("source_priority") or "").strip()
    if source_priority == "tushare_primary":
        if not _primary_generation_attestation_matches(
            _primary_attestation,
            tables=tables,
            metadata=metadata,
        ):
            raise FundamentalGenerationError(
                "tushare_primary generation requires an internal primary capability"
            )
    elif _primary_attestation is not None:
        raise FundamentalGenerationError(
            "primary generation capability cannot publish non-primary metadata"
        )
    elif _tables_claim_primary(tables):
        raise FundamentalGenerationError(
            "non-primary generation contains tushare_primary row claims"
        )
    base = _write_data_root(root)
    generations_root = _write_data_root(
        base / FUNDAMENTAL_GENERATIONS_DIRNAME
    )
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
        primary_provenance = (
            _primary_provenance_envelope(
                _primary_attestation,
                metadata=metadata,
                table_manifest=table_manifest,
            )
            if source_priority == "tushare_primary"
            and _primary_attestation is not None
            else None
        )
        manifest = {
            "schema_version": "cn-fundamental-generation.v1",
            "generation_id": generation_id,
            "status": "OK",
            "tables": table_manifest,
            "metadata": dict(metadata),
        }
        if primary_provenance is not None:
            manifest["primary_provenance"] = primary_provenance
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
        if primary_provenance is not None:
            pointer["primary_provenance"] = primary_provenance
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
    "load_fundamental_table",
    "publish_fundamental_generation",
    "resolve_fundamental_table_path",
]
