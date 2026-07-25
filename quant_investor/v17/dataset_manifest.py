"""Strict multipart dataset manifests for immutable v17 private sources.

The manifest is deliberately small: large Parquet/blob payloads remain in
content-addressed objects and are always hashed as streams.  This module does
not discover data, create sources, or grant authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import cmp_to_key
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from .contracts import (
    V17ContractError,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_identifier,
    require_nonempty_string,
)
from .semantic import canonical_json_bytes, require_sha256, validate_semantic_seal
from .storage import MAX_STREAM_OBJECT_BYTES, file_sha256

DATASET_MANIFEST_VERSION = "myquant.v17.dataset-manifest.v1"
DATASET_SCHEMA_DIGEST_VERSION = "myquant.v17.dataset-schema-digest.v1"
DATASET_CONTENT_SET_VERSION = "myquant.v17.dataset-content-set.v1"
DATASET_FORMATS = frozenset({"PARQUET", "BLOB"})
FORMAT_MEDIA_TYPES = {
    "PARQUET": "application/vnd.apache.parquet",
    "BLOB": "application/octet-stream",
}
FORMAT_SUFFIXES = {"PARQUET": "parquet", "BLOB": "blob"}

DATASET_MANIFEST_KEYS = frozenset(
    {
        "version",
        "dataset_id",
        "role",
        "format",
        "media_type",
        "schema",
        "primary_key",
        "partition_keys",
        "sort_keys",
        "shards",
        "total_row_count",
        "total_size_bytes",
        "content_set_sha256",
        "authority",
        "semantic_sha256",
    }
)
SCHEMA_ENTRY_KEYS = frozenset({"name", "logical_type", "nullable"})
SHARD_KEYS = frozenset(
    {
        "logical_name",
        "partition_values",
        "object_path",
        "byte_sha256",
        "size_bytes",
        "row_count",
        "min_key",
        "max_key",
        "schema_sha256",
    }
)

_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=-]{0,1023}$")


class V17DatasetManifestError(V17ContractError):
    """A dataset manifest or one of its immutable objects is invalid."""


def _array(raw: Any, *, label: str, maximum: int, nonempty: bool = False) -> list[Any]:
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise V17DatasetManifestError(f"{label} must be an array")
    values = list(raw)
    if nonempty and not values:
        raise V17DatasetManifestError(f"{label} must be nonempty")
    if len(values) > maximum:
        raise V17DatasetManifestError(f"{label} exceeds its fixed item limit")
    return values


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V17DatasetManifestError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result == 0:
        raise V17DatasetManifestError(f"{label} must be positive")
    return result


def _identifier_array(raw: Any, *, label: str) -> list[str]:
    values = _array(raw, label=label, maximum=64)
    normalized = [
        require_identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    ]
    if len(set(normalized)) != len(normalized):
        raise V17DatasetManifestError(f"{label} contains duplicates")
    return normalized


def _validate_schema(raw: Any) -> list[dict[str, Any]]:
    entries = _array(raw, label="schema", maximum=4096)
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise V17DatasetManifestError(f"schema[{index}] must be an object")
        require_exact_keys(entry, SCHEMA_ENTRY_KEYS, label=f"schema[{index}]")
        name = require_identifier(entry.get("name"), label=f"schema[{index}].name")
        if name in names:
            raise V17DatasetManifestError(f"duplicate schema field: {name}")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "logical_type": require_nonempty_string(
                    entry.get("logical_type"),
                    label=f"schema[{index}].logical_type",
                    max_chars=256,
                ),
                "nullable": require_bool(
                    entry.get("nullable"),
                    label=f"schema[{index}].nullable",
                ),
            }
        )
    return normalized


def dataset_schema_sha256(schema: Sequence[Mapping[str, Any]]) -> str:
    """Return the domain-separated digest for one canonical logical schema."""

    normalized = _validate_schema(schema)
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": DATASET_SCHEMA_DIGEST_VERSION,
                "schema": normalized,
            }
        )
    ).hexdigest()


def derive_dataset_object_path(byte_sha256: str, dataset_format: str) -> str:
    """Derive, never accept, the fixed repository-relative CAS object path."""

    digest = require_sha256(byte_sha256, label="dataset object byte SHA-256")
    if dataset_format not in DATASET_FORMATS:
        raise V17DatasetManifestError(f"unsupported dataset format: {dataset_format}")
    suffix = FORMAT_SUFFIXES[dataset_format]
    return f"data/private/v17_sources/objects/{digest[:2]}/{digest}.{suffix}"


def _logical_name(value: Any, *, label: str) -> str:
    name = require_nonempty_string(value, label=label, max_chars=1024)
    if "\\" in name or not _LOGICAL_NAME_RE.fullmatch(name):
        raise V17DatasetManifestError(f"{label} is not a canonical relative logical name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise V17DatasetManifestError(f"{label} is not a canonical relative logical name")
    if str(pure) != name:
        raise V17DatasetManifestError(f"{label} is not canonically normalized")
    return name


def _json_scalar(value: Any, *, label: str, allow_null: bool = True) -> Any:
    if value is None:
        if allow_null:
            return None
        raise V17DatasetManifestError(f"{label} cannot be null")
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise V17DatasetManifestError(f"{label} must be a finite JSON scalar")


def _key_tuple(raw: Any, *, label: str, length: int) -> list[Any]:
    values = _array(raw, label=label, maximum=64, nonempty=True)
    if len(values) != length:
        raise V17DatasetManifestError(f"{label} length must equal primary_key length")
    return [
        _json_scalar(value, label=f"{label}[{index}]", allow_null=False)
        for index, value in enumerate(values)
    ]


def _compare_scalar(left: Any, right: Any, *, label: str) -> int:
    left_numeric = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_numeric = isinstance(right, (int, float)) and not isinstance(right, bool)
    if left_numeric and right_numeric:
        return (left > right) - (left < right)
    if type(left) is not type(right) or not isinstance(left, (str, bool)):
        raise V17DatasetManifestError(f"{label} has incomparable key value types")
    return (left > right) - (left < right)


def _compare_key_tuples(left: Sequence[Any], right: Sequence[Any], *, label: str) -> int:
    if len(left) != len(right):
        raise V17DatasetManifestError(f"{label} key lengths differ")
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        compared = _compare_scalar(
            left_value,
            right_value,
            label=f"{label}[{index}]",
        )
        if compared:
            return compared
    return 0


def dataset_content_set_sha256(shards: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered complete set of immutable shard descriptors."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": DATASET_CONTENT_SET_VERSION,
                "shards": [dict(item) for item in shards],
            }
        )
    ).hexdigest()


def _validate_shards(
    raw: Any,
    *,
    dataset_format: str,
    partition_keys: Sequence[str],
    primary_key: Sequence[str],
    expected_schema_sha256: str,
) -> list[dict[str, Any]]:
    items = _array(raw, label="shards", maximum=100_000, nonempty=True)
    normalized: list[dict[str, Any]] = []
    logical_names: set[str] = set()
    object_paths: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise V17DatasetManifestError(f"shards[{index}] must be an object")
        require_exact_keys(item, SHARD_KEYS, label=f"shards[{index}]")
        logical_name = _logical_name(
            item.get("logical_name"), label=f"shards[{index}].logical_name"
        )
        if logical_name in logical_names:
            raise V17DatasetManifestError(f"duplicate shard logical_name: {logical_name}")
        logical_names.add(logical_name)

        partition_raw = item.get("partition_values")
        if not isinstance(partition_raw, Mapping):
            raise V17DatasetManifestError(f"shards[{index}].partition_values must be an object")
        require_exact_keys(
            partition_raw,
            set(partition_keys),
            label=f"shards[{index}].partition_values",
        )
        partition_values = {
            key: _json_scalar(
                partition_raw[key],
                label=f"shards[{index}].partition_values.{key}",
                allow_null=False,
            )
            for key in partition_keys
        }
        byte_sha256 = require_sha256(
            item.get("byte_sha256"),
            label=f"shards[{index}].byte_sha256",
        )
        expected_object_path = derive_dataset_object_path(byte_sha256, dataset_format)
        object_path = require_nonempty_string(
            item.get("object_path"),
            label=f"shards[{index}].object_path",
            max_chars=2048,
        )
        if object_path != expected_object_path:
            raise V17DatasetManifestError(
                f"shards[{index}].object_path is not derived from its byte SHA"
            )
        if object_path in object_paths:
            raise V17DatasetManifestError(f"duplicate shard object_path: {object_path}")
        object_paths.add(object_path)
        size_bytes = _positive_integer(
            item.get("size_bytes"),
            label=f"shards[{index}].size_bytes",
        )
        if size_bytes > MAX_STREAM_OBJECT_BYTES:
            raise V17DatasetManifestError(f"shards[{index}].size_bytes exceeds fixed limit")
        row_count = _nonnegative_integer(
            item.get("row_count"),
            label=f"shards[{index}].row_count",
        )
        shard_schema_sha256 = require_sha256(
            item.get("schema_sha256"),
            label=f"shards[{index}].schema_sha256",
        )
        if shard_schema_sha256 != expected_schema_sha256:
            raise V17DatasetManifestError(f"shards[{index}].schema_sha256 mismatch")

        if dataset_format == "PARQUET":
            if row_count == 0:
                raise V17DatasetManifestError(f"shards[{index}] Parquet shard must be nonempty")
            min_key = _key_tuple(
                item.get("min_key"),
                label=f"shards[{index}].min_key",
                length=len(primary_key),
            )
            max_key = _key_tuple(
                item.get("max_key"),
                label=f"shards[{index}].max_key",
                length=len(primary_key),
            )
            if (
                _compare_key_tuples(
                    min_key,
                    max_key,
                    label=f"shards[{index}]",
                )
                > 0
            ):
                raise V17DatasetManifestError(f"shards[{index}] min_key exceeds max_key")
        else:
            if row_count != 0 or item.get("min_key") is not None or item.get("max_key") is not None:
                raise V17DatasetManifestError(
                    f"shards[{index}] blob must have zero rows and null key bounds"
                )
            min_key = None
            max_key = None

        normalized.append(
            {
                "logical_name": logical_name,
                "partition_values": partition_values,
                "object_path": object_path,
                "byte_sha256": byte_sha256,
                "size_bytes": size_bytes,
                "row_count": row_count,
                "min_key": min_key,
                "max_key": max_key,
                "schema_sha256": shard_schema_sha256,
            }
        )

    def compare_canonical_shards(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
        left_partition = canonical_json_bytes(left["partition_values"])
        right_partition = canonical_json_bytes(right["partition_values"])
        if left_partition != right_partition:
            return (left_partition > right_partition) - (left_partition < right_partition)
        if dataset_format == "PARQUET":
            compared = _compare_key_tuples(
                left["min_key"],
                right["min_key"],
                label="canonical shard order",
            )
            if compared:
                return compared
        return (str(left["logical_name"]) > str(right["logical_name"])) - (
            str(left["logical_name"]) < str(right["logical_name"])
        )

    if normalized != sorted(normalized, key=cmp_to_key(compare_canonical_shards)):
        raise V17DatasetManifestError("shards are not in canonical deterministic order")

    if dataset_format == "PARQUET":
        groups: dict[bytes, list[dict[str, Any]]] = {}
        for item in normalized:
            groups.setdefault(canonical_json_bytes(item["partition_values"]), []).append(item)

        def compare_shards(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
            compared = _compare_key_tuples(
                left["min_key"],
                right["min_key"],
                label="shard range",
            )
            if compared:
                return compared
            return (str(left["logical_name"]) > str(right["logical_name"])) - (
                str(left["logical_name"]) < str(right["logical_name"])
            )

        for partition, group in groups.items():
            range_order = sorted(group, key=cmp_to_key(compare_shards))
            for previous, current in zip(range_order, range_order[1:]):
                if (
                    _compare_key_tuples(
                        previous["max_key"],
                        current["min_key"],
                        label=f"partition {partition.decode('utf-8')}",
                    )
                    >= 0
                ):
                    raise V17DatasetManifestError("same-partition shard key ranges overlap")
    return normalized


def validate_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact v1 shape, tags, totals, ranges, and both digests."""

    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, DATASET_MANIFEST_KEYS, label="dataset manifest")
    if sealed.get("version") != DATASET_MANIFEST_VERSION:
        raise V17DatasetManifestError("unsupported dataset manifest version")
    dataset_id = require_identifier(sealed.get("dataset_id"), label="dataset_id")
    role = require_identifier(sealed.get("role"), label="role")
    dataset_format = sealed.get("format")
    if dataset_format not in DATASET_FORMATS:
        raise V17DatasetManifestError("format must be PARQUET or BLOB")
    media_type = require_nonempty_string(sealed.get("media_type"), label="media_type")
    if media_type != FORMAT_MEDIA_TYPES[dataset_format]:
        raise V17DatasetManifestError("media_type does not match dataset format")

    schema = _validate_schema(sealed.get("schema"))
    primary_key = _identifier_array(sealed.get("primary_key"), label="primary_key")
    partition_keys = _identifier_array(sealed.get("partition_keys"), label="partition_keys")
    sort_keys = _identifier_array(sealed.get("sort_keys"), label="sort_keys")
    schema_names = {str(entry["name"]) for entry in schema}
    for label, keys in (
        ("primary_key", primary_key),
        ("partition_keys", partition_keys),
        ("sort_keys", sort_keys),
    ):
        missing = sorted(set(keys) - schema_names)
        if missing:
            raise V17DatasetManifestError(f"{label} fields missing from schema: {missing}")
    if dataset_format == "PARQUET":
        if not schema or not primary_key:
            raise V17DatasetManifestError("Parquet schema and primary_key must be nonempty")
    elif schema or primary_key or partition_keys or sort_keys:
        raise V17DatasetManifestError("blob dataset cannot declare tabular schema or keys")

    schema_digest = dataset_schema_sha256(schema)
    shards = _validate_shards(
        sealed.get("shards"),
        dataset_format=dataset_format,
        partition_keys=partition_keys,
        primary_key=primary_key,
        expected_schema_sha256=schema_digest,
    )
    total_row_count = _nonnegative_integer(
        sealed.get("total_row_count"),
        label="total_row_count",
    )
    total_size_bytes = _positive_integer(
        sealed.get("total_size_bytes"),
        label="total_size_bytes",
    )
    if total_row_count != sum(int(item["row_count"]) for item in shards):
        raise V17DatasetManifestError("total_row_count does not equal shard rows")
    if total_size_bytes != sum(int(item["size_bytes"]) for item in shards):
        raise V17DatasetManifestError("total_size_bytes does not equal shard bytes")
    declared_content_set = require_sha256(
        sealed.get("content_set_sha256"),
        label="content_set_sha256",
    )
    if declared_content_set != dataset_content_set_sha256(shards):
        raise V17DatasetManifestError("content_set_sha256 mismatch")
    require_authority_false(sealed.get("authority"))
    return {
        "version": DATASET_MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "role": role,
        "format": dataset_format,
        "media_type": media_type,
        "schema": schema,
        "primary_key": primary_key,
        "partition_keys": partition_keys,
        "sort_keys": sort_keys,
        "shards": shards,
        "total_row_count": total_row_count,
        "total_size_bytes": total_size_bytes,
        "content_set_sha256": declared_content_set,
        "authority": False,
        "semantic_sha256": sealed["semantic_sha256"],
    }


def _stat_signature(entry: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        entry.st_dev,
        entry.st_ino,
        entry.st_nlink,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
        stat.S_IMODE(entry.st_mode),
    )


def _canonical_arrow_key_scalar(value: Any, *, label: str) -> Any:
    if value is None:
        raise V17DatasetManifestError(f"{label} cannot be null")
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() == timezone.utc.utcoffset(value):
            return value.isoformat().replace("+00:00", "Z")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return _json_scalar(value, label=label, allow_null=False)


def _same_scalar(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _scan_parquet_contract(
    parquet_file: Any,
    *,
    partition_values: Mapping[str, Any],
    primary_key: Sequence[str],
    min_key: Sequence[Any],
    max_key: Sequence[Any],
    sort_keys: Sequence[str],
) -> int:
    scan_columns = list(dict.fromkeys([*partition_values, *primary_key, *sort_keys]))
    observed_min: list[Any] | None = None
    observed_max: list[Any] | None = None
    previous_sort: list[Any] | None = None
    observed_rows = 0
    for batch in parquet_file.iter_batches(
        batch_size=65_536,
        columns=scan_columns,
        use_threads=False,
    ):
        values_by_column = batch.to_pydict()
        for row_index in range(batch.num_rows):
            observed_rows += 1
            for key, expected in partition_values.items():
                observed = _canonical_arrow_key_scalar(
                    values_by_column[key][row_index],
                    label=f"Parquet partition key {key}",
                )
                if not _same_scalar(observed, expected):
                    raise V17DatasetManifestError(
                        f"Parquet row disagrees with partition_values: {key}"
                    )
            primary_values = [
                _canonical_arrow_key_scalar(
                    values_by_column[key][row_index],
                    label=f"Parquet primary key {key}",
                )
                for key in primary_key
            ]
            if (
                observed_min is None
                or _compare_key_tuples(
                    primary_values,
                    observed_min,
                    label="Parquet primary key minimum",
                )
                < 0
            ):
                observed_min = primary_values
            if (
                observed_max is None
                or _compare_key_tuples(
                    primary_values,
                    observed_max,
                    label="Parquet primary key maximum",
                )
                > 0
            ):
                observed_max = primary_values
            if sort_keys:
                sort_values = [
                    _canonical_arrow_key_scalar(
                        values_by_column[key][row_index],
                        label=f"Parquet sort key {key}",
                    )
                    for key in sort_keys
                ]
                if (
                    previous_sort is not None
                    and _compare_key_tuples(
                        previous_sort,
                        sort_values,
                        label="Parquet sort order",
                    )
                    > 0
                ):
                    raise V17DatasetManifestError("Parquet rows violate declared sort_keys")
                previous_sort = sort_values
    if observed_min is None or observed_max is None:
        raise V17DatasetManifestError("Parquet shard has no scannable primary keys")
    if len(observed_min) != len(min_key) or any(
        not _same_scalar(observed, declared) for observed, declared in zip(observed_min, min_key)
    ):
        raise V17DatasetManifestError("Parquet actual min_key mismatch")
    if len(observed_max) != len(max_key) or any(
        not _same_scalar(observed, declared) for observed, declared in zip(observed_max, max_key)
    ):
        raise V17DatasetManifestError("Parquet actual max_key mismatch")
    return observed_rows


def _validate_parquet_metadata(
    path: Path,
    *,
    expected_schema: Sequence[Mapping[str, Any]],
    expected_row_count: int,
    partition_values: Mapping[str, Any],
    primary_key: Sequence[str],
    min_key: Sequence[Any],
    max_key: Sequence[Any],
    sort_keys: Sequence[str],
) -> None:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - declared package dependency
        raise V17DatasetManifestError("pyarrow unavailable for Parquet verification") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise V17DatasetManifestError(f"Parquet object unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            parquet_file = parquet.ParquetFile(stream)
            observed_row_count = int(parquet_file.metadata.num_rows)
            observed_schema = [
                {
                    "name": field.name,
                    "logical_type": str(field.type),
                    "nullable": bool(field.nullable),
                }
                for field in parquet_file.schema_arrow
            ]
            if observed_schema != list(expected_schema):
                raise V17DatasetManifestError(f"Parquet logical schema mismatch: {path}")
            scanned_row_count = _scan_parquet_contract(
                parquet_file,
                partition_values=partition_values,
                primary_key=primary_key,
                min_key=min_key,
                max_key=max_key,
                sort_keys=sort_keys,
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_signature(before) != _stat_signature(after):
        raise V17DatasetManifestError(f"Parquet object changed during metadata read: {path}")
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise V17DatasetManifestError(f"Parquet object path unavailable: {path}") from exc
    if _stat_signature(path_after) != _stat_signature(before):
        raise V17DatasetManifestError(f"Parquet object path changed during metadata read: {path}")
    if observed_row_count != expected_row_count:
        raise V17DatasetManifestError(f"Parquet row count mismatch: {path}")
    if scanned_row_count != expected_row_count:
        raise V17DatasetManifestError(f"Parquet scanned row count mismatch: {path}")


def validate_dataset_objects(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate every referenced CAS object and Parquet metadata in place."""

    manifest = validate_dataset_manifest(payload)
    root = Path(os.path.abspath(os.fspath(repo_root)))
    for shard in manifest["shards"]:
        object_path = str(shard["object_path"])
        path = root.joinpath(*PurePosixPath(object_path).parts)
        try:
            before = os.lstat(path)
        except OSError as exc:
            raise V17DatasetManifestError(f"dataset object unavailable: {object_path}") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise V17DatasetManifestError(f"dataset object identity invalid: {object_path}")
        if before.st_size != shard["size_bytes"]:
            raise V17DatasetManifestError(f"dataset object size mismatch: {object_path}")
        if file_sha256(path) != shard["byte_sha256"]:
            raise V17DatasetManifestError(f"dataset object byte SHA mismatch: {object_path}")
        after = os.lstat(path)
        if _stat_signature(after) != _stat_signature(before):
            raise V17DatasetManifestError(
                f"dataset object changed during validation: {object_path}"
            )
        if manifest["format"] == "PARQUET":
            _validate_parquet_metadata(
                path,
                expected_schema=manifest["schema"],
                expected_row_count=int(shard["row_count"]),
                partition_values=shard["partition_values"],
                primary_key=manifest["primary_key"],
                min_key=shard["min_key"],
                max_key=shard["max_key"],
                sort_keys=manifest["sort_keys"],
            )
    return manifest


__all__ = [
    "DATASET_CONTENT_SET_VERSION",
    "DATASET_FORMATS",
    "DATASET_MANIFEST_VERSION",
    "DATASET_SCHEMA_DIGEST_VERSION",
    "V17DatasetManifestError",
    "dataset_content_set_sha256",
    "dataset_schema_sha256",
    "derive_dataset_object_path",
    "validate_dataset_manifest",
    "validate_dataset_objects",
]
