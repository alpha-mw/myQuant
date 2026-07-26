"""Pure, offline planning for the protocol-v2 source hash DAG.

This module reads caller-pinned local source files and returns exact write
intents.  It never creates directories, writes files, calls a provider, or
performs publication.  Storage is responsible for exclusive creation, locked
CAS, and exact-byte readback of the returned intents.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import inspect
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final

from quant_investor.v17_v2_contract.canonical import (
    canonical_json_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    require_ascii_casefold_unique,
    require_sha256,
)
from quant_investor.v17_v2_contract.limits import LIMITS
from quant_investor.v17_v2_contract.namespace import derive_content_object_path
from quant_investor.v17_v2_contract.schema_validation import (
    validate_canonical_schema_bytes,
)
from quant_investor.v17_v2_contract.validators import (
    DATASET_MANIFEST_VERSION,
    DATASET_SCHEMA_DIGEST_VERSION,
    DATASET_SUMMARY_VERSION,
    GENERATION_CATALOG_VERSION,
    OBSERVATION_DISPOSITION_VERSION,
    PROTOCOL_VERSION,
    SOURCE_BINDING_SET_VERSION,
    SOURCE_LOCATOR_VERSION,
    SOURCE_MANIFEST_VERSION,
    SOURCE_ROLE_MATRIX_VERSION,
    document_byte_sha256,
    require_runtime_usable_source_role_matrix,
    seal_semantic,
    validate_dataset_manifest,
    validate_document_identity,
    validate_source_hash_dag,
)

_ZERO_SHA256: Final = "0" * 64
_SOURCE_OBJECT_VERSION: Final = "myquant.v17.v2.source-object.v1"
_GENERATION_CATALOG_SCHEMA: Final = "myquant.v17.v2.generation-catalog.schema.v1"
_TABLE_ORDERING: Final = "stage-role-table_id-summary_path-summary_sha-dataset_path-dataset_sha"
_BINDING_ORDERING: Final = (
    "stage-role-catalog_path-catalog_sha-summary_path-summary_sha-"
    "dataset_path-dataset_sha-disposition_id"
)
_ID_FIELDS: Final = (
    "artifact_id",
    "catalog_id",
    "dataset_id",
    "disposition_id",
    "input_id",
    "locator_id",
    "manifest_id",
    "overlay_id",
    "pointer_id",
    "policy_id",
    "snapshot_id",
)
_ARTIFACT_ID_FIELD_BY_VERSION: Final = {
    "myquant.v17.v2.macro-overlay.v1": "overlay_id",
    "myquant.v17.v2.market-pointer.v1": "pointer_id",
    "myquant.v17.v2.market-snapshot-manifest.v1": "manifest_id",
    "myquant.v17.v2.markov-overlay.v1": "overlay_id",
    "myquant.v17.v2.portfolio-required-inputs.v1": "input_id",
    "myquant.v17.v2.risk-policy-snapshot.v1": "policy_id",
}
_CATALOG_ROLE_BY_DATASET_ROLE: Final = {
    "H00300_total_return_dataset": "pit_generation_catalog",
    "cn_open_day_calendar_dataset": "pit_generation_catalog",
    "corporate_actions_dataset": "pit_generation_catalog",
    "deep_evidence_dataset": "fundamental_generation_catalog",
    "fundamental_raw_tables_dataset": "fundamental_generation_catalog",
    "market_bars_dataset": "pit_generation_catalog",
    "official_delisting_cash_dataset": "pit_generation_catalog",
}


class SourcePlanningError(ValueError):
    """Raised before any write when an offline source plan is unsafe."""

    exit_code = 2


@dataclass(frozen=True)
class SourceFile:
    """One real local source file pinned by exact bytes and one exact role."""

    path: Path
    expected_sha256: str
    role: str


@dataclass(frozen=True)
class DatasetRecordSpec:
    """Registry-derived, immutable metadata for one dataset role."""

    catalog_role: str
    record_schema_id: str
    schema: tuple[Mapping[str, Any], ...]
    primary_key: tuple[str, ...]
    partition_keys: tuple[str, ...]
    sort_keys: tuple[str, ...]
    valid_time_field: str
    available_time_field: str


@dataclass(frozen=True)
class DatasetShardFacts:
    """Facts derived from the exact pinned Parquet bytes by an offline reader."""

    logical_name: str
    partition_values: Mapping[str, Any]
    row_count: int
    min_key: tuple[Any, ...]
    max_key: tuple[Any, ...]
    observation_key_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True)
class WriteIntent:
    """An immutable exact-byte request for the storage layer."""

    sequence: int
    kind: str
    relative_path: str
    payload: bytes
    byte_sha256: str
    mode: int = 0o600


@dataclass(frozen=True)
class SourceDagPlan:
    """Validated DAG documents plus their ordered, locator-last write intents."""

    scope_id: str
    write_intents: tuple[WriteIntent, ...]
    source_objects: Mapping[str, bytes]
    dataset_manifests: Mapping[str, Mapping[str, Any]]
    observation_dispositions: Mapping[str, Mapping[str, Any]]
    source_manifest: Mapping[str, Any]
    source_manifest_path: str
    generation_catalogs: Mapping[str, Mapping[str, Any]]
    summaries: Mapping[str, Mapping[str, Any]]
    source_binding_set: Mapping[str, Any]
    source_binding_set_path: str
    source_locator: Mapping[str, Any]
    source_locator_path: str


RecordRegistryValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
RecordSpecResolver = Callable[[str, Mapping[str, Any]], DatasetRecordSpec]
DatasetInspector = Callable[
    [SourceFile, bytes, DatasetRecordSpec],
    DatasetShardFacts,
]
ObjectCrossValidator = Callable[
    [str, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]
DagValidator = Callable[..., Mapping[str, Any]]


def _fail(message: str, *, cause: BaseException | None = None) -> None:
    error = SourcePlanningError(message)
    if cause is None:
        raise error
    raise error from cause


def _validate_instant(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        _fail(f"{label} must be an RFC3339 instant")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        _fail(f"{label} must be an RFC3339 instant", cause=exc)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{label} must include a timezone")
    return parsed


def _default_record_registry_validator(
    registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        from quant_investor.v17_v2_contract.validators import (
            require_runtime_usable_dataset_record_schema_registry,
            validate_dataset_record_schema_registry,
        )
    except ImportError as exc:
        _fail("packaged dataset record registry validator is unavailable", cause=exc)
    validated = validate_dataset_record_schema_registry(registry)
    approved = require_runtime_usable_dataset_record_schema_registry()
    if validated != approved:
        _fail("dataset record registry is not the exact packaged registry")
    return validated


def _default_record_spec_resolver(
    role: str,
    registry: Mapping[str, Any],
) -> DatasetRecordSpec:
    records = registry.get("records")
    if type(records) is not list:
        _fail("dataset record registry records must be an array")
    matches = [row for row in records if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        _fail(f"dataset record registry must contain exactly one row for role {role}")
    row = matches[0]
    catalog_role = _CATALOG_ROLE_BY_DATASET_ROLE.get(role)
    if catalog_role is None:
        _fail(f"dataset role has no frozen generation-catalog mapping: {role}")
    try:
        return DatasetRecordSpec(
            catalog_role=catalog_role,
            record_schema_id=str(row["record_schema_id"]),
            schema=tuple(dict(value) for value in row["logical_fields"]),
            primary_key=tuple(str(value) for value in row["primary_key"]),
            partition_keys=tuple(str(value) for value in row["partition_keys"]),
            sort_keys=tuple(str(value) for value in row["sort_keys"]),
            valid_time_field=str(row["effective_time_field"]),
            available_time_field=str(row["available_time_field"]),
        )
    except (KeyError, TypeError) as exc:
        _fail(f"dataset record registry row is malformed for role {role}", cause=exc)


def _slug(role: str) -> str:
    value = role.lower().replace("_", "-").replace(".", "-").replace(":", "-")
    if not value or len(value) > 96 or not value[0].isalnum():
        _fail(f"role cannot produce a safe path identifier: {role!r}")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in value):
        _fail(f"role cannot produce a safe path identifier: {role!r}")
    return value


def _secure_chain(
    absolute_path: str,
) -> tuple[list[int], tuple[tuple[int, int, int], ...]]:
    """Open every path component without following a symlink."""

    if not os.path.isabs(absolute_path) or os.path.normpath(absolute_path) != absolute_path:
        _fail("source path must be a normalized absolute path")
    parts = PurePosixPath(absolute_path).parts[1:]
    if not parts:
        _fail("source path must identify a file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    try:
        current = os.open("/", directory_flags)
        descriptors.append(current)
        root_stat = os.fstat(current)
        identities.append((root_stat.st_dev, root_stat.st_ino, stat.S_IFMT(root_stat.st_mode)))
        for index, component in enumerate(parts):
            is_leaf = index == len(parts) - 1
            opened = os.open(
                component,
                flags if is_leaf else directory_flags,
                dir_fd=current,
            )
            descriptors.append(opened)
            current = opened
            observed = os.fstat(opened)
            expected_kind = stat.S_IFREG if is_leaf else stat.S_IFDIR
            if stat.S_IFMT(observed.st_mode) != expected_kind:
                _fail("source path contains a non-directory ancestor or non-regular leaf")
            identities.append((observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)))
        return descriptors, tuple(identities)
    except (OSError, SourcePlanningError) as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, SourcePlanningError):
            raise
        _fail("source path is missing, changed, or contains a symlink", cause=exc)


def _secure_read_exact(
    source: SourceFile,
    *,
    source_root: Path,
    maximum_bytes: int,
) -> bytes:
    source_path = os.path.normpath(os.fspath(source.path))
    root_path = os.path.normpath(os.fspath(source_root))
    if not os.path.isabs(source_path) or not os.path.isabs(root_path):
        _fail("source_root and source paths must be absolute")
    try:
        within_root = os.path.commonpath((source_path, root_path)) == root_path
    except ValueError as exc:
        _fail("source path and source_root are on incompatible roots", cause=exc)
    if not within_root or source_path == root_path:
        _fail("source path escapes source_root")
    try:
        expected_sha256 = require_sha256(
            source.expected_sha256,
            label=f"{source.role} expected SHA-256",
        )
    except IdentityContractError as exc:
        _fail(str(exc), cause=exc)

    descriptors, before_chain = _secure_chain(source_path)
    leaf = descriptors[-1]
    try:
        before = os.fstat(leaf)
        if before.st_nlink != 1:
            _fail(f"source file is hardlinked: {source_path}")
        if before.st_size <= 0:
            _fail(f"source file is empty: {source_path}")
        if before.st_size > maximum_bytes:
            _fail(f"source file exceeds the exact-byte planning limit: {source_path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(leaf, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"source file truncated during read: {source_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(leaf, 1) != b"":
            _fail(f"source file grew during read: {source_path}")
        raw = b"".join(chunks)
        after = os.fstat(leaf)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
        ) or after.st_nlink != 1:
            _fail(f"source file changed during read: {source_path}")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass

    check_descriptors, after_chain = _secure_chain(source_path)
    try:
        check = os.fstat(check_descriptors[-1])
        if check.st_nlink != 1 or after_chain != before_chain:
            _fail(f"source path changed during read: {source_path}")
    finally:
        for descriptor in reversed(check_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        _fail(f"source SHA-256 mismatch for role {source.role}")
    return raw


def _artifact_ref(
    document: Mapping[str, Any],
    relative_path: str,
    *,
    artifact_id: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": str(document["version"]),
        "relative_path": relative_path,
        "byte_sha256": document_byte_sha256(document),
        "semantic_sha256": str(document["semantic_sha256"]),
    }


def _raw_source_ref(
    *,
    source_id: str,
    object_path: str,
    byte_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_id": source_id,
        "artifact_version": _SOURCE_OBJECT_VERSION,
        "relative_path": object_path,
        "byte_sha256": byte_sha256,
        "semantic_sha256": _ZERO_SHA256,
    }


def _sealed_document(**values: Any) -> dict[str, Any]:
    return seal_semantic(
        {
            "protocol_version": PROTOCOL_VERSION,
            "authority": False,
            **values,
        }
    )


def _document_artifact_id(document: Mapping[str, Any]) -> str:
    version = document.get("version")
    field = _ARTIFACT_ID_FIELD_BY_VERSION.get(version)
    if field is not None:
        value = document.get(field)
        if type(value) is not str:
            _fail(f"OBJECT source document lacks {field}")
        return value
    candidates = [value for field in _ID_FIELDS if type((value := document.get(field))) is str]
    if len(candidates) != 1:
        _fail("OBJECT source document must expose exactly one supported artifact ID")
    return candidates[0]


def _default_object_cross_validator(
    role: str,
    role_row: Mapping[str, Any],
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    schema_version = role_row.get("schema_version")
    if type(schema_version) is not str or not schema_version.endswith(".schema.v1"):
        _fail(f"OBJECT role lacks one frozen packaged schema: {role}")
    artifact_version = schema_version.removesuffix(".schema.v1") + ".v1"
    try:
        validated = validate_document_identity(
            document,
            expected_version=artifact_version,
        )
    except ValueError as exc:
        _fail(f"OBJECT source cross-validation failed for role {role}", cause=exc)
    if validated.get("role") != role or validated.get("phase") != role_row.get("phase"):
        _fail(f"OBJECT source does not carry exact role and phase: {role}")
    return validated


def _validate_object_bytes(
    *,
    role: str,
    role_row: Mapping[str, Any],
    raw: bytes,
    cross_validator: ObjectCrossValidator,
) -> dict[str, Any]:
    schema_version = role_row.get("schema_version")
    if type(schema_version) is not str or not schema_version.endswith(".schema.v1"):
        _fail(f"OBJECT role lacks one frozen packaged schema: {role}")
    artifact_version = schema_version.removesuffix(".schema.v1") + ".v1"
    try:
        schema_validated = validate_canonical_schema_bytes(
            raw,
            expected_version=artifact_version,
        )
        relationship_validated = cross_validator(role, role_row, schema_validated)
    except ValueError as exc:
        _fail(f"OBJECT source validation failed for role {role}", cause=exc)
    if dict(relationship_validated) != schema_validated:
        _fail(f"OBJECT cross-validator changed the accepted document: {role}")
    return dict(schema_validated)


def _schema_digest(schema: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": DATASET_SCHEMA_DIGEST_VERSION,
                "schema": list(schema),
            }
        )
    ).hexdigest()


def _content_set_digest(shards: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant.v17.v2.dataset-content-set.v1",
                "shards": list(shards),
            }
        )
    ).hexdigest()


def _validate_generated_document(
    document: Mapping[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    raw = canonical_resource_bytes(document)
    try:
        validated = validate_canonical_schema_bytes(
            raw,
            expected_version=expected_version,
        )
    except ValueError as exc:
        _fail(f"generated {expected_version} failed its sole packaged schema", cause=exc)
    if validated != document:
        _fail(f"generated {expected_version} changed across canonical validation")
    return validated


def _write_intent(
    *,
    sequence: int,
    kind: str,
    relative_path: str,
    payload: bytes,
) -> WriteIntent:
    if type(payload) is not bytes:
        _fail("write intent payload must be exact bytes")
    return WriteIntent(
        sequence=sequence,
        kind=kind,
        relative_path=relative_path,
        payload=payload,
        byte_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _invoke_dag_validator(
    validator: DagValidator,
    *,
    dataset_record_schema_registry: Mapping[str, Any],
    dag: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        parameters = inspect.signature(validator).parameters.values()
    except (TypeError, ValueError) as exc:
        _fail("source DAG validator must expose an inspectable signature", cause=exc)
    accepts_registry = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == "dataset_record_schema_registry"
        for parameter in parameters
    )
    kwargs = dict(dag)
    if accepts_registry:
        kwargs["dataset_record_schema_registry"] = dataset_record_schema_registry
    try:
        return validator(**kwargs)
    except ValueError as exc:
        _fail("generated source DAG failed cross-document validation", cause=exc)


def plan_source_dag(
    *,
    source_root: Path,
    sources: Sequence[SourceFile],
    cutoff: str,
    created_at: str,
    source_role_matrix: Mapping[str, Any],
    dataset_record_schema_registry: Mapping[str, Any],
    dataset_inspector: DatasetInspector,
    record_registry_validator: RecordRegistryValidator = (_default_record_registry_validator),
    record_spec_resolver: RecordSpecResolver = _default_record_spec_resolver,
    role_matrix_validator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = require_runtime_usable_source_role_matrix,
    object_cross_validator: ObjectCrossValidator = _default_object_cross_validator,
    dag_validator: DagValidator = validate_source_hash_dag,
) -> SourceDagPlan:
    """Build and validate a complete exact-byte source DAG without writing it.

    The role matrix and dataset-record registry are validated before any source
    file is opened.  Every required role must have exactly one source; optional
    roles may be omitted but cannot be duplicated or substituted.
    """

    if isinstance(sources, (str, bytes, bytearray)) or not isinstance(sources, Sequence):
        _fail("sources must be an ordered sequence")
    if not sources or any(type(source) is not SourceFile for source in sources):
        _fail("sources must contain SourceFile values")
    cutoff_instant = _validate_instant(cutoff, label="cutoff")
    created_instant = _validate_instant(created_at, label="created_at")
    if created_instant < cutoff_instant:
        _fail("created_at precedes cutoff")
    if record_registry_validator is None or record_spec_resolver is None:
        _fail("dataset record registry validation and resolution are required")
    if dataset_inspector is None or object_cross_validator is None or dag_validator is None:
        _fail("dataset, object, and DAG cross-validators are required")

    try:
        role_matrix = dict(role_matrix_validator(source_role_matrix))
        record_registry = dict(record_registry_validator(dataset_record_schema_registry))
    except ValueError as exc:
        _fail("source registries are not complete and runtime usable", cause=exc)
    if (
        role_matrix.get("completeness") != "COMPLETE"
        or role_matrix.get("runtime_usable") is not True
        or role_matrix.get("pending_registry") != []
    ):
        _fail("source role matrix must be COMPLETE with no pending registry")
    role_rows: dict[str, Mapping[str, Any]] = {}
    for raw_row in role_matrix.get("roles", []):
        if not isinstance(raw_row, Mapping) or type(raw_row.get("role")) is not str:
            _fail("source role matrix contains an invalid role row")
        role = str(raw_row["role"])
        if role in role_rows:
            _fail(f"duplicate source role matrix role: {role}")
        role_rows[role] = dict(raw_row)
    try:
        require_ascii_casefold_unique(
            sorted(role_rows),
            label="source role matrix roles",
        )
        source_roles = require_ascii_casefold_unique(
            [source.role for source in sources],
            label="source input roles",
        )
    except IdentityContractError as exc:
        _fail(str(exc), cause=exc)
    unexpected_roles = sorted(set(source_roles) - set(role_rows))
    if unexpected_roles:
        _fail(f"source inputs contain unregistered roles: {unexpected_roles}")
    missing_roles = sorted(
        role
        for role, row in role_rows.items()
        if row.get("required") is True and role not in set(source_roles)
    )
    if missing_roles:
        _fail(f"required source roles are missing: {missing_roles}")
    source_paths = [os.path.normpath(os.fspath(source.path)) for source in sources]
    if len(set(source_paths)) != len(source_paths) or len(
        {path.casefold() for path in source_paths}
    ) != len(source_paths):
        _fail("source input paths have an exact or casefold collision")

    raw_by_role: dict[str, bytes] = {}
    object_path_by_role: dict[str, str] = {}
    source_objects: dict[str, bytes] = {}
    object_documents: dict[str, Mapping[str, Any]] = {}
    source_id_by_role: dict[str, str] = {}
    ordered_sources = sorted(sources, key=lambda item: item.role)
    for source in ordered_sources:
        row = role_rows[source.role]
        kind = row.get("kind")
        extension = source.path.suffix.lower()
        if extension == ".csv":
            _fail(f"CSV fallback is forbidden for role {source.role}")
        if kind == "DATASET":
            if extension != ".parquet":
                _fail(f"DATASET role requires an exact Parquet source: {source.role}")
            suffix = "parquet"
        elif kind == "OBJECT":
            if extension != ".json":
                _fail(f"OBJECT role requires an exact JSON source: {source.role}")
            suffix = "json"
        else:
            _fail(f"unsupported source role kind for {source.role}: {kind!r}")
        raw = _secure_read_exact(
            source,
            source_root=source_root,
            maximum_bytes=LIMITS["max_shard_bytes"],
        )
        digest = hashlib.sha256(raw).hexdigest()
        object_path = str(derive_content_object_path(digest, suffix=suffix))
        if object_path in source_objects:
            _fail("two source roles resolve to the same content object")
        raw_by_role[source.role] = raw
        object_path_by_role[source.role] = object_path
        source_objects[object_path] = raw
        source_id_by_role[source.role] = f"source-{_slug(source.role)}"
        if kind == "OBJECT" and row.get("schema_version") != _GENERATION_CATALOG_SCHEMA:
            object_documents[source.role] = _validate_object_bytes(
                role=source.role,
                role_row=row,
                raw=raw,
                cross_validator=object_cross_validator,
            )
        elif kind == "OBJECT":
            try:
                load_canonical_resource(raw, label=f"catalog provenance {source.role}")
            except ValueError as exc:
                _fail(
                    f"catalog provenance must be canonical JSON: {source.role}",
                    cause=exc,
                )

    scope_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_version": PROTOCOL_VERSION,
                "cutoff": cutoff,
                "sources": [
                    {
                        "role": source.role,
                        "byte_sha256": source.expected_sha256,
                    }
                    for source in ordered_sources
                ],
            }
        )
    ).hexdigest()
    scope_id = f"source-{scope_digest[:24]}"
    role_matrix_sha = hashlib.sha256(canonical_resource_bytes(role_matrix)).hexdigest()

    dataset_manifests: dict[str, Mapping[str, Any]] = {}
    dataset_refs: dict[str, Mapping[str, Any]] = {}
    dispositions: dict[str, Mapping[str, Any]] = {}
    disposition_refs: dict[str, Mapping[str, Any]] = {}
    record_specs: dict[str, DatasetRecordSpec] = {}
    shard_facts_by_role: dict[str, DatasetShardFacts] = {}
    for source in ordered_sources:
        row = role_rows[source.role]
        if row.get("kind") != "DATASET":
            continue
        try:
            record_spec = record_spec_resolver(source.role, record_registry)
        except (KeyError, TypeError, ValueError) as exc:
            _fail(
                f"dataset role is not exactly bound in the record registry: {source.role}",
                cause=exc,
            )
        if type(record_spec) is not DatasetRecordSpec:
            _fail(f"record spec resolver returned the wrong type for {source.role}")
        if record_spec.catalog_role not in role_rows:
            _fail(f"dataset record spec references an unknown catalog role: {source.role}")
        catalog_row = role_rows[record_spec.catalog_role]
        if (
            catalog_row.get("kind") != "OBJECT"
            or catalog_row.get("schema_version") != _GENERATION_CATALOG_SCHEMA
        ):
            _fail(f"dataset record spec references a non-catalog role: {source.role}")
        try:
            facts = dataset_inspector(source, raw_by_role[source.role], record_spec)
        except (TypeError, ValueError) as exc:
            _fail(f"dataset inspection failed for role {source.role}", cause=exc)
        if type(facts) is not DatasetShardFacts:
            _fail(f"dataset inspector returned the wrong type for {source.role}")
        if (
            type(facts.row_count) is not int
            or facts.row_count <= 0
            or len(facts.min_key) != len(record_spec.primary_key)
            or len(facts.max_key) != len(record_spec.primary_key)
        ):
            _fail(f"dataset inspector returned invalid row/key facts for {source.role}")
        if set(facts.partition_values) != set(record_spec.partition_keys):
            _fail(f"dataset inspector partition keys mismatch for {source.role}")
        try:
            observation_hashes = tuple(
                sorted(
                    require_sha256(value, label=f"{source.role} observation key SHA-256")
                    for value in facts.observation_key_sha256s
                )
            )
        except IdentityContractError as exc:
            _fail(str(exc), cause=exc)
        if len(set(observation_hashes)) != len(observation_hashes):
            _fail(f"dataset observation key SHA-256 values are duplicated: {source.role}")
        record_specs[source.role] = record_spec
        shard_facts_by_role[source.role] = facts

        role_slug = _slug(source.role)
        dataset_id = f"{scope_id}-{role_slug}"
        dataset_path = (
            "data/private/v17_sources/protocol-v2/manifests/" f"{dataset_id}.dataset.json"
        )
        object_path = object_path_by_role[source.role]
        raw = raw_by_role[source.role]
        schema = [dict(value) for value in record_spec.schema]
        shard = {
            "logical_name": facts.logical_name,
            "partition_values": dict(facts.partition_values),
            "object_path": object_path,
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "row_count": facts.row_count,
            "min_key": list(facts.min_key),
            "max_key": list(facts.max_key),
            "schema_sha256": _schema_digest(schema),
        }
        dataset = _sealed_document(
            version=DATASET_MANIFEST_VERSION,
            dataset_id=dataset_id,
            role=source.role,
            format="PARQUET",
            media_type="application/vnd.apache.parquet",
            schema=schema,
            primary_key=list(record_spec.primary_key),
            partition_keys=list(record_spec.partition_keys),
            sort_keys=list(record_spec.sort_keys),
            shards=[shard],
            total_row_count=facts.row_count,
            total_size_bytes=len(raw),
            content_set_sha256=_content_set_digest([shard]),
        )
        _validate_generated_document(dataset, expected_version=DATASET_MANIFEST_VERSION)
        try:
            validate_dataset_manifest(dataset, source_objects={object_path: raw})
        except ValueError as exc:
            _fail(f"dataset manifest cross-validation failed for {source.role}", cause=exc)
        dataset_manifests[dataset_path] = dataset
        dataset_ref = _artifact_ref(dataset, dataset_path, artifact_id=dataset_id)
        dataset_refs[source.role] = dataset_ref

        disposition_id = f"{dataset_id}-disposition"
        disposition_path = (
            "data/private/v17_sources/protocol-v2/manifests/" f"{dataset_id}.disposition.json"
        )
        disposition = _sealed_document(
            version=OBSERVATION_DISPOSITION_VERSION,
            disposition_id=disposition_id,
            scope_id=dataset_id,
            stage=str(row["phase"]),
            market="CN",
            cutoff=cutoff,
            dataset_manifest_ref=dataset_ref,
            status="UNREADY",
            effect="MARK_STATISTICALLY_UNREADY",
            reason_code="exact_bytes_planned_unready",
            observation_key_sha256s=list(observation_hashes),
        )
        _validate_generated_document(
            disposition,
            expected_version=OBSERVATION_DISPOSITION_VERSION,
        )
        dispositions[disposition_path] = disposition
        disposition_refs[source.role] = _artifact_ref(
            disposition,
            disposition_path,
            artifact_id=disposition_id,
        )

    manifest_path = f"data/private/v17_sources/protocol-v2/manifests/{scope_id}.json"
    source_rows: list[dict[str, Any]] = []
    for source in ordered_sources:
        role = source.role
        if role in object_documents:
            document = object_documents[role]
            source_ref = _artifact_ref(
                document,
                object_path_by_role[role],
                artifact_id=_document_artifact_id(document),
            )
        else:
            source_ref = _raw_source_ref(
                source_id=source_id_by_role[role],
                object_path=object_path_by_role[role],
                byte_sha256=hashlib.sha256(raw_by_role[role]).hexdigest(),
            )
        source_rows.append(
            {
                "source_id": source_id_by_role[role],
                "role": role,
                "availability": "AVAILABLE",
                "source_ref": source_ref,
            }
        )
    source_rows.sort(key=lambda value: (value["role"], value["source_id"]))
    source_manifest = _sealed_document(
        version=SOURCE_MANIFEST_VERSION,
        manifest_id=scope_id,
        role_matrix_ref={
            "resource_name": "source_role_matrix.v1.json",
            "resource_version": SOURCE_ROLE_MATRIX_VERSION,
            "byte_sha256": role_matrix_sha,
        },
        market="CN",
        cutoff=cutoff,
        created_at=created_at,
        source_ordering="role-source_id-ascending",
        sources=source_rows,
        dataset_manifest_refs=sorted(
            dataset_refs.values(),
            key=lambda value: (
                value["artifact_id"],
                value["relative_path"],
                value["byte_sha256"],
            ),
        ),
        observation_disposition_refs=sorted(
            disposition_refs.values(),
            key=lambda value: (
                value["artifact_id"],
                value["relative_path"],
                value["byte_sha256"],
            ),
        ),
    )
    _validate_generated_document(source_manifest, expected_version=SOURCE_MANIFEST_VERSION)
    manifest_ref = _artifact_ref(
        source_manifest,
        manifest_path,
        artifact_id=scope_id,
    )

    summaries: dict[str, Mapping[str, Any]] = {}
    summary_refs: dict[str, Mapping[str, Any]] = {}
    for role in sorted(dataset_refs):
        summary_id = f"{scope_id}-{_slug(role)}-summary"
        summary = _sealed_document(
            version=DATASET_SUMMARY_VERSION,
            summary_id=summary_id,
            source_manifest_ref=manifest_ref,
            dataset_manifest_ref=dataset_refs[role],
            row_count=shard_facts_by_role[role].row_count,
        )
        _validate_generated_document(summary, expected_version=DATASET_SUMMARY_VERSION)
        summary_sha = document_byte_sha256(summary)
        summary_path = str(derive_content_object_path(summary_sha, suffix="json"))
        summaries[summary_path] = summary
        summary_refs[role] = _artifact_ref(
            summary,
            summary_path,
            artifact_id=summary_id,
        )

    catalog_tables: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(dataset_refs):
        spec = record_specs[role]
        row = role_rows[role]
        summary_ref = summary_refs[role]
        catalog_tables.setdefault(spec.catalog_role, []).append(
            {
                "stage": str(row["phase"]),
                "role": role,
                "table_id": f"{scope_id}-{_slug(role)}-table",
                "dataset_manifest_ref": dataset_refs[role],
                "summary_ref": {
                    **summary_ref,
                    "dataset_manifest_ref": dataset_refs[role],
                },
                "record_schema_id": spec.record_schema_id,
                "primary_key": list(spec.primary_key),
                "valid_time_field": spec.valid_time_field,
                "available_time_field": spec.available_time_field,
                "selection_policy": ("available_at_or_before_cutoff_then_latest_valid_revision"),
                "conflict_policy": "conflict_is_invalid_no_fallback",
            }
        )
    generation_catalogs: dict[str, Mapping[str, Any]] = {}
    catalog_refs: dict[str, Mapping[str, Any]] = {}
    for catalog_role, tables in sorted(catalog_tables.items()):
        if catalog_role not in source_id_by_role:
            _fail(f"catalog role lacks one exact source input: {catalog_role}")
        tables.sort(
            key=lambda value: (
                value["stage"],
                value["role"],
                value["table_id"],
                value["summary_ref"]["relative_path"],
                value["summary_ref"]["byte_sha256"],
                value["dataset_manifest_ref"]["relative_path"],
                value["dataset_manifest_ref"]["byte_sha256"],
            )
        )
        catalog_id = f"{scope_id}-{_slug(catalog_role)}"
        catalog_path = (
            "data/private/v17_sources/protocol-v2/manifests/" f"{catalog_id}.catalog.json"
        )
        catalog = _sealed_document(
            version=GENERATION_CATALOG_VERSION,
            catalog_id=catalog_id,
            generation_id=f"{catalog_id}-generation",
            role=catalog_role,
            phase=str(role_rows[catalog_role]["phase"]),
            market="CN",
            cutoff=cutoff,
            created_at=created_at,
            source_manifest_ref=manifest_ref,
            table_ordering=_TABLE_ORDERING,
            tables=tables,
        )
        _validate_generated_document(catalog, expected_version=GENERATION_CATALOG_VERSION)
        generation_catalogs[catalog_path] = catalog
        catalog_refs[catalog_role] = _artifact_ref(
            catalog,
            catalog_path,
            artifact_id=catalog_id,
        )
    unused_catalog_roles = sorted(
        role
        for role, row in role_rows.items()
        if (
            row.get("schema_version") == _GENERATION_CATALOG_SCHEMA
            and role in source_id_by_role
            and role not in catalog_refs
        )
    )
    if unused_catalog_roles:
        _fail(f"catalog roles have no exact dataset table closure: {unused_catalog_roles}")

    bindings: list[dict[str, Any]] = []
    for role in sorted(dataset_refs):
        spec = record_specs[role]
        disposition_ref = disposition_refs[role]
        disposition_path = str(disposition_ref["relative_path"])
        disposition = dispositions[disposition_path]
        bindings.append(
            {
                "stage": str(role_rows[role]["phase"]),
                "role": role,
                "catalog_ref": catalog_refs[spec.catalog_role],
                "summary_ref": summary_refs[role],
                "dataset_manifest_ref": dataset_refs[role],
                "disposition_id": disposition["disposition_id"],
                "observation_disposition_ref": disposition_ref,
            }
        )
    bindings.sort(
        key=lambda value: (
            value["stage"],
            value["role"],
            value["catalog_ref"]["relative_path"],
            value["catalog_ref"]["byte_sha256"],
            value["summary_ref"]["relative_path"],
            value["summary_ref"]["byte_sha256"],
            value["dataset_manifest_ref"]["relative_path"],
            value["dataset_manifest_ref"]["byte_sha256"],
            value["disposition_id"],
        )
    )
    binding_set_id = f"{scope_id}-bindings"
    binding_set_path = "data/private/v17_sources/protocol-v2/manifests/" f"{scope_id}.bindings.json"
    binding_set = _sealed_document(
        version=SOURCE_BINDING_SET_VERSION,
        binding_set_id=binding_set_id,
        market="CN",
        cutoff=cutoff,
        source_manifest_ref=manifest_ref,
        binding_ordering=_BINDING_ORDERING,
        bindings=bindings,
    )
    _validate_generated_document(binding_set, expected_version=SOURCE_BINDING_SET_VERSION)

    locator_id = f"{scope_id}-locator"
    locator_path = f"data/private/v17_sources/protocol-v2/locators/{locator_id}.json"
    locator = _sealed_document(
        version=SOURCE_LOCATOR_VERSION,
        locator_id=locator_id,
        market="CN",
        cutoff=cutoff,
        created_at=created_at,
        binding_set_ref=_artifact_ref(
            binding_set,
            binding_set_path,
            artifact_id=binding_set_id,
        ),
    )
    _validate_generated_document(locator, expected_version=SOURCE_LOCATOR_VERSION)

    dag = {
        "source_role_matrix": role_matrix,
        "source_objects": source_objects,
        "dataset_manifests": dataset_manifests,
        "observation_dispositions": dispositions,
        "source_manifest": source_manifest,
        "source_manifest_path": manifest_path,
        "generation_catalogs": generation_catalogs,
        "summaries": summaries,
        "source_binding_set": binding_set,
        "source_binding_set_path": binding_set_path,
        "source_locator": locator,
        "source_locator_path": locator_path,
    }
    validated_locator = _invoke_dag_validator(
        dag_validator,
        dataset_record_schema_registry=record_registry,
        dag=dag,
    )
    if dict(validated_locator) != locator:
        _fail("source DAG validator did not return the exact terminal locator")

    intents: list[WriteIntent] = []
    for path, raw in sorted(source_objects.items()):
        intents.append(
            _write_intent(
                sequence=0,
                kind="SOURCE_OBJECT",
                relative_path=path,
                payload=raw,
            )
        )
    for path, document in sorted(dataset_manifests.items()):
        intents.append(
            _write_intent(
                sequence=10,
                kind="DATASET_MANIFEST",
                relative_path=path,
                payload=canonical_resource_bytes(document),
            )
        )
    for path, document in sorted(dispositions.items()):
        intents.append(
            _write_intent(
                sequence=10,
                kind="OBSERVATION_DISPOSITION",
                relative_path=path,
                payload=canonical_resource_bytes(document),
            )
        )
    intents.append(
        _write_intent(
            sequence=20,
            kind="SOURCE_MANIFEST",
            relative_path=manifest_path,
            payload=canonical_resource_bytes(source_manifest),
        )
    )
    for path, document in sorted(summaries.items()):
        intents.append(
            _write_intent(
                sequence=30,
                kind="DATASET_SUMMARY",
                relative_path=path,
                payload=canonical_resource_bytes(document),
            )
        )
    for path, document in sorted(generation_catalogs.items()):
        intents.append(
            _write_intent(
                sequence=40,
                kind="GENERATION_CATALOG",
                relative_path=path,
                payload=canonical_resource_bytes(document),
            )
        )
    intents.append(
        _write_intent(
            sequence=50,
            kind="SOURCE_BINDING_SET",
            relative_path=binding_set_path,
            payload=canonical_resource_bytes(binding_set),
        )
    )
    intents.append(
        _write_intent(
            sequence=60,
            kind="SOURCE_LOCATOR",
            relative_path=locator_path,
            payload=canonical_resource_bytes(locator),
        )
    )
    intents.sort(key=lambda value: (value.sequence, value.relative_path))
    intent_paths = [intent.relative_path for intent in intents]
    if len(intent_paths) != len(set(intent_paths)) or len(
        {path.casefold() for path in intent_paths}
    ) != len(intent_paths):
        _fail("planned write paths have an exact or casefold collision")
    if intents[-1].kind != "SOURCE_LOCATOR":
        _fail("source locator must be the final write intent")
    for intent in intents:
        if hashlib.sha256(intent.payload).hexdigest() != intent.byte_sha256:
            _fail(f"write intent byte identity mismatch: {intent.relative_path}")

    return SourceDagPlan(
        scope_id=scope_id,
        write_intents=tuple(intents),
        source_objects=dict(source_objects),
        dataset_manifests=dict(dataset_manifests),
        observation_dispositions=dict(dispositions),
        source_manifest=dict(source_manifest),
        source_manifest_path=manifest_path,
        generation_catalogs=dict(generation_catalogs),
        summaries=dict(summaries),
        source_binding_set=dict(binding_set),
        source_binding_set_path=binding_set_path,
        source_locator=dict(locator),
        source_locator_path=locator_path,
    )


__all__ = [
    "DatasetInspector",
    "DatasetRecordSpec",
    "DatasetShardFacts",
    "ObjectCrossValidator",
    "RecordRegistryValidator",
    "RecordSpecResolver",
    "SourceDagPlan",
    "SourceFile",
    "SourcePlanningError",
    "WriteIntent",
    "plan_source_dag",
]
