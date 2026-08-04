"""Read-only, bounded compatibility reader for explicitly allowed V17 v4 graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final, Mapping, Sequence

from quant_investor.v17_v4_contract import (
    artifact_identity_field as v4_identity_field,
    load_canonical_artifact as load_v4_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.resources import (
    read_packaged_asset as read_v4_packaged_asset,
)
from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    validate_semantic_sha,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.resources import (
    COMPATIBILITY_POLICY_PATH,
    load_compatibility_policy,
    read_packaged_asset,
    verify_predecessor,
)

_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY: Final = getattr(os, "O_DIRECTORY", 0)
_ARTIFACT_REF_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
)
_SOURCE_REF_FIELDS: Final = frozenset(
    {
        "as_of",
        "available_at",
        "byte_sha256",
        "media_type",
        "relative_path",
        "role",
        "status",
    }
)
_POLICY_REF_FIELDS: Final = frozenset(
    {
        "byte_sha256",
        "relative_path",
        "semantic_sha256",
        "version",
    }
)
_ALLOWED_REGIME_POLICY_PATHS: Final = frozenset(
    {
        "resources/regime_inference_policy.v1.json",
        "resources/regime_inference_policy.v2.json",
    }
)
_FORBIDDEN_TRUE_FIELDS: Final = frozenset(
    {
        "broker",
        "broker_authority",
        "canary_evidence_eligible",
        "execution",
        "execution_authority",
        "formal_activation_eligible",
        "formal_research_publication",
        "formal_research_publication_eligible",
        "order",
        "performance_evidence_eligible",
        "production_default_eligible",
        "promotion_eligible",
        "provider_authority",
        "provider_invoked",
        "research_runtime_default",
        "trade",
    }
)
_V4_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class V4CompatibilityError(RuntimeError):
    """Raised when a V17 v4 predecessor input cannot be trusted."""

    exit_code = 2


@dataclass(frozen=True)
class V4ClosureNode:
    artifact_id: str
    byte_sha256: str
    relative_path: str
    semantic_sha256: str
    validation_mode: str
    version: str


@dataclass(frozen=True)
class V4TerminalBinding:
    available_at: str
    byte_sha256: str
    media_type: str
    pointer: str
    relative_path: str
    role: str


@dataclass(frozen=True)
class V4CompatibilityRead:
    closure: tuple[V4ClosureNode, ...]
    compatibility_policy_byte_sha256: str
    document: Mapping[str, Any]
    documents: Mapping[str, Mapping[str, Any]]
    predecessor_git_commit: str
    predecessor_package_manifest_byte_sha256: str
    predecessor_package_manifest_relative_path: str
    predecessor_protocol_version: str
    predecessor_runtime_manifest_byte_sha256: str
    predecessor_runtime_manifest_relative_path: str
    root_ref: Mapping[str, str]
    terminal_bindings: tuple[V4TerminalBinding, ...]


def _instant(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        raise V4CompatibilityError(f"{label} must be a UTC-second timestamp")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise V4CompatibilityError(f"{label} must be a UTC-second timestamp") from exc
    if parsed.strftime(_UTC_FORMAT) != value:
        raise V4CompatibilityError(f"{label} is not canonical")
    return parsed


def _file_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_workspace_root(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V4CompatibilityError("workspace_root must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise V4CompatibilityError("workspace_root is unavailable") from exc
    if resolved != path or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise V4CompatibilityError("workspace_root must be a real canonical directory")
    return path


def _case_exact_entry(directory_fd: int, name: str) -> None:
    try:
        entries = os.listdir(directory_fd)
    except OSError as exc:
        raise V4CompatibilityError("trusted directory cannot be enumerated") from exc
    matches = [entry for entry in entries if entry.casefold() == name.casefold()]
    if matches != [name]:
        raise V4CompatibilityError("path component is absent or casefold-ambiguous")


def _secure_read_relative(
    workspace_root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    try:
        normalized = require_relative_path(relative_path)
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    parts = PurePosixPath(normalized).parts
    root_fd = -1
    directory_fd = -1
    file_fd = -1
    try:
        root_fd = os.open(workspace_root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        directory_fd = root_fd
        for part in parts[:-1]:
            _case_exact_entry(directory_fd, part)
            before = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise V4CompatibilityError("path parent is not a real directory")
            child_fd = os.open(
                part,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=directory_fd,
            )
            after = os.fstat(child_fd)
            if _file_fingerprint(before) != _file_fingerprint(after):
                os.close(child_fd)
                raise V4CompatibilityError("path parent changed during open")
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = child_fd
        name = parts[-1]
        _case_exact_entry(directory_fd, name)
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes:
            raise V4CompatibilityError("artifact is not a bounded owner file")
        file_fd = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        if _file_fingerprint(before) != _file_fingerprint(opened):
            raise V4CompatibilityError("artifact changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(file_fd, min(1_048_576, remaining))
            if not chunk:
                raise V4CompatibilityError("artifact was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            raise V4CompatibilityError("artifact grew during read")
        closed = os.fstat(file_fd)
        if _file_fingerprint(opened) != _file_fingerprint(closed):
            raise V4CompatibilityError("artifact changed during read")
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            raise V4CompatibilityError("artifact read length mismatch")
        return raw
    except V4CompatibilityError:
        raise
    except OSError as exc:
        raise V4CompatibilityError("artifact secure read failed") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0 and directory_fd != root_fd:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _allowed_row(policy: Mapping[str, Any], version: str) -> Mapping[str, Any]:
    rows = [row for row in policy["allowed_artifacts"] if row["version"] == version]
    if len(rows) != 1:
        raise V4CompatibilityError(f"V17 v4 artifact version is not allowed: {version}")
    return rows[0]


def _path_allowed(relative_path: str, row: Mapping[str, Any]) -> None:
    if not any(
        relative_path == prefix or relative_path.startswith(f"{prefix}/")
        for prefix in row["allowed_path_prefixes"]
    ):
        raise V4CompatibilityError("V17 v4 artifact is outside its allowed namespace")


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _walk(value: Any, *, pointer: str = "") -> tuple[tuple[str, Any], ...]:
    result: list[tuple[str, Any]] = [(pointer or "/", value)]
    if type(value) is dict:
        for key in sorted(value):
            result.extend(_walk(value[key], pointer=f"{pointer}/{_escape_pointer(key)}"))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.extend(_walk(item, pointer=f"{pointer}/{index}"))
    return tuple(result)


def _resolve_pointer(value: Any, pattern: str) -> tuple[tuple[str, Any], ...]:
    tokens = pattern.split("/")[1:]
    result: list[tuple[str, Any]] = []

    def visit(current: Any, position: int, pointer: str) -> None:
        if position == len(tokens):
            result.append((pointer or "/", current))
            return
        token = tokens[position].replace("~1", "/").replace("~0", "~")
        if token == "*":
            if type(current) is list:
                for index, child in enumerate(current):
                    visit(child, position + 1, f"{pointer}/{index}")
            elif type(current) is dict:
                for key in sorted(current):
                    visit(
                        current[key],
                        position + 1,
                        f"{pointer}/{_escape_pointer(key)}",
                    )
            return
        if type(current) is dict and token in current:
            visit(
                current[token],
                position + 1,
                f"{pointer}/{_escape_pointer(token)}",
            )
        elif type(current) is list and token.isdigit() and int(token) < len(current):
            visit(current[int(token)], position + 1, f"{pointer}/{token}")

    visit(value, 0, "")
    return tuple(result)


def _is_artifact_ref(value: Any) -> bool:
    return type(value) is dict and set(value) == _ARTIFACT_REF_FIELDS


def _is_packaged_regime_policy_ref(value: Any) -> bool:
    return type(value) is dict and set(value) == _POLICY_REF_FIELDS


def _validate_packaged_regime_policy_ref(value: Any, *, label: str) -> None:
    if not _is_packaged_regime_policy_ref(value):
        raise V4CompatibilityError(f"{label} is not an exact packaged policy reference")
    try:
        relative_path = require_relative_path(
            value["relative_path"],
            label=f"{label}.relative_path",
        )
        byte_sha = require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        semantic_sha = require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
        version = require_identifier(value["version"], label=f"{label}.version")
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    if relative_path not in _ALLOWED_REGIME_POLICY_PATHS:
        raise V4CompatibilityError(f"{label} packaged policy path is not allowed")
    try:
        raw = read_v4_packaged_asset(relative_path)
        document = load_canonical_resource(raw, label=relative_path)
        if type(document) is not dict:
            raise V4CompatibilityError(f"{label} packaged policy is not an object")
        validate_semantic_sha(document)
    except V4CompatibilityError:
        raise
    except Exception as exc:
        raise V4CompatibilityError(f"{label} packaged policy verification failed") from exc
    if (
        hashlib.sha256(raw).hexdigest() != byte_sha
        or document.get("semantic_sha256") != semantic_sha
        or document.get("version") != version
    ):
        raise V4CompatibilityError(f"{label} packaged policy identity mismatch")


def _validate_artifact_ref(
    value: Any,
    *,
    expected_strategy_id: str,
    decision_cutoff: datetime,
    target_versions: Sequence[str],
    label: str,
) -> dict[str, str]:
    if not _is_artifact_ref(value):
        raise V4CompatibilityError(f"{label} is not an exact artifact reference")
    try:
        artifact_id = require_identifier(value["artifact_id"], label=f"{label}.artifact_id")
        version = require_identifier(
            value["artifact_version"],
            label=f"{label}.artifact_version",
        )
        byte_sha = require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        semantic_sha = require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
        path = require_relative_path(value["relative_path"], label=f"{label}.relative_path")
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    if version not in target_versions:
        raise V4CompatibilityError(f"{label} target version is not allowed")
    if value["strategy_id"] != expected_strategy_id:
        raise V4CompatibilityError(f"{label} strategy binding mismatch")
    if _instant(value["cutoff"], label=f"{label}.cutoff") > decision_cutoff:
        raise V4CompatibilityError(f"{label} cutoff is in the future")
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": byte_sha,
        "cutoff": value["cutoff"],
        "relative_path": path,
        "semantic_sha256": semantic_sha,
        "strategy_id": expected_strategy_id,
    }


def _validate_source_binding(
    value: Any,
    *,
    decision_cutoff: datetime,
    pointer: str,
) -> V4TerminalBinding:
    if type(value) is not dict or set(value) != _SOURCE_REF_FIELDS:
        raise V4CompatibilityError(f"{pointer} is not an exact terminal source binding")
    try:
        sha = require_sha256(value["byte_sha256"], label=f"{pointer}.byte_sha256")
        path = require_relative_path(value["relative_path"], label=f"{pointer}.relative_path")
        role = require_identifier(value["role"], label=f"{pointer}.role")
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    available_at = value["available_at"]
    if (
        value["status"] != "VERIFIED"
        or type(value["media_type"]) is not str
        or not value["media_type"]
        or _instant(available_at, label=f"{pointer}.available_at") > decision_cutoff
    ):
        raise V4CompatibilityError(f"{pointer} terminal source binding is invalid")
    return V4TerminalBinding(
        available_at=available_at,
        byte_sha256=sha,
        media_type=value["media_type"],
        pointer=pointer,
        relative_path=path,
        role=role,
    )


def _validate_cardinality(cardinality: str, count: int, *, pointer: str) -> None:
    if (
        (cardinality == "EXACT_ONE" and count != 1)
        or (cardinality in {"ONE_OR_MORE", "ONE_PER_PARENT_ROW"} and count < 1)
        or (cardinality == "ZERO_OR_ONE" and count > 1)
    ):
        raise V4CompatibilityError(f"{pointer} cardinality mismatch")


def _validate_no_authority(value: Any) -> None:
    for pointer, child in _walk(value):
        key = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
        if (
            key == "authority"
            and type(child) is dict
            and any(item is not False for item in child.values())
        ):
            raise V4CompatibilityError("V17 v4 artifact grants authority")
        if key in _FORBIDDEN_TRUE_FIELDS and child is not False:
            raise V4CompatibilityError("V17 v4 artifact grants authority")


def _registered_document(
    raw: bytes,
    *,
    path: str,
    version: str,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    try:
        validated = load_v4_artifact(raw, expected_version=version, label=path)
        if type(validated) is dict:
            document = dict(validated)
        elif hasattr(validated, "as_dict"):
            document = dict(validated.as_dict())
        else:
            raise V4CompatibilityError("V17 v4 registered artifact is not readable")
        identity_field = v4_identity_field(version)
        if (
            identity_field != row["identity_field"]
            or document.get("protocol_version") != "myquant.v17.v4"
        ):
            raise V4CompatibilityError("V17 v4 registered identity mismatch")
        artifact_id = require_identifier(document[identity_field], label=identity_field)
        semantic_sha = require_sha256(document["semantic_sha256"])
    except V4CompatibilityError:
        raise
    except Exception as exc:
        raise V4CompatibilityError("V17 v4 schema or semantic validation failed") from exc
    return dict(document), artifact_id, semantic_sha


def _generic_terminal_document(
    raw: bytes,
    *,
    path: str,
    expected_ref: Mapping[str, str],
) -> tuple[dict[str, Any], str, str]:
    try:
        value = load_canonical_resource(raw, label=path)
        if type(value) is not dict:
            raise V4CompatibilityError("terminal artifact root must be an object")
        document = validate_semantic_sha(value)
        artifact_id = require_identifier(document["artifact_id"], label="artifact_id")
        version = require_identifier(document["artifact_version"], label="artifact_version")
        semantic_sha = require_sha256(document["semantic_sha256"])
    except V4CompatibilityError:
        raise
    except Exception as exc:
        raise V4CompatibilityError("terminal artifact generic seal failed") from exc
    if (
        artifact_id != expected_ref["artifact_id"]
        or version != expected_ref["artifact_version"]
        or document.get("protocol_version") != "myquant.v17.v4"
        or document.get("authority") != _V4_NO_AUTHORITY
        or document.get("strategy_id") != expected_ref["strategy_id"]
        or document.get("cutoff") != expected_ref["cutoff"]
    ):
        raise V4CompatibilityError("terminal artifact identity mismatch")
    return dict(document), artifact_id, semantic_sha


def _parquet_metadata(
    raw: bytes,
    *,
    expected_ref: Mapping[str, str],
    limits: Mapping[str, int],
) -> tuple[str, str]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(io.BytesIO(raw))
        metadata = parquet.metadata
        schema_metadata = parquet.schema_arrow.metadata or {}
        decoded = {
            key.decode("ascii"): value.decode("ascii") for key, value in schema_metadata.items()
        }
        artifact_id = require_identifier(decoded["artifact_id"], label="parquet artifact_id")
        semantic_sha = require_sha256(
            decoded["semantic_sha256"],
            label="parquet semantic_sha256",
        )
    except Exception as exc:
        raise V4CompatibilityError("V17 v4 parquet metadata validation failed") from exc
    if (
        metadata.num_rows > limits["max_parquet_rows"]
        or metadata.num_row_groups > limits["max_parquet_row_groups"]
    ):
        raise V4CompatibilityError("V17 v4 parquet resource limit exceeded")
    if (
        artifact_id != expected_ref["artifact_id"]
        or decoded.get("artifact_version") != expected_ref["artifact_version"]
        or decoded.get("schema_version") != expected_ref["artifact_version"]
        or decoded.get("cutoff") != expected_ref["cutoff"]
        or decoded.get("strategy_id") != expected_ref["strategy_id"]
        or semantic_sha != expected_ref["semantic_sha256"]
        or _instant(decoded.get("available_at"), label="parquet.available_at")
        > _instant(expected_ref["cutoff"], label="parquet.ref.cutoff")
    ):
        raise V4CompatibilityError("V17 v4 parquet reference binding mismatch")
    return artifact_id, semantic_sha


def _decode_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    payload_json = document.get("payload_json")
    if type(payload_json) is not str:
        raise V4CompatibilityError("stage payload_json is absent")
    raw = payload_json.encode("utf-8", errors="strict")
    try:
        payload = load_canonical_resource(raw, label="stage payload_json")
    except Exception as exc:
        raise V4CompatibilityError("stage payload_json is not canonical") from exc
    if type(payload) is not dict or hashlib.sha256(raw).hexdigest() != document.get(
        "payload_sha256"
    ):
        raise V4CompatibilityError("stage payload_json hash mismatch")
    return dict(payload)


def read_v4_artifact(
    workspace_root: str | os.PathLike[str],
    *,
    relative_path: str,
    expected_byte_sha256: str,
    expected_strategy_id: str,
    decision_cutoff: str,
) -> V4CompatibilityRead:
    """Read one root and its exact allowlisted V17 v4 dependency closure."""

    root = _canonical_workspace_root(workspace_root)
    try:
        expected_sha = require_sha256(expected_byte_sha256)
        strategy = require_identifier(expected_strategy_id, label="expected_strategy_id")
        path = require_relative_path(relative_path)
    except IdentityContractError as exc:
        raise V4CompatibilityError(str(exc)) from exc
    cutoff = _instant(decision_cutoff, label="decision_cutoff")
    predecessor = verify_predecessor()
    policy = load_compatibility_policy()
    limits = policy["closure_limits"]
    policy_raw = read_packaged_asset(COMPATIBILITY_POLICY_PATH)
    nodes: dict[str, V4ClosureNode] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    terminal_bindings: dict[tuple[str, str], V4TerminalBinding] = {}
    active: set[str] = set()
    total_bytes = 0

    def visit(
        child_path: str,
        child_sha: str,
        *,
        expected_ref: Mapping[str, str] | None,
        depth: int,
        is_root: bool = False,
    ) -> None:
        nonlocal total_bytes
        if depth > limits["max_depth"]:
            raise V4CompatibilityError("V17 v4 closure depth limit exceeded")
        if child_path in active:
            raise V4CompatibilityError("V17 v4 closure cycle detected")
        existing = nodes.get(child_path)
        if existing is not None:
            if (
                existing.byte_sha256 != child_sha
                or expected_ref is not None
                and (
                    existing.artifact_id != expected_ref["artifact_id"]
                    or existing.semantic_sha256 != expected_ref["semantic_sha256"]
                    or existing.version != expected_ref["artifact_version"]
                )
            ):
                raise V4CompatibilityError("V17 v4 duplicate closure node conflict")
            return
        if len(nodes) >= limits["max_nodes"]:
            raise V4CompatibilityError("V17 v4 closure node limit exceeded")
        active.add(child_path)
        try:
            raw = _secure_read_relative(
                root,
                child_path,
                max_bytes=limits["max_artifact_bytes"],
            )
            observed_sha = hashlib.sha256(raw).hexdigest()
            if observed_sha != child_sha:
                raise V4CompatibilityError("V17 v4 artifact byte SHA-256 mismatch")
            total_bytes += len(raw)
            if total_bytes > limits["max_closure_bytes"]:
                raise V4CompatibilityError("V17 v4 closure byte limit exceeded")
            if expected_ref is None:
                try:
                    value = load_canonical_resource(raw, label=child_path)
                except Exception as exc:
                    raise V4CompatibilityError("V17 v4 root is not canonical JSON") from exc
                if type(value) is not dict or type(value.get("version")) is not str:
                    raise V4CompatibilityError("V17 v4 root version is absent")
                version = value["version"]
            else:
                version = expected_ref["artifact_version"]
            row = _allowed_row(policy, version)
            _path_allowed(child_path, row)
            if is_root and row["root_admissible"] is not True:
                raise V4CompatibilityError("V17 v4 artifact is not an admissible root")
            mode = row["validation_mode"]
            if mode == "V4_PARQUET_METADATA":
                if expected_ref is None:
                    raise V4CompatibilityError("binary leaf cannot be a closure root")
                artifact_id, semantic_sha = _parquet_metadata(
                    raw,
                    expected_ref=expected_ref,
                    limits=limits,
                )
                document: dict[str, Any] | None = None
            elif mode == "V4_GENERIC_CANONICAL_TERMINAL":
                if expected_ref is None:
                    raise V4CompatibilityError("terminal leaf cannot be a closure root")
                document, artifact_id, semantic_sha = _generic_terminal_document(
                    raw,
                    path=child_path,
                    expected_ref=expected_ref,
                )
                _validate_no_authority(document)
            else:
                document, artifact_id, semantic_sha = _registered_document(
                    raw,
                    path=child_path,
                    version=version,
                    row=row,
                )
                _validate_no_authority(document)
                if document.get("strategy_id") != strategy:
                    raise V4CompatibilityError("V17 v4 strategy binding mismatch")
                artifact_cutoff = _instant(document.get("cutoff"), label="artifact.cutoff")
                if artifact_cutoff > cutoff:
                    raise V4CompatibilityError("V17 v4 artifact cutoff is in the future")
                available_at = document.get("available_at")
                if (
                    available_at is not None
                    and _instant(
                        available_at,
                        label="artifact.available_at",
                    )
                    > cutoff
                ):
                    raise V4CompatibilityError("V17 v4 artifact availability is in the future")
            if expected_ref is not None and (
                artifact_id != expected_ref["artifact_id"]
                or semantic_sha != expected_ref["semantic_sha256"]
            ):
                raise V4CompatibilityError("V17 v4 artifact reference identity mismatch")
            nodes[child_path] = V4ClosureNode(
                artifact_id=artifact_id,
                byte_sha256=observed_sha,
                relative_path=child_path,
                semantic_sha256=semantic_sha,
                validation_mode=mode,
                version=version,
            )
            if document is None:
                return
            documents[child_path] = document
            if mode != "V4_REGISTERED_JSON":
                return
            declared_ref_pointers: set[str] = set()
            terminal_source_pointers: set[str] = set()
            follow_refs: list[tuple[str, dict[str, str]]] = []
            for edge in row["transitive_edges"]:
                edge_mode = edge["mode"]
                if edge_mode == "DECODED_REF_SCAN":
                    payload = _decode_payload(document)
                    resolved = tuple(
                        (f"/@decoded(payload_json){pointer}", value)
                        for pointer, value in _walk(payload)
                        if _is_artifact_ref(value)
                    )
                    partial = [
                        pointer
                        for pointer, value in _walk(payload)
                        if type(value) is dict
                        and ("relative_path" in value or "byte_sha256" in value)
                        and not _is_artifact_ref(value)
                    ]
                    if partial:
                        raise V4CompatibilityError(
                            "stage payload contains a partial external reference"
                        )
                else:
                    resolved = _resolve_pointer(document, edge["json_pointer"])
                _validate_cardinality(
                    edge["cardinality"],
                    len(resolved),
                    pointer=edge["json_pointer"],
                )
                for actual_pointer, value in resolved:
                    if edge_mode == "TERMINAL_SOURCE_BINDING":
                        binding = _validate_source_binding(
                            value,
                            decision_cutoff=cutoff,
                            pointer=f"{child_path}:{actual_pointer}",
                        )
                        terminal_bindings[(child_path, actual_pointer)] = binding
                        terminal_source_pointers.add(actual_pointer)
                        continue
                    reference = _validate_artifact_ref(
                        value,
                        expected_strategy_id=strategy,
                        decision_cutoff=cutoff,
                        target_versions=edge["target_versions"],
                        label=f"{child_path}:{actual_pointer}",
                    )
                    declared_ref_pointers.add(actual_pointer)
                    follow_refs.append((actual_pointer, reference))
            discovered_refs = {
                pointer for pointer, value in _walk(document) if _is_artifact_ref(value)
            }
            packaged_policy_pointers = {
                pointer
                for pointer, value in _walk(document)
                if _is_packaged_regime_policy_ref(value)
            }
            for pointer in sorted(packaged_policy_pointers):
                resolved_value = dict(_walk(document))[pointer]
                _validate_packaged_regime_policy_ref(
                    resolved_value,
                    label=f"{child_path}:{pointer}",
                )
            ordinary_declared = {
                pointer
                for pointer in declared_ref_pointers
                if not pointer.startswith("/@decoded(payload_json)")
            }
            if discovered_refs != ordinary_declared:
                raise V4CompatibilityError(
                    "artifact contains a hidden or undeclared transitive reference"
                )
            for pointer, value in _walk(document):
                if (
                    type(value) is dict
                    and ("relative_path" in value or "byte_sha256" in value)
                    and not _is_artifact_ref(value)
                    and pointer not in terminal_source_pointers
                    and pointer not in packaged_policy_pointers
                ):
                    raise V4CompatibilityError("artifact contains a partial external reference")
            for _, reference in sorted(
                follow_refs,
                key=lambda item: (
                    item[1]["relative_path"],
                    item[1]["byte_sha256"],
                    item[0],
                ),
            ):
                visit(
                    reference["relative_path"],
                    reference["byte_sha256"],
                    expected_ref=reference,
                    depth=depth + 1,
                )
        finally:
            active.discard(child_path)

    visit(path, expected_sha, expected_ref=None, depth=0, is_root=True)
    root_node = nodes[path]
    root_document = documents[path]
    root_ref = {
        "artifact_id": root_node.artifact_id,
        "artifact_version": root_node.version,
        "byte_sha256": root_node.byte_sha256,
        "cutoff": str(root_document["cutoff"]),
        "relative_path": root_node.relative_path,
        "semantic_sha256": root_node.semantic_sha256,
        "strategy_id": strategy,
    }
    return V4CompatibilityRead(
        closure=tuple(nodes[key] for key in sorted(nodes)),
        compatibility_policy_byte_sha256=hashlib.sha256(policy_raw).hexdigest(),
        document=dict(root_document),
        documents={key: dict(documents[key]) for key in sorted(documents)},
        predecessor_git_commit=predecessor["source_git_commit"],
        predecessor_package_manifest_byte_sha256=predecessor.get(
            "package_manifest_byte_sha256",
            "",
        ),
        predecessor_package_manifest_relative_path=policy["predecessor"][
            "package_manifest_relative_path"
        ],
        predecessor_protocol_version=predecessor.get("protocol_version", ""),
        predecessor_runtime_manifest_byte_sha256=predecessor.get(
            "runtime_manifest_byte_sha256",
            "",
        ),
        predecessor_runtime_manifest_relative_path=policy["predecessor"][
            "runtime_manifest_relative_path"
        ],
        root_ref=root_ref,
        terminal_bindings=tuple(terminal_bindings[key] for key in sorted(terminal_bindings)),
    )


__all__ = [
    "V4ClosureNode",
    "V4CompatibilityError",
    "V4CompatibilityRead",
    "V4TerminalBinding",
    "read_v4_artifact",
]
