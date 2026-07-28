"""Pure protocol-v2 namespace mapping and collision classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Any, Final, Mapping

from .canonical import load_canonical_resource
from .identities import (
    IdentityContractError,
    require_path_id,
    require_registry_token,
    require_sha256,
)
from .limits import ContractLimitError, require_nonnegative_int

NAMESPACE_MAP_VERSION: Final = "myquant.v17.v2.namespace-map.v1"
PROTOCOL_VERSION: Final = "myquant.v17.v2"
RESULTS_ROOT: Final = PurePosixPath("results/v17_shadow/protocol-v2")
SOURCES_ROOT: Final = PurePosixPath("data/private/v17_sources/protocol-v2")
_RESOURCE_PATH: Final = Path(__file__).with_name("resources") / "namespace_map.v1.json"


class NamespaceContractError(ValueError):
    """Raised when namespace identity or inventory data is invalid."""

    exit_code = 2


class NodeKind(str, Enum):
    MISSING = "MISSING"
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"
    SYMLINK = "SYMLINK"
    BROKEN_SYMLINK = "BROKEN_SYMLINK"
    OTHER = "OTHER"


class CollisionKind(str, Enum):
    CLEAR = "CLEAR"
    EXPECTED_EXISTING = "EXPECTED_EXISTING"
    CROSS_PROTOCOL_ID_COLLISION = "CROSS_PROTOCOL_ID_COLLISION"
    INVENTORY_INCOMPLETE = "INVENTORY_INCOMPLETE"
    INVENTORY_INCONSISTENT = "INVENTORY_INCONSISTENT"
    ANCESTOR_SYMLINK = "ANCESTOR_SYMLINK"
    ANCESTOR_BROKEN_SYMLINK = "ANCESTOR_BROKEN_SYMLINK"
    ANCESTOR_NOT_DIRECTORY = "ANCESTOR_NOT_DIRECTORY"
    LEAF_SYMLINK = "LEAF_SYMLINK"
    LEAF_BROKEN_SYMLINK = "LEAF_BROKEN_SYMLINK"
    LEAF_KIND_COLLISION = "LEAF_KIND_COLLISION"


@dataclass(frozen=True)
class NamespaceSpec:
    namespace_id: str
    kind: str
    path_template: str


@dataclass(frozen=True)
class CollisionReport:
    outcome: CollisionKind
    target: str
    conflict_path: str | None

    @property
    def is_collision(self) -> bool:
        return self.outcome not in {CollisionKind.CLEAR, CollisionKind.EXPECTED_EXISTING}

    @property
    def safe_to_initialize(self) -> bool:
        return self.outcome is CollisionKind.CLEAR


@dataclass(frozen=True)
class ContentObjectExpectation:
    byte_sha256: str
    size_bytes: int
    metadata_sha256: str
    suffix: str


@dataclass(frozen=True)
class ContentObjectObservation:
    path: str
    kind: NodeKind
    mode: int
    link_count: int
    size_bytes: int
    byte_sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class ContentReuseDecision:
    allowed: bool
    reason: str


def _canonical_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if type(value) is not str or "\\" in value:
        raise NamespaceContractError(f"{label} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise NamespaceContractError(f"{label} must be a canonical POSIX relative path")
    return path


def _load_namespace_specs() -> Mapping[str, NamespaceSpec]:
    try:
        payload = load_canonical_resource(
            _RESOURCE_PATH.read_bytes(),
            label="v17 v2 namespace map",
        )
    except (OSError, ValueError) as exc:
        raise NamespaceContractError("v17 v2 namespace map is invalid") from exc
    if type(payload) is not dict or set(payload) != {
        "namespaces",
        "protocol_version",
        "results_root",
        "sources_root",
        "version",
    }:
        raise NamespaceContractError("v17 v2 namespace map shape mismatch")
    if payload["version"] != NAMESPACE_MAP_VERSION:
        raise NamespaceContractError("v17 v2 namespace map version mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise NamespaceContractError("v17 v2 namespace map protocol version mismatch")
    if payload["results_root"] != str(RESULTS_ROOT):
        raise NamespaceContractError("v17 v2 results root mismatch")
    if payload["sources_root"] != str(SOURCES_ROOT):
        raise NamespaceContractError("v17 v2 sources root mismatch")
    entries = payload["namespaces"]
    if type(entries) is not list:
        raise NamespaceContractError("v17 v2 namespaces must be an array")
    specs: dict[str, NamespaceSpec] = {}
    previous_id: str | None = None
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"id", "kind", "path_template"}:
            raise NamespaceContractError(f"namespace entry {index} shape mismatch")
        namespace_id = entry["id"]
        if type(namespace_id) is not str:
            raise NamespaceContractError(f"namespace entry {index} ID is invalid")
        if previous_id is not None and namespace_id <= previous_id:
            raise NamespaceContractError("namespace entries are not canonically ordered")
        previous_id = namespace_id
        if namespace_id in specs:
            raise NamespaceContractError(f"duplicate namespace ID: {namespace_id}")
        kind = entry["kind"]
        if kind not in {"directory", "file"}:
            raise NamespaceContractError(f"namespace {namespace_id} kind is invalid")
        template = entry["path_template"]
        canonical_template = template.replace("{run_id}", "run-id")
        path = _canonical_relative_path(
            canonical_template,
            label=f"namespace {namespace_id} path template",
        )
        if path != RESULTS_ROOT and RESULTS_ROOT not in path.parents:
            if path != SOURCES_ROOT and SOURCES_ROOT not in path.parents:
                raise NamespaceContractError(f"namespace {namespace_id} escapes protocol-v2 roots")
        if "{" in canonical_template or "}" in canonical_template:
            raise NamespaceContractError(
                f"namespace {namespace_id} has an unknown path placeholder"
            )
        specs[namespace_id] = NamespaceSpec(namespace_id, kind, template)
    return MappingProxyType(specs)


NAMESPACE_MAP: Final[Mapping[str, NamespaceSpec]] = _load_namespace_specs()
NAMESPACE_IDS: Final = frozenset(NAMESPACE_MAP)


def namespace_spec(namespace_id: Any) -> NamespaceSpec:
    try:
        token = require_registry_token(
            namespace_id,
            registry=NAMESPACE_IDS,
            label="namespace ID",
        )
    except IdentityContractError as exc:
        raise NamespaceContractError(str(exc)) from exc
    return NAMESPACE_MAP[token]


def namespace_path(namespace_id: Any, *, run_id: Any | None = None) -> PurePosixPath:
    """Resolve one frozen relative path without touching the filesystem."""

    spec = namespace_spec(namespace_id)
    needs_run_id = "{run_id}" in spec.path_template
    if needs_run_id:
        try:
            canonical_run_id = require_path_id(run_id, label="run_id")
        except IdentityContractError as exc:
            raise NamespaceContractError(str(exc)) from exc
        value = spec.path_template.replace("{run_id}", canonical_run_id)
    else:
        if run_id is not None:
            raise NamespaceContractError(f"{spec.namespace_id} does not accept run_id")
        value = spec.path_template
    return _canonical_relative_path(value, label=f"{spec.namespace_id} path")


def node_kind_from_mode(mode: Any, *, link_target_exists: bool | None = None) -> NodeKind:
    """Classify caller-supplied ``lstat`` mode bits without making a syscall."""

    if type(mode) is not int or mode < 0:
        raise NamespaceContractError("lstat mode must be a nonnegative integer")
    if stat.S_ISLNK(mode):
        if type(link_target_exists) is not bool:
            raise NamespaceContractError("symlink inventory must state target existence")
        return NodeKind.SYMLINK if link_target_exists else NodeKind.BROKEN_SYMLINK
    if link_target_exists is not None:
        raise NamespaceContractError("target existence applies only to symlinks")
    if stat.S_ISREG(mode):
        return NodeKind.FILE
    if stat.S_ISDIR(mode):
        return NodeKind.DIRECTORY
    return NodeKind.OTHER


def _inventory_by_path(inventory: Mapping[Any, Any]) -> dict[str, NodeKind]:
    if not isinstance(inventory, Mapping):
        raise NamespaceContractError("lstat inventory must be a mapping")
    normalized: dict[str, NodeKind] = {}
    for raw_path, raw_kind in inventory.items():
        path = str(_canonical_relative_path(raw_path, label="inventory path"))
        if path in normalized:
            raise NamespaceContractError(f"duplicate normalized inventory path: {path}")
        try:
            kind = raw_kind if type(raw_kind) is NodeKind else NodeKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise NamespaceContractError(f"invalid node kind for {path}") from exc
        normalized[path] = kind
    return normalized


def classify_namespace_collision(
    target: str | PurePosixPath,
    *,
    expected_kind: str,
    inventory: Mapping[Any, Any],
    cross_protocol_identity_present: bool = False,
) -> CollisionReport:
    """Classify an exact target from complete caller-provided ``lstat`` facts."""

    if type(cross_protocol_identity_present) is not bool:
        raise NamespaceContractError("cross-protocol identity flag must be boolean")
    target_path = _canonical_relative_path(str(target), label="collision target")
    if (
        target_path != RESULTS_ROOT
        and RESULTS_ROOT not in target_path.parents
        and target_path != SOURCES_ROOT
        and SOURCES_ROOT not in target_path.parents
    ):
        raise NamespaceContractError("collision target is outside protocol-v2 roots")
    target_text = str(target_path)
    if expected_kind not in {"directory", "file"}:
        raise NamespaceContractError("expected kind must be directory or file")
    if cross_protocol_identity_present:
        return CollisionReport(
            CollisionKind.CROSS_PROTOCOL_ID_COLLISION,
            target_text,
            None,
        )
    facts = _inventory_by_path(inventory)
    required = [
        str(PurePosixPath(*target_path.parts[:index]))
        for index in range(1, len(target_path.parts) + 1)
    ]
    for path in required:
        if path not in facts:
            return CollisionReport(CollisionKind.INVENTORY_INCOMPLETE, target_text, path)

    missing_ancestor: str | None = None
    for path in required[:-1]:
        kind = facts[path]
        if missing_ancestor is not None:
            if kind is not NodeKind.MISSING:
                return CollisionReport(
                    CollisionKind.INVENTORY_INCONSISTENT,
                    target_text,
                    path,
                )
            continue
        if kind is NodeKind.MISSING:
            missing_ancestor = path
        elif kind is NodeKind.SYMLINK:
            return CollisionReport(CollisionKind.ANCESTOR_SYMLINK, target_text, path)
        elif kind is NodeKind.BROKEN_SYMLINK:
            return CollisionReport(
                CollisionKind.ANCESTOR_BROKEN_SYMLINK,
                target_text,
                path,
            )
        elif kind is not NodeKind.DIRECTORY:
            return CollisionReport(CollisionKind.ANCESTOR_NOT_DIRECTORY, target_text, path)

    leaf_kind = facts[target_text]
    if missing_ancestor is not None and leaf_kind is not NodeKind.MISSING:
        return CollisionReport(
            CollisionKind.INVENTORY_INCONSISTENT,
            target_text,
            target_text,
        )
    if leaf_kind is NodeKind.MISSING:
        return CollisionReport(CollisionKind.CLEAR, target_text, None)
    if leaf_kind is NodeKind.SYMLINK:
        return CollisionReport(CollisionKind.LEAF_SYMLINK, target_text, target_text)
    if leaf_kind is NodeKind.BROKEN_SYMLINK:
        return CollisionReport(
            CollisionKind.LEAF_BROKEN_SYMLINK,
            target_text,
            target_text,
        )
    expected_node = NodeKind.FILE if expected_kind == "file" else NodeKind.DIRECTORY
    if leaf_kind is expected_node:
        return CollisionReport(CollisionKind.EXPECTED_EXISTING, target_text, None)
    return CollisionReport(CollisionKind.LEAF_KIND_COLLISION, target_text, target_text)


def derive_content_object_path(byte_sha256: Any, *, suffix: Any) -> PurePosixPath:
    """Derive the only valid content-addressed v2 source-object path."""

    try:
        digest = require_sha256(byte_sha256, label="content object SHA-256")
        canonical_suffix = require_registry_token(
            suffix,
            registry=frozenset({"blob", "json", "parquet"}),
            label="content object suffix",
        )
    except IdentityContractError as exc:
        raise NamespaceContractError(str(exc)) from exc
    return namespace_path("SOURCE_OBJECTS") / digest[:2] / f"{digest}.{canonical_suffix}"


def classify_content_object_reuse(
    expected: ContentObjectExpectation,
    observed: ContentObjectObservation,
) -> ContentReuseDecision:
    """Permit reuse only for an exact, private, single-link regular object."""

    try:
        digest = require_sha256(expected.byte_sha256, label="expected byte SHA-256")
        metadata = require_sha256(
            expected.metadata_sha256,
            label="expected metadata SHA-256",
        )
        expected_size = require_nonnegative_int(
            expected.size_bytes,
            label="expected object size",
        )
        derived = str(derive_content_object_path(digest, suffix=expected.suffix))
    except (IdentityContractError, ContractLimitError, NamespaceContractError) as exc:
        raise NamespaceContractError(str(exc)) from exc
    checks = (
        (type(observed.kind) is NodeKind and observed.kind is NodeKind.FILE, "not_regular"),
        (observed.mode == 0o600 and type(observed.mode) is int, "mode_not_0600"),
        (
            observed.link_count == 1 and type(observed.link_count) is int,
            "link_count_not_one",
        ),
        (observed.path == derived, "path_mismatch"),
        (
            type(observed.size_bytes) is int and observed.size_bytes == expected_size,
            "size_mismatch",
        ),
        (observed.byte_sha256 == digest, "digest_mismatch"),
        (observed.metadata_sha256 == metadata, "metadata_mismatch"),
    )
    for valid, reason in checks:
        if not valid:
            return ContentReuseDecision(False, reason)
    return ContentReuseDecision(True, "exact_reuse")


__all__ = [
    "CollisionKind",
    "CollisionReport",
    "ContentObjectExpectation",
    "ContentObjectObservation",
    "ContentReuseDecision",
    "NAMESPACE_IDS",
    "NAMESPACE_MAP",
    "NAMESPACE_MAP_VERSION",
    "NamespaceContractError",
    "NamespaceSpec",
    "NodeKind",
    "PROTOCOL_VERSION",
    "RESULTS_ROOT",
    "SOURCES_ROOT",
    "classify_content_object_reuse",
    "classify_namespace_collision",
    "derive_content_object_path",
    "namespace_path",
    "namespace_spec",
    "node_kind_from_mode",
]
