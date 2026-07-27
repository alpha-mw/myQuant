"""Pure namespace and isolation policy for protocol v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import PurePosixPath
from string import Formatter
from typing import Any, Final

from .canonical import CanonicalContractError, validate_semantic_sha
from .identities import (
    IdentityContractError,
    require_path_id,
    require_registry_token,
    require_sha256,
)
from .resources import PackageResourceError, load_packaged_json

SOURCES_ROOT: Final = "data/private/v17_v3_sources"
RUNS_ROOT: Final = "data/private/v17_v3_runs"
SHADOW_RESULTS_ROOT: Final = "results/v17_v3_shadow"
FORMAL_RESEARCH_RESULTS_ROOT: Final = "results/v17_v3_formal_research"
ROOTS: Final = (
    SOURCES_ROOT,
    RUNS_ROOT,
    SHADOW_RESULTS_ROOT,
    FORMAL_RESEARCH_RESULTS_ROOT,
)
FORBIDDEN_V2_COMPONENTS: Final = frozenset(
    {
        "protocol-v2",
        "v17_v2",
        "v17_v2_contract",
        "v17_v2_formal_research",
        "v17_v2_runs",
        "v17_v2_runtime",
        "v17_v2_shadow",
        "v17_v2_sources",
    }
)


class NamespaceContractError(ValueError):
    """Raised when a v3 path is ambiguous, aliased, or outside its roots."""

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
    OUTSIDE_V3_ROOTS = "OUTSIDE_V3_ROOTS"
    V2_NAMESPACE_COLLISION = "V2_NAMESPACE_COLLISION"
    CASEFOLD_COLLISION = "CASEFOLD_COLLISION"
    INVENTORY_INCOMPLETE = "INVENTORY_INCOMPLETE"
    INVENTORY_INCONSISTENT = "INVENTORY_INCONSISTENT"
    ANCESTOR_SYMLINK = "ANCESTOR_SYMLINK"
    ANCESTOR_BROKEN_SYMLINK = "ANCESTOR_BROKEN_SYMLINK"
    ANCESTOR_NOT_DIRECTORY = "ANCESTOR_NOT_DIRECTORY"
    LEAF_SYMLINK = "LEAF_SYMLINK"
    LEAF_BROKEN_SYMLINK = "LEAF_BROKEN_SYMLINK"
    LEAF_KIND_COLLISION = "LEAF_KIND_COLLISION"


@dataclass(frozen=True)
class CollisionReport:
    outcome: CollisionKind
    target: str
    conflict_path: str | None = None

    @property
    def is_collision(self) -> bool:
        return self.outcome not in {CollisionKind.CLEAR, CollisionKind.EXPECTED_EXISTING}


def canonical_relative_path(value: Any, *, label: str = "path") -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise NamespaceContractError(f"{label} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise NamespaceContractError(f"{label} must be a canonical POSIX relative path")
    if any(part.casefold() in FORBIDDEN_V2_COMPONENTS for part in path.parts):
        raise NamespaceContractError(f"{label} collides with a forbidden v2 namespace")
    return path


def root_for_path(value: Any) -> str:
    path = canonical_relative_path(value)
    matches = [
        root for root in ROOTS if path == PurePosixPath(root) or PurePosixPath(root) in path.parents
    ]
    if len(matches) != 1:
        raise NamespaceContractError("path must be contained by exactly one v3 root")
    return matches[0]


def source_run_path(*, run_id: Any) -> PurePosixPath:
    try:
        token = require_path_id(run_id, label="run_id")
    except IdentityContractError as exc:
        raise NamespaceContractError(str(exc)) from exc
    return PurePosixPath(RUNS_ROOT) / token


@lru_cache(maxsize=1)
def namespace_templates() -> dict[str, tuple[str, str]]:
    try:
        payload = load_packaged_json("resources/namespace_map.v1.json")
        validate_semantic_sha(payload)
    except (PackageResourceError, CanonicalContractError) as exc:
        raise NamespaceContractError("v3 namespace map is invalid") from exc
    if (
        payload.get("version") != "myquant.v17.v3.namespace-map.v1"
        or payload.get("protocol_version") != "myquant.v17.v3"
        or payload.get("roots") != list(ROOTS)
    ):
        raise NamespaceContractError("v3 namespace map identity or roots mismatch")
    rows = payload.get("namespaces")
    if type(rows) is not list or not rows:
        raise NamespaceContractError("v3 namespace map has no namespaces")
    result: dict[str, tuple[str, str]] = {}
    previous: str | None = None
    for index, row in enumerate(rows):
        if (
            type(row) is not dict
            or set(row) != {"kind", "namespace_id", "path_template"}
            or row["kind"] not in {"directory", "file"}
            or type(row["namespace_id"]) is not str
            or type(row["path_template"]) is not str
        ):
            raise NamespaceContractError(f"namespace row {index} shape mismatch")
        namespace_id = row["namespace_id"]
        if previous is not None and namespace_id <= previous:
            raise NamespaceContractError("namespace rows are not in ASCII order")
        previous = namespace_id
        if namespace_id in result:
            raise NamespaceContractError("duplicate namespace id")
        result[namespace_id] = (row["kind"], row["path_template"])
    return result


def derive_namespace_path(namespace_id: Any, /, **tokens: Any) -> PurePosixPath:
    templates = namespace_templates()
    try:
        token = require_registry_token(
            namespace_id,
            registry=templates,
            label="namespace_id",
        )
    except IdentityContractError as exc:
        raise NamespaceContractError(str(exc)) from exc
    template = templates[token][1]
    placeholders = tuple(
        field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None
    )
    if set(tokens) != set(placeholders) or len(placeholders) != len(set(placeholders)):
        raise NamespaceContractError("namespace template arguments are incomplete or extra")
    normalized: dict[str, str] = {}
    try:
        for field_name in placeholders:
            if field_name == "byte_sha256":
                normalized[field_name] = require_sha256(
                    tokens[field_name],
                    label=field_name,
                )
            else:
                normalized[field_name] = require_path_id(
                    tokens[field_name],
                    label=field_name,
                )
    except IdentityContractError as exc:
        raise NamespaceContractError(str(exc)) from exc
    path = canonical_relative_path(template.format(**normalized), label=token)
    root_for_path(str(path))
    return path


def shadow_run_path(*, strategy_id: Any, run_id: Any) -> PurePosixPath:
    return derive_namespace_path(
        "SHADOW_RUN",
        strategy_id=strategy_id,
        run_id=run_id,
    )


def formal_run_path(*, strategy_id: Any, run_id: Any) -> PurePosixPath:
    return derive_namespace_path(
        "FORMAL_RUN",
        strategy_id=strategy_id,
        run_id=run_id,
    )


def validate_namespace_inventory(inventory: Mapping[Any, Any]) -> dict[str, NodeKind]:
    """Validate complete caller-supplied lstat facts without touching the filesystem."""

    if not isinstance(inventory, Mapping):
        raise NamespaceContractError("namespace inventory must be a mapping")
    result: dict[str, NodeKind] = {}
    casefold_paths: dict[str, str] = {}
    root_paths = tuple(PurePosixPath(root) for root in ROOTS)
    for raw_path, raw_kind in inventory.items():
        path = str(canonical_relative_path(raw_path, label="inventory path"))
        try:
            kind = raw_kind if type(raw_kind) is NodeKind else NodeKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise NamespaceContractError(f"invalid namespace inventory row: {path}") from exc
        posix_path = PurePosixPath(path)
        if not any(
            posix_path == root or posix_path in root.parents or root in posix_path.parents
            for root in root_paths
        ):
            raise NamespaceContractError(f"inventory path is unrelated to v3 roots: {path}")
        collision_key = path.casefold()
        previous = casefold_paths.get(collision_key)
        if previous is not None and previous != path:
            raise NamespaceContractError(
                f"namespace ASCII-casefold collision: {previous!r} and {path!r}"
            )
        if path in result:
            raise NamespaceContractError(f"duplicate namespace path: {path}")
        if kind in {NodeKind.SYMLINK, NodeKind.BROKEN_SYMLINK}:
            raise NamespaceContractError(f"symlink namespace collision: {path}")
        casefold_paths[collision_key] = path
        result[path] = kind
    return result


def classify_namespace_collision(
    target: Any,
    *,
    expected_kind: str,
    inventory: Mapping[Any, Any],
) -> CollisionReport:
    try:
        path = canonical_relative_path(target, label="collision target")
        root_for_path(str(path))
    except NamespaceContractError as exc:
        outcome = (
            CollisionKind.V2_NAMESPACE_COLLISION
            if "v2 namespace" in str(exc)
            else CollisionKind.OUTSIDE_V3_ROOTS
        )
        return CollisionReport(outcome, str(target))
    if expected_kind not in {"directory", "file"}:
        raise NamespaceContractError("expected_kind must be directory or file")
    try:
        facts = validate_namespace_inventory(inventory)
    except NamespaceContractError as exc:
        if "casefold collision" in str(exc):
            return CollisionReport(CollisionKind.CASEFOLD_COLLISION, str(path))
        if "symlink namespace collision" not in str(exc):
            raise
        conflict = str(exc).rsplit(": ", 1)[-1]
        raw_kind = inventory.get(conflict)
        try:
            kind = raw_kind if type(raw_kind) is NodeKind else NodeKind(raw_kind)
        except (TypeError, ValueError):
            kind = NodeKind.SYMLINK
        if conflict == str(path):
            outcome = (
                CollisionKind.LEAF_BROKEN_SYMLINK
                if kind is NodeKind.BROKEN_SYMLINK
                else CollisionKind.LEAF_SYMLINK
            )
        else:
            outcome = (
                CollisionKind.ANCESTOR_BROKEN_SYMLINK
                if kind is NodeKind.BROKEN_SYMLINK
                else CollisionKind.ANCESTOR_SYMLINK
            )
        return CollisionReport(outcome, str(path), conflict)

    required = [str(PurePosixPath(*path.parts[:index])) for index in range(1, len(path.parts) + 1)]
    for required_path in required:
        if required_path not in facts:
            return CollisionReport(
                CollisionKind.INVENTORY_INCOMPLETE,
                str(path),
                required_path,
            )
    missing_ancestor: str | None = None
    for ancestor in required[:-1]:
        kind = facts[ancestor]
        if missing_ancestor is not None:
            if kind is not NodeKind.MISSING:
                return CollisionReport(
                    CollisionKind.INVENTORY_INCONSISTENT,
                    str(path),
                    ancestor,
                )
            continue
        if kind is NodeKind.MISSING:
            missing_ancestor = ancestor
        elif kind is not NodeKind.DIRECTORY:
            return CollisionReport(
                CollisionKind.ANCESTOR_NOT_DIRECTORY,
                str(path),
                ancestor,
            )
    leaf = facts[str(path)]
    if missing_ancestor is not None and leaf is not NodeKind.MISSING:
        return CollisionReport(
            CollisionKind.INVENTORY_INCONSISTENT,
            str(path),
            str(path),
        )
    if leaf is NodeKind.MISSING:
        return CollisionReport(CollisionKind.CLEAR, str(path))
    expected = NodeKind.FILE if expected_kind == "file" else NodeKind.DIRECTORY
    if leaf is expected:
        return CollisionReport(CollisionKind.EXPECTED_EXISTING, str(path))
    return CollisionReport(CollisionKind.LEAF_KIND_COLLISION, str(path), str(path))


__all__ = [
    "CollisionKind",
    "CollisionReport",
    "FORMAL_RESEARCH_RESULTS_ROOT",
    "FORBIDDEN_V2_COMPONENTS",
    "NodeKind",
    "ROOTS",
    "RUNS_ROOT",
    "SHADOW_RESULTS_ROOT",
    "SOURCES_ROOT",
    "NamespaceContractError",
    "canonical_relative_path",
    "classify_namespace_collision",
    "derive_namespace_path",
    "formal_run_path",
    "namespace_templates",
    "root_for_path",
    "shadow_run_path",
    "source_run_path",
    "validate_namespace_inventory",
]
