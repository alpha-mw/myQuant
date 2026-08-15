"""Deterministic dependency, custody, and runtime-closure inventory resolver."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import ast
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Final

from ..contracts import get_contract, seal_artifact
from .canonical import (
    SHA256_RE,
    assert_no_symlink_components,
    canonical_json_bytes,
    canonical_relative_path,
    parse_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
    workspace_path,
    write_idempotent_bytes,
)
from .errors import (
    BROKEN_LOCAL_IMPORT,
    BROKEN_REFERENCE,
    CLASSIFICATION_COLLISION,
    DYNAMIC_IMPORT_ALLOWLIST_UNUSED,
    LEGACY_TEST_STILL_PRESENT,
    MODULE_COLLISION,
    PATH_COLLISION,
    POINTER_FILENAME_UNAPPROVED,
    REPLACEMENT_TEST_MAP_INVALID,
    REPLACEMENT_TEST_MISSING,
    REFERENCE_COLLISION,
    REFERENCE_HASH_MISMATCH,
    RUNTIME_ROOT_COLLISION,
    SYMLINK_REFUSED,
    TRACKED_ROOT_COLLISION,
    TRACKED_ROOT_MISSING,
    UNCLASSIFIED_PATH,
    UNPARSEABLE_JSON,
    UNSAFE_PATH,
    UnifiedCutoverError,
)
from .parsers import (
    ConfigEdge,
    module_name_for_path,
    parse_python_imports,
    parse_shell_config_edges,
    parse_toml_bytes,
    parse_toml_config_edges,
    parse_yaml_config_edges,
)
from .rules import (
    ACTIVE_AUTHORITY,
    ACTIVE_CALLER,
    CUSTODY_CLASSIFICATIONS,
    CUSTODY_ONLY,
    LEGACY_INACTIVE,
    NON_AUTHORITY_SHADOW,
    CutoverRules,
    GraphSeed,
    LoadedRules,
    TrackedRoot,
    RuntimeRoot,
    load_bootstrap_decision,
    load_dynamic_allowlist,
    load_legacy_custody_scope,
    load_legacy_seeds,
    load_replacement_test_map,
    load_rules,
    path_matches_glob,
    pointer_filename_matches,
)

INVENTORY_KIND: Final = "system.migration.inventory"
INVENTORY_PAYLOAD_FIELDS: Final = (
    "inventory_id",
    "status",
    "rules_ref",
    "dynamic_import_allowlist_ref",
    "legacy_seed_manifest_ref",
    "legacy_custody_scope_ref",
    "replacement_test_map_ref",
    "bootstrap_decision_ref",
    "tracked_roots",
    "runtime_roots",
    "files",
    "edges",
    "summary",
)
INVENTORY_CONTRACT_SHA256: Final = get_contract(INVENTORY_KIND).contract_sha256


@dataclass(frozen=True)
class FileObservation:
    relative_path: str
    origin: str
    byte_sha256: str
    bytes: int
    classification: str | None = None
    classification_reason: str | None = None

    def with_classification(self, classification: str, reason: str) -> "FileObservation":
        return replace(
            self,
            classification=classification,
            classification_reason=reason,
        )


@dataclass(frozen=True, order=True)
class DependencyEdge:
    source: str
    target: str
    kind: str
    line: int | None


@dataclass(frozen=True)
class InventoryResolution:
    document: Mapping[str, Any]
    raw: bytes


def _root_contains(root: str, path: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _path_kind(relative_path: str) -> str:
    suffix = PurePosixPath(relative_path).suffix.lower()
    return {
        ".json": "JSON",
        ".jsonl": "JSON_LINES",
        ".py": "PYTHON",
        ".sh": "SHELL",
        ".toml": "TOML",
        ".yaml": "YAML",
        ".yml": "YAML",
    }.get(suffix, "BYTES")


def _assert_no_root_overlap(paths: Sequence[str], *, code: str, label: str) -> None:
    ordered = sorted(paths)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if _root_contains(first, second) or _root_contains(second, first):
                raise UnifiedCutoverError(code, f"overlapping {label}: {first} and {second}")


def _git_tracked_paths(workspace_root: Path) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace_root), "ls-files", "--stage", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnifiedCutoverError(
            TRACKED_ROOT_MISSING,
            "exact tracked-file inventory requires a readable Git index",
        ) from exc
    result: list[str] = []
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise UnifiedCutoverError(
                TRACKED_ROOT_COLLISION, "Git index record is invalid"
            ) from exc
        if stage != "0":
            raise UnifiedCutoverError(TRACKED_ROOT_COLLISION, f"unmerged Git index path: {path}")
        if mode in {"120000", "160000"}:
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"tracked symlink/submodule refused: {path}")
        if mode not in {"100644", "100755"}:
            raise UnifiedCutoverError(
                TRACKED_ROOT_COLLISION, f"unsupported Git mode {mode}: {path}"
            )
        result.append(canonical_relative_path(path, label="tracked path"))
    if result != sorted(set(result)):
        raise UnifiedCutoverError(TRACKED_ROOT_COLLISION, "Git tracked paths are not unique")
    return tuple(result)


def _git_baseline_paths(
    workspace_root: Path, *, commit: str, expected_tree: str
) -> tuple[str, ...]:
    try:
        observed_tree = subprocess.run(
            ["git", "-C", str(workspace_root), "rev-parse", f"{commit}^{{tree}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        completed = subprocess.run(
            ["git", "-C", str(workspace_root), "ls-tree", "-r", "--name-only", "-z", commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MAP_INVALID,
            "approved baseline Git objects are unavailable",
        ) from exc
    if observed_tree != expected_tree:
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MAP_INVALID, "approved baseline tree identity mismatch"
        )
    try:
        paths = tuple(
            canonical_relative_path(item.decode("utf-8"), label="baseline path")
            for item in completed.stdout.split(b"\0")
            if item
        )
    except UnicodeDecodeError as exc:
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MAP_INVALID, "baseline contains a non-UTF-8 path"
        ) from exc
    if paths != tuple(sorted(set(paths))):
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MAP_INVALID, "baseline paths are not sorted and unique"
        )
    return paths


def _test_nodes(raw: bytes, *, relative_path: str) -> set[tuple[str, ...]]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=relative_path)
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MISSING, f"replacement test cannot be parsed: {relative_path}"
        ) from exc
    result: set[tuple[str, ...]] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.add((node.name,))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.add((node.name, child.name))
    return result


def _validate_replacement_test_closure(
    root: Path,
    *,
    replacement_map: Any,
    current_tracked_paths: Sequence[str],
) -> None:
    baseline_paths = _git_baseline_paths(
        root,
        commit=replacement_map.baseline_commit,
        expected_tree=replacement_map.baseline_tree,
    )
    expected_legacy = tuple(
        path
        for path in baseline_paths
        if any(
            path_matches_glob(path, pattern)
            for pattern in replacement_map.legacy_test_seed_patterns
        )
    )
    mapped = tuple(entry.baseline_test_path for entry in replacement_map.entries)
    if mapped != expected_legacy:
        missing = sorted(set(expected_legacy) - set(mapped))
        extra = sorted(set(mapped) - set(expected_legacy))
        raise UnifiedCutoverError(
            REPLACEMENT_TEST_MAP_INVALID,
            f"replacement map is incomplete (missing={missing[:1]}, extra={extra[:1]})",
        )
    current = frozenset(current_tracked_paths)
    retained = sorted(current.intersection(mapped))
    if retained:
        raise UnifiedCutoverError(
            LEGACY_TEST_STILL_PRESENT,
            f"mapped legacy test is still tracked: {retained[0]}",
        )
    parsed_nodes: dict[str, set[tuple[str, ...]]] = {}
    for entry in replacement_map.entries:
        for selector in entry.replacement_test_selectors:
            test_path, *nodes = selector.split("::")
            if test_path not in current:
                raise UnifiedCutoverError(
                    REPLACEMENT_TEST_MISSING,
                    f"replacement test is not tracked: {test_path}",
                )
            if test_path not in parsed_nodes:
                _observation, raw = _observe_file(root, test_path, origin="REPLACEMENT_TEST")
                parsed_nodes[test_path] = _test_nodes(raw, relative_path=test_path)
            if tuple(nodes) not in parsed_nodes[test_path]:
                raise UnifiedCutoverError(
                    REPLACEMENT_TEST_MISSING,
                    f"replacement test node is absent: {selector}",
                )


def _filter_tracked_paths(
    all_paths: Sequence[str],
    *,
    rules: CutoverRules,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    roots = [row.path for row in rules.tracked_roots]
    _assert_no_root_overlap(roots, code=TRACKED_ROOT_COLLISION, label="tracked roots")
    selected: list[str] = []
    statuses: list[dict[str, Any]] = []
    for root in rules.tracked_roots:
        matches = [path for path in all_paths if _root_contains(root.path, path)]
        if not matches and root.required:
            raise UnifiedCutoverError(
                TRACKED_ROOT_MISSING, f"tracked root has no files: {root.path}"
            )
        statuses.append(
            {
                "path": root.path,
                "required": root.required,
                "classification": root.classification,
                "status": "PRESENT" if matches else "ABSENT_OPTIONAL",
                "file_count": len(matches),
            }
        )
        selected.extend(matches)
    if selected != sorted(set(selected)):
        raise UnifiedCutoverError(TRACKED_ROOT_COLLISION, "tracked roots selected a path twice")
    return tuple(selected), statuses


def _tracked_direct_classifications(
    paths: Sequence[str], *, rules: CutoverRules
) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for path in paths:
        matches = [root for root in rules.tracked_roots if _root_contains(root.path, path)]
        if len(matches) != 1:
            raise UnifiedCutoverError(
                TRACKED_ROOT_COLLISION, f"tracked path has {len(matches)} roots: {path}"
            )
        classification = matches[0].classification
        if classification is not None:
            result[path] = (classification, "EXACT_TRACKED_ROOT_CLASSIFICATION")
    return result


def _external_inventory(
    codex_home: Path,
    *,
    rules: CutoverRules,
) -> tuple[
    dict[str, FileObservation],
    dict[str, bytes],
    dict[str, tuple[str, str]],
    list[dict[str, Any]],
]:
    observations: dict[str, FileObservation] = {}
    raw_by_path: dict[str, bytes] = {}
    direct: dict[str, tuple[str, str]] = {}
    statuses: list[dict[str, Any]] = []
    physical_roots = [row.path for row in rules.external_roots]
    inventory_prefixes = [row.inventory_prefix for row in rules.external_roots]
    _assert_no_root_overlap(
        physical_roots, code=TRACKED_ROOT_COLLISION, label="external physical roots"
    )
    _assert_no_root_overlap(
        inventory_prefixes, code=TRACKED_ROOT_COLLISION, label="external inventory prefixes"
    )
    for external in rules.external_roots:
        absolute = codex_home.joinpath(*PurePosixPath(external.path).parts)
        try:
            metadata = os.lstat(absolute)
        except FileNotFoundError:
            if external.required:
                raise UnifiedCutoverError(
                    TRACKED_ROOT_MISSING,
                    f"required external root is missing: CODEX_HOME/{external.path}",
                )
            statuses.append(
                {
                    "path": f"CODEX_HOME/{external.path}",
                    "inventory_prefix": external.inventory_prefix,
                    "required": False,
                    "classification": external.classification,
                    "origin": "EXTERNAL",
                    "status": "ABSENT_OPTIONAL",
                    "file_count": 0,
                }
            )
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(
                SYMLINK_REFUSED, f"external root is a symlink: CODEX_HOME/{external.path}"
            )
        if stat.S_ISREG(metadata.st_mode):
            physical_and_logical = ((absolute, external.inventory_prefix),)
        elif stat.S_ISDIR(metadata.st_mode):
            physical_and_logical = tuple(
                (
                    codex_home.joinpath(*PurePosixPath(path).parts),
                    canonical_relative_path(
                        f"{external.inventory_prefix}/"
                        f"{PurePosixPath(path).relative_to(external.path).as_posix()}"
                    ),
                )
                for path in _walk_real_files(codex_home, external.path)
            )
        else:
            raise UnifiedCutoverError(
                UNSAFE_PATH, f"external root is not a file/directory: {external.path}"
            )
        for physical, logical in physical_and_logical:
            if logical in observations:
                raise UnifiedCutoverError(
                    PATH_COLLISION, f"external logical path collision: {logical}"
                )
            raw = read_stable_regular_file(physical, label=f"external source {logical}")
            observations[logical] = FileObservation(
                logical, "EXTERNAL", sha256_bytes(raw), len(raw)
            )
            raw_by_path[logical] = raw
            direct[logical] = (
                external.classification,
                "EXACT_EXTERNAL_ROOT_CLASSIFICATION",
            )
        statuses.append(
            {
                "path": f"CODEX_HOME/{external.path}",
                "inventory_prefix": external.inventory_prefix,
                "required": external.required,
                "classification": external.classification,
                "origin": "EXTERNAL",
                "status": "PRESENT",
                "file_count": len(physical_and_logical),
            }
        )
    return observations, raw_by_path, direct, statuses


def _observe_file(
    root: Path,
    relative_path: str,
    *,
    origin: str,
) -> tuple[FileObservation, bytes]:
    relative = canonical_relative_path(relative_path)
    assert_no_symlink_components(root, relative, include_leaf=True)
    path = workspace_path(root, relative)
    raw = read_stable_regular_file(path, label=f"{origin.lower()} file {relative}")
    return FileObservation(relative, origin, sha256_bytes(raw), len(raw)), raw


def _walk_real_files(root: Path, relative_root: str) -> tuple[str, ...]:
    relative = canonical_relative_path(relative_root, label="runtime root")
    absolute = root.joinpath(*PurePosixPath(relative).parts)
    try:
        root_stat = os.lstat(absolute)
    except FileNotFoundError:
        return ()
    if stat.S_ISLNK(root_stat.st_mode):
        raise UnifiedCutoverError(SYMLINK_REFUSED, f"runtime root is a symlink: {relative}")
    if stat.S_ISREG(root_stat.st_mode):
        return (relative,)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise UnifiedCutoverError(UNSAFE_PATH, f"runtime root is not a file/directory: {relative}")
    result: list[str] = []

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise UnifiedCutoverError(UNSAFE_PATH, f"cannot scan runtime root {relative}") from exc
        for entry in entries:
            child_relative = (prefix / entry.name).as_posix()
            try:
                if entry.is_symlink():
                    raise UnifiedCutoverError(
                        SYMLINK_REFUSED, f"runtime symlink refused: {child_relative}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), prefix / entry.name)
                elif entry.is_file(follow_symlinks=False):
                    result.append(child_relative)
                else:
                    raise UnifiedCutoverError(
                        UNSAFE_PATH, f"non-regular runtime entry refused: {child_relative}"
                    )
            finally:
                entry.close() if hasattr(entry, "close") else None

    visit(absolute, PurePosixPath(relative))
    return tuple(result)


def _is_exception(path: str, rules: CutoverRules) -> bool:
    return any(path_matches_glob(path, pattern) for pattern in rules.custody_exceptions)


def _runtime_root_for(path: str, runtime_roots: Sequence[RuntimeRoot]) -> RuntimeRoot | None:
    matches = [row for row in runtime_roots if _root_contains(row.path, path)]
    if len(matches) > 1:
        raise UnifiedCutoverError(RUNTIME_ROOT_COLLISION, f"runtime roots collide at {path}")
    return matches[0] if matches else None


def _reference_expected_sha(container: Mapping[str, Any], path_key: str) -> str | None:
    stem = path_key[: -len("_path")] if path_key.endswith("_path") else path_key
    candidates = (
        f"{stem}_byte_sha256",
        f"{stem}_sha256",
        "byte_sha256",
        "sha256",
    )
    values: list[str] = []
    for key in candidates:
        value = container.get(key)
        if type(value) is str and SHA256_RE.fullmatch(value):
            values.append(value)
    unique = sorted(set(values))
    if len(unique) > 1:
        raise UnifiedCutoverError(
            REFERENCE_COLLISION, f"reference {path_key} has conflicting exact SHA-256 fields"
        )
    return unique[0] if unique else None


def _json_path_refs(
    value: Any,
    *,
    reference_keys: frozenset[str],
) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []
    if type(value) is dict:
        for key in sorted(value):
            item = value[key]
            if key in reference_keys and type(item) is str:
                result.append((item, _reference_expected_sha(value, key)))
            result.extend(_json_path_refs(item, reference_keys=reference_keys))
    elif type(value) is list:
        for item in value:
            result.extend(_json_path_refs(item, reference_keys=reference_keys))
    return result


def _resolve_runtime_reference(
    root: Path,
    *,
    source_path: str,
    referenced: str,
    runtime_roots: Sequence[RuntimeRoot],
) -> str:
    canonical = canonical_relative_path(referenced, label=f"reference from {source_path}")
    workspace_candidate = canonical
    parent_candidate = (PurePosixPath(source_path).parent / canonical).as_posix()
    candidates: list[str] = []
    for candidate in sorted({workspace_candidate, parent_candidate}):
        if _runtime_root_for(candidate, runtime_roots) is None:
            continue
        absolute = root.joinpath(*PurePosixPath(candidate).parts)
        try:
            metadata = os.lstat(absolute)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"runtime reference is symlink: {candidate}")
        if stat.S_ISREG(metadata.st_mode):
            candidates.append(candidate)
    if not candidates:
        raise UnifiedCutoverError(
            BROKEN_REFERENCE, f"broken runtime reference {referenced!r} from {source_path}"
        )
    if len(candidates) != 1:
        raise UnifiedCutoverError(
            REFERENCE_COLLISION,
            f"runtime reference {referenced!r} from {source_path} resolves ambiguously",
        )
    return candidates[0]


def _pointer_seed_paths(
    root: Path,
    runtime_root: RuntimeRoot,
    *,
    rules: CutoverRules,
) -> tuple[str, ...]:
    explicit = [
        seed.value
        for seed in (*rules.entrypoint_seeds, *rules.shadow_seeds)
        if seed.kind == "pointer" and _root_contains(runtime_root.path, seed.value)
    ]
    if explicit:
        approved_patterns = (
            *rules.pointer_filename_rules["active"],
            *rules.pointer_filename_rules["reachable"],
        )
        for path in explicit:
            if not pointer_filename_matches(PurePosixPath(path).name, approved_patterns):
                raise UnifiedCutoverError(
                    POINTER_FILENAME_UNAPPROVED, f"unapproved pointer filename: {path}"
                )
        return tuple(sorted(set(explicit)))
    all_files = _walk_real_files(root, runtime_root.path)
    pointers = tuple(
        path
        for path in all_files
        if pointer_filename_matches(
            PurePosixPath(path).name, rules.pointer_filename_rules["active"]
        )
    )
    if runtime_root.required and not pointers:
        raise UnifiedCutoverError(
            BROKEN_REFERENCE, f"required pointer root has no approved pointer: {runtime_root.path}"
        )
    return pointers


def _runtime_inventory(
    root: Path,
    *,
    rules: CutoverRules,
) -> tuple[
    dict[str, FileObservation], dict[str, bytes], list[DependencyEdge], list[dict[str, Any]]
]:
    paths = [row.path for row in rules.runtime_roots]
    _assert_no_root_overlap(paths, code=RUNTIME_ROOT_COLLISION, label="runtime roots")
    observations: dict[str, FileObservation] = {}
    raw_by_path: dict[str, bytes] = {}
    edges: list[DependencyEdge] = []
    statuses: list[dict[str, Any]] = []

    def add(path: str, runtime_root: RuntimeRoot, *, reason: str) -> None:
        if path in observations:
            existing = observations[path]
            if existing.classification != runtime_root.classification:
                raise UnifiedCutoverError(
                    CLASSIFICATION_COLLISION, f"runtime path has two classifications: {path}"
                )
            return
        classification = runtime_root.classification
        if _is_exception(path, rules):
            if classification != CUSTODY_ONLY:
                raise UnifiedCutoverError(
                    CLASSIFICATION_COLLISION,
                    f"custody exception {path} is assigned {classification}",
                )
            reason = "CUSTODY_EXCEPTION_NO_TRAVERSAL"
        observation, raw = _observe_file(root, path, origin="RUNTIME")
        observations[path] = observation.with_classification(classification, reason)
        raw_by_path[path] = raw

    for runtime_root in rules.runtime_roots:
        absolute = root.joinpath(*PurePosixPath(runtime_root.path).parts)
        try:
            metadata = os.lstat(absolute)
        except FileNotFoundError:
            if runtime_root.required:
                raise UnifiedCutoverError(
                    BROKEN_REFERENCE, f"required runtime root is missing: {runtime_root.path}"
                )
            statuses.append(
                {
                    "path": runtime_root.path,
                    "required": False,
                    "classification": runtime_root.classification,
                    "traversal": runtime_root.traversal,
                    "status": "ABSENT_OPTIONAL",
                    "file_count": 0,
                }
            )
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(
                SYMLINK_REFUSED, f"runtime root is symlink: {runtime_root.path}"
            )
        before_count = len(observations)
        if runtime_root.traversal in {"INVENTORY_ONLY", "BOUNDARY"}:
            for path in _walk_real_files(root, runtime_root.path):
                add(
                    path,
                    runtime_root,
                    reason=(
                        "INDEPENDENT_BOUNDARY_NO_TRAVERSAL"
                        if runtime_root.traversal == "BOUNDARY"
                        else "RUNTIME_INVENTORY_NO_TRAVERSAL"
                    ),
                )
        else:
            queue = deque(_pointer_seed_paths(root, runtime_root, rules=rules))
            queued = set(queue)
            while queue:
                path = queue.popleft()
                add(path, runtime_root, reason="REACHABLE_POINTER_CLOSURE")
                if _is_exception(path, rules):
                    continue
                raw = raw_by_path[path]
                if PurePosixPath(path).suffix.lower() != ".json":
                    continue
                try:
                    document = parse_json_bytes(raw, label=f"runtime JSON {path}")
                except UnifiedCutoverError as exc:
                    raise UnifiedCutoverError(
                        UNPARSEABLE_JSON, f"reachable runtime JSON cannot be parsed: {path}"
                    ) from exc
                for referenced, expected_sha in _json_path_refs(
                    document,
                    reference_keys=frozenset(rules.json_reference_keys),
                ):
                    target = _resolve_runtime_reference(
                        root,
                        source_path=path,
                        referenced=referenced,
                        runtime_roots=rules.runtime_roots,
                    )
                    target_root = _runtime_root_for(target, rules.runtime_roots)
                    assert target_root is not None
                    if target_root.classification != runtime_root.classification:
                        raise UnifiedCutoverError(
                            CLASSIFICATION_COLLISION,
                            f"runtime reference crosses custody classifications: {path} -> {target}",
                        )
                    target_observation, target_raw = _observe_file(root, target, origin="RUNTIME")
                    if expected_sha is not None and target_observation.byte_sha256 != expected_sha:
                        raise UnifiedCutoverError(
                            REFERENCE_HASH_MISMATCH,
                            f"runtime reference SHA-256 mismatch: {path} -> {target}",
                        )
                    edges.append(DependencyEdge(path, target, "JSON_EXACT_REF", None))
                    if target not in queued:
                        queued.add(target)
                        queue.append(target)
                    raw_by_path.setdefault(target, target_raw)
        statuses.append(
            {
                "path": runtime_root.path,
                "required": runtime_root.required,
                "classification": runtime_root.classification,
                "traversal": runtime_root.traversal,
                "status": "PRESENT",
                "file_count": len(observations) - before_count,
            }
        )
    return observations, raw_by_path, edges, statuses


def _console_scripts(raw_by_path: Mapping[str, bytes]) -> dict[str, str]:
    raw = raw_by_path.get("pyproject.toml")
    if raw is None:
        return {}
    document = parse_toml_bytes(raw, label="pyproject.toml")
    project = document.get("project")
    scripts = project.get("scripts", {}) if type(project) is dict else {}
    if type(scripts) is not dict:
        raise UnifiedCutoverError(UNPARSEABLE_JSON, "pyproject project.scripts is invalid")
    result: dict[str, str] = {}
    for key, value in scripts.items():
        if type(key) is not str or type(value) is not str:
            raise UnifiedCutoverError(UNPARSEABLE_JSON, "pyproject project.scripts is invalid")
        result[key] = value
    return result


def _resolve_module_target(
    module: str,
    *,
    module_index: Mapping[str, str],
    source_path: str,
    edge_kind: str,
) -> str | None:
    if module == "<BROKEN_RELATIVE_IMPORT>":
        raise UnifiedCutoverError(BROKEN_LOCAL_IMPORT, f"broken relative import in {source_path}")
    exact = module_index.get(module)
    if exact is not None:
        return exact
    parts = module.split(".")
    local_tops = {name.split(".", 1)[0] for name in module_index}
    if parts and parts[0] in local_tops:
        for length in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:length])
            if parent in module_index and edge_kind == "AST_FROM_MEMBER":
                return None
        raise UnifiedCutoverError(
            BROKEN_LOCAL_IMPORT,
            f"local module {module!r} imported by {source_path} is not tracked",
        )
    return None


def _source_graph(
    root: Path,
    *,
    observations: Mapping[str, FileObservation],
    raw_by_path: Mapping[str, bytes],
    rules: CutoverRules,
    allowlist_entries: Mapping[tuple[str, int, str], Any],
) -> tuple[dict[str, set[str]], list[DependencyEdge], set[tuple[str, int, str]]]:
    module_index: dict[str, str] = {}
    module_meta: dict[str, tuple[str, bool]] = {}
    for path in sorted(observations):
        named = module_name_for_path(path)
        if named is None:
            continue
        module, is_package = named
        if module in module_index:
            raise UnifiedCutoverError(
                MODULE_COLLISION, f"module {module} maps to {module_index[module]} and {path}"
            )
        module_index[module] = path
        module_meta[path] = (module, is_package)

    console_scripts = _console_scripts(raw_by_path)
    adjacency: dict[str, set[str]] = defaultdict(set)
    edges: list[DependencyEdge] = []
    used_allowlist: set[tuple[str, int, str]] = set()
    for path in sorted(observations):
        if _is_exception(path, rules):
            continue
        raw = raw_by_path[path]
        suffix = PurePosixPath(path).suffix.lower()
        if suffix == ".py":
            module, is_package = module_meta[path]
            parsed = parse_python_imports(
                raw,
                relative_path=path,
                module_name=module,
                is_package=is_package,
                allowlist=allowlist_entries,
            )
            used_allowlist.update(parsed.used_allowlist_keys)
            for imported in parsed.imports:
                target = _resolve_module_target(
                    imported.module,
                    module_index=module_index,
                    source_path=path,
                    edge_kind=imported.kind,
                )
                if target is not None:
                    adjacency[path].add(target)
                    edges.append(DependencyEdge(path, target, imported.kind, imported.line))
        else:
            config_edges: Sequence[ConfigEdge] = ()
            if suffix == ".toml":
                config_edges = parse_toml_config_edges(raw, relative_path=path)
            elif suffix in {".yaml", ".yml"}:
                config_edges = parse_yaml_config_edges(
                    raw,
                    relative_path=path,
                    console_scripts=console_scripts,
                )
            elif suffix == ".sh":
                config_edges = parse_shell_config_edges(
                    raw,
                    relative_path=path,
                    console_scripts=console_scripts,
                )
            elif suffix == ".json":
                parse_json_bytes(raw, label=f"tracked JSON {path}")
            for edge in config_edges:
                if edge.target_kind == "module":
                    target = _resolve_module_target(
                        edge.target,
                        module_index=module_index,
                        source_path=path,
                        edge_kind=edge.source_kind,
                    )
                    if target is None:
                        continue
                else:
                    target = canonical_relative_path(edge.target, label=f"shell path from {path}")
                    if target not in observations:
                        raise UnifiedCutoverError(
                            BROKEN_REFERENCE, f"shell entrypoint {path} targets untracked {target}"
                        )
                adjacency[path].add(target)
                edges.append(DependencyEdge(path, target, edge.source_kind, edge.line))
    return adjacency, edges, used_allowlist


def _seed_path(
    seed: GraphSeed,
    *,
    observations: Mapping[str, FileObservation],
    module_index: Mapping[str, str],
) -> str:
    if seed.kind == "module":
        target = module_index.get(seed.value)
        if target is None:
            raise UnifiedCutoverError(
                BROKEN_REFERENCE, f"graph seed module is missing: {seed.value}"
            )
        return target
    if seed.kind == "pointer":
        if seed.value not in observations:
            raise UnifiedCutoverError(BROKEN_REFERENCE, f"pointer seed is missing: {seed.value}")
        return seed.value
    if seed.value not in observations:
        raise UnifiedCutoverError(BROKEN_REFERENCE, f"graph seed path is missing: {seed.value}")
    return seed.value


def _reachable(adjacency: Mapping[str, set[str]], seeds: Iterable[str]) -> set[str]:
    result: set[str] = set()
    queue = deque(sorted(set(seeds)))
    while queue:
        path = queue.popleft()
        if path in result:
            continue
        result.add(path)
        queue.extend(sorted(adjacency.get(path, ())))
    return result


def _classify_tracked(
    observations: Mapping[str, FileObservation],
    *,
    rules: CutoverRules,
    adjacency: Mapping[str, set[str]],
    legacy_manifest_entries: Sequence[Any],
    direct_classifications: Mapping[str, tuple[str, str]],
) -> dict[str, FileObservation]:
    module_index = {
        named[0]: path for path in observations if (named := module_name_for_path(path)) is not None
    }
    legacy_seeds: list[GraphSeed] = []
    for entry in legacy_manifest_entries:
        observed = observations.get(entry.relative_path)
        if observed is None or observed.byte_sha256 != entry.byte_sha256:
            continue
        legacy_seeds.append(
            GraphSeed("module", entry.module)
            if entry.module is not None
            else GraphSeed("path", entry.relative_path)
        )

    classifications: dict[str, set[str]] = defaultdict(set)
    reasons: dict[tuple[str, str], str] = {}
    seed_groups = (
        (ACTIVE_CALLER, rules.entrypoint_seeds),
        (NON_AUTHORITY_SHADOW, rules.shadow_seeds),
        (LEGACY_INACTIVE, tuple(legacy_seeds)),
    )
    for classification, seeds in seed_groups:
        source_seeds = [seed for seed in seeds if seed.kind != "pointer"]
        paths = [
            _seed_path(seed, observations=observations, module_index=module_index)
            for seed in source_seeds
        ]
        for path in _reachable(adjacency, paths):
            classifications[path].add(classification)
            reasons[(path, classification)] = f"{classification}_GRAPH_REACHABLE"

    result: dict[str, FileObservation] = {}
    for path, observation in observations.items():
        if _is_exception(path, rules):
            result[path] = observation.with_classification(
                CUSTODY_ONLY, "CUSTODY_EXCEPTION_NO_TRAVERSAL"
            )
            continue
        direct = direct_classifications.get(path)
        if direct is not None:
            classification, reason = direct
            classifications[path].add(classification)
            reasons[(path, classification)] = reason
        if not classifications[path]:
            matches = [
                classification
                for classification in CUSTODY_CLASSIFICATIONS
                if any(
                    path_matches_glob(path, pattern)
                    for pattern in rules.classification_fallbacks[classification]
                )
            ]
            for classification in matches:
                classifications[path].add(classification)
                reasons[(path, classification)] = "EXACT_FALLBACK_PATTERN"
        if not classifications[path]:
            raise UnifiedCutoverError(UNCLASSIFIED_PATH, f"no custody class for {path}")
        if len(classifications[path]) != 1:
            raise UnifiedCutoverError(
                CLASSIFICATION_COLLISION,
                f"{path} has custody classes {sorted(classifications[path])}",
            )
        classification = next(iter(classifications[path]))
        result[path] = observation.with_classification(
            classification,
            reasons[(path, classification)],
        )
    return result


def _file_row(observation: FileObservation) -> dict[str, Any]:
    assert observation.classification is not None
    assert observation.classification_reason is not None
    return {
        "relative_path": observation.relative_path,
        "origin": observation.origin,
        "file_kind": _path_kind(observation.relative_path),
        "byte_sha256": observation.byte_sha256,
        "bytes": observation.bytes,
        "classification": observation.classification,
        "classification_reason": observation.classification_reason,
    }


def _exact_ref(path: str, raw: bytes) -> dict[str, Any]:
    return {"relative_path": path, "byte_sha256": sha256_bytes(raw), "bytes": len(raw)}


def resolve_unified_cutover_inventory(
    workspace_root: str | os.PathLike[str],
    *,
    created_at: str,
    rules_path: str | os.PathLike[str] | None = None,
    tracked_paths: Sequence[str] | None = None,
    codex_home: str | os.PathLike[str] | None = None,
) -> InventoryResolution:
    """Resolve the exact source graph and runtime custody closure without writes."""

    root = Path(workspace_root).resolve(strict=True)
    loaded_rules: LoadedRules = load_rules(
        root,
        rules_path if rules_path is not None else "operations/unified_cutover/rules.json",
    )
    rules = loaded_rules.rules
    custody_scope = load_legacy_custody_scope(root, rules.legacy_custody_scope)
    allowlist = load_dynamic_allowlist(root, rules.dynamic_import_allowlist)
    legacy = load_legacy_seeds(root, rules.legacy_seed_manifest)
    replacement_map = load_replacement_test_map(root, rules.replacement_test_map)
    bootstrap = load_bootstrap_decision(root, rules.bootstrap_decision)

    all_tracked = (
        _git_tracked_paths(root)
        if tracked_paths is None
        else tuple(sorted(canonical_relative_path(path) for path in tracked_paths))
    )
    if len(all_tracked) != len(set(all_tracked)):
        raise UnifiedCutoverError(TRACKED_ROOT_COLLISION, "supplied tracked paths collide")
    if tracked_paths is None:
        _validate_replacement_test_closure(
            root,
            replacement_map=replacement_map,
            current_tracked_paths=all_tracked,
        )
    tracked_selected, tracked_statuses = _filter_tracked_paths(all_tracked, rules=rules)
    tracked_observations: dict[str, FileObservation] = {}
    tracked_raw: dict[str, bytes] = {}
    casefold_paths: dict[str, str] = {}
    for path in tracked_selected:
        folded = path.casefold()
        if folded in casefold_paths and casefold_paths[folded] != path:
            raise UnifiedCutoverError(PATH_COLLISION, f"case-fold path collision: {path}")
        casefold_paths[folded] = path
        observation, raw = _observe_file(root, path, origin="TRACKED")
        tracked_observations[path] = observation
        tracked_raw[path] = raw

    configured_codex_home = (
        Path(codex_home)
        if codex_home is not None
        else Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    )
    external_observations, external_raw, external_direct, external_statuses = _external_inventory(
        configured_codex_home, rules=rules
    )
    for path in external_observations:
        if path in tracked_observations:
            raise UnifiedCutoverError(PATH_COLLISION, f"tracked/external path collision: {path}")
        folded = path.casefold()
        if folded in casefold_paths and casefold_paths[folded] != path:
            raise UnifiedCutoverError(PATH_COLLISION, f"case-fold path collision: {path}")
        casefold_paths[folded] = path

    source_observations = {**tracked_observations, **external_observations}
    source_raw = {**tracked_raw, **external_raw}
    direct_classifications = {
        **_tracked_direct_classifications(tracked_selected, rules=rules),
        **external_direct,
    }

    adjacency, source_edges, used_allowlist = _source_graph(
        root,
        observations=source_observations,
        raw_by_path=source_raw,
        rules=rules,
        allowlist_entries=allowlist.entries,
    )
    unused = sorted(set(allowlist.entries) - used_allowlist)
    if unused:
        first = unused[0]
        raise UnifiedCutoverError(
            DYNAMIC_IMPORT_ALLOWLIST_UNUSED,
            f"unused dynamic import allowlist key: {first[0]}:{first[1]}:{first[2]}",
        )
    tracked_classified = _classify_tracked(
        source_observations,
        rules=rules,
        adjacency=adjacency,
        legacy_manifest_entries=legacy.entries,
        direct_classifications=direct_classifications,
    )

    runtime_observations, _runtime_raw, runtime_edges, runtime_statuses = _runtime_inventory(
        root, rules=rules
    )
    for path in runtime_observations:
        if path in tracked_classified:
            raise UnifiedCutoverError(PATH_COLLISION, f"tracked/runtime path collision: {path}")
        folded = path.casefold()
        if folded in casefold_paths and casefold_paths[folded] != path:
            raise UnifiedCutoverError(PATH_COLLISION, f"case-fold path collision: {path}")
        casefold_paths[folded] = path

    all_observations = {**tracked_classified, **runtime_observations}
    files = [_file_row(all_observations[path]) for path in sorted(all_observations)]
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "edge_kind": edge.kind,
            "line": edge.line,
        }
        for edge in sorted(
            set(source_edges + runtime_edges),
            key=lambda item: (
                item.source,
                item.target,
                item.kind,
                -1 if item.line is None else item.line,
            ),
        )
    ]
    counts = {classification: 0 for classification in CUSTODY_CLASSIFICATIONS}
    bytes_by_class = {classification: 0 for classification in CUSTODY_CLASSIFICATIONS}
    for row in files:
        counts[row["classification"]] += 1
        bytes_by_class[row["classification"]] += row["bytes"]
    payload_body = {
        "status": "COMPLETE",
        "rules_ref": _exact_ref(loaded_rules.path.relative_to(root).as_posix(), loaded_rules.raw),
        "dynamic_import_allowlist_ref": _exact_ref(rules.dynamic_import_allowlist, allowlist.raw),
        "legacy_seed_manifest_ref": _exact_ref(rules.legacy_seed_manifest, legacy.raw),
        "legacy_custody_scope_ref": _exact_ref(rules.legacy_custody_scope, custody_scope.raw),
        "replacement_test_map_ref": _exact_ref(rules.replacement_test_map, replacement_map.raw),
        "bootstrap_decision_ref": _exact_ref(rules.bootstrap_decision, bootstrap.raw),
        "tracked_roots": [*tracked_statuses, *external_statuses],
        "runtime_roots": runtime_statuses,
        "files": files,
        "edges": edges,
        "summary": {
            "file_count": len(files),
            "edge_count": len(edges),
            "total_bytes": sum(row["bytes"] for row in files),
            "file_count_by_classification": counts,
            "bytes_by_classification": bytes_by_class,
            "baseline_custody_facts": rules.baseline_custody_facts,
            "blocker_codes": [],
        },
    }
    payload = {
        **payload_body,
        "inventory_id": "inventory-" + sha256_bytes(canonical_json_bytes(payload_body)),
    }
    if set(payload) != set(INVENTORY_PAYLOAD_FIELDS):
        raise AssertionError("inventory payload fields drifted from the compiled builder")
    document = seal_artifact(
        INVENTORY_KIND,
        payload,
        created_at=created_at,
        contract_sha256=INVENTORY_CONTRACT_SHA256,
    )
    raw = canonical_json_bytes(document)
    return InventoryResolution(document, raw)


def resolve_inventory(*args: Any, **kwargs: Any) -> InventoryResolution:
    """Compatibility spelling for callers that do not need the full long name."""

    return resolve_unified_cutover_inventory(*args, **kwargs)


def write_inventory(path: str | os.PathLike[str], resolution: InventoryResolution) -> bool:
    return write_idempotent_bytes(Path(path), resolution.raw)


__all__ = [
    "DependencyEdge",
    "FileObservation",
    "INVENTORY_CONTRACT_SHA256",
    "INVENTORY_KIND",
    "InventoryResolution",
    "resolve_inventory",
    "resolve_unified_cutover_inventory",
    "write_inventory",
]
