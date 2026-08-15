"""Repo-controlled rules for deterministic unified-runtime cutover inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .canonical import (
    SHA256_RE,
    canonical_json_bytes,
    canonical_relative_path,
    parse_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)
from .errors import (
    REPLACEMENT_TEST_MAP_INVALID,
    RULES_NON_CANONICAL,
    RULES_SCHEMA_INVALID,
    RULES_UNAVAILABLE,
    UnifiedCutoverError,
)

RULES_KIND: Final = "system.migration.rules"
DYNAMIC_ALLOWLIST_KIND: Final = "system.migration.dynamic_import_allowlist"
LEGACY_SEED_KIND: Final = "system.migration.legacy_seed_manifest"
SOURCE_TO_TARGET_KIND: Final = "system.migration.source_to_target"
REPLACEMENT_TEST_MAP_KIND: Final = "system.migration.replacement_test_map"
LEGACY_CUSTODY_SCOPE_KIND: Final = "system.migration.legacy_custody_scope"

RULES_RELATIVE_PATH: Final = "operations/unified_cutover/rules.json"
DYNAMIC_ALLOWLIST_RELATIVE_PATH: Final = "operations/unified_cutover/dynamic-import-allowlist.json"
LEGACY_SEED_RELATIVE_PATH: Final = "docs/migrations/unified-cutover/legacy-seed-manifest.json"
SOURCE_TO_TARGET_RELATIVE_PATH: Final = "operations/unified_cutover/source-to-target.json"
REPLACEMENT_TEST_MAP_RELATIVE_PATH: Final = (
    "docs/migrations/unified-cutover/replacement-test-map.json"
)
LEGACY_CUSTODY_SCOPE_RELATIVE_PATH: Final = (
    "docs/migrations/unified-cutover/legacy-custody-scope.json"
)
BOOTSTRAP_DECISION_RELATIVE_PATH: Final = "operations/unified_cutover/bootstrap-decision.json"
BOOTSTRAP_DECISION_KIND: Final = "factor.bootstrap_decision"
BOOTSTRAP_DECISION_BYTE_SHA256: Final = (
    "f4add792c25eafa61730dfc839e1e5d6cd9c81de25b3c47b455359d26fb2ce95"
)
LEGACY_BASELINE_COMMIT: Final = "8612ce51d8cb2b13076af60ac059a921dc104129"
LEGACY_BASELINE_TREE: Final = "475efa1db72501a94691b3719d06b85d17c53b57"
LEGACY_CUSTODY_SCOPE_BYTE_SHA256: Final = (
    "6958544bfee31d8d956256757b712821e285f0c0d87eb0f891d8427f232d2dac"
)
BASELINE_CUSTODY_FACTS_SHA256: Final = (
    "2ebfe0a73a4af63c8f047ff6600d1fd854e1c3d94c7ef953fb55a81df27f873d"
)

UNIFIED_ACTIVE_POINTER: Final = "results/system/_active.json"
PERMANENT_MARKER: Final = "results/system/_migration_complete.json"
UNIFIED_OBJECT_ROOT: Final = "results/system/objects"
UNIFIED_GENERATION_ROOT: Final = "results/system/generations"
UNIFIED_POINTER_HISTORY_ROOT: Final = "results/system/pointer_history"
ARCHIVE_ROOT_TEMPLATE: Final = "data/private/system_archives/{cutover_id}/authority_closure"

ACTIVE_AUTHORITY: Final = "ACTIVE_AUTHORITY"
ACTIVE_CALLER: Final = "ACTIVE_CALLER"
NON_AUTHORITY_SHADOW: Final = "NON_AUTHORITY_SHADOW"
INDEPENDENT_SOURCE: Final = "INDEPENDENT_SOURCE"
LEGACY_INACTIVE: Final = "LEGACY_INACTIVE"
CUSTODY_ONLY: Final = "CUSTODY_ONLY"
CUSTODY_CLASSIFICATIONS: Final = (
    ACTIVE_AUTHORITY,
    ACTIVE_CALLER,
    NON_AUTHORITY_SHADOW,
    INDEPENDENT_SOURCE,
    LEGACY_INACTIVE,
    CUSTODY_ONLY,
)
CUSTODY_EXCEPTIONS: Final = (
    "data/private/system_archives/**",
    "docs/migrations/unified-cutover/**",
    "tests/fixtures/legacy/**",
)

_RULE_FIELDS: Final = {
    "bootstrap_decision",
    "kind",
    "contract_sha256",
    "tracked_roots",
    "external_roots",
    "entrypoint_seeds",
    "shadow_seeds",
    "classification_fallbacks",
    "json_reference_keys",
    "custody_exceptions",
    "dynamic_import_allowlist",
    "legacy_seed_manifest",
    "legacy_custody_scope",
    "replacement_test_map",
    "source_to_target_table",
    "unified_layout",
}
_UNIFIED_LAYOUT_FIELDS: Final = {
    "active_pointer",
    "permanent_marker",
    "object_root",
    "generation_root",
    "pointer_history_root",
    "archive_root_template",
}


def _contract_sha256(kind: str, fields: Sequence[str]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "field_names": sorted(fields),
                "kind": kind,
                "strict_fields": True,
            }
        )
    )


RULES_CONTRACT_SHA256: Final = _contract_sha256(RULES_KIND, tuple(_RULE_FIELDS))
DYNAMIC_ALLOWLIST_CONTRACT_SHA256: Final = _contract_sha256(
    DYNAMIC_ALLOWLIST_KIND, ("kind", "contract_sha256", "entries")
)
LEGACY_SEED_CONTRACT_SHA256: Final = _contract_sha256(
    LEGACY_SEED_KIND,
    (
        "kind",
        "contract_sha256",
        "baseline_commit",
        "baseline_tree",
        "entries",
        "module_seed_prefixes",
        "path_seed_patterns",
        "removed_entrypoint_tokens",
        "runtime_seed_patterns",
        "seed_set_sha256",
    ),
)
SOURCE_TO_TARGET_CONTRACT_SHA256: Final = _contract_sha256(
    SOURCE_TO_TARGET_KIND, ("kind", "contract_sha256", "mappings")
)
REPLACEMENT_TEST_MAP_CONTRACT_SHA256: Final = _contract_sha256(
    REPLACEMENT_TEST_MAP_KIND,
    (
        "kind",
        "contract_sha256",
        "baseline_commit",
        "baseline_tree",
        "legacy_test_seed_patterns",
        "entries",
        "baseline_test_set_sha256",
        "map_sha256",
    ),
)
LEGACY_CUSTODY_SCOPE_CONTRACT_SHA256: Final = _contract_sha256(
    LEGACY_CUSTODY_SCOPE_KIND,
    (
        "kind",
        "contract_sha256",
        "baseline_custody_facts",
        "pointer_filename_rules",
        "retired_active_paths",
        "runtime_roots",
    ),
)


def _fail(detail: str) -> None:
    raise UnifiedCutoverError(RULES_SCHEMA_INVALID, detail)


def _exact_mapping(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _fail(f"{label} must contain exactly {sorted(fields)}")
    return dict(value)


def _sorted_unique_strings(value: Any, *, label: str, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not list or (nonempty and not value):
        _fail(f"{label} must be a {'non-empty ' if nonempty else ''}list")
    if any(type(item) is not str or not item for item in value):
        _fail(f"{label} must contain non-empty strings")
    result = tuple(value)
    if list(result) != sorted(set(result)):
        _fail(f"{label} must be sorted and unique")
    return result


def path_matches_glob(relative_path: str, pattern: str) -> bool:
    """Match canonical POSIX paths, with directory ``/**`` including descendants."""

    relative = canonical_relative_path(relative_path)
    if not isinstance(pattern, str) or not pattern or "\\" in pattern:
        _fail("glob pattern is invalid")
    if pattern == "**":
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return relative == prefix or relative.startswith(prefix + "/")
    return PurePosixPath(relative).match(pattern)


@dataclass(frozen=True)
class TrackedRoot:
    path: str
    required: bool
    classification: str | None


@dataclass(frozen=True)
class ExternalRoot:
    base: str
    path: str
    inventory_prefix: str
    required: bool
    classification: str


@dataclass(frozen=True)
class RuntimeRoot:
    path: str
    required: bool
    classification: str
    traversal: str


@dataclass(frozen=True)
class GraphSeed:
    kind: str
    value: str


@dataclass(frozen=True)
class UnifiedLayout:
    active_pointer: str
    permanent_marker: str
    object_root: str
    generation_root: str
    pointer_history_root: str
    archive_root_template: str

    def archive_root(self, cutover_id: str) -> str:
        if not cutover_id or "/" in cutover_id or ".." in cutover_id:
            _fail("cutover_id is unsafe")
        return canonical_relative_path(
            self.archive_root_template.format(cutover_id=cutover_id),
            label="archive root",
        )


@dataclass(frozen=True)
class CutoverRules:
    baseline_custody_facts: Mapping[str, Any]
    tracked_roots: tuple[TrackedRoot, ...]
    external_roots: tuple[ExternalRoot, ...]
    runtime_roots: tuple[RuntimeRoot, ...]
    entrypoint_seeds: tuple[GraphSeed, ...]
    shadow_seeds: tuple[GraphSeed, ...]
    classification_fallbacks: Mapping[str, tuple[str, ...]]
    pointer_filename_rules: Mapping[str, tuple[str, ...]]
    json_reference_keys: tuple[str, ...]
    custody_exceptions: tuple[str, ...]
    dynamic_import_allowlist: str
    legacy_seed_manifest: str
    legacy_custody_scope: str
    replacement_test_map: str
    source_to_target_table: str
    bootstrap_decision: str
    unified_layout: UnifiedLayout


@dataclass(frozen=True)
class LoadedRules:
    path: Path
    raw: bytes
    sha256: str
    document: Mapping[str, Any]
    rules: CutoverRules


@dataclass(frozen=True)
class DynamicImportAllowance:
    relative_path: str
    line: int
    ast_sha256: str
    modules: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.relative_path, self.line, self.ast_sha256)


@dataclass(frozen=True)
class LoadedDynamicAllowlist:
    raw: bytes
    sha256: str
    entries: Mapping[tuple[str, int, str], DynamicImportAllowance]


@dataclass(frozen=True)
class LegacySeed:
    relative_path: str
    byte_sha256: str
    module: str | None


@dataclass(frozen=True)
class LoadedLegacySeeds:
    raw: bytes
    sha256: str
    seed_set_sha256: str
    baseline_commit: str
    baseline_tree: str
    entries: tuple[LegacySeed, ...]
    module_seed_prefixes: tuple[str, ...]
    path_seed_patterns: tuple[str, ...]
    runtime_seed_patterns: tuple[str, ...]
    removed_entrypoint_tokens: tuple[str, ...]


@dataclass(frozen=True)
class LoadedBootstrapDecision:
    relative_path: str
    raw: bytes
    sha256: str
    document: Mapping[str, Any]


@dataclass(frozen=True)
class RetiredActivePath:
    relative_path: str
    byte_sha256: str
    disposition: str
    behavior_reason: str


@dataclass(frozen=True)
class LoadedLegacyCustodyScope:
    relative_path: str
    raw: bytes
    sha256: str
    baseline_custody_facts: Mapping[str, Any]
    runtime_roots: tuple[RuntimeRoot, ...]
    pointer_filename_rules: Mapping[str, tuple[str, ...]]
    retired_active_paths: tuple[RetiredActivePath, ...]


@dataclass(frozen=True)
class ReplacementTestEntry:
    baseline_test_path: str
    disposition: str
    replacement_test_selectors: tuple[str, ...]
    behavior_reason: str


@dataclass(frozen=True)
class LoadedReplacementTestMap:
    relative_path: str
    raw: bytes
    sha256: str
    baseline_commit: str
    baseline_tree: str
    legacy_test_seed_patterns: tuple[str, ...]
    entries: tuple[ReplacementTestEntry, ...]


def _parse_roots(value: Any, *, runtime: bool) -> tuple[TrackedRoot, ...] | tuple[RuntimeRoot, ...]:
    if type(value) is not list:
        _fail("roots must be a list")
    seen: set[str] = set()
    rows: list[TrackedRoot] | list[RuntimeRoot] = []
    for index, raw in enumerate(value):
        if runtime:
            row = _exact_mapping(
                raw,
                {"path", "required", "classification", "traversal"},
                label=f"runtime_roots[{index}]",
            )
        else:
            row = _exact_mapping(
                raw,
                {"path", "required", "classification"},
                label=f"tracked_roots[{index}]",
            )
        path = canonical_relative_path(row["path"], label="root path")
        if path in seen:
            _fail(f"duplicate root path: {path}")
        seen.add(path)
        if type(row["required"]) is not bool:
            _fail(f"root required flag is invalid: {path}")
        if runtime:
            classification = row["classification"]
            traversal = row["traversal"]
            if classification not in CUSTODY_CLASSIFICATIONS:
                _fail(f"runtime root classification is invalid: {path}")
            if traversal not in {"POINTER_CLOSURE", "INVENTORY_ONLY", "BOUNDARY"}:
                _fail(f"runtime root traversal is invalid: {path}")
            rows.append(RuntimeRoot(path, row["required"], classification, traversal))
        else:
            classification = row["classification"]
            if classification is not None and classification not in CUSTODY_CLASSIFICATIONS:
                _fail(f"tracked root classification is invalid: {path}")
            rows.append(TrackedRoot(path, row["required"], classification))
    if [row.path for row in rows] != sorted(row.path for row in rows):
        _fail("roots must be sorted by path")
    return tuple(rows)


def _parse_external_roots(value: Any) -> tuple[ExternalRoot, ...]:
    if type(value) is not list:
        _fail("external_roots must be a list")
    rows: list[ExternalRoot] = []
    keys: list[tuple[str, str, str]] = []
    for index, raw in enumerate(value):
        row = _exact_mapping(
            raw,
            {"base", "path", "inventory_prefix", "required", "classification"},
            label=f"external_roots[{index}]",
        )
        if row["base"] != "CODEX_HOME":
            _fail("external root base must be CODEX_HOME")
        path = canonical_relative_path(row["path"], label="external path")
        prefix = canonical_relative_path(row["inventory_prefix"], label="inventory prefix")
        if type(row["required"]) is not bool or row["classification"] not in {
            ACTIVE_CALLER,
            LEGACY_INACTIVE,
        }:
            _fail("external root metadata is invalid")
        keys.append((row["base"], path, prefix))
        rows.append(
            ExternalRoot(
                row["base"],
                path,
                prefix,
                row["required"],
                row["classification"],
            )
        )
    if keys != sorted(set(keys)):
        _fail("external roots must be sorted and unique")
    return tuple(rows)


def _parse_seeds(value: Any, *, label: str) -> tuple[GraphSeed, ...]:
    if type(value) is not list:
        _fail(f"{label} must be a list")
    result: list[GraphSeed] = []
    keys: list[tuple[str, str]] = []
    for index, raw in enumerate(value):
        row = _exact_mapping(raw, {"kind", "value"}, label=f"{label}[{index}]")
        kind = row["kind"]
        item = row["value"]
        if kind not in {"module", "path", "pointer"} or type(item) is not str or not item:
            _fail(f"{label}[{index}] is invalid")
        if kind in {"path", "pointer"}:
            item = canonical_relative_path(item, label=f"{label}[{index}].value")
        keys.append((kind, item))
        result.append(GraphSeed(kind, item))
    if keys != sorted(set(keys)):
        _fail(f"{label} must be sorted and unique")
    return tuple(result)


def validate_rules_document(document: Mapping[str, Any]) -> CutoverRules:
    row = _exact_mapping(document, _RULE_FIELDS, label="rules")
    if row["kind"] != RULES_KIND or row["contract_sha256"] != RULES_CONTRACT_SHA256:
        _fail("rules kind or immutable contract SHA-256 is unsupported")

    fallbacks_raw = row["classification_fallbacks"]
    if type(fallbacks_raw) is not dict or set(fallbacks_raw) != set(CUSTODY_CLASSIFICATIONS):
        _fail("classification_fallbacks must contain every exact custody classification")
    fallbacks: dict[str, tuple[str, ...]] = {}
    for classification in CUSTODY_CLASSIFICATIONS:
        fallbacks[classification] = _sorted_unique_strings(
            fallbacks_raw[classification],
            label=f"classification_fallbacks.{classification}",
            nonempty=False,
        )

    custody = _sorted_unique_strings(row["custody_exceptions"], label="custody_exceptions")
    if custody != tuple(sorted(CUSTODY_EXCEPTIONS)):
        _fail("custody_exceptions differ from the exact cutover exceptions")

    reference_keys = _sorted_unique_strings(row["json_reference_keys"], label="json_reference_keys")

    layout_raw = _exact_mapping(row["unified_layout"], _UNIFIED_LAYOUT_FIELDS, label="layout")
    for key in _UNIFIED_LAYOUT_FIELDS:
        if type(layout_raw[key]) is not str or not layout_raw[key]:
            _fail(f"unified_layout.{key} is invalid")
    layout = UnifiedLayout(
        active_pointer=canonical_relative_path(layout_raw["active_pointer"]),
        permanent_marker=canonical_relative_path(layout_raw["permanent_marker"]),
        object_root=canonical_relative_path(layout_raw["object_root"]),
        generation_root=canonical_relative_path(layout_raw["generation_root"]),
        pointer_history_root=canonical_relative_path(layout_raw["pointer_history_root"]),
        archive_root_template=layout_raw["archive_root_template"],
    )
    if layout != UnifiedLayout(
        UNIFIED_ACTIVE_POINTER,
        PERMANENT_MARKER,
        UNIFIED_OBJECT_ROOT,
        UNIFIED_GENERATION_ROOT,
        UNIFIED_POINTER_HISTORY_ROOT,
        ARCHIVE_ROOT_TEMPLATE,
    ):
        _fail("unified_layout differs from the canonical cutover layout")
    layout.archive_root("layout-check")

    allowlist = canonical_relative_path(row["dynamic_import_allowlist"])
    legacy = canonical_relative_path(row["legacy_seed_manifest"])
    legacy_custody_scope = canonical_relative_path(row["legacy_custody_scope"])
    replacement_test_map = canonical_relative_path(row["replacement_test_map"])
    table = canonical_relative_path(row["source_to_target_table"])
    bootstrap = canonical_relative_path(row["bootstrap_decision"])

    return CutoverRules(
        baseline_custody_facts={},
        tracked_roots=_parse_roots(row["tracked_roots"], runtime=False),
        external_roots=_parse_external_roots(row["external_roots"]),
        runtime_roots=(),
        entrypoint_seeds=_parse_seeds(row["entrypoint_seeds"], label="entrypoint_seeds"),
        shadow_seeds=_parse_seeds(row["shadow_seeds"], label="shadow_seeds"),
        classification_fallbacks=fallbacks,
        pointer_filename_rules={"active": (), "reachable": ()},
        json_reference_keys=reference_keys,
        custody_exceptions=custody,
        dynamic_import_allowlist=allowlist,
        legacy_seed_manifest=legacy,
        legacy_custody_scope=legacy_custody_scope,
        replacement_test_map=replacement_test_map,
        source_to_target_table=table,
        bootstrap_decision=bootstrap,
        unified_layout=layout,
    )


def _load_canonical_mapping(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = read_stable_regular_file(path, label=label, max_bytes=4 * 1024 * 1024)
    except UnifiedCutoverError:
        raise
    except OSError as exc:  # pragma: no cover - safe reader normalizes this
        raise UnifiedCutoverError(RULES_UNAVAILABLE, f"{label} is unavailable") from exc
    try:
        document = parse_json_bytes(raw, label=label, require_canonical=True)
    except UnifiedCutoverError as exc:
        raise UnifiedCutoverError(RULES_NON_CANONICAL, f"{label}: {exc.detail}") from exc
    if type(document) is not dict:
        raise UnifiedCutoverError(RULES_SCHEMA_INVALID, f"{label} must be an object")
    return raw, document


def load_legacy_custody_scope(root: Path, relative_path: str) -> LoadedLegacyCustodyScope:
    relative = canonical_relative_path(relative_path, label="legacy custody scope path")
    raw, document = _load_canonical_mapping(root / relative, label="legacy custody scope")
    if sha256_bytes(raw) != LEGACY_CUSTODY_SCOPE_BYTE_SHA256:
        _fail("legacy custody scope bytes differ from the approved exact custody record")
    if set(document) != {
        "kind",
        "contract_sha256",
        "baseline_custody_facts",
        "pointer_filename_rules",
        "retired_active_paths",
        "runtime_roots",
    } or (
        document.get("kind") != LEGACY_CUSTODY_SCOPE_KIND
        or document.get("contract_sha256") != LEGACY_CUSTODY_SCOPE_CONTRACT_SHA256
    ):
        _fail("legacy custody scope kind, contract, or fields are invalid")
    facts = document.get("baseline_custody_facts")
    if (
        type(facts) is not dict
        or sha256_bytes(canonical_json_bytes(facts)) != BASELINE_CUSTODY_FACTS_SHA256
    ):
        _fail("legacy custody facts differ from the read-only cutover evidence")
    runtime_roots = _parse_roots(document.get("runtime_roots"), runtime=True)
    pointer_raw = document.get("pointer_filename_rules")
    if type(pointer_raw) is not dict or set(pointer_raw) != {"active", "reachable"}:
        _fail("legacy pointer filename rules fields are not exact")
    pointer_rules: dict[str, tuple[str, ...]] = {}
    for key in ("active", "reachable"):
        patterns = _sorted_unique_strings(
            pointer_raw[key], label=f"legacy pointer_filename_rules.{key}"
        )
        if any("/" in pattern or pattern in {".", ".."} for pattern in patterns):
            _fail("legacy pointer filename patterns must be basenames")
        pointer_rules[key] = patterns
    retired_raw = document.get("retired_active_paths")
    if type(retired_raw) is not list or not retired_raw:
        _fail("retired active paths must be a non-empty list")
    retired_active_paths: list[RetiredActivePath] = []
    serialized_paths: list[str] = []
    for index, raw_entry in enumerate(retired_raw):
        row = _exact_mapping(
            raw_entry,
            {"relative_path", "byte_sha256", "disposition", "behavior_reason"},
            label=f"retired_active_paths[{index}]",
        )
        relative_path = canonical_relative_path(row["relative_path"], label="retired active path")
        byte_sha256 = row["byte_sha256"]
        disposition = row["disposition"]
        behavior_reason = row["behavior_reason"]
        if type(byte_sha256) is not str or SHA256_RE.fullmatch(byte_sha256) is None:
            _fail("retired active path byte SHA-256 is invalid")
        if disposition not in {"REPLACED", "BEHAVIOR_INTENTIONALLY_REMOVED"}:
            _fail("retired active path disposition is invalid")
        if (
            type(behavior_reason) is not str
            or not behavior_reason.strip()
            or behavior_reason != behavior_reason.strip()
        ):
            _fail("retired active path behavior reason is invalid")
        serialized_paths.append(relative_path)
        retired_active_paths.append(
            RetiredActivePath(
                relative_path,
                byte_sha256,
                disposition,
                behavior_reason,
            )
        )
    if serialized_paths != sorted(set(serialized_paths)):
        _fail("retired active paths must be sorted and unique")
    return LoadedLegacyCustodyScope(
        relative,
        raw,
        sha256_bytes(raw),
        facts,
        runtime_roots,
        pointer_rules,
        tuple(retired_active_paths),
    )


def load_rules(
    workspace_root: str | Path,
    rules_path: str | Path = RULES_RELATIVE_PATH,
) -> LoadedRules:
    root = Path(workspace_root).resolve(strict=True)
    path = Path(rules_path)
    if not path.is_absolute():
        path = root / canonical_relative_path(path.as_posix(), label="rules path")
    raw, document = _load_canonical_mapping(path, label="cutover rules")
    base_rules = validate_rules_document(document)
    scope = load_legacy_custody_scope(root, base_rules.legacy_custody_scope)
    rules = replace(
        base_rules,
        baseline_custody_facts=scope.baseline_custody_facts,
        runtime_roots=scope.runtime_roots,
        pointer_filename_rules=scope.pointer_filename_rules,
    )
    return LoadedRules(path, raw, sha256_bytes(raw), document, rules)


def load_dynamic_allowlist(root: Path, relative_path: str) -> LoadedDynamicAllowlist:
    raw, document = _load_canonical_mapping(
        root / canonical_relative_path(relative_path), label="dynamic import allowlist"
    )
    if set(document) != {"kind", "contract_sha256", "entries"} or (
        document.get("kind") != DYNAMIC_ALLOWLIST_KIND
        or document.get("contract_sha256") != DYNAMIC_ALLOWLIST_CONTRACT_SHA256
    ):
        _fail("dynamic import allowlist kind or contract SHA-256 is invalid")
    entries_raw = document.get("entries")
    if type(entries_raw) is not list:
        _fail("dynamic import allowlist entries must be a list")
    entries: dict[tuple[str, int, str], DynamicImportAllowance] = {}
    serialized_keys: list[tuple[str, int, str]] = []
    for index, raw_entry in enumerate(entries_raw):
        row = _exact_mapping(
            raw_entry,
            {"relative_path", "line", "ast_sha256", "modules"},
            label=f"allowlist.entries[{index}]",
        )
        relative = canonical_relative_path(row["relative_path"])
        line = row["line"]
        digest = row["ast_sha256"]
        if type(line) is not int or line <= 0:
            _fail("allowlist line must be a positive integer")
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            _fail("allowlist AST SHA-256 is invalid")
        modules = _sorted_unique_strings(row["modules"], label="allowlist modules")
        if any(not _valid_module_name(module) for module in modules):
            _fail("allowlist module name is invalid")
        allowance = DynamicImportAllowance(relative, line, digest, modules)
        if allowance.key in entries:
            _fail("dynamic import allowlist contains a duplicate exact key")
        entries[allowance.key] = allowance
        serialized_keys.append(allowance.key)
    if serialized_keys != sorted(serialized_keys):
        _fail("dynamic import allowlist entries must be sorted by exact key")
    return LoadedDynamicAllowlist(raw, sha256_bytes(raw), entries)


def _valid_module_name(value: str) -> bool:
    return bool(value) and all(part.isidentifier() for part in value.split("."))


def load_legacy_seeds(root: Path, relative_path: str) -> LoadedLegacySeeds:
    raw, document = _load_canonical_mapping(
        root / canonical_relative_path(relative_path), label="legacy seed manifest"
    )
    if set(document) != {
        "kind",
        "contract_sha256",
        "baseline_commit",
        "baseline_tree",
        "entries",
        "module_seed_prefixes",
        "path_seed_patterns",
        "removed_entrypoint_tokens",
        "runtime_seed_patterns",
        "seed_set_sha256",
    } or (
        document.get("kind") != LEGACY_SEED_KIND
        or document.get("contract_sha256") != LEGACY_SEED_CONTRACT_SHA256
    ):
        _fail("legacy seed manifest kind or contract SHA-256 is invalid")
    rows = document.get("entries")
    if type(rows) is not list:
        _fail("legacy seed entries must be a list")
    baseline_commit = document.get("baseline_commit")
    baseline_tree = document.get("baseline_tree")
    if (
        type(baseline_commit) is not str
        or len(baseline_commit) != 40
        or any(character not in "0123456789abcdef" for character in baseline_commit)
        or type(baseline_tree) is not str
        or len(baseline_tree) != 40
        or any(character not in "0123456789abcdef" for character in baseline_tree)
    ):
        _fail("legacy baseline commit/tree identities are invalid")
    if baseline_commit != LEGACY_BASELINE_COMMIT or baseline_tree != LEGACY_BASELINE_TREE:
        _fail("legacy baseline identity differs from the approved cutover baseline")
    module_prefixes = _sorted_unique_strings(
        document.get("module_seed_prefixes"),
        label="legacy module_seed_prefixes",
    )
    if any(not _valid_module_prefix(prefix) for prefix in module_prefixes):
        _fail("legacy module seed prefix is invalid")
    path_patterns = _sorted_unique_strings(
        document.get("path_seed_patterns"),
        label="legacy path_seed_patterns",
    )
    runtime_patterns = _sorted_unique_strings(
        document.get("runtime_seed_patterns"),
        label="legacy runtime_seed_patterns",
        nonempty=False,
    )
    removed_tokens = _sorted_unique_strings(
        document.get("removed_entrypoint_tokens"),
        label="legacy removed_entrypoint_tokens",
        nonempty=False,
    )
    expected_set_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "baseline_commit": baseline_commit,
                "baseline_tree": baseline_tree,
                "entries": rows,
                "module_seed_prefixes": list(module_prefixes),
                "path_seed_patterns": list(path_patterns),
                "removed_entrypoint_tokens": list(removed_tokens),
                "runtime_seed_patterns": list(runtime_patterns),
            }
        )
    )
    if document.get("seed_set_sha256") != expected_set_sha:
        _fail("legacy seed_set_sha256 mismatch")
    entries: list[LegacySeed] = []
    keys: list[str] = []
    for index, raw_entry in enumerate(rows):
        row = _exact_mapping(
            raw_entry,
            {"relative_path", "byte_sha256", "module"},
            label=f"legacy.entries[{index}]",
        )
        relative = canonical_relative_path(row["relative_path"])
        digest = row["byte_sha256"]
        module = row["module"]
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            _fail("legacy seed byte_sha256 is invalid")
        if module is not None and (type(module) is not str or not _valid_module_name(module)):
            _fail("legacy seed module is invalid")
        keys.append(relative)
        entries.append(LegacySeed(relative, digest, module))
    if keys != sorted(set(keys)):
        _fail("legacy seed entries must be sorted and unique")
    for entry in entries:
        if not any(path_matches_glob(entry.relative_path, pattern) for pattern in path_patterns):
            _fail(f"legacy entry is outside all path seed patterns: {entry.relative_path}")
        if entry.module is not None and not any(
            entry.module == prefix or entry.module.startswith(prefix) for prefix in module_prefixes
        ):
            _fail(f"legacy module is outside all module seed prefixes: {entry.module}")
    return LoadedLegacySeeds(
        raw,
        sha256_bytes(raw),
        expected_set_sha,
        baseline_commit,
        baseline_tree,
        tuple(entries),
        module_prefixes,
        path_patterns,
        runtime_patterns,
        removed_tokens,
    )


def _valid_module_prefix(value: str) -> bool:
    if not value or value.startswith(".") or value.endswith("."):
        return False
    return all(part.isidentifier() for part in value.split("."))


def pointer_filename_matches(filename: str, patterns: Sequence[str]) -> bool:
    if not filename or "/" in filename:
        return False
    return any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)


def load_bootstrap_decision(root: Path, relative_path: str) -> LoadedBootstrapDecision:
    relative = canonical_relative_path(relative_path, label="bootstrap decision path")
    raw, document = _load_canonical_mapping(root / relative, label="bootstrap decision")
    expected = {
        "kind": BOOTSTRAP_DECISION_KIND,
        "decision_source_id": "user-approved-unified-runtime-cutover",
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": "NOT_CLAIMED",
        "factor_weights": [
            {
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "weight": "0.500000000000",
            },
            {
                "factor_id": "pv_low_dollar_volume_5d",
                "weight": "0.500000000000",
            },
        ],
        "control_factor_ids": ["pv_blend_volstab19x2_mom90_amihud5_w75"],
        "prospective_evidence_claimed": False,
        "activation_authorized": False,
    }
    if document != expected or sha256_bytes(raw) != BOOTSTRAP_DECISION_BYTE_SHA256:
        _fail("bootstrap decision differs from the exact user-approved payload")
    return LoadedBootstrapDecision(relative, raw, sha256_bytes(raw), document)


def load_replacement_test_map(root: Path, relative_path: str) -> LoadedReplacementTestMap:
    relative = canonical_relative_path(relative_path, label="replacement test map path")
    raw, document = _load_canonical_mapping(root / relative, label="replacement test map")

    def invalid(detail: str) -> None:
        raise UnifiedCutoverError(REPLACEMENT_TEST_MAP_INVALID, detail)

    fields = {
        "kind",
        "contract_sha256",
        "baseline_commit",
        "baseline_tree",
        "legacy_test_seed_patterns",
        "entries",
        "baseline_test_set_sha256",
        "map_sha256",
    }
    if set(document) != fields:
        invalid("replacement test map fields are not exact")
    if (
        document.get("kind") != REPLACEMENT_TEST_MAP_KIND
        or document.get("contract_sha256") != REPLACEMENT_TEST_MAP_CONTRACT_SHA256
        or document.get("baseline_commit") != LEGACY_BASELINE_COMMIT
        or document.get("baseline_tree") != LEGACY_BASELINE_TREE
    ):
        invalid("replacement test map identity is unsupported")
    try:
        patterns = _sorted_unique_strings(
            document.get("legacy_test_seed_patterns"),
            label="legacy_test_seed_patterns",
        )
    except UnifiedCutoverError as exc:
        invalid(exc.detail)
    rows = document.get("entries")
    if type(rows) is not list or not rows:
        invalid("replacement test entries must be a non-empty list")
    entries: list[ReplacementTestEntry] = []
    serialized_paths: list[str] = []
    for index, raw_entry in enumerate(rows):
        if type(raw_entry) is not dict or set(raw_entry) != {
            "baseline_test_path",
            "disposition",
            "replacement_test_selectors",
            "behavior_reason",
        }:
            invalid(f"replacement test entry {index} fields are not exact")
        try:
            baseline_path = canonical_relative_path(
                raw_entry["baseline_test_path"], label="baseline test path"
            )
        except UnifiedCutoverError as exc:
            invalid(exc.detail)
        if not baseline_path.startswith("tests/") or not baseline_path.endswith(".py"):
            invalid("baseline test path is not a Python test")
        if not any(path_matches_glob(baseline_path, pattern) for pattern in patterns):
            invalid(f"baseline test lies outside seed patterns: {baseline_path}")
        disposition = raw_entry["disposition"]
        selectors_raw = raw_entry["replacement_test_selectors"]
        reason = raw_entry["behavior_reason"]
        if type(selectors_raw) is not list or selectors_raw != sorted(set(selectors_raw)):
            invalid("replacement selectors must be a sorted unique list")
        selectors: list[str] = []
        for selector in selectors_raw:
            if type(selector) is not str or selector.count("::") < 1:
                invalid("replacement selector must include an exact test node")
            test_path, *nodes = selector.split("::")
            try:
                test_path = canonical_relative_path(test_path, label="replacement test path")
            except UnifiedCutoverError as exc:
                invalid(exc.detail)
            if (
                not test_path.startswith("tests/")
                or not test_path.endswith(".py")
                or any(not node or not node.isidentifier() for node in nodes)
            ):
                invalid("replacement selector is not canonical")
            selectors.append("::".join((test_path, *nodes)))
        if type(reason) is not str or not reason.strip() or reason != reason.strip():
            invalid("every replacement/removal row requires an exact behavior reason")
        if disposition == "REPLACED":
            if not selectors:
                invalid("REPLACED row has no exact stable selector")
        elif disposition == "BEHAVIOR_INTENTIONALLY_REMOVED":
            if selectors:
                invalid("intentionally removed row must not name replacement selectors")
        else:
            invalid("replacement disposition is unsupported")
        serialized_paths.append(baseline_path)
        entries.append(
            ReplacementTestEntry(
                baseline_path,
                disposition,
                tuple(selectors),
                reason,
            )
        )
    if serialized_paths != sorted(set(serialized_paths)):
        invalid("replacement entries must be sorted and unique by baseline path")
    expected_set_sha = sha256_bytes(canonical_json_bytes(serialized_paths))
    if document.get("baseline_test_set_sha256") != expected_set_sha:
        invalid("baseline test set SHA-256 mismatch")
    map_body = {
        "baseline_commit": document["baseline_commit"],
        "baseline_tree": document["baseline_tree"],
        "baseline_test_set_sha256": expected_set_sha,
        "entries": rows,
        "legacy_test_seed_patterns": list(patterns),
    }
    if document.get("map_sha256") != sha256_bytes(canonical_json_bytes(map_body)):
        invalid("replacement map SHA-256 mismatch")
    return LoadedReplacementTestMap(
        relative,
        raw,
        sha256_bytes(raw),
        LEGACY_BASELINE_COMMIT,
        LEGACY_BASELINE_TREE,
        patterns,
        tuple(entries),
    )


__all__ = [
    "ACTIVE_AUTHORITY",
    "ACTIVE_CALLER",
    "ARCHIVE_ROOT_TEMPLATE",
    "BASELINE_CUSTODY_FACTS_SHA256",
    "BOOTSTRAP_DECISION_BYTE_SHA256",
    "BOOTSTRAP_DECISION_KIND",
    "BOOTSTRAP_DECISION_RELATIVE_PATH",
    "CUSTODY_CLASSIFICATIONS",
    "CUSTODY_EXCEPTIONS",
    "CUSTODY_ONLY",
    "CutoverRules",
    "DYNAMIC_ALLOWLIST_CONTRACT_SHA256",
    "DYNAMIC_ALLOWLIST_KIND",
    "DynamicImportAllowance",
    "ExternalRoot",
    "GraphSeed",
    "INDEPENDENT_SOURCE",
    "LEGACY_SEED_CONTRACT_SHA256",
    "LEGACY_BASELINE_COMMIT",
    "LEGACY_BASELINE_TREE",
    "LEGACY_CUSTODY_SCOPE_CONTRACT_SHA256",
    "LEGACY_CUSTODY_SCOPE_BYTE_SHA256",
    "LEGACY_CUSTODY_SCOPE_KIND",
    "LEGACY_CUSTODY_SCOPE_RELATIVE_PATH",
    "LEGACY_SEED_KIND",
    "LEGACY_INACTIVE",
    "LoadedBootstrapDecision",
    "LoadedDynamicAllowlist",
    "LoadedLegacySeeds",
    "LoadedLegacyCustodyScope",
    "LoadedReplacementTestMap",
    "LoadedRules",
    "NON_AUTHORITY_SHADOW",
    "PERMANENT_MARKER",
    "RULES_CONTRACT_SHA256",
    "RULES_KIND",
    "RULES_RELATIVE_PATH",
    "REPLACEMENT_TEST_MAP_CONTRACT_SHA256",
    "REPLACEMENT_TEST_MAP_KIND",
    "REPLACEMENT_TEST_MAP_RELATIVE_PATH",
    "ReplacementTestEntry",
    "RetiredActivePath",
    "RuntimeRoot",
    "SOURCE_TO_TARGET_RELATIVE_PATH",
    "SOURCE_TO_TARGET_CONTRACT_SHA256",
    "SOURCE_TO_TARGET_KIND",
    "TrackedRoot",
    "UNIFIED_ACTIVE_POINTER",
    "UNIFIED_GENERATION_ROOT",
    "UNIFIED_OBJECT_ROOT",
    "UNIFIED_POINTER_HISTORY_ROOT",
    "UnifiedLayout",
    "load_dynamic_allowlist",
    "load_bootstrap_decision",
    "load_legacy_seeds",
    "load_legacy_custody_scope",
    "load_rules",
    "load_replacement_test_map",
    "path_matches_glob",
    "pointer_filename_matches",
    "validate_rules_document",
]
