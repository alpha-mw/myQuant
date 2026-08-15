from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.migration import (
    BOOTSTRAP_DECISION_BYTE_SHA256,
    LEGACY_CUSTODY_SCOPE_BYTE_SHA256,
)
from quant_investor.migration.errors import (
    BROKEN_REFERENCE,
    CLASSIFICATION_COLLISION,
    REPLACEMENT_TEST_MAP_INVALID,
    SYMLINK_REFUSED,
    UNCLASSIFIED_PATH,
    UNPARSEABLE_JSON,
    UnifiedCutoverError,
)
from quant_investor.migration.resolver import resolve_unified_cutover_inventory
from quant_investor.migration.parsers import module_name_for_path
from quant_investor.migration.rules import (
    BOOTSTRAP_DECISION_RELATIVE_PATH,
    LEGACY_CUSTODY_SCOPE_RELATIVE_PATH,
    REPLACEMENT_TEST_MAP_RELATIVE_PATH,
    RULES_RELATIVE_PATH,
    load_bootstrap_decision,
    load_legacy_custody_scope,
    load_legacy_seeds,
    load_replacement_test_map,
    load_rules,
    path_matches_glob,
)

from test_unified_migration_helpers import (
    exact_runtime_root,
    make_test_workspace,
    sha,
    write_canonical,
)


CREATED_AT = "2026-08-14T00:00:00Z"


def _resolved_imports(relative_path: str, raw: bytes) -> set[str]:
    tree = ast.parse(raw.decode("utf-8"), filename=relative_path)
    named = module_name_for_path(relative_path)
    current_module, is_package = named if named is not None else ("", False)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                package = (
                    current_module.split(".")
                    if is_package
                    else current_module.split(".")[:-1]
                )
                ascend = node.level - 1
                prefix = (
                    package[: len(package) - ascend]
                    if ascend < len(package)
                    else []
                )
                module_parts = (node.module or "").split(".") if node.module else []
                base = ".".join(prefix + module_parts)
            if base:
                result.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    result.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if not isinstance(first, ast.Constant) or type(first.value) is not str:
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                result.add(first.value)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                result.add(first.value)
    return result


def _closed_workspace(root: Path) -> tuple[Path, tuple[str, ...]]:
    object_raw = canonical_json_bytes({"payload": "authority"})
    shadow_root = exact_runtime_root(
        classification="NON_AUTHORITY_SHADOW", traversal="POINTER_CLOSURE"
    )
    custody_root = exact_runtime_root(
        classification="CUSTODY_ONLY", traversal="INVENTORY_ONLY"
    )
    object_path = f"{shadow_root}/object.json"
    pointer = {
        "object_byte_sha256": sha(object_raw),
        "object_path": object_path,
    }
    (root / object_path).parent.mkdir(parents=True, exist_ok=True)
    (root / object_path).write_bytes(object_raw)
    write_canonical(root / shadow_root / "_active.json", pointer)
    archive = root / custody_root / "cutover-test/authority_closure/historical.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"not-json-and-never-traversed")
    return make_test_workspace(
        root,
        source_files={
            "src/main.py": b"import src.lib\n",
            "src/lib.py": b"VALUE = 1\n",
        },
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )


def test_repo_rules_and_bootstrap_are_exact_canonical_no_newline() -> None:
    root = Path(__file__).resolve().parents[2]
    loaded = load_rules(root, RULES_RELATIVE_PATH)
    assert loaded.raw == canonical_json_bytes(dict(loaded.document))
    assert not loaded.raw.endswith(b"\n")
    assert loaded.rules.legacy_seed_manifest == (
        "docs/migrations/unified-cutover/legacy-seed-manifest.json"
    )
    assert loaded.rules.replacement_test_map == REPLACEMENT_TEST_MAP_RELATIVE_PATH
    assert loaded.rules.legacy_custody_scope == LEGACY_CUSTODY_SCOPE_RELATIVE_PATH
    scope = load_legacy_custody_scope(root, loaded.rules.legacy_custody_scope)
    assert scope.sha256 == LEGACY_CUSTODY_SCOPE_BYTE_SHA256
    scope_document = parse_canonical_json_bytes(scope.raw)
    custody_only_values = [
        row["path"] for row in scope_document["runtime_roots"]
    ]
    custody_only_values.extend(
        pattern
        for patterns in scope_document["pointer_filename_rules"].values()
        for pattern in patterns
    )
    custody_only_values.extend(
        row["relative_path"]
        for row in scope_document["baseline_custody_facts"].values()
    )
    custody_only_values.extend(
        row["relative_path"] for row in scope_document["retired_active_paths"]
    )
    def scalar_strings(value: object) -> set[str]:
        if type(value) is dict:
            return {
                item
                for nested in value.values()
                for item in scalar_strings(nested)
            }
        if type(value) is list:
            return {item for nested in value for item in scalar_strings(nested)}
        return {value} if type(value) is str else set()

    active_rule_values = scalar_strings(loaded.document)
    assert all(value not in active_rule_values for value in custody_only_values)

    bootstrap = load_bootstrap_decision(root, BOOTSTRAP_DECISION_RELATIVE_PATH)
    assert bootstrap.sha256 == BOOTSTRAP_DECISION_BYTE_SHA256
    assert not bootstrap.raw.endswith(b"\n")
    assert bootstrap.document["factor_weights"] == [
        {
            "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
            "weight": "0.500000000000",
        },
        {"factor_id": "pv_low_dollar_volume_5d", "weight": "0.500000000000"},
    ]
    assert bootstrap.document["control_factor_ids"] == [
        "pv_blend_volstab19x2_mom90_amihud5_w75"
    ]
    assert bootstrap.document["activation_authorized"] is False

    macro_baseline_sha256 = (
        "c4e7f566f36371417218096d6c7cca02b1be1bee9c752e0d290bb02ea2678a46"
    )
    macro_rows = [
        entry
        for entry in scope.retired_active_paths
        if entry.byte_sha256 == macro_baseline_sha256
    ]
    assert len(macro_rows) == 1
    assert macro_rows[0].disposition == "REPLACED"
    assert not (root / macro_rows[0].relative_path).exists()
    macro_test_path = root / "tests/unit/test_macro_maintenance.py"
    macro_tree = ast.parse(
        macro_test_path.read_text(encoding="utf-8"), filename=str(macro_test_path)
    )
    macro_test_nodes = {
        node.name
        for node in macro_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "test_module_never_imports_retired_macro_mart",
        "test_dry_run_is_zero_write_and_commit_requires_live",
        "test_live_capture_is_hash_bound_before_both_publishers",
    }.issubset(macro_test_nodes)


def test_active_automation_is_read_only_and_has_no_legacy_bypass() -> None:
    root = Path(__file__).resolve().parents[2]
    rules = load_rules(root, RULES_RELATIVE_PATH).rules
    legacy = load_legacy_seeds(root, rules.legacy_seed_manifest)
    scope = load_legacy_custody_scope(root, rules.legacy_custody_scope)
    retired_automation_modules = {
        named[0]
        for entry in scope.retired_active_paths
        if (named := module_name_for_path(entry.relative_path)) is not None
        and named[0].startswith("quant_investor.automation.")
    }
    forbidden_imports = tuple(
        sorted({*legacy.module_seed_prefixes, *retired_automation_modules})
    )
    automation_root = root / "quant_investor/automation"
    imports_by_path: dict[str, set[str]] = {}
    violations: list[tuple[str, str]] = []
    for path in sorted(automation_root.glob("*.py")):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        imported_modules = _resolved_imports(relative, raw)
        imports_by_path[relative] = imported_modules
        for imported in imported_modules:
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in forbidden_imports
            ):
                violations.append((relative, imported))
        tree = ast.parse(raw.decode("utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            else:
                continue
            if called_name == "run_unified_pipeline":
                violations.append((relative, called_name))
    assert violations == [], f"active automation bypass remains: {violations[:1]}"

    analysis_imports = imports_by_path[
        "quant_investor/automation/analysis_runner.py"
    ]
    assert "quant_investor.mainline.read_public_run" in analysis_imports
    daily_imports = imports_by_path["quant_investor/automation/daily_runner.py"]
    assert "quant_investor.strategy_records.history.HistoryLoader" in daily_imports
    assert "quant_investor.automation.persistence" not in daily_imports
    assert "quant_investor.automation.report_builder" not in daily_imports


def test_frozen_legacy_executable_seeds_are_absent_from_active_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    rules = load_rules(root, RULES_RELATIVE_PATH).rules
    legacy = load_legacy_seeds(root, rules.legacy_seed_manifest)
    scope = load_legacy_custody_scope(root, rules.legacy_custody_scope)
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    checkout_paths = tuple(
        sorted(
            path
            for raw in completed.stdout.split(b"\0")
            if raw
            for path in [raw.decode("utf-8")]
            if (root / path).is_file()
        )
    )
    active_paths = tuple(
        path
        for path in checkout_paths
        if not any(path_matches_glob(path, pattern) for pattern in rules.custody_exceptions)
    )

    frozen_paths = {entry.relative_path for entry in legacy.entries}
    retained_paths = sorted(frozen_paths.intersection(active_paths))
    assert retained_paths == [], f"frozen source remains active: {retained_paths[:1]}"
    retired_active_paths = {
        entry.relative_path for entry in scope.retired_active_paths
    }
    retained_retired_paths = sorted(retired_active_paths.intersection(active_paths))
    assert retained_retired_paths == [], (
        f"retired active path remains: {retained_retired_paths[:1]}"
    )
    for entry in scope.retired_active_paths:
        baseline_raw = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{legacy.baseline_commit}:{entry.relative_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(baseline_raw).hexdigest() == entry.byte_sha256
    pattern_reappearances = sorted(
        path
        for path in active_paths
        if any(path_matches_glob(path, pattern) for pattern in legacy.path_seed_patterns)
    )
    assert pattern_reappearances == [], (
        f"frozen executable path seed reappeared: {pattern_reappearances[:1]}"
    )

    stable_module_namespaces = {
        entry.module
        for entry in legacy.entries
        if entry.module is not None
        and not (root / entry.relative_path).exists()
        and (root / Path(entry.relative_path).with_suffix("") / "__init__.py").is_file()
    }
    assert len(stable_module_namespaces) == 1
    forbidden_prefixes = tuple(
        prefix
        for prefix in legacy.module_seed_prefixes
        if prefix not in stable_module_namespaces
    )
    retired_modules = {
        named[0]
        for entry in scope.retired_active_paths
        if (named := module_name_for_path(entry.relative_path)) is not None
    }
    forbidden_import_prefixes = tuple(sorted({*forbidden_prefixes, *retired_modules}))

    import_violations: list[tuple[str, str]] = []
    for path in active_paths:
        if not path.endswith(".py"):
            continue
        for imported in _resolved_imports(path, (root / path).read_bytes()):
            if any(
                imported == prefix or imported.startswith(prefix)
                for prefix in forbidden_import_prefixes
            ):
                import_violations.append((path, imported))
    assert import_violations == [], f"retired import remains: {import_violations[:1]}"

    forbidden_modules = sorted(
        retired_modules
        | {
            entry.module
            for entry in legacy.entries
            if entry.module is not None
            and not any(
                entry.module == namespace or entry.module.startswith(namespace + ".")
                for namespace in stable_module_namespaces
            )
        }
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, json, sys\n"
                "resolved = []\n"
                "for module in json.loads(sys.argv[1]):\n"
                "    try:\n"
                "        spec = importlib.util.find_spec(module)\n"
                "    except (ImportError, ModuleNotFoundError, AttributeError):\n"
                "        spec = None\n"
                "    if spec is not None:\n"
                "        resolved.append(module)\n"
                "print(json.dumps(resolved))\n"
            ),
            json.dumps(forbidden_modules),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    importable = json.loads(probe.stdout)
    assert importable == [], f"retired Python import still resolves: {importable[:1]}"

    collision_entry = next(
        entry
        for entry in legacy.entries
        if entry.module in stable_module_namespaces
    )
    baseline = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{legacy.baseline_commit}:{collision_entry.relative_path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    old_tree = ast.parse(baseline.decode("utf-8"), filename=collision_entry.relative_path)
    old_public_types = sorted(
        node.name
        for node in old_tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )
    collision_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib,json,sys;"
                "mod=importlib.import_module(sys.argv[1]);"
                "names=json.loads(sys.argv[2]);"
                "print(json.dumps([name for name in names if hasattr(mod,name)]))"
            ),
            collision_entry.module,
            json.dumps(old_public_types),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    retained_old_types = json.loads(collision_probe.stdout)
    assert retained_old_types == [], (
        f"stable replacement re-exports retired protocol types: {retained_old_types[:1]}"
    )

    removed_console_names = {
        value.split()[0] for value in legacy.removed_entrypoint_tokens
    }
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    console_scripts = pyproject.get("project", {}).get("scripts", {})
    assert type(console_scripts) is dict
    assert removed_console_names.isdisjoint(console_scripts)

    exact_text_tokens = {
        *frozen_paths,
        *retired_active_paths,
        *retired_modules,
        *(
            entry.module
            for entry in legacy.entries
            if entry.module is not None and entry.module not in stable_module_namespaces
        ),
        *legacy.removed_entrypoint_tokens,
        *legacy.runtime_seed_patterns,
    }
    text_suffixes = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
    text_violations: list[tuple[str, str]] = []
    for path in active_paths:
        if Path(path).suffix.lower() not in text_suffixes:
            continue
        if path.startswith("tests/") and path.endswith(".py"):
            continue
        text = (root / path).read_text(encoding="utf-8")
        for token in exact_text_tokens:
            boundary = rf"(?<![A-Za-z0-9_./-]){re.escape(token)}(?![A-Za-z0-9_./-])"
            if re.search(boundary, text):
                text_violations.append((path, token))
                break
    assert text_violations == [], (
        f"retired executable identity remains outside custody: {text_violations[:1]}"
    )


def test_double_resolution_is_byte_identical_and_archive_bytes_are_not_traversed(
    tmp_path: Path,
) -> None:
    rules_path, tracked = _closed_workspace(tmp_path)
    first = resolve_unified_cutover_inventory(
        tmp_path,
        created_at=CREATED_AT,
        rules_path=rules_path,
        tracked_paths=tracked,
    )
    second = resolve_unified_cutover_inventory(
        tmp_path,
        created_at=CREATED_AT,
        rules_path=rules_path,
        tracked_paths=tracked,
    )
    assert first.raw == second.raw
    assert first.document == second.document
    payload = first.document["payload"]
    files = {row["relative_path"]: row for row in payload["files"]}
    custody_root = exact_runtime_root(
        classification="CUSTODY_ONLY", traversal="INVENTORY_ONLY"
    )
    shadow_root = exact_runtime_root(
        classification="NON_AUTHORITY_SHADOW", traversal="POINTER_CLOSURE"
    )
    archive_path = f"{custody_root}/cutover-test/authority_closure/historical.json"
    pointer_path = f"{shadow_root}/_active.json"
    object_path = f"{shadow_root}/object.json"
    assert files[archive_path]["classification"] == "CUSTODY_ONLY"
    assert (
        files[archive_path]["classification_reason"]
        == "CUSTODY_EXCEPTION_NO_TRAVERSAL"
    )
    assert files[pointer_path]["classification"] == "NON_AUTHORITY_SHADOW"
    assert files[object_path]["classification"] == "NON_AUTHORITY_SHADOW"
    assert payload["bootstrap_decision_ref"]["byte_sha256"] == (
        BOOTSTRAP_DECISION_BYTE_SHA256
    )
    assert payload["summary"]["blocker_codes"] == []


@pytest.mark.parametrize(
    ("pointer_raw", "expected_code"),
    [
        (b"not-json", UNPARSEABLE_JSON),
        (
            canonical_json_bytes(
                {
                    "object_byte_sha256": "0" * 64,
                    "object_path": "missing.json",
                }
            ),
            BROKEN_REFERENCE,
        ),
    ],
)
def test_unparseable_or_broken_reachable_pointer_fails_closed(
    tmp_path: Path, pointer_raw: bytes, expected_code: str
) -> None:
    shadow_root = exact_runtime_root(
        classification="NON_AUTHORITY_SHADOW", traversal="POINTER_CLOSURE"
    )
    pointer = tmp_path / shadow_root / "_active.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(pointer_raw)
    rules_path, tracked = make_test_workspace(
        tmp_path,
        source_files={"src/main.py": b"VALUE = 1\n"},
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )
    with pytest.raises(UnifiedCutoverError) as exc:
        resolve_unified_cutover_inventory(
            tmp_path,
            created_at=CREATED_AT,
            rules_path=rules_path,
            tracked_paths=tracked,
        )
    assert exc.value.code == expected_code


def test_symlink_unclassified_and_classification_collision_are_hard_stops(
    tmp_path: Path,
) -> None:
    symlink_root = tmp_path / "symlink"
    shadow_root = exact_runtime_root(
        classification="NON_AUTHORITY_SHADOW", traversal="POINTER_CLOSURE"
    )
    real = symlink_root / shadow_root / "real.json"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"{}")
    (real.parent / "_active.json").symlink_to(real.name)
    rules_path, tracked = make_test_workspace(
        symlink_root,
        source_files={"src/main.py": b"VALUE = 1\n"},
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )
    with pytest.raises(UnifiedCutoverError) as symlink:
        resolve_unified_cutover_inventory(
            symlink_root,
            created_at=CREATED_AT,
            rules_path=rules_path,
            tracked_paths=tracked,
        )
    assert symlink.value.code == SYMLINK_REFUSED

    unclassified_root = tmp_path / "unclassified"
    rules_path, tracked = make_test_workspace(
        unclassified_root,
        source_files={
            "src/main.py": b"VALUE = 1\n",
            "src/orphan.py": b"VALUE = 2\n",
        },
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )
    with pytest.raises(UnifiedCutoverError) as unclassified:
        resolve_unified_cutover_inventory(
            unclassified_root,
            created_at=CREATED_AT,
            rules_path=rules_path,
            tracked_paths=tracked,
        )
    assert unclassified.value.code == UNCLASSIFIED_PATH

    collision_root = tmp_path / "collision"
    fallbacks = {
        "ACTIVE_AUTHORITY": ["src/orphan.py"],
        "ACTIVE_CALLER": [],
        "CUSTODY_ONLY": [],
        "INDEPENDENT_SOURCE": [],
        "LEGACY_INACTIVE": ["src/orphan.py"],
        "NON_AUTHORITY_SHADOW": [],
    }
    rules_path, tracked = make_test_workspace(
        collision_root,
        source_files={
            "src/main.py": b"VALUE = 1\n",
            "src/orphan.py": b"VALUE = 2\n",
        },
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
        classification_fallbacks=fallbacks,
    )
    with pytest.raises(UnifiedCutoverError) as collision:
        resolve_unified_cutover_inventory(
            collision_root,
            created_at=CREATED_AT,
            rules_path=rules_path,
            tracked_paths=tracked,
        )
    assert collision.value.code == CLASSIFICATION_COLLISION


def test_replacement_map_rejects_unresolved_rows(tmp_path: Path) -> None:
    rules_path, _tracked = make_test_workspace(
        tmp_path,
        source_files={"src/main.py": b"VALUE = 1\n"},
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )
    rules = load_rules(tmp_path, rules_path).rules
    path = tmp_path / rules.replacement_test_map
    document = parse_canonical_json_bytes(path.read_bytes())
    broken = deepcopy(document)
    unresolved = next(
        row
        for row in broken["entries"]
        if row["disposition"] == "BEHAVIOR_INTENTIONALLY_REMOVED"
    )
    unresolved["disposition"] = "REPLACED"
    body = {
        "baseline_commit": broken["baseline_commit"],
        "baseline_tree": broken["baseline_tree"],
        "baseline_test_set_sha256": broken["baseline_test_set_sha256"],
        "entries": broken["entries"],
        "legacy_test_seed_patterns": broken["legacy_test_seed_patterns"],
    }
    broken["map_sha256"] = sha(canonical_json_bytes(body))
    write_canonical(path, broken)
    with pytest.raises(UnifiedCutoverError) as exc:
        load_replacement_test_map(tmp_path, rules.replacement_test_map)
    assert exc.value.code == REPLACEMENT_TEST_MAP_INVALID
