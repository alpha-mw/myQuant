from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.migration.rules import (
    LEGACY_CUSTODY_SCOPE_RELATIVE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_canonical(path: Path, value: Any) -> bytes:
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def make_test_workspace(
    root: Path,
    *,
    source_files: dict[str, bytes],
    entrypoint_seeds: list[dict[str, str]],
    shadow_seeds: list[dict[str, str]] | None = None,
    classification_fallbacks: dict[str, list[str]] | None = None,
    json_reference_keys: list[str] | None = None,
) -> tuple[Path, tuple[str, ...]]:
    template = parse_canonical_json_bytes(
        (REPO_ROOT / "operations/unified_cutover/rules.json").read_bytes()
    )
    rules = deepcopy(template)
    rules["tracked_roots"] = [
        {"path": "src", "required": True, "classification": None}
    ]
    rules["external_roots"] = []
    rules["entrypoint_seeds"] = entrypoint_seeds
    rules["shadow_seeds"] = shadow_seeds or []
    rules["classification_fallbacks"] = classification_fallbacks or {
        "ACTIVE_AUTHORITY": [],
        "ACTIVE_CALLER": [],
        "CUSTODY_ONLY": [],
        "INDEPENDENT_SOURCE": [],
        "LEGACY_INACTIVE": [],
        "NON_AUTHORITY_SHADOW": [],
    }
    rules["json_reference_keys"] = json_reference_keys or ["object_path"]

    for relative, raw in source_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    for relative in (
        rules["bootstrap_decision"],
        rules["dynamic_import_allowlist"],
        rules["legacy_custody_scope"],
        rules["legacy_seed_manifest"],
        rules["replacement_test_map"],
        rules["source_to_target_table"],
    ):
        source = REPO_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    rules_path = root / "operations/unified_cutover/rules.json"
    write_canonical(rules_path, rules)
    return rules_path, tuple(sorted(source_files))


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_runtime_root(*, classification: str, traversal: str) -> str:
    scope = parse_canonical_json_bytes(
        (REPO_ROOT / LEGACY_CUSTODY_SCOPE_RELATIVE_PATH).read_bytes()
    )
    matches = [
        row["path"]
        for row in scope["runtime_roots"]
        if row["classification"] == classification and row["traversal"] == traversal
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one exact runtime root for {classification}/{traversal}"
        )
    return matches[0]
