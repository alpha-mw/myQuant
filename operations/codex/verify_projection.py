#!/usr/bin/env python3
"""Verify the inert Codex skill and automation deployment projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tomllib
from typing import Any, Final

MANIFEST_RELATIVE: Final = Path("operations/codex/projection-manifest.json")
LEGACY_SEED_RELATIVE: Final = Path("docs/migrations/unified-cutover/legacy-seed-manifest.json")
LEGACY_SEED_KIND: Final = "system.migration.legacy_seed_manifest"
LEGACY_BASELINE_COMMIT: Final = "8612ce51d8cb2b13076af60ac059a921dc104129"
LEGACY_BASELINE_TREE: Final = "475efa1db72501a94691b3719d06b85d17c53b57"
SKILL_SOURCE_ROOT: Final = "operations/codex/skills/myquant"
SKILL_INSTALLED_ROOT: Final = "/Users/maxwell/.codex/skills/myquant"
AUTOMATION_SOURCE: Final = "operations/codex/automations/myquant-2/automation.toml"
AUTOMATION_TARGET: Final = "/Users/maxwell/.codex/automations/myquant-2/automation.toml"
REQUIRED_FILE_MODE: Final = "0644"

SKILL_RELATIVE_PATHS: Final = (
    "SKILL.md",
    "agents/openai.yaml",
    "references/investment-research-and-portfolio.md",
    "references/operations-and-verification.md",
)
AUTOMATION_INCLUDED_FIELDS: Final = (
    "version",
    "id",
    "kind",
    "name",
    "prompt",
    "rrule",
    "model",
    "reasoning_effort",
    "execution_environment",
    "target",
    "cwds",
)
AUTOMATION_EXCLUDED_FIELDS: Final = ("status", "created_at", "updated_at")


class ProjectionVerificationError(ValueError):
    """Raised when repository deployment bytes do not match the manifest."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_relative(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise ProjectionVerificationError(f"{label} must be relative text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProjectionVerificationError(f"{label} is not a canonical relative path")
    return value


def _read_regular(path: Path, *, expected_mode: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProjectionVerificationError(f"missing projection file: {path}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ProjectionVerificationError(f"projection symlink rejected: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise ProjectionVerificationError(f"projection path is not a regular file: {path}")
    observed_mode = f"{stat.S_IMODE(before.st_mode):04o}"
    if observed_mode != expected_mode:
        raise ProjectionVerificationError(f"projection mode mismatch for {path}: {observed_mode}")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ProjectionVerificationError(f"projection file is unreadable: {path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ProjectionVerificationError(f"projection file changed while read: {path}")
    return raw


def _walk_regular_tree(root: Path) -> list[tuple[str, str, bytes]]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ProjectionVerificationError("skill source root is missing") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ProjectionVerificationError("skill source root must be a real directory")

    rows: list[tuple[str, str, bytes]] = []
    stack = [root]
    while stack:
        parent = stack.pop()
        try:
            with os.scandir(parent) as iterator:
                entries = sorted(iterator, key=lambda row: row.name.encode("utf-8"))
        except OSError as exc:
            raise ProjectionVerificationError("skill tree cannot be enumerated") from exc
        directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ProjectionVerificationError(f"skill entry is unreadable: {path}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ProjectionVerificationError(f"skill symlink rejected: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ProjectionVerificationError(f"non-regular skill entry rejected: {path}")
            mode = f"{stat.S_IMODE(metadata.st_mode):04o}"
            relative = path.relative_to(root).as_posix()
            raw = _read_regular(path, expected_mode=mode)
            rows.append((relative, mode, raw))
        stack.extend(reversed(directories))
    return sorted(rows, key=lambda row: row[0].encode("utf-8"))


def _load_canonical_json(path: Path, *, expected_mode: str = REQUIRED_FILE_MODE) -> dict[str, Any]:
    raw = _read_regular(path, expected_mode=expected_mode)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionVerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ProjectionVerificationError(f"JSON is not exact compact canonical bytes: {path}")
    return value


def _load_removed_entrypoint_tokens(repository_root: Path) -> tuple[str, ...]:
    manifest = _load_canonical_json(repository_root / LEGACY_SEED_RELATIVE)
    expected_fields = (
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
    )
    if set(manifest) != set(expected_fields):
        raise ProjectionVerificationError("legacy seed manifest fields are not exact")
    expected_contract_sha256 = _sha256(
        canonical_json_bytes(
            {
                "field_names": sorted(expected_fields),
                "kind": LEGACY_SEED_KIND,
                "strict_fields": True,
            }
        )
    )
    if (
        manifest["kind"] != LEGACY_SEED_KIND
        or manifest["contract_sha256"] != expected_contract_sha256
    ):
        raise ProjectionVerificationError("legacy seed manifest contract hash mismatch")
    if (
        manifest["baseline_commit"] != LEGACY_BASELINE_COMMIT
        or manifest["baseline_tree"] != LEGACY_BASELINE_TREE
    ):
        raise ProjectionVerificationError("legacy seed baseline identity mismatch")
    seed_fields = {
        key: manifest[key]
        for key in (
            "baseline_commit",
            "baseline_tree",
            "entries",
            "module_seed_prefixes",
            "path_seed_patterns",
            "removed_entrypoint_tokens",
            "runtime_seed_patterns",
        )
    }
    if _sha256(canonical_json_bytes(seed_fields)) != manifest["seed_set_sha256"]:
        raise ProjectionVerificationError("legacy seed set hash mismatch")
    tokens = manifest["removed_entrypoint_tokens"]
    if (
        type(tokens) is not list
        or not tokens
        or any(type(value) is not str or not value for value in tokens)
        or tokens != sorted(set(tokens), key=lambda value: value.encode("utf-8"))
    ):
        raise ProjectionVerificationError("removed entrypoint tokens are not canonical")
    return tuple(tokens)


def reject_removed_entrypoint_tokens(
    documents: Mapping[str, str], removed_tokens: Sequence[str]
) -> None:
    for label, text in documents.items():
        for token in removed_tokens:
            if token in text:
                raise ProjectionVerificationError(
                    f"removed entrypoint token occurs in deployment source: {label}"
                )


def validate_automation_projection(
    document: Mapping[str, Any],
    *,
    included_fields: Sequence[str] = AUTOMATION_INCLUDED_FIELDS,
    excluded_fields: Sequence[str] = AUTOMATION_EXCLUDED_FIELDS,
) -> dict[str, Any]:
    if type(document) is not dict:
        raise ProjectionVerificationError("automation TOML must decode to one table")
    if tuple(included_fields) != AUTOMATION_INCLUDED_FIELDS:
        raise ProjectionVerificationError("automation included fields are not exact")
    if tuple(excluded_fields) != AUTOMATION_EXCLUDED_FIELDS:
        raise ProjectionVerificationError("automation excluded fields are not exact")
    if set(document) != set(included_fields) | set(excluded_fields):
        raise ProjectionVerificationError("automation field set is not exact")
    expected_values = {
        "version": 1,
        "id": "myquant-2",
        "kind": "cron",
        "name": "myQuant 统一因子周度健康复核",
        "rrule": "RRULE:FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=30;BYSECOND=0",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "execution_environment": "local",
        "target": {
            "type": "project",
            "project_id": "local-f9bd22cbbeff01393d5315615e57cb9a",
        },
        "cwds": ["/Users/maxwell/mySpace/myQuant"],
    }
    for field, expected in expected_values.items():
        if document[field] != expected:
            raise ProjectionVerificationError(f"automation {field} mismatch")
    prompt = document["prompt"]
    if type(prompt) is not str or not prompt:
        raise ProjectionVerificationError("automation prompt must be non-empty text")
    if document["status"] != "PAUSED":
        raise ProjectionVerificationError("automation status must remain PAUSED")
    for field in ("created_at", "updated_at"):
        if type(document[field]) is not int or document[field] <= 0:
            raise ProjectionVerificationError(f"automation {field} must be a positive integer")
    return {field: document[field] for field in included_fields}


def _verify_skill_tree(repository_root: Path, skill_manifest: Mapping[str, Any]) -> dict[str, str]:
    expected_fields = {
        "source_root",
        "installed_root",
        "skill_tree_identity_sha256",
        "rules",
        "files",
    }
    if type(skill_manifest) is not dict or set(skill_manifest) != expected_fields:
        raise ProjectionVerificationError("skill manifest fields are not exact")
    if skill_manifest["source_root"] != SKILL_SOURCE_ROOT:
        raise ProjectionVerificationError("skill source root mismatch")
    if skill_manifest["installed_root"] != SKILL_INSTALLED_ROOT:
        raise ProjectionVerificationError("skill installed root mismatch")
    expected_rules = {
        "allowed_type": "regular_file",
        "copy_mode": "byte_for_byte",
        "follow_symlinks": False,
        "identity_fields": ["relative_path", "mode", "byte_sha256"],
        "path_order": "repository_relative_utf8_lexicographic",
        "reject_extra_files": True,
        "replace_destination_tree": True,
    }
    if skill_manifest["rules"] != expected_rules:
        raise ProjectionVerificationError("skill tree rules mismatch")

    root = repository_root / SKILL_SOURCE_ROOT
    observed = _walk_regular_tree(root)
    if [row[0] for row in observed] != list(SKILL_RELATIVE_PATHS):
        raise ProjectionVerificationError("skill tree has missing or extra files")
    file_rows = skill_manifest["files"]
    if type(file_rows) is not list or len(file_rows) != len(observed):
        raise ProjectionVerificationError("skill manifest file count mismatch")

    identity_rows: list[dict[str, str]] = []
    text_documents: dict[str, str] = {}
    for index, (relative, mode, raw) in enumerate(observed):
        row = file_rows[index]
        expected_row_fields = {
            "relative_path",
            "source",
            "installed_target",
            "mode",
            "byte_sha256",
        }
        if type(row) is not dict or set(row) != expected_row_fields:
            raise ProjectionVerificationError("skill file row fields are not exact")
        _canonical_relative(row["relative_path"], label="skill relative_path")
        source = f"{SKILL_SOURCE_ROOT}/{relative}"
        installed_target = f"{SKILL_INSTALLED_ROOT}/{relative}"
        digest = _sha256(raw)
        if mode != REQUIRED_FILE_MODE:
            raise ProjectionVerificationError(f"skill file mode drift: {relative}")
        expected_row = {
            "relative_path": relative,
            "source": source,
            "installed_target": installed_target,
            "mode": mode,
            "byte_sha256": digest,
        }
        if row != expected_row:
            raise ProjectionVerificationError(f"skill file mapping mismatch: {relative}")
        identity_rows.append({"relative_path": relative, "mode": mode, "byte_sha256": digest})
        try:
            text_documents[source] = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProjectionVerificationError(f"skill file is not UTF-8: {relative}") from exc

    observed_identity = _sha256(canonical_json_bytes(identity_rows))
    if observed_identity != skill_manifest["skill_tree_identity_sha256"]:
        raise ProjectionVerificationError("skill tree identity hash mismatch")
    return text_documents


def _verify_automation(
    repository_root: Path, automation_manifest: Mapping[str, Any]
) -> dict[str, str]:
    expected_fields = {
        "source",
        "installed_target",
        "mode",
        "byte_sha256",
        "semantic_projection_sha256",
        "included_fields",
        "excluded_fields",
        "required_status",
        "activation_performed",
    }
    if type(automation_manifest) is not dict or set(automation_manifest) != expected_fields:
        raise ProjectionVerificationError("automation manifest fields are not exact")
    if automation_manifest["source"] != AUTOMATION_SOURCE:
        raise ProjectionVerificationError("automation source mismatch")
    if automation_manifest["installed_target"] != AUTOMATION_TARGET:
        raise ProjectionVerificationError("automation installed target mismatch")
    if automation_manifest["mode"] != REQUIRED_FILE_MODE:
        raise ProjectionVerificationError("automation manifest mode mismatch")
    if automation_manifest["required_status"] != "PAUSED":
        raise ProjectionVerificationError("automation required status mismatch")
    if automation_manifest["activation_performed"] is not False:
        raise ProjectionVerificationError("automation must remain inactive")

    path = repository_root / AUTOMATION_SOURCE
    raw = _read_regular(path, expected_mode=REQUIRED_FILE_MODE)
    if _sha256(raw) != automation_manifest["byte_sha256"]:
        raise ProjectionVerificationError("automation byte hash mismatch")
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectionVerificationError("automation TOML is invalid") from exc
    projection = validate_automation_projection(
        document,
        included_fields=automation_manifest["included_fields"],
        excluded_fields=automation_manifest["excluded_fields"],
    )
    if (
        _sha256(canonical_json_bytes(projection))
        != automation_manifest["semantic_projection_sha256"]
    ):
        raise ProjectionVerificationError("automation semantic projection hash mismatch")
    return {AUTOMATION_SOURCE: raw.decode("utf-8")}


def verify_projection(repository_root: Path | None = None) -> dict[str, Any]:
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    manifest = _load_canonical_json(root / MANIFEST_RELATIVE)
    expected_fields = {
        "kind",
        "contract_sha256",
        "external_deployment_performed",
        "skill_tree",
        "automation",
    }
    if set(manifest) != expected_fields:
        raise ProjectionVerificationError("projection manifest fields are not exact")
    if manifest["kind"] != "codex-external-deployment-projection":
        raise ProjectionVerificationError("projection manifest kind mismatch")
    if manifest["external_deployment_performed"] is not False:
        raise ProjectionVerificationError("external deployment must remain unperformed")
    contract = dict(manifest)
    expected_contract_sha256 = contract.pop("contract_sha256")
    if _sha256(canonical_json_bytes(contract)) != expected_contract_sha256:
        raise ProjectionVerificationError("projection contract hash mismatch")

    deployment_text = _verify_skill_tree(root, manifest["skill_tree"])
    deployment_text.update(_verify_automation(root, manifest["automation"]))
    readme_path = root / "operations/codex/README.md"
    deployment_text["operations/codex/README.md"] = _read_regular(
        readme_path, expected_mode=REQUIRED_FILE_MODE
    ).decode("utf-8")
    removed_tokens = _load_removed_entrypoint_tokens(root)
    reject_removed_entrypoint_tokens(deployment_text, removed_tokens)

    manifest_readback = _read_regular(
        root / MANIFEST_RELATIVE,
        expected_mode=REQUIRED_FILE_MODE,
    )
    if manifest_readback != canonical_json_bytes(manifest):
        raise ProjectionVerificationError("projection manifest changed during verification")

    return {
        "kind": "codex-projection-verification",
        "verified": True,
        "contract_sha256": expected_contract_sha256,
        "manifest_byte_sha256": _sha256(manifest_readback),
        "skill_tree_identity_sha256": manifest["skill_tree"]["skill_tree_identity_sha256"],
        "automation_semantic_projection_sha256": manifest["automation"][
            "semantic_projection_sha256"
        ],
        "automation_status": "PAUSED",
        "external_deployment_performed": False,
        "activation_performed": False,
    }


def main() -> int:
    try:
        result = verify_projection()
    except ProjectionVerificationError as exc:
        print(f"CODEX_PROJECTION_BLOCKED:{exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
