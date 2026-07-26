#!/usr/bin/env python3
"""Run the Phase 0 diff check through a fresh isolated Git index.

The helper never imports the evidence-index CLI.  It carries the same closed
Phase 0 path registry and source-binding algorithm, verifies that registry
against the checked index source with ``ast``, and routes every mutating Git
operation to a new work-root index.  The repository's real index, refs, and
object database are guarded before and after execution.  Only the final real
``git diff --check`` result is returned to the session receipt producer.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
ALTERNATE_INDEX_NAME = "phase0-alternate.index"
MAX_GIT_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_PHASE0_PATHS = 4096
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
GIT_VERSION_RE = re.compile(r"^git version \S+(?: .*)?$", re.ASCII)
SOURCE_BINDING_KEYS = {
    "base_commit",
    "binary_diff_sha256",
    "porcelain_sha256",
    "source_state_sha256",
    "untracked_inventory_sha256",
}

PHASE0_ALLOWED_PATTERN_REGISTRY = frozenset(
    {
        "docs/architecture/v17_v2_contract.md",
        "docs/runbooks/v17_shadow_operations.md",
        "quant_investor/v17_v2_contract/**",
        "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
        "scripts/schemas/v17_offline_dependency_evidence.v2.schema.json",
        "scripts/schemas/v17_phase0_command_receipt.v2.schema.json",
        "scripts/schemas/v17_phase0_evidence_index.v2.schema.json",
        "scripts/schemas/v17_phase0_gate_manifest.v2.schema.json",
        "scripts/schemas/v17_phase0_hash_freeze.v2.schema.json",
        "scripts/schemas/v17_phase0_main_suite_receipt.v1.schema.json",
        "scripts/schemas/v17_phase0_package_evidence.v2.schema.json",
        "scripts/schemas/v17_phase0_pre_existing_classification.v2.schema.json",
        "scripts/schemas/v17_phase0_session.v2.schema.json",
        "scripts/schemas/v17_phase0_skip_baseline.v2.schema.json",
        "scripts/schemas/v17_phase0_unpublished_failure.v2.schema.json",
        "scripts/v17_offline_dependency_evidence.py",
        "scripts/v17_phase0_diff_check.py",
        "scripts/v17_phase0_evidence_index.py",
        "scripts/v17_phase0_evidence_session.py",
        "scripts/v17_phase0_main_suite_harness.py",
        "scripts/v17_phase0_main_suite_wrapper.py",
        "scripts/v17_phase0_package_evidence.py",
        "scripts/v17_phase0_skip_baseline.py",
        "tests/unit/test_v17_offline_dependency_evidence.py",
        "tests/unit/test_v17_phase0_diff_check.py",
        "tests/unit/test_v17_phase0_evidence_index.py",
        "tests/unit/test_v17_phase0_main_suite_contract.py",
        "tests/unit/test_v17_phase0_evidence_session.py",
        "tests/unit/test_v17_phase0_package_evidence.py",
        "tests/unit/test_v17_phase0_skip_baseline.py",
        "tests/unit/test_v17_v2_*.py",
    }
)
FORBIDDEN_V16_ROOTS = (
    PurePosixPath("results/v16"),
    PurePosixPath("results/v16_operator_advisory"),
)


class DiffCheckError(RuntimeError):
    """Raised when isolated diff execution cannot prove repository integrity."""

    exit_code = 2


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DiffCheckError("value is not canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise DiffCheckError(f"{label} component is unavailable: {current}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise DiffCheckError(f"{label} cannot contain symlink components")


def _resolve_existing_directory(
    path: Path,
    *,
    label: str,
    owner_private: bool,
) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise DiffCheckError(f"{label} must be a normalized absolute path")
    _assert_no_symlink_components(path, label=label)
    try:
        observed = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DiffCheckError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(observed.st_mode) or resolved != path:
        raise DiffCheckError(f"{label} must be a concrete directory")
    if observed.st_uid != os.getuid():
        raise DiffCheckError(f"{label} must be owned by the current user")
    if owner_private and stat.S_IMODE(observed.st_mode) != 0o700:
        raise DiffCheckError(f"{label} must have mode 0700")
    return resolved


def _require_fresh_work_root(path: Path, *, repo_root: Path) -> Path:
    work = _resolve_existing_directory(path, label="diff work root", owner_private=True)
    if _path_within(work, repo_root) or _path_within(repo_root, work):
        raise DiffCheckError("diff work root cannot overlap the repository")
    for forbidden in FORBIDDEN_V16_ROOTS:
        protected = repo_root / Path(*forbidden.parts)
        if _path_within(work, protected) or _path_within(protected, work):
            raise DiffCheckError("diff work root cannot overlap protected v16 roots")
    try:
        names = tuple(entry.name for entry in os.scandir(work))
    except OSError as exc:
        raise DiffCheckError("diff work root cannot be listed") from exc
    if names:
        raise DiffCheckError("diff work root must be fresh and empty")
    return work


def _make_private_directory(path: Path, *, parent: Path, label: str) -> Path:
    if path.parent != parent or path.exists():
        raise DiffCheckError(f"{label} must be a new direct child")
    try:
        path.mkdir(mode=0o700)
        observed = path.lstat()
    except OSError as exc:
        raise DiffCheckError(f"{label} cannot be created") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_uid != os.getuid()
    ):
        raise DiffCheckError(f"{label} must be owner-private 0700")
    return path


def _base_git_environment(*, home: Path | None = None, tmp: Path | None = None) -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": str(home) if home is not None else "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp) if tmp is not None else "/private/tmp",
    }


def _execute(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise DiffCheckError(f"command could not start: {argv[0]}") from exc
    if len(completed.stdout) + len(completed.stderr) > MAX_GIT_CAPTURE_BYTES:
        raise DiffCheckError("Git command output exceeds the capture limit")
    return completed.returncode, completed.stdout, completed.stderr


def _git_bytes(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    env = _base_git_environment() if environment is None else dict(environment)
    returncode, stdout, _stderr = _execute(argv, cwd=cwd, environment=env)
    if returncode != 0:
        raise DiffCheckError(f"Git preflight command failed: {list(argv)!r}")
    return stdout


def _resolve_repo_root(repo_root: Path) -> Path:
    repo = _resolve_existing_directory(repo_root, label="repo root", owner_private=False)
    raw = _git_bytes(("git", "rev-parse", "--show-toplevel"), cwd=repo)
    try:
        top = Path(raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeError, OSError) as exc:
        raise DiffCheckError("git returned an invalid top-level path") from exc
    if top != repo:
        raise DiffCheckError("repo root must be the exact Git worktree top level")
    return repo


def _repo_relative_path(value: str, *, label: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DiffCheckError(f"{label} must be safe repository-relative POSIX")
    for forbidden in FORBIDDEN_V16_ROOTS:
        if pure == forbidden or forbidden in pure.parents:
            raise DiffCheckError(f"{label} enters a protected v16 root")
    return value


def _decode_nul_paths(raw: bytes, *, label: str) -> list[str]:
    if raw and not raw.endswith(b"\0"):
        raise DiffCheckError(f"{label} is not NUL terminated")
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            decoded = item.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise DiffCheckError(f"{label} contains a non-UTF-8 path") from exc
        paths.append(_repo_relative_path(decoded, label=label))
    if len(paths) != len(set(paths)) or len({path.casefold() for path in paths}) != len(paths):
        raise DiffCheckError(f"{label} contains duplicate/casefold-colliding paths")
    return paths


def _glob_regex(pattern: str) -> re.Pattern[str]:
    fragments = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                fragments.append(".*")
                index += 2
                continue
            fragments.append("[^/]*")
        elif character == "?":
            fragments.append("[^/]")
        else:
            fragments.append(re.escape(character))
        index += 1
    fragments.append("$")
    return re.compile("".join(fragments), re.ASCII)


_PHASE0_PATTERN_REGEXES = tuple(
    (pattern, _glob_regex(pattern))
    for pattern in sorted(PHASE0_ALLOWED_PATTERN_REGISTRY, key=lambda item: item.encode("utf-8"))
)


def _phase0_path_allowed(path: str) -> bool:
    return any(regex.fullmatch(path) is not None for _pattern, regex in _PHASE0_PATTERN_REGEXES)


def _registry_from_index_source(repo_root: Path) -> frozenset[str]:
    path = repo_root / "scripts" / "v17_phase0_evidence_index.py"
    raw, _observed = _stable_regular_bytes(path, label="Phase 0 evidence-index source")
    try:
        tree = ast.parse(raw.decode("utf-8", errors="strict"), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise DiffCheckError("cannot parse Phase 0 evidence-index source") from exc
    assignments = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "PHASE0_ALLOWED_PATTERN_REGISTRY"
            for target in targets
        ):
            continue
        assignments.append(node.value)
    if len(assignments) != 1:
        raise DiffCheckError("evidence-index registry assignment is not unique")
    expression = assignments[0]
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "frozenset"
        or len(expression.args) != 1
        or expression.keywords
    ):
        raise DiffCheckError("evidence-index registry must be one literal frozenset")
    try:
        value = ast.literal_eval(expression.args[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise DiffCheckError("evidence-index registry is not literal") from exc
    if not isinstance(value, (set, frozenset)) or not all(isinstance(item, str) for item in value):
        raise DiffCheckError("evidence-index registry contains invalid values")
    return frozenset(value)


def _assert_registry_matches_index(repo_root: Path) -> None:
    if _registry_from_index_source(repo_root) != PHASE0_ALLOWED_PATTERN_REGISTRY:
        raise DiffCheckError("diff helper Phase 0 registry drifted from evidence index")


def _stable_regular_bytes(
    path: Path,
    *,
    label: str,
    size_limit: int = MAX_GIT_CAPTURE_BYTES,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise DiffCheckError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(before.st_mode):
        raise DiffCheckError(f"{label} must be a regular non-symlink file")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > size_limit:
                raise DiffCheckError(f"{label} exceeds the size limit")
            chunks.append(chunk)
        raw = b"".join(chunks)
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        raise DiffCheckError(f"{label} cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise DiffCheckError(f"{label} disappeared during read") from exc
    signature = _stat_signature(before)
    if (
        signature != _stat_signature(opened)
        or signature != _stat_signature(after_fd)
        or signature != _stat_signature(after)
        or len(raw) != before.st_size
    ):
        raise DiffCheckError(f"{label} changed during read")
    return raw, before


def _stable_untracked(repo_root: Path, relative: str) -> dict[str, Any]:
    path = repo_root / Path(*PurePosixPath(relative).parts)
    try:
        before = path.lstat()
    except OSError as exc:
        raise DiffCheckError(f"untracked path disappeared: {relative}") from exc
    base = {
        "mode": _mode_string(before.st_mode),
        "path": relative,
        "size_bytes": before.st_size,
    }
    if stat.S_ISREG(before.st_mode):
        raw, observed = _stable_regular_bytes(path, label=f"untracked file {relative}")
        if _stat_signature(before) != _stat_signature(observed):
            raise DiffCheckError(f"untracked file drift: {relative}")
        return {
            **base,
            "sha256": _sha256(raw),
            "symlink_target": None,
            "type": "file",
        }
    if stat.S_ISLNK(before.st_mode):
        try:
            target_before = os.readlink(path)
            after = path.lstat()
            target_after = os.readlink(path)
            raw_target = target_before.encode("utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise DiffCheckError(f"untracked symlink is invalid: {relative}") from exc
        if _stat_signature(before) != _stat_signature(after) or target_before != target_after:
            raise DiffCheckError(f"untracked symlink drift: {relative}")
        return {
            **base,
            "sha256": _sha256(raw_target),
            "symlink_target": target_before,
            "type": "symlink",
        }
    raise DiffCheckError(f"unsupported untracked node: {relative}")


def _raw_binding(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _source_snapshot(repo_root: Path) -> dict[str, Any]:
    environment = _base_git_environment()
    head_raw = _git_bytes(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=repo_root,
        environment=environment,
    )
    try:
        base_commit = head_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise DiffCheckError("git HEAD is not ASCII") from exc
    if COMMIT_RE.fullmatch(base_commit) is None:
        raise DiffCheckError("git HEAD is not a full object id")
    porcelain = _git_bytes(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        cwd=repo_root,
        environment=environment,
    )
    binary_diff = _git_bytes(
        (
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            base_commit,
            "--",
        ),
        cwd=repo_root,
        environment=environment,
    )
    tracked_raw = _git_bytes(
        (
            "git",
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            base_commit,
            "--",
        ),
        cwd=repo_root,
        environment=environment,
    )
    untracked_raw = _git_bytes(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
        environment=environment,
    )
    tracked = _decode_nul_paths(tracked_raw, label="tracked dirty paths")
    untracked_paths = _decode_nul_paths(untracked_raw, label="untracked inventory")
    dirty_paths = sorted(
        set(tracked).union(untracked_paths),
        key=lambda value: value.encode("utf-8"),
    )
    untracked = [
        _stable_untracked(repo_root, path)
        for path in sorted(untracked_paths, key=lambda value: value.encode("utf-8"))
    ]
    public = {
        "base_commit": base_commit,
        "binary_diff_from_base": _raw_binding(binary_diff),
        "dirty_paths": dirty_paths,
        "porcelain_v1_z": _raw_binding(porcelain),
        "untracked": untracked,
    }
    return {
        **public,
        "source_state_sha256": _sha256(_canonical_bytes(public)),
        "_guards": {
            "binary_diff": binary_diff,
            "head": head_raw,
            "porcelain": porcelain,
            "tracked": tracked_raw,
            "untracked": untracked_raw,
        },
    }


def _public_source_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_guards"}


def _source_binding_from_snapshot(value: Mapping[str, Any]) -> dict[str, str]:
    public = _public_source_state(value)
    return {
        "base_commit": public["base_commit"],
        "binary_diff_sha256": public["binary_diff_from_base"]["sha256"],
        "porcelain_sha256": public["porcelain_v1_z"]["sha256"],
        "source_state_sha256": public["source_state_sha256"],
        "untracked_inventory_sha256": _sha256(_canonical_bytes(public["untracked"])),
    }


def _validate_source_binding(value: Mapping[str, Any]) -> dict[str, str]:
    if type(value) is not dict or set(value) != SOURCE_BINDING_KEYS:
        raise DiffCheckError("expected source binding has invalid keys")
    result: dict[str, str] = {}
    for key in SOURCE_BINDING_KEYS:
        item = value[key]
        if type(item) is not str:
            raise DiffCheckError(f"expected source binding {key} must be a string")
        if key == "base_commit":
            if COMMIT_RE.fullmatch(item) is None:
                raise DiffCheckError("expected source binding base commit is invalid")
        elif SHA256_RE.fullmatch(item) is None:
            raise DiffCheckError(f"expected source binding {key} is invalid")
        result[key] = item
    return result


def _assert_source_snapshot_equal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if _canonical_bytes(_public_source_state(before)) != _canonical_bytes(
        _public_source_state(after)
    ) or before.get("_guards") != after.get("_guards"):
        raise DiffCheckError("repository source state changed during diff check")


def _resolve_git_path(repo_root: Path, name: str) -> tuple[bytes, Path]:
    raw = _git_bytes(("git", "rev-parse", "--git-path", name), cwd=repo_root)
    try:
        text = raw.decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        raise DiffCheckError(f"git returned invalid {name} path") from exc
    if not text:
        raise DiffCheckError(f"git returned empty {name} path")
    lexical = Path(text)
    if not lexical.is_absolute():
        lexical = Path(os.path.abspath(repo_root / lexical))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise DiffCheckError(f"real Git {name} path is unavailable") from exc
    return raw, resolved


def _real_index_snapshot(repo_root: Path) -> dict[str, Any]:
    path_raw, path = _resolve_git_path(repo_root, "index")
    raw, observed = _stable_regular_bytes(path, label="real Git index")
    if observed.st_nlink != 1:
        raise DiffCheckError("real Git index must not be hardlinked")
    return {
        "path_raw": path_raw,
        "path": str(path),
        "sha256": _sha256(raw),
        "stat": list(_stat_signature(observed)),
    }


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def scan(path: Path, relative: str) -> None:
        try:
            observed = path.lstat()
            names_before = tuple(sorted(entry.name for entry in os.scandir(path)))
        except OSError as exc:
            raise DiffCheckError("Git object directory is unstable") from exc
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise DiffCheckError("Git object directory contains invalid directory node")
        rows.append(
            {
                "path": relative,
                "stat": list(_stat_signature(observed)[:6]),
                "type": "directory",
            }
        )
        for name in names_before:
            child = path / name
            child_relative = name if not relative else f"{relative}/{name}"
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise DiffCheckError("Git object inventory changed during scan") from exc
            if stat.S_ISDIR(child_stat.st_mode) and not stat.S_ISLNK(child_stat.st_mode):
                scan(child, child_relative)
            elif stat.S_ISREG(child_stat.st_mode):
                raw, stable_stat = _stable_regular_bytes(
                    child,
                    label=f"Git object {child_relative}",
                )
                rows.append(
                    {
                        "path": child_relative,
                        "sha256": _sha256(raw),
                        "stat": list(_stat_signature(stable_stat)[:7]),
                        "type": "file",
                    }
                )
            else:
                raise DiffCheckError("Git object inventory contains symlink/special node")
        try:
            after = path.lstat()
            names_after = tuple(sorted(entry.name for entry in os.scandir(path)))
        except OSError as exc:
            raise DiffCheckError("Git object directory drifted") from exc
        if _stat_signature(observed) != _stat_signature(after) or names_before != names_after:
            raise DiffCheckError("Git object directory drifted during scan")

    scan(root, "")
    return rows


def _repository_guard(repo_root: Path) -> dict[str, Any]:
    index = _real_index_snapshot(repo_root)
    objects_raw, objects = _resolve_git_path(repo_root, "objects")
    if not objects.is_dir():
        raise DiffCheckError("Git objects path must be a directory")
    environment = _base_git_environment()
    return {
        "count_objects": _git_bytes(
            ("git", "count-objects", "-v"),
            cwd=repo_root,
            environment=environment,
        ),
        "head": _git_bytes(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=repo_root,
            environment=environment,
        ),
        "index": index,
        "object_inventory": _directory_inventory(objects),
        "objects_path": str(objects),
        "objects_path_raw": objects_raw,
        "refs": _git_bytes(
            (
                "git",
                "for-each-ref",
                "--sort=refname",
                "--format=%(refname)%00%(objectname)%00%(symref)%00",
            ),
            cwd=repo_root,
            environment=environment,
        ),
    }


def _phase0_untracked_paths(repo_root: Path) -> list[str]:
    raw = _git_bytes(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo_root,
    )
    all_untracked = _decode_nul_paths(raw, label="Git untracked inventory")
    selected = sorted(
        (path for path in all_untracked if _phase0_path_allowed(path)),
        key=lambda value: value.encode("utf-8"),
    )
    if len(selected) > MAX_PHASE0_PATHS:
        raise DiffCheckError("Phase 0 untracked inventory exceeds the closed limit")
    return selected


def _assert_intent_to_add(
    repo_root: Path,
    *,
    paths: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    if not paths:
        return
    raw = _git_bytes(
        ("git", "ls-files", "--stage", "-z", "--", *paths),
        cwd=repo_root,
        environment=environment,
    )
    observed: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, raw_path = item.split(b"\t", 1)
            _mode, object_name, stage = metadata.decode("ascii", errors="strict").split(" ")
            path = raw_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise DiffCheckError("alternate index stage inventory is malformed") from exc
        if stage != "0" or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", object_name) is None:
            raise DiffCheckError("Phase 0 untracked path is not intent-to-add")
        observed.append(path)
    if observed != list(paths):
        raise DiffCheckError("alternate index does not contain every Phase 0 untracked path")
    diff_names = _decode_nul_paths(
        _git_bytes(
            ("git", "diff", "--name-only", "-z", "--", *paths),
            cwd=repo_root,
            environment=environment,
        ),
        label="alternate-index intent-to-add diff inventory",
    )
    if diff_names != list(paths):
        raise DiffCheckError("Phase 0 untracked paths are not active intent-to-add entries")


def _validated_object_sink_inventory(
    repo_root: Path,
    object_sink: Path,
) -> list[dict[str, Any]]:
    format_raw = _git_bytes(
        ("git", "rev-parse", "--show-object-format"),
        cwd=repo_root,
    )
    try:
        object_format = format_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise DiffCheckError("Git object format is invalid") from exc
    if object_format == "sha1":
        empty_blob = hashlib.sha1(b"blob 0\0").hexdigest()
    elif object_format == "sha256":
        empty_blob = hashlib.sha256(b"blob 0\0").hexdigest()
    else:
        raise DiffCheckError("unsupported Git object format")
    inventory = _directory_inventory(object_sink)
    observed_paths = {(row["path"], row["type"]) for row in inventory if row["path"]}
    allowed_paths = {
        (empty_blob[:2], "directory"),
        (f"{empty_blob[:2]}/{empty_blob[2:]}", "file"),
    }
    if observed_paths not in (set(), allowed_paths):
        raise DiffCheckError("alternate object sink contains more than the empty blob")
    return inventory


def run_isolated_diff_check(
    *,
    repo_root: Path,
    work_root: Path,
    expected_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only the final real ``git diff --check`` command/result.

    Internal Git preparation and integrity probes are deliberately not exposed
    to callers or command receipts.
    """

    repo = _resolve_repo_root(repo_root)
    work = _require_fresh_work_root(work_root, repo_root=repo)
    expected = _validate_source_binding(expected_source_binding)
    _assert_registry_matches_index(repo)
    source_before = _source_snapshot(repo)
    if _source_binding_from_snapshot(source_before) != expected:
        raise DiffCheckError("repository source binding does not match the frozen session")
    guard_before = _repository_guard(repo)
    phase0_paths = _phase0_untracked_paths(repo)

    home = _make_private_directory(work / "home", parent=work, label="diff HOME")
    tmp = _make_private_directory(work / "tmp", parent=work, label="diff TMPDIR")
    object_sink = _make_private_directory(
        work / "objects",
        parent=work,
        label="alternate Git object directory",
    )
    alternate_index = work / ALTERNATE_INDEX_NAME
    if alternate_index.exists():
        raise DiffCheckError("alternate Git index must not pre-exist")
    real_objects = guard_before["objects_path"]
    environment = {
        **_base_git_environment(home=home, tmp=tmp),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": real_objects,
        "GIT_INDEX_FILE": str(alternate_index),
        "GIT_OBJECT_DIRECTORY": str(object_sink),
    }
    read_tree_code, _read_tree_stdout, _read_tree_stderr = _execute(
        ("git", "read-tree", "HEAD"),
        cwd=repo,
        environment=environment,
    )
    if read_tree_code != 0:
        raise DiffCheckError("git read-tree HEAD failed for alternate index")
    if phase0_paths:
        add_code, _add_stdout, _add_stderr = _execute(
            ("git", "add", "-N", "--", *phase0_paths),
            cwd=repo,
            environment=environment,
        )
        if add_code != 0:
            raise DiffCheckError("git add -N failed for Phase 0 untracked paths")
    _assert_intent_to_add(repo, paths=phase0_paths, environment=environment)
    object_sink_after_add = _validated_object_sink_inventory(repo, object_sink)
    try:
        alternate_stat = alternate_index.lstat()
    except OSError as exc:
        raise DiffCheckError("alternate Git index was not created") from exc
    if (
        not stat.S_ISREG(alternate_stat.st_mode)
        or alternate_stat.st_uid != os.getuid()
        or alternate_stat.st_nlink != 1
    ):
        raise DiffCheckError("alternate Git index identity is unsafe")

    version_code, version_stdout, version_stderr = _execute(
        ("git", "--version"),
        cwd=repo,
        environment=environment,
    )
    try:
        tool_version = version_stdout.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise DiffCheckError("git --version output is invalid") from exc
    if version_code != 0 or version_stderr or GIT_VERSION_RE.fullmatch(tool_version) is None:
        raise DiffCheckError("git --version probe failed")

    final_argv = ["git", "diff", "--check"]
    final_code, final_stdout, final_stderr = _execute(
        final_argv,
        cwd=repo,
        environment=environment,
    )
    source_after = _source_snapshot(repo)
    _assert_source_snapshot_equal(source_before, source_after)
    if _source_binding_from_snapshot(source_after) != expected:
        raise DiffCheckError("source binding drifted during diff check")
    guard_after = _repository_guard(repo)
    if guard_after != guard_before:
        raise DiffCheckError("real Git index/refs/object identity changed during diff check")
    if _validated_object_sink_inventory(repo, object_sink) != object_sink_after_add:
        raise DiffCheckError("alternate Git object sink changed after intent-to-add")
    return {
        "argv": final_argv,
        "cwd": str(repo),
        "environment": environment,
        "exit_code": final_code if final_code >= 0 else None,
        "signal": -final_code if final_code < 0 else None,
        "stderr": final_stderr,
        "stdout": final_stdout,
        "tool_version": tool_version,
    }


def _load_expected_source_binding(path: Path) -> dict[str, Any]:
    raw, _observed = _stable_regular_bytes(path, label="expected source binding JSON")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DiffCheckError("expected source binding JSON cannot contain a BOM")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DiffCheckError("expected source binding JSON is invalid") from exc
    if raw != _canonical_bytes(value) + b"\n":
        raise DiffCheckError("expected source binding JSON must be canonical plus newline")
    return _validate_source_binding(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-source-binding-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected = _load_expected_source_binding(args.expected_source_binding_json)
        result = run_isolated_diff_check(
            repo_root=args.repo_root,
            work_root=args.work_root,
            expected_source_binding=expected,
        )
    except DiffCheckError as exc:
        print(f"v17 Phase 0 diff check failed: {exc}", file=sys.stderr)
        return exc.exit_code
    sys.stdout.buffer.write(result["stdout"])
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(result["stderr"])
    sys.stderr.buffer.flush()
    exit_code = result["exit_code"]
    return 128 + result["signal"] if exit_code is None else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
