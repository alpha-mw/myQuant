"""Workspace organization rules for the myQuant repository."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT_DIRS = (
    "quant_investor",
    "web",
    "frontend",
    "tests",
    "docs",
    "data",
    "results",
    "reports",
)

ENVIRONMENT_ROLES = {
    "venv": "Current Web and main-flow compatible environment.",
    ".venv": "Current script-side environment.",
    ".venv-managed": "Candidate managed environment kept for migration only.",
}

EXPLICIT_CLEANUP_DIRS = (
    Path(".cache"),
    Path(".pytest_cache"),
    Path(".uv-cache"),
    Path("frontend") / "dist",
)
_PATH_AUDIT_FILES = (
    Path(".venv/bin/activate"),
    Path(".venv/bin/activate.bat"),
    Path(".venv/bin/activate.csh"),
    Path(".venv/bin/activate.fish"),
    Path(".venv/bin/activate.nu"),
)

_PATH_AUDIT_GLOBS = (
    ".claude/worktrees/*/.git",
)

_SCAN_EXCLUDE_ROOTS = {
    ".git",
    ".venv",
    ".venv-managed",
    ".uv-python",
    "venv",
    "data",
    "results",
    "reports",
}


def _coerce_legacy_workspace_roots(
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None,
) -> tuple[Path, ...]:
    if legacy_roots is None:
        return ()
    return tuple(Path(root).expanduser() for root in legacy_roots)


def get_repo_root(root: Path | None = None) -> Path:
    """Return the workspace root used by cleanup helpers."""
    if root is not None:
        return root.resolve()
    return Path(__file__).resolve().parents[1]


def get_runtime_tmp_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Return the runtime tmp directories reserved for local transient output."""
    repo_root = get_repo_root(root)
    return repo_root / "results" / "tmp", repo_root / "reports" / "tmp"


def ensure_runtime_tmp_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Create runtime tmp directories when they do not already exist."""
    tmp_dirs = get_runtime_tmp_dirs(root)
    for path in tmp_dirs:
        path.mkdir(parents=True, exist_ok=True)
    return tmp_dirs


def describe_environment_roles(root: Path | None = None) -> list[dict[str, object]]:
    """Describe the current Python environment directories kept in the repo."""
    repo_root = get_repo_root(root)
    descriptions: list[dict[str, object]] = []
    for relative_name, role in ENVIRONMENT_ROLES.items():
        path = repo_root / relative_name
        descriptions.append(
            {
                "relative_path": relative_name,
                "path": path,
                "role": role,
                "exists": path.exists(),
            }
        )
    return descriptions


def _should_skip_tree(repo_root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    if not relative.parts:
        return False
    if relative.parts[0] in _SCAN_EXCLUDE_ROOTS:
        return True
    return "node_modules" in relative.parts


def iter_cleanup_targets(root: Path | None = None) -> list[Path]:
    """Collect safe-to-delete cache directories inside the workspace."""
    repo_root = get_repo_root(root)
    targets: dict[str, Path] = {}

    for relative_dir in EXPLICIT_CLEANUP_DIRS:
        path = repo_root / relative_dir
        if path.exists():
            targets[str(path)] = path

    for current_root, dir_names, _file_names in os.walk(repo_root):
        current_path = Path(current_root)
        if _should_skip_tree(repo_root, current_path):
            dir_names[:] = []
            continue

        if current_path.name == "__pycache__":
            targets[str(current_path)] = current_path
            dir_names[:] = []
            continue

        dir_names[:] = [
            directory
            for directory in dir_names
            if not _should_skip_tree(repo_root, current_path / directory)
        ]

    return sorted(targets.values(), key=lambda path: path.relative_to(repo_root).as_posix())


def remove_cleanup_targets(paths: list[Path]) -> list[Path]:
    """Delete collected cleanup targets and return the removed paths."""
    removed: list[Path] = []
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def iter_workspace_path_audit_targets(root: Path | None = None) -> list[Path]:
    """Collect local text files that may retain a moved workspace root."""
    repo_root = get_repo_root(root)
    targets: dict[str, Path] = {}

    for relative_path in _PATH_AUDIT_FILES:
        path = repo_root / relative_path
        if path.exists():
            targets[str(path)] = path

    for pattern in _PATH_AUDIT_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                targets[str(path)] = path

    return sorted(targets.values(), key=lambda path: path.relative_to(repo_root).as_posix())


def find_legacy_workspace_root_references(
    root: Path | None = None,
    *,
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, object]]:
    """Report local operational files that still point at the legacy workspace root."""
    repo_root = get_repo_root(root)
    resolved_legacy_roots = _coerce_legacy_workspace_roots(legacy_roots)
    findings: list[dict[str, object]] = []

    for path in iter_workspace_path_audit_targets(repo_root):
        text = path.read_text(encoding="utf-8")
        for legacy_root in resolved_legacy_roots:
            legacy_text = str(legacy_root)
            if legacy_text not in text:
                continue
            findings.append(
                {
                    "relative_path": path.relative_to(repo_root).as_posix(),
                    "path": path,
                    "legacy_root": legacy_root,
                }
            )
            break

    return findings


def replace_legacy_workspace_root_references(
    root: Path | None = None,
    *,
    new_root: Path | None = None,
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    """Rewrite legacy workspace roots inside local operational text files."""
    repo_root = get_repo_root(root)
    target_root = get_repo_root(new_root) if new_root is not None else repo_root
    resolved_legacy_roots = _coerce_legacy_workspace_roots(legacy_roots)
    updated_paths: list[Path] = []

    for path in iter_workspace_path_audit_targets(repo_root):
        original_text = path.read_text(encoding="utf-8")
        updated_text = original_text

        for legacy_root in resolved_legacy_roots:
            updated_text = updated_text.replace(str(legacy_root), str(target_root))

        if updated_text == original_text:
            continue

        path.write_text(updated_text, encoding="utf-8")
        updated_paths.append(path)

    return updated_paths
