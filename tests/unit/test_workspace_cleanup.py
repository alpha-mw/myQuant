"""Workspace cleanup contract tests."""

from __future__ import annotations

from scripts.workspace_cleanup import main as cleanup_main
from scripts.workspace_layout import (
    describe_environment_roles,
    ensure_runtime_tmp_dirs,
    find_legacy_workspace_root_references,
    iter_cleanup_targets,
    replace_legacy_workspace_root_references,
)


def test_iter_cleanup_targets_only_collects_safe_workspace_caches(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "quant_investor" / "__pycache__").mkdir(parents=True)
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".uv-cache").mkdir()
    (tmp_path / "frontend" / "dist").mkdir(parents=True)
    (tmp_path / "venv" / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "data" / "__pycache__").mkdir(parents=True)
    (tmp_path / "results" / "__pycache__").mkdir(parents=True)
    (tmp_path / "frontend" / "node_modules" / "pkg" / "__pycache__").mkdir(parents=True)

    targets = [path.relative_to(tmp_path).as_posix() for path in iter_cleanup_targets(tmp_path)]

    assert targets == [
        ".pytest_cache",
        ".uv-cache",
        "__pycache__",
        "frontend/dist",
        "quant_investor/__pycache__",
    ]


def test_ensure_runtime_tmp_dirs_creates_expected_directories(tmp_path):
    results_tmp, reports_tmp = ensure_runtime_tmp_dirs(tmp_path)

    assert results_tmp == tmp_path / "results" / "tmp"
    assert reports_tmp == tmp_path / "reports" / "tmp"
    assert results_tmp.is_dir()
    assert reports_tmp.is_dir()


def test_describe_environment_roles_reports_current_presence(tmp_path):
    (tmp_path / "venv").mkdir()
    (tmp_path / ".venv-managed").mkdir()

    roles = {
        item["relative_path"]: item
        for item in describe_environment_roles(tmp_path)
    }

    assert roles["venv"]["exists"] is True
    assert roles[".venv"]["exists"] is False
    assert roles[".venv-managed"]["exists"] is True


def test_workspace_cleanup_script_applies_cleanup_and_prepares_tmp_dirs(tmp_path, capsys):
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "frontend" / "dist").mkdir(parents=True)

    exit_code = cleanup_main(["--root", str(tmp_path), "--apply", "--show-envs"])

    assert exit_code == 0
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "frontend" / "dist").exists()
    assert (tmp_path / "results" / "tmp").is_dir()
    assert (tmp_path / "reports" / "tmp").is_dir()

    stdout = capsys.readouterr().out
    assert "workspace cleanup mode: apply" in stdout
    assert "removed 2 directories" in stdout


def test_workspace_path_audit_finds_and_repairs_legacy_local_roots(tmp_path):
    legacy_root = "/legacy/workspace/myQuant"
    activate = tmp_path / ".venv" / "bin" / "activate"
    linked_git = tmp_path / ".claude" / "worktrees" / "demo" / ".git"

    activate.parent.mkdir(parents=True)
    linked_git.parent.mkdir(parents=True)
    activate.write_text(f"VIRTUAL_ENV='{legacy_root}/.venv'\n", encoding="utf-8")
    linked_git.write_text(
        f"gitdir: {legacy_root}/.git/worktrees/demo\n",
        encoding="utf-8",
    )

    findings = {
        item["relative_path"]: item
        for item in find_legacy_workspace_root_references(tmp_path)
    }

    assert sorted(findings) == [
        ".claude/worktrees/demo/.git",
        ".venv/bin/activate",
    ]

    updated = [
        path.relative_to(tmp_path).as_posix()
        for path in replace_legacy_workspace_root_references(tmp_path)
    ]

    assert sorted(updated) == [
        ".claude/worktrees/demo/.git",
        ".venv/bin/activate",
    ]
    assert str(tmp_path / ".venv") in activate.read_text(encoding="utf-8")
    assert str(tmp_path / ".git" / "worktrees" / "demo") in linked_git.read_text(
        encoding="utf-8"
    )
    assert find_legacy_workspace_root_references(tmp_path) == []
