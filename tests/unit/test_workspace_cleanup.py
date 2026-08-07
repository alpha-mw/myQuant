"""Workspace cleanup contract tests."""

from __future__ import annotations

import json

from scripts.workspace_cleanup import main as cleanup_main
from scripts.workspace_layout import (
    PROTECTED_INVENTORY_PATHS,
    build_cleanup_inventory,
    describe_environment_roles,
    ensure_runtime_tmp_dirs,
    find_legacy_workspace_root_references,
    iter_cleanup_targets,
    replace_legacy_workspace_root_references,
)


EXPECTED_CODE_RETIREMENT_CANDIDATES = sum(
    1
    for _path, classification, _reason in PROTECTED_INVENTORY_PATHS
    if classification == "code_retirement_candidate"
)


def test_iter_cleanup_targets_only_collects_safe_workspace_caches(tmp_path):
    (tmp_path / ".cache" / "quant_investor").mkdir(parents=True)
    (tmp_path / ".mypy_cache").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "quant_investor" / "__pycache__").mkdir(parents=True)
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".uv-cache").mkdir()
    (tmp_path / "results" / "htmlcov").mkdir(parents=True)
    (tmp_path / "venv" / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "data" / "__pycache__").mkdir(parents=True)
    (tmp_path / "results" / "__pycache__").mkdir(parents=True)

    targets = [path.relative_to(tmp_path).as_posix() for path in iter_cleanup_targets(tmp_path)]

    assert targets == [
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".uv-cache",
        "__pycache__",
        "quant_investor/__pycache__",
        "results/htmlcov",
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
    (tmp_path / "results" / "htmlcov").mkdir(parents=True)

    exit_code = cleanup_main(["--root", str(tmp_path), "--apply", "--show-envs"])

    assert exit_code == 0
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "results" / "htmlcov").exists()
    assert (tmp_path / "results" / "tmp").is_dir()
    assert (tmp_path / "reports" / "tmp").is_dir()

    stdout = capsys.readouterr().out
    assert "workspace cleanup mode: apply" in stdout
    assert "removed 2 directories" in stdout


def test_cleanup_inventory_classifies_protected_sources_and_delete_candidates(tmp_path):
    (tmp_path / ".mypy_cache").mkdir()
    (tmp_path / "results" / "htmlcov").mkdir(parents=True)
    (tmp_path / "data" / "parquet" / "cn").mkdir(parents=True)
    (tmp_path / "data" / "parquet" / "cn" / "_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "data" / "parquet" / "cn" / "bars").mkdir()
    (tmp_path / "data" / "raw_backups" / "tushare").mkdir(parents=True)
    (tmp_path / "reports" / "storage" / "csv_quarantine").mkdir(parents=True)
    (tmp_path / "results" / "strategy_records" / "CN").mkdir(parents=True)
    (tmp_path / "quant_investor" / "kline_backends").mkdir(parents=True)
    (tmp_path / "quant_investor" / "agents").mkdir(parents=True)
    (tmp_path / "quant_investor" / "agents" / "kline_agent.py").write_text(
        "# retired kline agent\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "agents" / "subagents").mkdir(parents=True)
    (tmp_path / "quant_investor" / "agents" / "subagents" / "kline_agent.py").write_text(
        "# retired kline subagent\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "intelligence.py").write_text(
        "# retired kronos intelligence layer\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "kronos_predictor.py").write_text(
        "# retired kronos predictor\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "_vendor" / "chronos").mkdir(parents=True)
    (tmp_path / "quant_investor" / "_vendor" / "chronos_loader.py").write_text(
        "# retired chronos loader\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "_vendor" / "kronos_model").mkdir(parents=True)

    manifest = build_cleanup_inventory(tmp_path)
    items = {item["relative_path"]: item for item in manifest["items"]}

    assert items[".mypy_cache"]["classification"] == "safe_cache"
    assert items[".mypy_cache"]["delete_allowed"] is True
    assert items["results/htmlcov"]["classification"] == "derived_artifact"
    assert items["results/htmlcov"]["delete_allowed"] is True
    assert items["data/parquet/cn/_latest.json"]["classification"] == "active_runtime_source"
    assert items["data/parquet/cn/_latest.json"]["delete_allowed"] is False
    assert items["data/raw_backups/tushare"]["classification"] == "duplicate_restore_source"
    assert items["reports/storage/csv_quarantine"]["delete_allowed"] is False
    assert items["results/strategy_records"]["classification"] == "strategy_evidence"
    assert items["quant_investor/kline_backends"]["classification"] == "code_retirement_candidate"
    assert items["quant_investor/agents/kline_agent.py"]["classification"] == "code_retirement_candidate"
    assert (
        items["quant_investor/agents/subagents/kline_agent.py"]["classification"]
        == "code_retirement_candidate"
    )
    assert items["quant_investor/intelligence.py"]["classification"] == "code_retirement_candidate"
    assert items["quant_investor/kronos_predictor.py"]["classification"] == "code_retirement_candidate"
    assert items["quant_investor/_vendor/chronos"]["classification"] == "code_retirement_candidate"
    assert (
        items["quant_investor/_vendor/chronos_loader.py"]["classification"]
        == "code_retirement_candidate"
    )
    assert items["quant_investor/_vendor/kronos_model"]["classification"] == "code_retirement_candidate"
    assert manifest["summary"]["safe_cache"] == 1
    assert manifest["summary"]["code_retirement_candidate"] == (
        EXPECTED_CODE_RETIREMENT_CANDIDATES
    )
    assert manifest["delete_candidate_count"] == 2


def test_cleanup_inventory_keeps_missing_code_retirement_candidates(tmp_path):
    manifest = build_cleanup_inventory(tmp_path)
    items = {item["relative_path"]: item for item in manifest["items"]}

    candidate = items["quant_investor/kronos_predictor.py"]
    assert candidate["classification"] == "code_retirement_candidate"
    assert candidate["exists"] is False
    assert candidate["delete_allowed"] is False
    assert manifest["summary"]["code_retirement_candidate"] == (
        EXPECTED_CODE_RETIREMENT_CANDIDATES
    )


def test_workspace_cleanup_script_writes_inventory_manifest(tmp_path, capsys):
    (tmp_path / ".pytest_cache").mkdir()
    output_dir = tmp_path / "reports" / "project_cleanup" / "fixture"

    exit_code = cleanup_main(
        [
            "--root",
            str(tmp_path),
            "--skip-runtime-dirs",
            "--inventory",
            "--inventory-output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "cleanup_inventory.json").read_text())
    assert payload["schema_version"] == "myquant.cleanup_inventory.v1"
    assert payload["delete_candidate_count"] == 1
    assert (output_dir / "cleanup_inventory.md").read_text().startswith(
        "# Workspace Cleanup Inventory"
    )

    stdout = capsys.readouterr().out
    assert "workspace cleanup mode: dry-run" in stdout
    assert "cleanup inventory manifest:" in stdout


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
        for item in find_legacy_workspace_root_references(tmp_path, legacy_roots=[legacy_root])
    }

    assert sorted(findings) == [
        ".claude/worktrees/demo/.git",
        ".venv/bin/activate",
    ]

    updated = [
        path.relative_to(tmp_path).as_posix()
        for path in replace_legacy_workspace_root_references(
            tmp_path,
            legacy_roots=[legacy_root],
        )
    ]

    assert sorted(updated) == [
        ".claude/worktrees/demo/.git",
        ".venv/bin/activate",
    ]
    assert str(tmp_path / ".venv") in activate.read_text(encoding="utf-8")
    assert str(tmp_path / ".git" / "worktrees" / "demo") in linked_git.read_text(
        encoding="utf-8"
    )
    assert find_legacy_workspace_root_references(tmp_path, legacy_roots=[legacy_root]) == []
