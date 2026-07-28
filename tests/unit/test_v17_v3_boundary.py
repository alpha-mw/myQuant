from __future__ import annotations

import ast
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_V2_TREES = {
    "quant_investor/v17_v2_contract": (
        46,
        "58de840f785feea78d3f7d69f5199bfe45f2a60f70102f62f3821d5f523657e2",
    ),
    "quant_investor/v17_v2_runtime": (
        19,
        "72e577bcb959ffc029feca3191bf3a10ec2ab79257dc0af13060b3ccb1816eff",
    ),
}


def _tree_sha256(relative_root: str) -> tuple[int, str]:
    root = REPO_ROOT / relative_root
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store"
    )
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return len(paths), digest.hexdigest()


def test_v17_v2_contract_and_runtime_trees_remain_byte_frozen() -> None:
    for relative_root, expected in FROZEN_V2_TREES.items():
        assert _tree_sha256(relative_root) == expected


def test_v17_v3_python_sources_do_not_import_v2() -> None:
    roots = (
        REPO_ROOT / "quant_investor/v17_v3_contract",
        REPO_ROOT / "quant_investor/v17_v3_runtime",
    )
    sources = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    assert sources, "v17 v3 packages must exist"
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert not any("v17_v2" in module for module in imported_modules), path


def test_v17_v3_python_sources_have_no_live_provider_or_network_imports() -> None:
    forbidden_import_roots = {
        "aiohttp",
        "httpx",
        "openai",
        "requests",
        "socket",
        "subprocess",
        "tushare",
        "urllib",
        "yfinance",
    }
    roots = (
        REPO_ROOT / "quant_investor/v17_v3_contract",
        REPO_ROOT / "quant_investor/v17_v3_runtime",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.partition(".")[0])
            assert not imported_roots & forbidden_import_roots, path


def test_v17_v3_has_a_dedicated_entrypoint_and_does_not_replace_v15() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'quant-investor = "quant_investor.cli.main:main"' in pyproject
    assert 'quant-investor-v17-v3 = "quant_investor.v17_v3_runtime.cli:main"' in pyproject

    entrypoints = (REPO_ROOT / "docs/architecture/entrypoints_and_versioning.md").read_text(
        encoding="utf-8"
    )
    assert "V15" in entrypoints
    assert "quant-investor-v17-v3" in entrypoints
