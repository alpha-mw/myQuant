from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from quant_investor.factors.governance_v5 import build_governance_policy

ROOT = Path(__file__).resolve().parents[2]

V4_SHA256 = {
    "quant_investor/factors/aquant_expression.py": (
        "775b012c8042315b4950d3624a77e585a8356e7e3cf0613fc0f9559f926f88e0"
    ),
    "quant_investor/factors/governance_protocol_v4.py": (
        "a0e9abd2333f551e9ea7000796eecfce6248c2fcb09ba4e8704ccda01c3de281"
    ),
    "quant_investor/factors/production_set_carrier_v4.py": (
        "4ed13db694514d70cb8bcb36c0b1f01eeab81e27a5d108c5fd100eee2f9da700"
    ),
}


def _tree_sha(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*.py")):
        rows.append(
            path.relative_to(root).as_posix().encode("utf-8")
            + b"\0"
            + hashlib.sha256(path.read_bytes()).digest()
        )
    return hashlib.sha256(b"\n".join(rows)).hexdigest()


def test_v4_source_bytes_remain_at_current_main_baseline():
    for relative_path, expected in V4_SHA256.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_v5_is_not_registered_in_legacy_factor_runtime_or_automation():
    runtime = (ROOT / "quant_investor/factors/runtime.py").read_text(encoding="utf-8")
    automation = (ROOT / "scripts/daily_factor_mining_automation.py").read_text(encoding="utf-8")
    assert "aquant_expression_v5" not in runtime
    assert "mine_quant_branch_factors_v5" not in automation


def test_v5_package_has_no_external_or_mutating_imports():
    forbidden = {
        "aiohttp",
        "requests",
        "tushare",
        "yfinance",
        "quant_investor.broker",
        "quant_investor.execution",
        "quant_investor.portfolio_backtest",
        "quant_investor.factors.production_control_v1",
        "quant_investor.factors.registry_store",
    }
    source_paths = [
        *sorted((ROOT / "quant_investor/factors/governance_v5").glob("*.py")),
        ROOT / "quant_investor/factors/aquant_expression_v5.py",
    ]
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not imports & forbidden, path


def test_pure_policy_builder_does_not_mutate_factor_source_tree():
    factor_root = ROOT / "quant_investor/factors"
    before = _tree_sha(factor_root)
    build_governance_policy(
        created_at="2026-08-08T00:00:00Z",
        coverage_threshold="0.800000000000",
        label_horizon_sessions=20,
        minimum_prospective_paths=2,
    )
    assert _tree_sha(factor_root) == before
