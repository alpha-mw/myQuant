#!/usr/bin/env python3
"""Verify the isolated V17 v3 Phase A delivery without external side effects."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_investor.v17_v3_contract import PROTOCOL_VERSION, verify_package
from quant_investor.v17_v3_runtime.authority import (
    DELIVERY_STATUS,
    authority_envelope,
)

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
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
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
)


class PhaseAGateError(RuntimeError):
    """The additive Phase A implementation failed a frozen safety boundary."""


def _tree_identity(relative_root: str) -> tuple[int, str]:
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


def _verify_offline_import_boundary() -> int:
    source_count = 0
    for relative_root in (
        "quant_investor/v17_v3_contract",
        "quant_investor/v17_v3_runtime",
    ):
        for path in (REPO_ROOT / relative_root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            source_count += 1
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_roots: set[str] = set()
            imported_modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name)
                        imported_roots.add(alias.name.partition(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
                    imported_roots.add(node.module.partition(".")[0])
            if imported_roots & FORBIDDEN_IMPORT_ROOTS:
                raise PhaseAGateError(f"live provider or network import: {path}")
            if any("v17_v2" in module for module in imported_modules):
                raise PhaseAGateError(f"v17 v2 import from v3 source: {path}")
    if source_count == 0:
        raise PhaseAGateError("v17 v3 Python source inventory is empty")
    return source_count


def run_gate() -> dict[str, Any]:
    package_inventory = verify_package()
    observed_v2: dict[str, dict[str, Any]] = {}
    for relative_root, expected in FROZEN_V2_TREES.items():
        observed = _tree_identity(relative_root)
        if observed != expected:
            raise PhaseAGateError(f"frozen v17 v2 tree changed: {relative_root}")
        observed_v2[relative_root] = {
            "file_count": observed[0],
            "tree_sha256": observed[1],
        }

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    v15_entrypoint = 'quant-investor = "quant_investor.cli.main:main"'
    v3_entrypoint = 'quant-investor-v17-v3 = "quant_investor.v17_v3_runtime.cli:main"'
    if v15_entrypoint not in pyproject:
        raise PhaseAGateError("V15 production/default entrypoint changed")
    if v3_entrypoint not in pyproject:
        raise PhaseAGateError("dedicated V17 v3 entrypoint is absent")

    authority = authority_envelope()
    if any(authority.values()):
        raise PhaseAGateError("Phase A authority envelope must be entirely false")
    if DELIVERY_STATUS != "NOT_ACTIVATED_DATA_BLOCKED":
        raise PhaseAGateError("Phase A delivery status is not fail-closed")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": DELIVERY_STATUS,
        "package_verified": True,
        "package_asset_count": len(package_inventory),
        "v3_python_source_count": _verify_offline_import_boundary(),
        "v2_frozen_trees": observed_v2,
        "v15_default_entrypoint_unchanged": True,
        "dedicated_v3_entrypoint": True,
        "authority": authority,
        "provider_calls": False,
        "llm_calls": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
    }


def main() -> int:
    try:
        result = run_gate()
    except Exception as exc:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "PHASE_A_GATE_FAILED",
            "error_type": type(exc).__name__,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
