from __future__ import annotations

import ast
import re
from pathlib import Path

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_theme_likelihood_symbol_exists_under_bayesian_package() -> None:
    bayesian_root = REPO_ROOT / "quant_investor" / "bayesian"

    for path in bayesian_root.rglob("*.py"):
        assert "theme_likelihood" not in path.read_text(encoding="utf-8")


def test_no_theme_in_canonical_branch_order_source() -> None:
    assert "theme" not in CANONICAL_BRANCH_ORDER
    assert CANONICAL_BRANCH_ORDER == (
        "quant",
        "fundamental",
        "intelligence",
        "macro",
    )


def test_replay_calibration_not_imported_by_dag_context() -> None:
    source = (REPO_ROOT / "quant_investor" / "market" / "dag" / "context.py").read_text(
        encoding="utf-8"
    )

    assert "themes.replay" not in source
    assert "themes.calibration" not in source


def test_theme_boost_diagnostics_not_imported_by_dag_context() -> None:
    source = (REPO_ROOT / "quant_investor" / "market" / "dag" / "context.py").read_text(
        encoding="utf-8"
    )

    assert "theme_boost_diagnostics" not in source


def test_theme_agent_not_imported_by_research_dag() -> None:
    source = (REPO_ROOT / "quant_investor" / "market" / "dag" / "research.py").read_text(
        encoding="utf-8"
    )

    assert "ThemeAgent" not in source
    assert "theme_agent" not in source


def test_theme_governance_not_wired_into_authoritative_components() -> None:
    checked_paths = [
        REPO_ROOT / "quant_investor" / "bayesian" / "types.py",
        REPO_ROOT / "quant_investor" / "bayesian" / "likelihood.py",
        REPO_ROOT / "quant_investor" / "bayesian" / "posterior.py",
        REPO_ROOT / "quant_investor" / "agents" / "risk_guard.py",
        REPO_ROOT / "quant_investor" / "agents" / "portfolio_constructor.py",
        REPO_ROOT / "quant_investor" / "branch_config.py",
    ]

    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        assert "theme_governance" not in source, path
        assert "admitted_shadow" not in source, path


def test_no_external_network_imports_in_theme_modules() -> None:
    scanned_paths = [
        *sorted((REPO_ROOT / "quant_investor" / "themes").rglob("*.py")),
        REPO_ROOT / "quant_investor" / "funnel" / "theme_boost_diagnostics.py",
    ]
    forbidden_import_roots = {
        "aiohttp",
        "httpx",
        "requests",
        "tushare",
        "urllib",
        "yfinance",
    }
    forbidden_provider_strings = ("OpenAI", "Anthropic", "LLMClient")

    for path in scanned_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])

        assert not (imported_roots & forbidden_import_roots), path

        scrubbed = re.sub(r"no[_-]?llm", "", source, flags=re.IGNORECASE)
        for provider in forbidden_provider_strings:
            assert provider not in scrubbed, path
        assert re.search(r"\bllm\b", scrubbed, flags=re.IGNORECASE) is None, path
