from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from quant_investor.cli import main as cli_main


ROOT = Path(__file__).resolve().parents[2]
RETIRED_IMPORTS = (
    "quant_investor.themes",
    "quant_investor.agents.theme_agent",
    "quant_investor.funnel.theme_candidate_pool",
    "quant_investor.funnel.theme_boost_diagnostics",
    "quant_investor.market.dag.theme_context",
    "quant_investor.monitoring.theme_holding_guard",
    "scripts.run_theme_protocol_v2",
)
ACTIVE_FILES = (
    "quant_investor/config.py",
    "quant_investor/funnel/deterministic_funnel.py",
    "quant_investor/market/dag/context.py",
    "quant_investor/market/dag/decision.py",
    "quant_investor/agents/risk_guard.py",
    "quant_investor/agents/portfolio_constructor.py",
    "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
    "quant_investor/monitoring/cn_aggressive_daily_review.py",
    "scripts/export_cn_aggressive_dashboard_data.py",
    "scripts/check_cn_dashboard_export.py",
    "portfolio_dashboard/index.html",
    "portfolio_dashboard/app.js",
    "portfolio_dashboard/js/data.js",
    "portfolio_dashboard/js/metrics.js",
    "portfolio_dashboard/js/ui.js",
    "portfolio_dashboard/js/generated_records.js",
)


@pytest.mark.parametrize("module_name", RETIRED_IMPORTS)
def test_retired_theme_modules_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_active_v15_surfaces_have_no_theme_contract_tokens() -> None:
    violations: list[str] = []
    for relative in ACTIVE_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "theme" in text.casefold():
            violations.append(relative)
    assert violations == []


def test_current_cli_has_no_theme_commands_or_flags() -> None:
    parser = cli_main._build_parser()
    help_text = parser.format_help().casefold()
    assert "theme" not in help_text
