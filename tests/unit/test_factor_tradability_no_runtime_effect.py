from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_RUNTIME_MODULES = [
    ROOT / "daily_runner.py",
    ROOT / "quant_investor" / "automation" / "daily_runner.py",
    ROOT / "quant_investor" / "automation" / "analysis_runner.py",
    ROOT / "quant_investor" / "automation" / "history_loader.py",
    ROOT / "quant_investor" / "automation" / "persistence.py",
    ROOT / "quant_investor" / "automation" / "report_builder.py",
    ROOT / "quant_investor" / "portfolio_optimizer.py",
    ROOT / "quant_investor" / "agents" / "risk_guard.py",
    ROOT / "quant_investor" / "agents" / "portfolio_constructor.py",
    ROOT / "quant_investor" / "bayesian" / "posterior.py",
    ROOT / "quant_investor" / "pipeline" / "mainline.py",
    ROOT / "quant_investor" / "market" / "dag" / "research.py",
    ROOT / "quant_investor" / "market" / "dag" / "decision.py",
    ROOT / "quant_investor" / "market" / "dag" / "shortlist.py",
    ROOT / "quant_investor" / "market" / "dag_executor.py",
    ROOT / "quant_investor" / "market" / "analyze.py",
    ROOT / "quant_investor" / "market" / "run_pipeline.py",
    ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py",
]

TRADABILITY_AUDIT_NAMES = [
    "quant_investor.factors.tradability",
    "FactorTradabilityAuditStore",
    "AShareTradabilityMask",
    "FactorTradabilityAuditReport",
    "FactorExecutionFeasibilityReport",
    "build_ashare_tradability_mask",
    "build_tradability_audit_report",
    "audit_factor_weight_execution_feasibility",
    "render_tradability_audit_markdown",
    "render_execution_feasibility_markdown",
]


def test_tradability_audit_helpers_are_absent_from_runtime_decision_modules() -> None:
    checked_paths = []

    for path in FORBIDDEN_RUNTIME_MODULES:
        if not path.exists():
            continue
        checked_paths.append(path)
        text = path.read_text(encoding="utf-8")
        for helper_name in TRADABILITY_AUDIT_NAMES:
            assert helper_name not in text, f"{helper_name} leaked into {path}"

    assert checked_paths


def test_tradability_audit_does_not_touch_order_action_or_weight_paths() -> None:
    tracker_path = ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py"
    tracker_text = tracker_path.read_text(encoding="utf-8")

    for helper_name in TRADABILITY_AUDIT_NAMES:
        assert helper_name not in tracker_text
    assert "tradability_audit" not in tracker_text
    assert "execution_feasibility" not in tracker_text
