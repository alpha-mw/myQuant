from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_RUNTIME_MODULES = [
    ROOT / "daily_runner.py",
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

EXECUTION_COST_NAMES = [
    "quant_investor.factors.execution_cost",
    "FactorExecutionCostSimulationStore",
    "FactorExecutionCostConfig",
    "FactorExecutionCostSimulationReport",
    "ExecutionAdjustedBacktestRun",
    "build_execution_cost_simulation_report",
    "build_execution_adjusted_backtest_run",
    "render_execution_cost_report_markdown",
    "execution_cost",
    "execution_adjusted_runs",
]


def test_execution_cost_helpers_are_absent_from_runtime_decision_modules() -> None:
    checked_paths = []

    for path in FORBIDDEN_RUNTIME_MODULES:
        if not path.exists():
            continue
        checked_paths.append(path)
        text = path.read_text(encoding="utf-8")
        for helper_name in EXECUTION_COST_NAMES:
            assert helper_name not in text, f"{helper_name} leaked into {path}"

    assert checked_paths


def test_execution_cost_does_not_touch_order_action_or_weight_paths() -> None:
    tracker_path = ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py"
    tracker_text = tracker_path.read_text(encoding="utf-8")

    for helper_name in EXECUTION_COST_NAMES:
        assert helper_name not in tracker_text
    assert "orders.csv" in tracker_text
    assert "action_taken_today" in tracker_text
