from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_RUNTIME_MODULES = [
    ROOT / "quant_investor" / "portfolio_optimizer.py",
    ROOT / "quant_investor" / "agents" / "risk_guard.py",
    ROOT / "quant_investor" / "agents" / "portfolio_constructor.py",
    ROOT / "quant_investor" / "bayesian" / "posterior.py",
    ROOT / "quant_investor" / "pipeline" / "mainline.py",
    ROOT / "quant_investor" / "market" / "dag" / "research.py",
]

ALLOWED_REPORTING_MODULES = [
    ROOT / "quant_investor" / "observability.py",
    ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py",
    ROOT / "quant_investor" / "factors" / "report.py",
]

SHADOW_HELPER_NAMES = [
    "load_factor_library_shadow_status",
    "render_factor_library_shadow_markdown",
]


def test_factor_shadow_helpers_are_absent_from_runtime_decision_modules() -> None:
    checked_paths = []

    for path in FORBIDDEN_RUNTIME_MODULES:
        if not path.exists():
            continue
        checked_paths.append(path)
        text = path.read_text(encoding="utf-8")
        for helper_name in SHADOW_HELPER_NAMES:
            assert helper_name not in text, f"{helper_name} leaked into {path}"

    assert checked_paths


def test_factor_shadow_helpers_remain_confined_to_reporting_surfaces() -> None:
    observed_allowed_usage = []

    for path in ALLOWED_REPORTING_MODULES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if any(helper_name in text for helper_name in SHADOW_HELPER_NAMES):
            observed_allowed_usage.append(path)

    assert ROOT / "quant_investor" / "factors" / "report.py" in observed_allowed_usage
    assert (
        ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py"
        in observed_allowed_usage
    )
