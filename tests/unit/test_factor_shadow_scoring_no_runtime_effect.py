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
    ROOT / "quant_investor" / "market" / "dag" / "decision.py",
    ROOT / "quant_investor" / "market" / "dag" / "shortlist.py",
    ROOT / "quant_investor" / "market" / "dag_executor.py",
    ROOT / "quant_investor" / "market" / "analyze.py",
]

SHADOW_SCORING_NAMES = [
    "shadow_scoring",
    "ShadowScoringConfig",
    "ShadowFactorScore",
    "ShadowCandidateScore",
    "ShadowScoringComparisonReport",
    "build_shadow_candidate_scores",
    "build_shadow_scoring_comparison_report",
    "render_shadow_scoring_comparison_markdown",
    "FactorShadowScoringStore",
]


def test_shadow_scoring_helpers_are_absent_from_runtime_decision_modules() -> None:
    checked_paths = []

    for path in FORBIDDEN_RUNTIME_MODULES:
        if not path.exists():
            continue
        checked_paths.append(path)
        text = path.read_text(encoding="utf-8")
        for helper_name in SHADOW_SCORING_NAMES:
            assert helper_name not in text, f"{helper_name} leaked into {path}"

    assert checked_paths


def test_shadow_scoring_is_standalone_and_tracker_is_not_touched_by_phase11() -> None:
    tracker_path = ROOT / "quant_investor" / "monitoring" / "cn_aggressive_portfolio_tracker.py"
    tracker_text = tracker_path.read_text(encoding="utf-8")

    assert "因子影子评分对比（只读）" not in tracker_text
    assert "build_shadow_scoring_comparison_report" not in tracker_text
    assert "FactorShadowScoringStore" not in tracker_text
