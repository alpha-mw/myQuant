from __future__ import annotations

from pathlib import Path

from quant_investor.bayesian.types import LikelihoodSet
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.themes.shadow import build_theme_production_overlay_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_theme_production_overlay_default_off_is_no_theme_baseline() -> None:
    overlay = build_theme_production_overlay_diagnostics(
        funnel_boost_enabled=False,
        risk_guard_enabled=False,
        portfolio_cap_enabled=False,
    )

    assert overlay["production_decision_source"] == "no_theme_baseline"
    assert overlay["control_decision_source"] == "no_theme_baseline"
    assert overlay["theme_overlay_applied_to_baseline"] is False
    assert overlay["theme_overlay_modules"] == {
        "funnel_boost": False,
        "risk_guard": False,
        "portfolio_cap": False,
    }
    assert overlay["canonical_branch_unchanged"] is True
    assert overlay["theme_likelihood_added"] is False
    assert overlay["posterior_formula_changed"] is False


def test_theme_production_overlay_all_explicit_toggles_formalize_baseline() -> None:
    overlay = build_theme_production_overlay_diagnostics(
        funnel_boost_enabled=True,
        risk_guard_enabled=True,
        portfolio_cap_enabled=True,
    )

    assert overlay["production_decision_source"] == "theme_overlay_baseline"
    assert overlay["control_decision_source"] == "no_theme_baseline"
    assert overlay["theme_overlay_applied_to_baseline"] is True
    assert overlay["theme_overlay_modules"] == {
        "funnel_boost": True,
        "risk_guard": True,
        "portfolio_cap": True,
    }
    assert overlay["canonical_branch_unchanged"] is True
    assert overlay["theme_likelihood_added"] is False
    assert overlay["posterior_formula_changed"] is False


def test_theme_overlay_keeps_v14_branch_and_bayesian_boundaries() -> None:
    assert CANONICAL_BRANCH_ORDER == (
        "quant",
        "fundamental",
        "macro",
    )
    assert "theme" not in CANONICAL_BRANCH_ORDER

    likelihoods = LikelihoodSet(
        quant_likelihood=0.61,
        fundamental_likelihood=0.52,
    )
    assert not hasattr(likelihoods, "theme_likelihood")
    assert "theme_likelihood" not in likelihoods.to_dict()

    bayesian_root = REPO_ROOT / "quant_investor" / "bayesian"
    for path in (
        bayesian_root / "likelihood.py",
        bayesian_root / "types.py",
        bayesian_root / "posterior.py",
    ):
        assert "theme_likelihood" not in path.read_text(encoding="utf-8")
