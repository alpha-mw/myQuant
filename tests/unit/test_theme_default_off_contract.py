from __future__ import annotations

import os
from pathlib import Path

from quant_investor.agent_protocol import ActionLabel
from quant_investor.agents.theme_agent import ThemeAgent
from quant_investor.bayesian.types import LikelihoodSet
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.config import Config, MAINLINE_ENV_DEFAULTS
from quant_investor.market.dag.theme_context import (
    persist_theme_governance_artifact,
    persist_theme_rotation_snapshot,
)


def test_theme_production_config_defaults_are_on_without_shadow() -> None:
    expected_default_on = {
        "THEME_SCANNER_ENABLED": "1",
        "THEME_FUNNEL_BOOST_ENABLED": "1",
        "THEME_RISK_GUARD_ENABLED": "1",
        "THEME_PORTFOLIO_CAP_ENABLED": "1",
        "THEME_SNAPSHOT_ENABLED": "1",
        "THEME_POOL_ENABLED": "1",
        "THEME_POOL_REQUIRED": "1",
        "THEME_POOL_USE_MARKOV_POLICY": "1",
        "THEME_POOL_FALLBACK_TO_RAW_SCORE": "1",
    }
    expected_default_off = {
        "THEME_SNAPSHOT_SAVE_DISABLED": "0",
        "THEME_HOLDING_GUARD_ENABLED": "0",
        "THEME_CROWDING_ENABLED": "0",
        "THEME_CONCEPT_MEMBERSHIP_ENABLED": "0",
        "THEME_CONCEPT_MEMBERSHIP_REQUIRED": "0",
        "THEME_STAT_CLUSTER_ENABLED": "0",
        "THEME_SHADOW_MODE_ENABLED": "0",
        "THEME_GOVERNANCE_ENABLED": "0",
        "THEME_GOVERNANCE_ARTIFACT_ENABLED": "0",
    }

    for key, expected in expected_default_on.items():
        assert MAINLINE_ENV_DEFAULTS[key] == expected
        if key not in os.environ:
            assert getattr(Config, key) is True

    for key, expected in expected_default_off.items():
        assert MAINLINE_ENV_DEFAULTS[key] == expected
        if key not in os.environ:
            assert getattr(Config, key) is False

    assert MAINLINE_ENV_DEFAULTS["THEME_FUNNEL_BOOST_SCORE_SOURCE"] == "raw"
    if "THEME_FUNNEL_BOOST_SCORE_SOURCE" not in os.environ:
        assert Config.THEME_FUNNEL_BOOST_SCORE_SOURCE == "raw"
    assert MAINLINE_ENV_DEFAULTS["THEME_POOL_SCORE_SOURCE"] == "smoothed"
    if "THEME_POOL_SCORE_SOURCE" not in os.environ:
        assert Config.THEME_POOL_SCORE_SOURCE == "smoothed"
    assert MAINLINE_ENV_DEFAULTS["THEME_POOL_MIN_THEME_SCORE"] == "0.58"
    assert MAINLINE_ENV_DEFAULTS["THEME_POOL_MIN_SYMBOL_SCORE"] == "0.55"
    assert MAINLINE_ENV_DEFAULTS["THEME_HOLDING_GUARD_TIGHTEN_RATIO"] == "0.5"
    assert MAINLINE_ENV_DEFAULTS["THEME_CROWDING_MIN_UNIVERSE"] == "30"
    assert MAINLINE_ENV_DEFAULTS["THEME_CONCEPT_MEMBERSHIP_PATH"] == "data/theme_membership.jsonl"
    assert MAINLINE_ENV_DEFAULTS["THEME_CONCEPT_PRIMARY_MARGIN"] == "0.05"


def test_no_theme_likelihood_in_bayesian_types() -> None:
    likelihoods = LikelihoodSet(
        quant_likelihood=0.61,
        fundamental_likelihood=0.52,
        intelligence_likelihood=0.49,
    )

    assert not hasattr(likelihoods, "theme_likelihood")
    assert "theme_likelihood" not in likelihoods.to_dict()
    assert likelihoods.as_list() == [
        ("quant", 0.61),
        ("fundamental", 0.52),
        ("intelligence", 0.49),
    ]
    assert len(likelihoods.as_list()) == 3


def test_canonical_branch_order_has_no_theme() -> None:
    assert "theme" not in CANONICAL_BRANCH_ORDER
    assert CANONICAL_BRANCH_ORDER == (
        "quant",
        "fundamental",
        "intelligence",
        "macro",
    )


def test_theme_agent_is_non_canonical_metadata_only() -> None:
    verdict = ThemeAgent().run({"symbol": "000001.SZ"})

    assert verdict.metadata["branch_name"] == "theme"
    assert "theme" not in CANONICAL_BRANCH_ORDER
    assert verdict.final_score == 0.0
    assert verdict.final_confidence == 0.0
    assert verdict.action == ActionLabel.HOLD
    assert verdict.metadata["theme_data_available"] is False
    assert "theme_data_unavailable" in verdict.diagnostic_notes


def test_theme_snapshot_disabled_does_not_write(tmp_path: Path) -> None:
    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "success", "symbol_scores": {"000001.SZ": 0.8}},
        enabled=False,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert status["enabled"] is False
    assert status["status"] == "disabled"
    assert status["path"] == ""
    assert status["diagnostic_notes"] == ["theme_snapshot_disabled"]
    assert list(tmp_path.rglob("*.json")) == []


def test_theme_governance_artifact_disabled_does_not_write(tmp_path: Path) -> None:
    status = persist_theme_governance_artifact(
        theme_governance={
            "schema_version": "theme_governance.v1",
            "status": "success",
            "decisions": [],
        },
        enabled=False,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert status["enabled"] is False
    assert status["status"] == "disabled"
    assert status["path"] == ""
    assert status["diagnostic_notes"] == ["theme_governance_artifact_disabled"]
    assert list(tmp_path.rglob("*.json")) == []
