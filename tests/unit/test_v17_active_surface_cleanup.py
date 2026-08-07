from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor import config as config_module

ROOT = Path(__file__).resolve().parents[2]

RETIRED_ACTIVE_DEFAULTS = {
    "FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256",
    "FUNDAMENTAL_RESEARCH_ACTIVATION_PATH",
    "FUNDAMENTAL_RESEARCH_OVERLAY_MODE",
    "FUNDAMENTAL_RESEARCH_ROOT",
    "MARKOV_REGIME_ENABLED",
    "MARKOV_REGIME_EXECUTION_TARGET",
    "MARKOV_REGIME_HISTORY_PATH",
    "MARKOV_REGIME_MAX_REFERENCE_SYMBOLS",
    "MARKOV_REGIME_MIN_MARKET_SAMPLE",
    "MARKOV_REGIME_PERSIST_ENABLED",
    "MARKOV_REGIME_REFERENCE_UNIVERSE_CN",
    "MARKOV_REGIME_REFERENCE_UNIVERSE_US",
    "RISK_GUARD_SINGLE_NAME_WEIGHT_CAP",
}


def test_retired_runtime_controls_are_not_active_defaults_or_attributes() -> None:
    assert RETIRED_ACTIVE_DEFAULTS.isdisjoint(config_module.MAINLINE_ENV_DEFAULTS)
    assert not hasattr(config_module.Config, "PIPELINE_MODE")
    assert not hasattr(config_module.Config, "DECISION_ENGINE")
    assert not hasattr(config_module.Config, "DEFAULT_AGENT_TOTAL_TIMEOUT_SECONDS")
    assert not hasattr(config_module.Config, "MAINLINE_ENV_KEYS")


@pytest.mark.parametrize("retired_key", config_module.RETIRED_ENV_KEYS)
def test_retired_environment_controls_fail_loudly(
    retired_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(retired_key, "legacy-value")
    with pytest.raises(RuntimeError, match="retired myQuant environment keys"):
        config_module._reject_retired_env_keys()


def test_legacy_single_layer_regime_detector_is_removed() -> None:
    assert not (ROOT / "quant_investor" / "regime_detector.py").exists()
    cli_source = (ROOT / "quant_investor" / "cli" / "main.py").read_text(encoding="utf-8")
    assert "regime_detector" not in cli_source
    assert "intelligence.regime" not in cli_source


def test_web_workspace_layer_is_removed() -> None:
    for relative_path in (
        "web",
        "frontend",
        "run_web.sh",
        "vercel.json",
        "docker-compose.yml",
        "quant_investor/run_history_store.py",
    ):
        assert not (ROOT / relative_path).exists(), relative_path

    cli_source = (ROOT / "quant_investor" / "cli" / "main.py").read_text(encoding="utf-8")
    assert "uvicorn" not in cli_source
    assert "web.main" not in cli_source
    assert "run_web_api" not in cli_source


def test_web_only_dependencies_are_not_declared() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("fastapi", "uvicorn", "pydantic", "httpx"):
        assert package not in pyproject, package
        assert package not in requirements, package
    assert "web.main:app" not in pyproject


def test_superseded_contract_generations_are_removed() -> None:
    """v17_v4_contract is the only surviving generation of the V17 contract package."""

    for relative_path in (
        "quant_investor/v17",
        "quant_investor/v17_v2_contract",
        "quant_investor/v17_v2_runtime",
        "quant_investor/v17_v3_contract",
    ):
        assert not (ROOT / relative_path).exists(), relative_path

    assert (ROOT / "quant_investor" / "v17_v4_contract").is_dir()


def test_unreferenced_modules_are_removed() -> None:
    for relative_path in (
        "quant_investor/fetch_us_index_components.py",
        "quant_investor/fetch_complete_us_universe.py",
        "quant_investor/reencode_csv_for_excel.py",
        "quant_investor/macro/tushare_normalizer.py",
        "quant_investor/_vendor",
        "quant_investor/data/storage",
        "quant_investor/monitoring",
        "quant_investor/intelligence/decision",
        "quant_investor/market/resources",
    ):
        assert not (ROOT / relative_path).exists(), relative_path


def test_runtime_entrypoints_do_not_import_the_evidence_archive() -> None:
    """The governance evidence archive is SHA-pinned and must stay off runtime paths.

    See docs/architecture/evidence_archive_boundary.md.
    """

    for relative_path in (
        "quant_investor/cli/main.py",
        "quant_investor/automation/daily_runner.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "governance_" not in source, relative_path


def test_evidence_archive_boundary_is_documented() -> None:
    doc = ROOT / "docs" / "architecture" / "evidence_archive_boundary.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "FIXED_EXISTING_PROJECT_SHA256" in text


def test_mypy_configuration_does_not_hide_removed_packages() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    stale_paths = {
        "quant_investor/agent_orchestrator",
        "quant_investor/agents",
        "quant_investor/_vendor",
        "quant_investor/v17_v3_contract",
        "quant_investor/enhanced_data_layer",
        "quant_investor/forecast_snapshot_store",
        "quant_investor/learning",
        "quant_investor/market/run_pipeline",
        "quant_investor/monitoring",
        "quant_investor/reporting",
    }
    assert not any(path in pyproject for path in stale_paths)
