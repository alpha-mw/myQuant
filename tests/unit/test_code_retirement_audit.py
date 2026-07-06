from __future__ import annotations

import json
from pathlib import Path

from scripts.workspace_layout import (
    build_code_retirement_reference_audit,
    write_code_retirement_reference_audit,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE7_RETIRED_STEMS = {
    "advanced_risk_metrics",
    "factor_analyzer",
    "news_analysis",
    "sentiment_analysis",
    "signal_calibration",
    "stress_tester",
    "var_calculator",
    "financial_analysis",
    "risk_management_layer",
}
PHASE7_RETIRED_MODULES = {
    "quant_investor/" + stem + ".py"
    for stem in PHASE7_RETIRED_STEMS
}


def test_repository_retirement_candidates_have_no_production_references():
    manifest = build_code_retirement_reference_audit(ROOT)

    offenders = {
        candidate["relative_path"]: candidate["production_reference_count"]
        for candidate in manifest["candidates"]
        if candidate["production_reference_count"]
    }

    assert offenders == {}


def test_repository_retired_runtime_sources_are_removed_from_code_tree():
    manifest = build_code_retirement_reference_audit(ROOT)

    present = {
        candidate["relative_path"]
        for candidate in manifest["candidates"]
        if candidate["exists"]
    }

    assert manifest["candidate_count"] == 17
    assert present == set()


def test_code_retirement_audit_keeps_missing_candidates_in_manifest(tmp_path):
    manifest = build_code_retirement_reference_audit(tmp_path)
    candidates = {
        candidate["relative_path"]: candidate
        for candidate in manifest["candidates"]
    }

    assert manifest["candidate_count"] == 17
    assert candidates["quant_investor/kronos_predictor.py"]["exists"] is False
    assert candidates["quant_investor/kronos_predictor.py"]["reference_count"] == 0


def test_phase7_retired_modules_are_removed_and_unreferenced():
    manifest = build_code_retirement_reference_audit(ROOT)
    candidates = {
        candidate["relative_path"]: candidate
        for candidate in manifest["candidates"]
    }

    assert PHASE7_RETIRED_MODULES <= set(candidates)
    for relative_path in PHASE7_RETIRED_MODULES:
        assert candidates[relative_path]["exists"] is False
        assert candidates[relative_path]["reference_count"] == 0


def test_code_retirement_audit_classifies_reference_contexts(tmp_path):
    (tmp_path / "quant_investor").mkdir()
    (tmp_path / "quant_investor" / "kronos_predictor.py").write_text(
        "# retired predictor\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "pipeline").mkdir()
    (tmp_path / "quant_investor" / "pipeline" / "mainline.py").write_text(
        "from quant_investor.kronos_predictor import KronosIntegrator\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_kronos_legacy.py").write_text(
        "import quant_investor.kronos_predictor\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "legacy.md").write_text(
        "`quant_investor/kronos_predictor.py` is retired.\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "workspace_layout.py").write_text(
        "Path('quant_investor') / 'kronos_predictor.py'\n",
        encoding="utf-8",
    )

    manifest = build_code_retirement_reference_audit(tmp_path)
    candidates = {
        candidate["relative_path"]: candidate
        for candidate in manifest["candidates"]
    }
    candidate = candidates["quant_investor/kronos_predictor.py"]

    assert manifest["schema_version"] == "myquant.code_retirement_reference_audit.v1"
    assert candidate["module_name"] == "quant_investor.kronos_predictor"
    assert candidate["reference_summary"] == {
        "cleanup_manifest_reference": 1,
        "docs_reference": 1,
        "production_reference": 1,
        "test_reference": 1,
    }
    assert candidate["production_reference_count"] == 1
    assert candidate["reference_count"] == 4


def test_code_retirement_audit_writes_json_and_markdown(tmp_path):
    (tmp_path / "quant_investor").mkdir()
    (tmp_path / "quant_investor" / "intelligence.py").write_text(
        "# retired intelligence layer\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_intelligence_legacy.py").write_text(
        "from quant_investor.intelligence import IntelligenceLayerEngine\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports" / "project_cleanup" / "retirement"
    paths = write_code_retirement_reference_audit(tmp_path, output_dir=output_dir)

    payload = json.loads((output_dir / "code_retirement_reference_audit.json").read_text())
    assert payload["schema_version"] == "myquant.code_retirement_reference_audit.v1"
    assert paths["json"].endswith("code_retirement_reference_audit.json")
    assert paths["md"].endswith("code_retirement_reference_audit.md")
    assert (output_dir / "code_retirement_reference_audit.md").read_text().startswith(
        "# Code Retirement Reference Audit"
    )


def test_code_retirement_audit_does_not_match_generic_branch_words(tmp_path):
    (tmp_path / "quant_investor").mkdir()
    (tmp_path / "quant_investor" / "intelligence.py").write_text(
        "# retired intelligence layer\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "_vendor" / "chronos").mkdir(parents=True)
    (tmp_path / "quant_investor" / "market").mkdir()
    (tmp_path / "quant_investor" / "market" / "intelligence_mart.py").write_text(
        "branch_name = 'intelligence'\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "market" / "chronos_note.py").write_text(
        "import chronos\n",
        encoding="utf-8",
    )

    manifest = build_code_retirement_reference_audit(tmp_path)
    candidates = {
        candidate["relative_path"]: candidate
        for candidate in manifest["candidates"]
    }

    assert candidates["quant_investor/intelligence.py"]["reference_count"] == 0
    assert candidates["quant_investor/_vendor/chronos"]["reference_count"] == 0


def test_code_retirement_audit_distinguishes_same_stem_modules(tmp_path):
    (tmp_path / "quant_investor" / "agents" / "subagents").mkdir(parents=True)
    (tmp_path / "quant_investor" / "agents" / "kline_agent.py").write_text(
        "# retired kline agent\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "agents" / "subagents" / "kline_agent.py").write_text(
        "# retired kline subagent\n",
        encoding="utf-8",
    )
    (tmp_path / "quant_investor" / "agent_orchestrator.py").write_text(
        "from quant_investor.agents.kline_agent import KlineAgent\n",
        encoding="utf-8",
    )

    manifest = build_code_retirement_reference_audit(tmp_path)
    candidates = {
        candidate["relative_path"]: candidate
        for candidate in manifest["candidates"]
    }

    assert candidates["quant_investor/agents/kline_agent.py"]["production_reference_count"] == 1
    assert (
        candidates["quant_investor/agents/subagents/kline_agent.py"]["production_reference_count"]
        == 0
    )
