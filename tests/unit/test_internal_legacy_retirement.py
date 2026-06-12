from __future__ import annotations

import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_production_paths_do_not_import_parallel_pipeline_directly():
    targets = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "agents" / "quant_agent.py",
        ROOT / "quant_investor" / "agents" / "intelligence_agent.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "ParallelResearchPipeline" not in text, (
            f"{path} still depends on legacy batch pipeline"
        )


def test_v13_production_paths_do_not_import_retired_kline_or_kronos_runtime():
    targets = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "market" / "analyze.py",
        ROOT / "quant_investor" / "market" / "dag_executor.py",
        ROOT / "quant_investor" / "market" / "run_pipeline.py",
        ROOT / "quant_investor" / "agents" / "quant_agent.py",
        ROOT / "quant_investor" / "agents" / "intelligence_agent.py",
        ROOT / "quant_investor" / "branch_config.py",
    ]
    forbidden_markers = [
        "quant_investor.intelligence",
        "kronos_predictor",
        "kline_backends",
        "KronosIntegrator",
        "ChronosBackend",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in text, (
                f"{path} still imports retired runtime marker: {marker}"
            )


def test_subagents_package_exports_only_v13_runtime_agents():
    sys.modules.pop("quant_investor.agents.subagents", None)
    sys.modules.pop("quant_investor.agents.subagents.kline_agent", None)

    subagents = importlib.import_module("quant_investor.agents.subagents")

    assert set(subagents.__all__) == {
        "FundamentalSubAgent",
        "IntelligenceSubAgent",
        "MacroSubAgent",
        "QuantSubAgent",
        "SpecializedRiskSubAgent",
    }
    assert subagents.FundamentalSubAgent.__name__ == "FundamentalSubAgent"
    assert subagents.IntelligenceSubAgent.__name__ == "IntelligenceSubAgent"
    assert subagents.MacroSubAgent.__name__ == "MacroSubAgent"
    assert subagents.QuantSubAgent.__name__ == "QuantSubAgent"
    assert subagents.SpecializedRiskSubAgent.__name__ == "SpecializedRiskSubAgent"
    assert not hasattr(subagents, "KLineSubAgent")
    assert "quant_investor.agents.subagents.kline_agent" not in sys.modules


def test_review_prompt_defaults_are_v13_four_branch_only():
    from quant_investor.agents import prompts
    from quant_investor.versioning import CURRENT_BRANCH_ORDER

    assert set(prompts.BRANCH_SYSTEM_PROMPTS) == set(CURRENT_BRANCH_ORDER)
    assert set(prompts.CONVICTION_DEVIATION_CAP) == set(CURRENT_BRANCH_ORDER)
    assert set(prompts.BRANCH_OVERLAY_SCORE_CAP) == set(CURRENT_BRANCH_ORDER)
    assert set(prompts.BRANCH_OVERLAY_CONFIDENCE_CAP) == set(CURRENT_BRANCH_ORDER)
    assert "kline" not in prompts.MASTER_SYSTEM_PROMPT.lower()
    assert "kronos" not in prompts.MASTER_SYSTEM_PROMPT.lower()
    assert "chronos" not in prompts.MASTER_SYSTEM_PROMPT.lower()
    assert "5个分支" not in prompts.MASTER_SYSTEM_PROMPT
    assert "5 个分支" not in prompts.MASTER_SYSTEM_PROMPT


def test_agent_contract_default_exports_do_not_include_retired_kline_contracts():
    from quant_investor.agents import agent_contracts

    assert "KLineAgentInput" not in agent_contracts.__all__
    assert "KLineAgentOutput" not in agent_contracts.__all__
    assert hasattr(agent_contracts, "KLineAgentInput")
    assert hasattr(agent_contracts, "KLineAgentOutput")


def test_tests_no_longer_depend_on_legacy_batch_pipeline_or_mainline_helpers():
    forbidden_markers = [
        "from quant_investor.pipeline.parallel_research_pipeline import",
        "import quant_investor.pipeline.parallel_research_pipeline",
        "ParallelResearchPipeline(",
        "._run_review_layer(",
        "._run_unified_control_chain(",
    ]
    allowed = {
        ROOT / "tests" / "unit" / "test_internal_legacy_retirement.py",
    }

    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in text, (
                f"{path} still contains legacy dependency marker: {marker}"
            )


def test_current_architecture_docs_do_not_describe_retired_mainline():
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "architecture" / "research_pipeline_and_protocols.md",
        ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md",
        ROOT / "docs" / "modules" / "module_map.md",
    ]
    forbidden_markers = [
        "ParallelResearchPipeline",
        "parallel_research_pipeline.py",
        "Kronos Transformer + Amazon Chronos",
        "混合预测后端",
        "不提供架构切换参数和兼容别名",
        "kline_backends",
        "kline_agent.py",
    ]

    for path in docs:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in text, (
                f"{path} still describes retired mainline: {marker}"
            )


def test_tooling_config_does_not_reference_deleted_retired_modules():
    config = ROOT / "pyproject.toml"
    text = config.read_text(encoding="utf-8")
    forbidden_markers = [
        "kronos_predictor",
        "kline_backends",
        "intelligence\\.py",
    ]

    for marker in forbidden_markers:
        assert marker not in text, (
            f"{config} still references deleted retired module: {marker}"
        )
