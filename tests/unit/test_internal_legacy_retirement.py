from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_production_paths_do_not_import_parallel_pipeline_directly():
    targets = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "agents" / "quant_agent.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "ParallelResearchPipeline" not in text, (
            f"{path} still depends on legacy batch pipeline"
        )


def test_v15_production_paths_do_not_import_retired_kline_or_kronos_runtime():
    targets = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "market" / "analyze.py",
        ROOT / "quant_investor" / "market" / "dag_executor.py",
        ROOT / "quant_investor" / "market" / "run_pipeline.py",
        ROOT / "quant_investor" / "agents" / "quant_agent.py",
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


def test_subagents_package_exports_only_v15_runtime_agents():
    sys.modules.pop("quant_investor.agents.subagents", None)
    sys.modules.pop("quant_investor.agents.subagents.kline_agent", None)

    subagents = importlib.import_module("quant_investor.agents.subagents")

    assert set(subagents.__all__) == {
        "FundamentalSubAgent",
        "MacroSubAgent",
        "QuantSubAgent",
        "SpecializedRiskSubAgent",
    }
    assert subagents.FundamentalSubAgent.__name__ == "FundamentalSubAgent"
    assert subagents.MacroSubAgent.__name__ == "MacroSubAgent"
    assert subagents.QuantSubAgent.__name__ == "QuantSubAgent"
    assert subagents.SpecializedRiskSubAgent.__name__ == "SpecializedRiskSubAgent"
    assert not hasattr(subagents, "KLineSubAgent")
    assert not hasattr(subagents, "IntelligenceSubAgent")
    assert "quant_investor.agents.subagents.kline_agent" not in sys.modules


def test_intelligence_agent_modules_are_physically_deleted():
    assert not (ROOT / "quant_investor" / "agents" / "intelligence_agent.py").exists()
    assert not (ROOT / "quant_investor" / "agents" / "subagents" / "intelligence_agent.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.agents.intelligence_agent")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.agents.subagents.intelligence_agent")


@pytest.mark.parametrize(
    "module_name",
    [
        "quant_investor.agents.intelligence_agent",
        "quant_investor.agents.subagents.intelligence_agent",
        "quant_investor.ensemble_judge",
        "quant_investor.market.intelligence_mart",
        "quant_investor.monitoring.intelligence_monitor",
    ],
)
def test_sourceless_loader_blocks_retired_modules(module_name, tmp_path):
    from quant_investor._sourceless import (
        _QuantInvestorSourcelessFinder,
        load_shadowed_module,
    )

    fake_pyc = tmp_path / "retired.pyc"
    fake_pyc.write_bytes(b"not a real pyc")

    with pytest.raises(ModuleNotFoundError, match="retired in v15"):
        _QuantInvestorSourcelessFinder().find_spec(module_name)
    with pytest.raises(ModuleNotFoundError, match="retired in v15"):
        load_shadowed_module(module_name, fake_pyc)


def test_intelligence_monitor_module_is_physically_deleted():
    assert not (ROOT / "quant_investor" / "monitoring" / "intelligence_monitor.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.monitoring.intelligence_monitor")


def test_current_protocol_constructors_reject_intelligence_fields():
    from pydantic import ValidationError

    from quant_investor.agent_protocol import BayesianDecisionRecord, ReportBundle
    from quant_investor.agents.agent_contracts import BaseBranchAgentInput

    with pytest.raises(TypeError):
        BayesianDecisionRecord(intelligence_likelihood=0.7)
    with pytest.raises(TypeError):
        ReportBundle(intelligence={"score": 0.7})
    with pytest.raises(ValidationError):
        BaseBranchAgentInput(
            branch_name="intelligence",
            base_score=0.1,
            final_score=0.1,
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        BaseBranchAgentInput(
            branch_name="quant",
            base_score=0.1,
            final_score=0.1,
            confidence=0.5,
            catalyst_summary={"legacy": True},
        )


def test_current_protocol_constructors_reject_nested_intelligence_maps():
    from pydantic import ValidationError

    from quant_investor.agent_protocol import (
        BayesianDecisionRecord,
        BranchOverlayVerdict,
        BranchVerdict,
        ReportBundle,
        StockReviewBundle,
    )
    from quant_investor.agents.agent_contracts import MasterAgentInput

    with pytest.raises(ValueError, match="non-v15 branch keys"):
        ReportBundle(branch_verdicts={"intelligence": BranchVerdict()})
    with pytest.raises(ValueError, match="non-v15 branch keys"):
        StockReviewBundle(branch_summaries={"intelligence": BranchVerdict()})
    with pytest.raises(ValueError, match="Non-v15 branch overlay"):
        BranchOverlayVerdict(branch_name="intelligence")
    with pytest.raises(ValueError, match="likelihood fields must match v15"):
        BayesianDecisionRecord(likelihoods={"intelligence_likelihood": 0.8})
    with pytest.raises(ValidationError, match="non-v15 branch keys"):
        MasterAgentInput(branch_results={"intelligence": {}})


def test_legacy_ensemble_judge_is_physically_deleted():
    assert not (ROOT / "quant_investor" / "ensemble_judge.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.ensemble_judge")


def test_review_prompt_defaults_are_v15_three_branch_only():
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
    assert "IntelligenceAgentInput" not in agent_contracts.__all__
    assert "IntelligenceAgentOutput" not in agent_contracts.__all__
    assert not hasattr(agent_contracts, "IntelligenceAgentInput")
    assert not hasattr(agent_contracts, "IntelligenceAgentOutput")


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
