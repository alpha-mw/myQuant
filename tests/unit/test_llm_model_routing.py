from __future__ import annotations

from pathlib import Path

from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.pipeline.mainline import QuantInvestor
from quant_investor.llm_provider_priority import resolve_runtime_role_models


ROOT = Path(__file__).resolve().parents[2]


def test_mainline_preserves_explicit_branch_and_master_models():
    branch_config, master_config = resolve_runtime_role_models(
        agent_model="qwen3.5-plus",
        master_model="deepseek-reasoner",
    )

    orchestrator = AgentOrchestrator(
        branch_model=branch_config.primary_model,
        master_model=master_config.primary_model,
    )

    assert orchestrator.branch_model == "qwen3.5-plus"
    assert orchestrator.master_model == "deepseek-reasoner"
    assert not hasattr(QuantInvestor, "_resolve_agent_models")


def test_mainline_paths_do_not_import_provider_sdks_directly():
    mainline_paths = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "agents" / "llm_client.py",
        ROOT / "quant_investor" / "agents" / "orchestrator.py",
        ROOT / "quant_investor" / "agents" / "subagent.py",
        ROOT / "quant_investor" / "agents" / "master_agent.py",
    ]

    forbidden_tokens = [
        "import openai",
        "from openai",
        "import anthropic",
        "from anthropic",
    ]

    for path in mainline_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} should route through quant_investor.llm_gateway"
