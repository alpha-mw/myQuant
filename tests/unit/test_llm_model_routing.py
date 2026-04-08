from __future__ import annotations

from pathlib import Path

from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.pipeline.mainline import QuantInvestor


ROOT = Path(__file__).resolve().parents[2]


def test_mainline_uses_shared_branch_model_and_distinct_master_model():
    agent_model, master_model = QuantInvestor._resolve_agent_models(
        agent_model="moonshot-v1-32k",
        master_model="deepseek-chat",
    )

    orchestrator = AgentOrchestrator(branch_model=agent_model, master_model=master_model)

    assert orchestrator.branch_model == "moonshot-v1-32k"
    assert orchestrator.master_model == "deepseek-chat"


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
