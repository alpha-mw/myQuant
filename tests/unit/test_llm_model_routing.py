from __future__ import annotations

from pathlib import Path

from quant_investor.agents.llm_client import _build_openai_compatible
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.pipeline.mainline import QuantInvestor
from quant_investor.llm_gateway import _build_openai_compatible_request


ROOT = Path(__file__).resolve().parents[2]


def test_quant_branch_keeps_llm_brainstorm_disabled_in_research_core():
    text = (ROOT / "quant_investor" / "pipeline" / "parallel_research_pipeline.py").read_text(encoding="utf-8")

    assert "enable_llm=False" in text


def test_mainline_uses_shared_branch_model_and_distinct_master_model():
    agent_model, master_model = QuantInvestor._resolve_agent_models(
        agent_model="claude-3-5-sonnet",
        master_model="gpt-5.4-mini",
    )

    orchestrator = AgentOrchestrator(branch_model=agent_model, master_model=master_model)

    assert orchestrator.branch_model == "claude-3-5-sonnet"
    assert orchestrator.master_model == "gpt-5.4-mini"


def test_mainline_paths_do_not_import_provider_sdks_directly():
    mainline_paths = [
        ROOT / "quant_investor" / "pipeline" / "parallel_research_pipeline.py",
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
        "google.generativeai",
    ]

    for path in mainline_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} should route through quant_investor.llm_gateway"


def test_agent_openai_request_uses_completion_tokens_for_gpt5_models():
    _url, _headers, body = _build_openai_compatible(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=256,
        response_json=False,
        api_key="test-key",
        base_url="https://api.openai.com/v1/chat/completions",
    )

    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body


def test_gateway_openai_request_keeps_max_tokens_for_non_gpt5_models():
    _url, _headers, body = _build_openai_compatible_request(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=128,
        response_json=False,
        api_key="test-key",
        base_url="https://api.openai.com/v1/chat/completions",
    )

    assert body["max_tokens"] == 128
    assert "max_completion_tokens" not in body
