from __future__ import annotations

from quant_investor.agent_orchestrator import AgentOrchestrator, ControlChainOrchestrator


def test_control_chain_orchestrator_is_the_canonical_control_chain_entrypoint():
    assert AgentOrchestrator is ControlChainOrchestrator
