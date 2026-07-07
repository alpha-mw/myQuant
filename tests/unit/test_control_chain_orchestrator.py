from __future__ import annotations

import importlib
import sys

import pytest

from quant_investor.control_chain import AgentOrchestrator, ControlChainOrchestrator


def test_control_chain_orchestrator_is_the_canonical_control_chain_entrypoint():
    assert AgentOrchestrator is ControlChainOrchestrator


def test_legacy_agent_orchestrator_module_warns_and_reexports():
    sys.modules.pop("quant_investor.agent_orchestrator", None)

    with pytest.warns(DeprecationWarning, match="control_chain"):
        legacy_module = importlib.import_module("quant_investor.agent_orchestrator")

    assert legacy_module.AgentOrchestrator is ControlChainOrchestrator
    assert legacy_module.ControlChainOrchestrator is ControlChainOrchestrator
