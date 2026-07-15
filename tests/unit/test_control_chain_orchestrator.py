from __future__ import annotations

import importlib
import sys

import pytest

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.branch_contracts import BranchResult
from quant_investor.control_chain import AgentOrchestrator, ControlChainOrchestrator


def test_control_chain_orchestrator_is_the_canonical_control_chain_entrypoint():
    assert AgentOrchestrator is ControlChainOrchestrator


def test_legacy_agent_orchestrator_module_warns_and_reexports():
    sys.modules.pop("quant_investor.agent_orchestrator", None)

    with pytest.warns(DeprecationWarning, match="control_chain"):
        legacy_module = importlib.import_module("quant_investor.agent_orchestrator")

    assert legacy_module.AgentOrchestrator is ControlChainOrchestrator
    assert legacy_module.ControlChainOrchestrator is ControlChainOrchestrator


def test_structured_research_rejects_retired_branch_keys():
    verdict = BranchVerdict(
        agent_name="retired",
        thesis="legacy payload",
        final_score=0.0,
        final_confidence=0.0,
    )

    with pytest.raises(ValueError, match="非 v14 canonical branch: intelligence"):
        ControlChainOrchestrator._normalize_research_by_symbol(
            {"000001.SZ": {"intelligence": verdict}}
        )


def test_precomputed_research_rejects_retired_branch_keys():
    with pytest.raises(ValueError, match="not a v14 branch"):
        BranchResult(branch_name="intelligence")
    with pytest.raises(ValueError, match="非 v14 canonical branch: intelligence"):
        ControlChainOrchestrator._validate_branch_names(
            {"intelligence": object()},
            context="branch_results",
        )


def test_structured_research_accepts_exact_v14_three_branch_map():
    branch_map = {
        branch_name: BranchVerdict(
            agent_name=branch_name,
            thesis=f"{branch_name} payload",
            final_score=0.0,
            final_confidence=0.5,
        )
        for branch_name in ("quant", "fundamental", "macro")
    }

    normalized = ControlChainOrchestrator._normalize_research_by_symbol(
        {"000001.SZ": branch_map}
    )

    assert tuple(normalized["000001.SZ"]) == ("quant", "fundamental", "macro")
