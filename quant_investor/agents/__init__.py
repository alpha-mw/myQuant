"""
V13 review-layer agent surface.

This package exposes advisory review-layer contracts and helpers. The runtime
research surface is the v13 four-branch set; deterministic control-chain gates
remain authoritative.
"""

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BranchAgentInput,
    BranchAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
    SymbolRecommendation,
)
from quant_investor.agents.llm_client import LLMClient, LLMCallError, has_any_provider
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.agents.stock_reviewers import BranchOverlayReviewer, MasterICAgent
from quant_investor.agents.subagent import BranchSubAgent, RiskSubAgent

__all__ = [
    "AgentEnhancedStrategy",
    "AgentOrchestrator",
    "BranchAgentInput",
    "BranchAgentOutput",
    "BranchSubAgent",
    "BranchOverlayReviewer",
    "LLMCallError",
    "LLMClient",
    "MasterAgent",
    "MasterAgentInput",
    "MasterAgentOutput",
    "MasterICAgent",
    "RiskAgentInput",
    "RiskAgentOutput",
    "RiskSubAgent",
    "SymbolRecommendation",
    "has_any_provider",
]
