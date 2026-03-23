"""
V10 Multi-Agent 层。

为五路研究分支 + 风控各配备专属 SubAgent，
由 IC Master Agent 综合辩论产出最终投资建议。
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
from quant_investor.agents.subagent import BranchSubAgent, RiskSubAgent

__all__ = [
    "AgentEnhancedStrategy",
    "AgentOrchestrator",
    "BranchAgentInput",
    "BranchAgentOutput",
    "BranchSubAgent",
    "LLMCallError",
    "LLMClient",
    "MasterAgent",
    "MasterAgentInput",
    "MasterAgentOutput",
    "RiskAgentInput",
    "RiskAgentOutput",
    "RiskSubAgent",
    "SymbolRecommendation",
    "has_any_provider",
]
