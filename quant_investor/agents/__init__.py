"""
V10.1 Multi-Agent 层。

为五路研究分支各配备专属 SubAgent（首席分析师），
风控模块配备增强版 Risk SubAgent，
由 IC Master Agent 主持多轮辩论产出最终投资建议。
"""

from quant_investor.agents.agent_contracts import (
    AgentEnhancedStrategy,
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    BranchAgentInput,
    BranchAgentOutput,
    FundamentalAgentInput,
    FundamentalAgentOutput,
    ICDebateRound,
    IntelligenceAgentInput,
    IntelligenceAgentOutput,
    KLineAgentInput,
    KLineAgentOutput,
    MacroAgentInput,
    MacroAgentOutput,
    MasterAgentInput,
    MasterAgentOutput,
    QuantAgentInput,
    QuantAgentOutput,
    RiskAgentInput,
    RiskAgentOutput,
    SymbolRecommendation,
)
from quant_investor.agents.llm_client import LLMCallError, LLMClient, has_any_provider
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.agents.orchestrator import AgentOrchestrator
from quant_investor.agents.subagent import BaseSubAgent, BranchSubAgent, RiskSubAgent
from quant_investor.agents.subagents import (
    FundamentalSubAgent,
    IntelligenceSubAgent,
    KLineSubAgent,
    MacroSubAgent,
    QuantSubAgent,
    SpecializedRiskSubAgent,
)

__all__ = [
    # Contracts
    "AgentEnhancedStrategy",
    "BaseBranchAgentInput",
    "BaseBranchAgentOutput",
    "BranchAgentInput",
    "BranchAgentOutput",
    "FundamentalAgentInput",
    "FundamentalAgentOutput",
    "ICDebateRound",
    "IntelligenceAgentInput",
    "IntelligenceAgentOutput",
    "KLineAgentInput",
    "KLineAgentOutput",
    "MacroAgentInput",
    "MacroAgentOutput",
    "MasterAgentInput",
    "MasterAgentOutput",
    "QuantAgentInput",
    "QuantAgentOutput",
    "RiskAgentInput",
    "RiskAgentOutput",
    "SymbolRecommendation",
    # LLM
    "LLMCallError",
    "LLMClient",
    "has_any_provider",
    # Agents
    "AgentOrchestrator",
    "BaseSubAgent",
    "BranchSubAgent",
    "FundamentalSubAgent",
    "IntelligenceSubAgent",
    "KLineSubAgent",
    "MacroSubAgent",
    "MasterAgent",
    "QuantSubAgent",
    "RiskSubAgent",
    "SpecializedRiskSubAgent",
]
