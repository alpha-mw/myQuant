"""
Three-layer architecture contracts.

These are internal structure objects used by the global-context, per-symbol
research, and portfolio decision layers.
"""

from quant_investor.contracts.global_context import GlobalContext
from quant_investor.contracts.portfolio import PortfolioDecision
from quant_investor.contracts.selection import ShortlistItem
from quant_investor.contracts.symbol_research import SymbolResearchPacket

__all__ = [
    "GlobalContext",
    "PortfolioDecision",
    "ShortlistItem",
    "SymbolResearchPacket",
]
