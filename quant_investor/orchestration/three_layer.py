from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_investor.branch_contracts import BranchResult, TradeRecommendation, UnifiedDataBundle
from quant_investor.contracts import GlobalContext, PortfolioDecision, ShortlistItem, SymbolResearchPacket
from quant_investor.global_context.builder import GlobalContextBuilder
from quant_investor.portfolio.decision import build_portfolio_decisions
from quant_investor.selection.prefilter import build_shortlist_items
from quant_investor.symbol_research.builder import build_symbol_research_packets


class ThreeLayerOrchestrator:
    """Thin orchestration façade for the three-layer research DAG."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.global_context_builder = GlobalContextBuilder(cache_dir=cache_dir)

    def build_global_context(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        phase1_context: dict[str, Any] | None = None,
        branch_results: dict[str, BranchResult] | None = None,
        calibrated_signals: dict[str, Any] | None = None,
        risk_result: Any | None = None,
        force_refresh: bool = False,
    ) -> GlobalContext:
        return self.global_context_builder.build(
            data_bundle=data_bundle,
            phase1_context=phase1_context,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            risk_result=risk_result,
            force_refresh=force_refresh,
        )

    def build_symbol_packets(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        global_context: GlobalContext,
    ) -> dict[str, SymbolResearchPacket]:
        return build_symbol_research_packets(
            data_bundle=data_bundle,
            branch_results=branch_results,
            calibrated_signals=calibrated_signals,
            global_context=global_context,
        )

    def build_shortlist(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        symbol_packets: dict[str, SymbolResearchPacket],
        global_context: GlobalContext,
        max_items: int = 12,
    ) -> list[ShortlistItem]:
        return build_shortlist_items(
            data_bundle=data_bundle,
            symbol_packets=symbol_packets,
            global_context=global_context,
            max_items=max_items,
        )

    def build_portfolio_decisions(
        self,
        *,
        shortlist: list[ShortlistItem],
        trade_recommendations: list[TradeRecommendation],
        global_context: GlobalContext,
    ) -> list[PortfolioDecision]:
        return build_portfolio_decisions(
            shortlist=shortlist,
            trade_recommendations=trade_recommendations,
            global_context=global_context,
        )
