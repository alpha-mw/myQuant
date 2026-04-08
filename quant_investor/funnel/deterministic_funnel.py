"""Deterministic funnel — compress full market to candidate set.

Consumes quant + kline BranchResults (full-market) and a GlobalContext,
then applies a pipeline of gates and ranking to produce a compressed
candidate set of ~300-500 symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.candidate_filter import (
    DataQualityGate,
    LiquidityGate,
    TradabilityGate,
)
from quant_investor.logger import get_logger

_logger = get_logger("DeterministicFunnel")


@dataclass
class FunnelConfig:
    """Tuning knobs for the deterministic funnel."""

    max_candidates: int = 400
    liquidity_percentile_min: float = 0.10
    min_composite_score: float = -1.0  # disabled by default
    quant_weight: float = 0.55
    kline_weight: float = 0.45


@dataclass
class FunnelOutput:
    """Result of the deterministic funnel pass."""

    candidates: list[str] = field(default_factory=list)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    excluded_symbols: dict[str, str] = field(default_factory=dict)
    funnel_metadata: dict[str, Any] = field(default_factory=dict)


class DeterministicFunnel:
    """Full-market first-pass engine.

    Pipeline:
    1. Data quality gate
    2. Tradability gate
    3. Liquidity gate
    4. Composite score ranking (quant + kline)
    5. Top-N cutoff
    """

    def __init__(self, config: FunnelConfig | None = None) -> None:
        self.config = config or FunnelConfig()

    def run(
        self,
        *,
        quant_result: BranchResult,
        kline_result: BranchResult,
        global_context: GlobalContext,
    ) -> FunnelOutput:
        all_symbols = list(global_context.universe_tiers.get("researchable", global_context.universe_symbols))
        all_excluded: dict[str, str] = {}

        # Gate 1: data quality
        symbols, excluded = DataQualityGate().filter(all_symbols, global_context)
        all_excluded.update(excluded)

        # Gate 2: tradability
        symbols, excluded = TradabilityGate().filter(symbols, global_context)
        all_excluded.update(excluded)

        # Gate 3: liquidity
        symbols, excluded = LiquidityGate(
            percentile_min=self.config.liquidity_percentile_min,
        ).filter(symbols, global_context)
        all_excluded.update(excluded)

        # Score: weighted composite of quant + kline symbol_scores
        quant_scores = quant_result.symbol_scores or {}
        kline_scores = kline_result.symbol_scores or {}
        qw = self.config.quant_weight
        kw = self.config.kline_weight

        composite: dict[str, float] = {}
        for symbol in symbols:
            qs = float(quant_scores.get(symbol, 0.0))
            ks = float(kline_scores.get(symbol, 0.0))
            composite[symbol] = qw * qs + kw * ks

        # Filter by minimum composite score
        if self.config.min_composite_score > -1.0:
            for symbol in list(composite):
                if composite[symbol] < self.config.min_composite_score:
                    all_excluded[symbol] = f"below_min_score_{self.config.min_composite_score}"
                    del composite[symbol]

        # Rank and cutoff
        ranked = sorted(composite.items(), key=lambda item: item[1], reverse=True)
        top_n = ranked[: self.config.max_candidates]
        candidates = [symbol for symbol, _ in top_n]
        candidate_scores = {symbol: score for symbol, score in top_n}

        _logger.info(
            "Funnel: %d total -> %d after gates -> %d candidates (max %d)",
            len(all_symbols),
            len(composite),
            len(candidates),
            self.config.max_candidates,
        )

        return FunnelOutput(
            candidates=candidates,
            candidate_scores=candidate_scores,
            excluded_symbols=all_excluded,
            funnel_metadata={
                "total_universe": len(all_symbols),
                "after_gates": len(composite),
                "final_candidates": len(candidates),
                "max_candidates": self.config.max_candidates,
                "quant_weight": qw,
                "kline_weight": kw,
                "excluded_count": len(all_excluded),
            },
        )
