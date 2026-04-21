"""Hierarchical prior builder for the Bayesian decision layer.

Priors encode background belief about a symbol's probability of outperformance
before any branch evidence is considered.
"""

from __future__ import annotations

import math
from typing import Any

from quant_investor.agent_protocol import GlobalContext
from quant_investor.bayesian.types import PriorSet

# Regime -> base-rate mapping.  These represent the unconditional probability
# that a randomly-selected stock outperforms the market in this regime.
_REGIME_BASE_RATES: dict[str, float] = {
    "趋势上涨": 0.55,
    "趋势下跌": 0.35,
    "震荡低波": 0.48,
    "震荡高波": 0.42,
    "未知": 0.45,
}

# Weights for compositing the hierarchical prior.
_PRIOR_WEIGHTS: dict[str, float] = {
    "market": 0.30,
    "regime": 0.25,
    "sector": 0.15,
    "tradability": 0.15,
    "data_quality": 0.15,
}


def _clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, value))


class HierarchicalPriorBuilder:
    """Build a hierarchical prior for a single symbol."""

    def build_prior(
        self,
        symbol: str,
        global_context: GlobalContext,
    ) -> PriorSet:
        regime_label = global_context.macro_regime or "未知"

        # 1. Market prior — overall market base rate
        market_prior = _REGIME_BASE_RATES.get(regime_label, 0.45)

        # 2. Regime prior — same as market but can be overridden
        regime_prior = _REGIME_BASE_RATES.get(regime_label, 0.45)

        # 3. Sector / style prior
        sector_prior = self._sector_prior(symbol, global_context)

        # 4. Tradability prior
        tradability_prior = self._tradability_prior(symbol, global_context)

        # 5. Data quality prior
        data_quality_prior = self._data_quality_prior(symbol, global_context)

        # Composite: weighted geometric mean (in log space)
        weights = _PRIOR_WEIGHTS
        log_sum = (
            weights["market"] * math.log(market_prior)
            + weights["regime"] * math.log(regime_prior)
            + weights["sector"] * math.log(sector_prior)
            + weights["tradability"] * math.log(tradability_prior)
            + weights["data_quality"] * math.log(data_quality_prior)
        )
        composite = _clamp(math.exp(log_sum))

        return PriorSet(
            market_prior=market_prior,
            regime_prior=regime_prior,
            sector_prior=sector_prior,
            tradability_prior=tradability_prior,
            data_quality_prior=data_quality_prior,
            composite_prior=composite,
        )

    @staticmethod
    def _sector_prior(symbol: str, ctx: GlobalContext) -> float:
        """Derive sector prior from style/industry exposure data."""
        selection_profile = str(
            (ctx.metadata or {}).get("selection_profile", {}).get("funnel_profile", "classic")
            if isinstance((ctx.metadata or {}).get("selection_profile", {}), dict)
            else "classic"
        ).strip().lower()
        if selection_profile != "momentum_leader":
            return 0.50
        exposures = ctx.style_exposures or {}
        if not exposures:
            return 0.50  # neutral if no sector data
        sector_score = exposures.get(symbol, exposures.get("default", 0.50))
        if isinstance(sector_score, dict):
            sector_score = sector_score.get("prior", 0.50)
        return _clamp(float(sector_score))

    @staticmethod
    def _tradability_prior(symbol: str, ctx: GlobalContext) -> float:
        """Penalize symbols that are near illiquidity thresholds."""
        suspended = set(ctx.liquidity_filter.get("suspended", []))
        if symbol in suspended:
            return 0.10
        illiquid = set(ctx.liquidity_filter.get("illiquid", []))
        if symbol in illiquid:
            return 0.20
        scores = ctx.liquidity_filter.get("liquidity_scores", {})
        if scores and symbol in scores:
            # Map percentile rank to prior: higher liquidity -> higher prior
            return _clamp(0.30 + 0.40 * float(scores[symbol]))
        return 0.50

    @staticmethod
    def _data_quality_prior(symbol: str, ctx: GlobalContext) -> float:
        """Penalize symbols with known data quality issues."""
        if symbol in ctx.data_quality_quarantine:
            return 0.15
        for issue in ctx.data_quality_issues:
            if issue.symbol == symbol and issue.severity in ("error", "critical"):
                return 0.25
        return 0.50
