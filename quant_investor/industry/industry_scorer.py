"""Exact forward industry score formula."""

from __future__ import annotations

from dataclasses import dataclass

from quant_investor.industry.industry_context import IndustryContext


@dataclass(frozen=True)
class IndustryScore:
    industry_id: str
    base_score: float
    confidence: float
    score: float


def score_industry_context(context: IndustryContext) -> IndustryScore:
    """Apply the exact revised [0, 1] industry formula and confidence."""

    if not isinstance(context, IndustryContext):
        raise ValueError("context must be IndustryContext")
    base_score = (
        0.15 * context.demand_score
        + 0.10 * context.supply_score
        + 0.10 * context.inventory_score
        + 0.10 * context.pricing_power
        + 0.05 * context.capex_score
        + 0.15 * context.earnings_revision
        + 0.10 * context.policy_score
        + 0.10 * context.market_confirmation
        + 0.05 * context.narrative_strength
        + 0.10 * (1.0 - context.crowding_risk)
    )
    return IndustryScore(
        industry_id=context.industry_id,
        base_score=base_score,
        confidence=context.confidence,
        score=base_score * context.confidence,
    )


industry_score = score_industry_context


__all__ = ["IndustryScore", "industry_score", "score_industry_context"]
