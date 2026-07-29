"""Compatibility facade for the additive industry forward model."""

from quant_investor.industry.industry_context import IndustryContext
from quant_investor.industry.industry_evidence_store import (
    IndustryEvidence,
    IndustryEvidenceStore,
)
from quant_investor.industry.industry_scorer import (
    IndustryScore,
    industry_score,
    score_industry_context,
)

__all__ = [
    "IndustryContext",
    "IndustryEvidence",
    "IndustryEvidenceStore",
    "IndustryScore",
    "industry_score",
    "score_industry_context",
]
