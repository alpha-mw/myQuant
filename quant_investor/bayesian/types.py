"""Bayesian decision layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PriorSet:
    """Hierarchical prior components for a single symbol."""

    market_prior: float = 0.50
    regime_prior: float = 0.50
    sector_prior: float = 0.50
    tradability_prior: float = 0.50
    data_quality_prior: float = 0.50
    composite_prior: float = 0.50
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_prior": self.market_prior,
            "regime_prior": self.regime_prior,
            "sector_prior": self.sector_prior,
            "tradability_prior": self.tradability_prior,
            "data_quality_prior": self.data_quality_prior,
            "composite_prior": self.composite_prior,
        }


@dataclass
class LikelihoodSet:
    """Per-signal-family likelihood values for a single symbol."""

    kline_likelihood: float = 0.50
    quant_likelihood: float = 0.50
    fundamental_likelihood: float = 0.50
    intelligence_likelihood: float = 0.50
    correlation_matrix: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kline_likelihood": self.kline_likelihood,
            "quant_likelihood": self.quant_likelihood,
            "fundamental_likelihood": self.fundamental_likelihood,
            "intelligence_likelihood": self.intelligence_likelihood,
            "correlation_matrix": dict(self.correlation_matrix),
        }

    def as_list(self) -> list[tuple[str, float]]:
        return [
            ("kline", self.kline_likelihood),
            ("quant", self.quant_likelihood),
            ("fundamental", self.fundamental_likelihood),
            ("intelligence", self.intelligence_likelihood),
        ]


@dataclass
class PosteriorResult:
    """Full posterior output for a single symbol."""

    symbol: str = ""
    company_name: str = ""
    prior: PriorSet = field(default_factory=PriorSet)
    likelihoods: LikelihoodSet = field(default_factory=LikelihoodSet)
    raw_posterior: float = 0.50
    correlation_discounted_posterior: float = 0.50
    posterior_win_rate: float = 0.50
    posterior_expected_alpha: float = 0.0
    posterior_confidence: float = 0.50
    posterior_action_score: float = 0.0
    posterior_edge_after_costs: float = 0.0
    posterior_capacity_penalty: float = 0.0
    rank: int = 0
    coverage_discount: float = 0.0
    fallback_penalty: float = 0.0
    correlation_discount: float = 0.0
    data_quality_penalty: float = 0.0
    regime_adjustment: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    action_threshold_used: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "prior": self.prior.to_dict(),
            "likelihoods": self.likelihoods.to_dict(),
            "raw_posterior": self.raw_posterior,
            "correlation_discounted_posterior": self.correlation_discounted_posterior,
            "posterior_win_rate": self.posterior_win_rate,
            "posterior_expected_alpha": self.posterior_expected_alpha,
            "posterior_confidence": self.posterior_confidence,
            "posterior_action_score": self.posterior_action_score,
            "posterior_edge_after_costs": self.posterior_edge_after_costs,
            "posterior_capacity_penalty": self.posterior_capacity_penalty,
            "rank": self.rank,
            "coverage_discount": self.coverage_discount,
            "correlation_discount": self.correlation_discount,
            "evidence_sources": list(self.evidence_sources),
        }
