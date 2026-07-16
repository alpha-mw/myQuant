"""Bayesian decision layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from quant_investor.artifact_validation import (
    require_finite_structure,
    validate_posterior_numeric_fields,
)
from quant_investor.versioning import LIKELIHOOD_SCHEMA_VERSION


def _require_probability(value: float, field_name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return probability


def _require_finite_metadata(metadata: dict[str, Any]) -> None:
    numeric_fields = (
        "history_confidence",
        "avg_reliability",
        "recall_bias",
        "momentum_strength",
        "fake_breakout_penalty",
        "crowding_penalty",
        "market_pressure",
    )
    for field_name in numeric_fields:
        if field_name not in metadata:
            continue
        value = float(metadata[field_name])
        if not math.isfinite(value):
            raise ValueError(f"Likelihood metadata {field_name} must be finite.")
    branch_weights = metadata.get("branch_weights", {})
    if branch_weights and not isinstance(branch_weights, dict):
        raise ValueError("Likelihood metadata branch_weights must be a mapping.")
    unexpected_weight_branches = sorted(
        set(dict(branch_weights or {})) - {"quant", "fundamental"}
    )
    if unexpected_weight_branches:
        raise ValueError(
            "Likelihood metadata has unexpected branch weights: "
            + ", ".join(unexpected_weight_branches)
        )
    for branch_name, raw_weight in dict(branch_weights or {}).items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Likelihood metadata branch weight must be finite and non-negative: "
                f"{branch_name}={raw_weight!r}."
            )
    calibration_samples = metadata.get("calibration_samples", {})
    if calibration_samples and not isinstance(calibration_samples, dict):
        raise ValueError("Likelihood metadata calibration_samples must be a mapping.")
    unexpected_sample_branches = sorted(
        set(dict(calibration_samples or {})) - {"quant", "fundamental"}
    )
    if unexpected_sample_branches:
        raise ValueError(
            "Likelihood metadata has unexpected calibration branches: "
            + ", ".join(unexpected_sample_branches)
        )


def _require_empty_correlation_matrix(value: dict[str, float]) -> None:
    if value:
        raise ValueError(
            "v15 likelihood correlation_matrix must be empty; "
            "cross-likelihood correlation is not enabled."
        )


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

    def validate(self) -> None:
        for field_name in (
            "market_prior",
            "regime_prior",
            "sector_prior",
            "tradability_prior",
            "data_quality_prior",
            "composite_prior",
        ):
            setattr(self, field_name, _require_probability(getattr(self, field_name), field_name))

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "market_prior": self.market_prior,
            "regime_prior": self.regime_prior,
            "sector_prior": self.sector_prior,
            "tradability_prior": self.tradability_prior,
            "data_quality_prior": self.data_quality_prior,
            "composite_prior": self.composite_prior,
        }


@dataclass(init=False)
class LikelihoodSet:
    """Per-signal-family likelihood values for a single symbol."""

    schema_version: str = LIKELIHOOD_SCHEMA_VERSION
    quant_likelihood: float = 0.50
    fundamental_likelihood: float = 0.50
    correlation_matrix: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        schema_version: str = LIKELIHOOD_SCHEMA_VERSION,
        quant_likelihood: float = 0.50,
        fundamental_likelihood: float = 0.50,
        correlation_matrix: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if str(schema_version) != LIKELIHOOD_SCHEMA_VERSION:
            raise ValueError(
                "Likelihood schema mismatch: "
                f"expected {LIKELIHOOD_SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        self.schema_version = LIKELIHOOD_SCHEMA_VERSION
        self.quant_likelihood = _require_probability(
            quant_likelihood,
            "quant_likelihood",
        )
        self.fundamental_likelihood = _require_probability(
            fundamental_likelihood,
            "fundamental_likelihood",
        )
        resolved_correlation_matrix = dict(correlation_matrix or {})
        _require_empty_correlation_matrix(resolved_correlation_matrix)
        self.correlation_matrix = resolved_correlation_matrix
        self.metadata = dict(metadata or {})

    def validate(self) -> None:
        if self.schema_version != LIKELIHOOD_SCHEMA_VERSION:
            raise ValueError(
                "Likelihood schema mismatch: expected "
                f"{LIKELIHOOD_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        self.quant_likelihood = _require_probability(
            self.quant_likelihood,
            "quant_likelihood",
        )
        self.fundamental_likelihood = _require_probability(
            self.fundamental_likelihood,
            "fundamental_likelihood",
        )
        _require_empty_correlation_matrix(self.correlation_matrix)
        _require_finite_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "quant_likelihood": self.quant_likelihood,
            "fundamental_likelihood": self.fundamental_likelihood,
            "correlation_matrix": dict(self.correlation_matrix),
        }

    def as_list(self) -> list[tuple[str, float]]:
        return [
            ("quant", self.quant_likelihood),
            ("fundamental", self.fundamental_likelihood),
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

    def validate(self) -> None:
        self.prior.validate()
        self.likelihoods.validate()
        self.raw_posterior = _require_probability(self.raw_posterior, "raw_posterior")
        self.correlation_discounted_posterior = _require_probability(
            self.correlation_discounted_posterior,
            "correlation_discounted_posterior",
        )
        validate_posterior_numeric_fields(self)
        unexpected_sources = sorted(set(self.evidence_sources) - {"quant", "fundamental"})
        if unexpected_sources:
            raise ValueError(
                "PosteriorResult has unexpected evidence sources: "
                + ", ".join(unexpected_sources)
            )
        require_finite_structure(self.metadata, path="PosteriorResult.metadata")

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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
