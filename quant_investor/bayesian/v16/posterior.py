"""Recomputable v16 Bayesian posterior with equal branch evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from quant_investor.bayesian.v16.bootstrap import BlockBootstrapArtifact
from quant_investor.bayesian.v16.return_calibration import (
    ReturnCalibrationEstimate,
    ReturnCalibrationModel,
)
from quant_investor.bayesian.v16.types import (
    CANONICAL_CORRELATION_KEYS,
    LikelihoodSet,
    PosteriorResult,
    PriorSet,
)
from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER

_EPSILON = 1e-8
_EQUAL_BRANCH_WEIGHT = 0.25


def _safe_logit(probability: float) -> float:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability must be finite and in [0, 1]; got {probability!r}.")
    value = max(_EPSILON, min(1.0 - _EPSILON, value))
    return math.log(value / (1.0 - value))


def _sigmoid(log_odds: float) -> float:
    value = float(log_odds)
    if not math.isfinite(value):
        raise ValueError(f"Log odds must be finite; got {log_odds!r}.")
    if value >= 20.0:
        return 1.0 - _EPSILON
    if value <= -20.0:
        return _EPSILON
    return 1.0 / (1.0 + math.exp(-value))


def compute_equal_weight_evidence_increments(
    *,
    base_rate: float,
    branch_probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Compute ``0.25 * (logit(p_branch) - logit(p0))`` per branch."""

    if set(branch_probabilities) != set(CANONICAL_BRANCH_ORDER):
        raise ValueError("branch_probabilities must contain exactly the four v16 branches.")
    base_logit = _safe_logit(base_rate)
    return {
        branch_name: _EQUAL_BRANCH_WEIGHT
        * (_safe_logit(branch_probabilities[branch_name]) - base_logit)
        for branch_name in CANONICAL_BRANCH_ORDER
    }


def compute_correlation_vif(correlation_matrix: Mapping[str, float]) -> float:
    """Return the four-branch equal-weight VIF.

    ``VIF = max(1, 1 + (2/4) * sum(max(rho_ij, 0)))``.  Negative
    correlations never increase evidence strength; missing pairs mean zero
    correlation and are not missing branch likelihoods.
    """

    if set(correlation_matrix) != set(CANONICAL_CORRELATION_KEYS):
        missing = sorted(set(CANONICAL_CORRELATION_KEYS) - set(correlation_matrix))
        unexpected = sorted(set(correlation_matrix) - set(CANONICAL_CORRELATION_KEYS))
        raise ValueError(
            "correlation_matrix must contain exactly all six OOS branch pairs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    positive_correlation_sum = 0.0
    for pair, raw_correlation in correlation_matrix.items():
        correlation = float(raw_correlation)
        if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
            raise ValueError(f"Correlation {pair!r} must be finite and in [-1, 1].")
        positive_correlation_sum += max(correlation, 0.0)
    return max(
        1.0,
        1.0 + (2.0 / float(len(CANONICAL_BRANCH_ORDER))) * positive_correlation_sum,
    )


@dataclass(frozen=True)
class CostComponents:
    """Explicit cost decomposition; partial costs cannot produce an edge."""

    fee: float | None = None
    slippage: float | None = None
    market_impact: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("fee", "slippage", "market_impact"):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")

    @property
    def total(self) -> float | None:
        values = (self.fee, self.slippage, self.market_impact)
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values if value is not None)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "fee": self.fee,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
        }


class BayesianPosteriorEngine:
    """Combine four branch probabilities and calibrate returns separately."""

    def __init__(
        self,
        *,
        return_calibration_model: ReturnCalibrationModel,
        bootstrap_artifact: BlockBootstrapArtifact,
    ) -> None:
        self.return_calibration_model = return_calibration_model
        if not isinstance(self.return_calibration_model, ReturnCalibrationModel):
            raise TypeError("return_calibration_model must implement estimate().")
        if not isinstance(bootstrap_artifact, BlockBootstrapArtifact):
            raise TypeError(
                "BayesianPosteriorEngine requires a 1000-replicate " "BlockBootstrapArtifact."
            )
        self.bootstrap_artifact = bootstrap_artifact

    def compute_posterior(
        self,
        prior: PriorSet,
        likelihoods: LikelihoodSet,
        *,
        symbol: str = "",
        company_name: str = "",
        costs: CostComponents | None = None,
        regime: str | None = None,
        is_degraded: Mapping[str, bool] | None = None,
    ) -> PosteriorResult:
        """Compute the v16 posterior without policy or action outputs.

        ``regime`` is accepted only for caller transition and has no Bayesian
        effect.  A degraded branch fails closed instead of being replaced by a
        neutral likelihood.  Unless fee, slippage, and market impact are all
        supplied, edge after costs remains unknown (``None``).
        """

        del regime
        prior.validate()
        likelihoods.validate()
        degraded = dict(is_degraded or {})
        unexpected_degraded = sorted(set(degraded) - set(CANONICAL_BRANCH_ORDER))
        if unexpected_degraded:
            raise ValueError(
                "is_degraded has noncanonical branches: " + ", ".join(unexpected_degraded)
            )
        failed_branches = [
            branch_name
            for branch_name in CANONICAL_BRANCH_ORDER
            if degraded.get(branch_name, False)
        ]
        if failed_branches:
            raise ValueError(
                "v16 posterior refuses degraded-branch fallback: " + ", ".join(failed_branches)
            )

        branch_probabilities = dict(likelihoods.as_list())
        branch_increments = compute_equal_weight_evidence_increments(
            base_rate=prior.base_rate,
            branch_probabilities=branch_probabilities,
        )
        raw_increment = sum(
            branch_increments[branch_name] for branch_name in CANONICAL_BRANCH_ORDER
        )
        vif = compute_correlation_vif(likelihoods.correlation_matrix)
        vif_shrink = 1.0 / math.sqrt(vif)
        adjusted_increment = raw_increment * vif_shrink
        posterior_logit = _safe_logit(prior.base_rate) + adjusted_increment
        posterior_win_rate = _sigmoid(posterior_logit)

        return_estimate = self.return_calibration_model.estimate(
            branch_probabilities=branch_probabilities,
            base_rate=prior.base_rate,
            vif_shrink=vif_shrink,
        )
        if not isinstance(return_estimate, ReturnCalibrationEstimate):
            raise TypeError(
                "return_calibration_model.estimate() must return " "ReturnCalibrationEstimate."
            )
        win_rate_interval_90, expected_alpha_interval_90 = self.bootstrap_artifact.intervals(
            posterior_logit=posterior_logit,
            expected_alpha=return_estimate.expected_alpha,
        )
        resolved_costs = costs or CostComponents()
        if not isinstance(resolved_costs, CostComponents):
            raise TypeError("costs must be CostComponents or None.")
        edge_after_costs = (
            None
            if resolved_costs.total is None
            else return_estimate.expected_alpha - resolved_costs.total
        )

        return PosteriorResult(
            schema_version=PosteriorResult.__dataclass_fields__["schema_version"].default,
            symbol=symbol,
            company_name=company_name,
            prior=prior,
            likelihoods=likelihoods,
            posterior_win_rate=posterior_win_rate,
            posterior_expected_alpha=return_estimate.expected_alpha,
            posterior_edge_after_costs=edge_after_costs,
            posterior_win_rate_interval_90=win_rate_interval_90,
            posterior_expected_alpha_interval_90=expected_alpha_interval_90,
            raw_evidence_increment=raw_increment,
            correlation_adjusted_evidence_increment=adjusted_increment,
            correlation_vif=vif,
            correlation_vif_shrink=vif_shrink,
            branch_evidence_increments=branch_increments,
            evidence_sources=list(CANONICAL_BRANCH_ORDER),
            metadata={
                "formula": (
                    "logit(posterior)=logit(p0)+" "sum(0.25*(logit(p_branch)-logit(p0)))/sqrt(VIF)"
                ),
                "vif": vif,
                "lambda": vif_shrink,
                "costs": resolved_costs.to_dict(),
                "return_calibration_model": type(self.return_calibration_model).__name__,
                "return_model_equal_weight_evidence": (return_estimate.equal_weight_evidence),
                "return_model_correlation_adjusted_equal_weight_evidence": (
                    return_estimate.correlation_adjusted_equal_weight_evidence
                ),
                "bootstrap_artifact_id": self.bootstrap_artifact.artifact_id,
                "bootstrap_artifact_sha256": (self.bootstrap_artifact.artifact_sha256),
                "bootstrap_iterations": self.bootstrap_artifact.iterations,
                "bootstrap_method": self.bootstrap_artifact.method,
                "retrieval_evidence_used": False,
            },
        )


__all__ = [
    "BayesianPosteriorEngine",
    "CostComponents",
    "compute_correlation_vif",
    "compute_equal_weight_evidence_increments",
]
