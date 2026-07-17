"""Receipt-bound single base-rate prior for Bayesian v16."""

from __future__ import annotations

from dataclasses import dataclass

from quant_investor.bayesian.v16.training import TrainingReceipt
from quant_investor.bayesian.v16.types import PriorSet


@dataclass(frozen=True)
class BaseRateEvidence:
    positive_count: int
    total_count: int
    receipt: TrainingReceipt
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or isinstance(self.positive_count, bool)
            or not isinstance(self.positive_count, int)
            or self.total_count <= 0
            or not 0 <= self.positive_count <= self.total_count
        ):
            raise ValueError("Base-rate evidence counts are invalid.")
        self.receipt.require_sample_count(self.total_count)
        if self.beta_prior_alpha <= 0.0 or self.beta_prior_beta <= 0.0:
            raise ValueError("Base-rate beta-binomial smoothing must be positive.")

    @property
    def base_rate(self) -> float:
        return (self.positive_count + self.beta_prior_alpha) / (
            self.total_count + self.beta_prior_alpha + self.beta_prior_beta
        )


class BaseRatePriorBuilder:
    """Build the sole v16 prior from qualifying five-year evidence."""

    def __init__(self, evidence: BaseRateEvidence) -> None:
        if not isinstance(evidence, BaseRateEvidence):
            raise TypeError("BaseRatePriorBuilder requires BaseRateEvidence.")
        self.evidence = evidence

    def build_prior(
        self,
        symbol: str,
        global_context: object | None = None,
    ) -> PriorSet:
        del symbol, global_context
        return PriorSet(
            base_rate=self.evidence.base_rate,
            receipt=self.evidence.receipt,
            metadata={
                "source": "five_year_purged_embargoed_base_rate",
                "training_receipt_id": self.evidence.receipt.receipt_id,
                "training_evidence_sha256": self.evidence.receipt.evidence_sha256,
                "positive_count": self.evidence.positive_count,
                "total_count": self.evidence.total_count,
                "beta_prior_alpha": self.evidence.beta_prior_alpha,
                "beta_prior_beta": self.evidence.beta_prior_beta,
            },
        )


__all__ = ["BaseRateEvidence", "BaseRatePriorBuilder"]
