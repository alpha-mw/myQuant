"""Receipt-bound robust return models for Bayesian v16 expected alpha."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol, runtime_checkable

from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.training import TrainingReceipt

ROBUST_RETURN_MODEL_TYPE = "huber-regression.v1"


def _finite(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _logit(probability: float) -> float:
    value = _finite(probability, "probability")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Return-model probabilities must be in [0, 1].")
    value = max(1e-8, min(1.0 - 1e-8, value))
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class ReturnCalibrationEstimate:
    expected_alpha: float
    equal_weight_evidence: float
    correlation_adjusted_equal_weight_evidence: float

    def __post_init__(self) -> None:
        _finite(self.expected_alpha, "expected_alpha")
        _finite(self.equal_weight_evidence, "equal_weight_evidence")
        _finite(
            self.correlation_adjusted_equal_weight_evidence,
            "correlation_adjusted_equal_weight_evidence",
        )


@runtime_checkable
class ReturnCalibrationModel(Protocol):
    def estimate(
        self,
        *,
        branch_probabilities: Mapping[str, float],
        base_rate: float,
        vif_shrink: float,
    ) -> ReturnCalibrationEstimate: ...


@dataclass(frozen=True)
class RobustReturnModelArtifact:
    """Parameters emitted by an independently trained robust return model."""

    artifact_id: str
    parameters_sha256: str
    receipt: TrainingReceipt
    intercept: float
    aggregate_coefficient: float
    model_type: str = ROBUST_RETURN_MODEL_TYPE

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, TrainingReceipt):
            raise TypeError("Return model artifact requires a TrainingReceipt.")
        if self.model_type != ROBUST_RETURN_MODEL_TYPE:
            raise ValueError("v16 expected alpha requires the robust return model type.")
        if not str(self.artifact_id).strip():
            raise ValueError("Return model artifact_id must be non-empty.")
        digest = str(self.parameters_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Return model parameters_sha256 must be lowercase SHA-256 hex.")
        _finite(self.intercept, "return_model.intercept")
        _finite(
            self.aggregate_coefficient,
            "return_model.aggregate_coefficient",
        )


class ArtifactReturnCalibration:
    """Evaluate an independently trained robust-model artifact offline."""

    def __init__(self, artifact: RobustReturnModelArtifact) -> None:
        if not isinstance(artifact, RobustReturnModelArtifact):
            raise TypeError("ArtifactReturnCalibration requires a trained artifact.")
        self.artifact = artifact

    def estimate(
        self,
        *,
        branch_probabilities: Mapping[str, float],
        base_rate: float,
        vif_shrink: float,
    ) -> ReturnCalibrationEstimate:
        if set(branch_probabilities) != set(CANONICAL_BRANCH_ORDER):
            raise ValueError("branch_probabilities must contain exactly the four v16 branches.")
        shrink = _finite(vif_shrink, "vif_shrink")
        if not 0.0 < shrink <= 1.0:
            raise ValueError("vif_shrink must be in (0, 1].")
        base_logit = _logit(base_rate)
        equal_weight_evidence = 0.25 * sum(
            _logit(branch_probabilities[branch_name]) - base_logit
            for branch_name in CANONICAL_BRANCH_ORDER
        )
        adjusted_evidence = equal_weight_evidence * shrink
        expected_alpha = (
            float(self.artifact.intercept)
            + float(self.artifact.aggregate_coefficient) * adjusted_evidence
        )
        return ReturnCalibrationEstimate(
            expected_alpha=expected_alpha,
            equal_weight_evidence=equal_weight_evidence,
            correlation_adjusted_equal_weight_evidence=adjusted_evidence,
        )


__all__ = [
    "ArtifactReturnCalibration",
    "ROBUST_RETURN_MODEL_TYPE",
    "ReturnCalibrationEstimate",
    "ReturnCalibrationModel",
    "RobustReturnModelArtifact",
]
