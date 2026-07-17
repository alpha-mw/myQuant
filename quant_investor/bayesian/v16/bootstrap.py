"""Precomputed 1000-replicate time-block bootstrap intervals for v16."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from quant_investor.bayesian.v16.training import TrainingReceipt

BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.90
BOOTSTRAP_METHOD = "time-block-bootstrap.v1"


def _finite_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    materialized = tuple(float(value) for value in values)
    if len(materialized) != BOOTSTRAP_ITERATIONS:
        raise ValueError(f"{field_name} must contain exactly 1000 replicates.")
    if any(not math.isfinite(value) for value in materialized):
        raise ValueError(f"{field_name} contains a non-finite replicate.")
    return materialized


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _sigmoid(log_odds: float) -> float:
    if log_odds >= 20.0:
        return 1.0 - 1e-8
    if log_odds <= -20.0:
        return 1e-8
    return 1.0 / (1.0 + math.exp(-log_odds))


@dataclass(frozen=True)
class BlockBootstrapArtifact:
    """Offsets produced by 1000 chronological block-bootstrap refits."""

    artifact_id: str
    artifact_sha256: str
    receipt: TrainingReceipt
    block_length_days: int
    block_count: int
    win_rate_logit_offsets: tuple[float, ...]
    expected_alpha_offsets: tuple[float, ...]
    iterations: int = BOOTSTRAP_ITERATIONS
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL
    method: str = BOOTSTRAP_METHOD

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, TrainingReceipt):
            raise TypeError("Bootstrap artifact requires a TrainingReceipt.")
        if not str(self.artifact_id).strip():
            raise ValueError("Bootstrap artifact_id must be non-empty.")
        digest = str(self.artifact_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Bootstrap artifact_sha256 must be lowercase SHA-256 hex.")
        if self.method != BOOTSTRAP_METHOD:
            raise ValueError("v16 intervals require time-block bootstrap.")
        if self.iterations != BOOTSTRAP_ITERATIONS:
            raise ValueError("v16 intervals require exactly 1000 bootstrap iterations.")
        if not math.isclose(
            self.confidence_level,
            BOOTSTRAP_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("v16 bootstrap confidence level must equal 0.90.")
        if self.block_length_days <= 0 or self.block_count < 2:
            raise ValueError("Time-block bootstrap needs positive blocks and at least two blocks.")
        object.__setattr__(
            self,
            "win_rate_logit_offsets",
            _finite_tuple(self.win_rate_logit_offsets, "win_rate_logit_offsets"),
        )
        object.__setattr__(
            self,
            "expected_alpha_offsets",
            _finite_tuple(self.expected_alpha_offsets, "expected_alpha_offsets"),
        )

    def intervals(
        self,
        *,
        posterior_logit: float,
        expected_alpha: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if not math.isfinite(posterior_logit) or not math.isfinite(expected_alpha):
            raise ValueError("Bootstrap point estimates must be finite.")
        lower_probability = (1.0 - self.confidence_level) / 2.0
        upper_probability = 1.0 - lower_probability
        win_rate_replicates = tuple(
            _sigmoid(posterior_logit + offset) for offset in self.win_rate_logit_offsets
        )
        alpha_replicates = tuple(expected_alpha + offset for offset in self.expected_alpha_offsets)
        return (
            (
                _quantile(win_rate_replicates, lower_probability),
                _quantile(win_rate_replicates, upper_probability),
            ),
            (
                _quantile(alpha_replicates, lower_probability),
                _quantile(alpha_replicates, upper_probability),
            ),
        )


__all__ = [
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_METHOD",
    "BlockBootstrapArtifact",
]
