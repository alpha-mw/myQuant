"""Evidence-trained equal-frequency likelihood calibration for v16."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Protocol, runtime_checkable

from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.training import TrainingReceipt

BUCKET_COUNT = 5
DEFAULT_MIN_SAMPLES_PER_BRANCH = 25


def _finite(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


@dataclass(frozen=True)
class CalibrationObservation:
    sample_id: str
    branch_name: str
    score: float
    positive_outcome: bool

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise ValueError("Calibration sample_id must be non-empty.")
        if self.branch_name not in CANONICAL_BRANCH_ORDER:
            raise ValueError(f"Unsupported v16 calibration branch: {self.branch_name!r}.")
        _finite(self.score, "CalibrationObservation.score")
        if not isinstance(self.positive_outcome, bool):
            raise ValueError("positive_outcome must be bool.")


@dataclass(frozen=True)
class CalibrationBucket:
    bucket_index: int
    lower_score: float
    upper_score: float
    sample_count: int
    positive_count: int
    calibrated_probability: float

    def __post_init__(self) -> None:
        if not 0 <= self.bucket_index < BUCKET_COUNT:
            raise ValueError("Calibration bucket_index is outside [0, 4].")
        lower = _finite(self.lower_score, "CalibrationBucket.lower_score")
        upper = _finite(self.upper_score, "CalibrationBucket.upper_score")
        if lower > upper:
            raise ValueError("Calibration bucket bounds are reversed.")
        if self.sample_count <= 0 or not 0 <= self.positive_count <= self.sample_count:
            raise ValueError("Calibration bucket counts are invalid.")
        probability = _finite(
            self.calibrated_probability,
            "CalibrationBucket.calibrated_probability",
        )
        if not 0.0 < probability < 1.0:
            raise ValueError("Smoothed calibration probability must be in (0, 1).")


@runtime_checkable
class LikelihoodCalibrationModel(Protocol):
    def calibration_stats(
        self,
        branch_name: str,
        score: float,
    ) -> Mapping[str, float | str]: ...


@dataclass(frozen=True)
class CalibrationStore:
    """Pure in-memory artifact built only from qualifying historical evidence."""

    receipt: TrainingReceipt
    buckets_by_branch: Mapping[str, tuple[CalibrationBucket, ...]]
    beta_prior_alpha: float
    beta_prior_beta: float
    min_samples_per_branch: int

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, TrainingReceipt):
            raise TypeError("CalibrationStore requires a TrainingReceipt.")
        if set(self.buckets_by_branch) != set(CANONICAL_BRANCH_ORDER):
            raise ValueError("CalibrationStore must contain all four v16 branches.")
        for branch_name in CANONICAL_BRANCH_ORDER:
            buckets = self.buckets_by_branch[branch_name]
            if len(buckets) != BUCKET_COUNT:
                raise ValueError("Each v16 branch must have exactly five buckets.")
            if tuple(bucket.bucket_index for bucket in buckets) != tuple(range(BUCKET_COUNT)):
                raise ValueError("Calibration buckets must be in canonical index order.")
            counts = [bucket.sample_count for bucket in buckets]
            if max(counts) - min(counts) > 1:
                raise ValueError("Calibration buckets are not equal-frequency.")
            for left, right in zip(buckets, buckets[1:]):
                if left.upper_score > right.lower_score:
                    raise ValueError("Calibration bucket score ordering is invalid.")
            if sum(counts) < self.min_samples_per_branch:
                raise ValueError("Calibration branch sample threshold is not met.")
        if self.beta_prior_alpha <= 0.0 or self.beta_prior_beta <= 0.0:
            raise ValueError("Beta-binomial smoothing parameters must be positive.")

    @classmethod
    def from_training_evidence(
        cls,
        observations: Iterable[CalibrationObservation],
        *,
        receipt: TrainingReceipt,
        min_samples_per_branch: int = DEFAULT_MIN_SAMPLES_PER_BRANCH,
        beta_prior_alpha: float = 1.0,
        beta_prior_beta: float = 1.0,
    ) -> "CalibrationStore":
        materialized = tuple(observations)
        if (
            isinstance(min_samples_per_branch, bool)
            or not isinstance(min_samples_per_branch, int)
            or min_samples_per_branch < BUCKET_COUNT
        ):
            raise ValueError("min_samples_per_branch must be at least five.")
        if beta_prior_alpha <= 0.0 or beta_prior_beta <= 0.0:
            raise ValueError("Beta-binomial smoothing parameters must be positive.")
        grouped: dict[str, list[CalibrationObservation]] = {
            branch_name: [] for branch_name in CANONICAL_BRANCH_ORDER
        }
        seen: set[tuple[str, str]] = set()
        for observation in materialized:
            if not isinstance(observation, CalibrationObservation):
                raise TypeError("Calibration evidence must contain CalibrationObservation.")
            key = (observation.branch_name, observation.sample_id)
            if key in seen:
                raise ValueError(f"Duplicate calibration sample: {key!r}.")
            seen.add(key)
            grouped[observation.branch_name].append(observation)

        buckets_by_branch: dict[str, tuple[CalibrationBucket, ...]] = {}
        for branch_name in CANONICAL_BRANCH_ORDER:
            rows = sorted(
                grouped[branch_name],
                key=lambda item: (item.score, item.sample_id),
            )
            receipt.require_sample_count(len(rows))
            if len(rows) < min_samples_per_branch:
                raise ValueError(
                    f"Calibration branch {branch_name!r} has insufficient samples: "
                    f"{len(rows)} < {min_samples_per_branch}."
                )
            base_size, remainder = divmod(len(rows), BUCKET_COUNT)
            branch_buckets: list[CalibrationBucket] = []
            offset = 0
            for bucket_index in range(BUCKET_COUNT):
                size = base_size + (1 if bucket_index < remainder else 0)
                chunk = rows[offset : offset + size]
                offset += size
                wins = sum(1 for row in chunk if row.positive_outcome)
                probability = (wins + beta_prior_alpha) / (
                    len(chunk) + beta_prior_alpha + beta_prior_beta
                )
                branch_buckets.append(
                    CalibrationBucket(
                        bucket_index=bucket_index,
                        lower_score=chunk[0].score,
                        upper_score=chunk[-1].score,
                        sample_count=len(chunk),
                        positive_count=wins,
                        calibrated_probability=probability,
                    )
                )
            buckets_by_branch[branch_name] = tuple(branch_buckets)
        return cls(
            receipt=receipt,
            buckets_by_branch=buckets_by_branch,
            beta_prior_alpha=float(beta_prior_alpha),
            beta_prior_beta=float(beta_prior_beta),
            min_samples_per_branch=min_samples_per_branch,
        )

    def calibration_stats(
        self,
        branch_name: str,
        score: float,
    ) -> dict[str, float | str]:
        if branch_name not in CANONICAL_BRANCH_ORDER:
            raise ValueError(f"Unsupported v16 calibration branch: {branch_name!r}.")
        value = _finite(score, "calibration score")
        buckets = self.buckets_by_branch[branch_name]
        selected = buckets[-1]
        for bucket in buckets:
            if value <= bucket.upper_score:
                selected = bucket
                break
        return {
            "bucket": str(selected.bucket_index),
            "probability": selected.calibrated_probability,
            "sample_size": float(selected.sample_count),
            "source": "five_year_equal_frequency_beta_binomial",
            "training_receipt_id": self.receipt.receipt_id,
        }


__all__ = [
    "BUCKET_COUNT",
    "CalibrationBucket",
    "CalibrationObservation",
    "CalibrationStore",
    "DEFAULT_MIN_SAMPLES_PER_BRANCH",
    "LikelihoodCalibrationModel",
]
