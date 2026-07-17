"""Shared evidence receipt for trained Bayesian v16 components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

TARGET_DEFINITION = "CN_20D_NET_EXCESS_VS_CSI300_GT_0"
BENCHMARK = "CSI300"
HORIZON_DAYS = 20
TRAINING_RECEIPT_SCHEMA_VERSION = "v16.bayesian-training-receipt.v1"


@dataclass(frozen=True)
class TrainingReceipt:
    receipt_id: str
    evidence_sha256: str
    training_start: str
    training_end: str
    sample_count: int
    purged: bool
    embargo_complete: bool
    embargo_days: int
    target_definition: str = TARGET_DEFINITION
    benchmark: str = BENCHMARK
    horizon_days: int = HORIZON_DAYS
    lookback_years: int = 5
    schema_version: str = TRAINING_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RECEIPT_SCHEMA_VERSION:
            raise ValueError("Bayesian v16 training receipt schema mismatch.")
        if not str(self.receipt_id).strip():
            raise ValueError("Training receipt_id must be non-empty.")
        digest = str(self.evidence_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Training evidence_sha256 must be lowercase SHA-256 hex.")
        if self.target_definition != TARGET_DEFINITION:
            raise ValueError("Training target must be 20D net excess return vs CSI300 > 0.")
        if (
            self.benchmark != BENCHMARK
            or isinstance(self.horizon_days, bool)
            or self.horizon_days != HORIZON_DAYS
        ):
            raise ValueError("Training receipt must use CSI300 and a 20-day horizon.")
        if isinstance(self.lookback_years, bool) or self.lookback_years != 5:
            raise ValueError("Bayesian v16 training evidence must cover five years.")
        try:
            start = date.fromisoformat(self.training_start)
            end = date.fromisoformat(self.training_end)
        except ValueError as exc:
            raise ValueError("Training dates must use YYYY-MM-DD.") from exc
        try:
            expected_start = end.replace(year=end.year - 5)
        except ValueError:
            expected_start = end.replace(year=end.year - 5, day=28)
        if start != expected_start:
            raise ValueError("Bayesian v16 training evidence must span exactly five years.")
        if not isinstance(self.purged, bool) or not isinstance(self.embargo_complete, bool):
            raise ValueError("Training purge and embargo flags must be bool.")
        if not self.purged or not self.embargo_complete:
            raise ValueError("Bayesian v16 training must complete purge and embargo.")
        if isinstance(self.embargo_days, bool) or self.embargo_days < self.horizon_days:
            raise ValueError("Training embargo must be at least the target horizon.")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("Training sample_count must be positive.")

    def require_sample_count(self, actual: int) -> None:
        if actual != self.sample_count:
            raise ValueError(
                "Training receipt sample_count mismatch: "
                f"receipt={self.sample_count}, actual={actual}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "evidence_sha256": self.evidence_sha256,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "sample_count": self.sample_count,
            "target_definition": self.target_definition,
            "benchmark": self.benchmark,
            "horizon_days": self.horizon_days,
            "lookback_years": self.lookback_years,
            "purged": self.purged,
            "embargo_complete": self.embargo_complete,
            "embargo_days": self.embargo_days,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingReceipt":
        expected = {
            "schema_version",
            "receipt_id",
            "evidence_sha256",
            "training_start",
            "training_end",
            "sample_count",
            "target_definition",
            "benchmark",
            "horizon_days",
            "lookback_years",
            "purged",
            "embargo_complete",
            "embargo_days",
        }
        if set(payload) != expected:
            raise ValueError("Training receipt payload keys mismatch.")
        if not isinstance(payload["purged"], bool) or not isinstance(
            payload["embargo_complete"], bool
        ):
            raise ValueError("Training receipt flags must be bool.")
        integer_fields = (
            "sample_count",
            "horizon_days",
            "lookback_years",
            "embargo_days",
        )
        if any(
            isinstance(payload[field_name], bool) or not isinstance(payload[field_name], int)
            for field_name in integer_fields
        ):
            raise ValueError("Training receipt numeric fields must be integers.")
        return cls(
            schema_version=str(payload["schema_version"]),
            receipt_id=str(payload["receipt_id"]),
            evidence_sha256=str(payload["evidence_sha256"]),
            training_start=str(payload["training_start"]),
            training_end=str(payload["training_end"]),
            sample_count=payload["sample_count"],
            target_definition=str(payload["target_definition"]),
            benchmark=str(payload["benchmark"]),
            horizon_days=payload["horizon_days"],
            lookback_years=payload["lookback_years"],
            purged=payload["purged"],
            embargo_complete=payload["embargo_complete"],
            embargo_days=payload["embargo_days"],
        )


__all__ = [
    "BENCHMARK",
    "HORIZON_DAYS",
    "TARGET_DEFINITION",
    "TRAINING_RECEIPT_SCHEMA_VERSION",
    "TrainingReceipt",
]
