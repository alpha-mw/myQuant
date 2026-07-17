"""Strict four-branch likelihood mapper for Bayesian v16."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from quant_investor.bayesian.v16.calibration import CalibrationStore
from quant_investor.bayesian.v16.types import LikelihoodSet
from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
)
from quant_investor.v16.candidate_pipeline import FourBranchEvidence


@dataclass
class BranchLikelihoodInput:
    """Minimal schema-bound branch input accepted by the v16 mapper."""

    branch_name: str
    final_score: float
    final_confidence: float
    symbol_scores: dict[str, float] = field(default_factory=dict)
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    architecture_version: str = ARCHITECTURE_VERSION
    branch_schema_version: str = BRANCH_SCHEMA_VERSION

    def validate(self) -> None:
        if self.branch_name not in CANONICAL_BRANCH_ORDER:
            raise ValueError(f"Noncanonical v16 branch: {self.branch_name!r}.")
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                "v16 likelihood branch architecture mismatch: "
                f"expected {ARCHITECTURE_VERSION!r}, got {self.architecture_version!r}."
            )
        if self.branch_schema_version != BRANCH_SCHEMA_VERSION:
            raise ValueError(
                "v16 likelihood branch schema mismatch: "
                f"expected {BRANCH_SCHEMA_VERSION!r}, got {self.branch_schema_version!r}."
            )
        _finite_float(self.final_score, f"{self.branch_name}.final_score")
        _finite_float(
            self.final_confidence,
            f"{self.branch_name}.final_confidence",
            minimum=0.0,
            maximum=1.0,
        )


def _finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}; got {value!r}.")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}; got {value!r}.")
    return resolved


class SignalLikelihoodMapper:
    """Map exactly quant/fundamental/macro/llm into v16 likelihoods."""

    def __init__(
        self,
        *,
        calibration_store: CalibrationStore,
        correlation_matrix: Mapping[str, float] | None = None,
        recall_context: Mapping[str, Any] | None = None,
        global_context: Any | None = None,
    ) -> None:
        if not isinstance(calibration_store, CalibrationStore):
            raise TypeError("SignalLikelihoodMapper requires an evidence-trained CalibrationStore.")
        self.calibration_store = calibration_store
        self.correlation_matrix = dict(correlation_matrix or {})
        # Retained as non-evidentiary context for API stability.  Retrieval data
        # is intentionally never read when producing a v16 likelihood.
        self.recall_context = dict(recall_context or {})
        self.global_context = global_context

    @staticmethod
    def _fundamental_generation_is_explicitly_unconfirmed(
        result: BranchLikelihoodInput,
        symbol: str,
    ) -> bool:
        metadata = dict(result.metadata or {})
        statuses = metadata.get("fundamental_data_generation_status_by_symbol")
        generations = metadata.get("fundamental_data_generation_by_symbol")
        if statuses is None and generations is None:
            return False
        status_map = dict(statuses or {})
        generation_map = dict(generations or {})
        return not (
            str(generation_map.get(symbol) or "").strip()
            and str(status_map.get(symbol) or "").strip() == "confirmed"
        )

    def _branch_likelihood(
        self,
        branch_name: str,
        result: BranchLikelihoodInput,
        symbol: str,
    ) -> tuple[float, dict[str, float | str]]:
        score = _finite_float(
            result.symbol_scores.get(symbol, result.final_score),
            f"{branch_name}.score",
        )
        confidence = _finite_float(
            result.final_confidence,
            f"{branch_name}.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        calibration = self.calibration_store.calibration_stats(branch_name, score)
        likelihood = _finite_float(
            calibration["probability"],
            f"{branch_name}.calibrated_probability",
            minimum=0.0,
            maximum=1.0,
        )
        return likelihood, {
            "score": score,
            "confidence": confidence,
            "calibration_probability": float(calibration["probability"]),
            "calibration_sample_size": float(calibration["sample_size"]),
            "calibration_source": str(calibration["source"]),
        }

    def compute_likelihoods(
        self,
        *,
        branch_results: Mapping[str, BranchLikelihoodInput],
        symbol: str,
        candidate_symbols: set[str] | None = None,
    ) -> LikelihoodSet:
        """Return a complete v16 likelihood or fail closed.

        Missing, failed, swapped, or stale-schema branches are errors.  A
        non-candidate cannot be represented by synthetic neutral likelihoods.
        """

        expected = set(CANONICAL_BRANCH_ORDER)
        actual = set(branch_results)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            raise ValueError(
                "v16 likelihood mapping requires exactly all four canonical "
                "branches; " + "; ".join(details)
            )
        if candidate_symbols is not None and symbol not in candidate_symbols:
            raise ValueError("v16 likelihood mapping refuses a non-candidate neutral fallback.")

        branch_metadata: dict[str, dict[str, float | str]] = {}
        likelihood_values: dict[str, float] = {}
        for branch_name in CANONICAL_BRANCH_ORDER:
            result = branch_results[branch_name]
            if not isinstance(result, BranchLikelihoodInput):
                raise ValueError(
                    "v16 likelihood branch results must contain " "BranchLikelihoodInput objects."
                )
            result.validate()
            if result.branch_name != branch_name:
                raise ValueError(
                    "Likelihood branch result key/name mismatch: "
                    f"{branch_name!r} != {result.branch_name!r}."
                )
            if not result.success:
                raise ValueError(f"v16 likelihood branch {branch_name!r} is not successful.")
            if (
                branch_name == "fundamental"
                and self._fundamental_generation_is_explicitly_unconfirmed(result, symbol)
            ):
                raise ValueError(
                    "v16 likelihood requires confirmed Fundamental generation evidence."
                )
            likelihood, metadata = self._branch_likelihood(branch_name, result, symbol)
            likelihood_values[branch_name] = likelihood
            branch_metadata[branch_name] = metadata

        return LikelihoodSet(
            quant_likelihood=likelihood_values["quant"],
            fundamental_likelihood=likelihood_values["fundamental"],
            macro_likelihood=likelihood_values["macro"],
            llm_likelihood=likelihood_values["llm"],
            receipt=self.calibration_store.receipt,
            correlation_matrix=self.correlation_matrix,
            metadata={
                "evidence_sources": list(CANONICAL_BRANCH_ORDER),
                "branch_weights": {branch_name: 0.25 for branch_name in CANONICAL_BRANCH_ORDER},
                "branch_calibration": branch_metadata,
                "retrieval_evidence_used": False,
            },
        )

    def compute_from_sealed_evidence(
        self,
        evidence: FourBranchEvidence,
    ) -> LikelihoodSet:
        """Calibrate the formal sealed Q/F/M/LLM evidence, never retrieval notes."""

        if not isinstance(evidence, FourBranchEvidence):
            raise TypeError("evidence must be an exact FourBranchEvidence.")
        branch_results = {
            branch.branch: BranchLikelihoodInput(
                branch_name=branch.branch,
                final_score=branch.raw_score,
                final_confidence=branch.confidence,
                symbol_scores={evidence.symbol: branch.raw_score},
                metadata={"formal_evidence_ids": list(branch.evidence_ids)},
            )
            for branch in evidence.branches
        }
        return self.compute_likelihoods(
            branch_results=branch_results,
            symbol=evidence.symbol,
            candidate_symbols={evidence.symbol},
        )


__all__ = [
    "BranchLikelihoodInput",
    "SignalLikelihoodMapper",
]
