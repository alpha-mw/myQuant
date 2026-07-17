"""Strict v16 Bayesian data contracts.

The current runtime accepts one base-rate prior and exactly four likelihoods.
Historical v15 payloads must be handled outside this module; they cannot be
silently re-labelled as v16 values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from quant_investor.artifact_validation import require_finite_structure
from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.training import TrainingReceipt
from quant_investor.bayesian.v16.versioning import (
    LIKELIHOOD_SCHEMA_VERSION,
    POSTERIOR_SCHEMA_VERSION,
    PRIOR_SCHEMA_VERSION,
)

_RETIRED_POSTERIOR_KEYS = frozenset(
    {
        "posterior_action_score",
        "kill_switch",
        "action_threshold",
        "action_threshold_used",
    }
)


def _require_probability(value: float, field_name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return probability


def _require_finite(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _require_interval(
    value: Sequence[float],
    field_name: str,
    *,
    probability: bool,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly [lower, upper].")
    lower = _require_finite(value[0], f"{field_name}[0]")
    upper = _require_finite(value[1], f"{field_name}[1]")
    if lower > upper:
        raise ValueError(f"{field_name} lower bound must not exceed its upper bound.")
    if probability and (lower < 0.0 or upper > 1.0):
        raise ValueError(f"{field_name} must stay within [0, 1].")
    return (lower, upper)


def _reject_retired_posterior_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _RETIRED_POSTERIOR_KEYS:
                raise ValueError(f"{path}.{key} is retired in the v16 posterior contract.")
            _reject_retired_posterior_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_retired_posterior_keys(child, path=f"{path}[{index}]")


def correlation_key(branch_a: str, branch_b: str) -> str:
    """Return the deterministic key for one canonical unordered pair."""

    if branch_a == branch_b:
        raise ValueError("A branch cannot be correlated with itself.")
    try:
        index_a = CANONICAL_BRANCH_ORDER.index(branch_a)
        index_b = CANONICAL_BRANCH_ORDER.index(branch_b)
    except ValueError as exc:
        raise ValueError(
            f"Correlation branches must be canonical: {branch_a!r}, {branch_b!r}."
        ) from exc
    first, second = (branch_a, branch_b) if index_a < index_b else (branch_b, branch_a)
    return f"{first}_{second}"


CANONICAL_CORRELATION_KEYS = frozenset(
    correlation_key(branch_a, branch_b)
    for index, branch_a in enumerate(CANONICAL_BRANCH_ORDER)
    for branch_b in CANONICAL_BRANCH_ORDER[index + 1 :]
)


def _validate_correlation_matrix(value: Mapping[str, float]) -> dict[str, float]:
    matrix = {str(key): raw_value for key, raw_value in value.items()}
    missing = sorted(CANONICAL_CORRELATION_KEYS - set(matrix))
    unexpected = sorted(set(matrix) - CANONICAL_CORRELATION_KEYS)
    if missing or unexpected:
        raise ValueError(
            "Likelihood correlation_matrix must contain exactly all six OOS pairs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        key: _require_correlation(raw_value, f"correlation_matrix.{key}")
        for key, raw_value in matrix.items()
    }


def _require_correlation(value: float, field_name: str) -> float:
    correlation = _require_finite(value, field_name)
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1]; got {value!r}.")
    return correlation


@dataclass(init=False)
class PriorSet:
    """A single unconditional base rate, before branch evidence."""

    schema_version: str = PRIOR_SCHEMA_VERSION
    base_rate: float = field(init=False)
    receipt: TrainingReceipt = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        base_rate: float,
        receipt: TrainingReceipt,
        schema_version: str = PRIOR_SCHEMA_VERSION,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if str(schema_version) != PRIOR_SCHEMA_VERSION:
            raise ValueError(
                "Prior schema mismatch: "
                f"expected {PRIOR_SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        self.schema_version = PRIOR_SCHEMA_VERSION
        self.base_rate = _require_probability(base_rate, "base_rate")
        if not isinstance(receipt, TrainingReceipt):
            raise TypeError("v16 PriorSet requires a qualifying TrainingReceipt.")
        self.receipt = receipt
        self.metadata = dict(metadata or {})
        self.validate()

    def validate(self) -> None:
        if self.schema_version != PRIOR_SCHEMA_VERSION:
            raise ValueError(
                "Prior schema mismatch: "
                f"expected {PRIOR_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        self.base_rate = _require_probability(self.base_rate, "base_rate")
        if not isinstance(self.receipt, TrainingReceipt):
            raise TypeError("v16 PriorSet training receipt is invalid.")
        require_finite_structure(self.metadata, path="PriorSet.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "base_rate": self.base_rate,
            "training_receipt": self.receipt.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriorSet":
        expected = {"schema_version", "base_rate", "training_receipt"}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "v16 prior payload must contain exactly schema_version, base_rate, "
                "and training_receipt; "
                f"got {sorted(actual)!r}."
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            base_rate=float(payload["base_rate"]),
            receipt=TrainingReceipt.from_dict(dict(payload["training_receipt"])),
        )


@dataclass(init=False)
class LikelihoodSet:
    """Exactly four current branch likelihoods for one symbol."""

    schema_version: str = LIKELIHOOD_SCHEMA_VERSION
    quant_likelihood: float
    fundamental_likelihood: float
    macro_likelihood: float
    llm_likelihood: float
    receipt: TrainingReceipt = field(init=False)
    correlation_matrix: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        quant_likelihood: float,
        fundamental_likelihood: float,
        macro_likelihood: float,
        llm_likelihood: float,
        receipt: TrainingReceipt,
        schema_version: str = LIKELIHOOD_SCHEMA_VERSION,
        correlation_matrix: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if str(schema_version) != LIKELIHOOD_SCHEMA_VERSION:
            raise ValueError(
                "Likelihood schema mismatch: "
                f"expected {LIKELIHOOD_SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        self.schema_version = LIKELIHOOD_SCHEMA_VERSION
        self.quant_likelihood = quant_likelihood
        self.fundamental_likelihood = fundamental_likelihood
        self.macro_likelihood = macro_likelihood
        self.llm_likelihood = llm_likelihood
        if not isinstance(receipt, TrainingReceipt):
            raise TypeError("v16 LikelihoodSet requires a calibration TrainingReceipt.")
        self.receipt = receipt
        self.correlation_matrix = dict(correlation_matrix or {})
        self.metadata = dict(metadata or {})
        self.validate()

    def validate(self) -> None:
        if self.schema_version != LIKELIHOOD_SCHEMA_VERSION:
            raise ValueError(
                "Likelihood schema mismatch: "
                f"expected {LIKELIHOOD_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        if not isinstance(self.receipt, TrainingReceipt):
            raise TypeError("v16 LikelihoodSet calibration receipt is invalid.")
        for branch_name in CANONICAL_BRANCH_ORDER:
            field_name = f"{branch_name}_likelihood"
            setattr(
                self,
                field_name,
                _require_probability(getattr(self, field_name), field_name),
            )
        self.correlation_matrix = _validate_correlation_matrix(self.correlation_matrix)
        branch_weights = self.metadata.get("branch_weights")
        if branch_weights is not None:
            if not isinstance(branch_weights, Mapping):
                raise ValueError("Likelihood metadata branch_weights must be a mapping.")
            expected = set(CANONICAL_BRANCH_ORDER)
            if set(branch_weights) != expected:
                raise ValueError(
                    "Likelihood metadata branch_weights must contain exactly the "
                    f"v16 branches {list(CANONICAL_BRANCH_ORDER)!r}."
                )
            for branch_name in CANONICAL_BRANCH_ORDER:
                weight = _require_finite(
                    branch_weights[branch_name],
                    f"metadata.branch_weights.{branch_name}",
                )
                if not math.isclose(weight, 0.25, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("v16 likelihood branch weights must each equal 0.25.")
        require_finite_structure(self.metadata, path="LikelihoodSet.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            **{
                f"{branch_name}_likelihood": getattr(self, f"{branch_name}_likelihood")
                for branch_name in CANONICAL_BRANCH_ORDER
            },
            "correlation_matrix": dict(self.correlation_matrix),
            "training_receipt": self.receipt.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LikelihoodSet":
        expected = {
            "schema_version",
            *(f"{branch_name}_likelihood" for branch_name in CANONICAL_BRANCH_ORDER),
            "correlation_matrix",
            "training_receipt",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "v16 likelihood payload has incomplete or unexpected fields; "
                f"expected {sorted(expected)!r}, got {sorted(actual)!r}."
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            quant_likelihood=float(payload["quant_likelihood"]),
            fundamental_likelihood=float(payload["fundamental_likelihood"]),
            macro_likelihood=float(payload["macro_likelihood"]),
            llm_likelihood=float(payload["llm_likelihood"]),
            correlation_matrix=dict(payload["correlation_matrix"] or {}),
            receipt=TrainingReceipt.from_dict(dict(payload["training_receipt"])),
        )

    def as_list(self) -> list[tuple[str, float]]:
        self.validate()
        return [
            (branch_name, getattr(self, f"{branch_name}_likelihood"))
            for branch_name in CANONICAL_BRANCH_ORDER
        ]


@dataclass
class PosteriorResult:
    """The v16 Bayesian estimate; policy/action decisions live downstream."""

    prior: PriorSet
    likelihoods: LikelihoodSet
    posterior_win_rate: float
    posterior_expected_alpha: float
    posterior_edge_after_costs: float | None
    posterior_win_rate_interval_90: tuple[float, float]
    posterior_expected_alpha_interval_90: tuple[float, float]
    schema_version: str = POSTERIOR_SCHEMA_VERSION
    symbol: str = ""
    company_name: str = ""
    raw_evidence_increment: float = 0.0
    correlation_adjusted_evidence_increment: float = 0.0
    correlation_vif: float = 1.0
    correlation_vif_shrink: float = 1.0
    branch_evidence_increments: dict[str, float] = field(default_factory=dict)
    evidence_sources: list[str] = field(default_factory=list)
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != POSTERIOR_SCHEMA_VERSION:
            raise ValueError(
                "Posterior schema mismatch: "
                f"expected {POSTERIOR_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        self.prior.validate()
        self.likelihoods.validate()
        self.posterior_win_rate = _require_probability(
            self.posterior_win_rate, "posterior_win_rate"
        )
        self.posterior_expected_alpha = _require_finite(
            self.posterior_expected_alpha, "posterior_expected_alpha"
        )
        if self.posterior_edge_after_costs is not None:
            self.posterior_edge_after_costs = _require_finite(
                self.posterior_edge_after_costs,
                "posterior_edge_after_costs",
            )
        self.posterior_win_rate_interval_90 = _require_interval(
            self.posterior_win_rate_interval_90,
            "posterior_win_rate_interval_90",
            probability=True,
        )
        self.posterior_expected_alpha_interval_90 = _require_interval(
            self.posterior_expected_alpha_interval_90,
            "posterior_expected_alpha_interval_90",
            probability=False,
        )
        self.raw_evidence_increment = _require_finite(
            self.raw_evidence_increment, "raw_evidence_increment"
        )
        self.correlation_adjusted_evidence_increment = _require_finite(
            self.correlation_adjusted_evidence_increment,
            "correlation_adjusted_evidence_increment",
        )
        self.correlation_vif = _require_finite(self.correlation_vif, "correlation_vif")
        self.correlation_vif_shrink = _require_finite(
            self.correlation_vif_shrink, "correlation_vif_shrink"
        )
        if self.correlation_vif < 1.0:
            raise ValueError("correlation_vif must be at least 1.0.")
        if not 0.0 < self.correlation_vif_shrink <= 1.0:
            raise ValueError("correlation_vif_shrink must be in (0, 1].")
        expected_branches = set(CANONICAL_BRANCH_ORDER)
        if set(self.branch_evidence_increments) != expected_branches:
            raise ValueError(
                "branch_evidence_increments must contain exactly the four v16 branches."
            )
        for branch_name, value in self.branch_evidence_increments.items():
            self.branch_evidence_increments[branch_name] = _require_finite(
                value, f"branch_evidence_increments.{branch_name}"
            )
        if tuple(self.evidence_sources) != CANONICAL_BRANCH_ORDER:
            raise ValueError(
                "evidence_sources must contain all four v16 branches in canonical order."
            )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError(f"rank must be a non-negative integer; got {self.rank!r}.")
        _reject_retired_posterior_keys(self.metadata, path="PosteriorResult.metadata")
        require_finite_structure(self.metadata, path="PosteriorResult.metadata")

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "prior": self.prior.to_dict(),
            "likelihoods": self.likelihoods.to_dict(),
            "posterior_win_rate": self.posterior_win_rate,
            "posterior_expected_alpha": self.posterior_expected_alpha,
            "posterior_edge_after_costs": self.posterior_edge_after_costs,
            "posterior_win_rate_interval_90": list(self.posterior_win_rate_interval_90),
            "posterior_expected_alpha_interval_90": list(self.posterior_expected_alpha_interval_90),
            "raw_evidence_increment": self.raw_evidence_increment,
            "correlation_adjusted_evidence_increment": (
                self.correlation_adjusted_evidence_increment
            ),
            "correlation_vif": self.correlation_vif,
            "correlation_vif_shrink": self.correlation_vif_shrink,
            "branch_evidence_increments": {
                branch_name: self.branch_evidence_increments[branch_name]
                for branch_name in CANONICAL_BRANCH_ORDER
            },
            "evidence_sources": list(self.evidence_sources),
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PosteriorResult":
        expected = {
            "schema_version",
            "symbol",
            "company_name",
            "prior",
            "likelihoods",
            "posterior_win_rate",
            "posterior_expected_alpha",
            "posterior_edge_after_costs",
            "posterior_win_rate_interval_90",
            "posterior_expected_alpha_interval_90",
            "raw_evidence_increment",
            "correlation_adjusted_evidence_increment",
            "correlation_vif",
            "correlation_vif_shrink",
            "branch_evidence_increments",
            "evidence_sources",
            "rank",
            "metadata",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "v16 posterior payload has incomplete, retired, or unexpected "
                f"fields; expected {sorted(expected)!r}, got {sorted(actual)!r}."
            )
        edge = payload["posterior_edge_after_costs"]
        return cls(
            schema_version=str(payload["schema_version"]),
            symbol=str(payload["symbol"]),
            company_name=str(payload["company_name"]),
            prior=PriorSet.from_dict(dict(payload["prior"])),
            likelihoods=LikelihoodSet.from_dict(dict(payload["likelihoods"])),
            posterior_win_rate=float(payload["posterior_win_rate"]),
            posterior_expected_alpha=float(payload["posterior_expected_alpha"]),
            posterior_edge_after_costs=(None if edge is None else float(edge)),
            posterior_win_rate_interval_90=tuple(payload["posterior_win_rate_interval_90"]),
            posterior_expected_alpha_interval_90=tuple(
                payload["posterior_expected_alpha_interval_90"]
            ),
            raw_evidence_increment=float(payload["raw_evidence_increment"]),
            correlation_adjusted_evidence_increment=float(
                payload["correlation_adjusted_evidence_increment"]
            ),
            correlation_vif=float(payload["correlation_vif"]),
            correlation_vif_shrink=float(payload["correlation_vif_shrink"]),
            branch_evidence_increments=dict(payload["branch_evidence_increments"]),
            evidence_sources=list(payload["evidence_sources"]),
            rank=int(payload["rank"]),
            metadata=dict(payload["metadata"]),
        )


__all__ = [
    "CANONICAL_CORRELATION_KEYS",
    "LikelihoodSet",
    "PosteriorResult",
    "PriorSet",
    "correlation_key",
]
