from __future__ import annotations

from dataclasses import fields
import math

import pytest

from quant_investor import versioning as active_versioning
from quant_investor.bayesian.v16 import (
    ARCHITECTURE_VERSION,
    DEFAULT_BRANCH_WEIGHTS,
    ArtifactReturnCalibration,
    BaseRateEvidence,
    BaseRatePriorBuilder,
    BayesianPosteriorEngine,
    BlockBootstrapArtifact,
    BranchLikelihoodInput,
    CalibrationObservation,
    CalibrationStore,
    CostComponents,
    LikelihoodSet,
    PosteriorResult,
    PriorSet,
    ReturnCalibrationEstimate,
    RobustReturnModelArtifact,
    SignalLikelihoodMapper,
    compute_correlation_vif,
    compute_equal_weight_evidence_increments,
)
from quant_investor.bayesian.v16.training import TrainingReceipt
from quant_investor.bayesian.v16.types import CANONICAL_CORRELATION_KEYS
from quant_investor.bayesian.v16.versioning import (
    LIKELIHOOD_SCHEMA_VERSION,
    POSTERIOR_SCHEMA_VERSION,
)
from quant_investor.v16.candidate_pipeline import (
    FormalBranchEvidence,
    FourBranchEvidence,
)

BRANCHES = ("quant", "fundamental", "macro", "llm")
SHA = "a" * 64


def _receipt(sample_count: int, receipt_id: str = "receipt-1") -> TrainingReceipt:
    return TrainingReceipt(
        receipt_id=receipt_id,
        evidence_sha256=SHA,
        training_start="2021-01-01",
        training_end="2026-01-01",
        sample_count=sample_count,
        purged=True,
        embargo_complete=True,
        embargo_days=20,
    )


def _calibration_observations(samples_per_branch: int = 25) -> list[CalibrationObservation]:
    return [
        CalibrationObservation(
            sample_id=f"{branch_name}-{index:03d}",
            branch_name=branch_name,
            score=-1.0 + 2.0 * index / max(samples_per_branch - 1, 1),
            positive_outcome=index >= samples_per_branch // 2,
        )
        for branch_name in BRANCHES
        for index in range(samples_per_branch)
    ]


def _calibration_store(samples_per_branch: int = 25) -> CalibrationStore:
    observations = _calibration_observations(samples_per_branch)
    return CalibrationStore.from_training_evidence(
        observations,
        receipt=_receipt(samples_per_branch, "likelihood-training"),
    )


def _prior(base_positive: int = 40, total: int = 100) -> PriorSet:
    evidence = BaseRateEvidence(
        positive_count=base_positive,
        total_count=total,
        receipt=_receipt(total, "base-rate-training"),
    )
    return BaseRatePriorBuilder(evidence).build_prior("A")


def _likelihoods(
    probability: float = 0.60,
    *,
    correlations: dict[str, float] | None = None,
) -> LikelihoodSet:
    resolved_correlations = correlations or {key: 0.0 for key in CANONICAL_CORRELATION_KEYS}
    return LikelihoodSet(
        quant_likelihood=probability,
        fundamental_likelihood=probability,
        macro_likelihood=probability,
        llm_likelihood=probability,
        correlation_matrix=resolved_correlations,
        receipt=_calibration_store().receipt,
    )


def _bootstrap() -> BlockBootstrapArtifact:
    win_offsets = tuple((index - 499.5) / 2500.0 for index in range(1000))
    alpha_offsets = tuple((index - 499.5) / 50000.0 for index in range(1000))
    return BlockBootstrapArtifact(
        artifact_id="bootstrap-1",
        artifact_sha256="b" * 64,
        receipt=_receipt(100, "bootstrap-training"),
        block_length_days=20,
        block_count=60,
        win_rate_logit_offsets=win_offsets,
        expected_alpha_offsets=alpha_offsets,
    )


def _return_model() -> ArtifactReturnCalibration:
    artifact = RobustReturnModelArtifact(
        artifact_id="return-model-1",
        parameters_sha256="c" * 64,
        receipt=_receipt(100, "return-model-training"),
        intercept=0.001,
        aggregate_coefficient=0.04,
    )
    return ArtifactReturnCalibration(artifact)


def _engine() -> BayesianPosteriorEngine:
    return BayesianPosteriorEngine(
        return_calibration_model=_return_model(),
        bootstrap_artifact=_bootstrap(),
    )


def _branch_inputs() -> dict[str, BranchLikelihoodInput]:
    return {
        branch_name: BranchLikelihoodInput(
            branch_name=branch_name,
            final_score=0.20,
            final_confidence=0.80,
            symbol_scores={"A": 0.25},
            metadata={"reliability": 0.70},
        )
        for branch_name in BRANCHES
    }


def test_v16_is_explicit_and_does_not_pre_activate_v15_defaults() -> None:
    assert active_versioning.ARCHITECTURE_VERSION == "15.0.0-stable"
    assert ARCHITECTURE_VERSION == "16.0.0"
    assert DEFAULT_BRANCH_WEIGHTS == {
        "quant": 0.25,
        "fundamental": 0.25,
        "macro": 0.25,
        "llm": 0.25,
    }


def test_training_receipt_requires_five_year_purged_embargoed_20d_csi300() -> None:
    with pytest.raises(ValueError, match="five years"):
        TrainingReceipt(
            receipt_id="short",
            evidence_sha256=SHA,
            training_start="2024-01-01",
            training_end="2026-01-01",
            sample_count=100,
            purged=True,
            embargo_complete=True,
            embargo_days=20,
        )
    with pytest.raises(ValueError, match="purge and embargo"):
        TrainingReceipt(
            receipt_id="not-purged",
            evidence_sha256=SHA,
            training_start="2021-01-01",
            training_end="2026-01-01",
            sample_count=100,
            purged=False,
            embargo_complete=True,
            embargo_days=20,
        )


def test_v16_prior_has_one_base_rate_and_fails_without_evidence() -> None:
    with pytest.raises(TypeError):
        BaseRatePriorBuilder()  # type: ignore[call-arg]
    prior = _prior(base_positive=40, total=100)
    assert prior.base_rate == pytest.approx(41 / 102)
    assert set(prior.to_dict()) == {
        "schema_version",
        "base_rate",
        "training_receipt",
    }
    with pytest.raises(TypeError):
        PriorSet(base_rate=0.50)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PriorSet(composite_prior=0.50)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="schema mismatch"):
        PriorSet(
            base_rate=0.50,
            receipt=_receipt(100),
            schema_version="v15.hierarchical-prior",
        )


def test_calibration_store_builds_five_equal_frequency_smoothed_buckets() -> None:
    store = _calibration_store()
    for branch_name in BRANCHES:
        buckets = store.buckets_by_branch[branch_name]
        assert len(buckets) == 5
        assert [bucket.sample_count for bucket in buckets] == [5, 5, 5, 5, 5]
        assert all(0.0 < bucket.calibrated_probability < 1.0 for bucket in buckets)
        assert buckets[0].calibrated_probability == pytest.approx(1 / 7)
        assert buckets[-1].calibrated_probability == pytest.approx(6 / 7)

    sparse = _calibration_observations(samples_per_branch=4)
    with pytest.raises(ValueError, match="insufficient samples"):
        CalibrationStore.from_training_evidence(
            sparse,
            receipt=_receipt(4, "sparse"),
        )


def test_v16_likelihood_requires_all_four_values_current_schema_and_receipt() -> None:
    with pytest.raises(TypeError):
        LikelihoodSet(  # type: ignore[call-arg]
            quant_likelihood=0.60,
            fundamental_likelihood=0.60,
            macro_likelihood=0.60,
            llm_likelihood=0.60,
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        LikelihoodSet(
            schema_version="likelihood-schema.v15.two-likelihood",
            quant_likelihood=0.60,
            fundamental_likelihood=0.60,
            macro_likelihood=0.60,
            llm_likelihood=0.60,
            receipt=_calibration_store().receipt,
        )
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        LikelihoodSet.from_dict(
            {
                "schema_version": LIKELIHOOD_SCHEMA_VERSION,
                "quant_likelihood": 0.60,
                "fundamental_likelihood": 0.60,
                "correlation_matrix": {},
            }
        )


def test_v16_mapper_requires_trained_store_and_refuses_neutral_fallback() -> None:
    with pytest.raises(TypeError):
        SignalLikelihoodMapper()  # type: ignore[call-arg]
    mapper = SignalLikelihoodMapper(
        calibration_store=_calibration_store(),
        correlation_matrix={key: 0.0 for key in CANONICAL_CORRELATION_KEYS},
        recall_context={"top_picks": ["A"]},
    )
    inputs = _branch_inputs()
    inputs.pop("llm")
    with pytest.raises(ValueError, match="exactly all four"):
        mapper.compute_likelihoods(branch_results=inputs, symbol="A")

    with pytest.raises(ValueError, match="non-candidate neutral fallback"):
        mapper.compute_likelihoods(
            branch_results=_branch_inputs(),
            symbol="A",
            candidate_symbols=set(),
        )

    mapped = mapper.compute_likelihoods(branch_results=_branch_inputs(), symbol="A")
    assert [name for name, _value in mapped.as_list()] == list(BRANCHES)
    assert mapped.metadata["retrieval_evidence_used"] is False
    assert mapped.metadata["branch_weights"] == DEFAULT_BRANCH_WEIGHTS

    sealed = FourBranchEvidence(
        symbol="A",
        branches=tuple(
            FormalBranchEvidence(
                symbol="A",
                branch=branch_name,
                raw_score=0.25,
                confidence=0.80,
                evidence_ids=(f"{branch_name}-evidence",),
            )
            for branch_name in BRANCHES
        ),
    )
    from_sealed = mapper.compute_from_sealed_evidence(sealed)
    assert [name for name, _value in from_sealed.as_list()] == list(BRANCHES)


def test_equal_weight_increment_and_vif_shrink_are_recomputable() -> None:
    prior = _prior()
    base_rate = prior.base_rate
    branch_probabilities = {
        "quant": 0.60,
        "fundamental": 0.70,
        "macro": 0.55,
        "llm": 0.65,
    }
    increments = compute_equal_weight_evidence_increments(
        base_rate=base_rate,
        branch_probabilities=branch_probabilities,
    )

    def logit(value: float) -> float:
        return math.log(value / (1.0 - value))

    for branch_name, probability in branch_probabilities.items():
        assert increments[branch_name] == pytest.approx(
            0.25 * (logit(probability) - logit(base_rate))
        )

    fully_correlated = {key: 1.0 for key in CANONICAL_CORRELATION_KEYS}
    with pytest.raises(ValueError, match="all six OOS"):
        compute_correlation_vif({})
    assert compute_correlation_vif(fully_correlated) == pytest.approx(4.0)
    likelihoods = LikelihoodSet(
        quant_likelihood=branch_probabilities["quant"],
        fundamental_likelihood=branch_probabilities["fundamental"],
        macro_likelihood=branch_probabilities["macro"],
        llm_likelihood=branch_probabilities["llm"],
        correlation_matrix=fully_correlated,
        receipt=_calibration_store().receipt,
    )
    result = _engine().compute_posterior(prior, likelihoods)
    assert result.correlation_vif == pytest.approx(4.0)
    assert result.correlation_vif_shrink == pytest.approx(0.5)
    assert result.raw_evidence_increment == pytest.approx(sum(increments.values()))
    assert result.correlation_adjusted_evidence_increment == pytest.approx(
        sum(increments.values()) * 0.5
    )
    expected_logit = logit(base_rate) + sum(increments.values()) * 0.5
    assert result.posterior_win_rate == pytest.approx(1.0 / (1.0 + math.exp(-expected_logit)))
    assert result.metadata["vif"] == pytest.approx(4.0)
    assert result.metadata["lambda"] == pytest.approx(0.5)
    assert result.metadata["return_model_equal_weight_evidence"] == pytest.approx(
        result.raw_evidence_increment
    )


class _InjectedReturnModel:
    def __init__(self, expected_alpha: float) -> None:
        self.expected_alpha = expected_alpha

    def estimate(self, **_kwargs: object) -> ReturnCalibrationEstimate:
        return ReturnCalibrationEstimate(
            expected_alpha=self.expected_alpha,
            equal_weight_evidence=0.0,
            correlation_adjusted_equal_weight_evidence=0.0,
        )


def test_return_model_and_block_bootstrap_are_required_no_heuristic_defaults() -> None:
    with pytest.raises(TypeError):
        BayesianPosteriorEngine()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exactly 1000"):
        BlockBootstrapArtifact(
            artifact_id="bad-bootstrap",
            artifact_sha256="b" * 64,
            receipt=_receipt(100),
            block_length_days=20,
            block_count=60,
            win_rate_logit_offsets=(0.0,) * 999,
            expected_alpha_offsets=(0.0,) * 1000,
        )

    high = BayesianPosteriorEngine(
        return_calibration_model=_InjectedReturnModel(0.07),
        bootstrap_artifact=_bootstrap(),
    ).compute_posterior(_prior(), _likelihoods())
    low = BayesianPosteriorEngine(
        return_calibration_model=_InjectedReturnModel(-0.02),
        bootstrap_artifact=_bootstrap(),
    ).compute_posterior(_prior(), _likelihoods())
    assert high.posterior_win_rate == pytest.approx(low.posterior_win_rate)
    assert high.posterior_expected_alpha == pytest.approx(0.07)
    assert low.posterior_expected_alpha == pytest.approx(-0.02)
    assert high.metadata["bootstrap_iterations"] == 1000
    assert high.metadata["bootstrap_method"] == "time-block-bootstrap.v1"


def test_robust_return_model_uses_only_equal_weight_aggregate() -> None:
    model = _return_model()
    probabilities = {
        "quant": 0.72,
        "fundamental": 0.61,
        "macro": 0.48,
        "llm": 0.57,
    }
    swapped = {
        "quant": probabilities["llm"],
        "fundamental": probabilities["macro"],
        "macro": probabilities["fundamental"],
        "llm": probabilities["quant"],
    }
    first = model.estimate(
        branch_probabilities=probabilities,
        base_rate=0.40,
        vif_shrink=0.75,
    )
    second = model.estimate(
        branch_probabilities=swapped,
        base_rate=0.40,
        vif_shrink=0.75,
    )
    assert first.equal_weight_evidence == pytest.approx(second.equal_weight_evidence)
    assert first.expected_alpha == pytest.approx(second.expected_alpha)


def test_edge_requires_fee_slippage_and_market_impact() -> None:
    engine = _engine()
    prior = _prior()
    likelihoods = _likelihoods()
    missing = engine.compute_posterior(
        prior,
        likelihoods,
        costs=CostComponents(fee=0.001, slippage=0.002),
    )
    complete = engine.compute_posterior(
        prior,
        likelihoods,
        costs=CostComponents(fee=0.001, slippage=0.002, market_impact=0.003),
    )
    assert missing.posterior_edge_after_costs is None
    assert complete.posterior_edge_after_costs == pytest.approx(
        complete.posterior_expected_alpha - 0.006
    )


def test_v16_posterior_contract_has_bootstrap_intervals_and_no_policy_fields() -> None:
    field_names = {item.name for item in fields(PosteriorResult)}
    assert {
        "posterior_win_rate",
        "posterior_expected_alpha",
        "posterior_edge_after_costs",
        "posterior_win_rate_interval_90",
        "posterior_expected_alpha_interval_90",
    } <= field_names
    assert "posterior_action_score" not in field_names
    assert "action_threshold_used" not in field_names

    result = _engine().compute_posterior(_prior(), _likelihoods())
    payload = result.to_dict()
    assert payload["schema_version"] == POSTERIOR_SCHEMA_VERSION
    assert "posterior_action_score" not in payload
    assert "kill_switch" not in payload
    assert "action_threshold" not in payload
    assert len(payload["posterior_win_rate_interval_90"]) == 2
    assert len(payload["posterior_expected_alpha_interval_90"]) == 2
    assert PosteriorResult.from_dict(payload).to_dict() == payload

    legacy = dict(payload)
    legacy["posterior_action_score"] = 0.7
    with pytest.raises(ValueError, match="retired, or unexpected"):
        PosteriorResult.from_dict(legacy)


def test_v16_posterior_refuses_degraded_branch_fallback() -> None:
    with pytest.raises(ValueError, match="refuses degraded-branch fallback"):
        _engine().compute_posterior(
            _prior(),
            _likelihoods(),
            is_degraded={"macro": True},
        )
