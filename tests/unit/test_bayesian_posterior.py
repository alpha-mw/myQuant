"""Tests for Bayesian posterior engine."""

from __future__ import annotations

import pytest

from quant_investor.bayesian.likelihood import SignalLikelihoodMapper, _score_to_likelihood
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.branch_contracts import BranchResult


class TestScoreToLikelihood:
    def test_strong_positive_score_gives_high_likelihood(self):
        ll = _score_to_likelihood(score=0.80, confidence=0.90, reliability=0.80)
        assert ll > 0.65

    def test_strong_negative_score_gives_low_likelihood(self):
        ll = _score_to_likelihood(score=-0.80, confidence=0.90, reliability=0.80)
        assert ll < 0.35

    def test_zero_score_gives_neutral_likelihood(self):
        ll = _score_to_likelihood(score=0.0, confidence=0.50, reliability=0.50)
        assert 0.45 <= ll <= 0.55

    def test_low_confidence_keeps_likelihood_near_neutral(self):
        ll = _score_to_likelihood(score=0.80, confidence=0.10, reliability=0.10)
        assert 0.45 <= ll <= 0.60

    def test_output_is_clamped(self):
        ll = _score_to_likelihood(score=2.0, confidence=1.0, reliability=1.0)
        assert 0.05 <= ll <= 0.95


class TestSignalLikelihoodMapper:
    def test_all_branches_available(self):
        branches = {
            "kline": BranchResult(branch_name="kline", final_score=0.3, final_confidence=0.7,
                                  symbol_scores={"A": 0.3}, metadata={"reliability": 0.7}),
            "quant": BranchResult(branch_name="quant", final_score=0.5, final_confidence=0.8,
                                  symbol_scores={"A": 0.5}, metadata={"reliability": 0.75}),
            "fundamental": BranchResult(branch_name="fundamental", final_score=0.2, final_confidence=0.6,
                                        symbol_scores={"A": 0.2}, metadata={"reliability": 0.6}),
            "intelligence": BranchResult(branch_name="intelligence", final_score=0.1, final_confidence=0.5,
                                          symbol_scores={"A": 0.1}, metadata={"reliability": 0.55}),
        }
        mapper = SignalLikelihoodMapper()
        ls = mapper.compute_likelihoods(branch_results=branches, symbol="A")
        assert ls.kline_likelihood > 0.50  # positive score
        assert ls.quant_likelihood > 0.50  # positive score
        assert len(ls.as_list()) == 4

    def test_non_candidate_gets_neutral_for_expensive_branches(self):
        branches = {
            "kline": BranchResult(branch_name="kline", final_score=0.3, final_confidence=0.7,
                                  symbol_scores={"A": 0.3}, metadata={"reliability": 0.7}),
            "quant": BranchResult(branch_name="quant", final_score=0.5, final_confidence=0.8,
                                  symbol_scores={"A": 0.5}, metadata={"reliability": 0.75}),
        }
        mapper = SignalLikelihoodMapper()
        ls = mapper.compute_likelihoods(
            branch_results=branches, symbol="A", candidate_symbols=set(),
        )
        assert ls.fundamental_likelihood == 0.50
        assert ls.intelligence_likelihood == 0.50

    def test_missing_branch_gives_neutral(self):
        branches = {}
        mapper = SignalLikelihoodMapper()
        ls = mapper.compute_likelihoods(branch_results=branches, symbol="A")
        assert ls.kline_likelihood == 0.50
        assert ls.quant_likelihood == 0.50


class TestBayesianPosteriorEngine:
    def test_neutral_inputs_give_neutral_posterior(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        likelihoods = LikelihoodSet()  # all 0.50

        result = engine.compute_posterior(prior, likelihoods, symbol="A")
        assert isinstance(result, PosteriorResult)
        assert 0.45 <= result.posterior_win_rate <= 0.55
        assert result.posterior_expected_alpha == pytest.approx(0.0, abs=0.01)

    def test_bullish_evidence_raises_posterior(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        likelihoods = LikelihoodSet(
            kline_likelihood=0.75,
            quant_likelihood=0.80,
            fundamental_likelihood=0.70,
            intelligence_likelihood=0.65,
        )

        result = engine.compute_posterior(prior, likelihoods, symbol="A")
        assert result.posterior_win_rate > 0.65
        assert result.posterior_expected_alpha > 0.0
        assert result.posterior_action_score > 0.50

    def test_bearish_evidence_lowers_posterior(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        likelihoods = LikelihoodSet(
            kline_likelihood=0.25,
            quant_likelihood=0.20,
            fundamental_likelihood=0.30,
            intelligence_likelihood=0.35,
        )

        result = engine.compute_posterior(prior, likelihoods, symbol="A")
        assert result.posterior_win_rate < 0.35
        assert result.posterior_expected_alpha < 0.0

    def test_correlation_discount_applied(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)

        # Two correlated branches with strong agreement
        likelihoods_high = LikelihoodSet(
            kline_likelihood=0.80,
            quant_likelihood=0.80,
            fundamental_likelihood=0.50,
            intelligence_likelihood=0.50,
        )

        result = engine.compute_posterior(prior, likelihoods_high, symbol="A")
        assert result.correlation_discount > 0.0
        # Posterior should be high but not as high as if branches were independent
        assert result.posterior_win_rate > 0.60

    def test_degraded_backend_penalty(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        likelihoods = LikelihoodSet(kline_likelihood=0.80, quant_likelihood=0.80)

        normal = engine.compute_posterior(prior, likelihoods, symbol="A")
        degraded = engine.compute_posterior(
            prior, likelihoods, symbol="A",
            is_degraded={"kline": True},
        )

        assert degraded.fallback_penalty > 0.0
        # Degraded result should have lower posterior
        assert degraded.posterior_win_rate < normal.posterior_win_rate

    def test_coverage_discount_with_missing_branches(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        # Only kline evidence, others neutral
        likelihoods = LikelihoodSet(kline_likelihood=0.75)

        result = engine.compute_posterior(prior, likelihoods, symbol="A")
        assert result.coverage_discount > 0.0
        assert result.posterior_confidence < 1.0

    def test_regime_aware_thresholds(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.50)
        likelihoods = LikelihoodSet(kline_likelihood=0.70, quant_likelihood=0.70)

        bull = engine.compute_posterior(prior, likelihoods, symbol="A", regime="趋势上涨")
        bear = engine.compute_posterior(prior, likelihoods, symbol="A", regime="趋势下跌")

        # Bear regime should have higher buy threshold
        assert bear.action_threshold_used > bull.action_threshold_used

    def test_posterior_result_to_dict(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.55)
        likelihoods = LikelihoodSet(kline_likelihood=0.70)
        result = engine.compute_posterior(
            prior, likelihoods, symbol="A", company_name="Test Corp",
        )
        d = result.to_dict()
        assert d["symbol"] == "A"
        assert d["company_name"] == "Test Corp"
        assert "prior" in d
        assert "likelihoods" in d
        assert "posterior_win_rate" in d

    def test_strong_bear_prior_with_neutral_evidence(self):
        engine = BayesianPosteriorEngine()
        prior = PriorSet(composite_prior=0.30)
        likelihoods = LikelihoodSet()  # all neutral 0.50

        result = engine.compute_posterior(prior, likelihoods, symbol="A")
        # Should remain bearish since evidence is neutral
        assert result.posterior_win_rate < 0.40
