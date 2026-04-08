"""Tests for Bayesian hierarchical prior builder."""

from __future__ import annotations

import pytest

from quant_investor.agent_protocol import DataQualityIssue, GlobalContext
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.bayesian.types import PriorSet


def _make_context(
    regime: str = "未知",
    quarantine: list[str] | None = None,
    suspended: list[str] | None = None,
) -> GlobalContext:
    return GlobalContext(
        macro_regime=regime,
        data_quality_quarantine=quarantine or [],
        liquidity_filter={"suspended": suspended or []},
    )


class TestHierarchicalPriorBuilder:
    def test_bull_regime_raises_market_prior(self):
        ctx = _make_context(regime="趋势上涨")
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("000001.SZ", ctx)
        assert prior.market_prior == 0.55
        assert prior.regime_prior == 0.55

    def test_bear_regime_lowers_market_prior(self):
        ctx = _make_context(regime="趋势下跌")
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("000001.SZ", ctx)
        assert prior.market_prior == 0.35

    def test_quarantined_symbol_has_low_data_quality_prior(self):
        ctx = _make_context(quarantine=["BAD.SZ"])
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("BAD.SZ", ctx)
        assert prior.data_quality_prior == 0.15

    def test_suspended_symbol_has_low_tradability_prior(self):
        ctx = _make_context(suspended=["SUSP.SZ"])
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("SUSP.SZ", ctx)
        assert prior.tradability_prior == 0.10

    def test_normal_symbol_gets_neutral_priors(self):
        ctx = _make_context(regime="震荡低波")
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("000001.SZ", ctx)
        assert prior.data_quality_prior == 0.50
        assert prior.tradability_prior == 0.50
        assert prior.sector_prior == 0.50
        assert 0.01 < prior.composite_prior < 0.99

    def test_composite_prior_is_clamped(self):
        ctx = _make_context(regime="趋势上涨")
        builder = HierarchicalPriorBuilder()
        prior = builder.build_prior("000001.SZ", ctx)
        assert 0.01 <= prior.composite_prior <= 0.99

    def test_prior_to_dict(self):
        prior = PriorSet(market_prior=0.55, composite_prior=0.52)
        d = prior.to_dict()
        assert d["market_prior"] == 0.55
        assert "composite_prior" in d
