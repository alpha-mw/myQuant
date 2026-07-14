from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    GlobalContext,
    ICDecision,
    PortfolioPlan,
    RiskDecision,
    ShortlistItem,
    SymbolResearchPacket,
)
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.decision import (
    _run_bayesian_selection_phase,
    _run_portfolio_construction_phase,
)
from quant_investor.regime.types import REGIME_RANGE_HIGH_VOL, REGIME_TREND_DOWN


def test_turnover_cap_is_forwarded_to_risk_guard_and_portfolio_constructor() -> None:
    captured: dict[str, object] = {}

    class FakeRiskGuard:
        def run(self, payload: dict[str, object]) -> RiskDecision:
            captured["risk_constraints"] = dict(payload["constraints"])
            return RiskDecision(
                status=AgentStatus.SUCCESS,
                action_cap=ActionLabel.BUY,
                gross_exposure_cap=0.45,
                max_weight=0.50,
                position_limits={"000001.SZ": 0.50},
            )

    class FakeICCoordinator:
        def run(self, payload: dict[str, object]) -> ICDecision:
            return ICDecision(
                symbol="000001.SZ",
                action=ActionLabel.BUY,
                selected_symbols=["000001.SZ"],
                final_score=0.8,
                final_confidence=0.7,
            )

    class FakePortfolioConstructor:
        def run(self, payload: dict[str, object]) -> PortfolioPlan:
            captured["portfolio_risk_limits"] = dict(payload["risk_limits"])
            return PortfolioPlan(
                target_exposure=0.30,
                target_gross_exposure=0.30,
                target_net_exposure=0.30,
                cash_ratio=0.70,
                target_weights={"000001.SZ": 0.30},
                target_positions={"000001.SZ": 0.30},
            )

    state = _run_portfolio_construction_phase(
        shortlist=[ShortlistItem(symbol="000001.SZ", rank_score=0.9)],
        branch_summaries={"quant": BranchVerdict(agent_name="quant", thesis="ok")},
        macro_verdict=BranchVerdict(metadata={"target_gross_exposure": 0.80}),
        global_context=GlobalContext(
            risk_budget={
                "target_exposure": 0.45,
                "max_single_weight": 0.50,
                "turnover_cap": 0.30,
            }
        ),
        data_quality_issues=[],
        ic_hints_by_symbol={},
        research_by_symbol={"000001.SZ": {"quant": BranchVerdict(agent_name="quant", thesis="ok")}},
        tradability_snapshot={
            "000001.SZ": {
                "is_tradable": True,
                "sector": "Banking",
                "liquidity_score": 1.0,
                "momentum_strength": 0.8,
            }
        },
        funnel_summary={},
        bayesian_records=[],
        candidate_symbols=["000001.SZ"],
        portfolio_master_output=None,
        portfolio_master_meta={},
        risk_guard_cls=FakeRiskGuard,
        ic_coordinator_cls=FakeICCoordinator,
        portfolio_constructor_cls=FakePortfolioConstructor,
        attach_symbol_to_ic_decision_fn=lambda decision, **kwargs: decision,
    )

    assert captured["risk_constraints"]["turnover_cap"] == pytest.approx(0.30)
    assert captured["portfolio_risk_limits"]["turnover_cap"] == pytest.approx(0.30)
    assert state.portfolio_decision.target_weights == {"000001.SZ": 0.30}


def test_bayesian_decision_record_metadata_includes_markov_regime() -> None:
    class PriorBuilder:
        def build_prior(self, symbol: str, global_context: GlobalContext) -> PriorSet:
            return PriorSet(composite_prior=0.50)

    class LikelihoodMapper:
        def __init__(self, **kwargs: object) -> None:
            pass

        def compute_likelihoods(
            self,
            *,
            branch_results: dict[str, object],
            symbol: str,
            candidate_symbols: set[str],
        ) -> LikelihoodSet:
            return LikelihoodSet(quant_likelihood=0.60)

    class PosteriorEngine:
        def compute_posterior(
            self,
            prior: PriorSet,
            likelihoods: LikelihoodSet,
            *,
            symbol: str,
            company_name: str,
            regime: str,
            is_degraded: dict[str, bool],
        ) -> PosteriorResult:
            assert regime == REGIME_RANGE_HIGH_VOL
            return PosteriorResult(
                symbol=symbol,
                company_name=company_name,
                prior=prior,
                likelihoods=likelihoods,
                posterior_win_rate=0.55,
                posterior_expected_alpha=0.01,
                posterior_confidence=0.60,
                posterior_action_score=0.42,
                posterior_edge_after_costs=0.008,
                posterior_capacity_penalty=0.0,
            )

    global_context = GlobalContext(
        macro_regime=REGIME_RANGE_HIGH_VOL,
        regime_params={
            "markov": {
                "enabled": True,
                "execution_mode": "production",
                "production_eligible": True,
                "regime_scope": "full_market",
                "scope_key": "CN:full_market:full_a:symbols_50",
                "dominant_regime": REGIME_RANGE_HIGH_VOL,
                "confidence": 0.61,
                "transition_risk": 0.72,
                "probabilities": {
                    REGIME_RANGE_HIGH_VOL: 0.45,
                    REGIME_TREND_DOWN: 0.27,
                },
            }
        },
    )

    macro_verdict = BranchVerdict(agent_name="macro")
    state = _run_bayesian_selection_phase(
        candidate_symbols=["000001.SZ"],
        company_name_map={"000001.SZ": "平安银行"},
        symbol_research_packets={
            "000001.SZ": SymbolResearchPacket(symbol="000001.SZ", category="bank")
        },
        research_by_symbol={},
        branch_summaries={
            "quant": BranchVerdict(agent_name="quant"),
            "fundamental": BranchVerdict(agent_name="fundamental"),
            "macro": macro_verdict,
        },
        branch_results={
            "quant": BranchResult(branch_name="quant"),
            "fundamental": BranchResult(branch_name="fundamental"),
            "macro": BranchResult(branch_name="macro"),
        },
        macro_verdict=macro_verdict,
        global_context=global_context,
        model_roles=SimpleNamespace(
            agent_layer_enabled=False,
            metadata={},
            resolved_master_model="",
            resolved_branch_model="",
        ),
        resolver_snapshot={},
        data_quality_issues=[],
        top_k=1,
        all_symbols=["000001.SZ"],
        funnel_output=SimpleNamespace(excluded_symbols={}, funnel_metadata={}),
        provider_health={},
        master_timeout=0.0,
        master_reasoning_effort="",
        master_model_resolution=SimpleNamespace(resolved_model="", fallback_model=""),
        master_candidate_models=[],
        recall_context=None,
        hierarchical_prior_builder_cls=PriorBuilder,
        likelihood_mapper_cls=LikelihoodMapper,
        posterior_engine_cls=PosteriorEngine,
        master_agent_cls=object,
        llm_client_cls=object,
        portfolio_master_advisory_fn=lambda **kwargs: (None, {}),
    )

    metadata = state.bayesian_records[0].metadata["markov_regime"]
    assert metadata["dominant_regime"] == REGIME_RANGE_HIGH_VOL
    assert metadata["confidence"] == pytest.approx(0.61)
    assert metadata["transition_risk"] == pytest.approx(0.72)
    assert metadata["probabilities"][REGIME_TREND_DOWN] == pytest.approx(0.27)
