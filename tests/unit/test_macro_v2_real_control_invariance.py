from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    Direction,
    GlobalContext,
    ShortlistItem,
)
from quant_investor.agents.ic_coordinator import ICCoordinator
from quant_investor.agents.portfolio_constructor import PortfolioConstructor
from quant_investor.agents.risk_guard import RiskGuard
from quant_investor.bayesian import (
    BayesianPosteriorEngine,
    HierarchicalPriorBuilder,
    SignalLikelihoodMapper,
)
from quant_investor.branch_contracts import BranchResult
from quant_investor.macro.store import publish_observations
from quant_investor.market.dag import context as dag_context
from quant_investor.market.dag.assembly import _attach_symbol_to_ic_decision
from quant_investor.market.dag.decision import _run_portfolio_construction_phase


SYMBOL = "000001.SZ"


class _EmptyCalibrationHistory:
    """Deterministic no-history boundary; all decision classes remain real."""

    @staticmethod
    def calibration_stats(_branch_name: str, _score: float) -> dict[str, float]:
        return {
            "probability": 0.50,
            "sample_size": 0.0,
            "recent_failure_rate": 0.0,
        }


def _observation() -> dict[str, object]:
    return {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": "2024-04-30",
        "release_at": "2024-05-01T06:00:00+00:00",
        "available_at": "2024-05-01T06:00:00+00:00",
        "vintage_id": "initial",
        "value": 50.2,
        "unit": "index",
        "frequency": "monthly",
        "source_system": "nbs_official",
        "source_record_id": "nbs:pmi:2024-04",
        "source_url": "https://www.stats.gov.cn/data/pmi",
        "fetched_at": "2024-05-01T06:05:00+00:00",
        "quality_status": "pass",
    }


def _branch_results() -> dict[str, BranchResult]:
    return {
        "quant": BranchResult(
            branch_name="quant",
            score=0.68,
            confidence=0.82,
            symbol_scores={SYMBOL: 0.68},
            metadata={"reliability": 0.70},
        ),
        "fundamental": BranchResult(
            branch_name="fundamental",
            score=0.52,
            confidence=0.76,
            symbol_scores={SYMBOL: 0.52},
            metadata={
                "reliability": 0.60,
                "fundamental_data_generation_by_symbol": {
                    SYMBOL: "control-invariance-fixture"
                },
                "fundamental_data_generation_status_by_symbol": {
                    SYMBOL: "confirmed"
                },
            },
        ),
    }


def _branch_verdicts() -> tuple[dict[str, BranchVerdict], BranchVerdict]:
    macro = BranchVerdict(
        agent_name="MacroAgent",
        thesis="macro context remains advisory",
        status=AgentStatus.SUCCESS,
        direction=Direction.NEUTRAL,
        action=ActionLabel.HOLD,
        final_score=0.15,
        final_confidence=0.70,
        metadata={"target_gross_exposure": 0.65},
    )
    verdicts = {
        "quant": BranchVerdict(
            agent_name="QuantAgent",
            symbol=SYMBOL,
            thesis="quant evidence is positive",
            status=AgentStatus.SUCCESS,
            direction=Direction.BULLISH,
            action=ActionLabel.BUY,
            final_score=0.68,
            final_confidence=0.82,
        ),
        "fundamental": BranchVerdict(
            agent_name="FundamentalAgent",
            symbol=SYMBOL,
            thesis="fundamental evidence is positive",
            status=AgentStatus.SUCCESS,
            direction=Direction.BULLISH,
            action=ActionLabel.BUY,
            final_score=0.52,
            final_confidence=0.76,
        ),
        "macro": macro,
    }
    return verdicts, macro


def _base_context() -> GlobalContext:
    return GlobalContext(
        market="CN",
        universe_key="full_a",
        latest_trade_date="20240510",
        universe_symbols=[SYMBOL],
        symbol_name_map={SYMBOL: "Ping An Bank"},
        industry_map={SYMBOL: "Banking"},
        liquidity_filter={"liquidity_scores": {SYMBOL: 0.90}},
        macro_regime="趋势上涨",
        cross_section_quant={"breadth": 0.58},
        risk_budget={
            "target_exposure": 0.65,
            "max_single_weight": 0.20,
            "sector_bucket_limit": 2,
        },
        metadata={
            "selection_profile": {"funnel_profile": "momentum_leader"},
            "symbol_market_state": {
                SYMBOL: {
                    "industry": "Banking",
                    "momentum_strength": 0.62,
                    "fake_breakout_risk": 0.10,
                }
            },
            "candidate_sector_counts": {"Banking": 1},
        },
    )


def _real_control_surface(
    base_context: GlobalContext,
    observer_metadata: dict[str, object],
) -> dict[str, object]:
    global_context = deepcopy(base_context)
    global_context.metadata["macro_v2_observer"] = deepcopy(observer_metadata)

    prior = HierarchicalPriorBuilder().build_prior(SYMBOL, global_context)
    likelihoods = SignalLikelihoodMapper(
        calibration_store=_EmptyCalibrationHistory(),
        global_context=global_context,
    ).compute_likelihoods(
        branch_results=_branch_results(),
        symbol=SYMBOL,
        candidate_symbols={SYMBOL},
    )
    posterior = BayesianPosteriorEngine().compute_posterior(
        prior,
        likelihoods,
        symbol=SYMBOL,
        company_name="Ping An Bank",
        regime=global_context.macro_regime,
        is_degraded={"quant": False, "fundamental": False},
    )

    branch_verdicts, macro_verdict = _branch_verdicts()
    shortlist = [
        ShortlistItem(
            symbol=SYMBOL,
            company_name="Ping An Bank",
            rank_score=posterior.posterior_action_score,
            action=ActionLabel.BUY,
            confidence=posterior.posterior_confidence,
            expected_upside=posterior.posterior_expected_alpha,
            metadata={
                "history_confidence": posterior.metadata["history_confidence"],
                "momentum_strength": posterior.metadata["momentum_strength"],
                "fake_breakout_penalty": posterior.metadata[
                    "fake_breakout_penalty"
                ],
                "posterior_edge_after_costs": posterior.posterior_edge_after_costs,
            },
        )
    ]
    control_state = _run_portfolio_construction_phase(
        shortlist=shortlist,
        branch_summaries=branch_verdicts,
        macro_verdict=macro_verdict,
        global_context=global_context,
        data_quality_issues=[],
        ic_hints_by_symbol={},
        research_by_symbol={SYMBOL: branch_verdicts},
        tradability_snapshot={
            SYMBOL: {
                "company_name": "Ping An Bank",
                "is_tradable": True,
                "sector": "Banking",
                "industry": "Banking",
                "liquidity_score": 0.90,
                "momentum_strength": 0.62,
                "fake_breakout_risk": 0.10,
            }
        },
        funnel_summary={"final_candidates": 1},
        bayesian_records=[posterior],
        candidate_symbols=[SYMBOL],
        portfolio_master_output=None,
        portfolio_master_meta={"status": "disabled", "advisory_only": True},
        risk_guard_cls=RiskGuard,
        ic_coordinator_cls=ICCoordinator,
        portfolio_constructor_cls=PortfolioConstructor,
        attach_symbol_to_ic_decision_fn=_attach_symbol_to_ic_decision,
    )

    return {
        "likelihoods": asdict(likelihoods),
        "posterior": asdict(posterior),
        "risk": control_state.risk_decision.to_dict(),
        "ic": [decision.to_dict() for decision in control_state.ic_decisions],
        "portfolio": control_state.portfolio_plan.to_dict(),
    }


def test_real_control_chain_is_invariant_to_active_macro_observer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observations_root = tmp_path / "macro_observations"
    output_root = tmp_path / "observer_results"
    published = publish_observations(
        [_observation()],
        root=observations_root,
        run_id="observer-control-invariance",
    )
    assert published["promoted"] is True

    monkeypatch.setattr(
        dag_context.config,
        "MACRO_V2_OBSERVATIONS_PATH",
        str(observations_root),
    )
    monkeypatch.setattr(
        dag_context.config,
        "MACRO_V2_OBSERVER_OUTPUT_DIR",
        str(output_root),
    )
    monkeypatch.setattr(dag_context.config, "MACRO_V2_PRODUCTION_ENABLED", False)
    monkeypatch.setattr(
        dag_context.config,
        "MACRO_V2_PRODUCTION_KILL_SWITCH",
        True,
    )

    monkeypatch.setattr(dag_context.config, "MACRO_V2_OBSERVER_ENABLED", False)
    monkeypatch.setattr(dag_context.config, "MACRO_V2_OBSERVER_KILL_SWITCH", True)
    disabled = dag_context._macro_v2_observer_metadata(
        market="CN",
        as_of="20240510",
    )

    monkeypatch.setattr(dag_context.config, "MACRO_V2_OBSERVER_ENABLED", True)
    monkeypatch.setattr(dag_context.config, "MACRO_V2_OBSERVER_KILL_SWITCH", False)
    active = dag_context._macro_v2_observer_metadata(
        market="CN",
        as_of="20240510",
    )

    assert disabled["active"] is False
    assert active["active"] is True
    assert active["applied"] is False
    assert active["production_eligible"] is False
    assert active["observation_generation"]["generation_id"] == (
        "observer-control-invariance"
    )
    assert active["snapshot_hash"]
    assert disabled != active

    base_context = _base_context()
    disabled_surface = _real_control_surface(base_context, disabled)
    active_surface = _real_control_surface(base_context, active)

    assert active_surface == disabled_surface
    assert active_surface["likelihoods"]["quant_likelihood"] != 0.50
    assert active_surface["likelihoods"]["fundamental_likelihood"] != 0.50
    assert active_surface["portfolio"]["target_weights"]
