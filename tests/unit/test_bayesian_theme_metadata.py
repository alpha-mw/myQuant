from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    Direction,
    GlobalContext,
    SymbolResearchPacket,
)
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.decision import _run_bayesian_selection_phase
from quant_investor.market.dag import decision as decision_module
from quant_investor.market.dag.theme_context import extract_symbol_theme_metadata


class FakeContext:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata


def test_extract_symbol_theme_metadata_success_from_theme_rotation():
    context = FakeContext(
        {
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": {"000001.SZ": 0.78},
                "symbol_primary_theme": {"000001.SZ": "industry::AI"},
                "symbol_phase": {"000001.SZ": "confirmed_rotation"},
                "symbol_risk_flags": {"000001.SZ": ["theme_low_breadth"]},
                "theme_scores": {
                    "industry::AI": {
                        "theme_name": "AI",
                        "score": 72.5,
                        "confidence": 0.66,
                        "member_count": 18,
                    }
                },
                "top_themes": [],
            }
        }
    )

    metadata = extract_symbol_theme_metadata(
        global_context=context,
        symbol="000001.SZ",
    )

    assert metadata["available"] is True
    assert metadata["symbol_score"] == pytest.approx(0.78)
    assert metadata["symbol_score_100"] == pytest.approx(78.0)
    assert metadata["primary_theme_id"] == "industry::AI"
    assert metadata["primary_theme_name"] == "AI"
    assert metadata["phase"] == "confirmed_rotation"
    assert metadata["risk_flags"] == ["theme_low_breadth"]
    assert metadata["theme_score"] == pytest.approx(72.5)
    assert metadata["theme_confidence"] == pytest.approx(0.66)
    assert metadata["theme_member_count"] == 18


def test_counterfactual_control_inputs_rebuild_fundamental_branch() -> None:
    actual_fundamental = BranchVerdict(
        agent_name="fundamental",
        symbol="000001.SZ",
        final_score=0.40,
        final_confidence=0.95,
        thesis="actual generic overlay thesis",
        direction=Direction.BULLISH,
        action=ActionLabel.BUY,
        investment_risks=["actual overlay risk"],
        metadata={
            "fundamental_research_runtime": {"applied": True},
            "overlay": {"model": "actual-review"},
            "fundamental_deterministic_control_input": {
                "thesis": "deterministic thesis",
                "status": "SUCCESS",
                "direction": "neutral",
                "action": "hold",
                "confidence_label": "medium",
                "final_score": 0.10,
                "final_confidence": 0.60,
                "investment_risks": ["deterministic risk"],
                "coverage_notes": ["deterministic coverage"],
                "diagnostic_notes": [],
            },
        },
    )
    research = {
        "000001.SZ": {
            "quant": BranchVerdict(agent_name="quant", final_score=0.10),
            "fundamental": actual_fundamental,
            "intelligence": BranchVerdict(agent_name="intelligence", final_score=0.05),
            "macro": BranchVerdict(agent_name="macro", final_score=0.0),
        }
    }

    rebuilt, summaries = decision_module._build_counterfactual_control_inputs(
        research_by_symbol=research,
        counterfactual_by_symbol={
            "000001.SZ": {
                "basis": "without_dossier",
                "fundamental_score": -0.40,
            }
        },
    )

    alternative = rebuilt["000001.SZ"]["fundamental"]
    assert alternative is not actual_fundamental
    assert alternative.final_score == pytest.approx(-0.40)
    assert alternative.direction == Direction.BEARISH
    assert alternative.action == ActionLabel.SELL
    assert alternative.thesis == "deterministic thesis"
    assert alternative.final_confidence == pytest.approx(0.60)
    assert alternative.investment_risks == ["deterministic risk"]
    assert "overlay" not in alternative.metadata
    assert "fundamental_research_runtime" not in alternative.metadata
    assert alternative.metadata["fundamental_research_variant"] == "without_dossier"
    assert summaries["fundamental"].final_score == pytest.approx(-0.40)
    assert actual_fundamental.final_score == pytest.approx(0.40)


def test_extract_symbol_theme_metadata_disabled():
    context = FakeContext(
        {
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "disabled",
            }
        }
    )

    metadata = extract_symbol_theme_metadata(
        global_context=context,
        symbol="000001.SZ",
    )

    assert metadata["available"] is False
    assert metadata["status"] == "disabled"
    assert metadata["symbol_score"] == 0.0


def test_extract_symbol_theme_metadata_missing_symbol():
    context = FakeContext(
        {
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": {"000002.SZ": 0.55},
            }
        }
    )

    metadata = extract_symbol_theme_metadata(
        global_context=context,
        symbol="000001.SZ",
    )

    assert metadata["available"] is False
    assert metadata["status"] == "unavailable"


def test_extract_symbol_theme_metadata_fallback_aliases():
    context = FakeContext(
        {
            "symbol_theme_score": {"000001.SZ": 0.66},
            "symbol_primary_theme": {"000001.SZ": "industry::AI"},
            "symbol_theme_phase": {"000001.SZ": "early_acceleration"},
            "theme_scores": {
                "industry::AI": {
                    "theme_name": "AI",
                    "score": 71.0,
                    "confidence": 0.7,
                    "member_count": 8,
                }
            },
        }
    )

    metadata = extract_symbol_theme_metadata(
        global_context=context,
        symbol="000001.SZ",
    )

    assert metadata["available"] is True
    assert metadata["status"] == "success"
    assert metadata["symbol_score"] == pytest.approx(0.66)
    assert metadata["primary_theme_id"] == "industry::AI"
    assert metadata["primary_theme_name"] == "AI"
    assert metadata["phase"] == "early_acceleration"
    assert metadata["theme_score"] == pytest.approx(71.0)
    assert metadata["theme_confidence"] == pytest.approx(0.7)
    assert metadata["theme_member_count"] == 8


def test_extract_symbol_theme_metadata_malformed_safe():
    context = FakeContext(
        {
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": None,
                "theme_scores": "bad",
                "symbol_risk_flags": {"000001.SZ": "not-a-list"},
            }
        }
    )

    metadata = extract_symbol_theme_metadata(
        global_context=context,
        symbol="000001.SZ",
    )

    assert metadata["available"] is False
    assert metadata["status"] == "unavailable"
    assert metadata["risk_flags"] == []
    assert metadata["symbol_score"] == 0.0


def test_bayesian_record_metadata_integration_if_feasible():
    class PriorBuilder:
        def build_prior(self, symbol: str, global_context: GlobalContext) -> PriorSet:
            return PriorSet(composite_prior=0.50)

    class LikelihoodMapper:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def compute_likelihoods(
            self,
            *,
            branch_results: dict[str, object],
            symbol: str,
            candidate_symbols: set[str],
        ) -> LikelihoodSet:
            fundamental = branch_results.get("fundamental")
            score = (
                float(fundamental.symbol_scores.get(symbol, 0.0))
                if fundamental is not None
                else 0.0
            )
            return LikelihoodSet(quant_likelihood=score)

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
            return PosteriorResult(
                symbol=symbol,
                company_name=company_name,
                prior=prior,
                likelihoods=likelihoods,
                posterior_win_rate=0.61,
                posterior_expected_alpha=0.011,
                posterior_confidence=0.62,
                posterior_action_score=likelihoods.quant_likelihood,
                posterior_edge_after_costs=0.009,
                posterior_capacity_penalty=0.002,
                evidence_sources=["quant"],
                metadata={"momentum_strength": 0.12},
            )

    global_context = GlobalContext(
        macro_regime="未知",
        risk_budget={},
        metadata={
            "theme_rotation": {
                "schema_version": "theme_rotation.v1",
                "status": "success",
                "symbol_scores": {"000001.SZ": 0.78},
                "symbol_primary_theme": {"000001.SZ": "industry::AI"},
                "symbol_phase": {"000001.SZ": "confirmed_rotation"},
                "symbol_risk_flags": {"000001.SZ": ["theme_low_breadth"]},
                "theme_scores": {
                    "industry::AI": {
                        "theme_name": "AI",
                        "score": 72.5,
                        "confidence": 0.66,
                        "member_count": 18,
                    }
                },
            }
        },
    )

    state = _run_bayesian_selection_phase(
        candidate_symbols=["000001.SZ"],
        company_name_map={"000001.SZ": "Ping An Bank"},
        symbol_research_packets={
            "000001.SZ": SymbolResearchPacket(symbol="000001.SZ", category="bank")
        },
        research_by_symbol={
            "000001.SZ": {
                "fundamental": BranchVerdict(
                    metadata={
                        "deterministic_base_score": 0.10,
                        "fundamental_research_runtime": {
                            "request_id": "req-1",
                            "counterfactual": True,
                            "counterfactual_adjusted_score": 0.40,
                            "blockers": [],
                        },
                    }
                )
            }
        },
        branch_summaries={},
        branch_results={
            "fundamental": BranchResult(
                branch_name="fundamental",
                score=0.10,
                confidence=0.60,
                symbol_scores={"000001.SZ": 0.10},
            )
        },
        macro_verdict=BranchVerdict(),
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
        funnel_output=SimpleNamespace(
            excluded_symbols=[],
            funnel_metadata={
                "theme_pool": {
                    "policy": {"regime": "震荡低波"},
                    "symbols": {
                        "000001.SZ": {
                            "admitted": True,
                            "source": "risk_watch",
                            "primary_theme_id": "industry::AI",
                            "primary_theme_name": "AI",
                            "theme_score": 0.725,
                            "symbol_theme_score": 0.78,
                            "theme_pool_score": 0.42,
                            "bucket": "risk_watch_fake_breakout",
                            "phase": "confirmed_rotation",
                            "risk_flags": ["theme_fake_breakout_risk"],
                            "candidate_intent": "research_candidate_not_buy_signal",
                            "score_penalty": 0.22,
                            "theme_forced_admission": True,
                            "theme_pool_reason": "admitted",
                        }
                    },
                }
            },
        ),
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

    record = state.bayesian_records[0]
    assert record.posterior_action_score == pytest.approx(0.10)
    counterfactual = record.metadata["fundamental_research_counterfactual"]
    assert counterfactual["basis"] == "with_dossier"
    assert counterfactual["fundamental_score"] == pytest.approx(0.40)
    assert counterfactual["posterior_action_score"] == pytest.approx(0.40)
    assert counterfactual["rank"] == 1
    assert state.counterfactual_bayesian_records[0].posterior_action_score == pytest.approx(0.40)
    assert (
        state.counterfactual_bayesian_records[0].metadata["fundamental_research_variant"]
        == "with_dossier"
    )
    assert record.metadata["theme_rotation"]["available"] is True
    assert record.metadata["theme_rotation"]["primary_theme_id"] == "industry::AI"
    assert record.metadata["theme_pool"]["bucket"] == "risk_watch_fake_breakout"
    assert record.metadata["theme_pool"]["source"] == "risk_watch"
    assert record.metadata["theme_pool"]["risk_flags"] == ["theme_fake_breakout_risk"]
    assert record.metadata["theme_pool"]["score_penalty"] == pytest.approx(0.22)
    assert record.metadata["theme_pool"]["theme_forced_admission"] is True
