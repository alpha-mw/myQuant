from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.agent_protocol import (
    AgentStatus,
    BranchVerdict,
    GlobalContext,
    SymbolResearchPacket,
)
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.decision import _run_bayesian_selection_phase
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
            assert is_degraded == {"quant": False, "fundamental": True}
            return PosteriorResult(
                symbol=symbol,
                company_name=company_name,
                prior=prior,
                likelihoods=likelihoods,
                posterior_win_rate=0.61,
                posterior_expected_alpha=0.011,
                posterior_confidence=0.62,
                posterior_action_score=0.432,
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

    macro_verdict = BranchVerdict(agent_name="macro")
    state = _run_bayesian_selection_phase(
        candidate_symbols=["000001.SZ"],
        company_name_map={"000001.SZ": "Ping An Bank"},
        symbol_research_packets={
            "000001.SZ": SymbolResearchPacket(symbol="000001.SZ", category="bank")
        },
        research_by_symbol={},
        branch_summaries={
            "quant": BranchVerdict(status=AgentStatus.SUCCESS),
            "fundamental": BranchVerdict(status=AgentStatus.DEGRADED),
            "macro": macro_verdict,
        },
        branch_results={
            "quant": BranchResult(branch_name="quant"),
            "fundamental": BranchResult(
                branch_name="fundamental",
                metadata={"degraded_reason": "fundamental_evidence_incomplete"},
            ),
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
    assert record.posterior_action_score == pytest.approx(0.432)
    assert record.metadata["theme_rotation"]["available"] is True
    assert record.metadata["theme_rotation"]["primary_theme_id"] == "industry::AI"
    assert record.metadata["theme_pool"]["bucket"] == "risk_watch_fake_breakout"
    assert record.metadata["theme_pool"]["source"] == "risk_watch"
    assert record.metadata["theme_pool"]["risk_flags"] == ["theme_fake_breakout_risk"]
    assert record.metadata["theme_pool"]["score_penalty"] == pytest.approx(0.22)
    assert record.metadata["theme_pool"]["theme_forced_admission"] is True


def _selection_gate_kwargs(
    *,
    branch_results: dict[str, object],
    branch_summaries: dict[str, object],
    macro_verdict: object,
) -> dict[str, object]:
    return {
        "candidate_symbols": [],
        "company_name_map": {},
        "symbol_research_packets": {},
        "research_by_symbol": {},
        "branch_summaries": branch_summaries,
        "branch_results": branch_results,
        "macro_verdict": macro_verdict,
        "global_context": object(),
        "model_roles": object(),
        "resolver_snapshot": {},
        "data_quality_issues": [],
        "top_k": 1,
        "all_symbols": [],
        "funnel_output": object(),
        "provider_health": {},
        "master_timeout": 0.0,
        "master_reasoning_effort": "",
        "master_model_resolution": object(),
        "master_candidate_models": [],
        "recall_context": None,
        "hierarchical_prior_builder_cls": object,
        "likelihood_mapper_cls": object,
        "posterior_engine_cls": object,
        "master_agent_cls": object,
        "llm_client_cls": object,
        "portfolio_master_advisory_fn": lambda **kwargs: (None, {}),
    }


@pytest.mark.parametrize("payload_name", ["branch_results", "branch_summaries"])
@pytest.mark.parametrize("missing_branch", CANONICAL_BRANCH_ORDER)
def test_bayesian_selection_rejects_each_missing_canonical_branch(
    payload_name: str,
    missing_branch: str,
) -> None:
    macro_verdict = BranchVerdict(agent_name="macro")
    payloads: dict[str, dict[str, object]] = {
        "branch_results": {
            name: BranchResult(branch_name=name) for name in CANONICAL_BRANCH_ORDER
        },
        "branch_summaries": {
            "quant": BranchVerdict(agent_name="quant"),
            "fundamental": BranchVerdict(agent_name="fundamental"),
            "macro": macro_verdict,
        },
    }
    payloads[payload_name].pop(missing_branch)

    with pytest.raises(
        ValueError,
        match=rf"Bayesian selection {payload_name}.*missing branches: {missing_branch}",
    ):
        _run_bayesian_selection_phase(
            **_selection_gate_kwargs(
                branch_results=payloads["branch_results"],
                branch_summaries=payloads["branch_summaries"],
                macro_verdict=macro_verdict,
            )
        )


def test_bayesian_selection_rejects_macro_verdict_summary_drift() -> None:
    macro_summary = BranchVerdict(agent_name="macro", final_score=0.1)

    with pytest.raises(ValueError, match="macro_verdict must match"):
        _run_bayesian_selection_phase(
            **_selection_gate_kwargs(
                branch_results={
                    name: BranchResult(branch_name=name)
                    for name in CANONICAL_BRANCH_ORDER
                },
                branch_summaries={
                    "quant": BranchVerdict(agent_name="quant"),
                    "fundamental": BranchVerdict(agent_name="fundamental"),
                    "macro": macro_summary,
                },
                macro_verdict=BranchVerdict(agent_name="macro", final_score=0.2),
            )
        )


def test_bayesian_selection_rejects_branch_result_key_name_swap() -> None:
    macro_verdict = BranchVerdict(agent_name="macro")
    branch_results = {
        "quant": BranchResult(branch_name="fundamental"),
        "fundamental": BranchResult(branch_name="quant"),
        "macro": BranchResult(branch_name="macro"),
    }

    with pytest.raises(ValueError, match="key/name mismatch"):
        _run_bayesian_selection_phase(
            **_selection_gate_kwargs(
                branch_results=branch_results,
                branch_summaries={
                    "quant": BranchVerdict(agent_name="quant"),
                    "fundamental": BranchVerdict(agent_name="fundamental"),
                    "macro": macro_verdict,
                },
                macro_verdict=macro_verdict,
            )
        )


def test_bayesian_selection_validates_nested_retired_branch_metadata() -> None:
    macro_verdict = BranchVerdict(agent_name="macro")
    branch_results = {
        "quant": BranchResult(
            branch_name="quant",
            metadata={"intelligence_score": 0.9},
        ),
        "fundamental": BranchResult(branch_name="fundamental"),
        "macro": BranchResult(branch_name="macro"),
    }

    with pytest.raises(ValueError, match="retired Intelligence key"):
        _run_bayesian_selection_phase(
            **_selection_gate_kwargs(
                branch_results=branch_results,
                branch_summaries={
                    "quant": BranchVerdict(agent_name="quant"),
                    "fundamental": BranchVerdict(agent_name="fundamental"),
                    "macro": macro_verdict,
                },
                macro_verdict=macro_verdict,
            )
        )
