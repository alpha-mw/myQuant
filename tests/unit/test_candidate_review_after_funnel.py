from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
)
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.market.dag_executor import execute_market_dag
from quant_investor.market.shared_csv_reader import SharedCSVReadResult
from quant_investor.model_roles import ModelRoleResolution


def _frame(seed: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-03-01", periods=40),
            "close": [10.0 + seed + idx * 0.1 for idx in range(40)],
            "volume": [1_000_000 + idx * 1_000 for idx in range(40)],
        }
    )


@dataclass
class _FakeFunnelOutput:
    candidates: list[str]
    candidate_scores: dict[str, float]
    excluded_symbols: dict[str, str]
    funnel_metadata: dict[str, object]


class _FakeReader:
    def __init__(self, *args, **kwargs):
        self._frames = {
            "A": _frame(0.0),
            "B": _frame(1.0),
            "C": _frame(2.0),
            "D": _frame(3.0),
        }

    def list_symbols(self, universe_key: str = "full_a"):
        return list(self._frames)

    def read_symbol_frame(self, symbol: str, *, universe_key: str = "full_a"):
        return SharedCSVReadResult(
            frame=self._frames[symbol],
            path=f"/tmp/{symbol}.csv",
            symbol=symbol,
            category="full_a",
            universe_key=universe_key,
            resolver_trace={"resolution_strategy": "logical_full_a"},
            issues=[],
        )

    def snapshot(self):
        return {
            "resolution_strategy": "logical_full_a",
            "directory_priority": ["full_a"],
            "physical_directories_used_for_full_a": ["/tmp/full_a"],
        }


def test_candidate_review_only_runs_after_funnel(monkeypatch):
    import quant_investor.market.dag_executor as dag_module

    reviewed: dict[str, list[str]] = {"kline_shortlist": [], "fundamental": [], "intelligence": [], "kline_full_market": []}

    class _FakeFunnel:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *, quant_result, kline_result, global_context):
            assert set(kline_result.symbol_scores) == {"A", "B", "C", "D"}
            return _FakeFunnelOutput(
                candidates=["A", "B"],
                candidate_scores={"A": 0.9, "B": 0.8},
                excluded_symbols={"C": "rank_cutoff", "D": "rank_cutoff"},
                funnel_metadata={"after_gates": 4, "final_candidates": 2},
            )

    def _fake_kline_run(self, payload):
        symbol = list(payload["stock_pool"])[0] if payload.get("stock_pool") else ""
        mode = str(payload.get("mode", "")).lower()
        if mode == "full_market":
            reviewed["kline_full_market"] = list(payload["stock_pool"])
            return BranchVerdict(
                agent_name="kline",
                thesis="full market screen",
                final_score=0.3,
                final_confidence=0.6,
                metadata={
                    "legacy_symbol_scores": {"A": 0.9, "B": 0.8, "C": 0.2, "D": 0.1},
                },
            )
        reviewed["kline_shortlist"].append(symbol)
        return BranchVerdict(
            agent_name="kline",
            thesis=f"kline shortlist {symbol}",
            symbol=symbol,
            final_score=0.5,
            final_confidence=0.7,
            metadata={"legacy_symbol_scores": {symbol: 0.5}},
        )

    def _fake_fundamental_run(self, payload):
        symbol = list(payload["stock_pool"])[0]
        reviewed["fundamental"].append(symbol)
        return BranchVerdict(
            agent_name="fundamental",
            thesis=f"fundamental {symbol}",
            symbol=symbol,
            final_score=0.4,
            final_confidence=0.7,
        )

    def _fake_intelligence_run(self, payload):
        symbol = list(payload["stock_pool"])[0]
        reviewed["intelligence"].append(symbol)
        return BranchVerdict(
            agent_name="intelligence",
            thesis=f"intelligence {symbol}",
            symbol=symbol,
            final_score=0.3,
            final_confidence=0.65,
        )

    def _fake_macro_run(self, payload):
        return BranchVerdict(
            agent_name="macro",
            thesis="macro stable",
            final_score=0.2,
            final_confidence=0.8,
            metadata={"regime": "neutral", "target_gross_exposure": 0.5},
        )

    def _fake_prior(self, symbol, global_context):
        return PriorSet(composite_prior=0.55)

    def _fake_likelihoods(self, *, branch_results, symbol, candidate_symbols=None):
        return LikelihoodSet(
            quant_likelihood=0.6,
            kline_likelihood=0.7,
            fundamental_likelihood=0.65,
            intelligence_likelihood=0.6,
        )

    def _fake_posterior(self, prior, likelihoods, *, symbol, company_name, regime, is_degraded):
        rank_score = 0.9 if symbol == "A" else 0.8
        return PosteriorResult(
            symbol=symbol,
            company_name=company_name,
            prior=prior,
            likelihoods=likelihoods,
            posterior_win_rate=0.62,
            posterior_expected_alpha=0.11,
            posterior_confidence=0.78,
            posterior_action_score=rank_score,
            posterior_edge_after_costs=0.08,
            posterior_capacity_penalty=0.01,
            evidence_sources=["quant", "kline", "fundamental", "intelligence"],
            action_threshold_used=0.55,
        )

    def _fake_master(*args, **kwargs):
        return None, {"status": "fallback", "confidence": 0.5, "portfolio_narrative": "fallback"}

    def _fake_risk_run(self, payload):
        return RiskDecision(gross_exposure_cap=0.5, target_exposure_cap=0.5, max_weight=0.2)

    def _fake_ic_run(self, payload):
        return ICDecision(
            action=ActionLabel.BUY,
            final_score=0.6,
            final_confidence=0.7,
        )

    def _fake_portfolio_run(self, payload):
        symbols: list[str] = []
        for decision in payload["ic_decisions"]:
            for symbol in list(decision.selected_symbols):
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        weights = {symbol: 0.25 for symbol in symbols}
        return PortfolioPlan(
            target_exposure=0.5,
            target_gross_exposure=0.5,
            target_net_exposure=0.5,
            cash_ratio=0.5,
            target_weights=weights,
            target_positions=weights,
            position_limits={symbol: 0.25 for symbol in symbols},
        )

    def _fake_narrator_run(self, payload):
        return ReportBundle(
            markdown_report="# report",
            shortlist=list(payload.get("shortlist", [])),
            portfolio_decision=payload.get("portfolio_decision"),
            execution_trace=payload.get("execution_trace"),
            what_if_plan=payload.get("what_if_plan"),
            metadata={"funnel_summary": payload.get("funnel_summary", {})},
        )

    monkeypatch.setattr(dag_module, "SharedCSVReader", _FakeReader)
    monkeypatch.setattr(dag_module, "DeterministicFunnel", _FakeFunnel)
    monkeypatch.setattr(dag_module.KlineAgent, "run", _fake_kline_run)
    monkeypatch.setattr(dag_module.FundamentalAgent, "run", _fake_fundamental_run)
    monkeypatch.setattr(dag_module.IntelligenceAgent, "run", _fake_intelligence_run)
    monkeypatch.setattr(dag_module.MacroAgent, "run", _fake_macro_run)
    monkeypatch.setattr(dag_module.HierarchicalPriorBuilder, "build_prior", _fake_prior)
    monkeypatch.setattr(dag_module.SignalLikelihoodMapper, "compute_likelihoods", _fake_likelihoods)
    monkeypatch.setattr(dag_module.BayesianPosteriorEngine, "compute_posterior", _fake_posterior)
    monkeypatch.setattr(dag_module, "_portfolio_master_advisory", _fake_master)
    monkeypatch.setattr(dag_module.RiskGuard, "run", _fake_risk_run)
    monkeypatch.setattr(dag_module.ICCoordinator, "run", _fake_ic_run)
    monkeypatch.setattr(dag_module.PortfolioConstructor, "run", _fake_portfolio_run)
    monkeypatch.setattr(dag_module.NarratorAgent, "run", _fake_narrator_run)
    monkeypatch.setattr(dag_module, "_load_company_name_map", lambda market: {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"})
    monkeypatch.setattr(dag_module, "detect_provider_health", lambda **kwargs: {"kline": {"mode": "hybrid"}})
    monkeypatch.setattr(
        dag_module,
        "resolve_model_role",
        lambda **kwargs: ModelRoleResolution(
            role=str(kwargs.get("role", "")),
            primary_model="deepseek-chat" if str(kwargs.get("role", "")) == "branch" else "moonshot-v1-128k",
            fallback_model="",
            resolved_model="deepseek-chat" if str(kwargs.get("role", "")) == "branch" else "moonshot-v1-128k",
            provider_available=True,
        ),
    )

    result = execute_market_dag(
        market="CN",
        universe="full_a",
        mode="sample",
        batch_size=4,
        total_capital=1_000_000,
        top_k=2,
        enable_agent_layer=False,
        verbose=False,
    )

    assert reviewed["kline_full_market"] == ["A", "B", "C", "D"]
    assert reviewed["kline_shortlist"] == ["A", "B"]
    assert reviewed["fundamental"] == ["A", "B"]
    assert reviewed["intelligence"] == ["A", "B"]
    assert result["global_context"].universe_tiers["shortlistable"] == ["A", "B"]
    assert list(result["portfolio_decision"].target_weights) == ["A", "B"]
    assert result["portfolio_decision"].what_if_plan is not None
    assert result["portfolio_decision"].execution_trace is not None
