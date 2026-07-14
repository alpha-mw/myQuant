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
from quant_investor.factors.runtime import RuntimeFactorScore
from quant_investor.market.branch_readiness import (
    BranchDataReadiness,
    BranchGovernanceReport,
    STATUS_BLOCK,
    STATUS_PASS,
    SOURCE_TUSHARE,
)
from quant_investor.market.dag_executor import execute_market_dag
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.market.runtime_profile import MarketRuntimeProfiler
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
    single_read_count = 0
    batch_read_count = 0
    batch_read_columns: tuple[str, ...] = ()
    batch_read_start_date = ""

    def __init__(self, *args, **kwargs):
        type(self).single_read_count = 0
        type(self).batch_read_count = 0
        type(self).batch_read_columns = ()
        type(self).batch_read_start_date = ""
        self._frames = {
            "A": _frame(0.0),
            "B": _frame(1.0),
            "C": _frame(2.0),
            "D": _frame(3.0),
        }

    def list_symbols(self, universe_key: str = "full_a"):
        return list(self._frames)

    def read_symbol_frame(self, symbol: str, *, universe_key: str = "full_a"):
        type(self).single_read_count += 1
        return self._read_symbol_frame(symbol, universe_key=universe_key)

    def read_symbol_frames(
        self,
        symbols,
        *,
        universe_key: str = "full_a",
        columns=None,
        start_date: str = "",
    ):
        type(self).batch_read_count += 1
        type(self).batch_read_columns = tuple(str(column) for column in (columns or ()))
        type(self).batch_read_start_date = str(start_date or "")
        return {
            symbol: self._read_symbol_frame(symbol, universe_key=universe_key)
            for symbol in symbols
        }

    def _read_symbol_frame(self, symbol: str, *, universe_key: str = "full_a"):
        return MarketDataReadResult(
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
    import quant_investor.market.dag.context as dag_context
    import quant_investor.market.dag.packets as dag_packets

    reviewed: dict[str, list[str]] = {"fundamental": []}
    frame_summary_calls = {"count": 0}
    provider_health_calls = {"count": 0}
    original_frame_summary = dag_packets._frame_summary

    def _counting_frame_summary(frame):
        frame_summary_calls["count"] += 1
        return original_frame_summary(frame)

    def _counting_provider_health(**kwargs):
        provider_health_calls["count"] += 1
        return {
            "agent": {"model": str(kwargs.get("agent_model", "")), "available": False},
            "master": {"model": str(kwargs.get("master_model", "")), "available": False},
        }

    def _governance_blocked_quant(_frames):
        return RuntimeFactorScore(
            symbol_scores={str(symbol): 0.0 for symbol in _frames},
            governance_status="governance_blocked",
            factor_mode="governance_blocked",
            production_eligible=False,
            runtime_blockers=["fixture_governance_blocked"],
        )

    class _FakeFunnel:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *, quant_result, global_context):
            assert set(quant_result.symbol_scores) == {"A", "B", "C", "D"}
            return _FakeFunnelOutput(
                candidates=["A", "B"],
                candidate_scores={"A": 0.9, "B": 0.8},
                excluded_symbols={"C": "rank_cutoff", "D": "rank_cutoff"},
                funnel_metadata={"after_gates": 4, "final_candidates": 2},
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
            fundamental_likelihood=0.65,
        )

    def _fake_posterior(
        self,
        prior,
        likelihoods,
        *,
        symbol,
        company_name,
        regime,
        is_degraded,
    ):
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
            evidence_sources=["quant", "fundamental"],
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

    monkeypatch.setattr(dag_module, "MarketDataReader", _FakeReader)
    monkeypatch.setattr(dag_packets, "_frame_summary", _counting_frame_summary)
    monkeypatch.setattr(
        dag_packets,
        "score_with_mined_factors",
        _governance_blocked_quant,
    )
    monkeypatch.setattr(dag_module, "DeterministicFunnel", _FakeFunnel)
    monkeypatch.setattr(dag_module.FundamentalAgent, "run", _fake_fundamental_run)
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
    monkeypatch.setattr(dag_module, "detect_provider_health", _counting_provider_health)
    monkeypatch.setattr(
        dag_context,
        "assess_branch_data_readiness",
        lambda **kwargs: BranchGovernanceReport(
            run_id="fixture",
            market="CN",
            category="full_a",
            as_of="2026-03-01",
            readiness={
                branch: BranchDataReadiness(
                    branch=branch,
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                )
                for branch in ("quant", "fundamental", "macro")
            },
            blocked_symbols=[],
            quantifiable_universe=["A", "B", "C", "D"],
            investable_universe=["A", "B"],
            branch_data={},
        ),
    )
    monkeypatch.setattr(
        dag_context,
        "write_branch_readiness_report",
        lambda report: {"json": "fixture.json", "md": "fixture.md", "csv": "fixture.csv"},
    )
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

    runtime_profiler = MarketRuntimeProfiler(market="CN", universe="full_a", categories=["full_a"])

    result = execute_market_dag(
        market="CN",
        universe="full_a",
        mode="sample",
        batch_size=4,
        total_capital=1_000_000,
        top_k=2,
        data_snapshot={"local_latest_trade_date": "20260301", "freshness_mode": "stable"},
        enable_agent_layer=False,
        verbose=False,
        runtime_profiler=runtime_profiler,
    )

    assert reviewed["fundamental"] == ["A", "B"]
    assert not hasattr(dag_module, "IntelligenceAgent")
    assert result["global_context"].universe_tiers["shortlistable"] == ["A", "B"]
    assert list(result["portfolio_decision"].target_weights) == ["A", "B"]
    assert result["portfolio_decision"].what_if_plan is not None
    assert result["portfolio_decision"].execution_trace is not None
    quant_summary = result["branch_summaries"]["quant"]
    assert result["global_context"].cross_section_quant["breadth"] == 1.0
    assert quant_summary.final_score == 0.0
    assert quant_summary.final_confidence == 0.0
    assert quant_summary.metadata["production_quant_evidence"] is False
    assert quant_summary.metadata["cross_section_diagnostic_only"] is True
    assert _FakeReader.batch_read_count == 1
    assert _FakeReader.single_read_count == 0
    assert {"ts_code", "trade_date", "close", "vol", "amount"}.issubset(
        set(_FakeReader.batch_read_columns)
    )
    assert _FakeReader.batch_read_start_date == "20250105"
    stage_names = {stage["name"] for stage in runtime_profiler.stages}
    assert {
        "dag_symbol_list",
        "dag_batch_read",
        "dag_tradability_snapshot",
        "dag_quant_context",
        "dag_cross_section_quant",
        "dag_market_snapshot",
        "dag_macro_verdict",
        "dag_global_quant_verdict",
        "dag_quant_branch_result",
        "dag_funnel",
        "dag_branch_readiness",
        "dag_candidate_research",
        "dag_bayesian_selection",
        "dag_control_chain",
        "dag_reporting_artifacts",
    }.issubset(stage_names)
    batch_stage = next(stage for stage in runtime_profiler.stages if stage["name"] == "dag_batch_read")
    assert batch_stage["metadata"]["batch_result_count"] == 4
    assert batch_stage["metadata"]["per_symbol_fallback_count"] == 0
    assert batch_stage["metadata"]["projected_column_count"] >= 5
    assert batch_stage["metadata"]["runtime_lookback_calendar_days"] == 420
    assert batch_stage["metadata"]["runtime_lookback_start_date"] == _FakeReader.batch_read_start_date
    assert frame_summary_calls["count"] <= 8
    assert provider_health_calls["count"] == 1


def test_holding_single_review_runs_branches_when_readiness_blocks_symbol(monkeypatch):
    import quant_investor.market.dag_executor as dag_module
    import quant_investor.market.dag.context as dag_context

    reviewed: dict[str, list[str]] = {"fundamental": []}

    class _SingleFunnel:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *, quant_result, global_context):
            return _FakeFunnelOutput(
                candidates=["A"],
                candidate_scores={"A": 0.9},
                excluded_symbols={},
                funnel_metadata={"after_gates": 1, "final_candidates": 1},
            )

    def _fake_fundamental_run(self, payload):
        symbol = list(payload["stock_pool"])[0]
        reviewed["fundamental"].append(symbol)
        return BranchVerdict(
            agent_name="fundamental",
            thesis=f"fundamental {symbol}",
            symbol=symbol,
            final_score=0.2,
            final_confidence=0.4,
            investment_risks=["fundamental readiness blocked; limited evidence"],
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
        return PriorSet(composite_prior=0.5)

    def _fake_likelihoods(self, *, branch_results, symbol, candidate_symbols=None):
        return LikelihoodSet(
            quant_likelihood=0.5,
            fundamental_likelihood=0.4,
        )

    def _fake_posterior(
        self,
        prior,
        likelihoods,
        *,
        symbol,
        company_name,
        regime,
        is_degraded,
    ):
        return PosteriorResult(
            symbol=symbol,
            company_name=company_name,
            prior=prior,
            likelihoods=likelihoods,
            posterior_win_rate=0.5,
            posterior_expected_alpha=0.0,
            posterior_confidence=0.4,
            posterior_action_score=0.3,
            posterior_edge_after_costs=0.0,
            evidence_sources=["quant", "fundamental"],
            action_threshold_used=0.55,
        )

    def _fake_risk_run(self, payload):
        return RiskDecision(
            gross_exposure_cap=0.5,
            target_exposure_cap=0.5,
            max_weight=0.2,
            blocked_symbols=["A"],
        )

    def _fake_ic_run(self, payload):
        return ICDecision(action=ActionLabel.HOLD, final_score=0.3, final_confidence=0.4)

    def _fake_portfolio_run(self, payload):
        return PortfolioPlan(target_exposure=0.0, target_gross_exposure=0.0, cash_ratio=1.0)

    def _fake_narrator_run(self, payload):
        return ReportBundle(
            markdown_report="# report",
            shortlist=list(payload.get("shortlist", [])),
            portfolio_decision=payload.get("portfolio_decision"),
            execution_trace=payload.get("execution_trace"),
            what_if_plan=payload.get("what_if_plan"),
        )

    monkeypatch.setattr(dag_module, "MarketDataReader", _FakeReader)
    monkeypatch.setattr(dag_module, "DeterministicFunnel", _SingleFunnel)
    monkeypatch.setattr(dag_module.FundamentalAgent, "run", _fake_fundamental_run)
    monkeypatch.setattr(dag_module.MacroAgent, "run", _fake_macro_run)
    monkeypatch.setattr(dag_module.HierarchicalPriorBuilder, "build_prior", _fake_prior)
    monkeypatch.setattr(
        dag_module.SignalLikelihoodMapper,
        "compute_likelihoods",
        _fake_likelihoods,
    )
    monkeypatch.setattr(
        dag_module.BayesianPosteriorEngine,
        "compute_posterior",
        _fake_posterior,
    )
    monkeypatch.setattr(
        dag_module,
        "_portfolio_master_advisory",
        lambda *args, **kwargs: (None, {"status": "fallback"}),
    )
    monkeypatch.setattr(dag_module.RiskGuard, "run", _fake_risk_run)
    monkeypatch.setattr(dag_module.ICCoordinator, "run", _fake_ic_run)
    monkeypatch.setattr(dag_module.PortfolioConstructor, "run", _fake_portfolio_run)
    monkeypatch.setattr(dag_module.NarratorAgent, "run", _fake_narrator_run)
    monkeypatch.setattr(dag_module, "_load_company_name_map", lambda market: {"A": "Alpha"})
    monkeypatch.setattr(dag_module, "detect_provider_health", lambda **_kwargs: {})
    monkeypatch.setattr(
        dag_context,
        "assess_branch_data_readiness",
        lambda **kwargs: BranchGovernanceReport(
            run_id="fixture",
            market="CN",
            category="full_a",
            as_of="2026-03-01",
            readiness={
                "quant": BranchDataReadiness(
                    branch="quant",
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                ),
                "fundamental": BranchDataReadiness(
                    branch="fundamental",
                    status=STATUS_BLOCK,
                    coverage_ratio=0.0,
                    source_priority=SOURCE_TUSHARE,
                ),
                "macro": BranchDataReadiness(
                    branch="macro",
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                ),
            },
            blocked_symbols=["A"],
            quantifiable_universe=["A"],
            investable_universe=[],
            branch_data={},
        ),
    )
    monkeypatch.setattr(
        dag_context,
        "write_branch_readiness_report",
        lambda report: {"json": "fixture.json", "md": "fixture.md", "csv": "fixture.csv"},
    )

    result = execute_market_dag(
        market="CN",
        symbols=["A"],
        universe="full_a",
        mode="sample",
        batch_size=1,
        total_capital=1_000_000,
        top_k=1,
        data_snapshot={"local_latest_trade_date": "20260301", "freshness_mode": "stable"},
        enable_agent_layer=False,
        verbose=False,
        recall_context={"holding_symbol": "A"},
    )

    assert reviewed == {"fundamental": ["A"]}
    assert set(result["branch_verdicts_by_symbol"]["A"]) == {
        "quant",
        "fundamental",
        "macro",
    }
    assert result["global_context"].metadata["holding_review_branch_readiness_override"] is True
    assert result["global_context"].universe_tiers["shortlistable"] == ["A"]


def test_holding_single_review_runs_branches_when_funnel_excludes_symbol(monkeypatch):
    import quant_investor.market.dag_executor as dag_module
    import quant_investor.market.dag.context as dag_context

    reviewed: dict[str, list[str]] = {"fundamental": []}

    class _EmptyRequiredThemeFunnel:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *, quant_result, global_context):
            return _FakeFunnelOutput(
                candidates=[],
                candidate_scores={},
                excluded_symbols={"A": "theme_pool_not_core"},
                funnel_metadata={
                    "after_gates": 1,
                    "final_candidates": 0,
                    "theme_pool": {
                        "enabled": True,
                        "required": True,
                        "status": "applied",
                    },
                },
            )

    def _fake_fundamental_run(self, payload):
        symbol = list(payload["stock_pool"])[0]
        reviewed["fundamental"].append(symbol)
        return BranchVerdict(
            agent_name="fundamental",
            thesis=f"fundamental {symbol}",
            symbol=symbol,
            final_score=0.2,
            final_confidence=0.4,
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
        return PriorSet(composite_prior=0.5)

    def _fake_likelihoods(self, *, branch_results, symbol, candidate_symbols=None):
        return LikelihoodSet(
            quant_likelihood=0.5,
            fundamental_likelihood=0.4,
        )

    def _fake_posterior(
        self,
        prior,
        likelihoods,
        *,
        symbol,
        company_name,
        regime,
        is_degraded,
    ):
        return PosteriorResult(
            symbol=symbol,
            company_name=company_name,
            prior=prior,
            likelihoods=likelihoods,
            posterior_win_rate=0.5,
            posterior_expected_alpha=0.0,
            posterior_confidence=0.4,
            posterior_action_score=0.3,
            posterior_edge_after_costs=0.0,
            evidence_sources=["quant", "fundamental"],
            action_threshold_used=0.55,
        )

    def _fake_risk_run(self, payload):
        return RiskDecision(
            gross_exposure_cap=0.5,
            target_exposure_cap=0.5,
            max_weight=0.2,
            blocked_symbols=["A"],
        )

    def _fake_ic_run(self, payload):
        return ICDecision(action=ActionLabel.HOLD, final_score=0.3, final_confidence=0.4)

    def _fake_portfolio_run(self, payload):
        return PortfolioPlan(target_exposure=0.0, target_gross_exposure=0.0, cash_ratio=1.0)

    def _fake_narrator_run(self, payload):
        return ReportBundle(
            markdown_report="# report",
            shortlist=list(payload.get("shortlist", [])),
            portfolio_decision=payload.get("portfolio_decision"),
            execution_trace=payload.get("execution_trace"),
            what_if_plan=payload.get("what_if_plan"),
        )

    monkeypatch.setattr(dag_module, "MarketDataReader", _FakeReader)
    monkeypatch.setattr(dag_module, "DeterministicFunnel", _EmptyRequiredThemeFunnel)
    monkeypatch.setattr(dag_module.FundamentalAgent, "run", _fake_fundamental_run)
    monkeypatch.setattr(dag_module.MacroAgent, "run", _fake_macro_run)
    monkeypatch.setattr(dag_module.HierarchicalPriorBuilder, "build_prior", _fake_prior)
    monkeypatch.setattr(
        dag_module.SignalLikelihoodMapper,
        "compute_likelihoods",
        _fake_likelihoods,
    )
    monkeypatch.setattr(
        dag_module.BayesianPosteriorEngine,
        "compute_posterior",
        _fake_posterior,
    )
    monkeypatch.setattr(
        dag_module,
        "_portfolio_master_advisory",
        lambda *args, **kwargs: (None, {"status": "fallback"}),
    )
    monkeypatch.setattr(dag_module.RiskGuard, "run", _fake_risk_run)
    monkeypatch.setattr(dag_module.ICCoordinator, "run", _fake_ic_run)
    monkeypatch.setattr(dag_module.PortfolioConstructor, "run", _fake_portfolio_run)
    monkeypatch.setattr(dag_module.NarratorAgent, "run", _fake_narrator_run)
    monkeypatch.setattr(dag_module, "_load_company_name_map", lambda market: {"A": "Alpha"})
    monkeypatch.setattr(dag_module, "detect_provider_health", lambda **_kwargs: {})
    monkeypatch.setattr(
        dag_context,
        "assess_branch_data_readiness",
        lambda **kwargs: BranchGovernanceReport(
            run_id="fixture",
            market="CN",
            category="full_a",
            as_of="2026-03-01",
            readiness={
                "quant": BranchDataReadiness(
                    branch="quant",
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                ),
                "fundamental": BranchDataReadiness(
                    branch="fundamental",
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                ),
                "macro": BranchDataReadiness(
                    branch="macro",
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                    source_priority=SOURCE_TUSHARE,
                ),
            },
            blocked_symbols=[],
            quantifiable_universe=["A"],
            investable_universe=["A"],
            branch_data={},
        ),
    )
    monkeypatch.setattr(
        dag_context,
        "write_branch_readiness_report",
        lambda report: {"json": "fixture.json", "md": "fixture.md", "csv": "fixture.csv"},
    )

    result = execute_market_dag(
        market="CN",
        symbols=["A"],
        universe="full_a",
        mode="sample",
        batch_size=1,
        total_capital=1_000_000,
        top_k=1,
        data_snapshot={"local_latest_trade_date": "20260301", "freshness_mode": "stable"},
        enable_agent_layer=False,
        verbose=False,
        recall_context={"holding_symbol": "A"},
    )

    assert reviewed == {"fundamental": ["A"]}
    assert set(result["branch_verdicts_by_symbol"]["A"]) == {
        "quant",
        "fundamental",
        "macro",
    }
    assert result["global_context"].metadata["holding_review_funnel_override"] is True
    assert result["global_context"].universe_tiers["shortlistable"] == ["A"]
