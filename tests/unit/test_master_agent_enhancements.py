from __future__ import annotations

import pandas as pd

from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.agents.agent_contracts import MasterAgentInput, WhatIfScenario
from quant_investor.agents.master_agent import MasterAgent
from quant_investor.pipeline.parallel_research_pipeline import ParallelResearchPipeline


def _make_frame(end: pd.Timestamp, periods: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "close": [10.0 + index * 0.1 for index in range(periods)],
            "symbol": ["000001.SZ"] * periods,
        }
    )


def test_unified_data_bundle_stale_symbols_detects_old_frames():
    now = pd.Timestamp.now().normalize()
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ", "600519.SH"],
        symbol_data={
            "000001.SZ": _make_frame(now),
            "600519.SH": _make_frame(now - pd.Timedelta(days=12)),
        },
        metadata={"end_date": now.strftime("%Y%m%d")},
    )

    assert bundle.stale_symbols(max_stale_days=3) == ["600519.SH"]


def test_parallel_pipeline_validate_data_bundle_filters_synthetic_and_stale():
    now = pd.Timestamp.now().normalize()
    current_symbol = "000001.SZ"
    stale_symbol = "600519.SH"
    synthetic_symbol = "000858.SZ"
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=[current_symbol, stale_symbol, synthetic_symbol],
        symbol_data={
            current_symbol: _make_frame(now),
            stale_symbol: _make_frame(now - pd.Timedelta(days=12)),
            synthetic_symbol: _make_frame(now),
        },
        fundamentals={
            current_symbol: {},
            stale_symbol: {},
            synthetic_symbol: {},
        },
        event_data={current_symbol: [], stale_symbol: [], synthetic_symbol: []},
        sentiment_data={current_symbol: {}, stale_symbol: {}, synthetic_symbol: {}},
        metadata={
            "end_date": now.strftime("%Y%m%d"),
            "symbol_provenance": {
                current_symbol: {"data_source_status": "real", "is_synthetic": False},
                stale_symbol: {"data_source_status": "real", "is_synthetic": False},
                synthetic_symbol: {"data_source_status": "synthetic_fallback", "is_synthetic": True},
            },
        },
    )

    pipeline = ParallelResearchPipeline(stock_pool=list(bundle.symbols), market="CN", verbose=False)
    validated = pipeline._validate_data_bundle(bundle)

    assert validated.symbols == [current_symbol]
    assert list(validated.symbol_data) == [current_symbol]
    assert validated.metadata["data_gate"]["blocked_symbols"] == [synthetic_symbol, stale_symbol]
    assert validated.metadata["data_source_status"] == "real"
    assert validated.metadata["research_mode"] == "production"


def test_master_agent_normalizes_price_fields_and_what_if_scenarios():
    agent = MasterAgent(llm_client=object(), model="dummy", timeout=1.0, max_tokens=16)
    raw = {
        "final_conviction": "buy",
        "final_score": 0.4,
        "confidence": 0.8,
        "top_picks": [
            {
                "symbol": "000001.SZ",
                "action": "buy",
                "conviction": "buy",
                "rationale": "structured setup",
                "target_weight": 0.15,
                "entry_price": 10.0,
                "target_price": 12.0,
                "stop_loss": 9.0,
                "what_if_scenarios": [
                    {
                        "scenario": "加仓",
                        "trigger_condition": "突破 11.0",
                        "expected_outcome": "趋势延续",
                        "probability": 0.6,
                    }
                ],
            },
            {
                "symbol": "600519.SH",
                "action": "sell",
                "conviction": "sell",
                "rationale": "risk off",
                "target_weight": 0.0,
                "entry_price": 20.0,
                "target_price": 18.0,
                "stop_loss": 18.5,
            },
        ],
    }
    agent_input = MasterAgentInput(
        branch_reports={},
        risk_report=None,
        ensemble_baseline={"aggregate_score": 0.35},
        market_regime="default",
        candidate_symbols=["000001.SZ", "600519.SH"],
    )

    output = agent._parse_and_bound(raw, agent_input)

    first = output.top_picks[0]
    assert first.position_size_pct == 0.15
    assert round(first.risk_reward_ratio or 0.0, 2) == 2.0
    assert isinstance(first.what_if_scenarios[0], WhatIfScenario)

    second = output.top_picks[1]
    assert second.entry_price is None
    assert second.target_price is None
    assert second.stop_loss is None


def test_phased_research_pipeline_builds_phase1_context(monkeypatch):
    now = pd.Timestamp.now().normalize()
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ"],
        symbol_data={"000001.SZ": _make_frame(now)},
        fundamentals={"000001.SZ": {}},
        event_data={"000001.SZ": []},
        sentiment_data={"000001.SZ": {}},
        metadata={
            "end_date": now.strftime("%Y%m%d"),
            "symbol_provenance": {
                "000001.SZ": {"data_source_status": "real", "is_synthetic": False}
            },
        },
    )

    pipeline = ParallelResearchPipeline(stock_pool=["000001.SZ"], market="CN", verbose=False)
    phase1_results = {
        "macro": BranchResult(
            branch_name="macro",
            score=0.25,
            confidence=0.7,
            signals={"macro_regime": "bull", "liquidity_signal": "🟢"},
            symbol_scores={"000001.SZ": 0.25},
            success=True,
            metadata={"branch_mode": "macro_terminal", "reliability": 0.9},
        ),
        "quant": BranchResult(
            branch_name="quant",
            score=0.18,
            confidence=0.66,
            signals={
                "alpha_factors": ["mom"],
                "factor_exposures": {"000001.SZ": {"mom": 0.3}},
            },
            symbol_scores={"000001.SZ": 0.18},
            success=True,
            metadata={"branch_mode": "alpha_research", "reliability": 0.8},
        ),
    }
    phase2_results = {
        "kline": BranchResult(branch_name="kline", score=0.1, confidence=0.6, symbol_scores={"000001.SZ": 0.1}),
        "fundamental": BranchResult(branch_name="fundamental", score=0.2, confidence=0.65, symbol_scores={"000001.SZ": 0.2}),
        "intelligence": BranchResult(branch_name="intelligence", score=0.05, confidence=0.55, symbol_scores={"000001.SZ": 0.05}),
    }
    seen = {}

    def _fake_run_branch_group(self, data_bundle_arg, branch_names):
        seen["phase1_branch_names"] = list(branch_names)
        return phase1_results

    def _fake_run_symbol_parallel_phase(self, data_bundle=None, branch_names=None, phase1_context=None, **kwargs):
        seen["phase1_context"] = dict(phase1_context)
        seen["phase2_branch_names"] = list(branch_names)
        return phase2_results

    monkeypatch.setattr(ParallelResearchPipeline, "_run_branch_group", _fake_run_branch_group)
    monkeypatch.setattr(ParallelResearchPipeline, "_run_symbol_parallel_phase", _fake_run_symbol_parallel_phase)

    results = pipeline._run_branches_phased(bundle, execution_log=[])

    assert seen["phase1_branch_names"] == ["macro", "quant"]
    assert seen["phase2_branch_names"] == ["kline", "fundamental", "intelligence"]
    assert seen["phase1_context"]["macro_regime"] == "bull"
    assert bundle.metadata["phase1_context"]["macro_regime"] == "bull"
    assert set(results) == {"macro", "quant", "kline", "fundamental", "intelligence"}
