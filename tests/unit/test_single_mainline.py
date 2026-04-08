from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_investor.agent_orchestrator as agent_orchestrator_module
import quant_investor.pipeline.mainline as mainline_module
from quant_investor.agent_orchestrator import AgentOrchestrator
from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ConfidenceLabel,
    Direction,
    RiskDecision,
    RiskLevel,
)
from quant_investor.agents.agent_contracts import MasterAgentOutput, SymbolRecommendation
from quant_investor.branch_contracts import BranchResult, PortfolioStrategy, UnifiedDataBundle
from quant_investor.pipeline.mainline import QuantInvestor


ROOT = Path(__file__).resolve().parents[2]


def _make_price_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "close": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0],
            "volume": [2_000_000] * 6,
            "symbol": [symbol] * 6,
        }
    )


def _make_data_bundle(symbols: list[str]) -> UnifiedDataBundle:
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=list(symbols),
        symbol_data={symbol: _make_price_frame(symbol) for symbol in symbols},
        fundamentals={symbol: {"sector": "bank"} for symbol in symbols},
        macro_data={"regime": "neutral"},
    )
    bundle.stock_pool = list(symbols)
    bundle.stock_data = bundle.symbol_data
    bundle.current_prices = {symbol: 10.0 + index for index, symbol in enumerate(symbols)}
    bundle.stock_names = {symbol: symbol for symbol in symbols}
    bundle.fundamental_data = bundle.fundamentals
    return bundle


def _make_verdict(
    agent_name: str,
    score: float,
    confidence: float,
    *,
    symbol: str | None = None,
    metadata: dict[str, object] | None = None,
) -> BranchVerdict:
    if score >= 0.15:
        direction = Direction.BULLISH
        action = ActionLabel.BUY
    elif score <= -0.15:
        direction = Direction.BEARISH
        action = ActionLabel.SELL
    else:
        direction = Direction.NEUTRAL
        action = ActionLabel.HOLD
    return BranchVerdict(
        agent_name=agent_name,
        thesis=f"{agent_name} thesis for {symbol or 'market'}",
        symbol=symbol,
        status=AgentStatus.SUCCESS,
        direction=direction,
        action=action,
        confidence_label=ConfidenceLabel.HIGH if confidence >= 0.7 else ConfidenceLabel.MEDIUM,
        final_score=score,
        final_confidence=confidence,
        investment_risks=[f"投资风险: {agent_name} review"],
        coverage_notes=[f"覆盖说明: {agent_name} covered"],
        diagnostic_notes=[f"工程异常: {agent_name} cached"],
        metadata=dict(metadata or {}),
    )


def _make_research_by_symbol(symbols: list[str]) -> dict[str, dict[str, BranchVerdict]]:
    branch_scores = {
        "kline": 0.62,
        "quant": 0.58,
        "fundamental": 0.66,
        "intelligence": 0.41,
    }
    return {
        symbol: {
            branch_name: _make_verdict(f"{branch_name.title()}Agent", score, 0.8, symbol=symbol)
            for branch_name, score in branch_scores.items()
        }
        for symbol in symbols
    }


def _make_branch_results(symbols: list[str]) -> dict[str, BranchResult]:
    branch_scores = {
        "kline": 0.62,
        "quant": 0.58,
        "fundamental": 0.66,
        "intelligence": 0.41,
        "macro": 0.22,
    }
    results: dict[str, BranchResult] = {}
    for branch_name, score in branch_scores.items():
        signals = {}
        if branch_name == "macro":
            signals = {
                "macro_regime": "neutral",
                "risk_level": "neutral",
                "liquidity_signal": "🟡",
            }
        results[branch_name] = BranchResult(
            branch_name=branch_name,
            signals=signals,
            symbol_scores={symbol: score - index * 0.03 for index, symbol in enumerate(symbols)},
            final_score=score,
            final_confidence=0.8,
            explanation=f"{branch_name} explanation",
            conclusion=f"{branch_name} conclusion",
            investment_risks=[f"投资风险: {branch_name} risk"],
            coverage_notes=[f"覆盖说明: {branch_name} covered"],
            diagnostic_notes=[f"工程异常: {branch_name} cached"],
            success=True,
            metadata={"branch_mode": "fixture"},
        )
    return results


def _make_macro_verdict() -> BranchVerdict:
    return _make_verdict(
        "MacroAgent",
        0.22,
        0.76,
        metadata={"target_gross_exposure": 0.85, "style_bias": "balanced_quality"},
    )


def test_plain_pytest_bootstrap_adds_project_root_to_sys_path():
    assert str(ROOT) in sys.path


def test_mainline_raises_when_requested_symbol_has_no_local_csv(monkeypatch):
    investor = QuantInvestor(stock_pool=["000001.SZ"], verbose=False, enable_agent_layer=False)

    monkeypatch.setattr(
        mainline_module,
        "build_market_data_snapshot",
        lambda **kwargs: {
            "market": "CN",
            "universe_key": "full_a",
            "local_latest_trade_date": "20260326",
            "freshness_mode": "stable",
            "category_symbol_counts": {"full_a": 0},
            "date_distribution_top": [],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 1,
            "summary_text": "请求标的本地无数据。",
            "missing_requested_symbols": ["000001.SZ"],
            "unreadable_requested_symbols": [],
        },
    )

    with pytest.raises(ValueError, match="000001.SZ"):
        investor.run()


def test_master_review_adapter_ignores_target_weight_and_portfolio_narrative():
    investor = QuantInvestor(stock_pool=["000001.SZ"], verbose=False, enable_agent_layer=False)
    master_output = MasterAgentOutput(
        final_conviction="buy",
        final_score=0.36,
        confidence=0.84,
        top_picks=[
            SymbolRecommendation(
                symbol="000001.SZ",
                action="buy",
                conviction="strong_buy",
                rationale="structured rationale",
                target_weight=0.95,
            )
        ],
        conviction_drivers=["driver"],
        debate_resolution=["resolution"],
        portfolio_narrative="free text should not enter final allocation",
        risk_adjusted_exposure=0.75,
    )

    hints = investor._build_ic_hints_by_symbol(master_output)

    assert set(hints["000001.SZ"]) == {"score", "confidence", "action", "rationale_points"}
    assert hints["000001.SZ"]["action"] == "buy"
    assert "target_weight" not in hints["000001.SZ"]
    assert "portfolio_narrative" not in hints["000001.SZ"]


def test_control_chain_keeps_risk_veto_over_buy_hints():
    class HardVetoRiskGuard:
        def run(self, _payload):
            return RiskDecision(
                status=AgentStatus.VETOED,
                risk_level=RiskLevel.EXTREME,
                hard_veto=True,
                veto=True,
                action_cap=ActionLabel.AVOID,
                max_weight=0.0,
                gross_exposure_cap=0.0,
                target_exposure_cap=0.0,
                blocked_symbols=["000001.SZ"],
                position_limits={"000001.SZ": 0.0},
                reasons=["RiskGuard hard veto"],
            )

    orchestrator = AgentOrchestrator(risk_guard=HardVetoRiskGuard())
    result = orchestrator.run_with_structured_research(
        data_bundle=_make_data_bundle(["000001.SZ"]),
        macro_verdict=_make_macro_verdict(),
        research_by_symbol=_make_research_by_symbol(["000001.SZ"]),
        constraints={},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={},
        ic_hints_by_symbol={
            "000001.SZ": {
                "score": 0.9,
                "confidence": 0.95,
                "action": "buy",
                "rationale_points": ["aggressive master review"],
                "target_weight": 0.95,
                "portfolio_narrative": "should be ignored",
            }
        },
        persist_outputs=False,
    )

    assert result["risk_by_symbol"]["000001.SZ"].veto is True
    assert result["ic_by_symbol"]["000001.SZ"].metadata["llm_hint_applied"] is True
    assert result["portfolio_plan"].target_weights == {}
    assert result["report_bundle"].risk_decision.veto is True
    assert result["report_bundle"].portfolio_plan.target_weights == {}


def test_control_chain_is_deterministic_for_identical_inputs():
    orchestrator = AgentOrchestrator()
    data_bundle = _make_data_bundle(["000001.SZ", "600519.SH"])
    macro_verdict = _make_macro_verdict()
    research_by_symbol = _make_research_by_symbol(["000001.SZ", "600519.SH"])
    ic_hints = {
        symbol: {
            "score": 0.55 - index * 0.05,
            "confidence": 0.82,
            "action": "buy",
            "rationale_points": [f"{symbol} structured hint"],
        }
        for index, symbol in enumerate(["000001.SZ", "600519.SH"])
    }

    first = orchestrator.run_with_structured_research(
        data_bundle=data_bundle,
        macro_verdict=macro_verdict,
        research_by_symbol=research_by_symbol,
        constraints={},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={},
        ic_hints_by_symbol=ic_hints,
        persist_outputs=False,
    )
    second = orchestrator.run_with_structured_research(
        data_bundle=data_bundle,
        macro_verdict=macro_verdict,
        research_by_symbol=research_by_symbol,
        constraints={},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={},
        ic_hints_by_symbol=ic_hints,
        persist_outputs=False,
    )

    assert first["portfolio_plan"] == second["portfolio_plan"]
    assert first["portfolio_plan"].metadata["deterministic"] is True
    assert first["report_bundle"].portfolio_plan == second["report_bundle"].portfolio_plan


def test_precomputed_research_bridge_reuses_macro_branch_context_without_second_macro_run(monkeypatch):
    symbols = ["000001.SZ"]

    def _unexpected_macro_run(self, payload):
        raise AssertionError(f"unexpected macro rerun: {payload}")

    monkeypatch.setattr(agent_orchestrator_module.MacroAgent, "run", _unexpected_macro_run)

    orchestration = AgentOrchestrator().run_with_precomputed_research(
        data_bundle=_make_data_bundle(symbols),
        branch_results=_make_branch_results(symbols),
        constraints={},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={},
        persist_outputs=False,
    )

    assert orchestration["macro_verdict"] is not None
    assert orchestration["report_bundle"].macro_verdict is not None
