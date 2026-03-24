"""
market / pipeline / reporting 联合集成回归。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_investor.market.analyze as market_analyze
from quant_investor import QuantInvestorV9
from quant_investor.agent_orchestrator import AgentOrchestrator
from quant_investor.agent_protocol import ActionLabel, BranchVerdict, PortfolioPlan, ReportBundle
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.enhanced_data_layer import EnhancedDataLayer


def _make_frame(symbol: str, scale: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = np.linspace(100, 100 * (1 + 0.08 * scale), len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 1_500_000),
            "amount": close * 1_500_000,
            "symbol": symbol,
            "market": "CN",
            "forward_ret_5d": pd.Series(close).shift(-5) / pd.Series(close) - 1,
        }
    )


def _patch_runtime(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(
        EnhancedDataLayer,
        "fetch_and_process",
        lambda self, symbol, start_date, end_date, label_periods=5: frame.copy(),
    )
    monkeypatch.setattr(
        "quant_investor.pipeline.parallel_research_pipeline.create_terminal",
        lambda market: type(
            "_FakeTerminal",
            (),
            {
                "generate_risk_report": lambda self: SimpleNamespace(
                    overall_signal="🟢",
                    overall_risk_level="低风险",
                    recommendation="积极布局",
                ),
            },
        )(),
    )


def _make_market_results():
    return {
        "hs300": [
            {
                "stock_count": 20,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished", "Traceback: hidden stack"],
                "branches": {
                    "kline": {
                        "score": 0.12,
                        "confidence": 0.62,
                        "conclusion": "K线结论偏正。",
                        "support_drivers": ["趋势稳定。"],
                        "drag_drivers": ["波动仍需观察。"],
                        "investment_risks": ["估值扩张后回撤风险仍在。"],
                        "coverage_notes": ["K线数据 20/20 标的已覆盖。"],
                        "diagnostic_notes": ["Could not infer frequency"],
                    },
                    "quant": {
                        "score": 0.09,
                        "confidence": 0.58,
                        "conclusion": "量化结论偏正。",
                        "support_drivers": ["因子信号稳定。"],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                    },
                    "fundamental": {
                        "score": 0.06,
                        "confidence": 0.54,
                        "conclusion": "基本面结论偏正。",
                        "support_drivers": ["盈利质量稳定。"],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": ["文档语义 12/20 标的已覆盖。"],
                        "diagnostic_notes": [],
                    },
                    "intelligence": {
                        "score": 0.07,
                        "confidence": 0.57,
                        "conclusion": "智能融合结论偏正。",
                        "support_drivers": ["事件面中性偏正。"],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                    },
                    "macro": {
                        "score": 0.03,
                        "confidence": 0.52,
                        "conclusion": "宏观结论中性偏稳。",
                        "support_drivers": ["流动性中性。"],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                    },
                },
                "strategy": {
                    "target_exposure": 0.44,
                    "style_bias": "均衡",
                    "candidate_symbols": ["600000.SH"],
                    "risk_summary": {"risk_level": "normal"},
                },
                "recommendations": [
                    {
                        "symbol": "600000.SH",
                        "action": "buy",
                        "data_source_status": "real",
                        "suggested_weight": 0.12,
                        "current_price": 10.2,
                        "recommended_entry_price": 10.0,
                        "target_price": 11.4,
                        "stop_loss_price": 9.2,
                        "expected_upside": 0.14,
                        "model_expected_return": 0.11,
                        "consensus_score": 0.32,
                        "confidence": 0.58,
                        "branch_positive_count": 4,
                        "branch_scores": {
                            "kline": 0.20,
                            "quant": 0.10,
                            "fundamental": 0.08,
                            "intelligence": 0.06,
                            "macro": 0.03,
                        },
                        "risk_flags": ["等待回踩"],
                    }
                ],
            }
        ]
    }


class _CountingMacroAgent:
    def run(self, payload):
        return BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观中性偏稳。",
            final_score=0.1,
            final_confidence=0.8,
            metadata={
                "regime": "balanced",
                "target_gross_exposure": 0.6,
                "style_bias": "balanced_quality",
            },
        )


class _FakeResearchAgent:
    def __init__(self, agent_name: str, score_map: dict[str, float], risk_symbol: str | None = None) -> None:
        self.agent_name = agent_name
        self.score_map = score_map
        self.risk_symbol = risk_symbol

    def run(self, payload):
        symbol = list(payload.get("stock_pool") or payload["data_bundle"].symbols)[0]
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=f"{self.agent_name} 对 {symbol} 给出结构化判断。",
            symbol=symbol,
            final_score=self.score_map[symbol],
            final_confidence=0.72,
            investment_risks=["fraud investigation ongoing"] if symbol == self.risk_symbol else [],
        )


def test_full_market_and_single_symbol_paths_share_protocol_objects(monkeypatch, tmp_path):
    frame = _make_frame("000001.SZ")
    _patch_runtime(monkeypatch, frame)
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})

    single_symbol = QuantInvestorV9(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()
    full_market = market_analyze.generate_full_report(
        _make_market_results(),
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=1,
    )

    assert isinstance(single_symbol.agent_report_bundle, ReportBundle)
    assert isinstance(single_symbol.agent_portfolio_plan, PortfolioPlan)
    assert single_symbol.agent_ic_decisions
    assert isinstance(full_market["report_bundle"], ReportBundle)
    assert full_market["report_bundle"].stock_cards
    assert full_market["report_bundle"].branch_conclusions
    assert full_market["report_bundle"].coverage_summary
    assert single_symbol.agent_report_bundle.portfolio_plan.target_positions == single_symbol.agent_portfolio_plan.target_positions
    first_symbol = next(iter(single_symbol.agent_ic_decisions))
    assert single_symbol.agent_report_bundle.ic_decisions[0].action == single_symbol.agent_ic_decisions[first_symbol].action
    assert "## 三句话执行摘要" in full_market["report_bundle"].markdown_report
    assert "Traceback" not in full_market["report_bundle"].markdown_report


def test_risk_guard_veto_still_cannot_be_bypassed_in_integrated_report():
    symbols = ["AAA", "BBB"]
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=symbols,
        symbol_data={
            "AAA": _make_frame("AAA", scale=1.0),
            "BBB": _make_frame("BBB", scale=0.8),
        },
        fundamentals={
            "AAA": {"sector": "defensive"},
            "BBB": {"sector": "growth"},
        },
    )

    output = AgentOrchestrator(
        macro_agent=_CountingMacroAgent(),
        kline_agent=_FakeResearchAgent("KlineAgent", {"AAA": 0.45, "BBB": 0.50}),
        quant_agent=_FakeResearchAgent("QuantAgent", {"AAA": 0.40, "BBB": 0.55}, risk_symbol="BBB"),
        fundamental_agent=_FakeResearchAgent("FundamentalAgent", {"AAA": 0.35, "BBB": 0.48}),
        intelligence_agent=_FakeResearchAgent("IntelligenceAgent", {"AAA": 0.30, "BBB": 0.42}),
    ).run(
        data_bundle=bundle,
        constraints={
            "gross_exposure_cap": 0.7,
            "max_weight": 0.35,
            "sector_caps": {"defensive": 0.5, "growth": 0.5},
            "veto_keywords": ["fraud"],
        },
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={
            "AAA": {"is_tradable": True, "sector": "defensive", "liquidity_score": 1.0},
            "BBB": {"is_tradable": True, "sector": "growth", "liquidity_score": 1.0},
        },
        persist_outputs=False,
    )

    assert output["ic_by_symbol"]["BBB"].status.name == "VETOED"
    assert output["ic_by_symbol"]["BBB"].action is ActionLabel.HOLD
    assert "BBB" not in output["portfolio_plan"].target_positions
    assert all(card["symbol"] != "BBB" for card in output["report_bundle"].stock_cards)
