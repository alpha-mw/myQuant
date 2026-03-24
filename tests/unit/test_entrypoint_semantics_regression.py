"""
入口语义联合回归测试。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_investor
from quant_investor.agent_protocol import PortfolioPlan, ReportBundle
from quant_investor.enhanced_data_layer import EnhancedDataLayer


def _make_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=100)
    close = np.linspace(100, 110, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 1_000_000),
            "amount": close * 1_000_000,
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


def test_v8_v9_v10_entrypoints_remain_semantically_distinct():
    assert quant_investor.QuantInvestor is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorCurrent is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorLatest is quant_investor.QuantInvestorV10
    assert quant_investor.QuantInvestorV8 is not quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorV8 is not quant_investor.QuantInvestorV10
    assert quant_investor.QuantInvestorV8.__module__.endswith("quant_investor_v8")
    assert quant_investor.QuantInvestorV9.__module__.endswith("quant_investor_v9")
    assert quant_investor.QuantInvestorV10.__module__.endswith("quant_investor_v10")


def test_v9_and_v10_results_expose_unified_protocol_sidecars(monkeypatch):
    frame = _make_frame("000001.SZ")
    _patch_runtime(monkeypatch, frame)

    v9 = quant_investor.QuantInvestorV9(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()
    v10 = quant_investor.QuantInvestorV10(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
        enable_agent_layer=False,
    ).run()

    assert isinstance(v9.agent_report_bundle, ReportBundle)
    assert isinstance(v9.agent_portfolio_plan, PortfolioPlan)
    assert v9.agent_ic_decisions
    assert isinstance(v10.agent_report_bundle, ReportBundle)
    assert isinstance(v10.agent_portfolio_plan, PortfolioPlan)
    assert v10.agent_ic_decisions
    assert v9.agent_report_bundle.report_protocol_version == v9.report_protocol_version
    assert v10.agent_report_bundle.report_protocol_version == v10.report_protocol_version
