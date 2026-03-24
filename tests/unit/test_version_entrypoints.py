"""
版本入口与版本字段测试。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_investor
from quant_investor.agent_orchestrator import AgentOrchestrator
from quant_investor.agent_protocol import BranchVerdict
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.pipeline import QuantInvestor as PipelineQuantInvestor
from quant_investor.pipeline.current import CurrentPipelineResult, QuantInvestor as CurrentQuantInvestor
from quant_investor.versioning import (
    ARCHITECTURE_VERSION_CURRENT,
    BRANCH_SCHEMA_VERSION_V9,
    IC_PROTOCOL_VERSION,
    REPORT_PROTOCOL_VERSION,
)


def _make_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = np.linspace(100, 108, len(dates))
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


class _FakeMacroAgent:
    def run(self, payload):
        return BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观中性，维持中等暴露。",
            final_score=0.1,
            final_confidence=0.8,
            metadata={
                "regime": "balanced",
                "target_gross_exposure": 0.5,
                "style_bias": "balanced",
            },
        )


class _FakeResearchAgent:
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def run(self, payload):
        symbol = list(payload.get("stock_pool") or payload["data_bundle"].symbols)[0]
        return BranchVerdict(
            agent_name=self.agent_name,
            thesis=f"{self.agent_name} 对 {symbol} 保持正面判断。",
            symbol=symbol,
            final_score=0.35,
            final_confidence=0.7,
        )


def test_version_entrypoints_distinguish_v8_v9_and_current_alias():
    assert quant_investor.QuantInvestor is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorCurrent is quant_investor.QuantInvestorV9
    assert PipelineQuantInvestor is quant_investor.QuantInvestorV9
    assert CurrentQuantInvestor is quant_investor.QuantInvestorV9
    assert quant_investor.CurrentPipelineResult is CurrentPipelineResult
    assert quant_investor.QuantInvestorV8 is not quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorV8.__module__.endswith("quant_investor_v8")
    assert quant_investor.QuantInvestorV9.__module__.endswith("quant_investor_v9")


def test_quant_investor_default_entrypoint_returns_current_version_fields(monkeypatch):
    frame = _make_frame("000001.SZ")
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

    result = quant_investor.QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()

    assert isinstance(result, CurrentPipelineResult)
    assert result.architecture_version == ARCHITECTURE_VERSION_CURRENT
    assert result.branch_schema_version == BRANCH_SCHEMA_VERSION_V9
    assert result.ic_protocol_version == IC_PROTOCOL_VERSION
    assert result.report_protocol_version == REPORT_PROTOCOL_VERSION
    assert result.final_strategy.architecture_version == ARCHITECTURE_VERSION_CURRENT
    assert result.final_strategy.branch_schema_version == BRANCH_SCHEMA_VERSION_V9
    assert result.final_strategy.ic_protocol_version == IC_PROTOCOL_VERSION
    assert result.final_strategy.report_protocol_version == REPORT_PROTOCOL_VERSION


def test_agent_orchestrator_status_manifest_includes_protocol_versions(tmp_path):
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=["AAA"],
        symbol_data={"AAA": _make_frame("AAA")},
        fundamentals={"AAA": {"sector": "defensive"}},
    )
    output = AgentOrchestrator(
        macro_agent=_FakeMacroAgent(),
        kline_agent=_FakeResearchAgent("KlineAgent"),
        quant_agent=_FakeResearchAgent("QuantAgent"),
        fundamental_agent=_FakeResearchAgent("FundamentalAgent"),
        intelligence_agent=_FakeResearchAgent("IntelligenceAgent"),
    ).run(
        data_bundle=bundle,
        constraints={"gross_exposure_cap": 0.5, "max_weight": 0.3},
        existing_portfolio={"current_weights": {}},
        tradability_snapshot={"AAA": {"is_tradable": True, "sector": "defensive", "liquidity_score": 1.0}},
        persist_dir=tmp_path,
    )

    assert output["architecture_version"] == ARCHITECTURE_VERSION_CURRENT
    assert output["branch_schema_version"] == BRANCH_SCHEMA_VERSION_V9
    assert output["ic_protocol_version"] == IC_PROTOCOL_VERSION
    assert output["report_protocol_version"] == REPORT_PROTOCOL_VERSION

    manifest = json.loads(Path(output["persisted_paths"]["manifest"]).read_text(encoding="utf-8"))
    assert manifest["architecture_version"] == ARCHITECTURE_VERSION_CURRENT
    assert manifest["branch_schema_version"] == BRANCH_SCHEMA_VERSION_V9
    assert manifest["ic_protocol_version"] == IC_PROTOCOL_VERSION
    assert manifest["report_protocol_version"] == REPORT_PROTOCOL_VERSION
