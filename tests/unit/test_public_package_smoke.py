"""
公共入口与构建面烟测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import quant_investor
import quant_investor.cli.main as cli_main
from quant_investor.enhanced_data_layer import EnhancedDataLayer


def test_public_package_exports():
    assert hasattr(quant_investor, "QuantInvestor")
    assert hasattr(quant_investor, "QuantInvestorCurrent")
    assert hasattr(quant_investor, "QuantInvestorV8")
    assert hasattr(quant_investor, "QuantInvestorV9")
    assert hasattr(quant_investor, "QuantInvestorV10")
    assert hasattr(quant_investor, "QuantInvestorLatest")
    assert hasattr(quant_investor, "BranchResult")
    assert quant_investor.QuantInvestor is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorCurrent is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorLatest is quant_investor.QuantInvestorV10
    assert quant_investor.QuantInvestorV8 is not quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorV8 is not quant_investor.QuantInvestorV10


def test_cli_market_download_dispatches(monkeypatch):
    captured = {}

    def _fake_run_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_download", _fake_run_download)
    cli_main.main(["market", "download", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_cli_market_analyze_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_analysis(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_analysis", _fake_run_market_analysis)
    cli_main.main(["market", "analyze", "--market", "US", "--mode", "sample"])

    assert captured["market"] == "US"
    assert captured["mode"] == "sample"


def test_cli_market_backtest_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_backtest(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_backtest", _fake_run_market_backtest)
    cli_main.main(["market", "backtest", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_pyproject_only_packages_quant_investor():
    pyproject_text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'packages = ["quant_investor"]' in pyproject_text
    assert 'quant-investor = "quant_investor.cli.main:main"' in pyproject_text


def test_cli_research_accepts_architecture_flag(monkeypatch):
    captured = {}

    class _FakeLatest:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

        def print_report(self):
            return None

    monkeypatch.setattr(cli_main, "QuantInvestorLatest", _FakeLatest)
    cli_main.main(["research", "run", "--stocks", "000001.SZ", "--architecture", "latest"])

    assert captured["stock_pool"] == ["000001.SZ"]
    assert captured["enable_agent_layer"] is True
    assert captured["enable_branch_debate"] is True


def test_v9_one_symbol_mock_run_includes_version_fields(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = np.linspace(100, 110, len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(dates), 1_000_000),
            "amount": close * 1_000_000,
            "symbol": "000001.SZ",
            "market": "CN",
            "forward_ret_5d": pd.Series(close).shift(-5) / pd.Series(close) - 1,
        }
    )

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
                "generate_risk_report": lambda self: type(
                    "_Report",
                    (),
                    {
                        "overall_signal": "🟢",
                        "overall_risk_level": "低风险",
                        "recommendation": "积极布局",
                    },
                )(),
            },
        )(),
    )

    result = quant_investor.QuantInvestorV9(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()

    assert result.architecture_version == "9.0.0-current"
    assert result.branch_schema_version == "v9-fundamental-first-class"
    assert result.ic_protocol_version
    assert result.report_protocol_version
    assert result.calibration_schema_version
    assert result.debate_template_version
    assert result.final_strategy.architecture_version == result.architecture_version
    assert result.final_strategy.ic_protocol_version == result.ic_protocol_version
    assert result.final_strategy.report_protocol_version == result.report_protocol_version
