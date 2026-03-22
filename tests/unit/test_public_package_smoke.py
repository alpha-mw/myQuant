"""
公共入口与构建面烟测。
"""

from __future__ import annotations

from pathlib import Path

import quant_investor
import quant_investor.cli.main as cli_main


def test_public_package_exports():
    assert hasattr(quant_investor, "QuantInvestorV8")
    assert hasattr(quant_investor, "BranchResult")


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
