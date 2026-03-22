"""
全市场批量分析报告测试
"""

from __future__ import annotations

from pathlib import Path

import quant_investor.market.analyze as cn_batch
import quant_investor.market.analyze as us_batch


def _make_cn_all_results():
    return {
        "hs300": [
            {
                "stock_count": 30,
                "branches": {
                    "kline": {"score": 0.12},
                    "quant": {"score": 0.08},
                    "llm_debate": {"score": 0.05},
                    "intelligence": {"score": 0.07},
                    "macro": {"score": 0.02},
                },
                "strategy": {
                    "target_exposure": 0.46,
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
                        "recommended_entry_price": 10.0,
                        "current_price": 10.2,
                        "target_price": 11.4,
                        "stop_loss_price": 9.2,
                        "expected_upside": 0.14,
                        "model_expected_return": 0.11,
                        "consensus_score": 0.32,
                        "confidence": 0.56,
                        "branch_positive_count": 4,
                        "lot_size": 100,
                        "entry_price_range": {"low": 9.8, "high": 10.6},
                        "risk_flags": ["波动率中等"],
                        "position_management": ["首次建仓 60%"],
                    }
                ],
            }
        ]
    }


def _make_us_all_results():
    return {
        "large_cap": [
            {
                "stock_count": 25,
                "branches": {
                    "kline": {"score": 0.12},
                    "quant": {"score": 0.09},
                    "llm_debate": {"score": 0.04},
                    "intelligence": {"score": 0.08},
                    "macro": {"score": 0.03},
                },
                "strategy": {
                    "target_exposure": 0.44,
                    "style_bias": "成长",
                    "candidate_symbols": ["AAPL"],
                    "risk_summary": {"risk_level": "normal"},
                },
                "recommendations": [
                    {
                        "symbol": "AAPL",
                        "action": "buy",
                        "data_source_status": "real",
                        "suggested_weight": 0.10,
                        "recommended_entry_price": 180.0,
                        "current_price": 181.5,
                        "target_price": 198.0,
                        "stop_loss_price": 167.4,
                        "expected_upside": 0.10,
                        "model_expected_return": 0.08,
                        "consensus_score": 0.27,
                        "confidence": 0.58,
                        "branch_positive_count": 4,
                        "lot_size": 1,
                        "entry_price_range": {"low": 178.5, "high": 183.0},
                        "risk_flags": ["等待回踩"],
                        "position_management": ["目标价附近分批止盈"],
                    }
                ],
            }
        ]
    }


def test_cn_report_uses_recommended_entry_price(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cn_batch,
        "load_stock_names",
        lambda market="CN", refresh=False: {"600000.SH": "浦发银行"},
    )
    monkeypatch.setattr(
        cn_batch,
        "get_stock_name",
        lambda symbol, market="CN": "浦发银行",
    )

    output = cn_batch.generate_full_report(
        _make_cn_all_results(),
        market="CN",
        output_dir=str(tmp_path / "cn_reports"),
        total_capital=1_000_000,
        top_k=1,
    )

    report_text = Path(output["trade_report"]).read_text(encoding="utf-8")

    assert "| 1 | 600000.SH | 浦发银行 | 沪深300 (大盘股) | ¥10.20 | ¥10.00 |" in report_text
    assert "最大亏损: -8.0%" in report_text


def test_us_report_includes_kline_branch_average(monkeypatch, tmp_path):
    monkeypatch.setattr(
        us_batch,
        "load_stock_names",
        lambda market="US", refresh=False: {"AAPL": "Apple Inc."},
    )
    monkeypatch.setattr(
        us_batch,
        "get_stock_name",
        lambda symbol, market="US": "Apple Inc.",
    )

    output = us_batch.generate_full_report(
        _make_us_all_results(),
        market="US",
        output_dir=str(tmp_path / "us_reports"),
        total_capital=1_000_000,
        top_k=1,
    )

    report_text = Path(output["trade_report"]).read_text(encoding="utf-8")

    assert "kline: +0.120" in report_text
