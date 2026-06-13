"""Market full-report renderer boundary tests."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_full_report_helpers_and_sections_are_split_and_reexported():
    from quant_investor.market import full_report

    helpers = importlib.import_module("quant_investor.market.full_report_helpers")
    sections = importlib.import_module("quant_investor.market.full_report_sections")

    assert full_report._dedupe_text is helpers._dedupe_text
    assert full_report._sanitize_text is helpers._sanitize_text
    assert full_report._canonical_branch_map is helpers._canonical_branch_map
    assert full_report._derive_stock_conclusion is helpers._derive_stock_conclusion
    assert full_report._build_market_summary is helpers._build_market_summary
    assert full_report.ExecutiveSummaryBuilder is sections.ExecutiveSummaryBuilder
    assert full_report.ActionConsistencyGuard is sections.ActionConsistencyGuard
    assert full_report.DiagnosticsBucketizer is sections.DiagnosticsBucketizer
    assert full_report.ConclusionRenderer is sections.ConclusionRenderer
    assert full_report._aggregate_branch_summary is sections._aggregate_branch_summary


def test_full_report_renderer_writes_named_stock_report(monkeypatch, tmp_path):
    from quant_investor.market import full_report

    monkeypatch.setattr(
        full_report,
        "load_stock_names",
        lambda market="CN", refresh=False: {"600000.SH": "浦发银行"},
    )
    monkeypatch.setattr(
        full_report,
        "get_stock_name",
        lambda symbol, market="CN": "浦发银行",
    )

    output = full_report.generate_full_report(
        {
            "hs300": [
                {
                    "stock_count": 1,
                    "batch_id": 1,
                    "execution_log": [],
                    "branches": {
                        "quant": {
                            "score": 0.12,
                            "confidence": 0.62,
                            "conclusion": "量化结论偏正。",
                        }
                    },
                    "strategy": {
                        "target_exposure": 0.3,
                        "style_bias": "均衡",
                        "risk_summary": {"risk_level": "normal"},
                    },
                    "recommendations": [
                        {
                            "symbol": "600000.SH",
                            "company_name": "浦发银行",
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
                            "entry_price_range": {
                                "low": 9.8,
                                "high": 10.6,
                            },
                            "risk_flags": [],
                            "category_name": "沪深300 (大盘股)",
                            "macro_score": 0.0,
                        }
                    ],
                }
            ]
        },
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=1,
    )

    report_text = Path(output["trade_report"]).read_text(encoding="utf-8")

    assert "| 1 | 600000.SH | 浦发银行 |" in report_text
