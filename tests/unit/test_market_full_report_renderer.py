"""Market full-report renderer boundary tests."""

from __future__ import annotations

from pathlib import Path

from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)


CURRENT_MARKET_ENVELOPE = {
    "architecture_version": ARCHITECTURE_VERSION,
    "branch_schema_version": BRANCH_SCHEMA_VERSION,
    "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
    "ic_protocol_version": IC_PROTOCOL_VERSION,
    "report_protocol_version": REPORT_PROTOCOL_VERSION,
}


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
                    **CURRENT_MARKET_ENVELOPE,
                    "stock_count": 1,
                    "batch_id": 1,
                    "execution_log": [],
                    "branches": {
                        "quant": {
                            "score": 0.12,
                            "confidence": 0.62,
                            "conclusion": "量化结论偏正。",
                        },
                        "fundamental": {
                            "score": 0.08,
                            "confidence": 0.58,
                            "conclusion": "基本面结论偏正。",
                        },
                        "macro": {
                            "score": 0.0,
                            "confidence": 0.5,
                            "conclusion": "宏观结论中性。",
                        },
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
                            "branch_positive_count": 3,
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
                    "analysis_meta": {
                        **CURRENT_MARKET_ENVELOPE,
                        "market": "CN",
                        "universe": "hs300",
                    },
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
