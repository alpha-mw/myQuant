"""
设计意图 6：full-market 报告必须统一复用 NarratorAgent -> ReportBundle。

当前预期：
- quant_investor.market 主线通过
- web full-market 入口失败，暴露报告栈未收口 blocker
"""

from __future__ import annotations

from pathlib import Path

import quant_investor.market.analyze as market_analyze
from quant_investor.agent_protocol import ReportBundle
from web.services import analysis_service
from web.tasks import run_analysis_job


def _make_all_results():
    return {
        "hs300": [
            {
                "stock_count": 12,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
                    "kline": {
                        "score": 0.12,
                        "confidence": 0.60,
                        "conclusion": "K线结论偏正。",
                        "support_drivers": ["趋势稳定。"],
                        "drag_drivers": ["短线波动仍在。"],
                        "investment_risks": ["波动回撤风险仍在。"],
                        "coverage_notes": ["K线数据 12/12 标的已覆盖。"],
                        "diagnostic_notes": ["Could not infer frequency"],
                    },
                    "quant": {
                        "score": 0.10,
                        "confidence": 0.58,
                        "conclusion": "量化结论偏正。",
                        "support_drivers": ["因子动量稳定。"],
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
                        "coverage_notes": ["文档语义 8/12 标的已覆盖。"],
                        "diagnostic_notes": [],
                    },
                    "intelligence": {
                        "score": 0.07,
                        "confidence": 0.56,
                        "conclusion": "智能融合结论偏正。",
                        "support_drivers": ["事件面中性偏正。"],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                    },
                    "macro": {
                        "score": 0.02,
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
                    "target_exposure": 0.42,
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
                            "macro": 0.02,
                        },
                        "risk_flags": ["provider_missing", "等待回踩"],
                    }
                ],
            }
        ]
    }


def test_quant_investor_full_market_path_returns_report_bundle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})

    output = market_analyze.generate_full_report(
        _make_all_results(),
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=1,
    )

    bundle = output["report_bundle"]
    assert isinstance(bundle, ReportBundle)
    assert Path(output["trade_report"]).read_text(encoding="utf-8") == bundle.markdown_report


def test_web_full_market_surface_should_expose_report_bundle_protocol(monkeypatch) -> None:
    def _fake_run_job(payload):
        return {
            "target_exposure": 0.35,
            "style_bias": "均衡",
            "trade_recommendations": [
                {
                    "symbol": "000001.SZ",
                    "current_price": 10.0,
                    "recommended_entry_price": 9.8,
                    "target_price": 11.2,
                    "stop_loss_price": 9.2,
                    "suggested_weight": 0.12,
                }
            ],
            "risk": {
                "volatility": 0.15,
                "max_drawdown": 0.08,
                "sharpe_ratio": 1.1,
                "warnings": [],
            },
            "branches": [
                {
                    "branch_name": name,
                    "score": 0.10 if name != "macro" else 0.02,
                    "confidence": 0.60,
                    "top_symbols": ["000001.SZ"],
                    "risks": [],
                }
                for name in analysis_service.BRANCH_ORDER
            ],
        }

    monkeypatch.setattr(run_analysis_job, "run_job", _fake_run_job)

    result = analysis_service._run_market_analysis(
        {
            "market": "CN",
            "targets": ["000001.SZ"],
            "risk": {"capital": 1_000_000.0, "max_single_position": 0.2, "risk_level": "中等"},
            "portfolio": {"candidate_limit": 12},
            "branches": {
                name: {"enabled": True, "settings": {"backend": "heuristic"}}
                for name in analysis_service.BRANCH_ORDER
            },
            "llm_debate": {"enabled": False, "models": [], "assignments": []},
        }
    )

    assert "report_bundle" in result, (
        "web full-market 入口仍只返回 report_markdown，自建 _build_market_report，"
        "未统一复用 NarratorAgent -> ReportBundle；"
        "这是 web 入口/报告栈未收口，不是 quant_investor.market 主线缺少实现。"
    )

