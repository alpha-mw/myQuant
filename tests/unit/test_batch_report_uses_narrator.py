"""
full-market batch 报告统一复用 NarratorAgent 的测试。
"""

from __future__ import annotations

from pathlib import Path

import quant_investor.market.analyze as market_analyze
from quant_investor.agent_protocol import (
    ActionLabel,
    BranchVerdict,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
)


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


def test_batch_path_returns_report_bundle_and_uses_narrator_markdown(monkeypatch, tmp_path):
    captured = {}
    expected_bundle = ReportBundle(
        headline="批量报告头条",
        summary="批量报告摘要",
        macro_verdict=BranchVerdict(
            agent_name="MacroAgent",
            thesis="宏观中性。",
            final_score=0.1,
            final_confidence=0.7,
            metadata={"regime": "balanced", "target_gross_exposure": 0.4, "style_bias": "balanced"},
        ),
        ic_decisions=[
            ICDecision(
                thesis="600000.SH 保持买入。",
                action=ActionLabel.BUY,
                final_score=0.3,
                final_confidence=0.6,
                selected_symbols=["600000.SH"],
            )
        ],
        portfolio_plan=PortfolioPlan(
            target_exposure=0.12,
            target_gross_exposure=0.12,
            target_net_exposure=0.12,
            cash_ratio=0.88,
            target_positions={"600000.SH": 0.12},
        ),
        markdown_report="# 来自 NarratorAgent 的统一 markdown\n",
        executive_summary=["一", "二", "三"],
        market_view="市场观点",
        branch_conclusions={"kline": "K线结论"},
        stock_cards=[{"symbol": "600000.SH", "target_weight": 0.12, "display_action": "买入", "one_line_conclusion": "一句话"}],
        coverage_summary=["共整理 1 条覆盖说明，涉及 1/5 个分支。"],
        appendix_diagnostics=["共归档 1 条工程诊断，涉及 1/5 个分支。"],
    )

    def _fake_run(self, payload):
        captured.update(payload)
        return expected_bundle

    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(market_analyze.NarratorAgent, "run", _fake_run)

    output = market_analyze.generate_full_report(
        _make_all_results(),
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=1,
    )

    assert isinstance(output["report_bundle"], ReportBundle)
    assert output["report_bundle"] is expected_bundle
    assert set(captured) == {
        "macro_verdict",
        "branch_summaries",
        "ic_decisions",
        "portfolio_plan",
        "run_diagnostics",
    }
    assert Path(output["trade_report"]).read_text(encoding="utf-8") == expected_bundle.markdown_report


def test_batch_route_report_bundle_contains_required_sections(monkeypatch, tmp_path):
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
    assert bundle.branch_conclusions
    assert bundle.stock_cards
    assert bundle.coverage_summary
    assert len(bundle.executive_summary) == 3
    assert "Could not infer frequency" not in bundle.markdown_report
    assert "provider_missing" not in bundle.markdown_report
