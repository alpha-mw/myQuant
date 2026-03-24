"""
全市场批量分析报告测试。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import quant_investor.market.analyze as cn_batch
import quant_investor.market.analyze as us_batch


def _make_branch(score: float, confidence: float, conclusion: str, *, debate_status: str = "skipped"):
    return {
        "score": score,
        "confidence": confidence,
        "conclusion": conclusion,
        "support_drivers": ["核心驱动项稳定。"],
        "drag_drivers": ["短期拖累仍需观察。"],
        "investment_risks": ["估值扩张后回撤风险仍在。"] if score > 0 else ["景气度偏弱。"],
        "coverage_notes": ["文档语义 28/30 标的已覆盖。"],
        "diagnostic_notes": ["Could not infer frequency"] if "K线" in conclusion else [],
        "module_coverage": {
            "core": {
                "label": "核心模块",
                "available_symbols": 28,
                "total_symbols": 30,
                "status": "active",
            }
        },
        "debate_status": debate_status,
    }


def _make_cn_all_results(
    *,
    branch_positive_count: int = 4,
    confidence: float = 0.56,
    macro_score: float = 0.02,
    stock_risk_flags: list[str] | None = None,
    debate_status: str = "skipped",
):
    return {
        "hs300": [
            {
                "stock_count": 30,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
                    "kline": _make_branch(0.12, 0.62, "K线结论偏正。", debate_status=debate_status),
                    "quant": _make_branch(0.08, 0.58, "量化结论偏正。", debate_status=debate_status),
                    "fundamental": {
                        **_make_branch(0.05, 0.54, "基本面结论偏正。", debate_status=debate_status),
                        "coverage_notes": [
                            "盈利预测 20/30 标的已覆盖。",
                            "文档语义 18/30 标的已覆盖。",
                        ],
                        "module_coverage": {
                            "forecast": {
                                "label": "盈利预测",
                                "available_symbols": 20,
                                "total_symbols": 30,
                                "status": "active",
                            },
                            "documents": {
                                "label": "文档语义",
                                "available_symbols": 18,
                                "total_symbols": 30,
                                "status": "active",
                            },
                        },
                    },
                    "intelligence": _make_branch(0.07, 0.57, "智能融合结论偏正。", debate_status=debate_status),
                    "macro": _make_branch(macro_score, 0.51, "宏观结论中性偏稳。", debate_status=debate_status),
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
                        "confidence": confidence,
                        "branch_positive_count": branch_positive_count,
                        "lot_size": 100,
                        "entry_price_range": {"low": 9.8, "high": 10.6},
                        "risk_flags": stock_risk_flags or ["波动率中等"],
                        "position_management": ["首次建仓 60%"],
                        "branch_scores": {
                            "kline": 0.20,
                            "quant": 0.10,
                            "fundamental": 0.08,
                            "intelligence": 0.06,
                            "macro": macro_score,
                        },
                        "category_name": "沪深300 (大盘股)",
                        "macro_score": macro_score,
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
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
                    "kline": _make_branch(0.12, 0.62, "K线结论偏正。"),
                    "quant": _make_branch(0.09, 0.57, "量化结论偏正。"),
                    "fundamental": _make_branch(0.04, 0.52, "基本面结论中性偏正。"),
                    "intelligence": _make_branch(0.08, 0.58, "智能融合结论偏正。"),
                    "macro": _make_branch(0.03, 0.50, "宏观结论中性。"),
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
                        "branch_scores": {
                            "kline": 0.22,
                            "quant": 0.11,
                            "fundamental": 0.05,
                            "intelligence": 0.09,
                            "macro": 0.03,
                        },
                        "category_name": "大盘股 (S&P 500)",
                        "macro_score": 0.03,
                    }
                ],
            }
        ]
    }


def _read_report(output: dict[str, str]) -> str:
    return Path(output["trade_report"]).read_text(encoding="utf-8")


def test_report_top_contains_three_line_executive_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert "## 三句话执行摘要" in report_text
    assert report_text.count("- ") >= 3


def test_each_branch_section_has_non_empty_conclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    for label in ["K线", "量化", "基本面", "智能融合", "宏观"]:
        match = re.search(rf"### {label}分支\n- 结论: (.+)\n", report_text)
        assert match is not None
        assert match.group(1).strip()


def test_each_recommended_stock_has_non_empty_one_line_conclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    match = re.search(r"- 一句话结论: (.+)\n", report_text)
    assert match is not None
    assert match.group(1).strip()


def test_main_report_hides_raw_frequency_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert "Could not infer frequency" not in report_text
    assert "部分批次 K 线深度模型未完成频率对齐，已自动回退统计预测。" in report_text


def test_provider_missing_tokens_do_not_enter_main_body(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(
        _make_cn_all_results(stock_risk_flags=["provider_missing", "snapshot_missing"]),
        market="CN",
        output_dir=str(tmp_path),
    )
    report_text = _read_report(output)

    assert "provider_missing" not in report_text
    assert "snapshot_missing" not in report_text


def test_counts_include_units(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert "条覆盖说明" in report_text
    assert "条工程诊断" in report_text
    assert re.search(r"\d+/\d+ 个分支", report_text)


def test_debate_status_unknown_does_not_render(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(
        _make_cn_all_results(debate_status="unknown"),
        market="CN",
        output_dir=str(tmp_path),
    )
    report_text = _read_report(output)

    assert "unknown" not in report_text


def test_report_does_not_use_bare_parenthesized_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert re.search(r"\(\d+\)", report_text) is None


def test_analyze_batch_uses_current_stable_entrypoint_and_fast_screen(monkeypatch):
    captured = {}

    class _FakeCurrent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                final_strategy=SimpleNamespace(
                    trade_recommendations=[],
                    target_exposure=0.35,
                    style_bias="均衡",
                    candidate_symbols=[],
                    position_limits={},
                    branch_consensus={},
                    risk_summary={},
                    execution_notes=[],
                    research_mode="production",
                ),
                branch_results={},
                execution_log=[],
            )

    monkeypatch.setattr(cn_batch, "QuantInvestor", _FakeCurrent)

    result = cn_batch.analyze_batch(
        symbols=["600000.SH", "600519.SH"],
        category="hs300",
        batch_id=1,
        market="CN",
        verbose=False,
    )

    assert result is not None
    assert captured["stock_pool"] == ["600000.SH", "600519.SH"]
    assert "enable_agent_layer" not in captured
    assert captured["enable_branch_debate"] is False
    assert captured["enable_document_semantics"] is False
    assert captured["kline_backend"] == "heuristic"


def test_full_market_report_hides_raw_traceback_and_info_log(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})

    payload = _make_cn_all_results()
    payload["hs300"][0]["execution_log"] = ["[INFO] batch finished", "Traceback: hidden stack"]
    payload["hs300"][0]["branches"]["fundamental"]["diagnostic_notes"] = [
        "ValueError: hidden stack should not surface",
    ]

    output = cn_batch.generate_full_report(payload, market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert "[INFO]" not in report_text
    assert "Traceback" not in report_text
    assert "ValueError:" not in report_text
    assert "运行日志摘要" in report_text or "工程异常" in report_text


def test_trade_data_embeds_protocol_report_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(us_batch, "load_stock_names", lambda market="US", refresh=False: {})

    output = us_batch.generate_full_report(
        _make_us_all_results(),
        market="US",
        output_dir=str(tmp_path / "us_reports"),
        total_capital=1_000_000,
        top_k=1,
    )
    trade_data = json.loads(Path(output["trade_data"]).read_text(encoding="utf-8"))

    assert "report_bundle" in trade_data
    assert trade_data["report_bundle"]["markdown_report"].startswith("# 投资研究执行报告")
