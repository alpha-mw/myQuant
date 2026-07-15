"""
全市场批量分析报告测试
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

import quant_investor.market.analyze as cn_batch
import quant_investor.market.analyze as us_batch
from quant_investor.market.full_report import MarketArtifactContractError
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


def _make_branch(score: float, confidence: float, conclusion: str, *, debate_status: str = "skipped"):
    return {
        "score": score,
        "confidence": confidence,
        "conclusion": conclusion,
        "support_drivers": ["核心驱动项稳定。"],
        "drag_drivers": ["短期拖累仍需观察。"],
        "investment_risks": ["估值扩张后回撤风险仍在。"] if score > 0 else ["景气度偏弱。"],
        "coverage_notes": ["文档语义 28/30 标的已覆盖。"],
        "diagnostic_notes": ["Could not infer frequency"] if "legacy retired" in conclusion else [],
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
    branch_positive_count: int = 3,
    confidence: float = 0.56,
    macro_score: float = 0.02,
    stock_risk_flags: list[str] | None = None,
    debate_status: str = "skipped",
):
    return {
        "hs300": [
            {
                **CURRENT_MARKET_ENVELOPE,
                "stock_count": 30,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
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
                        "confidence": confidence,
                        "branch_positive_count": branch_positive_count,
                        "lot_size": 100,
                        "entry_price_range": {"low": 9.8, "high": 10.6},
                        "risk_flags": stock_risk_flags or ["波动率中等"],
                        "position_management": ["首次建仓 60%"],
                        "branch_scores": {
                            "quant": 0.10,
                            "fundamental": 0.08,
                            "macro": macro_score,
                        },
                        "category_name": "沪深300 (大盘股)",
                        "macro_score": macro_score,
                    }
                ],
                "analysis_meta": {
                    **CURRENT_MARKET_ENVELOPE,
                    "market": "CN",
                    "universe": "hs300",
                },
            }
        ]
    }


def _make_us_all_results():
    return {
        "large_cap": [
            {
                **CURRENT_MARKET_ENVELOPE,
                "stock_count": 25,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
                    "quant": _make_branch(0.09, 0.57, "量化结论偏正。"),
                    "fundamental": _make_branch(0.04, 0.52, "基本面结论中性偏正。"),
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
                        "branch_positive_count": 3,
                        "lot_size": 1,
                        "entry_price_range": {"low": 178.5, "high": 183.0},
                        "risk_flags": ["等待回踩"],
                        "position_management": ["目标价附近分批止盈"],
                        "branch_scores": {
                            "quant": 0.11,
                            "fundamental": 0.05,
                            "macro": 0.03,
                        },
                        "category_name": "大盘股 (S&P 500)",
                        "macro_score": 0.03,
                    }
                ],
                "analysis_meta": {
                    **CURRENT_MARKET_ENVELOPE,
                    "market": "US",
                    "universe": "large_cap",
                },
            }
        ]
    }


def _read_report(output: dict[str, str]) -> str:
    return Path(output["trade_report"]).read_text(encoding="utf-8")


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

    report_text = _read_report(output)

    assert "| 1 | 600000.SH | 浦发银行 |" in report_text
    assert "¥10.20 | ¥10.00 |" in report_text
    assert "最大亏损: -8.0%" in report_text


def test_us_report_uses_only_current_three_branch_average(monkeypatch, tmp_path):
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

    report_text = _read_report(output)

    assert "kline: +0.120" not in report_text
    assert "K线分支" not in report_text


@pytest.mark.parametrize(
    ("field_name", "stale_value"),
    [
        ("architecture_version", None),
        ("branch_schema_version", "branch-schema.v13.four-branch"),
        ("likelihood_schema_version", "likelihood-schema.v13.three-likelihood"),
        ("ic_protocol_version", "ic-protocol.v13.four-branch"),
        ("report_protocol_version", "report-protocol.v13.four-branch"),
    ],
)
def test_full_report_rejects_unversioned_or_stale_batch_before_writing(
    tmp_path,
    field_name,
    stale_value,
):
    all_results = deepcopy(_make_cn_all_results())
    batch = all_results["hs300"][0]
    if stale_value is None:
        batch.pop(field_name)
    else:
        batch[field_name] = stale_value
    output_dir = tmp_path / "rejected"

    with pytest.raises(MarketArtifactContractError, match=field_name):
        cn_batch.generate_full_report(
            all_results,
            market="CN",
            output_dir=str(output_dir),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("invalid_branch", "error_pattern"),
    [
        ("intelligence", "retired Intelligence key"),
        ("kline", "unsupported branches"),
    ],
)
def test_full_report_rejects_noncanonical_branch_before_writing(
    tmp_path,
    invalid_branch,
    error_pattern,
):
    all_results = deepcopy(_make_cn_all_results())
    all_results["hs300"][0]["branches"][invalid_branch] = _make_branch(
        0.1,
        0.6,
        "retired branch",
    )
    output_dir = tmp_path / "rejected"

    with pytest.raises(MarketArtifactContractError, match=error_pattern):
        cn_batch.generate_full_report(
            all_results,
            market="CN",
            output_dir=str(output_dir),
        )

    assert not output_dir.exists()


def test_full_report_rejects_missing_canonical_branch_before_writing(tmp_path):
    all_results = deepcopy(_make_cn_all_results())
    all_results["hs300"][0]["branches"].pop("macro")
    output_dir = tmp_path / "rejected"

    with pytest.raises(MarketArtifactContractError, match="missing branches: macro"):
        cn_batch.generate_full_report(
            all_results,
            market="CN",
            output_dir=str(output_dir),
        )

    assert not output_dir.exists()


def test_full_report_rejects_nested_retired_key_before_writing(tmp_path):
    all_results = deepcopy(_make_cn_all_results())
    all_results["hs300"][0]["analysis_meta"]["intelligence_snapshot"] = {
        "score": 1.0,
    }
    output_dir = tmp_path / "rejected"

    with pytest.raises(
        MarketArtifactContractError,
        match="analysis_meta.intelligence_snapshot",
    ):
        cn_batch.generate_full_report(
            all_results,
            market="CN",
            output_dir=str(output_dir),
        )

    assert not output_dir.exists()


def test_full_report_allows_retired_name_in_prose_values(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cn_batch,
        "load_stock_names",
        lambda market="CN", refresh=False: {},
    )
    monkeypatch.setattr(
        cn_batch,
        "get_stock_name",
        lambda symbol, market="CN": symbol,
    )
    all_results = deepcopy(_make_cn_all_results())
    all_results["hs300"][0]["analysis_meta"]["migration_note"] = (
        "Intelligence was retired from the alpha architecture."
    )

    output = cn_batch.generate_full_report(
        all_results,
        market="CN",
        output_dir=str(tmp_path / "accepted"),
    )

    assert Path(output["trade_report"]).exists()


@pytest.mark.parametrize(
    ("market", "legacy_root", "all_results"),
    [
        ("CN", "results/cn_analysis_full", _make_cn_all_results()),
        ("US", "results/us_analysis_full", _make_us_all_results()),
    ],
)
def test_full_report_rejects_retired_unversioned_output_roots(
    monkeypatch,
    tmp_path,
    market,
    legacy_root,
    all_results,
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="read-only"):
        cn_batch.generate_full_report(
            deepcopy(all_results),
            market=market,
            output_dir=legacy_root,
        )

    assert not (tmp_path / legacy_root).exists()


def test_candidate_index_validates_artifact_and_rejects_legacy_root(
    monkeypatch,
    tmp_path,
):
    from quant_investor.market.full_report import save_candidate_index

    unversioned = deepcopy(_make_cn_all_results())
    unversioned["hs300"][0].pop("architecture_version")
    custom_dir = tmp_path / "custom"
    with pytest.raises(MarketArtifactContractError, match="architecture_version"):
        save_candidate_index(unversioned, market="CN", output_dir=str(custom_dir))
    assert not custom_dir.exists()

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        save_candidate_index(
            _make_cn_all_results(),
            market="CN",
            output_dir="results/cn_analysis_full",
        )
    assert not (tmp_path / "results" / "cn_analysis_full").exists()


def test_report_top_contains_three_line_executive_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert "## 三句话执行摘要" in report_text


def test_each_branch_section_has_non_empty_conclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    for label in ["量化", "基本面", "宏观"]:
        match = re.search(rf"### {label}分支\n- 平均得分: .*?\n- 结论: (.+)\n", report_text)
        assert match is not None
        assert match.group(1).strip()


def test_each_recommended_stock_has_non_empty_one_line_conclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    match = re.search(r"- 一句话结论: (.+)\n", report_text)
    assert match is not None
    assert match.group(1).strip()


def test_provider_missing_tokens_do_not_enter_stock_risk_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

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
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(_make_cn_all_results(), market="CN", output_dir=str(tmp_path))
    report_text = _read_report(output)

    assert re.search(r"\d+/\d+ 批次", report_text)
    assert re.search(r"\d+/\d+ 标的", report_text)


def test_action_guard_blocks_buy_when_support_is_low(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(
        _make_cn_all_results(branch_positive_count=2, confidence=0.30, macro_score=-0.35),
        market="CN",
        output_dir=str(tmp_path),
    )
    trade_data = json.loads(Path(output["trade_data"]).read_text(encoding="utf-8"))
    recommendation = trade_data["recommendations"][0]

    assert recommendation["action"] in {"观察", "轻仓试错"}
    assert recommendation["action"] != "buy"

    report_text = _read_report(output)
    assert "执行动作: 买入" not in report_text


def test_debate_status_unknown_does_not_render(monkeypatch, tmp_path):
    monkeypatch.setattr(cn_batch, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(cn_batch, "get_stock_name", lambda symbol, market="CN": symbol)

    output = cn_batch.generate_full_report(
        _make_cn_all_results(debate_status="unknown"),
        market="CN",
        output_dir=str(tmp_path),
    )
    report_text = _read_report(output)

    assert "unknown" not in report_text
