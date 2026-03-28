from __future__ import annotations

import quant_investor.versioning as versioning

from tests.integration._helpers import run_stubbed_full_market_path, run_stubbed_quant_path


def test_reporting_protocol_consistency(monkeypatch, tmp_path):
    single_run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ"],
        thesis_prefix="protocol-single",
        veto=False,
        kline_backend="hybrid",
    )
    full_market_run = run_stubbed_full_market_path(
        monkeypatch,
        tmp_path,
        symbols=["000001.SZ", "600519.SH"],
        category="core",
    )

    single_bundle = single_run.result.agent_report_bundle
    full_bundle = full_market_run.report_bundle

    for bundle in [single_bundle, full_bundle]:
        assert bundle.architecture_version == versioning.ARCHITECTURE_VERSION
        assert bundle.branch_schema_version == versioning.BRANCH_SCHEMA_VERSION
        assert bundle.ic_protocol_version == versioning.IC_PROTOCOL_VERSION
        assert bundle.report_protocol_version == versioning.REPORT_PROTOCOL_VERSION
        assert bundle.metadata.get("narrator_read_only") is True
        assert bundle.markdown_report.strip()

    final_report = single_run.result.final_report
    for header in [
        "## 三句话执行摘要",
        "## 市场观点",
        "## 分支结论",
        "## 推荐标的卡片",
        "## 数据覆盖摘要",
        "## 附录：工程诊断",
        "## 数据概览",
        "## 市场概览",
        "## 分析过程",
        "## SubAgent 决策过程、逻辑和依据",
        "## Master Agent 决策过程、逻辑和依据",
        "## 最终投资建议",
        "## 仓位、买卖指令",
        "## 下一步计划",
    ]:
        assert header in final_report

    assert single_bundle.portfolio_plan.target_weights == single_run.artifacts["portfolio_plan"].target_weights
    assert full_bundle.portfolio_plan.target_weights == full_market_run.artifacts["portfolio_plan"].target_weights
    assert single_bundle.ic_decision.action == single_run.artifacts["ic_decision"].action
    assert full_bundle.ic_decision.action == full_market_run.artifacts["ic_decision"].action
