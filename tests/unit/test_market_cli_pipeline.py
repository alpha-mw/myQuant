from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import quant_investor.cli.main as cli_main
import quant_investor.market.analyze as market_analyze
import quant_investor.market.run_pipeline as market_pipeline


def test_market_maintain_cli_dispatches_to_maintenance(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_maintenance(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli_main, "run_market_maintenance", _run_market_maintenance)

    cli_main.main(
        [
            "market",
            "maintain",
            "--market",
            "CN",
            "--years",
            "5",
            "--workers",
            "6",
            "--max-rounds",
            "3",
            "--fail-on-incomplete",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["years"] == 5
    assert captured["max_workers"] == 6
    assert captured["max_rounds"] == 3
    assert captured["fail_on_incomplete"] is True


def test_market_analyze_cli_passes_agent_layer_args(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured.update(kwargs)
        return {"results": {}, "reports": {}}

    monkeypatch.setattr(cli_main, "run_market_analysis", _run_market_analysis)

    cli_main.main(
        [
            "market",
            "analyze",
            "--market",
            "CN",
            "--mode",
            "sample",
            "--category",
            "hs300",
            "--no-agent-layer",
            "--agent-model",
            "deepseek-chat",
            "--master-model",
            "deepseek-chat",
            "--agent-timeout",
            "30",
            "--master-timeout",
            "60",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["mode"] == "sample"
    assert captured["categories"] == ["hs300"]
    assert captured["enable_agent_layer"] is False
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["master_model"] == "deepseek-chat"
    assert captured["agent_timeout"] == 30.0
    assert captured["master_timeout"] == 60.0


def test_market_run_cli_dispatches_to_unified_pipeline(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_pipeline(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "reports": {}, "download": {}, "timing": {}}

    monkeypatch.setattr(cli_main, "run_market_pipeline", _run_market_pipeline)

    cli_main.main(
        [
            "market",
            "run",
            "--market",
            "CN",
            "--category",
            "hs300",
            "--mode",
            "sample",
            "--skip-download",
            "--agent-model",
            "deepseek-chat",
            "--master-model",
            "deepseek-chat",
            "--agent-timeout",
            "25",
            "--master-timeout",
            "55",
            "--years",
            "5",
            "--workers",
            "6",
            "--max-download-rounds",
            "3",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]
    assert captured["mode"] == "sample"
    assert captured["skip_download"] is True
    assert captured["force_download"] is False
    assert captured["enable_agent_layer"] is True
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["master_model"] == "deepseek-chat"
    assert captured["agent_timeout"] == 25.0
    assert captured["master_timeout"] == 55.0
    assert captured["years"] == 5
    assert captured["workers"] == 6
    assert captured["max_download_rounds"] == 3


def test_market_download_cli_uses_maintenance_alias(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_maintenance(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli_main, "run_market_maintenance", _run_market_maintenance)

    cli_main.main(
        [
            "market",
            "download",
            "--market",
            "CN",
            "--years",
            "3",
            "--workers",
            "4",
            "--batch-size",
            "50",
            "--max-rounds",
            "2",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["years"] == 3
    assert captured["max_workers"] == 4
    assert captured["batch_size"] == 50
    assert captured["max_rounds"] == 2
    assert captured["deprecated_alias"] is True


def test_unified_pipeline_stage1_builds_advisory_snapshot(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(
        market_pipeline,
        "build_market_data_snapshot",
        lambda **kwargs: {
            "market": "CN",
            "universe_key": "hs300",
            "local_latest_trade_date": "20260326",
            "freshness_mode": "stable",
            "category_symbol_counts": {"hs300": 1},
            "date_distribution_top": [{"trade_date": "20260326", "symbol_count": 1}],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 0,
            "summary_text": "本地 A 股数据更新至 20260326。",
        },
    )
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        enable_agent_layer=True,
        agent_model="deepseek-chat",
        master_model="deepseek-chat",
        agent_timeout=20.0,
        master_timeout=40.0,
        verbose=False,
    )

    assert output["download"]["status"] == "snapshot_only"
    assert output["download"]["reason"] == "analysis_uses_local_data_snapshot"
    assert output["download"]["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert captured_analysis["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert captured_analysis["enable_agent_layer"] is True
    assert captured_analysis["agent_model"] == "deepseek-chat"
    assert captured_analysis["master_model"] == "deepseek-chat"
    assert captured_analysis["master_reasoning_effort"] == "high"
    assert captured_analysis["agent_timeout"] == 20.0
    assert captured_analysis["master_timeout"] == 40.0


def test_unified_pipeline_uses_local_snapshot_even_when_data_is_stale(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(
        market_pipeline,
        "build_market_data_snapshot",
        lambda **kwargs: {
            "market": "CN",
            "universe_key": "hs300",
            "local_latest_trade_date": "20260325",
            "freshness_mode": "stable",
            "category_symbol_counts": {"hs300": 1},
            "date_distribution_top": [{"trade_date": "20260325", "symbol_count": 1}],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 1,
            "summary_text": "本地 A 股数据当前更新至 20260325，分析继续使用现有本地数据。",
        },
    )
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        max_download_rounds=2,
        verbose=False,
    )

    assert output["download"]["status"] == "snapshot_only"
    assert output["download"]["data_snapshot"]["local_latest_trade_date"] == "20260325"
    assert captured_analysis["master_reasoning_effort"] == "high"
    assert captured_analysis["categories"] == ["hs300"]
    assert captured_analysis["mode"] == "sample"
    assert captured_analysis["data_snapshot"]["data_quality_issue_count"] == 1


def test_unified_pipeline_skip_download_becomes_compatibility_warning(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(
        market_pipeline,
        "build_market_data_snapshot",
        lambda **kwargs: {
            "market": "CN",
            "universe_key": "hs300",
            "local_latest_trade_date": "20260326",
            "freshness_mode": "stable",
            "category_symbol_counts": {"hs300": 2},
            "date_distribution_top": [{"trade_date": "20260326", "symbol_count": 2}],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 0,
            "summary_text": "本地 A 股数据更新至 20260326。",
        },
    )
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        skip_download=True,
        verbose=False,
    )

    assert output["download"]["status"] == "snapshot_only"
    assert output["download"]["warning"] == "skip_download_ignored"
    assert captured_analysis["market"] == "CN"


def test_unified_pipeline_skip_stage1_becomes_compatibility_warning(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(
        market_pipeline,
        "build_market_data_snapshot",
        lambda **kwargs: {
            "market": "CN",
            "universe_key": "hs300",
            "local_latest_trade_date": "20260326",
            "freshness_mode": "stable",
            "category_symbol_counts": {"hs300": 1},
            "date_distribution_top": [{"trade_date": "20260326", "symbol_count": 1}],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 0,
            "summary_text": "本地 A 股数据更新至 20260326。",
        },
    )
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        skip_stage1=True,
        verbose=False,
    )

    assert output["download"]["status"] == "snapshot_only"
    assert output["download"]["warning"] == "skip_stage1_ignored"
    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert captured_analysis["market"] == "CN"
    assert captured_analysis["categories"] == ["hs300"]


def test_run_market_analysis_exposes_role_metadata(monkeypatch, tmp_path):
    captured_dag: dict[str, Any] = {}

    class _Payload:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload
            for key, value in payload.items():
                setattr(self, key, value)

        def to_dict(self) -> dict[str, Any]:
            return dict(self._payload)

    def _fake_execute_market_dag(**kwargs):
        captured_dag.update(kwargs)
        return {
            "model_role_metadata": _Payload(
                {
                    "branch_model": "deepseek-reasoner",
                    "master_model": "moonshot-v1-128k",
                    "agent_layer_enabled": True,
                }
            ),
            "execution_trace": _Payload(
                {
                    "key_parameters": {"batch_count": 1},
                    "steps": [
                        {
                            "stage": "master_synthesis",
                            "role": "master",
                            "model": "moonshot-v1-128k",
                            "success": True,
                            "conclusion": "master synthesis ok",
                        }
                    ],
                    "final_deterministic_outcome": {"selected_count": 1},
                }
            ),
            "what_if_plan": _Payload(
                {
                    "scenarios": [
                        {
                            "scenario_name": "macro_turns_weaker",
                            "trigger": "macro weakens",
                            "monitoring_indicators": ["macro"],
                            "action": "reduce risk",
                            "position_adjustment_rule": "downsize",
                            "rerun_full_market_daily_path": True,
                        }
                    ]
                }
            ),
            "global_context": _Payload(
                {
                    "market": "CN",
                    "universe_key": "hs300",
                    "universe_symbols": ["000001.SZ"],
                    "metadata": {
                        "data_snapshot": {
                            "market": "CN",
                            "universe_key": "hs300",
                            "local_latest_trade_date": "20260326",
                            "freshness_mode": "stable",
                            "category_symbol_counts": {"hs300": 1},
                            "date_distribution_top": [{"trade_date": "20260326", "symbol_count": 1}],
                            "data_directories": ["data/cn_market_full/hs300"],
                            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
                            "data_quality_issue_count": 0,
                            "summary_text": "本地 A 股数据更新至 20260326。",
                        }
                    },
                }
            ),
            "portfolio_decision": _Payload({"target_exposure": 0.4, "shortlist": [{"symbol": "000001.SZ"}]}),
            "symbol_research_packets": {"000001.SZ": _Payload({"symbol": "000001.SZ", "company_name": "平安银行"})},
            "shortlist": [_Payload({"symbol": "000001.SZ", "company_name": "平安银行"})],
            "review_bundle": _Payload(
                {
                    "ic_hints_by_symbol": {"000001.SZ": {"action": "buy"}},
                    "branch_schema_version": "branch-schema.v12.unified-mainline",
                    "ic_protocol_version": "ic.v1",
                    "report_protocol_version": "report.v1",
                }
            ),
            "branch_summaries": {},
            "data_quality_issues": [],
            "resolver": {"resolution_strategy": "logical_full_a"},
            "report_bundle": SimpleNamespace(markdown_report="", executive_summary=[], market_view=[]),
        }

    monkeypatch.setattr(market_analyze, "execute_market_dag", _fake_execute_market_dag)
    monkeypatch.setattr(
        market_analyze,
        "get_market_settings",
        lambda market: SimpleNamespace(
            market=market,
            default_batch_size=1,
            analysis_output_dir=tmp_path,
            market_name="中国A股",
            report_flag="CN",
            currency_symbol="¥",
        ),
    )
    monkeypatch.setattr(market_analyze, "normalize_categories", lambda _market, categories=None: list(categories or ["hs300"]))
    monkeypatch.setattr(
        market_analyze,
        "_synthesize_legacy_analysis_results_from_dag",
        lambda **kwargs: {
            "hs300": [
                {
                    "batch_id": 1,
                    "strategy": {"candidate_symbols": ["000001.SZ"]},
                    "recommendations": [],
                    "execution_log": ["[INFO] completed"],
                }
            ]
        },
    )
    monkeypatch.setattr(
        market_analyze,
        "generate_full_report",
        lambda *args, **kwargs: {
            "summary_report": str(tmp_path / "summary.md"),
            "trade_report": str(tmp_path / "trade.md"),
            "trade_data": str(tmp_path / "trade.json"),
            "candidate_index": str(tmp_path / "cand.json"),
        },
    )

    output = market_analyze.run_market_analysis(
        market="CN",
        mode="sample",
        categories=["hs300"],
        total_capital=1_000_000,
        top_k=1,
        verbose=False,
        data_snapshot={
            "market": "CN",
            "universe_key": "hs300",
            "local_latest_trade_date": "20260326",
            "freshness_mode": "stable",
            "category_symbol_counts": {"hs300": 1},
            "date_distribution_top": [{"trade_date": "20260326", "symbol_count": 1}],
            "data_directories": ["data/cn_market_full/hs300"],
            "resolver_priority": ["hs300", "zz500", "zz1000", "other"],
            "data_quality_issue_count": 0,
            "summary_text": "本地 A 股数据更新至 20260326。",
        },
        enable_agent_layer=True,
        agent_model="deepseek-reasoner",
        master_model="moonshot-v1-128k",
        agent_timeout=20.0,
        master_timeout=45.0,
    )

    assert captured_dag["agent_model"] == "deepseek-reasoner"
    assert captured_dag["master_model"] == "moonshot-v1-128k"
    assert captured_dag["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert output["analysis_meta"]["model_role_metadata"]["branch_model"] == "deepseek-reasoner"
    assert output["analysis_meta"]["model_role_metadata"]["master_model"] == "moonshot-v1-128k"
    assert output["analysis_meta"]["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert any(
        step["stage"] == "master_synthesis"
        for step in output["analysis_meta"]["execution_trace"]["steps"]
    )
    assert output["analysis_meta"]["what_if_plan"]["scenarios"][0]["scenario_name"] == "macro_turns_weaker"
