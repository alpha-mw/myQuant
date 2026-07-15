from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

import quant_investor.cli.main as cli_main
import quant_investor.market.analyze as market_analyze
import quant_investor.market.run_pipeline as market_pipeline
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)


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
    assert captured["storage_mode"] == "auto"


def test_market_maintain_cli_accepts_parquet_direct_storage_mode(monkeypatch):
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
            "--storage-mode",
            "parquet-direct",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["storage_mode"] == "parquet-direct"


def test_market_maintain_cli_dispatches_staged_options(monkeypatch):
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
            "--staged",
            "--resume",
            "--batch-size",
            "200",
            "--max-batches-per-run",
            "2",
            "--min-symbol-success-rate",
            "0.95",
            "--target-date",
            "20260316",
            "--daily-window",
            "--fail-on-incomplete",
            "false",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["staged"] is True
    assert captured["resume"] is True
    assert captured["batch_size"] == 200
    assert captured["max_batches_per_run"] == 2
    assert captured["min_symbol_success_rate"] == 0.95
    assert captured["target_date"] == "20260316"
    assert captured["daily_window"] is True
    assert captured["fail_on_incomplete"] is False


def test_market_maintain_cli_uses_staged_batch_size_default(monkeypatch):
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
            "--staged",
        ]
    )

    assert captured["batch_size"] == 200
    assert captured["staged"] is True


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
            "qwen3.5-plus",
            "--agent-fallback-model",
            "qwen3.5-flash",
            "--master-model",
            "deepseek-reasoner",
            "--master-fallback-model",
            "moonshot-v1-128k",
            "--agent-timeout",
            "30",
            "--master-timeout",
            "60",
            "--funnel-profile",
            "momentum_leader",
            "--max-candidates",
            "150",
            "--trend-windows",
            "15",
            "45",
            "120",
            "--volume-spike-threshold",
            "1.5",
            "--breakout-distance-pct",
            "0.05",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["mode"] == "sample"
    assert captured["categories"] == ["hs300"]
    assert captured["enable_agent_layer"] is False
    assert captured["agent_model"] == "qwen3.5-plus"
    assert captured["agent_fallback_model"] == "qwen3.5-flash"
    assert captured["master_model"] == "deepseek-reasoner"
    assert captured["master_fallback_model"] == "moonshot-v1-128k"
    assert captured["agent_timeout"] == 30.0
    assert captured["master_timeout"] == 60.0
    assert captured["funnel_profile"] == "momentum_leader"
    assert captured["max_candidates"] == 150
    assert captured["shortlist_size"] == 50
    assert captured["trend_windows"] == [15, 45, 120]
    assert captured["volume_spike_threshold"] == 1.5
    assert captured["breakout_distance_pct"] == 0.05


def test_cli_timeout_defaults_are_long_running():
    parser = cli_main._build_parser()

    research_args = parser.parse_args(
        [
            "research",
            "run",
            "--market",
            "CN",
            "--stocks",
            "600000.SH",
        ]
    )
    analyze_args = parser.parse_args(
        [
            "market",
            "analyze",
            "--market",
            "CN",
        ]
    )
    run_args = parser.parse_args(
        [
            "market",
            "run",
            "--market",
            "CN",
        ]
    )

    assert research_args.agent_timeout == 180.0
    assert research_args.master_timeout == 900.0
    assert research_args.max_candidates == 500
    assert analyze_args.agent_timeout == 180.0
    assert analyze_args.master_timeout == 900.0
    assert analyze_args.max_candidates == 500
    assert run_args.agent_timeout == 180.0
    assert run_args.master_timeout == 900.0
    assert run_args.max_candidates == 500


def test_research_cli_rejects_retired_intelligence_switch():
    parser = cli_main._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "research",
                "run",
                "--market",
                "CN",
                "--stocks",
                "600000.SH",
                "--no-intelligence",
            ]
        )


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
            "qwen3.5-plus",
            "--agent-fallback-model",
            "qwen3.5-flash",
            "--master-model",
            "deepseek-reasoner",
            "--master-fallback-model",
            "moonshot-v1-128k",
            "--agent-timeout",
            "25",
            "--master-timeout",
            "55",
            "--funnel-profile",
            "momentum_leader",
            "--max-candidates",
            "180",
            "--trend-windows",
            "20",
            "60",
            "120",
            "--volume-spike-threshold",
            "1.4",
            "--breakout-distance-pct",
            "0.04",
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
    assert captured["agent_model"] == "qwen3.5-plus"
    assert captured["agent_fallback_model"] == "qwen3.5-flash"
    assert captured["master_model"] == "deepseek-reasoner"
    assert captured["master_fallback_model"] == "moonshot-v1-128k"
    assert captured["agent_timeout"] == 25.0
    assert captured["master_timeout"] == 55.0
    assert captured["funnel_profile"] == "momentum_leader"
    assert captured["max_candidates"] == 180
    assert captured["shortlist_size"] == 50
    assert captured["trend_windows"] == [20, 60, 120]
    assert captured["volume_spike_threshold"] == 1.4
    assert captured["breakout_distance_pct"] == 0.04


def test_market_storage_internal_cli_dispatches(monkeypatch):
    captured: dict[str, Any] = {}

    def _storage_validate(**kwargs):
        captured["storage_validate"] = kwargs
        return {"status": "passed"}

    def _storage_validate_clean(**kwargs):
        captured["storage_validate_clean"] = kwargs
        return {"status": "passed"}

    def _materialize_serving(**kwargs):
        captured["materialize_serving"] = kwargs
        return {"status": "materialized"}

    def _materialize_features(**kwargs):
        captured["materialize_features"] = kwargs
        return {"status": "materialized"}

    def _storage_diff(**kwargs):
        captured["storage_diff"] = kwargs
        return {"status": "passed"}

    monkeypatch.setattr(cli_main, "run_storage_validate", _storage_validate)
    monkeypatch.setattr(cli_main, "run_storage_validate_clean", _storage_validate_clean)
    monkeypatch.setattr(cli_main, "run_materialize_serving", _materialize_serving)
    monkeypatch.setattr(cli_main, "run_materialize_features", _materialize_features)
    monkeypatch.setattr(cli_main, "run_storage_diff", _storage_diff)

    cli_main.main(["market", "storage-validate", "--market", "CN"])
    cli_main.main(["market", "storage-validate-clean", "--market", "CN"])
    cli_main.main(["market", "materialize-serving", "--market", "CN"])
    cli_main.main(["market", "materialize-features", "--market", "CN", "--trade-date", "20260103"])
    cli_main.main(["market", "storage-diff", "--market", "CN"])

    assert captured["storage_validate"]["market"] == "CN"
    assert captured["storage_validate_clean"]["market"] == "CN"
    assert captured["materialize_serving"]["market"] == "CN"
    assert captured["materialize_features"] == {"market": "CN", "trade_date": "20260103"}
    assert captured["storage_diff"]["market"] == "CN"


def test_market_download_alias_requires_category():
    with pytest.raises(SystemExit):
        cli_main.main(
            [
                "market",
                "download",
                "--market",
                "CN",
            ]
        )


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
            "runtime_profile": {
                "market": "CN",
                "universe": "hs300",
                "stages": [{"name": "dag_symbol_list", "duration_ms": 1.0, "metadata": {}}],
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
        review_model_priority=["deepseek-chat"],
        agent_timeout=20.0,
        master_timeout=40.0,
        verbose=False,
    )

    assert output["download"]["status"] == "snapshot_only"
    assert output["download"]["reason"] == "analysis_uses_local_data_snapshot"
    assert output["download"]["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert output["runtime_profile"]["stages"][0]["name"] == "dag_symbol_list"
    assert captured_analysis["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert captured_analysis["enable_agent_layer"] is True
    assert captured_analysis["review_model_priority"] == ["deepseek-chat"]
    assert captured_analysis["master_reasoning_effort"] == "high"
    assert captured_analysis["agent_timeout"] == 20.0
    assert captured_analysis["master_timeout"] == 40.0
    assert captured_analysis["shortlist_size"] == 50


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
    assert captured_analysis["agent_timeout"] == 180.0
    assert captured_analysis["master_timeout"] == 900.0
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


def test_unified_pipeline_forwards_recall_context(monkeypatch):
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

    recall_context = {
        "source": "strategy_records",
        "market": "CN",
        "recent_symbols": ["600000.SH"],
    }
    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        recall_context=recall_context,
        verbose=False,
    )

    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert captured_analysis["recall_context"] == recall_context


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
                    "branch_schema_version": BRANCH_SCHEMA_VERSION,
                    "ic_protocol_version": IC_PROTOCOL_VERSION,
                    "report_protocol_version": REPORT_PROTOCOL_VERSION,
                }
            ),
            "branch_summaries": {
                "quant": {"score": 0.1},
                "fundamental": {"score": 0.1},
                "macro": {"score": 0.0},
            },
            "data_quality_issues": [],
            "resolver": {"resolution_strategy": "logical_full_a"},
            "report_bundle": SimpleNamespace(
                architecture_version=ARCHITECTURE_VERSION,
                branch_schema_version=BRANCH_SCHEMA_VERSION,
                likelihood_schema_version=LIKELIHOOD_SCHEMA_VERSION,
                ic_protocol_version=IC_PROTOCOL_VERSION,
                report_protocol_version=REPORT_PROTOCOL_VERSION,
                markdown_report="",
                executive_summary=[],
                market_view=[],
            ),
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
    assert captured_dag["runtime_profiler"] is not None
    assert captured_dag["agent_model"] == "deepseek-reasoner"
    assert captured_dag["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert output["analysis_meta"]["model_role_metadata"]["branch_model"] == "deepseek-reasoner"
    assert output["analysis_meta"]["master_model"] == "moonshot-v1-128k"
    assert output["analysis_meta"]["bayesian_shortlist_symbols"] == ["000001.SZ"]
    assert output["analysis_meta"]["bayesian_record_count"] == 0
    assert output["analysis_meta"]["data_snapshot"]["local_latest_trade_date"] == "20260326"
    assert any(
        step["stage"] == "master_synthesis"
        for step in output["analysis_meta"]["execution_trace"]["steps"]
    )
    assert output["analysis_meta"]["what_if_plan"]["scenarios"][0]["scenario_name"] == "macro_turns_weaker"
    assert output["runtime_profile"]["stages"][-1]["name"] == "analysis_report_persistence"
    assert output["analysis_meta"]["runtime_profile"] == output["runtime_profile"]
    assert output["architecture_version"] == ARCHITECTURE_VERSION
    assert output["analysis_meta"]["likelihood_schema_version"] == (
        LIKELIHOOD_SCHEMA_VERSION
    )
    assert output["analysis_meta"]["ic_protocol_version"] == IC_PROTOCOL_VERSION
    runtime_profile_json = output["reports"]["runtime_profile_json"]
    runtime_profile_md = output["reports"]["runtime_profile_md"]
    assert json.loads(Path(runtime_profile_json).read_text(encoding="utf-8"))["stages"][-1]["name"] == "analysis_report_persistence"
    assert Path(runtime_profile_md).read_text(encoding="utf-8").startswith("# Market Runtime Profile")
