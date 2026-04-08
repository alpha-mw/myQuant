from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import daily_runner
import quant_investor.market.analyze as market_analyze
import quant_investor.market.run_pipeline as market_pipeline

from tests.unit.test_full_market_batch_reports import _make_cn_all_results


def _make_pipeline_result() -> dict[str, object]:
    return {
        "analysis": _make_cn_all_results(),
        "reports": {},
        "download": {
            "status": "downloaded",
            "reason": "stale_or_incomplete",
            "completeness_after": {"blocking_incomplete_count": 0},
        },
        "categories": ["hs300"],
        "timing": {
            "download_seconds": 1.5,
            "analysis_seconds": 2.5,
            "total_seconds": 4.0,
        },
        "analysis_meta": {
            "market": "CN",
            "batch_count": 1,
            "total_stocks": 1,
            "category_count": 1,
            "branch_model": "deepseek-reasoner",
            "master_model": "moonshot-v1-128k",
            "agent_layer_enabled": True,
            "model_role_metadata": {
                "branch_model": "deepseek-reasoner",
                "master_model": "moonshot-v1-128k",
                "branch_provider": "deepseek",
                "master_provider": "kimi",
                "branch_timeout": 20.0,
                "master_timeout": 45.0,
                "agent_layer_enabled": True,
                "branch_role": "per-stock analysis",
                "master_role": "master synthesis / portfolio-level judgment before deterministic risk and sizing",
            },
            "execution_trace": {
                "key_parameters": {"batch_count": 1, "category_count": 1},
                "steps": [
                    {
                        "stage": "data_check_download",
                        "role": "system",
                        "model": "deterministic",
                        "success": True,
                        "conclusion": "download finished",
                    },
                    {
                        "stage": "master_synthesis",
                        "role": "master",
                        "model": "moonshot-v1-128k",
                        "success": True,
                        "conclusion": "master synthesis completed",
                    },
                ],
                "final_deterministic_outcome": {
                    "selected_count": 1,
                    "target_exposure": 0.4,
                    "max_single_weight": 0.4,
                    "hard_veto": False,
                },
            },
            "what_if_plan": {
                "scenarios": [
                    {
                        "scenario_name": "macro_turns_weaker",
                        "trigger": "macro weakens",
                        "monitoring_indicators": ["macro_score"],
                        "action": "reduce risk",
                        "position_adjustment_rule": "downsize",
                        "rerun_full_market_daily_path": True,
                    },
                    {
                        "scenario_name": "single_name_stop_loss_or_reversal",
                        "trigger": "stop loss or signal reversal",
                        "monitoring_indicators": ["price", "signal"],
                        "action": "trim",
                        "position_adjustment_rule": "reduce to zero or cap",
                        "rerun_full_market_daily_path": False,
                    },
                    {
                        "scenario_name": "candidate_set_decays",
                        "trigger": "candidate set shrinks",
                        "monitoring_indicators": ["candidate_count"],
                        "action": "rerun scan",
                        "position_adjustment_rule": "shift to higher confidence names",
                        "rerun_full_market_daily_path": True,
                    },
                ]
            },
            "ic_hints_by_symbol": {"000001.SZ": {"action": "buy"}},
        },
    }


def test_load_config_backfills_role_split_defaults(tmp_path):
    config_file = tmp_path / "daily_config.py"
    config_file.write_text(
        "DAILY_CONFIG = {\n"
        '    "market": "CN",\n'
        '    "risk_level": "中等",\n'
        "}\n",
        encoding="utf-8",
    )

    cfg = daily_runner.load_config(str(config_file))

    assert cfg["agent_model"] == "moonshot-v1-128k"
    assert cfg["master_model"] == "moonshot-v1-128k"
    assert cfg["master_reasoning_effort"] == ""
    assert cfg["enable_agent_layer"] is True
    assert cfg["skip_stage1"] is False
    assert cfg["kline_backend"] == "heuristic"


def test_bootstrap_project_venv_reexecs_when_not_running_inside_venv(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = list(argv)
        raise RuntimeError("execv intercepted")

    monkeypatch.setattr(daily_runner.sys, "prefix", "/opt/homebrew")
    monkeypatch.setattr(daily_runner.os, "execv", _fake_execv)

    with pytest.raises(RuntimeError, match="execv intercepted"):
        daily_runner._bootstrap_project_venv()

    assert captured["path"].endswith(".venv/bin/python")
    assert captured["argv"][0].endswith(".venv/bin/python")
    assert captured["argv"][1:] == list(daily_runner.sys.argv)


def test_analysis_runner_forwards_role_split_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "reports": {}, "download": {}, "timing": {}, "analysis_meta": {}}

    monkeypatch.setattr(market_pipeline, "run_unified_pipeline", _fake_pipeline)
    monkeypatch.setattr("quant_investor.model_roles.has_provider_for_model", lambda _model: True)

    runner = daily_runner.AnalysisRunner()
    runner.run(
        {
            "market": "CN",
            "agent_model": "deepseek-reasoner",
            "master_model": "moonshot-v1-128k",
            "master_reasoning_effort": "high",
            "agent_timeout": 20.0,
            "master_timeout": 45.0,
            "top_k": 12,
            "total_capital": 1_000_000,
            "years": 3,
            "workers": 4,
            "enable_agent_layer": True,
            "skip_stage1": True,
            "skip_download": False,
        }
    )

    assert captured["mode"] == "batch"
    assert captured["enable_agent_layer"] is True
    assert captured["agent_model"] == "deepseek-reasoner"
    assert captured["master_model"] == "moonshot-v1-128k"
    assert captured["master_reasoning_effort"] == "high"
    assert captured["agent_timeout"] == 20.0
    assert captured["master_timeout"] == 45.0
    assert captured["agent_fallback_model"] == "deepseek-reasoner"
    assert captured["master_fallback_model"] == "deepseek-chat"
    assert captured["skip_stage1"] is True


def test_report_builder_renders_run_context(monkeypatch):
    monkeypatch.setattr(
        market_analyze,
        "get_stock_name",
        lambda symbol, market="CN": "浦发银行" if symbol == "600000.SH" else symbol,
    )

    builder = daily_runner.ReportBuilder()
    report = builder.build(
        _make_pipeline_result(),
        {
            "market": "CN",
            "risk_level": "中等",
            "total_capital": 1_000_000,
            "agent_model": "deepseek-reasoner",
            "master_model": "moonshot-v1-128k",
            "master_reasoning_effort": "high",
            "kline_backend": "heuristic",
            "top_k": 20,
            "agent_timeout": 20.0,
            "master_timeout": 45.0,
            "enable_agent_layer": True,
            "skip_download": False,
            "years": 3,
            "workers": 4,
            "schedule_time": "17:30",
        },
        history=[],
    )

    for header in [
        "## § 4 模型角色与执行轨迹",
        "## 模型角色元数据",
        "## 执行 Trace",
        "## What-If 响应计划",
        "Master synthesis / portfolio-level judgment before deterministic risk and sizing",
        "Master Agent reasoning",
        "macro_turns_weaker",
        "candidate_set_decays",
        "600000.SH 浦发银行",
    ]:
        assert header in report


def test_run_once_persists_role_split_and_default_models(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHistoryLoader:
        def load_recent(self, *_args, **_kwargs):
            return []

        def format_context_section(self, runs):
            return "_暂无历史分析记录。_"

    class FakeRunner:
        def run(self, cfg):
            captured["cfg"] = dict(cfg)
            return _make_pipeline_result()

    class FakePersistenceManager:
        def save(self, report_md, pipeline_result, config):
            captured["report_md"] = report_md
            captured["pipeline_result"] = pipeline_result
            captured["config"] = dict(config)
            return "job-123"

    monkeypatch.setattr(daily_runner, "HistoryLoader", FakeHistoryLoader)
    monkeypatch.setattr(daily_runner, "AnalysisRunner", lambda: FakeRunner())
    monkeypatch.setattr(daily_runner, "PersistenceManager", lambda: FakePersistenceManager())
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(market_analyze, "get_stock_name", lambda symbol, market="CN": symbol)

    config = {
        "market": "CN",
        "risk_level": "中等",
        "total_capital": 1_000_000,
        "agent_model": "deepseek-reasoner",
        "master_model": "moonshot-v1-128k",
        "master_reasoning_effort": "high",
        "kline_backend": "heuristic",
        "top_k": 20,
        "agent_timeout": 20.0,
        "master_timeout": 45.0,
        "enable_agent_layer": True,
        "skip_stage1": True,
        "skip_download": False,
        "years": 3,
        "workers": 4,
        "schedule_time": "17:30",
        "report_dir": str(Path("reports") / "daily"),
        "history_lookback": 5,
        "backend_host": "127.0.0.1",
        "backend_port": 8000,
    }

    job_id = daily_runner.run_once(config)

    assert job_id == "job-123"
    assert captured["cfg"]["agent_model"] == "deepseek-reasoner"
    assert captured["cfg"]["master_model"] == "moonshot-v1-128k"
    assert captured["cfg"]["master_reasoning_effort"] == "high"
    assert captured["cfg"]["skip_stage1"] is True
    assert "## § 4 模型角色与执行轨迹" in str(captured["report_md"])
    assert "## What-If 响应计划" in str(captured["report_md"])
    assert captured["pipeline_result"]["analysis_meta"]["model_role_metadata"]["branch_model"] == "deepseek-reasoner"


def test_daily_main_default_path_calls_run_once_without_extra_flags(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        daily_runner,
        "load_config",
        lambda _path=None: {
            "market": "CN",
            "risk_level": "中等",
            "total_capital": 1_000_000,
            "agent_model": "deepseek-reasoner",
            "master_model": "moonshot-v1-128k",
            "master_reasoning_effort": "high",
            "kline_backend": "heuristic",
            "top_k": 20,
            "agent_timeout": 20.0,
            "master_timeout": 45.0,
            "enable_agent_layer": True,
            "skip_stage1": False,
            "skip_download": False,
            "years": 3,
            "workers": 4,
            "schedule_time": "17:30",
            "report_dir": "reports/daily",
            "history_lookback": 5,
            "backend_host": "127.0.0.1",
            "backend_port": 8000,
        },
    )
    monkeypatch.setattr(
        daily_runner,
        "run_once",
        lambda cfg, skip_download=False, skip_stage1=False: captured.update(
            {"cfg": cfg, "skip_download": skip_download, "skip_stage1": skip_stage1}
        ) or "job",
    )
    monkeypatch.setattr(daily_runner.sys, "argv", ["daily_runner.py"])

    daily_runner.main()

    assert captured["skip_download"] is False
    assert captured["skip_stage1"] is False
    assert captured["cfg"]["agent_model"] == "deepseek-reasoner"
    assert captured["cfg"]["master_model"] == "moonshot-v1-128k"
    assert captured["cfg"]["skip_stage1"] is False


def test_run_once_skip_stage1_forwards_flag(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHistoryLoader:
        def load_recent(self, *_args, **_kwargs):
            return []

        def format_context_section(self, runs):
            return "_暂无历史分析记录。_"

    class FakeRunner:
        def run(self, cfg):
            captured["cfg"] = dict(cfg)
            return _make_pipeline_result()

    class FakePersistenceManager:
        def save(self, report_md, pipeline_result, config):
            captured["config"] = dict(config)
            return "job-456"

    monkeypatch.setattr(daily_runner, "HistoryLoader", FakeHistoryLoader)
    monkeypatch.setattr(daily_runner, "AnalysisRunner", lambda: FakeRunner())
    monkeypatch.setattr(daily_runner, "PersistenceManager", lambda: FakePersistenceManager())
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(market_analyze, "get_stock_name", lambda symbol, market="CN": symbol)

    job_id = daily_runner.run_once(
        {
            "market": "CN",
            "risk_level": "中等",
            "total_capital": 1_000_000,
            "agent_model": "deepseek-reasoner",
            "master_model": "moonshot-v1-128k",
            "master_reasoning_effort": "high",
            "kline_backend": "heuristic",
            "top_k": 20,
            "agent_timeout": 20.0,
            "master_timeout": 45.0,
            "enable_agent_layer": True,
            "skip_download": False,
            "years": 3,
            "workers": 4,
            "schedule_time": "17:30",
            "report_dir": str(Path("reports") / "daily"),
            "history_lookback": 5,
            "backend_host": "127.0.0.1",
            "backend_port": 8000,
        },
        skip_stage1=True,
    )

    assert job_id == "job-456"
    assert captured["cfg"]["skip_stage1"] is True
    assert captured["config"]["skip_stage1"] is True
