from __future__ import annotations

import json
from pathlib import Path

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


def _write_strategy_record(
    root: Path,
    *,
    market: str,
    strategy: str,
    timestamp: str,
    report_title: str,
    symbol: str = "600000.SH",
    action: str = "buy",
) -> Path:
    run_dir = root / "results" / "strategy_records" / market / strategy / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis_report.md").write_text(
        f"# {report_title}\n\n- symbol: {symbol}\n- action: {action}\n",
        encoding="utf-8",
    )
    (run_dir / "orders.csv").write_text(
        "symbol,action,shares,price\n"
        f"{symbol},{action},100,10.5\n",
        encoding="utf-8",
    )
    (run_dir / "ledger.csv").write_text(
        "symbol,shares,current_price\n"
        f"{symbol},100,10.8\n",
        encoding="utf-8",
    )
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir()
    (raw_dir / "ignored_report.md").write_text("# ignored", encoding="utf-8")
    return run_dir


def test_load_config_backfills_daily_defaults_without_web_runtime(tmp_path):
    config_file = tmp_path / "daily_config.py"
    config_file.write_text(
        "DAILY_CONFIG = {\n"
        '    "market": "CN",\n'
        '    "risk_level": "中等",\n'
        "}\n",
        encoding="utf-8",
    )

    cfg = daily_runner.load_config(str(config_file))

    assert cfg["review_model_priority"] == [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.5-plus",
    ]
    assert cfg["master_reasoning_effort"] == ""
    assert cfg["enable_agent_layer"] is True
    assert cfg["skip_stage1"] is False
    assert cfg["funnel_profile"] == "momentum_leader"
    assert cfg["funnel_max_candidates"] == 200
    assert cfg["trend_windows"] == [20, 60, 120]
    assert cfg["volume_spike_threshold"] == 1.35
    assert cfg["breakout_distance_pct"] == 0.06
    assert cfg["agent_timeout"] == 180.0
    assert cfg["master_timeout"] == 900.0
    assert cfg["kline_backend"] == "heuristic"
    assert "agent_model" not in cfg
    assert "agent_fallback_model" not in cfg
    assert cfg["master_model"] == "moonshot-v1-128k"
    assert cfg["master_fallback_model"] == "deepseek-reasoner"
    assert "pipeline_mode" not in cfg
    assert "history_lookback" not in cfg
    assert "backend_host" not in cfg
    assert "backend_port" not in cfg


def test_load_config_normalizes_legacy_model_fields_into_review_priority(tmp_path):
    config_file = tmp_path / "daily_config.py"
    config_file.write_text(
        "DAILY_CONFIG = {\n"
        '    "agent_model": "qwen-plus",\n'
        '    "agent_fallback_model": "moonshot-v1-128k",\n'
        '    "master_model": "deepseek-chat",\n'
        "}\n",
        encoding="utf-8",
    )

    cfg = daily_runner.load_config(str(config_file))

    assert cfg["review_model_priority"] == [
        "qwen3.5-plus",
        "moonshot-v1-128k",
        "deepseek-chat",
    ]
    assert cfg["agent_model"] == "qwen3.5-plus"
    assert cfg["agent_fallback_model"] == "moonshot-v1-128k"
    assert cfg["master_model"] == "deepseek-chat"
    assert cfg["master_fallback_model"] == "deepseek-reasoner"


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


def test_history_loader_reads_recent_strategy_records_and_builds_recall_context(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_runner, "ROOT", tmp_path)
    strategy_root = tmp_path / "results" / "strategy_records" / "CN" / "alpha_one"
    strategy_root.mkdir(parents=True, exist_ok=True)
    (strategy_root / "latest_notes_payload.md").write_text("ignore latest payload", encoding="utf-8")

    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260401_1000",
        report_title="old report",
        symbol="600001.SH",
        action="buy",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260402_1000",
        report_title="day2 report",
        symbol="600002.SH",
        action="hold",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260403_1000",
        report_title="day3 report",
        symbol="600003.SH",
        action="sell",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260404_1000",
        report_title="day4 report",
        symbol="600004.SH",
        action="buy",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260405_1000",
        report_title="day5 report",
        symbol="600005.SH",
        action="hold",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_one",
        timestamp="20260406_0900",
        report_title="day6 first run",
        symbol="600006.SH",
        action="buy",
    )
    _write_strategy_record(
        tmp_path,
        market="CN",
        strategy="alpha_two",
        timestamp="20260406_1500",
        report_title="day6 second run",
        symbol="600007.SH",
        action="sell",
    )

    loader = daily_runner.HistoryLoader()
    history = loader.load_recent(market="CN")
    recall_context = loader.build_recall_context(history, market="CN")

    assert {item["date"] for item in history} == {
        "20260402",
        "20260403",
        "20260404",
        "20260405",
        "20260406",
    }
    assert len([item for item in history if item["date"] == "20260406"]) == 2
    assert all("raw_exports" not in path for item in history for path in item["csv_files"] + item["markdown_files"])
    assert recall_context["source"] == "strategy_records"
    assert recall_context["market"] == "CN"
    assert recall_context["window_dates"] == ["20260406", "20260405", "20260404", "20260403", "20260402"]
    assert "600007.SH" in recall_context["recent_symbols"]
    assert any(item["action"] == "sell" for item in recall_context["recent_actions"])
    assert "ignore latest payload" not in json.dumps(recall_context, ensure_ascii=False)


def test_analysis_runner_forwards_review_priority_and_recall_context(monkeypatch):
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
            "agent_model": "qwen3.5-plus",
            "agent_fallback_model": "qwen3.5-flash",
            "master_model": "deepseek-reasoner",
            "master_fallback_model": "moonshot-v1-128k",
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
        },
        recall_context={"source": "strategy_records", "recent_symbols": ["600000.SH"]},
    )

    assert captured["mode"] == "batch"
    assert captured["enable_agent_layer"] is True
    assert captured["review_model_priority"] == [
        "qwen3.5-plus",
        "qwen3.5-flash",
        "deepseek-reasoner",
        "moonshot-v1-128k",
    ]
    assert captured["agent_model"] == "qwen3.5-plus"
    assert captured["agent_fallback_model"] == "qwen3.5-flash"
    assert captured["master_model"] == "deepseek-reasoner"
    assert captured["master_fallback_model"] == "moonshot-v1-128k"
    assert captured["master_reasoning_effort"] == "high"
    assert captured["agent_timeout"] == 20.0
    assert captured["master_timeout"] == 45.0
    assert captured["skip_stage1"] is True
    assert captured["recall_context"] == {"source": "strategy_records", "recent_symbols": ["600000.SH"]}


def test_analysis_runner_preserves_master_role_override_from_daily_config(monkeypatch):
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
            "review_model_priority": [
                "qwen3.5-flash",
                "deepseek-chat",
                "moonshot-v1-128k",
                "qwen3.5-plus",
            ],
            "master_model": "moonshot-v1-128k",
            "master_fallback_model": "deepseek-reasoner",
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
        },
        recall_context={"source": "strategy_records", "recent_symbols": ["600000.SH"]},
    )

    assert captured["agent_model"] == "qwen3.5-flash"
    assert captured["agent_fallback_model"] == "deepseek-chat"
    assert captured["master_model"] == "moonshot-v1-128k"
    assert captured["master_fallback_model"] == "deepseek-reasoner"


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
            "review_model_priority": ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"],
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
        "master synthesis / portfolio-level judgment before deterministic risk and sizing",
        "Master Agent reasoning",
        "macro_turns_weaker",
        "candidate_set_decays",
        "600000.SH 浦发银行",
    ]:
        assert header in report


def test_run_once_builds_recall_context_and_returns_report_path(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHistoryLoader:
        def load_recent(self, *_args, **_kwargs):
            return [{"date": "20260409", "strategy": "alpha_one", "record_dir": "fake"}]

        def build_recall_context(self, runs, market="CN"):
            captured["history_runs"] = list(runs)
            return {"source": "strategy_records", "market": market, "recent_symbols": ["600000.SH"]}

        def format_context_section(self, runs):
            return "history context from strategy records"

    class FakeRunner:
        def run(self, cfg, recall_context=None):
            captured["cfg"] = dict(cfg)
            captured["recall_context"] = dict(recall_context or {})
            return _make_pipeline_result()

    class FakePersistenceManager:
        def save(self, report_md, pipeline_result, config):
            captured["report_md"] = report_md
            captured["pipeline_result"] = pipeline_result
            captured["config"] = dict(config)
            return "reports/daily/generated.md"

    monkeypatch.setattr(daily_runner, "HistoryLoader", FakeHistoryLoader)
    monkeypatch.setattr(daily_runner, "AnalysisRunner", lambda: FakeRunner())
    monkeypatch.setattr(daily_runner, "PersistenceManager", lambda: FakePersistenceManager())
    monkeypatch.setattr(market_analyze, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(market_analyze, "get_stock_name", lambda symbol, market="CN": symbol)

    config = {
        "market": "CN",
        "risk_level": "中等",
        "total_capital": 1_000_000,
        "review_model_priority": ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"],
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
    }

    report_path = daily_runner.run_once(config)

    assert report_path == "reports/daily/generated.md"
    assert captured["cfg"]["review_model_priority"] == ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"]
    assert captured["cfg"]["master_reasoning_effort"] == "high"
    assert captured["cfg"]["skip_stage1"] is True
    assert captured["recall_context"] == {
        "source": "strategy_records",
        "market": "CN",
        "recent_symbols": ["600000.SH"],
    }
    assert captured["history_runs"] == [{"date": "20260409", "strategy": "alpha_one", "record_dir": "fake"}]
    assert "history context from strategy records" in str(captured["report_md"])
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
            "review_model_priority": ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"],
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
    assert captured["cfg"]["review_model_priority"] == ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"]
    assert captured["cfg"]["skip_stage1"] is False


def test_run_once_skip_stage1_forwards_flag(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHistoryLoader:
        def load_recent(self, *_args, **_kwargs):
            return []

        def build_recall_context(self, runs, market="CN"):
            return {"source": "strategy_records", "market": market, "recent_symbols": []}

        def format_context_section(self, runs):
            return "_暂无历史分析记录。_"

    class FakeRunner:
        def run(self, cfg, recall_context=None):
            captured["cfg"] = dict(cfg)
            captured["recall_context"] = dict(recall_context or {})
            return _make_pipeline_result()

    class FakePersistenceManager:
        def save(self, report_md, pipeline_result, config):
            captured["config"] = dict(config)
            return "reports/daily/generated.md"

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
            "review_model_priority": ["deepseek-reasoner", "moonshot-v1-128k", "qwen3.5-plus"],
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
        },
        skip_stage1=True,
    )

    assert job_id == "reports/daily/generated.md"
    assert captured["cfg"]["skip_stage1"] is True
    assert captured["config"]["skip_stage1"] is True
    assert captured["recall_context"]["source"] == "strategy_records"


def test_print_last_report_reads_latest_strategy_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(daily_runner, "ROOT", tmp_path)
    _write_strategy_record(
        tmp_path,
        market="US",
        strategy="simulated_portfolio_10000",
        timestamp="20260401_2206",
        report_title="older us report",
        symbol="AAPL",
        action="hold",
    )
    _write_strategy_record(
        tmp_path,
        market="US",
        strategy="simulated_portfolio_10000",
        timestamp="20260402_2214",
        report_title="latest us report",
        symbol="MSFT",
        action="buy",
    )

    daily_runner.print_last_report({"market": "US"})
    out = capsys.readouterr().out

    assert "latest us report" in out
    assert "older us report" not in out


def test_daily_main_backend_only_flag_is_removed(monkeypatch):
    monkeypatch.setattr(daily_runner.sys, "argv", ["daily_runner.py", "--backend-only"])

    with pytest.raises(SystemExit):
        daily_runner.main()


def test_daily_runner_source_is_decoupled_from_web():
    source = Path(daily_runner.__file__).read_text(encoding="utf-8")

    assert "web.services.run_history_store" not in source
    assert "web.main:app" not in source
