from __future__ import annotations

from pathlib import Path

from quant_investor.agent_protocol import ReportBundle

from tests.integration._helpers import assert_protocol_bundle, run_stubbed_full_market_path


def test_full_market_batch_end_to_end(monkeypatch, tmp_path):
    run = run_stubbed_full_market_path(
        monkeypatch,
        tmp_path,
        symbols=["000001.SZ", "600519.SH"],
        category="core",
    )

    report_bundle = run.report_bundle
    reports = run.output["reports"]
    dag_kwargs = run.captured_dag_kwargs[0]

    assert dag_kwargs["market"] == "CN"
    assert dag_kwargs["categories"] == ["core"]
    assert dag_kwargs["master_reasoning_effort"] == "high"
    assert isinstance(report_bundle, ReportBundle)
    assert report_bundle.agent_name == "NarratorAgent"
    assert_protocol_bundle(
        report_bundle,
        risk_decision=run.artifacts["risk_decision"],
        ic_decision=run.artifacts["ic_decision"],
        portfolio_plan=run.artifacts["portfolio_plan"],
    )
    assert Path(reports["summary_report"]).exists()
    assert Path(reports["trade_report"]).exists()
    assert Path(reports["trade_data"]).exists()
    assert Path(reports["candidate_index"]).exists()


def test_full_market_batch_passes_agent_layer_kwargs(monkeypatch, tmp_path):
    run = run_stubbed_full_market_path(
        monkeypatch,
        tmp_path,
        symbols=["000001.SZ", "600519.SH"],
        category="core",
        analysis_kwargs={
            "enable_agent_layer": True,
            "agent_model": "deepseek-chat",
            "master_model": "deepseek-chat",
            "master_reasoning_effort": "high",
            "agent_timeout": 30.0,
            "master_timeout": 60.0,
        },
    )

    dag_kwargs = run.captured_dag_kwargs[0]

    assert dag_kwargs["enable_agent_layer"] is True
    assert dag_kwargs["agent_model"] == "deepseek-chat"
    assert dag_kwargs["master_model"] == "deepseek-chat"
    assert dag_kwargs["master_reasoning_effort"] == "high"
    assert dag_kwargs["agent_timeout"] == 30.0
    assert dag_kwargs["master_timeout"] == 60.0
