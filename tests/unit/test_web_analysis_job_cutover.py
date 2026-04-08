from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from web.tasks.run_analysis_job import _risk_summary, run_job


def test_run_job_uses_public_quant_investor_entrypoint(monkeypatch):
    captured: dict[str, object] = {}

    class FakeQuantInvestor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self):
            strategy = SimpleNamespace(
                research_mode="production",
                target_exposure=0.35,
                style_bias="均衡",
                sector_preferences=[],
                candidate_symbols=["000001.SZ"],
                execution_notes=["dag path"],
                trade_recommendations=[
                    SimpleNamespace(
                        symbol="000001.SZ",
                        action="buy",
                        current_price=10.0,
                        recommended_entry_price=10.0,
                        target_price=11.0,
                        stop_loss_price=9.0,
                        suggested_weight=0.2,
                        suggested_amount=200000.0,
                        suggested_shares=20000,
                        confidence=0.8,
                        consensus_score=0.6,
                        branch_positive_count=3,
                        trend_regime="震荡",
                        risk_flags=[],
                    )
                ],
            )
            return SimpleNamespace(
                total_time=1.2,
                final_strategy=strategy,
                branch_results={},
                risk_results=None,
                final_report="# report",
                execution_log=["done"],
            )

    monkeypatch.setattr("quant_investor.pipeline.QuantInvestor", FakeQuantInvestor)

    result = run_job(
        {
            "targets": ["000001.SZ"],
            "stocks": ["000001.SZ"],
            "market": "CN",
            "mode": "single",
            "branches": {
                "kline": {"enabled": True, "settings": {"backend": "hybrid"}},
                "quant": {"enabled": True, "settings": {}},
                "llm_debate": {"enabled": False, "settings": {}},
                "intelligence": {"enabled": True, "settings": {}},
                "macro": {"enabled": True, "settings": {}},
            },
            "risk": {"capital": 1_000_000, "risk_level": "中等"},
            "portfolio": {"candidate_limit": 5},
            "llm_debate": {"enabled": False, "assignments": []},
        }
    )

    assert captured["kwargs"]["stock_pool"] == ["000001.SZ"]
    assert captured["kwargs"]["kline_backend"] == "hybrid"
    assert result["candidate_symbols"] == ["000001.SZ"]
    assert result["report_markdown"] == "# report"


def test_risk_summary_returns_defaults_when_risk_results_is_none():
    result = SimpleNamespace(risk_results=None)
    request = {"risk": {"capital": 500_000, "max_single_position": 0.15}}
    summary = _risk_summary(result, request)

    assert summary["risk_level"] == "unknown"
    assert summary["volatility"] == 0.0
    assert summary["max_single_position"] == 0.15
    assert summary["warnings"] == []


def test_risk_summary_extracts_populated_risk_results():
    risk_result = SimpleNamespace(
        risk_level="中等",
        risk_metrics=SimpleNamespace(volatility=0.22, max_drawdown=0.18, sharpe_ratio=1.3),
        risk_warnings=["流动性偏低"],
    )
    result = SimpleNamespace(risk_results=risk_result)
    request = {"risk": {}}
    summary = _risk_summary(result, request)

    assert summary["risk_level"] == "中等"
    assert summary["volatility"] == pytest.approx(0.22)
    assert summary["warnings"] == ["流动性偏低"]
    assert summary["stress_test"] == "流动性偏低"


def test_run_analysis_subprocess_timeout_raises(monkeypatch, tmp_path):
    import web.services.analysis_service as svc

    monkeypatch.setattr(svc, "WEB_ANALYSIS_DIR", str(tmp_path))
    monkeypatch.setattr(svc, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(svc, "_ensure_results_dir", lambda: None)
    monkeypatch.setattr(svc, "_analysis_python", lambda: "python3")
    monkeypatch.setattr(svc, "_result_file_for", lambda aid: tmp_path / f"{aid}.json")

    def _timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python3", timeout=60)

    monkeypatch.setattr(subprocess, "run", _timeout_run)

    with pytest.raises(RuntimeError, match="分析超时"):
        svc.run_analysis(
            {"targets": ["000001.SZ"], "market": "CN", "mode": "single"}
        )


def test_run_job_empty_targets_still_runs(monkeypatch):
    class FakeQuantInvestor:
        def __init__(self, **kwargs):
            pass

        def run(self):
            strategy = SimpleNamespace(
                research_mode="production",
                target_exposure=0.0,
                style_bias="均衡",
                sector_preferences=[],
                candidate_symbols=[],
                execution_notes=[],
                trade_recommendations=[],
            )
            return SimpleNamespace(
                total_time=0.1,
                final_strategy=strategy,
                branch_results={},
                risk_results=None,
                final_report="",
                execution_log=[],
            )

    monkeypatch.setattr("quant_investor.pipeline.QuantInvestor", FakeQuantInvestor)

    result = run_job({"targets": [], "market": "CN", "mode": "single"})
    assert result["candidate_symbols"] == []
    assert result["trade_recommendations"] == []

