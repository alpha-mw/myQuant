from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)
from web.models.analysis_models import AnalysisRunRequest
from web.services.analysis_service import (
    BRANCH_ORDER,
    _normalize_request_payload,
    get_analysis_options,
)
from web.tasks.run_analysis_job import _risk_summary, run_job


def _canonical_pipeline_branches():
    return {
        branch_name: SimpleNamespace(
            branch_name=branch_name,
            symbol_scores={},
            score=0.0,
            confidence=0.5,
            explanation=f"{branch_name} neutral",
            risks=[],
            signals={},
            metadata={},
        )
        for branch_name in ("quant", "fundamental", "macro")
    }


def test_analysis_request_preserves_non_intelligence_web_branch_contracts():
    assert BRANCH_ORDER == [
        "kline",
        "quant",
        "fundamental",
        "llm_debate",
        "macro",
    ]
    request = AnalysisRunRequest.model_validate(
        {
            "targets": ["000001.SZ"],
            "branches": {
                "kline": {"enabled": True},
                "kronos": {"enabled": False},
                "llm_debate": {"enabled": False},
            },
        }
    )
    assert set(request.branches) == {"kline", "kronos", "llm_debate"}

    canonical_request = AnalysisRunRequest.model_validate(
        {
            "targets": ["000001.SZ"],
            "branches": {"quant": {"settings": {"factor_pack": "core"}}},
        }
    )
    assert "enabled" not in canonical_request.model_dump()["branches"]["quant"]

    normalized = _normalize_request_payload(
        {
            "targets": ["000001.SZ"],
            "branches": {
                "kronos": {
                    "enabled": False,
                    "settings": {"backend": "hybrid"},
                },
                "llm_debate": {"enabled": False},
            },
        }
    )
    assert "kronos" not in normalized["branches"]
    assert normalized["branches"]["kline"]["enabled"] is False
    assert normalized["branches"]["kline"]["settings"]["backend"] == "hybrid"
    assert normalized["branches"]["llm_debate"]["enabled"] is False
    for branch_name in ("quant", "fundamental", "macro"):
        assert "enabled" not in normalized["branches"][branch_name]

    for branch_name in ("quant", "fundamental", "macro"):
        with pytest.raises(ValueError, match=rf"branches\.{branch_name}\.enabled"):
            AnalysisRunRequest.model_validate(
                {
                    "targets": ["000001.SZ"],
                    "branches": {branch_name: {"enabled": False}},
                }
            )

    with pytest.raises(ValueError, match="enable_macro"):
        AnalysisRunRequest.model_validate(
            {"targets": ["000001.SZ"], "enable_macro": False}
        )

    options = get_analysis_options()
    for branch_name in ("quant", "fundamental", "macro"):
        assert "enabled" not in options["branch_defaults"][branch_name]
        for preset in options["presets"]:
            assert "enabled" not in preset["defaults"]["branches"][branch_name]

    with pytest.raises(ValueError, match="Intelligence.*branches.intelligence"):
        AnalysisRunRequest.model_validate(
            {
                "targets": ["000001.SZ"],
                "branches": {"intelligence": {"enabled": True}},
            }
        )
    with pytest.raises(ValueError, match="enable_intelligence"):
        AnalysisRunRequest.model_validate(
            {
                "targets": ["000001.SZ"],
                "enable_intelligence": True,
            }
        )
    with pytest.raises(ValueError, match="enable_intelligence"):
        _normalize_request_payload(
            {
                "targets": ["000001.SZ"],
                "enable_intelligence": True,
            }
        )


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
                architecture_version=ARCHITECTURE_VERSION,
                branch_schema_version=BRANCH_SCHEMA_VERSION,
                likelihood_schema_version=LIKELIHOOD_SCHEMA_VERSION,
                report_protocol_version=REPORT_PROTOCOL_VERSION,
                total_time=1.2,
                final_strategy=strategy,
                branch_results=_canonical_pipeline_branches(),
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
                "quant": {"settings": {}},
                "fundamental": {"settings": {}},
                "macro": {"settings": {}},
            },
            "kline_backend": "hybrid",
            "risk": {"capital": 1_000_000, "risk_level": "中等"},
            "portfolio": {"candidate_limit": 5},
            "llm_debate": {"enabled": False, "assignments": []},
        }
    )

    assert captured["kwargs"]["stock_pool"] == ["000001.SZ"]
    assert captured["kwargs"]["kline_backend"] == "hybrid"
    assert captured["kwargs"]["enable_quant"] is True
    assert captured["kwargs"]["enable_fundamental"] is True
    assert captured["kwargs"]["enable_macro"] is True
    assert result["candidate_symbols"] == ["000001.SZ"]
    assert result["report_markdown"] == "# report"
    assert [branch["branch_name"] for branch in result["branches"]] == BRANCH_ORDER
    assert all(
        branch["enabled"] is True
        for branch in result["branches"]
        if branch["branch_name"] in {"quant", "fundamental", "macro"}
    )
    assert {
        key: result[key]
        for key in (
            "architecture_version",
            "branch_schema_version",
            "likelihood_schema_version",
            "report_protocol_version",
        )
    } == {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
    }


def test_run_job_rejects_removed_intelligence_contract():
    with pytest.raises(ValueError, match="enable_intelligence"):
        run_job(
            {
                "targets": ["000001.SZ"],
                "enable_intelligence": True,
            }
        )

    with pytest.raises(ValueError, match="Intelligence.*branches.intelligence"):
        run_job(
            {
                "targets": ["000001.SZ"],
                "branches": {"intelligence": {"enabled": True}},
            }
        )


def test_run_job_rejects_missing_canonical_pipeline_branch(monkeypatch):
    class FakeQuantInvestor:
        def __init__(self, **kwargs):
            pass

        def run(self):
            branches = _canonical_pipeline_branches()
            branches.pop("macro")
            return SimpleNamespace(
                architecture_version=ARCHITECTURE_VERSION,
                branch_schema_version=BRANCH_SCHEMA_VERSION,
                likelihood_schema_version=LIKELIHOOD_SCHEMA_VERSION,
                report_protocol_version=REPORT_PROTOCOL_VERSION,
                branch_results=branches,
            )

    monkeypatch.setattr("quant_investor.pipeline.QuantInvestor", FakeQuantInvestor)

    with pytest.raises(ValueError, match="missing branches: macro"):
        run_job({"targets": ["000001.SZ"], "market": "CN", "mode": "single"})


@pytest.mark.parametrize(
    "retired_key",
    ["intelligence_weight", "INTELLIGENCE_MODE", "IntelligenceOverlay"],
)
def test_analysis_request_recursively_rejects_intelligence_named_keys(retired_key):
    payload = {
        "targets": ["000001.SZ"],
        "branches": {
            "quant": {
                "settings": {
                    "nested": [{retired_key: 0.2}],
                }
            }
        },
    }

    with pytest.raises(ValueError, match=rf"Intelligence.*{retired_key}"):
        AnalysisRunRequest.model_validate(payload)
    with pytest.raises(ValueError, match=rf"Intelligence.*{retired_key}"):
        _normalize_request_payload(payload)
    with pytest.raises(ValueError, match=rf"Intelligence.*{retired_key}"):
        run_job(payload)


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
                architecture_version=ARCHITECTURE_VERSION,
                branch_schema_version=BRANCH_SCHEMA_VERSION,
                likelihood_schema_version=LIKELIHOOD_SCHEMA_VERSION,
                report_protocol_version=REPORT_PROTOCOL_VERSION,
                total_time=0.1,
                final_strategy=strategy,
                branch_results=_canonical_pipeline_branches(),
                risk_results=None,
                final_report="",
                execution_log=[],
            )

    monkeypatch.setattr("quant_investor.pipeline.QuantInvestor", FakeQuantInvestor)

    result = run_job({"targets": [], "market": "CN", "mode": "single"})
    assert result["candidate_symbols"] == []
    assert result["trade_recommendations"] == []
