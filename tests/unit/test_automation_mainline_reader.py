from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.automation import analysis_runner, daily_runner
from quant_investor.mainline import MainlineError


def test_analysis_runner_reads_only_the_active_mainline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_read(workspace_root: Path, *, strategy_id: str) -> dict[str, object]:
        calls.append((workspace_root, strategy_id))
        return {
            "status": "ACTIVE",
            "active_generation_id": "a" * 64,
            "mainline_state": "READY",
            "investment_state": "READY",
            "blockers": [],
            "result": {"decision": "HOLD"},
        }

    monkeypatch.setattr(analysis_runner, "read_public_run", fake_read)
    result = analysis_runner.AnalysisRunner().run(
        {"market": "CN", "strategy_id": "registered-strategy"},
        recall_context={"untrusted": "ignored"},
    )

    assert calls == [(analysis_runner.PROJECT_ROOT, "registered-strategy")]
    assert result["active_generation_id"] == "a" * 64


def test_analysis_runner_rejects_non_cn_market() -> None:
    with pytest.raises(MainlineError) as exc_info:
        analysis_runner.AnalysisRunner().run({"market": "US"})
    assert exc_info.value.public_fields["blockers"] == ["MARKET_UNSUPPORTED"]


def test_daily_runner_emits_generation_bound_result_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeHistoryLoader:
        def load_recent(self, **kwargs: object) -> list[dict[str, object]]:
            assert kwargs["strategy"] == "registered-strategy"
            return []

        def build_recall_context(
            self, runs: list[dict[str, object]], *, market: str
        ) -> dict[str, object]:
            assert runs == []
            assert market == "CN"
            return {"window_dates": []}

    class FakeAnalysisRunner:
        def run(self, config: dict[str, object], **kwargs: object) -> dict[str, object]:
            assert config["market"] == "CN"
            assert kwargs["recall_context"] == {"window_dates": []}
            return {
                "status": "ACTIVE",
                "active_generation_id": "b" * 64,
                "mainline_state": "READY",
                "investment_state": "READY",
                "blockers": [],
                "result": None,
            }

    monkeypatch.setattr(daily_runner, "HistoryLoader", FakeHistoryLoader)
    monkeypatch.setattr(daily_runner, "AnalysisRunner", FakeAnalysisRunner)
    result = daily_runner.run_once(
        {
            "market": "CN",
            "history_strategy": "registered-strategy",
            "review_model_priority": [],
        }
    )

    assert result["active_generation_id"] == "b" * 64
    assert capsys.readouterr().out == (
        '{"active_generation_id":"'
        + "b" * 64
        + '","blockers":[],"investment_state":"READY",'
        '"mainline_state":"READY","result":null,"status":"ACTIVE"}\n'
    )


def test_automation_sources_have_no_pipeline_or_legacy_history_fallback() -> None:
    source = Path(analysis_runner.__file__).read_text(encoding="utf-8")
    daily_source = Path(daily_runner.__file__).read_text(encoding="utf-8")
    combined = source + daily_source
    assert "run_unified_pipeline" not in combined
    assert "_run_automation_data_update_preflight" not in combined
    assert "automation.history_loader" not in combined
    assert "PersistenceManager" not in daily_source
    assert "ReportBuilder" not in daily_source
