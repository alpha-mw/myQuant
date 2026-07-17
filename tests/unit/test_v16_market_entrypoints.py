from __future__ import annotations

from quant_investor.market import analyze
from quant_investor.market import run_pipeline


def test_market_analyze_v16_returns_pending_without_v15_persistence(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return {
            "decision_protocol": "v16",
            "status": "pending_codex_stage1",
            "v16_stage1": {
                "status": "pending_codex_stage1",
                "run_id": "run-1",
                "review": {
                    "state": "S1_PREPARED",
                    "stage1_request_path": "stage1/request.prepared.json",
                },
            },
            "formal_shortlist_generated": False,
            "new_risk_authorized": False,
        }

    monkeypatch.setattr(analyze, "execute_market_dag", fake_execute)
    monkeypatch.setattr(
        analyze,
        "persist_market_analysis_outputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("v15 persistence must not run")),
    )
    result = analyze.run_market_analysis(
        market="CN",
        mode="batch",
        decision_protocol="v16",
        data_snapshot={"local_latest_trade_date": "20260717"},
        verbose=False,
    )

    assert captured["decision_protocol"] == "v16"
    assert result["v16_stage1"]["run_id"] == "run-1"
    assert result["results"] == []
    assert result["analysis_meta"]["new_risk_authorized"] is False


def test_market_run_v16_forwards_protocol_and_returns_pending(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_pipeline,
        "_run_download_stage",
        lambda **kwargs: (
            {"data_snapshot": {"local_latest_trade_date": "20260717"}},
            0.0,
        ),
    )

    def fake_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "results": [],
            "reports": {"stage1_request": "stage1/request.prepared.json"},
            "analysis_meta": {"status": "pending_codex_stage1"},
            "runtime_profile": {},
            "v16_stage1": {"run_id": "run-2", "status": "pending_codex_stage1"},
        }

    monkeypatch.setattr(run_pipeline, "run_market_analysis", fake_analysis)
    result = run_pipeline.run_unified_pipeline(
        market="CN",
        decision_protocol="v16",
        verbose=False,
    )

    assert captured["decision_protocol"] == "v16"
    assert result["status"] == "pending_codex_stage1"
    assert result["v16_stage1"]["run_id"] == "run-2"
