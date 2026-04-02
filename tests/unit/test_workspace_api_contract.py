from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import quant_investor.pipeline as pipeline_module
from web.services.research_runner import job_manager
from web.services.run_history_store import history_store
from web.workspace_app import app


class _FakeLLMUsage:
    total_calls = 2
    total_prompt_tokens = 128
    total_completion_tokens = 64
    estimated_cost_usd = 0.0123


class _FakeResult:
    final_report = "# Research Report"
    layer_timings = {"data": 0.12, "review": 0.34}
    execution_log = ["data ready", "report ready"]
    llm_usage_summary = _FakeLLMUsage()
    final_strategy = None
    master_review_output = None


def _make_payload(**overrides):
    payload = {
        "stock_pool": ["000001.SZ", "600519.SH"],
        "market": "CN",
        "capital": 1_000_000,
        "risk_level": "中等",
        "lookback_years": 1,
        "kline_backend": "hybrid",
        "enable_macro": True,
        "enable_quant": True,
        "enable_kline": True,
        "enable_fundamental": True,
        "enable_intelligence": True,
        "enable_agent_layer": True,
        "agent_model": "",
        "master_model": "",
        "agent_timeout": 15,
        "master_timeout": 30,
    }
    payload.update(overrides)
    return payload


def _wait_for_terminal_state(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        response = client.get(f"/api/research/{job_id}")
        response.raise_for_status()
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state in time")


@pytest.fixture
def workspace_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(history_store, "_db_path", tmp_path / "web_runs.db")
    monkeypatch.setattr(history_store, "_local", threading.local())

    with job_manager._lock:
        job_manager._jobs.clear()

    with TestClient(app) as client:
        yield client

    with job_manager._lock:
        job_manager._jobs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Original contract tests
# ─────────────────────────────────────────────────────────────────────────────

def test_research_run_request_rejects_empty_stock_pool_and_invalid_market(workspace_client: TestClient):
    empty_pool = workspace_client.post("/api/research/run", json=_make_payload(stock_pool=[]))
    invalid_market = workspace_client.post("/api/research/run", json=_make_payload(market="HK"))

    assert empty_pool.status_code == 422
    assert invalid_market.status_code == 422


def test_completed_run_persists_without_sse_consumer(workspace_client: TestClient, monkeypatch):
    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return _FakeResult()

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    response = workspace_client.post(
        "/api/research/run",
        json=_make_payload(preset_id="preset-123"),
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]

    terminal = _wait_for_terminal_state(workspace_client, job_id)

    assert terminal["status"] == "completed"
    assert terminal["error"] is None
    assert terminal["result_summary"]["stock_pool"] == ["000001.SZ", "600519.SH"]
    assert terminal["result_summary"]["market"] == "CN"

    with job_manager._lock:
        job_manager._jobs.clear()

    fallback_job = workspace_client.get(f"/api/research/{job_id}")
    fallback_report = workspace_client.get(f"/api/research/{job_id}/report")
    history = workspace_client.get("/api/research/history/list?page=1&per_page=10")

    assert fallback_job.status_code == 200
    assert fallback_job.json()["status"] == "completed"
    assert fallback_job.json()["result_summary"]["total_time"] is not None

    assert fallback_report.status_code == 200
    assert fallback_report.json()["markdown"] == "# Research Report"

    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["status"] == "completed"
    assert history.json()["items"][0]["preset_id"] == "preset-123"

    stored_run = history_store.get_run(job_id)
    assert stored_run is not None
    assert stored_run["status"] == "completed"
    assert stored_run["error"] == ""


def test_failed_run_fallback_exposes_error_and_summary_without_sse(workspace_client: TestClient, monkeypatch):
    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    response = workspace_client.post(
        "/api/research/run",
        json=_make_payload(stock_pool=["AAPL"], market="US"),
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]

    terminal = _wait_for_terminal_state(workspace_client, job_id)

    assert terminal["status"] == "failed"
    assert terminal["error"] == "simulated failure"
    assert terminal["result_summary"]["market"] == "US"
    assert terminal["result_summary"]["stock_pool"] == ["AAPL"]
    assert terminal["result_summary"]["total_time"] is not None

    with job_manager._lock:
        job_manager._jobs.clear()

    fallback_job = workspace_client.get(f"/api/research/{job_id}")
    history = workspace_client.get("/api/research/history/list?page=1&per_page=10")

    assert fallback_job.status_code == 200
    assert fallback_job.json()["status"] == "failed"
    assert fallback_job.json()["error"] == "simulated failure"
    assert fallback_job.json()["progress_pct"] == 1.0
    assert fallback_job.json()["result_summary"]["market"] == "US"
    assert fallback_job.json()["result_summary"]["stock_pool"] == ["AAPL"]

    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["status"] == "failed"

    stored_run = history_store.get_run(job_id)
    assert stored_run is not None
    assert stored_run["status"] == "failed"
    assert stored_run["error"] == "simulated failure"


# ─────────────────────────────────────────────────────────────────────────────
# New contract tests
# ─────────────────────────────────────────────────────────────────────────────

def test_universe_presets_cn_includes_all_a(workspace_client: TestClient):
    resp = workspace_client.get("/api/universe/CN/presets")
    assert resp.status_code == 200
    keys = [p["key"] for p in resp.json()["presets"]]
    assert "all_a" in keys
    label = next(p["label"] for p in resp.json()["presets"] if p["key"] == "all_a")
    assert "A股" in label or "全部" in label


def test_universe_resolve_cn_falls_back_to_local_snapshot_when_tushare_unavailable(
    workspace_client: TestClient,
    monkeypatch,
):
    from quant_investor.data._tushare_client import TushareClientPool

    def _raise_query(self, api_name: str, **kwargs):
        raise RuntimeError(f"{api_name} unavailable in test")

    monkeypatch.setattr(TushareClientPool, "_instance", None)
    monkeypatch.setattr(TushareClientPool, "query", _raise_query)

    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={"keys": ["hs300"], "operation": "replace"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 300
    assert body["resolved_keys"] == ["hs300"]
    assert body["selection_meta"]["per_key_counts"] == {"hs300": 300}
    assert "000001.SZ" in body["symbols"]
    assert "600519.SH" in body["symbols"]


def test_universe_resolve_cn_uses_latest_tushare_trade_date(
    workspace_client: TestClient,
    monkeypatch,
):
    from quant_investor.data._tushare_client import TushareClientPool

    def _fake_query(self, api_name: str, **kwargs):
        assert api_name == "index_weight"
        return pd.DataFrame(
            [
                {"trade_date": "20240131", "con_code": "000001.SZ"},
                {"trade_date": "20240131", "con_code": "000002.SZ"},
                {"trade_date": "20231229", "con_code": "600519.SH"},
            ]
        )

    monkeypatch.setattr(TushareClientPool, "_instance", None)
    monkeypatch.setattr(TushareClientPool, "query", _fake_query)

    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={"keys": ["hs300"], "operation": "replace"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["symbols"] == ["000001.SZ", "000002.SZ"]
    assert body["selection_meta"]["per_key_counts"] == {"hs300": 2}


def test_universe_resolve_dedupes_and_sorts(workspace_client: TestClient, monkeypatch):
    """POST /api/universe/CN/resolve with two keys that share symbols deduplicates correctly."""
    from web.routers import universe as universe_mod

    async def _fake_cn(key: str):
        mapping = {
            "hs300": ["000001.SZ", "600519.SH", "000002.SZ"],
            "zz500": ["600519.SH", "000003.SZ", "000001.SZ"],
        }
        return mapping.get(key)

    monkeypatch.setattr(universe_mod, "_fetch_cn_symbols", _fake_cn)

    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={"keys": ["hs300", "zz500"], "operation": "replace"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert body["symbols"] == sorted({"000001.SZ", "600519.SH", "000002.SZ", "000003.SZ"})
    assert set(body["resolved_keys"]) == {"hs300", "zz500"}
    assert body["selection_meta"]["total_before_dedupe"] == 6
    assert body["selection_meta"]["total_after_dedupe"] == 4


def test_universe_resolve_merge_unions_with_existing_pool(workspace_client: TestClient, monkeypatch):
    from web.routers import universe as universe_mod

    async def _fake_cn(key: str):
        if key == "hs300":
            return ["000001.SZ", "600519.SH"]
        return None

    monkeypatch.setattr(universe_mod, "_fetch_cn_symbols", _fake_cn)

    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={
            "keys": ["hs300"],
            "operation": "merge",
            "existing_pool": ["000001.SZ", "EXTRA.SZ"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "EXTRA.SZ" in body["symbols"]
    assert "000001.SZ" in body["symbols"]
    assert body["selection_meta"]["operation"] == "merge"
    assert body["selection_meta"]["merged_from_existing"] == 2


def test_universe_resolve_unknown_key_returns_404(workspace_client: TestClient, monkeypatch):
    from web.routers import universe as universe_mod

    async def _fake_cn(_key: str):
        return None

    monkeypatch.setattr(universe_mod, "_fetch_cn_symbols", _fake_cn)

    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={"keys": ["nonexistent_key"]},
    )
    assert resp.status_code == 404


def test_universe_resolve_rejects_empty_keys(workspace_client: TestClient):
    resp = workspace_client.post(
        "/api/universe/CN/resolve",
        json={"keys": []},
    )
    assert resp.status_code == 422


def test_run_request_accepts_stock_input_mode_fields(workspace_client: TestClient, monkeypatch):
    """ResearchRunRequest now accepts stock_input_mode, universe_keys, universe_operation."""

    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return _FakeResult()

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    payload = _make_payload(
        stock_input_mode="multi",
        universe_keys=["hs300", "zz500"],
        universe_operation="replace",
    )
    resp = workspace_client.post("/api/research/run", json=payload)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    terminal = _wait_for_terminal_state(workspace_client, job_id)
    assert terminal["status"] == "completed"

    stored = history_store.get_run(job_id)
    assert stored is not None
    meta = json.loads(stored["selection_meta_json"])
    assert meta["stock_input_mode"] == "multi"
    assert meta["universe_keys"] == ["hs300", "zz500"]
    assert meta["universe_operation"] == "replace"


def test_completed_run_persists_trade_records(workspace_client: TestClient, monkeypatch, tmp_path):
    """Completed runs with agent-layer top_picks auto-save trade records."""
    from types import SimpleNamespace

    actionable_pick = SimpleNamespace(symbol="000001.SZ", action="buy", rationale="strong momentum")
    hold_pick = SimpleNamespace(symbol="600519.SH", action="hold", rationale="wait")

    class FakeMasterReviewOutput:
        final_conviction = "buy"
        top_picks = [actionable_pick, hold_pick]
        conviction_drivers = ["driver one", "driver two"]

    class FakeResultWithTrades:
        final_report = "# Report"
        layer_timings = {}
        execution_log = []
        llm_usage_summary = _FakeLLMUsage()
        final_strategy = None
        master_review_output = FakeMasterReviewOutput()

    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return FakeResultWithTrades()

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    resp = workspace_client.post("/api/research/run", json=_make_payload())
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    _wait_for_terminal_state(workspace_client, job_id)

    trades = history_store.get_recent_trade_records(limit=10)
    assert any(t["symbol"] == "000001.SZ" and t["job_id"] == job_id for t in trades)
    assert all(t["symbol"] != "600519.SH" for t in trades)
    assert all(t["outcome_status"] == "pending" for t in trades)
    assert all(t["status"] == "suggested" for t in trades)
    workspace_learning_dir = tmp_path / "data" / "workspace_learning"
    saved_files = list(workspace_learning_dir.glob(f"{job_id}_*.json"))
    assert len(saved_files) == 1


def test_startup_context_returns_structured_response(workspace_client: TestClient, monkeypatch):
    """GET /api/research/startup-context returns recent_runs, suggested_trades, recall_summary."""

    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return _FakeResult()

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    # Run something so there is history
    resp = workspace_client.post("/api/research/run", json=_make_payload())
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    _wait_for_terminal_state(workspace_client, job_id)

    # Clear in-memory jobs so the store is authoritative
    with job_manager._lock:
        job_manager._jobs.clear()

    ctx_resp = workspace_client.get("/api/research/startup-context")
    assert ctx_resp.status_code == 200
    body = ctx_resp.json()
    assert "recent_runs" in body
    assert "suggested_trades" in body
    assert "recall_summary" in body
    assert isinstance(body["recent_runs"], list)
    assert body["recall_summary"]["run_count"] >= 1


def test_delete_run_cleans_trade_records_and_workspace_learning(workspace_client: TestClient, monkeypatch, tmp_path):
    from types import SimpleNamespace

    pick = SimpleNamespace(symbol="000001.SZ", action="buy", rationale="cleanup target")

    class FakeMasterReviewOutput:
        final_conviction = "buy"
        top_picks = [pick]
        conviction_drivers = ["cleanup driver"]

    class FakeResultWithTrades:
        final_report = "# Report"
        layer_timings = {}
        execution_log = []
        llm_usage_summary = _FakeLLMUsage()
        final_strategy = None
        master_review_output = FakeMasterReviewOutput()

    class FakeQuantInvestor:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return FakeResultWithTrades()

    monkeypatch.setattr(pipeline_module, "QuantInvestor", FakeQuantInvestor)

    resp = workspace_client.post("/api/research/run", json=_make_payload())
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    _wait_for_terminal_state(workspace_client, job_id)

    workspace_learning_dir = tmp_path / "data" / "workspace_learning"
    assert list(workspace_learning_dir.glob(f"{job_id}_*.json"))
    assert any(trade["job_id"] == job_id for trade in history_store.get_recent_trade_records(limit=10))

    delete_resp = workspace_client.delete(f"/api/research/{job_id}")
    assert delete_resp.status_code == 200
    assert history_store.get_run(job_id) is None
    assert not list(workspace_learning_dir.glob(f"{job_id}_*.json"))
    assert all(trade["job_id"] != job_id for trade in history_store.get_recent_trade_records(limit=10))

    startup_resp = workspace_client.get("/api/research/startup-context")
    assert startup_resp.status_code == 200
    assert all(trade["job_id"] != job_id for trade in startup_resp.json()["suggested_trades"])


def test_settings_patch_persists_kimi_api_key(workspace_client: TestClient, tmp_path, monkeypatch):
    """PATCH /api/settings/ with kimi_api_key writes to .env and os.environ."""
    import os
    from web.routers import settings as settings_mod

    env_file = tmp_path / ".env"
    monkeypatch.setattr(settings_mod, "_ENV_PATH", env_file)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    resp = workspace_client.patch(
        "/api/settings/",
        json={"kimi_api_key": "test-kimi-key-abc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "KIMI_API_KEY" in body["updated"]

    # os.environ updated
    assert os.environ.get("KIMI_API_KEY") == "test-kimi-key-abc"

    # .env written
    assert env_file.exists()
    content = env_file.read_text()
    assert "KIMI_API_KEY=test-kimi-key-abc" in content


def test_settings_patch_updates_existing_env_line(workspace_client: TestClient, tmp_path, monkeypatch):
    """PATCH /api/settings/ updates an existing key in .env without duplicating it."""
    import os
    from web.routers import settings as settings_mod

    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nOPENAI_API_KEY=old-value\nOTHER=x\n", encoding="utf-8")
    monkeypatch.setattr(settings_mod, "_ENV_PATH", env_file)

    workspace_client.patch("/api/settings/", json={"openai_api_key": "new-value"})

    content = env_file.read_text()
    assert content.count("OPENAI_API_KEY=") == 1
    assert "OPENAI_API_KEY=new-value" in content
    assert "OTHER=x" in content


def test_settings_get_masks_credentials(workspace_client: TestClient, monkeypatch):
    """GET /api/settings/ returns masked values for set keys."""
    import os

    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdefghijklmnop")

    resp = workspace_client.get("/api/settings/")
    assert resp.status_code == 200
    creds = {c["env_key"]: c for c in resp.json()["credentials"]}
    openai_cred = creds["OPENAI_API_KEY"]
    assert openai_cred["is_set"] is True
    assert "sk-a" in openai_cred["masked_value"]
    assert "mnop" in openai_cred["masked_value"]
    assert "abcdefghijkl" not in openai_cred["masked_value"]
    # Kimi key should appear in the registry
    assert "KIMI_API_KEY" in creds


def test_settings_get_includes_database_summaries(workspace_client: TestClient, tmp_path, monkeypatch):
    stock_db_path = tmp_path / "stock_database.db"
    stock_db_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("DB_PATH", str(stock_db_path))

    created_at = "2026-03-29T00:00:00+00:00"
    history_store.save_run(
        job_id="job-db-1",
        created_at=created_at,
        status="completed",
        request_json="{}",
        stock_pool='["AAPL"]',
    )
    history_store.save_trade_records(
        "job-db-1",
        [{"trade_id": "trade-1", "symbol": "AAPL", "direction": "buy", "rationale": "test"}],
    )

    resp = workspace_client.get("/api/settings/")
    assert resp.status_code == 200

    body = resp.json()
    assert body["db_path"] == str(stock_db_path)
    assert body["stock_db"]["exists"] is True
    assert body["stock_db"]["path"] == str(stock_db_path.resolve())
    assert body["workspace_db"]["exists"] is True
    assert body["workspace_db"]["run_count"] == 1
    assert body["workspace_db"]["completed_runs"] == 1
    assert body["workspace_db"]["failed_runs"] == 0
    assert body["workspace_db"]["pending_trades"] == 1
    assert body["workspace_db"]["last_run_at"] == created_at
    assert body["workspace_db"]["path"].endswith("web_runs.db")


def test_sse_progress_event_includes_phase_fields(workspace_client: TestClient, monkeypatch):
    """SSE progress events contain phase_key and phase_label alongside progress_pct."""
    from web.services import research_runner as runner_mod

    # Verify the runner emits the richer progress payload
    job_class = runner_mod.ResearchJob
    assert hasattr(job_class, "__dataclass_fields__")
    assert "phase_key" in job_class.__dataclass_fields__
    assert "phase_label" in job_class.__dataclass_fields__

    # Verify _estimate_phase returns a 3-tuple
    result = runner_mod._estimate_phase("starting kline forecast")
    assert result is not None
    pct, key, label = result
    assert 0 < pct <= 1.0
    assert key == "kline"
    assert label  # non-empty


def test_stream_endpoint_uses_native_sse_frames(workspace_client: TestClient, monkeypatch):
    """GET /api/research/{job_id}/stream returns standard SSE without extra deps."""
    from types import SimpleNamespace

    async def fake_stream_logs(_job_id: str):
        yield {"event": "log", "data": "hello"}
        yield {"event": "progress", "data": '{"progress_pct": 0.5}'}
        yield {"event": "completed", "data": '{"status": "completed"}'}

    monkeypatch.setattr(
        job_manager,
        "get_job",
        lambda _job_id: SimpleNamespace(job_id="job-1"),
    )
    monkeypatch.setattr(job_manager, "stream_logs", fake_stream_logs)

    response = workspace_client.get("/api/research/job-1/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: log" in response.text
    assert "data: hello" in response.text
    assert 'event: progress' in response.text
    assert 'data: {"progress_pct": 0.5}' in response.text


def test_recall_context_regression_deterministic_chain(workspace_client: TestClient, monkeypatch):
    """
    Regression: recall_context on agent contracts is advisory-only.
    BaseBranchAgentInput and MasterAgentInput both accept it with empty default,
    confirming no hard dependency on its presence.
    """
    from quant_investor.agents.agent_contracts import BaseBranchAgentInput, MasterAgentInput

    # Instantiation without recall_context should work (default empty)
    branch_input = BaseBranchAgentInput(
        branch_name="test",
        base_score=0.1,
        final_score=0.1,
        confidence=0.5,
    )
    assert branch_input.recall_context == {}

    master_input = MasterAgentInput()
    assert master_input.recall_context == {}

    # Instantiation with recall_context should also work
    branch_with_ctx = BaseBranchAgentInput(
        branch_name="test",
        base_score=0.1,
        final_score=0.1,
        confidence=0.5,
        recall_context={"conviction": "buy", "top_picks": [{"symbol": "AAPL"}]},
    )
    assert branch_with_ctx.recall_context["conviction"] == "buy"
