from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app
from web.services.analysis_service import _normalize_web_result
from web.services import analysis_service, data_service, portfolio_service, settings_service


def _make_frontend_dist(root: Path) -> Path:
    dist = root / "dist"
    assets_dir = dist / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>myQuant test shell</div></body></html>",
        encoding="utf-8",
    )
    (dist / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok')", encoding="utf-8")
    return dist


def test_api_endpoints_smoke(tmp_path, monkeypatch):
    stock_db = tmp_path / "stock.db"
    conn = sqlite3.connect(stock_db)
    conn.execute(
        """
        CREATE TABLE stock_list (
            ts_code TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            market TEXT,
            list_date TEXT,
            is_hs300 INTEGER DEFAULT 0,
            is_zz500 INTEGER DEFAULT 0,
            is_zz1000 INTEGER DEFAULT 0,
            last_update TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE daily_data (
            ts_code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO stock_list
        (ts_code, name, industry, market, list_date, is_hs300, is_zz500, is_zz1000, last_update)
        VALUES ('000001.SZ', '平安银行', '银行', 'CN', '19910403', 1, 0, 0, '2026-03-16T00:00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO daily_data
        (ts_code, trade_date, open, high, low, close, volume, amount)
        VALUES ('000001.SZ', '20260312', 10, 11, 9, 10.5, 1000, 10500)
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(data_service, "STOCK_DB_PATH", str(stock_db))
    monkeypatch.setattr(data_service, "_ensure_missing_stock_data_downloaded", lambda *args, **kwargs: None)

    app = create_app(frontend_dist=_make_frontend_dist(tmp_path))
    client = TestClient(app)

    health = client.get("/api/v1/health")
    settings = client.get("/api/v1/settings")
    statistics = client.get("/api/v1/data/statistics")
    options = client.get("/api/v1/analysis/options")
    jobs = client.get("/api/v1/analysis/jobs?limit=1")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    assert settings.status_code == 200
    assert isinstance(settings.json()["credentials"], list)

    assert statistics.status_code == 200
    assert "total_stocks" in statistics.json()

    assert options.status_code == 200
    assert "presets" in options.json()

    assert jobs.status_code == 200
    assert isinstance(jobs.json(), list)


def test_frontend_deep_links_return_index_html(tmp_path):
    app = create_app(frontend_dist=_make_frontend_dist(tmp_path))
    client = TestClient(app)

    for path in ["/", "/history", "/settings", "/stocks/000001.SZ", "/research?id=test"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "myQuant test shell" in response.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log('ok')" in asset.text

    missing_asset = client.get("/assets/missing.js")
    assert missing_asset.status_code == 404

    missing_api = client.get("/api/v1/missing")
    assert missing_api.status_code == 404
    assert missing_api.json() == {"detail": "Not Found"}


def test_settings_accept_dashscope_and_return_env_key(tmp_path, monkeypatch):
    app = create_app(frontend_dist=_make_frontend_dist(tmp_path))
    client = TestClient(app)
    env_file = tmp_path / ".env"

    monkeypatch.setattr(settings_service, "ENV_FILE", env_file)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    response = client.put("/api/v1/settings", json={"dashscope_api_key": "sk-test-dashscope"})
    assert response.status_code == 200

    data = response.json()
    dashscope = next(item for item in data["credentials"] if item["env_key"] == "DASHSCOPE_API_KEY")

    assert dashscope["is_set"] is True
    assert dashscope["masked_value"]
    assert "DASHSCOPE_API_KEY=sk-test-dashscope" in env_file.read_text(encoding="utf-8")


def test_analysis_branch_payload_is_normalized_when_optional_fields_are_null():
    result = _normalize_web_result(
        {
            "analysis_id": "test",
            "request": {"targets": ["000001.SZ"], "branches": {"kline": {"enabled": True, "settings": None}}},
            "branches": [
                {
                    "branch_name": "kline",
                    "score": 0.1,
                    "confidence": 0.8,
                    "explanation": "ok",
                    "risks": None,
                    "top_symbols": None,
                    "signals": None,
                    "metadata": None,
                }
            ],
        }
    )

    branch = result["branches"][0]
    assert branch["risks"] == []
    assert branch["top_symbols"] == []
    assert branch["signals"] == {}
    assert branch["metadata"] == {}
    assert isinstance(branch["settings"], dict)
    assert branch["settings"]["prediction_horizon"] == "20d"


def test_market_mode_expands_all_downloaded_symbols_from_selected_market(tmp_path, monkeypatch):
    stock_db = tmp_path / "stock.db"
    conn = sqlite3.connect(stock_db)
    conn.execute("CREATE TABLE stock_list (ts_code TEXT, market TEXT)")
    conn.execute("CREATE TABLE daily_data (ts_code TEXT, trade_date TEXT)")
    conn.execute("INSERT INTO stock_list VALUES ('000001.SZ', 'CN')")
    conn.execute("INSERT INTO stock_list VALUES ('600519.SH', 'CN')")
    conn.execute("INSERT INTO stock_list VALUES ('AAPL', 'US')")
    conn.execute("INSERT INTO daily_data VALUES ('000001.SZ', '20260312')")
    conn.execute("INSERT INTO daily_data VALUES ('AAPL', '20260312')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(analysis_service, "STOCK_DB_PATH", str(stock_db))

    normalized = analysis_service._normalize_request_payload({"mode": "market", "market": "CN"})
    assert normalized["mode"] == "market"
    assert normalized["targets"] == ["000001.SZ"]


def test_run_analysis_uses_market_batch_executor_for_market_mode(tmp_path, monkeypatch):
    web_analysis_dir = tmp_path / "web_analysis"
    jobs_dir = web_analysis_dir / "jobs"

    monkeypatch.setattr(analysis_service, "WEB_ANALYSIS_DIR", web_analysis_dir)
    monkeypatch.setattr(analysis_service, "JOB_DIR", jobs_dir)
    monkeypatch.setattr(analysis_service, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(
        analysis_service,
        "_normalize_request_payload",
        lambda payload: {
            "mode": "market",
            "market": "CN",
            "preset": "quick_scan",
            "targets": [f"{index:06d}.SZ" for index in range(120)],
            "stocks": [f"{index:06d}.SZ" for index in range(120)],
            "risk": {"capital": 1_000_000, "risk_level": "中等", "max_single_position": 0.2},
            "portfolio": {"candidate_limit": 8},
            "branches": {"llm_debate": {"enabled": False, "settings": {}}},
            "llm_debate": {"enabled": False, "assignments": []},
        },
    )

    captured: dict[str, object] = {}

    def fake_market_runner(payload, progress_callback=None):
        captured["payload"] = payload
        if progress_callback is not None:
            progress_callback({"status_message": "全市场分批扫描中", "progress": {"current_batch": 1, "total_batches": 2}})
        return {
            "analysis_id": payload["analysis_id"],
            "created_at": "2026-03-17T10:00:00",
            "source": "web",
            "request": payload,
            "target_exposure": 0.35,
            "style_bias": "均衡",
            "candidate_symbols": ["000001.SZ"],
            "execution_notes": [],
            "branches": [],
            "risk": {"risk_level": "中等", "warnings": []},
            "trade_recommendations": [],
            "report_markdown": "ok",
            "execution_log": [],
        }

    monkeypatch.setattr(analysis_service, "_run_market_analysis", fake_market_runner)
    monkeypatch.setattr(analysis_service.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call subprocess for market mode")))
    monkeypatch.setattr(analysis_service, "_normalize_web_result", lambda result: result)
    monkeypatch.setattr(analysis_service, "_save_analysis_session", lambda result: None)

    result = analysis_service.run_analysis({"mode": "market", "market": "CN"})

    assert result["analysis_id"]
    assert captured["payload"]["mode"] == "market"


def test_market_running_job_has_extended_stale_timeout(tmp_path, monkeypatch):
    frontend_dist = _make_frontend_dist(tmp_path)
    results_dir = tmp_path / "results"
    web_analysis_dir = results_dir / "web_analysis"
    jobs_dir = web_analysis_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    created_at = (now - timedelta(minutes=30)).isoformat(timespec="seconds")
    updated_at = (now - timedelta(minutes=15)).isoformat(timespec="seconds")

    monkeypatch.setattr(analysis_service, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_service, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(analysis_service, "WEB_ANALYSIS_DIR", web_analysis_dir)
    monkeypatch.setattr(analysis_service, "JOB_DIR", jobs_dir)

    analysis_service._job_file_for("20260317_100000_000001").write_text(
        json.dumps(
            {
                "ok": True,
                "job_id": "20260317_100000_000001",
                "status": "running",
                "created_at": created_at,
                "updated_at": updated_at,
                "result": None,
                "error": None,
                "mode": "market",
                "target_count": 5000,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(frontend_dist=frontend_dist))
    response = client.get("/api/v1/analysis/jobs?limit=5")

    assert response.status_code == 200
    assert response.json()[0]["status"] == "running"


def test_create_analysis_job_autoloads_unknown_manual_symbol(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        data_service,
        "ensure_symbols_available",
        lambda symbols, market: ["600519.SH"],
    )
    monkeypatch.setattr(analysis_service, "_write_job_payload", lambda job_id, payload: None)

    class DummyThread:
        def __init__(self, target, args, daemon, name):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon
            captured["name"] = name

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(analysis_service.threading, "Thread", DummyThread)

    job = analysis_service.create_analysis_job(
        {"mode": "single", "market": "CN", "targets": ["600519"]}
    )

    worker_args = captured["args"]
    assert job["status"] == "queued"
    assert captured["started"] is True
    assert worker_args[1]["targets"] == ["600519.SH"]
    assert worker_args[1]["stocks"] == ["600519.SH"]


def test_stale_running_job_is_reconciled_and_saved_to_history(tmp_path, monkeypatch):
    frontend_dist = _make_frontend_dist(tmp_path)
    results_dir = tmp_path / "results"
    web_analysis_dir = results_dir / "web_analysis"
    jobs_dir = web_analysis_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(analysis_service, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_service, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(analysis_service, "WEB_ANALYSIS_DIR", web_analysis_dir)
    monkeypatch.setattr(analysis_service, "JOB_DIR", jobs_dir)

    result_payload = {
        "analysis_id": "20260315_010101",
        "created_at": "2026-03-15T01:01:01",
        "request": {
            "mode": "single",
            "market": "CN",
            "preset": "quick_scan",
            "targets": ["000001.SZ"],
            "branches": {"kronos": {"enabled": True, "settings": {"prediction_horizon": "20d"}}},
        },
        "branches": [],
        "candidate_symbols": ["000001.SZ"],
        "target_exposure": 0.4,
        "risk": {"risk_level": "中等", "warnings": []},
        "report_markdown": "test",
        "execution_log": [],
        "trade_recommendations": [],
    }
    analysis_service._result_file_for("20260315_010101").write_text(
        json.dumps(result_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    analysis_service._job_file_for("20260315_010101_000001").write_text(
        json.dumps(
            {
                "ok": True,
                "job_id": "20260315_010101_000001",
                "status": "running",
                "created_at": "2026-03-15T01:01:01",
                "updated_at": "2026-03-15T01:01:01",
                "result": None,
                "error": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(frontend_dist=frontend_dist))

    jobs = client.get("/api/v1/analysis/jobs?limit=5")
    history = client.get("/api/v1/analysis/history?limit=5")

    assert jobs.status_code == 200
    assert jobs.json()[0]["status"] == "completed"
    assert jobs.json()[0]["result"]["analysis_id"] == "20260315_010101"

    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["analysis_id"] == "20260315_010101"


def test_analysis_history_delete_endpoints_remove_saved_results(tmp_path, monkeypatch):
    frontend_dist = _make_frontend_dist(tmp_path)
    results_dir = tmp_path / "results"
    web_analysis_dir = results_dir / "web_analysis"
    jobs_dir = web_analysis_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(analysis_service, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_service, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(analysis_service, "WEB_ANALYSIS_DIR", web_analysis_dir)
    monkeypatch.setattr(analysis_service, "JOB_DIR", jobs_dir)

    normalized = _normalize_web_result(
        {
            "analysis_id": "20260315_020202",
            "created_at": "2026-03-15T02:02:02",
            "request": {"targets": ["000001.SZ"], "market": "CN", "mode": "single", "preset": "quick_scan"},
            "branches": [],
            "candidate_symbols": [],
            "target_exposure": 0.2,
            "risk": {"risk_level": "中等", "warnings": []},
        }
    )
    analysis_service._save_analysis_session(normalized)
    analysis_service._result_file_for("20260315_020202").write_text(
        json.dumps(normalized, ensure_ascii=False),
        encoding="utf-8",
    )

    client = TestClient(create_app(frontend_dist=frontend_dist))

    single_delete = client.delete("/api/v1/analysis/20260315_020202")
    assert single_delete.status_code == 200
    assert single_delete.json()["deleted_count"] >= 1

    analysis_service._save_analysis_session(normalized)
    analysis_service._result_file_for("20260315_020202").write_text(
        json.dumps(normalized, ensure_ascii=False),
        encoding="utf-8",
    )
    clear_all = client.delete("/api/v1/analysis/history")
    history = client.get("/api/v1/analysis/history?limit=5")

    assert clear_all.status_code == 200
    assert clear_all.json()["deleted_count"] >= 1
    assert history.status_code == 200
    assert history.json()["total"] == 0


def test_portfolio_crud_endpoints(tmp_path, monkeypatch):
    frontend_dist = _make_frontend_dist(tmp_path)
    monkeypatch.setattr(portfolio_service, "APP_DB_PATH", str(tmp_path / "app.db"))

    client = TestClient(create_app(frontend_dist=frontend_dist))

    holding = client.post(
        "/api/v1/portfolio/holdings",
        json={"account_name": "主账户", "symbol": "000001.SZ", "market": "CN", "quantity": 200, "cost_basis": 12.5},
    )
    watchlist = client.post(
        "/api/v1/portfolio/watchlist",
        json={"symbol": "600519.SH", "market": "CN", "priority": "high", "notes": "观察回调"},
    )
    state = client.get("/api/v1/portfolio")

    assert holding.status_code == 200
    assert watchlist.status_code == 200
    assert state.status_code == 200
    assert state.json()["summary"]["holdings_count"] == 1
    assert state.json()["summary"]["account_count"] == 1
    assert state.json()["summary"]["accounts"] == ["主账户"]
    assert state.json()["summary"]["watchlist_count"] == 1
    assert state.json()["summary"]["holding_symbols"] == ["000001.SZ"]
    holding_id = state.json()["holdings"][0]["holding_id"]

    delete_holding = client.delete(f"/api/v1/portfolio/holdings/{holding_id}")
    delete_watchlist = client.delete("/api/v1/portfolio/watchlist/600519.SH")
    final_state = client.get("/api/v1/portfolio")

    assert delete_holding.status_code == 200
    assert delete_watchlist.status_code == 200
    assert final_state.json()["summary"]["holdings_count"] == 0
    assert final_state.json()["summary"]["watchlist_count"] == 0


def test_data_endpoints_only_list_downloaded_stocks(tmp_path, monkeypatch):
    frontend_dist = _make_frontend_dist(tmp_path)
    stock_db = tmp_path / "stock.db"

    conn = sqlite3.connect(stock_db)
    conn.execute(
        """
        CREATE TABLE stock_list (
            ts_code TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            market TEXT,
            list_date TEXT,
            is_hs300 INTEGER DEFAULT 0,
            is_zz500 INTEGER DEFAULT 0,
            is_zz1000 INTEGER DEFAULT 0,
            last_update TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE daily_data (
            ts_code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO stock_list
        (ts_code, name, industry, market, list_date, is_hs300, is_zz500, is_zz1000, last_update)
        VALUES
        ('000001.SZ', '平安银行', '银行', 'CN', '19910403', 1, 0, 0, '2026-03-16T00:00:00'),
        ('000002.SZ', '万科A', '地产', 'CN', '19910129', 1, 0, 0, '2026-03-16T00:00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO daily_data
        (ts_code, trade_date, open, high, low, close, volume, amount)
        VALUES ('000001.SZ', '20260312', 10, 11, 9, 10.5, 1000, 10500)
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("web.services.data_service.STOCK_DB_PATH", str(stock_db))
    monkeypatch.setattr("web.services.data_service._ensure_missing_stock_data_downloaded", lambda *args, **kwargs: None)

    client = TestClient(create_app(frontend_dist=frontend_dist))

    statistics = client.get("/api/v1/data/statistics")
    stocks = client.get("/api/v1/data/stocks?limit=10")
    missing_detail = client.get("/api/v1/data/stocks/000002.SZ")

    assert statistics.status_code == 200
    assert statistics.json()["total_stocks"] == 1
    assert statistics.json()["cn_count"] == 1
    assert statistics.json()["hs300_count"] == 1
    assert statistics.json()["stocks_with_data"] == 1

    assert stocks.status_code == 200
    assert stocks.json()["total"] == 1
    assert [item["ts_code"] for item in stocks.json()["items"]] == ["000001.SZ"]

    assert missing_detail.status_code == 404
