from __future__ import annotations

from fastapi.testclient import TestClient


def _dto() -> dict:
    ref = {
        "schema_id": "example.ref.v1",
        "relative_path": "results/v17_mainline/example.json",
        "byte_sha256": "a" * 64,
    }
    return {
        "schema_id": "myquant.v17.v4.mainline-public-run.v1",
        "protocol": "myquant.v17.v4",
        "canonical_strategy_id": "cn-mainline",
        "run_id": "run-1",
        "state": "ACTIVE",
        "market": "CN_A_SHARE",
        "capability": "RESEARCH_PORTFOLIO",
        "authority_source": "FORMAL_V17_V4",
        "authority_flags": {
            "broker_calls": False,
            "execution_calls": False,
            "llm_control_calls": False,
            "order_calls": False,
            "provider_calls": False,
            "selector_writes": False,
            "trade_calls": False,
        },
        "read_only": True,
        "selector_used": False,
        "fallback_used": False,
        "active_pointer_ref": ref,
        "mainline_run_ref": ref,
        "formal_output_ref": ref,
        "portfolio_output_ref": ref,
        "source_closure_ref": ref,
        "cash_weight": "0.10",
        "gross_weight": "0.90",
        "targets": [
            {
                "symbol": "000001.SZ",
                "current_target": "0.20",
                "final_target": "0.30",
                "lane": "SELECTION_POOL",
            }
        ],
        "semantic_sha256": "b" * 64,
    }


def test_web_exposes_only_read_only_research_result(monkeypatch, tmp_path) -> None:
    from web.routers import research
    from web.workspace_app import create_app

    expected = _dto()
    captured = {}

    def fake_read_public_run(workspace_root, *, strategy_id, expected_pointer_sha256=None):
        captured.update(
            workspace_root=workspace_root,
            strategy_id=strategy_id,
            expected_pointer_sha256=expected_pointer_sha256,
        )
        return expected

    monkeypatch.setattr(research, "read_public_run", fake_read_public_run)
    client = TestClient(create_app(frontend_dist=tmp_path / "absent", auth_token=""))

    response = client.get(
        "/api/research/cn-mainline?expected_pointer_sha256=" + "c" * 64
    )
    assert response.status_code == 200
    assert response.json() == expected
    assert captured["strategy_id"] == "cn-mainline"
    assert captured["expected_pointer_sha256"] == "c" * 64

    for method, path in (
        ("post", "/api/research/run"),
        ("get", "/api/research/history/list"),
        ("get", "/api/v4/research-runs/cn-mainline"),
    ):
        assert getattr(client, method)(path).status_code in {404, 405, 409, 422}


def test_web_maps_closed_code_without_schema_migration(monkeypatch, tmp_path) -> None:
    from quant_investor.v17_mainline import V17MainlineError
    from web.routers import research
    from web.workspace_app import create_app

    def unavailable(*args, **kwargs):
        raise V17MainlineError("V17_MAINLINE_UNINITIALIZED")

    monkeypatch.setattr(research, "read_public_run", unavailable)
    client = TestClient(create_app(frontend_dist=tmp_path / "absent", auth_token=""))
    response = client.get("/api/research/cn-mainline")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "V17_MAINLINE_UNINITIALIZED"


def test_web_uninitialized_workspace_is_no_write(monkeypatch, tmp_path) -> None:
    from web.routers import research
    from web.workspace_app import create_app

    monkeypatch.setattr(research, "PROJECT_ROOT", tmp_path)
    client = TestClient(create_app(frontend_dist=tmp_path / "absent", auth_token=""))

    response = client.get("/api/research/cn-mainline")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "V17_MAINLINE_UNINITIALIZED"
    assert list(tmp_path.iterdir()) == []


def test_legacy_web_app_removes_analysis_and_portfolio(monkeypatch, tmp_path) -> None:
    from web.app import create_app

    client = TestClient(create_app(frontend_dist=tmp_path / "absent"))
    paths = {route.path for route in client.app.routes}
    assert not any(path.startswith("/api/v1/analysis") for path in paths)
    assert not any(path.startswith("/api/v1/portfolio") for path in paths)
    assert "/api/research/{strategy_id}" in paths
