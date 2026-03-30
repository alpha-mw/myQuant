"""Workspace web runtime smoke tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import web.main as web_main
from web.api import app as exported_app
from web.api import create_app


def test_web_main_entrypoint_uses_workspace_app():
    assert web_main.app is exported_app
    assert web_main.app.title == "myQuant Research Workspace"


def test_workspace_app_serves_frontend_dist_and_preserves_api_boundary(
    tmp_path,
):
    dist = tmp_path / "frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<html><body>workspace</body></html>",
        encoding="utf-8",
    )
    (assets / "app.js").write_text(
        "console.log('workspace');",
        encoding="utf-8",
    )

    client = TestClient(create_app(frontend_dist=dist))

    root = client.get("/")
    detail = client.get("/history/test-id")
    asset = client.get("/assets/app.js")
    missing_api = client.get("/api/unknown")

    assert root.status_code == 200
    assert "workspace" in root.text

    assert detail.status_code == 200
    assert "workspace" in detail.text

    assert asset.status_code == 200
    assert "console.log('workspace');" in asset.text

    assert missing_api.status_code == 404


def test_workspace_app_exposes_api_health():
    client = TestClient(
        create_app(frontend_dist=Path("/tmp/non-existent-workspace-dist"))
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "12.0.0"}
