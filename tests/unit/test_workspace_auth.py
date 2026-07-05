"""Tests for the optional workspace Bearer auth and binding safety rails.

The workspace API is single-user and unauthenticated by design when bound to
loopback. These tests pin the opt-in protections: WORKSPACE_AUTH_TOKEN gates
/api/* (except /api/health and CORS preflight), CORS refuses a wildcard origin
while credentials are allowed, and non-loopback binding without a token warns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.workspace_app as workspace_app_module
from web.config import warn_if_insecure_binding
from web.workspace_app import create_app

TOKEN = "unit-test-token"


def _client(tmp_path: Path, auth_token: str | None) -> TestClient:
    # Point at a nonexistent dist dir so frontend catch-all routes are not
    # registered; TestClient is used without a context manager so the lifespan
    # (and its DB init) never runs.
    return TestClient(
        create_app(tmp_path / "no_dist", auth_token=auth_token),
        raise_server_exceptions=False,
    )


def test_health_endpoint_is_exempt_from_auth(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token=TOKEN)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_api_requires_bearer_token_when_configured(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token=TOKEN)

    response = client.get("/api/settings")
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_wrong_token_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token=TOKEN)

    response = client.get(
        "/api/settings", headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_valid_token_passes_through_to_router(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token=TOKEN)

    # A nonexistent API path must reach the router (404), proving the
    # middleware admitted the request rather than short-circuiting it.
    response = client.get(
        "/api/definitely-not-a-route",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 404


def test_options_preflight_is_not_blocked(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token=TOKEN)

    response = client.options(
        "/api/definitely-not-a-route",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code != 401


def test_no_token_configured_leaves_api_open(tmp_path: Path) -> None:
    client = _client(tmp_path, auth_token="")

    response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404


def test_token_is_read_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKSPACE_AUTH_TOKEN", TOKEN)
    client = TestClient(
        create_app(tmp_path / "no_dist"), raise_server_exceptions=False
    )

    assert client.get("/api/settings").status_code == 401
    assert (
        client.get(
            "/api/health",
        ).status_code
        == 200
    )


def test_wildcard_cors_origin_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_app_module, "CORS_ORIGINS", ["*"])

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        create_app(tmp_path / "no_dist", auth_token="")


def test_warns_on_non_loopback_binding_without_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("WORKSPACE_AUTH_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="web.config"):
        warn_if_insecure_binding("0.0.0.0")

    assert any("WORKSPACE_AUTH_TOKEN" in record.message for record in caplog.records)


def test_no_warning_on_loopback_binding(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("WORKSPACE_AUTH_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="web.config"):
        warn_if_insecure_binding("127.0.0.1")

    assert not caplog.records


def test_no_warning_when_token_configured(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("WORKSPACE_AUTH_TOKEN", TOKEN)

    with caplog.at_level(logging.WARNING, logger="web.config"):
        warn_if_insecure_binding("0.0.0.0")

    assert not caplog.records
