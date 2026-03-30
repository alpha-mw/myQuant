"""Frontend workspace route contract tests."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_FILE = ROOT / "frontend" / "src" / "App.tsx"


def test_workspace_frontend_routes_only_expose_new_information_architecture():
    source = APP_FILE.read_text(encoding="utf-8")

    assert 'path="/"' in source
    assert 'Navigate to="/research" replace' in source
    assert 'path="/research"' in source
    assert 'path="/history"' in source
    assert 'path="/history/:jobId"' in source
    assert 'path="/settings"' in source

    for removed_route in [
        "/market",
        "/stocks",
        "/watchlists",
        "/analysis",
        "/data",
        "/regime",
    ]:
        assert f'path="{removed_route}"' not in source


def test_workspace_frontend_app_does_not_import_removed_legacy_pages():
    source = APP_FILE.read_text(encoding="utf-8")

    for removed_page in [
        "Dashboard",
        "MarketStatus",
        "DataExplorer",
        "StockDetail",
        "AnalysisHub",
        "AnalysisHistory",
        "Watchlists",
        "RegimeMonitor",
        "components/layout/AppShell",
        "components/layout/Sidebar",
    ]:
        assert removed_page not in source
