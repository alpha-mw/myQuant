"""Web application configuration."""

import os
import sys
from pathlib import Path

from quant_investor.env_loading import load_env_file

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

load_env_file(PROJECT_ROOT / ".env")

# API settings
#
# The workspace API has no built-in multi-user auth, so it binds loopback by
# default. Opt into LAN/remote exposure explicitly via API_HOST, and set
# WORKSPACE_AUTH_TOKEN when doing so (enforced as a Bearer token on /api/*).
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def workspace_auth_token() -> str:
    """Read the optional workspace Bearer token at call time (test-friendly)."""

    return os.environ.get("WORKSPACE_AUTH_TOKEN", "").strip()


def warn_if_insecure_binding(host: str | None) -> None:
    """Warn loudly when binding beyond loopback without an auth token."""

    resolved_host = (host or API_HOST).strip()
    if resolved_host in _LOOPBACK_HOSTS:
        return
    if workspace_auth_token():
        return
    import logging

    logging.getLogger("web.config").warning(
        "myQuant workspace is binding to %s with NO authentication configured. "
        "Anyone who can reach this host can trigger analysis jobs, read masked "
        "credential status, and overwrite API keys via /api/settings. Set "
        "WORKSPACE_AUTH_TOKEN, or bind API_HOST=127.0.0.1.",
        resolved_host,
    )


# CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]

# Database paths
STOCK_DB_PATH = os.environ.get("DB_PATH", str(PROJECT_ROOT / "data" / "stock_database.db"))
APP_DB_PATH = os.environ.get("APP_DB_PATH", str(PROJECT_ROOT / "data" / "app.db"))

# Results / runtime paths
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJECT_ROOT / "results")))
WEB_ANALYSIS_DIR = Path(os.environ.get("WEB_ANALYSIS_DIR", str(RESULTS_DIR / "web_analysis")))
PROJECT_VENV_PYTHON = next(
    (
        path
        for path in (
            PROJECT_ROOT / ".venv" / "bin" / "python",
            PROJECT_ROOT / "venv" / "bin" / "python",
        )
        if path.exists()
    ),
    Path(sys.executable),
)

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
