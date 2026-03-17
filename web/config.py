"""Web application configuration."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load .env file
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _stripped = _line.strip()
            if _stripped and not _stripped.startswith("#") and "=" in _stripped:
                _key, _, _val = _stripped.partition("=")
                _key, _val = _key.strip(), _val.strip()
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# CORS
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Database paths
STOCK_DB_PATH = os.environ.get("DB_PATH", str(PROJECT_ROOT / "data" / "stock_database.db"))
APP_DB_PATH = str(PROJECT_ROOT / "data" / "app.db")

# Results / runtime paths
RESULTS_DIR = PROJECT_ROOT / "results"
WEB_ANALYSIS_DIR = RESULTS_DIR / "web_analysis"
PROJECT_VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Unified scripts path (for importing existing modules)
UNIFIED_SCRIPTS_PATH = str(PROJECT_ROOT / "scripts" / "unified")
