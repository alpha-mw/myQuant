"""Service layer for reading/writing application settings."""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from web.config import PROJECT_ROOT
from web.services.run_history_store import history_store

logger = logging.getLogger(__name__)

# Credential names used by the quant system
CREDENTIAL_KEYS = [
    ("TUSHARE_TOKEN", "Tushare"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
    ("GOOGLE_API_KEY", "Google"),
    ("FRED_API_KEY", "FRED"),
    ("FINNHUB_API_KEY", "Finnhub"),
    ("DASHSCOPE_API_KEY", "Dashscope"),
]

ENV_FILE = PROJECT_ROOT / ".env"


def _mask(value: str, keep_prefix: int = 4, keep_suffix: int = 2) -> str:
    if not value:
        return ""
    if len(value) <= keep_prefix + keep_suffix:
        return "*" * len(value)
    return f"{value[:keep_prefix]}{'*' * (len(value) - keep_prefix - keep_suffix)}{value[-keep_suffix:]}"


def get_credentials_status() -> list[dict]:
    result = []
    for env_name, display_name in CREDENTIAL_KEYS:
        val = os.environ.get(env_name, "").strip()
        result.append({
            "name": display_name,
            "env_key": env_name,
            "is_set": bool(val),
            "masked_value": _mask(val) if val else "",
        })
    return result


def get_backtest_defaults() -> dict:
    return {
        "initial_cash": float(os.environ.get("INITIAL_CASH", "1000000")),
        "commission_rate": float(os.environ.get("COMMISSION_RATE", "0.0003")),
        "stamp_duty_rate": float(os.environ.get("STAMP_DUTY_RATE", "0.001")),
        "slippage": float(os.environ.get("SLIPPAGE", "0.001")),
    }


def _file_summary(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "size_bytes": None,
            "modified_at": None,
        }

    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _workspace_db_summary() -> dict[str, object]:
    db_path = history_store._db_path.resolve()
    summary = _file_summary(db_path)
    if not db_path.exists():
        return {
            **summary,
            "run_count": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "preset_count": 0,
            "pending_trades": 0,
            "last_run_at": None,
        }

    conn = history_store._conn()

    def scalar(query: str, fallback: int = 0) -> int:
        try:
            row = conn.execute(query).fetchone()
        except sqlite3.DatabaseError:
            return fallback
        return int(row[0]) if row and row[0] is not None else fallback

    try:
        last_run_row = conn.execute("SELECT MAX(created_at) FROM runs").fetchone()
    except sqlite3.DatabaseError:
        last_run_row = None

    return {
        **summary,
        "run_count": scalar("SELECT COUNT(*) FROM runs"),
        "completed_runs": scalar("SELECT COUNT(*) FROM runs WHERE status = 'completed'"),
        "failed_runs": scalar("SELECT COUNT(*) FROM runs WHERE status = 'failed'"),
        "preset_count": scalar("SELECT COUNT(*) FROM presets"),
        "pending_trades": scalar("SELECT COUNT(*) FROM trade_records WHERE outcome_status = 'pending'"),
        "last_run_at": str(last_run_row[0]) if last_run_row and last_run_row[0] else None,
    }


def get_settings() -> dict:
    stock_db_path = Path(os.environ.get("DB_PATH", "data/stock_database.db"))
    return {
        "credentials": get_credentials_status(),
        "backtest": get_backtest_defaults(),
        "db_path": str(stock_db_path),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        "stock_db": _file_summary(stock_db_path),
        "workspace_db": _workspace_db_summary(),
    }


def update_env_file(updates: dict[str, str]) -> None:
    """Update .env file with new values. Creates the file if missing."""
    existing: dict[str, str] = {}
    raw_lines: list[str] = []

    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                raw_lines.append(line)
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, val = stripped.partition("=")
                    existing[key.strip()] = val.strip()

    # Merge updates
    for key, value in updates.items():
        if value is not None:
            existing[key] = value
            # Also update the live process environment
            os.environ[key] = value

    # Atomic rewrite: write to temp file then replace
    fd, tmp_path = tempfile.mkstemp(dir=ENV_FILE.parent, suffix=".env.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            written_keys: set[str] = set()
            for line in raw_lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.partition("=")[0].strip()
                    if key in existing:
                        f.write(f"{key}={existing[key]}\n")
                        written_keys.add(key)
                        continue
                f.write(line)
            # Write new keys
            for key, value in existing.items():
                if key not in written_keys:
                    f.write(f"{key}={value}\n")
        os.replace(tmp_path, ENV_FILE)
        logger.info("Updated .env with keys: %s", list(updates.keys()))
    except Exception:
        logger.exception("Failed to write .env file")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def update_settings(updates: dict) -> dict:
    """Apply settings updates and return the new settings."""
    env_updates: dict[str, str] = {}

    key_mapping = {
        "tushare_token": "TUSHARE_TOKEN",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "deepseek_api_key": "DEEPSEEK_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "fred_api_key": "FRED_API_KEY",
        "finnhub_api_key": "FINNHUB_API_KEY",
        "dashscope_api_key": "DASHSCOPE_API_KEY",
        "initial_cash": "INITIAL_CASH",
        "commission_rate": "COMMISSION_RATE",
        "stamp_duty_rate": "STAMP_DUTY_RATE",
        "slippage": "SLIPPAGE",
        "log_level": "LOG_LEVEL",
    }

    for field, env_key in key_mapping.items():
        val = updates.get(field)
        if val is not None:
            env_updates[env_key] = str(val)

    if env_updates:
        update_env_file(env_updates)

    return get_settings()
