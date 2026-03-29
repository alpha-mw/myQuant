"""Settings and LLM model availability endpoints."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from web.models.research_models import LLMModelOption, LLMModelsResponse
from web.models.settings_models import (
    BacktestDefaults,
    CredentialStatus,
    DatabaseFileSummary,
    SettingsResponse,
    SettingsUpdateRequest,
    WorkspaceDatabaseSummary,
)
from web.services.run_history_store import history_store

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Canonical (name, env_key) registry shared by GET and PATCH
_CREDENTIAL_REGISTRY: list[tuple[str, str]] = [
    ("Tushare",       "TUSHARE_TOKEN"),
    ("OpenAI",        "OPENAI_API_KEY"),
    ("Anthropic",     "ANTHROPIC_API_KEY"),
    ("DeepSeek",      "DEEPSEEK_API_KEY"),
    ("Google Gemini", "GOOGLE_API_KEY"),
    ("Finnhub",       "FINNHUB_API_KEY"),
    ("Aliyun Qwen",   "DASHSCOPE_API_KEY"),
    ("Moonshot Kimi", "KIMI_API_KEY"),
    ("FRED",          "FRED_API_KEY"),
]

_NUMERIC_REGISTRY: list[tuple[str, str]] = [
    ("initial_cash",      "INITIAL_CASH"),
    ("commission_rate",   "COMMISSION_RATE"),
    ("stamp_duty_rate",   "STAMP_DUTY_RATE"),
    ("slippage",          "SLIPPAGE"),
]

# Maps SettingsUpdateRequest field names → env keys for credentials
_CREDENTIAL_FIELD_MAP: list[tuple[str, str]] = [
    ("tushare_token",      "TUSHARE_TOKEN"),
    ("openai_api_key",     "OPENAI_API_KEY"),
    ("anthropic_api_key",  "ANTHROPIC_API_KEY"),
    ("deepseek_api_key",   "DEEPSEEK_API_KEY"),
    ("google_api_key",     "GOOGLE_API_KEY"),
    ("fred_api_key",       "FRED_API_KEY"),
    ("finnhub_api_key",    "FINNHUB_API_KEY"),
    ("dashscope_api_key",  "DASHSCOPE_API_KEY"),
    ("kimi_api_key",       "KIMI_API_KEY"),
]

_ENV_PATH = Path(".env")


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _persist_to_env(updates: dict[str, str]) -> None:
    """Write key=value pairs to repo-root .env, creating the file if absent."""
    if not updates:
        return

    # Read existing lines preserving order/comments
    lines: list[str] = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()

    existing_keys: dict[str, int] = {}  # key → line index
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            existing_keys[k] = i

    for env_key, value in updates.items():
        if env_key in existing_keys:
            lines[existing_keys[env_key]] = f"{env_key}={value}"
        else:
            lines.append(f"{env_key}={value}")

    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_summary(path: Path) -> DatabaseFileSummary:
    resolved = path.resolve()
    if not resolved.exists():
        return DatabaseFileSummary(
            path=str(resolved),
            exists=False,
            size_bytes=None,
            modified_at=None,
        )

    stat = resolved.stat()
    return DatabaseFileSummary(
        path=str(resolved),
        exists=True,
        size_bytes=int(stat.st_size),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


def _workspace_db_summary() -> WorkspaceDatabaseSummary:
    db_path = history_store._db_path.resolve()
    base_summary = _file_summary(db_path)

    summary = WorkspaceDatabaseSummary(
        path=base_summary.path,
        exists=base_summary.exists,
        size_bytes=base_summary.size_bytes,
        modified_at=base_summary.modified_at,
    )
    if not db_path.exists():
        return summary

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

    return WorkspaceDatabaseSummary(
        path=summary.path,
        exists=summary.exists,
        size_bytes=summary.size_bytes,
        modified_at=summary.modified_at,
        run_count=scalar("SELECT COUNT(*) FROM runs"),
        completed_runs=scalar("SELECT COUNT(*) FROM runs WHERE status = 'completed'"),
        failed_runs=scalar("SELECT COUNT(*) FROM runs WHERE status = 'failed'"),
        preset_count=scalar("SELECT COUNT(*) FROM presets"),
        pending_trades=scalar("SELECT COUNT(*) FROM trade_records WHERE outcome_status = 'pending'"),
        last_run_at=str(last_run_row[0]) if last_run_row and last_run_row[0] else None,
    )


@router.get("/", response_model=SettingsResponse)
async def get_settings():
    credentials = []
    for name, key in _CREDENTIAL_REGISTRY:
        val = os.environ.get(key, "")
        credentials.append(
            CredentialStatus(
                name=name,
                env_key=key,
                is_set=bool(val),
                masked_value=_mask(val),
            )
        )

    stock_db_path = Path(os.environ.get("DB_PATH", "data/stock_database.db"))

    return SettingsResponse(
        credentials=credentials,
        backtest=BacktestDefaults(
            initial_cash=float(os.environ.get("INITIAL_CASH", 1_000_000)),
            commission_rate=float(os.environ.get("COMMISSION_RATE", 0.0003)),
            stamp_duty_rate=float(os.environ.get("STAMP_DUTY_RATE", 0.001)),
            slippage=float(os.environ.get("SLIPPAGE", 0.001)),
        ),
        db_path=str(stock_db_path),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        stock_db=_file_summary(stock_db_path),
        workspace_db=_workspace_db_summary(),
    )


@router.get("/models", response_model=LLMModelsResponse)
async def get_models():
    from quant_investor.agents.llm_client import has_provider_for_model
    from quant_investor.llm_gateway import (
        LLM_MODEL_PRICING_REGISTRY,
        LLM_PROVIDER_REGISTRY,
        detect_provider,
    )

    models = []
    for model_id, pricing in LLM_MODEL_PRICING_REGISTRY.items():
        provider_name = detect_provider(model_id)
        provider_spec = LLM_PROVIDER_REGISTRY.get(provider_name)
        label = f"{model_id} ({provider_name})" if provider_spec else model_id

        models.append(
            LLMModelOption(
                id=model_id,
                provider=provider_name,
                label=label,
                available=has_provider_for_model(model_id),
                prompt_price=pricing.prompt_usd_per_1m,
                completion_price=pricing.completion_usd_per_1m,
            )
        )

    return LLMModelsResponse(models=models)


@router.patch("/")
async def update_settings(request: SettingsUpdateRequest):
    updated: list[str] = []
    env_writes: dict[str, str] = {}

    # Credential fields
    for field_name, env_key in _CREDENTIAL_FIELD_MAP:
        value = getattr(request, field_name, None)
        if value is not None:
            os.environ[env_key] = value
            env_writes[env_key] = value
            updated.append(env_key)

    # Numeric / string config fields
    for field_name, env_key in [
        ("initial_cash",    "INITIAL_CASH"),
        ("commission_rate", "COMMISSION_RATE"),
        ("stamp_duty_rate", "STAMP_DUTY_RATE"),
        ("slippage",        "SLIPPAGE"),
        ("log_level",       "LOG_LEVEL"),
    ]:
        value = getattr(request, field_name, None)
        if value is not None:
            str_value = str(value)
            os.environ[env_key] = str_value
            env_writes[env_key] = str_value
            updated.append(env_key)

    _persist_to_env(env_writes)

    return {"ok": True, "updated": updated}
