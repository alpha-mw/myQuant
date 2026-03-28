"""Settings and LLM model availability endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from web.models.research_models import LLMModelOption, LLMModelsResponse
from web.models.settings_models import (
    BacktestDefaults,
    CredentialStatus,
    SettingsResponse,
    SettingsUpdateRequest,
)

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

    return SettingsResponse(
        credentials=credentials,
        backtest=BacktestDefaults(
            initial_cash=float(os.environ.get("INITIAL_CASH", 1_000_000)),
            commission_rate=float(os.environ.get("COMMISSION_RATE", 0.0003)),
            stamp_duty_rate=float(os.environ.get("STAMP_DUTY_RATE", 0.001)),
            slippage=float(os.environ.get("SLIPPAGE", 0.001)),
        ),
        db_path=os.environ.get("DB_PATH", "data/stock_database.db"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
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
