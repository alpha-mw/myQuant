from __future__ import annotations

from pathlib import Path

from quant_investor.config import Config
from quant_investor.llm_gateway import LLM_PROVIDER_ENV_KEYS


ROOT = Path(__file__).resolve().parents[2]


def _env_keys_from_example() -> set[str]:
    keys: set[str] = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def test_mainline_env_inventory_matches_env_example():
    example_keys = _env_keys_from_example()
    config_keys = set(Config.MAINLINE_ENV_KEYS)

    assert example_keys == config_keys
    assert set(LLM_PROVIDER_ENV_KEYS) <= config_keys


def test_archived_env_keys_are_not_in_mainline_inventory():
    config_keys = set(Config.MAINLINE_ENV_KEYS)

    assert "FINNHUB_API_KEY" not in config_keys
