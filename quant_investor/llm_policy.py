"""
Local LLM execution policy.

myQuant can still keep deterministic research artifacts locally while routing
LLM-dependent interpretation to Codex.  The policy is intentionally small and
centralized so provider keys alone do not imply local LLM calls are allowed.
"""

from __future__ import annotations

import os
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes", "on", "codex"}
FALSE_VALUES = {"0", "false", "no", "off", "local", "none", ""}

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return bool(default)


def local_llm_disabled() -> bool:
    """Return True when local runtime LLM calls should be skipped."""

    if _env_flag("MYQUANT_ENABLE_LOCAL_LLM", default=False):
        return False
    if _env_flag("MYQUANT_DISABLE_LOCAL_LLM", default=False):
        return True
    return str(os.getenv("MYQUANT_LLM_HANDOFF", "")).strip().lower() == "codex"


def apply_local_llm_policy(enable_agent_layer: bool) -> bool:
    """Apply the global local-LLM policy to an agent-layer request."""

    return bool(enable_agent_layer) and not local_llm_disabled()


def llm_handoff_reason() -> str:
    if local_llm_disabled():
        return "local_llm_disabled_codex_handoff"
    return ""


def llm_handoff_metadata() -> dict[str, str | bool]:
    disabled = local_llm_disabled()
    return {
        "local_llm_disabled": disabled,
        "llm_handoff": "codex" if disabled else "",
        "handoff_reason": llm_handoff_reason(),
    }
