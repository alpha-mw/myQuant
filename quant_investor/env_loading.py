"""Shared .env loading helpers.

The project-wide contract is conservative: values already present in
``os.environ`` are not overwritten unless a caller opts in explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _parse_env_lines(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].strip()
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        parsed_value = value.strip()
        if (
            len(parsed_value) >= 2
            and parsed_value[0] == parsed_value[-1]
            and parsed_value[0] in {"'", '"'}
        ):
            parsed_value = parsed_value[1:-1]
        values[key] = parsed_value
    return values


def read_env_file_values(path: str | Path | None = None) -> dict[str, str]:
    """Read key/value pairs from a .env file without mutating process env."""

    env_path = Path(path) if path is not None else DEFAULT_ENV_FILE
    if not env_path.exists():
        return {}
    try:
        from dotenv import dotenv_values

        parsed = dotenv_values(env_path)
        return {
            str(key): str(value)
            for key, value in parsed.items()
            if key and value is not None
        }
    except ImportError:
        return _parse_env_lines(env_path.read_text(encoding="utf-8").splitlines())


def apply_env_values(values: Mapping[str, str], *, override: bool = False) -> dict[str, str]:
    """Apply parsed env values and return the keys set by this call."""

    applied: dict[str, str] = {}
    for key, value in values.items():
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        if not override and normalized_key in os.environ:
            continue
        os.environ[normalized_key] = str(value)
        applied[normalized_key] = str(value)
    return applied


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a .env file into ``os.environ`` without overriding by default."""

    return apply_env_values(read_env_file_values(path), override=override)


__all__ = [
    "DEFAULT_ENV_FILE",
    "PROJECT_ROOT",
    "apply_env_values",
    "load_env_file",
    "read_env_file_values",
]
