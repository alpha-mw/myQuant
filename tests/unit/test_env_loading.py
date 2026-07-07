from __future__ import annotations

import os
from pathlib import Path

from quant_investor.env_loading import load_env_file, read_env_file_values


def test_load_env_file_does_not_override_existing_process_env(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TUSHARE_TOKEN=file-token",
                "API_HOST=0.0.0.0",
                "WORKSPACE_AUTH_TOKEN=file-workspace-token",
                "INITIAL_CASH=2000000",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("WORKSPACE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("INITIAL_CASH", raising=False)

    applied = load_env_file(env_file)

    assert os.environ["API_HOST"] == "127.0.0.1"
    assert os.environ["TUSHARE_TOKEN"] == "file-token"
    assert os.environ["WORKSPACE_AUTH_TOKEN"] == "file-workspace-token"
    assert os.environ["INITIAL_CASH"] == "2000000"
    assert applied == {
        "TUSHARE_TOKEN": "file-token",
        "WORKSPACE_AUTH_TOKEN": "file-workspace-token",
        "INITIAL_CASH": "2000000",
    }


def test_load_env_file_can_override_when_explicit(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_HOST=0.0.0.0\n", encoding="utf-8")
    monkeypatch.setenv("API_HOST", "127.0.0.1")

    applied = load_env_file(env_file, override=True)

    assert applied == {"API_HOST": "0.0.0.0"}
    assert os.environ["API_HOST"] == "0.0.0.0"


def test_read_env_file_values_handles_comments_export_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# comment
export KIMI_API_KEY=sk-kimi
DEEPSEEK_API_KEY="sk-deepseek"
DASHSCOPE_API_KEY='sk-dashscope'
""".strip(),
        encoding="utf-8",
    )

    assert read_env_file_values(env_file) == {
        "KIMI_API_KEY": "sk-kimi",
        "DEEPSEEK_API_KEY": "sk-deepseek",
        "DASHSCOPE_API_KEY": "sk-dashscope",
    }
