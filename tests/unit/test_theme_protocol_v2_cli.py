from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_theme_protocol_v2 import main


def test_cli_formal_mode_without_calendar_fails_closed(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot = tmp_path / "theme.json"
    snapshot.write_text(
        json.dumps(
            {
                "theme_scores": {
                    "tech::ai": {
                        "theme_id": "tech::ai",
                        "theme_name": "人工智能",
                        "score": 80,
                        "attention": 0.8,
                        "market_confirmation": 0.8,
                        "member_count": 1,
                    }
                },
                "symbol_theme_memberships": {"000001.SZ": ["tech::ai"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="formal mode requires --trading-dates"):
        main(
            [
                "--theme-snapshot",
                str(snapshot),
                "--as-of",
                "2026-07-10",
                "--formal-enabled",
                "--formal-kill-switch",
            ]
        )
    assert capsys.readouterr().out == ""
