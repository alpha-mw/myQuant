from __future__ import annotations

import json

from quant_investor.v17_v4_runtime.cli import main


def test_v17_v4_status_is_research_only_and_has_no_mainline_authority(capsys) -> None:
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["research_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["production_authority"] is False
    assert payload["authority"] == {
        "broker": False,
        "execution": False,
        "mainline_authority": False,
        "order": False,
        "production": False,
        "research_only": True,
        "trade": False,
    }


def test_v17_v4_cli_exposes_only_forward_research_surfaces() -> None:
    parser = __import__(
        "quant_investor.v17_v4_runtime.cli",
        fromlist=["_parser"],
    )._parser()
    commands = set(parser._subparsers._group_actions[0].choices)

    assert commands == {
        "deep-v3-compile",
        "factor-set-status",
        "forward-shadow-readiness",
        "forward-shadow-status",
        "run-forward",
        "status",
        "verify",
    }
