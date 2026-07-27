from __future__ import annotations

import json
from pathlib import Path

from quant_investor.v17_v4_runtime.authority import authority_envelope
from quant_investor.v17_v4_runtime.cli import main


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v17_v4_scaffold_has_no_authority() -> None:
    assert authority_envelope() == {
        "protocol_version": "myquant.v17.v4",
        "state": "V15_DEFAULT",
        "formal_research_publication": False,
        "research_runtime_default": False,
        "execution": False,
        "broker": False,
        "order": False,
        "trade": False,
    }


def test_v17_v4_cli_is_explicit_and_v15_default_is_unchanged() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'quant-investor = "quant_investor.cli.main:main"' in pyproject
    assert 'quant-investor-v17-v4 = "quant_investor.v17_v4_runtime.cli:main"' in pyproject
    assert 'default="v15"' in (
        REPO_ROOT / "quant_investor" / "cli" / "main.py"
    ).read_text(encoding="utf-8")


def test_v17_v4_verify_is_no_write_and_all_side_effects_are_false(
    capsys,
) -> None:
    assert main(["verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PUBLIC_SURFACES_AVAILABLE_NOT_DEFAULT"
    assert payload["package_verified"] is True
    assert payload["package_asset_count"] > 0
    for field in (
        "provider_calls",
        "llm_control_calls",
        "execution_calls",
        "broker_calls",
        "order_calls",
        "trade_calls",
        "selector_writes",
    ):
        assert payload[field] is False
