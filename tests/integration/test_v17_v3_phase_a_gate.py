from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_v17_v3_phase_a_gate.py"


def test_v17_v3_phase_a_gate_is_offline_isolated_and_data_blocked() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "myquant.v17.v3"
    assert payload["status"] == "NOT_ACTIVATED_DATA_BLOCKED"
    assert payload["package_verified"] is True
    assert payload["v15_default_entrypoint_unchanged"] is True
    assert payload["dedicated_v3_entrypoint"] is True
    assert payload["authority"] == {
        "broker_authority": False,
        "execution_authority": False,
        "formal_research_publication_authority": False,
        "order_authority": False,
        "production_default": False,
        "trade_authority": False,
    }
    for counter in (
        "provider_calls",
        "llm_calls",
        "broker_calls",
        "order_calls",
        "trade_calls",
    ):
        assert payload[counter] is False
