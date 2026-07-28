from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from quant_investor.v17_v2_runtime import RuntimeGate


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_import_never_loads_legacy_v17() -> None:
    code = """
import json
import sys
import quant_investor.v17_v2_runtime as runtime
print(json.dumps({
    "legacy": sorted(name for name in sys.modules if name == "quant_investor.v17" or name.startswith("quant_investor.v17.")),
    "protocol": runtime.PROTOCOL_VERSION,
    "authority": runtime.RUNTIME_AUTHORITY,
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "authority": False,
        "legacy": [],
        "protocol": "myquant.v17.v2",
    }


def test_preimport_gate_is_zero_write(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    decision = RuntimeGate(tmp_path).classify(
        "SHADOW_PREPARE",
        "run-1",
        version="ABSENT",
        state="MISSING",
        checkpoint="PRE_IMPORT",
    )
    assert decision.allowed
    assert decision.matrix_rule == "absent-prepare-gate"
    assert decision.allowed_write_namespaces == ()
    assert decision.retry_cas == "EMPTY"
    assert tuple(tmp_path.iterdir()) == before


def test_matrix_rejection_is_zero_write(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    decision = RuntimeGate(tmp_path).classify(
        "SHADOW_RECEIVE",
        "run-1",
        version="ABSENT",
        state="MISSING",
        checkpoint="PRE_IMPORT",
    )
    assert not decision.allowed
    assert decision.matrix_rule == "absent-missing-other-actions"
    assert decision.allowed_write_namespaces == ()
    assert tuple(tmp_path.iterdir()) == before
