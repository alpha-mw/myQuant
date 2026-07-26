from __future__ import annotations

import json

from quant_investor.v17_v2_runtime.cli import main
from quant_investor.v17_v2_runtime.service import verify_runtime


def test_verify_runtime_proves_complete_non_authoritative_phase1() -> None:
    readiness = verify_runtime()
    assert readiness.matrix_status == "COMPLETE"
    assert readiness.runtime_usable is True
    assert readiness.pending_registry == ()
    assert readiness.authority is False
    assert readiness.packaged_asset_count >= 36
    assert readiness.implementation_binding_count >= 10


def test_cli_verify_is_machine_readable_and_non_authoritative(capsys) -> None:
    assert main(["verify"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protocol_version"] == "myquant.v17.v2"
    assert output["matrix_status"] == "COMPLETE"
    assert output["runtime_usable"] is True
    assert output["pending_registry"] == []
    assert output["authority"] is False


def test_cli_preimport_rejection_has_no_authority(capsys) -> None:
    assert (
        main(
            [
                "gate",
                "--action",
                "SHADOW_PREPARE",
                "--run-id",
                "run-001",
                "--version",
                "unknown",
                "--state",
                "UNKNOWN",
                "--checkpoint",
                "PRE_IMPORT",
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is False
    assert output["allowed_write_namespaces"] == []
    assert output["authority"] is False


def test_cli_analyze_rejects_noncanonical_input_without_fallback(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"cutoff": "2026-07-24T15:00:00+08:00"}\n')
    assert main(["analyze", "--input", str(input_path)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "BLOCKED"
    assert output["authority"] is False
    assert "canonical" in output["detail"]
