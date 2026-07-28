from __future__ import annotations

import hashlib
import json

import pytest

from quant_investor.v17_v2_contract.validators import (
    SourceAdmissionDisposition,
    SourceAdmissionOutcome,
)
from quant_investor.v17_v2_runtime.cli import main
from quant_investor.v17_v2_runtime.pipeline import PipelineInput
from quant_investor.v17_v2_runtime import service as service_module
from quant_investor.v17_v2_runtime.service import (
    RuntimeServiceError,
    admit_source_bundle,
    pipeline_input_from_mapping,
    verify_runtime,
)


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


def test_pipeline_wire_envelope_round_trips_through_service_parser() -> None:
    value = PipelineInput(
        cutoff="2026-07-24T15:00:00+08:00",
        strategy_id="cn-shadow",
        fundamental_rows=(),
        fundamental_history=(),
        forward_observations=(),
        price_history={},
        timing_observations=(),
        deep_response={},
        sealed_evidence_ids={},
        holdings={},
        cash="1",
        nav="1",
        risk_policy={},
        cost_policy={},
        tradability={},
        risk_model={},
        clusters={},
        macro={},
        markov={},
        portfolio_candidates=(),
    )
    assert pipeline_input_from_mapping(value.to_wire()).to_wire() == value.to_wire()
    invalid = {**value.to_wire(), "authority": True}
    with pytest.raises(RuntimeServiceError, match="authority"):
        pipeline_input_from_mapping(invalid)


def test_cli_exposes_read_only_source_admission_surface(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["admit-sources", "--help"])
    assert exc.value.code == 0
    assert "--commit" in capsys.readouterr().out


def test_source_admission_surface_calls_exact_validator_without_writing(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"PAR1exact")
    observed = {}

    def fake_admit(**kwargs):
        observed.update(kwargs)
        return SourceAdmissionOutcome(
            disposition=SourceAdmissionDisposition.ADMITTED,
            locator={"locator_id": "locator-001"},
            locator_byte_sha256="a" * 64,
            input_bindings=(("role", "id", "version", "path", "b" * 64, "c" * 64),),
            unavailable_required_roles=(),
        )

    monkeypatch.setattr(service_module, "admit_runtime_source_hash_dag", fake_admit)
    payload = {
        "source_root": str(tmp_path),
        "source_objects": [
            {
                "relative_path": (
                    "data/private/v17_sources/protocol-v2/objects/aa/" + "a" * 64 + ".parquet"
                ),
                "absolute_path": str(source),
                "expected_sha256": hashlib.sha256(b"PAR1exact").hexdigest(),
            }
        ],
        "dataset_manifests": {},
        "observation_dispositions": {},
        "source_manifest": {},
        "source_manifest_path": ("data/private/v17_sources/protocol-v2/manifests/source.json"),
        "generation_catalogs": {},
        "summaries": {},
        "source_binding_set": {},
        "source_binding_set_path": (
            "data/private/v17_sources/protocol-v2/manifests/source.bindings.json"
        ),
        "source_locator": {},
        "source_locator_path": ("data/private/v17_sources/protocol-v2/locators/locator-001.json"),
    }
    result = admit_source_bundle(payload, workspace_root=tmp_path, commit=False)
    assert result.disposition == "ADMITTED"
    assert result.committed is False
    assert result.committed_path_count == 0
    assert observed["source_objects"]
    assert not (tmp_path / "data").exists()
