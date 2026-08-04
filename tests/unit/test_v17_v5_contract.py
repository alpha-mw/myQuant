from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any

import pytest

from quant_investor.v17_v5_contract import (
    load_compatibility_policy,
    seal_semantic,
    validate_artifact,
    verify_package,
    verify_predecessor,
    verify_runtime_build,
)
from quant_investor.v17_v5_contract.resources import (
    COMPATIBILITY_POLICY_PATH,
    PackageResourceError,
    read_packaged_asset,
)
from quant_investor.v17_v5_contract.schema_validation import SchemaValidationError
from quant_investor.v17_v5_contract.validators import ArtifactContractError, NO_AUTHORITY
from quant_investor.v17_v5_contract.validators import (
    V4_REGIME_EVIDENCE_V3_RUNTIME_SHA256,
    V4_REGIME_EVIDENCE_V3_SCHEMA_SHA256,
    V4_REGIME_INFERENCE_POLICY_V2_SHA256,
    V4_V2_PUBLICATION_BLOCK_CLI_SHA256,
)


def _predecessor_binding() -> dict[str, Any]:
    policy = load_compatibility_policy()
    policy_raw = read_packaged_asset(COMPATIBILITY_POLICY_PATH)
    predecessor = policy["predecessor"]
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "binding_id": "v17.v4.predecessor.phase0",
            "compatibility_policy_ref": {
                "artifact_id": policy["artifact_id"],
                "byte_sha256": hashlib.sha256(policy_raw).hexdigest(),
                "relative_path": (
                    "quant_investor/v17_v5_contract/" "resources/v4_compatibility_policy.v4.json"
                ),
                "semantic_sha256": policy["semantic_sha256"],
                "version": policy["version"],
            },
            "protocol_version": "myquant.v17.v5",
            "regime_evidence_v3_runtime_sha256": V4_REGIME_EVIDENCE_V3_RUNTIME_SHA256,
            "regime_evidence_v3_schema_sha256": V4_REGIME_EVIDENCE_V3_SCHEMA_SHA256,
            "regime_inference_policy_v2_sha256": V4_REGIME_INFERENCE_POLICY_V2_SHA256,
            "source_git_commit": predecessor["source_git_commit"],
            "source_package_asset_count": 114,
            "source_package_manifest_byte_sha256": (predecessor["package_manifest_byte_sha256"]),
            "source_package_manifest_relative_path": (
                predecessor["package_manifest_relative_path"]
            ),
            "source_protocol_version": "myquant.v17.v4",
            "source_runtime_manifest_byte_sha256": (predecessor["runtime_manifest_byte_sha256"]),
            "source_runtime_manifest_relative_path": (
                predecessor["runtime_manifest_relative_path"]
            ),
            "source_runtime_source_count": 34,
            "v2_cli_source_sha256": V4_V2_PUBLICATION_BLOCK_CLI_SHA256,
            "v2_publication_status": "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE",
            "version": "myquant.v17.v5.v4-predecessor-binding.v4",
        }
    )


def test_v5_package_runtime_and_predecessor_are_closed() -> None:
    package = verify_package()
    runtime = verify_runtime_build()
    predecessor = verify_predecessor()

    assert len(package) == 24
    assert set(runtime) == {
        "v17_v5_runtime/__init__.py",
        "v17_v5_runtime/authority.py",
        "v17_v5_runtime/cli.py",
        "v17_v5_runtime/factor_diagnostics.py",
        "v17_v5_runtime/factor_lifecycle.py",
        "v17_v5_runtime/factor_regime_diagnostics.py",
        "v17_v5_runtime/factor_regime_origin_inventory.py",
        "v17_v5_runtime/regime_chain_deployability.py",
        "v17_v5_runtime/v4_compat_reader.py",
        "v17_v5_runtime/v4_factor_adapter.py",
        "v17_v5_runtime/v4_regime_adapter.py",
    }
    assert predecessor == {
        "package_asset_count": 114,
        "package_manifest_byte_sha256": (
            "a603b5f3e5f012548e3c3a224ba32ffc62b072d6555849887369f48f45012449"
        ),
        "protocol_version": "myquant.v17.v4",
        "regime_evidence_v3_runtime_sha256": V4_REGIME_EVIDENCE_V3_RUNTIME_SHA256,
        "regime_evidence_v3_schema_sha256": V4_REGIME_EVIDENCE_V3_SCHEMA_SHA256,
        "regime_inference_policy_v2_sha256": V4_REGIME_INFERENCE_POLICY_V2_SHA256,
        "runtime_manifest_byte_sha256": (
            "9f3e6ebc2bc9283b5d81113630d2dad68eef6bec0eddd0fcd28077a5153edfbe"
        ),
        "runtime_source_count": 34,
        "source_git_commit": "6a2fa23dec68d87eb686464a86d8ba8997416310",
        "status": "PINNED_AND_VERIFIED",
        "v2_cli_source_sha256": V4_V2_PUBLICATION_BLOCK_CLI_SHA256,
        "v2_publication_status": "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE",
    }


def test_v4_predecessor_binding_validates_without_granting_authority() -> None:
    artifact = _predecessor_binding()

    assert validate_artifact(artifact) == artifact
    assert artifact["authority"] == NO_AUTHORITY


def test_v4_predecessor_binding_rejects_authority_or_manifest_drift() -> None:
    artifact = _predecessor_binding()
    authority_mutation = dict(artifact)
    authority_mutation.pop("semantic_sha256")
    authority_mutation["authority"] = {**NO_AUTHORITY, "execution": True}
    with pytest.raises(Exception):
        validate_artifact(seal_semantic(authority_mutation))

    manifest_mutation = dict(artifact)
    manifest_mutation.pop("semantic_sha256")
    manifest_mutation["source_package_manifest_byte_sha256"] = "0" * 64
    with pytest.raises(ArtifactContractError, match="predecessor binding"):
        validate_artifact(seal_semantic(manifest_mutation))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", "v17.v4.compatibility.policy.other"),
        ("byte_sha256", "0" * 64),
        (
            "relative_path",
            "quant_investor/v17_v5_contract/resources/other_policy.json",
        ),
        ("semantic_sha256", "1" * 64),
    ],
)
def test_v4_predecessor_binding_rejects_compatibility_policy_drift(
    field: str,
    value: str,
) -> None:
    artifact = _predecessor_binding()
    artifact.pop("semantic_sha256")
    policy_ref = dict(artifact["compatibility_policy_ref"])
    policy_ref[field] = value
    artifact["compatibility_policy_ref"] = policy_ref

    with pytest.raises(
        ArtifactContractError,
        match="compatibility policy identity",
    ):
        validate_artifact(seal_semantic(artifact))


def test_v4_predecessor_binding_rejects_compatibility_policy_version_drift() -> None:
    artifact = _predecessor_binding()
    artifact.pop("semantic_sha256")
    policy_ref = dict(artifact["compatibility_policy_ref"])
    policy_ref["version"] = "myquant.v17.v5.v4-compatibility-policy.v1"
    artifact["compatibility_policy_ref"] = policy_ref

    with pytest.raises(SchemaValidationError, match="does not match const"):
        validate_artifact(seal_semantic(artifact))


def test_compatibility_policy_is_exact_release_rc_1_allowlist() -> None:
    policy = load_compatibility_policy()

    assert [row["version"] for row in policy["allowed_artifacts"]] == [
        "myquant.v17.v4.existing-factor-inventory.v1",
        "myquant.v17.v4.factor-universe-observation.v1",
        "myquant.v17.v4.forward-evaluation-receipt.v1",
        "myquant.v17.v4.forward-evidence-origin-inventory.v1",
        "myquant.v17.v4.forward-factor-input-bundle.v1",
        "myquant.v17.v4.forward-label.v1",
        "myquant.v17.v4.forward-observation-run.v1",
        "myquant.v17.v4.forward-run-request.v1",
        "myquant.v17.v4.forward-source-locator.v1",
        "myquant.v17.v4.forward-source-parquet.v1",
        "myquant.v17.v4.forward-source-slice-manifest.v1",
        "myquant.v17.v4.forward-stage-output.v1",
        "myquant.v17.v4.forward-stage-receipt.v1",
        "myquant.v17.v4.regime-calendar-terminal.v1",
        "myquant.v17.v4.regime-chain-anchor.v1",
        "myquant.v17.v4.regime-evidence.v1",
        "myquant.v17.v4.regime-evidence.v2",
        "myquant.v17.v4.regime-evidence.v3",
        "myquant.v17.v4.regime-feature-snapshot.v1",
        "myquant.v17.v4.regime-market-terminal.v1",
        "myquant.v17.v4.regime-model-snapshot.v1",
        "myquant.v17.v4.regime-model-snapshot.v2",
        "myquant.v17.v4.regime-pit-membership-terminal.v1",
        "myquant.v17.v4.regime-segment-anchor.v1",
        "myquant.v17.v4.regime-source-locator-terminal.v1",
        "myquant.v17.v4.regime-state-checkpoint.v1",
        "myquant.v17.v4.regime-transition-matrix-snapshot.v1",
        "myquant.v17.v4.regime-transition-matrix-snapshot.v2",
        "myquant.v17.v4.research-shadow-factor-set.v1",
        "myquant.v17.v4.shadow-factor-selection-audit.v1",
    ]
    assert [row["version"] for row in policy["allowed_artifacts"] if row["root_admissible"]] == [
        "myquant.v17.v4.forward-evaluation-receipt.v1",
        "myquant.v17.v4.regime-evidence.v1",
        "myquant.v17.v4.regime-evidence.v2",
        "myquant.v17.v4.regime-evidence.v3",
    ]
    assert policy["forbidden_import_prefixes"] == ["quant_investor.v17_v4_runtime"]
    assert (
        "quant_investor.v17_v4_runtime.provisional_forward" in policy["forbidden_v4_writer_modules"]
    )
    assert all(value is False for value in policy["authority"].values())


def test_factor_diagnostic_policy_is_descriptive_only() -> None:
    from quant_investor.v17_v5_contract import load_factor_diagnostic_policy

    policy = load_factor_diagnostic_policy()

    assert policy["sample_policy"] == {
        "descriptive_coverage_minimum_origins": 60,
        "descriptive_coverage_minimum_symbols_per_origin": 100,
        "horizon_sessions": 20,
        "inference_gate_passed": False,
        "naturally_matured_only": True,
    }
    assert policy["statuses"] == ["ACCUMULATING", "UNAVAILABLE", "UNOBSERVED"]
    assert all(value is False for value in policy["authority"].values())


@pytest.mark.parametrize(
    "relative_path",
    [
        "resources/factor_diagnostic_policy.v1.json",
        "schemas/factor_diagnostic.v1.schema.json",
        "../v17_v5_runtime/factor_diagnostics.py",
    ],
)
def test_factor_diagnostic_policy_schema_and_runtime_tamper_fail_closed(
    tmp_path: Path,
    relative_path: str,
) -> None:
    from quant_investor.v17_v5_contract.resources import verify_package as verify_copy

    source_quant = Path(__file__).resolve().parents[2] / "quant_investor"
    target_quant = tmp_path / "quant_investor"
    shutil.copytree(source_quant / "v17_v5_contract", target_quant / "v17_v5_contract")
    shutil.copytree(source_quant / "v17_v5_runtime", target_quant / "v17_v5_runtime")
    target = target_quant / "v17_v5_contract" / relative_path
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(PackageResourceError):
        verify_copy(package_root=target_quant / "v17_v5_contract")


def test_v4_compatibility_policy_and_reader_are_manifest_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    package = verify_package()
    runtime = verify_runtime_build()

    assert (
        package["resources/v4_compatibility_policy.v2.json"]
        == hashlib.sha256(
            (
                root / "quant_investor/v17_v5_contract/resources/" "v4_compatibility_policy.v2.json"
            ).read_bytes()
        ).hexdigest()
    )
    assert (
        runtime["v17_v5_runtime/v4_compat_reader.py"]
        == hashlib.sha256(
            (root / "quant_investor/v17_v5_runtime/v4_compat_reader.py").read_bytes()
        ).hexdigest()
    )
