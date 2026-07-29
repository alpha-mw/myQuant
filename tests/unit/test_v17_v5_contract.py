from __future__ import annotations

import hashlib

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
    read_packaged_asset,
)
from quant_investor.v17_v5_contract.schema_validation import SchemaValidationError
from quant_investor.v17_v5_contract.validators import ArtifactContractError, NO_AUTHORITY


def _predecessor_binding() -> dict[str, object]:
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
                    "quant_investor/v17_v5_contract/" "resources/v4_compatibility_policy.v1.json"
                ),
                "semantic_sha256": policy["semantic_sha256"],
                "version": policy["version"],
            },
            "protocol_version": "myquant.v17.v5",
            "source_git_commit": predecessor["source_git_commit"],
            "source_package_manifest_byte_sha256": (predecessor["package_manifest_byte_sha256"]),
            "source_package_manifest_relative_path": (
                predecessor["package_manifest_relative_path"]
            ),
            "source_protocol_version": "myquant.v17.v4",
            "source_runtime_manifest_byte_sha256": (predecessor["runtime_manifest_byte_sha256"]),
            "source_runtime_manifest_relative_path": (
                predecessor["runtime_manifest_relative_path"]
            ),
            "version": "myquant.v17.v5.v4-predecessor-binding.v1",
        }
    )


def test_v5_package_runtime_and_predecessor_are_closed() -> None:
    package = verify_package()
    runtime = verify_runtime_build()
    predecessor = verify_predecessor()

    assert len(package) == 5
    assert set(runtime) == {
        "v17_v5_runtime/__init__.py",
        "v17_v5_runtime/authority.py",
        "v17_v5_runtime/cli.py",
        "v17_v5_runtime/v4_compat_reader.py",
    }
    assert predecessor == {
        "package_asset_count": 93,
        "package_manifest_byte_sha256": (
            "fdc0aba035cdfff243df1a191431c84cfd7638fd0d94d877c7b37b29d5bc6875"
        ),
        "protocol_version": "myquant.v17.v4",
        "runtime_manifest_byte_sha256": (
            "09700937c1fac82b2e3bbd405f1cbe7d31e71faea6a6c71e2d57d0c8c2b87b04"
        ),
        "runtime_source_count": 30,
        "source_git_commit": "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e",
        "status": "PINNED_AND_VERIFIED",
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
    policy_ref["version"] = "myquant.v17.v5.v4-compatibility-policy.v2"
    artifact["compatibility_policy_ref"] = policy_ref

    with pytest.raises(SchemaValidationError, match="does not match const"):
        validate_artifact(seal_semantic(artifact))


def test_compatibility_policy_is_exact_phase0_allowlist() -> None:
    policy = load_compatibility_policy()

    assert [row["version"] for row in policy["allowed_artifacts"]] == [
        "myquant.v17.v4.regime-evidence.v1"
    ]
    assert policy["allowed_artifacts"][0]["transitive_edges"] == []
    assert policy["forbidden_import_prefixes"] == ["quant_investor.v17_v4_runtime"]
    assert all(value is False for value in policy["authority"].values())
