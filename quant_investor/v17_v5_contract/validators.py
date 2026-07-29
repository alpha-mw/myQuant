"""Semantic validators for V17 v5 Phase-0 artifacts."""

from __future__ import annotations

from typing import Any, Final, Mapping

from .canonical import CanonicalContractError, validate_semantic_sha
from .identities import (
    IdentityContractError,
    require_git_commit,
    require_identifier,
    require_relative_path,
    require_sha256,
)

NO_AUTHORITY: Final = {
    "broker": False,
    "canary": False,
    "execution": False,
    "factor_governance_write": False,
    "formal_activation": False,
    "formal_research_publication": False,
    "llm": False,
    "order": False,
    "portfolio": False,
    "promotion": False,
    "provider": False,
    "research_runtime_default": False,
    "selector": False,
    "trade": False,
}
PREDECESSOR_BINDING_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v1"
V4_COMPATIBILITY_POLICY_ID: Final = "v17.v4.compatibility.policy.phase0"
V4_COMPATIBILITY_POLICY_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v1"
V4_COMPATIBILITY_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v1.json"
)
V4_COMPATIBILITY_POLICY_BYTE_SHA256: Final = (
    "480d89a7c0804427510f4a32c70195a55085acf4389d40edc939a08851bfec47"
)
V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256: Final = (
    "69334ee3b1065bc923bbbbb06b57f5aa87e59d9c2c353b811a6de33e44bb786c"
)
V4_SOURCE_GIT_COMMIT: Final = "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e"
V4_PACKAGE_MANIFEST_SHA256: Final = (
    "fdc0aba035cdfff243df1a191431c84cfd7638fd0d94d877c7b37b29d5bc6875"
)
V4_RUNTIME_MANIFEST_SHA256: Final = (
    "09700937c1fac82b2e3bbd405f1cbe7d31e71faea6a6c71e2d57d0c8c2b87b04"
)


class ArtifactContractError(ValueError):
    """Raised when a schema-valid V17 v5 artifact violates semantics."""

    exit_code = 2


def _validate_predecessor_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(document["binding_id"], label="binding_id")
        require_git_commit(document["source_git_commit"])
        require_sha256(document["source_package_manifest_byte_sha256"])
        require_sha256(document["source_runtime_manifest_byte_sha256"])
        require_relative_path(document["source_package_manifest_relative_path"])
        require_relative_path(document["source_runtime_manifest_relative_path"])
        policy = document["compatibility_policy_ref"]
        require_identifier(policy["artifact_id"], label="compatibility policy artifact_id")
        require_sha256(policy["byte_sha256"], label="compatibility policy byte SHA-256")
        require_sha256(policy["semantic_sha256"], label="compatibility policy semantic SHA-256")
        require_relative_path(policy["relative_path"], label="compatibility policy path")
    except (CanonicalContractError, IdentityContractError, KeyError, TypeError) as exc:
        raise ArtifactContractError("V17 v4 predecessor binding is invalid") from exc
    if document["authority"] != NO_AUTHORITY:
        raise ArtifactContractError("V17 v5 predecessor binding grants authority")
    if policy != {
        "artifact_id": V4_COMPATIBILITY_POLICY_ID,
        "byte_sha256": V4_COMPATIBILITY_POLICY_BYTE_SHA256,
        "relative_path": V4_COMPATIBILITY_POLICY_PATH,
        "semantic_sha256": V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256,
        "version": V4_COMPATIBILITY_POLICY_VERSION,
    }:
        raise ArtifactContractError("V17 v4 compatibility policy identity mismatch")
    if (
        document["protocol_version"] != "myquant.v17.v5"
        or document["source_protocol_version"] != "myquant.v17.v4"
        or document["source_git_commit"] != V4_SOURCE_GIT_COMMIT
        or document["source_package_manifest_byte_sha256"] != V4_PACKAGE_MANIFEST_SHA256
        or document["source_runtime_manifest_byte_sha256"] != V4_RUNTIME_MANIFEST_SHA256
    ):
        raise ArtifactContractError("V17 v4 predecessor binding identity mismatch")
    return document


def validate_typed_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> dict[str, Any]:
    if not schema_checked:
        from .schema_validation import validate_schema_version

        validate_schema_version(payload, payload.get("version"))
    if payload.get("version") == PREDECESSOR_BINDING_VERSION:
        return _validate_predecessor_binding(payload)
    raise ArtifactContractError("unsupported V17 v5 artifact version")


__all__ = [
    "ArtifactContractError",
    "NO_AUTHORITY",
    "PREDECESSOR_BINDING_VERSION",
    "V4_COMPATIBILITY_POLICY_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_ID",
    "V4_COMPATIBILITY_POLICY_PATH",
    "V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256",
    "V4_COMPATIBILITY_POLICY_VERSION",
    "V4_PACKAGE_MANIFEST_SHA256",
    "V4_RUNTIME_MANIFEST_SHA256",
    "V4_SOURCE_GIT_COMMIT",
    "validate_typed_artifact",
]
