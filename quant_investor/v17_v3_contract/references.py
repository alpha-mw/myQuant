"""Single authoritative artifact-identity and seven-field reference builder."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .canonical import canonical_resource_bytes
from .identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_cutoff,
)
from .namespace import NamespaceContractError, root_for_path
from .schema_validation import (
    SchemaValidationError,
    artifact_identity_field,
    validate_artifact,
)


class ArtifactReferenceError(ValueError):
    """Raised when an exact artifact reference cannot be constructed."""

    exit_code = 2


def build_artifact_ref(
    document: Mapping[str, Any],
    raw_bytes: bytes,
    relative_path: str,
) -> dict[str, str]:
    """Validate one canonical artifact and return its exact seven-field reference."""

    if type(document) is not dict or type(raw_bytes) is not bytes:
        raise ArtifactReferenceError("artifact document and exact bytes are required")
    try:
        validate_artifact(document)
        if raw_bytes != canonical_resource_bytes(document):
            raise ArtifactReferenceError("artifact bytes do not match canonical document bytes")
        version = require_opaque_id(document.get("version"), label="artifact version")
        identity_field = artifact_identity_field(version)
        artifact_id = require_opaque_id(
            document.get(identity_field),
            label=identity_field,
        )
        strategy_id = require_opaque_id(
            document.get("strategy_id"),
            label="strategy_id",
        )
        cutoff = require_utc_cutoff(document.get("cutoff"), label="cutoff")
        semantic = require_sha256(
            document.get("semantic_sha256"),
            label="semantic_sha256",
        )
        root_for_path(relative_path)
    except (
        IdentityContractError,
        NamespaceContractError,
        SchemaValidationError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, ArtifactReferenceError):
            raise
        raise ArtifactReferenceError("artifact reference construction failed") from exc
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "cutoff": cutoff,
        "relative_path": relative_path,
        "semantic_sha256": semantic,
        "strategy_id": strategy_id,
    }


__all__ = ["ArtifactReferenceError", "build_artifact_ref"]
