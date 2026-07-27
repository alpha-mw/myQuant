"""Schema-first artifact construction, references, and governed persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any, Mapping

from quant_investor.v17_v3_contract import (
    build_artifact_ref,
    load_canonical_artifact,
    validate_artifact,
)
from quant_investor.v17_v3_contract.canonical import (
    CanonicalContractError,
    canonical_resource_bytes,
    seal_semantic,
)

from .storage import SecureStore, WriteResult


class ArtifactRuntimeError(ValueError):
    """A governed runtime artifact failed its registered contract."""

    exit_code = 2


@dataclass(frozen=True)
class RuntimeArtifact:
    """One canonical, typed artifact bound to its governed relative path."""

    relative_path: PurePosixPath
    document: Mapping[str, Any]
    raw: bytes
    byte_sha256: str

    @property
    def reference(self) -> dict[str, str]:
        return artifact_reference(
            relative_path=self.relative_path,
            document=self.document,
            raw=self.raw,
        )


def seal_typed_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal and validate before any governed write."""

    try:
        document = seal_semantic(dict(payload))
        validate_artifact(document)
    except (CanonicalContractError, RuntimeError, ValueError) as exc:
        raise ArtifactRuntimeError("artifact failed registered validation") from exc
    return document


def load_typed_artifact(
    raw: bytes,
    *,
    label: str,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Strictly decode and typed-validate exact stored bytes."""

    if type(raw) is not bytes:
        raise ArtifactRuntimeError(f"{label} must be exact bytes")
    try:
        validated = load_canonical_artifact(
            raw,
            label=label,
            expected_version=expected_version,
        )
        if hasattr(validated, "as_dict") and callable(validated.as_dict):
            value = validated.as_dict()
        elif isinstance(validated, Mapping):
            value = dict(validated)
        else:
            raise ArtifactRuntimeError(f"{label} root must be an object")
    except (CanonicalContractError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ArtifactRuntimeError):
            raise
        raise ArtifactRuntimeError(f"{label} failed registered validation") from exc
    return value


def runtime_artifact(
    *,
    relative_path: str | PurePosixPath,
    document: Mapping[str, Any],
) -> RuntimeArtifact:
    """Validate an in-memory document and bind its canonical bytes and path."""

    try:
        validate_artifact(document)
        raw = canonical_resource_bytes(document)
    except (CanonicalContractError, RuntimeError, ValueError) as exc:
        raise ArtifactRuntimeError("artifact failed registered validation") from exc
    return RuntimeArtifact(
        PurePosixPath(str(relative_path)),
        dict(document),
        raw,
        hashlib.sha256(raw).hexdigest(),
    )


def artifact_reference(
    *,
    relative_path: str | PurePosixPath,
    document: Mapping[str, Any],
    raw: bytes,
) -> dict[str, str]:
    """Build the exact seven-field v3 artifact reference."""

    try:
        return build_artifact_ref(
            dict(document),
            raw,
            str(relative_path),
        )
    except (RuntimeError, ValueError) as exc:
        raise ArtifactRuntimeError("artifact reference construction failed") from exc


def write_typed_exact_once(
    store: SecureStore,
    artifact: RuntimeArtifact,
) -> WriteResult:
    """Persist immutable typed bytes and typed-validate exact readback."""

    validate_artifact(artifact.document)
    result = store.write_exact_once(artifact.relative_path, artifact.raw)
    readback = store.read(artifact.relative_path, result.byte_sha256)
    observed = load_typed_artifact(
        readback,
        label="written artifact",
        expected_version=str(artifact.document["version"]),
    )
    if observed != dict(artifact.document):
        raise ArtifactRuntimeError("typed artifact readback mismatch")
    return result


def replace_typed_cas(
    store: SecureStore,
    artifact: RuntimeArtifact,
    *,
    expected_sha256: str,
) -> WriteResult:
    """CAS one mutable typed pointer and validate exact readback."""

    validate_artifact(artifact.document)
    result = store.replace_cas(
        artifact.relative_path,
        expected_sha256,
        artifact.raw,
    )
    readback = store.read(artifact.relative_path, result.byte_sha256)
    observed = load_typed_artifact(
        readback,
        label="written pointer",
        expected_version=str(artifact.document["version"]),
    )
    if observed != dict(artifact.document):
        raise ArtifactRuntimeError("typed pointer readback mismatch")
    return result


__all__ = [
    "ArtifactRuntimeError",
    "RuntimeArtifact",
    "artifact_reference",
    "load_typed_artifact",
    "replace_typed_cas",
    "runtime_artifact",
    "seal_typed_artifact",
    "write_typed_exact_once",
]
