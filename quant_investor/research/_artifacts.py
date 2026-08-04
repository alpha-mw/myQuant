"""Canonical helpers shared by the sanitized V17 research runtime."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import PurePosixPath
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.schema_validation import validate_artifact

PROTOCOL_VERSION: Final = "myquant.v17.v4"
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class ResearchArtifactError(RuntimeError):
    """Raised when research-only artifact construction fails closed."""


def timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ResearchArtifactError(f"RESEARCH_{label.upper()}_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchArtifactError(f"RESEARCH_{label.upper()}_INVALID") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise ResearchArtifactError(f"RESEARCH_{label.upper()}_INVALID")
    return value


def session(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 10:
        raise ResearchArtifactError(f"RESEARCH_{label.upper()}_INVALID")
    try:
        if datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
            raise ValueError
    except ValueError as exc:
        raise ResearchArtifactError(f"RESEARCH_{label.upper()}_INVALID") from exc
    return value


def artifact_identity(document: Mapping[str, Any], *, field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal(document: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    result = dict(document)
    result[identity_field] = artifact_identity(result, field=identity_field)
    sealed = seal_semantic(result)
    validate_artifact(sealed)
    return sealed


def artifact_ref(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    relative_path: PurePosixPath,
) -> dict[str, str]:
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": str(relative_path),
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def sorted_refs(values: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(value) for value in values),
        key=lambda row: (
            str(row["relative_path"]).encode("ascii"),
            str(row["byte_sha256"]).encode("ascii"),
        ),
    )


__all__ = [
    "NO_AUTHORITY",
    "PROTOCOL_VERSION",
    "ResearchArtifactError",
    "artifact_identity",
    "artifact_ref",
    "seal",
    "session",
    "sorted_refs",
    "timestamp",
]
