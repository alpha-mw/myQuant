"""Exact strategy-identity declaration validation without alias selection."""

from __future__ import annotations

import os
from typing import Any, Final, Mapping

from .contracts import (
    ArtifactRef,
    IDENTITY_DECLARATION_PROTOCOL,
    IDENTITY_DECLARATION_SCHEMA_ID,
    PortfolioCycleError,
    VerifiedStrategyIdentity,
    parse_canonical_json,
    require_exact_fields,
    require_identifier,
    require_sha256,
    require_text,
    require_timestamp,
)
from .exact_io import ExactReader

_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "historical_label",
        "canonical_strategy_id",
        "declared_by",
        "declared_at",
        "authority_kind",
        "provenance",
        "semantic_sha256",
    }
)


def validate_strategy_identity_declaration(
    value: Mapping[str, Any], *, expected_historical_label: str | None = None
) -> dict[str, str]:
    code = "PORTFOLIO_CYCLE_IDENTITY_INVALID"
    document = require_exact_fields(value, _FIELDS, label="identity declaration", code=code)
    if (
        document.get("schema_id") != IDENTITY_DECLARATION_SCHEMA_ID
        or document.get("protocol") != IDENTITY_DECLARATION_PROTOCOL
    ):
        raise PortfolioCycleError(code, "identity declaration schema/protocol mismatch")
    historical_label = require_text(
        document.get("historical_label"),
        label="historical_label",
        code=code,
        max_length=160,
    )
    if expected_historical_label is not None:
        expected = require_text(
            expected_historical_label,
            label="expected_historical_label",
            code=code,
            max_length=160,
        )
        if historical_label != expected:
            raise PortfolioCycleError(
                "PORTFOLIO_CYCLE_IDENTITY_MISMATCH",
                "historical label does not match the explicit caller " "reference",
            )
    strategy = require_identifier(
        document.get("canonical_strategy_id"),
        label="canonical_strategy_id",
        code=code,
    )
    declared_by = require_text(
        document.get("declared_by"),
        label="declared_by",
        code=code,
        max_length=200,
    )
    declared_at = document.get("declared_at")
    require_timestamp(declared_at, label="declared_at", code=code)
    authority_kind = document.get("authority_kind")
    if authority_kind != "owner_declaration":
        raise PortfolioCycleError(
            code,
            "authority_kind must be owner_declaration; signed attestation "
            "is unavailable without a signature verifier",
        )
    provenance = require_text(
        document.get("provenance"),
        label="provenance",
        code=code,
        max_length=4000,
    )
    return {
        "historical_label": historical_label,
        "canonical_strategy_id": strategy,
        "declared_by": declared_by,
        "declared_at": declared_at,
        "authority_kind": authority_kind,
        "provenance": provenance,
    }


def resolve_strategy_identity(
    workspace_root: str | os.PathLike[str],
    *,
    declaration_path: str,
    declaration_sha256: str,
    expected_historical_label: str | None = None,
) -> VerifiedStrategyIdentity:
    digest = require_sha256(declaration_sha256, label="declaration_sha256")
    stored = ExactReader(workspace_root).read(declaration_path, expected_sha256=digest)
    document = parse_canonical_json(stored.data)
    normalized = validate_strategy_identity_declaration(
        document, expected_historical_label=expected_historical_label
    )
    return VerifiedStrategyIdentity(
        verified=True,
        declaration_ref=ArtifactRef(
            schema_id=IDENTITY_DECLARATION_SCHEMA_ID,
            relative_path=stored.relative_path,
            byte_sha256=stored.byte_sha256,
        ),
        historical_label=normalized["historical_label"],
        canonical_strategy_id=normalized["canonical_strategy_id"],
        declared_by=normalized["declared_by"],
        declared_at=normalized["declared_at"],
        authority_kind=normalized["authority_kind"],  # type: ignore[arg-type]
        provenance=normalized["provenance"],
    )


__all__ = [
    "resolve_strategy_identity",
    "validate_strategy_identity_declaration",
]
