"""Closed schema registry for V17 v5."""

from __future__ import annotations

from typing import Any, Final, Mapping

from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError as DraftSchemaValidationError,
    validate_instance_against_schema,
)

from .canonical import CanonicalContractError, load_canonical_resource

PROTOCOL_VERSION: Final = "myquant.v17.v5"
_ARTIFACT_REGISTRY: Final = {
    "myquant.v17.v5.factor-diagnostic.v1": (
        "schemas/factor_diagnostic.v1.schema.json",
        "diagnostic_id",
    ),
    "myquant.v17.v5.factor-lifecycle-diagnostic.v1": (
        "schemas/factor_lifecycle_diagnostic.v1.schema.json",
        "lifecycle_diagnostic_id",
    ),
    "myquant.v17.v5.factor-regime-origin-inventory.v1": (
        "schemas/factor_regime_origin_inventory.v1.schema.json",
        "inventory_id",
    ),
    "myquant.v17.v5.factor-regime-origin-inventory.v2": (
        "schemas/factor_regime_origin_inventory.v2.schema.json",
        "inventory_id",
    ),
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v1": (
        "schemas/regime_conditioned_factor_diagnostic.v1.schema.json",
        "diagnostic_id",
    ),
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v2": (
        "schemas/regime_conditioned_factor_diagnostic.v2.schema.json",
        "diagnostic_id",
    ),
    "myquant.v17.v5.v4-predecessor-binding.v1": (
        "schemas/v4_predecessor_binding.v1.schema.json",
        "binding_id",
    ),
    "myquant.v17.v5.v4-predecessor-binding.v2": (
        "schemas/v4_predecessor_binding.v2.schema.json",
        "binding_id",
    ),
}


class SchemaValidationError(ValueError):
    """Raised when a V17 v5 schema or artifact is invalid."""

    exit_code = 2


def schema_versions() -> tuple[str, ...]:
    return tuple(sorted(_ARTIFACT_REGISTRY))


def schema_path_for_version(version: Any) -> str:
    if type(version) is not str or version not in _ARTIFACT_REGISTRY:
        raise SchemaValidationError(f"unknown V17 v5 artifact version: {version!r}")
    return _ARTIFACT_REGISTRY[version][0]


def artifact_identity_field(version: Any) -> str:
    schema_path_for_version(version)
    return _ARTIFACT_REGISTRY[version][1]


def validate_schema_version(instance: Any, version: Any) -> dict[str, Any]:
    if type(instance) is not dict:
        raise SchemaValidationError("V17 v5 artifact must be an object")
    from .resources import load_packaged_json

    schema = load_packaged_json(schema_path_for_version(version))
    try:
        validate_instance_against_schema(instance, schema)
    except DraftSchemaValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc
    return dict(instance)


def validate_artifact(instance: Any) -> dict[str, Any]:
    document = validate_schema_version(
        instance, instance.get("version") if type(instance) is dict else None
    )
    from .validators import validate_typed_artifact

    return validate_typed_artifact(document, schema_checked=True)


def load_canonical_artifact(
    raw: bytes,
    *,
    expected_version: str | None = None,
    label: str = "V17 v5 artifact",
) -> dict[str, Any]:
    try:
        instance = load_canonical_resource(raw, label=label)
    except CanonicalContractError as exc:
        raise SchemaValidationError(str(exc)) from exc
    if type(instance) is not dict:
        raise SchemaValidationError(f"{label} root must be an object")
    if expected_version is not None and instance.get("version") != expected_version:
        raise SchemaValidationError(f"{label} version mismatch")
    return validate_artifact(instance)


def registry() -> Mapping[str, tuple[str, str]]:
    return dict(_ARTIFACT_REGISTRY)


__all__ = [
    "PROTOCOL_VERSION",
    "SchemaValidationError",
    "artifact_identity_field",
    "load_canonical_artifact",
    "registry",
    "schema_path_for_version",
    "schema_versions",
    "validate_artifact",
    "validate_schema_version",
]
