"""Closed Draft 2020-12 schema subset and v4 artifact dispatch."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
import math
import re
from typing import Any, Final, Mapping

from .canonical import (
    CanonicalContractError,
    canonical_bytes,
    load_canonical_resource,
    validate_json_limits,
)
from .resources import PackageResourceError, load_packaged_json

PROTOCOL_VERSION: Final = "myquant.v17.v4"
_DRAFT: Final = "https://json-schema.org/draft/2020-12/schema"
_ARTIFACT_REGISTRY: Final = {
    "myquant.v17.v4.branch-output.v1": (
        "schemas/branch_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v4.calibration-origin-inventory.v1": (
        "schemas/calibration_origin_inventory.v1.schema.json",
        "inventory_id",
    ),
    "myquant.v17.v4.calibration-receipt.v1": (
        "schemas/calibration_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.canary-pointer.v1": (
        "schemas/canary_pointer.v1.schema.json",
        "pointer_id",
    ),
    "myquant.v17.v4.canary-receipt.v1": (
        "schemas/canary_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.default-eligibility-receipt.v1": (
        "schemas/default_eligibility_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.default-eligible-pointer.v1": (
        "schemas/default_eligible_pointer.v1.schema.json",
        "pointer_id",
    ),
    "myquant.v17.v4.dual-run-comparison.v1": (
        "schemas/dual_run_comparison.v1.schema.json",
        "comparison_id",
    ),
    "myquant.v17.v4.formal-activation-receipt.v1": (
        "schemas/formal_activation_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.formal-active-pointer.v1": (
        "schemas/formal_active_pointer.v1.schema.json",
        "pointer_id",
    ),
    "myquant.v17.v4.formal-output.v1": (
        "schemas/formal_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v4.fusion-promotion-receipt.v1": (
        "schemas/fusion_promotion_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v4.historical-canary-policy.v1": (
        "schemas/historical_canary_policy.v1.schema.json",
        "policy_id",
    ),
    "myquant.v17.v4.initial-pool-output.v1": (
        "schemas/initial_pool_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v4.pit-catalog-pointer.v1": (
        "schemas/pit_catalog_pointer.v1.schema.json",
        "pointer_id",
    ),
    "myquant.v17.v4.pit-generation-catalog.v1": (
        "schemas/pit_generation_catalog.v1.schema.json",
        "catalog_id",
    ),
    "myquant.v17.v4.preselect-locator.v1": (
        "schemas/preselect_locator.v1.schema.json",
        "locator_id",
    ),
    "myquant.v17.v4.total-return-labels.v1": (
        "schemas/total_return_labels.v1.schema.json",
        "label_id",
    ),
}
_SUPPORTED_KEYWORDS: Final = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "const",
        "enum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
        "x-ordering",
    }
)
_JSON_TYPES: Final = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_DATE_TIME_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_EXTERNAL_REF_RE: Final = re.compile(
    r"^(?P<file>[a-z0-9_]+\.v1\.schema\.json)#/\$defs/"
    r"(?P<name>[A-Za-z0-9_.-]+)$",
    re.ASCII,
)


class SchemaValidationError(ValueError):
    """Raised when a v4 schema or artifact fails closed validation."""

    exit_code = 2


def schema_path_for_version(version: Any) -> str:
    if type(version) is not str or version not in _ARTIFACT_REGISTRY:
        if type(version) is str and version.startswith("myquant.v17.v3."):
            raise SchemaValidationError("v3 artifact identity cannot be relabelled as v4")
        raise SchemaValidationError(f"unsupported v4 artifact version: {version!r}")
    return _ARTIFACT_REGISTRY[version][0]


def artifact_identity_field(version: Any) -> str:
    schema_path_for_version(version)
    return _ARTIFACT_REGISTRY[version][1]


def schema_versions() -> tuple[str, ...]:
    return tuple(sorted(_ARTIFACT_REGISTRY))


def _resolve_reference(
    reference: Any,
    *,
    root: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    if type(reference) is not str:
        raise SchemaValidationError("schema reference must be a string")
    if reference.startswith("#/$defs/"):
        name = reference.removeprefix("#/$defs/")
        definitions = root.get("$defs")
        if type(definitions) is not dict or name not in definitions:
            raise SchemaValidationError(f"unresolved local schema reference: {reference}")
        return definitions[name], root
    match = _EXTERNAL_REF_RE.fullmatch(reference)
    if match is None:
        raise SchemaValidationError(f"unsupported schema reference: {reference}")
    schema = load_packaged_json(f"schemas/{match.group('file')}")
    definitions = schema.get("$defs")
    name = match.group("name")
    if type(definitions) is not dict or name not in definitions:
        raise SchemaValidationError(
            f"unresolved external schema reference: {reference}"
        )
    return definitions[name], schema


def _preflight_node(
    node: Any,
    *,
    root: Mapping[str, Any],
    path: str,
    reference_stack: tuple[str, ...],
) -> None:
    if type(node) is bool:
        return
    if type(node) is not dict:
        raise SchemaValidationError(f"schema node must be object or boolean at {path}")
    unknown = sorted(set(node) - _SUPPORTED_KEYWORDS)
    if unknown:
        raise SchemaValidationError(f"unsupported schema keywords at {path}: {unknown}")
    reference = node.get("$ref")
    if reference is not None:
        if reference in reference_stack:
            raise SchemaValidationError(f"cyclic schema reference at {path}")
        target, target_root = _resolve_reference(reference, root=root)
        _preflight_node(
            target,
            root=target_root,
            path=f"{path}.$ref",
            reference_stack=(*reference_stack, reference),
        )
    declared = node.get("type")
    types = [declared] if type(declared) is str else declared or []
    if declared is not None and (
        type(types) is not list
        or not types
        or any(type(item) is not str or item not in _JSON_TYPES for item in types)
        or len(types) != len(set(types))
    ):
        raise SchemaValidationError(f"invalid schema type at {path}")
    if "object" in types and node.get("additionalProperties") is not False:
        raise SchemaValidationError(
            f"every v4 object schema must set additionalProperties=false at {path}"
        )
    if "array" in types and (
        type(node.get("uniqueItems")) is not bool
        or type(node.get("x-ordering")) is not str
        or not node["x-ordering"]
    ):
        raise SchemaValidationError(
            f"every v4 array schema must declare uniqueness and ordering at {path}"
        )
    properties = node.get("properties")
    if properties is not None:
        if type(properties) is not dict:
            raise SchemaValidationError(f"schema properties invalid at {path}")
        for name, child in properties.items():
            _preflight_node(
                child,
                root=root,
                path=f"{path}.properties.{name}",
                reference_stack=reference_stack,
            )
    required = node.get("required")
    if required is not None and (
        type(required) is not list
        or len(required) != len(set(required))
        or any(type(item) is not str for item in required)
        or (type(properties) is dict and not set(required).issubset(properties))
    ):
        raise SchemaValidationError(f"schema required list invalid at {path}")
    definitions = node.get("$defs")
    if definitions is not None:
        if type(definitions) is not dict:
            raise SchemaValidationError(f"schema $defs invalid at {path}")
        for name, child in definitions.items():
            _preflight_node(
                child,
                root=root,
                path=f"{path}.$defs.{name}",
                reference_stack=reference_stack,
            )
    if "items" in node:
        _preflight_node(
            node["items"],
            root=root,
            path=f"{path}.items",
            reference_stack=reference_stack,
        )
    for keyword in ("oneOf", "allOf"):
        branches = node.get(keyword)
        if branches is None:
            continue
        if type(branches) is not list or not branches:
            raise SchemaValidationError(f"schema {keyword} invalid at {path}")
        for index, branch in enumerate(branches):
            _preflight_node(
                branch,
                root=root,
                path=f"{path}.{keyword}[{index}]",
                reference_stack=reference_stack,
            )
    if "pattern" in node:
        try:
            re.compile(node["pattern"], re.ASCII)
        except (TypeError, re.error) as exc:
            raise SchemaValidationError(f"schema pattern invalid at {path}") from exc
    for keyword in ("minItems", "maxItems", "minLength", "maxLength"):
        if keyword in node and (type(node[keyword]) is not int or node[keyword] < 0):
            raise SchemaValidationError(f"schema {keyword} invalid at {path}")


def preflight_schema(schema: Mapping[str, Any]) -> None:
    if (
        type(schema) is not dict
        or schema.get("$schema") != _DRAFT
        or type(schema.get("$id")) is not str
    ):
        raise SchemaValidationError("schema envelope is invalid")
    canonical_bytes(schema)
    _preflight_node(schema, root=schema, path="$", reference_stack=())


def _matches_type(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return type(instance) is bool
    if expected == "integer":
        return type(instance) is int
    if expected == "number":
        return type(instance) in {int, float} and math.isfinite(float(instance))
    if expected == "string":
        return type(instance) is str
    if expected == "array":
        return type(instance) is list
    if expected == "object":
        return type(instance) is dict
    raise SchemaValidationError(f"unsupported runtime schema type: {expected}")


def _identity(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in {int, float}:
        return ("number", Decimal(str(value)))
    if type(value) is str:
        return ("string", value)
    if type(value) is list:
        return ("array", tuple(_identity(item) for item in value))
    if type(value) is dict:
        return ("object", tuple((key, _identity(value[key])) for key in sorted(value)))
    raise SchemaValidationError("unsupported JSON identity")


def _validate(instance: Any, node: Any, *, root: Mapping[str, Any], path: str) -> None:
    if type(node) is bool:
        if node:
            return
        raise SchemaValidationError(f"{path} rejected by false schema")
    if type(node) is not dict:
        raise SchemaValidationError(f"invalid executable schema at {path}")
    if "$ref" in node:
        target, target_root = _resolve_reference(node["$ref"], root=root)
        _validate(instance, target, root=target_root, path=path)
    for branch in node.get("allOf", ()):
        _validate(instance, branch, root=root, path=path)
    if "oneOf" in node:
        matches = 0
        for branch in node["oneOf"]:
            try:
                _validate(instance, branch, root=root, path=path)
            except SchemaValidationError:
                continue
            matches += 1
        if matches != 1:
            raise SchemaValidationError(f"{path} must match exactly one oneOf branch")
    declared = node.get("type")
    if declared is not None:
        types = [declared] if type(declared) is str else declared
        if not any(_matches_type(instance, item) for item in types):
            raise SchemaValidationError(f"{path} has the wrong JSON type")
    if "const" in node and _identity(instance) != _identity(node["const"]):
        raise SchemaValidationError(f"{path} does not match const")
    if "enum" in node and all(
        _identity(instance) != _identity(value) for value in node["enum"]
    ):
        raise SchemaValidationError(f"{path} is outside the closed enum")
    if type(instance) is str:
        if "minLength" in node and len(instance) < node["minLength"]:
            raise SchemaValidationError(f"{path} is shorter than minLength")
        if "maxLength" in node and len(instance) > node["maxLength"]:
            raise SchemaValidationError(f"{path} is longer than maxLength")
        if "pattern" in node and re.fullmatch(node["pattern"], instance, re.ASCII) is None:
            raise SchemaValidationError(f"{path} does not match the canonical pattern")
        if node.get("format") == "date-time":
            if _DATE_TIME_RE.fullmatch(instance) is None:
                raise SchemaValidationError(f"{path} is not a UTC timestamp")
            try:
                datetime.fromisoformat(instance[:-1] + "+00:00")
            except ValueError as exc:
                raise SchemaValidationError(f"{path} is not a real timestamp") from exc
    if type(instance) in {int, float} and type(instance) is not bool:
        if "minimum" in node and instance < node["minimum"]:
            raise SchemaValidationError(f"{path} is below minimum")
        if "maximum" in node and instance > node["maximum"]:
            raise SchemaValidationError(f"{path} is above maximum")
    if type(instance) is list:
        if "minItems" in node and len(instance) < node["minItems"]:
            raise SchemaValidationError(f"{path} has too few items")
        if "maxItems" in node and len(instance) > node["maxItems"]:
            raise SchemaValidationError(f"{path} has too many items")
        if node.get("uniqueItems") and len({_identity(item) for item in instance}) != len(
            instance
        ):
            raise SchemaValidationError(f"{path} must contain unique items")
        for index, item in enumerate(instance):
            if "items" in node:
                _validate(item, node["items"], root=root, path=f"{path}[{index}]")
    if type(instance) is dict:
        missing = [name for name in node.get("required", ()) if name not in instance]
        if missing:
            raise SchemaValidationError(f"{path} is missing required properties: {missing}")
        properties = node.get("properties", {})
        if node.get("additionalProperties") is False:
            unknown = sorted(set(instance) - set(properties))
            if unknown:
                raise SchemaValidationError(f"{path} contains additional properties: {unknown}")
        for name, child in instance.items():
            if name in properties:
                _validate(child, properties[name], root=root, path=f"{path}.{name}")


def validate_instance_against_schema(instance: Any, schema: Mapping[str, Any]) -> None:
    validate_json_limits(instance)
    preflight_schema(schema)
    _validate(instance, schema, root=schema, path="$")


def validate_schema_version(instance: Any, version: Any) -> dict[str, Any]:
    if type(instance) is not dict:
        raise SchemaValidationError("v4 artifact must be an object")
    schema = load_packaged_json(schema_path_for_version(version))
    validate_instance_against_schema(instance, schema)
    return dict(instance)


def validate_artifact(
    instance: Any,
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None = None,
) -> Any:
    if type(instance) is not dict:
        raise SchemaValidationError("v4 artifact must be an object")
    validate_schema_version(instance, instance.get("version"))
    try:
        from .validators import validate_typed_artifact

        return validate_typed_artifact(
            instance,
            schema_checked=True,
            artifact_loader=artifact_loader,
        )
    except (CanonicalContractError, PackageResourceError) as exc:
        raise SchemaValidationError(str(exc)) from exc


def load_canonical_artifact(
    raw: bytes,
    *,
    expected_version: str | None = None,
    label: str = "v4 artifact",
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None = None,
) -> Any:
    try:
        instance = load_canonical_resource(raw, label=label)
    except CanonicalContractError as exc:
        raise SchemaValidationError(str(exc)) from exc
    if type(instance) is not dict:
        raise SchemaValidationError(f"{label} root must be an object")
    if expected_version is not None and instance.get("version") != expected_version:
        raise SchemaValidationError(f"{label} version does not match expected_version")
    return validate_artifact(
        instance,
        artifact_loader=artifact_loader,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "SchemaValidationError",
    "artifact_identity_field",
    "load_canonical_artifact",
    "preflight_schema",
    "schema_path_for_version",
    "schema_versions",
    "validate_artifact",
    "validate_instance_against_schema",
    "validate_schema_version",
]
