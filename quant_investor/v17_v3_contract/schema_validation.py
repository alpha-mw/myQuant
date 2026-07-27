"""Closed Draft 2020-12 JSON Schema subset and artifact dispatch for v3."""

from __future__ import annotations

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
from .resources import (
    PackageResourceError,
    load_packaged_json,
    package_resource_session,
)

PROTOCOL_VERSION: Final = "myquant.v17.v3"
_DRAFT: Final = "https://json-schema.org/draft/2020-12/schema"
_ARTIFACT_REGISTRY: Final = {
    "myquant.v17.v3.calibration-gate-inputs.v1": (
        "schemas/calibration_gate_inputs.v1.schema.json",
        "input_id",
    ),
    "myquant.v17.v3.activation-pointer.v1": (
        "schemas/activation_pointer.v1.schema.json",
        "pointer_id",
    ),
    "myquant.v17.v3.activation-receipt.v1": (
        "schemas/activation_receipt.v1.schema.json",
        "receipt_id",
    ),
    "myquant.v17.v3.branch-output.v1": (
        "schemas/branch_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.deep-output.v1": (
        "schemas/deep_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.deep-research-inputs.v1": (
        "schemas/deep_research_inputs.v1.schema.json",
        "input_id",
    ),
    "myquant.v17.v3.factor-governance-readiness.v1": (
        "schemas/factor_governance_readiness.v1.schema.json",
        "readiness_id",
    ),
    "myquant.v17.v3.formal-latest.v1": (
        "schemas/formal_latest.v1.schema.json",
        "latest_id",
    ),
    "myquant.v17.v3.formal-research-output.v1": (
        "schemas/formal_research_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.fusion-calibration-inputs.v1": (
        "schemas/fusion_calibration_inputs.v1.schema.json",
        "input_id",
    ),
    "myquant.v17.v3.fusion-calibration-receipt.v1": (
        "schemas/fusion_calibration_receipt.v1.schema.json",
        "calibration_id",
    ),
    "myquant.v17.v3.fusion-output.v1": (
        "schemas/fusion_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.fusion-promotion-receipt.v1": (
        "schemas/fusion_promotion_receipt.v1.schema.json",
        "promotion_id",
    ),
    "myquant.v17.v3.initial-pool-output.v1": (
        "schemas/initial_pool_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.ledger.v1": ("schemas/ledger.v1.schema.json", "run_id"),
    "myquant.v17.v3.portfolio-overlay.v1": (
        "schemas/portfolio_overlay.v1.schema.json",
        "overlay_id",
    ),
    "myquant.v17.v3.portfolio-output.v1": (
        "schemas/portfolio_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.pretrade-permissions.v1": (
        "schemas/pretrade_permissions.v1.schema.json",
        "permissions_id",
    ),
    "myquant.v17.v3.provisional-factor-baseline.v1": (
        "schemas/provisional_factor_baseline.v1.schema.json",
        "baseline_id",
    ),
    "myquant.v17.v3.quant-preselection-inputs.v1": (
        "schemas/quant_preselection_inputs.v1.schema.json",
        "input_id",
    ),
    "myquant.v17.v3.shadow-latest.v1": (
        "schemas/shadow_latest.v1.schema.json",
        "latest_id",
    ),
    "myquant.v17.v3.shadow-output.v1": (
        "schemas/shadow_output.v1.schema.json",
        "output_id",
    ),
    "myquant.v17.v3.source-locator.v1": (
        "schemas/source_locator.v1.schema.json",
        "locator_id",
    ),
    "myquant.v17.v3.source-manifest.v1": (
        "schemas/source_manifest.v1.schema.json",
        "manifest_id",
    ),
    "myquant.v17.v3.unpublished-evidence.v1": (
        "schemas/unpublished_evidence.v1.schema.json",
        "evidence_id",
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
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_EXTERNAL_REF_RE: Final = re.compile(
    r"^(?P<file>[a-z0-9_]+\.v1\.schema\.json)#/\$defs/(?P<name>[A-Za-z0-9_.-]+)$",
    re.ASCII,
)


class SchemaValidationError(ValueError):
    """Raised when a schema or contract artifact fails closed validation."""

    exit_code = 2


def schema_path_for_version(version: Any) -> str:
    if type(version) is not str or version not in _ARTIFACT_REGISTRY:
        raise SchemaValidationError(f"unsupported v3 artifact version: {version!r}")
    return _ARTIFACT_REGISTRY[version][0]


def artifact_identity_field(version: Any) -> str:
    if type(version) is not str or version not in _ARTIFACT_REGISTRY:
        raise SchemaValidationError(f"unsupported v3 artifact version: {version!r}")
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
        raise SchemaValidationError(f"unresolved external schema reference: {reference}")
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
        raise SchemaValidationError(f"schema node must be an object or boolean at {path}")
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
    declared_type = node.get("type")
    types: list[str] = []
    if declared_type is not None:
        if type(declared_type) is str:
            types = [declared_type]
        elif type(declared_type) is list:
            types = declared_type
        if (
            not types
            or any(type(item) is not str or item not in _JSON_TYPES for item in types)
            or len(types) != len(set(types))
        ):
            raise SchemaValidationError(f"invalid schema type declaration at {path}")
    if "object" in types and node.get("additionalProperties") is not False:
        raise SchemaValidationError(
            f"every v3 object schema must set additionalProperties=false at {path}"
        )
    if "array" in types:
        if type(node.get("uniqueItems")) is not bool:
            raise SchemaValidationError(f"every v3 array schema must declare uniqueItems at {path}")
        if type(node.get("x-ordering")) is not str or not node["x-ordering"]:
            raise SchemaValidationError(f"every v3 array schema must declare x-ordering at {path}")
    if "required" in node:
        required = node["required"]
        if (
            type(required) is not list
            or any(type(item) is not str for item in required)
            or len(required) != len(set(required))
        ):
            raise SchemaValidationError(f"schema required list is invalid at {path}")
        properties = node.get("properties")
        if type(properties) is dict and not set(required).issubset(properties):
            raise SchemaValidationError(f"required property is undeclared at {path}")
    if "properties" in node:
        properties = node["properties"]
        if type(properties) is not dict:
            raise SchemaValidationError(f"schema properties must be an object at {path}")
        for name, child in properties.items():
            _preflight_node(
                child,
                root=root,
                path=f"{path}.properties.{name}",
                reference_stack=reference_stack,
            )
    if "$defs" in node:
        definitions = node["$defs"]
        if type(definitions) is not dict:
            raise SchemaValidationError(f"schema $defs must be an object at {path}")
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
        if keyword not in node:
            continue
        branches = node[keyword]
        if type(branches) is not list or not branches:
            raise SchemaValidationError(f"schema {keyword} must be nonempty at {path}")
        for index, branch in enumerate(branches):
            _preflight_node(
                branch,
                root=root,
                path=f"{path}.{keyword}[{index}]",
                reference_stack=reference_stack,
            )
    if "pattern" in node:
        if type(node["pattern"]) is not str:
            raise SchemaValidationError(f"schema pattern must be a string at {path}")
        try:
            re.compile(node["pattern"], re.ASCII)
        except re.error as exc:
            raise SchemaValidationError(f"invalid schema pattern at {path}") from exc
    for keyword in ("minItems", "maxItems", "minLength", "maxLength"):
        if keyword in node and (type(node[keyword]) is not int or node[keyword] < 0):
            raise SchemaValidationError(f"schema {keyword} is invalid at {path}")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (
            type(node[keyword]) not in {int, float} or not math.isfinite(float(node[keyword]))
        ):
            raise SchemaValidationError(f"schema {keyword} is invalid at {path}")


def preflight_schema(schema: Mapping[str, Any]) -> None:
    if type(schema) is not dict:
        raise SchemaValidationError("schema root must be an object")
    if schema.get("$schema") != _DRAFT or type(schema.get("$id")) is not str:
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


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is dict:
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    return left == right


def _json_identity(value: Any) -> tuple[Any, ...]:
    """Return a hashable identity with JSON Schema numeric equality semantics."""

    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in {int, float}:
        return ("number", Decimal(str(value)))
    if type(value) is str:
        return ("string", value)
    if type(value) is list:
        return ("array", tuple(_json_identity(item) for item in value))
    if type(value) is dict:
        return (
            "object",
            tuple((key, _json_identity(value[key])) for key in sorted(value)),
        )
    raise SchemaValidationError(f"unsupported JSON value in uniqueItems: {type(value).__name__}")


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
    if "allOf" in node:
        for branch in node["allOf"]:
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
    declared_type = node.get("type")
    if declared_type is not None:
        types = [declared_type] if type(declared_type) is str else declared_type
        if not any(_matches_type(instance, expected) for expected in types):
            raise SchemaValidationError(f"{path} has the wrong JSON type")
    if "const" in node and not _json_equal(instance, node["const"]):
        raise SchemaValidationError(f"{path} does not match const")
    if "enum" in node and not any(_json_equal(instance, value) for value in node["enum"]):
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
                raise SchemaValidationError(f"{path} is not a second-precision UTC timestamp")
            try:
                datetime.fromisoformat(instance.removesuffix("Z") + "+00:00")
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
        if node.get("uniqueItems"):
            identities = {_json_identity(item) for item in instance}
            if len(identities) != len(instance):
                raise SchemaValidationError(f"{path} must contain unique items")
        if "items" in node:
            for index, item in enumerate(instance):
                _validate(item, node["items"], root=root, path=f"{path}[{index}]")
    if type(instance) is dict:
        required = node.get("required", ())
        missing = [name for name in required if name not in instance]
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


@package_resource_session()
def validate_instance_against_schema(instance: Any, schema: Mapping[str, Any]) -> None:
    validate_json_limits(instance)
    preflight_schema(schema)
    _validate(instance, schema, root=schema, path="$")


def validate_schema_version(instance: Any, version: Any) -> dict[str, Any]:
    if type(instance) is not dict:
        raise SchemaValidationError("v3 artifact must be an object")
    schema = load_packaged_json(schema_path_for_version(version))
    validate_instance_against_schema(instance, schema)
    return dict(instance)


def validate_artifact(instance: Any) -> Any:
    """Validate schema, semantic SHA, and cross-document v3 invariants."""

    if type(instance) is not dict:
        raise SchemaValidationError("v3 artifact must be an object")
    version = instance.get("version")
    validate_schema_version(instance, version)
    try:
        from .validators import validate_typed_artifact

        return validate_typed_artifact(instance)
    except (CanonicalContractError, PackageResourceError) as exc:
        raise SchemaValidationError(str(exc)) from exc


def load_canonical_artifact(
    raw: bytes,
    *,
    expected_version: str | None = None,
    label: str = "v3 artifact",
) -> Any:
    """Parse exact canonical stored bytes, then run schema and typed validation."""

    try:
        instance = load_canonical_resource(raw, label=label)
    except CanonicalContractError as exc:
        raise SchemaValidationError(str(exc)) from exc
    if type(instance) is not dict:
        raise SchemaValidationError(f"{label} root must be an object")
    if expected_version is not None:
        schema_path_for_version(expected_version)
        if instance.get("version") != expected_version:
            raise SchemaValidationError(f"{label} version does not match expected_version")
    return validate_artifact(instance)


__all__ = [
    "PROTOCOL_VERSION",
    "SchemaValidationError",
    "preflight_schema",
    "load_canonical_artifact",
    "schema_path_for_version",
    "schema_versions",
    "validate_artifact",
    "validate_instance_against_schema",
    "validate_schema_version",
]
