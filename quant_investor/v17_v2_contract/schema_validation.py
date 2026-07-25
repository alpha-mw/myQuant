"""Hash-bound JSON Schema execution for protocol-v2 contract documents.

The repository intentionally does not depend on a general JSON Schema package.
This module therefore implements the closed Draft 2020-12 keyword subset used
by the 15 frozen protocol-v2 schemas.  Schema preflight rejects every keyword
outside that subset, so a future schema cannot silently exceed this executor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import math
import re
from typing import Any, Final, TypeVar

from .canonical import (
    CanonicalContractError,
    canonical_json_bytes,
    load_canonical_resource,
)
from .resources import load_packaged_json

T = TypeVar("T")

_DRAFT_2020_12: Final = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_PATH_BY_VERSION: Final = {
    "myquant.v17.v2.action-failure-receipt.v1": ("schemas/action_failure_receipt.v1.schema.json"),
    "myquant.v17.v2.dataset-manifest.v1": ("schemas/dataset_manifest.v1.schema.json"),
    "myquant.v17.v2.dataset-summary.v1": ("schemas/dataset_summary.v1.schema.json"),
    "myquant.v17.v2.deep-research-report.v1": ("schemas/deep_research_report.v1.schema.json"),
    "myquant.v17.v2.deep-research-request.v1": ("schemas/deep_research_request.v1.schema.json"),
    "myquant.v17.v2.deep-research-response.v1": ("schemas/deep_research_response.v1.schema.json"),
    "myquant.v17.v2.generation-catalog.v1": ("schemas/generation_catalog.v1.schema.json"),
    "myquant.v17.v2.observation-disposition.v1": ("schemas/observation_disposition.v1.schema.json"),
    "myquant.v17.v2.shadow-latest-pointer.v1": ("schemas/shadow_latest_pointer.v1.schema.json"),
    "myquant.v17.v2.shadow-ledger.v1": ("schemas/shadow_ledger.v1.schema.json"),
    "myquant.v17.v2.shadow-output.v1": ("schemas/shadow_output.v1.schema.json"),
    "myquant.v17.v2.source-binding-set.v1": ("schemas/source_binding_set.v1.schema.json"),
    "myquant.v17.v2.source-locator.v1": ("schemas/source_locator.v1.schema.json"),
    "myquant.v17.v2.source-manifest.v1": ("schemas/source_manifest.v1.schema.json"),
    "myquant.v17.v2.source-role-matrix.v1": ("schemas/source_role_matrix.v1.schema.json"),
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
        "else",
        "enum",
        "format",
        "if",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "oneOf",
        "pattern",
        "properties",
        "propertyNames",
        "required",
        "then",
        "title",
        "type",
        "uniqueItems",
        "x-canonical-order",
    }
)
_JSON_TYPES: Final = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_DATE_TIME_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?"
    r"(?:Z|[+-][0-9]{2}:[0-9]{2})$",
    re.ASCII,
)


class SchemaValidationError(ValueError):
    """Raised when a schema or contract instance fails closed validation."""

    exit_code = 2


def schema_path_for_version(version: str) -> str:
    """Return the sole frozen schema path for one artifact version."""

    if type(version) is not str or version not in _SCHEMA_PATH_BY_VERSION:
        raise SchemaValidationError(f"unsupported contract version: {version!r}")
    return _SCHEMA_PATH_BY_VERSION[version]


def _schema_array(value: Any, *, label: str, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise SchemaValidationError(f"{label} must be a {qualifier}array")
    return value


def _preflight_node(node: Any, *, root: Mapping[str, Any], path: str) -> None:
    if type(node) is bool:
        return
    if type(node) is not dict:
        raise SchemaValidationError(f"schema node must be an object or boolean at {path}")
    unknown = sorted(set(node) - _SUPPORTED_KEYWORDS)
    if unknown:
        raise SchemaValidationError(f"unsupported schema keywords at {path}: {unknown}")

    reference = node.get("$ref")
    if reference is not None:
        if (
            type(reference) is not str
            or re.fullmatch(r"#/\$defs/[A-Za-z0-9_.-]+", reference, re.ASCII) is None
        ):
            raise SchemaValidationError(f"unsupported schema reference at {path}")
        name = reference.removeprefix("#/$defs/")
        definitions = root.get("$defs")
        if type(definitions) is not dict or name not in definitions:
            raise SchemaValidationError(f"unresolved schema reference at {path}: {reference}")

    declared_type = node.get("type")
    if declared_type is not None:
        types = (
            [declared_type]
            if type(declared_type) is str
            else _schema_array(declared_type, label=f"{path}.type", nonempty=True)
        )
        if any(type(item) is not str or item not in _JSON_TYPES for item in types) or len(
            types
        ) != len(set(types)):
            raise SchemaValidationError(f"invalid schema type declaration at {path}")

    for keyword in ("$schema", "$id", "title", "format", "x-canonical-order"):
        if keyword in node and type(node[keyword]) is not str:
            raise SchemaValidationError(f"schema {keyword} must be a string at {path}")
    if "format" in node and node["format"] != "date-time":
        raise SchemaValidationError(f"unsupported schema format at {path}")
    if "pattern" in node:
        if type(node["pattern"]) is not str:
            raise SchemaValidationError(f"schema pattern must be a string at {path}")
        try:
            re.compile(node["pattern"], re.ASCII)
        except re.error as exc:
            raise SchemaValidationError(f"invalid schema pattern at {path}") from exc

    for keyword in (
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
    ):
        if keyword in node and (type(node[keyword]) is not int or node[keyword] < 0):
            raise SchemaValidationError(f"schema {keyword} must be a nonnegative integer at {path}")
    for lower, upper in (
        ("minItems", "maxItems"),
        ("minLength", "maxLength"),
        ("minProperties", "maxProperties"),
    ):
        if lower in node and upper in node and node[lower] > node[upper]:
            raise SchemaValidationError(f"schema {lower} exceeds {upper} at {path}")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (
            type(node[keyword]) not in {int, float} or not math.isfinite(float(node[keyword]))
        ):
            raise SchemaValidationError(f"schema {keyword} must be finite at {path}")
    if "minimum" in node and "maximum" in node and node["minimum"] > node["maximum"]:
        raise SchemaValidationError(f"schema minimum exceeds maximum at {path}")
    if "uniqueItems" in node and type(node["uniqueItems"]) is not bool:
        raise SchemaValidationError(f"schema uniqueItems must be boolean at {path}")

    if "enum" in node:
        values = _schema_array(node["enum"], label=f"{path}.enum", nonempty=True)
        if any(
            _json_equal(left, right)
            for index, left in enumerate(values)
            for right in values[:index]
        ):
            raise SchemaValidationError(f"schema enum contains duplicates at {path}")
    if "const" in node:
        canonical_json_bytes(node["const"])

    for keyword in ("$defs", "properties"):
        if keyword not in node:
            continue
        children = node[keyword]
        if type(children) is not dict or any(type(name) is not str for name in children):
            raise SchemaValidationError(f"schema {keyword} must be an object at {path}")
        for name, child in children.items():
            _preflight_node(child, root=root, path=f"{path}.{keyword}.{name}")

    if "required" in node:
        required = _schema_array(node["required"], label=f"{path}.required")
        if any(type(item) is not str for item in required) or len(required) != len(set(required)):
            raise SchemaValidationError(f"schema required must contain unique strings at {path}")
        properties = node.get("properties")
        if type(properties) is dict and not set(required).issubset(properties):
            raise SchemaValidationError(f"schema required field lacks property at {path}")

    if "additionalProperties" in node:
        additional = node["additionalProperties"]
        if type(additional) is not bool:
            _preflight_node(
                additional,
                root=root,
                path=f"{path}.additionalProperties",
            )
    for keyword in ("propertyNames", "items", "if", "then", "else"):
        if keyword in node:
            _preflight_node(node[keyword], root=root, path=f"{path}.{keyword}")
    for keyword in ("allOf", "oneOf"):
        if keyword not in node:
            continue
        branches = _schema_array(node[keyword], label=f"{path}.{keyword}", nonempty=True)
        for index, branch in enumerate(branches):
            _preflight_node(
                branch,
                root=root,
                path=f"{path}.{keyword}[{index}]",
            )


def preflight_packaged_schema(schema: Mapping[str, Any]) -> None:
    """Reject an unsupported or malformed packaged schema before execution."""

    if type(schema) is not dict:
        raise SchemaValidationError("schema root must be an object")
    if schema.get("$schema") != _DRAFT_2020_12:
        raise SchemaValidationError("unsupported JSON Schema dialect")
    if type(schema.get("$id")) is not str:
        raise SchemaValidationError("schema $id is missing")
    canonical_json_bytes(schema)
    _preflight_node(schema, root=schema, path="$")


def validate_instance_against_schema(
    instance: Any,
    schema: Mapping[str, Any],
) -> None:
    """Preflight and execute one closed-subset Draft 2020-12 schema."""

    preflight_packaged_schema(schema)
    _validate_instance(instance, schema, root=schema, path="$")


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is dict:
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    return left == right


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


def _valid_date_time(value: str) -> bool:
    if _DATE_TIME_RE.fullmatch(value) is None:
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_instance(
    instance: Any,
    node: Any,
    *,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if type(node) is bool:
        if node:
            return
        raise SchemaValidationError(f"{path} is rejected by false schema")
    if type(node) is not dict:
        raise SchemaValidationError(f"invalid executable schema node at {path}")

    reference = node.get("$ref")
    if reference is not None:
        definition = root["$defs"][reference.removeprefix("#/$defs/")]
        _validate_instance(instance, definition, root=root, path=path)

    if "type" in node:
        declared = node["type"]
        types = [declared] if type(declared) is str else declared
        if not any(_matches_type(instance, expected) for expected in types):
            raise SchemaValidationError(f"{path} has an invalid JSON type")
    if "const" in node and not _json_equal(instance, node["const"]):
        raise SchemaValidationError(f"{path} does not match const")
    if "enum" in node and not any(_json_equal(instance, item) for item in node["enum"]):
        raise SchemaValidationError(f"{path} is outside enum")

    for branch in node.get("allOf", []):
        _validate_instance(instance, branch, root=root, path=path)
    if "oneOf" in node:
        matches = 0
        for branch in node["oneOf"]:
            try:
                _validate_instance(instance, branch, root=root, path=path)
            except SchemaValidationError:
                continue
            matches += 1
        if matches != 1:
            raise SchemaValidationError(f"{path} must match exactly one oneOf branch")
    if "if" in node:
        try:
            _validate_instance(instance, node["if"], root=root, path=path)
        except SchemaValidationError:
            if "else" in node:
                _validate_instance(instance, node["else"], root=root, path=path)
        else:
            if "then" in node:
                _validate_instance(instance, node["then"], root=root, path=path)

    if type(instance) is dict:
        required = node.get("required", [])
        missing = sorted(set(required) - set(instance))
        if missing:
            raise SchemaValidationError(f"{path} is missing required keys: {missing}")
        if "minProperties" in node and len(instance) < node["minProperties"]:
            raise SchemaValidationError(f"{path} has too few properties")
        if "maxProperties" in node and len(instance) > node["maxProperties"]:
            raise SchemaValidationError(f"{path} has too many properties")
        if "propertyNames" in node:
            for key in instance:
                _validate_instance(
                    key,
                    node["propertyNames"],
                    root=root,
                    path=f"{path}.<property-name>",
                )
        properties = node.get("properties", {})
        for key, child_schema in properties.items():
            if key in instance:
                _validate_instance(
                    instance[key],
                    child_schema,
                    root=root,
                    path=f"{path}.{key}",
                )
        additional = node.get("additionalProperties", True)
        for key in set(instance) - set(properties):
            if additional is False:
                raise SchemaValidationError(f"{path} has additional property: {key}")
            if type(additional) is dict:
                _validate_instance(
                    instance[key],
                    additional,
                    root=root,
                    path=f"{path}.{key}",
                )

    if type(instance) is list:
        if "minItems" in node and len(instance) < node["minItems"]:
            raise SchemaValidationError(f"{path} has too few items")
        if "maxItems" in node and len(instance) > node["maxItems"]:
            raise SchemaValidationError(f"{path} has too many items")
        if node.get("uniqueItems") is True:
            for index, item in enumerate(instance):
                if any(_json_equal(item, prior) for prior in instance[:index]):
                    raise SchemaValidationError(f"{path} has duplicate items")
        if "items" in node:
            for index, item in enumerate(instance):
                _validate_instance(
                    item,
                    node["items"],
                    root=root,
                    path=f"{path}[{index}]",
                )

    if type(instance) is str:
        if "minLength" in node and len(instance) < node["minLength"]:
            raise SchemaValidationError(f"{path} is too short")
        if "maxLength" in node and len(instance) > node["maxLength"]:
            raise SchemaValidationError(f"{path} is too long")
        if "pattern" in node and re.search(node["pattern"], instance, re.ASCII) is None:
            raise SchemaValidationError(f"{path} does not match pattern")
        if node.get("format") == "date-time" and not _valid_date_time(instance):
            raise SchemaValidationError(f"{path} is not a valid date-time")

    if type(instance) in {int, float} and type(instance) is not bool:
        if not math.isfinite(float(instance)):
            raise SchemaValidationError(f"{path} is not finite")
        if "minimum" in node and instance < node["minimum"]:
            raise SchemaValidationError(f"{path} is below minimum")
        if "maximum" in node and instance > node["maximum"]:
            raise SchemaValidationError(f"{path} is above maximum")


def validate_mapping_against_packaged_schema(
    document: Mapping[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    """Validate one in-memory object against its hash-bound packaged schema."""

    if type(document) is not dict:
        raise SchemaValidationError("contract document root must be an object")
    schema_path = schema_path_for_version(expected_version)
    schema = load_packaged_json(schema_path)
    expected_schema_id = expected_version.removesuffix(".v1") + ".schema.v1"
    if schema.get("$id") != expected_schema_id or schema.get("properties", {}).get("version") != {
        "const": expected_version
    }:
        raise SchemaValidationError("packaged schema identity mismatch")
    validate_instance_against_schema(document, schema)
    return dict(document)


def validate_canonical_contract_bytes(
    raw: bytes,
    *,
    expected_version: str,
    cross_document_validator: Callable[[Mapping[str, Any]], T],
) -> T:
    """Run the mandatory canonical, packaged-schema, and relationship checks."""

    if cross_document_validator is None:
        raise SchemaValidationError("cross-document validator is required for acceptance")
    validated = validate_canonical_schema_bytes(
        raw,
        expected_version=expected_version,
    )
    return cross_document_validator(validated)


def validate_canonical_schema_bytes(
    raw: bytes,
    *,
    expected_version: str,
) -> dict[str, Any]:
    """Inspect canonical bytes against the packaged schema without accepting them.

    This helper deliberately omits relationship validation and therefore must
    not be used as an acceptance or publication gate.
    """

    try:
        document = load_canonical_resource(raw, label=expected_version)
    except CanonicalContractError as exc:
        raise SchemaValidationError(str(exc)) from exc
    if type(document) is not dict:
        raise SchemaValidationError("contract document root must be an object")
    return validate_mapping_against_packaged_schema(
        document,
        expected_version=expected_version,
    )


def packaged_schema_versions() -> tuple[str, ...]:
    """Return the exact supported artifact-version inventory."""

    return tuple(sorted(_SCHEMA_PATH_BY_VERSION))


__all__ = [
    "SchemaValidationError",
    "packaged_schema_versions",
    "preflight_packaged_schema",
    "schema_path_for_version",
    "validate_canonical_contract_bytes",
    "validate_canonical_schema_bytes",
    "validate_instance_against_schema",
    "validate_mapping_against_packaged_schema",
]
