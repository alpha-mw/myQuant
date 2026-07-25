"""Read-only, hash-bound access to packaged v17 policy resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

from .contracts import V17ContractError
from .semantic import canonical_json_bytes, require_sha256
from .storage import file_sha256, read_json

RESOURCE_DIR = Path(__file__).with_name("resources")
SCHEMA_DIR = Path(__file__).with_name("schemas")

CANONICAL_POLICY_RESOURCE_NAMES = (
    "shadow_policy.v1.json",
    "deep_research_template.v1.json",
    "state_machine.v1.json",
    "quant_factor_set.v1.json",
)

FROZEN_POLICY_RESOURCE_SHA256S = {
    "deep_research_template.v1.json": (
        "434cf726270d5f65eb7a0d2e2f2569363b281be23c16ee08304acb6962cc6537"
    ),
    "quant_factor_set.v1.json": "670d18dd8f164f3390ee9838b626a7fd893f699042e58ab71d303c022eb47c56",
    "shadow_policy.v1.json": "c55331e1d7e67e1a958491a616ccd6e2413ba3413fb90d53ea93d61d5e979137",
    "state_machine.v1.json": "241b9c784cc3623a1d81a7a706b15abe44bb0157ccdbd3da36a15ef4ff6f60f4",
}
FROZEN_OPERATIONAL_RESOURCE_SHA256S = {
    "retirement_scan_allowlist.json": (
        "54931a0a345f91da763fa1306953d9b8ad4a92bdfdfea5195cfef5b27d2f209d"
    ),
}
FROZEN_SCHEMA_SHA256S = {
    "dataset_manifest.v1.schema.json": (
        "46372f960ea9baf424f55d3abc603c8c171e95933f72e64c06fb52ada315061e"
    ),
    "deep_research_request.v2.schema.json": (
        "e690e320aa1c9afaec5407a6beb94037609604339133a336721ad1e83de6a7ab"
    ),
    "deep_research_response.v2.schema.json": (
        "a33ce2171f0a2624d6fa861b26d7ed12282582606a97550b953d92de0c9d65cc"
    ),
    "execution_cost_policy.v1.schema.json": (
        "3e1cf5aa3e05fee0fc798c07844dcbd93ce40904f9cc0cf440485352be1c7678"
    ),
    "generation_catalog.v1.schema.json": (
        "85eaf777c72bc5ef867275e5a8f494a81ff84fc632cc1777516c1d9f0a9be4c8"
    ),
    "holdings_snapshot.v1.schema.json": (
        "3d2d8ad9a9c54ff2651c282cf5aef0c75a81a41fbbfbed3099aa85731e04a981"
    ),
    "observation_disposition.v1.schema.json": (
        "49f8d80550772f376e5beaa63e814657c5f4340e5418c81a8268e27370f73cbe"
    ),
    "portfolio_risk_policy_snapshot.v1.schema.json": (
        "d3441aa64f2b5a5fdd650fe499ebff0a6b9077b5139f2ccc07519c004ad21136"
    ),
    "pretrade_result.v1.schema.json": (
        "5bce9ff0937d259297bcee458df0cefc807886243c9197026159aa538ba3d350"
    ),
    "regime_portfolio_overlay.v1.schema.json": (
        "89a3d31ab0b1817a0fa81e34a85eac6e79190e35bc387141196d1152a28d8061"
    ),
    "shadow_output.v1.schema.json": (
        "ad5298f44b3feb998f4f7a7ae7b2c63a2240531f4bf1e847d6433a7d81a0ff61"
    ),
    "shadow_state.v1.schema.json": (
        "e724ab3bbf17616d7d1de72e05f47fd823ae030624d76c9845a5c26b728c18a5"
    ),
    "source_manifest.v2.schema.json": (
        "e11bad313af50e7453265912fadad472c3c092c03e4d1d02b69db7b9ac773b92"
    ),
    "trade_permission.v1.schema.json": (
        "f16cd9cae575f63535981752fc8d26a98835062e835fd60815dbe8016f325688"
    ),
}

_SAFE_RESOURCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}\.json$")
_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "title",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "propertyNames",
        "items",
        "enum",
        "const",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "uniqueItems",
        "allOf",
        "anyOf",
        "oneOf",
        "if",
        "then",
        "else",
    }
)
_JSON_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)


def _schema_array(value: Any, *, label: str, nonempty: bool = False) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise V17ContractError(f"{label} must be an array")
    items = list(value)
    if nonempty and not items:
        raise V17ContractError(f"{label} must be nonempty")
    return items


def _check_supported_schema_node(
    node: Any,
    *,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, Mapping):
        raise V17ContractError(f"JSON Schema node must be an object or boolean: {path}")
    unknown = sorted(set(node) - _SUPPORTED_SCHEMA_KEYWORDS)
    if unknown:
        raise V17ContractError(f"unsupported JSON Schema keywords at {path}: {unknown}")
    reference = node.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise V17ContractError(f"only local $defs references are supported: {path}")
        definition_name = reference.removeprefix("#/$defs/")
        definitions = root.get("$defs")
        if (
            not definition_name
            or "/" in definition_name
            or not isinstance(definitions, Mapping)
            or definition_name not in definitions
        ):
            raise V17ContractError(f"unresolved JSON Schema reference at {path}: {reference}")
    declared_type = node.get("type")
    if declared_type is not None:
        types = (
            [declared_type]
            if isinstance(declared_type, str)
            else _schema_array(declared_type, label=f"{path}.type", nonempty=True)
        )
        if any(
            not isinstance(item, str) or item not in _JSON_SCHEMA_TYPES for item in types
        ) or len(set(types)) != len(types):
            raise V17ContractError(f"invalid JSON Schema type declaration at {path}")
    for keyword in ("$schema", "$id", "title", "format"):
        if keyword in node and not isinstance(node[keyword], str):
            raise V17ContractError(f"JSON Schema {keyword} must be a string at {path}")
    if "pattern" in node:
        pattern = node["pattern"]
        if not isinstance(pattern, str):
            raise V17ContractError(f"JSON Schema pattern must be a string at {path}")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise V17ContractError(f"invalid JSON Schema pattern at {path}") from exc
    for keyword in (
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
    ):
        if keyword in node and (
            isinstance(node[keyword], bool)
            or not isinstance(node[keyword], int)
            or node[keyword] < 0
        ):
            raise V17ContractError(f"JSON Schema {keyword} must be nonnegative at {path}")
    for lower, upper in (
        ("minLength", "maxLength"),
        ("minProperties", "maxProperties"),
        ("minItems", "maxItems"),
    ):
        if lower in node and upper in node and node[lower] > node[upper]:
            raise V17ContractError(f"JSON Schema {lower} exceeds {upper} at {path}")
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in node and (
            isinstance(node[keyword], bool)
            or not isinstance(node[keyword], (int, float))
            or not math.isfinite(float(node[keyword]))
        ):
            raise V17ContractError(f"JSON Schema {keyword} must be finite at {path}")
    if "minimum" in node and "maximum" in node and node["minimum"] > node["maximum"]:
        raise V17ContractError(f"JSON Schema minimum exceeds maximum at {path}")
    lower_bounds = [
        (node[keyword], keyword.startswith("exclusive"))
        for keyword in ("minimum", "exclusiveMinimum")
        if keyword in node
    ]
    upper_bounds = [
        (node[keyword], keyword.startswith("exclusive"))
        for keyword in ("maximum", "exclusiveMaximum")
        if keyword in node
    ]
    if lower_bounds and upper_bounds:
        lower_value, lower_exclusive = max(lower_bounds, key=lambda item: item[0])
        upper_value, upper_exclusive = min(upper_bounds, key=lambda item: item[0])
        if lower_value > upper_value or (
            lower_value == upper_value and (lower_exclusive or upper_exclusive)
        ):
            raise V17ContractError(f"JSON Schema numeric bounds are unsatisfiable at {path}")
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise V17ContractError(f"JSON Schema uniqueItems must be boolean at {path}")
    if "enum" in node:
        enum = _schema_array(node["enum"], label=f"{path}.enum", nonempty=True)
        encoded = [canonical_json_bytes(item) for item in enum]
        if len(set(encoded)) != len(encoded):
            raise V17ContractError(f"JSON Schema enum contains duplicates at {path}")
    if "const" in node:
        canonical_json_bytes(node["const"])

    child_maps: list[tuple[str, Mapping[str, Any]]] = []
    for keyword in ("$defs", "properties"):
        if keyword in node:
            value = node[keyword]
            if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
                raise V17ContractError(f"JSON Schema {keyword} must be an object at {path}")
            child_maps.append((keyword, value))
    for keyword, children in child_maps:
        for name, child in children.items():
            _check_supported_schema_node(
                child,
                root=root,
                path=f"{path}.{keyword}.{name}",
            )
    if "required" in node:
        required = _schema_array(node["required"], label=f"{path}.required")
        if any(not isinstance(item, str) for item in required) or len(set(required)) != len(
            required
        ):
            raise V17ContractError(f"JSON Schema required must contain unique strings at {path}")
        properties = node.get("properties")
        if isinstance(properties, Mapping) and not set(required).issubset(properties):
            raise V17ContractError(f"JSON Schema required field lacks a property at {path}")
    if "additionalProperties" in node:
        additional = node["additionalProperties"]
        if not isinstance(additional, bool):
            _check_supported_schema_node(
                additional,
                root=root,
                path=f"{path}.additionalProperties",
            )
    for keyword in ("propertyNames", "items", "if", "then", "else"):
        if keyword in node:
            _check_supported_schema_node(
                node[keyword],
                root=root,
                path=f"{path}.{keyword}",
            )
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword in node:
            branches = _schema_array(node[keyword], label=f"{path}.{keyword}", nonempty=True)
            for index, branch in enumerate(branches):
                _check_supported_schema_node(
                    branch,
                    root=root,
                    path=f"{path}.{keyword}[{index}]",
                )


def assert_supported_json_schema(schema: Mapping[str, Any]) -> None:
    """Fail closed on malformed or unsupported constructs in packaged schemas.

    This dependency-free structural gate is intentionally narrower than an
    arbitrary JSON Schema engine.  Every keyword used by this package is
    checked recursively, local references are resolved, and regex/bounds are
    validated.  Instance validation still uses ``jsonschema`` when available.
    """

    if not isinstance(schema, Mapping):
        raise V17ContractError("JSON Schema root must be an object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise V17ContractError("unsupported JSON Schema dialect")
    if not isinstance(schema.get("$id"), str):
        raise V17ContractError("JSON Schema $id missing")
    canonical_json_bytes(schema)
    _check_supported_schema_node(schema, root=schema, path="$")


def _fixed_resource_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not _SAFE_RESOURCE_NAME.fullmatch(name):
        raise V17ContractError("invalid package resource name")
    path = directory / name
    if path.parent != directory:
        raise V17ContractError("package resource path escaped fixed directory")
    if path.is_symlink():
        raise V17ContractError(f"symlink package resource rejected: {name}")
    if not path.is_file():
        raise V17ContractError(f"package resource unavailable: {name}")
    return path


def resource_path(name: str) -> Path:
    return _fixed_resource_path(RESOURCE_DIR, name)


def schema_path(name: str) -> Path:
    return _fixed_resource_path(SCHEMA_DIR, name)


def resource_byte_sha256(name: str) -> str:
    return file_sha256(resource_path(name))


def schema_byte_sha256(name: str) -> str:
    return file_sha256(schema_path(name))


def _read_hash_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = require_sha256(expected_sha256, label="expected resource SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise V17ContractError(f"package resource byte SHA-256 mismatch: {path.name}")
    return read_json(path)


def load_json_resource(name: str, *, expected_sha256: str) -> dict[str, Any]:
    """Load a canonical resource only when its external byte digest matches."""

    return _read_hash_bound(resource_path(name), expected_sha256)


def load_json_schema(name: str, *, expected_sha256: str) -> dict[str, Any]:
    payload = _read_hash_bound(schema_path(name), expected_sha256)
    assert_supported_json_schema(payload)
    return payload


def assert_resource_manifest(
    manifest: Mapping[str, Any],
    *,
    resource_directory: str | Path,
) -> dict[str, str]:
    """Verify an exact name-to-byte-SHA manifest for a clean resource package."""

    if not isinstance(manifest, Mapping) or not manifest:
        raise V17ContractError("resource manifest must be a nonempty object")
    root = Path(resource_directory)
    if root.is_symlink() or not root.is_dir():
        raise V17ContractError("resource directory is unavailable or ambiguous")
    observed: dict[str, str] = {}
    for name, expected_value in sorted(manifest.items()):
        path = _fixed_resource_path(root, name)
        expected = require_sha256(expected_value, label=f"resource SHA for {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise V17ContractError(f"resource byte SHA-256 mismatch: {name}")
        # Parse now so a hash-bound but invalid JSON file still fails closed.
        read_json(path)
        observed[name] = actual
    json_entries = [item for item in root.iterdir() if item.suffix == ".json"]
    if any(item.is_symlink() or not item.is_file() for item in json_entries):
        raise V17ContractError("resource directory contains an ambiguous JSON entry")
    actual_names = {item.name for item in json_entries}
    if actual_names != set(manifest):
        raise V17ContractError(
            "resource manifest does not exactly cover the JSON resource directory"
        )
    return observed


def json_schema_validation_available() -> bool:
    """Report whether optional jsonschema validation can run, without importing it."""

    try:
        import importlib.util

        return importlib.util.find_spec("jsonschema") is not None
    except (ImportError, ValueError):
        return False


def validate_against_schema(
    instance: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """Validate with jsonschema when installed; absence is an explicit failure."""

    if not json_schema_validation_available():
        raise V17ContractError("jsonschema validator unavailable")
    try:
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(dict(schema))
        jsonschema.Draft202012Validator(dict(schema)).validate(dict(instance))
    except Exception as exc:
        raise V17ContractError("JSON Schema validation failed") from exc


def assert_frozen_package_contracts() -> dict[str, dict[str, str]]:
    """Verify the exact reviewed v17 policy and schema package.

    Request-provided digests are not authority.  These code-owned manifests
    make an unreviewed package-resource or schema edit fail closed even when a
    caller reports the new digest accurately.
    """

    if set(FROZEN_POLICY_RESOURCE_SHA256S) != set(CANONICAL_POLICY_RESOURCE_NAMES):
        raise V17ContractError("frozen policy resource name set mismatch")
    observed_resources: dict[str, str] = {}
    for name, expected in sorted(FROZEN_POLICY_RESOURCE_SHA256S.items()):
        actual = resource_byte_sha256(name)
        if actual != expected:
            raise V17ContractError(f"frozen package resource drift: {name}")
        load_json_resource(name, expected_sha256=expected)
        observed_resources[name] = actual
    observed_operational_resources: dict[str, str] = {}
    for name, expected in sorted(FROZEN_OPERATIONAL_RESOURCE_SHA256S.items()):
        actual = resource_byte_sha256(name)
        if actual != expected:
            raise V17ContractError(f"frozen operational resource drift: {name}")
        load_json_resource(name, expected_sha256=expected)
        observed_operational_resources[name] = actual
    package_resource_names = {path.name for path in RESOURCE_DIR.glob("*.json")}
    expected_resource_names = set(FROZEN_POLICY_RESOURCE_SHA256S) | set(
        FROZEN_OPERATIONAL_RESOURCE_SHA256S
    )
    if package_resource_names != expected_resource_names:
        raise V17ContractError("frozen v17 package resource name set mismatch")
    schema_names = {path.name for path in SCHEMA_DIR.glob("*.json")}
    if schema_names != set(FROZEN_SCHEMA_SHA256S):
        raise V17ContractError("frozen v17 schema name set mismatch")
    observed_schemas: dict[str, str] = {}
    for name, expected in sorted(FROZEN_SCHEMA_SHA256S.items()):
        actual = schema_byte_sha256(name)
        if actual != expected:
            raise V17ContractError(f"frozen package schema drift: {name}")
        load_json_schema(name, expected_sha256=expected)
        observed_schemas[name] = actual

    policy = load_json_resource(
        "shadow_policy.v1.json",
        expected_sha256=FROZEN_POLICY_RESOURCE_SHA256S["shadow_policy.v1.json"],
    )
    expected_pipeline = [
        "fundamental_candidate_generation",
        "codex_fundamental_deep_research",
        "quant_timing",
        "regime_portfolio_overlay",
        "pretrade_and_portfolio_optimization",
        "shadow_result",
    ]
    if (
        policy.get("version") != "17.0.0"
        or policy.get("authority") is not False
        or policy.get("production_protocol") != "v15"
        or policy.get("mode") != "research_shadow_only"
        or policy.get("pipeline") != expected_pipeline
        or any(value is not False for value in policy.get("side_effects", {}).values())
    ):
        raise V17ContractError("shadow policy authority/pipeline invariant mismatch")
    fundamental = policy.get("fundamental", {})
    forward = policy.get("forward_calibration", {})
    optimizer = policy.get("optimizer", {})
    if (
        fundamental.get("top_n") != 24
        or fundamental.get("holdings_append_without_consuming_top_n") is not True
        or fundamental.get("failed_deep_research_may_backfill") is not False
        or forward.get("horizons_open_days") != [120, 252, 378]
        or forward.get("minimum_observations") != 100
        or forward.get("minimum_cross_section_dates") != 12
        or forward.get("minimum_symbols") != 20
        or optimizer.get("permission_mask_authoritative") is not True
        or optimizer.get("may_create_permission") is not False
        or policy.get("missing_enabled_overlay_or_risk_input_terminal")
        != "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
    ):
        raise V17ContractError("shadow policy-to-code invariant mismatch")
    return {
        "resources": observed_resources,
        "operational_resources": observed_operational_resources,
        "schemas": observed_schemas,
    }


__all__ = [
    "CANONICAL_POLICY_RESOURCE_NAMES",
    "FROZEN_OPERATIONAL_RESOURCE_SHA256S",
    "FROZEN_POLICY_RESOURCE_SHA256S",
    "FROZEN_SCHEMA_SHA256S",
    "RESOURCE_DIR",
    "SCHEMA_DIR",
    "assert_resource_manifest",
    "assert_frozen_package_contracts",
    "assert_supported_json_schema",
    "json_schema_validation_available",
    "load_json_resource",
    "load_json_schema",
    "resource_byte_sha256",
    "resource_path",
    "schema_byte_sha256",
    "schema_path",
    "validate_against_schema",
]
