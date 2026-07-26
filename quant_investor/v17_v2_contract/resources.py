"""Hash-bound access to protocol-v2 package resources.

The package manifest cannot contain its own byte hash without becoming
circular.  This module is therefore the independent source of truth for the
manifest hash and the exact packaged JSON inventory.  It performs no directory
discovery and grants no runtime authority.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Final, Mapping

from .canonical import CanonicalContractError, load_canonical_resource

PACKAGE_MANIFEST_PATH: Final = "resources/package_manifest.v1.json"
PACKAGE_MANIFEST_SHA256: Final = "6ca4956bd12a8d0908a3351182e0d97e1240032cc082f38005d80380c76e2ffe"
LEDGER_IMPLEMENTATION_MODULES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "quant_investor.v17_v2_contract.__init__": "__init__.py",
        "quant_investor.v17_v2_contract.action_matrix": "action_matrix.py",
        "quant_investor.v17_v2_contract.canonical": "canonical.py",
        "quant_investor.v17_v2_contract.identities": "identities.py",
        "quant_investor.v17_v2_contract.limits": "limits.py",
        "quant_investor.v17_v2_contract.namespace": "namespace.py",
        "quant_investor.v17_v2_contract.package_parity": "package_parity.py",
        "quant_investor.v17_v2_contract.resources": "resources.py",
        "quant_investor.v17_v2_contract.schema_validation": "schema_validation.py",
        "quant_investor.v17_v2_contract.validators": "validators.py",
    }
)
PACKAGE_ASSET_SHA256S: Final[Mapping[str, str]] = MappingProxyType(
    {
        "resources/action_matrix.v1.json": (
            "f342820aadd34005e718552e82b162b8efaee2e1046c184609d8080cf13e6434"
        ),
        "resources/deep_research_template.v1.json": (
            "15143aa020f7e78a5ac6772ab5c1125caf989bb7a2a0e9c072511f7c68871c85"
        ),
        "resources/limits.v1.json": (
            "37a515d37fc69e83808e8e0bfcdeda9d55119863e6c7cb02b25bdb63aaef2f53"
        ),
        "resources/main_suite_runtime_policy.v1.json": (
            "429a0ced02a047654ad5c813d70ca558036dd5e6a3e8fb142bc1cc1e7dc440e0"
        ),
        "resources/namespace_map.v1.json": (
            "c2dc96c18e486b75a48bb3ea140b92163af0dd7fd91e91c14a26c5fba76add81"
        ),
        PACKAGE_MANIFEST_PATH: PACKAGE_MANIFEST_SHA256,
        "resources/quant_factor_set.v1.json": (
            "8c46282029fe07f677a1a3a7efe1241a0a2b2b817f2e6db51082a84bc0b09c3d"
        ),
        "resources/shadow_policy.v1.json": (
            "9a1b16d1ea3f131a3842bdde2955615c3c14563cad262641d0f608542b2b662a"
        ),
        "resources/source_role_matrix.v1.json": (
            "c67ada1dab8c4692b97309f33fc30b658543c4fa9b2a162a71caacdd9abbcca9"
        ),
        "resources/state_machine.v1.json": (
            "6f21ea8b9b13ec4242730d65d0a7d48e1af7033713342e96b43bf3a80d60aee8"
        ),
        "schemas/action_failure_receipt.v1.schema.json": (
            "85aa9bcc211449a8cf0573a1f34839ab2d01bc58a044ec9616a9459da0e03510"
        ),
        "schemas/dataset_manifest.v1.schema.json": (
            "b6bc3fccbab3afbff4b961c7cf2d9ad20aaf7c427838de16455f7981de579d5f"
        ),
        "schemas/dataset_summary.v1.schema.json": (
            "997118874ec525ce1e2db87ac642ffd1de35aa5024b08c97387672fcd85442da"
        ),
        "schemas/deep_research_report.v1.schema.json": (
            "2fc2f64b8474e442909cfc4a21116e7c6d60c6232bcb529c0035e7c8d00986aa"
        ),
        "schemas/deep_research_request.v1.schema.json": (
            "3c9179f6494f08dcc3b2cef65f568bb328fc5c716e7404e8d5b42cd0d8941657"
        ),
        "schemas/deep_research_response.v1.schema.json": (
            "11d0709ecfbe1dc83fe1d040c59eebd5c3054a5f7aa1e42472005d775e5ead9e"
        ),
        "schemas/generation_catalog.v1.schema.json": (
            "20cf9871388afe55f1a47a43a3e51915b89d3dd8b1f482052f5008a955d332c2"
        ),
        "schemas/main_suite_runtime_policy.v1.schema.json": (
            "e64b6c3e04d5507153d94e4c1b1358b55fbcf129c2c6291fe3021cc9e0d56553"
        ),
        "schemas/observation_disposition.v1.schema.json": (
            "01a3d0e497e00b900f216f8a2597c1c5b15a22ba9ae05f8217c0923d1d585d95"
        ),
        "schemas/shadow_latest_pointer.v1.schema.json": (
            "004975f2143dc18d9f792043313f6ca0d61f0a8d1a66b7bdb204c0bd8475f6e6"
        ),
        "schemas/shadow_ledger.v1.schema.json": (
            "fb3ebda16183f58ac4b941016f07ba09dbc72bc0a1209898f59b006d01c364e5"
        ),
        "schemas/shadow_output.v1.schema.json": (
            "b7346c56b73d85e42c145a690c728c125ea12633e4b6e13babda5036820e6f9b"
        ),
        "schemas/source_binding_set.v1.schema.json": (
            "4c1265f23b00d48842e487f3bf78e6219ddd1559c351c14d9bc7edfe1370c882"
        ),
        "schemas/source_locator.v1.schema.json": (
            "144cc36d4b9915ea18bd4e67e0f38e962aac4ac1adac5d3fce6655f99be580ea"
        ),
        "schemas/source_manifest.v1.schema.json": (
            "ea48f6f04c0dd52c06e9ed0aeec7fd3922cc3dad7d6be21c44595f3c94b988c6"
        ),
        "schemas/source_role_matrix.v1.schema.json": (
            "08c13ea933799dbc1e9425f8ec10fbe91a2dda6a98ddf499c2a4bf2fed6be350"
        ),
    }
)


class PackageResourceError(RuntimeError):
    """Raised when a packaged contract asset is unknown, absent, or changed."""

    exit_code = 2


def package_asset_paths() -> tuple[str, ...]:
    """Return the exact frozen asset inventory in ASCII path order."""

    return tuple(sorted(PACKAGE_ASSET_SHA256S))


def read_packaged_asset(relative_path: str) -> bytes:
    """Read one allowlisted asset and verify its complete stored byte hash."""

    if type(relative_path) is not str or relative_path not in PACKAGE_ASSET_SHA256S:
        raise PackageResourceError(f"unknown protocol-v2 package asset: {relative_path!r}")
    target = files(__package__)
    for part in relative_path.split("/"):
        target = target.joinpath(part)
    try:
        raw = target.read_bytes()
    except (OSError, TypeError) as exc:
        raise PackageResourceError(
            f"protocol-v2 package asset is unreadable: {relative_path}"
        ) from exc
    observed = hashlib.sha256(raw).hexdigest()
    expected = PACKAGE_ASSET_SHA256S[relative_path]
    if observed != expected:
        raise PackageResourceError(
            f"protocol-v2 package asset byte SHA-256 mismatch: {relative_path}"
        )
    return raw


def load_packaged_json(relative_path: str) -> dict[str, Any]:
    """Load one hash-bound asset and require exact canonical stored JSON."""

    try:
        payload = load_canonical_resource(
            read_packaged_asset(relative_path),
            label=relative_path,
        )
    except CanonicalContractError as exc:
        raise PackageResourceError(str(exc)) from exc
    if type(payload) is not dict:
        raise PackageResourceError(
            f"protocol-v2 package asset root must be an object: {relative_path}"
        )
    return payload


def load_package_manifest() -> dict[str, Any]:
    """Load the independently hash-bound package manifest."""

    return load_packaged_json(PACKAGE_MANIFEST_PATH)


def expected_ledger_contract_bindings() -> dict[str, Any]:
    """Return the exact manifest-derived resource and schema ledger bindings."""

    manifest = load_package_manifest()
    resources = manifest.get("resources")
    schemas = manifest.get("schemas")
    if type(resources) is not list or type(schemas) is not list:
        raise PackageResourceError("package manifest binding inventories are invalid")
    resource_bindings: list[dict[str, str]] = []
    for index, value in enumerate(resources):
        if type(value) is not dict or set(value) != {
            "byte_sha256",
            "relative_path",
            "resource_version",
        }:
            raise PackageResourceError(
                f"package manifest resource binding is invalid at index {index}"
            )
        relative_path = value["relative_path"]
        if (
            type(relative_path) is not str
            or PACKAGE_ASSET_SHA256S.get(relative_path) != value["byte_sha256"]
        ):
            raise PackageResourceError(
                f"package manifest resource binding is not frozen at index {index}"
            )
        resource_bindings.append(
            {
                "binding_id": str(value["resource_version"]),
                "relative_path": relative_path,
                "byte_sha256": str(value["byte_sha256"]),
            }
        )
    schema_bindings: list[dict[str, str]] = []
    for index, value in enumerate(schemas):
        if type(value) is not dict or set(value) != {
            "artifact_version",
            "byte_sha256",
            "relative_path",
            "schema_id",
        }:
            raise PackageResourceError(
                f"package manifest schema binding is invalid at index {index}"
            )
        relative_path = value["relative_path"]
        if (
            type(relative_path) is not str
            or PACKAGE_ASSET_SHA256S.get(relative_path) != value["byte_sha256"]
        ):
            raise PackageResourceError(
                f"package manifest schema binding is not frozen at index {index}"
            )
        schema_bindings.append(
            {
                "binding_id": str(value["schema_id"]),
                "relative_path": relative_path,
                "byte_sha256": str(value["byte_sha256"]),
            }
        )
    resource_bindings.sort(
        key=lambda row: (
            row["binding_id"],
            row["relative_path"],
            row["byte_sha256"],
        )
    )
    schema_bindings.sort(
        key=lambda row: (
            row["binding_id"],
            row["relative_path"],
            row["byte_sha256"],
        )
    )
    return {
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "resource_bindings": resource_bindings,
        "schema_bindings": schema_bindings,
    }


def expected_ledger_implementation_bindings() -> list[dict[str, str]]:
    """Hash the exact package-owned Python implementation inventory."""

    root = files(__package__)
    rows: list[dict[str, str]] = []
    for module_id, relative_path in LEDGER_IMPLEMENTATION_MODULES.items():
        target = root
        for part in relative_path.split("/"):
            target = target.joinpath(part)
        try:
            raw = target.read_bytes()
        except (OSError, TypeError) as exc:
            raise PackageResourceError(
                f"ledger implementation module is unreadable: {relative_path}"
            ) from exc
        rows.append(
            {
                "module_id": module_id,
                "relative_path": relative_path,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    rows.sort(
        key=lambda row: (
            row["module_id"],
            row["relative_path"],
            row["byte_sha256"],
        )
    )
    return rows


def verify_packaged_assets() -> dict[str, str]:
    """Read and verify every frozen asset without mutating package state."""

    return {
        relative_path: hashlib.sha256(read_packaged_asset(relative_path)).hexdigest()
        for relative_path in package_asset_paths()
    }


__all__ = [
    "LEDGER_IMPLEMENTATION_MODULES",
    "PACKAGE_ASSET_SHA256S",
    "PACKAGE_MANIFEST_PATH",
    "PACKAGE_MANIFEST_SHA256",
    "PackageResourceError",
    "expected_ledger_contract_bindings",
    "expected_ledger_implementation_bindings",
    "load_package_manifest",
    "load_packaged_json",
    "package_asset_paths",
    "read_packaged_asset",
    "verify_packaged_assets",
]
