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
PACKAGE_MANIFEST_SHA256: Final = "0a9a43cfe0bcd9dd5036daf72fdea6187fbf290c1faf2d00ca2397bbd95a950c"
LEDGER_IMPLEMENTATION_MODULES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "quant_investor.v17_v2_contract.__init__": "v17_v2_contract/__init__.py",
        "quant_investor.v17_v2_contract.action_matrix": "v17_v2_contract/action_matrix.py",
        "quant_investor.v17_v2_contract.canonical": "v17_v2_contract/canonical.py",
        "quant_investor.v17_v2_contract.identities": "v17_v2_contract/identities.py",
        "quant_investor.v17_v2_contract.limits": "v17_v2_contract/limits.py",
        "quant_investor.v17_v2_contract.namespace": "v17_v2_contract/namespace.py",
        "quant_investor.v17_v2_contract.package_parity": "v17_v2_contract/package_parity.py",
        "quant_investor.v17_v2_contract.resources": "v17_v2_contract/resources.py",
        "quant_investor.v17_v2_contract.schema_validation": "v17_v2_contract/schema_validation.py",
        "quant_investor.v17_v2_contract.validators": "v17_v2_contract/validators.py",
        "quant_investor.v17_v2_runtime.__init__": "v17_v2_runtime/__init__.py",
        "quant_investor.v17_v2_runtime.algorithms.__init__": "v17_v2_runtime/algorithms/__init__.py",
        "quant_investor.v17_v2_runtime.algorithms._semantic": "v17_v2_runtime/algorithms/_semantic.py",
        "quant_investor.v17_v2_runtime.algorithms.deep_research": "v17_v2_runtime/algorithms/deep_research.py",
        "quant_investor.v17_v2_runtime.algorithms.forward_calibration": "v17_v2_runtime/algorithms/forward_calibration.py",
        "quant_investor.v17_v2_runtime.algorithms.fundamental_scoring": "v17_v2_runtime/algorithms/fundamental_scoring.py",
        "quant_investor.v17_v2_runtime.algorithms.optimizer": "v17_v2_runtime/algorithms/optimizer.py",
        "quant_investor.v17_v2_runtime.algorithms.permissions": "v17_v2_runtime/algorithms/permissions.py",
        "quant_investor.v17_v2_runtime.algorithms.quant_timing": "v17_v2_runtime/algorithms/quant_timing.py",
        "quant_investor.v17_v2_runtime.algorithms.regime_overlay": "v17_v2_runtime/algorithms/regime_overlay.py",
        "quant_investor.v17_v2_runtime.algorithms.transaction_cost": "v17_v2_runtime/algorithms/transaction_cost.py",
        "quant_investor.v17_v2_runtime.cli": "v17_v2_runtime/cli.py",
        "quant_investor.v17_v2_runtime.gate": "v17_v2_runtime/gate.py",
        "quant_investor.v17_v2_runtime.ledger": "v17_v2_runtime/ledger.py",
        "quant_investor.v17_v2_runtime.pipeline": "v17_v2_runtime/pipeline.py",
        "quant_investor.v17_v2_runtime.service": "v17_v2_runtime/service.py",
        "quant_investor.v17_v2_runtime.sources": "v17_v2_runtime/sources.py",
        "quant_investor.v17_v2_runtime.storage": "v17_v2_runtime/storage.py",
        "quant_investor.v17_v2_runtime.terminal": "v17_v2_runtime/terminal.py",
    }
)
PACKAGE_ASSET_SHA256S: Final[Mapping[str, str]] = MappingProxyType(
    {
        "resources/action_matrix.v1.json": "f342820aadd34005e718552e82b162b8efaee2e1046c184609d8080cf13e6434",
        "resources/dataset_record_schema_registry.v1.json": "3f33080d9782bad52911c15b29c035b1fe04af675e68928fc0b1b6f397bcbcc3",
        "resources/deep_research_template.v1.json": "15143aa020f7e78a5ac6772ab5c1125caf989bb7a2a0e9c072511f7c68871c85",
        "resources/limits.v1.json": "37a515d37fc69e83808e8e0bfcdeda9d55119863e6c7cb02b25bdb63aaef2f53",
        "resources/main_suite_runtime_policy.v1.json": "429a0ced02a047654ad5c813d70ca558036dd5e6a3e8fb142bc1cc1e7dc440e0",
        "resources/namespace_map.v1.json": "c2dc96c18e486b75a48bb3ea140b92163af0dd7fd91e91c14a26c5fba76add81",
        "resources/package_manifest.v1.json": PACKAGE_MANIFEST_SHA256,
        "resources/quant_factor_set.v1.json": "8c46282029fe07f677a1a3a7efe1241a0a2b2b817f2e6db51082a84bc0b09c3d",
        "resources/shadow_policy.v1.json": "9a1b16d1ea3f131a3842bdde2955615c3c14563cad262641d0f608542b2b662a",
        "resources/source_role_matrix.v1.json": "36a2a94699189ba1433b3679019362e51d701df9a99654f88640e0a0af4d3ddb",
        "resources/state_machine.v1.json": "6f21ea8b9b13ec4242730d65d0a7d48e1af7033713342e96b43bf3a80d60aee8",
        "schemas/action_failure_receipt.v1.schema.json": "85aa9bcc211449a8cf0573a1f34839ab2d01bc58a044ec9616a9459da0e03510",
        "schemas/dataset_manifest.v1.schema.json": "b6bc3fccbab3afbff4b961c7cf2d9ad20aaf7c427838de16455f7981de579d5f",
        "schemas/dataset_record_schema_registry.v1.schema.json": "3a747b3027e970c71e3e2c3ae4cc5d655b92208ab85fdfe334912c1936f6d24e",
        "schemas/dataset_summary.v1.schema.json": "997118874ec525ce1e2db87ac642ffd1de35aa5024b08c97387672fcd85442da",
        "schemas/deep_research_report.v1.schema.json": "2fc2f64b8474e442909cfc4a21116e7c6d60c6232bcb529c0035e7c8d00986aa",
        "schemas/deep_research_request.v1.schema.json": "3c9179f6494f08dcc3b2cef65f568bb328fc5c716e7404e8d5b42cd0d8941657",
        "schemas/deep_research_response.v1.schema.json": "11d0709ecfbe1dc83fe1d040c59eebd5c3054a5f7aa1e42472005d775e5ead9e",
        "schemas/generation_catalog.v1.schema.json": "20cf9871388afe55f1a47a43a3e51915b89d3dd8b1f482052f5008a955d332c2",
        "schemas/macro_overlay.v1.schema.json": "1f0dec2bda645498629cdd249504747ed8dda6a6683a632dfffcbc2965bafe65",
        "schemas/main_suite_runtime_policy.v1.schema.json": "e64b6c3e04d5507153d94e4c1b1358b55fbcf129c2c6291fe3021cc9e0d56553",
        "schemas/market_pointer.v1.schema.json": "3c6dee4ec7931a4190cafa6bc1354baa7412e5f2ddb15bb3f56d1e955824cc2e",
        "schemas/market_snapshot_manifest.v1.schema.json": "dbedfd788bf83d1b0e8e1742e8314e0b74d892f4f3a9e75532a71d73d0a8cf02",
        "schemas/markov_overlay.v1.schema.json": "59809f44b4f5fe9eef8b136f1b6ae275eae6b3b78b8d2b1bd7bd672fc77e7e38",
        "schemas/observation_disposition.v1.schema.json": "01a3d0e497e00b900f216f8a2597c1c5b15a22ba9ae05f8217c0923d1d585d95",
        "schemas/portfolio_output.v1.schema.json": "b05022161b6fd30bbed93d646f603b84b9a8e832efed9944c55188cc744eb84f",
        "schemas/portfolio_required_inputs.v1.schema.json": "3688189d1d82b9f66b87f06ea937f0a28fa106033705dc3a9e53b1f856665514",
        "schemas/rank_output.v1.schema.json": "a50f3c45308e08f66760dd050d8f765175bdf95e3d1a3d66363a014df23b2f99",
        "schemas/risk_policy_snapshot.v1.schema.json": "ce3850375ecdca1dc4223b0f37cc98087a1ae081a2e72232196fadb9627e1351",
        "schemas/shadow_latest_pointer.v1.schema.json": "004975f2143dc18d9f792043313f6ca0d61f0a8d1a66b7bdb204c0bd8475f6e6",
        "schemas/shadow_ledger.v1.schema.json": "fb3ebda16183f58ac4b941016f07ba09dbc72bc0a1209898f59b006d01c364e5",
        "schemas/shadow_output.v1.schema.json": "241590fa97a4e37005b32c953f3fec0ea763837c45c6c7c0edfa6708fa03a114",
        "schemas/source_binding_set.v1.schema.json": "4c1265f23b00d48842e487f3bf78e6219ddd1559c351c14d9bc7edfe1370c882",
        "schemas/source_locator.v1.schema.json": "144cc36d4b9915ea18bd4e67e0f38e962aac4ac1adac5d3fce6655f99be580ea",
        "schemas/source_manifest.v1.schema.json": "ea48f6f04c0dd52c06e9ed0aeec7fd3922cc3dad7d6be21c44595f3c94b988c6",
        "schemas/source_role_matrix.v1.schema.json": "6bcf5566e6511c9d5bb03dd18e096c4a33a36c986b8020f05b8faf2066c731dd",
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

    root = files("quant_investor")
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
