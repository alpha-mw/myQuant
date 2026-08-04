#!/usr/bin/env python3
"""Deterministically seal V17 v5 resources and refresh closed manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from quant_investor.v17_v5_contract.canonical import (
    canonical_resource_bytes,
    seal_semantic,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "quant_investor/v17_v5_contract"
RUNTIME = ROOT / "quant_investor/v17_v5_runtime"
QUANT_INVESTOR = ROOT / "quant_investor"
NO_AUTHORITY = {
    "broker": False,
    "canary": False,
    "execution": False,
    "factor_governance_write": False,
    "formal_activation": False,
    "formal_research_publication": False,
    "llm": False,
    "order": False,
    "portfolio": False,
    "promotion": False,
    "provider": False,
    "research_runtime_default": False,
    "selector": False,
    "trade": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_resource_bytes(seal_semantic(value)))


def _seal_existing_resource(path: Path) -> None:
    payload = json.loads(path.read_bytes())
    if type(payload) is not dict:
        raise RuntimeError(f"resource is not an object: {path}")
    payload.pop("semantic_sha256", None)
    _write_sealed(path, payload)


def _replace_digest(source: str, name: str, digest: str) -> str:
    pattern = rf"{name}: Final = " rf'(?:\(\n    )?"[0-9a-f]{{64}}"(?:\n\))?'
    if name == "PACKAGE_MANIFEST_SHA256":
        replacement = f'{name}: Final = "{digest}"'
    else:
        replacement = f'{name}: Final = (\n    "{digest}"\n)'
    updated, count = re.subn(pattern, replacement, source)
    if count != 1:
        raise RuntimeError(f"cannot update {name}")
    return updated


def main() -> None:
    resources_dir = CONTRACT / "resources"
    schemas_dir = CONTRACT / "schemas"
    for path in sorted(resources_dir.glob("*.json")):
        if path.name not in {
            "package_manifest.v1.json",
            "runtime_build_manifest.v1.json",
        }:
            _seal_existing_resource(path)
    for path in sorted(schemas_dir.glob("*.json")):
        path.write_bytes(canonical_resource_bytes(json.loads(path.read_bytes())))

    runtime_manifest = resources_dir / "runtime_build_manifest.v1.json"
    _write_sealed(
        runtime_manifest,
        {
            "array_order_semantics": {
                "/sources": "relative_path ASCII ascending",
            },
            "authority": NO_AUTHORITY,
            "protocol_version": "myquant.v17.v5",
            "sources": [
                {
                    "byte_sha256": _sha(path),
                    "relative_path": path.relative_to(QUANT_INVESTOR).as_posix(),
                }
                for path in sorted(RUNTIME.glob("*.py"))
            ],
            "version": "myquant.v17.v5.runtime-build-manifest.v1",
        },
    )

    asset_paths = sorted(
        [
            *[
                path
                for path in resources_dir.glob("*.json")
                if path.name != "package_manifest.v1.json"
            ],
            *list(schemas_dir.glob("*.json")),
        ],
        key=lambda path: path.relative_to(CONTRACT).as_posix(),
    )
    assets: list[dict[str, str]] = []
    for path in asset_paths:
        payload = json.loads(path.read_bytes())
        artifact_id = payload.get("artifact_id") or payload.get("version") or payload.get("$id")
        if type(artifact_id) is not str:
            raise RuntimeError(f"asset has no identity: {path}")
        assets.append(
            {
                "artifact_id": artifact_id,
                "byte_sha256": _sha(path),
                "relative_path": path.relative_to(CONTRACT).as_posix(),
            }
        )
    package_manifest = resources_dir / "package_manifest.v1.json"
    _write_sealed(
        package_manifest,
        {
            "array_order_semantics": {
                "/assets": "relative_path ASCII ascending",
                "/source_paths": "relative_path ASCII ascending",
            },
            "assets": assets,
            "authority": NO_AUTHORITY,
            "protocol_version": "myquant.v17.v5",
            "self_binding": {
                "byte_sha256_source": (
                    "quant_investor.v17_v5_contract.resources." "PACKAGE_MANIFEST_SHA256"
                ),
                "relative_path": "resources/package_manifest.v1.json",
            },
            "source_paths": sorted(path.name for path in CONTRACT.glob("*.py")),
            "version": "myquant.v17.v5.package-manifest.v1",
        },
    )

    validators = CONTRACT / "validators.py"
    source = validators.read_text(encoding="utf-8")
    compatibility_v1 = resources_dir / "v4_compatibility_policy.v1.json"
    compatibility_v1_payload = json.loads(compatibility_v1.read_bytes())
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V1_BYTE_SHA256",
        _sha(compatibility_v1),
    )
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V1_SEMANTIC_SHA256",
        compatibility_v1_payload["semantic_sha256"],
    )
    compatibility_v2 = resources_dir / "v4_compatibility_policy.v2.json"
    compatibility_v2_payload = json.loads(compatibility_v2.read_bytes())
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V2_BYTE_SHA256",
        _sha(compatibility_v2),
    )
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V2_SEMANTIC_SHA256",
        compatibility_v2_payload["semantic_sha256"],
    )
    compatibility_v3 = resources_dir / "v4_compatibility_policy.v3.json"
    compatibility_v3_payload = json.loads(compatibility_v3.read_bytes())
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V3_BYTE_SHA256",
        _sha(compatibility_v3),
    )
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_V3_SEMANTIC_SHA256",
        compatibility_v3_payload["semantic_sha256"],
    )
    compatibility = resources_dir / "v4_compatibility_policy.v4.json"
    compatibility_payload = json.loads(compatibility.read_bytes())
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_BYTE_SHA256",
        _sha(compatibility),
    )
    source = _replace_digest(
        source,
        "V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256",
        compatibility_payload["semantic_sha256"],
    )
    adapter = resources_dir / "v4_factor_evidence_adapter_policy.v1.json"
    adapter_payload = json.loads(adapter.read_bytes())
    source = _replace_digest(
        source,
        "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256",
        _sha(adapter),
    )
    source = _replace_digest(
        source,
        "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_SEMANTIC_SHA256",
        adapter_payload["semantic_sha256"],
    )
    regime_policy_v1 = resources_dir / "factor_regime_diagnostic_policy.v1.json"
    regime_policy_v1_payload = json.loads(regime_policy_v1.read_bytes())
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_BYTE_SHA256",
        _sha(regime_policy_v1),
    )
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_SEMANTIC_SHA256",
        regime_policy_v1_payload["semantic_sha256"],
    )
    regime_policy_v2 = resources_dir / "factor_regime_diagnostic_policy.v2.json"
    regime_policy_v2_payload = json.loads(regime_policy_v2.read_bytes())
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_BYTE_SHA256",
        _sha(regime_policy_v2),
    )
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_SEMANTIC_SHA256",
        regime_policy_v2_payload["semantic_sha256"],
    )
    regime_policy = resources_dir / "factor_regime_diagnostic_policy.v3.json"
    regime_policy_payload = json.loads(regime_policy.read_bytes())
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256",
        _sha(regime_policy),
    )
    source = _replace_digest(
        source,
        "FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256",
        regime_policy_payload["semantic_sha256"],
    )
    source, count = re.subn(
        r'V4_COMPATIBILITY_POLICY_ID: Final = "[^"]+"',
        ("V4_COMPATIBILITY_POLICY_ID: Final = " f'"{compatibility_payload["artifact_id"]}"'),
        source,
    )
    if count != 1:
        raise RuntimeError("cannot update V4_COMPATIBILITY_POLICY_ID")
    validators.write_text(source, encoding="utf-8")

    resources_module = CONTRACT / "resources.py"
    source = resources_module.read_text(encoding="utf-8")
    source = _replace_digest(
        source,
        "PACKAGE_MANIFEST_SHA256",
        _sha(package_manifest),
    )
    resources_module.write_text(source, encoding="utf-8")

    print(
        json.dumps(
            {
                "adapter_policy_byte_sha256": _sha(adapter),
                "compatibility_policy_byte_sha256": _sha(compatibility),
                "compatibility_policy_v3_byte_sha256": _sha(compatibility_v3),
                "compatibility_policy_v2_byte_sha256": _sha(compatibility_v2),
                "compatibility_policy_v1_byte_sha256": _sha(compatibility_v1),
                "factor_regime_diagnostic_policy_byte_sha256": _sha(regime_policy),
                "factor_regime_diagnostic_policy_v2_byte_sha256": _sha(regime_policy_v2),
                "factor_regime_diagnostic_policy_v1_byte_sha256": _sha(regime_policy_v1),
                "package_manifest_byte_sha256": _sha(package_manifest),
                "runtime_manifest_byte_sha256": _sha(runtime_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
