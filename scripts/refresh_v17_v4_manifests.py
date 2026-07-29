#!/usr/bin/env python3
"""Deterministically refresh the closed V17 v4 package/runtime manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from quant_investor.v17_v4_contract.canonical import (
    canonical_resource_bytes,
    seal_semantic,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "quant_investor/v17_v4_contract"
RUNTIME = ROOT / "quant_investor/v17_v4_runtime"
QUANT_INVESTOR = ROOT / "quant_investor"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(canonical_resource_bytes(seal_semantic(value)))


def main() -> None:
    forward_source_paths = [
        QUANT_INVESTOR / "factors/forward_evaluator.py",
        *sorted((QUANT_INVESTOR / "industry").glob("*.py")),
        *sorted((RUNTIME / "themes").glob("*.py")),
    ]
    if any(not path.is_file() for path in forward_source_paths):
        raise RuntimeError("forward runtime source inventory is incomplete")
    _write(
        CONTRACT / "resources/forward_runtime_source_manifest.v1.json",
        {
            "array_order_semantics": {"/sources": "relative_path ASCII ascending"},
            "authority": NO_AUTHORITY,
            "manifest_id": "v17-v4-forward-runtime-sources",
            "protocol_version": "myquant.v17.v4",
            "sources": [
                {
                    "byte_sha256": _sha(path),
                    "relative_path": path.relative_to(QUANT_INVESTOR).as_posix(),
                }
                for path in sorted(
                    forward_source_paths,
                    key=lambda value: value.relative_to(QUANT_INVESTOR).as_posix(),
                )
            ],
            "version": ("myquant.v17.v4.forward-runtime-source-manifest.v1"),
        },
    )

    runtime_manifest = CONTRACT / "resources/runtime_build_manifest.v1.json"
    runtime_rows = [
        {
            "byte_sha256": _sha(path),
            "relative_path": ("v17_v4_runtime/" + path.name),
        }
        for path in sorted(RUNTIME.glob("*.py"))
    ]
    _write(
        runtime_manifest,
        {
            "array_order_semantics": {"/sources": "relative_path ASCII ascending"},
            "authority": NO_AUTHORITY,
            "protocol_version": "myquant.v17.v4",
            "sources": runtime_rows,
            "version": "myquant.v17.v4.runtime-build-manifest.v1",
        },
    )

    asset_paths = sorted(
        [
            *[
                path
                for path in (CONTRACT / "resources").glob("*.json")
                if path.name != "package_manifest.v1.json"
            ],
            *list((CONTRACT / "schemas").glob("*.json")),
        ],
        key=lambda path: path.relative_to(CONTRACT).as_posix(),
    )
    assets: list[dict[str, str]] = []
    for path in asset_paths:
        payload = json.loads(path.read_bytes())
        artifact_id = payload.get("version") or payload.get("$id")
        if type(artifact_id) is not str:
            raise RuntimeError(f"asset has no identity: {path}")
        assets.append(
            {
                "artifact_id": artifact_id,
                "byte_sha256": _sha(path),
                "relative_path": path.relative_to(CONTRACT).as_posix(),
            }
        )
    package_manifest = CONTRACT / "resources/package_manifest.v1.json"
    _write(
        package_manifest,
        {
            "array_order_semantics": {
                "/assets": "relative_path ASCII ascending",
                "/source_paths": "relative_path ASCII ascending",
            },
            "assets": assets,
            "authority": NO_AUTHORITY,
            "protocol_version": "myquant.v17.v4",
            "self_binding": {
                "byte_sha256_source": (
                    "quant_investor.v17_v4_contract.resources." "PACKAGE_MANIFEST_SHA256"
                ),
                "relative_path": "resources/package_manifest.v1.json",
            },
            "source_paths": sorted(path.name for path in CONTRACT.glob("*.py")),
            "version": "myquant.v17.v4.package-manifest.v1",
        },
    )
    digest = _sha(package_manifest)
    resources = CONTRACT / "resources.py"
    source = resources.read_text(encoding="utf-8")
    updated, count = re.subn(
        (r"PACKAGE_MANIFEST_SHA256: Final = " r'(?:\(\n    )?"[0-9a-f]{64}"(?:\n\))?'),
        f'PACKAGE_MANIFEST_SHA256: Final = "{digest}"',
        source,
    )
    if count != 1:
        raise RuntimeError("cannot update PACKAGE_MANIFEST_SHA256")
    resources.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
