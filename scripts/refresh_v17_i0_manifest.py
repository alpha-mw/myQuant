#!/usr/bin/env python3
"""Deterministically refresh only the closed V17 I0 Python source manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from quant_investor.v17_v4_contract.canonical import canonical_resource_bytes, seal_semantic

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "quant_investor/intelligence"
MANIFEST = PACKAGE / "resources/package_manifest.v1.json"
VERSION = "myquant.v17.research-intelligence.package-manifest.v1"


def main() -> None:
    paths = sorted(
        PACKAGE.rglob("*.py"),
        key=lambda path: path.relative_to(PACKAGE).as_posix().encode("ascii"),
    )
    document = seal_semantic(
        {
            "array_order_semantics": {"/source_paths": "relative_path ASCII ascending"},
            "source_paths": [
                {
                    "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "relative_path": path.relative_to(PACKAGE).as_posix(),
                }
                for path in paths
            ],
            "version": VERSION,
        }
    )
    MANIFEST.write_bytes(canonical_resource_bytes(document))


if __name__ == "__main__":
    main()
