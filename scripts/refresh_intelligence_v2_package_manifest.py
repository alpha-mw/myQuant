#!/usr/bin/env python3
"""Deterministically refresh the closed Intelligence v2 package manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "quant_investor" / "intelligence_v2"
MANIFEST_PATH = PACKAGE_ROOT / "resources" / "package_manifest.v1.json"
FROZEN_V1_MANIFEST_SHA256 = "119e31882cbb3a68ffaf99eac2d6404d1c45e4284f46e5c8f54aa22b2cb908fc"
VERSION = "myquant.v17.intelligence-v2.package-manifest.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _row(path: Path) -> dict[str, str]:
    return {
        "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "relative_path": path.relative_to(PACKAGE_ROOT).as_posix(),
    }


def build_manifest_bytes() -> bytes:
    source_paths = sorted(
        PACKAGE_ROOT.rglob("*.py"),
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix().encode("ascii"),
    )
    resource_paths = sorted(
        (
            path
            for path in (PACKAGE_ROOT / "resources").rglob("*")
            if path.is_file() and path != MANIFEST_PATH
        ),
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix().encode("ascii"),
    )
    body: dict[str, Any] = {
        "array_order_semantics": {
            "/resource_paths": "relative_path ASCII ascending",
            "/source_paths": "relative_path ASCII ascending",
        },
        "frozen_v1_manifest_semantic_sha256": FROZEN_V1_MANIFEST_SHA256,
        "resource_paths": [_row(path) for path in resource_paths],
        "source_paths": [_row(path) for path in source_paths],
        "version": VERSION,
    }
    body["semantic_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return _canonical_bytes(body)


def main() -> None:
    if MANIFEST_PATH.is_symlink() or not MANIFEST_PATH.is_file():
        raise RuntimeError("Intelligence v2 package manifest path is unsafe")
    MANIFEST_PATH.write_bytes(build_manifest_bytes())


if __name__ == "__main__":
    main()
