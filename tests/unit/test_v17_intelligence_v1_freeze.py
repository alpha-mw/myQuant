from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from quant_investor.intelligence.package import verify_package


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT
    / "docs"
    / "architecture"
    / "resources"
    / "v17_intelligence_v1_freeze_manifest.json"
)
MANIFEST_VERSION = "myquant.v17.intelligence-v1-freeze-manifest.v1"
FROZEN_PACKAGE_SHA256 = (
    "6433bd5350129aac404271f49dde21598e512a418746ef613688315e7677e604"
)
R22_NORMALIZED_GOLDEN_SHA256 = (
    "3f33b8814e02ff2f2bbe9c3939f0562d4b485385e178b64ac83d369dbf13bbe1"
)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate freeze-manifest key: {key}")
        result[key] = value
    return result


def _canonical_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_manifest() -> dict[str, Any]:
    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )


def test_v1_freeze_manifest_is_sealed_and_exact() -> None:
    manifest = _load_manifest()
    assert set(manifest) == {
        "version",
        "frozen_at",
        "frozen_commit",
        "decision_protocol",
        "research_only",
        "production",
        "package_manifest_semantic_sha256",
        "r22_normalized_golden_sha256",
        "files",
        "semantic_sha256",
    }
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["decision_protocol"] == "myquant.v17.v4"
    assert manifest["research_only"] is True
    assert manifest["production"] is False
    assert manifest["package_manifest_semantic_sha256"] == FROZEN_PACKAGE_SHA256
    assert manifest["r22_normalized_golden_sha256"] == R22_NORMALIZED_GOLDEN_SHA256

    semantic_sha256 = manifest.pop("semantic_sha256")
    assert semantic_sha256 == _canonical_sha256(manifest)


def test_every_frozen_file_remains_byte_identical() -> None:
    rows = _load_manifest()["files"]
    assert isinstance(rows, list)
    paths = [str(row["path"]) for row in rows]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))

    for row in rows:
        assert set(row) == {"path", "category", "byte_sha256"}
        path = ROOT / row["path"]
        assert path.is_file() and not path.is_symlink(), row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["byte_sha256"]


def test_intelligence_v1_file_set_cannot_drift() -> None:
    frozen = {
        str(row["path"])
        for row in _load_manifest()["files"]
        if str(row["path"]).startswith("quant_investor/intelligence/")
    }
    package_root = ROOT / "quant_investor" / "intelligence"
    current = {
        path.relative_to(ROOT).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert current == frozen


def test_runtime_package_identity_matches_the_frozen_manifest() -> None:
    assert verify_package()["semantic_sha256"] == FROZEN_PACKAGE_SHA256
