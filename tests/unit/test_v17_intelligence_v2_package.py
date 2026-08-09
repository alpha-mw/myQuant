from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from quant_investor.intelligence_v2 import IntelligenceV2ContractError
from quant_investor.intelligence_v2._core import FROZEN_V1_MANIFEST_SHA256
from quant_investor.intelligence_v2.package import (
    PACKAGE_MANIFEST_VERSION,
    verify_package,
)

PACKAGE_ROOT = Path(__file__).parents[2] / "quant_investor" / "intelligence_v2"


def _isolated_package(tmp_path: Path) -> Path:
    target = tmp_path / "intelligence_v2"
    shutil.copytree(PACKAGE_ROOT, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target


def test_v2_package_manifest_verifies_exact_source_and_frozen_v1_sets() -> None:
    receipt = verify_package()

    assert receipt["version"] == PACKAGE_MANIFEST_VERSION
    assert receipt["frozen_v1_manifest_semantic_sha256"] == FROZEN_V1_MANIFEST_SHA256
    source_paths = [row["relative_path"] for row in receipt["source_paths"]]
    assert source_paths == sorted(source_paths, key=lambda value: value.encode("ascii"))
    assert "decision_v2/engine.py" in source_paths
    assert "llm_research/safety_collector.py" in source_paths
    assert "portfolio/constructor.py" in source_paths
    assert "publication/permits.py" in source_paths
    assert "sources/tushare/contracts.py" in source_paths
    assert "sources/tushare/probe.py" in source_paths
    assert receipt["resource_paths"] == [
        {
            "byte_sha256": "abf722ada40d4b12803321eb904351f77958e4763b2c5de6c665d0a2c8c3889d",
            "relative_path": "resources/frozen_v1_manifest.v1.json",
        }
    ]


@pytest.mark.parametrize("operation", ["tamper", "add", "delete", "rename"])
def test_v2_package_rejects_source_set_or_byte_drift(tmp_path: Path, operation: str) -> None:
    package_root = _isolated_package(tmp_path)
    source = package_root / "readiness.py"
    if operation == "tamper":
        source.write_bytes(source.read_bytes() + b"\n")
    elif operation == "add":
        (package_root / "unexpected.py").write_text("VALUE = 1\n", encoding="utf-8")
    elif operation == "delete":
        source.unlink()
    else:
        source.rename(package_root / "readiness_renamed.py")

    with pytest.raises(IntelligenceV2ContractError):
        verify_package(package_root)


def test_v2_package_rejects_duplicate_manifest_key(tmp_path: Path) -> None:
    package_root = _isolated_package(tmp_path)
    manifest_path = package_root / "resources" / "package_manifest.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"version":"myquant.v17.intelligence-v2.package-manifest.v1"}',
        '"version":"duplicate","version":"myquant.v17.intelligence-v2.package-manifest.v1"}',
    )
    assert json.loads(raw)["version"] == manifest["version"]
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(IntelligenceV2ContractError, match="duplicate key"):
        verify_package(package_root)
