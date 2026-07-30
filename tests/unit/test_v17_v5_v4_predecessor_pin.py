from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from quant_investor.v17_v5_contract import (
    load_compatibility_policy,
    load_compatibility_policy_v1,
    load_compatibility_policy_v2,
    verify_predecessor,
)
from quant_investor.v17_v5_contract.resources import (
    PackageResourceError,
    load_compatibility_policy as load_active_policy,
)
from quant_investor.v17_v5_contract.validators import (
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)

ROOT = Path(__file__).resolve().parents[2]
OLD_PREDECESSOR = "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e"
PACKAGE_MANIFEST_PATH = "quant_investor/v17_v4_contract/resources/package_manifest.v1.json"
RUNTIME_MANIFEST_PATH = "quant_investor/v17_v4_contract/resources/runtime_build_manifest.v1.json"


def _git_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_git_object_pin_verifies_every_v4_asset_and_runtime_source() -> None:
    assert V4_SOURCE_GIT_COMMIT == "73c5b6eea6c60d9a31865e176646687ffeee9d6a"
    package_raw = _git_bytes(V4_SOURCE_GIT_COMMIT, PACKAGE_MANIFEST_PATH)
    runtime_raw = _git_bytes(V4_SOURCE_GIT_COMMIT, RUNTIME_MANIFEST_PATH)
    assert hashlib.sha256(package_raw).hexdigest() == V4_PACKAGE_MANIFEST_SHA256
    assert hashlib.sha256(runtime_raw).hexdigest() == V4_RUNTIME_MANIFEST_SHA256

    package = json.loads(package_raw)
    runtime = json.loads(runtime_raw)
    assert len(package["assets"]) + 1 == 109
    assert len(runtime["sources"]) == 32
    for row in package["assets"]:
        path = f"quant_investor/v17_v4_contract/{row['relative_path']}"
        assert (
            hashlib.sha256(_git_bytes(V4_SOURCE_GIT_COMMIT, path)).hexdigest() == row["byte_sha256"]
        )
    for row in runtime["sources"]:
        path = f"quant_investor/{row['relative_path']}"
        assert (
            hashlib.sha256(_git_bytes(V4_SOURCE_GIT_COMMIT, path)).hexdigest() == row["byte_sha256"]
        )


def test_exact_merge_parent_and_v4_subtrees_preserve_sprint1e0a1_identity() -> None:
    parents = subprocess.run(
        ["git", "show", "-s", "--format=%P", "0a43f354e848290adaaf2400194c07851a57a6cb"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert parents.split() == [
        "3045f316cc8a085378011073a90b0f684957a0bf",
        V4_SOURCE_GIT_COMMIT,
    ]
    for subtree in ("v17_v4_contract", "v17_v4_runtime"):
        source_tree = subprocess.run(
            ["git", "rev-parse", f"{V4_SOURCE_GIT_COMMIT}:quant_investor/{subtree}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        current_tree = subprocess.run(
            ["git", "rev-parse", f"HEAD:quant_investor/{subtree}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert current_tree == source_tree


def test_active_v3_and_explicit_legacy_v1_v2_loaders_do_not_fallback() -> None:
    active = load_compatibility_policy()
    legacy_v2 = load_compatibility_policy_v2()
    legacy = load_compatibility_policy_v1()
    assert active["version"] == "myquant.v17.v5.v4-compatibility-policy.v3"
    assert active["predecessor"]["source_git_commit"] == V4_SOURCE_GIT_COMMIT
    assert legacy_v2["version"] == "myquant.v17.v5.v4-compatibility-policy.v2"
    assert (
        legacy_v2["predecessor"]["source_git_commit"] == "1da7ffb636a3254940525d746549d15e827f06ba"
    )
    assert legacy["version"] == "myquant.v17.v5.v4-compatibility-policy.v1"
    assert legacy["predecessor"]["source_git_commit"] == OLD_PREDECESSOR


def test_runtime_predecessor_verify_is_package_safe_without_git(tmp_path: Path) -> None:
    quant_root = tmp_path / "quant_investor"
    shutil.copytree(
        ROOT / "quant_investor/v17_v4_contract",
        quant_root / "v17_v4_contract",
    )
    shutil.copytree(
        ROOT / "quant_investor/v17_v4_runtime",
        quant_root / "v17_v4_runtime",
    )
    shutil.copytree(
        ROOT / "quant_investor/v17_v5_contract",
        quant_root / "v17_v5_contract",
    )
    assert not (tmp_path / ".git").exists()

    result = verify_predecessor(package_root=quant_root / "v17_v5_contract")

    assert result["source_git_commit"] == V4_SOURCE_GIT_COMMIT
    assert result["package_asset_count"] == 109
    assert result["runtime_source_count"] == 32


def test_missing_active_v3_policy_does_not_fall_back_to_valid_v1_v2(tmp_path: Path) -> None:
    target = tmp_path / "v17_v5_contract"
    shutil.copytree(ROOT / "quant_investor/v17_v5_contract", target)
    (target / "resources/v4_compatibility_policy.v3.json").unlink()
    assert (target / "resources/v4_compatibility_policy.v1.json").exists()
    assert (target / "resources/v4_compatibility_policy.v2.json").exists()

    with pytest.raises(PackageResourceError):
        load_active_policy(package_root=target)
