from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.system import (
    EMPTY,
    EMERGENCY_CONTROLLER_PATH,
    SystemImmutableConflict,
    SystemSecurityError,
    SystemStore,
    build_emergency_controller,
    build_suspended_generation,
    installed_code_manifest,
    verify_emergency_controller,
)
import quant_investor.system.controller as controller_module
import quant_investor.system.release as release_module

CREATED_AT = "2026-08-14T00:00:00Z"


def _target(tmp_path: Path) -> tuple[SystemStore, dict[str, Any], dict[str, Any], Path]:
    store = SystemStore(tmp_path)
    generation = build_suspended_generation(
        store,
        blockers=["EMERGENCY_TARGET"],
        created_at=CREATED_AT,
    )
    controller = build_emergency_controller(
        store,
        suspended_generation_id=generation["generation_id"],
    )
    path = tmp_path / str(EMERGENCY_CONTROLLER_PATH)
    return store, generation, controller, path


def test_emergency_controller_is_exact_stdlib_only_current_uid_mode_0500(
    tmp_path: Path,
) -> None:
    store, generation, controller, path = _target(tmp_path)
    raw = path.read_bytes()
    metadata = controller_module._parse_controller_metadata(raw)

    assert controller == verify_emergency_controller(
        store,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert controller["generation_id"] == generation["generation_id"]
    assert controller["manifest_sha256"] == generation["manifest_sha256"]
    assert controller["path"] == str(EMERGENCY_CONTROLLER_PATH)
    assert metadata["active_pointer_path"] == str(tmp_path / "results/system/_active.json")
    assert metadata["suspended_generation_id"] == generation["generation_id"]
    assert metadata["suspended_manifest_sha256"] == generation["manifest_sha256"]
    assert metadata["manifest_contract_sha256"] in raw.decode("utf-8")
    file_stat = path.stat(follow_symlinks=False)
    assert file_stat.st_uid == os.geteuid()
    assert file_stat.st_nlink == 1
    assert stat.S_IMODE(file_stat.st_mode) == 0o500

    tree = ast.parse(raw.decode("utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported == {
        "datetime",
        "fcntl",
        "hashlib",
        "json",
        "os",
        "secrets",
        "stat",
        "sys",
    }


def test_controller_only_cas_targets_embedded_suspended_generation_and_retains_history(
    tmp_path: Path,
) -> None:
    store, target, _, path = _target(tmp_path)
    prior = build_suspended_generation(
        store,
        blockers=["PRIOR_ACTIVE"],
        created_at="2026-08-14T00:01:00Z",
    )
    first = store.activate_generation(
        prior["generation_id"],
        expected_pointer_sha256=EMPTY,
        activated_at="2026-08-14T00:01:01Z",
        os_actor="test",
    )
    first_raw = (tmp_path / "results/system/_active.json").read_bytes()

    completed = subprocess.run(
        [sys.executable, str(path), first["pointer_byte_sha256"]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    report = json.loads(completed.stdout)
    assert report["generation_id"] == target["generation_id"]
    assert report["manifest_sha256"] == target["manifest_sha256"]
    assert report["state"] == "SYSTEM_SUSPENDED"
    active = store.read_active()
    assert active is not None
    assert active["generation_id"] == target["generation_id"]
    history = tmp_path / "results/system/pointer_history" / f"{first['pointer_byte_sha256']}.json"
    assert history.read_bytes() == first_raw

    active_before = (tmp_path / "results/system/_active.json").read_bytes()
    conflict = subprocess.run(
        [sys.executable, str(path), first["pointer_byte_sha256"]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert conflict.returncode == 2
    assert json.loads(conflict.stderr) == {"code": "SYSTEM_EMERGENCY_CAS_FAILED"}
    assert conflict.stdout == b""
    assert (tmp_path / "results/system/_active.json").read_bytes() == active_before


def test_controller_retention_requires_exact_readback_before_active_replace(
    tmp_path: Path,
) -> None:
    _, _, _, path = _target(tmp_path)
    namespace: dict[str, Any] = {
        "__file__": str(path),
        "__name__": "controller_readback_test",
    }
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    namespace["_read"] = lambda path, optional=False: b"readback-mismatch"
    controller_failure = namespace["ControllerFailure"]

    with pytest.raises(controller_failure):
        namespace["_retain_previous"](
            b'{"pointer":"prior"}',
            "a" * 64,
        )
    assert not (tmp_path / "results/system/_active.json").exists()


def test_controller_is_immutable_and_verifier_rejects_mode_tamper(tmp_path: Path) -> None:
    store, _, controller, path = _target(tmp_path)
    second = build_suspended_generation(
        store,
        blockers=["DIFFERENT_TARGET"],
        created_at="2026-08-14T00:02:00Z",
    )
    with pytest.raises(SystemImmutableConflict):
        build_emergency_controller(
            store,
            suspended_generation_id=second["generation_id"],
        )

    path.chmod(0o700)
    with pytest.raises(SystemSecurityError):
        verify_emergency_controller(store, expected_sha256=controller["byte_sha256"])


def test_controller_invalid_target_payload_fails_as_expected_validation_error(
    tmp_path: Path,
) -> None:
    _, generation, _, path = _target(tmp_path)
    manifest_path = (
        tmp_path / "results/system/generations" / generation["generation_id"] / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_bytes())
    manifest["payload"] = []
    forged_manifest = canonical_json_bytes(manifest)
    manifest_path.write_bytes(forged_manifest)
    manifest_path.chmod(0o600)

    metadata = controller_module._parse_controller_metadata(path.read_bytes())
    metadata["suspended_manifest_sha256"] = hashlib.sha256(forged_manifest).hexdigest()
    path.chmod(0o600)
    path.write_bytes(controller_module._controller_bytes(metadata))
    path.chmod(0o500)

    completed = subprocess.run(
        [sys.executable, str(path), EMPTY],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stderr) == {"code": "SYSTEM_EMERGENCY_CAS_FAILED"}
    assert not (tmp_path / "results/system/_active.json").exists()


def test_controller_rejects_suspended_target_with_factor_attestation_closure(
    tmp_path: Path,
) -> None:
    _, generation, _, path = _target(tmp_path)
    original_path = (
        tmp_path / "results/system/generations" / generation["generation_id"] / "manifest.json"
    )
    manifest = json.loads(original_path.read_bytes())
    manifest["payload"]["factor_source_object_refs"] = [{}]
    preimage = {
        "domain": "myquant-artifact",
        "kind": manifest["kind"],
        "contract_sha256": manifest["contract_sha256"],
        "identity_field": "assembly_id",
        "artifact_id": manifest["artifact_id"],
        "created_at": manifest["created_at"],
        "payload": manifest["payload"],
    }
    manifest["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
    forged_raw = canonical_json_bytes(manifest)
    forged_dir = tmp_path / "results/system/generations" / manifest["semantic_sha256"]
    forged_dir.mkdir(mode=0o700)
    forged_path = forged_dir / "manifest.json"
    forged_path.write_bytes(forged_raw)
    forged_path.chmod(0o600)

    metadata = controller_module._parse_controller_metadata(path.read_bytes())
    metadata["generation_manifest_path"] = str(forged_path)
    metadata["suspended_generation_id"] = manifest["semantic_sha256"]
    metadata["suspended_manifest_sha256"] = hashlib.sha256(forged_raw).hexdigest()
    path.chmod(0o600)
    path.write_bytes(controller_module._controller_bytes(metadata))
    path.chmod(0o500)

    completed = subprocess.run(
        [sys.executable, str(path), EMPTY],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stderr) == {"code": "SYSTEM_EMERGENCY_CAS_FAILED"}
    assert not (tmp_path / "results/system/_active.json").exists()


def test_installed_code_manifest_rejects_unsafe_mode_owner_hardlink_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "quant_investor"
    package.mkdir(mode=0o700)
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source.chmod(0o600)
    monkeypatch.setattr(release_module, "_PACKAGE_ROOT", package)

    assert installed_code_manifest() == release_module.installed_code_manifest()

    source.chmod(0o622)
    with pytest.raises(SystemSecurityError):
        installed_code_manifest()
    source.chmod(0o600)

    alias = package / "module-alias.py"
    os.link(source, alias)
    with pytest.raises(SystemSecurityError):
        installed_code_manifest()
    alias.unlink()

    linked_directory = package / "linked"
    linked_directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SystemSecurityError):
        installed_code_manifest()
    linked_directory.unlink()

    package.chmod(0o722)
    with pytest.raises(SystemSecurityError):
        installed_code_manifest()
    package.chmod(0o700)

    with monkeypatch.context() as owner_patch:
        owner_patch.setattr(os, "geteuid", lambda: os.getuid() + 1)
        with pytest.raises(SystemSecurityError):
            installed_code_manifest()
