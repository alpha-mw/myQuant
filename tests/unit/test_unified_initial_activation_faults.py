from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import quant_investor.factors.governance.production as production_module
import quant_investor.system.store as system_store_module

from quant_investor.cli.unified import factor_history, system_status, system_verify
from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    seal_artifact,
)
from quant_investor.mainline import MainlineStore
from quant_investor.system import (
    ACTIVE_POINTER_PATH,
    ACTIVATION_TRANSACTIONS_ROOT,
    MIGRATION_MARKER_PATH,
    SystemActivationAuthorizationError,
    SystemCASMismatch,
    SystemContractError,
    SystemError,
    SystemImmutableConflict,
    SystemMigrationMarkerAbsent,
    SystemPreconditionError,
    SystemStorageError,
    SystemStore,
    build_suspended_generation,
    build_prepared_activation_transaction,
    validate_activation_authorization,
    verify_emergency_controller,
)
from test_unified_system_bootstrap import _closure
from unified_activation_helpers import prepare_initial_activation


@pytest.fixture(autouse=True)
def _isolate_pointer_protocol_from_production_source_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fault injection remains below the independently tested source hard gate."""

    def isolated_receipt(**kwargs):
        sources = kwargs["verified_generation"]["manifest"]["payload"]["factor_source_object_refs"]
        return {
            "payload": {
                "calendar_authority_policy_ref": sources[0],
                "calendar_compilation_ref": sources[1],
                "calendar_capability_ref": None,
                "calendar_capture_execution_ref": None,
                "calendar_authorization_basis": {
                    "authority_route": "EXCHANGE_OFFICIAL",
                    "policy_ref": sources[0],
                    "compilation_ref": sources[1],
                    "capability_ref": None,
                    "capture_execution_ref": None,
                    "source_limitations": [],
                },
                "calendar_source_limitations": [],
            }
        }

    monkeypatch.setattr(
        production_module,
        "validate_production_bootstrap_generation_closure",
        isolated_receipt,
    )


def _case(tmp_path: Path) -> tuple[dict, dict, dict]:
    closure = _closure(tmp_path)
    store = closure["store"]
    generation = store.assemble_generation(**closure["kwargs"])
    prepared = prepare_initial_activation(
        store,
        generation,
        closure["release_ref"],
    )
    return closure, generation, prepared


def _marker_bytes(prepared: dict, generation: dict) -> bytes:
    authorization, marker = validate_activation_authorization(
        prepared["activation_authorization_raw"],
        final_cutover_authorization=prepared["final_cutover_authorization_raw"],
        migration_receipt=prepared["migration_receipt_raw"],
        target_active_pointer=prepared["target_active_pointer_raw"],
        target_generation_manifest=generation["manifest"],
        deployed_release_ref=prepared["deployed_release_ref"],
        current_uid=os.geteuid(),
    )
    assert authorization["kind"] == "system.activation_authorization"
    return canonical_json_bytes(marker)


def _direct_write(root: Path, relative: object, raw: bytes) -> None:
    path = root / str(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _emergency_pointer_raw(
    store: SystemStore,
    previous_pointer_sha256: str,
) -> bytes:
    controller = verify_emergency_controller(store)
    return canonical_json_bytes(
        {
            "generation_id": controller["generation_id"],
            "manifest_sha256": controller["manifest_sha256"],
            "previous_pointer_sha256": previous_pointer_sha256,
            "activated_at": "2026-08-14T00:00:02Z",
            "os_actor": f"uid:{os.geteuid()}:emergency-suspend",
        }
    )


def test_no_transaction_and_authorization_only_leave_pointer_absent(
    tmp_path: Path,
) -> None:
    closure, _generation, prepared = _case(tmp_path)
    store = closure["store"]
    assert store.read_active() is None

    authorization = parse_canonical_json_bytes(
        prepared["activation_authorization_raw"], label="authorization"
    )
    store.put_object(authorization)
    assert store.read_active() is None
    assert not (closure["workspace"] / str(ACTIVE_POINTER_PATH)).exists()
    assert not (closure["workspace"] / str(MIGRATION_MARKER_PATH)).exists()


def test_missing_fake_or_unresolved_final_cutover_authority_cannot_cas(
    tmp_path: Path,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    active_path = closure["workspace"] / str(ACTIVE_POINTER_PATH)

    missing = {**inputs, "final_cutover_authorization_raw": b""}
    with pytest.raises(SystemActivationAuthorizationError, match="exact non-empty"):
        store.activate_initial_generation(**missing)

    original = parse_canonical_json_bytes(
        inputs["final_cutover_authorization_raw"], label="final authorization"
    )
    fake_payload = dict(original["payload"])
    fake_payload["final_integration_commit"] = "f" * 40
    fake_payload["release_commit"] = "f" * 40
    fake = seal_artifact(
        "system.final_cutover_authorization",
        fake_payload,
        created_at=original["created_at"],
    )
    with pytest.raises(SystemPreconditionError, match="final|clean"):
        store.activate_initial_generation(
            **{**inputs, "final_cutover_authorization_raw": canonical_json_bytes(fake)}
        )

    unresolved_payload = dict(original["payload"])
    unresolved_ref = dict(unresolved_payload["main_checkout_adoption_ref"])
    unresolved_ref["byte_sha256"] = "f" * 64
    unresolved_payload["main_checkout_adoption_ref"] = unresolved_ref
    unresolved = seal_artifact(
        "system.final_cutover_authorization",
        unresolved_payload,
        created_at=original["created_at"],
    )
    with pytest.raises(SystemError):
        store.activate_initial_generation(
            **{
                **inputs,
                "final_cutover_authorization_raw": canonical_json_bytes(unresolved),
            }
        )
    assert not active_path.exists()


def test_prepared_transaction_without_pointer_can_complete_once(tmp_path: Path) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    authorization = parse_canonical_json_bytes(
        inputs["activation_authorization_raw"], label="authorization"
    )
    store.put_object(authorization)
    store.put_object(build_prepared_activation_transaction(authorization))

    result = store.activate_initial_generation(**inputs)
    assert result["activation"]["cas_performed"] is True
    assert store.read_active()["pointer"] == result["pointer"]


def test_pointer_only_crash_blocks_active_then_recovers_marker_without_second_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    original = store._storage._write_exact_once
    failed = False

    def fail_marker(value: object, raw: bytes, *, allow_reserved_authority: bool):
        nonlocal failed
        if str(value) == str(MIGRATION_MARKER_PATH) and not failed:
            failed = True
            raise SystemStorageError("injected marker publication failure")
        return original(value, raw, allow_reserved_authority=allow_reserved_authority)

    monkeypatch.setattr(store._storage, "_write_exact_once", fail_marker)
    with pytest.raises(SystemStorageError, match="injected marker"):
        store.activate_initial_generation(**inputs)

    pointer_path = closure["workspace"] / str(ACTIVE_POINTER_PATH)
    assert pointer_path.read_bytes() == inputs["target_active_pointer_raw"]
    assert not (closure["workspace"] / str(MIGRATION_MARKER_PATH)).exists()
    with pytest.raises(SystemMigrationMarkerAbsent):
        store.read_active()

    recovered = store.activate_initial_generation(**inputs)
    assert recovered["activation"]["cas_performed"] is False
    assert (
        recovered["pointer_byte_sha256"]
        == hashlib.sha256(inputs["target_active_pointer_raw"]).hexdigest()
    )


def test_completed_activation_replay_is_exact_and_never_repeats_cas(
    tmp_path: Path,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    first = store.activate_initial_generation(**inputs)
    second = store.activate_initial_generation(**inputs)
    assert first["activation"]["cas_performed"] is True
    assert second["activation"]["cas_performed"] is False
    assert first["pointer"] == second["pointer"]
    assert first["migration_completion"]["marker"] == second["migration_completion"]["marker"]
    assert first["deployed_release_verified"] is True
    assert second["deployed_release_verified"] is True


def test_initial_marker_remains_valid_after_descendant_release_commit(tmp_path: Path) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    activated = store.activate_initial_generation(**inputs)

    descendant = closure["workspace"] / ".descendant-release"
    descendant.write_bytes(b"legitimate-descendant-release\n")
    runner = closure["workspace"] / "quant_investor/migration/authority.py"
    runner.write_bytes(runner.read_bytes() + b"\n# descendant runner change\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(closure["workspace"]),
            "add",
            "-f",
            descendant.name,
            runner.relative_to(closure["workspace"]).as_posix(),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "-C", str(closure["workspace"]), "commit", "-q", "-m", "descendant"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    readback = store.read_active()
    assert readback is not None
    assert readback["pointer"] == activated["pointer"]
    assert readback["deployed_release_verified"] is True
    assert readback["migration_completion"]["marker"]["payload"]["migration_replay_refused"] is True


def test_default_initial_read_falls_back_to_historical_when_install_drifted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    activated = store.activate_initial_generation(**inputs)

    def reject_drifted_install(_release: object) -> None:
        raise SystemContractError("installed release differs")

    monkeypatch.setattr(system_store_module, "_verify_installed_release", reject_drifted_install)
    readback = store.read_active()
    assert readback["pointer"] == activated["pointer"]
    assert readback["deployed_release_verified"] is False
    assert readback["historical_release_verified"] is True
    completion = store.verify_migration_completion()
    assert completion["initial_pointer"] == activated["pointer"]

    import quant_investor.system as system_package

    monkeypatch.setattr(system_package, "SystemStore", lambda _workspace_root: store)
    active_verify = system_verify(
        workspace_root=str(closure["workspace"]),
        generation_id=None,
    )
    assert active_verify == {
        "status": "BLOCKED",
        "active_generation_id": activated["generation_id"],
        "generation_state": "OPERATIONAL",
        "verified": False,
        "blockers": ["SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"],
    }
    named_verify = system_verify(
        workspace_root=str(closure["workspace"]),
        generation_id=activated["generation_id"],
    )
    assert named_verify["status"] == "VERIFIED"
    assert named_verify["verified"] is True
    status_result = system_status(workspace_root=str(closure["workspace"]))
    assert status_result["capabilities"]["system"] == "PARTIAL"
    assert "SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED" in status_result["blockers"]
    assert factor_history(workspace_root=str(closure["workspace"])) == {
        "status": "BLOCKED",
        "active_generation_id": activated["generation_id"],
        "entries": [],
        "blockers": ["SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"],
    }
    assert MainlineStore(closure["workspace"], system_store=store).status(strategy_id="fixture")[
        "blockers"
    ] == ["DEPLOYED_RELEASE_NOT_VERIFIED"]

    with pytest.raises(SystemContractError, match="installed release differs"):
        store.read_active(deployed_release_ref=inputs["deployed_release_ref"])


def test_fresh_descendant_process_uses_historical_anchor_and_can_suspend(
    tmp_path: Path,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    activated = store.activate_initial_generation(**inputs)
    target_pointer = _emergency_pointer_raw(store, activated["pointer_byte_sha256"])
    marker_path = closure["workspace"] / str(MIGRATION_MARKER_PATH)
    marker_path.write_bytes(b"{}\n")
    marker_path.chmod(0o600)
    source_root = Path(__file__).resolve().parents[2]
    script = r"""
import base64
import json
import sys
import quant_investor.system.controller as controller
from quant_investor.system import SystemContractError, SystemStore

controller._CONTROLLER_BODY = controller._CONTROLLER_BODY + "\n# descendant implementation\n"

def reject_current_catalog(*_args, **_kwargs):
    raise SystemContractError("descendant compiled catalog must not reinterpret anchor")

SystemStore.read_contract_catalog = reject_current_catalog

store = SystemStore(sys.argv[1], source_root=sys.argv[4], source_root_id=sys.argv[5])
raw = base64.b64decode(sys.argv[3].encode("ascii"))
suspended = store.activate_suspended_generation(
    target_active_pointer_raw=raw,
    expected_pointer_sha256=sys.argv[2],
)
print(json.dumps({
    "after": suspended["generation_state"],
    "factor_authority": suspended["factor_authority"],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            "-c",
            script,
            str(closure["workspace"]),
            activated["pointer_byte_sha256"],
            base64.b64encode(target_pointer).decode("ascii"),
            str(store.source_root),
            store.source_root_id,
        ],
        cwd=source_root,
        env={**os.environ, "PYTHONPATH": str(source_root)},
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "after": "SYSTEM_SUSPENDED",
        "factor_authority": "BLOCKED",
    }
    assert marker_path.read_bytes() == b"{}\n"


@pytest.mark.parametrize("marker_state", ["absent", "malformed"])
def test_fixed_emergency_target_is_independent_from_marker_closure(
    tmp_path: Path,
    marker_state: str,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    activated = store.activate_initial_generation(**inputs)
    root = closure["workspace"]
    marker_path = root / str(MIGRATION_MARKER_PATH)
    if marker_state == "absent":
        marker_path.unlink()
        expected_marker: bytes | None = None
    else:
        expected_marker = b"{}\n"
        marker_path.write_bytes(expected_marker)
        marker_path.chmod(0o600)
    target_raw = _emergency_pointer_raw(store, activated["pointer_byte_sha256"])

    suspended = store.activate_suspended_generation(
        target_active_pointer_raw=target_raw,
        expected_pointer_sha256=activated["pointer_byte_sha256"],
    )

    assert suspended["generation_state"] == "SYSTEM_SUSPENDED"
    assert suspended["factor_authority"] == "BLOCKED"
    assert (root / str(ACTIVE_POINTER_PATH)).read_bytes() == target_raw
    if expected_marker is None:
        assert not marker_path.exists()
    else:
        assert marker_path.read_bytes() == expected_marker


@pytest.mark.parametrize(
    "mutation",
    ["second_target", "operational_kind", "wrong_manifest", "controller", "extra_field"],
)
def test_emergency_target_mutations_fail_before_cas_and_preserve_marker(
    tmp_path: Path,
    mutation: str,
) -> None:
    closure, generation, inputs = _case(tmp_path)
    store = closure["store"]
    activated = store.activate_initial_generation(**inputs)
    root = closure["workspace"]
    active_path = root / str(ACTIVE_POINTER_PATH)
    marker_path = root / str(MIGRATION_MARKER_PATH)
    active_before = active_path.read_bytes()
    marker_before = marker_path.read_bytes()
    controller = verify_emergency_controller(store)
    generation_id = controller["generation_id"]
    manifest_sha = controller["manifest_sha256"]
    if mutation == "second_target":
        second = build_suspended_generation(
            store,
            blockers=["UNSEALED_SECOND_TARGET"],
            created_at="2026-08-14T00:00:03Z",
        )
        generation_id = second["generation_id"]
        manifest_sha = second["manifest_sha256"]
    elif mutation == "operational_kind":
        generation_id = generation["generation_id"]
        manifest_sha = generation["manifest_sha256"]
    elif mutation == "wrong_manifest":
        manifest_sha = "f" * 64
    elif mutation == "controller":
        controller_path = root / "results/system/control/suspend.py"
        controller_path.chmod(0o600)
        controller_path.write_bytes(controller_path.read_bytes() + b"# tampered\n")
        controller_path.chmod(0o500)
    else:
        manifest_path = root / "results/system/generations" / generation_id / "manifest.json"
        manifest = parse_canonical_json_bytes(
            manifest_path.read_bytes(), label="suspended manifest"
        )
        manifest["payload"]["unexpected"] = True
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        manifest_path.chmod(0o600)
    target_raw = canonical_json_bytes(
        {
            "generation_id": generation_id,
            "manifest_sha256": manifest_sha,
            "previous_pointer_sha256": activated["pointer_byte_sha256"],
            "activated_at": "2026-08-14T00:00:02Z",
            "os_actor": f"uid:{os.geteuid()}:emergency-suspend",
        }
    )

    with pytest.raises(SystemError):
        store.activate_suspended_generation(
            target_active_pointer_raw=target_raw,
            expected_pointer_sha256=activated["pointer_byte_sha256"],
        )

    assert active_path.read_bytes() == active_before
    assert marker_path.read_bytes() == marker_before


def test_marker_only_and_different_pointer_are_never_overwritten(tmp_path: Path) -> None:
    closure, generation, inputs = _case(tmp_path)
    store = closure["store"]
    root = closure["workspace"]
    marker_raw = _marker_bytes(inputs, generation)
    _direct_write(root, MIGRATION_MARKER_PATH, marker_raw)
    with pytest.raises(SystemImmutableConflict, match="without active pointer"):
        store.activate_initial_generation(**inputs)

    (root / str(MIGRATION_MARKER_PATH)).unlink()
    pointer = parse_canonical_json_bytes(inputs["target_active_pointer_raw"], label="pointer")
    different = canonical_json_bytes({**pointer, "activated_at": "2026-08-14T00:00:02Z"})
    _direct_write(root, ACTIVE_POINTER_PATH, different)
    with pytest.raises(SystemCASMismatch) as exc:
        store.activate_initial_generation(**inputs)
    assert (
        exc.value.public_fields["observed_pointer_sha256"] == hashlib.sha256(different).hexdigest()
    )
    assert (root / str(ACTIVE_POINTER_PATH)).read_bytes() == different


def test_marker_and_prepared_tamper_block_exact_recovery(tmp_path: Path) -> None:
    closure, generation, inputs = _case(tmp_path)
    store = closure["store"]
    root = closure["workspace"]
    result = store.activate_initial_generation(**inputs)

    marker_path = root / str(MIGRATION_MARKER_PATH)
    marker_raw = _marker_bytes(inputs, generation)
    marker_path.write_bytes(marker_raw + b" ")
    marker_path.chmod(0o600)
    with pytest.raises(SystemImmutableConflict, match="marker conflicts"):
        store.activate_initial_generation(**inputs)

    marker_path.write_bytes(marker_raw)
    marker_path.chmod(0o600)
    prepared_index = (
        root / str(ACTIVATION_TRANSACTIONS_ROOT) / f"{result['pointer_byte_sha256']}.json"
    )
    prepared_index.write_bytes(b"{}\n")
    prepared_index.chmod(0o600)
    with pytest.raises(SystemImmutableConflict):
        store.activate_initial_generation(**inputs)


def test_uid_release_and_preimage_drift_fail_before_empty_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure, _generation, inputs = _case(tmp_path)
    store = closure["store"]
    root = closure["workspace"]

    import quant_investor.system.store as store_module

    actual_uid = os.geteuid()
    with monkeypatch.context() as uid_patch:
        uid_patch.setattr(
            store_module,
            "os",
            SimpleNamespace(geteuid=lambda: actual_uid + 1),
        )
        with pytest.raises(SystemActivationAuthorizationError, match="UID changed"):
            store.activate_initial_generation(**inputs)
    assert not (root / str(ACTIVE_POINTER_PATH)).exists()

    missing_release = {**inputs["deployed_release_ref"], "byte_sha256": "f" * 64}
    with pytest.raises(SystemContractError, match="deployed release identity"):
        store.activate_initial_generation(**{**inputs, "deployed_release_ref": missing_release})
    assert not (root / str(ACTIVE_POINTER_PATH)).exists()

    pointer = parse_canonical_json_bytes(inputs["target_active_pointer_raw"], label="pointer")
    drift = canonical_json_bytes({**pointer, "activated_at": "2026-08-14T00:00:03Z"})
    _direct_write(root, ACTIVE_POINTER_PATH, drift)
    with pytest.raises(SystemCASMismatch):
        store.activate_initial_generation(**inputs)
    assert (root / str(ACTIVE_POINTER_PATH)).read_bytes() == drift
