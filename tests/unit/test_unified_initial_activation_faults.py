from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from quant_investor.system import (
    ACTIVE_POINTER_PATH,
    ACTIVATION_TRANSACTIONS_ROOT,
    MIGRATION_MARKER_PATH,
    SystemActivationAuthorizationError,
    SystemCASMismatch,
    SystemContractError,
    SystemImmutableConflict,
    SystemMigrationMarkerAbsent,
    SystemStorageError,
    build_prepared_activation_transaction,
    validate_activation_authorization,
)
from test_unified_system_bootstrap import _closure
from unified_activation_helpers import prepare_initial_activation


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
    assert recovered["pointer_byte_sha256"] == hashlib.sha256(
        inputs["target_active_pointer_raw"]
    ).hexdigest()


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
    assert first["migration_completion"]["marker"] == second[
        "migration_completion"
    ]["marker"]


def test_marker_only_and_different_pointer_are_never_overwritten(tmp_path: Path) -> None:
    closure, generation, inputs = _case(tmp_path)
    store = closure["store"]
    root = closure["workspace"]
    marker_raw = _marker_bytes(inputs, generation)
    _direct_write(root, MIGRATION_MARKER_PATH, marker_raw)
    with pytest.raises(SystemImmutableConflict, match="without active pointer"):
        store.activate_initial_generation(**inputs)

    (root / str(MIGRATION_MARKER_PATH)).unlink()
    pointer = parse_canonical_json_bytes(
        inputs["target_active_pointer_raw"], label="pointer"
    )
    different = canonical_json_bytes(
        {**pointer, "activated_at": "2026-08-14T00:00:02Z"}
    )
    _direct_write(root, ACTIVE_POINTER_PATH, different)
    with pytest.raises(SystemCASMismatch) as exc:
        store.activate_initial_generation(**inputs)
    assert exc.value.public_fields["observed_pointer_sha256"] == hashlib.sha256(
        different
    ).hexdigest()
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
        root
        / str(ACTIVATION_TRANSACTIONS_ROOT)
        / f"{result['pointer_byte_sha256']}.json"
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
    monkeypatch.setattr(
        store_module,
        "os",
        SimpleNamespace(geteuid=lambda: actual_uid + 1),
    )
    with pytest.raises(SystemActivationAuthorizationError, match="UID changed"):
        store.activate_initial_generation(**inputs)
    assert not (root / str(ACTIVE_POINTER_PATH)).exists()
    monkeypatch.undo()

    missing_release = {**inputs["deployed_release_ref"], "byte_sha256": "f" * 64}
    with pytest.raises(SystemContractError, match="deployed release identity"):
        store.activate_initial_generation(
            **{**inputs, "deployed_release_ref": missing_release}
        )
    assert not (root / str(ACTIVE_POINTER_PATH)).exists()

    pointer = parse_canonical_json_bytes(
        inputs["target_active_pointer_raw"], label="pointer"
    )
    drift = canonical_json_bytes(
        {**pointer, "activated_at": "2026-08-14T00:00:03Z"}
    )
    _direct_write(root, ACTIVE_POINTER_PATH, drift)
    with pytest.raises(SystemCASMismatch):
        store.activate_initial_generation(**inputs)
    assert (root / str(ACTIVE_POINTER_PATH)).read_bytes() == drift
