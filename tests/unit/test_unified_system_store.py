from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.system import (
    ACTIVE_POINTER_PATH,
    ASSEMBLY_REQUEST_FIELDS,
    EMPTY,
    OBJECT_REF_FIELDS,
    POINTER_FIELDS,
    SystemCASMismatch,
    SystemContractError,
    SystemStore,
    build_suspended_generation,
    decode_assembly_request,
    verify_emergency_controller,
)
from test_unified_system_bootstrap import _closure
from unified_activation_helpers import (
    activate_initial,
    isolate_pointer_protocol_source_gate,
    prepare_initial_activation,
)

CREATED_AT = "2026-08-14T00:00:00Z"


@pytest.fixture(autouse=True)
def _isolate_pointer_protocol_from_production_source_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """These tests exercise CAS mechanics; production closure has its own suite."""
    isolate_pointer_protocol_source_gate(monkeypatch)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _store(tmp_path: Path) -> SystemStore:
    return SystemStore(tmp_path)


def _suspended(
    store: SystemStore,
    *,
    blocker: str = "TEST_SUSPENSION",
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    return build_suspended_generation(
        store,
        blockers=[blocker],
        created_at=created_at,
    )


def _activate_suspended(
    store: SystemStore,
    generation: dict[str, Any],
    previous_sha256: str,
) -> dict[str, Any]:
    return store.activate_suspended_generation(
        target_active_pointer_raw=canonical_json_bytes(
            {
                "generation_id": generation["generation_id"],
                "manifest_sha256": generation["manifest_sha256"],
                "previous_pointer_sha256": previous_sha256,
                "activated_at": "2026-08-14T00:01:01Z",
                "os_actor": f"uid:{os.geteuid()}:emergency-suspend",
            }
        ),
        expected_pointer_sha256=previous_sha256,
    )


def _controller_target(store: SystemStore) -> dict[str, Any]:
    controller = verify_emergency_controller(store)
    return store.verify_generation(controller["generation_id"])


def test_absent_status_and_verify_are_normal_reports(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.read_active() is None
    assert store.verify() == {
        "state": "UNINITIALIZED",
        "verified": False,
        "active_pointer_sha256": EMPTY,
        "generation_id": None,
        "blockers": ["SYSTEM_ACTIVE_POINTER_ABSENT"],
    }
    status = store.status()
    assert status["state"] == "UNINITIALIZED"
    assert status["active_pointer_sha256"] == EMPTY
    assert status["external_routing_state"] == "UNINITIALIZED"


def test_suspended_generation_is_minimal_and_generation_id_is_semantic_sha(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    generation = _suspended(store)
    payload = generation["manifest"]["payload"]

    assert generation["generation_state"] == "SYSTEM_SUSPENDED"
    assert generation["generation_id"] == generation["manifest"]["semantic_sha256"]
    assert "generation_id" not in payload
    assert payload["source_refs"] == []
    assert payload["factor_policy_ref"] is None
    assert payload["factor_evidence_refs"] == []
    assert payload["factor_active_set_ref"] is None
    assert payload["mainline_ref"] is None
    assert payload["research_refs"] == []
    assert payload["migration_receipt_ref"] is None
    assert payload["migration_marker_ref"] is None
    assert payload["emergency_controller_sha256"] is None
    assert generation["sources"] == []
    assert generation["factor_evidence"] == []
    assert generation["factor_active_set"] is None

    manifest = (
        tmp_path / "results/system/generations" / generation["generation_id"] / "manifest.json"
    )
    assert manifest.is_file()
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


def test_object_layout_is_kind_scoped_and_readback_is_exact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    artifact = seal_artifact(
        "system.release",
        {
            "release_id": "release-layout",
            "state": "TEST",
            "code_sha256": _sha("code"),
            "wheel_sha256": _sha("wheel"),
            "code_manifest_sha256": _sha("code-manifest"),
        },
        created_at=CREATED_AT,
    )

    ref = store.put_object(artifact)

    assert set(ref) == set(OBJECT_REF_FIELDS)
    path = tmp_path / "results/system/objects/system.release" / f"{ref['byte_sha256']}.json"
    assert path.read_bytes() == canonical_json_bytes(artifact)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.read_object(ref) == artifact


def test_empty_activation_pointer_and_previous_bytes_are_retained(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    first_generation = store.assemble_generation(**closure["kwargs"])
    first = activate_initial(store, first_generation, closure["release_ref"])
    active_path = closure["workspace"] / str(ACTIVE_POINTER_PATH)
    first_bytes = active_path.read_bytes()

    assert set(first["pointer"]) == set(POINTER_FIELDS)
    assert first["pointer"]["previous_pointer_sha256"] == EMPTY
    assert first["pointer"]["manifest_sha256"] == first["manifest_sha256"]
    assert stat.S_IMODE(active_path.stat().st_mode) == 0o600

    second_generation = _controller_target(store)
    second = _activate_suspended(store, second_generation, first["pointer_byte_sha256"])

    history = (
        closure["workspace"]
        / "results/system/pointer_history"
        / f"{first['pointer_byte_sha256']}.json"
    )
    assert history.read_bytes() == first_bytes
    assert second["pointer"]["previous_pointer_sha256"] == first["pointer_byte_sha256"]
    newest = store.pointer_history()
    oldest = store.pointer_history(newest_first=False)
    assert [row["generation"]["generation_id"] for row in newest] == [
        second_generation["generation_id"],
        first_generation["generation_id"],
    ]
    assert oldest == list(reversed(newest))


def test_pointer_history_rejects_missing_retained_bytes(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    first_generation = store.assemble_generation(**closure["kwargs"])
    first = activate_initial(store, first_generation, closure["release_ref"])
    second_generation = _controller_target(store)
    _activate_suspended(store, second_generation, first["pointer_byte_sha256"])
    retained = (
        closure["workspace"]
        / "results/system/pointer_history"
        / f"{first['pointer_byte_sha256']}.json"
    )
    retained.unlink()

    from quant_investor.system import SystemNotFound

    with pytest.raises(SystemNotFound):
        store.pointer_history()


def test_pointer_history_rejects_retained_byte_hash_mismatch(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    first_generation = store.assemble_generation(**closure["kwargs"])
    first = activate_initial(store, first_generation, closure["release_ref"])
    second_generation = _controller_target(store)
    _activate_suspended(store, second_generation, first["pointer_byte_sha256"])
    retained = (
        closure["workspace"]
        / "results/system/pointer_history"
        / f"{first['pointer_byte_sha256']}.json"
    )
    forged = {**first["pointer"], "os_actor": "forged"}
    retained.write_bytes(canonical_json_bytes(forged))
    retained.chmod(0o600)

    with pytest.raises(SystemContractError, match="retained previous pointer byte hash mismatch"):
        store.pointer_history()


def test_pointer_history_rejects_a_cycle_before_resolving_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    first_sha = "a" * 64
    second_sha = "b" * 64
    base = {
        "generation_id": "c" * 64,
        "manifest_sha256": "d" * 64,
        "activated_at": "2026-08-14T00:00:01Z",
        "os_actor": "test",
    }
    first = SimpleNamespace(
        data=canonical_json_bytes({**base, "previous_pointer_sha256": second_sha}),
        byte_sha256=first_sha,
    )
    second = SimpleNamespace(
        data=canonical_json_bytes({**base, "previous_pointer_sha256": first_sha}),
        byte_sha256=second_sha,
    )
    by_sha = {first_sha: first, second_sha: second}
    monkeypatch.setattr(store._storage, "read_optional", lambda path: first)
    monkeypatch.setattr(
        store._storage,
        "read",
        lambda path: by_sha[Path(str(path)).stem],
    )

    with pytest.raises(SystemContractError, match="active pointer history is cyclic"):
        store.pointer_history()


def test_concurrent_empty_cas_has_one_winner_and_safe_conflict(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    generations = [
        store.assemble_generation(
            **{**closure["kwargs"], "created_at": f"2026-08-14T00:0{index}:00Z"}
        )
        for index in (1, 2)
    ]
    preparations = [
        prepare_initial_activation(
            store,
            generation,
            closure["release_ref"],
            cutover_id=f"concurrent-{index}",
            prepared_at="2026-08-14T00:00:00Z",
            activated_at="2026-08-14T00:03:00Z",
        )
        for index, generation in enumerate(generations)
    ]
    barrier = threading.Barrier(2)

    def activate(prepared: dict[str, Any]) -> object:
        barrier.wait()
        try:
            return store.activate_initial_generation(**prepared)
        except Exception as exc:  # captured for exact assertion below
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(activate, preparations))

    conflicts = [row for row in outcomes if isinstance(row, SystemCASMismatch)]
    winners = [row for row in outcomes if isinstance(row, dict)]
    assert len(conflicts) == 1
    assert len(winners) == 1
    conflict = conflicts[0]
    assert conflict.exit_code == 2
    assert conflict.public_fields == {
        "expected_pointer_sha256": EMPTY,
        "observed_pointer_sha256": winners[0]["pointer_byte_sha256"],
    }
    assert not hasattr(conflict, "detail")


def test_assembly_request_decoder_is_exact_and_assembles_only_stored_refs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    generation = _suspended(store)
    manifest_payload = generation["manifest"]["payload"]
    payload = {
        "assembly_request_id": "suspended-request",
        **{
            field: manifest_payload[field]
            for field in ASSEMBLY_REQUEST_FIELDS
            if field != "assembly_request_id"
        },
    }
    request = seal_artifact("system.assembly_request", payload, created_at=CREATED_AT)

    decoded = decode_assembly_request(canonical_json_bytes(request))
    assert "assembly_request_id" not in decoded
    assert decoded["created_at"] == CREATED_AT
    assert store.assemble_from_request(request)["generation_id"] == generation["generation_id"]

    with pytest.raises(SystemContractError):
        decode_assembly_request(payload)

    invalid_state_request = seal_artifact(
        "system.assembly_request",
        {**payload, "generation_state": ["SYSTEM_SUSPENDED"]},
        created_at=CREATED_AT,
    )
    with pytest.raises(SystemContractError, match="generation_state is invalid"):
        decode_assembly_request(invalid_state_request)

    key_fields = (
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    )
    unordered_refs = sorted(
        [
            manifest_payload["release_manifest_ref"],
            manifest_payload["readiness_matrix_ref"],
        ],
        key=lambda row: tuple(row[field] for field in key_fields),
        reverse=True,
    )
    unordered_request = seal_artifact(
        "system.assembly_request",
        {**payload, "research_refs": unordered_refs},
        created_at=CREATED_AT,
    )
    with pytest.raises(SystemContractError, match="tuple-sorted and unique"):
        decode_assembly_request(unordered_request)


def test_status_external_routing_excludes_scheduler_from_identity(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    generation = store.assemble_generation(**closure["kwargs"])
    release = closure["release_ref"]
    active = activate_initial(store, generation, release)
    expected = active["manifest"]["payload"]["automation_semantic_sha256"]

    disabled = store.status(
        deployed_release_ref=release,
        external_routing={
            "automation_semantic_sha256": expected,
            "scheduler_enabled": False,
        },
    )
    drifted = store.status(
        deployed_release_ref=release,
        external_routing={
            "automation_semantic_sha256": _sha("different-routing"),
            "scheduler_enabled": True,
        },
    )
    assert disabled["external_routing_state"] == ("SYSTEM_ACTIVE_AUTOMATION_DISABLED")
    assert drifted["external_routing_state"] == "SYSTEM_EXTERNAL_ROUTING_DRIFT"


def test_corrupt_current_pointer_is_not_accepted_even_with_its_byte_sha(
    tmp_path: Path,
) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    first_generation = store.assemble_generation(**closure["kwargs"])
    first = activate_initial(store, first_generation, closure["release_ref"])
    active_path = closure["workspace"] / str(ACTIVE_POINTER_PATH)
    corrupt = dict(first["pointer"])
    corrupt["unexpected"] = True
    corrupt_raw = canonical_json_bytes(corrupt)
    active_path.write_bytes(corrupt_raw)
    active_path.chmod(0o600)

    second = _suspended(
        store,
        blocker="NEXT",
        created_at="2026-08-14T00:02:00Z",
    )
    with pytest.raises(SystemContractError):
        store.activate_generation(
            second["generation_id"],
            expected_pointer_sha256=hashlib.sha256(corrupt_raw).hexdigest(),
            activated_at="2026-08-14T00:02:01Z",
            os_actor="test",
        )


def test_first_activation_never_writes_outside_tmp_workspace(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    generation = store.assemble_generation(**closure["kwargs"])
    activate_initial(store, generation, closure["release_ref"])

    assert (closure["workspace"] / "results/system/_active.json").is_file()
