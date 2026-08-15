from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import threading
import tracemalloc
from typing import Any

import pytest
import quant_investor.system.store as system_store_module

from quant_investor.contracts import canonical_json_bytes, get_contract, seal_artifact
from quant_investor.system import (
    COMPONENT_REGISTRY_SHA256,
    EMPTY,
    MAXIMUM_DECODE_RESERVATION_BYTES,
    SOURCE_VERIFICATION_SNAPSHOT_CONTRACT_SHA256,
    SystemCASMismatch,
    SystemContractError,
    SystemImmutableConflict,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStore,
    VALIDATION_COMPLETION_CONTRACT_SHA256,
    VALIDATION_CUSTODY_RECORD_CONTRACT_SHA256,
    VALIDATION_INTENT_CONTRACT_SHA256,
    VALIDATION_PREPARED_CONTRACT_SHA256,
    object_ref_for_artifact,
    validate_candidate_transaction_intent,
    validate_custody_record,
    validate_source_verification_snapshot,
    validate_validation_completion,
    validate_validation_intent,
    validate_validation_prepared,
)
from quant_investor.system.validation_records import (
    build_custody_record,
    build_source_verification_snapshot,
    build_validation_completion,
    build_validation_intent,
    build_validation_prepared,
    completion_id,
    custody_record_id,
    prepared_id,
    validation_intent_id,
)

STAMP = "2026-08-14T00:00:00Z"
SHA = "1" * 64


def _release() -> dict[str, Any]:
    return seal_artifact(
        "system.release",
        {
            "release_id": "release-record-test",
            "state": "INSTALLED",
            "code_sha256": "2" * 64,
            "wheel_sha256": "3" * 64,
            "code_manifest_sha256": "4" * 64,
        },
        created_at=STAMP,
    )


def _ref(kind: str, artifact_id: str) -> dict[str, str]:
    definition = get_contract(kind)
    semantic = hashlib.sha256(f"semantic:{kind}:{artifact_id}".encode("utf-8")).hexdigest()
    byte_sha = hashlib.sha256(f"bytes:{kind}:{artifact_id}".encode("utf-8")).hexdigest()
    return {
        "kind": kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": semantic,
        "byte_sha256": byte_sha,
    }


def _records() -> dict[str, dict[str, Any]]:
    release_ref = object_ref_for_artifact(_release())
    request_ref = _ref("system.validation_run_request", "request-record-test")
    validator_ref = _ref("factor.validator_manifest", "validator-record-test")
    receipt_ref = _ref("factor.validation_receipt", "receipt-record-test")
    namespace = "factor-validation-namespace-record-test"
    intent_id = validation_intent_id(namespace, request_ref["artifact_id"])
    intent = build_validation_intent(
        candidate_state_pointer_sha256=EMPTY,
        candidate_state_ref=None,
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        factor_source_object_count=1,
        factor_source_stat_tree_sha256="5" * 64,
        factor_source_total_bytes=3,
        factor_validator_manifest_ref=validator_ref,
        installed_code_manifest_sha256="6" * 64,
        intent_id=intent_id,
        intrinsic_receipt_ref=receipt_ref,
        maximum_total_factor_source_bytes=10,
        plan_sha256="7" * 64,
        release_manifest_ref=release_ref,
        trusted_at=STAMP,
        validation_lane="BOOTSTRAP",
        validation_namespace_id=namespace,
        validation_profile_id="factor-bootstrap-contextual-validation",
        validation_request_ref=request_ref,
    )
    intent_raw = canonical_json_bytes(intent)
    context_ref = _ref("factor.contextual_validation_result", "context-record-test")
    attestation_ref = _ref("system.validation_attestation", "attestation-record-test")
    prepared = build_validation_prepared(
        contextual_result_ref=context_ref,
        intent_id=intent_id,
        intent_semantic_sha256=intent["semantic_sha256"],
        intent_sha256=hashlib.sha256(intent_raw).hexdigest(),
        plan_sha256="7" * 64,
        prepared_id=prepared_id(intent_id),
        trusted_at=STAMP,
        validation_attestation_ref=attestation_ref,
        validation_namespace_id=namespace,
        validation_request_ref=request_ref,
    )
    custody = build_custody_record(
        record_id=custody_record_id(attestation_ref),
        validation_request_ref=request_ref,
        attestation_ref=attestation_ref,
        contextual_result_ref=context_ref,
        release_manifest_ref=release_ref,
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        recorded_at=STAMP,
        os_actor="uid:501",
    )
    stat_identity = {
        "st_ctime_ns": 1,
        "st_dev": 2,
        "st_gid": 3,
        "st_ino": 4,
        "st_mode": 0o100600,
        "st_mtime_ns": 5,
        "st_nlink": 1,
        "st_size": 3,
        "st_uid": 501,
    }
    source_ref = _ref("system.source_object", "source-record-test")
    stat_row = {
        "source_binding_sha256": "8" * 64,
        "source_object_ref": source_ref,
        "stat_identity": stat_identity,
        "stat_identity_sha256": hashlib.sha256(canonical_json_bytes(stat_identity)).hexdigest(),
    }
    snapshot = build_source_verification_snapshot(
        factor_source_total_bytes=3,
        installed_code_manifest_sha256="6" * 64,
        maximum_total_factor_source_bytes=10,
        source_object_count=1,
        source_object_refs=[source_ref],
        source_stat_rows=[stat_row],
        source_stat_tree_sha256=hashlib.sha256(canonical_json_bytes([stat_row])).hexdigest(),
        unique_source_binding_count=1,
        validation_attestation_ref=attestation_ref,
    )
    completion = build_validation_completion(
        completion_id=completion_id(intent_id),
        contextual_result_ref=context_ref,
        custody_record_sha256=hashlib.sha256(canonical_json_bytes(custody)).hexdigest(),
        intent_semantic_sha256=intent["semantic_sha256"],
        intent_sha256=hashlib.sha256(intent_raw).hexdigest(),
        prepared_sha256=hashlib.sha256(canonical_json_bytes(prepared)).hexdigest(),
        source_verification_snapshot_sha256=hashlib.sha256(
            canonical_json_bytes(snapshot)
        ).hexdigest(),
        trusted_at=STAMP,
        validation_attestation_ref=attestation_ref,
        validation_namespace_id=namespace,
        validation_request_ref=request_ref,
    )
    return {
        "intent": intent,
        "prepared": prepared,
        "custody": custody,
        "snapshot": snapshot,
        "completion": completion,
    }


def test_frozen_special_record_contracts_and_semantic_tamper() -> None:
    assert VALIDATION_INTENT_CONTRACT_SHA256 == (
        "4c91bfd608e6b1409d95501ca7389ed62e0a112cdeba51c093ccce38dde9c435"
    )
    assert VALIDATION_PREPARED_CONTRACT_SHA256 == (
        "ddf9108cfa5ee5b7b228f271e8e7996ce49d5480cb52901ecf3e35b1bc6aacc0"
    )
    assert VALIDATION_CUSTODY_RECORD_CONTRACT_SHA256 == (
        "df7494449e9c5404b7cd6d51d40732151591cd7d3415c3b90709791e3796e6f1"
    )
    assert SOURCE_VERIFICATION_SNAPSHOT_CONTRACT_SHA256 == (
        "cafedfacc0a7ac5eaccf10f9bffd07b2c119af011e303f60e7b6a9c5b6c89693"
    )
    assert VALIDATION_COMPLETION_CONTRACT_SHA256 == (
        "eb982739c098a65e8ab5fa894b0fd7de0ac3678111939518ea31ca9d4fca2ef9"
    )
    records = _records()
    validators = {
        "intent": validate_validation_intent,
        "prepared": validate_validation_prepared,
        "custody": validate_custody_record,
        "snapshot": validate_source_verification_snapshot,
        "completion": validate_validation_completion,
    }
    for name, validator in validators.items():
        assert validator(canonical_json_bytes(records[name])) == records[name]
        tampered = copy.deepcopy(records[name])
        tampered["semantic_sha256"] = "f" * 64
        with pytest.raises(SystemContractError):
            validator(tampered)
        unknown = copy.deepcopy(records[name])
        unknown["unexpected"] = True
        with pytest.raises(SystemContractError):
            validator(unknown)


def _composite(
    namespace: str,
    identity: str,
    *,
    previous_composite_state_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    return seal_artifact(
        "factor.composite_state",
        {
            "composite_state_id": identity,
            "custody_namespace_id": namespace,
            "preregistration_ref": None,
            "cycle_state": "CANDIDATE",
            "transaction_sequence": 0,
            "previous_composite_state_ref": previous_composite_state_ref,
            "transaction_id": identity,
            "custody_record_count": 0,
            "custody_head_ref": None,
            "selection_ref": None,
            "signal_capture_count": 0,
            "signal_capture_head_ref": None,
            "observation_count": 0,
            "observation_head_ref": None,
            "execution_evidence_ref": None,
            "evaluation_ref": None,
            "admitted_set_ref": None,
            "intrinsic_receipt_ref": None,
            "resolved_signal_slot_count": 0,
            "resolved_label_slot_count": 0,
            "slot_tree_sha256": SHA,
            "terminal": False,
            "blockers": [],
            "last_stored_at": STAMP,
            "authority": "NON_AUTHORIZING",
        },
        created_at=STAMP,
    )


def test_candidate_state_is_per_namespace_nonauthorizing_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SystemStore(tmp_path)
    monkeypatch.setattr(system_store_module, "_utc_now", lambda: STAMP)
    namespace = "candidate-namespace"
    first_intent = store.begin_candidate_transaction(
        namespace,
        "candidate-one",
        expected_pointer_sha256=EMPTY,
        transaction_plan={"operation_request_sha256": "a" * 64, "input_refs": []},
    )
    assert set(first_intent) == {
        "domain",
        "intent_id",
        "validation_namespace_id",
        "transaction_id",
        "expected_pointer_sha256",
        "previous_candidate_state_ref",
        "transaction_plan",
        "transaction_plan_sha256",
        "trusted_at",
        "clock_source",
        "authority",
        "semantic_sha256",
    }
    assert first_intent["trusted_at"] == STAMP
    assert store.read_candidate_transaction(namespace, "candidate-one") == first_intent
    assert validate_candidate_transaction_intent(first_intent) == first_intent
    with pytest.raises(SystemImmutableConflict):
        store.begin_candidate_transaction(
            namespace,
            "candidate-one",
            expected_pointer_sha256=EMPTY,
            transaction_plan={"operation_request_sha256": "b" * 64, "input_refs": []},
        )
    first_ref = store.put_object(_composite(namespace, "candidate-one"))
    first = store.compare_and_swap_candidate_state(
        namespace,
        first_ref,
        expected_pointer_sha256=EMPTY,
    )
    assert first["candidate_state_ref"] == first_ref
    assert first["pointer"]["authority"] == "NON_AUTHORIZING"

    with pytest.raises(SystemCASMismatch) as mismatch:
        store.compare_and_swap_candidate_state(
            namespace,
            first_ref,
            expected_pointer_sha256=EMPTY,
        )
    assert mismatch.value.public_fields == {
        "expected_pointer_sha256": EMPTY,
        "observed_pointer_sha256": first["pointer_byte_sha256"],
    }

    store.begin_candidate_transaction(
        namespace,
        "candidate-two",
        expected_pointer_sha256=first["pointer_byte_sha256"],
        transaction_plan={"operation_request_sha256": "c" * 64, "input_refs": []},
    )
    second_ref = store.put_object(_composite(namespace, "candidate-two"))
    with pytest.raises(SystemContractError, match="intent"):
        store.compare_and_swap_candidate_state(
            namespace,
            second_ref,
            expected_pointer_sha256=first["pointer_byte_sha256"],
        )

    store.begin_candidate_transaction(
        namespace,
        "candidate-two-linked",
        expected_pointer_sha256=first["pointer_byte_sha256"],
        transaction_plan={"operation_request_sha256": "d" * 64, "input_refs": []},
    )
    linked_ref = store.put_object(
        _composite(
            namespace,
            "candidate-two-linked",
            previous_composite_state_ref=first_ref,
        )
    )
    second = store.compare_and_swap_candidate_state(
        namespace,
        linked_ref,
        expected_pointer_sha256=first["pointer_byte_sha256"],
    )
    assert second["candidate_state_ref"] == linked_ref


def test_candidate_transaction_pre_cas_retry_reuses_exact_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = "candidate-retry-namespace"
    transaction_id = "candidate-retry"
    plan = {"operation_request_sha256": "e" * 64, "input_refs": []}
    first_store = SystemStore(tmp_path)
    monkeypatch.setattr(system_store_module, "_utc_now", lambda: STAMP)
    first = first_store.begin_candidate_transaction(
        namespace,
        transaction_id,
        expected_pointer_sha256=EMPTY,
        transaction_plan=plan,
    )

    def forbidden_clock() -> str:
        raise AssertionError("retry sampled a second transaction timestamp")

    monkeypatch.setattr(system_store_module, "_utc_now", forbidden_clock)
    retry_store = SystemStore(tmp_path)
    retry = retry_store.begin_candidate_transaction(
        namespace,
        transaction_id,
        expected_pointer_sha256=EMPTY,
        transaction_plan=plan,
    )
    assert retry == first
    candidate_ref = retry_store.put_object(_composite(namespace, transaction_id))
    committed = retry_store.compare_and_swap_candidate_state(
        namespace,
        candidate_ref,
        expected_pointer_sha256=EMPTY,
    )
    assert committed["candidate_state_ref"] == candidate_ref


def test_candidate_transaction_clock_rollback_writes_no_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = "candidate-clock-namespace"
    store = SystemStore(tmp_path)
    monkeypatch.setattr(system_store_module, "_utc_now", lambda: STAMP)
    store.begin_candidate_transaction(
        namespace,
        "candidate-clock-one",
        expected_pointer_sha256=EMPTY,
        transaction_plan={"operation_request_sha256": "f" * 64, "input_refs": []},
    )
    first_ref = store.put_object(_composite(namespace, "candidate-clock-one"))
    first = store.compare_and_swap_candidate_state(
        namespace,
        first_ref,
        expected_pointer_sha256=EMPTY,
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        system_store_module,
        "_utc_now",
        lambda: "2026-08-13T23:59:59Z",
    )
    with pytest.raises(SystemPreconditionError) as rollback:
        store.begin_candidate_transaction(
            namespace,
            "candidate-clock-two",
            expected_pointer_sha256=first["pointer_byte_sha256"],
            transaction_plan={
                "operation_request_sha256": "0" * 64,
                "input_refs": [],
            },
        )
    assert rollback.value.code == "SYSTEM_CLOCK_ROLLBACK"
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_concurrent_candidate_transaction_begin_samples_one_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SystemStore(tmp_path)
    namespace = "candidate-concurrent-namespace"
    transaction_id = "candidate-concurrent"
    plan = {"operation_request_sha256": "9" * 64, "input_refs": []}
    barrier = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def trusted_time() -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
        return STAMP

    monkeypatch.setattr(system_store_module, "_utc_now", trusted_time)

    def begin() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                store.begin_candidate_transaction(
                    namespace,
                    transaction_id,
                    expected_pointer_sha256=EMPTY,
                    transaction_plan=plan,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=begin) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    assert calls == 1
    assert len(results) == 2 and results[0] == results[1]


def test_candidate_cas_requires_untampered_transaction_intent(tmp_path: Path) -> None:
    store = SystemStore(tmp_path)
    namespace = "candidate-required-intent-namespace"
    candidate = store.put_object(_composite(namespace, "candidate-without-intent"))
    with pytest.raises(SystemPreconditionError) as absent:
        store.compare_and_swap_candidate_state(
            namespace,
            candidate,
            expected_pointer_sha256=EMPTY,
        )
    assert absent.value.code == "SYSTEM_CANDIDATE_TRANSACTION_REQUIRED"
    assert store.read_candidate_state(namespace) is None

    intent = store.begin_candidate_transaction(
        namespace,
        "candidate-without-intent",
        expected_pointer_sha256=EMPTY,
        transaction_plan={"operation_request_sha256": "8" * 64, "input_refs": []},
    )
    path = next(tmp_path.rglob("intent.json"))
    tampered = {**intent, "semantic_sha256": "7" * 64}
    path.write_bytes(canonical_json_bytes(tampered))
    path.chmod(0o600)
    with pytest.raises(SystemContractError, match="semantic"):
        store.read_candidate_transaction(namespace, "candidate-without-intent")
    assert store.read_candidate_state(namespace) is None


def test_source_byte_seam_rejects_non_owner_only_mode(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir(mode=0o700)
    source = source_root / "sample.json"
    source.write_bytes(b"{}")
    source.chmod(0o600)
    store = SystemStore(tmp_path, source_root=source_root, max_source_bytes=16)
    ref = store.put_source_file(
        "sample.json",
        source_object_id="source-sample",
        media_type="application/json",
        source_format="JSON",
        created_at=STAMP,
    )
    payload, raw = store.read_source_object_bytes(ref, maximum_bytes=16)
    assert payload["byte_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(SystemSecurityError, match="bound"):
        store.read_source_object_bytes(ref, maximum_bytes=17)
    source.chmod(0o644)
    with pytest.raises(SystemSecurityError):
        store.read_source_object_bytes(ref, maximum_bytes=16)


def _stream_source(
    tmp_path: Path, *, name: str = "source.parquet"
) -> tuple[SystemStore, Path, dict[str, str]]:
    workspace = tmp_path / f"workspace-{name}"
    source_root = tmp_path / f"source-root-{name}"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    path = source_root / name
    with path.open("wb") as handle:
        for _ in range(6):
            handle.write(b"P" * (1024 * 1024))
    path.chmod(0o600)
    store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id=f"source-root-{name}",
    )
    ref = store.put_source_file(
        name,
        source_object_id=f"source-{name}",
        media_type="application/vnd.apache.parquet",
        source_format="PARQUET",
        created_at=STAMP,
    )
    return store, path, ref


def test_source_stream_is_seekable_and_never_materializes_whole_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, ref = _stream_source(tmp_path)

    def forbidden_byte_read(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("streaming seam used whole-file bytes")

    monkeypatch.setattr(
        store._source_storage,
        "read_workspace_file_bytes",
        forbidden_byte_read,
    )
    tracemalloc.start()
    try:
        with store.open_source_object(
            ref,
            maximum_bytes=8 * 1024 * 1024,
            decoded_reservation_bytes=1024 * 1024,
        ) as (payload, stream):
            assert payload["source_format"] == "PARQUET"
            assert stream.readable() is True
            assert stream.seekable() is True
            assert stream.tell() == 0
            assert stream.read(4) == b"PPPP"
            assert stream.seek(1024 * 1024) == 1024 * 1024
            target = bytearray(4)
            assert stream.readinto(target) == 4
            assert bytes(target) == b"PPPP"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def test_source_stream_detects_path_drift_and_releases_lease(tmp_path: Path) -> None:
    store, path, ref = _stream_source(tmp_path)
    with pytest.raises(SystemSecurityError, match="changed"):
        with store.open_source_object(
            ref,
            maximum_bytes=8 * 1024 * 1024,
            decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
        ):
            path.write_bytes(b"PAR1drift")
            path.chmod(0o600)

    with pytest.raises(SystemContractError, match="hash"):
        with store.open_source_object(
            ref,
            maximum_bytes=8 * 1024 * 1024,
            decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
        ):
            pass


def test_source_stream_weighted_budget_serializes_concurrent_decodes(
    tmp_path: Path,
) -> None:
    store, _, ref = _stream_source(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with store.open_source_object(
                ref,
                maximum_bytes=8 * 1024 * 1024,
                decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
            ):
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("first decode lease was not released by test")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def second() -> None:
        second_started.set()
        try:
            with store.open_source_object(
                ref,
                maximum_bytes=8 * 1024 * 1024,
                decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
            ):
                second_entered.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert second_started.wait(timeout=5)
    assert second_entered.wait(timeout=0.1) is False
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
    assert second_entered.is_set() is True
    assert errors == []


def test_source_stream_weighted_budget_is_process_global_across_stores(
    tmp_path: Path,
) -> None:
    first_store, _, first_ref = _stream_source(tmp_path, name="first.parquet")
    second_store, _, second_ref = _stream_source(tmp_path, name="second.parquet")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with first_store.open_source_object(
                first_ref,
                maximum_bytes=8 * 1024 * 1024,
                decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
            ):
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("first cross-store lease was not released")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def second() -> None:
        try:
            with second_store.open_source_object(
                second_ref,
                maximum_bytes=8 * 1024 * 1024,
                decoded_reservation_bytes=MAXIMUM_DECODE_RESERVATION_BYTES,
            ):
                second_entered.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert second_entered.wait(timeout=0.1) is False
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
    assert second_entered.is_set() is True
    assert errors == []


def test_system_core_has_no_concrete_factor_business_semantics() -> None:
    system_root = Path(__file__).parents[2] / "quant_investor" / "system"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(system_root.glob("*.py")))
    for forbidden in (
        "pv_low_dollar_volume_5d",
        "pv_blend_volstab19x2_mom90_amihud5_w75",
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "0.500000000000",
        "factor_definitions",
        "weight_total",
    ):
        assert forbidden not in text
