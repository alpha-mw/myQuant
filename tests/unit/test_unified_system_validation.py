from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import threading
from typing import Any

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    COMPONENT_REGISTRY_SHA256,
    MAXIMUM_VALIDATION_RSS_BYTES,
    SystemContractError,
    SystemImmutableConflict,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
    SystemStore,
)
from quant_investor.system.components import validation_profile
import quant_investor.system.validation as validation_module

STAMP = "2026-08-14T00:00:00Z"


def _request_roots(store: SystemStore) -> tuple[dict[str, str], ...]:
    release_ref = store.put_object(
        seal_artifact(
            "system.release",
            {
                "release_id": "validation-request-release",
                "state": "TEST",
                "code_sha256": "1" * 64,
                "wheel_sha256": "2" * 64,
                "code_manifest_sha256": "3" * 64,
            },
            created_at=STAMP,
        )
    )
    validator_ref = store.put_object(
        seal_artifact(
            "factor.validator_manifest",
            {
                "validator_manifest_id": "validation-request-validator",
                "release_manifest_ref": release_ref,
                "contextual_validator_component_ref": release_ref,
                "source_decoder_component_ref": release_ref,
                "implementation_rows": [],
                "validated_contracts": [],
                "authority": "NON_AUTHORIZING",
            },
            created_at=STAMP,
        )
    )
    receipt_ref = store.put_object(
        seal_artifact(
            "factor.validation_receipt",
            {
                "validation_receipt_id": "validation-request-receipt",
                "policy_ref": release_ref,
                "evidence_refs": [release_ref],
                "active_set_ref": release_ref,
                "validated": True,
                "authority": "NON_AUTHORIZING",
            },
            created_at=STAMP,
        )
    )
    return release_ref, validator_ref, receipt_ref


def test_validation_request_is_exact_deterministic_eight_field_closure(
    tmp_path: Path,
) -> None:
    store = SystemStore(tmp_path)
    release_ref, validator_ref, receipt_ref = _request_roots(store)
    result = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=validator_ref,
        intrinsic_receipt_ref=receipt_ref,
    )
    request = result["validation_request"]
    payload = request["payload"]
    assert set(payload) == {
        "validation_request_id",
        "validation_profile_id",
        "component_registry_sha256",
        "validation_namespace_id",
        "release_manifest_ref",
        "factor_validator_manifest_ref",
        "intrinsic_receipt_ref",
        "candidate_state_ref",
    }
    assert payload["validation_profile_id"] == BOOTSTRAP_VALIDATION_PROFILE
    assert payload["component_registry_sha256"] == COMPONENT_REGISTRY_SHA256
    assert payload["candidate_state_ref"] is None
    body = {key: value for key, value in payload.items() if key != "validation_request_id"}
    assert (
        payload["validation_request_id"]
        == hashlib.sha256(
            canonical_json_bytes({"domain": "myquant-validation-run-request-id", **body})
        ).hexdigest()
    )
    assert store.get_object(result["validation_request_ref"]) == request


def test_concurrent_validation_request_builds_publish_one_exact_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SystemStore(tmp_path)
    release_ref, validator_ref, receipt_ref = _request_roots(store)
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(2)
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def sampled_time() -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
            return "2026-08-14T00:00:01Z"

    monkeypatch.setattr("quant_investor.system.validation._utc_now", sampled_time)

    def build() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                store.build_validation_run_request(
                    release_manifest_ref=release_ref,
                    factor_validator_manifest_ref=validator_ref,
                    intrinsic_receipt_ref=receipt_ref,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert calls == 1
    assert results[0] == results[1]


def _runner_fixture(tmp_path: Path) -> dict[str, Any]:
    workspace = tmp_path / "runner-workspace"
    source_root = tmp_path / "runner-sources"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    source_path = source_root / "runner.parquet"
    source_path.write_bytes(b"PAR1-system-runner-protocol")
    source_path.chmod(0o600)
    store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="runner-source-root",
    )
    release_ref, validator_ref, receipt_ref = _request_roots(store)
    source_ref = store.put_source_file(
        "runner.parquet",
        source_object_id="runner-source-object",
        media_type="application/vnd.apache.parquet",
        source_format="PARQUET",
        created_at=STAMP,
    )
    request_result = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=validator_ref,
        intrinsic_receipt_ref=receipt_ref,
    )
    request_payload = request_result["validation_request"]["payload"]
    inspected = store.inspect_source_object(
        source_ref,
        full_hash=True,
        maximum_bytes=1024 * 1024,
    )
    binding_sha = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-source-binding",
                "source_root_id": inspected["source_root_id"],
                "relative_path": inspected["relative_path"],
            }
        )
    ).hexdigest()
    stat_identity = inspected["stat_identity"]
    stat_row = {
        "source_binding_sha256": binding_sha,
        "source_object_ref": source_ref,
        "stat_identity": stat_identity,
        "stat_identity_sha256": hashlib.sha256(canonical_json_bytes(stat_identity)).hexdigest(),
    }
    evidence_refs = [release_ref]
    plan = {
        "domain": "myquant-validation-run-plan",
        "validation_namespace_id": request_payload["validation_namespace_id"],
        "validation_profile_id": BOOTSTRAP_VALIDATION_PROFILE,
        "validation_lane": "BOOTSTRAP",
        "component_registry_sha256": COMPONENT_REGISTRY_SHA256,
        "release_manifest_ref": release_ref,
        "installed_code_manifest_sha256": "4" * 64,
        "factor_validator_manifest_ref": validator_ref,
        "contextual_validator_component_ref": release_ref,
        "source_decoder_component_ref": release_ref,
        "implementation_component_refs": [],
        "intrinsic_receipt_ref": receipt_ref,
        "policy_ref": release_ref,
        "evidence_refs": evidence_refs,
        "active_set_ref": release_ref,
        "candidate_state_ref": None,
        "candidate_state_pointer_sha256": "EMPTY",
        "source_attestation_refs": [],
        "source_object_refs": [source_ref],
        "source_stat_rows": [stat_row],
        "source_stat_tree_sha256": hashlib.sha256(canonical_json_bytes([stat_row])).hexdigest(),
        "factor_source_total_bytes": inspected["size"],
        "maximum_total_factor_source_bytes": 2 * 1024**3,
        "custody_record_refs": [],
        "custody_head_ref": None,
        "custody_tree_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        "compiled_contracts": [],
    }
    derived = {
        "profile": validation_profile(BOOTSTRAP_VALIDATION_PROFILE),
        "receipt": {},
        "candidate": None,
        "factor_manifest": store.get_object(validator_ref),
        "release": store.get_object(release_ref),
        "plan": plan,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
    }
    context_payload = {
        "contextual_result_id": "system-runner-context",
        "validation_namespace_id": plan["validation_namespace_id"],
        "lane": plan["validation_lane"],
        "intrinsic_receipt_ref": receipt_ref,
        "policy_ref": release_ref,
        "evidence_refs": evidence_refs,
        "active_set_ref": release_ref,
        "composite_state_ref": None,
        "factor_validator_manifest_ref": validator_ref,
        "contextual_validator_component_ref": release_ref,
        "source_decoder_component_ref": release_ref,
        "implementation_component_refs": [],
        "source_attestation_refs": [],
        "source_object_refs": [source_ref],
        "custody_record_refs": [],
        "custody_tree_sha256": plan["custody_tree_sha256"],
        "custody_head_ref": None,
        "validated": True,
        "blockers": [],
        "authority": "NON_AUTHORIZING",
    }
    return {
        "store": store,
        "workspace": workspace,
        "source_path": source_path,
        "source_ref": source_ref,
        "request": request_result["validation_request"],
        "request_ref": request_result["validation_request_ref"],
        "derived": derived,
        "context_payload": context_payload,
    }


def _install_runner_test_seams(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
    *,
    callback_count: list[int],
) -> None:
    def derive_plan(
        store: SystemStore,
        request_payload: dict[str, Any],
        *,
        full_source_hash: bool,
    ) -> dict[str, Any]:
        del store, request_payload, full_source_hash
        return copy.deepcopy(fixture["derived"])

    def invoke_callback(
        store: SystemStore,
        *,
        profile: dict[str, Any],
        validation_request: dict[str, Any],
        trusted_at: str,
    ) -> dict[str, Any]:
        del store, profile, validation_request, trusted_at
        callback_count[0] += 1
        return copy.deepcopy(fixture["context_payload"])

    monkeypatch.setattr(validation_module, "_derive_plan", derive_plan)
    monkeypatch.setattr(validation_module, "_invoke_callback", invoke_callback)
    monkeypatch.setattr(validation_module, "_utc_now", lambda: STAMP)


def test_prepared_recovery_never_reinvokes_callback_and_reconstructs_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    store = fixture["store"]
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)
    publish_custody = validation_module._publish_custody

    def crash_after_prepared(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise SystemStorageError("injected crash after prepared")

    monkeypatch.setattr(validation_module, "_publish_custody", crash_after_prepared)
    with pytest.raises(SystemStorageError, match="injected crash"):
        store.run_validation(fixture["request_ref"])
    assert callback_count == [1]
    assert list(fixture["workspace"].rglob("prepared.json"))
    assert list(fixture["workspace"].rglob("completion.json")) == []

    monkeypatch.setattr(validation_module, "_publish_custody", publish_custody)
    result = store.run_validation(fixture["request_ref"])
    assert callback_count == [1]
    assert set(result) == {
        "outcome",
        "validation_request",
        "validation_request_ref",
        "validation_intent",
        "validation_prepared",
        "contextual_result",
        "contextual_result_ref",
        "validation_attestation",
        "validation_attestation_ref",
        "custody_record",
        "source_verification_snapshot",
        "validation_completion",
        "completion_sha256",
    }
    assert result["outcome"] == "VALIDATED"
    assert store.run_validation(fixture["request_ref"]) == result
    assert callback_count == [1]

    custody_directories = list(
        (fixture["workspace"] / "results/system/validation_custody").iterdir()
    )
    assert len(custody_directories) == 1
    custody = custody_directories[0]
    assert custody.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in custody.iterdir()} == {
        "contextual_result.json",
        "attestation.json",
        "record.json",
    }
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in custody.iterdir())


def test_plain_self_sealed_attestation_is_inert_without_protected_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)
    result = fixture["store"].run_validation(fixture["request_ref"])
    forged_payload = copy.deepcopy(result["validation_attestation"]["payload"])
    forged_payload["attestation_id"] = "plain-self-sealed-attestation"
    forged = seal_artifact(
        "system.validation_attestation",
        forged_payload,
        created_at=STAMP,
    )
    forged_ref = fixture["store"].put_object(forged)

    with pytest.raises(SystemContractError):
        fixture["store"].resolve_validation_attestation(
            forged_ref,
            verification_level="full",
        )
    assert callback_count == [1]


def test_source_drift_after_prepared_blocks_completion_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)
    publish_custody = validation_module._publish_custody

    def assert_bound_source_stable(
        store: SystemStore,
        *,
        request_payload: dict[str, Any],
        derived: dict[str, Any],
    ) -> None:
        del request_payload, derived
        store.inspect_source_object(
            fixture["source_ref"],
            full_hash=True,
            maximum_bytes=1024 * 1024,
        )

    def drift_after_custody(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        result = publish_custody(*args, **kwargs)
        fixture["source_path"].write_bytes(b"PAR1-drifted-after-prepared")
        fixture["source_path"].chmod(0o600)
        return result

    monkeypatch.setattr(validation_module, "_publish_custody", drift_after_custody)
    monkeypatch.setattr(validation_module, "_assert_plan_stable", assert_bound_source_stable)
    with pytest.raises(SystemContractError):
        fixture["store"].run_validation(fixture["request_ref"])
    assert callback_count == [1]
    assert list(fixture["workspace"].rglob("completion.json")) == []


def _governed_file_snapshot(workspace: Path) -> dict[str, bytes]:
    root = workspace / "results/system"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_clock_rollback_retry_writes_no_additional_validation_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)

    def fail_callback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        callback_count[0] += 1
        raise SystemStorageError("injected callback interruption")

    monkeypatch.setattr(validation_module, "_invoke_callback", fail_callback)
    with pytest.raises(SystemStorageError, match="interruption"):
        fixture["store"].run_validation(fixture["request_ref"])
    before = _governed_file_snapshot(fixture["workspace"])

    monkeypatch.setattr(validation_module, "_utc_now", lambda: "2026-08-13T23:59:59Z")
    with pytest.raises(SystemPreconditionError) as rollback:
        fixture["store"].run_validation(fixture["request_ref"])
    assert rollback.value.code == "SYSTEM_CLOCK_ROLLBACK"
    assert _governed_file_snapshot(fixture["workspace"]) == before
    assert callback_count == [1]


def test_prepared_plan_drift_blocks_retry_without_callback_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)

    monkeypatch.setattr(
        validation_module,
        "_publish_custody",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SystemStorageError("injected crash after prepared")
        ),
    )
    with pytest.raises(SystemStorageError, match="injected crash"):
        fixture["store"].run_validation(fixture["request_ref"])
    assert callback_count == [1]

    fixture["derived"]["plan"]["installed_code_manifest_sha256"] = "9" * 64
    fixture["derived"]["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(fixture["derived"]["plan"])
    ).hexdigest()
    with pytest.raises(SystemImmutableConflict, match="intent plan"):
        fixture["store"].run_validation(fixture["request_ref"])
    assert callback_count == [1]
    assert list(fixture["workspace"].rglob("completion.json")) == []


def test_concurrent_validation_runs_publish_one_completion_and_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)
    callback_started = threading.Event()
    release_callback = threading.Event()
    original_invoke = validation_module._invoke_callback

    def blocking_callback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        callback_started.set()
        if not release_callback.wait(timeout=5):
            raise AssertionError("callback release was not signaled")
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(validation_module, "_invoke_callback", blocking_callback)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(fixture["store"].run_validation(fixture["request_ref"]))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert callback_started.wait(timeout=5)
    second.start()
    release_callback.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert first.is_alive() is False
    assert second.is_alive() is False
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert callback_count == [1]
    assert len(list(fixture["workspace"].rglob("completion.json"))) == 1


def test_corrupt_prepared_mapping_fails_closed_without_callback_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path)
    callback_count = [0]
    _install_runner_test_seams(monkeypatch, fixture, callback_count=callback_count)
    monkeypatch.setattr(
        validation_module,
        "_publish_custody",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SystemStorageError("injected crash after prepared")
        ),
    )
    with pytest.raises(SystemStorageError):
        fixture["store"].run_validation(fixture["request_ref"])
    prepared_path = next(fixture["workspace"].rglob("prepared.json"))
    prepared_path.write_bytes(b"{}")
    prepared_path.chmod(0o600)

    with pytest.raises(SystemContractError):
        fixture["store"].run_validation(fixture["request_ref"])
    assert callback_count == [1]
    assert list(fixture["workspace"].rglob("completion.json")) == []


def test_runner_rejects_process_rss_above_the_hard_limit_before_callback_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_module, "_open_fd_count", lambda: 0)
    monkeypatch.setattr(
        validation_module,
        "_resident_rss_bytes",
        lambda: MAXIMUM_VALIDATION_RSS_BYTES + 1,
    )
    with pytest.raises(SystemSecurityError, match="RSS"):
        validation_module._invoke_callback(
            object(),
            profile={
                "callback_module": "never.imported",
                "callback_qualified_name": "never",
                "validation_lane": "BOOTSTRAP",
            },
            validation_request={},
            trusted_at=STAMP,
        )


def test_invocation_worker_rejects_transient_peak_after_memory_is_released() -> None:
    def transient_peak(**_kwargs: Any) -> dict[str, bool]:
        allocation = bytearray(MAXIMUM_VALIDATION_RSS_BYTES)
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1
        del allocation
        return {"released_before_return": True}

    with pytest.raises(SystemSecurityError, match="RSS"):
        validation_module._run_callback_worker(
            transient_peak,
            store=object(),
            validation_request={},
            trusted_at=STAMP,
            maximum_seconds=30,
        )
