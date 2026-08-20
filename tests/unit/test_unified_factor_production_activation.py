"""Factor-only initial production authority, deliberately separate from System."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading

import pytest
import quant_investor.factors.production_authority as factor_authority

from quant_investor.contracts import (
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from quant_investor.cli.main import main
from quant_investor.cli.unified import (
    factor_production_activate,
    factor_production_signal,
    factor_production_status,
    factor_production_verify,
)
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.factors.production_authority import (
    FACTOR_ACTIVE_POINTER_PATH,
    FACTOR_PRODUCTION_MARKER_PATH,
    FactorReadOnlySystemCustody,
    FactorProductionStore,
)
from quant_investor.migration.canonical import write_idempotent_bytes
from quant_investor.migration.errors import UnifiedCutoverError
from quant_investor.system.store import SystemStore
from tests.unit import test_unified_factor_production_authority as source_fixture

STAMP = "2026-08-19T04:00:00Z"
ACTIVATED_AT = "2026-08-19T04:00:01Z"


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_factor_active_pointer_is_exactly_the_narrow_six_field_payload() -> None:
    active_pointer = {
        "factor_generation_id": "factor-production-generation-" + "1" * 64,
        "factor_generation_sha256": "2" * 64,
        "previous_pointer_sha256": "EMPTY",
        "activated_at": ACTIVATED_AT,
        "os_actor": f"uid:{os.geteuid()}",
        "authority_scope": "FACTOR_PRODUCTION",
    }
    pointer_raw = canonical_json_bytes(active_pointer)
    record_body = {**active_pointer, "pointer_raw_sha256": hashlib.sha256(pointer_raw).hexdigest()}
    record = seal_artifact(
        "factor.production_pointer",
        {
            "factor_production_pointer_id": "factor-production-pointer-" + _sha(record_body),
            **record_body,
        },
        created_at=ACTIVATED_AT,
    )
    assert factor_authority._factor_pointer_raw(record) == pointer_raw
    assert factor_authority.validate_factor_active_pointer(pointer_raw) == active_pointer
    system_shaped = {**active_pointer, "factor_generation_id": "1" * 64}
    with pytest.raises(FactorGovernanceError, match="generation identity"):
        factor_authority.validate_factor_active_pointer(canonical_json_bytes(system_shaped))


def test_factor_production_contracts_have_no_system_generation_linkage() -> None:
    for kind in (
        "factor.production_source_closure",
        "factor.production_recomputation_evidence",
        "factor.production_generation",
        "factor.production_generation_receipt",
        "factor.production_activation_bundle",
        "factor.production_marker",
    ):
        fields = get_contract(kind).required_payload_fields
        assert "generation_manifest_ref" not in fields
        assert "target_generation_manifest_ref" not in fields
    assert get_contract("factor.production_pointer").required_payload_fields == frozenset(
        {
            "factor_production_pointer_id",
            "factor_generation_id",
            "factor_generation_sha256",
            "previous_pointer_sha256",
            "activated_at",
            "os_actor",
            "authority_scope",
            "pointer_raw_sha256",
        }
    )
    assert get_contract("factor.production_market_pit_selection").identity_field == (
        "market_pit_selection_id"
    )


def test_only_public_factor_mutation_is_the_one_shot_cli_operator() -> None:
    for retired in ("prepare_factor_production_activation", "activate_factor_production"):
        assert retired not in factor_authority.__all__
        assert not hasattr(factor_authority, retired)


def _native_store_and_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FactorProductionStore, dict]:
    """Reuse the source lane's real native-custody fixture without a second tree.

    The source test creates strict nested calendar/PIT capture bundles, the
    typed Market snapshot bridge, and all closure/recompute artifacts in a
    temporary ``SystemStore``.  We capture that no-pointer store, then run the
    separate Factor authority preparation against its read-only custody.
    """

    captured: list[SystemStore] = []
    documents: list[dict] = []
    produced: dict[str, dict] = {}
    base_store = source_fixture.SystemStore
    base_evidence_builder = source_fixture.build_factor_production_recomputation_evidence

    class CapturingSystemStore(base_store):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured.append(self)

        def put_object(self, artifact: object) -> dict[str, str]:
            document = validate_artifact(artifact)  # type: ignore[arg-type]
            documents.append(document)
            return super().put_object(document)

    def capture_evidence(*args: object, **kwargs: object) -> dict:
        evidence = base_evidence_builder(*args, **kwargs)
        produced["recomputation"] = evidence
        return evidence

    monkeypatch.setattr(source_fixture, "SystemStore", CapturingSystemStore)
    monkeypatch.setattr(
        source_fixture, "build_factor_production_recomputation_evidence", capture_evidence
    )
    source_fixture.test_deep_factor_replay_rebuilds_trusted_calendar_and_market_binding(
        tmp_path, monkeypatch
    )
    assert len(captured) == 1
    by_kind = {document["kind"]: document for document in documents}
    factor_generation = by_kind["factor.production_generation"]
    source_closure = by_kind["factor.production_source_closure"]
    recomputation = produced["recomputation"]
    market_input = by_kind["factor.production_market_input"]
    legacy = by_kind["factor.production_legacy_zero_call_certificate"]
    expected_release_verification = source_closure["payload"]["release_install_verification"]

    def verify_current_release(_raw: bytes, *, repository_root: str | Path) -> dict[str, object]:
        assert Path(repository_root).resolve(strict=True) == release_repository_root
        return dict(expected_release_verification)

    monkeypatch.setattr(
        source_fixture.production_authority,
        "verify_running_release_install_input",
        verify_current_release,
    )
    release_repository_root = Path(
        by_kind["system.trusted_provider_calendar_capture_execution"]["payload"][
            "release_repository_root"
        ]
    )

    custody_store = captured[0]
    store = FactorProductionStore(
        custody_store.workspace_root,
        source_custody=FactorReadOnlySystemCustody(
            custody_store.workspace_root,
            source_root=custody_store.source_root,
            source_root_id=custody_store.source_root_id,
        ),
        release_repository_root=release_repository_root,
    )
    prepared = store.prepare_initial_activation(
        factor_generation=factor_generation,
        source_closure=source_closure,
        recomputation_evidence=recomputation,
        legacy_zero_call_certificate=legacy,
        market_input=market_input,
        prepared_at=STAMP,
        activated_at=ACTIVATED_AT,
    )
    return store, prepared


def _activate(store: FactorProductionStore, prepared: dict) -> dict:
    return store.activate_initial_generation(
        target_factor_pointer_raw=prepared["target_factor_pointer_raw"],
        factor_generation_receipt_raw=prepared["factor_generation_receipt_raw"],
        activation_bundle_raw=prepared["activation_bundle_raw"],
        prepared_transaction_raw=prepared["prepared_transaction_raw"],
        permanent_marker_raw=prepared["permanent_marker_raw"],
    )


def _take_factor_active_lock_in_process(
    workspace_root: str,
    barrier: multiprocessing.synchronize.Barrier,
    queue: multiprocessing.queues.Queue,
) -> None:
    try:
        barrier.wait(timeout=10)
        with FactorProductionStore(workspace_root)._active_lock():
            queue.put("OK")
    except BaseException as exc:  # pragma: no cover - asserted by parent process
        queue.put(f"ERROR:{type(exc).__name__}:{exc}")


def test_factor_initial_activation_isolated_from_system_and_has_exact_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root
    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()
    assert not (workspace / str(FACTOR_ACTIVE_POINTER_PATH)).exists()
    assert not (workspace / str(FACTOR_PRODUCTION_MARKER_PATH)).exists()
    assert (workspace / prepared["factor_generation_path"]).read_bytes() == prepared[
        "factor_generation_raw"
    ]

    result = _activate(store, prepared)
    assert result["activation"]["cas_performed"] is True
    assert result["factor_readiness"] == "READY"
    assert result["factor_authority"] == "ACTIVE"
    active_raw = (workspace / str(FACTOR_ACTIVE_POINTER_PATH)).read_bytes()
    assert factor_authority.validate_factor_active_pointer(active_raw) == {
        "factor_generation_id": result["factor_generation_id"],
        "factor_generation_sha256": prepared["target_factor_pointer"]["payload"][
            "factor_generation_sha256"
        ],
        "previous_pointer_sha256": "EMPTY",
        "activated_at": ACTIVATED_AT,
        "os_actor": f"uid:{os.geteuid()}",
        "authority_scope": "FACTOR_PRODUCTION",
    }
    assert {row["factor_id"]: row["weight"] for row in result["active_factors"]} == {
        "pv_low_dollar_volume_5d": "0.500000000000",
        "pv_blend_volstab19x2_mom90_amihud5_w80": "0.500000000000",
    }
    assert result["control_factors"][0]["factor_id"] == "pv_blend_volstab19x2_mom90_amihud5_w75"
    assert result["control_factors"][0]["weight"] == "0.000000000000"
    assert result["control_factors"][0]["selectable"] is False
    assert result["system_authority"] == "NONE"
    assert result["mainline_authority"] == "NONE"
    assert result["investment_authority"] == "NONE"
    assert result["order_authority"] == "NONE"
    assert result["trade_authority"] == "NONE"
    assert result["funds_transfer_authority"] == "NONE"
    assert result["system_pointer_touched"] is False
    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()
    assert SystemStore(workspace).read_active() is None


def test_public_factor_production_reads_require_marker_and_use_sealed_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root

    def forbidden_system_authority(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Factor production reads must not inspect System authority")

    monkeypatch.setattr(SystemStore, "read_active", forbidden_system_authority)
    initial = factor_production_status(workspace_root=str(workspace))
    assert initial["factor_authority"] == "INACTIVE"
    assert factor_production_verify(workspace_root=str(workspace))["verified"] is False
    with pytest.raises(FactorGovernanceError, match="complete active marker closure"):
        factor_production_signal(
            workspace_root=str(workspace),
            factor_id="pv_low_dollar_volume_5d",
        )

    with store._active_lock():
        store._write_initial_pointer_under_lock(prepared["target_factor_pointer_raw"])
    pending = factor_production_status(workspace_root=str(workspace))
    assert pending["factor_authority"] == "BLOCKED"
    assert pending["blockers"] == ["FACTOR_PRODUCTION_MARKER_ABSENT"]
    assert factor_production_verify(workspace_root=str(workspace))["verified"] is False
    with pytest.raises(FactorGovernanceError, match="complete active marker closure"):
        factor_production_signal(
            workspace_root=str(workspace),
            factor_id="pv_low_dollar_volume_5d",
        )

    recovered = _activate(store, prepared)
    assert recovered["activation"]["cas_performed"] is False
    assert recovered["activation"]["marker_only_recovery"] is True
    generation_payload = prepared["factor_generation"]["payload"]
    for factor_id in (
        "pv_low_dollar_volume_5d",
        "pv_blend_volstab19x2_mom90_amihud5_w80",
    ):
        signal = factor_production_signal(workspace_root=str(workspace), factor_id=factor_id)
        assert signal["command_status"] == "VERIFIED_ACTIVE_SIGNAL"
        assert signal["authority_domain"] == "FACTOR_PRODUCTION_ONLY"
        assert signal["system_runtime_state"] == "NOT_EVALUATED"
        assert signal["grants_system_authority"] is False
        assert signal["grants_trading_authority"] is False
        assert (
            signal["factor_generation_id"] == generation_payload["factor_production_generation_id"]
        )
        assert signal["signal_values"] == generation_payload["signal_values"][factor_id]
        assert signal["symbol_count"] == len(signal["signal_values"])
        assert signal["system_authority"] == "NONE"
        assert signal["trade_authority"] == "NONE"
    with pytest.raises(FactorGovernanceError, match="only active LOW or W80"):
        store.read_active_signal("pv_blend_volstab19x2_mom90_amihud5_w75")

    main(["factor", "production-verify", "--workspace-root", str(workspace)])
    cli_verify = json.loads(capsys.readouterr().out)
    assert cli_verify["command_status"] == "VERIFIED"
    assert cli_verify["verified"] is True
    main(
        [
            "factor",
            "production-signal",
            "--workspace-root",
            str(workspace),
            "--factor-id",
            "pv_low_dollar_volume_5d",
        ]
    )
    cli_signal = json.loads(capsys.readouterr().out)
    assert (
        cli_signal["signal_values"]
        == generation_payload["signal_values"]["pv_low_dollar_volume_5d"]
    )


def test_one_shot_public_operator_composes_prepare_current_gates_cas_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_investor.factors.governance.factor_production_prepare as prepare_module
    import quant_investor.factors.production_authority as authority_module
    import quant_investor.system as system_module

    workspace = tmp_path / "workspace"
    release_root = tmp_path / "release-root"
    market_root = tmp_path / "market-root"
    capture_root = tmp_path / "capture-root"
    source_root = workspace / "results/factors/preparations/op-a/sources"
    for directory in (workspace, release_root, market_root, capture_root, source_root):
        directory.mkdir(parents=True, mode=0o700)

    def ref(kind: str, identity: str, marker: str) -> dict[str, str]:
        return {
            "kind": kind,
            "contract_sha256": get_contract(kind).contract_sha256,
            "artifact_id": identity,
            "semantic_sha256": marker * 64,
            "byte_sha256": marker * 64,
        }

    generation_ref = ref("factor.production_generation", "generation-a", "7")
    source_ref = ref("factor.production_source_closure", "source-a", "8")
    recomputation_ref = ref("factor.production_recomputation_evidence", "recompute-a", "9")
    legacy_ref = ref("factor.production_legacy_zero_call_certificate", "legacy-a", "a")
    market_ref = ref("factor.production_market_input", "market-a", "b")
    prepare_calls: list[dict[str, object]] = []
    store_calls: list[dict[str, object]] = []
    artifacts = {
        generation_ref["byte_sha256"]: {"kind": "factor.production_generation"},
        source_ref["byte_sha256"]: {
            "kind": "factor.production_source_closure",
            "payload": {"legacy_zero_call_ref": legacy_ref, "market_input_ref": market_ref},
        },
        recomputation_ref["byte_sha256"]: {"kind": "factor.production_recomputation_evidence"},
        legacy_ref["byte_sha256"]: {"kind": "factor.production_legacy_zero_call_certificate"},
        market_ref["byte_sha256"]: {"kind": "factor.production_market_input"},
    }

    def prepare(**kwargs: object) -> dict[str, object]:
        prepare_calls.append(dict(kwargs))
        assert "legacy_scan_runner" not in kwargs and "process_runner" not in kwargs
        return {
            "operation_id": "op-a",
            "operation_inputs_sha256": "e" * 64,
            "operation_inputs_ref": {
                "relative_path": "operation-inputs.json",
                "byte_sha256": "e" * 64,
            },
            "source_root": "results/factors/preparations/op-a/sources",
            "source_root_id": "f" * 64,
            "release_repository_root": str(release_root),
            "factor_production_generation_ref": generation_ref,
            "factor_production_source_closure_ref": source_ref,
            "factor_production_recomputation_ref": recomputation_ref,
        }

    class SourceStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_object(self, reference: dict[str, str]) -> dict[str, object]:
            return artifacts[reference["byte_sha256"]]

    class FactorStore:
        @classmethod
        def from_system_source_custody(cls, *args: object, **kwargs: object) -> "FactorStore":
            store_calls.append({"args": args, **kwargs})
            return cls()

        def prepare_initial_activation(self, **kwargs: object) -> dict[str, bytes]:
            store_calls.append({"prepare": kwargs})
            return {
                "target_factor_pointer_raw": b"pointer",
                "factor_generation_receipt_raw": b"receipt",
                "activation_bundle_raw": b"bundle",
                "prepared_transaction_raw": b"prepared",
                "permanent_marker_raw": b"marker",
            }

        def activate_initial_generation(self, **kwargs: object) -> dict[str, object]:
            store_calls.append({"activate": kwargs})
            return {
                "factor_authority": "ACTIVE",
                "factor_readiness": "READY",
                "factor_generation_id": "factor-generation-a",
                "factor_generation_sha256": "1" * 64,
                "factor_pointer_byte_sha256": "2" * 64,
                "factor_pointer_semantic_sha256": "3" * 64,
                "marker_byte_sha256": "4" * 64,
                "marker_semantic_sha256": "5" * 64,
                "active_factors": [],
                "control_factors": [],
                "as_of": "20260817",
                "activation": {
                    "cas_performed": True,
                    "marker_only_recovery": False,
                },
            }

    monkeypatch.setattr(prepare_module, "prepare_factor_production", prepare)
    monkeypatch.setattr(
        authority_module,
        "verify_factor_production",
        lambda _root: {
            "factor_authority": "INACTIVE",
            "blockers": ["FACTOR_ACTIVE_POINTER_ABSENT"],
        },
    )
    monkeypatch.setattr(authority_module, "FactorProductionStore", FactorStore)
    monkeypatch.setattr(system_module, "SystemStore", SourceStore)
    result = factor_production_activate(
        workspace_root=str(workspace),
        market_data_root=str(market_root),
        calendar_capture_root=str(capture_root),
        expected_calendar_success_sha256="6" * 64,
        expected_empty=True,
    )
    assert result["command_status"] == "ACTIVATED"
    assert result["authority_domain"] == "FACTOR_PRODUCTION_ONLY"
    assert result["cas_performed"] is True
    assert result["operation_inputs_ref"]["byte_sha256"] == "e" * 64
    assert result["grants_system_authority"] is False
    assert result["grants_trading_authority"] is False
    assert len(prepare_calls) == 1
    assert any("prepare" in call for call in store_calls)
    assert any("activate" in call for call in store_calls)


def test_cli_one_shot_operator_runs_real_temp_source_prepare_and_single_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = source_fixture._factor_production_operator_fixture(tmp_path, monkeypatch)
    workspace = Path(fixture["workspace"])
    pointer = workspace / str(FACTOR_ACTIVE_POINTER_PATH)
    marker = workspace / str(FACTOR_PRODUCTION_MARKER_PATH)
    system_pointer = workspace / "results/system/_active.json"
    system_marker = workspace / "results/system/_migration_complete.json"
    assert not pointer.exists() and not marker.exists()
    assert not system_pointer.exists() and not system_marker.exists()

    protected: dict[Path, bytes] = {}
    for relative in (
        "results/broker/r4-sentinel.json",
        "results/orders/r4-sentinel.json",
        "results/trades/r4-sentinel.json",
        "results/funds/r4-sentinel.json",
        "results/portfolio/r4-sentinel.json",
        "results/strategy_records/r4-sentinel.json",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.write_bytes(canonical_json_bytes({"authority": "UNCHANGED"}))
        path.chmod(0o600)
        protected[path] = path.read_bytes()

    command = [
        "factor",
        "production-activate",
        "--workspace-root",
        str(workspace),
        "--market-data-root",
        str(fixture["market_root"]),
        "--calendar-capture-root",
        str(fixture["capture_root"]),
        "--expected-calendar-success-sha256",
        str(fixture["calendar_success_sha256"]),
        "--expected-empty",
    ]
    main(command)
    first = json.loads(capsys.readouterr().out)
    assert first["command_status"] == "ACTIVATED"
    assert first["authority_domain"] == "FACTOR_PRODUCTION_ONLY"
    assert first["cas_performed"] is True
    assert first["factor_readiness"] == "READY"
    assert first["factor_authority"] == "ACTIVE"
    assert {row["factor_id"]: row["weight"] for row in first["active_factors"]} == {
        "pv_low_dollar_volume_5d": "0.500000000000",
        "pv_blend_volstab19x2_mom90_amihud5_w80": "0.500000000000",
    }
    assert len(first["control_factors"]) == 1
    control = first["control_factors"][0]
    assert control["factor_id"] == "pv_blend_volstab19x2_mom90_amihud5_w75"
    assert control["role"] == "CONTROL_ONLY"
    assert control["weight"] == "0.000000000000"
    assert control["selectable"] is False
    assert first["grants_system_authority"] is False
    assert first["grants_trading_authority"] is False
    assert pointer.exists() and marker.exists()
    assert pointer.stat().st_nlink == 1 and marker.stat().st_nlink == 1
    assert not system_pointer.exists() and not system_marker.exists()
    assert all(path.read_bytes() == raw for path, raw in protected.items())
    pointer_before = pointer.read_bytes()
    marker_before = marker.read_bytes()

    with pytest.raises(SystemExit) as replay_exit:
        main(command)
    assert replay_exit.value.code == 2
    replay = json.loads(capsys.readouterr().out)
    assert replay["status"] == "BLOCKED"
    assert replay["blocker_code"] == "FACTOR_EXPECTED_EMPTY_FAILED"
    assert pointer.read_bytes() == pointer_before
    assert marker.read_bytes() == marker_before
    assert not system_pointer.exists() and not system_marker.exists()
    assert all(path.read_bytes() == raw for path, raw in protected.items())


def test_cli_recovers_exact_marker_after_before_marker_rename_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import quant_investor.factors.governance.factor_production_prepare as prepare_module

    fixture = source_fixture._factor_production_operator_fixture(tmp_path, monkeypatch)
    workspace = Path(fixture["workspace"])
    command = [
        "factor",
        "production-activate",
        "--workspace-root",
        str(workspace),
        "--market-data-root",
        str(fixture["market_root"]),
        "--calendar-capture-root",
        str(fixture["capture_root"]),
        "--expected-calendar-success-sha256",
        str(fixture["calendar_success_sha256"]),
        "--expected-empty",
    ]
    pointer = workspace / str(FACTOR_ACTIVE_POINTER_PATH)
    marker = workspace / str(FACTOR_PRODUCTION_MARKER_PATH)
    system_pointer = workspace / "results/system/_active.json"
    system_marker = workspace / "results/system/_migration_complete.json"
    for relative in (
        "results/broker/r5-sentinel.json",
        "results/orders/r5-sentinel.json",
        "results/trades/r5-sentinel.json",
        "results/funds/r5-sentinel.json",
        "results/portfolio/r5-sentinel.json",
        "results/strategy_records/r5-sentinel.json",
    ):
        sentinel = workspace / relative
        sentinel.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        sentinel.write_bytes(canonical_json_bytes({"authority": "UNCHANGED"}))
        sentinel.chmod(0o600)
    protected = {
        path: path.read_bytes()
        for path in workspace.glob("results/**/r5-sentinel.json")
        if path.is_file()
    }
    assert len(protected) == 6

    def fail_before_marker(point: str) -> None:
        if point == "BEFORE_MARKER_RENAME":
            raise FactorGovernanceError("injected R5 before marker rename")

    monkeypatch.setattr(
        factor_authority._FactorSecureStorage,
        "_test_fault_hook",
        fail_before_marker,
    )
    with pytest.raises(SystemExit) as first_exit:
        main(command)
    assert first_exit.value.code == 2
    capsys.readouterr()
    assert pointer.exists() and not marker.exists()
    pointer_before = pointer.read_bytes()
    assert pointer.stat().st_nlink == 1
    assert not system_pointer.exists() and not system_marker.exists()
    assert all(path.read_bytes() == raw for path, raw in protected.items())

    monkeypatch.setattr(factor_authority._FactorSecureStorage, "_test_fault_hook", None)

    def forbidden_prepare(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("pointer-only recovery must not run source preparation")

    monkeypatch.setattr(prepare_module, "prepare_factor_production", forbidden_prepare)
    main(command)
    recovered = json.loads(capsys.readouterr().out)
    assert recovered["command_status"] == "MARKER_RECOVERED"
    assert recovered["cas_performed"] is False
    assert recovered["marker_only_recovery"] is True
    assert recovered["factor_readiness"] == "READY"
    assert recovered["factor_authority"] == "ACTIVE"
    assert pointer.read_bytes() == pointer_before
    assert marker.exists() and marker.stat().st_nlink == 1
    marker_before = marker.read_bytes()
    assert not system_pointer.exists() and not system_marker.exists()
    assert all(path.read_bytes() == raw for path, raw in protected.items())

    with pytest.raises(SystemExit) as completed_exit:
        main(command)
    assert completed_exit.value.code == 2
    completed = json.loads(capsys.readouterr().out)
    assert completed["blocker_code"] == "FACTOR_EXPECTED_EMPTY_FAILED"
    assert pointer.read_bytes() == pointer_before
    assert marker.read_bytes() == marker_before


def test_factor_initial_activation_replay_is_idempotent_and_never_second_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    _activate(store, prepared)
    replay = _activate(store, prepared)
    assert replay["activation"]["cas_performed"] is False
    assert replay["activation"]["idempotent_replay"] is True
    assert replay["activation"]["marker_only_recovery"] is False


def test_concurrent_factor_initial_activation_has_one_empty_cas_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    assert store._source_custody is not None
    assert store._release_repository_root is not None
    peer = FactorProductionStore(
        store.workspace_root,
        source_custody=store._source_custody,
        release_repository_root=store._release_repository_root,
    )
    gate = threading.Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []

    def activate(store_instance: FactorProductionStore) -> None:
        try:
            gate.wait(timeout=10)
            results.append(_activate(store_instance, prepared))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=activate, args=(store,))
    second = threading.Thread(target=activate, args=(peer,))
    first.start()
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert sorted(result["activation"]["cas_performed"] for result in results) == [False, True]


def test_noncooperative_pointer_appearance_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    store = FactorProductionStore(workspace)
    target = canonical_json_bytes({"target": True})
    rival = canonical_json_bytes({"rival": True})
    original_rename = factor_authority._atomic_no_replace_rename

    def publish_rival_then_rename(
        source: str,
        destination: str,
        *,
        source_directory_fd: int,
        destination_directory_fd: int,
    ) -> None:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_directory_fd,
        )
        try:
            os.write(descriptor, rival)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original_rename(
            source,
            destination,
            source_directory_fd=source_directory_fd,
            destination_directory_fd=destination_directory_fd,
        )

    monkeypatch.setattr(factor_authority, "_atomic_no_replace_rename", publish_rival_then_rename)
    with store._active_lock():
        with pytest.raises(FactorGovernanceError, match="preimage is no longer EMPTY"):
            store._write_initial_pointer_under_lock(target)
    pointer = workspace / str(FACTOR_ACTIVE_POINTER_PATH)
    assert pointer.read_bytes() == rival
    assert pointer.stat().st_nlink == 1
    assert not list(pointer.parent.glob("._active.json.publish-*"))


@pytest.mark.parametrize(
    ("fault_point", "pointer_exists"),
    (("BEFORE_POINTER_RENAME", False), ("AFTER_POINTER_RENAME", True)),
)
def test_pointer_atomic_rename_fault_boundaries_never_expose_partial_bytes(
    tmp_path: Path, fault_point: str, pointer_exists: bool
) -> None:
    workspace = tmp_path / fault_point.lower()
    workspace.mkdir(mode=0o700)
    store = FactorProductionStore(workspace)
    target = canonical_json_bytes({"complete": True})

    def inject(point: str) -> None:
        if point == fault_point:
            raise FactorGovernanceError(f"injected {fault_point}")

    store._storage._fault_hook = inject
    with store._active_lock():
        with pytest.raises(FactorGovernanceError, match=fault_point):
            store._write_initial_pointer_under_lock(target)
    pointer = workspace / str(FACTOR_ACTIVE_POINTER_PATH)
    assert pointer.exists() is pointer_exists
    if pointer_exists:
        assert pointer.read_bytes() == target
        assert pointer.stat().st_nlink == 1
    assert not list(pointer.parent.glob("._active.json.publish-*"))


def test_marker_atomic_no_replace_supports_exact_replay_and_rejects_partial_conflict(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "marker"
    workspace.mkdir(mode=0o700)
    store = FactorProductionStore(workspace)
    marker_raw = canonical_json_bytes({"complete": True})
    with store._active_lock():
        first = store._write_permanent_marker_under_lock(marker_raw)
        replay = store._write_permanent_marker_under_lock(marker_raw)
    marker = workspace / str(FACTOR_PRODUCTION_MARKER_PATH)
    assert first.data == marker_raw == replay.data
    assert first.byte_sha256 == replay.byte_sha256
    assert marker.stat().st_nlink == 1

    conflicting_workspace = tmp_path / "conflicting-marker"
    conflicting_workspace.mkdir(mode=0o700)
    conflicting_store = FactorProductionStore(conflicting_workspace)
    with conflicting_store._active_lock():
        conflict = conflicting_workspace / str(FACTOR_PRODUCTION_MARKER_PATH)
        conflict.write_bytes(b"{")
        conflict.chmod(0o600)
        with pytest.raises(FactorGovernanceError, match="conflicts"):
            conflicting_store._write_permanent_marker_under_lock(marker_raw)
    assert conflict.read_bytes() == b"{"
    assert conflict.stat().st_nlink == 1


def test_atomic_no_replace_rename_has_no_unsupported_platform_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "unsupported-platform"
    workspace.mkdir(mode=0o700)
    store = FactorProductionStore(workspace)
    monkeypatch.setattr(factor_authority.sys, "platform", "unsupported-test-platform")
    with store._active_lock():
        with pytest.raises(FactorGovernanceError, match="unavailable on this platform"):
            store._write_initial_pointer_under_lock(canonical_json_bytes({"complete": True}))
    pointer = workspace / str(FACTOR_ACTIVE_POINTER_PATH)
    assert not pointer.exists()
    assert not list(pointer.parent.glob("._active.json.publish-*"))


def test_factor_active_lock_creation_is_reliable_under_parallel_first_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    lock_path = workspace / "results/factors/.active.lock"
    for _round in range(20):
        if lock_path.exists():
            lock_path.unlink()
        gate = threading.Barrier(8)
        failures: list[BaseException] = []
        acquired: list[int] = []

        def acquire(index: int) -> None:
            try:
                gate.wait(timeout=10)
                with FactorProductionStore(workspace)._active_lock():
                    acquired.append(index)
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        threads = [threading.Thread(target=acquire, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert not any(thread.is_alive() for thread in threads)
        assert failures == []
        assert sorted(acquired) == list(range(8))


def test_factor_active_lock_creation_is_reliable_across_processes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    lock_path = workspace / "results/factors/.active.lock"
    context = multiprocessing.get_context("spawn")
    for _round in range(5):
        if lock_path.exists():
            lock_path.unlink()
        barrier = context.Barrier(2)
        queue = context.Queue()
        processes = [
            context.Process(
                target=_take_factor_active_lock_in_process,
                args=(str(workspace), barrier, queue),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert not any(process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
        assert sorted(queue.get(timeout=5) for _ in processes) == ["OK", "OK"]


def test_marker_only_recovery_reuses_exact_prepared_marker_without_second_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root

    def fail_before_marker(point: str) -> None:
        if point == "BEFORE_MARKER_RENAME":
            raise FactorGovernanceError("injected marker publication failure")

    store._storage._fault_hook = fail_before_marker
    with pytest.raises(FactorGovernanceError, match="injected marker"):
        _activate(store, prepared)
    assert (workspace / str(FACTOR_ACTIVE_POINTER_PATH)).exists()
    assert not (workspace / str(FACTOR_PRODUCTION_MARKER_PATH)).exists()
    assert not list(
        (workspace / str(FACTOR_PRODUCTION_MARKER_PATH)).parent.glob(
            "._production_complete.json.publish-*"
        )
    )
    store._storage._fault_hook = lambda _point: None
    # Simulate post-CAS source-custody / checkout drift.  Exact pointer-only
    # recovery must use the sealed Factor mirrors, publish only the exact
    # marker, and never attempt a new current-source CAS gate.
    store._source_custody = None
    store._release_repository_root = None
    recovered = _activate(store, prepared)
    assert recovered["activation"]["cas_performed"] is False
    assert recovered["activation"]["marker_only_recovery"] is True
    assert recovered["factor_authority"] == "ACTIVE"


def test_reserved_factor_paths_reject_generic_writer_and_tampered_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root
    for path in (FACTOR_ACTIVE_POINTER_PATH, FACTOR_PRODUCTION_MARKER_PATH):
        with pytest.raises(FactorGovernanceError, match="reserved Factor authority"):
            store.write_exact_once(path, b"{}")
    tampered = dict(prepared)
    tampered["permanent_marker_raw"] = prepared["permanent_marker_raw"] + b" "
    with pytest.raises(FactorGovernanceError):
        _activate(store, tampered)
    assert not (workspace / str(FACTOR_ACTIVE_POINTER_PATH)).exists()
    assert not (workspace / str(FACTOR_PRODUCTION_MARKER_PATH)).exists()


@pytest.mark.parametrize(
    ("authority_path", "authority"),
    (
        ("results/factors/_active.json", "Factor"),
        ("results/factors/_production_complete.json", "Factor"),
        ("results/system/_active.json", "System"),
        ("results/system/_migration_complete.json", "System"),
    ),
)
def test_generic_cutover_publisher_cannot_write_factor_authority_paths(
    tmp_path: Path, authority_path: str, authority: str
) -> None:
    with pytest.raises(UnifiedCutoverError, match=f"reserved {authority} authority"):
        write_idempotent_bytes(tmp_path / authority_path, b"{}")


def test_generic_cutover_publisher_rejects_factor_path_aliases_before_write(tmp_path: Path) -> None:
    target = tmp_path / "results/factors"
    target.mkdir(mode=0o700, parents=True)
    lexical_alias = tmp_path / "results/factors/../factors/_active.json"
    with pytest.raises(UnifiedCutoverError, match="lexically canonical"):
        write_idempotent_bytes(lexical_alias, b"{}")
    symlink_alias = tmp_path / "factor-authority-alias"
    os.symlink(target, symlink_alias)
    with pytest.raises(UnifiedCutoverError, match="reserved Factor authority"):
        write_idempotent_bytes(symlink_alias / "_active.json", b"{}")
    assert not (target / "_active.json").exists()


@pytest.mark.parametrize(
    "descendant",
    (
        "results/factors/_active.json/child",
        "results/factors/_production_complete.json/child",
        "results/system/_active.json/child",
        "results/system/_migration_complete.json/child",
    ),
)
def test_generic_cutover_publisher_rejects_reserved_authority_descendants(
    tmp_path: Path, descendant: str
) -> None:
    with pytest.raises(UnifiedCutoverError, match="reserved (Factor|System) authority"):
        write_idempotent_bytes(tmp_path / descendant, b"{}")
    assert not (tmp_path / "results/factors/_active.json").exists()
    assert not (tmp_path / "results/factors/_production_complete.json").exists()
    assert not (tmp_path / "results/system/_active.json").exists()
    assert not (tmp_path / "results/system/_migration_complete.json").exists()


def test_factor_storage_refuses_symlink_casefold_hardlink_and_traversal_attacks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    target = tmp_path / "outside"
    target.mkdir(mode=0o700)
    results = workspace / "results"
    results.mkdir(mode=0o700)
    os.symlink(target, results / "factors")
    symlinked = FactorProductionStore(workspace)
    with pytest.raises(FactorGovernanceError, match="symlink"):
        symlinked.write_exact_once("results/factors/objects/unsafe.json", b"{}")

    (results / "factors").unlink()
    (results / "Factors").mkdir(mode=0o700)
    casefolded = FactorProductionStore(workspace)
    with pytest.raises(FactorGovernanceError, match="casefold"):
        casefolded.write_exact_once("results/factors/objects/unsafe.json", b"{}")

    (results / "Factors").rmdir()
    store = FactorProductionStore(workspace)
    stored = store.write_exact_once("results/factors/objects/test.json", b"{}")
    path = workspace / stored.relative_path
    os.link(path, path.with_name("test-link.json"))
    with pytest.raises(FactorGovernanceError, match="hard link"):
        store.read("results/factors/objects/test.json")
    with pytest.raises(FactorGovernanceError, match="outside"):
        store.write_exact_once("results/factors/../system/unsafe.json", b"{}")
    with pytest.raises(FactorGovernanceError, match="outside"):
        store.write_exact_once("results/system/_active.json", b"{}")
    for descendant in (
        "results/factors/_active.json/child",
        "results/factors/_production_complete.json/child",
    ):
        with pytest.raises(FactorGovernanceError, match="reserved Factor authority"):
            store.write_exact_once(descendant, b"{}")


def test_factor_prepare_does_not_read_or_write_system_active_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_system_active(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Factor source custody must not inspect System active authority")

    monkeypatch.setattr(SystemStore, "read_active", forbidden_system_active)
    # The fixture performs real source-custody preparation.  The monkeypatch
    # would immediately fail if Factor preparation treated System activation as
    # an input rather than its immutable source objects.
    store, _prepared_bundle = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root
    assert store._source_custody is not None
    assert not hasattr(store._source_custody, "_store")
    custody_methods = {
        name
        for name in dir(store._source_custody)
        if callable(getattr(store._source_custody, name)) and not name.startswith("_")
    }
    assert custody_methods == {"artifact_resolver", "source_resolver"}
    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()
    for forbidden in (
        "results/broker",
        "results/orders",
        "results/trades",
        "results/funds",
        "results/portfolio",
        "results/strategy_records",
    ):
        assert not (workspace / forbidden).exists()


def _protected_state_fingerprint(root: Path, relative_path: str) -> str:
    target = root / relative_path
    if not target.exists() and not target.is_symlink():
        return "ABSENT"
    if target.is_symlink():
        return "SYMLINK"
    if target.is_file():
        return "FILE:" + hashlib.sha256(target.read_bytes()).hexdigest()
    entries = []
    for path in sorted(target.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            entries.append((path.relative_to(target).as_posix(), "SYMLINK"))
        elif path.is_file():
            entries.append(
                (
                    path.relative_to(target).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return "DIRECTORY:" + hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def test_factor_activation_preserves_system_and_trading_protected_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, prepared = _native_store_and_prepared(tmp_path, monkeypatch)
    workspace = store.workspace_root
    protected_paths = (
        "results/system/_active.json",
        "results/system/_migration_complete.json",
        "results/broker",
        "results/orders",
        "results/trades",
        "results/funds",
        "results/portfolio",
        "results/strategy_records",
    )
    before = {path: _protected_state_fingerprint(workspace, path) for path in protected_paths}
    _activate(store, prepared)
    after = {path: _protected_state_fingerprint(workspace, path) for path in protected_paths}
    assert before == after
    assert set(before.values()) == {"ABSENT"}
