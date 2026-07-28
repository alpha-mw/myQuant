from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pytest

import quant_investor.factors.production_control_v1 as production_control
from quant_investor.factors.governance_canonical_replay_v4 import (
    readback_v4_evidence as canonical_readback_v4_evidence,
)
from quant_investor.factors.governance_protocol_v4 import (
    PRODUCTION_APPLY_ENABLED,
    protocol_hash,
)
from quant_investor.factors.governance_transaction_v4 import (
    build_activation_request_v4,
    build_factor_v4_transaction_plan,
)
from quant_investor.factors.production_control_v1 import (
    AUTHORIZATION_SCOPE,
    EMPTY_SHA256,
    ProductionControlCrash,
    ProductionControlError,
    ProductionControlStore,
    build_artifact_ref,
    build_authorization_receipt,
    build_pre_activation_eligibility,
    build_production_control_transaction,
    build_production_registry,
    build_rollback_authorization_receipt,
    build_runtime_contract_set,
    build_v4_evidence_set,
    build_v4_replay_set,
    canonical_file_bytes,
    validate_active_set_pointer,
    validate_control_receipt,
    validate_pre_activation_eligibility,
    validate_production_control_transaction,
    validate_production_registry,
    validate_rollback_authorization_receipt,
    validate_rollback_receipt,
)
from quant_investor.factors.runtime import production_factor_set_sha256

AS_OF = "2026-07-27"
ACTIVATED_AT = "2026-07-27T09:00:00Z"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _semantic(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sealed(schema_version: str, **values: Any) -> dict[str, Any]:
    result = {"schema_version": schema_version, **values}
    result["semantic_sha256"] = _semantic(result)
    return result


@pytest.fixture(autouse=True)
def _stub_verified_v4_replay_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def readback(evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "complete_chain_hash_binding_verified": True,
            "context_bindings_readback_verified": True,
            "evidence": deepcopy(evidence),
            "local_bytes_readback_verified": True,
            "quantitative_evidence_hash_binding_verified": True,
            "replay": {
                "replay_semantic_sha256": evidence[
                    "replay_semantic_sha256"
                ],
            },
            "replay_file_sha256": evidence["replay_file_sha256"],
        }

    monkeypatch.setattr(
        production_control,
        "readback_v4_evidence",
        readback,
    )


def _calendar() -> dict[str, Any]:
    cursor = date(2024, 1, 1)
    end = date(2026, 7, 28)
    sessions: list[str] = []
    while cursor < end:
        if cursor.weekday() < 5:
            sessions.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return {
        "schema_version": "factor-governance-open-session-calendar.v4",
        "market": "CN",
        "source": "strict_parquet_observed_trade_dates",
        "latest_pointer_sha256": _digest("pointer"),
        "manifest_sha256": _digest("manifest"),
        "open_session_dates": sessions,
    }


def _month_ends(calendar: dict[str, Any]) -> list[str]:
    by_month: dict[str, str] = {}
    for session in calendar["open_session_dates"]:
        by_month[session[:7]] = session
    return list(by_month.values())[-12:]


def _record(index: int, *, count: int = 5) -> dict[str, Any]:
    name = f"factor_{index}"
    family = f"family_{index}" if count == 5 else f"family_{index // 2}"
    calendar = _calendar()
    runtime_contract = {
        "schema_version": "factor-production-runtime-contract.v4",
        "factor_name": name,
    }
    return {
        "name": name,
        "family": family,
        "slot": f"{family}::slot_{index}",
        "state": "production_factor",
        "weight": 1.0,
        "calendar_sha256": _semantic(calendar),
        "gate_results": {str(gate_id): True for gate_id in range(1, 9)},
        "maturity": {
            "calendar": calendar,
            "month_end_rankic_dates": _month_ends(calendar),
            "forward_cohorts": [],
        },
        "bh_q_value": 0.05,
        "fdr_method": "benjamini_hochberg_by_family",
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": _semantic(runtime_contract),
        "runtime_contract_status": "verified",
        "evidence": {
            "factor_name": name,
            "replay_file_sha256": _digest(f"replay-file:{name}"),
            "replay_path": f"/private/tmp/{name}-replay.json",
            "schema_version": "factor-governance-replay-evidence.v4",
            "status": "verified",
            "replay_semantic_sha256": _digest(f"replay:{name}"),
            "runtime_contract_sha256": _semantic(runtime_contract),
        },
        "health": {
            "status": "healthy",
            "fresh": True,
            "data_blocked": False,
            "source_as_of": AS_OF,
        },
    }


def _proposal() -> dict[str, Any]:
    return {
        "action": "add_proposal",
        "challenger": "factor_4",
        "apply": False,
    }


def _source_artifacts(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    runtime_contracts = build_runtime_contract_set(records)
    evidence = build_v4_evidence_set(records)
    replay = build_v4_replay_set(records)
    names = [record["name"] for record in records]
    factor_set_sha = production_factor_set_sha256(names)
    plan = build_factor_v4_transaction_plan(
        transaction_id="source-plan",
        as_of=AS_OF,
        cadence="month_end",
        production_factor_count=5,
        expected_registry_file_sha256=_digest("source-before"),
        proposed_registry_file_sha256=_digest("source-after"),
        expected_production_factor_set_sha256=_digest("set-before"),
        proposed_production_factor_set_sha256=factor_set_sha,
        proposals=[_proposal()],
        wal_path="research/factor_v4/wal.jsonl",
        inverse_rollback_path="research/factor_v4/inverse.json",
    )
    request = build_activation_request_v4(
        request_id="source-request",
        as_of=AS_OF,
        transaction_plan_sha256=plan["transaction_plan_sha256"],
        proposed_registry_file_sha256=_digest("source-after"),
        proposed_production_factor_set_sha256=factor_set_sha,
        runtime_contracts_sha256=runtime_contracts[
            "runtime_contracts_sha256"
        ],
    )
    return {
        "runtime_contracts": runtime_contracts,
        "v4_activation_request": request,
        "v4_evidence": evidence,
        "v4_replay": replay,
        "v4_transaction_plan": plan,
    }


def _source_refs(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        role: build_artifact_ref(
            artifact,
            relative_path=f"reports/factor_governance/{role}.json",
        )
        for role, artifact in _source_artifacts(records).items()
    }


def _artifacts(
    *,
    count: int = 5,
    expected_registry_sha256: str = EMPTY_SHA256,
    expected_active_set_sha256: str = EMPTY_SHA256,
) -> tuple[dict[str, Any], ...]:
    records = [_record(index, count=count) for index in range(count)]
    source_artifacts = _source_artifacts(records)
    registry = build_production_registry(
        as_of=AS_OF,
        factor_records=records,
        source_refs={
            role: build_artifact_ref(
                artifact,
                relative_path=(
                    f"reports/factor_governance/{role}.json"
                ),
            )
            for role, artifact in source_artifacts.items()
        },
    )
    registry_ref = build_artifact_ref(
        registry,
        relative_path=(
            "data/private/factor_governance_production_control_v1/"
            "registry/current.json"
        ),
    )
    eligibility = build_pre_activation_eligibility(
        registry=registry,
        proposed_registry_ref=registry_ref,
    )
    authorization = build_authorization_receipt(
        receipt_id="authorization-1",
        authorized_by="codex_delegated_reviewer",
        issued_at="2026-07-27T08:00:00Z",
        expires_at="2026-07-27T10:00:00Z",
        transaction_plan_sha256=registry["source_refs"][
            "v4_transaction_plan"
        ]["semantic_sha256"],
        proposed_registry_sha256=registry_ref["byte_sha256"],
        production_factor_set_sha256=registry[
            "production_factor_set_sha256"
        ],
        runtime_contracts_sha256=registry["runtime_contracts_sha256"],
    )
    transaction = build_production_control_transaction(
        transaction_id="production-control-1",
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
        expected_registry_sha256=expected_registry_sha256,
        expected_active_set_sha256=expected_active_set_sha256,
        activated_at=ACTIVATED_AT,
        v4_activation_receipt_id="v4-activation-1",
        control_receipt_id="control-activation-1",
    )
    return (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    )


def test_five_factor_underfilled_set_is_eligible_without_relabelling() -> None:
    registry, eligibility, authorization, transaction, _ = _artifacts()

    assert PRODUCTION_APPLY_ENABLED is False
    assert authorization["authorization_scope"] == AUTHORIZATION_SCOPE
    assert authorization["activation_performed"] is False
    assert eligibility["eligible"] is True
    assert eligibility["production_factor_count"] == 5
    assert eligibility["pre_activation_healthy_factor_count"] == 5
    assert eligibility["blockers"] == []
    assert eligibility["readiness_without_activation"]["status"] == (
        "no_new_risk"
    )
    validate_production_registry(registry)
    validate_pre_activation_eligibility(eligibility, registry=registry)
    validate_production_control_transaction(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
    )
    assert transaction["readiness_after_activation"]["readiness"][
        "status"
    ] == "underfilled_accelerated_mining"
    assert not any(
        transaction["authority"][field]
        for field in ("account_new_risk", "broker", "execution", "order", "trade")
    )


def test_four_factors_and_legacy_relabel_are_blocked() -> None:
    records = [_record(index, count=4) for index in range(4)]
    registry = build_production_registry(
        as_of=AS_OF,
        factor_records=records,
        source_refs=_source_refs(records),
    )
    eligibility = build_pre_activation_eligibility(
        registry=registry,
        proposed_registry_ref=build_artifact_ref(
            registry,
            relative_path=(
                "data/private/factor_governance_production_control_v1/"
                "registry/current.json"
            ),
        ),
    )
    assert eligibility["eligible"] is False
    assert "production_factor_count_below_5" in eligibility["blockers"]

    legacy = deepcopy(_record(0))
    for field in (
        "family",
        "slot",
        "gate_results",
        "maturity",
        "bh_q_value",
        "fdr_method",
        "runtime_contract",
        "runtime_contract_sha256",
        "runtime_contract_status",
        "evidence",
        "health",
    ):
        legacy.pop(field)
    with pytest.raises(ProductionControlError):
        build_production_registry(
            as_of=AS_OF,
            factor_records=[legacy] * 5,
            source_refs=_source_refs([_record(index) for index in range(5)]),
        )


def test_stale_health_source_is_not_eligible() -> None:
    records = [_record(index) for index in range(5)]
    for record in records:
        record["health"]["source_as_of"] = "2026-07-17"
    registry = build_production_registry(
        as_of=AS_OF,
        factor_records=records,
        source_refs=_source_refs(records),
    )
    eligibility = build_pre_activation_eligibility(
        registry=registry,
        proposed_registry_ref=build_artifact_ref(
            registry,
            relative_path=(
                "data/private/factor_governance_production_control_v1/"
                "registry/current.json"
            ),
        ),
    )

    assert eligibility["eligible"] is False
    assert eligibility["source_as_of"] == "2026-07-17"
    assert (
        "health_source_open_session_lag_above_3:2026-07-17"
        in eligibility["blockers"]
    )
    assert (
        "health_source_calendar_day_lag_above_8:2026-07-17"
        in eligibility["blockers"]
    )


def test_coherent_caller_declared_replay_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production_control,
        "readback_v4_evidence",
        canonical_readback_v4_evidence,
    )
    forged_records = [_record(index) for index in range(5)]
    for record in forged_records:
        record["evidence"]["replay_semantic_sha256"] = _digest(
            f"forged:{record['name']}"
        )

    with pytest.raises(ProductionControlError):
        _source_artifacts(forged_records)


def test_apply_wal_cas_exact_readback_and_inverse_rollback(
    tmp_path: Path,
) -> None:
    (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    ) = _artifacts()
    store = ProductionControlStore((tmp_path / "control").resolve())

    receipt = store.apply(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
        source_artifacts=source_artifacts,
    )

    validate_control_receipt(receipt, transaction=transaction)
    assert receipt["active_for_production_research"] is True
    assert hashlib.sha256(store.registry_path.read_bytes()).hexdigest() == (
        transaction["proposed_registry_sha256"]
    )
    assert hashlib.sha256(store.active_set_path.read_bytes()).hexdigest() == (
        transaction["proposed_active_set_sha256"]
    )
    validate_active_set_pointer(
        json.loads(store.active_set_path.read_text())
    )
    wal = (
        store.root / f"wal/{transaction['transaction_id']}.jsonl"
    ).read_text()
    for state in (
        "PREPARED",
        "REGISTRY_COMMITTED",
        "V4_RECEIPT_ISSUED",
        "READINESS_RECOMPUTED",
        "ACTIVE_SET_COMMITTED",
    ):
        assert f'"state":"{state}"' in wal
    for directory in (
        store.root,
        store.root / "registry",
        store.root / "active_sets",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in (
        store.registry_path,
        store.active_set_path,
        store.registry_lock_path,
        store.active_set_lock_path,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    expired_rollback_authorization = build_rollback_authorization_receipt(
        receipt_id="rollback-authorization-expired",
        authorized_by="codex_delegated_reviewer",
        issued_at="2026-07-27T08:00:00Z",
        expires_at="2026-07-27T08:30:00Z",
        transaction=transaction,
    )
    with pytest.raises(ProductionControlError):
        store.rollback(
            transaction,
            receipt_id="rollback-expired",
            authorization_receipt=expired_rollback_authorization,
            recorded_at="2026-07-27T09:30:00Z",
        )
    assert hashlib.sha256(store.registry_path.read_bytes()).hexdigest() == (
        transaction["proposed_registry_sha256"]
    )
    assert hashlib.sha256(store.active_set_path.read_bytes()).hexdigest() == (
        transaction["proposed_active_set_sha256"]
    )

    rollback_authorization = build_rollback_authorization_receipt(
        receipt_id="rollback-authorization-1",
        authorized_by="codex_delegated_reviewer",
        issued_at="2026-07-27T09:15:00Z",
        expires_at="2026-07-27T10:00:00Z",
        transaction=transaction,
    )
    validate_rollback_authorization_receipt(
        rollback_authorization,
        transaction=transaction,
        observed_at="2026-07-27T09:30:00Z",
    )
    unbound_authorization = deepcopy(rollback_authorization)
    unbound_authorization["control_receipt_ref"]["byte_sha256"] = _digest(
        "different-control-receipt"
    )
    unbound_authorization["semantic_sha256"] = _semantic(
        unbound_authorization
    )
    with pytest.raises(ProductionControlError):
        store.rollback(
            transaction,
            receipt_id="rollback-unbound",
            authorization_receipt=unbound_authorization,
            recorded_at="2026-07-27T09:30:00Z",
        )

    rollback = store.rollback(
        transaction,
        receipt_id="rollback-1",
        authorization_receipt=rollback_authorization,
        recorded_at="2026-07-27T09:30:00Z",
    )
    validate_rollback_receipt(
        rollback,
        transaction=transaction,
        authorization_receipt=rollback_authorization,
    )
    assert rollback["rollback_performed"] is True
    assert not store.registry_path.exists()
    assert not store.active_set_path.exists()
    assert (
        store.root
        / "receipts/control_activations/control-activation-1.json"
    ).exists()
    assert (store.root / "receipts/v4_activations/v4-activation-1.json").exists()
    assert (
        store.root
        / "rollback_authorizations/rollback-authorization-1.json"
    ).exists()
    with pytest.raises(ProductionControlError):
        store.apply(
            transaction,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
            source_artifacts=source_artifacts,
        )


def test_crash_after_registry_cas_recovers_idempotently(
    tmp_path: Path,
) -> None:
    (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    ) = _artifacts()

    def crash(state: str) -> None:
        if state == "REGISTRY_COMMITTED":
            raise ProductionControlCrash(state)

    root = (tmp_path / "control").resolve()
    crashing = ProductionControlStore(root, fault_hook=crash)
    with pytest.raises(ProductionControlCrash):
        crashing.apply(
            transaction,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
            source_artifacts=source_artifacts,
        )
    assert crashing.registry_path.exists()
    assert not crashing.active_set_path.exists()

    recovered = ProductionControlStore(root)
    receipt = recovered.apply(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
        source_artifacts=source_artifacts,
    )
    assert receipt["active_for_production_research"] is True
    states = [
        json.loads(line)["state"]
        for line in (
            root / f"wal/{transaction['transaction_id']}.jsonl"
        ).read_text().splitlines()
    ]
    assert states == [
        "PREPARED",
        "REGISTRY_COMMITTED",
        "V4_RECEIPT_ISSUED",
        "READINESS_RECOMPUTED",
        "ACTIVE_SET_COMMITTED",
    ]


def test_cas_drift_and_expired_authorization_fail_closed(
    tmp_path: Path,
) -> None:
    (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    ) = _artifacts(
        expected_registry_sha256=_digest("different-registry")
    )
    store = ProductionControlStore((tmp_path / "control").resolve())
    with pytest.raises(ProductionControlError):
        store.apply(
            transaction,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
            source_artifacts=source_artifacts,
        )
    assert not store.registry_path.exists()
    assert not store.active_set_path.exists()
    assert not (store.root / f"wal/{transaction['transaction_id']}.jsonl").exists()

    tampered_sources = deepcopy(source_artifacts)
    tampered_sources["v4_evidence"]["factors"][0][
        "evidence_payload_sha256"
    ] = _digest("tampered-evidence-payload")
    clean_store = ProductionControlStore(
        (tmp_path / "tampered-control").resolve()
    )
    with pytest.raises(ProductionControlError):
        clean_store.apply(
            transaction,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
            source_artifacts=tampered_sources,
        )
    assert not (
        clean_store.root
        / f"transactions/{transaction['transaction_id']}.json"
    ).exists()

    expired = build_authorization_receipt(
        receipt_id="expired",
        authorized_by="codex_delegated_reviewer",
        issued_at="2026-07-27T06:00:00Z",
        expires_at="2026-07-27T07:00:00Z",
        transaction_plan_sha256=authorization["transaction_plan_sha256"],
        proposed_registry_sha256=authorization[
            "proposed_registry_sha256"
        ],
        production_factor_set_sha256=authorization[
            "production_factor_set_sha256"
        ],
        runtime_contracts_sha256=authorization[
            "runtime_contracts_sha256"
        ],
    )
    with pytest.raises(ProductionControlError):
        build_production_control_transaction(
            transaction_id="expired-transaction",
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=expired,
            expected_registry_sha256=EMPTY_SHA256,
            expected_active_set_sha256=EMPTY_SHA256,
            activated_at=ACTIVATED_AT,
            v4_activation_receipt_id="v4-expired",
            control_receipt_id="control-expired",
        )


def test_transaction_bytes_are_canonical_and_tamper_fails() -> None:
    registry, eligibility, authorization, transaction, _ = _artifacts()
    assert canonical_file_bytes(transaction).endswith(b"\n")
    tampered = deepcopy(transaction)
    tampered["proposed_registry_sha256"] = _digest("tampered")
    tampered["semantic_sha256"] = _semantic(tampered)
    with pytest.raises(ProductionControlError):
        validate_production_control_transaction(
            tampered,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
        )


def test_store_rejects_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ProductionControlError):
        ProductionControlStore(linked)
