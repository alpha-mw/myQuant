from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest

from quant_investor.factors.governance_protocol_v4 import protocol_hash, semantic_sha256
from quant_investor.factors.governance_transaction_v4 import (
    FactorV4ShadowTransactionStore,
    FactorGovernanceTransactionV4Error,
    activation_receipt_sha256,
    build_activation_request_v4,
    build_factor_v4_transaction_plan,
    canonical_shadow_registry_bytes_v4,
    shadow_file_sha256_v4,
    validate_activation_receipt_v4,
    validate_factor_v4_transaction_plan,
    validate_inverse_rollback_manifest_v4,
    validate_shadow_activation_receipt_v4,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _proposal(lower: float = 0.01) -> dict:
    return {
        "action": "replace_proposal",
        "incumbent": "old-factor",
        "challenger": "new-factor",
        "incremental_edge_ci95_lower": lower,
        "apply": False,
    }


def _plan() -> dict:
    return build_factor_v4_transaction_plan(
        transaction_id="factor-v4-test",
        as_of="2026-07-17",
        cadence="month_end",
        production_factor_count=10,
        expected_registry_file_sha256=_digest("registry-before"),
        proposed_registry_file_sha256=_digest("registry-after"),
        expected_production_factor_set_sha256=_digest("set-before"),
        proposed_production_factor_set_sha256=_digest("set-after"),
        proposals=[_proposal()],
        wal_path="research/factor_v4/wal.jsonl",
        inverse_rollback_path="research/factor_v4/inverse.json",
    )


def test_v4_transaction_is_inert_and_binds_wal_cas_and_inverse() -> None:
    plan = _plan()
    validated = validate_factor_v4_transaction_plan(plan)
    assert validated["status"] == "plan_ready"
    assert validated["plan_only"] is True
    assert validated["production_apply_enabled"] is False
    assert validated["registry_mutation_performed"] is False
    assert validated["wal"]["status"] == "planned_not_written"
    assert validated["wal"]["write_performed"] is False
    assert validated["cas"]["status"] == "planned_not_attempted"
    assert validated["cas"]["performed"] is False
    assert validated["inverse_rollback_plan"]["status"] == "planned_not_applied"
    assert validated["inverse_rollback_plan"]["performed"] is False


def test_v4_transaction_rejects_tampered_cas_and_nonpositive_target_edge() -> None:
    tampered = copy.deepcopy(_plan())
    tampered["cas"]["compare_registry_file_sha256"] = _digest("wrong")
    with pytest.raises(FactorGovernanceTransactionV4Error, match="CAS registry SHA"):
        validate_factor_v4_transaction_plan(tampered)

    blocked = build_factor_v4_transaction_plan(
        transaction_id="blocked",
        as_of="2026-07-17",
        cadence="month_end",
        production_factor_count=10,
        expected_registry_file_sha256=_digest("registry-before"),
        proposed_registry_file_sha256=_digest("registry-after"),
        expected_production_factor_set_sha256=_digest("set-before"),
        proposed_production_factor_set_sha256=_digest("set-after"),
        proposals=[_proposal(0.0)],
        wal_path="research/factor_v4/wal.jsonl",
        inverse_rollback_path="research/factor_v4/inverse.json",
    )
    assert blocked["status"] == "plan_blocked"
    assert any("incremental_edge" in item for item in blocked["blockers"])


def _receipt() -> dict:
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": _digest("transaction"),
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": _digest("factor-set"),
        "runtime_contracts_sha256": _digest("runtime-contracts"),
        "as_of": "2026-07-17",
    }
    payload = {
        "schema_version": "factor-governance-activation-receipt.v4",
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": "receipt-test",
        "status": "activated",
        "authorization_scope": "factor_v4_production_activation",
        "authorized_by": "Maxwell",
        "activated_at": "2026-07-17T15:30:00+08:00",
        "as_of": "2026-07-17",
        "transaction_plan_sha256": context["transaction_plan_sha256"],
        "registry_file_sha256": context["registry_file_sha256"],
        "production_factor_set_sha256": context["production_factor_set_sha256"],
        "runtime_contracts_sha256": context["runtime_contracts_sha256"],
        "activation_context_sha256": semantic_sha256(context),
        "activation_performed": True,
    }
    payload["receipt_sha256"] = activation_receipt_sha256(payload)
    return payload


def test_activation_request_is_not_receipt_and_receipt_is_same_day_hash_bound() -> None:
    request = build_activation_request_v4(
        request_id="request-test",
        as_of="2026-07-17",
        transaction_plan_sha256=_digest("transaction"),
        proposed_registry_file_sha256=_digest("registry"),
        proposed_production_factor_set_sha256=_digest("factor-set"),
        runtime_contracts_sha256=_digest("runtime-contracts"),
    )
    assert request["status"] == "pending_separate_human_authorization"
    assert request["activation_performed"] is False
    with pytest.raises(FactorGovernanceTransactionV4Error, match="fields invalid"):
        validate_activation_receipt_v4(request)

    receipt = _receipt()
    validated = validate_activation_receipt_v4(
        receipt,
        expected_as_of="2026-07-17",
        expected_protocol_hash=protocol_hash(),
        expected_registry_file_sha256=_digest("registry"),
        expected_production_factor_set_sha256=_digest("factor-set"),
        expected_runtime_contracts_sha256=_digest("runtime-contracts"),
    )
    assert validated["status"] == "activated"

    stale = copy.deepcopy(receipt)
    stale["activated_at"] = "2026-07-16T15:30:00+08:00"
    stale["receipt_sha256"] = activation_receipt_sha256(stale)
    with pytest.raises(FactorGovernanceTransactionV4Error, match="fresh"):
        validate_activation_receipt_v4(stale)


def test_independent_shadow_store_executes_atomic_cas_wal_receipt_and_rollback(
    tmp_path: Path,
) -> None:
    store = FactorV4ShadowTransactionStore(tmp_path / "factor-v4-shadow")
    before_registry = {
        "schema_version": "factor-v4-shadow-registry.v1",
        "factors": ["old-factor"],
    }
    after_registry = {
        "schema_version": "factor-v4-shadow-registry.v1",
        "factors": ["new-factor"],
    }
    before_sha = store.initialize_shadow_registry(before_registry)
    after_sha = shadow_file_sha256_v4(canonical_shadow_registry_bytes_v4(after_registry))
    plan = build_factor_v4_transaction_plan(
        transaction_id="shadow-cas-test",
        as_of="2026-07-17",
        cadence="month_end",
        production_factor_count=10,
        expected_registry_file_sha256=before_sha,
        proposed_registry_file_sha256=after_sha,
        expected_production_factor_set_sha256=_digest("set-before"),
        proposed_production_factor_set_sha256=_digest("set-after"),
        proposals=[_proposal()],
        wal_path=str(store.wal_path),
        inverse_rollback_path=str(store.inverse_manifest_path),
    )
    receipt = store.apply_shadow_transaction(
        plan,
        after_registry,
        authorization={
            "authorization_scope": "factor_v4_research_shadow",
            "authorized_by": "test-reviewer",
            "authorized_at": "2026-07-17T09:30:00+08:00",
            "receipt_id": "shadow-receipt-test",
            "runtime_contracts_sha256": _digest("runtime-contracts"),
        },
    )
    validated_receipt = validate_shadow_activation_receipt_v4(
        receipt,
        expected_registry_file_sha256=after_sha,
        expected_production_factor_set_sha256=_digest("set-after"),
    )
    assert validated_receipt["shadow_activation_performed"] is True
    assert validated_receipt["production_activation_performed"] is False
    with pytest.raises(FactorGovernanceTransactionV4Error, match="fields invalid"):
        validate_activation_receipt_v4(receipt)
    assert store.registry_file_sha256() == after_sha

    inverse = json.loads(store.inverse_manifest_path.read_text(encoding="utf-8"))
    validated_inverse = validate_inverse_rollback_manifest_v4(inverse)
    assert validated_inverse["rollback_performed"] is False
    assert validated_inverse["restore_registry_file_sha256"] == before_sha
    wal_before_rollback = store.wal_path.read_bytes()
    assert wal_before_rollback.count(b"\n") == 2

    rollback = store.rollback_shadow_transaction(
        authorization={
            "authorization_scope": "factor_v4_shadow_rollback",
            "authorized_by": "test-reviewer",
            "authorized_at": "2026-07-17T10:00:00+08:00",
        }
    )
    assert rollback["status"] == "shadow_rolled_back"
    assert rollback["production_activation_performed"] is False
    assert store.registry_file_sha256() == before_sha
    assert store.wal_path.read_bytes().startswith(wal_before_rollback)
    assert store.wal_path.read_bytes().count(b"\n") == 4
    for path in (
        store.registry_path,
        store.wal_path,
        store.receipt_path,
        store.inverse_manifest_path,
        store.lock_path,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_shadow_store_cas_failure_leaves_registry_and_wal_unchanged(tmp_path: Path) -> None:
    store = FactorV4ShadowTransactionStore(tmp_path / "factor-v4-shadow")
    before = {"schema_version": "factor-v4-shadow-registry.v1", "factors": ["old"]}
    after = {"schema_version": "factor-v4-shadow-registry.v1", "factors": ["new"]}
    before_sha = store.initialize_shadow_registry(before)
    after_sha = shadow_file_sha256_v4(canonical_shadow_registry_bytes_v4(after))
    plan = build_factor_v4_transaction_plan(
        transaction_id="stale-cas",
        as_of="2026-07-17",
        cadence="month_end",
        production_factor_count=10,
        expected_registry_file_sha256=_digest("stale-before"),
        proposed_registry_file_sha256=after_sha,
        expected_production_factor_set_sha256=_digest("set-before"),
        proposed_production_factor_set_sha256=_digest("set-after"),
        proposals=[_proposal()],
        wal_path=str(store.wal_path),
        inverse_rollback_path=str(store.inverse_manifest_path),
    )
    with pytest.raises(FactorGovernanceTransactionV4Error, match="CAS compare failed"):
        store.apply_shadow_transaction(
            plan,
            after,
            authorization={
                "authorization_scope": "factor_v4_research_shadow",
                "authorized_by": "test-reviewer",
                "authorized_at": "2026-07-17T09:30:00+08:00",
                "receipt_id": "shadow-receipt-test",
                "runtime_contracts_sha256": _digest("runtime-contracts"),
            },
        )
    assert store.registry_file_sha256() == before_sha
    assert not store.wal_path.exists()
