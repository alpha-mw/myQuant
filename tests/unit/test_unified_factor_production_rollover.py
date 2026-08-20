from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from quant_investor.contracts import canonical_json_bytes, get_contract
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.factors.production_authority import (
    FACTOR_ACTIVE_POINTER_PATH,
    _FactorSecureStorage,
    validate_factor_active_pointer,
)
from quant_investor.factors.production_rollover import (
    validate_daily_maintenance_receipt,
)
from quant_investor.cli.unified import factor_production_rollover


def _pointer(*, generation: str, previous: str, activated_at: str) -> bytes:
    return canonical_json_bytes(
        {
            "factor_generation_id": "factor-production-generation-" + generation,
            "factor_generation_sha256": generation,
            "previous_pointer_sha256": previous,
            "activated_at": activated_at,
            "os_actor": f"uid:{os.geteuid()}",
            "authority_scope": "FACTOR_PRODUCTION",
        }
    )


def _write(path: Path, value: object | bytes) -> str:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def test_successor_pointer_accepts_exact_sha_preimage_only() -> None:
    successor = _pointer(
        generation="2" * 64,
        previous="1" * 64,
        activated_at="2026-08-20T13:00:00Z",
    )
    assert validate_factor_active_pointer(successor)["previous_pointer_sha256"] == "1" * 64
    invalid = successor.replace(("1" * 64).encode(), b"UPPERCASE-NOT-A-SHA")
    with pytest.raises(FactorGovernanceError):
        validate_factor_active_pointer(invalid)


def test_rollover_contracts_are_distinct_from_initial_activation() -> None:
    assert get_contract("factor.production_rollover_bundle").identity_field == (
        "factor_production_rollover_bundle_id"
    )
    assert get_contract("factor.production_rollover_prepared").identity_field == (
        "factor_production_rollover_prepared_id"
    )
    assert get_contract("factor.production_rollover_commit").identity_field == (
        "factor_production_rollover_commit_id"
    )


def test_cooperative_pointer_replace_requires_exact_under_lock_preimage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    storage = _FactorSecureStorage(workspace)
    genesis = _pointer(
        generation="1" * 64,
        previous="EMPTY",
        activated_at="2026-08-19T13:00:00Z",
    )
    successor = _pointer(
        generation="2" * 64,
        previous=hashlib.sha256(genesis).hexdigest(),
        activated_at="2026-08-20T13:00:00Z",
    )
    with storage.exclusive_lock("results/factors/.active.lock"):
        storage.write_initial_pointer_under_lock(genesis)
    with storage.exclusive_lock("results/factors/.active.lock"):
        with pytest.raises(FactorGovernanceError, match="preimage changed"):
            storage.replace_active_pointer_under_lock(successor, expected_pointer_sha256="f" * 64)
    assert storage.read(FACTOR_ACTIVE_POINTER_PATH).data == genesis
    with storage.exclusive_lock("results/factors/.active.lock"):
        stored = storage.replace_active_pointer_under_lock(
            successor,
            expected_pointer_sha256=hashlib.sha256(genesis).hexdigest(),
        )
    assert stored.data == successor
    assert storage.read(FACTOR_ACTIVE_POINTER_PATH).data == successor


def _maintenance_attempt(tmp_path: Path, *, mode: str = "execute", status: str = "COMPLETE"):
    workspace = tmp_path / "workspace"
    attempt = workspace / "data/private/cn_daily_maintenance/attempts/run-1"
    pit_root = workspace / "data/parquet/cn/reference"
    pit_manifest_path = pit_root / "generation/manifest.json"
    pit_membership_path = pit_root / "generation/membership.json"
    pit_membership_sha = _write(pit_membership_path, {"records": []})
    pit_manifest_sha = _write(pit_manifest_path, {"generation_id": "pit-test"})
    pit_pointer_path = pit_root / "stock_basic_membership_latest.json"
    pit_pointer_sha = _write(
        pit_pointer_path,
        {
            "generation_id": "pit-test",
            "generation_manifest_path": str(pit_manifest_path),
            "generation_manifest_sha256": pit_manifest_sha,
            "canonical_path": str(pit_membership_path),
            "canonical_sha256": pit_membership_sha,
        },
    )
    market_manifest_path = workspace / "data/parquet/cn/snapshot.json"
    market_manifest_sha = _write(market_manifest_path, {"snapshot_id": "market-test"})
    market_pointer_path = workspace / "data/parquet/cn/_latest.json"
    market_pointer_sha = _write(
        market_pointer_path,
        {
            "latest_complete_trade_date": "20260820",
            "manifest_path": str(market_manifest_path),
            "coverage": {
                "pit_generation_id": "pit-test",
                "pit_generation_manifest_sha256": pit_manifest_sha,
                "pit_membership_sha256": pit_membership_sha,
            },
        },
    )
    history_path = attempt / "history.json"
    history_sha = _write(
        history_path,
        {
            "history_audit_status": "passed",
            "target_trade_date": "20260820",
            "effective_trade_date": "20260820",
            "audited_trade_dates_count": 100,
            "canonical": {"latest_sha256": market_pointer_sha},
        },
    )
    raw_response = b'{"code":0,"data":{"fields":[],"items":[]}}'
    raw_path = attempt / "close-session.raw.json"
    raw_sha = _write(raw_path, raw_response)
    close = {
        "target_trade_date": "20260820",
        "raw_response_path": str(raw_path),
        "raw_response_sha256": raw_sha,
    }
    close_path = attempt / "close-session-receipt.json"
    close_sha = _write(close_path, close)
    state = {
        "schema_version": "cn-daily-maintenance-state.v1",
        "status": status,
        "maintenance_status": status,
        "same_day_status": status,
        "fundamental_integrity_status": "READY",
        "fundamental_refresh_status": "HEALTH_ONLY",
        "mode": mode,
        "attempt_slot": "2020",
        "target_date": "20260820",
        "stage_states": {
            stage: "READY" for stage in ("PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE")
        },
        "blockers": [],
    }
    state_path = attempt / "state.json"
    state_sha = _write(state_path, state)
    stages = [
        {
            "stage": stage,
            "status": "READY",
            "write_performed": status == "COMPLETE",
            "blockers": [],
            "evidence": {},
        }
        for stage in ("PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE")
    ]
    stages[0]["evidence"] = {
        "pit_binding": {
            "generation_id": "pit-test",
            "generation_manifest_path": str(pit_manifest_path),
            "generation_manifest_sha256": pit_manifest_sha,
            "canonical_path": str(pit_membership_path),
            "canonical_sha256": pit_membership_sha,
            "discovery_pointer_path": str(pit_pointer_path),
            "discovery_pointer_sha256": pit_pointer_sha,
        }
    }
    stages[1]["evidence"] = {
        "pointer_path": str(market_pointer_path),
        "pointer_sha256": market_pointer_sha,
        "snapshot_manifest_path": str(market_manifest_path),
        "snapshot_manifest_sha256": market_manifest_sha,
    }
    stages[2]["evidence"] = {
        "audit_path": str(history_path),
        "audit_sha256": history_sha,
        "history_audit_status": "passed",
    }
    receipt = {
        "schema_version": "cn-daily-maintenance-attempt.v1",
        "status": status,
        "maintenance_status": status,
        "same_day_status": status,
        "fundamental_integrity_status": "READY",
        "fundamental_refresh_status": "HEALTH_ONLY",
        "mode": mode,
        "attempt_slot": "2020",
        "target_date": "20260820",
        "canonical_unchanged": status == "NO_ACTION",
        "canonical_write_count": 0 if status == "NO_ACTION" else 4,
        "usable_for_investment_research": "UNCONFIRMED",
        "close_session_receipt_ref": {"path": str(close_path), "sha256": close_sha},
        "stage_results": stages,
        "blockers": [],
        "protected_surfaces": [],
        "state_ref": {"path": str(state_path), "sha256": state_sha},
    }
    receipt_path = attempt / "attempt.json"
    receipt_sha = _write(receipt_path, receipt)
    return workspace, receipt_path, receipt_sha


def test_maintenance_receipt_accepts_only_exact_execute_success(tmp_path: Path) -> None:
    workspace, receipt_path, receipt_sha = _maintenance_attempt(tmp_path)
    result = validate_daily_maintenance_receipt(
        workspace_root=workspace,
        receipt_path=receipt_path,
        expected_receipt_sha256=receipt_sha,
    )
    assert result["target_date"] == "20260820"
    assert result["status"] == "COMPLETE"


def test_maintenance_receipt_rejects_shadow_and_sha_drift(tmp_path: Path) -> None:
    workspace, receipt_path, receipt_sha = _maintenance_attempt(tmp_path, mode="shadow")
    with pytest.raises(FactorGovernanceError, match="authoritative success"):
        validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=receipt_path,
            expected_receipt_sha256=receipt_sha,
        )
    with pytest.raises(FactorGovernanceError, match="SHA differs"):
        validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=receipt_path,
            expected_receipt_sha256="0" * 64,
        )


def test_public_rollover_reaches_indexed_post_cas_commit_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_investor.factors.production_authority as authority_module
    import quant_investor.factors.production_rollover as rollover_module

    maintenance = {
        "target_date": "20260820",
        "upstream_maintenance_status": "PARTIAL",
        "macro_status": "BLOCKED",
        "macro_blockers": ["MACRO_RELEASE_CONTRACT_BLOCKED"],
        "macro_used_by_factor": False,
        "core_closure": {},
    }
    monkeypatch.setattr(
        rollover_module,
        "validate_daily_maintenance_receipt",
        lambda **_kwargs: maintenance,
    )
    calls: list[dict[str, str]] = []

    class Store:
        def __init__(self, _workspace: str) -> None:
            pass

        def read(self, _path: str):
            return type("Stored", (), {"byte_sha256": "2" * 64})()

        def recover_rollover_for_inputs(self, **kwargs: str):
            calls.append(dict(kwargs))
            return {
                "factor_authority": "ACTIVE",
                "factor_readiness": "READY",
                "factor_generation_id": "factor-production-generation-" + "3" * 64,
                "factor_generation_sha256": "3" * 64,
                "factor_pointer_byte_sha256": "2" * 64,
                "factor_pointer_semantic_sha256": "4" * 64,
                "marker_byte_sha256": "5" * 64,
                "marker_semantic_sha256": "6" * 64,
                "active_factors": [],
                "control_factors": [],
                "as_of": "20260820",
                "rollover": {
                    "cas_performed": False,
                    "marker_only_recovery": False,
                    "commit_recovered": True,
                    "idempotent_replay": False,
                    "previous_pointer_sha256": "1" * 64,
                    "target_pointer_sha256": "2" * 64,
                    "rollover_commit_ref": {"kind": "factor.production_rollover_commit"},
                },
            }

    monkeypatch.setattr(authority_module, "FactorProductionStore", Store)
    result = factor_production_rollover(
        workspace_root=str(tmp_path),
        market_data_root=str(tmp_path),
        calendar_capture_root=str(tmp_path),
        expected_calendar_success_sha256="7" * 64,
        maintenance_receipt=str(tmp_path / "attempt.json"),
        expected_maintenance_receipt_sha256="8" * 64,
        expected_current_pointer_sha256="1" * 64,
    )

    assert result["command_status"] == "ROLLOVER_COMMIT_RECOVERED"
    assert calls == [
        {
            "expected_pointer_sha256": "1" * 64,
            "maintenance_sha256": "8" * 64,
        }
    ]
