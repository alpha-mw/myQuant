"""Actual confined storage/locks; pure saved resolution uses controlled native EOD admission."""

import os
import pickle
import time

import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.automatic_catchup_contract import document_ref, run_path
from quant_investor.operations.automatic_catchup_storage import (
    AutomaticRunStorage,
    PENDING,
    PENDING_SCHEMA,
    automatic_execution,
    require_dispatch_ownership,
)
from quant_investor.operations.daily_contract import ContractError
from quant_investor.system.errors import SystemSecurityError
from test_automatic_catchup_resolution import fixture, resolve
from test_daily_evidence_public_catchup import snapshot
from _public_catchup_fixture import put


def prepared(root, monkeypatch):
    request, ref, _, _, _ = fixture(root, monkeypatch)
    resolution = resolve(root, ref, request)["resolution"]
    path = run_path(ref, "resolution.v1.json")
    resolution_ref = document_ref(path, resolution)
    pending = {
        "schema_version": PENDING_SCHEMA,
        "state": "ACTIVE",
        "auto_request_ref": ref,
        "resolution_ref": resolution_ref,
    }
    return AutomaticRunStorage(str(root)), resolution_ref, resolution, pending


def test_pending_read_creates_nothing_and_second_lock_returns_promptly(tmp_path, monkeypatch):
    storage, _, _, _ = prepared(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    directories = set(tmp_path.rglob("*"))
    assert storage.pending() is None
    assert snapshot(tmp_path) == before and set(tmp_path.rglob("*")) == directories
    with storage.locked():
        other = AutomaticRunStorage(str(tmp_path))
        started = time.monotonic()
        with pytest.raises(ContractError, match="AUTO_RUN_BUSY"):
            with other.locked():
                pytest.fail("overlap acquired active run lock")
        assert time.monotonic() - started < 2
    with other.locked():
        other.require_lock()


@pytest.mark.parametrize(
    "path",
    [
        "results/system/_active.json",
        "results/operations/daily_production/CN/20260828/forbidden.json",
        "results/operations/daily_production/CN/automatic/"
        "aggressive_tech_manufacturing/arbitrary.json",
    ],
)
def test_storage_never_writes_outside_owned_paths(tmp_path, monkeypatch, path):
    storage, _, _, _ = prepared(tmp_path, monkeypatch)
    with storage.locked():
        before = snapshot(tmp_path)
        with pytest.raises(SystemSecurityError):
            storage.write(path, b"{}")
        assert snapshot(tmp_path) == before


def test_lease_is_exact_and_unknown_old_state_cannot_be_replaced(tmp_path, monkeypatch):
    storage, ref, resolution, pending = prepared(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="AUTO_RUN_LOCK_REQUIRED"):
        storage.set_pending(pending)
    with storage.locked():
        storage.write(ref["path"], canonical_json_bytes(resolution))
        stored = storage.set_pending(pending)
        before = snapshot(tmp_path)
        assert storage.set_pending(pending) == stored
        assert snapshot(tmp_path) == before
        put(tmp_path, PENDING, {**pending, "state": "UNKNOWN"})
        before = snapshot(tmp_path)
        with pytest.raises(ContractError, match="AUTO_PENDING_INVALID"):
            storage.set_pending({**pending, "state": "IDLE"})
        assert snapshot(tmp_path) == before


def test_crash_after_lease_replace_recovers_same_bytes(tmp_path, monkeypatch):
    storage, ref, resolution, pending = prepared(tmp_path, monkeypatch)
    with storage.locked():
        storage.write(ref["path"], canonical_json_bytes(resolution))
        original = os.replace

        def crash(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("synthetic crash after durable replacement")

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", crash)
            with pytest.raises(OSError, match="synthetic crash"):
                storage.set_pending(pending)
        assert storage.pending() == pending
        before = snapshot(tmp_path)
        storage.set_pending(pending)
        assert snapshot(tmp_path) == before


def test_capability_is_exact_nonserializable_revoked_and_closed_run_refused(tmp_path, monkeypatch):
    storage, ref, resolution, pending = prepared(tmp_path, monkeypatch)
    args = dict(
        workspace=str(tmp_path),
        request_ref=resolution["derived_request_ref"],
        request=resolution["derived_request"],
    )
    with pytest.raises(ContractError, match="AUTO_OWNED_REQUEST_INTERNAL_ONLY"):
        require_dispatch_ownership(**args)
    with storage.locked():
        storage.write(ref["path"], canonical_json_bytes(resolution))
        storage.set_pending(pending)
        with automatic_execution(storage, resolution_ref=ref, resolution=resolution) as capability:
            require_dispatch_ownership(**args)
            with pytest.raises(TypeError, match="cannot be serialized"):
                pickle.dumps(capability)
            with pytest.raises(ContractError, match="AUTO_CAPABILITY_BINDING_INVALID"):
                require_dispatch_ownership(
                    **{
                        **args,
                        "request_ref": {
                            **args["request_ref"],
                            "path": "copied-request.json",
                        },
                    }
                )
        with pytest.raises(ContractError, match="AUTO_CAPABILITY_EXPIRED"):
            capability.require(str(tmp_path), args["request_ref"])
        storage.write(run_path(resolution["auto_request_ref"], "closure.v1.json"), b"{}")
        with pytest.raises(ContractError, match="AUTO_RESOLUTION_CLOSED"):
            with automatic_execution(storage, resolution_ref=ref, resolution=resolution):
                pytest.fail("closed resolution received execution capability")


def test_generated_current_request_cannot_execute_through_public_dispatch_guard(tmp_path):
    request = {
        "action": "EXECUTE",
        "recipe_ref": {
            "path": "results/operations/daily_production/CN/20260828/catchup/"
            + "a" * 64
            + "/recipe.json",
            "sha256": "b" * 64,
        },
    }
    with pytest.raises(ContractError, match="CATCHUP_GENERATED_EXECUTE_INTERNAL_ONLY"):
        require_dispatch_ownership(
            workspace=str(tmp_path),
            request_ref={"path": "copied-current.json", "sha256": "c" * 64},
            request=request,
        )
