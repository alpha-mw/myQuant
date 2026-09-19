"""Retained future references reach the real Factor-loop recovery entry."""

import hashlib
from copy import deepcopy

import pytest

from quant_investor.market import daily_factor_loop as loop
from quant_investor.market import future_calendar_producer as producer
from quant_investor.factors import production_observation
from quant_investor.operations import core_pool
from test_future_calendar_context import context, state, REF


def setup(tmp_path, monkeypatch):
    instance = object.__new__(loop.DailyFactorLoop)
    instance.workspace = tmp_path
    instance.context = context()
    instance.context_sha256 = "c" * 64
    instance._core_release_ref = lambda: REF
    writes = []
    instance._save_state = lambda value: writes.append(deepcopy(value))
    value = state()
    value.update(phase="FUTURE_BOUND", next_session_calendar_failure_ref=REF)
    for alias in ("LOW", "W80"):
        value["core_observation_refs"][alias] = {
            "path": str(tmp_path / f"results/factors/observations/2026/09/02/{alias}.json"),
            "sha256": hashlib.sha256(alias.encode()).hexdigest(),
        }
    monkeypatch.setattr(
        loop,
        "_read_owner_file",
        lambda path, **kw: (path.stem.encode(), hashlib.sha256(path.stem.encode()).hexdigest()),
    )
    monkeypatch.setattr(
        production_observation,
        "validate_factor_production_observation",
        lambda raw: {
            "payload": {
                "factor_alias": raw.decode(),
                "signal_date": "20260902",
                "factor_pointer_sha256": "a" * 64,
                "factor_generation_sha256": "b" * 64,
            }
        },
    )
    calls = []

    def bind(**kw):
        assert kw["recovery"] is True and kw["context_sha"] == "c" * 64
        calls.append("retained-binding")
        return kw["state"]

    monkeypatch.setattr(producer, "bind_future_calendar", bind)
    monkeypatch.setattr(
        core_pool, "publish_core_pool", lambda **kw: calls.append(kw) or {"core_handoff_ref": REF}
    )
    return instance, value, calls, writes


def test_recovery_forwards_exact_retained_failure_before_core(tmp_path, monkeypatch):
    instance, value, calls, writes = setup(tmp_path, monkeypatch)
    instance._recover_core(value)
    assert calls[0] == "retained-binding"
    assert calls[1]["next_session_calendar_failure_ref"] == REF
    assert calls[1]["next_session_calendar_proof_ref"] is None
    assert writes == []


def test_recovery_forwards_exact_retained_proof(tmp_path, monkeypatch):
    instance, value, calls, _ = setup(tmp_path, monkeypatch)
    value.update(next_session_calendar_failure_ref=None, next_session_calendar_proof_ref=REF)
    instance._recover_core(value)
    assert calls[1]["next_session_calendar_proof_ref"] == REF
    assert calls[1]["next_session_calendar_failure_ref"] is None


def test_wrong_day_cannot_write_evidence_or_publish_core(tmp_path, monkeypatch):
    instance, value, calls, writes = setup(tmp_path, monkeypatch)
    value["trade_date"] = "20260903"
    with pytest.raises(ValueError, match="STATE_DATE_MISMATCH"):
        instance._recover_core(value)
    assert calls == writes == []


def test_invalid_retained_state_rejected_before_observation_writes(tmp_path, monkeypatch):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.daily_contract import ContractError

    instance = object.__new__(loop.DailyFactorLoop)
    instance.workspace = tmp_path
    instance.run_root = tmp_path / "maintenance"
    instance.run_root.mkdir(mode=0o700)
    instance.context = context()
    instance.context_sha256 = "c" * 64
    value = state()
    value["context_sha256"] = "d" * 64
    path = instance.run_root / "factor-loop-state.json"
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    monkeypatch.setattr(
        loop,
        "register_factor_production_observations",
        lambda *a, **kw: pytest.fail("invalid state reached writer"),
    )
    with pytest.raises(ContractError, match="STATE_INVALID"):
        instance.recover()


def test_core_completed_rejects_retained_mismatch_before_factor_work(tmp_path, monkeypatch):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.daily_contract import ContractError

    instance = object.__new__(loop.DailyFactorLoop)
    instance.workspace = tmp_path
    instance.run_root = tmp_path / "maintenance"
    instance.run_root.mkdir(mode=0o700)
    instance.context = context()
    instance.context_sha256 = "c" * 64
    instance.stages = {}
    value = state()
    value["context_sha256"] = "d" * 64
    path = instance.run_root / "factor-loop-state.json"
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    monkeypatch.setattr(
        loop,
        "validate_daily_maintenance_receipt",
        lambda **kw: {"target_date": "20260902", "close_session_receipt_ref": REF},
    )
    instance._attempt = lambda *a, **kw: pytest.fail("invalid state reached Factor stage")
    with pytest.raises(ContractError, match="STATE_INVALID"):
        instance.core_completed(REF)
    assert not instance.stages
