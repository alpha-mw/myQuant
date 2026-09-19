"""Mechanically extracted serialization preserves bytes and distinct owner guards."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import inspect
import pytest
from quant_investor.market import daily_maintenance as daily
from quant_investor.market.scope_transition import _seal_scope_transition_attempt


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("core", [False, True])
def test_daily_seal_matches_pinned_pre_refactor_bytes(tmp_path, monkeypatch, started, core):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 8, 13, tzinfo=timezone.utc)

    monkeypatch.setattr(daily, "datetime", Clock)
    monkeypatch.setattr(daily, "operation_provider_summary", lambda: {"provider_calls": False})
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o700)
    claim = tmp_path / "claim.json"
    claim.write_bytes(b'{"test_only":true}')
    claim.chmod(0o600)
    ref = {"path": str(claim), "sha256": hashlib.sha256(claim.read_bytes()).hexdigest()}
    for name, present in [("started.json", started), ("core-completion.json", core)]:
        if present:
            p = attempt / name
            p.write_bytes(b'{"test_only":true}')
            p.chmod(0o600)
    captured = []
    native_write = daily._write_once

    def recording(path, raw):
        captured.append((path.name, raw))
        return native_write(path, raw)

    monkeypatch.setattr(daily, "_write_once", recording)
    reference = (Path(__file__).parent / "fixtures/seal_attempt_pre_refactor.txt").read_bytes()
    assert (
        hashlib.sha256(reference).hexdigest()
        == "5945a7c6f05914b5f0060edcce5578ce0e771f5df4029f88aa984e42d3c49fbe"
    )
    namespace = dict(vars(daily))
    exec(compile(reference, "pinned-pre-refactor-test-reference", "exec"), namespace)
    payload = {
        "schema_version": "cn-daily-maintenance-attempt.v1",
        "status": "READY",
        "logical_claim_ref": ref,
    }
    state = {"status": "READY"}
    old = namespace["_seal_attempt"](
        attempt_root=attempt, payload=payload, state=state, logical_claim_ref=ref
    )
    expected = list(captured)
    before = {p.name: p.read_bytes() for p in attempt.iterdir()}
    # Only remove records produced by the test oracle, so both writers see the
    # same fresh attempt path. Native no-overwrite semantics remain unchanged.
    for name in ("state.json", "attempt.json", "ended.json"):
        (attempt / name).unlink()
    captured.clear()
    actual = daily._seal_attempt(
        attempt_root=attempt, payload=payload, state=state, logical_claim_ref=ref
    )
    assert actual == old and captured == expected
    assert [name for name, _ in captured] == ["state.json", "attempt.json", "ended.json"]
    assert before == {p.name: p.read_bytes() for p in attempt.iterdir()}


@pytest.mark.parametrize(
    "schema", ["cn-scope-transition-readiness.v1", "cn-scope-transition-attempt.v1"]
)
def test_scope_owner_seals_without_inventing_daily_claim(tmp_path, schema):
    payload = {"schema_version": schema, "status": "BLOCKED"}
    result = _seal_scope_transition_attempt(attempt_root=tmp_path, payload=payload, state=payload)
    assert result["schema_version"] == schema
    assert "logical_claim_ref" not in result
    assert all((tmp_path / name).exists() for name in ["state.json", "attempt.json", "ended.json"])


@pytest.mark.parametrize("fault", ["mismatch", "daily", "payload_claim", "state_claim"])
def test_scope_rejections_write_no_records(tmp_path, fault):
    payload = {"schema_version": "cn-scope-transition-attempt.v1", "status": "BLOCKED"}
    state = dict(payload)
    if fault == "mismatch":
        state["schema_version"] = "cn-scope-transition-readiness.v1"
    elif fault == "daily":
        payload["schema_version"] = state["schema_version"] = "cn-daily-maintenance-attempt.v1"
    else:
        (payload if fault == "payload_claim" else state)["logical_claim_ref"] = {}
    with pytest.raises(RuntimeError, match="SCOPE_TRANSITION_ATTEMPT_SCHEMA_INVALID"):
        _seal_scope_transition_attempt(attempt_root=tmp_path, payload=payload, state=state)
    assert list(tmp_path.iterdir()) == []


def test_daily_claim_remains_mandatory_and_checked_before_writes(tmp_path):
    assert (
        inspect.signature(daily._seal_attempt).parameters["logical_claim_ref"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(TypeError):
        daily._seal_attempt(attempt_root=tmp_path, payload={}, state={})
    assert list(tmp_path.iterdir()) == []
    claim = tmp_path / "claim.json"
    claim.write_bytes(b"{}")
    claim.chmod(0o600)
    with pytest.raises(daily.DailyMaintenanceError, match="LOGICAL_CLAIM_CHANGED_BEFORE_SEAL"):
        daily._seal_attempt(
            attempt_root=tmp_path,
            payload={},
            state={},
            logical_claim_ref={"path": str(claim), "sha256": "0" * 64},
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["claim.json"]
