from copy import deepcopy
from types import SimpleNamespace

import pytest

from quant_investor.market import future_calendar_producer as producer
from quant_investor.operations.daily_contract import ContractError
from test_future_calendar_context import context, state, REF


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(producer, "require_fixture_capability", lambda **kw: object())
    monkeypatch.setattr(producer, "future_calendar_outputs", lambda **kw: {})
    monkeypatch.setattr(producer, "selected_binding", lambda *a: (None, 0, None))
    monkeypatch.setattr(
        producer,
        "DailyJournal",
        lambda *a: SimpleNamespace(
            trade_date="20260902", root=tmp_path, storage=SimpleNamespace(read=lambda *a: None)
        ),
    )
    monkeypatch.setattr(
        producer,
        "capture_next_session_calendar",
        lambda **kw: pytest.fail("recovery called provider"),
    )
    saved = []
    failures = []
    monkeypatch.setattr(
        producer, "publish_next_session_failure", lambda **kw: failures.append(kw) or REF
    )
    args = dict(
        workspace=tmp_path,
        context=context(),
        context_sha="c" * 64,
        state=state(),
        save_state=lambda s: saved.append(deepcopy(s)),
        recovery=True,
    )
    return args, saved, failures


def capture():
    return {
        "capture_execution": {},
        "capture_success": {},
        "capture_execution_file_ref": {
            "relative_path": "cap/capture-execution.json",
            "byte_sha256": "a" * 64,
        },
        "capture_success_file_ref": {
            "relative_path": "cap/capture-success.json",
            "byte_sha256": "b" * 64,
        },
    }


def test_recovery_before_capture_records_failure_without_provider(monkeypatch, tmp_path):
    args, saved, failures = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(producer, "_retained_capture", lambda *a: None)
    out = producer.bind_future_calendar(**args)
    assert failures[0]["failure_code"] == "ACQUISITION_FAILED"
    assert out["phase"] == "FUTURE_BOUND" and saved[-1] == out


def test_capture_without_durable_provenance_cannot_be_adopted(monkeypatch, tmp_path):
    args, saved, failures = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(producer, "_retained_capture", lambda *a: capture())

    def missing(**kw):
        raise ContractError("PROVENANCE_UNAVAILABLE")

    monkeypatch.setattr(producer, "retained_fixture_evidence", missing)
    out = producer.bind_future_calendar(**args)
    assert failures[0]["failure_code"] == "PROVENANCE_UNAVAILABLE"
    assert out["next_session_calendar_proof_ref"] is None


def test_recovery_after_durable_evidence_finishes_publication_without_capture(
    monkeypatch, tmp_path
):
    args, saved, failures = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(producer, "_retained_capture", lambda *a: capture())
    monkeypatch.setattr(producer, "retained_fixture_evidence", lambda **kw: REF)
    monkeypatch.setattr(producer, "publish_bound_fixture_proof", lambda **kw: REF)
    out = producer.bind_future_calendar(**args)
    assert [s["phase"] for s in saved] == ["CAPTURE_BOUND", "FUTURE_BOUND"]
    assert out["next_session_calendar_proof_ref"] == REF and not failures


def test_selected_calendar_cannot_receive_future_evidence(monkeypatch, tmp_path):
    args, saved, failures = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(producer, "selected_binding", lambda *a: ("a" * 64, 0, None))
    with pytest.raises(ContractError, match="ALREADY_SELECTED"):
        producer.bind_future_calendar(**args)
    assert not saved and not failures


def test_saved_failure_is_read_only_during_recovery(monkeypatch, tmp_path):
    args, saved, failures = setup(monkeypatch, tmp_path)
    args["state"].update(phase="FUTURE_BOUND", next_session_calendar_failure_ref=REF)
    assert producer.bind_future_calendar(**args) == args["state"]
    assert not saved and not failures


def test_disabled_context_never_acquires(monkeypatch, tmp_path):
    args, saved, failures = setup(monkeypatch, tmp_path)
    args["context"]["next_session_calendar_mode"] = "DISABLED"
    args["recovery"] = False
    out = producer.bind_future_calendar(**args)
    assert out["phase"] == "FUTURE_BOUND" and not failures
    assert producer.core_future_arguments(out) == {
        "next_session_calendar_proof_ref": None,
        "next_session_calendar_failure_ref": None,
    }


def fresh_args(monkeypatch, tmp_path):
    from quant_investor.factors import production_rollover

    args, saved, failures = setup(monkeypatch, tmp_path)
    args["recovery"] = False
    monkeypatch.setattr(
        production_rollover, "_read_owner_file", lambda *a, **k: (b"release", "a" * 64)
    )
    return args, saved, failures


def test_native_failure_ref_is_retained_exactly(monkeypatch, tmp_path):
    from quant_investor.system.errors import SystemPreconditionError

    args, saved, failures = fresh_args(monkeypatch, tmp_path)
    ref = {"relative_path": "cap-failed/capture-failure.json", "byte_sha256": "e" * 64}
    exc = SystemPreconditionError("not published to diagnostic", code="NATIVE_FAILURE")
    exc.public_fields = {"capture_failure_file_ref": ref}

    def fail(**kw):
        raise exc

    monkeypatch.setattr(producer, "capture_next_session_calendar", fail)
    monkeypatch.setattr(producer, "_retained_native_failure", lambda *a: ref)
    producer.bind_future_calendar(**args)
    assert failures[0]["native_failure_ref"] == ref
    assert failures[0]["failure_code"] == "ACQUISITION_FAILED"
    assert "diagnostic" not in str(failures)
    assert saved[0]["phase"] == "CORE_READY"


@pytest.mark.parametrize(
    "code,expected",
    [
        ("NEXT_SESSION_HORIZON_INCOMPLETE", "HORIZON_INCOMPLETE"),
        ("NEXT_SESSION_EXCHANGE_DISAGREEMENT", "EXCHANGE_DISAGREEMENT"),
        ("NEXT_SESSION_EOD_NOT_OPEN", "EOD_NOT_OPEN"),
        ("NEXT_SESSION_NO_LATER_OPEN", "NO_LATER_OPEN"),
    ],
)
def test_projection_failure_keeps_capture_refs(monkeypatch, tmp_path, code, expected):
    args, saved, failures = fresh_args(monkeypatch, tmp_path)

    def fail(**kw):
        raise ContractError(code)

    monkeypatch.setattr(producer, "capture_next_session_calendar", fail)
    monkeypatch.setattr(producer, "_retained_capture", lambda *a: capture())
    producer.bind_future_calendar(**args)
    assert failures[0]["failure_code"] == expected
    assert failures[0]["execution_ref"] == capture()["capture_execution_file_ref"]
    assert failures[0]["phase"] == "POST_CAPTURE_VALIDATION"


def test_security_failure_is_not_downgraded_to_optional_failure(monkeypatch, tmp_path):
    from quant_investor.system.errors import SystemSecurityError

    args, saved, failures = fresh_args(monkeypatch, tmp_path)

    def fail(**kw):
        raise SystemSecurityError("tampered capture")

    monkeypatch.setattr(producer, "capture_next_session_calendar", fail)
    with pytest.raises(SystemSecurityError):
        producer.bind_future_calendar(**args)
    assert not failures and len(saved) == 1


def test_fresh_producer_persists_each_boundary_in_order(monkeypatch, tmp_path):
    args, saved, failures = fresh_args(monkeypatch, tmp_path)
    seen = []

    def acquire(**kw):
        assert saved[-1]["phase"] == "CORE_READY"
        seen.append("capture")
        return {"capture": capture()}

    monkeypatch.setattr(producer, "capture_next_session_calendar", acquire)
    monkeypatch.setattr(
        producer, "publish_fixture_evidence", lambda **kw: seen.append("evidence") or REF
    )

    def proof(**kw):
        assert saved[-1]["phase"] == "CAPTURE_BOUND"
        seen.append("proof")
        return REF

    monkeypatch.setattr(producer, "publish_bound_fixture_proof", proof)
    producer.bind_future_calendar(**args)
    assert seen == ["capture", "evidence", "proof"]
    assert [v["phase"] for v in saved] == ["CORE_READY", "CAPTURE_BOUND", "FUTURE_BOUND"]
