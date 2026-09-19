"""Production state/routing tests; native custody/installation seams are explicit."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib

import pytest

from quant_investor.market import future_calendar_producer as dispatch
from quant_investor.market import production_future_calendar as production
from quant_investor.market import next_session_acquisition as acquisition
from quant_investor.market import production_calendar_evidence as evidence
from quant_investor.market import _calendar_production_transport as recorder
from quant_investor.market.future_calendar_context import (
    PRODUCTION_CONTEXT_SCHEMA,
    PRODUCTION_STATE_SCHEMA,
    PRODUCTION_MODE,
    validate_loop_context,
    validate_future_state,
)
from quant_investor.operations.daily_contract import ContractError
from quant_investor.system.errors import SystemSecurityError
from test_future_calendar_context import context, state, REF
from test_future_calendar_producer import capture as base_capture


def capture():
    value = base_capture()
    root = "future-20260902-" + hashlib.sha256(b"unit-install-input").hexdigest()[:16]
    for role in ("execution", "success"):
        value["capture_" + role + "_file_ref"]["relative_path"] = (
            root + "/capture-" + role + ".json"
        )
    return value


def configured():
    return {
        **context(),
        "schema_version": PRODUCTION_CONTEXT_SCHEMA,
        "next_session_calendar_mode": PRODUCTION_MODE,
    }


def retained():
    value = state()
    value.pop("fixture_transport_evidence_ref")
    return {**value, "schema_version": PRODUCTION_STATE_SCHEMA, "transport_evidence_ref": None}


def validate(value, mode=PRODUCTION_MODE):
    return validate_future_state(
        value,
        context_sha256="c" * 64,
        trade_date="20260902",
        mode=mode,
        context_schema=PRODUCTION_CONTEXT_SCHEMA,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "v2_production",
        "v3_fixture",
        "both_refs",
        "wrong_state_version",
        "early_transport",
        "sealed_without_capture",
        "bound_without_transport",
    ],
)
def test_versioned_context_and_durable_state_boundaries(fault):
    ctx, value = configured(), retained()
    if fault == "v2_production":
        ctx["schema_version"] = "cn-daily-factor-loop.v2"
    elif fault == "v3_fixture":
        ctx["next_session_calendar_mode"] = "SYNTHETIC_FIXTURE_ONLY"
    elif fault == "both_refs":
        value["fixture_transport_evidence_ref"] = None
    elif fault == "wrong_state_version":
        value["schema_version"] = "cn-daily-factor-state.v2"
    elif fault == "early_transport":
        value["transport_evidence_ref"] = REF
    elif fault == "sealed_without_capture":
        value["phase"] = "CAPTURE_SEALED"
    elif fault == "bound_without_transport":
        value.update(
            phase="TRANSPORT_BOUND",
            future_capture_refs={
                "execution_ref": capture()["capture_execution_file_ref"],
                "success_ref": capture()["capture_success_file_ref"],
            },
        )
    with pytest.raises(ContractError):
        validate_loop_context(ctx)
        validate(value)


def setup(tmp_path, monkeypatch):
    ctx = configured()
    raw = b"unit-install-input"
    path = tmp_path / "release.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    ctx["release_install_input_ref"] = {
        "path": "release.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    ctx["release_repository_root"] = str(tmp_path)
    durable = {"capture": False, "transport": False, "proof": False, "scope": False}
    saved, calls, failures = [], [], []

    @contextmanager
    def scope(**kw):
        calls.append("scope")
        durable["scope"] = True
        try:
            yield object()
        finally:
            durable["scope"] = False

    def acquire(**kw):
        assert durable["scope"]
        calls.append("capture")
        durable["capture"] = True
        return {"capture": capture()}

    def transport(**kw):
        assert durable["scope"]
        calls.append("transport")
        durable["transport"] = True
        return REF

    def read_transport(*args):
        if not durable["transport"]:
            raise ContractError("PROVENANCE_UNAVAILABLE")
        return REF, {}

    def publish(**kw):
        assert not durable["scope"]
        calls.append("proof")
        durable["proof"] = True
        return REF

    monkeypatch.setattr(recorder, "_production_transport_scope", scope)
    monkeypatch.setattr(acquisition, "capture_next_session_calendar", acquire)
    monkeypatch.setattr(evidence, "publish_transport_evidence", transport)
    monkeypatch.setattr(
        production, "_retained_capture", lambda *a: capture() if durable["capture"] else None
    )
    monkeypatch.setattr(production, "_retained_native_failure", lambda *a: None)
    monkeypatch.setattr(production, "inspect_next_session_capture", lambda **kw: {})
    monkeypatch.setattr(production, "read_transport_evidence", read_transport)
    monkeypatch.setattr(production, "publish_production_next_session_proof", publish)
    monkeypatch.setattr(
        production, "publish_next_session_failure", lambda **kw: failures.append(kw) or REF
    )
    monkeypatch.setattr(production, "future_calendar_outputs", lambda **kw: {})
    monkeypatch.setattr(
        production,
        "read_next_session_proof",
        lambda **kw: {
            "synthetic": False,
            "live_eligible": True,
            "proof": {
                "transport_evidence_ref": REF,
                "execution_ref": capture()["capture_execution_file_ref"],
                "success_ref": capture()["capture_success_file_ref"],
            },
        },
    )
    args = dict(
        workspace=tmp_path,
        context=ctx,
        context_sha="c" * 64,
        state=retained(),
        save_state=lambda s: saved.append(deepcopy(s)),
    )
    return args, durable, saved, calls, failures


def forbid_fresh(monkeypatch):
    def forbidden(**kw):
        pytest.fail("recovery reached fresh scope/capture/transport publisher")

    monkeypatch.setattr(recorder, "_production_transport_scope", forbidden)
    monkeypatch.setattr(acquisition, "capture_next_session_calendar", forbidden)
    monkeypatch.setattr(evidence, "publish_transport_evidence", forbidden)


def test_fresh_dispatch_persists_all_boundaries_and_reuses_proof(tmp_path, monkeypatch):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    result = dispatch.bind_future_calendar(**args)
    assert [s["phase"] for s in saved] == [
        "CORE_READY",
        "CAPTURE_SEALED",
        "TRANSPORT_BOUND",
        "FUTURE_BOUND",
    ]
    assert calls == ["scope", "capture", "transport", "proof"]
    assert result["next_session_calendar_proof_ref"] == REF and not failures
    forbid_fresh(monkeypatch)
    args["save_state"] = lambda s: pytest.fail("bound recovery wrote")
    assert dispatch.bind_future_calendar(**args, recovery=True) == result


@pytest.mark.parametrize(
    "point,expected",
    [
        ("CORE_READY", "ACQUISITION_FAILED"),
        ("CAPTURE_SEALED", "PROVENANCE_UNAVAILABLE"),
        ("TRANSPORT_BOUND", "proof"),
        ("FUTURE_BOUND", "proof"),
    ],
)
def test_each_durable_crash_recovers_without_network(tmp_path, monkeypatch, point, expected):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)

    def save(value):
        saved.append(deepcopy(value))
        if value["phase"] == point:
            raise RuntimeError("unit crash")

    args["save_state"] = save
    with pytest.raises(RuntimeError, match="unit crash"):
        dispatch.bind_future_calendar(**args)
    args["state"] = saved[-1]
    args["save_state"] = lambda s: None
    before_capture_count = calls.count("capture")
    forbid_fresh(monkeypatch)
    result = dispatch.bind_future_calendar(**args, recovery=True)
    assert calls.count("capture") == before_capture_count
    if expected == "proof":
        assert result["next_session_calendar_proof_ref"] == REF and not failures
    else:
        assert failures[-1]["failure_code"] == expected
        assert result["next_session_calendar_proof_ref"] is None


def test_declared_transport_receipt_missing_is_integrity_error(tmp_path, monkeypatch):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    durable["capture"] = True
    args["state"].update(
        phase="TRANSPORT_BOUND",
        transport_evidence_ref=REF,
        future_capture_refs={
            "execution_ref": capture()["capture_execution_file_ref"],
            "success_ref": capture()["capture_success_file_ref"],
        },
    )
    forbid_fresh(monkeypatch)
    with pytest.raises(SystemSecurityError, match="TRANSPORT_BINDING_INVALID"):
        dispatch.bind_future_calendar(**args, recovery=True)
    assert not saved and not failures


def test_selected_calendar_cannot_gain_new_provenance(tmp_path, monkeypatch):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(production, "selected_binding", lambda *a: ("selected", 1, None))
    forbid_fresh(monkeypatch)
    with pytest.raises(ContractError, match="ALREADY_SELECTED"):
        dispatch.bind_future_calendar(**args)
    assert not saved and not calls and not failures


def test_disabled_v3_never_opens_transport(tmp_path, monkeypatch):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    args["context"]["next_session_calendar_mode"] = "DISABLED"
    forbid_fresh(monkeypatch)
    result = dispatch.bind_future_calendar(**args)
    assert result["phase"] == "FUTURE_BOUND" and not calls
    assert dispatch.core_future_arguments(result) == {
        "next_session_calendar_proof_ref": None,
        "next_session_calendar_failure_ref": None,
    }


def test_v3_core_recovery_forwards_exact_retained_refs(tmp_path, monkeypatch):
    from test_future_calendar_core_recovery import setup as setup_core

    instance, value, calls, writes = setup_core(tmp_path, monkeypatch)
    instance.context = configured()
    value["schema_version"] = PRODUCTION_STATE_SCHEMA
    value["transport_evidence_ref"] = value.pop("fixture_transport_evidence_ref")
    instance._recover_core(value)
    assert calls[0] == "retained-binding"
    assert calls[1]["next_session_calendar_failure_ref"] == REF
    assert writes == []


@pytest.mark.parametrize(
    "point,expected",
    [("capture", "PROVENANCE_UNAVAILABLE"), ("transport", "proof"), ("proof", "proof")],
)
def test_crash_before_state_publication_adopts_only_retained_evidence(
    tmp_path, monkeypatch, point, expected
):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    module, name = {
        "capture": (acquisition, "capture_next_session_calendar"),
        "transport": (evidence, "publish_transport_evidence"),
        "proof": (production, "publish_production_next_session_proof"),
    }[point]
    original = getattr(module, name)

    def interrupted(**kwargs):
        original(**kwargs)
        raise RuntimeError("after durable object before state")

    monkeypatch.setattr(module, name, interrupted)
    with pytest.raises(RuntimeError, match="after durable object"):
        dispatch.bind_future_calendar(**args)
    monkeypatch.setattr(module, name, original)
    args["state"] = deepcopy(saved[-1])
    args["save_state"] = lambda state: None
    captures = calls.count("capture")
    forbid_fresh(monkeypatch)
    result = dispatch.bind_future_calendar(**args, recovery=True)
    assert calls.count("capture") == captures
    if expected == "proof":
        assert result["next_session_calendar_proof_ref"] == REF and not failures
    else:
        assert failures[-1]["failure_code"] == expected


def test_core_published_recovery_preserves_original_state(tmp_path, monkeypatch):
    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    dispatch.bind_future_calendar(**args)
    args["state"].update(phase="CORE_PUBLISHED", core_handoff_ref=REF)
    args["save_state"] = lambda state: pytest.fail("completed state changed")
    forbid_fresh(monkeypatch)
    before = deepcopy(args["state"])
    assert dispatch.bind_future_calendar(**args, recovery=True) == before


def test_public_context_rejects_fixture_before_installation_or_directory_write(
    tmp_path, monkeypatch
):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market import daily_factor_loop
    from quant_investor.market._calendar_fixture_capability import _ACTIVE

    raw = canonical_json_bytes(configured())
    path = tmp_path / "context.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    monkeypatch.setattr(
        daily_factor_loop,
        "verify_running_release_install_input",
        lambda *a, **k: pytest.fail("fixture reached install verification"),
    )
    token = _ACTIVE.set(object())
    try:
        with pytest.raises(SystemSecurityError, match="TRANSPORT_ROUTE_UNAVAILABLE"):
            daily_factor_loop.read_factor_loop_context(
                workspace_root=str(tmp_path),
                context_path=str(path),
                context_sha256=hashlib.sha256(raw).hexdigest(),
            )
    finally:
        _ACTIVE.reset(token)
    assert list(tmp_path.iterdir()) == [path]


def test_internal_recovery_entry_also_rejects_fixture_scope(tmp_path, monkeypatch):
    from quant_investor.market._calendar_fixture_capability import _ACTIVE

    args, durable, saved, calls, failures = setup(tmp_path, monkeypatch)
    token = _ACTIVE.set(object())
    try:
        with pytest.raises(SystemSecurityError, match="TRANSPORT_ROUTE_UNAVAILABLE"):
            dispatch.bind_future_calendar(**args, recovery=True)
    finally:
        _ACTIVE.reset(token)
    assert not saved and not calls and not failures
